#pragma once
#if defined(__HIPCC__)
// Resident Q6_K decode head: reuse its quantized input across output rows.
constexpr int kPolyHeadIn = 3072;
constexpr int kPolyHeadBlocks = kPolyHeadIn / 256;
constexpr int kPolyHeadBlockBytes = 210;

#include "poly_head_aligned.cuh"
#include "poly_head_preload.cuh"
#include "poly_head_verify.cuh"

template<int NC>
__launch_bounds__(128)
__global__ void poly_head_q6k_pair_kernel(const Q6KBlock* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_out) {
    const int row0 = 8 * int(blockIdx.x) + 2 * int(threadIdx.y);
    if (row0 >= n_out) return;
    const int lane = int(threadIdx.x);
    constexpr int XB = kPolyHeadIn / 32;
    const int xoff = 4 * (lane / 16) + (lane % 16) / 8;
    float sum[2][NC] = {};
#pragma unroll 1
    for (int kb = 0; kb < kPolyHeadBlocks; ++kb) {
        int u[NC][2];
        float d8[NC][2];
#pragma unroll
        for (int col = 0; col < NC; ++col) {
            const Q81Block* xb = x + col * XB + kb * 8 + xoff;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                u[col][i] = reinterpret_cast<const int*>(xb[2 * i].qs)[lane % 8];
                d8[col][i] = __low2float(xb[2 * i].ds);
            }
        }
#pragma unroll
        for (int row = 0; row < 2; ++row) {
            const int actual_row = min(row0 + row, n_out - 1);
            const Q6KTraits::W weight = Q6KTraits::load(w + size_t(actual_row) * kPolyHeadBlocks + kb, lane);
#pragma unroll
            for (int col = 0; col < NC; ++col)
                sum[row][col] += q6_q8_dot_impl(weight.vl, weight.vh, u[col], weight.scales, weight.d, d8[col]);
        }
    }
#pragma unroll
    for (int row = 0; row < 2; ++row) {
#pragma unroll
        for (int col = 0; col < NC; ++col) {
            const float value = warp_sum(sum[row][col]);
            if (lane == 0 && row0 + row < n_out)
                y[size_t(col) * n_out + row0 + row] = value;
        }
    }
}

inline bool poly_head_pair_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_HEAD_PAIR");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

// A nonpersistent wave owns one output row. The four waves of a block write
// four independent rows, avoiding the native kernel's cross-wave reduction.
// For verify windows, each wave decodes its Q6_K weights once for all columns.
template<int NC>
__launch_bounds__(128)
__global__ void poly_head_q6k_wave_kernel(const Q6KBlock* __restrict__ w,
                                          const Q81Block* __restrict__ x,
                                          float* __restrict__ y, int n_out) {
    const int row = 4 * int(blockIdx.x) + int(threadIdx.y);
    if (row >= n_out) return;
    const int lane = int(threadIdx.x);
    const Q6KBlock* wr = w + size_t(row) * kPolyHeadBlocks;
    constexpr int XB = kPolyHeadIn / 32;
    float sum[NC] = {};
#pragma unroll 1
    for (int kb = 0; kb < kPolyHeadBlocks; ++kb) {
        const Q6KTraits::W weight = Q6KTraits::load(wr + kb, lane);
#pragma unroll
        for (int col = 0; col < NC; ++col)
            sum[col] += Q6KTraits::apply(weight, x + col * XB + kb * 8, lane);
    }
#pragma unroll
    for (int col = 0; col < NC; ++col) {
        const float value = warp_sum(sum[col]);
        if (lane == 0) y[size_t(col) * n_out + row] = value;
    }
}

// Two waves cooperate on a row. This halves the original kernel's shared
// reduction while keeping more weight reads in flight than one wave per row.
template<int NC>
__launch_bounds__(128)
__global__ void poly_head_q6k_two_wave_kernel(const Q6KBlock* __restrict__ w,
                                              const Q81Block* __restrict__ x,
                                              float* __restrict__ y, int n_out) {
    const int row = 2 * int(blockIdx.x) + int(threadIdx.y) / 2;
    const int half = int(threadIdx.y) & 1;
    const int lane = int(threadIdx.x);
    const Q6KBlock* wr = w + size_t(row < n_out ? row : 0) * kPolyHeadBlocks;
    constexpr int XB = kPolyHeadIn / 32;
    float sum[NC] = {};
#pragma unroll 1
    for (int kb = half; kb < kPolyHeadBlocks; kb += 2) {
        const Q6KTraits::W weight = Q6KTraits::load(wr + kb, lane);
#pragma unroll
        for (int col = 0; col < NC; ++col)
            sum[col] += Q6KTraits::apply(weight, x + col * XB + kb * 8, lane);
    }
    __shared__ float partial[2][NC][32];
    if (half) {
#pragma unroll
        for (int col = 0; col < NC; ++col)
            partial[int(threadIdx.y) / 2][col][lane] = sum[col];
    }
    __syncthreads();
    if (half) return;
#pragma unroll
    for (int col = 0; col < NC; ++col) {
        const float value = warp_sum(sum[col] + partial[int(threadIdx.y) / 2][col][lane]);
        if (lane == 0 && row < n_out) y[size_t(col) * n_out + row] = value;
    }
}

inline bool poly_head_two_wave_enabled(int n_in, int n_out, int ncols) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_HEAD_2WAVE");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on && n_in == kPolyHeadIn && n_out >= 65536 && ncols >= 1 && ncols <= 4;
}

inline void poly_head_two_wave(const void* weights, const void* x, float* y, int n_out,
                               int ncols, cudaStream_t stream) {
    const dim3 blocks(unsigned((n_out + 1) / 2));
    const dim3 threads(32, 4);
    const auto* w = static_cast<const Q6KBlock*>(weights);
    const auto* q = static_cast<const Q81Block*>(x);
    switch (ncols) {
    case 1: poly_head_q6k_two_wave_kernel<1><<<blocks, threads, 0, stream>>>(w, q, y, n_out); break;
    case 2: poly_head_q6k_two_wave_kernel<2><<<blocks, threads, 0, stream>>>(w, q, y, n_out); break;
    case 3: poly_head_q6k_two_wave_kernel<3><<<blocks, threads, 0, stream>>>(w, q, y, n_out); break;
    case 4: poly_head_q6k_two_wave_kernel<4><<<blocks, threads, 0, stream>>>(w, q, y, n_out); break;
    }
}

inline bool poly_head_wave_enabled(int n_in, int n_out, int ncols) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_HEAD_WAVE");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return (on || poly_head_pair_enabled() || (ncols == 1 && poly_head_preload_enabled()) ||
            (ncols >= 2 && poly_head_verify_enabled())) &&
           n_in == kPolyHeadIn && n_out >= 65536 && ncols >= 1 && ncols <= 4;
}

inline void poly_head_wave(const void* weights, const void* x, float* y, int n_out,
                           int ncols, cudaStream_t stream) {
    const dim3 blocks(unsigned((n_out + 3) / 4));
    const dim3 threads(32, 4);
    const auto* w = static_cast<const Q6KBlock*>(weights);
    const auto* q = static_cast<const Q81Block*>(x);
    if (ncols >= 2 && poly_head_verify_enabled()) {
        poly_head_verify(w, q, y, n_out, ncols, stream);
        return;
    }
    if (ncols == 1 && poly_head_preload_enabled()) {
        poly_head_preload(w, q, y, n_out, stream);
        return;
    }
    if (poly_head_pair_enabled()) {
        const dim3 pair_blocks(unsigned((n_out + 7) / 8));
        switch (ncols) {
        case 1: poly_head_q6k_pair_kernel<1><<<pair_blocks, threads, 0, stream>>>(w, q, y, n_out); break;
        case 2: poly_head_q6k_pair_kernel<2><<<pair_blocks, threads, 0, stream>>>(w, q, y, n_out); break;
        case 3: poly_head_q6k_pair_kernel<3><<<pair_blocks, threads, 0, stream>>>(w, q, y, n_out); break;
        case 4: poly_head_q6k_pair_kernel<4><<<pair_blocks, threads, 0, stream>>>(w, q, y, n_out); break;
        }
        return;
    }
    switch (ncols) {
    case 1: poly_head_q6k_wave_kernel<1><<<blocks, threads, 0, stream>>>(w, q, y, n_out); break;
    case 2: poly_head_q6k_wave_kernel<2><<<blocks, threads, 0, stream>>>(w, q, y, n_out); break;
    case 3: poly_head_q6k_wave_kernel<3><<<blocks, threads, 0, stream>>>(w, q, y, n_out); break;
    case 4: poly_head_q6k_wave_kernel<4><<<blocks, threads, 0, stream>>>(w, q, y, n_out); break;
    }
}
__device__ __forceinline__ int poly_head_load_i32(const uint8_t* p) {
    int v;
    __builtin_memcpy(&v, p, 4);
    return v;
}
template <int NC, bool GUARD_FINAL_PREFETCH = false>
__launch_bounds__(128)
__global__ void poly_head_q6k_kernel(const uint8_t* __restrict__ w, const Q81Block* __restrict__ x, float* __restrict__ y,
                                      int n_out, int row_step) {
    const int lane = int(threadIdx.x), wave = int(threadIdx.y);
    const int iqs = lane;
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int qh_word = 8 * (iqs / 16) + iqs % 8;
    const int slots = wave < kPolyHeadBlocks - 8 ? 3 : 2;
    constexpr int XB = kPolyHeadIn / 32;   // Q8_1 blocks per column
    __shared__ float etab[NC * XB];
    for (int i = wave * 32 + lane; i < NC * XB; i += 128) etab[i] = __low2float(x[i].ds);
    int u0[NC][3], u1[NC][3];
#pragma unroll
    for (int s = 0; s < 3; ++s) {
        const int kbx = wave + 4 * s < kPolyHeadBlocks ? wave + 4 * s : wave;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const Q81Block* xb = x + j * XB + kbx * 8 + bq8_offset;
            u0[j][s] = reinterpret_cast<const int*>(xb[0].qs)[iqs % 8];
            u1[j][s] = reinterpret_cast<const int*>(xb[2].qs)[iqs % 8];
        }
    }
    __syncthreads();
    const float* et = etab + wave * 8 + bq8_offset;   // + j * XB + s * 32 (+ 2)
    __shared__ float partial[2][3][NC][32];
    constexpr int P = s26ts::pow2_ceil(NC);
    const int my_col = s26ts::tsum_token<P>(lane);
    const bool writer = my_col < NC && lane == s26ts::tsum_lane<P>(my_col);
    int rl[3], rh[3], rs0[3], rs1[3];
    half rd[3];
    const std::size_t lane_ql = 4 * iqs, lane_qh = 128 + 4 * qh_word, lane_sc = 192 + scale_offset;
    int row = int(blockIdx.x);
    if (row < n_out) {
        const uint8_t* wr = w + std::size_t(row) * (kPolyHeadBlocks * kPolyHeadBlockBytes) + wave * kPolyHeadBlockBytes;
#pragma unroll
        for (int s = 0; s < 3; ++s) {
            if (s < slots) {
                const uint8_t* wb = wr + s * (4 * kPolyHeadBlockBytes);
                rl[s] = poly_head_load_i32(wb + lane_ql);
                rh[s] = poly_head_load_i32(wb + lane_qh);
                rs0[s] = reinterpret_cast<const int8_t*>(wb)[lane_sc];
                rs1[s] = reinterpret_cast<const int8_t*>(wb)[lane_sc + 4];
                rd[s] = *reinterpret_cast<const half*>(wb + 208);
            }
        }
    }
    int buf = 0;
    for (; row < n_out; row += row_step, buf ^= 1) {
        const bool has_next = row + row_step < n_out;
        const int next = has_next ? row + row_step : row;
        const uint8_t* wn = w + std::size_t(next) * (kPolyHeadBlocks * kPolyHeadBlockBytes) + wave * kPolyHeadBlockBytes;
        float tmp[NC];
#pragma unroll
        for (int j = 0; j < NC; ++j) tmp[j] = 0.0f;
#pragma unroll
        for (int s = 0; s < 3; ++s) {
            if (s < slots) {
                const int vl = rl[s];
                const int vh = rh[s] >> vh_shift;
                const int sc0 = rs0[s], sc1 = rs1[s];
                const float d = __half2float(rd[s]);
                const int vi0 = int((uint32_t((vl & 0x0f0f0f0f) | ((vh << 4) & 0x30303030)) + 0x60606060u) ^ 0x80808080u);
                const int vi1 = int((uint32_t(((vl >> 4) & 0x0f0f0f0f) | (vh & 0x30303030)) + 0x60606060u) ^ 0x80808080u);
                if (!GUARD_FINAL_PREFETCH || has_next) {
                    const uint8_t* wb = wn + s * (4 * kPolyHeadBlockBytes);
                    rl[s] = poly_head_load_i32(wb + lane_ql);
                    rh[s] = poly_head_load_i32(wb + lane_qh);
                    rs0[s] = reinterpret_cast<const int8_t*>(wb)[lane_sc];
                    rs1[s] = reinterpret_cast<const int8_t*>(wb)[lane_sc + 4];
                    rd[s] = *reinterpret_cast<const half*>(wb + 208);
                }
#pragma unroll
                for (int j = 0; j < NC; ++j) {
                    const float e0 = et[j * XB + s * 32], e1 = et[j * XB + s * 32 + 2];
                    float sumf = 0.0f;                                         // q6_q8_dot_impl's tail, verbatim
                    sumf += e0 * (STRATA_DP4A(vi0, u0[j][s], 0) * sc0);
                    sumf += e1 * (STRATA_DP4A(vi1, u1[j][s], 0) * sc1);
                    tmp[j] += d * sumf;
                }
            }
        }
        if (wave != 3) {
#pragma unroll
            for (int j = 0; j < NC; ++j) partial[buf][wave][j][lane] = tmp[j];
        }
        __syncthreads();
        if (wave == 3) {
            float v[P];
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                float t = partial[buf][0][j][lane];
                t += partial[buf][1][j][lane];
                t += partial[buf][2][j][lane];
                t += tmp[j];
                v[j] = t;
            }
#pragma unroll
            for (int j = NC; j < P; ++j) v[j] = 0.0f;
            const float sum = s26ts::tsum<P>(v, lane);
            if (writer) y[std::size_t(my_col) * n_out + row] = sum;
        }
    }
}


inline bool poly_head_wide_enabled(int n_in, int n_out, int ncols) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_HEAD_WIDE");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on && n_in == kPolyHeadIn && n_out >= 65536 && ncols == 1;
}
inline void poly_head_wide(const void* weights, const void* x, float* y, int n_out, cudaStream_t stream) {
    static const int requested_grid = [] {
        const char* v = std::getenv("STRATA_POLY_HEAD_WIDE_GRID");
        const int parsed = v ? std::atoi(v) : 0;
        return parsed > 0 ? std::max(256, std::min(16384, parsed)) : 0;
    }();
    const int grid = requested_grid ? std::min(n_out, requested_grid) : 4096;
    const auto* w = static_cast<const uint8_t*>(weights);
    const auto* q = static_cast<const Q81Block*>(x);
    if (requested_grid)
        poly_head_q6k_kernel<1, true><<<grid, dim3(32, 4), 0, stream>>>(w, q, y, n_out, grid);
    else
        poly_head_q6k_kernel<1><<<grid, dim3(32, 4), 0, stream>>>(w, q, y, n_out, grid);
}
#else
inline bool poly_head_wide_enabled(int, int, int) { return false; }
#endif
