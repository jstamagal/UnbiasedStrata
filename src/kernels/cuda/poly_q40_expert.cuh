// Optional Q4_0 expert decode layout for RDNA2 wave32. Included by native_mmvq.cu
// after Q40Block, Q81Block, MmvqSlots/MmvqIds, and slot_ptrs are defined.
#pragma once

#include "poly_q40_expert_block.cuh"

#if defined(__HIPCC__)
template<typename Weight, typename S, bool PARTIAL = false>
__global__ void __launch_bounds__(128) poly_q40_expert_wave_kernel(
        S slots, int n_in, int n_out) {
    const int j = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    const int row = 4 * int(blockIdx.x) + int(threadIdx.y);
    if constexpr (PARTIAL) {
        if (slots.table[slots.ids[j]] == nullptr) {
            if (lane == 0 && row < n_out) slots.y[size_t(j) * slots.y_stride + row] = 0.f;
            return;
        }
    }
    if (row >= n_out) return;
    const SlotPtrs at = slot_ptrs(slots, j);
    const auto* w = static_cast<const Weight*>(at.w);
    const auto* x = static_cast<const Q81Block*>(at.x);

    // A wave owns one output row. The two 16-value parts of a Q4_0 block
    // retain small_q8_dot's affine correction from the Q8_1 input sum.
    const int nblocks = n_in / 32;
    const Weight* wr = w + size_t(row) * nblocks;
    float sum = 0.0f;
    for (int k = lane; k < 2 * nblocks; k += 32) {
        const int kb = k >> 1;
        const int iqs = (k & 1) * 2;
        // Q4_1: d*q + m; the min term uses the sum of Q8_1's quantized
        // int8 codes, as the Q5_1 path does. Q4_0 keeps its centered sum.
        sum += small_q8_dot(wr + kb, x + kb, iqs);
    }
    sum = warp_sum(sum);
    if (lane == 0) at.y[row] = sum;
}

// Multiple rows share the same Q8_1 words and scales.
template<typename Weight, typename S, bool PARTIAL = false, int ROWS = 2>
__global__ void __launch_bounds__(128) poly_q40_expert_pair_kernel(
        S slots, int n_in, int n_out) {
    const int j = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    const int row0 = 4 * ROWS * int(blockIdx.x) + ROWS * int(threadIdx.y);
    if constexpr (PARTIAL) {
        if (slots.table[slots.ids[j]] == nullptr) {
            if (lane == 0) {
#pragma unroll
                for (int r = 0; r < ROWS; ++r)
                    if (row0 + r < n_out)
                        slots.y[size_t(j) * slots.y_stride + row0 + r] = 0.f;
            }
            return;
        }
    }
    if (row0 >= n_out) return;
    const SlotPtrs at = slot_ptrs(slots, j);
    const auto* w = static_cast<const Weight*>(at.w);
    const auto* x = static_cast<const Q81Block*>(at.x);
    const int nblocks = n_in / 32;
    const Weight* wr[ROWS];
    float sums[ROWS] = {};
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
        wr[r] = w + size_t(row0 + r < n_out ? row0 + r : row0) * nblocks;
    for (int k = lane; k < 2 * nblocks; k += 32) {
        const int kb = k >> 1, iqs = (k & 1) * 2;
        const Q81Block& xb = x[kb];
        int dots[ROWS] = {};
        int sumu = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int qlo = reinterpret_cast<const int*>(xb.qs)[iqs + i];
            const int qhi = reinterpret_cast<const int*>(xb.qs)[iqs + i + 4];
            if constexpr (std::is_same_v<Weight, Q41Block>)
                sumu = STRATA_DP4A(0x01010101, qhi, STRATA_DP4A(0x01010101, qlo, sumu));
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int a = load_int_b2(wr[r][kb].qs, iqs + i);
                dots[r] = STRATA_DP4A(a & 0x0f0f0f0f, qlo, dots[r]);
                dots[r] = STRATA_DP4A((a >> 4) & 0x0f0f0f0f, qhi, dots[r]);
            }
        }
        const float2 ds = __half22float2(xb.ds);
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if constexpr (std::is_same_v<Weight, Q41Block>) {
                const float2 dm = __half22float2(wr[r][kb].dm);
                sums[r] += float(dots[r]) * (dm.x * ds.x) + float(sumu) * (dm.y * ds.x);
            } else {
                sums[r] += float(wr[r][kb].d) * (dots[r] * ds.x - 4 * ds.y);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        sums[r] = warp_sum(sums[r]);
        if (lane == 0 && row0 + r < n_out) at.y[row0 + r] = sums[r];
    }
}

inline int poly_q40_expert_rows_mode() {
    static const int mode = [] {
        const char* v = std::getenv("STRATA_POLY_Q40_EXPERT_ROWS");
        if (!v) return 0;
        if (v[0] == '1' && v[1] == '\0') return 1;
        if (v[0] == '2' && v[1] == '\0') return 2;
        if (v[0] == '4' && v[1] == '\0') return 4;
        if (std::strcmp(v, "auto") == 0) return -1;
        return 0;
    }();
    return mode;
}

inline bool poly_q40_expert_pair_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_Q40_EXPERT_PAIR");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

inline bool poly_q40_expert_enabled(int n_in, int n_out) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_Q40_EXPERT_WAVE");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return (on || poly_q40_expert_pair_on() || poly_q40_expert_rows_mode() != 0 || poly_q40_expert_block_on()) &&
           ((n_in == 3072 && n_out == 2048) ||
            (n_in == 1024 && n_out == 3072));
}

inline bool poly_q40_expert_ids_enabled(int n_in, int n_out) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_EXPERT_IDS");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on && poly_q40_expert_enabled(n_in, n_out);
}

template<typename Weight, typename S, bool PARTIAL = false>
inline void poly_q4_expert_slots(const S& slots, int count,
                                 int n_in, int n_out, cudaStream_t stream) {
    const dim3 threads(32, 4);
    int rows = poly_q40_expert_rows_mode();
    if (rows == -1) {
        // Keep at least 512 workgroups when the actual routed entry count permits.
        rows = 1;
        if (size_t((n_out + 7) / 8) * size_t(count) >= 512) rows = 2;
        if (size_t((n_out + 15) / 16) * size_t(count) >= 512) rows = 4;
    }
    if (rows == 0) rows = poly_q40_expert_pair_on() ? 2 : 1;
    if (poly_q40_expert_block_on()) {
        poly_q40_expert_block_slots<Weight, S, PARTIAL>(slots, count, n_in, n_out, rows, stream);
        return;
    }
    if (rows == 4) {
        const dim3 blocks(unsigned((n_out + 15) / 16), unsigned(count));
        poly_q40_expert_pair_kernel<Weight, S, PARTIAL, 4><<<blocks, threads, 0, stream>>>(slots, n_in, n_out);
    } else if (rows == 2) {
        const dim3 blocks(unsigned((n_out + 7) / 8), unsigned(count));
        poly_q40_expert_pair_kernel<Weight, S, PARTIAL><<<blocks, threads, 0, stream>>>(slots, n_in, n_out);
    } else {
        const dim3 blocks(unsigned((n_out + 3) / 4), unsigned(count));
        poly_q40_expert_wave_kernel<Weight, S, PARTIAL><<<blocks, threads, 0, stream>>>(slots, n_in, n_out);
    }
}
template<typename S>
inline void poly_q40_expert_slots(const S& slots, int count,
                                  int n_in, int n_out, cudaStream_t stream) {
    poly_q4_expert_slots<Q40Block>(slots, count, n_in, n_out, stream);
}
template<typename S>
inline void poly_q41_expert_slots(const S& slots, int count,
                                  int n_in, int n_out, cudaStream_t stream) {
    poly_q4_expert_slots<Q41Block>(slots, count, n_in, n_out, stream);
}
inline void poly_q40_expert_ids_partial(const MmvqIds& ids, int count,
                                         int n_in, int n_out, cudaStream_t stream) {
    poly_q4_expert_slots<Q40Block, MmvqIds, true>(ids, count, n_in, n_out, stream);
}
inline void poly_q41_expert_ids_partial(const MmvqIds& ids, int count,
                                         int n_in, int n_out, cudaStream_t stream) {
    poly_q4_expert_slots<Q41Block, MmvqIds, true>(ids, count, n_in, n_out, stream);
}
#else
inline bool poly_q40_expert_enabled(int, int) { return false; }
#endif
