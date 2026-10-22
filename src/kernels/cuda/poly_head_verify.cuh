#pragma once

template<int NC, int ROWS>
__launch_bounds__(128)
__global__ void poly_head_q6k_verify_kernel(const Q6KBlock* __restrict__ w,
                                           const Q81Block* __restrict__ x,
                                           float* __restrict__ y, int n_out) {
    const int lane = int(threadIdx.x);
    const int wave = int(threadIdx.y);
    constexpr int XB = kPolyHeadIn / 32;
    const int xoff = 4 * (lane / 16) + (lane % 16) / 8;
    int u[3][NC][2];
    float d8[3][NC][2];
#pragma unroll
    for (int s = 0; s < 3; ++s) {
#pragma unroll
        for (int col = 0; col < NC; ++col) {
            const Q81Block* xb = x + col * XB + (wave + 4 * s) * 8 + xoff;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                u[s][col][i] = reinterpret_cast<const int*>(xb[2 * i].qs)[lane % 8];
                d8[s][col][i] = __low2float(xb[2 * i].ds);
            }
        }
    }
    __shared__ float partial[2][3][ROWS][NC][32];
    int buf = 0;
    for (int row0 = ROWS * int(blockIdx.x); row0 < n_out; row0 += ROWS * int(gridDim.x), buf ^= 1) {
        float sum[ROWS][NC] = {};
#pragma unroll
        for (int s = 0; s < 3; ++s) {
#pragma unroll
            for (int row = 0; row < ROWS; ++row) {
                const int actual_row = min(row0 + row, n_out - 1);
                const Q6KTraits::W weight = Q6KTraits::load(
                    w + size_t(actual_row) * kPolyHeadBlocks + wave + 4 * s, lane);
#pragma unroll
                for (int col = 0; col < NC; ++col)
                    sum[row][col] += q6_q8_dot_impl(
                        weight.vl, weight.vh, u[s][col], weight.scales, weight.d, d8[s][col]);
            }
        }
        if (wave > 0) {
#pragma unroll
            for (int row = 0; row < ROWS; ++row) {
#pragma unroll
                for (int col = 0; col < NC; ++col)
                    partial[buf][wave - 1][row][col][lane] = sum[row][col];
            }
        }
        __syncthreads();
        if (wave == 0) {
            constexpr int P = s26ts::pow2_ceil(ROWS * NC);
            float values[P];
#pragma unroll
            for (int row = 0; row < ROWS; ++row) {
#pragma unroll
                for (int col = 0; col < NC; ++col) {
#pragma unroll
                    for (int other = 0; other < 3; ++other)
                        sum[row][col] += partial[buf][other][row][col][lane];
                    values[row * NC + col] = sum[row][col];
                }
            }
#pragma unroll
            for (int i = ROWS * NC; i < P; ++i) values[i] = 0.0f;
            const float value = s26ts::tsum<P>(values, lane);
            const int output = s26ts::tsum_token<P>(lane);
            const int row = output / NC;
            const int col = output % NC;
            if (output < ROWS * NC && lane == s26ts::tsum_lane<P>(output) && row0 + row < n_out)
                y[size_t(col) * n_out + row0 + row] = value;
        }
    }
}

inline bool poly_head_verify_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_HEAD_VERIFY");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

template<int NC>
inline void poly_head_verify_cols(const Q6KBlock* w, const Q81Block* x, float* y,
                                  int n_out, cudaStream_t stream) {
    static const int rows = [] {
        const char* v = std::getenv("STRATA_POLY_HEAD_VERIFY_ROWS");
        const int parsed = v ? std::atoi(v) : 2;
        return parsed >= 4 ? 4 : parsed >= 2 ? 2 : 1;
    }();
    static const int requested_grid = [] {
        const char* v = std::getenv("STRATA_POLY_HEAD_VERIFY_GRID");
        const int parsed = v ? std::atoi(v) : 0;
        return parsed > 0 ? std::max(256, std::min(16384, parsed)) : 2048;
    }();
    const unsigned grid = unsigned(std::min((n_out + rows - 1) / rows, requested_grid));
    const dim3 threads(32, 4);
    if (rows == 1)
        poly_head_q6k_verify_kernel<NC, 1><<<grid, threads, 0, stream>>>(w, x, y, n_out);
    else if (rows == 2)
        poly_head_q6k_verify_kernel<NC, 2><<<grid, threads, 0, stream>>>(w, x, y, n_out);
    else
        poly_head_q6k_verify_kernel<NC, 4><<<grid, threads, 0, stream>>>(w, x, y, n_out);
}

inline void poly_head_verify(const Q6KBlock* w, const Q81Block* x, float* y,
                             int n_out, int ncols, cudaStream_t stream) {
    switch (ncols) {
    case 2: poly_head_verify_cols<2>(w, x, y, n_out, stream); break;
    case 3: poly_head_verify_cols<3>(w, x, y, n_out, stream); break;
    case 4: poly_head_verify_cols<4>(w, x, y, n_out, stream); break;
    }
}
