#pragma once

__launch_bounds__(128)
__global__ void poly_head_q6k_preload_kernel(const Q6KBlock* __restrict__ w,
                                            const Q81Block* __restrict__ x,
                                            float* __restrict__ y, int n_out) {
    const int lane = int(threadIdx.x);
    const int wave = int(threadIdx.y);
    const int xoff = 4 * (lane / 16) + (lane % 16) / 8;
    int u[kPolyHeadBlocks][2];
    float d8[kPolyHeadBlocks][2];
#pragma unroll
    for (int kb = 0; kb < kPolyHeadBlocks; ++kb) {
        const Q81Block* xb = x + kb * 8 + xoff;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[kb][i] = reinterpret_cast<const int*>(xb[2 * i].qs)[lane % 8];
            d8[kb][i] = __low2float(xb[2 * i].ds);
        }
    }
    for (int row = 4 * int(blockIdx.x) + wave; row < n_out; row += 4 * int(gridDim.x)) {
        float sum[4] = {};
        const Q6KBlock* wr = w + size_t(row) * kPolyHeadBlocks;
#pragma unroll
        for (int kb = 0; kb < kPolyHeadBlocks; ++kb) {
            const Q6KTraits::W weight = Q6KTraits::load(wr + kb, lane);
            sum[kb & 3] += q6_q8_dot_impl(weight.vl, weight.vh, u[kb], weight.scales, weight.d, d8[kb]);
        }
        const float value = warp_sum(((sum[0] + sum[1]) + sum[2]) + sum[3]);
        if (lane == 0) y[row] = value;
    }
}

inline bool poly_head_preload_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_HEAD_PRELOAD");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

inline void poly_head_preload(const Q6KBlock* weights, const Q81Block* x, float* y,
                              int n_out, cudaStream_t stream) {
    static const int requested_grid = [] {
        const char* v = std::getenv("STRATA_POLY_HEAD_PRELOAD_GRID");
        const int parsed = v ? std::atoi(v) : 0;
        return parsed > 0 ? std::max(256, std::min(16384, parsed)) : 2048;
    }();
    const unsigned grid = unsigned(std::min((n_out + 3) / 4, requested_grid));
    poly_head_q6k_preload_kernel<<<grid, dim3(32, 4), 0, stream>>>(weights, x, y, n_out);
}
