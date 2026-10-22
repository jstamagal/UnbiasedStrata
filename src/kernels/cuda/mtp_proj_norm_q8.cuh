#pragma once

template<int SUM, bool EMBED_READY>
__launch_bounds__(1024, 1)
__global__ void mtp_proj_norm_q8_kernel(const float* __restrict__ x0, const float* __restrict__ w0,
                                       const float* __restrict__ x1, const float* __restrict__ w1,
                                       float* y, Q81Block* __restrict__ yq, int n, float eps) {
    __shared__ float part[SUM / WARP];
    const bool second = blockIdx.y != 0;
    const bool ready = EMBED_READY && !second;
    const float* xr = (second ? x1 : x0) + size_t(blockIdx.x) * n;
    const float* w = second ? w1 : w0;
    float* yr = y + size_t(blockIdx.x) * (2 * n) + (second ? n : 0);
    Q81Block* qr = yq + size_t(blockIdx.x) * (2 * n / Q8K) + (second ? n / Q8K : 0);
    float scale = 1.0f;
    if (!ready) {
        if (threadIdx.x < SUM) {
            float ss = 0.0f;
            for (int i = threadIdx.x; i < n; i += SUM) ss = __fmaf_rn(xr[i], xr[i], ss);
            ss = warp_sum(ss);
            if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
        }
        __syncthreads();
        float sum = 0.0f;
        for (int k = 0; k < SUM / WARP; ++k) sum = __fadd_rn(sum, part[k]);
        scale = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, float(n)), eps)));
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xi = ready ? yr[i] : __fmul_rn(__fmul_rn(xr[i], scale), w[i]);
        if (!ready) yr[i] = xi;
        const float amax = warp_max(fabsf(xi));
        const float sum = warp_sum(xi);
        const float d = q8_1_finite(amax / 127.0f);
        qr[i / Q8K].qs[i % Q8K] = q8_1_quant(xi, d, amax);
        if (i % Q8K == 0) qr[i / Q8K].ds = q8_1_ds(d, sum);
    }
}

template<int SUM>
inline void mtp_proj_norm_q8_launch(const float* x0, const float* w0, const float* x1, const float* w1,
                                    float* y, Q81Block* q, int n, int rows, float eps,
                                    bool embed_ready, cudaStream_t stream) {
    const int block = std::max(SUM, std::min(1024, n));
    const dim3 grid(rows, 2);
    if (embed_ready)
        mtp_proj_norm_q8_kernel<SUM, true><<<grid, block, 0, stream>>>(x0, w0, x1, w1, y, q, n, eps);
    else
        mtp_proj_norm_q8_kernel<SUM, false><<<grid, block, 0, stream>>>(x0, w0, x1, w1, y, q, n, eps);
}
