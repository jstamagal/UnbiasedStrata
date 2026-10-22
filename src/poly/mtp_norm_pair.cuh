#pragma once

namespace {

__global__ void mtp_norm_pair_kernel(const float* __restrict__ x0, const float* __restrict__ w0,
                                     const float* __restrict__ x1, const float* __restrict__ w1,
                                     float* y, int n, float eps) {
    __shared__ float part[32];
    const bool second = blockIdx.y != 0;
    const float* xr = (second ? x1 : x0) + size_t(blockIdx.x) * n;
    const float* w = second ? w1 : w0;
    float* yr = y + size_t(blockIdx.x) * (2 * n) + (second ? n : 0);
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, part);
    const float scale = 1.0f / sqrtf(ss / float(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        yr[i] = w ? xr[i] * scale * w[i] : xr[i] * scale;
}

}

void rms_norm_pair(const float* x0, const float* w0, const float* x1, const float* w1,
                    float* y, int n, int rows, float eps, void* stream) {
    mtp_norm_pair_kernel<<<dim3(rows, 2), sum_threads(n), 0, cs(stream)>>>(x0, w0, x1, w1, y, n, eps);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("rms_norm_pair: ") + cudaGetErrorString(e));
}
