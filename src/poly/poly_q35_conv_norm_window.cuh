// Short-window Qwen3.5 GDN convolution and Q/K norm, in one launch.
// Include from qwen35_fuse.cu after CUDA headers.
#pragma once

namespace strata::poly {
namespace {

__global__ void poly_q35_conv_norm_window_kernel(const float* __restrict__ x,
                                                 float* __restrict__ state,
                                                 const float* __restrict__ w,
                                                 float* __restrict__ out,
                                                 int channels, int key_heads, int rows,
                                                 float eps, float* __restrict__ snap,
                                                 size_t snap_stride, unsigned snaps) {
    constexpr int dim = 128;
    __shared__ float part[dim / 32];
    const int group = blockIdx.x, lane = threadIdx.x, c = group * dim + lane;
    const float* wc = w + size_t(c) * 4;
    float* st = state + size_t(c) * 3;
    float s0 = st[0], s1 = st[1], s2 = st[2];
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    const bool query = group < key_heads;
    const bool key = group >= key_heads && group < 2 * key_heads;
    for (int row = 0; row < rows; ++row) {
        const size_t at = size_t(row) * channels + c;
        const float v = x[at];
        const float sum = s0 * w0 + s1 * w1 + s2 * w2 + v * w3;
        s0 = s1;
        s1 = s2;
        s2 = v;
        float y = sum * (1.f / (1.f + expf(-sum)));
        if (query || key) {
            float sq = y * y;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                sq += __shfl_xor_sync(0xffffffffu, sq, offset);
            if ((lane & 31) == 0) part[lane >> 5] = sq;
            __syncthreads();
            float ss = 0.f;
#pragma unroll
            for (int k = 0; k < dim / 32; ++k) ss += part[k];
            y = y / sqrtf(ss + eps);
            if (query) y *= 1.f / sqrtf(float(dim));
            __syncthreads();
        }
        out[at] = y;
        if (snap && row < 32 && ((snaps >> row) & 1u)) {
            float* dst = snap + size_t(__popc(snaps & ((1u << row) - 1u))) * snap_stride + size_t(c) * 3;
            dst[0] = s0;
            dst[1] = s1;
            dst[2] = s2;
        }
    }
    st[0] = s0;
    st[1] = s1;
    st[2] = s2;
}

} // namespace

void qwen35_conv_norm_window(const float* x, float* state, const float* w, float* out,
                                    int channels, int key_heads, int rows, float eps,
                                    float* snap, size_t snap_stride, unsigned snaps, void* stream) {
    poly_q35_conv_norm_window_kernel<<<channels / 128, 128, 0, (cudaStream_t) stream>>>(
        x, state, w, out, channels, key_heads, rows, eps, snap, snap_stride, snaps);
}

} // namespace strata::poly
