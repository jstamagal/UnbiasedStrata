#include "qwen35_fuse.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include "poly_q35_conv_norm_window.cuh"

namespace strata::poly {
namespace {

__device__ __forceinline__ float q35_warp_sum(float x) {
#pragma unroll
    for (int step = 16; step; step >>= 1) x += __shfl_xor_sync(0xffffffff, x, step, 32);
    return x;
}

__device__ __forceinline__ float q35_warp_max(float x) {
#pragma unroll
    for (int step = 16; step; step >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, step, 32));
    return x;
}

struct Q35Q81 { half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q35Q81) == 36);

__global__ void attn_gate_q8_kernel(float* att, const float* gate, Q35Q81* q8, int heads, int dim) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= heads * dim) return;
    const int h = i / dim, d = i % dim;
    const float g = gate[h * 2 * dim + d];
    const float x = att[i] * (1.f / (1.f + expf(-g)));
    att[i] = x;
    const float amax = q35_warp_max(fabsf(x));
    const float sum = q35_warp_sum(x);
    const float scale = kernels::q8_1_finite(amax / 127.f);
    q8[i / 32].qs[i % 32] = kernels::q8_1_quant(x, scale, amax);
    if ((i & 31) == 0) q8[i / 32].ds = kernels::q8_1_ds(scale, sum);
}

template <bool Gates>
__global__ void alpha_beta_kernel(const float* aw, const float* bw, const float* x, const float* dt,
                                  const float* ssm_a, float* a, float* b, int n_in, int n_out) {
    __shared__ float part[32];
    const int out = blockIdx.x;
    const int row = blockIdx.y;
    const bool beta = blockIdx.z != 0;
    const float* w = (beta ? bw : aw) + (size_t) out * n_in;
    const float* xr = x + (size_t) row * n_in;
    float acc = 0.f;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) acc += w[i] * xr[i];
    acc = q35_warp_sum(acc);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
    __syncthreads();
    float sum = 0.f;
    for (int k = 0; k < blockDim.x / 32; ++k) sum += part[k];
    if (threadIdx.x == 0) {
        if constexpr (Gates) {
            if (beta) sum = 1.f / (1.f + expf(-sum));
            else {
                const float g = sum + dt[out];
                sum = expf((g > 20.0f ? g : logf(1.0f + expf(g))) * ssm_a[out]);
            }
        }
        (beta ? b : a)[(size_t) row * n_out + out] = sum;
    }
}

__global__ void kv_store_pair_kernel(const float* __restrict__ key, const float* __restrict__ value,
                                     int8_t* __restrict__ key_codes, half* __restrict__ key_scales,
                                     int8_t* __restrict__ value_codes, half* __restrict__ value_scales,
                                     int n, int rows, int nb, long long pos,
                                     const long long* __restrict__ at) {
    const int gb = blockIdx.x * blockDim.x + threadIdx.x;
    const int one = nb * rows;
    if (gb >= 2 * one) return;
    const bool is_value = gb >= one;
    const int b = is_value ? gb - one : gb;
    const int trow = b / nb, blk = b % nb;
    const long long row = (at ? *at : pos) + trow;
    const float* x = is_value ? value : key;
    int8_t* codes = is_value ? value_codes : key_codes;
    half* scales = is_value ? value_scales : key_scales;
    const float* xr = x + (size_t) trow * n + blk * 32;
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(xr[j]));
    const float d = amax / 127.0f;
    const float id = d > 0.f ? 1.0f / d : 0.0f;
    scales[(size_t) row * nb + blk] = __float2half(d);
    int8_t* cr = codes + (size_t) row * n + blk * 32;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        float q = rintf(xr[j] * id);
        q = fminf(fmaxf(q, -127.0f), 127.0f);
        cr[j] = (int8_t) q;
    }
}

__global__ void shared_gate_kernel(const float* w, const float* x, float* gate_out,
                                   float* expert_out, int n) {
    __shared__ float part[32];
    __shared__ float scale;
    const int row = blockIdx.x;
    const float* xr = x + (size_t) row * n;
    float* yr = expert_out + (size_t) row * n;
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += w[i] * xr[i];
    acc = q35_warp_sum(acc);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int k = 0; k < blockDim.x / 32; ++k) sum += part[k];
        gate_out[row] = sum;
        scale = 1.f / (1.f + expf(-sum));
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] *= scale;
}

__global__ void conv_norm1_kernel(const float* x, float* state, const float* w, float* out, int dim,
                                  int k_heads, float eps, float* snap) {
    __shared__ float part[8];
    const int group = blockIdx.x, j = threadIdx.x, c = group * dim + j;
    const float* wc = w + (size_t) c * 4;
    float* st = state + (size_t) c * 3;
    const float s0 = st[0], s1 = st[1], s2 = st[2], v = x[c];
    const float z = s0 * wc[0] + s1 * wc[1] + s2 * wc[2] + v * wc[3];
    st[0] = s1;
    st[1] = s2;
    st[2] = v;
    if (snap) {
        float* dst = snap + (size_t) c * 3;
        dst[0] = s1;
        dst[1] = s2;
        dst[2] = v;
    }
    float y = z * (1.f / (1.f + expf(-z)));
    if (group < 2 * k_heads) {
        float ss = q35_warp_sum(y * y);
        if ((j & 31) == 0) part[j >> 5] = ss;
        __syncthreads();
        ss = 0.f;
        for (int k = 0; k < dim / 32; ++k) ss += part[k];
        if (group < k_heads) y = y / sqrtf(ss + eps) * (1.f / sqrtf((float) dim));
        else y = y / sqrtf(ss + eps);
    }
    out[c] = y;
}

}

void qwen35_alpha_beta(const float* alpha_w, const float* beta_w, const float* x,
                       float* alpha, float* beta, int n_in, int n_out, int rows, int threads, void* stream) {
    alpha_beta_kernel<false><<<dim3(n_out, rows, 2), threads, 0, (cudaStream_t) stream>>>(
        alpha_w, beta_w, x, nullptr, nullptr, alpha, beta, n_in, n_out);
}

void qwen35_alpha_beta_gates(const float* alpha_w, const float* beta_w, const float* x,
                             const float* dt, const float* a, float* alpha, float* beta,
                             int n_in, int n_out, int rows, int threads, void* stream) {
    alpha_beta_kernel<true><<<dim3(n_out, rows, 2), threads, 0, (cudaStream_t) stream>>>(
        alpha_w, beta_w, x, dt, a, alpha, beta, n_in, n_out);
}

void qwen35_conv_norm1(const float* x, float* state, const float* w, float* out, int channels,
                       int dim, int k_heads, float eps, float* snap, void* stream) {
    conv_norm1_kernel<<<channels / dim, dim, 0, (cudaStream_t) stream>>>(
        x, state, w, out, dim, k_heads, eps, snap);
}

void qwen35_kv_store_pair(const float* key, const float* value,
                          int8_t* key_codes, uint16_t* key_scales,
                          int8_t* value_codes, uint16_t* value_scales,
                          int n, int rows, int64_t pos, const long long* at, void* stream) {
    if ((n & 31) != 0) throw std::invalid_argument("qwen35_kv_store_pair: n is not a multiple of 32");
    const int blocks = (n >> 5) * rows * 2;
    if (!blocks) return;
    kv_store_pair_kernel<<<(blocks + 255) / 256, 256, 0, (cudaStream_t) stream>>>(
        key, value, key_codes, (half*) key_scales, value_codes, (half*) value_scales,
        n, rows, n >> 5, (long long) pos, at);
}

void qwen35_shared_gate(const float* w, const float* x, float* gate, float* expert_out,
                        int n, int rows, int threads, void* stream) {
    shared_gate_kernel<<<rows, threads, 0, (cudaStream_t) stream>>>(w, x, gate, expert_out, n);
}

void qwen35_attn_gate_q8(float* att, const float* gate, void* q8, int heads, int dim, void* stream) {
    const int n = heads * dim;
    attn_gate_q8_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(
        att, gate, static_cast<Q35Q81*>(q8), heads, dim);
}

}
