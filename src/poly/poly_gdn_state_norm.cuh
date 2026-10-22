// GDN state update and gated output norm in one launch. Include inside
// namespace strata::poly after kernels.cu's device helpers.
#pragma once

namespace {

__global__ void poly_gdn_state_norm_kernel(const float* __restrict__ qkv,
                                           const float* __restrict__ decay,
                                           const float* __restrict__ share,
                                           float* __restrict__ state,
                                           const float* __restrict__ z,
                                           const float* __restrict__ gamma,
                                           float* __restrict__ out,
                                           int value_heads, int key_heads, int rows,
                                           float eps, float* __restrict__ snap,
                                           size_t snap_stride, unsigned snaps) {
    constexpr int dim = 128, warps = 16, tiles = dim / warps, per_lane = dim / 32;
    const int head = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int key_head = head % key_heads;
    const size_t row_width = size_t(2 * key_heads + value_heads) * dim;
    __shared__ float raw[dim];
    __shared__ float norm_part[dim / 32];
    float cell[tiles][per_lane];
#pragma unroll
    for (int tile = 0; tile < tiles; ++tile) {
        const size_t base = (size_t(head) * dim + tile * warps + warp) * dim;
#pragma unroll
        for (int k = 0; k < per_lane; ++k)
            cell[tile][k] = state[base + lane + k * 32];
    }

    for (int row = 0; row < rows; ++row) {
        const float* x = qkv + size_t(row) * row_width;
        const float dc = decay[size_t(row) * value_heads + head];
        const float sh = share[size_t(row) * value_heads + head];
        float qa[per_lane], ka[per_lane];
#pragma unroll
        for (int k = 0; k < per_lane; ++k) {
            qa[k] = x[size_t(key_head) * dim + lane + k * 32];
            ka[k] = x[size_t(key_heads + key_head) * dim + lane + k * 32];
        }
        float* snapshot = nullptr;
        if (snap && row < 32 && ((snaps >> row) & 1u))
            snapshot = snap + size_t(__popc(snaps & ((1u << row) - 1u))) * snap_stride;
#pragma unroll
        for (int tile = 0; tile < tiles; ++tile) {
            const int col = tile * warps + warp;
            const size_t base = (size_t(head) * dim + col) * dim;
            float sk = 0.f;
#pragma unroll
            for (int k = 0; k < per_lane; ++k)
                sk = __fmaf_rn(__fmul_rn(dc, cell[tile][k]), ka[k], sk);
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                sk += __shfl_xor_sync(0xffffffffu, sk, offset);
            const float value = x[size_t(2 * key_heads + head) * dim + col];
            const float delta = __fmul_rn(__fsub_rn(value, sk), sh);
            float o = 0.f;
#pragma unroll
            for (int k = 0; k < per_lane; ++k) {
                const int i = lane + k * 32;
                const float next = __fmaf_rn(dc, cell[tile][k], __fmul_rn(ka[k], delta));
                cell[tile][k] = next;
                if (snapshot) snapshot[base + i] = next;
                o = __fmaf_rn(next, qa[k], o);
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                o += __shfl_xor_sync(0xffffffffu, o, offset);
            if (lane == 0) raw[col] = o;
        }
        __syncthreads();
        if (tid < dim) {
            const float o = raw[tid];
            float sq = o * o;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                sq += __shfl_xor_sync(0xffffffffu, sq, offset);
            if ((tid & 31) == 0) norm_part[tid >> 5] = sq;
        }
        __syncthreads();
        if (tid < dim) {
            const float os = norm_part[0] + norm_part[1] + norm_part[2] + norm_part[3];
            const size_t at = (size_t(row) * value_heads + head) * dim + tid;
            const float gate = z[at];
            const float o = raw[tid];
            out[at] = o / sqrtf(os / float(dim) + eps) * gamma[tid] *
                      (gate * (1.f / (1.f + expf(-gate))));
        }
        __syncthreads();
    }
#pragma unroll
    for (int tile = 0; tile < tiles; ++tile) {
        const size_t base = (size_t(head) * dim + tile * warps + warp) * dim;
#pragma unroll
        for (int k = 0; k < per_lane; ++k)
            state[base + lane + k * 32] = cell[tile][k];
    }
}

inline void poly_gdn_state_norm(const float* qkv, const float* decay, const float* share,
                                float* state, const float* z, const float* gamma, float* out,
                                int value_heads, int key_heads, int rows, float eps,
                                float* snap, size_t snap_stride, unsigned snaps, void* stream) {
    poly_gdn_state_norm_kernel<<<value_heads, 512, 0, (cudaStream_t) stream>>>(
        qkv, decay, share, state, z, gamma, out, value_heads, key_heads, rows,
        eps, snap, snap_stride, snaps);
}

} // namespace
