#pragma once

#include "poly_expert_reuse_group4_meta.cuh"

#if defined(__HIPCC__)
inline bool poly_expert_reuse_group4_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_EXPERT_REUSE_GROUP4");
        return (v && v[0] == '1' && v[1] == '\0') || poly_expert_reuse_group4_meta_on();
    }();
    return on;
}

struct PolyExpertGroup4 {
    int count;
    int entry[4];
};

__device__ __forceinline__ PolyExpertGroup4 poly_expert_group4_find(
        const int32_t* ids, int count, int j, int expert) {
    const int lane = int(threadIdx.x);
    PolyExpertGroup4 group{1, {j, -1, -1, -1}};
    int prior = 0;
    for (int base = 0; base < count; base += 32) {
        const int index = base + lane;
        const uint32_t matches = uint32_t(__ballot_sync(
            0xffffffffu, index < count && ids[index] == expert));
        const int at = j - base;
        uint32_t before = 0, later = 0;
        if (at >= 32) before = 0xffffffffu;
        else if (at < 0) later = 0xffffffffu;
        else {
            before = at == 0 ? 0u : (1u << at) - 1u;
            later = at == 31 ? 0u : ~((1u << (at + 1)) - 1u);
        }
        prior += __popc(matches & before);
        if (at >= 0 && at < 32 && (prior & 3)) {
            group.count = 0;
            return group;
        }
        uint32_t next = matches & later;
        while (next && group.count < 4) {
            const int bit = __ffs(next) - 1;
            if (group.count == 1) group.entry[1] = base + bit;
            else if (group.count == 2) group.entry[2] = base + bit;
            else group.entry[3] = base + bit;
            ++group.count;
            next &= next - 1u;
        }
        if (group.count == 4) break;
    }
    return group;
}

template<typename F, bool PARTIAL, int ROWS>
__global__ void __launch_bounds__(128) poly_expert_reuse_group4_kernel(
        MmvqIds slots, int count, int n_in, int n_out) {
    const int j = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    const int row0 = 4 * ROWS * int(blockIdx.x) + ROWS * int(threadIdx.y);
    if (row0 >= n_out) return;
    const int expert = slots.ids[j];
    PolyExpertGroup4 group;
    if (!poly_expert_group4_metadata(slots.reuse_group4, j, group.count, group.entry))
        group = poly_expert_group4_find(slots.ids, count, j, expert);
    if (group.count == 0) return;
    if constexpr (PARTIAL) {
        if (slots.table[expert] == nullptr) {
            if (lane == 0) {
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    if (c < group.count) {
#pragma unroll
                        for (int r = 0; r < ROWS; ++r)
                            if (row0 + r < n_out)
                                slots.y[size_t(group.entry[c]) * slots.y_stride + row0 + r] = 0.f;
                    }
                }
            }
            return;
        }
    }
    const SlotPtrs at = slot_ptrs(slots, j);
    const auto* w = static_cast<const typename F::Block*>(at.w);
    const Q81Block* x[4];
    float* y[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const SlotPtrs column = slot_ptrs(slots, c < group.count ? group.entry[c] : j);
        x[c] = static_cast<const Q81Block*>(column.x);
        y[c] = column.y;
    }
    const int nblocks = n_in / 32;
    float sums[4][ROWS] = {};
    for (int k = lane; k < 2 * nblocks; k += 32) {
        const int kb = k >> 1, iqs = 2 * (k & 1);
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (row0 + r < n_out) {
                const auto weight = F::load(w + size_t(row0 + r) * nblocks + kb, iqs);
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    if (c < group.count) sums[c][r] += F::apply(weight, x[c] + kb, iqs);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        if (c < group.count) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                sums[c][r] = warp_sum(sums[c][r]);
                if (lane == 0 && row0 + r < n_out) y[c][row0 + r] = sums[c][r];
            }
        }
    }
}

template<typename Weight, bool PARTIAL>
inline void poly_expert_reuse_group4(const MmvqIds& slots, int count, int n_in,
                                     int n_out, int rows, cudaStream_t stream) {
    using F = SmallTraits<Weight, 4>;
    const dim3 threads(32, 4);
    if (rows == 4) {
        const dim3 blocks(unsigned((n_out + 15) / 16), unsigned(count));
        poly_expert_reuse_group4_kernel<F, PARTIAL, 4><<<blocks, threads, 0, stream>>>(slots, count, n_in, n_out);
    } else if (rows == 2) {
        const dim3 blocks(unsigned((n_out + 7) / 8), unsigned(count));
        poly_expert_reuse_group4_kernel<F, PARTIAL, 2><<<blocks, threads, 0, stream>>>(slots, count, n_in, n_out);
    } else {
        const dim3 blocks(unsigned((n_out + 3) / 4), unsigned(count));
        poly_expert_reuse_group4_kernel<F, PARTIAL, 1><<<blocks, threads, 0, stream>>>(slots, count, n_in, n_out);
    }
}
#endif
