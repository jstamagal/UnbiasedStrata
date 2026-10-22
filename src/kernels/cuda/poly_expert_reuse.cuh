#pragma once

#if defined(__HIPCC__)
inline bool poly_expert_reuse_pair_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_EXPERT_REUSE_PAIR");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

template<typename F, bool PARTIAL, int ROWS>
__global__ void __launch_bounds__(128) poly_expert_reuse_pair_kernel(
        MmvqIds slots, int count, int n_in, int n_out) {
    const int j = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    const int row0 = 4 * ROWS * int(blockIdx.x) + ROWS * int(threadIdx.y);
    if (row0 >= n_out) return;
    const int expert = slots.ids[j];
    int prior = 0;
    for (int k = 0; k < j; ++k) prior += slots.ids[k] == expert;
    if (prior & 1) return;
    int partner = -1;
    for (int k = j + 1; k < count; ++k) {
        if (slots.ids[k] == expert) {
            partner = k;
            break;
        }
    }
    if constexpr (PARTIAL) {
        if (slots.table[expert] == nullptr) {
            if (lane == 0) {
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    if (row0 + r < n_out) {
                        slots.y[size_t(j) * slots.y_stride + row0 + r] = 0.f;
                        if (partner >= 0)
                            slots.y[size_t(partner) * slots.y_stride + row0 + r] = 0.f;
                    }
                }
            }
            return;
        }
    }
    const SlotPtrs at = slot_ptrs(slots, j);
    const auto* w = static_cast<const typename F::Block*>(at.w);
    const auto* x0 = static_cast<const Q81Block*>(at.x);
    const SlotPtrs other = partner >= 0 ? slot_ptrs(slots, partner) : at;
    const auto* x1 = static_cast<const Q81Block*>(other.x);
    const int nblocks = n_in / 32;
    float sum0[ROWS] = {}, sum1[ROWS] = {};
    for (int k = lane; k < 2 * nblocks; k += 32) {
        const int kb = k >> 1, iqs = 2 * (k & 1);
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (row0 + r < n_out) {
                const auto weight = F::load(w + size_t(row0 + r) * nblocks + kb, iqs);
                sum0[r] += F::apply(weight, x0 + kb, iqs);
                if (partner >= 0) sum1[r] += F::apply(weight, x1 + kb, iqs);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        sum0[r] = warp_sum(sum0[r]);
        if (lane == 0 && row0 + r < n_out) at.y[row0 + r] = sum0[r];
        if (partner >= 0) {
            sum1[r] = warp_sum(sum1[r]);
            if (lane == 0 && row0 + r < n_out) other.y[row0 + r] = sum1[r];
        }
    }
}

template<typename Weight, bool PARTIAL>
inline void poly_expert_reuse_pair(const MmvqIds& slots, int count, int n_in,
                                   int n_out, int rows, cudaStream_t stream) {
    using F = SmallTraits<Weight, 4>;
    const dim3 threads(32, 4);
    if (rows == 4) {
        const dim3 blocks(unsigned((n_out + 15) / 16), unsigned(count));
        poly_expert_reuse_pair_kernel<F, PARTIAL, 4><<<blocks, threads, 0, stream>>>(slots, count, n_in, n_out);
    } else if (rows == 2) {
        const dim3 blocks(unsigned((n_out + 7) / 8), unsigned(count));
        poly_expert_reuse_pair_kernel<F, PARTIAL, 2><<<blocks, threads, 0, stream>>>(slots, count, n_in, n_out);
    } else {
        const dim3 blocks(unsigned((n_out + 3) / 4), unsigned(count));
        poly_expert_reuse_pair_kernel<F, PARTIAL, 1><<<blocks, threads, 0, stream>>>(slots, count, n_in, n_out);
    }
}
#endif
