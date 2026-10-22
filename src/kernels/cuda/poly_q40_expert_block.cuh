#pragma once

#include "poly_q40_expert_weights.cuh"

#if defined(__HIPCC__)
inline bool poly_q40_expert_block_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_Q40_EXPERT_BLOCK");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

template<typename Weight, typename S, bool PARTIAL, int ROWS, bool VECTOR = false>
__global__ void __launch_bounds__(128) poly_q40_expert_block_kernel(
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
    for (int kb = lane; kb < nblocks; kb += 32) {
        const Q81Block& xb = x[kb];
        PolyExpertQ4Codes codes[ROWS];
        if constexpr (VECTOR) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) codes[r] = poly_expert_q4_codes(wr[r][kb].qs);
        }
        int dots[ROWS] = {};
        int sumu = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int qlo = reinterpret_cast<const int*>(xb.qs)[i];
            const int qhi = reinterpret_cast<const int*>(xb.qs)[i + 4];
            if constexpr (std::is_same_v<Weight, Q41Block>)
                sumu = STRATA_DP4A(0x01010101, qhi, STRATA_DP4A(0x01010101, qlo, sumu));
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                int a;
                if constexpr (VECTOR) a = codes[r].word[i];
                else a = load_int_b2(wr[r][kb].qs, i);
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
                sums[r] += float(wr[r][kb].d) * (dots[r] * ds.x - 8 * ds.y);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        sums[r] = warp_sum(sums[r]);
        if (lane == 0 && row0 + r < n_out) at.y[row0 + r] = sums[r];
    }
}

template<typename Weight, typename S, bool PARTIAL, bool VECTOR>
inline void poly_q40_expert_block_slots_impl(const S& slots, int count,
                                        int n_in, int n_out, int rows,
                                        cudaStream_t stream) {
    const dim3 threads(32, 4);
    if (rows == 4) {
        const dim3 blocks(unsigned((n_out + 15) / 16), unsigned(count));
        poly_q40_expert_block_kernel<Weight, S, PARTIAL, 4, VECTOR><<<blocks, threads, 0, stream>>>(slots, n_in, n_out);
    } else if (rows == 2) {
        const dim3 blocks(unsigned((n_out + 7) / 8), unsigned(count));
        poly_q40_expert_block_kernel<Weight, S, PARTIAL, 2, VECTOR><<<blocks, threads, 0, stream>>>(slots, n_in, n_out);
    } else {
        const dim3 blocks(unsigned((n_out + 3) / 4), unsigned(count));
        poly_q40_expert_block_kernel<Weight, S, PARTIAL, 1, VECTOR><<<blocks, threads, 0, stream>>>(slots, n_in, n_out);
    }
}
template<typename Weight, typename S, bool PARTIAL>
inline void poly_q40_expert_block_slots(const S& slots, int count,
                                        int n_in, int n_out, int rows,
                                        cudaStream_t stream) {
    if (poly_q40_expert_weight_vector_on())
        poly_q40_expert_block_slots_impl<Weight, S, PARTIAL, true>(slots, count, n_in, n_out, rows, stream);
    else
        poly_q40_expert_block_slots_impl<Weight, S, PARTIAL, false>(slots, count, n_in, n_out, rows, stream);
}
#endif
