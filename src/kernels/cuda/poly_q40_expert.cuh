// Optional Q4_0 expert decode layout for RDNA2 wave32. Included by native_mmvq.cu
// after Q40Block, Q81Block, MmvqSlots/MmvqIds, and slot_ptrs are defined.
#pragma once

#if defined(__HIPCC__)
template<typename Weight, typename S, bool PARTIAL = false>
__global__ void __launch_bounds__(128) poly_q40_expert_wave_kernel(
        S slots, int n_in, int n_out) {
    const int j = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    const int row = 4 * int(blockIdx.x) + int(threadIdx.y);
    if constexpr (PARTIAL) {
        if (slots.table[slots.ids[j]] == nullptr) {
            if (lane == 0 && row < n_out) slots.y[size_t(j) * slots.y_stride + row] = 0.f;
            return;
        }
    }
    if (row >= n_out) return;
    const SlotPtrs at = slot_ptrs(slots, j);
    const auto* w = static_cast<const Weight*>(at.w);
    const auto* x = static_cast<const Q81Block*>(at.x);

    // A wave owns one output row. The two 16-value parts of a Q4_0 block
    // retain small_q8_dot's affine correction from the Q8_1 input sum.
    const int nblocks = n_in / 32;
    const Weight* wr = w + size_t(row) * nblocks;
    float sum = 0.0f;
    for (int k = lane; k < 2 * nblocks; k += 32) {
        const int kb = k >> 1;
        const int iqs = (k & 1) * 2;
        // Q4_1: d*q + m; the min term uses the sum of Q8_1's quantized
        // int8 codes, as the Q5_1 path does. Q4_0 keeps its centered sum.
        sum += small_q8_dot(wr + kb, x + kb, iqs);
    }
    sum = warp_sum(sum);
    if (lane == 0) at.y[row] = sum;
}

// Two rows share the same Q8_1 words and scales. This is especially useful
// for the 1024-wide down matrix: each wave reads two complete output rows.
template<typename Weight, typename S, bool PARTIAL = false>
__global__ void __launch_bounds__(128) poly_q40_expert_pair_kernel(
        S slots, int n_in, int n_out) {
    const int j = int(blockIdx.y);
    const int lane = int(threadIdx.x);
    const int row0 = 8 * int(blockIdx.x) + 2 * int(threadIdx.y);
    const int row1 = row0 + 1;
    if constexpr (PARTIAL) {
        if (slots.table[slots.ids[j]] == nullptr) {
            if (lane == 0 && row0 < n_out) {
                float* y = slots.y + size_t(j) * slots.y_stride;
                y[row0] = 0.f;
                if (row1 < n_out) y[row1] = 0.f;
            }
            return;
        }
    }
    if (row0 >= n_out) return;
    const SlotPtrs at = slot_ptrs(slots, j);
    const auto* w = static_cast<const Weight*>(at.w);
    const auto* x = static_cast<const Q81Block*>(at.x);
    const int nblocks = n_in / 32;
    const Weight* w0 = w + size_t(row0) * nblocks;
    const Weight* w1 = w + size_t(row1 < n_out ? row1 : row0) * nblocks;
    float sum0 = 0.0f, sum1 = 0.0f;
    for (int k = lane; k < 2 * nblocks; k += 32) {
        const int kb = k >> 1, iqs = (k & 1) * 2;
        const Q81Block& xb = x[kb];
        int dot0 = 0, dot1 = 0, sumu = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int qlo = reinterpret_cast<const int*>(xb.qs)[iqs + i];
            const int qhi = reinterpret_cast<const int*>(xb.qs)[iqs + i + 4];
            if constexpr (std::is_same_v<Weight, Q41Block>)
                sumu = STRATA_DP4A(0x01010101, qhi, STRATA_DP4A(0x01010101, qlo, sumu));
            const int a = load_int_b2(w0[kb].qs, iqs + i);
            const int b = load_int_b2(w1[kb].qs, iqs + i);
            dot0 = STRATA_DP4A(a & 0x0f0f0f0f, qlo, dot0);
            dot0 = STRATA_DP4A((a >> 4) & 0x0f0f0f0f, qhi, dot0);
            dot1 = STRATA_DP4A(b & 0x0f0f0f0f, qlo, dot1);
            dot1 = STRATA_DP4A((b >> 4) & 0x0f0f0f0f, qhi, dot1);
        }
        const float2 ds = __half22float2(xb.ds);
        if constexpr (std::is_same_v<Weight, Q41Block>) {
            const float2 dm0 = __half22float2(w0[kb].dm);
            const float2 dm1 = __half22float2(w1[kb].dm);
            sum0 += float(dot0) * (dm0.x * ds.x) + float(sumu) * (dm0.y * ds.x);
            sum1 += float(dot1) * (dm1.x * ds.x) + float(sumu) * (dm1.y * ds.x);
        } else {
            sum0 += float(w0[kb].d) * (dot0 * ds.x - 4 * ds.y);
            sum1 += float(w1[kb].d) * (dot1 * ds.x - 4 * ds.y);
        }
    }
    sum0 = warp_sum(sum0);
    sum1 = warp_sum(sum1);
    if (lane == 0) {
        at.y[row0] = sum0;
        if (row1 < n_out) at.y[row1] = sum1;
    }
}

inline bool poly_q40_expert_pair_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_Q40_EXPERT_PAIR");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

inline bool poly_q40_expert_enabled(int n_in, int n_out) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_Q40_EXPERT_WAVE");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return (on || poly_q40_expert_pair_on()) &&
           ((n_in == 3072 && n_out == 2048) ||
            (n_in == 1024 && n_out == 3072));
}

inline bool poly_q40_expert_ids_enabled(int n_in, int n_out) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_EXPERT_IDS");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on && poly_q40_expert_enabled(n_in, n_out);
}

template<typename Weight, typename S, bool PARTIAL = false>
inline void poly_q4_expert_slots(const S& slots, int count,
                                 int n_in, int n_out, cudaStream_t stream) {
    const dim3 threads(32, 4);
    if (poly_q40_expert_pair_on()) {
        const dim3 blocks(unsigned((n_out + 7) / 8), unsigned(count));
        poly_q40_expert_pair_kernel<Weight, S, PARTIAL><<<blocks, threads, 0, stream>>>(slots, n_in, n_out);
    } else {
        const dim3 blocks(unsigned((n_out + 3) / 4), unsigned(count));
        poly_q40_expert_wave_kernel<Weight, S, PARTIAL><<<blocks, threads, 0, stream>>>(slots, n_in, n_out);
    }
}
template<typename S>
inline void poly_q40_expert_slots(const S& slots, int count,
                                  int n_in, int n_out, cudaStream_t stream) {
    poly_q4_expert_slots<Q40Block>(slots, count, n_in, n_out, stream);
}
template<typename S>
inline void poly_q41_expert_slots(const S& slots, int count,
                                  int n_in, int n_out, cudaStream_t stream) {
    poly_q4_expert_slots<Q41Block>(slots, count, n_in, n_out, stream);
}
inline void poly_q40_expert_ids_partial(const MmvqIds& ids, int count,
                                         int n_in, int n_out, cudaStream_t stream) {
    poly_q4_expert_slots<Q40Block, MmvqIds, true>(ids, count, n_in, n_out, stream);
}
inline void poly_q41_expert_ids_partial(const MmvqIds& ids, int count,
                                         int n_in, int n_out, cudaStream_t stream) {
    poly_q4_expert_slots<Q41Block, MmvqIds, true>(ids, count, n_in, n_out, stream);
}
#else
inline bool poly_q40_expert_enabled(int, int) { return false; }
#endif
