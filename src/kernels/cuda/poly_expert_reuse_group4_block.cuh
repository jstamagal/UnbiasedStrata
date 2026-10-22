#pragma once

#include "poly_q40_expert_weights.cuh"

#if defined(__HIPCC__)
inline bool poly_expert_reuse_group4_block_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_EXPERT_REUSE_GROUP4_BLOCK");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

template<typename Weight, int ROWS, bool VECTOR>
__device__ __forceinline__ void poly_expert_reuse_group4_block_accumulate(
        const Weight* w, const Q81Block* const (&x)[4], int count,
        int row0, int n_out, int nblocks, float (&sums)[4][ROWS]) {
    const int lane = int(threadIdx.x);
    const Weight* wr[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
        wr[r] = w + size_t(row0 + r < n_out ? row0 + r : row0) * nblocks;
    for (int kb = lane; kb < nblocks; kb += 32) {
        PolyExpertQ4Codes codes[ROWS];
        if constexpr (VECTOR) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) codes[r] = poly_expert_q4_codes(wr[r][kb].qs);
        }
        int dots[4][ROWS] = {};
        int sumu[4] = {};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int words[ROWS];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if constexpr (VECTOR) words[r] = codes[r].word[i];
                else words[r] = load_int_b2(wr[r][kb].qs, i);
            }
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                if (c < count) {
                    const int qlo = reinterpret_cast<const int*>(x[c][kb].qs)[i];
                    const int qhi = reinterpret_cast<const int*>(x[c][kb].qs)[i + 4];
                    if constexpr (std::is_same_v<Weight, Q41Block>)
                        sumu[c] = STRATA_DP4A(0x01010101, qhi,
                            STRATA_DP4A(0x01010101, qlo, sumu[c]));
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        dots[c][r] = STRATA_DP4A(words[r] & 0x0f0f0f0f, qlo, dots[c][r]);
                        dots[c][r] = STRATA_DP4A((words[r] >> 4) & 0x0f0f0f0f, qhi, dots[c][r]);
                    }
                }
            }
        }
        float2 ds[4];
#pragma unroll
        for (int c = 0; c < 4; ++c)
            if (c < count) ds[c] = __half22float2(x[c][kb].ds);
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if constexpr (std::is_same_v<Weight, Q41Block>) {
                const float2 dm = __half22float2(wr[r][kb].dm);
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    if (c < count)
                        sums[c][r] += float(dots[c][r]) * (dm.x * ds[c].x)
                                    + float(sumu[c]) * (dm.y * ds[c].x);
            } else {
                const float d = float(wr[r][kb].d);
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    if (c < count)
                        sums[c][r] += d * (dots[c][r] * ds[c].x - 8 * ds[c].y);
            }
        }
    }
}
#endif
