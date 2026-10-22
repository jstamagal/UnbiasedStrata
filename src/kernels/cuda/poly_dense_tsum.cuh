#pragma once
#if defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
inline bool poly_dense_tsum_on(int ncols) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_DENSE_TSUM");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    static const bool wide = [] {
        const char* v = std::getenv("STRATA_POLY_DENSE_TSUM_WIDE");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return ncols <= 4 ? on : wide;
}

template<int NC, int ROWS>
__device__ __forceinline__ bool poly_dense_tsum(float (&sum)[4][NC][ROWS], float* y,
                                               int n_out, int row0, int lane, bool enabled) {
    constexpr int VALUES = NC * ROWS;
    if constexpr (NC < 2 || NC > 8 || VALUES > 32) {
        return false;
    } else {
        if (!enabled) return false;
        constexpr int P = s26ts::pow2_ceil(VALUES);
        float values[P];
#pragma unroll
        for (int i = 0; i < VALUES; ++i) {
            float value = sum[0][i / ROWS][i % ROWS];
#pragma unroll
            for (int group = 1; group < 4; ++group) value += sum[group][i / ROWS][i % ROWS];
            values[i] = value;
        }
#pragma unroll
        for (int i = VALUES; i < P; ++i) values[i] = 0.f;
        const float result = s26ts::tsum<P>(values, lane);
        const int output = s26ts::tsum_token<P>(lane);
        if (output < VALUES && lane == s26ts::tsum_lane<P>(output)) {
            const int row = row0 + output % ROWS;
            if (row < n_out) y[size_t(output / ROWS) * n_out + row] = result;
        }
        return true;
    }
}
#endif
