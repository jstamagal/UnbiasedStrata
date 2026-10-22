#pragma once
#if defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
inline bool poly_dense_xscale_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_DENSE_X_SCALE");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

template<int T>
__device__ __forceinline__ half2 poly_dense_xscale(const Q81Block* x, bool broadcast) {
    static_assert(T == 2 || T == 4);
    if (!broadcast) return x->ds;
    const int lane = int(threadIdx.x);
    uint32_t packed = 0;
    if ((lane & (T - 1)) == 0) __builtin_memcpy(&packed, &x->ds, sizeof(packed));
    packed = __shfl(packed, lane & ~(T - 1), 32);
    half2 scale;
    __builtin_memcpy(&scale, &packed, sizeof(packed));
    return scale;
}
#endif
