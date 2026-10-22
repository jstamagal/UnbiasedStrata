#pragma once
#if defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
inline bool poly_dense_wscale_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_DENSE_W_SCALE");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

template<typename F>
__device__ __forceinline__ float poly_dense_wscale(const typename F::Block* w, bool broadcast) {
    if (!broadcast) return __half2float(w->d);
    const int lane = int(threadIdx.x);
    unsigned bits = 0;
    if ((lane & (F::T - 1)) == 0) bits = __half_as_ushort(w->d);
    bits = __shfl(bits, lane & ~(F::T - 1), 32);
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}
#endif
