#pragma once
#if defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
#include "poly_dense_wscale.cuh"
inline bool poly_dense_weight_vector_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_DENSE_WEIGHT_VEC");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

inline int poly_dense_weight_mode() {
    return int(poly_dense_weight_vector_on()) | (int(poly_dense_wscale_on()) << 1);
}

__device__ __forceinline__ int2 poly_dense_weight_pair(const uint8_t* p) {
    int2 result;
    const uintptr_t address = reinterpret_cast<uintptr_t>(p);
    const uintptr_t misalignment = address & 2;
    const auto* words = reinterpret_cast<const uint32_t*>(address - misalignment);
    const uint32_t first = words[0], middle = words[1];
    result.x = int(first);
    result.y = int(middle);
    if (misalignment) {
        // Include the preceding halfword, which belongs to this weight block.
        // The last halfword remains a 16-bit load to avoid reading past it.
        const uint32_t last = *reinterpret_cast<const uint16_t*>(p + 6);
        result.x = int((first >> 16) | (middle << 16));
        result.y = int((middle >> 16) | (last << 16));
    }
    return result;
}

template<typename F>
__device__ __forceinline__ typename F::W poly_dense_weight_load(
        const typename F::Block* w, int iqs, int weight_mode) {
    if (weight_mode == 0) return F::load(w, iqs);
    int2 codes;
    if (weight_mode & 1) {
        codes = poly_dense_weight_pair(reinterpret_cast<const uint8_t*>(w->qs) + 4 * iqs);
    } else {
        codes.x = load_int_b2(w->qs, iqs);
        codes.y = load_int_b2(w->qs, iqs + 1);
    }
    const float scale = poly_dense_wscale<F>(w, (weight_mode & 2) != 0);
    typename F::W result;
    if constexpr (std::is_same_v<F, SmallTraits<Q40Block, 4>>) {
        result.vi0[0] = codes.x & 0x0f0f0f0f;
        result.vi1[0] = (codes.x >> 4) & 0x0f0f0f0f;
        result.vi0[1] = codes.y & 0x0f0f0f0f;
        result.vi1[1] = (codes.y >> 4) & 0x0f0f0f0f;
        result.d = scale;
    } else {
        result.v[0] = codes.x;
        result.v[1] = codes.y;
        result.d0 = scale;
    }
    return result;
}
#endif
