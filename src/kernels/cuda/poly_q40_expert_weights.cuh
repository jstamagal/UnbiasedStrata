#pragma once

#if defined(__HIPCC__)
inline bool poly_q40_expert_weight_vector_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_Q40_EXPERT_WEIGHT_VEC");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

struct PolyExpertQ4Codes { int word[4]; };

__device__ __forceinline__ PolyExpertQ4Codes poly_expert_q4_codes(const uint8_t* p) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(p);
    const uintptr_t shift = address & 2;
    const auto* words = reinterpret_cast<const uint32_t*>(address - shift);
    const uint32_t a = words[0], b = words[1], c = words[2], d = words[3];
    PolyExpertQ4Codes result;
    if (shift) {
        // The preceding halfword is the scale in this same Q4_0 block.
        // Read its final halfword separately to remain within the block.
        const uint32_t last = *reinterpret_cast<const uint16_t*>(p + 14);
        result.word[0] = int((a >> 16) | (b << 16));
        result.word[1] = int((b >> 16) | (c << 16));
        result.word[2] = int((c >> 16) | (d << 16));
        result.word[3] = int((d >> 16) | (last << 16));
    } else {
        result.word[0] = int(a);
        result.word[1] = int(b);
        result.word[2] = int(c);
        result.word[3] = int(d);
    }
    return result;
}
#endif
