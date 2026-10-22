#pragma once

#if defined(__HIPCC__)
inline bool poly_expert_reuse_ballot_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_EXPERT_REUSE_BALLOT");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

struct PolyExpertPairMatch {
    int prior;
    int partner;
};

__device__ __forceinline__ PolyExpertPairMatch poly_expert_pair_ballot(
        const int32_t* ids, int count, int j, int expert) {
    const int lane = int(threadIdx.x);
    PolyExpertPairMatch result{0, -1};
    for (int base = 0; base < count; base += 32) {
        const int index = base + lane;
        const uint32_t matches = uint32_t(__ballot_sync(
            0xffffffffu, index < count && ids[index] == expert));
        const int at = j - base;
        uint32_t before = 0, later = 0;
        if (at >= 32) {
            before = 0xffffffffu;
        } else if (at < 0) {
            later = 0xffffffffu;
        } else {
            before = at == 0 ? 0u : (1u << at) - 1u;
            later = at == 31 ? 0u : ~((1u << (at + 1)) - 1u);
        }
        result.prior += __popc(matches & before);
        if (at >= 0 && at < 32 && (result.prior & 1)) return result;
        const uint32_t next = matches & later;
        if (next) {
            result.partner = base + __ffs(next) - 1;
            break;
        }
    }
    return result;
}
#endif
