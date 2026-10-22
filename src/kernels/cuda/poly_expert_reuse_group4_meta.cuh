#pragma once

#if defined(__HIPCC__)
inline bool poly_expert_reuse_group4_meta_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_EXPERT_REUSE_GROUP4_META");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

__device__ __forceinline__ bool poly_expert_group4_metadata(
        const int32_t* metadata, int j, int& count, int (&entries)[4]) {
    if (!metadata) return false;
    const int32_t* record = metadata + size_t(j) * 4;
    count = record[0];
    entries[0] = j;
#pragma unroll
    for (int c = 1; c < 4; ++c) entries[c] = c < count ? record[c] : j;
    return true;
}
#endif
