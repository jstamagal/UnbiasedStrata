#pragma once

inline bool poly_head_aligned_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_HEAD_ALIGNED");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

template<bool ALIGNED>
__device__ __forceinline__ Q6KTraits::W poly_head_q6k_load(const Q6KBlock* w, int lane, int kb) {
    if constexpr (ALIGNED) {
        // Twelve 210-byte blocks make each head row dword-aligned. Even
        // blocks can load each word directly; odd blocks retain short loads.
        if ((kb & 1) == 0) {
            Q6KTraits::W r;
            r.bq8_offset = 4 * (lane / 16) + (lane % 16) / 8;
            const int scale_offset = 8 * (lane / 16) + (lane % 16) / 4;
            const int vh_shift = 2 * ((lane % 16) / 8);
            r.vl = reinterpret_cast<const int*>(w->ql)[lane];
            r.vh = reinterpret_cast<const int*>(w->qh)[8 * (lane / 16) + lane % 8] >> vh_shift;
            r.scales = w->scales + scale_offset;
            r.d = w->d;
            return r;
        }
    }
    return Q6KTraits::load(w, lane);
}
