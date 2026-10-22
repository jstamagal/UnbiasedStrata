#pragma once

inline int expert_activation_q8_scalar_threads() {
    static const int threads = [] {
        const char* v = std::getenv("STRATA_POLY_EXPERT_ACT_Q8_SCALAR_THREADS");
        if (!v) return 256;
        char* end = nullptr;
        const long n = std::strtol(v, &end, 10);
        return end != v && *end == '\0' && (n == 64 || n == 128 || n == 256)
                   ? int(n) : 256;
    }();
    return threads;
}
