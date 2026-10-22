// Optional Linux placement for PolyStrata's host, HIP runtime, expert pool and helpers.
#pragma once

#include "strata/platform/aux_cpus.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strata::poly::cpu_placement {

#if defined(__linux__)
struct State {
    std::mutex mu;
    bool on = false;
    int host = -1, hip = -1, hip_tid = -1;
    std::vector<int> pool, aux;
    std::set<int> owned;
    std::atomic<bool> stop{false};
    std::thread sweeper;
    ~State() {
        stop.store(true, std::memory_order_release);
        if (sweeper.joinable()) sweeper.join();
    }
};

inline State& state() { static State s; return s; }
inline int tid() { return (int) syscall(SYS_gettid); }

inline bool place(int id, const std::vector<int>& cpus) {
    if (cpus.empty()) return false;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int cpu : cpus) CPU_SET(cpu, &mask);
    return sched_setaffinity(id, sizeof(mask), &mask) == 0;
}

inline std::vector<int> tasks() {
    std::vector<int> out;
    if (DIR* dir = opendir("/proc/self/task")) {
        while (const dirent* e = readdir(dir))
            if (e->d_name[0] >= '0' && e->d_name[0] <= '9') out.push_back(std::atoi(e->d_name));
        closedir(dir);
    }
    std::sort(out.begin(), out.end());
    return out;
}

inline bool hip_name(int id) {
    char path[64], name[64] = {};
    std::snprintf(path, sizeof(path), "/proc/self/task/%d/comm", id);
    FILE* file = std::fopen(path, "r");
    if (!file) return false;
    const bool read = std::fgets(name, sizeof(name), file) != nullptr;
    std::fclose(file);
    if (!read) return false;
    for (char& c : name) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    return std::strstr(name, "hip") || std::strstr(name, "hsa") ||
           std::strstr(name, "roc") || std::strstr(name, "kfd");
}

// Explicitly owned threads are never moved by the proc sweep. Of the others,
// identify the HIP runtime by its comm where possible, then by creation order.
inline void sweep() {
    State& s = state();
    if (!s.on) return;
    const std::vector<int> all = tasks();
    std::lock_guard<std::mutex> guard(s.mu);
    for (auto it = s.owned.begin(); it != s.owned.end(); )
        if (!std::binary_search(all.begin(), all.end(), *it)) it = s.owned.erase(it);
        else ++it;
    if (s.hip_tid > 0 && (!std::binary_search(all.begin(), all.end(), s.hip_tid) || s.owned.count(s.hip_tid)))
        s.hip_tid = -1;
    // A runtime thread may get its comm after creation; upgrade a provisional
    // creation-order choice when a named thread becomes visible.
    if (s.hip_tid < 0 || !hip_name(s.hip_tid)) {
        for (int id : all)
            if (!s.owned.count(id) && hip_name(id)) { s.hip_tid = id; break; }
        if (s.hip_tid < 0)
            for (int id : all)
                if (!s.owned.count(id)) { s.hip_tid = id; break; }
    }
    for (int id : all) {
        if (s.owned.count(id)) continue;
        if (id == s.hip_tid) place(id, {s.hip});
        else place(id, s.aux);
    }
}

inline bool configure(const char* spec, std::string& error) {
    if (!spec || !*spec) return true;
    State& s = state();
    std::string value(spec);
    std::vector<std::pair<std::string, std::string>> fields;
    size_t start = 0;
    while (start < value.size()) {
        const size_t end = value.find(',', start);
        const std::string part = value.substr(start, end == std::string::npos ? end : end - start);
        const size_t eq = part.find('=');
        if (eq != std::string::npos) {
            fields.emplace_back(part.substr(0, eq), part.substr(eq + 1));
        } else if (!fields.empty()) {
            fields.back().second += "," + part;
        } else {
            error = "STRATA_POLY_CPUS: expected host=,hip=,pool=,aux=";
            return false;
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    std::vector<int> host, hip, pool, aux;
    std::set<std::string> seen;
    for (const auto& [name, cpus] : fields) {
        std::vector<int>* target = name == "host" ? &host : name == "hip" ? &hip :
                                   name == "pool" ? &pool : name == "aux" ? &aux : nullptr;
        if (!target || !seen.insert(name).second || !strata::aux_cpus::parse_cpu_list(cpus, *target)) {
            error = "STRATA_POLY_CPUS: invalid or repeated " + name + " CPU list";
            return false;
        }
    }
    if (seen.size() != 4 || host.size() != 1 || hip.size() != 1 || pool.empty() || aux.empty()) {
        error = "STRATA_POLY_CPUS needs host=CPU,hip=CPU,pool=LIST,aux=LIST";
        return false;
    }
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        error = "STRATA_POLY_CPUS: cannot read the allowed CPUs";
        return false;
    }
    std::set<int> unique;
    for (int cpu : [&] { std::vector<int> all{host[0], hip[0]}; all.insert(all.end(), pool.begin(), pool.end());
                        all.insert(all.end(), aux.begin(), aux.end()); return all; }()) {
        if (cpu >= CPU_SETSIZE || !CPU_ISSET(cpu, &allowed) || !unique.insert(cpu).second) {
            error = "STRATA_POLY_CPUS: CPU lists must be distinct and allowed by this process";
            return false;
        }
    }
    // The host, HIP runtime, and each pool worker need distinct physical cores.
    std::vector<int> dedicated{host[0], hip[0]};
    dedicated.insert(dedicated.end(), pool.begin(), pool.end());
    std::set<std::pair<long, long>> cores;
    for (const auto& cpu : strata::aux_cpus::read_topology(dedicated)) {
        if (cpu.pkg < 0 || cpu.core < 0 || !cores.insert({cpu.pkg, cpu.core}).second) {
            error = "STRATA_POLY_CPUS: host, hip and pool need separate physical cores";
            return false;
        }
    }
    for (const auto& cpu : strata::aux_cpus::read_topology(aux)) {
        if (cpu.pkg < 0 || cpu.core < 0 || cores.count({cpu.pkg, cpu.core})) {
            error = "STRATA_POLY_CPUS: aux cannot share a physical core with host, hip or pool";
            return false;
        }
    }
    s.host = host[0]; s.hip = hip[0]; s.pool = std::move(pool); s.aux = std::move(aux);
    s.on = true;
    s.owned.insert(tid());
    if (!place(0, host)) {
        error = "STRATA_POLY_CPUS: cannot pin the host thread";
        s.on = false;
        return false;
    }
    s.sweeper = std::thread([] {
        State& a = state();
        { std::lock_guard<std::mutex> guard(a.mu); a.owned.insert(tid()); }
        place(0, a.aux);
        while (!a.stop.load(std::memory_order_acquire)) {
            sweep();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    return true;
}

// partition 0/1 selects the two pipeline engines' disjoint pool halves;
// -1 is the ordinary whole pool, -2 is the auxiliary staging pool.
inline int pool_threads(int requested, int partition = -1) {
    State& s = state();
    if (!s.on) return requested;
    const int half = (int) s.pool.size() / 2;
    const int available = partition == 0 ? half : partition == 1 ? (int) s.pool.size() - half :
                          partition == -2 ? (int) s.aux.size() : (int) s.pool.size();
    return std::min(requested, available + 1);
}

inline int configured_threads() {
    State& s = state();
    return s.on ? (int) s.pool.size() + 1 : 0;
}

inline void pin_pool(int worker, int partition = -1) {
    State& s = state();
    if (!s.on) return;
    const int self = tid();
    { std::lock_guard<std::mutex> guard(s.mu); s.owned.insert(self); if (s.hip_tid == self) s.hip_tid = -1; }
    if (partition == -2) { place(0, s.aux); return; }
    const int half = (int) s.pool.size() / 2;
    const int begin = partition == 1 ? half : 0;
    const int size = partition == 0 ? half : partition == 1 ? (int) s.pool.size() - half : (int) s.pool.size();
    if (size > 0) place(0, {s.pool[(size_t) (begin + worker % size)]});
}

inline void pin_aux() {
    State& s = state();
    if (!s.on) return;
    const int self = tid();
    { std::lock_guard<std::mutex> guard(s.mu); s.owned.insert(self); if (s.hip_tid == self) s.hip_tid = -1; }
    place(0, s.aux);
}

#else
inline bool configure(const char* spec, std::string& error) {
    if (spec && *spec) { error = "STRATA_POLY_CPUS is Linux-only"; return false; }
    return true;
}
inline int pool_threads(int requested, int = -1) { return requested; }
inline int configured_threads() { return 0; }
inline void pin_pool(int, int = -1) {}
inline void pin_aux() {}
inline void sweep() {}
#endif

}  // namespace strata::poly::cpu_placement
