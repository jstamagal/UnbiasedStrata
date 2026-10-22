#include "router_doorbell.hpp"

#include <cuda_runtime.h>

namespace strata::poly {
namespace {

__global__ void publish_kernel(uint32_t* word, uint32_t value) {
    __threadfence_system();
    *reinterpret_cast<volatile uint32_t*>(word) = value;
#if defined(__HIPCC__)
    // HIP may retain a volatile mapped-host store in L2 until a system fence.
    __threadfence_system();
#endif
}

__global__ void partial_publish_kernel(const float* x, const int32_t* ids, const float* wts,
                                       float* host_x, int32_t* host_ids, float* host_wts,
                                       int x_count, int route_count, uint32_t* ready, uint32_t* layer_ready) {
    const int t = threadIdx.x;
    for (int i = t; i < x_count; i += blockDim.x) host_x[i] = x[i];
    for (int i = t; i < route_count; i += blockDim.x) {
        host_ids[i] = ids[i];
        host_wts[i] = wts[i];
    }
    __threadfence_system();
    __syncthreads();
    if (t == 0) {
        volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(ready);
        const uint32_t value = *p + 1;
        *p = value;
        if (layer_ready) *reinterpret_cast<volatile uint32_t*>(layer_ready) = value;
        __threadfence_system();
    }
}

__global__ void partial_ring_kernel(uint32_t* ready) {
    __threadfence_system();
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(ready);
    *p = *p + 1;
    __threadfence_system();
}

__global__ void partial_wait_kernel(const uint32_t* ready, const uint32_t* done, const uint32_t* abort,
                                    void** slots, int n_expert) {
    const auto* r = reinterpret_cast<const volatile uint32_t*>(ready);
    const auto* d = reinterpret_cast<const volatile uint32_t*>(done);
    const auto* a = reinterpret_cast<const volatile uint32_t*>(abort);
    __shared__ int cancelled;
    if (threadIdx.x == 0) {
        const uint32_t need = *r;
        while (*d != need && *a == 0) { }
        cancelled = *a != 0;
    }
    __syncthreads();
    if (cancelled && slots)
        for (int i = threadIdx.x; i < n_expert; i += blockDim.x) slots[i] = nullptr;
    __threadfence_system();
}

__global__ void partial_wait_dma_kernel(const uint32_t* ready, const uint32_t* done, const uint32_t* abort,
                                        void* const* incoming, void** safe, int n_expert) {
    const auto* r = reinterpret_cast<const volatile uint32_t*>(ready);
    const auto* d = reinterpret_cast<const volatile uint32_t*>(done);
    const auto* a = reinterpret_cast<const volatile uint32_t*>(abort);
    __shared__ int landed;
    if (threadIdx.x == 0) {
        const uint32_t need = *r;
        while (*d != need && *a == 0) { }
        landed = *a == 0;
    }
    __syncthreads();
    __threadfence_system();
    for (int i = threadIdx.x; i < n_expert; i += blockDim.x)
        safe[i] = landed ? incoming[i] : nullptr;
    __threadfence_system();
}


__global__ void partial_add_kernel(const uint32_t* ready, const uint32_t* done, const uint32_t* abort,
                                   const float* cpu, float* sum, int count) {
    const auto* r = reinterpret_cast<const volatile uint32_t*>(ready);
    const auto* d = reinterpret_cast<const volatile uint32_t*>(done);
    const auto* a = reinterpret_cast<const volatile uint32_t*>(abort);
    __shared__ int cancelled;
    if (threadIdx.x == 0) {
        const uint32_t need = *r;
        while (*d != need && *a == 0) { }
        cancelled = *a != 0;
    }
    __syncthreads();
    if (cancelled) return;
    __threadfence_system();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
        sum[i] += cpu[i];
}

__global__ void partial_add_cached_kernel(const uint32_t* ready, const uint32_t* done, const uint32_t* abort,
                                          const void* const* slots, const int32_t* ids,
                                          int route_count, int n_expert, const float* cpu, float* sum, int count) {
    const auto* r = reinterpret_cast<const volatile uint32_t*>(ready);
    const auto* d = reinterpret_cast<const volatile uint32_t*>(done);
    const auto* a = reinterpret_cast<const volatile uint32_t*>(abort);
    __shared__ int needs_cpu;
    __shared__ int cancelled;
    if (threadIdx.x == 0) {
        needs_cpu = 0;
        cancelled = 0;
        for (int j = 0; j < route_count; ++j) {
            const int e = ids[j];
            if (e < 0 || e >= n_expert || !slots[e]) { needs_cpu = 1; break; }
        }
        if (needs_cpu) {
            const uint32_t need = *r;
            while (*d != need && *a == 0) { }
            cancelled = *a != 0;
        }
    }
    __syncthreads();
    if (!needs_cpu || cancelled) return;
    __threadfence_system();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
        sum[i] += cpu[i];
}

} // namespace

void router_doorbell_publish(uint32_t* device_word, uint32_t value, void* stream) {
    publish_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(device_word, value);
}

void partial_graph_publish(const float* x, const int32_t* ids, const float* wts,
                           float* host_x, int32_t* host_ids, float* host_wts,
                           int x_count, int route_count, uint32_t* ready, void* stream) {
    partial_publish_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(
        x, ids, wts, host_x, host_ids, host_wts, x_count, route_count, ready, nullptr);
}

void partial_graph_publish_runahead(const float* x, const int32_t* ids, const float* wts,
                                    float* host_x, int32_t* host_ids, float* host_wts,
                                    int x_count, int route_count, uint32_t* ready,
                                    uint32_t* layer_ready, void* stream) {
    partial_publish_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(
        x, ids, wts, host_x, host_ids, host_wts, x_count, route_count, ready, layer_ready);
}

void partial_graph_ring(uint32_t* ready, void* stream) {
    partial_ring_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(ready);
}

void partial_graph_wait(const uint32_t* ready, const uint32_t* done, const uint32_t* abort, void* stream) {
    partial_wait_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(ready, done, abort, nullptr, 0);
}

void partial_graph_wait_slots(const uint32_t* ready, const uint32_t* done, const uint32_t* abort,
                              void** slots, int n_expert, void* stream) {
    partial_wait_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(ready, done, abort, slots, n_expert);
}

void partial_graph_wait_dma(const uint32_t* ready, const uint32_t* done, const uint32_t* abort,
                            void* const* incoming, void** safe, int n_expert, void* stream) {
    partial_wait_dma_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(
        ready, done, abort, incoming, safe, n_expert);
}


void partial_graph_add(const uint32_t* ready, const uint32_t* done, const uint32_t* abort,
                       const float* cpu, float* sum, int count, void* stream) {
    const int blocks = (count + 255) / 256;
    partial_add_kernel<<<blocks < 32 ? blocks : 32, 256, 0, (cudaStream_t) stream>>>(
        ready, done, abort, cpu, sum, count);
}

void partial_graph_add_cached(const uint32_t* ready, const uint32_t* done, const uint32_t* abort,
                              const void* const* slots, const int32_t* ids, int route_count, int n_expert,
                              const float* cpu, float* sum, int count, void* stream) {
    const int blocks = (count + 255) / 256;
    partial_add_cached_kernel<<<blocks < 32 ? blocks : 32, 256, 0, (cudaStream_t) stream>>>(
        ready, done, abort, slots, ids, route_count, n_expert, cpu, sum, count);
}

} // namespace strata::poly
