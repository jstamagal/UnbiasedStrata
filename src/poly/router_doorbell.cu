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
                                       int x_count, int route_count, uint32_t* ready) {
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
        *p = *p + 1;
        __threadfence_system();
    }
}

__global__ void partial_ring_kernel(uint32_t* ready) {
    __threadfence_system();
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(ready);
    *p = *p + 1;
    __threadfence_system();
}

__global__ void partial_wait_kernel(const uint32_t* ready, const uint32_t* done, const uint32_t* abort) {
    const auto* r = reinterpret_cast<const volatile uint32_t*>(ready);
    const auto* d = reinterpret_cast<const volatile uint32_t*>(done);
    const auto* a = reinterpret_cast<const volatile uint32_t*>(abort);
    const uint32_t need = *r;
    while (*d != need && *a == 0) { }
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

} // namespace

void router_doorbell_publish(uint32_t* device_word, uint32_t value, void* stream) {
    publish_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(device_word, value);
}

void partial_graph_publish(const float* x, const int32_t* ids, const float* wts,
                           float* host_x, int32_t* host_ids, float* host_wts,
                           int x_count, int route_count, uint32_t* ready, void* stream) {
    partial_publish_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(
        x, ids, wts, host_x, host_ids, host_wts, x_count, route_count, ready);
}

void partial_graph_ring(uint32_t* ready, void* stream) {
    partial_ring_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(ready);
}

void partial_graph_wait(const uint32_t* ready, const uint32_t* done, const uint32_t* abort, void* stream) {
    partial_wait_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(ready, done, abort);
}


void partial_graph_add(const uint32_t* ready, const uint32_t* done, const uint32_t* abort,
                       const float* cpu, float* sum, int count, void* stream) {
    const int blocks = (count + 255) / 256;
    partial_add_kernel<<<blocks < 32 ? blocks : 32, 256, 0, (cudaStream_t) stream>>>(
        ready, done, abort, cpu, sum, count);
}

} // namespace strata::poly
