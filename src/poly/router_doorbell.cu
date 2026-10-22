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

} // namespace

void router_doorbell_publish(uint32_t* device_word, uint32_t value, void* stream) {
    publish_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(device_word, value);
}

void partial_graph_ring(uint32_t* ready, void* stream) {
    partial_ring_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(ready);
}

void partial_graph_wait(const uint32_t* ready, const uint32_t* done, const uint32_t* abort, void* stream) {
    partial_wait_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(ready, done, abort);
}

} // namespace strata::poly
