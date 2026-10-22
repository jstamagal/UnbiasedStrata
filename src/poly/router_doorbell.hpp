#pragma once

#include <cstdint>

namespace strata::poly {

// Publish completion of the router's host copies on a mapped pinned word.
void router_doorbell_publish(uint32_t* device_word, uint32_t value, void* stream);
void partial_graph_ring(uint32_t* ready, void* stream);
void partial_graph_publish(const float* x, const int32_t* ids, const float* wts,
                           float* host_x, int32_t* host_ids, float* host_wts,
                           int x_count, int route_count, uint32_t* ready, void* stream);
void partial_graph_publish_runahead(const float* x, const int32_t* ids, const float* wts,
                                    float* host_x, int32_t* host_ids, float* host_wts,
                                    int x_count, int route_count, uint32_t* ready,
                                    uint32_t* layer_ready, void* stream);
void partial_graph_wait(const uint32_t* ready, const uint32_t* done, const uint32_t* abort, void* stream);
void partial_graph_wait_slots(const uint32_t* ready, const uint32_t* done, const uint32_t* abort,
                              void** slots, int n_expert, void* stream);
void partial_graph_wait_dma(const uint32_t* ready, const uint32_t* done, const uint32_t* abort,
                            void* const* incoming, void** safe, int n_expert, void* stream);
void partial_graph_wait_dma_cached(const uint32_t* ready, const uint32_t* done, const uint32_t* abort,
                                   void* const* incoming, void** safe, const void* const* resident,
                                   const int32_t* ids, int route_count, int n_expert, void* stream);
void partial_graph_add(const uint32_t* ready, const uint32_t* done, const uint32_t* abort,
                       const float* cpu, float* sum, int count, void* stream);
void partial_graph_add_cached(const uint32_t* ready, const uint32_t* done, const uint32_t* abort,
                              const void* const* slots, const int32_t* ids, int route_count, int n_expert,
                              const float* cpu, float* sum, int count, void* stream);

}
