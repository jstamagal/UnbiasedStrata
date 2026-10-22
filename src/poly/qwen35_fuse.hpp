#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::poly {

void qwen35_alpha_beta(const float* alpha_w, const float* beta_w, const float* x,
                       float* alpha, float* beta, int n_in, int n_out, int rows, int threads, void* stream);
void qwen35_alpha_beta_gates(const float* alpha_w, const float* beta_w, const float* x,
                             const float* dt, const float* a, float* alpha, float* beta,
                             int n_in, int n_out, int rows, int threads, void* stream);

// One decode row: convolution and the Q/K head norms in the same block.
void qwen35_conv_norm1(const float* x, float* state, const float* w, float* out, int channels,
                       int dim, int k_heads, float eps, float* snap, void* stream, float* rollback = nullptr);

// Short decode windows: convolution and Q/K norm across rows, preserving state snapshots.
void qwen35_conv_norm_window(const float* x, float* state, const float* w, float* out,
                             int channels, int k_heads, int rows, float eps,
                             float* snap, size_t snap_stride, unsigned snaps, void* stream, float* rollback = nullptr);

// Store Q8_0 K and V cache rows in one launch, with a fixed or device-held starting position.
void qwen35_kv_store_pair(const float* key, const float* value,
                          int8_t* key_codes, uint16_t* key_scales,
                          int8_t* value_codes, uint16_t* value_scales,
                          int n, int rows, int64_t pos, const long long* at, void* stream);

// Dot the F32 shared-expert gate with each row, then apply its sigmoid to that row's output.
void qwen35_shared_gate(const float* w, const float* x, float* gate, float* expert_out,
                        int n, int rows, int threads, void* stream);

// Apply the attention output gate and write its Q8_1 image in one launch.
void qwen35_attn_gate_q8(float* att, const float* gate, void* q8, int heads, int dim, void* stream);

}
