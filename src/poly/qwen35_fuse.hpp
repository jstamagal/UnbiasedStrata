#pragma once

#include <cstddef>

namespace strata::poly {

void qwen35_alpha_beta(const float* alpha_w, const float* beta_w, const float* x,
                       float* alpha, float* beta, int n_in, int n_out, int rows, int threads, void* stream);
void qwen35_alpha_beta_gates(const float* alpha_w, const float* beta_w, const float* x,
                             const float* dt, const float* a, float* alpha, float* beta,
                             int n_in, int n_out, int rows, int threads, void* stream);

// One decode row: convolution and the Q/K head norms in the same block.
void qwen35_conv_norm1(const float* x, float* state, const float* w, float* out, int channels,
                       int dim, int k_heads, float eps, float* snap, void* stream);

// Short decode windows: convolution and Q/K norm across rows, preserving state snapshots.
void qwen35_conv_norm_window(const float* x, float* state, const float* w, float* out,
                             int channels, int k_heads, int rows, float eps,
                             float* snap, size_t snap_stride, unsigned snaps, void* stream);

// Apply the attention output gate and write its Q8_1 image in one launch.
void qwen35_attn_gate_q8(float* att, const float* gate, void* q8, int heads, int dim, void* stream);

}
