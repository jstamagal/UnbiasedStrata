// src/poly/qwen35.cpp - the layers of Qwen3.5 and Qwen3.6 with routed experts.  The reference for the math is
// llama.cpp's qwen35moe graph.
#include "qwen35.hpp"

#include "runtime.hpp"
#include "qwen35_fuse.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr int kConv = 4;   // what conv4_silu is written for; checked at load
constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum
constexpr size_t kScoreFloats = size_t(16) << 20;  // the attention's scores of the tokens it takes at a time
// A q8 prompt batch's product reads a block of keys at a time; before each block its K and V rows are dequantized
// into an fp16 workspace of this many rows.  Bounded, so it does not grow with the context: the natural block is
// about 16/3 * head_dim rows, well under this, so real prompts take the same products the fp16 path takes.
constexpr size_t kQ8WorkRows = 2048;

bool router_topk_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_ROUTER_TOPK");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

// Within a whole-stage partial graph, carry an interior layer's routed sum into
// the next layer's residual-add + input norm + Q8_1 launch.
bool tail_next_norm_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_TAIL_NEXT_NORM");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

}  // namespace

void Qwen35::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    n_gdn_ = n_attn_ = 0;
}

void Qwen35::check_draft_block() {
    if (blocks_[(size_t) n_layer_].gdn)
        throw std::runtime_error("the block that guesses ahead is not one this engine runs; --no-draft loads "
                                 "the model without it");
}

void Qwen35::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    n_head_ = shape_i("attention.head_count");
    n_kv_ = shape_i("attention.head_count_kv");
    head_dim_ = shape_i("attention.key_length");
    if (opt_.kv_format == KvFmt::kQ8_0 && (n_kv_ * head_dim_) % 32 != 0)
        throw std::runtime_error("--kv q8_0 needs n_kv*head_dim to be a multiple of 32 (ggml QK8_0); this model "
                                 "has " + std::to_string(n_kv_ * head_dim_));
    const MetaValue* rot = find("rope.dimension_count");
    n_rot_ = rot ? (int) rot->u : head_dim_;
    s_dim_ = shape_i("ssm.state_size");
    k_heads_ = shape_i("ssm.group_count");
    v_heads_ = shape_i("ssm.time_step_rank");
    d_inner_ = shape_i("ssm.inner_size");
    conv_dim_ = 2 * k_heads_ * s_dim_ + d_inner_;

    need(n_kv_ > 0 && n_head_ % n_kv_ == 0 && n_head_ <= 256, "bad attention.head_count");
    need(head_dim_ > 0 && shape_i("attention.value_length") == head_dim_,
         "the attention's keys and values differ in length");
    need(n_rot_ > 0 && n_rot_ % 2 == 0 && n_rot_ <= head_dim_, "bad rope.dimension_count");
    need(shape_i("ssm.conv_kernel") == kConv, "ssm.conv_kernel is not 4");
    need(s_dim_ > 0 && s_dim_ % 32 == 0 && s_dim_ <= 256, "ssm.state_size is not a multiple of 32 up to 256");
    need(k_heads_ > 0 && v_heads_ % k_heads_ == 0 && d_inner_ == v_heads_ * s_dim_, "bad ssm head counts");

    // The rotary embedding has an axis for time, two for an image and a spare one, and its sections say which
    // pairs turn by which.  A token of text has its position on the first three and none on the spare one, so
    // the kernel turns every pair by the position: that is wrong for a pair the sections leave to the spare axis.
    const MetaValue* sec = find("rope.dimension_sections");
    need(sec && sec->type == MetaType::ARRAY && sec->items.size() >= 3, "no rope.dimension_sections");
    int sections[4] = {}, covered = 0;
    for (size_t i = 0; i < 4 && i < sec->items.size(); ++i) covered += sections[i] = (int) sec->items[i].u;
    need(covered > 0, "empty rope.dimension_sections");
    for (int pair = 0; pair < n_rot_ / 2; ++pair) {
        const int s = pair % covered;
        need(s < 3 * sections[s % 3], "the rotary embedding leaves pairs to an axis text has no position on");
    }
    sections_ = covered;
    theta_scale_ = std::pow(shape_f("rope.freq_base"), -2.f / (float) n_rot_);
    // this architecture puts no limit on a SwiGLU's inputs
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;
}

void Qwen35::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    const bool mtp_rows_only = mtp_stage0_last_ && il == n_layer_;
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };

    L.attn_norm = f32("attn_norm.weight", n_embd_);
    if (!mtp_rows_only) L.ffn_norm = f32("post_attention_norm.weight", n_embd_);
    B.gdn = tensor(il, "attn_qkv.weight") != nullptr;
    L.fixed = true;   // the attention's launches read the position on the card when they are recorded
    if (B.gdn) {
        B.wqkv = mat("attn_qkv.weight", n_embd_, conv_dim_);
        B.wz = mat("attn_gate.weight", n_embd_, d_inner_);
        B.conv = f32("ssm_conv1d.weight", kConv, conv_dim_);
        B.dt = f32("ssm_dt.bias", v_heads_);
        B.a = f32("ssm_a", v_heads_);
        B.alpha = mat("ssm_alpha.weight", n_embd_, v_heads_);
        B.beta = mat("ssm_beta.weight", n_embd_, v_heads_);
        B.o_norm = f32("ssm_norm.weight", s_dim_);
        B.wo = mat("ssm_out.weight", d_inner_, n_embd_);
        ++n_gdn_;
    } else {
        // the query projection gives every head its query and then the gate of its output
        if (!mtp_rows_only) B.wq = mat("attn_q.weight", n_embd_, (int64_t) 2 * n_head_ * head_dim_);
        B.wk = mat("attn_k.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
        B.wv = mat("attn_v.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
        if (!mtp_rows_only) B.q_norm = f32("attn_q_norm.weight", head_dim_);
        B.k_norm = f32("attn_k_norm.weight", head_dim_);
        if (!mtp_rows_only) B.wo = mat("attn_output.weight", (int64_t) n_head_ * head_dim_, n_embd_);
        ++n_attn_;
    }
    if (!mtp_rows_only) {
        load_moe(il);
        B.sh_gate_inp = f32("ffn_gate_inp_shexp.weight", n_embd_);
    }
}

size_t Qwen35::recurrent_floats() const {
    return size_t(n_gdn_) * (size_t(3) * conv_dim_ + size_t(v_heads_) * s_dim_ * s_dim_);
}

int Qwen35::widest() const { return std::max(d_inner_, n_head_ * head_dim_); }

// The attention layers' caches, and each recurrent layer's convolution and delta-rule state in state_.
void Qwen35::alloc_caches() {
    if (opt_.kv_format == KvFmt::kQ8_0 && !axes_) {
        const size_t row_bytes = (size_t) n_kv_ * head_dim_ + (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
        std::fprintf(stderr, "strata-poly: q8_0 KV cache: %zu bytes a row\n", row_bytes);
    }
    const size_t past_n = size_t(3) * conv_dim_, s_n = size_t(v_heads_) * s_dim_ * s_dim_;
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
    float* st = state_;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (B.gdn) {
            B.past = st;
            st += past_n;
            B.s = st;
            st += s_n;
        } else if (opt_.kv_format == KvFmt::kQ8_0) {
            const size_t codes_row = (size_t) n_kv_ * head_dim_;                            // int8
            const size_t scales_row = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);  // fp16
            B.k_codes = (int8_t*) kv_alloc(codes_row, 1);
            B.k_scales = (uint16_t*) kv_alloc(scales_row, 1);
            B.v_codes = (int8_t*) kv_alloc(codes_row, 1);
            B.v_scales = (uint16_t*) kv_alloc(scales_row, 1);
        } else {
            B.keys = (uint16_t*) kv_alloc(row, 1);
            B.vals = (uint16_t*) kv_alloc(row, 1);
        }
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
    if (axes_) return;   // bind(): a call's scratch is every sequence's
    if (router_topk_on()) {
        router_ticket_ = (uint32_t*) dmalloc(sizeof(uint32_t), "fused router ticket");
        ck(cudaMemsetAsync(router_ticket_, 0, sizeof(uint32_t), cs(st_)), "fused router ticket");
    }
    axes_ = (int32_t*) dmalloc((size_t) rows_ * 3 * sizeof(int32_t), "a call's rotary positions");
    h_axes_.resize((size_t) rows_ * 3);
}

// Per-token buffers hold rows_ tokens, row after row.
void Qwen35::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * n_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    proj_ = fbuf(n * conv_dim_);
    qkv_ = fbuf(n * conv_dim_);
    z_ = fbuf(n * d_inner_);
    alpha_ = fbuf(n * v_heads_);
    beta_ = fbuf(n * v_heads_);
    gated_ = fbuf(n * d_inner_);

    qg_ = fbuf(2 * heads);
    q_ = fbuf(heads);
    kraw_ = fbuf(kv);
    k_ = fbuf(kv);
    v_ = fbuf(kv);
    att_ = fbuf(heads);
    score_ = fbuf((size_t) chunk_ * n_head_ * opt_.max_context);
    sg_ = fbuf(n);
    if (rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);   // a prompt batch's attention
    if (rows_ >= kStageRows && opt_.kv_format == KvFmt::kQ8_0) {
        // the K and V blocks a q8 prompt batch dequantizes, side by side; work_halves_ halves each
        work_halves_ = kQ8WorkRows * (size_t) n_kv_ * head_dim_;
        q8w_ = fbuf((2 * work_halves_ + 1) / 2);
    }
}

// A recurrent layer's convolution and delta-rule state lie one behind the other in state_ (alloc_caches()).
void Qwen35::save_layer(int il, int64_t, uint8_t* dst) {
    const Block& B = blocks_[(size_t) il];
    if (!B.past) return;
    const size_t n = size_t(3) * conv_dim_ + size_t(v_heads_) * s_dim_ * s_dim_;
    ck(cudaMemcpy(dst + (size_t) (B.past - state_) * sizeof(float), B.past, n * sizeof(float),
                  cudaMemcpyDeviceToHost),
       "save state");
}

size_t Qwen35::card_row_bytes() const {
    if (opt_.kv_format == KvFmt::kQ8_0)
        return (size_t) n_attn_ * 2 *
               ((size_t) n_kv_ * head_dim_ + (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t));
    return (size_t) n_attn_ * 2 * n_kv_ * head_dim_ * sizeof(uint16_t);
}

size_t Qwen35::mtp_row_bytes() const {
    if (opt_.kv_format == KvFmt::kQ8_0)
        return 2 * ((size_t) n_kv_ * head_dim_ + (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t));
    return 2 * (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
}

void Qwen35::mtp_download_rows(int64_t first, int n, uint8_t* dst, bool source) {
    const Block& B = blocks_[(size_t) n_layer_];
    if (n < 0 || n > 256) throw std::logic_error("MTP handoff takes at most 256 rows");
    const size_t source_bytes = source ? (size_t) n_embd_ * sizeof(float) : 0;
    const size_t bytes = (size_t) n * mtp_row_bytes() + source_bytes;
    static const bool direct_source = [] {
        const char* v = std::getenv("STRATA_POLY_MTP_COMPACT_HANDOFF");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    if (direct_source && n == 0) {
        if (source_bytes)
            ck(cudaMemcpyAsync(dst, hn_, source_bytes, cudaMemcpyDeviceToHost, cs(st_)), "MTP source handoff");
        ck(cudaStreamSynchronize(cs(st_)), "MTP source handoff");
        return;
    }
    if (!mtp_pack_) mtp_pack_ = (uint8_t*) dmalloc(256 * mtp_row_bytes() + (size_t) n_embd_ * sizeof(float),
                                                   "MTP packed handoff");
    if (opt_.kv_format == KvFmt::kQ8_0) {
        const size_t c = (size_t) n_kv_ * head_dim_, s = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
        mtp_pack_rows(B.k_codes + (size_t) first * c, (const uint8_t*) B.k_scales + (size_t) first * s,
                      B.v_codes + (size_t) first * c, (const uint8_t*) B.v_scales + (size_t) first * s,
                      hn_, mtp_pack_, (size_t) n * c, (size_t) n * s, source_bytes, st_);
    } else {
        const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
        mtp_pack_rows((const uint8_t*) B.keys + (size_t) first * row, nullptr,
                      (const uint8_t*) B.vals + (size_t) first * row, nullptr,
                      hn_, mtp_pack_, (size_t) n * row, 0, source_bytes, st_);
    }
    ck(cudaMemcpyAsync(dst, mtp_pack_, bytes, cudaMemcpyDeviceToHost, cs(st_)), "MTP row handoff");
    ck(cudaStreamSynchronize(cs(st_)), "MTP row handoff");
}

void Qwen35::mtp_upload_rows(int64_t first, int n, const uint8_t* src, bool wait, bool source) {
    Block& B = blocks_[(size_t) n_layer_];
    if (n < 0 || n > 256) throw std::logic_error("MTP handoff takes at most 256 rows");
    const size_t source_bytes = source ? (size_t) n_embd_ * sizeof(float) : 0;
    const size_t bytes = (size_t) n * mtp_row_bytes() + source_bytes;
    static const bool direct_source = [] {
        const char* v = std::getenv("STRATA_POLY_MTP_COMPACT_HANDOFF");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    if (direct_source && n == 0) {
        if (source_bytes)
            ck(cudaMemcpyAsync(hn_, src, source_bytes, cudaMemcpyHostToDevice, cs(st_)), "MTP source handoff");
        if (wait) ck(cudaStreamSynchronize(cs(st_)), "MTP source handoff");
        return;
    }
    if (!mtp_pack_) mtp_pack_ = (uint8_t*) dmalloc(256 * mtp_row_bytes() + (size_t) n_embd_ * sizeof(float),
                                                   "MTP packed handoff");
    ck(cudaMemcpyAsync(mtp_pack_, src, bytes, cudaMemcpyHostToDevice, cs(st_)), "MTP row handoff");
    if (opt_.kv_format == KvFmt::kQ8_0) {
        const size_t c = (size_t) n_kv_ * head_dim_, s = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
        mtp_unpack_rows(mtp_pack_, B.k_codes + (size_t) first * c, (uint8_t*) B.k_scales + (size_t) first * s,
                        B.v_codes + (size_t) first * c, (uint8_t*) B.v_scales + (size_t) first * s,
                        hn_, (size_t) n * c, (size_t) n * s, source_bytes, st_);
    } else {
        const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
        mtp_unpack_rows(mtp_pack_, (uint8_t*) B.keys + (size_t) first * row, nullptr,
                        (uint8_t*) B.vals + (size_t) first * row, nullptr,
                        hn_, (size_t) n * row, 0, source_bytes, st_);
    }
    if (wait) ck(cudaStreamSynchronize(cs(st_)), "MTP row handoff");
}

void Qwen35::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    if (opt_.kv_format == KvFmt::kQ8_0) {
        // q8_0: each attention block writes k_codes, k_scales, v_codes, v_scales, in that order, so that a load
        // can walk the same runs back.  Same layout as card_row_bytes() sums to (n * card_row_bytes() / n_attn_).
        const size_t crow = (size_t) n_kv_ * head_dim_,
                     srow = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
        const size_t at_c = (size_t) first * crow, at_s = (size_t) first * srow;
        const size_t n_c = (size_t) n * crow, n_s = (size_t) n * srow;
        for (const Block& B : blocks_) {
            if (!B.k_codes || n == 0) continue;
            ck(cudaMemcpy(dst, (const uint8_t*) B.k_codes + at_c, n_c, cudaMemcpyDeviceToHost), "save rows");
            dst += n_c;
            ck(cudaMemcpy(dst, (const uint8_t*) B.k_scales + at_s, n_s, cudaMemcpyDeviceToHost), "save rows");
            dst += n_s;
            ck(cudaMemcpy(dst, (const uint8_t*) B.v_codes + at_c, n_c, cudaMemcpyDeviceToHost), "save rows");
            dst += n_c;
            ck(cudaMemcpy(dst, (const uint8_t*) B.v_scales + at_s, n_s, cudaMemcpyDeviceToHost), "save rows");
            dst += n_s;
        }
        return;
    }
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row, bytes = (size_t) n * row;
    for (const Block& B : blocks_) {
        if (!B.keys || bytes == 0) continue;
        for (const uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy(dst, (const uint8_t*) cache + at, bytes, cudaMemcpyDeviceToHost), "save rows");
            dst += bytes;
        }
    }
}

void Qwen35::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    if (opt_.kv_format == KvFmt::kQ8_0) {
        // q8_0: mirror save_card_rows run for run, advancing by each run's own saved size (count * row) so
        // multi-layer blobs line up; the rows wanted start `first - from` rows into the run.
        const size_t crow = (size_t) n_kv_ * head_dim_,
                     srow = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
        const size_t at_c = (size_t) first * crow, at_s = (size_t) first * srow;
        const size_t n_c = (size_t) n * crow, n_s = (size_t) n * srow;
        const size_t sk_c = (size_t) (first - from) * crow, sk_s = (size_t) (first - from) * srow;
        const size_t all_c = (size_t) count * crow, all_s = (size_t) count * srow;
        for (const Block& B : blocks_) {
            if (!B.k_codes || n == 0) continue;
            ck(cudaMemcpy((uint8_t*) B.k_codes + at_c, src + sk_c, n_c, cudaMemcpyHostToDevice), "load rows");
            src += all_c;
            ck(cudaMemcpy((uint8_t*) B.k_scales + at_s, src + sk_s, n_s, cudaMemcpyHostToDevice), "load rows");
            src += all_s;
            ck(cudaMemcpy((uint8_t*) B.v_codes + at_c, src + sk_c, n_c, cudaMemcpyHostToDevice), "load rows");
            src += all_c;
            ck(cudaMemcpy((uint8_t*) B.v_scales + at_s, src + sk_s, n_s, cudaMemcpyHostToDevice), "load rows");
            src += all_s;
        }
        return;
    }
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row, bytes = (size_t) n * row;
    const size_t skip = (size_t) (first - from) * row, all = (size_t) count * row;
    for (const Block& B : blocks_) {
        if (!B.keys || bytes == 0) continue;
        for (uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy((uint8_t*) cache + at, src + skip, bytes, cudaMemcpyHostToDevice), "load rows");
            src += all;
        }
    }
}

// The gated delta net.  One projection gives the queries, keys and values, which a convolution over the last
// four tokens mixes; another gives the gate of the output.
void Qwen35::gdn(Block& B, int n) {
    const kernels::NativeMmvqItem gdn_proj[] = {
        {(int) B.wqkv.type, B.wqkv.p, proj_, (int) B.wqkv.n_out},
        {(int) B.wz.type, B.wz.p, z_, (int) B.wz.n_out},
    };
    static const bool gdn_set_on = [] {
        const char* v = std::getenv("STRATA_POLY_Q35_GDN_SET");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    const bool projected = n == 1 ? kernels::native_mmvq_set(gdn_proj, 2, xq_, n_embd_, 1, st_) :
        (gdn_set_on && n >= 2 && n <= 8 && !opt_.dump &&
         kernels::native_mmvq_set_explicit(gdn_proj, 2, xq_, n_embd_, n, st_));
    if (!projected) {
        lin(B.wqkv, xn_, proj_, n, xq_);
        lin(B.wz, xn_, z_, n, xq_);
    }
    static const bool fuse = [] {
        const char* v = std::getenv("STRATA_POLY_Q35_FUSE");
        return v && std::atoi(v) == 1;
    }();
    static const bool conv_window_on = [] {
        const char* v = std::getenv("STRATA_POLY_Q35_CONV_NORM_WINDOW");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    static const bool gates_window_on = [] {
        const char* v = std::getenv("STRATA_POLY_Q35_GATES_WINDOW");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    static const bool alpha_beta_on = [] {
        const char* v = std::getenv("STRATA_POLY_Q35_ALPHA_BETA");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    const bool conv_window = conv_window_on && n >= 2 && n <= 8 && s_dim_ == 128 && !rows_of_ && !opt_.dump;
    const bool conv_normed = (fuse && n == 1 && !opt_.dump) || conv_window;
    const bool gates_fused = !opt_.dump && ((fuse && n == 1) ||
                             (alpha_beta_on && n >= 1 && n <= 8) ||
                             (gates_window_on && n >= 2 && n <= 8)) &&
                             B.alpha.type == kTypeF32 && B.beta.type == kTypeF32;
    if (gates_fused) {
        qwen35_alpha_beta_gates(fp(B.alpha), fp(B.beta), xn_, fp(B.dt), fp(B.a), alpha_, beta_, n_embd_,
                                v_heads_, n, sum_threads(n_embd_), st_);
    } else if (fuse && n <= 8 && B.alpha.type == kTypeF32 && B.beta.type == kTypeF32) {
        qwen35_alpha_beta(fp(B.alpha), fp(B.beta), xn_, alpha_, beta_, n_embd_, v_heads_, n,
                          sum_threads(n_embd_), st_);
    } else {
        lin(B.alpha, xn_, alpha_, n, xq_);
        lin(B.beta, xn_, beta_, n, xq_);
    }
    if (rows_of_) {   // step(): each row with its own sequence's convolution and state
        for (int i = 0; i < n; ++i) {
            const int s = rows_of_[i].seq;
            const size_t c = (size_t) i * conv_dim_, h = (size_t) i * v_heads_, d = (size_t) i * d_inner_;
            if (conv_normed)
                qwen35_conv_norm1(proj_ + c, seq_state(s, B.past), fp(B.conv), qkv_ + c,
                                   conv_dim_, s_dim_, k_heads_, rms_eps_, nullptr, st_);
            else
                conv4_silu(proj_ + c, seq_state(s, B.past), fp(B.conv), qkv_ + c, conv_dim_, 1, st_);
            gdn_step(qkv_ + c, alpha_ + h, beta_ + h, fp(B.dt), fp(B.a), seq_state(s, B.s), z_ + d, fp(B.o_norm),
                     gated_ + d, s_dim_, v_heads_, k_heads_, 1, rms_eps_, st_, nullptr, 0, 0, conv_normed,
                     gates_fused);
        }
    } else {
        if (conv_window)
            qwen35_conv_norm_window(proj_, B.past, fp(B.conv), qkv_, conv_dim_, k_heads_, n,
                                     rms_eps_, snap_bits_ ? snap_at_ + (B.past - state_) : nullptr,
                                     rec_n_, snap_bits_, st_, pipe_save_into_ ? pipe_save_into_ + (B.past - state_) : nullptr);
        else if (conv_normed)
            qwen35_conv_norm1(proj_, B.past, fp(B.conv), qkv_, conv_dim_, s_dim_, k_heads_, rms_eps_,
                               (snap_bits_ & 1u) ? snap_at_ + (B.past - state_) : nullptr, st_,
                               pipe_save_into_ ? pipe_save_into_ + (B.past - state_) : nullptr);
        else
            conv4_silu(proj_, B.past, fp(B.conv), qkv_, conv_dim_, n, st_,
                       snap_bits_ ? snap_at_ + (B.past - state_) : nullptr, rec_n_, snap_bits_,
                       pipe_save_into_ ? pipe_save_into_ + (B.past - state_) : nullptr);
        gdn_step(qkv_, alpha_, beta_, fp(B.dt), fp(B.a), B.s, z_, fp(B.o_norm), gated_, s_dim_, v_heads_, k_heads_, n,
                 rms_eps_, st_, snap_bits_ ? snap_at_ + (B.s - state_) : nullptr, rec_n_, snap_bits_, conv_normed,
                 gates_fused, pipe_save_into_ ? pipe_save_into_ + (B.s - state_) : nullptr);
    }
    lin(B.wo, gated_, ao_, n);
}

// A row's position on the rotary embedding's three axes, as llama.cpp's mtmd counts them.  Text has the same on
// all three and counts up.  A picture's rows share the position the picture starts at on the first axis, and add
// their line to it on the second and their column on the third; the text after a picture goes on as many
// positions later as the picture has lines or columns, whichever is more.
void Qwen35::turn_of(int64_t row, int32_t* thw) const {
    int64_t skipped = 0;   // the rows before this one that took no position of their own
    for (const Image& im : images_) {
        if (row < im.at) break;
        if (row < im.at + im.n) {
            const int64_t p = im.at - skipped, j = row - im.at;
            thw[0] = (int32_t) p;
            thw[1] = (int32_t) (p + j / im.nx);
            thw[2] = (int32_t) (p + j % im.nx);
            return;
        }
        skipped += im.n - std::max(im.nx, im.n / im.nx);
    }
    thw[0] = thw[1] = thw[2] = (int32_t) (row - skipped);
}

// The rotary positions of n rows from row `pos`.  Rows of text alone count up from *first, and the result is
// null.  With a picture's rows among them the result is the positions of each row on the three axes, on the card.
const int32_t* Qwen35::turns(int64_t pos, int n, int64_t* first) {
    int32_t thw[3];
    turn_of(pos, thw);
    *first = thw[0];
    const bool pictures = std::any_of(images_.begin(), images_.end(), [&](const Image& im) {
        return im.at < pos + n && im.at + im.n > pos;
    });
    if (!pictures) return nullptr;
    if (axes_pos_ != pos || axes_n_ != n || axes_set_ != images_set_) {
        for (int t = 0; t < n; ++t) turn_of(pos + t, h_axes_.data() + (size_t) t * 3);
        // the launches that read the positions of the call before this one have to be through
        ck(cudaStreamSynchronize(cs(st_)), "rotary positions");
        ck(cudaMemcpy(axes_, h_axes_.data(), (size_t) n * 3 * sizeof(int32_t), cudaMemcpyHostToDevice),
           "rotary positions");
        axes_pos_ = pos;
        axes_n_ = n;
        axes_set_ = images_set_;
    }
    return axes_;
}

// The rows n tokens from position `pos` leave in an attention layer's caches: each head group's key, normed and
// turned by the position, and its value.  xn_ and xq_ hold the layer's input.
void Qwen35::attn_store(Block& B, int64_t pos, int n, bool projected) {
    int64_t first;
    const int32_t* axes = turns(pos, n, &first);
    const long long* at = recording_ ? at_ : nullptr;   // a recorded launch: the tokens from what at_ holds
    if (!projected) lin(B.wk, xn_, kraw_, n, xq_);
    head_norm_rope(kraw_, head_dim_, fp(B.k_norm), k_, head_dim_, n_kv_, n_rot_, theta_scale_, first, n, rms_eps_, st_,
                   nullptr, axes, sections_, at ? at + 1 : nullptr);
    if (!projected) lin(B.wv, xn_, v_, n, xq_);
    if (opt_.kv_format == KvFmt::kQ8_0) {
        static const bool pair_on = [] {
            const char* v = std::getenv("STRATA_POLY_Q35_KV_STORE_PAIR");
            return v && v[0] == '1' && v[1] == '\0';
        }();
        if (pair_on && n <= 8 && !opt_.dump)
            qwen35_kv_store_pair(k_, v_, B.k_codes, B.k_scales, B.v_codes, B.v_scales,
                                 n_kv_ * head_dim_, n, pos, at, st_);
        else if (at) {
            store_q8_at(k_, B.k_codes, B.k_scales, n_kv_ * head_dim_, at, n, 0, st_);
            store_q8_at(v_, B.v_codes, B.v_scales, n_kv_ * head_dim_, at, n, 0, st_);
        } else {
            store_q8(k_, B.k_codes, B.k_scales, n_kv_ * head_dim_, pos, n, st_);
            store_q8(v_, B.v_codes, B.v_scales, n_kv_ * head_dim_, pos, n, st_);
        }
    } else if (at) {
        store_f16_at(k_, B.keys, n_kv_ * head_dim_, at, n, 0, st_);
        store_f16_at(v_, B.vals, n_kv_ * head_dim_, at, n, 0, st_);
    } else {
        store_f16(k_, B.keys, n_kv_ * head_dim_, pos, n, st_);
        store_f16(v_, B.vals, n_kv_ * head_dim_, pos, n, st_);
    }
}

// The recordings of an attention layer's head are numbered by the rows they are sized for.
int Qwen35::head_key(int il, int n) const { return blocks_[(size_t) il].gdn ? 0 : room_key(n); }

int64_t Qwen35::turn(int64_t row) const {
    int32_t thw[3];
    turn_of(row, thw);
    return thw[0];
}

void Qwen35::attn(Block& B, int64_t pos, int n) {
    if (rows_of_) return attn_rows(B, n);
    int64_t first;
    const int32_t* axes = turns(pos, n, &first);
    const long long* at = recording_ ? at_ : nullptr;
    static const bool attn_set_on = [] {
        const char* v = std::getenv("STRATA_POLY_Q35_ATTN_SET");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    bool projected = false;
    if (attn_set_on && n >= 1 && n <= 8 && !opt_.dump) {
        const kernels::NativeMmvqItem projections[] = {
            {(int) B.wq.type, B.wq.p, qg_, (int) B.wq.n_out},
            {(int) B.wk.type, B.wk.p, kraw_, (int) B.wk.n_out},
            {(int) B.wv.type, B.wv.p, v_, (int) B.wv.n_out},
        };
        projected = kernels::native_mmvq_set_explicit(projections, 3, xq_, n_embd_, n, st_);
    }
    if (!projected) lin(B.wq, xn_, qg_, n, xq_);
    head_norm_rope(qg_, 2 * head_dim_, fp(B.q_norm), q_, head_dim_, n_head_, n_rot_, theta_scale_, first, n, rms_eps_,
                   st_, nullptr, axes, sections_, at ? at + 1 : nullptr);
    attn_store(B, pos, n, projected);
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    // a prompt batch through cuBLAS; a few tokens with the kernels that give a token the same bits in any company
    if (n >= kStageRows) {
        // a prompt batch: the fp16 path passes the caches where they lie (null caches and no workspace under q8);
        // under q8 each block of K and V the products read is dequantized into q8w_ (kwork then vwork)
        const uint16_t* kc = q8 ? nullptr : B.keys;
        const uint16_t* vc = q8 ? nullptr : B.vals;
        uint16_t* kw = q8 ? (uint16_t*) q8w_ : nullptr;
        uint16_t* vw = q8 ? kw + work_halves_ : nullptr;
        attend_batch(blas_, q_, kc, vc, pos, n, score_, (size_t) chunk_ * n_head_ * opt_.max_context, awork_,
                     att_, head_dim_, n_head_, n_kv_, scale, st_, 0, 0, false,
                     q8 ? Kvq8{B.k_codes, B.k_scales} : Kvq8{}, q8 ? Kvq8{B.v_codes, B.v_scales} : Kvq8{}, kw, vw,
                     q8 ? work_halves_ : 0);
    } else
        attend(q_, q8 ? nullptr : B.keys, q8 ? nullptr : B.vals, pos, n, score_, chunk_, att_, head_dim_, n_head_,
               n_kv_, scale, st_, 0, 0, false, at, at ? room(n) : 0, /*score_floats*/ 0,
               q8 ? Kvq8{B.k_codes, B.k_scales} : Kvq8{}, q8 ? Kvq8{B.v_codes, B.v_scales} : Kvq8{});
    static const bool fuse_attn_gate = [] {
        const char* v = std::getenv("STRATA_POLY_Q35_FUSE");
        return v && std::atoi(v) == 1;
    }();
    static const bool attn_gate_window_on = [] {
        const char* v = std::getenv("STRATA_POLY_Q35_ATTN_GATE_WINDOW");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    if (((fuse_attn_gate && n == 1) || (attn_gate_window_on && n >= 2 && n <= 8)) &&
        !opt_.dump && head_dim_ % 32 == 0 &&
        B.wo.type != kTypeF32 && B.wo.type != kTypeBF16) {
        qwen35_attn_gate_q8(att_, qg_ + head_dim_, q8_, n * n_head_, head_dim_, st_);
        lin(B.wo, att_, ao_, n, q8_);
    } else {
        mul_sigmoid(att_, qg_ + head_dim_, head_dim_, n * n_head_, 2 * head_dim_, false, st_);
        lin(B.wo, att_, ao_, n);
    }
}

// step()'s rows, each of its own sequence: the products for all of them at once, each row's position from at_, and
// each row's keys and values stored and attended to in its own sequence's caches.
void Qwen35::attn_rows(Block& B, int n) {
    const long long* at = at_;
    lin(B.wq, xn_, qg_, n, xq_);
    lin(B.wk, xn_, kraw_, n, xq_);
    lin(B.wv, xn_, v_, n, xq_);
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const size_t kv = (size_t) n_kv_ * head_dim_, hq = (size_t) n_head_ * head_dim_;
    for (int i = 0; i < n; ++i) {
        const Row& r = rows_of_[i];
        const long long* ai = at + 2 * i;
        // a row at a time: the turn takes one position and counts on from it, which suits a window's rows only
        head_norm_rope(qg_ + 2 * i * hq, 2 * head_dim_, fp(B.q_norm), q_ + i * hq, head_dim_, n_head_, n_rot_,
                       theta_scale_, 0, 1, rms_eps_, st_, nullptr, nullptr, sections_, ai + 1);
        head_norm_rope(kraw_ + i * kv, head_dim_, fp(B.k_norm), k_ + i * kv, head_dim_, n_kv_, n_rot_, theta_scale_, 0, 1,
                       rms_eps_, st_, nullptr, nullptr, sections_, ai + 1);
        Kvq8 kq{}, vq{};
        uint16_t *kc = nullptr, *vc = nullptr;
        if (q8) {
            kq = {(int8_t*) seq_buf(r.seq, B.k_codes), (uint16_t*) seq_buf(r.seq, B.k_scales)};
            vq = {(int8_t*) seq_buf(r.seq, B.v_codes), (uint16_t*) seq_buf(r.seq, B.v_scales)};
            store_q8_at(k_ + i * kv, (int8_t*) kq.codes, (uint16_t*) kq.scales, (int) kv, ai, 1, 0, st_);
            store_q8_at(v_ + i * kv, (int8_t*) vq.codes, (uint16_t*) vq.scales, (int) kv, ai, 1, 0, st_);
        } else {
            kc = (uint16_t*) seq_buf(r.seq, B.keys);
            vc = (uint16_t*) seq_buf(r.seq, B.vals);
            store_f16_at(k_ + i * kv, kc, (int) kv, ai, 1, 0, st_);
            store_f16_at(v_ + i * kv, vc, (int) kv, ai, 1, 0, st_);
        }
        const int64_t keep = pos_;
        pos_ = r.pos;
        const int sized = room(1);
        pos_ = keep;
        attend(q_ + i * hq, kc, vc, r.pos, 1, score_, chunk_, att_ + i * hq, head_dim_, n_head_, n_kv_, scale, st_, 0, 0,
               false, ai, sized, 0, kq, vq);
    }
    static const bool fuse_attn_rows_gate = [] {
        const char* v = std::getenv("STRATA_POLY_Q35_FUSE");
        return v && std::atoi(v) == 1;
    }();
    if (fuse_attn_rows_gate && n == 1 && !opt_.dump && head_dim_ % 32 == 0 &&
        B.wo.type != kTypeF32 && B.wo.type != kTypeBF16) {
        qwen35_attn_gate_q8(att_, qg_ + head_dim_, q8_, n_head_, head_dim_, st_);
        lin(B.wo, att_, ao_, n, q8_);
    } else {
        mul_sigmoid(att_, qg_ + head_dim_, head_dim_, n * n_head_, 2 * head_dim_, false, st_);
        lin(B.wo, att_, ao_, n);
    }
}

// A layer from its input to its router's copy.  A prompt batch's MoE (moe_rows) does its own routing.
void Qwen35::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    static const bool fuse = [] {
        const char* v = std::getenv("STRATA_POLY_Q35_FUSE");
        return v && std::atoi(v) == 1;
    }();
    const bool quick = fuse && n <= 8 && !opt_.dump;
    const bool carried = tail_next_norm_on() && partial_capture_ && n <= 8 && !opt_.dump &&
                         il > lo_ && layers_[(size_t) il - 1].moe;
    if (quick || carried)
        kernels::native_add_rms_norm_q8_1(r_, carried ? fo_ : nullptr, fp(L.attn_norm), xn_, xq_, n_embd_, n,
                                           rms_eps_, sum_threads(n_embd_), st_);
    else {
        rms_norm(r_, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
        kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    }
    dbg("attn_in", il, xn_, n_embd_);
    if (B.gdn) gdn(B, n);
    else attn(B, pos_, n);
    dbg("attn", il, ao_, n_embd_);
    if (quick)
        kernels::native_add_rms_norm_q8_1(r_, ao_, fp(L.ffn_norm), xn_, xq_, n_embd_, n, rms_eps_,
                                           sum_threads(n_embd_), st_);
    else {
        axpy(ao_, 1.f, r_, n * n_embd_, true, st_);
        rms_norm(r_, fp(L.ffn_norm), xn_, n_embd_, n, rms_eps_, st_);
        kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    }
    dbg("ffn_in", il, xn_, n_embd_);
    if (!batched(L, n)) moe_router(L, n);
}

void Qwen35::layer_tail(Layer&, int il, int n) {
    if (tail_next_norm_on() && partial_capture_ && n <= 8 && !opt_.dump && il + 1 < hi_) return;
    axpy(fo_, 1.f, r_, n * n_embd_, true, st_);
}

void Qwen35::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

const void* Qwen35::output_rows_q8(int first, int rows, float* h) {
    static const bool fuse = [] {
        const char* v = std::getenv("STRATA_POLY_HEAD_NORM_Q8");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    if (!fuse || opt_.dump || rows < 1 || rows > 8 || n_embd_ % 32 != 0 ||
        !kernels::native_mmvq_supported((int) output_.type))
        return Engine::output_rows_q8(first, rows, h);
    kernels::native_add_rms_norm_q8_1(r_ + (size_t) first * n_embd_, nullptr, fp(out_norm_), h, q8_,
                                      n_embd_, rows, rms_eps_, sum_threads(n_embd_), st_);
    return q8_;
}

// What ggml's MoE graph does for this model: a softmax over every expert's score, the n_used most likely, and
// their weights made to sum to one.
void Qwen35::route(int, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    float mx = logits[0];
    for (int e = 1; e < n_expert_; ++e) mx = std::max(mx, logits[e]);
    double all = 0.0;
    for (int e = 0; e < n_expert_; ++e) all += probs[e] = std::exp(logits[e] - mx);
    const float inv = (float) (1.0 / all);
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (probs[e] > probs[best]) best = e;
        ids[k] = best;
        w[k] = probs[best] * inv;
        sum += w[k];
        probs[best] = -1.f;
    }
    const float denom = std::max(sum, kMinWeightSum);
    for (int k = 0; k < n_used_; ++k) w[k] /= denom;
}

bool Qwen35::fused_router(const Layer& L, int n) const {
    return router_topk_on() && router_ticket_ && partial_capture_ && !opt_.dump &&
           n >= 1 && n <= 8 && n_embd_ >= 128 && sum_threads(n_embd_) >= n * 32 &&
           n_expert_ >= 1 && n_expert_ <= 512 && n_used_ >= 1 && n_used_ <= 32 &&
           n_used_ <= n_expert_ && L.gate_inp.type == kTypeF32;
}

void Qwen35::router_scores(const Layer& L, int n) {
    if (!fused_router(L, n)) Engine::router_scores(L, n);
}

bool Qwen35::route_card(int, int n, int32_t* ids, float* w) {
    route_rows(rl_, n_expert_, n_used_, n, 0, 1.f, nullptr, nullptr, true, 0.f, ids, w, st_);
    return true;
}

bool Qwen35::route_card_publish(int il, int n, int32_t* ids, float* w, float* host_x,
                                int32_t* host_ids, float* host_wts, uint32_t* ready,
                                uint32_t* layer_ready) {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_POLY_ROUTE_PUBLISH");
        const char* meta = std::getenv("STRATA_POLY_EXPERT_REUSE_META");
        const char* group4 = std::getenv("STRATA_POLY_EXPERT_REUSE_GROUP4_META");
        return (v && v[0] == '1' && v[1] == '\0') ||
               (meta && meta[0] == '1' && meta[1] == '\0') ||
               (group4 && group4[0] == '1' && group4[1] == '\0');
    }();
    const RoutePublish publish{xn_, host_x, host_ids, host_wts, n * n_embd_, ready, layer_ready,
                               d_reuse_partner_, false, d_reuse_group4_};
    const Layer& L = layers_[(size_t) il];
    if (fused_router(L, n)) {
        if (router_f32_publish(fp(L.gate_inp), xn_, rl_, n_embd_, n_expert_, n_used_, n,
                                router_ticket_, ids, w, publish, st_))
            return true;
        // The publisher can reject missing destinations; materialize the deferred scores before fallback.
        Engine::router_scores(L, n);
    }
    if (!enabled) return false;
    return route_rows_publish(rl_, n_expert_, n_used_, n, 0, 1.f, nullptr, nullptr, true, 0.f,
                              ids, w, publish, st_);
}

// The shared expert, times the sigmoid of its own gate.  It runs on the GPU while the host routes.
void Qwen35::moe_shared(const Layer& L, int il, int n) {
    const kernels::NativeMmvqItem shared_proj[] = {
        {(int) L.sh_gate.type, L.sh_gate.p, g_, (int) L.sh_gate.n_out},
        {(int) L.sh_up.type, L.sh_up.p, u_, (int) L.sh_up.n_out},
    };
    static const bool shared_set_on = [] {
        const char* v = std::getenv("STRATA_POLY_Q35_SHARED_SET");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    const bool projected = n == 1 ? kernels::native_mmvq_set(shared_proj, 2, xq_, n_embd_, 1, st_) :
        (shared_set_on && n >= 2 && n <= 8 && !opt_.dump &&
         kernels::native_mmvq_set_explicit(shared_proj, 2, xq_, n_embd_, n, st_));
    if (!projected) {
        lin(L.sh_gate, xn_, g_, n, xq_);
        lin(L.sh_up, xn_, u_, n, xq_);
    }
    static const bool fuse = [] {
        const char* v = std::getenv("STRATA_POLY_Q35_FUSE");
        return v && std::atoi(v) == 1;
    }();
    if (fuse && n <= 8 && !opt_.dump && std::isinf(clamp_shexp_[(size_t) il]) &&
        L.sh_down.type != kTypeF32 && L.sh_down.type != kTypeBF16 && n_ff_shexp_ % 32 == 0) {
        kernels::native_swiglu_quantize_q8_1(g_, u_, q8_, n_ff_shexp_, n, st_);
        lin(L.sh_down, h_, fo_, n, q8_);
    } else {
        swiglu_clamp(g_, u_, h_, n * n_ff_shexp_, clamp_shexp_[(size_t) il], st_);
        lin(L.sh_down, h_, fo_, n);
    }
    static const bool shared_gate_on = [] {
        const char* v = std::getenv("STRATA_POLY_Q35_SHARED_GATE");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    if (shared_gate_on && n <= 8 && !opt_.dump)
        qwen35_shared_gate(fp(blocks_[(size_t) il].sh_gate_inp), xn_, sg_, fo_,
                           n_embd_, n, sum_threads(n_embd_), st_);
    else {
        lin(blocks_[(size_t) il].sh_gate_inp, xn_, sg_, n);
        mul_sigmoid(fo_, sg_, n_embd_, n, 1, true, st_);
    }
}

// The draft block is one more attention layer.
void Qwen35::block_attend(int64_t pos, int n) { attn(blocks_[(size_t) n_layer_], pos, n); }
void Qwen35::block_rows(int64_t pos, int n) { attn_store(blocks_[(size_t) n_layer_], pos, n); }

}  // namespace strata::poly
