#pragma once
// ---- STRATA_POLY_MMVQ_ROW=1 (opt-in, wave32 AMD; measured on gfx1030): one wavefront per R rows and no shared memory.
// The default layout gives a row (or two) a block of four wavefronts whose lanes each walk a quarter of the row's
// weight blocks, park their sums in LDS, join, and let wavefront 0 add them up: for a 2 KB K-quant row that is
// mostly launching and joining.  Here ONE wavefront does a row: lane x plays the four threads (wavefront v = 0..3,
// lane x) of the default layout one after the other - the same weight blocks (kbx = tid / T, stride BPI) in the
// same order with the same F::load / F::apply - and adds the four sums in the order wavefront 0 adds the LDS
// partials, then the same butterfly.  Every output is BITWISE the default kernel's, for 1..8 columns and for the
// one-column kernels (which share the multi-column layout's thread mapping).  W wavefronts per block, each with its
// own R rows.
int g_row_r = -1, g_row_w = -1;   // native_mmvq_row_tune (tests, benchmarks): -1 = the environment / the table
int mmvq_row_env(const char* name, int dflt) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' ? std::atoi(v) : dflt;
}
// one wavefront's R rows from row0 (row0 < n_out): the body of both kernels below
template<typename F, int NCOLS, int R, bool TS>
__device__ __forceinline__ void mmvq_row_body(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                              float* __restrict__ y, int n_in, int n_out, int row0, int lane) {
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;
    float tot[NCOLS][R];
#pragma unroll
    for (int v = 0; v < WARPS; ++v) {                 // the default layout's wavefront v, lane `lane`
        const int tid = WARP * v + lane;
        const int kqs = F::kqs(tid);
        float tmp[NCOLS][R] = {};
        for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += F::BPI) {
            const int kby = kbx * F::KBY;
#pragma unroll
            for (int i = 0; i < R; ++i) {
                if (row0 + i < n_out) {
                    const typename F::W wv = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
#pragma unroll
                    for (int j = 0; j < NCOLS; ++j) tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < R; ++i) tot[j][i] = v == 0 ? tmp[j][i] : tot[j][i] + tmp[j][i];
    }
    if constexpr (TS) {
        constexpr int V = NCOLS * R, P = s26ts::pow2_ceil(V);
        float v[P];
#pragma unroll
        for (int k = 0; k < P; ++k) v[k] = k < V ? tot[k / R][k % R] : 0.0f;
        const float sum = s26ts::tsum<P>(v, lane);
        const int k = s26ts::tsum_token<P>(lane);
        const int j = k / R, i = k % R;
        if (lane == s26ts::tsum_lane<P>(k) && k < V && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = sum;
        return;
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const float sum = warp_sum(tot[j][i]);
            if (lane == 0 && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = sum;
        }
}
template<typename F, int NCOLS, int R, bool TS>
__launch_bounds__(256)
__global__ void native_mmvq_row_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                       float* __restrict__ y, int n_in, int n_out) {
    const int row0 = (int(blockIdx.x) * int(blockDim.y) + int(threadIdx.y)) * R;
    if (row0 >= n_out) return;
    mmvq_row_body<F, NCOLS, R, TS>(w, x, y, n_in, n_out, row0, int(threadIdx.x));
}
template<typename F, int NCOLS>
void row_launch_n(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int r, int wv, cudaStream_t s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const bool ts = s26_tsum_on();
    const unsigned blocks = unsigned((std::size_t(n_out) + std::size_t(r) * wv - 1) / (std::size_t(r) * wv));
    const dim3 threads(WARP, unsigned(wv));
#define STRATA_ROW_GO(R_) \
    do { if (ts) native_mmvq_row_kernel<F, NCOLS, R_, true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out); \
         else native_mmvq_row_kernel<F, NCOLS, R_, false><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out); } while (0)
    if (r == 1) STRATA_ROW_GO(1);
    else if (r == 2) STRATA_ROW_GO(2);
    else STRATA_ROW_GO(4);
#undef STRATA_ROW_GO
}
// Where it was measured faster on gfx1030 (2x Radeon PRO V620, hip_mmvq_row_parity --bench, 1.2 GB of weight copies
// in rotation so no cache holds them, STRATA_TSUM=1, microseconds a call, default -> row):
//   Q6_K 2560 -> 10240: 1 col 48.8 -> 49.5, 2: 50.8 -> 50.2, 3: 53.0 -> 51.1, 4: 58.7 -> 55.1, 6: 88.7 -> 66.0
//   Q5_K 2560 -> 10240: 43.8 -> 42.1, 48.8 -> 42.7, 65.5 -> 46.0, 86.1 -> 51.9, 109.5 -> 66.1
//   Q4_K 2560 -> 6144:  26.8 -> 25.1, 30.6 -> 25.7, 37.8 -> 28.0, 48.6 -> 30.4, 63.6 -> 39.5
//   IQ4_XS 2560 -> 6144: 28.2 -> 27.0, 29.9 -> 28.3, 32.8 -> 29.4, 35.3 -> 30.0, 42.0 -> 33.1
//   Q6_K 2560 -> 248320 (the head): 4 cols 1206 -> 1092, 6: 2011 -> 1423
// Not faster, and left on the default kernels: Q8_0 6144 -> 2560 (40.2 -> 40.4 at 4 columns: it already reads at
// the card's memory speed) and matrices of a few hundred rows (Q6_K 2560 -> 512: 6.6 -> 7.4).  Q3_K, Q4_0, Q5_0,
// IQ4_NL and Q2_0 are bitwise equal too but were not timed: they take this kernel only with STRATA_POLY_MMVQ_ROW=2 (every
// type and shape).
//   Q4_0 2560 -> 10240: 1 col 37.5 -> 35.6, 2: 37.4 -> 35.9, 4: 40.3 -> 36.8; 2560 -> 6144: 25.8 -> 24.8, 26.1 -> 25.0,
//   28.1 -> 25.3; its 6144 -> 2560 is not faster (24.5 -> 24.7, 24.8 -> 24.9, 28.2 -> 25.2): from 4,096 rows up only
template<typename F> struct RowTable { static constexpr int kind = 0, min_out = 1 << 30; };   // 0: only when forced
template<> struct RowTable<Q6KTraits> { static constexpr int kind = 2, min_out = 1024; };     // 1 row a wavefront up to 3 columns
template<> struct RowTable<Q5KTraits> { static constexpr int kind = 1, min_out = 1024; };     // ... up to 2 columns
template<> struct RowTable<Q4KTraits> { static constexpr int kind = 1, min_out = 1024; };
template<> struct RowTable<IQ4XSTraits> { static constexpr int kind = 1, min_out = 1024; };
template<> struct RowTable<SmallTraits<Q40Block, 4>> { static constexpr int kind = 1, min_out = 4096; };
template<typename F>
bool row_launch(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    static const int env_on = mmvq_row_env("STRATA_POLY_MMVQ_ROW", 0);
    static const int env_r = mmvq_row_env("STRATA_POLY_MMVQ_ROW_R", 0), env_w = mmvq_row_env("STRATA_POLY_MMVQ_ROW_W", 0);
    if (g_row_r == 0 || (g_row_r < 0 && env_on == 0)) return false;
    if (ncols < 1 || ncols > MAX_NCOLS) return false;
    const int kind = RowTable<F>::kind;
    if (g_row_r < 0 && env_on < 2 && (kind == 0 || n_out < RowTable<F>::min_out)) return false;   // the table: measured shapes only
    int r = g_row_r > 0 ? g_row_r : env_r > 0 ? env_r : (ncols <= (kind == 2 ? 3 : 2) ? 1 : 2);
    int wv = g_row_w > 0 ? g_row_w : env_w > 0 ? env_w : 1;
    r = r >= 4 ? (ncols <= 4 ? 4 : 2) : r >= 2 ? 2 : 1;      // 4 rows of 5..8 columns would be 32 sums a lane
    wv = wv >= 8 ? 8 : wv >= 4 ? 4 : wv >= 2 ? 2 : 1;
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ncols) {
        case 1: row_launch_n<F, 1>(weights, x_q8_1, y, n_in, n_out, r, wv, s); break;
        case 2: row_launch_n<F, 2>(weights, x_q8_1, y, n_in, n_out, r, wv, s); break;
        case 3: row_launch_n<F, 3>(weights, x_q8_1, y, n_in, n_out, r, wv, s); break;
        case 4: row_launch_n<F, 4>(weights, x_q8_1, y, n_in, n_out, r, wv, s); break;
        case 5: row_launch_n<F, 5>(weights, x_q8_1, y, n_in, n_out, r, wv, s); break;
        case 6: row_launch_n<F, 6>(weights, x_q8_1, y, n_in, n_out, r, wv, s); break;
        case 7: row_launch_n<F, 7>(weights, x_q8_1, y, n_in, n_out, r, wv, s); break;
        default: row_launch_n<F, 8>(weights, x_q8_1, y, n_in, n_out, r, wv, s); break;
    }
    return true;
}
// Fused launches for two or three matrices sharing one Q8_1 input.
template<typename F> constexpr int row_set_rows(int ncols) {
    return RowTable<F>::kind == 0 ? 2 : ncols <= (RowTable<F>::kind == 2 ? 3 : 2) ? 1 : 2;
}
struct RowSetArgs {
    const void* w[3];
    float* y[3];
    int n_out[3], type[3], end[3];   // end[i]: one past matrix i's last block
};
template<typename F, int NCOLS, bool TS>
__device__ __forceinline__ void row_set_item(const RowSetArgs& a, int it, int blk, const Q81Block* __restrict__ x, int n_in,
                                             int lane) {
    constexpr int R = row_set_rows<F>(NCOLS);
    const int row0 = blk * R;
    if (row0 >= a.n_out[it]) return;
    mmvq_row_body<F, NCOLS, R, TS>(static_cast<const typename F::Block*>(a.w[it]), x, a.y[it], n_in, a.n_out[it], row0, lane);
}
template<int NCOLS, bool TS>
__launch_bounds__(WARP)
__global__ void native_mmvq_rowset_kernel(RowSetArgs a, const Q81Block* __restrict__ x, int n_in) {
    int blk = int(blockIdx.x);
    const int it = blk < a.end[0] ? 0 : blk < a.end[1] ? 1 : 2;
    if (it > 0) blk -= a.end[it - 1];
    const int lane = int(threadIdx.x);
    switch (a.type[it]) {
        case 14: row_set_item<Q6KTraits, NCOLS, TS>(a, it, blk, x, n_in, lane); break;
        case 13: row_set_item<Q5KTraits, NCOLS, TS>(a, it, blk, x, n_in, lane); break;
        case 12: row_set_item<Q4KTraits, NCOLS, TS>(a, it, blk, x, n_in, lane); break;
        case 23: row_set_item<IQ4XSTraits, NCOLS, TS>(a, it, blk, x, n_in, lane); break;
        case 2: row_set_item<SmallTraits<Q40Block, 4>, NCOLS, TS>(a, it, blk, x, n_in, lane); break;
        case 6: row_set_item<SmallTraits<Q50Block, 4>, NCOLS, TS>(a, it, blk, x, n_in, lane); break;
        default: row_set_item<SmallTraits<Q80Block, 8>, NCOLS, TS>(a, it, blk, x, n_in, lane); break;   // 8
    }
}
int row_set_rows_of(int type, int ncols) {
    switch (type) {
        case 14: return row_set_rows<Q6KTraits>(ncols);
        case 13: return row_set_rows<Q5KTraits>(ncols);
        case 12: return row_set_rows<Q4KTraits>(ncols);
        case 23: return row_set_rows<IQ4XSTraits>(ncols);
        case 2: return row_set_rows<SmallTraits<Q40Block, 4>>(ncols);
        case 6: return row_set_rows<SmallTraits<Q50Block, 4>>(ncols);
        case 8: return row_set_rows<SmallTraits<Q80Block, 8>>(ncols);
        default: return 0;
    }
}
template<int NCOLS>
void row_set_launch(const RowSetArgs& a, const Q81Block* x, int n_in, unsigned blocks, cudaStream_t s) {
    if (s26_tsum_on()) native_mmvq_rowset_kernel<NCOLS, true><<<blocks, WARP, 0, s>>>(a, x, n_in);
    else native_mmvq_rowset_kernel<NCOLS, false><<<blocks, WARP, 0, s>>>(a, x, n_in);
}
#define STRATA_WAVE_MMVQ(...) \
    if (row_launch<__VA_ARGS__>(weights, x_q8_1, y, n_in, n_out, ncols, stream)) { launch_check(); return; }
