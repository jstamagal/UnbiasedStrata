#pragma once
#if defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
template<typename F, int NC, int ROWS>
__device__ __forceinline__ void poly_dense_prefetch_load(
        const typename F::Block* w, const Q81Block* x, int nb, int n_out, int row0, int kb, int iqs,
        typename PolyDenseFragment<F>::X (&activation)[NC], typename F::W (&weight)[ROWS],
        int weight_mode, bool broadcast_scale) {
    if (kb >= nb) return;
#pragma unroll
    for (int col = 0; col < NC; ++col)
        activation[col] = PolyDenseFragment<F>::load(x + size_t(col) * nb + kb, iqs, broadcast_scale);
#pragma unroll
    for (int row = 0; row < ROWS; ++row)
        if (row0 + row < n_out)
            weight[row] = poly_dense_weight_load<F>(w + size_t(row0 + row) * nb + kb, iqs, weight_mode);
}

template<typename F, int NC, int ROWS>
__launch_bounds__(128)
__global__ void poly_dense_prefetch_kernel(const typename F::Block* __restrict__ w,
                                          const Q81Block* __restrict__ x, float* __restrict__ y,
                                          int n_in, int n_out, int weight_mode, bool broadcast_scale,
                                          bool transpose_sum) {
    using Fragment = PolyDenseFragment<F>;
    const int row0 = (4 * int(blockIdx.x) + int(threadIdx.y)) * ROWS;
    if (row0 >= n_out) return;
    const int lane = int(threadIdx.x);
    const int nb = n_in / 32;
    const int iqs = F::kqs(lane);
    typename Fragment::X activation[NC] = {}, next_activation[NC] = {};
    typename F::W weight[ROWS] = {}, next_weight[ROWS] = {};
    float sum[4][NC][ROWS] = {};
    poly_dense_prefetch_load<F, NC, ROWS>(w, x, nb, n_out, row0, lane / F::T, iqs,
                                        activation, weight, weight_mode, broadcast_scale);
    for (int base = lane / F::T; base < nb; base += F::BPI) {
#pragma unroll
        for (int group = 0; group < 4; ++group) {
            const int kb = base + group * (32 / F::T);
            const int next_kb = group == 3 ? base + F::BPI : base + (group + 1) * (32 / F::T);
            poly_dense_prefetch_load<F, NC, ROWS>(w, x, nb, n_out, row0, next_kb, iqs,
                                                next_activation, next_weight, weight_mode, broadcast_scale);
            if (kb < nb) {
#pragma unroll
                for (int row = 0; row < ROWS; ++row) {
                    if (row0 + row < n_out) {
#pragma unroll
                        for (int col = 0; col < NC; ++col)
                            sum[group][col][row] += Fragment::dot(weight[row], activation[col]);
                    }
                }
            }
#pragma unroll
            for (int col = 0; col < NC; ++col) activation[col] = next_activation[col];
#pragma unroll
            for (int row = 0; row < ROWS; ++row) weight[row] = next_weight[row];
        }
    }
    if (poly_dense_tsum<NC, ROWS>(sum, y, n_out, row0, lane, transpose_sum)) return;
#pragma unroll
    for (int col = 0; col < NC; ++col) {
#pragma unroll
        for (int row = 0; row < ROWS; ++row) {
            float value = sum[0][col][row];
#pragma unroll
            for (int group = 1; group < 4; ++group) value += sum[group][col][row];
            value = warp_sum(value);
            if (lane == 0 && row0 + row < n_out) y[size_t(col) * n_out + row0 + row] = value;
        }
    }
}

template<typename F, int NC, int ROWS>
void poly_dense_prefetch_n(const typename F::Block* w, const Q81Block* x, float* y,
                           int n_in, int n_out, cudaStream_t stream) {
    const dim3 grid(unsigned((size_t(n_out) + 4 * ROWS - 1) / (4 * ROWS)));
    poly_dense_prefetch_kernel<F, NC, ROWS><<<grid, dim3(32, 4), 0, stream>>>(
        w, x, y, n_in, n_out, poly_dense_weight_mode(), poly_dense_xscale_on(), poly_dense_tsum_on(NC));
}

template<typename F, int ROWS>
void poly_dense_prefetch_rows(const typename F::Block* w, const Q81Block* x, float* y,
                              int n_in, int n_out, int ncols, cudaStream_t stream) {
    switch (ncols) {
    case 1: poly_dense_prefetch_n<F, 1, ROWS>(w, x, y, n_in, n_out, stream); break;
    case 2: poly_dense_prefetch_n<F, 2, ROWS>(w, x, y, n_in, n_out, stream); break;
    case 3: poly_dense_prefetch_n<F, 3, ROWS>(w, x, y, n_in, n_out, stream); break;
    case 4: poly_dense_prefetch_n<F, 4, ROWS>(w, x, y, n_in, n_out, stream); break;
    }
}

template<typename F>
bool poly_dense_prefetch(const void* weights, const void* x, float* y,
                         int n_in, int n_out, int ncols, cudaStream_t stream) {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_POLY_DENSE_PREFETCH");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    if (!enabled || ncols < 1 || ncols > 4) return false;
    static const int rows = [] {
        constexpr bool q40 = std::is_same_v<F, SmallTraits<Q40Block, 4>>;
        const char* v = std::getenv(q40 ? "STRATA_POLY_Q40_DENSE_ROWS" : "STRATA_POLY_Q80_DENSE_ROWS");
        return v && (v[0] == '1' || v[0] == '4') && v[1] == '\0' ? v[0] - '0' : 2;
    }();
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* q = static_cast<const Q81Block*>(x);
    if (rows == 1) poly_dense_prefetch_rows<F, 1>(w, q, y, n_in, n_out, ncols, stream);
    else if (rows == 4) poly_dense_prefetch_rows<F, 4>(w, q, y, n_in, n_out, ncols, stream);
    else poly_dense_prefetch_rows<F, 2>(w, q, y, n_in, n_out, ncols, stream);
    return true;
}
#endif
