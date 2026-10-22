#pragma once
#if defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
#include "poly_dense_weights.cuh"
#include "poly_dense_xscale.cuh"
#include "poly_dense_tsum.cuh"
template<typename F> struct PolyDenseFragment;

template<> struct PolyDenseFragment<SmallTraits<Q40Block, 4>> {
    using F = SmallTraits<Q40Block, 4>;
    struct X { int lo[2], hi[2]; float2 ds; };
    __device__ static X load(const Q81Block* x, int iqs, bool broadcast_scale) {
        X r;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            r.lo[i] = reinterpret_cast<const int*>(x->qs)[iqs + i];
            r.hi[i] = reinterpret_cast<const int*>(x->qs)[iqs + i + 4];
        }
        r.ds = __half22float2(poly_dense_xscale<F::T>(x, broadcast_scale));
        return r;
    }
    __device__ static float dot(const F::W& w, const X& x) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = STRATA_DP4A(w.vi0[i], x.lo[i], sumi);
            sumi = STRATA_DP4A(w.vi1[i], x.hi[i], sumi);
        }
        return w.d * (sumi * x.ds.x - 4 * x.ds.y);
    }
};

template<> struct PolyDenseFragment<SmallTraits<Q80Block, 8>> {
    using F = SmallTraits<Q80Block, 8>;
    struct X { int u[2]; float d; };
    __device__ static X load(const Q81Block* x, int iqs, bool broadcast_scale) {
        X r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.u[i] = reinterpret_cast<const int*>(x->qs)[iqs + i];
        r.d = __low2float(poly_dense_xscale<F::T>(x, broadcast_scale));
        return r;
    }
    __device__ static float dot(const F::W& w, const X& x) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) sumi = STRATA_DP4A(w.v[i], x.u[i], sumi);
        return w.d0 * x.d * float(sumi);
    }
};

// Each wave handles two rows with all four reference warp accumulators live.
// Adjacent groups issue independent loads; each group's own block order and
// the final group and lane sums retain the native MMVQ reduction order.
template<typename F, int NC>
__launch_bounds__(128)
__global__ void poly_dense_wave_kernel(const typename F::Block* __restrict__ w,
                                      const Q81Block* __restrict__ x,
                                      float* __restrict__ y, int n_in, int n_out, bool vector_load, bool broadcast_scale, bool transpose_sum) {
    using Fragment = PolyDenseFragment<F>;
    const int row0 = (4 * int(blockIdx.x) + int(threadIdx.y)) * 2;
    if (row0 >= n_out) return;
    const int lane = int(threadIdx.x);
    const int nb = n_in / 32;
    const int iqs = F::kqs(lane);
    float sum[4][NC][2] = {};
    for (int base = lane / F::T; base < nb; base += F::BPI) {
#pragma unroll
        for (int group = 0; group < 4; ++group) {
            const int kb = base + group * (32 / F::T);
            if (kb < nb) {
                typename Fragment::X activation[NC];
#pragma unroll
                for (int col = 0; col < NC; ++col)
                    activation[col] = Fragment::load(x + size_t(col) * nb + kb, iqs, broadcast_scale);
#pragma unroll
                for (int row = 0; row < 2; ++row) {
                    if (row0 + row < n_out) {
                        const typename F::W weight = poly_dense_weight_load<F>(w + size_t(row0 + row) * nb + kb, iqs, vector_load);
#pragma unroll
                        for (int col = 0; col < NC; ++col)
                            sum[group][col][row] += Fragment::dot(weight, activation[col]);
                    }
                }
            }
        }
    }
    if (poly_dense_tsum<NC, 2>(sum, y, n_out, row0, lane, transpose_sum)) return;
#pragma unroll
    for (int col = 0; col < NC; ++col) {
#pragma unroll
        for (int row = 0; row < 2; ++row) {
            float value = sum[0][col][row];
#pragma unroll
            for (int group = 1; group < 4; ++group) value += sum[group][col][row];
            value = warp_sum(value);
            if (lane == 0 && row0 + row < n_out) y[size_t(col) * n_out + row0 + row] = value;
        }
    }
}

#include "poly_dense_q80_rows.cuh"
#include "poly_dense_q40_rows.cuh"

template<typename F>
bool poly_dense_wave(const void* weights, const void* x, float* y,
                     int n_in, int n_out, int ncols, cudaStream_t stream) {
    constexpr bool q40 = std::is_same_v<F, SmallTraits<Q40Block, 4>>;
    constexpr bool q80 = std::is_same_v<F, SmallTraits<Q80Block, 8>>;
    if constexpr (!q40 && !q80) {
        return false;
    } else {
        static const bool enabled = [] {
            const char* v = std::getenv(q40 ? "STRATA_POLY_Q40_DENSE_WAVE" : "STRATA_POLY_Q80_DENSE_WAVE");
            return v && v[0] == '1' && v[1] == '\0';
        }();
        static const bool wide = [] {
            const char* v = std::getenv("STRATA_POLY_DENSE_WAVE_WIDE");
            return v && v[0] == '1' && v[1] == '\0';
        }();
        if (ncols < 1 || ncols > 8 || (ncols > 1 && !g_multi_exact)) return false;
        if (!(ncols <= 4 ? enabled : wide)) return false;
        if constexpr (q80) {
            if (poly_dense_q80_rows(weights, x, y, n_in, n_out, ncols, stream)) return true;
        }
        if constexpr (q40) {
            if (poly_dense_q40_rows(weights, x, y, n_in, n_out, ncols, stream)) return true;
        }
        const auto* w = static_cast<const typename F::Block*>(weights);
        const auto* q = static_cast<const Q81Block*>(x);
        const dim3 grid(unsigned((size_t(n_out) + 7) / 8));
        const dim3 threads(32, 4);
        switch (ncols) {
        case 1: poly_dense_wave_kernel<F, 1><<<grid, threads, 0, stream>>>(w, q, y, n_in, n_out, poly_dense_weight_vector_on(), poly_dense_xscale_on(), poly_dense_tsum_on()); break;
        case 2: poly_dense_wave_kernel<F, 2><<<grid, threads, 0, stream>>>(w, q, y, n_in, n_out, poly_dense_weight_vector_on(), poly_dense_xscale_on(), poly_dense_tsum_on()); break;
        case 3: poly_dense_wave_kernel<F, 3><<<grid, threads, 0, stream>>>(w, q, y, n_in, n_out, poly_dense_weight_vector_on(), poly_dense_xscale_on(), poly_dense_tsum_on()); break;
        case 4: poly_dense_wave_kernel<F, 4><<<grid, threads, 0, stream>>>(w, q, y, n_in, n_out, poly_dense_weight_vector_on(), poly_dense_xscale_on(), poly_dense_tsum_on()); break;
        case 5: poly_dense_wave_kernel<F, 5><<<grid, threads, 0, stream>>>(w, q, y, n_in, n_out, poly_dense_weight_vector_on(), poly_dense_xscale_on(), poly_dense_tsum_on()); break;
        case 6: poly_dense_wave_kernel<F, 6><<<grid, threads, 0, stream>>>(w, q, y, n_in, n_out, poly_dense_weight_vector_on(), poly_dense_xscale_on(), poly_dense_tsum_on()); break;
        case 7: poly_dense_wave_kernel<F, 7><<<grid, threads, 0, stream>>>(w, q, y, n_in, n_out, poly_dense_weight_vector_on(), poly_dense_xscale_on(), poly_dense_tsum_on()); break;
        case 8: poly_dense_wave_kernel<F, 8><<<grid, threads, 0, stream>>>(w, q, y, n_in, n_out, poly_dense_weight_vector_on(), poly_dense_xscale_on(), poly_dense_tsum_on()); break;
        }
        return true;
    }
}
#else
template<typename F>
bool poly_dense_wave(const void*, const void*, float*, int, int, int, cudaStream_t) { return false; }
#endif
