#pragma once
#if defined(__HIPCC__) && !defined(STRATA_HIP_GFX906)
template<int NC, int ROWS>
__launch_bounds__(128)
__global__ void poly_dense_q80_rows_kernel(const Q80Block* __restrict__ w,
                                          const Q81Block* __restrict__ x,
                                          float* __restrict__ y, int n_in, int n_out) {
    using F = SmallTraits<Q80Block, 8>;
    using Fragment = PolyDenseFragment<F>;
    const int row0 = (4 * int(blockIdx.x) + int(threadIdx.y)) * ROWS;
    if (row0 >= n_out) return;
    const int lane = int(threadIdx.x);
    const int nb = n_in / 32;
    const int iqs = F::kqs(lane);
    float sum[4][NC][ROWS] = {};
    for (int base = lane / F::T; base < nb; base += F::BPI) {
#pragma unroll
        for (int group = 0; group < 4; ++group) {
            const int kb = base + group * (32 / F::T);
            if (kb < nb) {
                typename Fragment::X activation[NC];
#pragma unroll
                for (int col = 0; col < NC; ++col)
                    activation[col] = Fragment::load(x + size_t(col) * nb + kb, iqs);
#pragma unroll
                for (int row = 0; row < ROWS; ++row) {
                    if (row0 + row < n_out) {
                        const F::W weight = F::load(w + size_t(row0 + row) * nb + kb, iqs);
#pragma unroll
                        for (int col = 0; col < NC; ++col)
                            sum[group][col][row] += Fragment::dot(weight, activation[col]);
                    }
                }
            }
        }
    }
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

template<int NC>
void poly_dense_q80_rows_n(const Q80Block* w, const Q81Block* x, float* y,
                           int n_in, int n_out, int rows, cudaStream_t stream) {
    const dim3 threads(32, 4);
    if (rows == 1) {
        const dim3 grid(unsigned((size_t(n_out) + 3) / 4));
        poly_dense_q80_rows_kernel<NC, 1><<<grid, threads, 0, stream>>>(w, x, y, n_in, n_out);
    } else {
        const dim3 grid(unsigned((size_t(n_out) + 15) / 16));
        poly_dense_q80_rows_kernel<NC, 4><<<grid, threads, 0, stream>>>(w, x, y, n_in, n_out);
    }
}

inline bool poly_dense_q80_rows(const void* weights, const void* x, float* y,
                                int n_in, int n_out, int ncols, cudaStream_t stream) {
    static const int rows = [] {
        const char* v = std::getenv("STRATA_POLY_Q80_DENSE_ROWS");
        return v && (v[0] == '1' || v[0] == '4') && v[1] == '\0' ? v[0] - '0' : 2;
    }();
    if (rows == 2 || ncols < 1 || ncols > 8) return false;
    const auto* w = static_cast<const Q80Block*>(weights);
    const auto* q = static_cast<const Q81Block*>(x);
    switch (ncols) {
    case 1: poly_dense_q80_rows_n<1>(w, q, y, n_in, n_out, rows, stream); break;
    case 2: poly_dense_q80_rows_n<2>(w, q, y, n_in, n_out, rows, stream); break;
    case 3: poly_dense_q80_rows_n<3>(w, q, y, n_in, n_out, rows, stream); break;
    case 4: poly_dense_q80_rows_n<4>(w, q, y, n_in, n_out, rows, stream); break;
    case 5: poly_dense_q80_rows_n<5>(w, q, y, n_in, n_out, rows, stream); break;
    case 6: poly_dense_q80_rows_n<6>(w, q, y, n_in, n_out, rows, stream); break;
    case 7: poly_dense_q80_rows_n<7>(w, q, y, n_in, n_out, rows, stream); break;
    case 8: poly_dense_q80_rows_n<8>(w, q, y, n_in, n_out, rows, stream); break;
    }
    return true;
}
#endif
