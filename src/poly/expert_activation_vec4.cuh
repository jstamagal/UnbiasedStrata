#pragma once

inline bool expert_activation_q8_vec4_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_POLY_EXPERT_ACT_Q8_VEC4");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return on;
}

template<bool GELU>
__global__ void __launch_bounds__(128) expert_activation_q8_vec4_kernel(
        const float* __restrict__ gate_up, ExpertQ81Block* __restrict__ out,
        int n, float limit) {
    const int group = int(threadIdx.x) / 8;
    const int lane = int(threadIdx.x) & 7;
    const int qb = int(blockIdx.x) * 16 + group;
    if (qb >= n / 32) return;
    const int col = qb * 32 + lane * 4;
    const float* gu = gate_up + size_t(blockIdx.y) * 2 * n;
    const float4 gv = *reinterpret_cast<const float4*>(gu + col);
    const float4 uv = *reinterpret_cast<const float4*>(gu + n + col);
    const float g[4] = {gv.x, gv.y, gv.z, gv.w};
    const float u[4] = {uv.x, uv.y, uv.z, uv.w};
    float v[4], sums[4], maxima[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        if constexpr (GELU) {
            v[k] = gelu(g[k]) * u[k];
        } else {
            const float gc = fminf(g[k], limit);
            const float uc = fminf(fmaxf(u[k], -limit), limit);
            v[k] = gc / (1.0f + expf(-gc)) * uc;
        }
        sums[k] = v[k];
        maxima[k] = fabsf(v[k]);
    }
    // The lane-zero sum has the same operation order as offsets 16,8,4,2,1
    // over the original 32 scalar lanes.
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            sums[k] += __shfl_xor_sync(0xffffffff, sums[k], offset, 8);
            maxima[k] = fmaxf(maxima[k], __shfl_xor_sync(0xffffffff, maxima[k], offset, 8));
        }
    }
    const float sum = (sums[0] + sums[2]) + (sums[1] + sums[3]);
    const float amax = fmaxf(fmaxf(maxima[0], maxima[2]), fmaxf(maxima[1], maxima[3]));
    const float d = strata::kernels::q8_1_finite(amax / 127.0f);
    uint32_t codes = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k)
        codes |= uint32_t(uint8_t(strata::kernels::q8_1_quant(v[k], d, amax))) << (k * 8);
    ExpertQ81Block& block = out[size_t(blockIdx.y) * (n / 32) + qb];
    reinterpret_cast<uint32_t*>(block.qs)[lane] = codes;
    if (lane == 0) block.ds = strata::kernels::q8_1_ds(d, sum);
}

inline void expert_activation_q8_vec4(const float* gate_up, void* out, int n,
                                       int rows, float limit, bool use_gelu, void* stream) {
    const dim3 blocks(unsigned((n / 32 + 15) / 16), unsigned(rows));
    if (use_gelu)
        expert_activation_q8_vec4_kernel<true><<<blocks, 128, 0, (cudaStream_t) stream>>>(
            gate_up, static_cast<ExpertQ81Block*>(out), n, limit);
    else
        expert_activation_q8_vec4_kernel<false><<<blocks, 128, 0, (cudaStream_t) stream>>>(
            gate_up, static_cast<ExpertQ81Block*>(out), n, limit);
}
