// SPDX-License-Identifier: Apache-2.0
// Adapted from oMLX qwen35_verify_qmm.py (sg8), MTPLX (Copyright 2026 Youssof Altoukhi) and Splash's
// linear_q4_sgmatrix.metal (https://github.com/incoai/splash). Exact half operands encode 128 + q;
// accumulation and the affine correction are f32. No quantization or approximate activation is introduced.
const int K = X_shape[1], N = Wq_shape[0], M = X_shape[0];
const int GROUPS = SC_shape[1], GS = K / GROUPS, PER = GROUPS / S;
const int sg = int(simdgroup_index_in_threadgroup), lane = int(thread_index_in_simdgroup);
const int qid = lane >> 2;
const int fm = (qid & 4) | ((lane >> 1) & 3);
const int fn = ((qid & 2) << 1) | ((lane & 1) << 1);
const int kx = (fm >> 1) * 8 + (fm & 1) * 4;
const int nbase = int(threadgroup_position_in_grid.x) * 8 * NT;
const device half* x0 = X + min(fn, M - 1) * K + kx;
const device half* x1 = X + min(fn + 1, M - 1) * K + kx;
float2 result[NT];
for (int c = 0; c < NT; ++c) result[c] = float2(0.0f);
for (int g = sg * PER; g < (sg + 1) * PER; ++g) {
    simdgroup_matrix<float, 8, 8> acc[NT];
    for (int c = 0; c < NT; ++c) acc[c] = simdgroup_matrix<float, 8, 8>(0.0f);
    float2 sum = float2(0.0f);
    for (int block = 0; block < GS / 32; ++block) {
        const int k = g * GS + block * 32;
        const half4 xa = *(const device half4*)(x0 + k);
        const half4 xb = *(const device half4*)(x1 + k);
        for (int j = 0; j < 4; ++j) sum += float2(float(xa[j]), float(xb[j]));
        for (int c = 0; c < NT; ++c) {
            const int n = min(nbase + c * 8 + fm, N - 1);
            const uint word = Wq[n * (K / 8) + g * (GS / 8) + block * 4 + (fn >> 1)];
            #pragma clang loop unroll(full)
            for (int j = 0; j < 4; ++j) {
                simdgroup_matrix<half, 8, 8> a, b;
                const uint pair = (((word >> (4 * j)) & 0x000F000Fu) << 3) | 0x58005800u;
                a.thread_elements()[0] = as_type<half>(ushort(pair));
                a.thread_elements()[1] = as_type<half>(ushort(pair >> 16));
                b.thread_elements()[0] = xa[j];
                b.thread_elements()[1] = xb[j];
                simdgroup_multiply_accumulate(acc[c], a, b, acc[c]);
            }
        }
    }
    sum += float2(simd_shuffle_xor(sum.x, 2), simd_shuffle_xor(sum.y, 2));
    sum += float2(simd_shuffle_xor(sum.x, 4), simd_shuffle_xor(sum.y, 4));
    sum += float2(simd_shuffle_xor(sum.x, 16), simd_shuffle_xor(sum.y, 16));
    for (int c = 0; c < NT; ++c) {
        const int n = min(nbase + c * 8 + fm, N - 1);
        const float sc = float(SC[n * GROUPS + g]), bi = float(BI[n * GROUPS + g]);
        const float2 a = float2(acc[c].thread_elements()[0], acc[c].thread_elements()[1]);
        result[c] += sc * a + (bi - 128.0f * sc) * sum;
    }
}
threadgroup float2 part[NT * S * 32];
for (int c = 0; c < NT; ++c) part[(c * S + sg) * 32 + lane] = result[c];
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    for (int c = 0; c < NT; ++c) {
        float2 sum = float2(0.0f);
        for (int p = 0; p < S; ++p) sum += part[(c * S + p) * 32 + lane];
        const int n = nbase + c * 8 + fm;
        if (n < N && fn < M) OUT[fn * N + n] = sum.x;
        if (n < N && fn + 1 < M) OUT[(fn + 1) * N + n] = sum.y;
    }
}
