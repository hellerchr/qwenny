// Adapted from oMLX m1_verify_mma.py / mlx-dspark skinny_qmm.py.
// MIT License
// Copyright (c) 2026 erahim3
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
// to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions
// of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// qwenny: f16 inputs, group-128 weights, f32 outputs (no half output rounding).
const int K = X_shape[1], N = Wq_shape[0], R = X_shape[0];
const int G = K / 64, KG = SC_shape[1], GS = K / KG;
const int lane = int(thread_index_in_simdgroup);
const int c = int(simdgroup_index_in_threadgroup);
const int fm = ((lane >> 4) << 2) + ((lane >> 1) & 3);
const int fn = (((lane >> 3) & 1) << 2) + ((lane & 1) << 1);
const int n0 = int(threadgroup_position_in_grid.x) * (8 * NT);
const int m0 = int(threadgroup_position_in_grid.y) * (8 * RT);
int nr[NT];
for (int t = 0; t < NT; ++t) nr[t] = min(n0 + 8 * t + fm, N - 1);
const device half* x0[RT];
const device half* x1[RT];
for (int r = 0; r < RT; ++r) {
    x0[r] = X + size_t(min(m0 + 8 * r + fn, R - 1)) * K;
    x1[r] = X + size_t(min(m0 + 8 * r + fn + 1, R - 1)) * K;
}
float acc[RT][NT][2];
for (int r = 0; r < RT; ++r)
    for (int t = 0; t < NT; ++t) { acc[r][t][0] = 0; acc[r][t][1] = 0; }
for (int g = c; g < G; g += S) {
    simdgroup_matrix<float, 8, 8> P[RT][NT];
    float sum0[RT], sum1[RT];
    for (int r = 0; r < RT; ++r) {
        sum0[r] = 0; sum1[r] = 0;
        for (int t = 0; t < NT; ++t) P[r][t] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
    uint4 w[NT];
    for (int t = 0; t < NT; ++t) {
        const int words = B == 4 ? K / 8 : K / 4;
        const device uint* wp = Wq + size_t(nr[t]) * words + (B == 4 ? 8 : 16) * g + (B == 4 ? fn : 2 * fn);
        w[t] = B == 4 ? uint4(wp[0], wp[1], 0, 0) : *(const device uint4*)wp;
    }
    half xa[RT][8], xb[RT][8];
    for (int r = 0; r < RT; ++r) {
        for (int e = 0; e < 8; ++e) {
            xa[r][e] = x0[r][64 * g + 8 * fm + e];
            xb[r][e] = x1[r][64 * g + 8 * fm + e];
            sum0[r] += float(xa[r][e]); sum1[r] += float(xb[r][e]);
        }
    }
    #pragma clang loop unroll(full)
    for (int s = 0; s < 8; ++s) {
        const int shift = B == 4 ? 4 * s : 8 * (s & 3);
        const uint mask = (B == 4 ? 0xFu : 0xFFu) << shift;
        const float pre = as_type<float>(uint(127 - shift) << 23);
        simdgroup_matrix<float, 8, 8> b[RT];
        for (int r = 0; r < RT; ++r) {
            b[r].thread_elements()[0] = float(xa[r][s]) * pre;
            b[r].thread_elements()[1] = float(xb[r][s]) * pre;
        }
        for (int t = 0; t < NT; ++t) {
            simdgroup_matrix<float, 8, 8> a;
            a.thread_elements()[0] = float(w[t][B == 4 ? 0 : s >> 2] & mask);
            a.thread_elements()[1] = float(w[t][B == 4 ? 1 : 2 + (s >> 2)] & mask);
            for (int r = 0; r < RT; ++r) simdgroup_multiply_accumulate(P[r][t], a, b[r], P[r][t]);
        }
    }
    for (int r = 0; r < RT; ++r) {
        for (int d = 2; d <= 16; d <<= 1) {
            if (d == 8) continue;
            sum0[r] += simd_shuffle_xor(sum0[r], ushort(d));
            sum1[r] += simd_shuffle_xor(sum1[r], ushort(d));
        }
    }
    for (int t = 0; t < NT; ++t) {
        const float sc = float(SC[size_t(nr[t]) * KG + (64 * g) / GS]);
        const float bi = float(BI[size_t(nr[t]) * KG + (64 * g) / GS]);
        for (int r = 0; r < RT; ++r) {
            acc[r][t][0] = fma(bi, sum0[r], fma(sc, P[r][t].thread_elements()[0], acc[r][t][0]));
            acc[r][t][1] = fma(bi, sum1[r], fma(sc, P[r][t].thread_elements()[1], acc[r][t][1]));
        }
    }
}
threadgroup float part[S * RT * NT * 64];
for (int r = 0; r < RT; ++r)
    for (int t = 0; t < NT; ++t)
        for (int e = 0; e < 2; ++e) part[((c * RT + r) * NT + t) * 64 + fm * 8 + fn + e] = acc[r][t][e];
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int i = c * 32 + lane; i < RT * NT * 64; i += S * 32) {
    float v = 0;
    for (int q = 0; q < S; ++q) v += part[q * (RT * NT * 64) + i];
    const int r = i / (NT * 64), t = (i / 64) % NT, e = i % 64;
    const int row = m0 + 8 * r + (e % 8), n = n0 + 8 * t + (e / 8);
    if (row < R && n < N) OUT[size_t(row) * N + n] = v;
}
