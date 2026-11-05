// Exact sequential gated delta rule. One simdgroup owns a value channel; its 128-dimensional state stays in
// registers across every token of the segment. All products, reductions and updates accumulate in f32.
const int hv = ST_shape[0], dv = ST_shape[1], dk = ST_shape[2];
const int lk = KQ_shape[1], len = E_shape[0];
const int h = int(thread_position_in_grid.y);
const int row = int(thread_position_in_grid.x) / 32;
const int lane = int(thread_index_in_simdgroup);
const int kh = h / (hv / lk);
float st[4];
#pragma clang loop unroll(full)
for (int i = 0; i < 4; ++i) {
    const int d = lane + 32 * i;
    st[i] = d < dk ? ST[(h * dv + row) * dk + d] : 0.0f;
    if (AT[0] == 0 && d < dk) CK[(h * dv + row) * dk + d] = st[i];
}
for (int t = 0; t < len; ++t) {
    float k[4], q[4];
    float sk = 0.0f, sq = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 4; ++i) {
        const int d = lane + 32 * i;
        k[i] = d < dk ? KQ[((t * lk + kh) * dk + d) * 2] : 0.0f;
        q[i] = d < dk ? KQ[((t * lk + kh) * dk + d) * 2 + 1] : 0.0f;
        sk = fma(st[i], k[i], sk);
        sq = fma(st[i], q[i], sq);
    }
    sk = simd_sum(sk);
    sq = simd_sum(sq);
    const float e = E[t * hv + h], b = BETA[t * hv + h];
    const float delta = b * (V[(t * hv + h) * dv + row] - e * sk);
    if (lane == 0) OUT[(t * hv + h) * dv + row] = fma(delta, DOT[t * lk + kh], e * sq);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 4; ++i) {
        st[i] = fma(delta, k[i], e * st[i]);
        const int d = lane + 32 * i;
        if (t + 1 == AT[0] && d < dk) CK[(h * dv + row) * dk + d] = st[i];
    }
}
#pragma clang loop unroll(full)
for (int i = 0; i < 4; ++i) {
    const int d = lane + 32 * i;
    if (d < dk) SF[(h * dv + row) * dk + d] = st[i];
}
