// Replay accepted tokens and process the next segment in one f32 recurrence. The checkpoint is the state
// after replay; only the new segment's outputs are returned. Identical lane layout and arithmetic to delta8.
const int hv = ST_shape[0], dv = ST_shape[1], dk = ST_shape[2], lk = KQ_shape[1];
const int len = E_shape[0], replay = RP[0];
const int h = int(threadgroup_position_in_grid.y), tid = int(thread_position_in_threadgroup.x);
const int row = int(threadgroup_position_in_grid.x) * 16 + tid / 8, seg = tid % 8;
if (row >= dv) return;
const int kh = h / (hv / lk);
float4 st[4];
#pragma clang loop unroll(full)
for (int i = 0; i < 4; ++i) {
    const int d = 4 * (seg + 8 * i);
    st[i] = *(const device float4*)(ST + (h * dv + row) * dk + d);
    if (replay == 0) *(device float4*)(CK + (h * dv + row) * dk + d) = st[i];
}
for (int t = 0; t < replay + len; ++t) {
    const bool old = t < replay;
    const int p0 = old ? t : t - replay;
    const device float* kp = old ? RKQ : KQ;
    const device float* vp = old ? RV : V;
    const device float* ep = old ? RE : E;
    const device float* bp = old ? RBETA : BETA;
    float4 k[4], q[4];
    float2 sk2 = 0.0f, sq2 = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 4; ++i) {
        const int d = 4 * (seg + 8 * i);
        const device float* p = kp + ((p0 * lk + kh) * dk + d) * 2;
        const float4 a = *(const device float4*)p, b = *(const device float4*)(p + 4);
        k[i] = float4(a.x,a.z,b.x,b.z); q[i] = float4(a.y,a.w,b.y,b.w);
        sk2 = fma(st[i].xy,k[i].xy,sk2); sk2 = fma(st[i].zw,k[i].zw,sk2);
        sq2 = fma(st[i].xy,q[i].xy,sq2); sq2 = fma(st[i].zw,q[i].zw,sq2);
    }
    float sk = sk2.x + sk2.y, sq = sq2.x + sq2.y;
    sk += simd_shuffle_xor(sk,4); sq += simd_shuffle_xor(sq,4);
    sk += simd_shuffle_xor(sk,2); sq += simd_shuffle_xor(sq,2);
    sk += simd_shuffle_xor(sk,1); sq += simd_shuffle_xor(sq,1);
    const float e = ep[p0 * hv + h], b = bp[p0 * hv + h];
    const float delta = b * (vp[(p0 * hv + h) * dv + row] - e * sk);
    if (!old && seg == 0) OUT[(p0 * hv + h) * dv + row] = fma(delta,DOT[p0 * lk + kh],e * sq);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 4; ++i) {
        st[i] = fma(float4(delta),k[i],float4(e) * st[i]);
        const int d = 4 * (seg + 8 * i);
        if (t + 1 == replay) *(device float4*)(CK + (h * dv + row) * dk + d) = st[i];
    }
}
#pragma clang loop unroll(full)
for (int i = 0; i < 4; ++i) {
    const int d = 4 * (seg + 8 * i);
    *(device float4*)(SF + (h * dv + row) * dk + d) = st[i];
}
