// Single-row affine int4 matrix-vector product for MTP. Unpack once, accumulate in f32 and return f32.
// Uses the qmv_fast-style lane layout (16 K values/lane), as in MLX and oMLX m1_verify_qmv.py.
const int K = X_shape[1], N = Wq_shape[0], KW = Wq_shape[1], KG = SC_shape[1], GS = K / KG;
const int lane = int(thread_index_in_simdgroup);
const int sg = int(simdgroup_index_in_threadgroup);
const int base = int(threadgroup_position_in_grid.x) * 8 + sg * 4;
float result[4] = {0.0f,0.0f,0.0f,0.0f};
for (int k = 0; k < K; k += 512) {
    const int g = (k + lane * 16) / GS;
    half4 wf[4][4];
    float scales[4], biases[4];
    #pragma clang loop unroll(full)
    for (int r = 0; r < 4; ++r) {
        const int n = min(base + r, N - 1);
        const device uint* p = Wq + n * KW + k / 8 + lane * 2;
        const uint a = p[0], b = p[1];
        wf[r][0] = half4(half(a & 15),half((a >> 4) & 15),half((a >> 8) & 15),half((a >> 12) & 15));
        wf[r][1] = half4(half((a >> 16) & 15),half((a >> 20) & 15),half((a >> 24) & 15),half((a >> 28) & 15));
        wf[r][2] = half4(half(b & 15),half((b >> 4) & 15),half((b >> 8) & 15),half((b >> 12) & 15));
        wf[r][3] = half4(half((b >> 16) & 15),half((b >> 20) & 15),half((b >> 24) & 15),half((b >> 28) & 15));
        scales[r] = float(SC[n * KG + g]);
        biases[r] = float(BI[n * KG + g]);
    }
    float acc[4] = {0.0f,0.0f,0.0f,0.0f};
    float4 sum4 = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 4; ++i) {
        const float4 x = float4(*(const device half4*)(X + k + lane * 16 + i * 4));
        sum4 += x;
        #pragma clang loop unroll(full)
        for (int r = 0; r < 4; ++r) acc[r] += dot(x, float4(wf[r][i]));
    }
    const float sum = (sum4.x + sum4.y) + (sum4.z + sum4.w);
    #pragma clang loop unroll(full)
    for (int r = 0; r < 4; ++r) result[r] += scales[r] * acc[r] + biases[r] * sum;
}
#pragma clang loop unroll(full)
for (int r = 0; r < 4; ++r) {
    const float v = simd_sum(result[r]);
    if (lane == 0 && base + r < N) OUT[base + r] = v;
}
