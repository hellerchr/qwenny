// SPDX-License-Identifier: Apache-2.0
// Adapted from oMLX qwen35_packed_linear.py (Apache-2.0): tile-contiguous checkpoint int4 codes and
// Metal 4 tensor ops. Native group-64/128 scales, half inputs, and float products/outputs; no requantization.
constexpr int TILE_N = 128;
constexpr int GROUPS = K / GS;
constexpr int GPS = GROUPS / KSPLIT;
constexpr int GPP = GPS / PARTS;
const int M = X_shape[0];
threadgroup float sums[GPS * 8];
threadgroup float partials[PARTS * 8 * TILE_N];
const uint tile = threadgroup_position_in_grid.x;
const uint split = threadgroup_position_in_grid.y;
const uint lane = thread_index_in_simdgroup;
const uint simd = simdgroup_index_in_threadgroup;
const uint part = simd / 8;
const uint gs = split * GPS;
const uint g0 = gs + part * GPP;
const uint n0 = tile * TILE_N;
const uint tid = simd * 32 + lane;
constexpr uint NT = PARTS * 8 * 32;
for (uint idx = simd; idx < 8 * GPS; idx += PARTS * 8) {
    const uint row = idx % 8;
    const uint gl = idx / 8;
    float v = 0.0f;
    if (row < M) {
        const device half* xr = X + row * K + (gs + gl) * GS;
        for (uint i = lane; i < GS; i += 32) v += float(xr[i]);
        v = simd_sum(v);
    }
    if (lane == 0) sums[gl * 8 + row] = v;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// Packed [tile][64-wide K block][column][32 bytes], nibbles unchanged.
const device uchar* wtile = (const device uchar*)W + ulong(tile) * (K / 64) * TILE_N * 32;
auto a = tensor((device half*)X, dextents<int, 2>{K, M}, array<int, 2>{1, K});
constexpr auto desc = matmul2d_descriptor(8, TILE_N, 64, false, true, false);
matmul2d<desc, execution_simdgroups<8>> op;
tensor<device uint4b_format, dextents<int, 2>, tensor_inline> bfirst(
    (device uchar*)wtile, dextents<int, 2>{64, TILE_N}, array<int, 2>{1, 64});
auto a_first = a.template slice<64, 8>(0, 0);
auto b_first = bfirst.template slice<64, TILE_N>(0, 0);
auto acc = op.template get_destination_cooperative_tensor<decltype(a_first), decltype(b_first), float>();
for (ushort i = 0; i < 4; ++i) acc[i] = 0.0f;
const auto first = acc.get_multidimensional_index(ushort(0));
const uint my_col = first[0];
const uint my_row = first[1];
for (uint gi = 0; gi < GPP; ++gi) {
    const uint g = g0 + gi;
    decltype(acc) dot;
    for (ushort i = 0; i < 4; ++i) dot[i] = 0.0f;
    for (uint j = 0; j < GS / 64; ++j) {
        auto as = a.template slice<64, 8>(g * GS + j * 64, 0);
        tensor<device uint4b_format, dextents<int, 2>, tensor_inline> bt(
            (device uchar*)wtile + ulong(g * (GS / 64) + j) * TILE_N * 32,
            dextents<int, 2>{64, TILE_N}, array<int, 2>{1, 64});
        auto bs = bt.template slice<64, TILE_N>(0, 0);
        decltype(acc) p;
        op.run(as, bs, p);
        for (ushort i = 0; i < 4; ++i) dot[i] += p[i];
    }
    const float xs = sums[(g - gs) * 8 + my_row];
    const uint at = (tile * GROUPS + g) * TILE_N + my_col;
    const float4 sc = float4(*(const device half4*)(SC + at));
    const float4 bi = float4(*(const device half4*)(BI + at));
    for (ushort i = 0; i < 4; ++i) acc[i] += dot[i] * sc[i] + xs * bi[i];
}
*(threadgroup float4*)(partials + (part * 8 + my_row) * TILE_N + my_col) =
    float4(acc[0], acc[1], acc[2], acc[3]);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = tid; i < uint(M) * TILE_N; i += NT) {
    const uint row = i / TILE_N, col = i % TILE_N;
    float v = 0.0f;
    for (uint p = 0; p < PARTS; ++p) v += partials[(p * 8 + row) * TILE_N + col];
    OUT[(split * M + row) * N + n0 + col] = v;
}
