#pragma once
#include <type_traits>
// Full attention over the KV cache, GQA with NH query and NKV key/value heads of HD dims (the including file defines
// them and attn_prep). The KV cache is int8 with an f32 scale per key (and value) and head,
// [slot][NKV][cap] (k = q * scale, q = round(k / scale), scale = max |k| / 127 over the head's dims), laid out for the
// mma fragments of the decode attention: keys [slot][NKV][cap][HD] with the dims of each 64-dim block reordered
// (kslot), values in blocks of 16 keys [slot][NKV][cap / 16][HD][16] with the keys of a block reordered (vslot). A step
// runs attn_prep (q, k, v of every token; k, v into the cache; into the output what the attention multiplies it by:
// the sigmoid gate), then the decode tokens (attn_mq by key splits, attn_combine) and the prompt tiles
// (attn_prefill on tensor cores), which multiply their result into that output. Scores are in base 2: attn_prep scales
// q by log2(e) over sqrt(HD).

#define KPS 32          // decode: keys per split at least (the host sizes grids the same way: attn.rs)
#define PK 64  // prefill: keys per stage
#define SROW (HD + 8)   // prefill: shared row stride in halves (528 bytes: ldmatrix rows on distinct banks)

static_assert(HD % 64 == 0, "lane layouts assume blocks of 64 dims in a head");

__device__ __forceinline__ size_t cache_head(int slot, int g, int cap) { return ((size_t)slot * NKV + g) * cap * HD; }
// index of a key's scale: cache_head / HD + position
__device__ __forceinline__ size_t scale_head(int slot, int g, int cap) { return ((size_t)slot * NKV + g) * cap; }

// Byte of dim d in a key row: byte b * 64 + t4 * 16 + s * 4 + e holds dim b * 64 + s * 16 + 2 t4 + (e & 1) + 8 (e >> 1):
// lane t4 of a quad reads the B fragments of 4 k16 steps (dims 2 t4, 2 t4 + 1, 2 t4 + 8, 2 t4 + 9 each) as 16 bytes
__device__ __forceinline__ int kslot(int d) {
    int b = d >> 6, w = d & 15, t4 = (w & 7) >> 1, e = (w & 1) | ((w >> 3) << 1);
    return b * 64 + t4 * 16 + ((d & 63) >> 4) * 4 + e;
}
// Byte of (position, dim) in a head's values: blocks of 16 keys, [block][dim][16]; key j of a block at byte t4 * 4 + e
// for j = 2 t4 + (e & 1) + 8 (e >> 1): lane t4 reads the B fragment of one n8 tile of dims (keys 2 t4, 2 t4 + 1,
// 2 t4 + 8, 2 t4 + 9) as 4 bytes
__device__ __forceinline__ size_t vslot(int pos, int d) {
    int j = pos & 15, t4 = (j & 7) >> 1, e = (j & 1) | ((j >> 3) << 1);
    return ((size_t)(pos >> 4) * HD + d) * 16 + t4 * 4 + e;
}
// The cache holds q + 128. Two of its bytes (sel) -> half2 1152 + q, exactly, in one prmt; the attention
// kernels subtract 1152 times the row sums of Q (scores) and of P (outputs) instead of converting every value
#define KV_OFF 1152.f
__device__ __forceinline__ uint32_t kv_half2(uint32_t w, uint32_t sel) { return __byte_perm(w, 0x64646464u, sel); }

// Keys per split of a decode token with `keys` keys: a multiple of 16, at least KPS, at most DSPLITS splits. A function of
// `keys` alone, so a token gets the same splits (the same sums) in any step: a draft as with one token per step. The
// size changes every 16 * DSPLITS keys; Step::upload cuts a short segment where it changes (Segments::cut).
#define DSPLITS 128
__device__ __forceinline__ int dchunk(int keys) { return max(KPS, ((keys + DSPLITS - 1) / DSPLITS + 15) / 16 * 16); }
__device__ __forceinline__ int nsplits(int keys, int z) { int c = dchunk(keys); return min(z, (keys + c - 1) / c); }

// A key (or value) head of a token into the cache, one warp, the lane's HD / 32 values v[j] of dims lane + 32 j: int8
// with the head's scale at this position, in the cache's layout (kslot, vslot)
__device__ __forceinline__ void kv_store(const float* v, bool key, int8_t* kc, int8_t* vc, float* ks, float* vs,
                                         int lane, int slot, int g, int pos, int cap) {
    int8_t* c = (key ? kc : vc) + cache_head(slot, g, cap);
    float mx = 0.f;
#pragma unroll
    for (int j = 0; j < HD / 32; j++) mx = fmaxf(mx, fabsf(v[j]));
    mx = warp_max(mx);
    float inv = mx > 0.f ? 127.f / mx : 0.f;
#pragma unroll
    for (int j = 0; j < HD / 32; j++) {
        int d = lane + 32 * j;
        c[key ? (size_t)pos * HD + kslot(d) : vslot(pos, d)] = (int8_t)(__float2int_rn(v[j] * inv) + 128);
    }
    if (lane == 0) (key ? ks : vs)[scale_head(slot, g, cap) + pos] = mx / 127.f;
}

// Decode attention for short segments (a decoding token and its drafts) on tensor cores. Block (seg, g, s): the rows
// (token, head) of segment `seg` (dseg: first index into dtok, tokens) for the GROUP query heads of kv head g, in RT
// tiles of 16 (row = token * GROUP + head), against key split s of the segment's last token (splits on 16-key blocks).
// Warp wt owns row tile wt over all head dims. Per block of 16 keys: S = Q K^T (Q f16 in shared memory via ldmatrix;
// K fragments read straight from the cache, one 16-byte load per lane and 64 dims, kslot), scaled per key and masked by
// each row's position, online softmax in the accumulator layout, then O += P V (P from S's accumulators, V fragments
// as 4-byte loads, vslot) with the value scales folded into P. The warps of a block read the same K and V (L1 serves
// all but the first); no tile computes another's S, and the small block (shared memory for Q only) keeps many blocks
// resident. One split: the block finishes its rows (out *= normalized result); else it writes the partials (max, sum,
// acc) for attn_combine.
#define MQ 8  // most tokens of a short segment (segments() in attn.rs)
template <int RT>
__device__ __forceinline__ void attn_mq_tc(const float* q, const int8_t* kc, const int8_t* vc, const float* ks,
                                           const float* vs, const int* dseg, const int* dtok, const int* tok_slot,
                                           const int* tok_pos, int cap, int z, float* part_ml, float* part_acc,
                                           float* out) {
    constexpr int NTD = HD / 8;  // n8 dim tiles
    extern __shared__ __align__(16) __half qsh[];  // [RT * 16][SROW]
    int seg = blockIdx.x, g = blockIdx.y, s = blockIdx.z, wt = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int r = lane >> 2, t4 = lane & 3;
    int first = dseg[2 * seg], nt = dseg[2 * seg + 1];
    int tlast = dtok[first + nt - 1], keys = tok_pos[tlast] + 1, ns = nsplits(keys, z);
    if (s >= ns) return;
    int chunk = dchunk(keys), kbeg = s * chunk, ke = min(keys, kbeg + chunk);
    for (int x = threadIdx.x; x < RT * 16 * HD / 4; x += 32 * RT) {
        int phys = x / (HD / 4), row = phys, d = (x % (HD / 4)) * 4, tok = row / GROUP;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tok < nt) v = *(const float4*)(q + ((size_t)dtok[first + tok] * NH + g * GROUP + row % GROUP) * HD + d);
        *(uint2*)&qsh[phys * SROW + d] = make_uint2(pack_half2(v.x, v.y), pack_half2(v.z, v.w));
    }
    __syncthreads();
    // this lane's rows (wt * 16 + r and + 8): the keys they see end at lim; KV_OFF times their sums of Q
    int lim[2];
    float qoff[2];
    {
        // lane l sums half of row l % 16; lanes l and l ^ 16 together the whole row
        float qs = 0.f;
        const __half* qr = qsh + (wt * 16 + (lane & 15)) * SROW + (lane >> 4) * (HD / 2);
        for (int d = 0; d < HD / 2; d += 2) {
            float2 f = __half22float2(*(const __half2*)(qr + d));
            qs += f.x + f.y;
        }
        qs += __shfl_xor_sync(0xffffffff, qs, 16);
        qoff[0] = KV_OFF * __shfl_sync(0xffffffff, qs, r);
        qoff[1] = KV_OFF * __shfl_sync(0xffffffff, qs, r + 8);
    }
#pragma unroll
    for (int h2 = 0; h2 < 2; h2++) {
        int tok = (wt * 16 + r + 8 * h2) / GROUP;
        lim[h2] = tok < nt ? min(ke, tok_pos[dtok[first + tok]] + 1) : 0;
    }
    float m[2], l[2], o[NTD][4], ps[2];  // ps: this lane's sum of the f16 P it fed (KV_OFF fold)
    m[0] = m[1] = -INFINITY;
    l[0] = l[1] = 0.f;
    ps[0] = ps[1] = 0.f;
#pragma unroll
    for (int j = 0; j < NTD; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[j][e] = 0.f;
    int slot = tok_slot[tlast];
    const int8_t* kh = kc + cache_head(slot, g, cap);
    const int8_t* vh = vc + cache_head(slot, g, cap);
    const float *ksh = ks + scale_head(slot, g, cap), *vsh = vs + scale_head(slot, g, cap);
    const __half* qa = qsh + (wt * 16 + (lane & 15)) * SROW + (lane >> 4) * 8;
    for (int kb = kbeg; kb < ke; kb += 16) {
        // K fragments (keys kb + 8 n + r), V words (32 lanes: contiguous 128-byte runs), scales
        uint4 kf[2][HD / 64];
#pragma unroll
        for (int n = 0; n < 2; n++)
#pragma unroll
            for (int b = 0; b < HD / 64; b++)
                kf[n][b] = *(const uint4*)(kh + (size_t)(kb + 8 * n + r) * HD + 64 * b + 16 * t4);
        uint32_t vw[NTD];
        const int8_t* vp = vh + (size_t)kb * HD + r * 16 + t4 * 4;
#pragma unroll
        for (int j = 0; j < NTD; j++) vw[j] = *(const uint32_t*)(vp + j * 8 * 16);
        float ksc[2][2], vsc[2][2];
#pragma unroll
        for (int n = 0; n < 2; n++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                int k = kb + 8 * n + 2 * t4 + e;
                ksc[n][e] = k < ke ? ksh[k] : 0.f;
                vsc[n][e] = k < ke ? vsh[k] : 0.f;
            }
        // S = Q K^T
        float sacc[2][4];
#pragma unroll
        for (int n = 0; n < 2; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) sacc[n][e] = 0.f;
#pragma unroll
        for (int kstep = 0; kstep < HD / 16; kstep++) {
            uint32_t bf[2][2];
#pragma unroll
            for (int n = 0; n < 2; n++) {
                uint4 u = kf[n][kstep >> 2];
                uint32_t w = (kstep & 3) == 0 ? u.x : (kstep & 3) == 1 ? u.y : (kstep & 3) == 2 ? u.z : u.w;
                bf[n][0] = kv_half2(w, 0x4140);
                bf[n][1] = kv_half2(w, 0x4342);
            }
            uint32_t a[4];
            ldsm_x4(a, qa + kstep * 16);
#pragma unroll
            for (int n = 0; n < 2; n++) mma16816(sacc[n], a, bf[n][0], bf[n][1]);
        }
        // scale, mask, online softmax per row; P (times the value scales) as the A fragments of P V
        uint32_t pa[4];
        float pv[2][4];
#pragma unroll
        for (int h2 = 0; h2 < 2; h2++) {
            float v[2][2], mx = -INFINITY;
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    int k = kb + 8 * n + 2 * t4 + e;
                    v[n][e] = k < lim[h2] ? (sacc[n][2 * h2 + e] - qoff[h2]) * ksc[n][e] : -INFINITY;
                    mx = fmaxf(mx, v[n][e]);
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
            float mn = fmaxf(m[h2], mx);
            float corr = mn == -INFINITY ? 1.f : exp2f(m[h2] - mn);
            m[h2] = mn;
            float sum = 0.f;
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    float p = v[n][e] == -INFINITY ? 0.f : exp2f(v[n][e] - mn);
                    sum += p;
                    pv[n][2 * h2 + e] = p * vsc[n][e];
                }
            l[h2] = l[h2] * corr + sum;
            ps[h2] *= corr;
            if (__any_sync(0xffffffff, corr != 1.f)) {  // only when some row's max grew
#pragma unroll
                for (int j = 0; j < NTD; j++) {
                    o[j][2 * h2] *= corr;
                    o[j][2 * h2 + 1] *= corr;
                }
            }
        }
        pa[0] = pack_half2(pv[0][0], pv[0][1]);
        pa[1] = pack_half2(pv[0][2], pv[0][3]);
        pa[2] = pack_half2(pv[1][0], pv[1][1]);
        pa[3] = pack_half2(pv[1][2], pv[1][3]);
        // the P the products see (f16): rows r (words 0, 2) and r + 8 (1, 3)
        float2 f0 = __half22float2(*(__half2*)&pa[0]), f2 = __half22float2(*(__half2*)&pa[2]);
        float2 f1 = __half22float2(*(__half2*)&pa[1]), f3 = __half22float2(*(__half2*)&pa[3]);
        ps[0] += f0.x + f0.y + f2.x + f2.y;
        ps[1] += f1.x + f1.y + f3.x + f3.y;
        // O += P V
#pragma unroll
        for (int j = 0; j < NTD; j++) mma16816(o[j], pa, kv_half2(vw[j], 0x4140), kv_half2(vw[j], 0x4342));
    }
#pragma unroll
    for (int h2 = 0; h2 < 2; h2++) {
        float L = l[h2], P = ps[h2];
        L += __shfl_xor_sync(0xffffffff, L, 1);
        L += __shfl_xor_sync(0xffffffff, L, 2);
        P += __shfl_xor_sync(0xffffffff, P, 1);
        P += __shfl_xor_sync(0xffffffff, P, 2);
#pragma unroll
        for (int j = 0; j < NTD; j++) {  // O = O' - KV_OFF * sum P
            o[j][2 * h2] -= KV_OFF * P;
            o[j][2 * h2 + 1] -= KV_OFF * P;
        }
        int row = wt * 16 + r + 8 * h2, tok = row / GROUP;
        if (tok >= nt) continue;
        int i = first + tok, t = dtok[i], head = g * GROUP + row % GROUP;
        if (ns == 1) {
            float f = 1.f / L;
            float* op = out + (size_t)t * MIX + head * HD + 2 * t4;
#pragma unroll
            for (int j = 0; j < NTD; j++) {
                float2 gt = *(float2*)(op + j * 8);
                *(float2*)(op + j * 8) = make_float2(gt.x * o[j][2 * h2] * f, gt.y * o[j][2 * h2 + 1] * f);
            }
        } else {
            size_t pi = ((size_t)i * NH + head) * z + s;
            float* pp = part_acc + pi * HD + 2 * t4;
#pragma unroll
            for (int j = 0; j < NTD; j++) *(float2*)(pp + j * 8) = make_float2(o[j][2 * h2], o[j][2 * h2 + 1]);
            if (t4 == 0) { part_ml[pi * 2] = m[h2]; part_ml[pi * 2 + 1] = L; }
        }
    }
}
#define ATTN_MQ(NAME, RT)                                                                                              \
    extern "C" __global__ void __launch_bounds__(32 * RT)                                                               \
        NAME(const float* q, const int8_t* kc, const int8_t* vc, const float* ks, const float* vs, const int* dseg,   \
             const int* dtok, const int* tok_slot, const int* tok_pos, int cap, int z, float* part_ml,                \
             float* part_acc, float* out) {                                                                            \
        attn_mq_tc<RT>(q, kc, vc, ks, vs, dseg, dtok, tok_slot, tok_pos, cap, z, part_ml, part_acc, out);             \
    }
ATTN_MQ(attn_mq1, 1)
ATTN_MQ(attn_mq2, 2)
ATTN_MQ(attn_mq3, 3)
static_assert(MQ * GROUP <= 48, "a short segment's rows fit three tiles");

// Merge the ns key splits of one query row and head (partial results at base, base + 1, ...) into out (HD values,
// times the gate there); HD threads. Warp 0 turns the splits' maxima into weights (shared), then thread x sums 4 dims
// over a quarter of the splits.
#define MAX_SPLITS 128  // most splits per token or tile (DSPLITS, TSPLITS in attn.rs)
__device__ __forceinline__ void combine_row(const float* part_ml, const float* part_acc, size_t base, int ns,
                                            float* out) {
    __shared__ float se[MAX_SPLITS], sL;
    __shared__ __align__(16) float4 sa[4][HD / 4];
    int d = threadIdx.x;
    if (d < 32) {
        float ms[MAX_SPLITS / 32], M = -INFINITY, L = 0.f;
#pragma unroll
        for (int j = 0; j < MAX_SPLITS / 32; j++) {
            int s = d + 32 * j;
            ms[j] = s < ns ? part_ml[(base + s) * 2] : -INFINITY;
            M = fmaxf(M, ms[j]);
        }
        M = warp_max(M);
#pragma unroll
        for (int j = 0; j < MAX_SPLITS / 32; j++) {
            int s = d + 32 * j;
            if (s < ns) {
                float e = ms[j] == -INFINITY ? 0.f : exp2f(ms[j] - M);
                se[s] = e;
                L += e * part_ml[(base + s) * 2 + 1];
            }
        }
        L = warp_sum(L);
        if (d == 0) sL = L;
    }
    __syncthreads();
    int d4 = d % (HD / 4), qq = d / (HD / 4);
    float4 A = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 8
    for (int s = qq; s < ns; s += 4) {
        float e = se[s];
        float4 a = *(const float4*)(part_acc + (base + s) * HD + 4 * d4);
        A.x += e * a.x; A.y += e * a.y; A.z += e * a.z; A.w += e * a.w;
    }
    sa[qq][d4] = A;
    __syncthreads();
    if (qq == 0) {
        float f = 1.f / sL;
        float4 b = sa[1][d4], c = sa[2][d4], e = sa[3][d4];
        float4* op = (float4*)(out + 4 * d4);
        float4 gt = *op;
        *op = make_float4(gt.x * (A.x + b.x + c.x + e.x) * f, gt.y * (A.y + b.y + c.y + e.y) * f,
                          gt.z * (A.z + b.z + c.z + e.z) * f, gt.w * (A.w + b.w + c.w + e.w) * f);
    }
}

// Merge the key splits of the decode tokens with more than one: grid (decode tokens, NH), HD threads.
extern "C" __global__ void __launch_bounds__(HD) attn_combine(const float* part_ml, const float* part_acc,
                                                              const int* dtok, const int* dkeys, int z, float* out) {
    int i = blockIdx.x, h = blockIdx.y, t = dtok[i];
    int ns = nsplits(dkeys[i], z);  // the splits of its segment (attn_mq)
    if (ns == 1) return;  // attn_mq finished it
    combine_row(part_ml, part_acc, ((size_t)i * NH + h) * z, ns, out + (size_t)t * MIX + h * HD);
}

#define PF_WARPS 4
#define PF_ROWS (16 * PF_WARPS)
#define PF_TT (PF_ROWS / GROUP)  // tokens per prompt tile (attn.rs: segments)

// Key splits of a prompt tile whose last key is keys - 1 when the grid has z of them: at least kps keys each, a
// whole number of PK-key blocks (`chunk` keys), none empty.
__device__ __forceinline__ int tile_splits(int keys, int z, int kps, int& chunk) {
    int ns = min(z, (keys + kps - 1) / kps);
    chunk = ((keys + ns - 1) / ns + PK - 1) / PK * PK;
    return (keys + chunk - 1) / chunk;
}

// Merge the key splits of the prompt tiles with more than one: grid (tiles * PF_TT tokens, NH), HD threads.
extern "C" __global__ void __launch_bounds__(HD) attn_combine_tiles(const float* part_ml, const float* part_acc,
                                                                    const int* ptile, const int* tok_pos, int z,
                                                                    int kps, float* out) {
    int tile = blockIdx.x / PF_TT, row = blockIdx.x % PF_TT, h = blockIdx.y;
    int t0 = ptile[2 * tile], len = ptile[2 * tile + 1], chunk;
    if (row >= len) return;
    int ns = tile_splits(tok_pos[t0] + len, z, kps, chunk);
    if (ns == 1) return;  // attn_prefill finished it
    combine_row(part_ml, part_acc, ((size_t)blockIdx.x * NH + h) * z, ns, out + (size_t)(t0 + row) * MIX + h * HD);
}

// Prompt attention (pth-flash-attn's shape): block (tile, g, s) of PF_WARPS warps over PF_ROWS rows, the (token,
// query head) pairs of the tile's up to PF_TT tokens for the GROUP query heads of kv head g, row = token * GROUP +
// head (GQA packing: every row reads the same keys), against key split s (keys from kbeg, a whole number of PK-key
// stages). The int8 keys and values of a stage come in with cp.async (PF_STAGES stages in flight, shared by the
// warps); each warp reads its mma fragments from them as the decode attention does (kslot, vslot), converts them in
// registers, scales S per key and folds the value scales into P. Q in shared memory as f16 (ldmatrix), online
// softmax in base 2, causal mask by position. One split: the block finishes its rows (out *= normalized result);
// else it writes the partials for attn_combine_tiles. Dynamic shared memory: PREFILL_SMEM bytes.
#define PF_STAGES 2
#define PREFILL_SMEM (PF_ROWS * SROW * 2 + PF_STAGES * 2 * PK * HD)
static_assert(PREFILL_SMEM <= 99 * 1024 && PK % 16 == 0 && PF_TT <= 16, "prefill attention: shared memory, keys per stage");

extern "C" __global__ void __launch_bounds__(32 * PF_WARPS)
    attn_prefill(const float* q, const int8_t* kc, const int8_t* vc, const float* ks, const float* vs, const int* ptile,
                 const int* tok_slot, const int* tok_pos, int cap, int z, int kps, float* part_ml,
                 float* part_acc, float* out) {
    extern __shared__ __align__(16) __half smem[];
    int tile = gridDim.x - 1 - blockIdx.x;  // the later tiles (more keys) first
    int t0 = ptile[2 * tile], len = ptile[2 * tile + 1], g = blockIdx.y, s = blockIdx.z;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r = lane >> 2, t4 = lane & 3;
    int slot = tok_slot[t0];
    int p0 = tok_pos[t0], plast = p0 + len - 1, chunk, ns = tile_splits(plast + 1, z, kps, chunk);
    if (s >= ns) return;
    int kbeg = s * chunk, kend = min(plast + 1, kbeg + chunk);  // a split ends on a stage boundary or at plast
    __half* Qs = smem;                                   // [PF_ROWS][SROW]
    int8_t* stg = (int8_t*)(smem + PF_ROWS * SROW);      // stage b: PK K rows, then PK / 16 V blocks
    size_t cb = cache_head(slot, g, cap), sb = scale_head(slot, g, cap);
    const int8_t *kh = kc + cb, *vh = vc + cb;
    auto fetch = [&](int k0, int b) {  // keys [k0, k0 + PK): both 16-byte aligned runs of PK * HD bytes
        int8_t* d = stg + b * 2 * PK * HD;
        for (int x = threadIdx.x; x < 2 * PK * HD / 16; x += 32 * PF_WARPS)
            cp_async16(d + 16 * x, x < PK * HD / 16 ? kh + (size_t)k0 * HD + 16 * x : vh + (size_t)k0 * HD + 16 * x - PK * HD);
        asm volatile("cp.async.commit_group;\n" ::);
    };
    int nst = (kend - kbeg + PK - 1) / PK;
    if (nst > 0) fetch(kbeg, 0);
    else asm volatile("cp.async.commit_group;\n" ::);
    // the block's rows of Q as f16
    for (int x = threadIdx.x; x < PF_ROWS * HD / 4; x += 32 * PF_WARPS) {
        int row = x / (HD / 4), d = (x % (HD / 4)) * 4, tok = row / GROUP;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tok < len && row < PF_TT * GROUP)
            v = *(const float4*)(q + ((size_t)(t0 + tok) * NH + g * GROUP + row % GROUP) * HD + d);
        *(uint2*)&Qs[row * SROW + d] = make_uint2(pack_half2(v.x, v.y), pack_half2(v.z, v.w));
    }
    __syncthreads();
    const __half* Qw = Qs + warp * 16 * SROW;  // this warp's 16 rows
    // KV_OFF times the sums of this lane's rows r and r + 8 of Q (lanes l and l ^ 16 sum halves of row l % 16)
    float qoff[2];
    {
        float qs = 0.f;
        const __half* qr = Qw + (lane & 15) * SROW + (lane >> 4) * (HD / 2);
        for (int d = 0; d < HD / 2; d += 2) {
            float2 f = __half22float2(*(const __half2*)(qr + d));
            qs += f.x + f.y;
        }
        qs += __shfl_xor_sync(0xffffffff, qs, 16);
        qoff[0] = KV_OFF * __shfl_sync(0xffffffff, qs, r);
        qoff[1] = KV_OFF * __shfl_sync(0xffffffff, qs, r + 8);
    }
    float o[HD / 8][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f}, ps[2] = {0.f, 0.f};
#pragma unroll
    for (int j = 0; j < HD / 8; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[j][e] = 0.f;
    // rows r and r + 8 of this warp see the keys below lim (none for rows past the tile)
    int lim[2];
#pragma unroll
    for (int h2 = 0; h2 < 2; h2++) {
        int row = warp * 16 + r + 8 * h2, tok = row / GROUP;
        lim[h2] = tok < len && row < PF_TT * GROUP ? min(kend, p0 + tok + 1) : 0;
    }
    const __half* qa = Qw + (lane & 15) * SROW + (lane >> 4) * 8;
    for (int it = 0; it < nst; it++) {
        int k0 = kbeg + PK * it;
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();  // stage it is in; everyone is done with the other buffer (stage it - 1)
        if (it + 1 < nst) fetch(k0 + PK, (it + 1) % PF_STAGES);
        const int8_t* st = stg + (it % PF_STAGES) * 2 * PK * HD;
        // S = Q K^T for the stage's PK keys (key k0 + 8 n + r on lane quad r): PK / 8 key tiles
        constexpr int NK = PK / 8;
        float sacc[NK][4];
#pragma unroll
        for (int n = 0; n < NK; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) sacc[n][e] = 0.f;
#pragma unroll
        for (int b = 0; b < HD / 64; b++) {
            uint4 kf[NK];
#pragma unroll
            for (int n = 0; n < NK; n++) kf[n] = *(const uint4*)(st + (8 * n + r) * HD + 64 * b + 16 * t4);
#pragma unroll
            for (int k4 = 0; k4 < 4; k4++) {
                uint32_t a[4];
                ldsm_x4(a, qa + (4 * b + k4) * 16);
#pragma unroll
                for (int n = 0; n < NK; n++) {
                    uint32_t w = k4 == 0 ? kf[n].x : k4 == 1 ? kf[n].y : k4 == 2 ? kf[n].z : kf[n].w;
                    mma16816(sacc[n], a, kv_half2(w, 0x4140), kv_half2(w, 0x4342));
                }
            }
        }
        float pv[NK][4];
        float2 ksc[NK], vsc[NK];  // the scales of keys 2 t4, 2 t4 + 1 of each tile, in pairs
#pragma unroll
        for (int n = 0; n < NK; n++) {
            int k = k0 + 8 * n + 2 * t4;
            ksc[n] = *(const float2*)(ks + sb + k);
            vsc[n] = *(const float2*)(vs + sb + k);
        }
        // scale, mask, online softmax; a stage below every row's position needs no mask (FULL)
        auto softmax = [&](auto full) {
            constexpr bool FULL = decltype(full)::value;
#pragma unroll
            for (int h2 = 0; h2 < 2; h2++) {
                float v[NK][2], mx = -INFINITY;
#pragma unroll
                for (int n = 0; n < NK; n++)
#pragma unroll
                    for (int e = 0; e < 2; e++) {
                        int k = k0 + 8 * n + 2 * t4 + e;
                        float x = (sacc[n][2 * h2 + e] - qoff[h2]) * (e ? ksc[n].y : ksc[n].x);
                        v[n][e] = FULL || k < lim[h2] ? x : -INFINITY;
                        mx = fmaxf(mx, v[n][e]);
                    }
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
                float mn = fmaxf(m[h2], mx);
                float corr = mn == -INFINITY ? 1.f : exp2f(m[h2] - mn);
                m[h2] = mn;
                float sum = 0.f;
#pragma unroll
                for (int n = 0; n < NK; n++)
#pragma unroll
                    for (int e = 0; e < 2; e++) {
                        int k = k0 + 8 * n + 2 * t4 + e;
                        float p = FULL ? exp2f(v[n][e] - mn) : v[n][e] == -INFINITY ? 0.f : exp2f(v[n][e] - mn);
                        sum += p;
                        pv[n][2 * h2 + e] = p * (FULL || k < kend ? (e ? vsc[n].y : vsc[n].x) : 0.f);
                    }
                l[h2] = l[h2] * corr + sum;
                ps[h2] *= corr;
                if (__any_sync(0xffffffff, corr != 1.f)) {  // only when some row's max grew
#pragma unroll
                    for (int j = 0; j < HD / 8; j++) {
                        o[j][2 * h2] *= corr;
                        o[j][2 * h2 + 1] *= corr;
                    }
                }
            }
        };
        if (k0 + PK <= min(p0 + 1, kend)) softmax(std::true_type{});
        else softmax(std::false_type{});
        // O += P V, 16 keys (one value block) per k16 step
#pragma unroll
        for (int kq = 0; kq < PK / 16; kq++) {
            uint32_t pa[4] = {pack_half2(pv[2 * kq][0], pv[2 * kq][1]), pack_half2(pv[2 * kq][2], pv[2 * kq][3]),
                              pack_half2(pv[2 * kq + 1][0], pv[2 * kq + 1][1]), pack_half2(pv[2 * kq + 1][2], pv[2 * kq + 1][3])};
            float2 f0 = __half22float2(*(__half2*)&pa[0]), f2 = __half22float2(*(__half2*)&pa[2]);
            float2 f1 = __half22float2(*(__half2*)&pa[1]), f3 = __half22float2(*(__half2*)&pa[3]);
            ps[0] += f0.x + f0.y + f2.x + f2.y;
            ps[1] += f1.x + f1.y + f3.x + f3.y;
            const int8_t* vp = st + PK * HD + 16 * kq * HD + r * 16 + t4 * 4;
#pragma unroll
            for (int j = 0; j < HD / 8; j++) {
                uint32_t w = *(const uint32_t*)(vp + j * 8 * 16);
                mma16816(o[j], pa, kv_half2(w, 0x4140), kv_half2(w, 0x4342));
            }
        }
    }
#pragma unroll
    for (int rr = 0; rr < 2; rr++) {
        l[rr] += __shfl_xor_sync(0xffffffff, l[rr], 1);
        l[rr] += __shfl_xor_sync(0xffffffff, l[rr], 2);
        ps[rr] += __shfl_xor_sync(0xffffffff, ps[rr], 1);
        ps[rr] += __shfl_xor_sync(0xffffffff, ps[rr], 2);
#pragma unroll
        for (int j = 0; j < HD / 8; j++) {  // O = O' - KV_OFF * sum P
            o[j][2 * rr] -= KV_OFF * ps[rr];
            o[j][2 * rr + 1] -= KV_OFF * ps[rr];
        }
    }
#pragma unroll
    for (int rr = 0; rr < 2; rr++) {
        int row = warp * 16 + r + 8 * rr, tok = row / GROUP, head = g * GROUP + row % GROUP;
        if (tok >= len || row >= PF_TT * GROUP) continue;
        if (ns > 1) {
            size_t pi = ((size_t)(tile * PF_TT + tok) * NH + head) * z + s;
            if (t4 == 0) { part_ml[pi * 2] = m[rr]; part_ml[pi * 2 + 1] = l[rr]; }
#pragma unroll
            for (int j = 0; j < HD / 8; j++)
                *(float2*)(part_acc + pi * HD + j * 8 + 2 * t4) = make_float2(o[j][rr * 2], o[j][rr * 2 + 1]);
            continue;
        }
        float f = 1.f / l[rr];
        float* op = out + (size_t)(t0 + tok) * MIX + head * HD + 2 * t4;
#pragma unroll
        for (int j = 0; j < HD / 8; j++) {
            float2 gt = *(float2*)(op + j * 8);
            *(float2*)(op + j * 8) = make_float2(gt.x * o[j][rr * 2] * f, gt.y * o[j][rr * 2 + 1] * f);
        }
    }
}
