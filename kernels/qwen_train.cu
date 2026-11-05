// Qwen3.5 LoRA training kernels (host: src/cuda/qwen/train.rs): the element-wise parts of a layer's forward pass and
// their backward; the matrix products are cuBLAS, the shared parts in train_common.cuh. A step holds T tokens of
// packed documents, row-major [T][dim]. The backward pass of a layer recomputes its forward first (only the residual
// stream before each layer is kept), so these kernels save nothing across layers. Built per model size (QWEN_27B:
// Qwen3.8-27B; QWEN_9B: Qwen3.5-9B; else Qwen3.5-4B) and per activation type (train_common.cuh).
//
// In-projection P [T][IN] (act), as the inference kernels read it:
//   Gated DeltaNet: [in_proj_qkv: q (LK_HEADS x LKD), k, v (LV_HEADS x LVD) | in_proj_z | in_proj_b | in_proj_a]
//   full attention: [q_proj: per head q (HD), gate (HD) | k_proj | v_proj]
// Attention tensors are head-major per document: Q of the document starting at token s with L tokens is [NH][L][HD]
// at offset s * NH * HD, K and V [NKV][L][HD] at s * NKV * HD.
#include <cuda_bf16.h>
#include <stdint.h>

#ifdef QWEN_27B
#define H 5120
#define FF 17408
#define NH 24
#define LV_HEADS 48
#elif defined(QWEN_9B)
#define H 4096
#define FF 12288
#define NH 16
#define LV_HEADS 32
#else
#define H 2560
#define FF 9216
#define NH 16
#define LV_HEADS 32
#endif
#define NKV 4
#define HD 256
#define GROUP (NH / NKV)
#define ROT 64
#define QG (2 * NH * HD)
#define LK_HEADS 16
#define LKD 128
#define LVD 128
#define CONV_K 4
#define QK (2 * LK_HEADS * LKD)              // q and k channels of the DeltaNet conv
#define LQKV (QK + LV_HEADS * LVD)            // conv channels: q, k, v
#define ZCOL LQKV                             // in-projection columns: z, b, a
#define BCOL (LQKV + LV_HEADS * LVD)
#define ACOL (BCOL + LV_HEADS)
#define DELTA_IN (ACOL + LV_HEADS)
#define ATTN_IN (QG + 2 * NKV * HD)
#define MIX (NH * HD)
static_assert(MIX == LV_HEADS * LVD, "both mixers write MIX columns");

#include "train_common.cuh"

// Residual step: h_out = h_in + x; y = w * h_out / rms(h_out), rs = 1 / rms(h_out). One block of 256 per token.
extern "C" __global__ void add_rms(const float* h_in, const act* x, const float* w, float* h_out, act* y, float* rs,
                                   float eps) {
    __shared__ float sh[32];
    size_t base = (size_t)blockIdx.x * H;
    float v[H / 256], ss = 0.f;
#pragma unroll
    for (int i = 0; i < H / 256; i++) {
        int c = threadIdx.x + i * 256;
        v[i] = h_in[base + c] + f(x[base + c]);
        h_out[base + c] = v[i];
        ss += v[i] * v[i];
    }
    float r = rsqrtf(block_sum(ss, sh) / H + eps);
    if (threadIdx.x == 0) rs[blockIdx.x] = r;
#pragma unroll
    for (int i = 0; i < H / 256; i++) {
        int c = threadIdx.x + i * 256;
        y[base + c] = to_act(w[c] * (v[i] * r));
    }
}

// Rows r0 .. r0 + n of a stored quantized matrix into dense act rows `out` (row stride cols): q [rows][words] holds
// 32 / bits values per word (value c of a row in word c * bits / 32, from the low bits up); per (row, group of
// `group` columns) an f32 scale and a u8 zero point: w = (q - zero) * scale. One thread per word, its values written
// in one vector store (consecutive threads: consecutive stores).
template <int PER>
struct alignas(PER * sizeof(act) >= 16 ? 16 : PER * sizeof(act)) ActVec {
    act a[PER];
};

template <int BITS>
__device__ __forceinline__ void dequant_word(const uint32_t* q, const float* sc, const uint8_t* zp, int words, int cols,
                                             int group, int r0, size_t i, act* out) {
    constexpr int PER = 32 / BITS;
    int r = r0 + (int)(i / words), w = i % words, c0 = w * PER;
    uint32_t v = q[(size_t)r * words + w];
    size_t g = (size_t)r * (cols / group) + c0 / group;  // a word's values share a group (group % PER == 0)
    float s = sc[g], z = zp[g];
    ActVec<PER> o;
#pragma unroll
    for (int k = 0; k < PER; k++) o.a[k] = to_act(((float)((v >> (k * BITS)) & ((1u << BITS) - 1)) - z) * s);
    *(ActVec<PER>*)(out + (size_t)(r - r0) * cols + c0) = o;
}

extern "C" __global__ void dequant(const uint32_t* q, const float* sc, const uint8_t* zp, int words, int cols,
                                   int group, int bits, int r0, int n, act* out) {
    size_t i = gid();
    if (i >= (size_t)n * words) return;
    if (bits == 4) dequant_word<4>(q, sc, zp, words, cols, group, r0, i, out);
    else dequant_word<8>(q, sc, zp, words, cols, group, r0, i, out);
}

// ================================ Gated DeltaNet ================================

__device__ __forceinline__ float4 ld4(const float* p) { return *(const float4*)p; }
__device__ __forceinline__ float dsilu(float a) {
    float s = sigmoid(a);
    return s * (1.f + a * (1.f - s));
}

// The conv pre-activation of channel c at token t: sum_j w[c][j] P[t - 3 + j][c] within the document
__device__ __forceinline__ float conv_at(const act* P, const float* cw, int t, int c, int ds) {
    float a = 0.f;
#pragma unroll
    for (int j = 0; j < CONV_K; j++) {
        int u = t - (CONV_K - 1) + j;
        if (u >= ds) a += cw[c * CONV_K + j] * f(P[(size_t)u * DELTA_IN + c]);
    }
    return a;
}

// softplus as torch (threshold 20) and its derivative
__device__ __forceinline__ float softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }

// grid (LQKV / 128, ceil(T / 4)), block 128: warp w of block (g, b) handles token 4b + w and channels 128g ..
// 128g + 127 (one q, k or v head), 4 per lane: cv = silu(conv), q and k l2-normed (eps 1e-6 inside the root, as FLA),
// q also / sqrt(LKD). Warp 0 of group 0 also writes the gates gb [T][2 LV_HEADS]: exp(g) with
// g = -exp(A_log) softplus(a + dt_bias), and beta = sigmoid(b).
extern "C" __global__ void __launch_bounds__(128) dn_conv(const act* P, const float* cw, const float* a_log,
                                                          const float* dt_bias, const int* doc_start, float* cv,
                                                          float* gb, int T) {
    int grp = blockIdx.x, lane = threadIdx.x & 31, t = blockIdx.y * 4 + (threadIdx.x >> 5);
    if (t >= T) return;
    int ds = doc_start[t], c = grp * 128 + lane * 4;
    float y[4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        y[i] = silu(conv_at(P, cw, t, c + i, ds));
        ss += y[i] * y[i];
    }
    if (grp < 2 * LK_HEADS) {
        float sc = rsqrtf(warp_sum(ss) + 1e-6f);
        if (grp < LK_HEADS) sc *= rsqrtf((float)LKD);
#pragma unroll
        for (int i = 0; i < 4; i++) y[i] *= sc;
    }
    *(float4*)(cv + (size_t)t * LQKV + c) = make_float4(y[0], y[1], y[2], y[3]);
    if (grp == 0) {
        for (int h = lane; h < LV_HEADS; h += 32) {
            float a = f(P[(size_t)t * DELTA_IN + ACOL + h]) + dt_bias[h], b = f(P[(size_t)t * DELTA_IN + BCOL + h]);
            gb[(size_t)t * 2 * LV_HEADS + h] = expf(-expf(a_log[h]) * softplus(a));
            gb[(size_t)t * 2 * LV_HEADS + LV_HEADS + h] = sigmoid(b);
        }
    }
}

// The gated delta rule per (document, value head h, slice of DCPB value columns), state S[k][v] of the head in f32
// registers: thread = value column col = slice * DCPB + lc and the DRPT key rows 4 * (rg + DLPC j) + i (i < 4) of
// it. Per token: S <- e S; delta = (v - S^T k) beta; S <- S + k delta^T; o = S^T q. Value head h reads key head
// h / (LV_HEADS / LK_HEADS) (Transformers' repeat_interleave).
// State checkpoints for the backward pass: the state before document-local token c * C_CK (c >= 1) at
// ck[doc_ck[d] + c - 1] [LV_HEADS][LVD (v)][LKD (k)]; chunk 0 starts from zeros.
#define C_CK 8
#define DCPB 16
#define DLPC 8
#define DRPT (LKD / DLPC)
#define DNT (DCPB * DLPC)
#define DNSLICE (LVD / DCPB)
#define TCH 8
static_assert(DNT == 128 && DRPT == 16, "dn_rule layout");

__device__ __forceinline__ float col_sum(float v) {
#pragma unroll
    for (int o = 1; o < DLPC; o <<= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ void ld_rows(float* x, const float* src, int rg) {
#pragma unroll
    for (int j = 0; j < DRPT / 4; j++) {
        float4 v = ld4(src + 4 * (rg + DLPC * j));
        x[4 * j] = v.x; x[4 * j + 1] = v.y; x[4 * j + 2] = v.z; x[4 * j + 3] = v.w;
    }
}

__device__ __forceinline__ void st_rows(float* dst, const float* x, int rg) {
#pragma unroll
    for (int j = 0; j < DRPT / 4; j++)
        *(float4*)(dst + 4 * (rg + DLPC * j)) = make_float4(x[4 * j], x[4 * j + 1], x[4 * j + 2], x[4 * j + 3]);
}

__device__ __forceinline__ size_t ck_at(const int* doc_ck, int d, int c, int h, int col) {
    return (((size_t)(doc_ck[d] + c - 1) * LV_HEADS + h) * LVD + col) * LKD;
}

// The m tokens from row `t` of a chunk into shared memory: q and k of key head kh, the value columns of the slice, and
// the gates e and beta of value head h (the caller synchronizes around it)
__device__ __forceinline__ void load_chunk(const float* cv, const float* gb, int t, int m, int kh, int h, int slice,
                                           float (*sq)[LKD], float (*sk)[LKD], float (*sv)[DCPB], float* se, float* sb) {
    for (int x = threadIdx.x; x < m * LKD / 4; x += DNT) {
        int tk = x / (LKD / 4), dd = (x % (LKD / 4)) * 4;
        const float* row = cv + (size_t)(t + tk) * LQKV + kh * LKD + dd;
        *(float4*)&sq[tk][dd] = ld4(row);
        *(float4*)&sk[tk][dd] = ld4(row + LK_HEADS * LKD);
    }
    for (int x = threadIdx.x; x < m * DCPB; x += DNT) {
        int tk = x / DCPB, j = x % DCPB;
        sv[tk][j] = cv[(size_t)(t + tk) * LQKV + QK + h * LVD + slice * DCPB + j];
    }
    if (threadIdx.x < m) {
        se[threadIdx.x] = gb[(size_t)(t + threadIdx.x) * 2 * LV_HEADS + h];
        sb[threadIdx.x] = gb[(size_t)(t + threadIdx.x) * 2 * LV_HEADS + LV_HEADS + h];
    }
}

// grid (DNSLICE, LV_HEADS, docs), block DNT: the raw output o [T][MIX] (f32); ck: checkpoints (null: none)
extern "C" __global__ void __launch_bounds__(DNT) dn_rule(const float* cv, const float* gb, float* o, float* ck,
                                                          const int* doc_start, const int* doc_len, const int* doc_ck) {
    __shared__ __align__(16) float sq[TCH][LKD], sk[TCH][LKD], sv[TCH][DCPB], se[TCH], sb[TCH];
    int slice = blockIdx.x, h = blockIdx.y, d = blockIdx.z, kh = h / (LV_HEADS / LK_HEADS);
    int tid = threadIdx.x, lane = tid & 31, rg = lane % DLPC, lc = (tid >> 5) * (32 / DLPC) + lane / DLPC;
    int col = slice * DCPB + lc, t0 = doc_start[d], len = doc_len[d];
    float S[DRPT];
#pragma unroll
    for (int i = 0; i < DRPT; i++) S[i] = 0.f;
    for (int c0 = 0; c0 < len; c0 += TCH) {
        int m = min(TCH, len - c0);
        __syncthreads();
        load_chunk(cv, gb, t0 + c0, m, kh, h, slice, sq, sk, sv, se, sb);
        __syncthreads();
        for (int tk = 0; tk < m; tk++) {
            int p = c0 + tk;
            if (ck && p > 0 && p % C_CK == 0) st_rows(ck + ck_at(doc_ck, d, p / C_CK, h, col), S, rg);
            float k[DRPT], q[DRPT];
            ld_rows(k, sk[tk], rg);
            ld_rows(q, sq[tk], rg);
            float dk = 0.f, dq = 0.f, kq = 0.f;
#pragma unroll
            for (int i = 0; i < DRPT; i++) { dk += S[i] * k[i]; dq += S[i] * q[i]; kq += k[i] * q[i]; }
            dk = col_sum(dk); dq = col_sum(dq); kq = col_sum(kq);
            float ex = se[tk], delta = (sv[tk][lc] - ex * dk) * sb[tk];
            if (rg == 0) o[(size_t)(t0 + p) * MIX + h * LVD + col] = ex * dq + delta * kq;
#pragma unroll
            for (int i = 0; i < DRPT; i++) S[i] = ex * S[i] + k[i] * delta;
        }
    }
}

// Backward of dn_rule, per (document, value head, slice) block from the last chunk of C_CK tokens to the first; the
// adjoint dS of the state stays in registers. A chunk starts from its checkpoint, and the state before each of its
// tokens is recomputed from there (O(C_CK^2) steps, no storage). Per token, with S the state before it:
//   u = e S^T k, delta = beta (v - u), S_t = e S + k delta^T, o = S_t^T q
//   dS += q do^T; dq = S_t do; d(delta) = dS^T k; dk = dS delta + e S du; dv = beta d(delta);
//   d(beta) = d(delta) . (v - u); du = -beta d(delta); dS~ = dS + k du^T; de = <dS~, S>; dS <- e dS~
// dq, dk (summed over the value heads of a key head and the slices: [T][q | k] like cv), de, d(beta) (gb's layout)
// are added atomically (zero them first); dv [T][MIX] is written.
extern "C" __global__ void __launch_bounds__(DNT) dn_rule_bwd(const float* cv, const float* gb, const float* dO,
                                                              const float* ck, const int* doc_start, const int* doc_len,
                                                              const int* doc_ck, float* dqk, float* dv, float* dgb) {
    __shared__ __align__(16) float sq[C_CK][LKD], sk[C_CK][LKD], sv[C_CK][DCPB], sd[C_CK][DCPB], se[C_CK], sb[C_CK];
    __shared__ __align__(16) float red[DNT / 32][2 * LKD];
    int slice = blockIdx.x, h = blockIdx.y, d = blockIdx.z, kh = h / (LV_HEADS / LK_HEADS);
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, rg = lane % DLPC, lc = warp * (32 / DLPC) + lane / DLPC;
    int col = slice * DCPB + lc, t0 = doc_start[d], len = doc_len[d];
    float dS[DRPT];
#pragma unroll
    for (int i = 0; i < DRPT; i++) dS[i] = 0.f;
    for (int c = (len - 1) / C_CK; c >= 0; c--) {
        int c0 = c * C_CK, m = min(C_CK, len - c0);
        __syncthreads();
        load_chunk(cv, gb, t0 + c0, m, kh, h, slice, sq, sk, sv, se, sb);
        for (int x = tid; x < m * DCPB; x += DNT) {
            int tk = x / DCPB, j = x % DCPB;
            sd[tk][j] = dO[(size_t)(t0 + c0 + tk) * MIX + h * LVD + slice * DCPB + j];
        }
        __syncthreads();
        float S0[DRPT];
        if (c == 0) {
#pragma unroll
            for (int i = 0; i < DRPT; i++) S0[i] = 0.f;
        } else {
            ld_rows(S0, ck + ck_at(doc_ck, d, c, h, col), rg);
        }
        for (int tk = m - 1; tk >= 0; tk--) {
            float S[DRPT], k[DRPT];
#pragma unroll
            for (int i = 0; i < DRPT; i++) S[i] = S0[i];
            for (int s = 0; s < tk; s++) {
                ld_rows(k, sk[s], rg);
                float u = 0.f;
#pragma unroll
                for (int i = 0; i < DRPT; i++) u += S[i] * k[i];
                float ex = se[s], delta = (sv[s][lc] - ex * col_sum(u)) * sb[s];
#pragma unroll
                for (int i = 0; i < DRPT; i++) S[i] = ex * S[i] + k[i] * delta;
            }
            float q[DRPT];
            ld_rows(k, sk[tk], rg);
            ld_rows(q, sq[tk], rg);
            float ex = se[tk], be = sb[tk], vj = sv[tk][lc], doj = sd[tk][lc];
            float u = 0.f;
#pragma unroll
            for (int i = 0; i < DRPT; i++) u += S[i] * k[i];
            u = ex * col_sum(u);
            float delta = be * (vj - u);
            float dqp[DRPT], dkp[DRPT], dd = 0.f;
#pragma unroll
            for (int i = 0; i < DRPT; i++) {
                dS[i] += q[i] * doj;
                dqp[i] = (ex * S[i] + k[i] * delta) * doj;
                dd += dS[i] * k[i];
            }
            dd = col_sum(dd);
            float du = -be * dd, de = 0.f;
#pragma unroll
            for (int i = 0; i < DRPT; i++) {
                dkp[i] = dS[i] * delta + ex * S[i] * du;
                float dst = dS[i] + k[i] * du;
                de += dst * S[i];
                dS[i] = ex * dst;
            }
            de = col_sum(de);
            size_t t = t0 + c0 + tk;
            if (rg == 0) dv[t * MIX + h * LVD + col] = be * dd;
            // de, d(beta): the same on a column's lanes; sum the warp's columns
            float db = dd * (vj - u);
#pragma unroll
            for (int o = DLPC; o < 32; o <<= 1) {
                de += __shfl_xor_sync(0xffffffff, de, o);
                db += __shfl_xor_sync(0xffffffff, db, o);
            }
            if (lane == 0) {
                atomicAdd(dgb + t * 2 * LV_HEADS + h, de);
                atomicAdd(dgb + t * 2 * LV_HEADS + LV_HEADS + h, db);
            }
            // dq, dk: sum over the warp's columns (lanes of equal rg), then over the warps
#pragma unroll
            for (int i = 0; i < DRPT; i++) {
#pragma unroll
                for (int o = DLPC; o < 32; o <<= 1) {
                    dqp[i] += __shfl_xor_sync(0xffffffff, dqp[i], o);
                    dkp[i] += __shfl_xor_sync(0xffffffff, dkp[i], o);
                }
            }
            if (lane < DLPC) {
                st_rows(red[warp], dqp, rg);
                st_rows(red[warp] + LKD, dkp, rg);
            }
            __syncthreads();
            for (int x = tid; x < 2 * LKD; x += DNT) {
                float sum = 0.f;
#pragma unroll
                for (int w = 0; w < DNT / 32; w++) sum += red[w][x];
                int ch = x < LKD ? kh * LKD + x : LK_HEADS * LKD + kh * LKD + x - LKD;
                atomicAdd(dqk + t * QK + ch, sum);
            }
            __syncthreads();
        }
    }
}

// grid (T, LV_HEADS / NORM_HEADS), block NORM_HEADS * 32: warp w normalizes value head blockIdx.y * NORM_HEADS + w of
// token blockIdx.x (gated RMSNorm, plain weight): y = w * o / rms(o) * silu(z) -> y [T][MIX] (act)
#define NORM_HEADS 16
static_assert(LV_HEADS % NORM_HEADS == 0, "dn_norm blocks");
extern "C" __global__ void dn_norm(const float* o, const act* P, const float* nw, act* y, float eps) {
    int t = blockIdx.x, h = blockIdx.y * NORM_HEADS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const float* src = o + (size_t)t * MIX + h * LVD + lane * 4;
    float4 v = ld4(src);
    float r = rsqrtf(warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w) / LVD + eps);
    float x[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int dd = lane * 4 + i;
        float z = f(P[(size_t)t * DELTA_IN + ZCOL + h * LVD + dd]);
        y[(size_t)t * MIX + h * LVD + dd] = to_act(nw[dd] * (x[i] * r) * silu(z));
    }
}

// Backward of dn_norm: dz into dP's z columns, d(o) over o in place (f32)
extern "C" __global__ void dn_norm_bwd(const act* dy, float* o, const act* P, const float* nw, act* dP, float eps) {
    int t = blockIdx.x, h = blockIdx.y * NORM_HEADS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    float* src = o + (size_t)t * MIX + h * LVD + lane * 4;
    float4 v = ld4(src);
    float x[4] = {v.x, v.y, v.z, v.w}, dn[4], s = 0.f;
    float r = rsqrtf(warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w) / LVD + eps);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int dd = lane * 4 + i;
        size_t pc = (size_t)t * DELTA_IN + ZCOL + h * LVD + dd;
        float z = f(P[pc]), g = f(dy[(size_t)t * MIX + h * LVD + dd]) * nw[dd], n = x[i] * r;
        dP[pc] = to_act(g * n * dsilu(z));
        dn[i] = g * silu(z);
        s += dn[i] * n;
    }
    float mean = warp_sum(s) / LVD;
    *(float4*)src = make_float4(r * (dn[0] - x[0] * r * mean), r * (dn[1] - x[1] * r * mean),
                                r * (dn[2] - x[2] * r * mean), r * (dn[3] - x[3] * r * mean));
}

// Backward of dn_conv up to the conv's pre-activation: grid and warps as dn_conv. From dqk (q, k) and dv to
// da = d(silu(conv)) * silu'(conv) [T][LQKV] (f32, into `da`, which may be cv); warp 0 of group 0 writes the gates'
// gradients into dP's a and b columns.
extern "C" __global__ void __launch_bounds__(128) dn_conv_bwd(const act* P, const float* cw, const float* a_log,
                                                              const float* dt_bias, const int* doc_start,
                                                              const float* dqk, const float* dv, const float* dgb,
                                                              const float* gb, float* da, act* dP, int T) {
    int grp = blockIdx.x, lane = threadIdx.x & 31, t = blockIdx.y * 4 + (threadIdx.x >> 5);
    if (t >= T) return;
    int ds = doc_start[t], c = grp * 128 + lane * 4;
    float a[4], y[4], g[4], ss = 0.f, dot = 0.f;
    float4 gv = c < QK ? ld4(dqk + (size_t)t * QK + c) : ld4(dv + (size_t)t * MIX + c - QK);
    g[0] = gv.x; g[1] = gv.y; g[2] = gv.z; g[3] = gv.w;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        a[i] = conv_at(P, cw, t, c + i, ds);
        y[i] = silu(a[i]);
        ss += y[i] * y[i];
        dot += y[i] * g[i];
    }
    if (grp < 2 * LK_HEADS) {  // y_out = s rho y, rho = (sum y^2 + eps)^-1/2: dy = s rho (g - rho^2 (y . g) y)
        float rho = rsqrtf(warp_sum(ss) + 1e-6f), sc = grp < LK_HEADS ? rsqrtf((float)LKD) : 1.f;
        dot = warp_sum(dot);
#pragma unroll
        for (int i = 0; i < 4; i++) g[i] = sc * rho * (g[i] - rho * rho * dot * y[i]);
    }
    *(float4*)(da + (size_t)t * LQKV + c) = make_float4(g[0] * dsilu(a[0]), g[1] * dsilu(a[1]), g[2] * dsilu(a[2]),
                                                        g[3] * dsilu(a[3]));
    if (grp == 0) {
        for (int h = lane; h < LV_HEADS; h += 32) {
            size_t gi = (size_t)t * 2 * LV_HEADS + h;
            float x = f(P[(size_t)t * DELTA_IN + ACOL + h]) + dt_bias[h];
            float dsp = x > 20.f ? 1.f : sigmoid(x);
            dP[(size_t)t * DELTA_IN + ACOL + h] = to_act(dgb[gi] * gb[gi] * -expf(a_log[h]) * dsp);
            float be = gb[gi + LV_HEADS];
            dP[(size_t)t * DELTA_IN + BCOL + h] = to_act(dgb[gi + LV_HEADS] * be * (1.f - be));
        }
    }
}

// The conv's input gradient: dP[u][c] = sum_j w[c][j] da[u + 3 - j][c] within the document (element-wise, T * LQKV)
extern "C" __global__ void dn_conv_in_bwd(const float* da, const float* cw, const int* doc_end, act* dP, int T) {
    size_t i = gid();
    if (i >= (size_t)T * LQKV) return;
    int u = i / LQKV, c = i % LQKV, de = doc_end[u];
    float x = 0.f;
#pragma unroll
    for (int j = 0; j < CONV_K; j++) {
        int tt = u + (CONV_K - 1) - j;
        if (tt < de) x += cw[c * CONV_K + j] * da[(size_t)tt * LQKV + c];
    }
    dP[(size_t)u * DELTA_IN + c] = to_act(x);
}

// ================================ gated full attention ================================

// Per token (one block, a warp per head): q heads (warps 0..NH-1) and k heads (NKV): RMSNorm (1 + w), RoPE on the
// first ROT dims (pairs (d, d + 32), d < 32), into the head-major Q, K; v heads into V. Lane owns dims lane + 32 j.
extern "C" __global__ void at_prep(const act* P, const float* q_norm, const float* k_norm, const float* inv_freq,
                                   const int* tok_pos, const int* doc_start, const int* doc_len, act* Q, act* K, act* V,
                                   float eps) {
    int t = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int s = doc_start[t], L = doc_len[t], i = t - s;
    const act* row = P + (size_t)t * ATTN_IN;
    if (w < NH + NKV) {
        bool isq = w < NH;
        int head = isq ? w : w - NH;
        const act* src = row + (isq ? head * 2 * HD : QG + head * HD);
        const float* nw = isq ? q_norm : k_norm;
        float v[HD / 32], ss = 0.f;
#pragma unroll
        for (int j = 0; j < HD / 32; j++) { v[j] = f(src[lane + 32 * j]); ss += v[j] * v[j]; }
        float r = rsqrtf(warp_sum(ss) / HD + eps);
#pragma unroll
        for (int j = 0; j < HD / 32; j++) v[j] = nw[lane + 32 * j] * (v[j] * r);
        float sn, cs;
        sincosf((float)tok_pos[t] * inv_freq[lane], &sn, &cs);
        float a = v[0], b = v[1];
        v[0] = a * cs - b * sn;
        v[1] = b * cs + a * sn;
        act* dst = isq ? Q + (size_t)s * NH * HD : K + (size_t)s * NKV * HD;
#pragma unroll
        for (int j = 0; j < HD / 32; j++) dst[((size_t)head * L + i) * HD + lane + 32 * j] = to_act(v[j]);
    } else {
        int head = w - NH - NKV;
        const act* src = row + QG + NKV * HD + head * HD;
        act* dst = V + (size_t)s * NKV * HD;
#pragma unroll
        for (int j = 0; j < HD / 32; j++) dst[((size_t)head * L + i) * HD + lane + 32 * j] = src[lane + 32 * j];
    }
}

// Backward of at_prep: dQ, dK, dV into dP's q, k, v columns (RoPE turned back, then the head norm)
extern "C" __global__ void at_prep_bwd(const act* P, const float* q_norm, const float* k_norm, const float* inv_freq,
                                       const int* tok_pos, const int* doc_start, const int* doc_len, const act* dQ,
                                       const act* dK, const act* dV, act* dP, float eps) {
    int t = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int s = doc_start[t], L = doc_len[t], i = t - s;
    const act* row = P + (size_t)t * ATTN_IN;
    act* drow = dP + (size_t)t * ATTN_IN;
    if (w < NH + NKV) {
        bool isq = w < NH;
        int head = isq ? w : w - NH, col = isq ? head * 2 * HD : QG + head * HD;
        const float* nw = isq ? q_norm : k_norm;
        const act* g = isq ? dQ + (size_t)s * NH * HD : dK + (size_t)s * NKV * HD;
        float x[HD / 32], d[HD / 32], ss = 0.f;
#pragma unroll
        for (int j = 0; j < HD / 32; j++) {
            x[j] = f(row[col + lane + 32 * j]);
            ss += x[j] * x[j];
            d[j] = f(g[((size_t)head * L + i) * HD + lane + 32 * j]);
        }
        float r = rsqrtf(warp_sum(ss) / HD + eps);
        float sn, cs;
        sincosf((float)tok_pos[t] * inv_freq[lane], &sn, &cs);
        float a = d[0], b = d[1];
        d[0] = a * cs + b * sn;
        d[1] = b * cs - a * sn;
        float xh[HD / 32], gw[HD / 32], m = 0.f;
#pragma unroll
        for (int j = 0; j < HD / 32; j++) {
            xh[j] = x[j] * r;
            gw[j] = d[j] * nw[lane + 32 * j];
            m += gw[j] * xh[j];
        }
        m = warp_sum(m) / HD;
#pragma unroll
        for (int j = 0; j < HD / 32; j++) drow[col + lane + 32 * j] = to_act(r * (gw[j] - xh[j] * m));
    } else {
        int head = w - NH - NKV;
        const act* g = dV + (size_t)s * NKV * HD;
#pragma unroll
        for (int j = 0; j < HD / 32; j++)
            drow[QG + NKV * HD + head * HD + lane + 32 * j] = g[((size_t)head * L + i) * HD + lane + 32 * j];
    }
}

// Causal softmax of scale * S for `rows` query rows of one document: S [rows][LP] (padded), row r is query r % L.
// One warp per row; P is zero outside the allowed keys, padding included.
extern "C" __global__ void at_softmax(const float* S, act* P, int rows, int L, int LP, float scale) {
    int r = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (r >= rows) return;
    int i = r % L;
    const float* srow = S + (size_t)r * LP;
    act* prow = P + (size_t)r * LP;
    float m = -INFINITY;
    for (int j = lane; j <= i; j += 32) m = fmaxf(m, srow[j] * scale);
    m = warp_max(m);
    float sum = 0.f;
    for (int j = lane; j <= i; j += 32) sum += expf(srow[j] * scale - m);
    float inv = 1.f / warp_sum(sum);
    for (int j = lane; j < LP; j += 32) prow[j] = to_act(j <= i ? expf(srow[j] * scale - m) * inv : 0.f);
}

// dS = scale * P * (dP - sum_j P * dP), per row
extern "C" __global__ void at_softmax_bwd(const act* P, const float* dP, act* dS, int rows, int L, int LP, float scale) {
    int r = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (r >= rows) return;
    const act* prow = P + (size_t)r * LP;
    const float* drow = dP + (size_t)r * LP;
    float d = 0.f;
    for (int j = lane; j < L; j += 32) d += f(prow[j]) * drow[j];
    d = warp_sum(d);
    for (int j = lane; j < LP; j += 32) dS[(size_t)r * LP + j] = to_act(j < L ? scale * f(prow[j]) * (drow[j] - d) : 0.f);
}

// y[t][h * HD + d] = O * sigmoid(gate[t][h][d]); one block of NH warps per token
extern "C" __global__ void at_gate(const act* O, const act* P, const int* doc_start, const int* doc_len, act* y) {
    int t = blockIdx.x, h = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int s = doc_start[t], L = doc_len[t], i = t - s;
    const act* o = O + (size_t)s * NH * HD + ((size_t)h * L + i) * HD;
#pragma unroll
    for (int j = 0; j < HD / 32; j++) {
        int dd = lane + 32 * j;
        float g = sigmoid(f(P[(size_t)t * ATTN_IN + h * 2 * HD + HD + dd]));
        y[(size_t)t * MIX + h * HD + dd] = to_act(f(o[dd]) * g);
    }
}

// dO = dy * sigmoid(gate); d(gate) = dy * O * sigmoid' into dP's gate columns
extern "C" __global__ void at_gate_bwd(const act* dy, const act* O, const act* P, const int* doc_start,
                                       const int* doc_len, act* dO, act* dP) {
    int t = blockIdx.x, h = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int s = doc_start[t], L = doc_len[t], i = t - s;
    size_t oi = (size_t)s * NH * HD + ((size_t)h * L + i) * HD;
#pragma unroll
    for (int j = 0; j < HD / 32; j++) {
        int dd = lane + 32 * j;
        size_t gc = (size_t)t * ATTN_IN + h * 2 * HD + HD + dd;
        float g = sigmoid(f(P[gc])), d = f(dy[(size_t)t * MIX + h * HD + dd]);
        dO[oi + dd] = to_act(d * g);
        dP[gc] = to_act(d * f(O[oi + dd]) * g * (1.f - g));
    }
}
