// Qwen3.5 kernels (Gated DeltaNet and gated full attention, see qwen_deltanet.cuh and attn.cuh). A step
// processes N tokens (any mix of sequences); activations are f32 [N][dim], the matrix products' inputs f16 (XIN);
// matrix products leave split-K partial products [split][N][rows] that the consumer sums. Norm weights arrive as (1 + w): Qwen3.5's RMSNorm scales by
// 1 + weight (the host adds the 1). Built once per model size (QWEN_27B: Qwen3.8-27B; QWEN_9B: Qwen3.5-9B; else
// Qwen3.5-4B).
#include "common.cuh"

#ifdef QWEN_27B
#define H 5120
#define NH 24          // full attention: query heads
#define LV_HEADS 48    // Gated DeltaNet: value heads
#elif defined(QWEN_9B)
#define H 4096
#define NH 16
#define LV_HEADS 32
#else
#define H 2560
#define NH 16
#define LV_HEADS 32
#endif
#define NKV 4          // full attention: key/value heads
#define HD 256         // ... head dim
#define GROUP (NH / NKV)
#define ROT 64         // rotary dims of a head (partial_rotary_factor 0.25): pairs (d, d + ROT / 2), d < ROT / 2
#define QG_ROWS (2 * NH * HD)                   // q_proj rows: per head [q (HD); gate (HD)]
#define ATTN_IN (QG_ROWS + 2 * NKV * HD)        // in-projection rows of a full-attention layer: q_proj, k_proj, v_proj
#define LK_HEADS 16    // Gated DeltaNet: key heads (q and k)
#define LKD 128        // ... key head dim
#define LVD 128        // ... value head dim
#define LQKV (2 * LK_HEADS * LKD + LV_HEADS * LVD)  // conv channels: q, k, v
#define DELTA_IN (LQKV + LV_HEADS * LVD)  // in-projection rows: in_proj_qkv, in_proj_z (in_proj_b, in_proj_a: a product of their own)
#define CONV_K 4       // causal conv kernel
#define MIX (NH * HD)  // full-attention output width (o_proj input)
#define VOCAB 248320
#define EPS 1e-6f

#include "dense.cuh"
#include "gemm.cuh"
GEMM_SET(i8, 8)
GEMM_SET(f16, 16)

// the stored int4 matrices, symmetric or with zero points: Marlin (core.rs Gemm::marlin)
#include "marlin.cuh"
MARLIN_SET(1, 8, 8)
MARLIN_SET(1, 16, 4)
MARLIN_SET(2, 16, 4)
MARLIN_SET(3, 16, 4)
MARLIN_SET(4, 16, 4)

// a LoRA adapter served unmerged (src/cuda/qwen/lora.rs)
#include "lora.cuh"

// Residual step: h += the sum of `splits` partial products x [split][n][H]; then y = w_next * h / rms(h) (the next
// layer's input norm, or the final norm). One block of AN threads per token.
#define AN 512
static_assert(H % AN == 0, "AN divides H");
extern "C" __global__ void __launch_bounds__(AN) add_rmsnorm(float* h, const float* x, int splits, int n,
                                                              const float* w_next, XIN* y, float eps) {
    __shared__ float sh[32];
    size_t base = (size_t)blockIdx.x * H;
    float v[H / AN], wn[H / AN], ss = 0.f;
#pragma unroll
    for (int i = 0; i < H / AN; i++) {
        int c = threadIdx.x + i * AN;
        v[i] = h[base + c] + x[base + c];
        wn[i] = w_next[c];
    }
#pragma unroll 4
    for (int s = 1; s < splits; s++)
#pragma unroll
        for (int i = 0; i < H / AN; i++) v[i] += x[(size_t)s * n * H + base + threadIdx.x + i * AN];
#pragma unroll
    for (int i = 0; i < H / AN; i++) {
        h[base + threadIdx.x + i * AN] = v[i];
        ss += v[i] * v[i];
    }
    float r = rsqrtf(block_sum(ss, sh) / H + eps);
#pragma unroll
    for (int i = 0; i < H / AN; i++) st_x(y + base + threadIdx.x + i * AN, wn[i] * (v[i] * r));
}

// out[t] = the row ids[t] of an int8 embedding table as stored: (E - 128) * a scale per 128 columns (S [rows][H / 128]);
// the first nkeys blocks clear the argmax keys of the step (as embed). The table lives in host memory the device maps
// (core.rs HostMapped): 16-byte reads over PCIe
extern "C" __global__ void embed_q8(const uint8_t* E, const float* S, const int* ids, float* out, unsigned long long* keys,
                                    int nkeys) {
    if (threadIdx.x == 0 && blockIdx.x < nkeys) keys[blockIdx.x] = 0;
    const uint4* row = (const uint4*)(E + (size_t)ids[blockIdx.x] * H);
    const float* sr = S + (size_t)ids[blockIdx.x] * (H / 128);
    float* o = out + (size_t)blockIdx.x * H;
    for (int i = threadIdx.x; i < H / 16; i += blockDim.x) {
        uint4 u = row[i];
        float sc = sr[i / 8];
        uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 16; j++) o[16 * i + j] = ((int)((w[j / 4] >> (8 * (j % 4))) & 255) - 128) * sc;
    }
}

// MTP input (the draft layer of speculative decoding), one block of 256 threads per token t:
// y[t] = [pre_e * e / rms(e); pre_h * g / rms(g)], e = the token's embedding (e [n][H], from embed), g = fw *
// h[rows[t]] / rms(h[rows[t]]): h is the residual stream before a final norm fw (the target's and its final norm, or the
// MTP layer's own and its norm).
extern "C" __global__ void __launch_bounds__(256) mtp_in(const float* emb, const float* h, const int* rows,
                                                         const float* fw, const float* pre_e, const float* pre_h,
                                                         XIN* y, float eps) {
    __shared__ float sh[32];
    const float* er = emb + (size_t)blockIdx.x * H;
    const float* hr = h + (size_t)rows[blockIdx.x] * H;
    float e[H / 256], g[H / 256], se = 0.f, sg = 0.f;
#pragma unroll
    for (int i = 0; i < H / 256; i++) {
        int c = threadIdx.x + i * 256;
        e[i] = er[c];
        g[i] = hr[c];
        se += e[i] * e[i];
        sg += g[i] * g[i];
    }
    float re = rsqrtf(block_sum(se, sh) / H + eps), rg = rsqrtf(block_sum(sg, sh) / H + eps), sg2 = 0.f;
#pragma unroll
    for (int i = 0; i < H / 256; i++) {
        g[i] = fw[threadIdx.x + i * 256] * (g[i] * rg);
        sg2 += g[i] * g[i];
    }
    float rg2 = rsqrtf(block_sum(sg2, sh) / H + eps);
    XIN* yr = y + (size_t)blockIdx.x * 2 * H;
#pragma unroll
    for (int i = 0; i < H / 256; i++) {
        int c = threadIdx.x + i * 256;
        st_x(yr + c, pre_e[c] * (e[i] * re));
        st_x(yr + H + c, pre_h[c] * (g[i] * rg2));
    }
}

// y = x as the matrix products' input type (the attention output, n * MIX values)
extern "C" __global__ void to_xin(const float* x, XIN* y, int len) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += gridDim.x * blockDim.x) st_x(y + i, x[i]);
}

// Sum `splits` partial products [split][n][ld] into y [n][ld] (layercheck)
extern "C" __global__ void sum_parts(const float* x, int splits, int n, int ld, float* y) {
    for (int c = threadIdx.x; c < ld; c += blockDim.x) y[(size_t)blockIdx.x * ld + c] = part_sum(x, splits, n, ld, blockIdx.x, c);
}

#include "attn.cuh"
#include "sampling.cuh"
#include "qwen_deltanet.cuh"

#define PREP_WARPS (NH + 2 * NKV)
static_assert(HD == 256 && ROT == 64, "attn_prep's lanes assume head dim 256 and 64 rotary dims");

// Per token (one block, a warp per head): the sum of the in-projection partial products. Warps 0..NH-1: query head
// w, RMSNorm (1 + w), RoPE, times qscale -> q [t][NH][HD]; sigmoid of its gate -> out [t][MIX], which the attention
// kernels multiply by. Warps NH..NH+NKV-1: key heads (norm, RoPE), then value heads, into the cache.
// Lane owns dims lane + 32 j: the RoPE pairs (d, d + 32), d < 32, are j = 0 and 1 of the same lane.
extern "C" __global__ void __launch_bounds__(32 * PREP_WARPS)
    attn_prep(const float* part, int splits, int n, const float* q_norm, const float* k_norm, const float* inv_freq,
              float qscale, float* q, float* out, int8_t* kc, int8_t* vc, float* ks, float* vs, const int* tok_slot,
              const int* tok_pos, int cap) {
    int t = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31, pos = tok_pos[t];
    int col;
    const float* nw;
    if (w < NH) { col = w * 2 * HD; nw = q_norm; }
    else if (w < NH + NKV) { col = QG_ROWS + (w - NH) * HD; nw = k_norm; }
    else { col = QG_ROWS + NKV * HD + (w - NH - NKV) * HD; nw = nullptr; }
    // all loads of a split slice at once (q warps: the gate too)
    float v[HD / 32], gl[HD / 32];
#pragma unroll
    for (int j = 0; j < HD / 32; j++) v[j] = gl[j] = 0.f;
    for (int sp = 0; sp < splits; sp++) {
        const float* p = part + ((size_t)sp * n + t) * ATTN_IN + col + lane;
#pragma unroll
        for (int j = 0; j < HD / 32; j++) v[j] += p[32 * j];
        if (w < NH)
#pragma unroll
            for (int j = 0; j < HD / 32; j++) gl[j] += p[HD + 32 * j];
    }
    if (nw) {
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < HD / 32; j++) ss += v[j] * v[j];
        float r = rsqrtf(warp_sum(ss) / HD + EPS);
#pragma unroll
        for (int j = 0; j < HD / 32; j++) v[j] = nw[lane + 32 * j] * (v[j] * r);
        float sn, cs;
        sincosf((float)pos * inv_freq[lane], &sn, &cs);
        float a = v[0], b = v[1];
        v[0] = a * cs - b * sn;
        v[1] = b * cs + a * sn;
    }
    if (w < NH) {
        float* qo = q + ((size_t)t * NH + w) * HD;
        float* go = out + (size_t)t * MIX + w * HD;
#pragma unroll
        for (int j = 0; j < HD / 32; j++) {
            qo[lane + 32 * j] = v[j] * qscale;
            go[lane + 32 * j] = 1.f / (1.f + expf(-gl[j]));
        }
    } else {
        kv_store(v, w < NH + NKV, kc, vc, ks, vs, lane, tok_slot[t], (w - NH) % NKV, pos, cap);
    }
}
