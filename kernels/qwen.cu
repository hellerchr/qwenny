// Qwen3.5 kernels (Gated DeltaNet and gated full attention, see qwen_deltanet.cuh and attn.cuh). A step
// processes N tokens (any mix of sequences); activations are f32 [N][dim], the matrix products' inputs f16 (XIN);
// matrix products leave split-K partial products [split][N][rows] that the consumer sums. Norm weights arrive as (1 + w): Qwen3.5's RMSNorm scales by
// 1 + weight (the host adds the 1). Built once per model size (QWEN_27B: Qwen3.8-27B; QWEN_9B: Qwen3.5-9B; else
// Qwen3.5-4B).
#define XIN __half
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

#include "stack.cuh"
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
