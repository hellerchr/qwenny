// Kernels of Granite (GraniteForCausalLM, Granite 4.2 8B and what is post-trained from it, e.g. TwIL-8B): GQA attention in
// every layer, RoPE over whole heads, no gates and no norms on q and k. The rest is stack.cuh's. Norm weights arrive as
// stored, the model's RMSNorm scales by the weight.
#define XIN __half
#include "common.cuh"

#define H 4096
#define NH 32          // query heads
#define NKV 8          // key/value heads
#define HD 128         // head dim
#define GROUP (NH / NKV)
#define MIX (NH * HD)  // attention output width (o_proj input)
#define ATTN_IN ((NH + 2 * NKV) * HD)  // in-projection rows: q_proj, k_proj, v_proj
#define VOCAB 100352

#include "stack.cuh"
#include "dflash.cuh"

static_assert(HD % 64 == 0, "attn_prep's RoPE pairs: dims lane + 32 j and lane + 32 j + HD / 2");

// One head of a token per block, one warp (NH + 2 NKV heads do not fit one block of at most 1024 threads): the sum of the
// head's in-projection partial products, RoPE on q and k (pairs (d, d + HD / 2), rotate_half in Transformers), then q heads
// times qscale into q [t][NH][HD] and 1 into the attention's output, k and v heads into the cache. Lane owns dims lane + 32 j.
extern "C" __global__ void __launch_bounds__(32)
    attn_prep(const float* part, int splits, int n, const float* inv_freq, float qscale, float* q, float* out, int8_t* kc,
              int8_t* vc, float* ks, float* vs, const int* tok_slot, const int* tok_pos, int cap) {
    int t = blockIdx.x, w = blockIdx.y, lane = threadIdx.x, pos = tok_pos[t];
    float v[HD / 32];
#pragma unroll
    for (int j = 0; j < HD / 32; j++) v[j] = 0.f;
    for (int sp = 0; sp < splits; sp++) {
        const float* p = part + ((size_t)sp * n + t) * ATTN_IN + w * HD + lane;
#pragma unroll
        for (int j = 0; j < HD / 32; j++) v[j] += p[32 * j];
    }
    if (w < NH + NKV) {
#pragma unroll
        for (int j = 0; j < HD / 64; j++) {
            float sn, cs;
            sincosf((float)pos * inv_freq[lane + 32 * j], &sn, &cs);
            float a = v[j], b = v[j + HD / 64];
            v[j] = a * cs - b * sn;
            v[j + HD / 64] = b * cs + a * sn;
        }
    }
    if (w < NH) {
        float* qo = q + ((size_t)t * NH + w) * HD;
        float* go = out + (size_t)t * MIX + w * HD;
#pragma unroll
        for (int j = 0; j < HD / 32; j++) {
            qo[lane + 32 * j] = v[j] * qscale;
            go[lane + 32 * j] = 1.f;
        }
    } else {
        kv_store(v, w < NH + NKV, kc, vc, ks, vs, lane, tok_slot[t], (w - NH) % NKV, pos, cap);
    }
}
