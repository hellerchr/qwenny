// What every model's kernels share: embedding and norms, matrix products, a served LoRA, the residual step, the attention
// over the KV cache and the sampler. The including file defines XIN, H, NH, NKV, HD, GROUP, MIX and VOCAB, and adds its
// attn_prep.
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
