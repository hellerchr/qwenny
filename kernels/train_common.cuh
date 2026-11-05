#pragma once
// The general parts of the LoRA training kernels (included by qwen_train.cu; the host side is src/cuda/learn.rs):
// the activation type of the build, RMSNorm, SwiGLU, cross-entropy and row copies,
// LoRA dropout, gradient clipping and AdamW over the LoRA arena. Built with `act` bf16, f32 with -DACT_F32 (checks) or
// f16 with -DACT_F16. The including file defines H (the hidden size) and FF (the MLP width).
#include <cuda_fp16.h>
#include "common.cuh"
#if defined(ACT_F32)
typedef float act;
__device__ __forceinline__ act to_act(float x) { return x; }
#elif defined(ACT_F16)
typedef __half act;
__device__ __forceinline__ act to_act(float x) { return __float2half(x); }
#else
typedef __nv_bfloat16 act;
__device__ __forceinline__ act to_act(float x) { return __float2bfloat16(x); }
#endif
// gradients of the norms' inputs (the products' dX): f32, or f16 in the f16 build
#ifdef ACT_F16
typedef __half grad;
#else
typedef float grad;
#endif
__device__ __forceinline__ float f(float x) { return x; }
__device__ __forceinline__ float f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float f(__half x) { return __half2float(x); }
__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

__device__ __forceinline__ float block_max(float v, float* sh) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    v = warp_max(v);
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float t = lane < (blockDim.x >> 5) ? sh[lane] : -INFINITY;
    t = warp_max(t);
    __syncthreads();
    return t;
}

__device__ __forceinline__ size_t gid() { return (size_t)blockIdx.x * blockDim.x + threadIdx.x; }

// ---- RMSNorm: one block of 256 threads per token ----

// y = w * x / rms(x); rs = 1 / rms(x)
extern "C" __global__ void rms_fwd(const float* x, const float* w, act* y, float* rs, float eps) {
    __shared__ float sh[32];
    size_t base = (size_t)blockIdx.x * H;
    float v[H / 256], ss = 0.f;
#pragma unroll
    for (int i = 0; i < H / 256; i++) { v[i] = x[base + threadIdx.x + i * 256]; ss += v[i] * v[i]; }
    float r = rsqrtf(block_sum(ss, sh) / H + eps);
    if (threadIdx.x == 0) rs[blockIdx.x] = r;
#pragma unroll
    for (int i = 0; i < H / 256; i++) {
        int c = threadIdx.x + i * 256;
        y[base + c] = to_act(w[c] * (v[i] * r));
    }
}

// Backward of y = w * x * rs: dx = rs * (g - xhat * mean(g * xhat)), g = dy * w, xhat = x * rs
template <class TX, class TY, class TD>
__device__ __forceinline__ void rms_bwd(const TX* x, const float* rs, const float* w, const TY* dy, TD* dx,
                                        bool acc) {
    __shared__ float sh[32];
    size_t base = (size_t)blockIdx.x * H;
    float r = rs[blockIdx.x];
    float xh[H / 256], g[H / 256], s = 0.f;
#pragma unroll
    for (int i = 0; i < H / 256; i++) {
        int c = threadIdx.x + i * 256;
        xh[i] = f(x[base + c]) * r;
        g[i] = f(dy[base + c]) * w[c];
        s += g[i] * xh[i];
    }
    float m = block_sum(s, sh) / H;
#pragma unroll
    for (int i = 0; i < H / 256; i++) {
        int c = threadIdx.x + i * 256;
        float d = r * (g[i] - xh[i] * m);
        if (acc) d += f(dx[base + c]);
        dx[base + c] = (TD)d;
    }
}

// The pre-norms of the residual stream: dh += d(norm input)
extern "C" __global__ void rms_bwd_pre(const float* x, const float* rs, const float* w, const grad* dy, float* dh) {
    rms_bwd<float, grad, float>(x, rs, w, dy, dh, true);
}

// The final norm, in place: dh holds d(output) and becomes d(input)
extern "C" __global__ void rms_bwd_final(const float* x, const float* rs, const float* w, float* dh) {
    rms_bwd<float, float, float>(x, rs, w, dh, dh, false);
}

// ---- feed-forward (the including file defines FF) ----

// w13 [T][2FF] = [a | b]: mid = silu(a) * b
extern "C" __global__ void swiglu_fwd(const act* w13, act* mid, int T) {
    size_t i = gid();
    if (i >= (size_t)T * FF) return;
    size_t t = i / FF, j = i % FF;
    float a = f(w13[t * 2 * FF + j]), b = f(w13[t * 2 * FF + FF + j]);
    mid[i] = to_act(a * sigmoid(a) * b);
}

// In place: w13 = [a | b] becomes [da | db] for d(mid)
extern "C" __global__ void swiglu_bwd(act* w13, const act* dmid, int T) {
    size_t i = gid();
    if (i >= (size_t)T * FF) return;
    size_t t = i / FF, j = i % FF;
    float a = f(w13[t * 2 * FF + j]), b = f(w13[t * 2 * FF + FF + j]), d = f(dmid[i]);
    float s = sigmoid(a);
    w13[t * 2 * FF + j] = to_act(d * b * s * (1.f + a * (1.f - s)));
    w13[t * 2 * FF + FF + j] = to_act(d * a * s);
}

// out[rows[b]] = x[b], scaled by w[b] when w is given (preference training: a weight per loss row)
extern "C" __global__ void scatter_rows(const float* x, const int* rows, const float* w, float* out) {
    float* dst = out + (size_t)rows[blockIdx.x] * H;
    const float s = w ? w[blockIdx.x] : 1.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) dst[i] = s * x[(size_t)blockIdx.x * H + i];
}

// Per row (one block): loss = logsumexp - logit[target], correct = argmax == target, and with dlog != 0
// dlog = scale * (softmax - onehot(target)).
extern "C" __global__ void cross_entropy(const float* logits, const int* targets, int V, float scale, act* dlog,
                                         float* loss, int* correct) {
    __shared__ float sh[32];
    __shared__ int shi[32];
    const float* row = logits + (size_t)blockIdx.x * V;
    float m = -INFINITY;
    int mi = 0;
    for (int j = threadIdx.x; j < V; j += blockDim.x)
        if (row[j] > m) { m = row[j]; mi = j; }
    // argmax: the lowest index among the maxima
    float M = block_max(m, sh);
    int cand = m == M ? mi : 0x7fffffff;
    for (int o = 16; o > 0; o >>= 1) cand = min(cand, __shfl_xor_sync(0xffffffff, cand, o));
    if ((threadIdx.x & 31) == 0) shi[threadIdx.x >> 5] = cand;
    __syncthreads();
    int best = 0x7fffffff;
    for (int w = 0; w < (int)(blockDim.x >> 5); w++) best = min(best, shi[w]);
    float sum = 0.f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) sum += expf(row[j] - M);
    sum = block_sum(sum, sh);
    int tg = targets[blockIdx.x];
    if (threadIdx.x == 0) {
        loss[blockIdx.x] = logf(sum) + M - row[tg];
        correct[blockIdx.x] = best == tg;
    }
    if (dlog) {
        float inv = 1.f / sum;
        act* drow = dlog + (size_t)blockIdx.x * V;
        for (int j = threadIdx.x; j < V; j += blockDim.x)
            drow[j] = to_act(scale * (expf(row[j] - M) * inv - (j == tg ? 1.f : 0.f)));
    }
}

// ---- rows of [n][H] ----

extern "C" __global__ void gather_rows(const act* x, const int* rows, act* out) {
    const act* src = x + (size_t)rows[blockIdx.x] * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) out[(size_t)blockIdx.x * H + i] = src[i];
}

// ---- LoRA dropout: the same mask in forward and backward, from (seed, index); 8 elements per thread ----

__device__ __forceinline__ uint32_t hash32(uint32_t seed, uint32_t i) {
    uint32_t x = i * 0x9E3779B1u ^ seed;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// keep with probability 1 - p (p in steps of 2^-24)
__device__ __forceinline__ bool keep(uint32_t seed, size_t i, uint32_t thr) { return (hash32(seed, (uint32_t)i) >> 8) >= thr; }

template <class T>
struct alignas(16) V8 {
    T a[8];
};

struct DropArgs {
    uint32_t seed, thr;
    float inv;
};

__device__ __forceinline__ DropArgs drop_args(unsigned long long seed, float p) {
    return {(uint32_t)(seed ^ (seed >> 32)), (uint32_t)(p * 16777216.f), 1.f / (1.f - p)};
}

extern "C" __global__ void dropout(const act* x, act* out, size_t n, float p, unsigned long long seed) {
    size_t i0 = gid() * 8;
    if (i0 >= n) return;
    DropArgs d = drop_args(seed, p);
    V8<act> v = *(const V8<act>*)(x + i0);
#pragma unroll
    for (int j = 0; j < 8; j++) v.a[j] = to_act(keep(d.seed, i0 + j, d.thr) ? f(v.a[j]) * d.inv : 0.f);
    *(V8<act>*)(out + i0) = v;
}

// dx += mask * g / (1 - p), dx a gradient buffer or act
extern "C" __global__ void dropout_add_grad(grad* dx, const float* g, size_t n, float p, unsigned long long seed) {
    size_t i0 = gid() * 8;
    if (i0 >= n) return;
    DropArgs d = drop_args(seed, p);
    V8<grad> x = *(V8<grad>*)(dx + i0);
    V8<float> gv = *(const V8<float>*)(g + i0);
#pragma unroll
    for (int j = 0; j < 8; j++) x.a[j] = (grad)(f(x.a[j]) + (keep(d.seed, i0 + j, d.thr) ? gv.a[j] * d.inv : 0.f));
    *(V8<grad>*)(dx + i0) = x;
}

extern "C" __global__ void dropout_add_act(act* dx, const float* g, size_t n, float p, unsigned long long seed) {
    size_t i0 = gid() * 8;
    if (i0 >= n) return;
    DropArgs d = drop_args(seed, p);
    V8<act> x = *(V8<act>*)(dx + i0);
    V8<float> gv = *(const V8<float>*)(g + i0);
#pragma unroll
    for (int j = 0; j < 8; j++) x.a[j] = to_act(f(x.a[j]) + (keep(d.seed, i0 + j, d.thr) ? gv.a[j] * d.inv : 0.f));
    *(V8<act>*)(dx + i0) = x;
}

// ---- optimizer over the LoRA arena (f32 parameters, gradients and moments; `pa` the act copy for cuBLAS) ----

extern "C" __global__ void sum_squares(const float* g, size_t n, float* out) {
    __shared__ float sh[32];
    float s = 0.f;
    for (size_t i = gid(); i < n; i += (size_t)gridDim.x * blockDim.x) s += g[i] * g[i];
    s = block_sum(s, sh);
    if (threadIdx.x == 0) atomicAdd(out, s);
}

// torch.optim.AdamW after clip_grad_norm_(max_norm): the gradient is scaled by min(1, max_norm / (norm + 1e-6));
// g holds the gradient times the loss scale, 1 / inv_scale
extern "C" __global__ void adamw(float* p, const float* g, float* m, float* v, act* pa, size_t n, const float* sumsq,
                                 float max_norm, float lr, float b1, float b2, float eps, float wd, float bc1,
                                 float bc2, float inv_scale) {
    size_t i = gid();
    if (i >= n) return;
    float clip = fminf(1.f, max_norm / (sqrtf(*sumsq) * inv_scale + 1e-6f));
    float gi = g[i] * inv_scale * clip;
    float mi = b1 * m[i] + (1.f - b1) * gi;
    float vi = b2 * v[i] + (1.f - b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float pi = p[i] * (1.f - lr * wd);
    pi -= lr / bc1 * mi / (sqrtf(vi) / sqrtf(bc2) + eps);
    p[i] = pi;
    pa[i] = to_act(pi);
}

extern "C" __global__ void to_act_copy(const float* p, act* pa, size_t n) {
    size_t i = gid();
    if (i < n) pa[i] = to_act(p[i]);
}
