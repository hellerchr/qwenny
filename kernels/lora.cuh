// A LoRA adapter applied unmerged (src/cuda/qwen/lora.rs): y = W x + s B (A x) per adapted module, s = alpha / r for
// the tokens of slots that use the adapter and 0 (nothing added) for the others. The products that read one input
// share one stacked A (a matrix product of the step, gemm.cuh: its split-K partial products xa [split][n][M], M =
// adapted modules * R); these kernels add s B xa to a plain product's partial products (split 0, [n][ldy]), or apply
// SwiGLU to a plain [gate; up] product in the SwiGLU epilogue's row order (core.rs row_order) with the delta added
// first. B: f16 [rows][R] in the product's row order. tab: per module of the stacked product the end of its rows
// (ascending; unused: INT_MAX), then its first column of xa (-1: not adapted); SwiGLU: module 0 gate, 1 up.
#pragma once

#define LORA_TT 8      // tokens per block
#define LORA_MAXM 256  // columns of xa at most (4 modules of rank 64)

// The block's tokens t0 .. t0 + LORA_TT: their scales and xa (the sum of A's partial products; 0 without the adapter)
__device__ __forceinline__ void lora_xa(const float* xa, int split, int M, int n, const int* tok_slot,
                                        const float* slot_scale, int t0, float (*xs)[LORA_MAXM], float* sc) {
    if (threadIdx.x < LORA_TT) {
        int t = t0 + threadIdx.x;
        sc[threadIdx.x] = t < n ? slot_scale[tok_slot[t]] : 0.f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < LORA_TT * M; i += blockDim.x) {
        int tl = i / M, m = i % M, t = t0 + tl;
        xs[tl][m] = sc[tl] != 0.f ? part_sum(xa, split, n, M, t, m) : 0.f;
    }
    __syncthreads();
}

template <int R>
__device__ __forceinline__ void lora_row(const __half* B, int o, float (&b)[R]) {
    const uint4* p = (const uint4*)(B + (size_t)o * R);
#pragma unroll
    for (int i = 0; i < R / 8; i++) {
        uint4 u = p[i];
        const __half2* h = (const __half2*)&u;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f = __half22float2(h[j]);
            b[8 * i + 2 * j] = f.x;
            b[8 * i + 2 * j + 1] = f.y;
        }
    }
}

template <int R>
__device__ __forceinline__ float lora_dot(const float (&b)[R], const float* x) {
    float d = 0.f;
#pragma unroll
    for (int j = 0; j < R; j++) d = fmaf(b[j], x[j], d);
    return d;
}

// y[t][o] += s_t * B[o] . xa[t][off(o) ..] for the rows o of a plain product (y: its split 0, [n][ldy]).
// Grid (rows / 256, n / LORA_TT), 256 threads.
template <int R>
__device__ void lora_add(const float* xa, int split, int M, const __half* B, int rows, const int* tab,
                         const int* tok_slot, const float* slot_scale, int n, float* y, int ldy) {
    __shared__ float xs[LORA_TT][LORA_MAXM];
    __shared__ float sc[LORA_TT];
    int t0 = blockIdx.y * LORA_TT;
    lora_xa(xa, split, M, n, tok_slot, slot_scale, t0, xs, sc);
    bool any = false;
#pragma unroll
    for (int i = 0; i < LORA_TT; i++) any |= sc[i] != 0.f;
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (!any || o >= rows) return;
    int mod = (o >= tab[0]) + (o >= tab[1]) + (o >= tab[2]);
    int off = tab[4 + mod];
    if (off < 0) return;
    float b[R];
    lora_row<R>(B, o, b);
#pragma unroll
    for (int tl = 0; tl < LORA_TT; tl++) {
        int t = t0 + tl;
        if (sc[tl] != 0.f) y[(size_t)t * ldy + o] += sc[tl] * lora_dot<R>(b, &xs[tl][off]);
    }
}

// The SwiGLU of a plain [gate; up] product y [n][ldy] (split 1, the SwiGLU epilogue's row order: 16-row tile p holds
// gate rows 8p .. 8p + 7, then their up rows) with the delta added: mid[t][8p + g] = silu(gate + dg) * (up + du), as
// the epilogue rounds it (every token: those without the adapter get what the epilogue gives them).
// Grid (rows / 2 / 256, n / LORA_TT), 256 threads.
template <int R>
__device__ void lora_silu(const float* xa, int split, int M, const __half* B, int rows, const int* tab,
                          const int* tok_slot, const float* slot_scale, int n, const float* y, int ldy, XIN* mid) {
    __shared__ float xs[LORA_TT][LORA_MAXM];
    __shared__ float sc[LORA_TT];
    int t0 = blockIdx.y * LORA_TT;
    lora_xa(xa, split, M, n, tok_slot, slot_scale, t0, xs, sc);
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= rows / 2) return;
    int og = 16 * (j / 8) + j % 8, ou = og + 8, offg = tab[4], offu = tab[5];
    float bg[R], bu[R];
    if (offg >= 0) lora_row<R>(B, og, bg);
    if (offu >= 0) lora_row<R>(B, ou, bu);
#pragma unroll
    for (int tl = 0; tl < LORA_TT; tl++) {
        int t = t0 + tl;
        if (t >= n) break;
        float g = y[(size_t)t * ldy + og], u = y[(size_t)t * ldy + ou];
        if (sc[tl] != 0.f) {
            if (offg >= 0) g += sc[tl] * lora_dot<R>(bg, &xs[tl][offg]);
            if (offu >= 0) u += sc[tl] * lora_dot<R>(bu, &xs[tl][offu]);
        }
        st_x(mid + (size_t)t * (rows / 2) + j, silu(g) * u);
    }
}

#define LORA_SET(R)                                                                                                  \
    extern "C" __global__ void __launch_bounds__(256) lora_add_r##R(const float* xa, int split, int M, const __half* B, \
        int rows, const int* tab, const int* tok_slot, const float* slot_scale, int n, float* y, int ldy) {           \
        lora_add<R>(xa, split, M, B, rows, tab, tok_slot, slot_scale, n, y, ldy);                                   \
    }                                                                                                                \
    extern "C" __global__ void __launch_bounds__(256) lora_silu_r##R(const float* xa, int split, int M,               \
        const __half* B, int rows, const int* tab, const int* tok_slot, const float* slot_scale, int n,               \
        const float* y, int ldy, XIN* mid) {                                                                          \
        lora_silu<R>(xa, split, M, B, rows, tab, tok_slot, slot_scale, n, y, ldy, mid);                             \
    }

LORA_SET(8)
LORA_SET(16)
LORA_SET(32)
LORA_SET(64)
