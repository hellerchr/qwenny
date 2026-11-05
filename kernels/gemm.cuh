// Matrix products on tensor cores, .cu files instantiate them with GEMM_SET.
#pragma once
#include "common.cuh"

// ---- matrix products on tensor cores ----
//
// Y[t][r] = W[r] . X[t]. W is packed per tile of 16 rows, in the register order of the mma A fragment: each lane
// holds 16 bytes per load, the values a0..a7 of 1 (f16) or 2 (int8, stored q + 128) k16 steps. WB is the weight
// width in bits (16, 8). int8 rows carry one f32 scale per 128 columns, stored [tile][group][16 rows]. (int4 matrices
// are Marlin's: marlin.cuh.)
// A block of WARPS warps takes WARPS consecutive tiles, a K range (split-K: blockIdx.y) and 8 * NTT tokens
// (blockIdx.z); X is staged in shared memory as f16 per chunk of KC columns.

#define WARPS 4      // decode kernels
#define PWARPS 8     // prompt kernels: more rows per block share the staged X (one 16-row tile per warp: 2 measured
                     // 1.8 % slower on the 27B: one block per SM)
#define PNTT 8       // prompt kernels: n8 token tiles per block
#define KC 128       // prompt kernels: columns staged per chunk, one scale group

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// two bytes (q + 128) -> half2 q, exactly: 0x64XX is 1024 + XX
__device__ __forceinline__ uint32_t u8x2_half2(uint32_t w, uint32_t sel) {
    uint32_t h = __byte_perm(w, 0x64646464u, sel);
    __half2 r = __hsub2(*(__half2*)&h, __half2(__float2half(1152.f), __float2half(1152.f)));
    return *(uint32_t*)&r;
}

// columns per 16-byte load of a lane, k16 steps in it
template <int WB> __host__ __device__ constexpr int cols_per_load() { return 256 / WB; }
template <int WB> __host__ __device__ constexpr int steps_per_load() { return 16 / WB; }

// The mma A fragment of k16 step `ks` (of a 128-column group) from the group's loads `w`
template <int WB>
__device__ __forceinline__ void a_frag(const uint4* w, int ks, uint32_t* a) {
    constexpr int SPU = steps_per_load<WB>();
    uint4 u = w[ks / SPU];
    int sub = ks % SPU;
    if (WB == 16) {
        a[0] = u.x; a[1] = u.y; a[2] = u.z; a[3] = u.w;
    } else {
        uint32_t lo = sub ? u.z : u.x, hi = sub ? u.w : u.y;
        a[0] = u8x2_half2(lo, 0x4140); a[1] = u8x2_half2(lo, 0x4342);
        a[2] = u8x2_half2(hi, 0x4140); a[3] = u8x2_half2(hi, 0x4342);
    }
}

enum { EPI_PLAIN = 0, EPI_SILU = 1, EPI_ARGMAX = 2 };

// Weights of decode steps stream through L2 once: load them evict-first (no L1 allocation), so they do not push
// the activations and small parameters out of L2.
__device__ __forceinline__ unsigned long long evict_first_policy() {
    unsigned long long p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}
__device__ __forceinline__ uint4 ld_stream(const uint4* a, unsigned long long pol) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(a), "l"(pol));
    return v;
}


// Order-preserving map of a float to an unsigned int; key = (ordered value, ~index): atomicMax picks the largest
// logit and, among equal ones, the lowest index.
__device__ __forceinline__ unsigned long long argmax_key(float v, int i) {
    uint32_t b = __float_as_uint(v);
    b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return ((unsigned long long)b << 32) | (0xffffffffu - (uint32_t)i);
}

// Matrix product epilogue: acc holds rows g, g + 8 of the warp's tile (c0, c1: row g, tokens 2 t4 and 2 t4 + 1 of
// each n8 tile; c2, c3: row g + 8). Y is already offset to the block's split for plain products; EPI_SILU writes
// the next product's input, XIN.
template <int NTT, int EPI>
__device__ __forceinline__ void epilogue(const float (&acc)[NTT][4], int tile, int tok0, int n, float* Y, int ldy,
                                         unsigned long long* keys) {
    int lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    if (EPI == EPI_SILU) {
#pragma unroll
        for (int j = 0; j < NTT; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                int t = tok0 + j * 8 + 2 * t4 + e;
                float a = acc[j][e], b = acc[j][2 + e];
                if (t < n) st_x((XIN*)Y + (size_t)t * ldy + tile * 8 + g, silu(a) * b);
            }
    } else {
#pragma unroll
        for (int j = 0; j < NTT; j++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                int t = tok0 + j * 8 + 2 * t4 + (i & 1), r = tile * 16 + g + (i >> 1) * 8;
                if (t < n) Y[(size_t)t * ldy + r] = acc[j][i];
            }
        if (EPI == EPI_ARGMAX) {
#pragma unroll
            for (int j = 0; j < NTT; j++)
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    int r = tile * 16 + g;
                    unsigned long long k = max(argmax_key(acc[j][e], r), argmax_key(acc[j][2 + e], r + 8));
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) k = max(k, __shfl_xor_sync(0xffffffff, k, o));
                    int t = tok0 + j * 8 + 2 * t4 + e;
                    if (g == 0 && t < n) atomicMax(keys + t, k);
                }
        }
    }
}

// WB: weight width (16, 8); NTT: n8 token tiles per block.
// xrows: X row of token t (null: t). EPI_PLAIN writes Y + split * n * ldy (partial products, summed by the
// consumer); EPI_SILU: tile p holds rows 8p..8p+7 of w1 and of w3, Y[t][8p + g] = silu(w1 . x) * (w3 . x);
// EPI_ARGMAX: plain, plus the running argmax key per token.
template <int WB, int NTT, int EPI>
__device__ __forceinline__ void gemm(const uint4* __restrict__ W, const float* __restrict__ S,
                                     const XIN* __restrict__ X, const int* __restrict__ xrows, int K, int ntiles,
                                     int n, int kper, float* __restrict__ Y, int ldy, unsigned long long* keys) {
    constexpr int TOK = 8 * NTT, PAD = KC + 8;  // row stride of 4 (mod 32) words: conflict-free B fragment loads
    constexpr int CPL = cols_per_load<WB>(), VPC = KC / CPL;  // uint4 per lane per chunk
    constexpr bool Q = WB == 8;  // Q: f32 scales applied after the products
    extern __shared__ __align__(16) __half xsm[];  // [2][TOK * PAD]: dynamic (PROMPT_SMEM bytes)
    __half (*xs)[TOK * PAD] = (__half (*)[TOK * PAD])xsm;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2;
    int tile = blockIdx.x * PWARPS + warp;
    int tok0 = blockIdx.z * TOK;
    int k0 = blockIdx.y * kper, k1 = min(K, k0 + kper), nch = (k1 - k0) / KC;
    bool active = tile < ntiles;
    const uint4* wp = W + (size_t)min(tile, ntiles - 1) * (K / CPL) * 32 + (k0 / CPL) * 32 + lane;
    const float* sp = S + ((size_t)min(tile, ntiles - 1) * (K / 128) + k0 / 128) * 16;

    // stage X columns [kc, kc + KC) of the block's tokens (cp.async, waited for before the barrier that ends the chunk
    // computed meanwhile)
    auto stage = [&](int c, int buf) {
        int kc = k0 + c * KC;
        for (int i = threadIdx.x; i < TOK * KC / 8; i += PWARPS * 32) {
            int r = i / (KC / 8), col = (i % (KC / 8)) * 8, t = tok0 + r;
            cp_async16(&xs[buf][r * PAD + col], X + (size_t)(t < n ? (xrows ? xrows[t] : t) : 0) * K + kc + col, t < n);
        }
    };
    uint4 wc[VPC], wn[VPC];
    float sc[2], sn[2];
    auto load = [&](int c, uint4* w, float* s) {
        if (!active || c >= nch) return;
#pragma unroll
        for (int v = 0; v < VPC; v++) w[v] = __ldg(wp + (size_t)(c * VPC + v) * 32);
#pragma unroll
        for (int r = 0; r < 2; r++) {
            if (Q) s[r] = __ldg(sp + c * 16 + g + 8 * r);
        }
    };
    float acc[NTT][4];
#pragma unroll
    for (int j = 0; j < NTT; j++)
#pragma unroll
        for (int i = 0; i < 4; i++) acc[j][i] = 0.f;

    load(0, wc, sc);
    stage(0, 0);
    cp_async_wait_all();
    __syncthreads();
    for (int c = 0; c < nch; c++) {
        int buf = c & 1;
        if (c + 1 < nch) {
            load(c + 1, wn, sn);
            stage(c + 1, buf ^ 1);
        }
        if (active) {
            const __half* xb = xs[buf];
            float tmp[NTT][4];
            if (Q) {
#pragma unroll
                for (int j = 0; j < NTT; j++)
#pragma unroll
                    for (int i = 0; i < 4; i++) tmp[j][i] = 0.f;
            }
#pragma unroll
            for (int ks = 0; ks < 8; ks++) {  // k16 steps of the chunk
                uint32_t a[4];
                a_frag<WB>(wc, ks, a);
#pragma unroll
                for (int jp = 0; jp < NTT / 2; jp++) {  // two n8 token tiles per ldmatrix.x4
                    uint32_t bf[4];
                    ldsm_x4(bf, xb + (jp * 16 + (lane & 7) + ((lane >> 4) << 3)) * PAD + ks * 16 + ((lane >> 3) & 1) * 8);
                    mma16816(Q ? tmp[2 * jp] : acc[2 * jp], a, bf[0], bf[1]);
                    mma16816(Q ? tmp[2 * jp + 1] : acc[2 * jp + 1], a, bf[2], bf[3]);
                }
            }
            if (Q) {
#pragma unroll
                for (int j = 0; j < NTT; j++) {
                    acc[j][0] += sc[0] * tmp[j][0]; acc[j][1] += sc[0] * tmp[j][1];
                    acc[j][2] += sc[1] * tmp[j][2]; acc[j][3] += sc[1] * tmp[j][3];
                }
            }
        }
        cp_async_wait_all();
        __syncthreads();
        if (c + 1 < nch) {
#pragma unroll
            for (int v = 0; v < VPC; v++) wc[v] = wn[v];
            sc[0] = sn[0]; sc[1] = sn[1];
        }
    }
    if (active)
        epilogue<NTT, EPI>(acc, tile, tok0, n, Y + (EPI == EPI_PLAIN ? (size_t)blockIdx.y * n * ldy : 0), ldy, keys);
}

// Decode-sized steps (8 * NTT tokens at most): the block stages its K range of X (KD / NTT columns at a time; one
// piece unless a full-K epilogue needs more), then each warp streams its tile's weights on its own, 8 loads per stage
// (128 columns of f16, 256 of int8), two stages in flight, each weight fragment used for NTT n8 token tiles.
#define KD 1024
#define DEC_LOADS 8  // loads of 16 bytes per lane and decode stage

// Block (x, y): tiles x * WARPS .., K range y * kper ..; xs: TOK * (KD / NTT + 8) halves of shared memory
template <int WB, int NTT, int EPI>
__device__ __forceinline__ void gemm_dec(const uint4* __restrict__ W, const float* __restrict__ S,
                                         const XIN* __restrict__ X, const int* __restrict__ xrows, int K,
                                         int ntiles, int n, int kper, float* __restrict__ Y, int ldy,
                                         unsigned long long* keys, __half* xs) {
    constexpr int CPL = cols_per_load<WB>(), TOK = 8 * NTT, PAD = KD / NTT + 8, VS = DEC_LOADS, SC = VS * CPL, G = SC / 128;
    constexpr bool Q = WB == 8;  // Q: f32 scales applied per group after the products
    int bx = blockIdx.x, by = blockIdx.y;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    int tile = bx * WARPS + warp;
    unsigned long long pol = evict_first_policy();
    float acc[NTT][4];
#pragma unroll
    for (int j = 0; j < NTT; j++)
#pragma unroll
        for (int i = 0; i < 4; i++) acc[j][i] = 0.f;
    // the block's K range in pieces of at most KD / NTT columns (one piece unless a full-K epilogue needs more)
    int kb = by * kper, kend = min(K, kb + kper);
    for (int k0 = kb; k0 < kend; k0 += KD / NTT) {
        int kr = min(kend, k0 + KD / NTT) - k0, ns = kr / SC;
        bool live = tile < ntiles;
        const uint4* wp = W + (size_t)tile * (K / CPL) * 32 + (k0 / CPL) * 32 + lane;
        const float* sp = S + ((size_t)tile * (K / 128) + k0 / 128) * 16;
        uint4 wa[VS], wb[VS];
        float sa[G][2], sb[G][2];
        auto load = [&](int st, uint4* w, float (*s)[2]) {
            if (!live || st >= ns) return;
#pragma unroll
            for (int v = 0; v < VS; v++) w[v] = ld_stream(wp + (size_t)(st * VS + v) * 32, pol);
#pragma unroll
            for (int q = 0; q < G; q++)
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    if (Q) s[q][r] = __ldg(sp + (st * G + q) * 16 + g + 8 * r);
                }
        };
        auto compute = [&](int st, const uint4* w, float (*s)[2]) {
#pragma unroll
            for (int q = 0; q < G; q++) {
                float tmp[NTT][4];
#pragma unroll
                for (int j = 0; j < NTT; j++)
#pragma unroll
                    for (int i = 0; i < 4; i++) tmp[j][i] = 0.f;
#pragma unroll
                for (int ks = 0; ks < 8; ks++) {
                    uint32_t a[4];
                    a_frag<WB>(w + q * (128 / CPL), ks, a);
#pragma unroll
                    for (int j = 0; j < NTT; j++) {
                        const __half* xr = xs + (j * 8 + g) * PAD + st * SC + q * 128 + ks * 16 + 2 * t4;
                        mma16816(Q ? tmp[j] : acc[j], a, *(const uint32_t*)xr, *(const uint32_t*)(xr + 8));
                    }
                }
                if (Q) {
#pragma unroll
                    for (int j = 0; j < NTT; j++) {
                        acc[j][0] += s[q][0] * tmp[j][0]; acc[j][1] += s[q][0] * tmp[j][1];
                        acc[j][2] += s[q][1] * tmp[j][2]; acc[j][3] += s[q][1] * tmp[j][3];
                    }
                }
            }
        };
        load(0, wa, sa);  // the first weights do not depend on X: in flight while X is staged
        if (k0 > kb) __syncthreads();
        // X is copied as it is (cp.async.cg reads through L2: coherent)
        for (int i = threadIdx.x; i < TOK * kr / 8; i += WARPS * 32) {
            int r = i / (kr / 8), col = (i % (kr / 8)) * 8;
            cp_async16(&xs[r * PAD + col], X + (size_t)(r < n ? (xrows ? xrows[r] : r) : 0) * K + k0 + col, r < n);
        }
        cp_async_wait_all();
        __syncthreads();
        if (!live) continue;
        for (int st = 0; st < ns; st += 2) {
            load(st + 1, wb, sb);
            compute(st, wa, sa);
            if (st + 1 >= ns) break;
            load(st + 2, wa, sa);
            compute(st + 1, wb, sb);
        }
    }
    if (tile < ntiles)
        epilogue<NTT, EPI>(acc, tile, 0, n, Y + (EPI == EPI_PLAIN ? (size_t)by * n * ldy : 0), ldy, keys);
}

#define GEMM_KERNEL(NAME, WB, NTT, DEC, EPI)                                                                      \
    extern "C" __global__ void __launch_bounds__((DEC ? WARPS : PWARPS) * 32)                                     \
        NAME(const uint4* W, const float* S, const XIN* X, const int* xrows, int K, int ntiles, int n, int kper,   \
             float* Y, int ldy, unsigned long long* keys) {                                                       \
        if (DEC) {                                                                                                \
            __shared__ __align__(16) __half xs[8 * NTT * (KD / NTT + 8)];                                         \
            gemm_dec<WB, NTT, EPI>(W, S, X, xrows, K, ntiles, n, kper, Y, ldy, keys, xs);                       \
        }                                                                                                         \
        else                                                                                                      \
            gemm<WB, NTT, EPI>(W, S, X, xrows, K, ntiles, n, kper, Y, ldy, keys);                                \
    }
#define GEMM_TILE(WT, TT, WB, NTT, DEC)                                     \
    GEMM_KERNEL(gemm_##WT##_##TT##_plain, WB, NTT, DEC, EPI_PLAIN)          \
    GEMM_KERNEL(gemm_##WT##_##TT##_silu, WB, NTT, DEC, EPI_SILU)            \
    GEMM_KERNEL(gemm_##WT##_##TT##_argmax, WB, NTT, DEC, EPI_ARGMAX)
// decode steps of up to 8, 16, 32 tokens (gemm_dec); prompt chunks: 64 tokens per block. WB: weight bits.
#define GEMM_SET(WT, WB)                                                    \
    GEMM_TILE(WT, t8, WB, 1, 1)                                             \
    GEMM_TILE(WT, t16, WB, 2, 1)                                            \
    GEMM_TILE(WT, t32, WB, 4, 1)                                            \
    GEMM_TILE(WT, t64, WB, PNTT, 0)
