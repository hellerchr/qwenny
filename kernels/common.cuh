// Helpers shared by every kernel file: warp and block reductions, f16 conversions, shared-memory copies.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

// The matrix products' input type: they round it to f16 anyway, so its producer writes f16 and it is staged with cp.async
#define XIN __half
__device__ __forceinline__ void st_x(XIN* p, float v) { *p = __float2half_rn(v); }

// Asynchronous copies global -> shared (cp.async; wait with cp.async.wait_group). !ok: zero-fill the 16 bytes.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool ok = true) {
    uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
}

// ldmatrix: four 8x8 b16 matrices from shared memory (lane i gives the address of row i % 8 of matrix i / 8)
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const __half* p) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// Sum over the block; every thread gets the result. sh holds one float per warp.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    v = warp_sum(v);
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float t = lane < (blockDim.x >> 5) ? sh[lane] : 0.f;
    t = warp_sum(t);
    __syncthreads();
    return t;
}

__device__ __forceinline__ uint32_t pack_half2(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *(uint32_t*)&h;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// Sum of the `splits` partial products [split][n][ld] at row t, column c
__device__ __forceinline__ float part_sum(const float* p, int splits, int n, int ld, int t, int c) {
    float v = 0.f;
    for (int s = 0; s < splits; s++) v += p[((size_t)s * n + t) * ld + c];
    return v;
}
