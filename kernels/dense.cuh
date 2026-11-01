// The embedding lookup and RMSNorm, for a hidden size H defined by the including file.
#pragma once
#include "common.cuh"

// out[t] = E[ids[t]]; the first MAX_OUT blocks also clear the argmax keys of the step
extern "C" __global__ void embed(const __nv_bfloat16* E, const int* ids, float* out, unsigned long long* keys,
                                 int nkeys) {
    if (threadIdx.x == 0 && blockIdx.x < nkeys) keys[blockIdx.x] = 0;
    const __nv_bfloat16* row = E + (size_t)ids[blockIdx.x] * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) out[(size_t)blockIdx.x * H + i] = __bfloat162float(row[i]);
}

// y = w * x / rms(x); one block of 256 threads per token
extern "C" __global__ void rmsnorm(const float* x, const float* w, XIN* y, float eps) {
    __shared__ float sh[32];
    const float* xr = x + (size_t)blockIdx.x * H;
    float v[H / 256], ss = 0.f;
#pragma unroll
    for (int i = 0; i < H / 256; i++) { v[i] = xr[threadIdx.x + i * 256]; ss += v[i] * v[i]; }
    float r = rsqrtf(block_sum(ss, sh) / H + eps);
#pragma unroll
    for (int i = 0; i < H / 256; i++) {
        int c = threadIdx.x + i * 256;
        st_x(y + (size_t)blockIdx.x * H + c, w[c] * (v[i] * r));
    }
}
