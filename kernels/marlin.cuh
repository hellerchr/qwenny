// Marlin (Elias Frantar, IST Austria; github.com/IST-DASLab/marlin, kernel marlin_cuda_kernel.cu), the FP16 x INT4
// matrix product of the Qwen models' stored int4 matrices (symmetric, groups of 128 columns), for prompt chunks and
// decode steps alike. Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at), licensed under the Apache
// License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0); distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND.
//
// Changed here: groups of 128 columns only (group_blocks 8; no per-column scales); results in f32 instead of f16:
// blocks that share an output slice sum their parts in f32 (in a scratch slot of the slice instead of the output),
// and the last one writes Y [m][n] f32 (a split-K product of one part, as the consumers read it) or, for the SwiGLU
// product, silu(gate) * up as f16; A and Y start at row `row0` of the step (the host launches one kernel per run of
// 64-token problems and one for the remainder, as Marlin's own launcher does); every pipeline stage holds its tile's
// scales (Marlin's stages share them per group, a race between the warps: fetch_to_shared).
//
// Names: m tokens (the rows of A = X [m][k] f16), n output rows of the weight matrix (Marlin's columns), k its
// columns. B is the weight matrix transposed, k x n, in Marlin's layout (core.rs pack_marlin): per 16 x 64 piece
// (k16 step, 64 outputs) each lane's 4 words hold the mma B fragments of four 16-output blocks, nibbles q + 8;
// scales f16 [k / 128][n], permuted per 64 outputs to match the fragments (with zero points, ZP: then (8 - z) * scale
// in the same layout).
#pragma once
#include "common.cuh"

namespace marlin {

__host__ __device__ constexpr int ceildiv(int a, int b) { return (a + b - 1) / b; }

// Instances of `Vec` organize groups of registers (inputs of tensor core operations): index accesses must be
// compile-time constants, hence `#pragma unroll` throughout.
template <typename T, int n>
struct Vec {
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
};

using I4 = Vec<int, 4>;
// mma.m16n8k16 fragments (PTX ISA "matrix fragments for mma.m16n8k16 with floating point type")
using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragS = Vec<half2, 1>;  // quantization scales

// Predicated asynchronous global -> shared copy (A: the last 16-row block of a step may be partial)
__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.cg.shared.global [%1], [%2], 16;\n"
        "}\n" ::"r"((int)pred), "r"(smem), "l"(glob_ptr));
}

// Asynchronous copy, evict-first in L2: the weights B are read once and must not push A and the partial sums out
__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .b64 p;\n"
        "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
        "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, p;\n"
        "}\n" ::"r"(smem), "l"(glob_ptr));
}

__device__ inline void cp_async_fence() { asm volatile("cp.async.commit_group;\n" ::); }

// Wait until at most `n` async copy groups are pending
template <int n>
__device__ inline void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

// m16n8k16 tensor core product, f16 inputs, f32 accumulation
__device__ inline void mma(const FragA& a_frag, const FragB& frag_b, FragC& frag_c) {
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    float* c = reinterpret_cast<float*>(&frag_c);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// A full 16x16 A fragment from shared memory, directly in tensor core layout
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
    uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(smem));
}

// Three-input logical operation by lookup table (the compiler does not always find the LOP3 itself)
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
    int res;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
    return res;
}

// An int32 of 4-bit values -> a B fragment of 4 f16 values q - 8, exactly (as FasterTransformer's interleaved
// numeric conversion: 0x64XX is 1024 + XX; the high nibbles are scaled by 1/16 in the same fma)
__device__ inline FragB dequant(int q) {
    const int LO = 0x000f000f;
    const int HI = 0x00f000f0;
    const int EX = 0x64006400;
    int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
    int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
    // the symmetric zero point 8 folded into SUB and ADD
    const int SUB = 0x64086408;
    const int MUL = 0x2c002c00;
    const int ADD = 0xd480d480;
    FragB frag_b;
    frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<const half2*>(&SUB));
    frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi), *reinterpret_cast<const half2*>(&MUL),
                        *reinterpret_cast<const half2*>(&ADD));
    return frag_b;
}

// Times the group's scale (f16; (q - 8) * scale is exact for bf16 scales: |q - 8| <= 8 times an 8-bit significand)
__device__ inline void scale(FragB& frag_b, FragS& frag_s, int i) {
    half2 s = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);
    frag_b[0] = __hmul2(frag_b[0], s);
    frag_b[1] = __hmul2(frag_b[1], s);
}

// With a zero point z: (q - 8) * scale + (8 - z) * scale, the second term stored (exact in f16: |8 - z| has at most 3
// significant bits), the sum rounded once: (q - z) * scale in f16, the dequantized weight rounded once to f16
__device__ inline void scale_z(FragB& frag_b, FragS& frag_s, FragS& frag_z, int i) {
    half2 s = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);
    half2 z = __half2half2(reinterpret_cast<__half*>(&frag_z)[i]);
    frag_b[0] = __hfma2(frag_b[0], s, z);
    frag_b[1] = __hfma2(frag_b[1], s, z);
}

// Wait until the lock reaches `count` (the slice's blocks before this one are done)
__device__ inline void barrier_acquire(int* lock, int count) {
    if (threadIdx.x == 0) {
        int state = -1;
        do
            asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
        while (state != count);
    }
    __syncthreads();
}

// Count this block as done (its writes visible first); the slice's last block resets the lock for the next product
__device__ inline void barrier_release(int* lock, bool reset = false) {
    __syncthreads();
    if (threadIdx.x == 0) {
        if (reset) {
            lock[0] = 0;
            return;
        }
        int val = 1;
        asm volatile("fence.acq_rel.gpu;\n");
        asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
    }
}

constexpr int THREADS = 256;  // 8 warps: 2 per scheduler, many registers each
constexpr int STAGES = 4;     // global -> shared pipeline
constexpr int GROUP_BLOCKS = 8;  // 16-column blocks of k per scale group (QG = 128)
// a scratch slot (floats): one 64-token x 256-output tile in fragment layout, the largest configuration's
constexpr int SLOT = 64 * 256;

enum { OUT_PLAIN = 0, OUT_SILU = 1, OUT_ARGMAX = 2 };

// Y [m][n] = A [m][k] . B [k][n] for the step's rows row0 .. row0 + prob_m (OUT_SILU: Y [m][n / 2] f16, see
// write_result; OUT_ARGMAX: as plain, and keys[row] the running argmax over the outputs below n_valid). The grid is
// one block per SM. red: gridDim.x scratch slots of SLOT floats; locks: gridDim.x ints, zero (and left zero).
template <int thread_m_blocks,  // 16-token blocks of a block's tile (m)
          int thread_n_blocks,  // 16-output blocks (n)
          int thread_k_blocks,  // 16-column blocks (k)
          int OUT,
          bool ZP>              // zero points: (8 - z) * scale after the scales in s, the same layout (scale_z)
__device__ __forceinline__ void mul(const int4* __restrict__ A, const int4* __restrict__ B, void* __restrict__ Y,
                                    const int4* __restrict__ s, int prob_m, int prob_n, int prob_k, int row0,
                                    float4* __restrict__ red, int* __restrict__ locks, unsigned long long* keys,
                                    int n_valid) {
    constexpr int threads = THREADS, stages = STAGES, group_blocks = GROUP_BLOCKS;
    // Each block processes one "stripe" of B of (roughly) the same size, which may span several column "slices"
    // (16 * thread_n_blocks outputs wide), as in this 3 x 3 example on 5 SMs:
    //   0 1 3
    //   0 2 3
    //   1 2 4
    // so that every SM is busy for any shape, with few cross-block reductions.

    // larger products run several 64-token problems ("parallel") side by side, for a better partition
    int parallel = 1;
    if (prob_m > 16 * thread_m_blocks) {
        parallel = prob_m / (16 * thread_m_blocks);
        prob_m = 16 * thread_m_blocks;
    }
    A += (size_t)row0 * prob_k / 8;
    int m0 = row0;  // the first row (of Y) of the current problem

    int k_tiles = prob_k / 16 / thread_k_blocks;
    int n_tiles = prob_n / 16 / thread_n_blocks;
    int iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);
    // stripes are a whole number of scale groups: no stripe starts in the middle of a group
    iters = (group_blocks / thread_k_blocks) * ceildiv(iters, (group_blocks / thread_k_blocks));

    int slice_row = (iters * blockIdx.x) % k_tiles;
    int slice_col_par = (iters * blockIdx.x) / k_tiles;
    int slice_col = slice_col_par;
    int slice_iters;      // block tiles in the current slice
    int slice_count = 0;  // blocks active in the current slice
    int slice_idx;        // index of this block in the slice, numbered bottom to top

    // parallel problems: remap indices and advance the pointers
    if (slice_col_par >= n_tiles) {
        A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 8;
        m0 += (slice_col_par / n_tiles) * 16 * thread_m_blocks;
        slice_col = slice_col_par % n_tiles;
    }

    // what synchronizing the current slice needs
    auto init_slice = [&]() {
        slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
        if (slice_iters < 0 || slice_col_par >= n_tiles * parallel) slice_iters = 0;
        if (slice_iters == 0) return;
        if (slice_row + slice_iters > k_tiles) slice_iters = k_tiles - slice_row;
        slice_count = 1;
        slice_idx = 0;
        int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
        if (col_first <= k_tiles * (slice_col_par + 1)) {
            int col_off = col_first - k_tiles * slice_col_par;
            slice_count = ceildiv(k_tiles - col_off, iters);
            if (col_off > 0) slice_count++;
            int delta_first = iters * blockIdx.x - col_first;
            if (delta_first < 0 || (col_off == 0 && delta_first == 0))
                slice_idx = slice_count - 1;
            else {
                slice_idx = slice_count - 1 - delta_first / iters;
                if (col_off > 0) slice_idx--;
            }
        }
        if (slice_col == n_tiles) {
            A += 16 * thread_m_blocks * prob_k / 8;
            m0 += 16 * thread_m_blocks;
            slice_col = 0;
        }
    };
    init_slice();

    int a_gl_stride = prob_k / 8;                                  // A row stride in global memory (int4)
    constexpr int a_sh_stride = 16 * thread_k_blocks / 8;          // ... of an A tile in shared memory
    constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;      // between subsequent A tiles in global memory
    int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);  // between accesses within a tile
    constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);  // between shared memory writes
    constexpr int a_sh_rd_delta_o = 2 * ((threads / 32) / (thread_n_blocks / 4));  // between shared tile reads
    constexpr int a_sh_rd_delta_i = a_sh_stride * 16;              // within a shared memory tile
    constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);  // size of a tile
    constexpr int a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta);  // shared write iterations per tile

    int b_gl_stride = 16 * prob_n / 32;
    constexpr int b_sh_stride = 32 * thread_n_blocks / 4;
    int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
    int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);
    constexpr int b_sh_wr_delta = threads;
    constexpr int b_sh_rd_delta = threads;
    constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
    constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

    int s_gl_stride = prob_n / 8;
    constexpr int s_sh_stride = 16 * thread_n_blocks / 8;
    constexpr int s_sh_stage = s_sh_stride;
    int s_gl_rd_delta = s_gl_stride;

    // global A read index of this thread
    int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
    a_gl_rd += a_gl_rd_delta_o * slice_row;
    // shared write index
    int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
    // shared read index
    int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;
    a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

    int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
    b_gl_rd += b_sh_stride * slice_col;
    b_gl_rd += b_gl_rd_delta_o * slice_row;
    int b_sh_wr = threadIdx.x;
    int b_sh_rd = threadIdx.x;

    int s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) + s_sh_stride * slice_col + threadIdx.x;
    int s_sh_wr = threadIdx.x;
    // scales of grouped quantization: a half2 tile in column-major layout
    int s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;

    // which threads must not read in which iterations: more threads than a tile needs, or a partial last 16 tokens
    bool a_sh_wr_pred[a_sh_wr_iters];
#pragma unroll
    for (int i = 0; i < a_sh_wr_iters; i++) a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;
    bool s_sh_wr_pred = threadIdx.x < s_sh_stride;

    // An XOR layout of the A tiles in shared memory makes both the writes and the fragment reads bank-conflict free:
    // the 16-byte pieces of 8 consecutive threads never share banks.
    auto transform_a = [&](int i) {
        int row = i / a_gl_rd_delta_o;
        return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
    };
    // all shared memory accesses are static (the main loop is unrolled): the transformed indices are precomputed
    int a_sh_wr_trans[a_sh_wr_iters];
#pragma unroll
    for (int i = 0; i < a_sh_wr_iters; i++) a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
    int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
#pragma unroll
    for (int i = 0; i < b_sh_wr_iters; i++) {
#pragma unroll
        for (int j = 0; j < thread_m_blocks; j++)
            a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
    }

    // B strides are not constant: one pointer per access of a tile breaks the dependencies between them
    const int4* B_ptr[b_sh_wr_iters];
#pragma unroll
    for (int i = 0; i < b_sh_wr_iters; i++) B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

    extern __shared__ int4 sh[];
    // the global fetch pipelines
    int4* sh_a = sh;
    int4* sh_b = sh_a + (stages * a_sh_stage);
    int4* sh_s = sh_b + (stages * b_sh_stage);
    int4* sh_z = sh_s + (stages * s_sh_stage);
    const int4* z = s + (size_t)(prob_k / 128) * s_gl_stride;
    // double buffers of shared memory reads
    FragA frag_a[2][thread_m_blocks];
    I4 frag_b_quant[2];
    FragC frag_c[thread_m_blocks][4][2];
    FragS frag_s[2][4];
    FragS frag_z[2][4];

    auto zero_accums = [&]() {
#pragma unroll
        for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++) reinterpret_cast<float*>(frag_c)[i] = 0;
    };

    // fetch the next A, B and s tile into the next pipeline stage
    auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
        if (pred) {
            int4* sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
            for (int i = 0; i < a_sh_wr_iters; i++) {
                cp_async4_pred(&sh_a_stage[a_sh_wr_trans[i]], &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
                               a_sh_wr_pred[i]);
            }
            int4* sh_b_stage = sh_b + b_sh_stage * pipe;
#pragma unroll
            for (int i = 0; i < b_sh_wr_iters; i++) {
                cp_async4_stream(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);
                B_ptr[i] += b_gl_rd_delta_o;
            }
            // the tile's group scales into its own stage (also when the group's previous tile has them: a stage that
            // shares another's scales would read them while the warp that fetches them overwrites that stage with a
            // later group's, the warps not synchronized in between; rare, timing-dependent wrong products)
            int4* sh_s_stage = sh_s + s_sh_stage * pipe;
            if (s_sh_wr_pred) cp_async4_stream(&sh_s_stage[s_sh_wr], &s[s_gl_rd]);
            if (ZP && s_sh_wr_pred) cp_async4_stream(&sh_z[s_sh_stage * pipe + s_sh_wr], &z[s_gl_rd]);
            if (pipe % (group_blocks / thread_k_blocks) == group_blocks / thread_k_blocks - 1) s_gl_rd += s_gl_rd_delta;
        }
        // a fence even when winding down the pipeline, so that waiting stays correct
        cp_async_fence();
    };

    // until the next tile is in shared memory: `stages - 2` fetches in flight, as the register loads are double
    // buffered and a stage must be fully read before it is overwritten
    auto wait_for_stage = [&]() {
        cp_async_wait<stages - 2>();
        __syncthreads();
    };

    // the next sub-tile from shared memory into the current register buffer (the scales reloaded each time: better
    // instruction scheduling than keeping them)
    auto fetch_to_registers = [&](int k, int pipe) {
        reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s[s_sh_stage * pipe + s_sh_rd];
        if (ZP) reinterpret_cast<int4*>(&frag_z[k % 2])[0] = sh_z[s_sh_stage * pipe + s_sh_rd];
        int4* sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
        for (int i = 0; i < thread_m_blocks; i++) ldsm4(frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
        int4* sh_b_stage = sh_b + b_sh_stage * pipe;
        frag_b_quant[k % 2] = *reinterpret_cast<I4*>(&sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);
    };

    // the tensor core products of a sub-tile; m the inner loop, overlapping dequantization and products
    auto matmul = [&](int k) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int b_quant = frag_b_quant[k % 2][j];
            int b_quant_shift = b_quant >> 8;
            FragB frag_b0 = dequant(b_quant);
            FragB frag_b1 = dequant(b_quant_shift);
            if (ZP) {
                scale_z(frag_b0, frag_s[k % 2][j], frag_z[k % 2][j], 0);
                scale_z(frag_b1, frag_s[k % 2][j], frag_z[k % 2][j], 1);
            } else {
                scale(frag_b0, frag_s[k % 2][j], 0);
                scale(frag_b1, frag_s[k % 2][j], 1);
            }
#pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
                mma(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);
                mma(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);
            }
        }
    };

    // The warps of a block split the tile's k range too (more warps at a moderate n width): their partial sums of
    // the same outputs are reduced in shared memory, logarithmically, with no needless reads or writes.
    auto thread_block_reduce = [&]() {
        constexpr int red_off = threads / b_sh_stride / 2;
        if (red_off >= 1) {
            int red_idx = threadIdx.x / b_sh_stride;
            constexpr int red_sh_stride = b_sh_stride * 4 * 2;
            constexpr int red_sh_delta = b_sh_stride;
            int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
#pragma unroll
            for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
#pragma unroll
                for (int i = red_off; i > 0; i /= 2) {
                    if (i <= red_idx && red_idx < 2 * i) {
#pragma unroll
                        for (int j = 0; j < 4 * 2; j++) {
                            int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
                            if (i < red_off) {
                                float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * j + red_sh_rd]);
                                float* c_wr = reinterpret_cast<float*>(&sh[red_sh_wr]);
#pragma unroll
                                for (int k = 0; k < 4; k++)
                                    reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] += c_rd[k] + c_wr[k];
                            }
                            sh[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
                        }
                    }
                    __syncthreads();
                }
                if (red_idx == 0) {
#pragma unroll
                    for (int i = 0; i < 4 * 2; i++) {
                        float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * i + red_sh_rd]);
#pragma unroll
                        for (int j = 0; j < 4; j++) reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] += c_rd[j];
                    }
                }
                __syncthreads();
            }
        }
    };

    // After the block reduce the first n-group of warps holds the tile: warp w outputs 64 w .. 64 w + 63 of the
    // slice, frag_c[i][j][h] the n8 tile at 16 j + 8 h (rows 16 i + g, + 8; outputs 2 t4, 2 t4 + 1).
    constexpr int active_threads = 32 * thread_n_blocks / 4;

    // Blocks that share a slice add their parts one after the other in f32 (slice_idx 0 first), in a scratch slot
    // instead of the output (which may be f16 or narrower): a float4 per FragC and thread, read and written whole.
    // The slot and lock are those of the slice's lowest block, the one whose stripe holds the slice's first tile: a
    // block is that for at most one shared slice, so gridDim.x slots serve any shape, and the slots stay in L2.
    auto global_reduce = [&](float4* slot, bool first, bool last) {
        if (threadIdx.x < active_threads) {
#pragma unroll
            for (int i = 0; i < thread_m_blocks * 4 * 2; i++) {
                float* c = reinterpret_cast<float*>(&frag_c) + 4 * i;
                float4* p = slot + i * active_threads + threadIdx.x;
                if (!first) {
                    float4 r = __ldcg(p);
                    c[0] += r.x; c[1] += r.y; c[2] += r.z; c[3] += r.w;
                }
                if (!last) __stcg(p, make_float4(c[0], c[1], c[2], c[3]));
            }
        }
    };

    // The slice's last block writes its rows below prob_m. OUT_PLAIN: f32 Y [m][n], 32-byte pieces per row and quad.
    // OUT_SILU: each 16-output block p holds 8 gate rows (n8 tile h = 0) and their 8 up rows (h = 1; core.rs
    // row_order), so a thread has gate and up of the same outputs 8 p + 2 t4 (+ 1): Y [m][n / 2] = silu(gate) * up, f16.
    auto write_result = [&]() {
        if (threadIdx.x >= active_threads) return;
        int lane = threadIdx.x % 32, g = lane / 4, t4 = lane % 4;
        int col = 16 * thread_n_blocks * slice_col + 64 * (threadIdx.x / 32);  // the warp's first output
#pragma unroll
        for (int i = 0; i < thread_m_blocks; i++)
#pragma unroll
            for (int r = 0; r < 2; r++) {
                int row = 16 * i + g + 8 * r;
                if (row >= prob_m) continue;
                size_t y0 = (size_t)(m0 + row) * (OUT == OUT_SILU ? prob_n / 2 : prob_n);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    FragC& c0 = frag_c[i][j][0];
                    FragC& c1 = frag_c[i][j][1];
                    if (OUT == OUT_SILU) {
                        half2 v = __floats2half2_rn(silu(c0[2 * r]) * c1[2 * r], silu(c0[2 * r + 1]) * c1[2 * r + 1]);
                        *reinterpret_cast<half2*>((__half*)Y + y0 + (col + 16 * j) / 2 + 2 * t4) = v;
                    } else {
                        float* y = (float*)Y + y0 + col + 16 * j + 2 * t4;
                        *reinterpret_cast<float2*>(y) = make_float2(c0[2 * r], c0[2 * r + 1]);
                        *reinterpret_cast<float2*>(y + 8) = make_float2(c1[2 * r], c1[2 * r + 1]);
                    }
                }
                if (OUT == OUT_ARGMAX) {
                    // the row's largest output of the thread, then of its quad (the lanes of row g), into keys
                    unsigned long long k = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++)
#pragma unroll
                        for (int h = 0; h < 2; h++)
#pragma unroll
                            for (int e = 0; e < 2; e++) {
                                int o = col + 16 * j + 8 * h + 2 * t4 + e;
                                float v = frag_c[i][j][h][2 * r + e];
                                if (o < n_valid) k = max(k, argmax_key(v, o));
                            }
                    // the quad's 4 lanes (row g) take this branch together: lanes of rows past prob_m do not
                    unsigned quad = 0xfu << (lane & ~3);
                    k = max(k, __shfl_xor_sync(quad, k, 1));
                    k = max(k, __shfl_xor_sync(quad, k, 2));
                    if (t4 == 0 && k) atomicMax(keys + m0 + row, k);
                }
            }
    };

    // start the global fetch and register load pipelines
    auto start_pipes = [&]() {
#pragma unroll
        for (int i = 0; i < stages - 1; i++) fetch_to_shared(i, i, i < slice_iters);
        zero_accums();
        wait_for_stage();
        fetch_to_registers(0, 0);
        a_gl_rd += a_gl_rd_delta_o * (stages - 1);
    };
    start_pipes();

    // main loop
    while (slice_iters) {
        // unrolled over both pipelines (both of even length, so the next iteration starts at index 0): all shared
        // memory accesses are static
#pragma unroll
        for (int pipe = 0; pipe < stages;) {
#pragma unroll
            for (int k = 0; k < b_sh_wr_iters; k++) {
                fetch_to_registers(k + 1, pipe % stages);
                if (k == b_sh_wr_iters - 2) {
                    fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);
                    pipe++;
                    wait_for_stage();
                }
                matmul(k);
            }
            slice_iters--;
            if (slice_iters == 0) break;
        }
        a_gl_rd += a_gl_rd_delta_o * stages;

        // the slice's results, then on to the next slice (other loop shapes compiled to slower code)
        if (slice_iters == 0) {
            cp_async_wait<0>();
            bool last = slice_idx == slice_count - 1;
            thread_block_reduce();
            if (slice_count > 1) {  // a global reduction only when several blocks share the slice
                int slot = k_tiles * slice_col_par / iters;
                barrier_acquire(&locks[slot], slice_idx);
                global_reduce(red + (size_t)slot * (SLOT / 4), slice_idx == 0, last);
                barrier_release(&locks[slot], last);
            }
            if (last) write_result();
            slice_row = 0;
            slice_col_par++;
            slice_col++;
            init_slice();
            if (slice_iters) {
                a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
#pragma unroll
                for (int i = 0; i < b_sh_wr_iters; i++) B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
                if (slice_col == 0) {
#pragma unroll
                    for (int i = 0; i < b_sh_wr_iters; i++) B_ptr[i] -= b_gl_stride;
                }
                s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
                start_pipes();
            }
        }
    }
}

}  // namespace marlin

// Instances: decode steps of up to 16 tokens take 128 x 128 tiles (k x n; a finer partition matters more than the
// tensor-pipe use), larger steps 64 x 256 tiles of 16 to 64 tokens. Dynamic shared memory: core.rs marlin_smem.
#define MARLIN_KERNEL(MB, NB, KB, OUT, ZP, NAME)                                                                     \
    extern "C" __global__ void __launch_bounds__(marlin::THREADS)                                                  \
        marlin_m##MB##_n##NB##_##NAME(const int4* A, const int4* B, void* Y, const int4* s, int prob_m, int prob_n, \
                                      int prob_k, int row0, float4* red, int* locks, unsigned long long* keys,     \
                                      int n_valid) {                                                                \
        marlin::mul<MB, NB, KB, OUT, ZP>(A, B, Y, s, prob_m, prob_n, prob_k, row0, red, locks, keys, n_valid);     \
    }
// per configuration: plain, silu, argmax (symmetric), then the same with zero points (_z)
#define MARLIN_SET(MB, NB, KB)                                    \
    MARLIN_KERNEL(MB, NB, KB, marlin::OUT_PLAIN, false, plain)    \
    MARLIN_KERNEL(MB, NB, KB, marlin::OUT_SILU, false, silu)      \
    MARLIN_KERNEL(MB, NB, KB, marlin::OUT_ARGMAX, false, argmax)  \
    MARLIN_KERNEL(MB, NB, KB, marlin::OUT_PLAIN, true, plain_z)   \
    MARLIN_KERNEL(MB, NB, KB, marlin::OUT_SILU, true, silu_z)     \
    MARLIN_KERNEL(MB, NB, KB, marlin::OUT_ARGMAX, true, argmax_z)

// The rows `rows[j]` of X [.][k] f16 into out [j][k] (Marlin reads its A rows contiguously): grid n, k / 8 per row
extern "C" __global__ void marlin_gather(const int4* x, const int* rows, int4* out, int k8) {
    const int4* row = x + (size_t)rows[blockIdx.x] * k8;
    for (int c = threadIdx.x; c < k8; c += blockDim.x) out[(size_t)blockIdx.x * k8 + c] = row[c];
}

// Write the 16-output block `blk` of the 64-output group `c64` of a Marlin matrix with n outputs (core.rs
// Mat::write_marlin; layout: pack_marlin): its words [k / 16][32 lanes] (per k16 step 2 n words, lane l's word of the
// block at 128 c64 + 4 l + blk) and its scales [tables * groups][16] (tables 2 with zero points). Grid: k / 16 blocks
// of 32 threads (k / 16 >= 2 groups).
extern "C" __global__ void marlin_put_block(unsigned* B, __half* S, const unsigned* words, const __half* sc, int n,
                                            int c64, int blk, int groups, int tables) {
    int k16 = blockIdx.x, lane = threadIdx.x;
    B[(size_t)k16 * 2 * n + (c64 * 32 + lane) * 4 + blk] = words[k16 * 32 + lane];
    if (k16 < groups * tables && lane < 16) {
        int c = blk * 16 + lane;
        S[(size_t)k16 * n + c64 * 64 + 8 * (c % 8) + c / 8] = sc[k16 * 16 + lane];
    }
}
