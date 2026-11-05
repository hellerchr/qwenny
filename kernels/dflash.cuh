// DFlash2's parallel draft model. The target residual stream is tapped after complete decoder layers; a fixed block
// holds one verified anchor and seven mask tokens. The drafter's heads are the including file's (granite.cu: 32 query, 8
// key/value heads of 128 dims); its RMSNorm epsilon is DF_EPS (rmsnorm takes it as an argument).
#define DF_HD HD
#define DF_NH NH
#define DF_NKV NKV
#define DF_BLOCK 8
#define DF_WINDOW 2048
#define DF_RANK 256
#define DF_TOPK 16
#define DF_GROUPS (H / 16)
#define DF_DYN (4 * DF_GROUPS)
#define DF_EPS 1e-6f

// Gather the target's tapped residual rows into fc's input. taps is [taps][target step tokens][H].
extern "C" __global__ void __launch_bounds__(256) dflash_gather_rows(const XIN* taps, const int* rows, XIN* out,
                                                                      int nrows, int ntaps, int target_rows) {
    size_t o = (size_t)blockIdx.x * ntaps * H;
    for (int t = 0; t < ntaps; t++) {
        const XIN* in = taps + ((size_t)t * target_rows + rows[blockIdx.x]) * H;
        for (int c = threadIdx.x; c < H; c += 256) out[o + t * H + c] = in[c];
    }
    (void)nrows;
}

// Sum a plain product's split-K partials into the f16 input of the next operation.
extern "C" __global__ void dflash_parts_xin(const float* x, int splits, int n, int ld, XIN* y) {
    for (int c = threadIdx.x; c < ld; c += blockDim.x)
        st_x(y + (size_t)blockIdx.x * ld + c, part_sum(x, splits, n, ld, blockIdx.x, c));
}

// Prepare of a two-tap grouped dynamic convolution. dynamic is [row][2 stages][2 taps][H / 16].
extern "C" __global__ void dflash_conv_prepare(const XIN* x, const XIN* dynamic, const float* base, XIN* out) {
    int row = blockIdx.x, t = row & (DF_BLOCK - 1);
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        int g = c / 16;
        size_t at = (size_t)row * H + c;
        float v = (base[c] + __half2float(dynamic[(size_t)row * DF_DYN + g])) * __half2float(x[at]);
        if (t > 0) {
            size_t prev = at - H;
            v += (base[H + c] + __half2float(dynamic[(size_t)row * DF_DYN + DF_GROUPS + g])) * __half2float(x[prev]);
        }
        st_x(out + at, v);
    }
}

// Finish: add the convolution of a sublayer's f32 output to the residual stream.
extern "C" __global__ void dflash_conv_finish(const float* x, const XIN* dynamic, const float* base, float* residual) {
    int row = blockIdx.x, t = row & (DF_BLOCK - 1);
    const XIN* dyn = dynamic + (size_t)row * DF_DYN + 2 * DF_GROUPS;
    const float* b = base + 2 * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        int g = c / 16;
        size_t at = (size_t)row * H + c;
        float v = (b[c] + __half2float(dyn[g])) * x[at];
        if (t > 0) v += (b[H + c] + __half2float(dyn[DF_GROUPS + g])) * x[at - H];
        residual[at] += v;
    }
}

__device__ __forceinline__ float df_rope_freq(int d) {
    return powf(1e7f, -2.0f * d / DF_HD);
}

// K/V of target-context features into a slot's sliding ring. k/v cache: [slot][ring position][DF_NKV][DF_HD].
extern "C" __global__ void dflash_context_kv(const float* parts, int splits, int n, const float* k_norm,
                                              const int* slots, const int* positions, XIN* kc, XIN* vc, int k0, int ld) {
    int row = blockIdx.x, head = threadIdx.x / 32, lane = threadIdx.x & 31;
    if (head >= DF_NKV) return;
    __shared__ float ks[DF_NKV * DF_HD];
    float k[DF_HD / 32], v[DF_HD / 32], ss = 0.f;
#pragma unroll
    for (int i = 0; i < DF_HD / 32; i++) {
        int c = lane + 32 * i;
        k[i] = part_sum(parts, splits, n, ld, row, (k0 + head) * DF_HD + c);
        v[i] = part_sum(parts, splits, n, ld, row, (k0 + DF_NKV + head) * DF_HD + c);
        ss += k[i] * k[i];
    }
    for (int d = 16; d > 0; d >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, d);
    float r = rsqrtf(ss / DF_HD + DF_EPS);
    size_t out = (((size_t)slots[row] * DF_WINDOW + (positions[row] & (DF_WINDOW - 1))) * DF_NKV + head) * DF_HD;
#pragma unroll
    for (int i = 0; i < DF_HD / 32; i++) {
        int c = lane + 32 * i;
        ks[head * DF_HD + c] = k_norm[c] * k[i] * r;
        st_x(vc + out + c, v[i]);
    }
    __syncwarp();
    for (int d = lane; d < DF_HD / 2; d += 32) {
        float a = ks[head * DF_HD + d], b = ks[head * DF_HD + d + DF_HD / 2];
        float angle = positions[row] * df_rope_freq(d), cs = cosf(angle), sn = sinf(angle);
        st_x(kc + out + d, a * cs - b * sn);
        st_x(kc + out + d + DF_HD / 2, b * cs + a * sn);
    }
}

// Q/K/V of an anchor + mask block. q is scaled for exp2 attention; noise K/V are separate from the context ring.
extern "C" __global__ void dflash_block_qkv(const float* parts, int splits, int n, const float* q_norm,
                                             const float* k_norm, const int* block_pos, float* qout, XIN* kout,
                                             XIN* vout) {
    int row = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    int position = block_pos[row / DF_BLOCK] + (row & (DF_BLOCK - 1));
    __shared__ float qs[DF_NH * DF_HD], ks[DF_NKV * DF_HD];
    for (int head = warp; head < DF_NH + DF_NKV; head += blockDim.x / 32) {
        bool isq = head < DF_NH;
        int h = isq ? head : head - DF_NH;
        float x[DF_HD / 32], ss = 0.f;
#pragma unroll
        for (int i = 0; i < DF_HD / 32; i++) {
            int c = lane + 32 * i;
            int col = isq ? head * DF_HD + c : (DF_NH + h) * DF_HD + c;
            x[i] = part_sum(parts, splits, n, (DF_NH + 2 * DF_NKV) * DF_HD, row, col);
            if (isq || head < DF_NH + DF_NKV) ss += x[i] * x[i];
        }
        for (int d = 16; d > 0; d >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, d);
        float r = rsqrtf(ss / DF_HD + DF_EPS);
        const float* norm = isq ? q_norm : k_norm;
        float* sh = isq ? qs + h * DF_HD : ks + h * DF_HD;
#pragma unroll
        for (int i = 0; i < DF_HD / 32; i++) {
            int c = lane + 32 * i;
            sh[c] = norm[c] * x[i] * r;
            if (!isq) {
                // V follows all Q and K columns and has neither norm nor RoPE.
                float vv = part_sum(parts, splits, n, (DF_NH + 2 * DF_NKV) * DF_HD, row,
                                    (DF_NH + DF_NKV + h) * DF_HD + c);
                st_x(vout + ((size_t)row * DF_NKV + h) * DF_HD + c, vv);
            }
        }
        __syncwarp();
        for (int d = lane; d < DF_HD / 2; d += 32) {
            float a = sh[d], b = sh[d + DF_HD / 2];
            float angle = position * df_rope_freq(d), cs = cosf(angle), sn = sinf(angle);
            sh[d] = a * cs - b * sn;
            sh[d + DF_HD / 2] = b * cs + a * sn;
        }
        __syncwarp();
        if (isq) {
            float scale = 1.4426950408889634f * rsqrtf((float)DF_HD);
#pragma unroll
            for (int i = 0; i < DF_HD / 32; i++) qout[((size_t)row * DF_NH + h) * DF_HD + lane + 32 * i] = sh[lane + 32 * i] * scale;
        } else {
#pragma unroll
            for (int i = 0; i < DF_HD / 32; i++) st_x(kout + ((size_t)row * DF_NKV + h) * DF_HD + lane + 32 * i, sh[lane + 32 * i]);
        }
    }
}

// Bidirectional eight-row attention over the last 2048 target-context positions and the block's own eight rows, split
// by key ranges: DF_SPLITS - 1 ranges of the context ring plus one split for the eight block keys. A thread block owns
// one KV head and one range; its warps are that KV head's query heads, each with all eight queries, so every K/V
// vector is read once for 8 x DF_GROUP dot products. Unnormalized partials go to part_o / part_ml, dflash_combine sums
// them. (One warp per head over all 2048 keys took 2.8 ms per layer on the 3090; the reads are ~8 MB.)
#define DF_SPLITS 65
#define DF_GROUP (DF_NH / DF_NKV)
extern "C" __global__ void dflash_attention(const float* q, const XIN* kc, const XIN* vc, const XIN* kn,
                                             const XIN* vn, const int* block_slot, const int* block_pos,
                                             float* part_o, float* part_ml) {
    int block = blockIdx.x, kv = blockIdx.y, split = blockIdx.z, lane = threadIdx.x & 31;
    int head = kv * DF_GROUP + threadIdx.x / 32;
    float qv[DF_BLOCK][DF_HD / 32];
#pragma unroll
    for (int t = 0; t < DF_BLOCK; t++)
#pragma unroll
        for (int i = 0; i < DF_HD / 32; i++)
            qv[t][i] = q[((size_t)(block * DF_BLOCK + t) * DF_NH + head) * DF_HD + lane + 32 * i];
    float m[DF_BLOCK], l[DF_BLOCK], acc[DF_BLOCK][DF_HD / 32];
#pragma unroll
    for (int t = 0; t < DF_BLOCK; t++) {
        m[t] = -INFINITY;
        l[t] = 0.f;
#pragma unroll
        for (int i = 0; i < DF_HD / 32; i++) acc[t][i] = 0.f;
    }
    int start = block_pos[block], slot = block_slot[block];
    bool noise = split == DF_SPLITS - 1;
    int lo, hi;
    if (noise) {
        lo = 0;
        hi = DF_BLOCK;
    } else {
        constexpr int per = DF_WINDOW / (DF_SPLITS - 1);
        lo = max(0, start - DF_WINDOW + 1) + split * per;
        hi = min(start, lo + per);
    }
    for (int key = lo; key < hi; key++) {
        int absolute = noise ? start + key : key;
        size_t at = noise ? (((size_t)(block * DF_BLOCK + key) * DF_NKV + kv) * DF_HD)
                          : ((((size_t)slot * DF_WINDOW + (key & (DF_WINDOW - 1))) * DF_NKV + kv) * DF_HD);
        const XIN* kp = noise ? kn + at : kc + at;
        const XIN* vp = noise ? vn + at : vc + at;
        float kval[DF_HD / 32], vval[DF_HD / 32];
#pragma unroll
        for (int i = 0; i < DF_HD / 32; i++) {
            kval[i] = __half2float(kp[lane + 32 * i]);
            vval[i] = __half2float(vp[lane + 32 * i]);
        }
#pragma unroll
        for (int t = 0; t < DF_BLOCK; t++) {
            if (!noise && start + t - absolute >= DF_WINDOW) continue;
            float score = 0.f;
#pragma unroll
            for (int i = 0; i < DF_HD / 32; i++) score += qv[t][i] * kval[i];
            for (int d = 16; d > 0; d >>= 1) score += __shfl_xor_sync(0xffffffff, score, d);
            float mn = fmaxf(m[t], score), alpha = exp2f(m[t] - mn), p = exp2f(score - mn);
            l[t] = l[t] * alpha + p;
            m[t] = mn;
#pragma unroll
            for (int i = 0; i < DF_HD / 32; i++) acc[t][i] = acc[t][i] * alpha + p * vval[i];
        }
    }
#pragma unroll
    for (int t = 0; t < DF_BLOCK; t++) {
        size_t o = ((size_t)(block * DF_SPLITS + split) * DF_NH + head) * DF_BLOCK + t;
#pragma unroll
        for (int i = 0; i < DF_HD / 32; i++) part_o[o * DF_HD + lane + 32 * i] = acc[t][i];
        if (lane == 0) { part_ml[2 * o] = m[t]; part_ml[2 * o + 1] = l[t]; }
    }
}

// The attention's split partials into the f16 input of o_proj: grid (block rows, query heads), DF_HD threads.
extern "C" __global__ void dflash_combine(const float* part_o, const float* part_ml, XIN* out) {
    int row = blockIdx.x, head = blockIdx.y, c = threadIdx.x, block = row / DF_BLOCK, t = row % DF_BLOCK;
    float mx = -INFINITY;
    for (int s = 0; s < DF_SPLITS; s++)
        mx = fmaxf(mx, part_ml[2 * (((size_t)(block * DF_SPLITS + s) * DF_NH + head) * DF_BLOCK + t)]);
    float sum = 0.f, v = 0.f;
    for (int s = 0; s < DF_SPLITS; s++) {
        size_t o = ((size_t)(block * DF_SPLITS + s) * DF_NH + head) * DF_BLOCK + t;
        float w = exp2f(part_ml[2 * o] - mx);
        sum += part_ml[2 * o + 1] * w;
        v += part_o[o * DF_HD + c] * w;
    }
    st_x(out + ((size_t)row * DF_NH + head) * DF_HD + c, v / sum);
}

// Split-K partial products [split][n][ld] summed into y [n][ld] over the whole grid (sum_parts runs one block per
// row: 7 blocks for the 248k-column output layer, 0.72 ms on the 3090).
extern "C" __global__ void dflash_sum(const float* x, int splits, int n, int ld, float* y) {
    size_t len = (size_t)n * ld;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < len; i += (size_t)gridDim.x * blockDim.x) {
        float v = 0.f;
        for (int s = 0; s < splits; s++) v += x[s * len + i];
        y[i] = v;
    }
}

// Rows 1..7 of every draft block feed the shared target LM head and the selector's hidden projection.
extern "C" __global__ void dflash_head_rows(const XIN* x, XIN* out, int rows) {
    int block = blockIdx.x / 7, r = blockIdx.x % 7 + 1;
    const XIN* in = x + (size_t)(block * DF_BLOCK + r) * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x) out[(size_t)blockIdx.x * H + c] = in[c];
    (void)rows;
}

// Parallel top 16, adapted from ExLlamaV3's dflash2.cu: 61 blocks per vocabulary row, warp-level
// candidate lists, and the last block merging the partial lists. One read of the logits, all SMs occupied.
template <int N>
__device__ __forceinline__ void df_warp_top16(float (&x)[N], int (&id)[N], float* out_v, int* out_ix) {
    int lane = threadIdx.x & 31;
    float cur = -INFINITY;
    int arg = 0;
#pragma unroll
    for (int i = 0; i < N; i++) if (x[i] > cur) { cur = x[i]; arg = i; }
#pragma unroll
    for (int k = 0; k < DF_TOPK; k++) {
        float best = cur;
        int best_lane = lane;
        for (int d = 16; d > 0; d >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, best, d);
            int ol = __shfl_xor_sync(0xffffffff, best_lane, d);
            if (ov > best || (ov == best && ol < best_lane)) { best = ov; best_lane = ol; }
        }
        int best_id = -1;
#pragma unroll
        for (int i = 0; i < N; i++) if (i == arg) best_id = id[i];
        best_id = __shfl_sync(0xffffffff, best_id, best_lane);
        if (lane == 0) { out_v[k] = best; out_ix[k] = best_id; }
        if (lane == best_lane) {
#pragma unroll
            for (int i = 0; i < N; i++) if (i == arg) x[i] = -INFINITY;
            cur = -INFINITY; arg = 0;
#pragma unroll
            for (int i = 0; i < N; i++) if (x[i] > cur) { cur = x[i]; arg = i; }
        }
    }
}

template <int N>
__device__ __forceinline__ void df_block_top16(float (&x)[N], int (&id)[N], float* out_v, int* out_ix,
                                                float* s_wv, int* s_wix) {
    constexpr int M = DF_TOPK / 4;
    int t = threadIdx.x, warp = t / 32, lane = t & 31;
    df_warp_top16<N>(x, id, s_wv + warp * DF_TOPK, s_wix + warp * DF_TOPK);
    __syncthreads();
    if (warp == 0) {
        float y[M];
        int yid[M];
#pragma unroll
        for (int i = 0; i < M; i++) { y[i] = s_wv[lane * M + i]; yid[i] = s_wix[lane * M + i]; }
        df_warp_top16<M>(y, yid, out_v, out_ix);
    }
    __syncthreads();
}

// ncols: the row length of `logits`, the vocabulary
extern "C" __global__ void __launch_bounds__(256) dflash_top16(const float* logits, int splits, int ncols,
                                                                 float* part_v, int* part_ix, int* counters, float* values, int* ids) {
    constexpr int E = 16;
    __shared__ float s_wv[8 * DF_TOPK];
    __shared__ int s_wix[8 * DF_TOPK];
    __shared__ bool last;
    int row = blockIdx.x / splits, split = blockIdx.x % splits, t = threadIdx.x;
    const float* r = logits + (size_t)row * ncols;
    int c0 = split * 256 * E;
    float x[E];
    int id[E];
#pragma unroll
    for (int l = 0; l < 4; l++) {
        int c = c0 + l * 1024 + t * 4;
        if (c + 4 <= ncols) {
            float4 v = *(const float4*)(r + c);
            x[l * 4] = v.x; x[l * 4 + 1] = v.y; x[l * 4 + 2] = v.z; x[l * 4 + 3] = v.w;
        } else {
#pragma unroll
            for (int j = 0; j < 4; j++) x[l * 4 + j] = c + j < ncols ? r[c + j] : -INFINITY;
        }
#pragma unroll
        for (int j = 0; j < 4; j++) id[l * 4 + j] = c + j;
    }
    float* pv = part_v + (size_t)blockIdx.x * DF_TOPK;
    int* pi = part_ix + (size_t)blockIdx.x * DF_TOPK;
    df_block_top16<E>(x, id, pv, pi, s_wv, s_wix);
    __threadfence();
    if (t == 0) last = atomicAdd(counters + row, 1) == splits - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    constexpr int M = DF_TOPK / 4;
    float y[M];
    int yid[M];
    const float* rv = part_v + (size_t)row * splits * DF_TOPK;
    const int* ri = part_ix + (size_t)row * splits * DF_TOPK;
#pragma unroll
    for (int i = 0; i < M; i++) {
        int j = t * M + i;
        y[i] = j < splits * DF_TOPK ? rv[j] : -INFINITY;
        yid[i] = j < splits * DF_TOPK ? ri[j] : -1;
    }
    __shared__ float out_v[DF_TOPK];
    __shared__ int out_ix[DF_TOPK];
    df_block_top16<M>(y, yid, out_v, out_ix, s_wv, s_wix);
    if (t < DF_TOPK) {
        values[row * DF_TOPK + t] = out_v[t];
        ids[row * DF_TOPK + t] = out_ix[t];
    }
    if (t == 0) counters[row] = 0;
}

__device__ __forceinline__ float df_bf16(unsigned short x) {
    unsigned int v = (unsigned int)x << 16;
    return __uint_as_float(v);
}

// The reference's walk (dflash CandidateSelector.select at temperature 0, ExLlamaV3's dflash2_selector_walk): per row
// the candidate of the highest unary + <A(previous token) * gate, B(candidate)>, then the next row from it.
extern "C" __global__ void __launch_bounds__(256) dflash_selector(const float* unary, const int* cands,
                                                                          const float* gate, const unsigned short* predecessor,
                                                                          const unsigned short* successor, const int* anchors, int* out) {
    __shared__ float ag[DF_RANK], scores[DF_TOPK];
    __shared__ int pred;
    int t = threadIdx.x, warp = t / 32, lane = t & 31;
    if (t == 0) pred = anchors[blockIdx.x];
    __syncthreads();
    for (int row = 0; row < DF_BLOCK - 1; row++) {
        int at = blockIdx.x * (DF_BLOCK - 1) + row;
        for (int r = t; r < DF_RANK; r += 256) ag[r] = df_bf16(predecessor[(size_t)pred * DF_RANK + r]) * gate[at * DF_RANK + r];
        __syncthreads();
        for (int c = warp; c < DF_TOPK; c += 8) {
            int token = cands[at * DF_TOPK + c];
            float dot = 0.f;
            for (int r = lane; r < DF_RANK; r += 32) dot += ag[r] * df_bf16(successor[(size_t)token * DF_RANK + r]);
            for (int d = 16; d > 0; d >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, d);
            if (lane == 0) scores[c] = unary[at * DF_TOPK + c] + dot;
        }
        __syncthreads();
        if (t == 0) {
            int best = 0;
            for (int c = 1; c < DF_TOPK; c++) if (scores[c] > scores[best]) best = c;
            pred = cands[at * DF_TOPK + best];
            out[at] = pred;
        }
        __syncthreads();
    }
}
