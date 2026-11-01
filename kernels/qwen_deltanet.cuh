#pragma once
// Gated DeltaNet (linear attention), three kernels per layer and step:
//  delta_conv: causal depthwise conv (kernel 4) + SiLU over the q, k, v channels, l2 norm of q and k per head
//              (q also scaled by 1/sqrt(LKD)) -> cv [n][CVW]; also exp(g) and beta per value head after the channels.
//              One warp per (token, 128 channels).
//  delta_rule: the gated delta rule per (segment, value head, slice of CPB value columns): the recurrent state slice
//              lives in registers while the block walks the segment's tokens; raw output o -> out [n][MIX]. Also
//              stores the segment's last conv inputs in the conv ring (after delta_conv has read it).
//  delta_norm: y = norm_w * o / rms(o) * silu(z) per value head (y: the out-projection's input, XIN); updates the
//              per-slot metadata.
//
// Rollback (speculative drafts): a step may restart a slot up to KEEP positions before where its last step ended,
// inside that step. Per slot there are two state buffers. After a step, buffer `ebuf` holds the state at its end and
// the other one the state at the checkpoint ck = max(first position, end - KEEP), with the per-token inputs of the
// delta rule (k, v, exp(g), beta) for positions ck .. end - 1 in that buffer's replay bank. A step that continues at
// `end` reads ebuf; one that restarts earlier reads the checkpoint and replays the accepted tokens from its bank.
// For a short step (<= KEEP tokens) continuing at `end`, the input state is itself the new checkpoint: the step
// writes its end state to the other buffer and nothing else, so decode moves no more state than without rollback.
// The conv ring needs nothing: it is indexed by position and a restarted step overwrites the rejected entries.

#define CR 32       // conv ring: the inputs at the last CR positions per slot and channel (>= KEEP + CONV_K - 1)
#define KEEP 16     // rollback reach: a step may restart at most KEEP positions before the previous step's end
#define TCH 8       // tokens per round of delta_rule (two rounds staged in shared memory)
#define CPB 16      // value columns per delta_rule block
#define NSLICE (LVD / CPB)
#define REP (LK_HEADS * LKD + LV_HEADS * LVD + 2 * LV_HEADS)  // replay entry: k, v, exp(g), beta
#define META 4      // per slot: end, ck, ebuf, unused
// A slot's memory in a layer (`sst` floats per slot, page-aligned so that each slot's part maps on its own: see
// deltanet.rs): the two state buffers [2][LV_HEADS][LVD (v)][LKD (k)], the two replay banks [2][KEEP][REP], the conv
// ring [CR][LQKV]
#define SLOT_STATE (LV_HEADS * LVD * LKD)
#define SLOT_BANK (KEEP * REP)
#define SLOT_RING (2 * SLOT_STATE + 2 * SLOT_BANK)
#define CVW (LQKV + 2 * LV_HEADS)  // delta_conv output row: q, k, v, exp(g), beta

static_assert(LKD == 128 && LVD == 128, "the thread layouts assume 128-dim heads");
static_assert(CR >= KEEP + CONV_K - 1 && (CR & (CR - 1)) == 0, "conv ring");

__device__ __forceinline__ float4 ld4(const float* p) { return *(const float4*)p; }

__device__ __forceinline__ float4 part_sum4(const float* p, int splits, int n, int t, int c) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int s = 0; s < splits; s++) {
        float4 x = ld4(p + ((size_t)s * n + t) * DELTA_IN + c);
        v.x += x.x; v.y += x.y; v.z += x.z; v.w += x.w;
    }
    return v;
}

// What a segment's step does with the slot's state, from the slot's metadata (the same in delta_rule and delta_norm)
struct Plan {
    int in;     // state buffer to start from (unless zero)
    int r;      // tokens to replay from the checkpoint first (from position `from`, out of bank `in`)
    int from;
    int ck;     // the new checkpoint position
    int cbuf;   // buffer (and replay bank) of the new checkpoint
    int ebuf;   // buffer of the new end state
    bool zero;  // start from zeros (a new sequence)
    bool wck;   // write the checkpoint state (else the input state stays as the checkpoint)
};

__device__ Plan plan(const int* meta, int slot, int p0, int len) {
    int end = meta[slot * META], ck = meta[slot * META + 1], eb = meta[slot * META + 2];
    Plan p;
    p.ck = max(p0, p0 + len - KEEP);
    p.r = 0;
    p.from = 0;
    p.zero = p0 == 0 || end < 0;
    if (p.zero) {
        p.in = 0;
    } else if (p0 == end) {
        p.in = eb;
    } else {  // restart inside the last step: ck <= p0 < end by contract (no state exists for anything earlier)
        p.in = 1 - eb;
        p.from = ck;
        p.r = min(max(p0 - ck, 0), KEEP);
    }
    p.wck = p.zero || p.r > 0 || p.ck > p0;
    p.cbuf = p.wck ? 1 - p.in : p.in;
    p.ebuf = 1 - p.cbuf;
    return p;
}

// grid (LQKV / 128, ceil(n / 4)), block 128: warp w of block (g, b) handles token 4b + w, channels 128g .. 128g + 127
// (a q, k or v head), 4 per lane. Inputs at earlier positions come from the step (same slot) or the conv ring.
extern "C" __global__ void __launch_bounds__(128) delta_conv(const float* p, int splits, int n, const float* ba,
                                                             int ba_splits, const float* cw,
                                                             const float* a_log, const float* dt_bias,
                                                             const float* mem, long long sst, float* cv,
                                                             const int* tok_slot, const int* tok_pos) {
    int grp = blockIdx.x, lane = threadIdx.x & 31;
    int t = blockIdx.y * 4 + (threadIdx.x >> 5);
    if (t >= n) return;
    int c = grp * 128 + lane * 4, slot = tok_slot[t], pos = tok_pos[t];
    float w[4][CONV_K];  // [channel][tap]
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float4 v = ld4(cw + (c + i) * CONV_K);
        w[i][0] = v.x; w[i][1] = v.y; w[i][2] = v.z; w[i][3] = v.w;
    }
    float y[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < CONV_K; j++) {
        int q = pos - (CONV_K - 1) + j, tt = t - (CONV_K - 1) + j;
        float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q >= 0) {
            if (tt >= 0 && tok_slot[tt] == slot) x = part_sum4(p, splits, n, tt, c);  // same segment
            else x = ld4(mem + slot * sst + SLOT_RING + (size_t)(q & (CR - 1)) * LQKV + c);
        }
        y[0] += w[0][j] * x.x; y[1] += w[1][j] * x.y; y[2] += w[2][j] * x.z; y[3] += w[3][j] * x.w;
    }
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) { y[i] = silu(y[i]); ss += y[i] * y[i]; }
    if (grp < 2 * LK_HEADS) {  // q or k: l2 norm (eps as in FLA's l2norm), q also / sqrt(LKD)
        float sc = rsqrtf(warp_sum(ss) + 1e-6f);
        if (grp < LK_HEADS) sc *= rsqrtf((float)LKD);
#pragma unroll
        for (int i = 0; i < 4; i++) y[i] *= sc;
    }
    *(float4*)(cv + (size_t)t * CVW + c) = make_float4(y[0], y[1], y[2], y[3]);
    if (grp == 0) {  // the gates of value head v: exp(g) = exp(-exp(A_log) softplus(a + dt_bias)), sigmoid(b)
        for (int v = lane; v < LV_HEADS; v += 32) {
            float b = part_sum(ba, ba_splits, n, 2 * LV_HEADS, t, v);
            float a = part_sum(ba, ba_splits, n, 2 * LV_HEADS, t, LV_HEADS + v) + dt_bias[v];
            float sp = a > 20.f ? a : log1pf(expf(a));  // softplus (torch's threshold 20)
            cv[(size_t)t * CVW + LQKV + v] = expf(-expf(a_log[v]) * sp);
            cv[(size_t)t * CVW + LQKV + LV_HEADS + v] = 1.f / (1.f + expf(-b));
        }
    }
}

// grid (NSLICE, LV_HEADS, segs), block NT. Thread: value column col = slice * CPB + lane / LPC + warp * 32 / LPC and
// the RPT key rows 4 * (lane % LPC + LPC j) + i (i < 4) of the state S[k][v] of value head h (key head h / 2), in
// registers. State buffers: per slot [2][LV_HEADS][LVD (v)][LKD (k)] f32 (SLOT_STATE). Per token: S *= e (= exp(g));
// delta = (v - S^T k) beta; S += k delta^T; o = S^T q, computed from the old S in one pass: o = e S^T q + delta (k . q).
// The block's columns are independent of all other blocks: a segment runs as NSLICE * LV_HEADS blocks in parallel,
// each walking the segment's tokens in order, TCH per round; cp.async stages the next round while one runs.
#define LPC 4                   // lanes per value column
#define RPT (LKD / LPC)         // key rows per thread
#define NT (CPB * LPC)          // threads per block
#define KT (NT / TCH)            // threads per token computing k . q
static_assert(NT % 32 == 0 && NT % TCH == 0 && KT <= 32 && (KT & (KT - 1)) == 0 && TCH * CPB / 4 + 2 * TCH <= NT &&
              LKD / 4 <= NT && TCH * LKD / 4 % NT == 0, "delta_rule layout");

__device__ __forceinline__ void ld_rows(float* x, const float* src, int rg) {
#pragma unroll
    for (int j = 0; j < RPT / 4; j++) {
        float4 v = ld4(src + 4 * (rg + LPC * j));
        x[4 * j] = v.x; x[4 * j + 1] = v.y; x[4 * j + 2] = v.z; x[4 * j + 3] = v.w;
    }
}

__device__ __forceinline__ void st_rows(float* dst, const float* x, int rg) {
#pragma unroll
    for (int j = 0; j < RPT / 4; j++)
        *(float4*)(dst + 4 * (rg + LPC * j)) = make_float4(x[4 * j], x[4 * j + 1], x[4 * j + 2], x[4 * j + 3]);
}

// sum over the LPC lanes of a column
__device__ __forceinline__ float col_sum(float v) {
#pragma unroll
    for (int o = 1; o < LPC; o <<= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}


struct Round {
    float q[TCH][LKD], k[TCH][LKD], v[TCH][CPB], e[TCH], b[TCH], kq[TCH];
};

extern "C" __global__ void __launch_bounds__(NT) delta_rule(
    const float* p, int splits, int n, const float* cv, float* mem, long long sst, const int* meta, float* out,
    const int* seg_start, const int* seg_len, const int* seg_slot, const int* seg_pos0) {
    __shared__ __align__(16) Round sm[2];
    int slice = blockIdx.x, h = blockIdx.y, sg = blockIdx.z, kh = h / (LV_HEADS / LK_HEADS);
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, rg = lane % LPC;
    int lc = warp * (32 / LPC) + lane / LPC, col = slice * CPB + lc;
    int t0 = seg_start[sg], len = seg_len[sg], slot = seg_slot[sg], p0 = seg_pos0[sg];
    Plan pl = plan(meta, slot, p0, len);
    float* sm0 = mem + slot * sst;
    auto sbuf = [&](int b) { return sm0 + (size_t)b * SLOT_STATE + ((size_t)h * LVD + col) * LKD; };
    auto entry = [&](int b, int pos) { return sm0 + 2 * SLOT_STATE + (size_t)b * SLOT_BANK + (size_t)(pos % KEEP) * REP; };
    // stage round c0 / TCH into sm[r]: q and k of the key head, the block's v columns, the gates of head h
    auto stage = [&](int c0, int r) {
        int m = min(TCH, len - c0);
        Round& R = sm[r];
#pragma unroll
        for (int i = 0; i < TCH * LKD / 4 / NT; i++) {
            int x = tid + NT * i, tk = x / (LKD / 4), d = (x % (LKD / 4)) * 4;
            if (tk < m) {
                const float* row = cv + (size_t)(t0 + c0 + tk) * CVW + kh * LKD + d;
                cp_async16(&R.q[tk][d], row);
                cp_async16(&R.k[tk][d], row + LK_HEADS * LKD);
            }
        }
        if (tid < TCH * CPB / 4) {
            int tk = tid / (CPB / 4), d = (tid % (CPB / 4)) * 4;
            if (tk < m) cp_async16(&R.v[tk][d], cv + (size_t)(t0 + c0 + tk) * CVW + 2 * LK_HEADS * LKD + h * LVD + slice * CPB + d);
        } else if (tid >= NT - 2 * TCH) {
            int x = tid - (NT - 2 * TCH), tk = x % TCH, which = x / TCH;
            if (tk < m) cp_async4(which ? &R.b[tk] : &R.e[tk], cv + (size_t)(t0 + c0 + tk) * CVW + LQKV + which * LV_HEADS + h);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    stage(0, 0);

    float S[RPT];
    if (pl.zero) {
#pragma unroll
        for (int i = 0; i < RPT; i++) S[i] = 0.f;
    } else {
        ld_rows(S, sbuf(pl.in), rg);
    }
    // replay the accepted tokens between the checkpoint and the restart position
    for (int i = 0; i < pl.r; i++) {
        const float* e = entry(pl.in, pl.from + i);
        // the same order of operations as the step's own tokens below (four partial sums), so that a replayed state is
        // bit for bit the state one token per step reaches
        float k[RPT], dk4[4] = {0.f, 0.f, 0.f, 0.f};
        ld_rows(k, e + kh * LKD, rg);
#pragma unroll
        for (int j = 0; j < RPT; j++) dk4[j & 3] += S[j] * k[j];
        float dk = col_sum((dk4[0] + dk4[1]) + (dk4[2] + dk4[3]));
        float ex = e[LK_HEADS * LKD + LV_HEADS * LVD + h], be = e[LK_HEADS * LKD + LV_HEADS * LVD + LV_HEADS + h];
        float delta = (e[LK_HEADS * LKD + h * LVD + col] - ex * dk) * be;
#pragma unroll
        for (int j = 0; j < RPT; j++) S[j] = ex * S[j] + k[j] * delta;
    }

    for (int c0 = 0, r = 0; c0 < len; c0 += TCH, r ^= 1) {
        int m = min(TCH, len - c0);
        if (c0 + TCH < len) stage(c0 + TCH, r ^ 1);
        else asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        __syncthreads();
        Round& R = sm[r];
        {  // k . q per token: KT threads per token
            int tk = tid / KT, d = tid % KT;
            float x = 0.f;
            if (tk < m) {
#pragma unroll
                for (int i = 0; i < LKD / KT; i++) x += R.q[tk][d + KT * i] * R.k[tk][d + KT * i];
            }
#pragma unroll
            for (int o = 1; o < KT; o <<= 1) x += __shfl_xor_sync(0xffffffff, x, o);
            if (d == 0 && tk < m) R.kq[tk] = x;
        }
        // the delta rule inputs of the tokens from the new checkpoint on, for a later restart
        for (int tk = max(0, pl.ck - (p0 + c0)); tk < m; tk++) {
            float* e = entry(pl.cbuf, p0 + c0 + tk);
            if (slice == 0 && h % 2 == 0 && tid < LKD / 4) *(float4*)(e + kh * LKD + tid * 4) = ld4(&R.k[tk][tid * 4]);
            if (tid < CPB) e[LK_HEADS * LKD + h * LVD + slice * CPB + tid] = R.v[tk][tid];
            if (slice == 0 && tid == 0) {
                e[LK_HEADS * LKD + LV_HEADS * LVD + h] = R.e[tk];
                e[LK_HEADS * LKD + LV_HEADS * LVD + LV_HEADS + h] = R.b[tk];
            }
        }
        __syncthreads();
        for (int tk = 0; tk < m; tk++) {
            if (pl.wck && p0 + c0 + tk == pl.ck) st_rows(sbuf(pl.cbuf), S, rg);
            float k[RPT], q[RPT];
            ld_rows(k, R.k[tk], rg);
            ld_rows(q, R.q[tk], rg);
            float dk4[4] = {0.f, 0.f, 0.f, 0.f}, dq4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int j = 0; j < RPT; j++) {
                dk4[j & 3] += S[j] * k[j];
                dq4[j & 3] += S[j] * q[j];
            }
            float dk = col_sum((dk4[0] + dk4[1]) + (dk4[2] + dk4[3])), dq = col_sum((dq4[0] + dq4[1]) + (dq4[2] + dq4[3]));
            float ex = R.e[tk], delta = (R.v[tk][lc] - ex * dk) * R.b[tk];
            if (rg == 0) out[(size_t)(t0 + c0 + tk) * MIX + h * LVD + col] = ex * dq + delta * R.kq[tk];
#pragma unroll
            for (int j = 0; j < RPT; j++) S[j] = ex * S[j] + k[j] * delta;
        }
        __syncthreads();  // done with sm[r] before it is staged again
    }
    st_rows(sbuf(pl.ebuf), S, rg);
    // conv ring: the inputs of the segment's last CR tokens, this block's share of the channels (delta_conv, which
    // reads the ring, has finished); the last blocks' shares are shorter when the blocks do not divide them
    const int CH = (LQKV + NSLICE * LV_HEADS - 1) / (NSLICE * LV_HEADS);
    int c0 = (h * NSLICE + slice) * CH, ch = max(0, min(CH, LQKV - c0)), m = min(len, CR);
    for (int i = tid; i < m * ch; i += NT) {
        int tk = len - m + i / ch, c = c0 + i % ch;
        sm0[SLOT_RING + (size_t)((p0 + tk) & (CR - 1)) * LQKV + c] = part_sum(p, splits, n, DELTA_IN, t0 + tk, c);
    }
}

// grid (n, LV_HEADS / NORM_HEADS), block NORM_HEADS * 32: warp w normalizes value head blockIdx.y * NORM_HEADS + w of
// token blockIdx.x (gated RMSNorm with the plain weight) from delta_rule's out into y; block (0, 0) then records each
// segment's step in the slot metadata (after delta_rule has read it).
#define NORM_HEADS 16
static_assert(LV_HEADS % NORM_HEADS == 0, "delta_norm blocks");
extern "C" __global__ void __launch_bounds__(NORM_HEADS * 32) delta_norm(
    const float* p, int splits, int n, const float* nw, const float* out, XIN* y, int* meta, const int* seg_len,
    const int* seg_slot, const int* seg_pos0, int segs) {
    int t = blockIdx.x, h = blockIdx.y * NORM_HEADS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    size_t o = (size_t)t * MIX + h * LVD + lane * 4;
    float4 v = ld4(out + o), z = part_sum4(p, splits, n, t, LQKV + h * LVD + lane * 4), w = ld4(nw + lane * 4);
    float r = rsqrtf(warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w) / LVD + EPS);
    st_x(y + o, w.x * (v.x * r) * silu(z.x));
    st_x(y + o + 1, w.y * (v.y * r) * silu(z.y));
    st_x(y + o + 2, w.z * (v.z * r) * silu(z.z));
    st_x(y + o + 3, w.w * (v.w * r) * silu(z.w));
    if (t == 0 && blockIdx.y == 0) {
        for (int s = threadIdx.x; s < segs; s += blockDim.x) {
            int slot = seg_slot[s], p0 = seg_pos0[s], len = seg_len[s];
            Plan pl = plan(meta, slot, p0, len);
            meta[slot * META] = p0 + len;
            meta[slot * META + 1] = pl.ck;
            meta[slot * META + 2] = pl.ebuf;
        }
    }
}
