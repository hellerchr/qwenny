// Sampling on the GPU, for a vocabulary of VOCAB tokens defined by the including file.
#pragma once
#include "common.cuh"

// ---- sampling ----

#define SCAP 2048

#define SEEN_WORDS (VOCAB / 32)

// Tokens seen per slot (prompt and generated so far, for the repetition penalty). One block per segment: a segment
// at position 0 starts a new sequence and clears its slot; then the tokens mark[mark_at[sg] .. mark_at[sg + 1]]
// (those committed since the slot's last step) are set. Draft tokens of a step are not: the sampler checks them.
extern "C" __global__ void mark_seen(unsigned* seen, const int* mark, const int* mark_at, const int* seg_slot,
                                     const int* seg_pos0) {
    int sg = blockIdx.x;
    unsigned* sb = seen + (size_t)seg_slot[sg] * SEEN_WORDS;
    if (seg_pos0[sg] == 0) {
        for (int i = threadIdx.x; i < SEEN_WORDS; i += blockDim.x) sb[i] = 0;
        __syncthreads();
    }
    for (int i = mark_at[sg] + threadIdx.x; i < mark_at[sg + 1]; i += blockDim.x) {
        int id = mark[i];
        atomicOr(sb + (id >> 5), 1u << (id & 31));
    }
}

// Bitonic sort of cand[0 .. np), np a power of two, 1024 threads: ascending or descending
template <bool ASC>
__device__ __forceinline__ void bitonic(unsigned long long* cand, int np, int tid) {
    for (int size = 2; size <= np; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < np; i += 1024) {
                int o = i ^ stride;
                if (o > i) {
                    bool up = (i & size) == 0;
                    unsigned long long a = cand[i], c = cand[o];
                    if ((ASC ? a > c : a < c) == up) { cand[i] = c; cand[o] = a; }
                }
            }
            __syncthreads();
        }
}

// Inclusive prefix sum over the block (1024 threads, one value each)
__device__ __forceinline__ float block_scan(float v, float* sh) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        float x = __shfl_up_sync(0xffffffff, v, o);
        if (lane >= o) v += x;
    }
    if (lane == 31) sh[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float w = sh[lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            float x = __shfl_up_sync(0xffffffff, w, o);
            if (lane >= o) w += x;
        }
        sh[lane] = w;
    }
    __syncthreads();
    float r = v + (warp > 0 ? sh[warp - 1] : 0.f);
    __syncthreads();
    return r;
}

// The draft distributions q of sampled drafts (draft_sample), per slot and draft index t of its chain (QD at most):
// the kept tokens in ascending id order with their probabilities (QCAP at most: top_k <= 1024), their count, and the
// token drawn from them. The target's sampler reads them for the rows whose draft is marked sampled.
#define QD 16
#define QCAP 1024

// The k largest of val(0 .. W - 1) as (ordered value << 32 | ~index), sorted descending in cand; returns how many
// cand holds (at least k unless W < k). Radix select: narrow (prefix, mask) on the ordered values until the
// candidates at or above the bin holding the k-th largest fit in shared memory.
template <class Val>
__device__ __forceinline__ int top_cands(Val val, int W, int k, unsigned* hist, unsigned long long* cand, float* sh,
                                         unsigned* sv) {
    int tid = threadIdx.x;
    unsigned& s_bin = sv[0];
    unsigned& s_above = sv[1];
    unsigned& s_cnt = sv[2];
    unsigned prefix = 0, mask = 0, above = 0, lo = 0;
    for (int level = 0; level < 3; level++) {
        int shift = level == 0 ? 20 : level == 1 ? 8 : 0, nb = level < 2 ? 4096 : 256;
        for (int b = tid; b < nb; b += 1024) hist[b] = 0;
        __syncthreads();
        for (int i = tid; i < W; i += 1024) {
            unsigned u = ordered(val(i));
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & (nb - 1)], 1u);
        }
        __syncthreads();
        // the bin where the count from the top reaches k: thread t sums bins [nb - per (t + 1), nb - per t)
        int per = (nb + 1023) / 1024, top = nb - per * tid;
        float mine = 0.f;
        for (int q = 0; q < per; q++)
            if (top - 1 - q >= 0) mine += hist[top - 1 - q];
        float cum = block_scan(mine, sh);
        if (tid == 0) { s_bin = 0; s_above = above; }
        __syncthreads();
        if (top > 0 && above + cum >= k && above + cum - mine < k) {
            unsigned c = above + (unsigned)(cum - mine);
            for (int q = 0; q < per; q++) {
                int b = top - 1 - q;
                if (c + hist[b] >= (unsigned)k) { s_bin = b; s_above = c; break; }
                c += hist[b];
            }
        }
        __syncthreads();
        unsigned b = s_bin;
        above = s_above;
        prefix |= b << shift;
        mask |= (unsigned)(nb - 1) << shift;
        lo = prefix;
        if (above + hist[b] <= SCAP) break;
    }
    // candidates: every value >= the lower edge of that bin
    if (tid == 0) s_cnt = 0;
    __syncthreads();
    for (int i = tid; i < W; i += 1024) {
        unsigned u = ordered(val(i));
        if (u >= lo) {
            unsigned at = atomicAdd(&s_cnt, 1u);
            if (at < SCAP) cand[at] = ((unsigned long long)u << 32) | (0xffffffffu - (unsigned)i);
        }
    }
    __syncthreads();
    int cnt = min((int)s_cnt, SCAP), np = 1;
    while (np < cnt) np <<= 1;
    for (int i = cnt + tid; i < np; i += 1024) cand[i] = 0;
    __syncthreads();
    bitonic<false>(cand, np, tid);
    return cnt;
}

__device__ __forceinline__ int cand_index(unsigned long long c) { return (int)(0xffffffffu - (unsigned)c); }

// Softmax of the top k (sorted in cand) at the temperature and the top-p cut: thread tid < k gets p (unnormalized)
// and cum (inclusive prefix of p); returns the kept count (the smallest prefix with mass >= top_p, at least one) and
// its mass in *mass.
template <class Val>
__device__ __forceinline__ int kept(Val val, const unsigned long long* cand, int k, float temp, float top_p, float* sh,
                                    unsigned* sv, float& p, float& cum, float* mass) {
    int tid = threadIdx.x;
    float vmax = val(cand_index(cand[0]));
    p = tid < k ? __expf((val(cand_index(cand[tid])) - vmax) / temp) : 0.f;
    cum = block_scan(p, sh);
    if (tid == k - 1) sh[0] = cum;
    __syncthreads();
    float total = sh[0];
    __syncthreads();
    if (tid == 0) sv[2] = k;
    __syncthreads();
    if (top_p < 1.f && tid < k && cum / total >= top_p) atomicMin(&sv[2], tid + 1);
    __syncthreads();
    int keep = sv[2];
    if (tid == keep - 1) sh[1] = cum;
    __syncthreads();
    *mass = sh[1];
    __syncthreads();
    return keep;
}

// q of token `id` in a stored draft distribution (0 when it is not among its tokens)
__device__ __forceinline__ float q_of(const int* qid, const float* qp, int n, int id) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1, v = qid[mid];
        if (v == id) return qp[mid];
        if (v < id) lo = mid + 1; else hi = mid - 1;
    }
    return 0.f;
}

// One block of 1024 threads per logit row, in vLLM's order: repetition penalty (tokens seen in the row's slot, and
// the step's tokens of the row's sequence up to the row's own), presence penalty (subtracted for the tokens the
// request generated: the slot's outputs set, and the step's drafts before the row), temperature, top-k, top-p, then a draw with the host's uniform numbers. params[row]: temperature, top_k, top_p,
// repetition penalty, u1, u2, draft (-1: none; d <= -2: the draft -d - 2, drawn by draft_sample), presence penalty
// (0 for rows that end a prompt: no output yet). Greedy rows (temperature 0 or top_k 1) take the argmax of the
// penalized logits.
// With a draft d (speculative decoding) from the draft layer's argmax: accept it with probability p(d) (greedy: if it
// is the argmax), else draw from p without d, using u2. With a sampled draft d ~ q (q stored for the row's slot and
// its index in the segment's drafts): accept it with probability min(1, p(d) / q(d)), else draw from
// max(0, p - q) normalized. Either way the result follows p exactly. out_tok[row] = token, out_acc[row] = d accepted.
extern "C" __global__ void __launch_bounds__(1024)
    sample_rows(const float* logits, const int* out_rows, const int* tok_slot, const int* ids, const int* row_first,
                const unsigned* seen, const unsigned* outs, const float* params, const int* qtok, const int* qn,
                const int* qid, const float* qp, int* out_tok, int* out_acc) {
    __shared__ unsigned hist[4096];
    __shared__ unsigned long long cand[SCAP];  // (ordered value << 32 | ~index): sorts descending by value
    __shared__ float sh[32];
    __shared__ unsigned sv[3];
    __shared__ unsigned s_bin;
    int j = blockIdx.x, tid = threadIdx.x;
    const float* L = logits + (size_t)j * VOCAB;
    const float* pr = params + j * 8;
    float temp = pr[0], top_p = pr[2], rep = pr[3], u01 = pr[4], u2 = pr[5], pres = pr[7];
    int top_k = (int)pr[1], draft = (int)pr[6];
    int row_tok = out_rows[j], first = row_first[j];
    // a sampled draft: its distribution, if it is still the one stored for the row (else as a draft of the argmax)
    int qrow = -1;
    if (draft <= -2) {
        draft = -draft - 2;
        int t = row_tok - first, q = tok_slot[row_tok] * QD + t;
        if (t < QD && qtok[q] == draft && qn[q] > 0) qrow = q;
    }
    const unsigned* sb = seen + (size_t)tok_slot[row_tok] * SEEN_WORDS;
    const unsigned* ob = outs + (size_t)tok_slot[row_tok] * SEEN_WORDS;
    auto val = [&](int i) {
        float v = L[i];
        if (rep != 1.f || pres != 0.f) {
            bool step = false;  // among this step's drafts so far
            for (int q = first + 1; q <= row_tok && !step; q++) step = ids[q] == i;
            if (rep != 1.f && (step || ((sb[i >> 5] >> (i & 31)) & 1))) v = v > 0.f ? v / rep : v * rep;
            if (pres != 0.f && (step || ((ob[i >> 5] >> (i & 31)) & 1))) v -= pres;
        }
        return v;
    };
    bool greedy = temp <= 0.f || top_k == 1;
    // more candidates than a block sorts (top_k 0: the whole vocabulary): the host's row (Step::sample)
    if (!greedy && (top_k == 0 || top_k > QCAP)) {
        if (tid == 0) out_acc[j] = -1;
        return;
    }
    int k = greedy ? 1 : min(top_k, VOCAB);
    int cnt = top_cands(val, VOCAB, k, hist, cand, sh, sv);
    auto token = [&](int r) { return cand_index(cand[r]); };
    if (greedy) {
        if (tid == 0) { out_tok[j] = token(0); out_acc[j] = draft >= 0 && token(0) == draft; }
        return;
    }
    k = min(k, cnt);
    float p, cum, mass;
    int keep = kept(val, cand, k, temp, top_p, sh, sv, p, cum, &mass);
    // the draft's probability among the kept tokens
    if (tid == 0) { sh[2] = 0.f; s_bin = 0xffffffffu; }
    __syncthreads();
    if (draft >= 0 && tid < keep && token(tid) == draft) { sh[2] = p / mass; s_bin = tid; }
    __syncthreads();
    float pd = sh[2];
    unsigned at_d = s_bin;
    if (qrow >= 0) {
        const int* qi = qid + (size_t)qrow * QCAP;
        const float* qv = qp + (size_t)qrow * QCAP;
        int nq = qn[qrow];
        float qd = q_of(qi, qv, nq, draft);
        if (u01 * qd < pd) {
            if (tid == 0) { out_tok[j] = draft; out_acc[j] = 1; }
            return;
        }
        // the residual max(0, p - q) over the kept tokens (p's support), drawn with u2
        float r = tid < keep ? fmaxf(p / mass - q_of(qi, qv, nq, token(tid)), 0.f) : 0.f;
        float cr = block_scan(r, sh);
        if (tid == 1023) sh[3] = cr;
        if (tid == 0) { sv[0] = 0xffffffffu; sv[1] = 0; }
        __syncthreads();
        float target = u2 * sh[3];
        if (r > 0.f && cr >= target) atomicMin(&sv[0], tid);
        if (r > 0.f) atomicMax(&sv[1], tid);
        __syncthreads();
        if (tid == 0) {
            // none crossed (rounding at the top end): the last token with residual mass. A rejection leaves residual
            // mass (q(d) > p(d)); only rounding can leave none, then the most likely token.
            unsigned at = sv[0] != 0xffffffffu ? sv[0] : sv[1];
            out_tok[j] = token(sh[3] > 0.f ? (int)at : 0);
            out_acc[j] = 0;
        }
        return;
    }
    if (draft >= 0 && u01 < pd) {
        if (tid == 0) { out_tok[j] = draft; out_acc[j] = 1; }
        return;
    }
    // draw from the kept tokens (without the rejected draft)
    float target = draft >= 0 ? u2 * (mass - pd * mass) : u01 * mass;
    float c2 = cum - (at_d != 0xffffffffu && (unsigned)tid >= at_d ? pd * mass : 0.f);
    bool live = tid < keep && (unsigned)tid != at_d;
    if (tid == 0) sv[2] = keep - 1;
    __syncthreads();
    if (live && c2 >= target) atomicMin(&sv[2], tid);
    __syncthreads();
    if (tid == 0) {
        int r = sv[2];
        if ((unsigned)r == at_d) r = r + 1 < keep ? r + 1 : r - 1;  // only when the draft was the last one left
        out_tok[j] = token(r);
        out_acc[j] = 0;
    }
}

// Sampled drafts: one block of 1024 threads per output row of a draft step, over its W logits (rows ld apart; the draft
// head's rows: token head_ids[i] for column i; null: column i is token i). The row's sampling as sample_rows does it
// (penalties of the row's slot and of its chain's drafts before t, temperature, top-k, top-p), then a draw d ~ q with
// u; q is stored for (slot, t) (see QD), d in drafts[row]. params[row]: temperature, top_k, top_p, repetition penalty,
// u, t, -, presence penalty. Rows need a temperature > 0 (top_k 1: the argmax, sample::argmax_draft_row).
extern "C" __global__ void __launch_bounds__(1024)
    draft_sample(const float* logits, int W, int ld, const int* head_ids, const int* out_rows, const int* tok_slot,
                 const unsigned* seen, const unsigned* outs, const float* params, int* drafts, int* qtok, int* qn,
                 int* qid, float* qp) {
    __shared__ unsigned hist[4096];
    __shared__ unsigned long long cand[SCAP];
    __shared__ float sh[32];
    __shared__ unsigned sv[3];
    int j = blockIdx.x, tid = threadIdx.x;
    const float* L = logits + (size_t)j * ld;
    const float* pr = params + j * 8;
    float temp = pr[0], top_p = pr[2], rep = pr[3], u = pr[4], pres = pr[7];
    int top_k = (int)pr[1], t = (int)pr[5];
    int slot = tok_slot[out_rows[j]], q = slot * QD + t;
    const unsigned* sb = seen + (size_t)slot * SEEN_WORDS;
    const unsigned* ob = outs + (size_t)slot * SEEN_WORDS;
    const int* chain = qtok + slot * QD;
    auto id = [&](int i) { return head_ids ? head_ids[i] : i; };
    auto val = [&](int i) {
        float v = L[i];
        if (rep != 1.f || pres != 0.f) {
            int x = id(i);
            bool step = false;  // among the chain's drafts before this one
            for (int c = 0; c < t && !step; c++) step = chain[c] == x;
            if (rep != 1.f && (step || ((sb[x >> 5] >> (x & 31)) & 1))) v = v > 0.f ? v / rep : v * rep;
            if (pres != 0.f && (step || ((ob[x >> 5] >> (x & 31)) & 1))) v -= pres;
        }
        return v;
    };
    int k = min(top_k > 0 ? min(top_k, W) : 1024, QCAP);
    int cnt = top_cands(val, W, k, hist, cand, sh, sv);
    k = min(k, cnt);
    float p, cum, mass;
    int keep = kept(val, cand, k, temp, top_p, sh, sv, p, cum, &mass);
    // the draw: the first kept token whose prefix reaches u * mass
    float target = u * mass;
    if (tid == 0) sv[2] = keep - 1;
    __syncthreads();
    if (tid < keep && cum >= target) atomicMin(&sv[2], tid);
    __syncthreads();
    int x = tid < keep ? id(cand_index(cand[tid])) : 0x7fffffff;
    float prob = tid < keep ? p / mass : 0.f;
    if (tid == (int)sv[2]) { drafts[j] = x; qtok[q] = x; qn[q] = keep; }
    __syncthreads();
    // q in ascending id order (for q_of): (id << 32 | probability bits), bitonic sort over the kept tokens
    int np = 1;
    while (np < keep) np <<= 1;
    for (int i = keep + tid; i < np; i += 1024) cand[i] = ~0ull;
    if (tid < keep) cand[tid] = ((unsigned long long)(unsigned)x << 32) | __float_as_uint(prob);
    __syncthreads();
    bitonic<true>(cand, np, tid);
    if (tid < keep) {
        qid[(size_t)q * QCAP + tid] = (int)(cand[tid] >> 32);
        qp[(size_t)q * QCAP + tid] = __uint_as_float((unsigned)cand[tid]);
    }
}
