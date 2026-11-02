//! Qwen inference on MLX. A step's tokens share their matrix products; attention and DeltaNet run per contiguous
//! sequence segment against that slot's state. Sampling is the shared host sampler.
//!
//! The KV cache holds what CUDA's holds: int8 values with an f32 scale per position and head (q = round(x * 127 /
//! max|x|), read as q * max|x| / 127), dequantized to f32 for attention. DeltaNet rolls back as CUDA's does
//! (kernels/qwen_deltanet.cuh): a step may restart a slot at most KEEP positions before its last step's end, inside
//! that step; per layer the slot keeps the state at the checkpoint ck = max(first position, end - KEEP), the delta
//! rule's inputs of the positions ck .. end and the raw convolution inputs of the last KEEP + 3 positions, and a
//! restart replays the accepted tokens from the checkpoint (delta::token, the same operations as the step's own).
use super::delta::{self, CH};
use super::weights::{self as w, CONV_K, Fused, HD, Layer, LKD, LK_HEADS, LVD, MResult, NKV, ROT, THETA, Weights};
use crate::engine::{Engine, Tok};
use crate::sample::sample_row;
use anyhow::{Context, Result, ensure};
use mlx_rs::fast::ScaledDotProductAttentionMask;
use mlx_rs::ops::indexing::{IndexOp, TryIndexMutOp, argmax_axis};
use mlx_rs::{Array, Dtype, fast, nn, ops, transforms};
use std::path::Path;

const MAX_TOK: usize = 2048;
const MAX_OUT: usize = 64;
/// draft tokens per step (all decoding requests together), as CUDA's SPEC_TOK
const SPEC_TOK: usize = 16;
const KV_CHUNK: usize = 1024;
/// rollback reach: as CUDA's KEEP
const KEEP: usize = 16;
/// segments up to this length run token by token (decode, drafts); longer ones chunked (delta::chunked)
const SEQ_MAX: usize = 16;

struct Segment {
    a: usize,
    b: usize,
    slot: usize,
    pos0: usize,
    /// replay this many tokens from the checkpoint (a restart inside the last step), else continue at the end
    replay: Option<usize>,
    /// index into the slot's conv history of position pos0 - 3
    hfrom: usize,
    /// the new checkpoint and the first position of the new conv history
    ck: usize,
    hpos: isize,
}

/// One slot's attention cache in one layer: int8 keys and values [NKV][cap][HD], f32 scales [NKV][cap][1]
struct Kv {
    k: Array,
    ks: Array,
    v: Array,
    vs: Array,
}

/// The delta rule's inputs of the positions ck .. end (delta::Tok, per token)
struct Bank {
    kq: Array,
    v: Array,
    e: Array,
    b: Array,
    dot: Array,
}

impl Bank {
    fn tok(&self, i: i32) -> MResult<(Array, Array, Array, Array, Array)> {
        let r = |x: &Array| delta::row(x, i);
        Ok((r(&self.kq)?, r(&self.v)?, r(&self.e)?, r(&self.b)?, r(&self.dot)?))
    }
}

/// One slot's DeltaNet layer: the state [Hv][Dv][Dk] at its end and at the checkpoint, the bank, the raw conv inputs
/// [n][C] of the positions hpos .. end (hpos -3 for a new sequence: zeros)
struct DeltaState {
    state: Array,
    ck_state: Array,
    bank: Option<Bank>,
    hist: Array,
}

pub struct Model {
    w: Weights,
    slots: usize,
    max_ctx: usize,
    /// per slot, per attention layer
    kv: Vec<Vec<Kv>>,
    cap: Vec<usize>,
    /// per slot, per DeltaNet layer
    delta: Vec<Vec<DeltaState>>,
    zero_state: Array,
    zero_hist: Array,
    /// per slot: the end of its last step, its checkpoint and the first position of its conv history
    end: Vec<usize>,
    ck: Vec<usize>,
    hpos: Vec<isize>,
    seen: Vec<Vec<u64>>,
    outs: Vec<Vec<u32>>,
    logits: Option<Array>,
    /// per logit row: (slot, first token of its segment, its token)
    rows: Vec<(usize, usize, usize)>,
    ids: Vec<u32>,
    /// compiled per-token stages shared by all DeltaNet layers (delta::token; the segment's convolution and inputs;
    /// its output norm and gate)
    fused: SegFns,
}

struct SegFns {
    token: Fused,
    prep: Fused,
    post: Fused,
}

impl Model {
    pub fn load(dir: &Path, max_ctx: usize, slots: usize, int8: bool) -> Result<Self> {
        ensure!(max_ctx > 0 && slots > 0, "context and slots must be positive");
        w::init()?;
        let w = Weights::load(dir, Dtype::Float16, int8)?;
        let (_, n_delta) = w.counts();
        let d = w.d;
        let zero_state = ops::zeros::<f32>(&[d.lv_heads, LVD, LKD])?;
        let zero_hist = ops::zeros::<f32>(&[CONV_K - 1, d.conv_width()])?;
        let fresh = || DeltaState { state: zero_state.clone(), ck_state: zero_state.clone(), bank: None,
                                    hist: zero_hist.clone() };
        let delta = (0..slots).map(|_| (0..n_delta).map(|_| fresh()).collect()).collect();
        let vocab = d.vocab;
        Ok(Model { w, slots, max_ctx, kv: (0..slots).map(|_| Vec::new()).collect(), cap: vec![0; slots], delta,
                   zero_state, zero_hist, end: vec![0; slots], ck: vec![0; slots], hpos: vec![-3; slots],
                   seen: vec![vec![0; vocab.div_ceil(64)]; slots], outs: vec![vec![0; vocab / 32]; slots],
                   logits: None, rows: Vec::new(), ids: Vec::new(), fused: SegFns::new(d) })
    }

    fn reserve(&mut self, slot: usize, end: usize) -> Result<()> {
        if end <= self.cap[slot] {
            return Ok(());
        }
        let cap = end.next_multiple_of(KV_CHUNK).min(self.max_ctx);
        ensure!(end <= cap, "position {end} beyond the context of {}", self.max_ctx);
        let more = cap - self.cap[slot];
        let q = || ops::zeros::<i8>(&[NKV, more as i32, HD]);
        let s = || ops::zeros::<f32>(&[NKV, more as i32, 1]);
        if self.cap[slot] == 0 {
            let n_attn = self.w.counts().0;
            self.kv[slot] = (0..n_attn).map(|_| Ok(Kv { k: q()?, ks: s()?, v: q()?, vs: s()? })).collect::<MResult<_>>()?;
        } else {
            let (q, s) = (q()?, s()?);
            for c in self.kv[slot].iter_mut() {
                c.k = ops::concatenate(&[&c.k, &q], 1)?;
                c.ks = ops::concatenate(&[&c.ks, &s], 1)?;
                c.v = ops::concatenate(&[&c.v, &q], 1)?;
                c.vs = ops::concatenate(&[&c.vs, &s], 1)?;
            }
        }
        self.cap[slot] = cap;
        Ok(())
    }

    /// A new sequence in `slot`: zero recurrent state and conv history
    fn reset(&mut self, slot: usize) {
        for st in self.delta[slot].iter_mut() {
            *st = DeltaState { state: self.zero_state.clone(), ck_state: self.zero_state.clone(), bank: None,
                               hist: self.zero_hist.clone() };
        }
        self.end[slot] = 0;
        self.ck[slot] = 0;
        self.hpos[slot] = -3;
        self.seen[slot].fill(0);
    }

    fn forward(&mut self, toks: &[Tok], segs: &[Segment], out: &[usize]) -> MResult<Option<Array>> {
        let t = toks.len() as i32;
        let ids: Vec<i32> = toks.iter().map(|x| x.id as i32).collect();
        let h = self.w.embed.take_axis(&Array::from_slice(&ids, &[t]), 0)?.as_dtype(Dtype::Float32)?;
        let pos: Vec<i32> = toks.iter().map(|x| x.pos as i32).collect();
        let rope = rope_tables(&pos)?;
        let mut cache = Cache { kv: &mut self.kv, delta: &mut self.delta, segs, rope, masks: masks(segs)?, d: self.w.d,
                                fused: &mut self.fused };
        let h = self.w.layers(&mut cache, h)?;
        if out.is_empty() {
            return Ok(None);
        }
        let rows: Vec<i32> = out.iter().map(|&r| r as i32).collect();
        let hf = w::rms(&h.take_axis(&Array::from_slice(&rows, &[rows.len() as i32]), 0)?, &self.w.final_norm)?
            .as_dtype(self.w.dtype)?;
        let logits = match &self.w.head {
            Some(head) => head.apply(&hf),
            None => ops::matmul(&hf, ops::transpose(&self.w.embed)?)?.as_dtype(Dtype::Float32),
        }?;
        Ok(Some(logits))
    }
}

/// cos and sin [T][1][ROT], split-half (nontraditional) RoPE, theta 10^7 and 64 rotary dimensions.
fn rope_tables(pos: &[i32]) -> MResult<[(Array, Array); 2]> {
    let inv: Vec<f32> = (0..ROT / 2).map(|d| THETA.powf(-2.0 * d as f64 / ROT as f64) as f32).collect();
    let (mut c, mut s) = (Vec::with_capacity(pos.len() * ROT as usize), Vec::with_capacity(pos.len() * ROT as usize));
    for &p in pos {
        for _ in 0..2 {
            for &f in &inv {
                let a = p as f32 * f;
                c.push(a.cos());
                s.push(a.sin());
            }
        }
    }
    let c = Array::from_slice(&c, &[pos.len() as i32, ROT]).reshape(&[pos.len() as i32, 1, ROT])?;
    let s = Array::from_slice(&s, &[pos.len() as i32, ROT]).reshape(&[pos.len() as i32, 1, ROT])?;
    Ok([(c.clone(), s.clone()), (c, s)])
}

fn masks(segs: &[Segment]) -> MResult<Vec<Option<Array>>> {
    segs.iter().map(|g| {
        let len = (g.b - g.a) as i32;
        if len == 1 {
            return Ok(None);
        }
        let end = (g.pos0 + g.b - g.a) as i32;
        let qp = ops::arange::<_, i32>(g.pos0 as i32, end, 1)?.reshape(&[len, 1])?;
        let kp = ops::arange::<_, i32>(0, end, 1)?.reshape(&[1, end])?;
        Ok(Some(ops::le(&kp, &qp)?))
    }).collect()
}

struct Cache<'a> {
    kv: &'a mut [Vec<Kv>],
    delta: &'a mut [Vec<DeltaState>],
    segs: &'a [Segment],
    rope: [(Array, Array); 2],
    masks: Vec<Option<Array>>,
    d: w::Dims,
    fused: &'a mut SegFns,
}

impl SegFns {
    fn new(d: w::Dims) -> Self {
        SegFns {
            // [s, kq, v, e, b, dot] -> [s, o]
            token: Fused::new(|a: &[Array]| {
                let (s, o) = delta::token(&a[0], delta::Tok { kq: &a[1], v: &a[2], e: &a[3], b: &a[4], dot: &a[5] })?;
                Ok(vec![s, o])
            }),
            // [conv history (3 rows), raw rows, conv weights] -> [kq, v, k . q, conv inputs from pos0 - 3 on]
            prep: Fused::new(move |a: &[Array]| {
                let (q, k, v, ext) = conv_qkv(&a[0], &a[1], &a[2], d)?;
                let dot = ops::sum_axis(ops::multiply(&q, &k)?, -1, false)?;
                Ok(vec![ops::stack(&[&k, &q], -1)?, v, dot, ext])
            }),
            // [o, norm weight, z] -> [the out-projection's input]
            post: Fused::new(move |a: &[Array]| Ok(vec![out_gate(&a[0], &a[1], &a[2], d)?])),
        }
    }

    fn token(&mut self, small: bool, s: &Array, t: delta::Tok) -> MResult<(Array, Array)> {
        let mut r = self.token.call(small, &[s.clone(), t.kq.clone(), t.v.clone(), t.e.clone(), t.b.clone(), t.dot.clone()])?;
        let o = r.pop().unwrap();
        Ok((r.pop().unwrap(), o))
    }
}

/// The causal conv (kernel 4) and SiLU of a segment's raw qkv rows [len][C] after the 3 rows before (`hist`), then q
/// and k per key head l2-normalized (q also / sqrt(Dk)) [len][Lk][Dk], v [len][Hv][Dv]; and the conv inputs of
/// positions pos0 - 3 .. pos0 + len.
fn conv_qkv(hist: &Array, raw: &Array, cw: &Array, d: w::Dims) -> MResult<(Array, Array, Array, Array)> {
    let (len, lv) = (raw.shape()[0], d.lv_heads);
    let ext = ops::concatenate(&[hist, raw], 0)?;
    let mut cv = ops::multiply(ext.index(..len), cw.index((.., 0)))?;
    for j in 1..CONV_K {
        cv = ops::add(&cv, ops::multiply(ext.index(j..j + len), cw.index((.., j)))?)?;
    }
    let cv = nn::silu(cv)?;
    let qkvw = ops::split_at_indices(&cv, &[d.key_width(), 2 * d.key_width()], -1)?;
    let q = l2(&qkvw[0].reshape(&[len, LK_HEADS, LKD])?)?;
    let k = l2(&qkvw[1].reshape(&[len, LK_HEADS, LKD])?)?;
    let q = ops::multiply(q, Array::from_f32(1.0 / (LKD as f32).sqrt()))?;
    Ok((q, k, qkvw[2].reshape(&[len, lv, LVD])?, ext))
}

/// q or k [len][Lk][Dk] for each value head [len][Hv][Dk]: every value head reads its key head (adjacent groups)
fn repeat(x: &Array, lv: i32) -> MResult<Array> {
    let len = x.shape()[0];
    ops::broadcast_to(ops::expand_dims(x, 2)?, &[len, LK_HEADS, lv / LK_HEADS, LKD])?.reshape(&[len, lv, LKD])
}

/// norm_w * o / rms(o) * silu(z) per value head: o [len][Hv][Dv], z [len][Hv * Dv]
fn out_gate(o: &Array, norm_w: &Array, z: &Array, d: w::Dims) -> MResult<Array> {
    let len = o.shape()[0];
    let yn = fast::rms_norm(o, Some(norm_w), w::EPS)?;
    ops::multiply(&yn, nn::silu(z.reshape(&[len, d.lv_heads, LVD])?)?)
}

/// An array's own copy (a slice shares, and keeps alive, the buffer of what it slices); x * 1 is exact
fn own(x: &Array) -> MResult<Array> {
    ops::multiply(x, Array::from_f32(1.0))
}

/// n zero rows in front and `back` behind of x [len][..]
fn pad_rows(x: &Array, front: i32, back: i32) -> MResult<Array> {
    if front == 0 && back == 0 {
        return Ok(x.clone());
    }
    let mut sh = x.shape().to_vec();
    let mut parts = Vec::with_capacity(3);
    if front > 0 {
        sh[0] = front;
        parts.push(ops::zeros::<f32>(&sh)?);
    }
    parts.push(x.clone());
    if back > 0 {
        sh[0] = back;
        parts.push(ops::zeros::<f32>(&sh)?);
    }
    ops::concatenate(&parts, 0)
}

impl w::Mix for Cache<'_> {
    fn rope(&self) -> (Array, Array) {
        self.rope[0].clone()
    }

    fn attend(&mut self, ai: usize, q: &Array, kq: &Array, ks: &Array, vq: &Array, vs: &Array) -> MResult<Array> {
        let mut parts = Vec::with_capacity(self.segs.len());
        for (g, mask) in self.segs.iter().zip(&self.masks) {
            let (a, b, len) = (g.a as i32, g.b as i32, (g.b - g.a) as i32);
            let heads = |x: &Array| ops::swap_axes(&x.index(a..b), 0, 1);
            let (p0, end) = (g.pos0 as i32, (g.pos0 + len as usize) as i32);
            let c = &mut self.kv[g.slot][ai];
            c.k.try_index_mut((.., p0..end, ..), &heads(&kq)?)?;
            c.ks.try_index_mut((.., p0..end, ..), &heads(&ks)?)?;
            c.v.try_index_mut((.., p0..end, ..), &heads(&vq)?)?;
            c.vs.try_index_mut((.., p0..end, ..), &heads(&vs)?)?;
            let read = |x: &Array, s: &Array| -> MResult<Array> {
                ops::expand_dims(ops::multiply(x.index((.., ..end, ..)).as_dtype(Dtype::Float32)?, s.index((.., ..end, ..)))?, 0)
            };
            let mask = mask.as_ref().map(ScaledDotProductAttentionMask::Array);
            let o = fast::scaled_dot_product_attention(ops::expand_dims(heads(q)?, 0)?, read(&c.k, &c.ks)?,
                                                       read(&c.v, &c.vs)?, 1.0 / (HD as f32).sqrt(), mask, None)?;
            parts.push(ops::swap_axes(&o.reshape(&[self.d.nh, len, HD])?, 0, 1)?);
        }
        ops::concatenate(&parts, 0)
    }

    fn delta(&mut self, di: usize, layer: &Layer, small: bool, qkv: &Array, z: &Array, beta: &Array, log_decay: &Array,
             decay: &Array) -> MResult<Array> {
        let d = self.d;
        let (cw, norm_w) = (layer.conv_w.as_ref().unwrap(), layer.delta_norm.as_ref().unwrap());
        let mut parts = Vec::with_capacity(self.segs.len());
        for g in self.segs {
            let (a, b, len) = (g.a as i32, g.b as i32, (g.b - g.a) as i32);
            let st = &mut self.delta[g.slot][di];
            let fused = &mut *self.fused;
            let raw = qkv.index(a..b);
            // the conv inputs of positions pos0 - 3 .. pos0 from the history (zeros before a sequence's start)
            let h0 = g.hfrom as i32;
            let hist = st.hist.index(h0..h0 + CONV_K - 1);
            let keep = (g.hpos - (g.pos0 as isize - 3)) as i32;
            let (e, lg, bt) = (decay.index(a..b), log_decay.index(a..b), beta.index(a..b));
            // the state the segment starts from
            let mut s = match g.replay {
                None => st.state.clone(),
                Some(r) => {
                    let mut s = st.ck_state.clone();
                    for i in 0..r as i32 {
                        let (kq, v, e, b, dot) = st.bank.as_ref().expect("a bank to replay from").tok(i)?;
                        s = fused.token(small, &s, delta::Tok { kq: &kq, v: &v, e: &e, b: &b, dot: &dot })?.0;
                    }
                    s
                }
            };
            let cko = (g.ck - g.pos0) as i32;
            let o;
            if len as usize <= SEQ_MAX {
                // token by token; the checkpoint is where the segment starts (len <= KEEP)
                debug_assert!(cko == 0);
                let r = fused.prep.call(small, &[hist, raw, cw.clone()])?;
                let (kq, v, dot) = (&r[0], &r[1], &r[2]);
                st.hist = r[3].index(keep..);
                st.ck_state = s.clone();
                let mut ys = Vec::with_capacity(len as usize);
                for t in 0..len {
                    let r = |x: &Array| delta::row(x, t);
                    let (s2, y) = fused.token(small, &s, delta::Tok { kq: &r(kq)?, v: &r(v)?, e: &r(&e)?, b: &r(&bt)?,
                                                                     dot: &r(dot)? })?;
                    s = s2;
                    ys.push(y);
                }
                st.bank = Some(Bank { kq: kq.clone(), v: v.clone(), e, b: bt, dot: dot.clone() });
                o = ops::stack(&ys, 0)?;
            } else {
                let (q1, k1, v, ext) = conv_qkv(&hist, &raw, cw, d)?;
                st.hist = own(&ext.index(keep..))?;
                let dot = ops::sum_axis(ops::multiply(&q1, &k1)?, -1, false)?;
                let (q, k) = (repeat(&q1, d.lv_heads)?, repeat(&k1, d.lv_heads)?);
                // chunks aligned so that the checkpoint is a chunk boundary: zero tokens in front, and behind to whole
                // chunks
                let front = (CH - cko % CH) % CH;
                let back = (CH - (front + len) % CH) % CH;
                let p = |x: &Array| pad_rows(x, front, back);
                let (y, s_end, s_ck) = delta::chunked(&s, &p(&q)?, &p(&k)?, &p(&v)?, &p(&lg)?, &p(&bt)?, front + cko)?;
                st.ck_state = s_ck;
                s = s_end;
                let tail = |x: &Array| own(&x.index(cko..));
                st.bank = Some(Bank { kq: ops::stack(&[&k1.index(cko..), &q1.index(cko..)], -1)?, v: tail(&v)?, e: tail(&e)?,
                                      b: tail(&bt)?, dot: tail(&dot)? });
                o = y.index(front..front + len);
            }
            st.state = s;
            parts.push(fused.post.call(small, &[o, norm_w.clone(), z.index(a..b)])?.remove(0));
        }
        ops::concatenate(&parts, 0)?.reshape(&[qkv.shape()[0], d.value_width()])
    }
}

/// x / sqrt(sum(x^2) + eps), over each head's key dimension, in f32.
fn l2(x: &Array) -> MResult<Array> {
    let ss = ops::sum_axis(ops::square(x)?, -1, true)?;
    ops::multiply(x, ops::rsqrt(ops::add(ss, Array::from_f32(w::EPS))?)?)
}

impl Engine for Model {
    fn slots(&self) -> usize { self.slots }
    fn max_ctx(&self) -> usize { self.max_ctx }
    fn max_tok(&self) -> usize { MAX_TOK }
    fn max_out(&self) -> usize { MAX_OUT }
    fn spec_tok(&self) -> usize { SPEC_TOK }
    fn vocab(&self) -> usize { self.w.d.vocab }

    fn step(&mut self, toks: &[Tok], out: &[usize], marks: Option<&[Vec<u32>]>) -> Result<()> {
        ensure!((1..=MAX_TOK).contains(&toks.len()) && out.len() <= MAX_OUT, "step of {} tokens, {} rows", toks.len(), out.len());
        ensure!(out.iter().all(|&r| r < toks.len()), "an output row beyond the step");
        let mut segs: Vec<Segment> = Vec::new();
        for (i, t) in toks.iter().enumerate() {
            ensure!((t.slot as usize) < self.slots && (t.pos as usize) < self.max_ctx && (t.id as usize) < self.w.d.vocab,
                    "token {t:?} out of range");
            match segs.last_mut() {
                Some(g) if g.slot == t.slot as usize && g.pos0 + (g.b - g.a) == t.pos as usize => g.b = i + 1,
                _ => segs.push(Segment { a: i, b: i + 1, slot: t.slot as usize, pos0: t.pos as usize, replay: None,
                                         hfrom: 0, ck: 0, hpos: 0 }),
            }
        }
        let mut used = vec![false; self.slots];
        for g in &segs {
            ensure!(!std::mem::replace(&mut used[g.slot], true), "slot {} twice in one step", g.slot);
        }
        for (k, g) in segs.iter_mut().enumerate() {
            let s = g.slot;
            if g.pos0 == 0 {
                self.reset(s);
            }
            let (end, len) = (self.end[s], g.b - g.a);
            if g.pos0 != end {
                ensure!(self.ck[s] <= g.pos0 && g.pos0 < end,
                        "slot {s} restarts at {}: its last step ended at {end}, rollback reaches back to {}",
                        g.pos0, self.ck[s]);
                g.replay = Some(g.pos0 - self.ck[s]);
            }
            g.hfrom = (g.pos0 as isize - 3 - self.hpos[s]) as usize;
            let new_end = g.pos0 + len;
            g.ck = g.pos0.max(new_end.saturating_sub(KEEP));
            g.hpos = (g.pos0 as isize - 3).max(new_end as isize - KEEP as isize - 3);
            self.reserve(s, new_end)?;
            for &id in marks.and_then(|m| m.get(k)).into_iter().flatten() {
                ensure!((id as usize) < self.w.d.vocab, "mark token {id} out of range");
                self.seen[s][id as usize / 64] |= 1 << (id % 64);
            }
        }
        let logits = self.forward(toks, &segs, out)?;
        // Commit logits and all lazy state writes together; no graph may span engine steps.
        let kv = self.kv.iter().flatten().flat_map(|c| [&c.k, &c.ks, &c.v, &c.vs]);
        let delta = self.delta.iter().flatten().flat_map(|st| {
            [&st.state, &st.ck_state, &st.hist].into_iter()
                .chain(st.bank.iter().flat_map(|b| [&b.kq, &b.v, &b.e, &b.b, &b.dot]))
        });
        transforms::eval(logits.iter().chain(kv).chain(delta))?;
        for g in &segs {
            self.end[g.slot] = g.pos0 + g.b - g.a;
            self.ck[g.slot] = g.ck;
            self.hpos[g.slot] = g.hpos;
        }
        self.rows = out.iter().map(|&r| {
            let g = segs.iter().find(|g| (g.a..g.b).contains(&r)).unwrap();
            (g.slot, g.a, r)
        }).collect();
        self.ids = toks.iter().map(|t| t.id).collect();
        self.logits = logits;
        Ok(())
    }

    fn logits(&self) -> Result<Vec<f32>> {
        if self.rows.is_empty() { return Ok(Vec::new()); }
        w::host(self.logits.as_ref().context("no step yet")?)
    }

    fn logits_row(&self, j: usize) -> Result<Vec<f32>> {
        w::host(&self.logits.as_ref().context("no step yet")?.index(j as i32))
    }

    fn argmax(&self) -> Result<Vec<u32>> {
        if self.rows.is_empty() { return Ok(Vec::new()); }
        let a = argmax_axis(self.logits.as_ref().context("no step yet")?, -1, false)?.as_dtype(Dtype::Uint32)?;
        a.eval()?;
        Ok(a.as_slice::<u32>().to_vec())
    }

    fn set_outputs(&mut self, slot: u32, words: &[u32]) -> Result<()> {
        ensure!((slot as usize) < self.slots && words.len() == self.w.d.vocab / 32,
                "outputs of slot {slot}: {} words", words.len());
        self.outs[slot as usize].copy_from_slice(words);
        Ok(())
    }

    fn sample(&mut self, params: &[[f32; 8]]) -> Result<Vec<(u32, bool)>> {
        ensure!(params.len() == self.rows.len(), "{} sampling rows for {} logit rows", params.len(), self.rows.len());
        if params.is_empty() { return Ok(Vec::new()); }
        let all = self.logits()?;
        let vocab = self.w.d.vocab;
        Ok(self.rows.iter().zip(params).enumerate().map(|(j, (&(slot, first, tok), p))| {
            let step = |id: u32| self.ids[first + 1..=tok].contains(&id);
            let seen = |id: u32| self.seen[slot][id as usize / 64] >> (id % 64) & 1 == 1 || step(id);
            let generated = |id: u32| self.outs[slot][id as usize / 32] >> (id % 32) & 1 == 1 || step(id);
            sample_row(&all[j * vocab..(j + 1) * vocab], seen, generated, p)
        }).collect())
    }

    fn release(&mut self, slot: u32) -> Result<()> {
        ensure!((slot as usize) < self.slots, "slot {slot} out of range");
        let s = slot as usize;
        self.reset(s);
        self.kv[s].clear();
        self.cap[s] = 0;
        Ok(())
    }

    fn sync(&self) -> Result<()> { Ok(()) }
}
