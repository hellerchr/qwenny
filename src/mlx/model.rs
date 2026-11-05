//! Qwen inference on MLX. A step's tokens share their matrix products; attention and DeltaNet run per contiguous
//! sequence segment against that slot's state. Sampling is the shared host sampler.
//!
//! The KV cache holds what CUDA's holds: int8 values with an f32 scale per position and head (q = round(x * 127 /
//! max|x|), read as q * max|x| / 127), dequantized to f32 for attention. DeltaNet rolls back as CUDA's does
//! (kernels/qwen_deltanet.cuh): a step may restart a slot at most KEEP positions before its last step's end, inside
//! that step; per layer the slot keeps the state at the checkpoint ck = max(first position, end - KEEP), the delta
//! rule's inputs of the positions ck .. end and the raw convolution inputs of the last KEEP + 3 positions, and a
//! restart replays the accepted tokens from the checkpoint (delta::token, the same operations as the step's own).
#[cfg(not(has_metal))]
use super::delta::{self, CH, pad_rows, repeat};
use super::weights::{self as w, CONV_K, Fused, HD, LK_HEADS, LKD, LVD, Layer, MResult, NKV, ROT, THETA, Weights};
use crate::engine::{DraftSampling, Engine, Tok};
use crate::sample::{kept_dist, sample_row, sample_row_q};
use anyhow::{Context, Result, ensure};
use mlx_rs::fast::ScaledDotProductAttentionMask;
use mlx_rs::ops::indexing::{IndexOp, TryIndexMutOp, argmax_axis};
use mlx_rs::{Array, Dtype, fast, nn, ops, transforms};
use std::path::Path;

pub const MAX_TOK: usize = 2048;
const MAX_OUT: usize = 64;
/// draft tokens per step (all decoding requests together), as CUDA's SPEC_TOK
const SPEC_TOK: usize = 16;
const KV_CHUNK: usize = 1024;
/// rollback reach: as CUDA's KEEP
const KEEP: usize = 16;
/// segments up to this length run token by token (decode, drafts); longer ones chunked (delta::chunked)
const SEQ_MAX: usize = 16;

/// A sampled draft distribution (token, probability) and the token it ends in
type DraftDist = (Vec<(u32, f32)>, u32);

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

#[cfg(not(has_metal))]
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

impl DeltaState {
    fn new(zero_state: &Array, zero_hist: &Array) -> Self {
        DeltaState { state: zero_state.clone(), ck_state: zero_state.clone(), bank: None, hist: zero_hist.clone() }
    }
}

/// Every KV cache's arrays (to evaluate)
fn kv_arrays(kv: &[Vec<Kv>]) -> impl Iterator<Item = &Array> {
    kv.iter().flatten().flat_map(|c| [&c.k, &c.ks, &c.v, &c.vs])
}

/// Every DeltaNet state's arrays (to evaluate)
fn delta_arrays(delta: &[Vec<DeltaState>]) -> impl Iterator<Item = &Array> {
    delta.iter().flatten().flat_map(|st| {
        [&st.state, &st.ck_state, &st.hist]
            .into_iter()
            .chain(st.bank.iter().flat_map(|b| [&b.kq, &b.v, &b.e, &b.b, &b.dot]))
    })
}

/// Rows `rows` of x
fn take_rows(x: &Array, rows: impl Iterator<Item = usize>) -> MResult<Array> {
    let rows: Vec<i32> = rows.map(|r| r as i32).collect();
    x.take_axis(Array::from_slice(&rows, &[rows.len() as i32]), 0)
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
    /// Target residual stream of the last step (all rows, for teacher-forced MTP cache fills).
    hidden: Option<Array>,
    /// Sampled draft distributions, by slot and chain index; kept until target verification.
    draft_dist: Vec<Vec<DraftDist>>,
    draft_end: Vec<usize>,
    /// per logit row: (slot, first token of its segment, its token)
    rows: Vec<(usize, usize, usize)>,
    ids: Vec<u32>,
    fused: DeltaStages,
}

/// Compiled stages of a segment shared by all DeltaNet layers: per token the delta rule (delta::token; Metal has its own
/// kernels), the convolution and inputs, the output norm and gate
struct DeltaStages {
    #[cfg(not(has_metal))]
    token: Fused,
    prep: Fused,
    post: Fused,
}

impl Model {
    pub fn load(dir: &Path, max_ctx: usize, slots: usize, int8: bool) -> Result<Self> {
        ensure!(max_ctx > 0 && slots > 0, "context and slots must be positive");
        w::init()?;
        let w = Weights::load(dir, int8)?;
        let (_, n_delta) = w.counts();
        let d = w.d;
        let zero_state = ops::zeros::<f32>(&[d.lv_heads, LVD, LKD])?;
        let zero_hist = ops::zeros::<f32>(&[CONV_K - 1, d.conv_width()])?;
        let delta =
            (0..slots).map(|_| (0..n_delta).map(|_| DeltaState::new(&zero_state, &zero_hist)).collect()).collect();
        let vocab = d.vocab;
        Ok(Model {
            w,
            slots,
            max_ctx,
            kv: (0..slots).map(|_| Vec::new()).collect(),
            cap: vec![0; slots],
            delta,
            zero_state,
            zero_hist,
            end: vec![0; slots],
            ck: vec![0; slots],
            hpos: vec![-3; slots],
            seen: vec![vec![0; vocab.div_ceil(64)]; slots],
            outs: vec![vec![0; vocab / 32]; slots],
            logits: None,
            hidden: None,
            draft_dist: vec![Vec::new(); slots],
            draft_end: vec![0; slots],
            rows: Vec::new(),
            ids: Vec::new(),
            fused: DeltaStages::new(d),
        })
    }

    /// Whether token `id` is in slot `slot`'s seen set (repetition penalty)
    fn is_seen(&self, slot: usize, id: u32) -> bool {
        self.seen[slot][id as usize / 64] >> (id % 64) & 1 == 1
    }

    /// Whether token `id` is among slot `slot`'s outputs (presence penalty)
    fn is_output(&self, slot: usize, id: u32) -> bool {
        self.outs[slot][id as usize / 32] >> (id % 32) & 1 == 1
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
            let n_attn = self.w.counts().0 + usize::from(self.w.mtp.is_some());
            self.kv[slot] =
                (0..n_attn).map(|_| Ok(Kv { k: q()?, ks: s()?, v: q()?, vs: s()? })).collect::<MResult<_>>()?;
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
            *st = DeltaState::new(&self.zero_state, &self.zero_hist);
        }
        self.end[slot] = 0;
        self.ck[slot] = 0;
        self.hpos[slot] = -3;
        self.seen[slot].fill(0);
        self.draft_end[slot] = 0;
        self.draft_dist[slot].clear();
    }

    /// A step's segments: per contiguous run of positions of a slot, each slot once
    fn segments(&self, toks: &[Tok]) -> Result<Vec<Segment>> {
        let mut segs: Vec<Segment> = Vec::new();
        let mut used = vec![false; self.slots];
        for (i, t) in toks.iter().enumerate() {
            let s = t.slot as usize;
            ensure!(
                s < self.slots && (t.pos as usize) < self.max_ctx && (t.id as usize) < self.w.d.vocab,
                "token {t:?} out of range"
            );
            match segs.last_mut() {
                Some(g) if g.slot == s && g.pos0 + g.b - g.a == t.pos as usize => g.b = i + 1,
                _ => {
                    ensure!(!std::mem::replace(&mut used[s], true), "slot {s} twice in one step");
                    segs.push(Segment {
                        a: i,
                        b: i + 1,
                        slot: s,
                        pos0: t.pos as usize,
                        replay: None,
                        hfrom: 0,
                        ck: 0,
                        hpos: 0,
                    });
                }
            }
        }
        Ok(segs)
    }

    /// One teacher-forced MTP step, or one link using the preceding MTP link's residual stream.
    fn mtp_forward(&mut self, toks: &[Tok], h: &Array, out: &[usize]) -> Result<(Array, Option<Array>)> {
        let segs = self.segments(toks)?;
        for g in &segs {
            self.reserve(g.slot, g.pos0 + g.b - g.a)?;
        }
        let e = take_rows(&self.w.embed, toks.iter().map(|t| t.id as usize))?;
        let ai = self.w.counts().0;
        let m = self.w.mtp.as_mut().context("no MTP layer")?;
        let fused = ops::concatenate(&[w::rms(&e.as_dtype(Dtype::Float32)?, &m.pre_e)?, w::rms(h, &m.pre_h)?], -1)?
            .as_dtype(Dtype::Float16)?;
        let x = m.fc.apply_draft(&fused)?;
        let pos: Vec<i32> = toks.iter().map(|t| t.pos as i32).collect();
        let mut cache = Cache {
            kv: &mut self.kv,
            delta: &mut self.delta,
            segs: &segs,
            rope: rope_tables(&pos)?,
            d: self.w.d,
            fused: &mut self.fused,
        };
        let h = m.forward(&mut cache, &x, ai)?;
        let logits = if out.is_empty() {
            None
        } else {
            let hf = w::rms(&take_rows(&h, out.iter().copied())?, &m.norm)?.as_dtype(Dtype::Float16)?;
            let head = m.head.as_ref().or(self.w.head.as_ref());
            Some(match head {
                Some(head) => head.apply_draft(&hf)?,
                None => ops::matmul(&hf, ops::transpose(&self.w.embed)?)?.as_dtype(Dtype::Float32)?,
            })
        };
        transforms::eval([&h].into_iter().chain(logits.iter()).chain(kv_arrays(&self.kv)))?;
        for g in &segs {
            self.draft_end[g.slot] = g.pos0 + g.b - g.a;
        }
        Ok((h, logits))
    }

    fn forward(&mut self, toks: &[Tok], segs: &[Segment], out: &[usize]) -> MResult<Option<Array>> {
        let h = take_rows(&self.w.embed, toks.iter().map(|t| t.id as usize))?.as_dtype(Dtype::Float32)?;
        let pos: Vec<i32> = toks.iter().map(|x| x.pos as i32).collect();
        let rope = rope_tables(&pos)?;
        let mut cache =
            Cache { kv: &mut self.kv, delta: &mut self.delta, segs, rope, d: self.w.d, fused: &mut self.fused };
        let prompt = segs.iter().any(|g| g.pos0 == 0) || toks.len() > SPEC_TOK;
        let h = self.w.layers(&mut cache, h, prompt)?;
        self.hidden = self.w.mtp.as_ref().map(|_| h.clone());
        if out.is_empty() {
            return Ok(None);
        }
        let hf = w::rms(&take_rows(&h, out.iter().copied())?, &self.w.final_norm)?.as_dtype(Dtype::Float16)?;
        let logits = match &self.w.head {
            Some(head) => head.apply(&hf),
            None => ops::matmul(&hf, ops::transpose(&self.w.embed)?)?.as_dtype(Dtype::Float32),
        }?;
        Ok(Some(logits))
    }
}

/// cos and sin [T][1][ROT], split-half (nontraditional) RoPE, theta 10^7 and 64 rotary dimensions.
fn rope_tables(pos: &[i32]) -> MResult<(Array, Array)> {
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
    Ok((c, s))
}

struct Cache<'a> {
    kv: &'a mut [Vec<Kv>],
    delta: &'a mut [Vec<DeltaState>],
    segs: &'a [Segment],
    rope: (Array, Array),
    d: w::Dims,
    fused: &'a mut DeltaStages,
}

impl DeltaStages {
    fn new(d: w::Dims) -> Self {
        DeltaStages {
            // [s, kq, v, e, b, dot] -> [s, o]
            #[cfg(not(has_metal))]
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

    #[cfg(not(has_metal))]
    fn token(&mut self, small: bool, s: &Array, t: delta::Tok) -> MResult<(Array, Array)> {
        let mut r =
            self.token.call(small, &[s.clone(), t.kq.clone(), t.v.clone(), t.e.clone(), t.b.clone(), t.dot.clone()])?;
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

impl w::Mix for Cache<'_> {
    fn submit(&self, h: &Array) -> MResult<()> {
        transforms::async_eval([h].into_iter().chain(kv_arrays(self.kv)).chain(delta_arrays(self.delta)))
    }

    fn rope(&self) -> (Array, Array) {
        self.rope.clone()
    }

    fn attend(&mut self, ai: usize, q: &Array, kq: &Array, ks: &Array, vq: &Array, vs: &Array) -> MResult<Array> {
        let mut parts = Vec::with_capacity(self.segs.len());
        for g in self.segs {
            let (a, b, len) = (g.a as i32, g.b as i32, (g.b - g.a) as i32);
            let heads = |x: &Array| ops::swap_axes(x.index(a..b), 0, 1);
            let (p0, end) = (g.pos0 as i32, (g.pos0 + len as usize) as i32);
            let c = &mut self.kv[g.slot][ai];
            c.k.try_index_mut((.., p0..end, ..), &heads(kq)?)?;
            c.ks.try_index_mut((.., p0..end, ..), &heads(ks)?)?;
            c.v.try_index_mut((.., p0..end, ..), &heads(vq)?)?;
            c.vs.try_index_mut((.., p0..end, ..), &heads(vs)?)?;
            // Dequantize the cache to f16 (int8 q * f32 scale) and run attention in f16 with f32 accumulation, as
            // CUDA's tensor-core attention product. f16 inputs take MLX's fused flash-attention kernel; f32 inputs
            // fall back to a path that materializes the [len][end] scores.
            let read = |x: &Array, s: &Array| -> MResult<Array> {
                ops::expand_dims(
                    ops::multiply(x.index((.., ..end, ..)).as_dtype(Dtype::Float32)?, s.index((.., ..end, ..)))?
                        .as_dtype(Dtype::Float16)?,
                    0,
                )
            };
            // Query positions pos0..end are the last `len` of the `end` cached keys, so the built-in causal mask is
            // exact (it needs no materialized array); a single decode query attends to everything (no mask).
            let mask = (len > 1).then_some(ScaledDotProductAttentionMask::Causal);
            let qh = ops::expand_dims(heads(q)?.as_dtype(Dtype::Float16)?, 0)?;
            let o = fast::scaled_dot_product_attention(
                qh,
                read(&c.k, &c.ks)?,
                read(&c.v, &c.vs)?,
                1.0 / (HD as f32).sqrt(),
                mask,
                None,
            )?;
            parts.push(ops::swap_axes(&o.reshape(&[self.d.nh, len, HD])?.as_dtype(Dtype::Float32)?, 0, 1)?);
        }
        ops::concatenate(&parts, 0)
    }

    #[cfg_attr(has_metal, allow(unused_variables))]
    fn delta(
        &mut self,
        di: usize,
        layer: &Layer,
        small: bool,
        qkv: &Array,
        z: &Array,
        beta: &Array,
        log_decay: &Array,
        decay: &Array,
    ) -> MResult<Array> {
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
            let (e, bt) = (decay.index(a..b), beta.index(a..b));
            // the state the segment starts from; on Metal a short segment's replay runs fused with it, below
            let mut s = match g.replay {
                None => st.state.clone(),
                Some(r) => {
                    let mut s = st.ck_state.clone();
                    let bank = st.bank.as_ref();
                    #[cfg(has_metal)]
                    if r > 0 && len as usize > SEQ_MAX {
                        let bank = bank.expect("a bank to replay from");
                        let r = r as i32;
                        s = super::metal::delta(
                            &s,
                            &bank.kq.index(..r),
                            &bank.v.index(..r),
                            &bank.e.index(..r),
                            &bank.b.index(..r),
                            &bank.dot.index(..r),
                            0,
                        )?
                        .0;
                    }
                    #[cfg(not(has_metal))]
                    for i in 0..r as i32 {
                        let (kq, v, e, b, dot) = bank.expect("a bank to replay from").tok(i)?;
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
                #[cfg(has_metal)]
                {
                    let (s2, ck, y) = match g.replay {
                        Some(r) if r > 0 => {
                            let bank = st.bank.as_ref().expect("a bank to replay from");
                            super::metal::delta_replay(
                                &s,
                                [&bank.kq, &bank.v, &bank.e, &bank.b],
                                r as i32,
                                [kq, v, &e, &bt, dot],
                            )?
                        }
                        _ => {
                            let (s2, _, y) = super::metal::delta(&s, kq, v, &e, &bt, dot, 0)?;
                            (s2, s.clone(), y)
                        }
                    };
                    st.ck_state = ck;
                    s = s2;
                    o = y;
                }
                #[cfg(not(has_metal))]
                {
                    let mut ys = Vec::with_capacity(len as usize);
                    for t in 0..len {
                        let r = |x: &Array| delta::row(x, t);
                        let (s2, y) = fused.token(
                            small,
                            &s,
                            delta::Tok { kq: &r(kq)?, v: &r(v)?, e: &r(&e)?, b: &r(&bt)?, dot: &r(dot)? },
                        )?;
                        s = s2;
                        ys.push(y);
                    }
                    o = ops::stack(&ys, 0)?;
                }
                st.bank = Some(Bank { kq: kq.clone(), v: v.clone(), e, b: bt, dot: dot.clone() });
            } else {
                let (q1, k1, v, ext) = conv_qkv(&hist, &raw, cw, d)?;
                st.hist = own(&ext.index(keep..))?;
                let dot = ops::sum_axis(ops::multiply(&q1, &k1)?, -1, false)?;
                #[cfg(has_metal)]
                let (y, s_end, s_ck) = {
                    let (s_end, s_ck, y) =
                        super::metal::delta(&s, &ops::stack(&[&k1, &q1], -1)?, &v, &e, &bt, &dot, cko)?;
                    (y, s_end, s_ck)
                };
                #[cfg(not(has_metal))]
                let (y, s_end, s_ck) = {
                    let (q, k) = (repeat(&q1, d.lv_heads)?, repeat(&k1, d.lv_heads)?);
                    let lg = log_decay.index(a..b);
                    // chunks aligned so that the checkpoint is a chunk boundary: zero tokens in front, and behind to whole
                    // chunks
                    let front = (CH - cko % CH) % CH;
                    let back = (CH - (front + len) % CH) % CH;
                    let p = |x: &Array| pad_rows(x, front, back);
                    let (y, s_end, s_ck) =
                        delta::chunked(&s, &p(&q)?, &p(&k)?, &p(&v)?, &p(&lg)?, &p(&bt)?, front + cko)?;
                    (y.index(front..front + len), s_end, s_ck)
                };
                st.ck_state = s_ck;
                s = s_end;
                let tail = |x: &Array| own(&x.index(cko..));
                st.bank = Some(Bank {
                    kq: ops::stack(&[&k1.index(cko..), &q1.index(cko..)], -1)?,
                    v: tail(&v)?,
                    e: tail(&e)?,
                    b: tail(&bt)?,
                    dot: tail(&dot)?,
                });
                o = y;
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
    fn slots(&self) -> usize {
        self.slots
    }
    fn max_ctx(&self) -> usize {
        self.max_ctx
    }
    fn max_tok(&self) -> usize {
        MAX_TOK
    }
    fn max_out(&self) -> usize {
        MAX_OUT
    }
    fn spec_tok(&self) -> usize {
        SPEC_TOK
    }
    fn vocab(&self) -> usize {
        self.w.d.vocab
    }

    fn step(&mut self, toks: &[Tok], out: &[usize], marks: Option<&[Vec<u32>]>) -> Result<()> {
        ensure!(
            (1..=MAX_TOK).contains(&toks.len()) && out.len() <= MAX_OUT,
            "step of {} tokens, {} rows",
            toks.len(),
            out.len()
        );
        ensure!(out.iter().all(|&r| r < toks.len()), "an output row beyond the step");
        let mut segs = self.segments(toks)?;
        ensure!(marks.is_none_or(|m| m.len() == segs.len()), "marks per segment");
        for (k, g) in segs.iter_mut().enumerate() {
            let s = g.slot;
            if g.pos0 == 0 {
                self.reset(s);
            }
            let (end, len) = (self.end[s], g.b - g.a);
            if g.pos0 != end {
                ensure!(
                    self.ck[s] <= g.pos0 && g.pos0 < end,
                    "slot {s} restarts at {}: its last step ended at {end}, rollback reaches back to {}",
                    g.pos0,
                    self.ck[s]
                );
                g.replay = Some(g.pos0 - self.ck[s]);
            }
            g.hfrom = (g.pos0 as isize - 3 - self.hpos[s]) as usize;
            let new_end = g.pos0 + len;
            g.ck = g.pos0.max(new_end.saturating_sub(KEEP));
            g.hpos = (g.pos0 as isize - 3).max(new_end as isize - KEEP as isize - 3);
            self.reserve(s, new_end)?;
            let own: Vec<u32>;
            let marked = match marks {
                Some(m) => &m[k],
                None => {
                    own = toks[g.a..g.b].iter().map(|t| t.id).collect();
                    &own
                }
            };
            for &id in marked {
                ensure!((id as usize) < self.w.d.vocab, "mark token {id} out of range");
                self.seen[s][id as usize / 64] |= 1 << (id % 64);
            }
        }
        let committed: Vec<u32> = marks.map_or_else(|| toks.iter().map(|t| t.id).collect(), <[_]>::concat);
        self.w.learn_drafts(&committed)?;
        let logits = self.forward(toks, &segs, out)?;
        // Commit logits and all lazy state writes together; no graph may span engine steps.
        transforms::eval(
            logits.iter().chain(self.hidden.iter()).chain(kv_arrays(&self.kv)).chain(delta_arrays(&self.delta)),
        )?;
        for g in &segs {
            self.end[g.slot] = g.pos0 + g.b - g.a;
            self.ck[g.slot] = g.ck;
            self.hpos[g.slot] = g.hpos;
        }
        self.rows = out
            .iter()
            .map(|&r| {
                let g = segs.iter().find(|g| (g.a..g.b).contains(&r)).unwrap();
                (g.slot, g.a, r)
            })
            .collect();
        self.ids = toks.iter().map(|t| t.id).collect();
        self.logits = logits;
        Ok(())
    }

    fn logits(&self) -> Result<Vec<f32>> {
        if self.rows.is_empty() {
            return Ok(Vec::new());
        }
        w::host(self.logits.as_ref().context("no step yet")?)
    }

    fn logits_row(&self, j: usize) -> Result<Vec<f32>> {
        w::host(&self.logits.as_ref().context("no step yet")?.index(j as i32))
    }

    fn argmax(&self) -> Result<Vec<u32>> {
        if self.rows.is_empty() {
            return Ok(Vec::new());
        }
        let a = argmax_axis(self.logits.as_ref().context("no step yet")?, -1, false)?.as_dtype(Dtype::Uint32)?;
        a.eval()?;
        Ok(a.as_slice::<u32>().to_vec())
    }

    fn set_outputs(&mut self, slot: u32, words: &[u32]) -> Result<()> {
        ensure!(
            (slot as usize) < self.slots && words.len() == self.w.d.vocab / 32,
            "outputs of slot {slot}: {} words",
            words.len()
        );
        self.outs[slot as usize].copy_from_slice(words);
        Ok(())
    }

    fn sample(&mut self, params: &[[f32; 8]]) -> Result<Vec<(u32, bool)>> {
        ensure!(params.len() == self.rows.len(), "{} sampling rows for {} logit rows", params.len(), self.rows.len());
        if params.is_empty() {
            return Ok(Vec::new());
        }
        let all = self.logits()?;
        let vocab = self.w.d.vocab;
        Ok(self
            .rows
            .iter()
            .zip(params)
            .enumerate()
            .map(|(j, (&(slot, first, tok), p))| {
                let step = |id: u32| self.ids[first + 1..=tok].contains(&id);
                let seen = |id: u32| self.is_seen(slot, id) || step(id);
                let generated = |id: u32| self.is_output(slot, id) || step(id);
                if p[6] <= -2.0 {
                    let mut row = *p;
                    row[6] = -p[6] - 2.0;
                    let t = tok - first;
                    sample_row_q(&all[j * vocab..(j + 1) * vocab], seen, generated, &row, &self.draft_dist[slot][t].0)
                } else {
                    sample_row(&all[j * vocab..(j + 1) * vocab], seen, generated, p)
                }
            })
            .collect())
    }

    fn has_draft(&self) -> bool {
        self.w.mtp.is_some()
    }
    fn draft_capacity(&self) -> usize {
        KEEP.min(SPEC_TOK.saturating_sub(1))
    }

    fn draft(
        &mut self,
        toks: &[Tok],
        rows: &[usize],
        out: &[usize],
        k: usize,
        sampling: &[Option<DraftSampling>],
    ) -> Result<Vec<Vec<u32>>> {
        ensure!(
            toks.len() == rows.len() && !toks.is_empty() && toks.len() <= MAX_TOK,
            "MTP tokens/hidden rows mismatch"
        );
        ensure!(k <= KEEP && out.len() <= MAX_OUT && out.iter().all(|&r| r < toks.len()), "MTP draft shape");
        ensure!(sampling.is_empty() || sampling.len() == out.len(), "MTP sampling rows mismatch");
        let hidden = self.hidden.as_ref().context("MTP needs a target step")?;
        ensure!(rows.iter().all(|&r| r < hidden.shape()[0] as usize), "MTP hidden row beyond target step");
        // Match qwenny's MTP reference: the backbone's final hidden state, after its final norm.
        let h = w::rms(&take_rows(hidden, rows.iter().copied())?, &self.w.final_norm)?;
        let mut chains = vec![Vec::new(); out.len()];
        for &r in out {
            self.draft_dist[toks[r].slot as usize].clear();
        }
        let (mut h, mut logits) = self.mtp_forward(toks, &h, if k == 0 { &[] } else { out })?;
        let mut cur: Vec<Tok> = out.iter().map(|&r| toks[r]).collect();
        for t in 0..k {
            if out.is_empty() {
                break;
            }
            let l = logits.take().context("no MTP logits")?;
            let ids = &self.w.mtp.as_ref().unwrap().ids;
            ensure!(l.shape()[1] as usize == ids.len(), "MTP head and draft vocabulary sizes differ");
            let greedy = argmax_axis(&l, -1, false)?.as_dtype(Dtype::Uint32)?;
            greedy.eval()?;
            let am = greedy.as_slice::<u32>();
            for j in 0..cur.len() {
                let slot = cur[j].slot as usize;
                let id = if let Some(s) = sampling.get(j).and_then(Option::as_ref) {
                    let log = w::host(&l.index(j as i32))?;
                    // Expand the fixed head into token-id order: ties and penalty membership use real ids.
                    let mut full = vec![f32::NEG_INFINITY; self.w.d.vocab];
                    for (&id, &v) in ids.iter().zip(&log) {
                        full[id as usize] = v;
                    }
                    let seen = |id: u32| {
                        self.is_seen(slot, id)
                            || toks.iter().any(|x| x.slot as usize == slot && x.id == id)
                            || chains[j].contains(&id)
                    };
                    let generated = |id: u32| self.is_output(slot, id) || chains[j].contains(&id);
                    let (p, kept) = kept_dist(&full, &seen, &generated, &s.row);
                    let u = s.u.get(t).copied().unwrap_or(0.5);
                    let mut cum = 0.0;
                    let at = p
                        .iter()
                        .position(|&v| {
                            cum += v;
                            cum >= u
                        })
                        .unwrap_or(p.len() - 1);
                    let id = kept[at];
                    let mut q: Vec<_> = kept.into_iter().zip(p).collect();
                    q.sort_by_key(|x| x.0);
                    self.draft_dist[slot].push((q, id));
                    id
                } else {
                    ids[am[j] as usize]
                };
                chains[j].push(id);
                cur[j].id = id;
                cur[j].pos += 1;
            }
            if t + 1 < k {
                if t == 0 {
                    h = take_rows(&h, out.iter().copied())?;
                }
                let hnorm = w::rms(&h, &self.w.mtp.as_ref().unwrap().norm)?;
                let all: Vec<usize> = (0..cur.len()).collect();
                (h, logits) = self.mtp_forward(&cur, &hnorm, &all)?;
            }
        }
        Ok(chains)
    }

    fn draft_q(&self, slot: u32, t: usize) -> Result<(Vec<(u32, f32)>, u32)> {
        self.draft_dist.get(slot as usize).and_then(|q| q.get(t)).cloned().context("no sampled MTP draft")
    }

    fn release(&mut self, slot: u32) -> Result<()> {
        ensure!((slot as usize) < self.slots, "slot {slot} out of range");
        let s = slot as usize;
        self.reset(s);
        self.kv[s].clear();
        self.cap[s] = 0;
        Ok(())
    }

    fn warmup(&mut self, draft_depth: usize) -> Result<()> {
        if self.max_ctx < 4 {
            return Ok(());
        }
        // Empty marks: these synthetic inputs never affect the learned vocabulary or penalties.
        let marks = vec![Vec::new()];
        self.step(&[Tok { id: 0, slot: 0, pos: 0 }], &[0], Some(&marks))?;
        self.step(&[Tok { id: 0, slot: 0, pos: 1 }], &[0], Some(&marks))?;
        let n = (draft_depth + 1).min(self.max_ctx - 2).min(SPEC_TOK);
        let toks: Vec<Tok> = (0..n).map(|i| Tok { id: 0, slot: 0, pos: 2 + i as u32 }).collect();
        let rows: Vec<usize> = (0..n).collect();
        self.step(&toks, &rows, Some(&marks))?;
        if draft_depth > 0 && self.has_draft() && 2 + n + draft_depth < self.max_ctx {
            let sampling =
                DraftSampling { row: [1.0, 20.0, 0.95, 1.0, 0.0, 0.0, -1.0, 0.0], u: vec![0.5; draft_depth] };
            self.draft(&toks, &rows, &[n - 1], draft_depth, &[Some(sampling)])?;
        }
        self.release(0)?;
        self.logits = None;
        self.hidden = None;
        self.rows.clear();
        self.ids.clear();
        Ok(())
    }

    fn sync(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "QWENNY_TEST_MTP_MODEL=<dir>: adaptive head growth, replacement and verification on a model with int4 MTP"]
    fn adaptive_probe() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var("QWENNY_TEST_MTP_MODEL")?);
        let mut m = Model::load(&dir, 8192, 1, true)?;
        let seed = m.w.mtp.as_ref().unwrap().ids.clone();
        ensure!(m.w.mtp.as_ref().unwrap().adaptive.is_some(), "adaptive head disabled");
        let pt: Vec<Tok> = seed[..8].iter().enumerate().map(|(i, &id)| Tok { id, slot: 0, pos: i as u32 }).collect();
        m.run(&pt, &[7])?;
        let ds = DraftSampling { row: [1.0, 20.0, 0.95, 1.0, 0.0, 0.0, -1.0, 0.0], u: vec![0.25] };
        m.draft(&[Tok { id: seed[1], slot: 0, pos: 7 }], &[7], &[0], 1, &[Some(ds)])?;
        let before_q = m.draft_q(0, 0)?;
        let vocab = m.vocab();
        let missing = |m: &Model, count: usize| -> Vec<u32> {
            let mut have = vec![false; vocab];
            for &id in &m.w.mtp.as_ref().unwrap().ids {
                have[id as usize] = true;
            }
            (0..vocab as u32).filter(|&id| !have[id as usize]).take(count).collect()
        };
        let verify_rows = |m: &Model, rows: &[usize]| -> Result<()> {
            let head = m.w.mtp.as_ref().unwrap();
            let ids: Vec<u32> = rows.iter().map(|&r| head.ids[r]).collect();
            let idx = Array::from_slice(&ids, &[ids.len() as i32]).as_dtype(Dtype::Int32)?;
            let r: Vec<i32> = rows.iter().map(|&r| r as i32).collect();
            let ridx = Array::from_slice(&r, &[r.len() as i32]);
            for (src, draft) in
                m.w.head.as_ref().unwrap().arrays().into_iter().zip(head.head.as_ref().unwrap().arrays())
            {
                let equal = ops::all(ops::eq(src.take_axis(&idx, 0)?, draft.take_axis(&ridx, 0)?)?, None)?;
                equal.eval()?;
                ensure!(equal.item_exact::<bool>(), "draft row differs from the stored target output matrix");
            }
            Ok(())
        };
        let new = missing(&m, 16);
        m.w.learn_drafts(&new[..15])?;
        assert_eq!(m.w.mtp.as_ref().unwrap().ids.len(), seed.len());
        m.w.learn_drafts(&new[15..])?;
        assert_eq!(&m.w.mtp.as_ref().unwrap().ids[seed.len()..], &new);
        verify_rows(&m, &(seed.len()..seed.len() + 16).collect::<Vec<_>>())?;
        assert_eq!(m.draft_q(0, 0)?, before_q, "head growth changed a distribution already drawn");
        let remaining = crate::draft_vocab::MAX_ROWS - m.w.mtp.as_ref().unwrap().ids.len();
        let extra = missing(&m, remaining);
        m.w.learn_drafts(&extra)?;
        assert_eq!(m.w.mtp.as_ref().unwrap().ids.len(), crate::draft_vocab::MAX_ROWS);
        for _ in 0..3 {
            m.w.learn_drafts(&seed[..16])?;
        }
        let replace = missing(&m, 16);
        m.w.learn_drafts(&replace)?;
        let head = m.w.mtp.as_ref().unwrap();
        for id in &seed[..16] {
            assert!(head.ids.contains(id), "a hot row was evicted");
        }
        let changed: Vec<usize> = replace.iter().map(|id| head.ids.iter().position(|v| v == id).unwrap()).collect();
        verify_rows(&m, &changed)?;
        assert_eq!(m.draft_q(0, 0)?, before_q, "replacement changed an existing sampled distribution");
        let saved_ids = m.w.mtp.as_ref().unwrap().ids.clone();
        drop(m);
        let restored = crate::draft_vocab::Adaptive::load(&dir, vocab, &seed)?;
        assert_eq!(restored.ids(), saved_ids, "learned vocabulary did not survive reload");
        println!(
            "adaptive head: 16-row growth, packed target rows, cold replacement, hot-row retention, old q and reload OK"
        );
        Ok(())
    }
}
