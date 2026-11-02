//! Qwen inference on MLX. A step's tokens share their matrix products; attention and DeltaNet run per contiguous
//! sequence segment against that slot's state. Drafting is disabled (`spec_tok == 1`) because a rejected DeltaNet
//! token requires recurrent-state rollback, not merely overwritten KV entries. Sampling is the shared host sampler.
use super::weights::{self as w, CONV_K, HD, Layer, LKD, LK_HEADS, LVD, MResult, NKV, ROT, THETA, Weights};
use crate::engine::{Engine, Tok};
use crate::sample::sample_row;
use anyhow::{Context, Result, ensure};
use mlx_rs::fast::ScaledDotProductAttentionMask;
use mlx_rs::ops::indexing::{IndexOp, TryIndexMutOp, argmax_axis};
use mlx_rs::{Array, Dtype, fast, nn, ops, transforms};
use std::path::Path;

/// Bound the pure-operations DeltaNet recurrence's graph size. The Metal recurrent kernel can raise this later.
const MAX_TOK: usize = 64;
const MAX_OUT: usize = 64;
const KV_CHUNK: usize = 1024;

struct Segment {
    a: usize,
    b: usize,
    slot: usize,
    pos0: usize,
}

pub struct Model {
    w: Weights,
    slots: usize,
    max_ctx: usize,
    /// per slot, per attention layer: keys and values [NKV][capacity][HD], f32
    kv: Vec<Vec<(Array, Array)>>,
    cap: Vec<usize>,
    /// per slot, per DeltaNet layer: the last three raw qkv rows [3][C]
    conv: Vec<Vec<Array>>,
    /// per slot, per DeltaNet layer: [Hv][Dv][Dk], f32
    state: Vec<Vec<Array>>,
    /// Logical end of each slot. Backward restarts are rejected until recurrent rollback is implemented.
    end: Vec<usize>,
    seen: Vec<Vec<u64>>,
    outs: Vec<Vec<u32>>,
    logits: Option<Array>,
    /// per logit row: (slot, first token of its segment, its token)
    rows: Vec<(usize, usize, usize)>,
    ids: Vec<u32>,
}

impl Model {
    pub fn load(dir: &Path, max_ctx: usize, slots: usize, int8: bool) -> Result<Self> {
        ensure!(max_ctx > 0 && slots > 0, "context and slots must be positive");
        w::init()?;
        let w = Weights::load(dir, Dtype::Float16, int8)?;
        let (n_attn, n_delta) = w.counts();
        let d = w.d;
        let kv = (0..slots).map(|_| Vec::with_capacity(n_attn)).collect();
        let conv = (0..slots).map(|_| {
            (0..n_delta).map(|_| ops::zeros::<f32>(&[CONV_K - 1, d.conv_width()])).collect::<MResult<Vec<_>>>()
        }).collect::<MResult<Vec<_>>>()?;
        let state = (0..slots).map(|_| {
            (0..n_delta).map(|_| ops::zeros::<f32>(&[d.lv_heads, LVD, LKD])).collect::<MResult<Vec<_>>>()
        }).collect::<MResult<Vec<_>>>()?;
        let vocab = d.vocab;
        Ok(Model { w, slots, max_ctx, kv, cap: vec![0; slots], conv, state, end: vec![0; slots],
                   seen: vec![vec![0; vocab.div_ceil(64)]; slots], outs: vec![vec![0; vocab / 32]; slots],
                   logits: None, rows: Vec::new(), ids: Vec::new() })
    }

    fn reserve(&mut self, slot: usize, end: usize) -> Result<()> {
        if end <= self.cap[slot] {
            return Ok(());
        }
        let cap = end.next_multiple_of(KV_CHUNK).min(self.max_ctx);
        ensure!(end <= cap, "position {end} beyond the context of {}", self.max_ctx);
        let zeros = |n: usize| ops::zeros::<f32>(&[NKV, n as i32, HD]);
        if self.cap[slot] == 0 {
            let n_attn = self.w.counts().0;
            self.kv[slot] = (0..n_attn).map(|_| Ok((zeros(cap)?, zeros(cap)?))).collect::<MResult<_>>()?;
        } else {
            let more = zeros(cap - self.cap[slot])?;
            for (k, v) in self.kv[slot].iter_mut() {
                *k = ops::concatenate(&[&*k, &more], 1)?;
                *v = ops::concatenate(&[&*v, &more], 1)?;
            }
        }
        self.cap[slot] = cap;
        Ok(())
    }

    fn reset(&mut self, slot: usize) -> MResult<()> {
        for h in self.conv[slot].iter_mut() {
            *h = ops::zeros::<f32>(h.shape())?;
        }
        for s in self.state[slot].iter_mut() {
            *s = ops::zeros::<f32>(s.shape())?;
        }
        self.seen[slot].fill(0);
        Ok(())
    }

    fn forward(&mut self, toks: &[Tok], segs: &[Segment], out: &[usize]) -> MResult<Option<Array>> {
        let t = toks.len() as i32;
        let ids: Vec<i32> = toks.iter().map(|x| x.id as i32).collect();
        let h = self.w.embed.take_axis(&Array::from_slice(&ids, &[t]), 0)?.as_dtype(Dtype::Float32)?;
        let pos: Vec<i32> = toks.iter().map(|x| x.pos as i32).collect();
        let rope = rope_tables(&pos)?;
        let mut cache = Cache { kv: &mut self.kv, conv: &mut self.conv, state: &mut self.state, segs, rope,
                                masks: masks(segs)?, d: self.w.d };
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
    kv: &'a mut [Vec<(Array, Array)>],
    conv: &'a mut [Vec<Array>],
    state: &'a mut [Vec<Array>],
    segs: &'a [Segment],
    rope: [(Array, Array); 2],
    masks: Vec<Option<Array>>,
    d: w::Dims,
}

impl w::Mix for Cache<'_> {
    fn rope(&self) -> (Array, Array) {
        self.rope[0].clone()
    }

    fn attend(&mut self, ai: usize, q: &Array, k: &Array, v: &Array) -> MResult<Array> {
        // The CUDA cache's effective values: int8 q = round(x * 127 / max|x|), read as q * max|x| / 127.
        // Keep the dequantized values in f32 for now; the scales and quantized rows are exactly the cache's values.
        let k = quantize_kv(k)?;
        let v = quantize_kv(v)?;
        let (k, v) = (&k, &v);
        let mut parts = Vec::with_capacity(self.segs.len());
        for (g, mask) in self.segs.iter().zip(&self.masks) {
            let (a, b, len) = (g.a as i32, g.b as i32, (g.b - g.a) as i32);
            let heads = |x: &Array| ops::swap_axes(&x.index(a..b), 0, 1);
            let end = g.pos0 + len as usize;
            let (kc, vc) = &mut self.kv[g.slot][ai];
            kc.try_index_mut((.., g.pos0 as i32..end as i32, ..), &heads(k)?)?;
            vc.try_index_mut((.., g.pos0 as i32..end as i32, ..), &heads(v)?)?;
            let keys = ops::expand_dims(kc.index((.., ..end as i32, ..)), 0)?;
            let values = ops::expand_dims(vc.index((.., ..end as i32, ..)), 0)?;
            let mask = mask.as_ref().map(ScaledDotProductAttentionMask::Array);
            let o = fast::scaled_dot_product_attention(ops::expand_dims(heads(q)?, 0)?, keys, values,
                                                       1.0 / (HD as f32).sqrt(), mask, None)?;
            parts.push(ops::swap_axes(&o.reshape(&[self.d.nh, len, HD])?, 0, 1)?);
        }
        ops::concatenate(&parts, 0)
    }

    fn delta(&mut self, di: usize, layer: &Layer, qkv: &Array, z: &Array, ba: &Array) -> MResult<Array> {
        let d = self.d;
        let (c, lv) = (d.conv_width(), d.lv_heads);
        let ba = ops::split_equal(ba, 2, -1)?;
        let beta = nn::sigmoid(&ba[0])?;
        let gates = ops::add(&ba[1], layer.dt_bias.as_ref().unwrap())?;
        let decay = ops::exp(ops::negative(ops::multiply(ops::exp(layer.a_log.as_ref().unwrap())?, nn::softplus(gates)?)?)?)?;
        let mut parts = Vec::with_capacity(self.segs.len());
        for g in self.segs {
            let (a, b, len) = (g.a as i32, g.b as i32, (g.b - g.a) as i32);
            let raw = qkv.index(a..b);
            // Three previous raw rows followed by this segment. A new sequence's history was zeroed at step entry.
            let ext = ops::concatenate(&[&self.conv[g.slot][di], &raw], 0)?;
            let cw = layer.conv_w.as_ref().unwrap();
            let mut cv = ops::multiply(ext.index(..len), cw.index((.., 0)))?;
            for j in 1..CONV_K {
                cv = ops::add(&cv, ops::multiply(ext.index(j..j + len), cw.index((.., j)))?)?;
            }
            self.conv[g.slot][di] = ext.index(ext.shape()[0] - (CONV_K - 1)..);
            let cv = nn::silu(cv)?;
            let qkvw = ops::split_at_indices(&cv, &[d.key_width(), 2 * d.key_width()], -1)?;
            let q = l2(&qkvw[0].reshape(&[len, LK_HEADS, LKD])?)?;
            let k = l2(&qkvw[1].reshape(&[len, LK_HEADS, LKD])?)?;
            let q = ops::multiply(q, Array::from_f32(1.0 / (LKD as f32).sqrt()))?;
            let v = qkvw[2].reshape(&[len, lv, LVD])?;
            // Every value head reads its assigned key head (adjacent groups, not a tiled repeat of the whole list).
            let repeat = |x: &Array| -> MResult<Array> {
                let group = lv / LK_HEADS;
                ops::broadcast_to(ops::expand_dims(x, 2)?, &[len, LK_HEADS, group, LKD])?.reshape(&[len, lv, LKD])
            };
            let (q, k) = (repeat(&q)?, repeat(&k)?);
            let mut state = self.state[g.slot][di].clone();
            let mut ys = Vec::with_capacity(len as usize);
            for t in 0..len {
                let kt = k.index(t).reshape(&[lv, 1, LKD])?;
                let qt = q.index(t).reshape(&[lv, 1, LKD])?;
                let vt = v.index(t);
                let bt = beta.index(g.a as usize as i32 + t).reshape(&[lv, 1])?;
                let gt = decay.index(g.a as usize as i32 + t).reshape(&[lv, 1, 1])?;
                let old = ops::multiply(&state, &gt)?;
                let prediction = ops::sum_axis(ops::multiply(&old, &kt)?, 2, false)?;
                let delta = ops::multiply(ops::subtract(&vt, &prediction)?, &bt)?;
                state = ops::add(&old, ops::multiply(&kt, &delta.reshape(&[lv, LVD, 1])?)?)?;
                ys.push(ops::sum_axis(ops::multiply(&state, &qt)?, 2, false)?);
            }
            self.state[g.slot][di] = state;
            let y = ops::stack(&ys, 0)?;
            let yn = fast::rms_norm(&y, Some(layer.delta_norm.as_ref().unwrap()), w::EPS)?;
            let zz = z.index(a..b).reshape(&[len, lv, LVD])?;
            parts.push(ops::multiply(&yn, nn::silu(zz)?)?);
        }
        ops::concatenate(&parts, 0)?.reshape(&[qkv.shape()[0], d.value_width()])
    }
}

fn quantize_kv(x: &Array) -> MResult<Array> {
    let mx = ops::max_axis(ops::abs(x)?, -1, true)?;
    let scale = ops::divide(&mx, Array::from_f32(127.0))?;
    let inv = ops::select(ops::gt(&mx, Array::from_f32(0.0))?,
                          ops::divide(Array::from_f32(127.0), &mx)?, Array::from_f32(0.0))?;
    let q = ops::round(ops::multiply(x, inv)?, None)?;
    ops::multiply(&q, &scale)
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
    /// One token per decoding request: no n-gram or MTP drafts until DeltaNet rollback exists.
    fn spec_tok(&self) -> usize { 1 }
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
                _ => segs.push(Segment { a: i, b: i + 1, slot: t.slot as usize, pos0: t.pos as usize }),
            }
        }
        let mut ends = self.end.clone();
        for (k, g) in segs.iter().enumerate() {
            if g.pos0 == 0 {
                self.reset(g.slot)?;
                ends[g.slot] = 0;
            }
            ensure!(g.pos0 == ends[g.slot], "slot {} restarts at {}, after {}: recurrent rollback is not available on MLX",
                    g.slot, g.pos0, ends[g.slot]);
            self.reserve(g.slot, g.pos0 + g.b - g.a)?;
            for &id in marks.and_then(|m| m.get(k)).into_iter().flatten() {
                ensure!((id as usize) < self.w.d.vocab, "mark token {id} out of range");
                self.seen[g.slot][id as usize / 64] |= 1 << (id % 64);
            }
            ends[g.slot] = g.pos0 + g.b - g.a;
        }
        let logits = self.forward(toks, &segs, out)?;
        // Commit logits and all lazy state writes together; no graph may span engine steps.
        let state = self.kv.iter().flatten().flat_map(|(k, v)| [k, v])
            .chain(self.conv.iter().flatten()).chain(self.state.iter().flatten());
        transforms::eval(logits.iter().chain(state))?;
        self.end = ends;
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
        self.end[slot as usize] = 0;
        Ok(())
    }

    fn sync(&self) -> Result<()> { Ok(()) }
}
