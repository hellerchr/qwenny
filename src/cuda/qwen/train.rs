//! LoRA training of Qwen3.5 on CUDA: every linear layer of the language model (Unsloth's targets: q/k/v/o_proj,
//! in_proj_qkv/z/b/a, out_proj, gate/up/down_proj), packed documents, cuBLAS products (crate::cuda::learn), the
//! element-wise parts in kernels/qwen_train.cu. Only the residual stream before each layer is kept: the backward pass
//! of a layer recomputes its forward first (gradient checkpointing per layer), so activation memory is one layer's.
//! The Gated DeltaNet's backward walks each document backwards from state checkpoints every C_CK tokens that the
//! recomputed forward writes. The loss and the output layer run on the loss rows only.
//!
//! The frozen weights stay as the checkpoint stores them: 16-bit, or int4 / int8 (compressed-tensors, e.g. the
//! AutoRound Qwen3.8-27B), dequantized into one scratch matrix just before each product (`Weight`); the output layer
//! in slices of vocabulary rows. The embedding stays in the checkpoint on the host: a micro-batch's rows are read from
//! it and uploaded.
use super::{Dims, EPS, HD, LK_HEADS, LKD, LVD, NKV, attn, norm};
use crate::checkpoint::Checkpoint;
use crate::cuda::learn::{Arg::{F, I, P}, Buf, Group, Layout, Learn, Scratch};
use crate::training::{Micro, Opts, f32_bytes};
use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::LaunchConfig;
use std::{collections::HashMap, fs, path::Path};

mod int8;

const PTX_4B: [&str; 3] = [include_str!(concat!(env!("OUT_DIR"), "/qwen4b_train_bf16.ptx")),
                           include_str!(concat!(env!("OUT_DIR"), "/qwen4b_train_f32.ptx")),
                           include_str!(concat!(env!("OUT_DIR"), "/qwen4b_train_f16.ptx"))];
const PTX_9B: [&str; 3] = [include_str!(concat!(env!("OUT_DIR"), "/qwen9b_train_bf16.ptx")),
                           include_str!(concat!(env!("OUT_DIR"), "/qwen9b_train_f32.ptx")),
                           include_str!(concat!(env!("OUT_DIR"), "/qwen9b_train_f16.ptx"))];
const PTX_27B: [&str; 3] = [include_str!(concat!(env!("OUT_DIR"), "/qwen27b_train_bf16.ptx")),
                            include_str!(concat!(env!("OUT_DIR"), "/qwen27b_train_f32.ptx")),
                            include_str!(concat!(env!("OUT_DIR"), "/qwen27b_train_f16.ptx"))];

const KERNELS: [&str; 23] = [
    "dequant", "rms_fwd", "add_rms", "rms_bwd_pre", "rms_bwd_final", "swiglu_fwd", "swiglu_bwd", "scatter_rows",
    "scatter_rows_scaled",
    "cross_entropy", "dn_conv", "dn_rule", "dn_rule_bwd", "dn_norm", "dn_norm_bwd", "dn_conv_bwd", "dn_conv_in_bwd",
    "at_prep", "at_prep_bwd", "at_softmax", "at_softmax_bwd", "at_gate", "at_gate_bwd",
];

/// Rows of the output layer per matrix product (logits f32 and their gradient)
const CHUNK: usize = 256;
/// DeltaNet state checkpoints every C_CK tokens, value columns per dn_rule block (C_CK, DCPB in the kernels)
const C_CK: usize = 8;
const NSLICE: usize = 8;
/// Value heads per block of dn_norm (NORM_HEADS in the kernels)
const NORM_HEADS: usize = 16;
/// LoRA U columns per group (in-projection up to 4 modules, out-projection, gate/up, down) in units of the rank
const U_COLS: [usize; 4] = [0, 4, 5, 7];
const U_RANKS: usize = 8;

/// Rows of a frozen matrix on the device: act values, or as the checkpoint stores them quantized (`words` u32 per
/// row, an f32 scale and a u8 zero point per row and group)
enum Part {
    Dense { w: Buf, rows: usize },
    Quant { q: Buf, sc: Buf, zp: Buf, rows: usize, words: usize, group: usize, bits: usize },
}

impl Part {
    fn rows(&self) -> usize {
        match self {
            Part::Dense { rows, .. } | Part::Quant { rows, .. } => *rows,
        }
    }
}

/// A frozen matrix [rows][cols], stacked from checkpoint matrices: one dense part is used in place, anything else is
/// put together in the scratch for each product (Trainer::wrows).
struct Weight {
    parts: Vec<Part>,
    rows: usize,
    cols: usize,
}

impl Weight {
    fn in_place(&self) -> bool {
        matches!(self.parts[..], [Part::Dense { .. }])
    }
}

struct Layer {
    delta: bool,
    /// [in_proj_qkv; in_proj_z; in_proj_b; in_proj_a] or [q_proj; k_proj; v_proj]
    w_in: Weight,
    in_width: usize,
    /// out_proj or o_proj [H][MIX]
    w_out: Weight,
    /// [gate_proj; up_proj] [2FF][H], down_proj [H][FF]
    w13: Weight,
    w_down: Weight,
    /// 1 + w (f32)
    in_norm: Buf,
    post_norm: Buf,
    /// DeltaNet: conv1d weight [LQKV][4], A_log, dt_bias, gated norm weight (plain); attention: q_norm, k_norm (1 + w)
    f: [Option<Buf>; 4],
    g: [Group; 4],
}

pub struct Trainer {
    l: Learn,
    d: Dims,
    tmax: usize,
    lmax: usize,
    /// the checkpoint: the embedding is read from it per micro-batch
    w: Checkpoint,
    embed_name: String,
    /// the output layer (the embedding when tied)
    lm_head: Weight,
    /// dequantized weights of one product: `scr_rows` rows of the widest matrix
    scr: Buf,
    scr_elems: usize,
    final_norm: Buf,
    inv_freq: Buf,
    layers: Vec<Layer>,
    /// residual stream before each layer and after the last: [layers + 1][T][H] f32
    hs: Buf,
    rs_in: Buf,
    rs_mid: Buf,
    rs_fin: Buf,
    xn: Buf,
    xn2: Buf,
    /// in-projection output and its gradient [T][in_width]
    p: Buf,
    dp: Buf,
    /// DeltaNet: conv output [T][LQKV] (f32; da in the backward), gates [T][2 lv_heads], raw output / its gradient
    /// [T][MIX], state checkpoints, gradients of q and k, v, the gates
    cv: Buf,
    gb: Buf,
    o: Buf,
    ck: Buf,
    ck_rows: usize,
    dqk: Buf,
    dvv: Buf,
    dgb: Buf,
    /// attention: head-major Q, K, V, O and their gradients; scores of one kv group
    qa: Buf,
    ka: Buf,
    va: Buf,
    oa: Buf,
    dqa: Buf,
    dka: Buf,
    dva: Buf,
    doa: Buf,
    att_s: Buf,
    att_p: Buf,
    att_dp: Buf,
    att_ds: Buf,
    /// mixer output (out-projection input) and its gradient [T][MIX]
    y: Buf,
    dy: Buf,
    s_op: Buf,
    hmid: Buf,
    w13: Buf,
    mid: Buf,
    dmid: Buf,
    u: Buf,
    dx: Buf,
    dh: Buf,
    dact: Buf,
    meta: [Buf; 5], // ids, pos, dstart, dend, dlen
    docs: [Buf; 3], // start, len, first checkpoint
    rows: Buf,
    targets: Buf,
    gath: Buf,
    logits: Buf,
    dlog: Buf,
    loss_row: Buf,
    correct: Buf,
    /// per loss row: the weight of its loss (pass_weighted)
    row_w: Buf,
}

fn cfg3(grid: (usize, usize, usize), block: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (grid.0 as u32, grid.1 as u32, grid.2 as u32), block_dim: (block as u32, 1, 1), shared_mem_bytes: 0 }
}

impl Trainer {
    /// `tmax`: most tokens of a micro-batch, `lmax`: the longest document.
    pub fn new(dir: &Path, o: Opts, tmax: usize, lmax: usize, seed: u64) -> Result<Self> {
        ensure!(!(o.f32 && o.f16), "--f32 and --fast16 exclude each other");
        ensure!(o.rank <= 16, "rank above 16 does not fit the U columns");
        let config: serde_json::Value = serde_json::from_str(&fs::read_to_string(dir.join("config.json"))?)?;
        let tc = config.get("text_config").unwrap_or(&config);
        let d = Dims::parse(tc)?;
        let ptx = match d.h {
            2560 => PTX_4B,
            4096 => PTX_9B,
            5120 => PTX_27B,
            _ => bail!("training kernels are built for Qwen3.5-4B, Qwen3.5-9B and Qwen3.8-27B"),
        };
        let (s, module) = crate::cuda::core::open(ptx[if o.f32 { 1 } else if o.f16 { 2 } else { 0 }])?;
        let es = if o.f32 { 4 } else { 2 };
        let alloc = |bytes: usize| crate::cuda::learn::alloc(&s, bytes);
        let upload = |bytes: &[u8]| crate::cuda::learn::upload(&s, bytes);
        let w = Checkpoint::open(dir)?;
        // DeltaNet's in-projection columns: [in_proj_qkv; in_proj_z; in_proj_b; in_proj_a] (DELTA_IN of the training
        // kernels; the inference engine runs in_proj_b/a as a product of their own)
        let delta_in = d.delta_in() + 2 * d.lv_heads;
        // 16-bit checkpoints too large for the GPU (Qwen3.5-9B) are held as int8 (rows.rs)
        let types = tc["layer_types"].as_array().context("layer_types")?;
        let layer_params: usize = types.iter().map(|t| d.h * (3 * d.ff + d.mix() + match t.as_str() {
            Some("linear_attention") => delta_in,
            _ => d.attn_in(),
        })).sum();
        let q8 = !w.prequantized() && (2 * layer_params) as f64 > int8::INT8_ABOVE;
        // weights stacked by rows: stored quantized with integer zero points as they are, else bf16 as stored, widened
        // for the f32 build or rounded for f16 (consecutive dense matrices in one part)
        let mat = |names: &[String]| -> Result<Weight> {
            let (mut parts, mut dense, mut dense_rows, mut rows, mut cols) = (Vec::new(), Vec::new(), 0, 0, 0);
            for n in names {
                let base = n.strip_suffix(".weight").context("a .weight")?;
                let quant = if w.is_quant(base) {
                    let q = w.quant_matrix(base)?;
                    q.zero_points().filter(|_| q.group % (32 / q.bits) == 0).map(|z| (q, z))
                } else {
                    None
                };
                if let Some((q, zp)) = quant {
                    if dense_rows > 0 {
                        parts.push(Part::Dense { w: upload(&std::mem::take(&mut dense))?, rows: std::mem::take(&mut dense_rows) });
                    }
                    let words = q.cols * q.bits / 32;
                    parts.push(Part::Quant { q: upload(&q.q.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?,
                                             sc: upload(f32_bytes(&q.scales))?, zp: upload(&zp)?, rows: q.rows, words,
                                             group: q.group, bits: q.bits });
                    rows += q.rows;
                    cols = q.cols;
                    continue;
                }
                let (b, shape) = w.bf16(n)?;
                if q8 {
                    if dense_rows > 0 {
                        parts.push(Part::Dense { w: upload(&std::mem::take(&mut dense))?, rows: std::mem::take(&mut dense_rows) });
                    }
                    let (q, sc, zp) = int8::quantize(&b, shape[1]);
                    parts.push(Part::Quant { q: upload(&q)?, sc: upload(f32_bytes(&sc))?, zp: upload(&zp)?, rows: shape[0],
                                             words: shape[1] / 4, group: int8::GROUP, bits: 8 });
                    rows += shape[0];
                    cols = shape[1];
                    continue;
                }
                if o.f32 {
                    dense.extend(b.iter().flat_map(|&x| f32::from_bits((x as u32) << 16).to_le_bytes()));
                } else if o.f16 {
                    dense.extend(b.iter().flat_map(|&x| half::f16::from_f32(f32::from_bits((x as u32) << 16)).to_le_bytes()));
                } else {
                    dense.extend(b.iter().flat_map(|x| x.to_le_bytes()));
                }
                dense_rows += shape[0];
                rows += shape[0];
                cols = shape[1];
            }
            if dense_rows > 0 {
                parts.push(Part::Dense { w: upload(&dense)?, rows: dense_rows });
            }
            Ok(Weight { parts, rows, cols })
        };
        let vec = |v: Vec<f32>| upload(f32_bytes(&v));
        let n = if o.layers > 0 { o.layers.min(types.len()) } else { types.len() };
        let mut lay = Layout::new(o.rank);
        let mut layers = Vec::new();
        for (i, kind) in types.iter().take(n).enumerate() {
            let p = format!("model.language_model.layers.{i}");
            let peft = format!("base_model.model.{p}");
            let delta = match kind.as_str() {
                Some("linear_attention") => true,
                Some("full_attention") => false,
                other => bail!("layer {i}: unknown type {other:?}"),
            };
            let (mixer, ins, out): (&str, Vec<(&str, usize)>, &str) = if delta {
                ("linear_attn", vec![("in_proj_qkv", d.lqkv()), ("in_proj_z", d.lv_heads * LVD), ("in_proj_b", d.lv_heads),
                                     ("in_proj_a", d.lv_heads)], "out_proj")
            } else {
                ("self_attn", vec![("q_proj", 2 * d.nh * HD), ("k_proj", NKV * HD), ("v_proj", NKV * HD)], "o_proj")
            };
            let g = [lay.group(&format!("{peft}.{mixer}"), d.h, &ins), lay.group(&format!("{peft}.{mixer}"), d.mix(), &[(out, d.h)]),
                     lay.group(&format!("{peft}.mlp"), d.h, &[("gate_proj", d.ff), ("up_proj", d.ff)]),
                     lay.group(&format!("{peft}.mlp"), d.ff, &[("down_proj", d.h)])];
            let fl = |n: &str| -> Result<Option<Buf>> { Ok(Some(vec(w.floats(&format!("{p}.{mixer}.{n}"))?.0)?)) };
            let f = if delta {
                [fl("conv1d.weight")?, fl("A_log")?, fl("dt_bias")?, fl("norm.weight")?]
            } else {
                [Some(vec(norm(&w, &format!("{p}.self_attn.q_norm.weight"))?)?),
                 Some(vec(norm(&w, &format!("{p}.self_attn.k_norm.weight"))?)?), None, None]
            };
            let ly = Layer {
                delta,
                w_in: mat(&ins.iter().map(|(m, _)| format!("{p}.{mixer}.{m}.weight")).collect::<Vec<_>>())?,
                in_width: if delta { delta_in } else { d.attn_in() },
                w_out: mat(&[format!("{p}.{mixer}.{out}.weight")])?,
                w13: mat(&[format!("{p}.mlp.gate_proj.weight"), format!("{p}.mlp.up_proj.weight")])?,
                w_down: mat(&[format!("{p}.mlp.down_proj.weight")])?,
                in_norm: vec(norm(&w, &format!("{p}.input_layernorm.weight"))?)?,
                post_norm: vec(norm(&w, &format!("{p}.post_attention_layernorm.weight"))?)?,
                f,
                g,
            };
            // the matrices as stacked from the checkpoint against the widths the kernels assume (a mismatch, e.g. a
            // changed Dims, would read misplaced columns without any other error)
            let shapes = [(&ly.w_in, ly.in_width, d.h), (&ly.w_out, d.h, d.mix()), (&ly.w13, 2 * d.ff, d.h), (&ly.w_down, d.h, d.ff)];
            for (wt, rows, cols) in shapes {
                ensure!(wt.rows == rows && wt.cols == cols, "layer {i}: a matrix of {} x {}, the kernels take {rows} x {cols}",
                        wt.rows, wt.cols);
            }
            layers.push(ly);
        }
        let init = lay.init(seed);
        let (h, mix, lvh) = (d.h, d.mix(), d.lv_heads);
        let in_max = delta_in.max(d.attn_in());
        let lp = lmax.next_multiple_of(8);
        let att = (d.nh / NKV) * lmax * lp;
        let e = "model.language_model.embed_tokens.weight".to_string();
        let lm_head = mat(&[if w.has("lm_head.weight") { "lm_head.weight".to_string() } else { e.clone() }])?;
        // the scratch: the widest matrix put together for a product; the output layer in slices of as many elements
        // (a layer's four matrices side by side: put together once for its recompute and backward)
        let widest = layers.iter().map(|l| [&l.w_in, &l.w_out, &l.w13, &l.w_down].iter().filter(|w| !w.in_place())
            .map(|w| w.rows * w.cols).sum::<usize>()).max().unwrap_or(0);
        let scr_elems = if lm_head.in_place() { widest } else { widest.max(32768 * d.h) };
        let ck_rows = tmax.div_ceil(C_CK);
        let inv_freq: Vec<f32> = (0..attn::ROT / 2).map(|i| attn::THETA.powf(-2.0 * i as f64 / attn::ROT as f64) as f32).collect();
        let t = Trainer {
            lm_head,
            scr: alloc(scr_elems * es)?,
            scr_elems,
            final_norm: vec(norm(&w, "model.language_model.norm.weight")?)?,
            inv_freq: vec(inv_freq)?,
            hs: alloc((n + 1) * tmax * h * 4)?,
            rs_in: alloc(tmax * 4)?,
            rs_mid: alloc(tmax * 4)?,
            rs_fin: alloc(tmax * 4)?,
            xn: alloc(tmax * h * es)?,
            xn2: alloc(tmax * h * es)?,
            p: alloc(tmax * in_max * es)?,
            dp: alloc(tmax * in_max * es)?,
            cv: alloc(tmax * d.lqkv() * 4)?,
            gb: alloc(tmax * 2 * lvh * 4)?,
            o: alloc(tmax * mix * 4)?,
            ck: alloc(ck_rows * lvh * LVD * LKD * 4)?,
            ck_rows,
            dqk: alloc(tmax * 2 * LK_HEADS * LKD * 4)?,
            dvv: alloc(tmax * mix * 4)?,
            dgb: alloc(tmax * 2 * lvh * 4)?,
            qa: alloc(tmax * mix * es)?,
            ka: alloc(tmax * NKV * HD * es)?,
            va: alloc(tmax * NKV * HD * es)?,
            oa: alloc(tmax * mix * es)?,
            dqa: alloc(tmax * mix * es)?,
            dka: alloc(tmax * NKV * HD * es)?,
            dva: alloc(tmax * NKV * HD * es)?,
            doa: alloc(tmax * mix * es)?,
            att_s: alloc(att * 4)?,
            att_p: alloc(att * es)?,
            att_dp: alloc(att * 4)?,
            att_ds: alloc(att * es)?,
            y: alloc(tmax * mix * es)?,
            dy: alloc(tmax * mix * es)?,
            s_op: alloc(tmax * h * es)?,
            hmid: alloc(tmax * h * 4)?,
            w13: alloc(tmax * 2 * d.ff * es)?,
            mid: alloc(tmax * d.ff * es)?,
            dmid: alloc(tmax * d.ff * es)?,
            u: alloc(tmax * U_RANKS * o.rank * es)?,
            dx: alloc(tmax * h * if o.f16 { 2 } else { 4 })?,
            dh: alloc(tmax * h * 4)?,
            dact: alloc(tmax * h * es)?,
            meta: [alloc(tmax * 4)?, alloc(tmax * 4)?, alloc(tmax * 4)?, alloc(tmax * 4)?, alloc(tmax * 4)?],
            docs: [alloc(tmax * 4)?, alloc(tmax * 4)?, alloc(tmax * 4)?],
            rows: alloc(tmax * 4)?,
            targets: alloc(tmax * 4)?,
            gath: alloc(tmax * h * es)?,
            logits: alloc(CHUNK * d.vocab * 4)?,
            dlog: alloc(CHUNK * d.vocab * es)?,
            loss_row: alloc(tmax * 4)?,
            correct: alloc(tmax * 4)?,
            row_w: alloc(tmax * 4)?,
            l: Learn::new(s.clone(), &module, &KERNELS, o.clone(), lay, &init,
                          Scratch { tmax, inp: d.ff.max(mix).max(in_max), ldu: U_RANKS * o.rank })?,
            layers,
            d,
            tmax,
            lmax,
            w,
            embed_name: e.strip_suffix(".weight").unwrap().to_string(),
        };
        t.l.s.synchronize()?;
        Ok(t)
    }

    /// Rows r0..r1 of a frozen matrix as act [r1 - r0][cols]: in place, or put together in the scratch from element
    /// `at` on
    fn wrows_at(&self, wt: &Weight, r0: usize, r1: usize, at: usize) -> Result<u64> {
        let (l, es, cols) = (&self.l, self.l.es, wt.cols);
        if let [Part::Dense { w, .. }] = &wt.parts[..] {
            return Ok(w.p + (r0 * cols * es) as u64);
        }
        ensure!(at + (r1 - r0) * cols <= self.scr_elems, "{} rows of {cols} in a scratch of {}", r1 - r0, self.scr_elems);
        let scr = self.scr.p + (at * es) as u64;
        let mut row = 0;
        for part in &wt.parts {
            let (a, b) = (row.max(r0), (row + part.rows()).min(r1));
            if a < b {
                let dst = scr + ((a - r0) * cols * es) as u64;
                match part {
                    Part::Dense { w, .. } => l.copy(dst, w.p + ((a - row) * cols * es) as u64, (b - a) * cols * es)?,
                    Part::Quant { q, sc, zp, words, group, bits, .. } => {
                        l.elementwise("dequant", (b - a) * words, &[P(q.p), P(sc.p), P(zp.p), I(*words as i32), I(cols as i32),
                                                                   I(*group as i32), I(*bits as i32), I((a - row) as i32),
                                                                   I((b - a) as i32), P(dst)])?;
                    }
                }
            }
            row += part.rows();
        }
        Ok(scr)
    }

    fn wrows(&self, wt: &Weight, r0: usize, r1: usize) -> Result<u64> {
        self.wrows_at(wt, r0, r1, 0)
    }

    /// Layer `i`'s matrices [w_in, w_out, w13, w_down] as act, those not in place side by side in the scratch
    fn layer_weights(&self, i: usize) -> Result<[u64; 4]> {
        let ly = &self.layers[i];
        let mut at = 0;
        let mut out = [0; 4];
        for (o, w) in out.iter_mut().zip([&ly.w_in, &ly.w_out, &ly.w13, &ly.w_down]) {
            *o = self.wrows_at(w, 0, w.rows, at)?;
            if !w.in_place() {
                at += w.rows * w.cols;
            }
        }
        Ok(out)
    }

    fn hs(&self, l: usize) -> u64 {
        self.hs.p + (l * self.tmax * self.d.h * 4) as u64
    }

    fn u(&self, g: usize) -> u64 {
        self.u.p + (U_COLS[g] * self.l.o.rank * self.l.es) as u64
    }

    fn upload(&self, mb: &Micro) -> Result<()> {
        ensure!(mb.tokens() <= self.tmax, "micro-batch of {} tokens, room for {}", mb.tokens(), self.tmax);
        ensure!(mb.docs.iter().all(|d| d.1 <= self.lmax), "document longer than {}", self.lmax);
        for (b, v) in self.meta.iter().zip([&mb.ids, &mb.pos, &mb.dstart, &mb.dend, &mb.dlen]) {
            self.l.htod(b.p, v)?;
        }
        // DeltaNet checkpoints: chunks 1.. of each document (chunk 0 starts from zeros)
        let (mut start, mut len, mut ck0, mut n) = (Vec::new(), Vec::new(), Vec::new(), 0);
        for &(s0, l) in &mb.docs {
            start.push(s0 as i32);
            len.push(l as i32);
            ck0.push(n as i32);
            n += l.div_ceil(C_CK) - 1;
        }
        ensure!(n <= self.ck_rows, "{n} DeltaNet checkpoints, room for {}", self.ck_rows);
        for (b, v) in self.docs.iter().zip([&start, &len, &ck0]) {
            self.l.htod(b.p, v)?;
        }
        self.l.htod(self.rows.p, &mb.rows)?;
        self.l.htod(self.targets.p, &mb.targets)?;
        self.l.s.synchronize()?;
        Ok(())
    }

    /// The Gated DeltaNet of a layer: p (in-projection) -> y; `ck`: write the state checkpoints (for the backward)
    fn delta_fwd(&self, ly: &Layer, t: usize, docs: usize, ck: bool) -> Result<()> {
        let (l, d) = (&self.l, &self.d);
        let m = |i: usize| self.meta[i].p;
        let f = |i: usize| ly.f[i].as_ref().unwrap().p;
        l.launch_cfg("dn_conv", cfg3((d.lqkv() / 128, t.div_ceil(4), 1), 128),
                     &[P(self.p.p), P(f(0)), P(f(1)), P(f(2)), P(m(2)), P(self.cv.p), P(self.gb.p), I(t as i32)])?;
        l.launch_cfg("dn_rule", cfg3((NSLICE, d.lv_heads, docs), 128),
                     &[P(self.cv.p), P(self.gb.p), P(self.o.p), P(if ck { self.ck.p } else { 0 }), P(self.docs[0].p),
                       P(self.docs[1].p), P(self.docs[2].p)])?;
        l.launch_cfg("dn_norm", cfg3((t, d.lv_heads / NORM_HEADS, 1), NORM_HEADS * 32), &[P(self.o.p), P(self.p.p), P(f(3)), P(self.y.p), F(EPS)])
    }

    /// dy (the mixer output's gradient) -> dp
    fn delta_bwd(&self, ly: &Layer, t: usize, docs: usize) -> Result<()> {
        let (l, d) = (&self.l, &self.d);
        let m = |i: usize| self.meta[i].p;
        let f = |i: usize| ly.f[i].as_ref().unwrap().p;
        l.launch_cfg("dn_norm_bwd", cfg3((t, d.lv_heads / NORM_HEADS, 1), NORM_HEADS * 32), &[P(self.dy.p), P(self.o.p), P(self.p.p), P(f(3)), P(self.dp.p), F(EPS)])?;
        l.zero(self.dqk.p, t * 2 * LK_HEADS * LKD * 4)?;
        l.zero(self.dgb.p, t * 2 * d.lv_heads * 4)?;
        l.launch_cfg("dn_rule_bwd", cfg3((NSLICE, d.lv_heads, docs), 128),
                     &[P(self.cv.p), P(self.gb.p), P(self.o.p), P(self.ck.p), P(self.docs[0].p), P(self.docs[1].p),
                       P(self.docs[2].p), P(self.dqk.p), P(self.dvv.p), P(self.dgb.p)])?;
        l.launch_cfg("dn_conv_bwd", cfg3((d.lqkv() / 128, t.div_ceil(4), 1), 128),
                     &[P(self.p.p), P(f(0)), P(f(1)), P(f(2)), P(m(2)), P(self.dqk.p), P(self.dvv.p), P(self.dgb.p),
                       P(self.gb.p), P(self.cv.p), P(self.dp.p), I(t as i32)])?;
        l.elementwise("dn_conv_in_bwd", t * d.lqkv(), &[P(self.cv.p), P(f(0)), P(m(3)), P(self.dp.p), I(t as i32)])
    }

    /// Head-major offsets of document (s0, len) and kv group g: Q / O, K / V (elements)
    fn group_at(&self, s0: usize, len: usize, g: usize) -> (u64, u64) {
        let group = self.d.nh / NKV;
        ((s0 * self.d.nh * HD + g * group * len * HD) as u64, (s0 * NKV * HD + g * len * HD) as u64)
    }

    /// Scores and softmax of one document's kv group into att_p
    fn scores(&self, s0: usize, len: usize, g: usize) -> Result<()> {
        let (l, es) = (&self.l, self.l.es as u64);
        let rows = self.d.nh / NKV * len;
        let lp = len.next_multiple_of(8);
        let (qo, ko) = self.group_at(s0, len, g);
        l.mm(false, true, rows, len, HD, 1.0, self.qa.p + qo * es, HD, self.ka.p + ko * es, HD, 0.0, self.att_s.p, lp, true)?;
        l.launch("at_softmax", rows.div_ceil(8), 256, &[P(self.att_s.p), P(self.att_p.p), I(rows as i32), I(len as i32),
                                                         I(lp as i32), F(1.0 / (HD as f32).sqrt())])
    }

    fn attn_fwd(&self, ly: &Layer, t: usize, mb: &Micro) -> Result<()> {
        let (l, es) = (&self.l, self.l.es as u64);
        let m = |i: usize| self.meta[i].p;
        let f = |i: usize| ly.f[i].as_ref().unwrap().p;
        l.launch("at_prep", t, 32 * (self.d.nh + 2 * NKV), &[P(self.p.p), P(f(0)), P(f(1)), P(self.inv_freq.p), P(m(1)),
            P(m(2)), P(m(4)), P(self.qa.p), P(self.ka.p), P(self.va.p), F(EPS)])?;
        for &(s0, len) in &mb.docs {
            for g in 0..NKV {
                self.scores(s0, len, g)?;
                let (qo, ko) = self.group_at(s0, len, g);
                let rows = self.d.nh / NKV * len;
                l.mm(false, false, rows, HD, len, 1.0, self.att_p.p, len.next_multiple_of(8), self.va.p + ko * es, HD, 0.0,
                     self.oa.p + qo * es, HD, false)?;
            }
        }
        l.launch("at_gate", t, 32 * self.d.nh, &[P(self.oa.p), P(self.p.p), P(m(2)), P(m(4)), P(self.y.p)])
    }

    fn attn_bwd(&self, ly: &Layer, t: usize, mb: &Micro) -> Result<()> {
        let (l, es) = (&self.l, self.l.es as u64);
        let m = |i: usize| self.meta[i].p;
        let f = |i: usize| ly.f[i].as_ref().unwrap().p;
        l.launch("at_gate_bwd", t, 32 * self.d.nh, &[P(self.dy.p), P(self.oa.p), P(self.p.p), P(m(2)), P(m(4)),
                                                     P(self.doa.p), P(self.dp.p)])?;
        let scale = 1.0 / (HD as f32).sqrt();
        for &(s0, len) in &mb.docs {
            let (rows, lp) = (self.d.nh / NKV * len, len.next_multiple_of(8));
            for g in 0..NKV {
                self.scores(s0, len, g)?;
                let (qo, ko) = self.group_at(s0, len, g);
                let (q, k, v) = (self.qa.p + qo * es, self.ka.p + ko * es, self.va.p + ko * es);
                let (dout, dq, dk, dv) = (self.doa.p + qo * es, self.dqa.p + qo * es, self.dka.p + ko * es, self.dva.p + ko * es);
                l.mm(false, true, rows, len, HD, 1.0, dout, HD, v, HD, 0.0, self.att_dp.p, lp, true)?;
                l.launch("at_softmax_bwd", rows.div_ceil(8), 256, &[P(self.att_p.p), P(self.att_dp.p), P(self.att_ds.p),
                                                                     I(rows as i32), I(len as i32), I(lp as i32), F(scale)])?;
                l.mm(false, false, rows, HD, len, 1.0, self.att_ds.p, lp, k, HD, 0.0, dq, HD, false)?;
                l.mm(true, false, len, HD, rows, 1.0, self.att_ds.p, lp, q, HD, 0.0, dk, HD, false)?;
                l.mm(true, false, len, HD, rows, 1.0, self.att_p.p, lp, dout, HD, 0.0, dv, HD, false)?;
            }
        }
        l.launch("at_prep_bwd", t, 32 * (self.d.nh + 2 * NKV), &[P(self.p.p), P(f(0)), P(f(1)), P(self.inv_freq.p), P(m(1)),
            P(m(2)), P(m(4)), P(self.dqa.p), P(self.dka.p), P(self.dva.p), P(self.dp.p), F(EPS)])
    }

    /// Layer `i` forward from xn (its input norm of hs[i]). `recompute` (the backward's): DeltaNet checkpoints, and the
    /// MLP stops at its down projection's LoRA U (hs[i + 1] is known). Else hs[i + 1] and xn = the next norm of it.
    #[allow(clippy::too_many_arguments)]
    fn layer_fwd(&self, i: usize, wts: [u64; 4], mb: &Micro, train: bool, step: u64, micro: usize, recompute: bool)
    -> Result<()> {
        let (l, ly, d, t) = (&self.l, &self.layers[i], &self.d, mb.tokens());
        let sd = |g: usize| Learn::seed(step, micro, i, g);
        let iw = ly.in_width;
        l.mm(false, true, t, iw, d.h, 1.0, self.xn.p, d.h, wts[0], d.h, 0.0, self.p.p, iw, false)?;
        l.lora_fwd(&ly.g[0], self.xn.p, t, self.p.p, iw, self.u(0), train, sd(0))?;
        if ly.delta {
            self.delta_fwd(ly, t, mb.docs.len(), recompute)?;
        } else {
            self.attn_fwd(ly, t, mb)?;
        }
        l.mm(false, true, t, d.h, d.mix(), 1.0, self.y.p, d.mix(), wts[1], d.mix(), 0.0, self.s_op.p, d.h, false)?;
        l.lora_fwd(&ly.g[1], self.y.p, t, self.s_op.p, d.h, self.u(1), train, sd(1))?;
        l.launch("add_rms", t, 256, &[P(self.hs(i)), P(self.s_op.p), P(ly.post_norm.p), P(self.hmid.p), P(self.xn2.p),
                                      P(self.rs_mid.p), F(EPS)])?;
        l.mm(false, true, t, 2 * d.ff, d.h, 1.0, self.xn2.p, d.h, wts[2], d.h, 0.0, self.w13.p, 2 * d.ff, false)?;
        l.lora_fwd(&ly.g[2], self.xn2.p, t, self.w13.p, 2 * d.ff, self.u(2), train, sd(2))?;
        l.elementwise("swiglu_fwd", t * d.ff, &[P(self.w13.p), P(self.mid.p), I(t as i32)])?;
        if recompute {
            return l.lora_u(&ly.g[3], self.mid.p, t, self.u(3), train, sd(3));
        }
        l.mm(false, true, t, d.h, d.ff, 1.0, self.mid.p, d.ff, wts[3], d.ff, 0.0, self.s_op.p, d.h, false)?;
        l.lora_fwd(&ly.g[3], self.mid.p, t, self.s_op.p, d.h, self.u(3), train, sd(3))?;
        let (next, rs) = match self.layers.get(i + 1) {
            Some(n) => (n.in_norm.p, self.rs_in.p),
            None => (self.final_norm.p, self.rs_fin.p),
        };
        l.launch("add_rms", t, 256, &[P(self.hmid.p), P(self.s_op.p), P(next), P(self.hs(i + 1)), P(self.xn.p), P(rs), F(EPS)])
    }

    /// Backward of layer `i`: dh holds d hs[i + 1] and becomes d hs[i]
    fn layer_bwd(&self, i: usize, mb: &Micro, step: u64, micro: usize) -> Result<()> {
        let (l, ly, d, t) = (&self.l, &self.layers[i], &self.d, mb.tokens());
        let sd = |g: usize| Learn::seed(step, micro, i, g);
        l.launch("rms_fwd", t, 256, &[P(self.hs(i)), P(ly.in_norm.p), P(self.xn.p), P(self.rs_in.p), F(EPS)])?;
        let wts = self.layer_weights(i)?;
        self.layer_fwd(i, wts, mb, true, step, micro, true)?;
        let (h, ff, mix, iw) = (d.h, d.ff, d.mix(), ly.in_width);
        // MLP
        l.elementwise("to_act_copy", t * h, &[P(self.dh.p), P(self.dact.p), P((t * h) as u64)])?;
        l.mm(false, false, t, ff, h, 1.0, self.dact.p, h, wts[3], ff, 0.0, self.dmid.p, ff, false)?;
        l.lora_bwd(&ly.g[3], self.dact.p, h, self.mid.p, t, self.u(3), self.dmid.p, false, sd(3))?;
        l.elementwise("swiglu_bwd", t * ff, &[P(self.w13.p), P(self.dmid.p), I(t as i32)])?;
        l.mm(false, false, t, h, 2 * ff, 1.0, self.w13.p, 2 * ff, wts[2], h, 0.0, self.dx.p, h, l.g32())?;
        l.lora_bwd(&ly.g[2], self.w13.p, 2 * ff, self.xn2.p, t, self.u(2), self.dx.p, true, sd(2))?;
        l.launch("rms_bwd_pre", t, 256, &[P(self.hmid.p), P(self.rs_mid.p), P(ly.post_norm.p), P(self.dx.p), P(self.dh.p)])?;
        // mixer
        l.elementwise("to_act_copy", t * h, &[P(self.dh.p), P(self.dact.p), P((t * h) as u64)])?;
        l.mm(false, false, t, mix, h, 1.0, self.dact.p, h, wts[1], mix, 0.0, self.dy.p, mix, false)?;
        l.lora_bwd(&ly.g[1], self.dact.p, h, self.y.p, t, self.u(1), self.dy.p, false, sd(1))?;
        if ly.delta {
            self.delta_bwd(ly, t, mb.docs.len())?;
        } else {
            self.attn_bwd(ly, t, mb)?;
        }
        l.mm(false, false, t, h, iw, 1.0, self.dp.p, iw, wts[0], h, 0.0, self.dx.p, h, l.g32())?;
        l.lora_bwd(&ly.g[0], self.dp.p, iw, self.xn.p, t, self.u(0), self.dx.p, true, sd(0))?;
        l.launch("rms_bwd_pre", t, 256, &[P(self.hs(i)), P(self.rs_in.p), P(ly.in_norm.p), P(self.dx.p), P(self.dh.p)])
    }

    /// Forward over the micro-batch; with `train` (dropout on) also the backward pass, adding the gradients of
    /// `scale` * (summed loss) to the arena. Returns (summed loss, correct argmax) over the micro-batch's loss rows.
    pub fn pass(&self, mb: &Micro, train: bool, scale: f32, step: u64, micro: usize) -> Result<(f64, usize)> {
        let (loss, correct) = self.run(mb, train, scale, None, step, micro)?;
        Ok((loss.iter().map(|&x| x as f64).sum(), correct))
    }

    /// A training pass with a weight per loss row that depends on the losses (preference training): after the
    /// forward pass `weigh` gets the loss (NLL) of every loss row and returns their weights w; the gradients of
    /// sum_i w_i * loss_i are added to the arena. Returns the rows' losses. The output layer's backward runs once
    /// with weight 1 and its rows are scaled afterwards (the rest of the backward pass is linear in them).
    pub fn pass_weighted(&self, mb: &Micro, step: u64, micro: usize, weigh: &mut dyn FnMut(&[f32]) -> Result<Vec<f32>>)
        -> Result<Vec<f32>> {
        Ok(self.run(mb, true, 1.0, Some(weigh), step, micro)?.0)
    }

    fn run(&self, mb: &Micro, train: bool, scale: f32, weigh: Option<&mut dyn FnMut(&[f32]) -> Result<Vec<f32>>>, step: u64,
           micro: usize) -> Result<(Vec<f32>, usize)> {
        self.upload(mb)?;
        let (l, t, h, es) = (&self.l, mb.tokens(), self.d.h, self.l.es as u64);
        let ids: Vec<u32> = mb.ids.iter().map(|&i| i as u32).collect();
        l.htod(self.hs(0), &self.w.rows(&self.embed_name, &ids)?)?;
        l.launch("rms_fwd", t, 256, &[P(self.hs(0)), P(self.layers[0].in_norm.p), P(self.xn.p), P(self.rs_in.p), F(EPS)])?;
        for i in 0..self.layers.len() {
            self.layer_fwd(i, self.layer_weights(i)?, mb, train, step, micro, false)?;
        }

        // the output layer on the loss rows only; xn holds the final norm
        let na = mb.rows.len();
        if na > 0 {
            l.launch("gather_rows", na, 256, &[P(self.xn.p), P(self.rows.p), P(self.gath.p)])?;
        }
        // the output layer in slices of vocabulary rows that fit the scratch (one slice when it is used in place)
        let lm = &self.lm_head;
        let vocab = self.d.vocab;
        let slice = if lm.in_place() { vocab } else { self.scr_elems / h };
        let out_w = |v0: usize, v1: usize| self.wrows(lm, v0, v1);
        for c0 in (0..na).step_by(CHUNK) {
            let c = CHUNK.min(na - c0);
            let g = self.gath.p + (c0 * h) as u64 * es;
            for v0 in (0..vocab).step_by(slice) {
                let v1 = (v0 + slice).min(vocab);
                l.mm(false, true, c, v1 - v0, h, 1.0, g, h, out_w(v0, v1)?, h, 0.0, self.logits.p + (v0 * 4) as u64, vocab, true)?;
            }
            l.launch("cross_entropy", c, 1024, &[P(self.logits.p), P(self.targets.p + c0 as u64 * 4), I(vocab as i32),
                F(scale * l.loss_scale), P(if train { self.dlog.p } else { 0 }), P(self.loss_row.p + c0 as u64 * 4),
                P(self.correct.p + c0 as u64 * 4)])?;
            if train {
                for v0 in (0..vocab).step_by(slice) {
                    let v1 = (v0 + slice).min(vocab);
                    l.mm(false, false, c, h, v1 - v0, 1.0, self.dlog.p + v0 as u64 * es, vocab, out_w(v0, v1)?, h,
                         if v0 == 0 { 0.0 } else { 1.0 }, l.tmp32.p + (c0 * h * 4) as u64, h, true)?;
                }
            }
        }
        let loss: Vec<f32> = l.dtoh(self.loss_row.p, na)?;
        let correct: Vec<i32> = l.dtoh(self.correct.p, na)?;
        let correct = correct.iter().filter(|&&c| c != 0).count();
        if !train {
            return Ok((loss, correct));
        }

        l.zero(self.dh.p, t * h * 4)?;
        let weights = match weigh {
            Some(f) => {
                let w = f(&loss)?;
                ensure!(w.len() == na, "{} row weights for {na} loss rows", w.len());
                Some(w)
            }
            None => None,
        };
        if na > 0 {
            match &weights {
                None => l.launch("scatter_rows", na, 256, &[P(l.tmp32.p), P(self.rows.p), P(self.dh.p)])?,
                Some(w) => {
                    l.htod(self.row_w.p, w)?;
                    l.launch("scatter_rows_scaled", na, 256, &[P(l.tmp32.p), P(self.rows.p), P(self.row_w.p), P(self.dh.p)])?;
                }
            }
        }
        let n = self.layers.len();
        l.launch("rms_bwd_final", t, 256, &[P(self.hs(n)), P(self.rs_fin.p), P(self.final_norm.p), P(self.dh.p)])?;
        for i in (0..n).rev() {
            self.layer_bwd(i, mb, step, micro)?;
        }
        if weights.is_some() {
            l.s.synchronize()?; // the weights' host copy outlives their upload
        }
        Ok((loss, correct))
    }
}

/// What the training loop (crate::training) asks of the trainer besides `pass`
impl Trainer {
    /// Most tokens of a micro-batch
    pub fn tmax(&self) -> usize {
        self.tmax
    }

    pub fn opts(&self) -> &Opts {
        &self.l.o
    }

    /// LoRA tensors: PEFT name, rows, columns
    pub fn tensors(&self) -> Vec<(String, usize, usize)> {
        self.l.tensor_shapes()
    }

    /// Loss per loss row of the last pass
    pub fn row_losses(&self, n: usize) -> Result<Vec<f32>> {
        self.l.dtoh(self.loss_row.p, n)
    }

    pub fn zero_grads(&mut self) -> Result<()> {
        self.l.zero_grads()
    }

    /// clip_grad_norm_(max_norm) and one AdamW step; the gradient norm before clipping (NaN: step skipped)
    pub fn optimizer_step(&mut self, lr: f32, max_norm: f32, wd: f32) -> Result<f32> {
        self.l.optimizer_step(lr, max_norm, wd)
    }

    /// The LoRA tensors by PEFT name (their gradients with `grads`)
    pub fn read_params(&self, grads: bool) -> Result<HashMap<String, Vec<f32>>> {
        self.l.read_params(grads)
    }

    pub fn write_params(&mut self, values: &HashMap<String, Vec<f32>>) -> Result<()> {
        self.l.write_params(values)
    }
}
