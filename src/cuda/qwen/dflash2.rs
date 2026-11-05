//! DFlash2: one parallel eight-row draft (an anchor and seven mask rows), conditioned on residual-stream taps of the
//! target. The target verifies the selected chain exactly as it verifies MTP's drafts. This module owns only proposal
//! generation and its sliding KV cache; accepted/rejected target tokens remain the server's business.
//!
//! Granite's drafter (tools/train_dflash2.py) has five layers and taps the target after layers 1, 10, 19, 28 and 37. Kernels:
//! kernels/dflash.cuh, in kernels/granite.cu.
use super::{Embed, Stack};
use crate::checkpoint::Checkpoint;
use crate::engine::Tok;
use crate::cuda::core::{self, Epi, Graphs, Mat, Step, cfg, part_len};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, PushKernelArg};
use std::{cell::RefCell, fs, path::Path, sync::Arc};

pub const BLOCK: usize = 8;
pub const DRAFTS: usize = BLOCK - 1;
/// the drafter's RMSNorm epsilon (DF_EPS in the kernels)
const EPS: f32 = 1e-6;
const WINDOW: usize = 2048;
const HD: usize = 128;
const RANK: usize = 256;
const TOPK: usize = 16;
const MAX_BLOCKS: usize = 16;
const MAX_HEAD: usize = MAX_BLOCKS * DRAFTS;
/// key ranges of the draft attention: 64 of the context window and one of the block (DF_SPLITS in the kernels)
const SPLITS: usize = 65;

struct DKernels {
    gather: CudaFunction,
    parts_xin: CudaFunction,
    conv_prepare: CudaFunction,
    conv_finish: CudaFunction,
    context_kv: CudaFunction,
    block_qkv: CudaFunction,
    attention: CudaFunction,
    combine: CudaFunction,
    sum: CudaFunction,
    head_rows: CudaFunction,
    top16: CudaFunction,
    selector: CudaFunction,
}

impl DKernels {
    fn load(m: &Arc<cudarc::driver::CudaModule>) -> Result<Self> {
        let f = |n| m.load_function(n);
        Ok(Self { gather: f("dflash_gather_rows")?, parts_xin: f("dflash_parts_xin")?,
                  conv_prepare: f("dflash_conv_prepare")?, conv_finish: f("dflash_conv_finish")?,
                  context_kv: f("dflash_context_kv")?, block_qkv: f("dflash_block_qkv")?,
                  attention: f("dflash_attention")?, combine: f("dflash_combine")?, sum: f("dflash_sum")?,
                  head_rows: f("dflash_head_rows")?, top16: f("dflash_top16")?, selector: f("dflash_selector")? })
    }
}

struct Layer {
    qkv: Mat,
    /// K/V without Q, for the target-context update (the Q columns would be wasted work there)
    kv: Mat,
    o: Mat,
    gate_up: Mat,
    down: Mat,
    attn_conv: Mat,
    mlp_conv: Mat,
    attn_base: CudaSlice<f32>,
    mlp_base: CudaSlice<f32>,
    input_norm: CudaSlice<f32>,
    post_norm: CudaSlice<f32>,
    q_norm: CudaSlice<f32>,
    k_norm: CudaSlice<f32>,
    kc: CudaSlice<u16>,
    vc: CudaSlice<u16>,
}

impl Layer {
    fn load(s: &Arc<CudaStream>, w: &Checkpoint, i: usize, slots: usize, nh: usize, nkv: usize, int8: bool) -> Result<Self> {
        let p = format!("layers.{i}");
        let mat = |name: &str, silu: bool| Mat::stack(s, w, &[format!("{p}.{name}.weight")], silu, int8);
        let qkv = Mat::stack(s, w, &["q_proj", "k_proj", "v_proj"].map(|n| format!("{p}.self_attn.{n}.weight")),
                             false, int8)?;
        let kv = Mat::stack(s, w, &["k_proj", "v_proj"].map(|n| format!("{p}.self_attn.{n}.weight")),
                            false, int8)?;
        ensure!(qkv.tiles * 16 == (nh + 2 * nkv) * HD && kv.tiles * 16 == 2 * nkv * HD,
                "{p}: attention projections for another geometry");
        let fl = |name: &str| -> Result<CudaSlice<f32>> { Ok(s.clone_htod(&w.floats(&format!("{p}.{name}"))?.0)?) };
        let cache = slots * WINDOW * nkv * HD;
        Ok(Self {
            qkv, kv,
            o: mat("self_attn.o_proj", false)?,
            gate_up: Mat::stack(s, w, &[format!("{p}.mlp.gate_proj.weight"), format!("{p}.mlp.up_proj.weight")],
                                true, int8)?,
            down: mat("mlp.down_proj", false)?, attn_conv: mat("attention_conv.kernel_projection", false)?,
            mlp_conv: mat("mlp_conv.kernel_projection", false)?,
            attn_base: fl("attention_conv.base_kernel")?, mlp_base: fl("mlp_conv.base_kernel")?,
            input_norm: fl("input_layernorm.weight")?, post_norm: fl("post_attention_layernorm.weight")?,
            q_norm: fl("self_attn.q_norm.weight")?, k_norm: fl("self_attn.k_norm.weight")?,
            kc: s.alloc_zeros(cache)?, vc: s.alloc_zeros(cache)?,
        })
    }

    fn state_bytes(&self) -> usize {
        self.kc.len() * 2 + self.vc.len() * 2
    }
}

pub struct DFlash2 {
    k: DKernels,
    /// the draft pass's own step (its token count changes between the stages)
    st: RefCell<Step>,
    /// the draft pass as CUDA graphs, per (context rows, blocks, rows of the target step)
    graphs: Graphs<(usize, usize, usize)>,
    taps: Vec<usize>,
    mask: u32,
    nh: usize,
    nkv: usize,
    h: usize,
    fc: Mat,
    hidden_norm: CudaSlice<f32>,
    norm: CudaSlice<f32>,
    layers: Vec<Layer>,
    hidden_projection: Mat,
    predecessor: CudaSlice<u16>,
    successor: CudaSlice<u16>,
    // Target-context and draft-block metadata, separate so one graph can use both.
    rows: CudaSlice<i32>,
    ctx_slots: CudaSlice<i32>,
    ctx_pos: CudaSlice<i32>,
    ids: CudaSlice<i32>,
    block_slots: CudaSlice<i32>,
    block_pos: CudaSlice<i32>,
    anchors: CudaSlice<i32>,
    gather: CudaSlice<u16>,
    context: CudaSlice<u16>,
    part: CudaSlice<f32>,
    residual: CudaSlice<f32>,
    xin: CudaSlice<u16>,
    conv: CudaSlice<u16>,
    dynamic: CudaSlice<u16>,
    sub: CudaSlice<f32>,
    q: CudaSlice<f32>,
    kn: CudaSlice<u16>,
    vn: CudaSlice<u16>,
    /// the attention's split partials (dflash_attention): unnormalized outputs and (max, sum) per row and head
    part_o: CudaSlice<f32>,
    part_ml: CudaSlice<f32>,
    mix: CudaSlice<u16>,
    mid: CudaSlice<u16>,
    head_in: CudaSlice<u16>,
    logits: CudaSlice<f32>,
    unary: CudaSlice<f32>,
    cands: CudaSlice<i32>,
    top_part_v: CudaSlice<f32>,
    top_part_i: CudaSlice<i32>,
    top_counters: CudaSlice<i32>,
    gate: CudaSlice<f32>,
    drafts: CudaSlice<i32>,
}

impl DFlash2 {
    pub fn load(x: &Stack, lm_head: &Mat, dir: &Path, int8: bool) -> Result<Self> {
        let config: serde_json::Value = serde_json::from_str(&fs::read_to_string(dir.join("config.json"))?)?;
        let dc = &config["dflash_config"];
        let get = |k: &str| config[k].as_u64().map(|v| v as usize);
        let (h, ff, nh, nkv) = (get("hidden_size").context("DFlash2 hidden_size")?,
                                get("intermediate_size").context("DFlash2 intermediate_size")?,
                                get("num_attention_heads").context("DFlash2 num_attention_heads")?,
                                get("num_key_value_heads").context("DFlash2 num_key_value_heads")?);
        ensure!(h == x.d.h && ff <= x.d.ff && ff % 256 == 0 && nh == x.d.nh && nkv == x.d.nkv && get("head_dim") == Some(HD) && x.d.hd == HD,
                "DFlash2 geometry {h}/{ff}, {nh}/{nkv} heads does not fit this target's kernels ({}/{}, {}/{} heads of {})",
                x.d.h, x.d.ff, x.d.nh, x.d.nkv, x.d.hd);
        ensure!(get("sliding_window") == Some(WINDOW), "DFlash2 must use a {WINDOW}-position sliding window");
        ensure!(dc["block_size"].as_u64() == Some(BLOCK as u64) && dc["selector_rank"].as_u64() == Some(RANK as u64)
                && dc["selector_top_k"].as_u64() == Some(TOPK as u64) && dc["conv_kernel_size"].as_u64() == Some(2)
                && dc["conv_group_size"].as_u64() == Some(16), "DFlash2 block/selector/convolution configuration is not supported");
        ensure!(config.get("output_multiplier").and_then(|v| v.as_f64()).unwrap_or(1.0) == 1.0
                && config.get("final_logit_softcapping").and_then(|v| v.as_f64()).unwrap_or(0.0) == 0.0,
                "DFlash2 logit scaling/softcapping is not supported");
        let taps: Vec<usize> = serde_json::from_value(dc["target_layer_ids"].clone()).context("DFlash2 target_layer_ids")?;
        ensure!(!taps.is_empty() && taps.len() <= 5 && taps.iter().all(|&t| t < x.layers.len()), "DFlash2 target taps {taps:?}");
        let nl = get("num_hidden_layers").context("DFlash2 num_hidden_layers")?;
        let w = Checkpoint::open(dir)?;
        let (s, d) = (&x.s, &x.d);
        let layers = (0..nl).map(|i| Layer::load(s, &w, i, x.st.slots, nh, nkv, int8)).collect::<Result<Vec<_>>>()?;
        let fc = Mat::stack(s, &w, &["fc.weight".into()], false, int8)?;
        ensure!(fc.cols == taps.len() * h && fc.tiles * 16 == h, "DFlash2 fc is {} x {}, want {h} x {}", fc.tiles * 16, fc.cols, taps.len() * h);
        let fl = |name: &str| -> Result<CudaSlice<f32>> { Ok(s.clone_htod(&w.floats(name)?.0)?) };
        let bf = |name: &str| -> Result<CudaSlice<u16>> {
            let (v, shape) = w.bf16(name)?;
            ensure!(shape == [d.vocab, RANK], "{name}: shape {shape:?}");
            Ok(s.clone_htod(&v)?)
        };
        let hidden_projection = Mat::stack(s, &w, &["candidate_selector.hidden_projection.weight".into()], false, int8)?;
        let dyn_width = h / 4;
        for l in &layers {
            ensure!(l.attn_conv.tiles * 16 == dyn_width && l.mlp_conv.tiles * 16 == dyn_width,
                    "DFlash2 dynamic-convolution width");
        }
        let z = |n: usize| s.alloc_zeros::<f32>(n);
        let zh = |n: usize| s.alloc_zeros::<u16>(n);
        let zi = |n: usize| s.alloc_zeros::<i32>(n);
        // the output layer's partial products for MAX_HEAD rows at most (not MAX_TOK: 2 GB for the 248k vocabulary)
        let part = {
            let mut mats = vec![&fc, &hidden_projection];
            for l in &layers {
                mats.extend([&l.qkv, &l.kv, &l.o, &l.gate_up, &l.down, &l.attn_conv, &l.mlp_conv]);
            }
            let rows = |m: &Mat| core::DEC_TILES.iter().zip(m.split).map(|(&n, s)| n * s).max().unwrap_or(0).max(MAX_HEAD);
            part_len(mats).max(rows(lm_head) * lm_head.tiles * 16)
        };
        Ok(Self {
            k: DKernels::load(&x.m)?,
            graphs: Graphs::new(Some(x.pool.clone())),
            st: RefCell::new(Step::new(s, &x.m, x.st.slots, x.st.max_ctx, d.vocab, super::attn::segments(d.nh / d.nkv))?),
            taps, mask: dc["mask_token_id"].as_u64().context("DFlash2 mask_token_id")? as u32,
            nh, nkv, h, fc, hidden_norm: fl("hidden_norm.weight")?, norm: fl("norm.weight")?, layers, hidden_projection,
            predecessor: bf("candidate_selector.predecessor_codebook")?, successor: bf("candidate_selector.successor_codebook")?,
            rows: zi(core::MAX_TOK)?, ctx_slots: zi(core::MAX_TOK)?, ctx_pos: zi(core::MAX_TOK)?,
            ids: zi(MAX_BLOCKS * BLOCK)?, block_slots: zi(MAX_BLOCKS)?, block_pos: zi(MAX_BLOCKS)?, anchors: zi(MAX_BLOCKS)?,
            gather: zh(core::MAX_TOK * taps_len(dc)? * h)?, context: zh(core::MAX_TOK * h)?,
            part: z(part)?, residual: z(core::MAX_TOK * h)?, xin: zh(core::MAX_TOK * h)?,
            conv: zh(core::MAX_TOK * h)?, dynamic: zh(core::MAX_TOK * dyn_width)?, sub: z(core::MAX_TOK * h)?,
            q: z(MAX_BLOCKS * BLOCK * nh * HD)?, kn: zh(MAX_BLOCKS * BLOCK * nkv * HD)?, vn: zh(MAX_BLOCKS * BLOCK * nkv * HD)?,
            part_o: z(MAX_BLOCKS * SPLITS * BLOCK * nh * HD)?, part_ml: z(MAX_BLOCKS * SPLITS * BLOCK * nh * 2)?, mix: zh(MAX_BLOCKS * BLOCK * nh * HD)?, mid: zh(MAX_BLOCKS * BLOCK * ff)?,
            head_in: zh(MAX_HEAD * h)?, logits: z(MAX_HEAD * d.vocab)?, unary: z(MAX_HEAD * TOPK)?, cands: zi(MAX_HEAD * TOPK)?,
            top_part_v: z(MAX_HEAD * 61 * TOPK)?, top_part_i: zi(MAX_HEAD * 61 * TOPK)?, top_counters: zi(MAX_HEAD)?,
            gate: z(MAX_HEAD * RANK)?, drafts: zi(MAX_HEAD)?,
        })
    }

    pub fn taps(&self) -> &[usize] {
        &self.taps
    }

    fn state_bytes(&self) -> usize {
        self.layers.iter().map(Layer::state_bytes).sum::<usize>() / self.st.borrow().slots
    }

    pub fn image_bytes(&self) -> usize {
        self.state_bytes()
    }

    pub fn save(&self, s: &Arc<CudaStream>, slot: usize, out: &mut Vec<u8>) -> Result<()> {
        let per = WINDOW * self.nkv * HD;
        for l in &self.layers {
            for c in [&l.kc, &l.vc] {
                let v = s.clone_dtoh(&c.slice(slot * per..(slot + 1) * per))?;
                out.extend(v.iter().flat_map(|x| x.to_le_bytes()));
            }
        }
        Ok(())
    }

    pub fn load_state(&mut self, s: &Arc<CudaStream>, slot: usize, data: &[u8]) -> Result<usize> {
        let per = WINDOW * self.nkv * HD;
        let mut at = 0;
        for l in &mut self.layers {
            for c in [&mut l.kc, &mut l.vc] {
                let v: Vec<u16> = data.get(at..at + per * 2).context("short DFlash2 state")?.chunks_exact(2)
                    .map(|x| u16::from_le_bytes(x.try_into().unwrap())).collect();
                s.memcpy_htod(&v, &mut c.slice_mut(slot * per..(slot + 1) * per))?;
                at += per * 2;
            }
        }
        Ok(at)
    }

    /// Update the draft's target-context cache from the target taps at `rows`, then propose at most `k` tokens after
    /// each anchor at `out` (indices into `toks`). `target_rows` is the row count of the target step that made `taps`.
    #[allow(clippy::too_many_arguments)]
    pub fn draft(&mut self, x: &Stack, embed: &Embed, lm_head: &Mat, target_taps: &CudaSlice<u16>, target_rows: usize,
                 toks: &[Tok], rows: &[usize], out: &[usize], k: usize) -> Result<Vec<Vec<u32>>> {
        ensure!(rows.len() == toks.len() && !toks.is_empty() && rows.iter().all(|&r| r < target_rows), "DFlash2 context rows");
        ensure!(out.len() <= MAX_BLOCKS && out.iter().all(|&o| o < toks.len()), "DFlash2 output rows");
        let s = &x.s;
        let r: Vec<i32> = rows.iter().map(|&r| r as i32).collect();
        let slots: Vec<i32> = toks.iter().map(|t| t.slot as i32).collect();
        let pos: Vec<i32> = toks.iter().map(|t| t.pos as i32).collect();
        s.memcpy_htod(&r, &mut self.rows.slice_mut(..r.len()))?;
        s.memcpy_htod(&slots, &mut self.ctx_slots.slice_mut(..slots.len()))?;
        s.memcpy_htod(&pos, &mut self.ctx_pos.slice_mut(..pos.len()))?;
        let mut ids = Vec::with_capacity(out.len() * BLOCK);
        let mut bslots = Vec::with_capacity(out.len());
        let mut bpos = Vec::with_capacity(out.len());
        let mut anchors = Vec::with_capacity(out.len());
        for &o in out {
            let a = toks[o];
            anchors.push(a.id as i32);
            bslots.push(a.slot as i32);
            bpos.push(a.pos as i32 + 1);
            for j in 0..BLOCK {
                ids.push(if j == 0 { a.id } else { self.mask } as i32);
            }
        }
        if !out.is_empty() {
            s.memcpy_htod(&ids, &mut self.ids.slice_mut(..ids.len()))?;
            s.memcpy_htod(&bslots, &mut self.block_slots.slice_mut(..bslots.len()))?;
            s.memcpy_htod(&bpos, &mut self.block_pos.slice_mut(..bpos.len()))?;
            s.memcpy_htod(&anchors, &mut self.anchors.slice_mut(..anchors.len()))?;
        }
        // one graph per shape (QWENNY_DRAFT_STATS times each stage instead: synchronizing breaks a capture)
        let shape = (toks.len(), out.len(), target_rows);
        if std::env::var_os("QWENNY_DRAFT_STATS").is_some() {
            self.launch(x, embed, lm_head, target_taps, target_rows, toks.len(), out.len())?;
        } else {
            self.graphs.run(s, toks.len().max(out.len() * BLOCK), shape,
                            || self.launch(x, embed, lm_head, target_taps, target_rows, toks.len(), out.len()))?;
        }
        if out.is_empty() {
            return Ok(Vec::new());
        }
        let d = s.clone_dtoh(&self.drafts.slice(..out.len() * DRAFTS))?;
        Ok(d.chunks(DRAFTS).map(|d| d.iter().take(k).map(|&t| t as u32).collect()).collect())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(&self, x: &Stack, embed: &Embed, lm_head: &Mat, taps: &CudaSlice<u16>, target_rows: usize,
              nctx: usize, nblocks: usize) -> Result<()> {
        let (s, k, ntaps) = (&x.s, &self.k, self.taps.len());
        let profile = std::env::var_os("QWENNY_DRAFT_STATS").is_some();
        let mut marks = Vec::new();
        let mut last = std::time::Instant::now();
        let mut mark = |name: &'static str| -> Result<()> {
            if profile {
                s.synchronize()?;
                let now = std::time::Instant::now();
                marks.push((name, now.duration_since(last).as_secs_f64() * 1e3));
                last = now;
            }
            Ok(())
        };
        self.st.borrow_mut().n = nctx;
        let ni = nctx as i32;
        unsafe {
            s.launch_builder(&k.gather).arg(taps).arg(&self.rows).arg(&self.gather).arg(&ni).arg(&(ntaps as i32))
                .arg(&(target_rows as i32)).launch(cfg((nctx, 1, 1), 256))?;
        }
        let split = x.gemm_with(&self.st.borrow(), &self.fc, &self.gather, &self.part, Epi::Plain)?;
        self.sum_parts(x, split, nctx, self.h, &self.residual)?;
        self.rmsnorm(x, nctx, &self.hidden_norm, &self.context)?;
        for l in &self.layers {
            let split = x.gemm_with(&self.st.borrow(), &l.kv, &self.context, &self.part, Epi::Plain)?;
            unsafe {
                s.launch_builder(&k.context_kv).arg(&self.part).arg(&(split as i32)).arg(&ni).arg(&l.k_norm)
                    .arg(&self.ctx_slots).arg(&self.ctx_pos).arg(&l.kc).arg(&l.vc).arg(&0i32)
                    .arg(&((2 * self.nkv * HD) as i32)).launch(cfg((nctx, 1, 1), 32 * self.nkv))?;
            }
        }
        mark("context")?;
        if nblocks == 0 {
            return Ok(());
        }
        let nb = nblocks * BLOCK;
        self.st.borrow_mut().n = nb;
        embed.launch(s, &x.k, &self.ids, &self.residual, &self.st.borrow().keys, 0, nb)?;
        for l in &self.layers {
            self.rmsnorm(x, nb, &l.input_norm, &self.xin)?;
            self.conv_prepare(x, nb, &l.attn_conv, &l.attn_base)?;
            let split = x.gemm_with(&self.st.borrow(), &l.qkv, &self.conv, &self.part, Epi::Plain)?;
            unsafe {
                s.launch_builder(&k.block_qkv).arg(&self.part).arg(&(split as i32)).arg(&(nb as i32)).arg(&l.q_norm)
                    .arg(&l.k_norm).arg(&self.block_pos).arg(&self.q).arg(&self.kn).arg(&self.vn)
                    .launch(cfg((nb, 1, 1), 256))?;
                s.launch_builder(&k.attention).arg(&self.q).arg(&l.kc).arg(&l.vc).arg(&self.kn).arg(&self.vn)
                    .arg(&self.block_slots).arg(&self.block_pos).arg(&self.part_o).arg(&self.part_ml)
                    .launch(cfg((nblocks, self.nkv, SPLITS), 32 * self.nh / self.nkv))?;
                s.launch_builder(&k.combine).arg(&self.part_o).arg(&self.part_ml).arg(&self.mix)
                    .launch(cfg((nb, self.nh, 1), HD))?;
            }
            let split = x.gemm_with(&self.st.borrow(), &l.o, &self.mix, &self.part, Epi::Plain)?;
            self.sum_parts(x, split, nb, self.h, &self.sub)?;
            self.conv_finish(x, nb, &l.attn_base)?;
            self.rmsnorm(x, nb, &l.post_norm, &self.xin)?;
            self.conv_prepare(x, nb, &l.mlp_conv, &l.mlp_base)?;
            x.gemm_with(&self.st.borrow(), &l.gate_up, &self.conv, &self.mid, Epi::Silu)?;
            let split = x.gemm_with(&self.st.borrow(), &l.down, &self.mid, &self.part, Epi::Plain)?;
            self.sum_parts(x, split, nb, self.h, &self.sub)?;
            self.conv_finish(x, nb, &l.mlp_base)?;
        }
        mark("draft layers")?;
        self.rmsnorm(x, nb, &self.norm, &self.xin)?;
        let nh = nblocks * DRAFTS;
        self.st.borrow_mut().n = nh;
        unsafe {
            s.launch_builder(&k.head_rows).arg(&self.xin).arg(&self.head_in).arg(&(nh as i32)).launch(cfg((nh, 1, 1), 256))?;
        }
        let cols = x.d.vocab;
        let split = x.gemm_with(&self.st.borrow(), lm_head, &self.head_in, &self.part, Epi::Plain)?;
        self.sum_parts(x, split, nh, cols, &self.logits)?;
        mark("lm head")?;
        unsafe {
            let (splits, ncols) = (cols.div_ceil(4096) as i32, cols as i32);
            s.launch_builder(&k.top16).arg(&self.logits).arg(&splits).arg(&ncols)
                .arg(&self.top_part_v).arg(&self.top_part_i).arg(&self.top_counters).arg(&self.unary).arg(&self.cands)
                .launch(cfg((nh * splits as usize, 1, 1), 256))?;
        }
        mark("top16")?;
        let split = x.gemm_with(&self.st.borrow(), &self.hidden_projection, &self.head_in, &self.part, Epi::Plain)?;
        self.sum_parts(x, split, nh, RANK, &self.gate)?;
        unsafe {
            s.launch_builder(&k.selector).arg(&self.unary).arg(&self.cands).arg(&self.gate).arg(&self.predecessor)
                .arg(&self.successor).arg(&self.anchors).arg(&self.drafts).launch(cfg((nblocks, 1, 1), 256))?;
        }
        mark("selector")?;
        if profile {
            eprintln!("DFlash2 stages: {marks:?}");
        }
        Ok(())
    }

    fn sum_parts(&self, x: &Stack, split: usize, n: usize, ld: usize, out: &CudaSlice<f32>) -> Result<()> {
        unsafe {
            x.s.launch_builder(&self.k.sum).arg(&self.part).arg(&(split as i32)).arg(&(n as i32)).arg(&(ld as i32))
                .arg(out).launch(cfg(((n * ld).div_ceil(256).min(1024), 1, 1), 256))?;
        }
        Ok(())
    }

    fn rmsnorm(&self, x: &Stack, n: usize, w: &CudaSlice<f32>, out: &CudaSlice<u16>) -> Result<()> {
        unsafe {
            x.s.launch_builder(&x.k.rmsnorm).arg(&self.residual).arg(w).arg(out).arg(&EPS).launch(cfg((n, 1, 1), 256))?;
        }
        Ok(())
    }

    fn conv_prepare(&self, x: &Stack, n: usize, proj: &Mat, base: &CudaSlice<f32>) -> Result<()> {
        let split = x.gemm_with(&self.st.borrow(), proj, &self.xin, &self.part, Epi::Plain)?;
        unsafe {
            x.s.launch_builder(&self.k.parts_xin).arg(&self.part).arg(&(split as i32)).arg(&(n as i32))
                .arg(&(self.h as i32 / 4)).arg(&self.dynamic).launch(cfg((n, 1, 1), 256))?;
            x.s.launch_builder(&self.k.conv_prepare).arg(&self.xin).arg(&self.dynamic).arg(base).arg(&self.conv)
                .launch(cfg((n, 1, 1), 256))?;
        }
        Ok(())
    }

    fn conv_finish(&self, x: &Stack, n: usize, base: &CudaSlice<f32>) -> Result<()> {
        unsafe {
            x.s.launch_builder(&self.k.conv_finish).arg(&self.sub).arg(&self.dynamic).arg(base).arg(&self.residual)
                .launch(cfg((n, 1, 1), 256))?;
        }
        Ok(())
    }
}

fn taps_len(dc: &serde_json::Value) -> Result<usize> {
    Ok(dc["target_layer_ids"].as_array().context("DFlash2 target_layer_ids")?.len())
}
