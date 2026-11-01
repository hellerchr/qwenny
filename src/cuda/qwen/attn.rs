//! Full attention of Qwen3.5 (every 4th layer): q (with a per-dim output gate), k, v from the in-projection, RMSNorm
//! (1 + w) on q and k per head, RoPE on the first ROT dims of each head, GQA over a bf16 KV cache, sigmoid gate.
//! Kernels in kernels/qwen_attn.cuh: attn_prep (every token: q to a scratch buffer, k and v into the cache, the gate into
//! the output), attn_decode + attn_combine (single-token segments: flash-decoding over key splits) and attn_prefill +
//! attn_combine_tiles (longer segments in tiles of 16 tokens on tensor cores, split by keys when few tiles).
use super::{Ctx, Dims, HD, NKV};
use crate::checkpoint::Checkpoint;
use crate::cuda::core::{MAX_TOK, Segments, Step, cfg};
use crate::cuda::vmm::{Pool, Region};
use anyhow::{Result, ensure};
use cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Single tokens use the decode attention (at most 64 per step), longer segments the prompt attention in tiles of 16
/// tokens (one warp holds 16 query rows: mma m16). Short segments (drafts) go to the tiles too: their tokens share
/// the key reads on tensor cores, where on the decode path each token would read the whole cache.
pub fn segments(group: usize) -> Segments {
    Segments { short: 8, max_dec: 64, ptile: 64 / group, cut: DCUT }
}
/// Rotary dims of a head and RoPE base; must match ROT in kernels/qwen.cu (Dims::of checks the config against them)
pub const ROT: usize = 64;
pub const THETA: f64 = 1e7;
/// Decode: keys per split at least (KPS in the kernels), most splits per segment, most (decode token, split) pairs per
/// step (the partial-result buffers).
const KPS: usize = 32;
/// Key splits of every decode step (DSPLITS in the kernels): a token's splits depend on its keys alone (dchunk), so
/// drafts sum their attention as one token per step does; the split size changes at multiples of DCUT keys.
const DSPLITS: usize = 128;
const DCUT: usize = 16 * DSPLITS;
const PAIRS: usize = 64 * DSPLITS;
/// Prompt tiles: most key splits per tile, most (tile, split) pairs per step (16 rows each, the same buffers), keys
/// per split at least per tile of the step (splits pay off when few tiles stream long contexts; with many tiles the
/// GPU is busy and merging costs bandwidth)
const TSPLITS: usize = 128;
const TPAIRS: usize = 256;
const TKPS: usize = 64;
/// Warps of attn_mq1..3: one per row tile
const MQ_WARPS: [usize; 3] = [1, 2, 3];
/// attn_mq of `rt` row tiles: dynamic shared memory for Q (rt * 16 rows of HD + 8 halves; K and V come straight from
/// the cache)
fn mq_smem(rt: usize) -> usize {
    rt * 16 * (HD + 8) * 2
}

/// attn_prefill: a block per (tile of 64 / GROUP tokens, kv head) of 4 warps over the 64 (token, query head) rows;
/// dynamic shared memory: Q (64 rows of HD + 8 halves) and 2 stages of 64 int8 keys and values (PREFILL_SMEM)
fn prefill_smem() -> usize {
    64 * (HD + 8) * 2 + 2 * 2 * 64 * HD
}

/// The attention kernels (from the model's module)
pub struct Kernels {
    prep: CudaFunction,
    /// attn_mq1..3: short segments of up to 16, 32, 48 rows (tokens * query heads of a kv head)
    decode: [CudaFunction; 3],
    /// query heads per kv head
    group: usize,
    combine: CudaFunction,
    prefill: CudaFunction,
    combine_tiles: CudaFunction,
}

impl Kernels {
    pub fn load(_s: &Arc<CudaStream>, m: &Arc<CudaModule>, d: &Dims) -> Result<Self> {
        let prefill = m.load_function("attn_prefill")?;
        prefill.set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, prefill_smem() as i32)?;
        let decode = [m.load_function("attn_mq1")?, m.load_function("attn_mq2")?, m.load_function("attn_mq3")?];
        for (rt, f) in decode.iter().enumerate() {
            f.set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, mq_smem(rt + 1) as i32)?;
        }
        Ok(Kernels {
            prep: m.load_function("attn_prep")?,
            decode,
            group: d.nh / NKV,
            combine: m.load_function("attn_combine")?,
            prefill,
            combine_tiles: m.load_function("attn_combine_tiles")?,
        })
    }
}

/// Per-step scratch the layers share: q [MAX_TOK][nh][HD], partial results per (decode token or tile row, head, split)
pub struct Scratch {
    q: CudaSlice<f32>,
    ml: CudaSlice<f32>,
    acc: CudaSlice<f32>,
}

impl Scratch {
    pub fn new(s: &Arc<CudaStream>, d: &Dims) -> Result<Self> {
        let rows = PAIRS.max(TPAIRS * 16) * d.nh;
        Ok(Scratch { q: s.alloc_zeros(MAX_TOK * d.nh * HD)?, ml: s.alloc_zeros(rows * 2)?, acc: s.alloc_zeros(rows * HD)? })
    }
}

/// Positions a slot's part of the cache is reserved for, a multiple of CAP_ALIGN: then every slot's keys and values
/// of a head (cap * HD bytes) and its scales (NKV * cap floats) start on a page of 2 MiB and map on their own
const CAP_ALIGN: usize = 131072;

/// One full-attention layer: its norms, RoPE table and KV cache.
pub struct Attn {
    q_norm: CudaSlice<f32>,
    k_norm: CudaSlice<f32>,
    inv_freq: CudaSlice<f32>,
    /// int8 [slot][NKV][cap][HD], an f32 scale per position: ks, vs [slot][NKV][cap]; mapped per slot for the
    /// positions it holds (`mapped`, a multiple of a page's positions), the scales of a slot as a whole
    kc: Region,
    vc: Region,
    ks: Region,
    vs: Region,
    mapped: Vec<usize>,
    cap: usize,
}

impl Attn {
    /// `prefix`: e.g. "model.language_model.layers.3.self_attn"
    pub fn new(s: &Arc<CudaStream>, pool: &Arc<Pool>, w: &Checkpoint, prefix: &str, slots: usize, max_ctx: usize)
               -> Result<Self> {
        let q_norm = super::norm(w, &format!("{prefix}.q_norm.weight"))?;
        let k_norm = super::norm(w, &format!("{prefix}.k_norm.weight"))?;
        ensure!(q_norm.len() == HD && k_norm.len() == HD, "{prefix}: norms of {} and {}", q_norm.len(), k_norm.len());
        let inv_freq: Vec<f32> = (0..ROT / 2).map(|d| THETA.powf(-2.0 * d as f64 / ROT as f64) as f32).collect();
        let cap = max_ctx.div_ceil(CAP_ALIGN) * CAP_ALIGN;
        ensure!((NKV * cap * 4) % pool.page() == 0 && (cap * HD) % pool.page() == 0, "pages of {} bytes", pool.page());
        let cache = slots * NKV * cap * HD;
        Ok(Attn {
            q_norm: s.clone_htod(&q_norm)?,
            k_norm: s.clone_htod(&k_norm)?,
            inv_freq: s.clone_htod(&inv_freq)?,
            kc: Region::reserve(s, pool, cache)?,
            vc: Region::reserve(s, pool, cache)?,
            ks: Region::reserve(s, pool, cache / HD * 4)?,
            vs: Region::reserve(s, pool, cache / HD * 4)?,
            mapped: vec![0; slots],
            cap,
        })
    }

    /// Positions per page of a head's keys (or values)
    fn page_pos(&self) -> usize {
        self.kc.page_size() / HD
    }

    /// Map slot `slot`'s cache for positions [0, len) (and its scales); false when the pages ran out
    pub fn map(&mut self, slot: usize, len: usize) -> Result<bool> {
        let len = len.min(self.cap).div_ceil(self.page_pos()) * self.page_pos();
        if len <= self.mapped[slot] {
            return Ok(true);
        }
        let sc = slot * NKV * self.cap * 4;
        if self.mapped[slot] == 0 && !(self.ks.map(sc, sc + NKV * self.cap * 4)? && self.vs.map(sc, sc + NKV * self.cap * 4)?) {
            return Ok(false);
        }
        for g in 0..NKV {
            let at = (slot * NKV + g) * self.cap * HD;
            if !(self.kc.map(at, at + len * HD)? && self.vc.map(at, at + len * HD)?) {
                return Ok(false);
            }
        }
        self.mapped[slot] = len;
        Ok(true)
    }

    /// Device bytes slot `slot` maps holding `len` positions: whole pages of positions, the scales of the slot whole
    pub fn slot_bytes(&self, len: usize) -> usize {
        let len = len.min(self.cap).div_ceil(self.page_pos()) * self.page_pos();
        2 * NKV * (len * HD + self.cap * 4)
    }

    /// Bytes of a slot's cache for positions [from, to) (save, load; `from` a multiple of 16): per kv head its keys,
    /// values (in blocks of 16 positions) and their scales
    pub fn saved_bytes(from: usize, to: usize) -> usize {
        NKV * ((to - from) * HD + (to.div_ceil(16) * 16 - from) * HD + 2 * (to - from) * 4)
    }

    /// Slot `slot`'s cache for positions [from, to) (mapped) into `out` (saved_bytes long)
    pub fn save(&self, slot: usize, from: usize, to: usize, out: &mut [u8]) -> Result<()> {
        ensure!(from % 16 == 0 && from <= to && to <= self.mapped[slot], "slot {slot}: positions {from}..{to} of {}", self.mapped[slot]);
        ensure!(out.len() == Self::saved_bytes(from, to), "{} bytes for positions {from}..{to}", out.len());
        let (vl, n, mut o) = (to.div_ceil(16) * 16 - from, to - from, out);
        for g in 0..NKV {
            let (at, sc) = (((slot * NKV + g) * self.cap + from) * HD, ((slot * NKV + g) * self.cap + from) * 4);
            let (k, rest) = o.split_at_mut(n * HD);
            let (v, rest) = rest.split_at_mut(vl * HD);
            let (ks, rest) = rest.split_at_mut(n * 4);
            let (vs, rest) = rest.split_at_mut(n * 4);
            self.kc.read(at, k)?;
            self.vc.read(at, v)?;
            self.ks.read(sc, ks)?;
            self.vs.read(sc, vs)?;
            o = rest;
        }
        Ok(())
    }

    /// Put a cache from save for positions [from, to) into slot `slot`; the bytes it took; None when the pages ran out
    pub fn load(&mut self, slot: usize, from: usize, to: usize, data: &[u8]) -> Result<Option<usize>> {
        let nb = Self::saved_bytes(from, to);
        ensure!(from % 16 == 0 && data.len() >= nb, "a cache of {} bytes for positions {from}..{to}, not {nb}", data.len());
        if !self.map(slot, to)? {
            return Ok(None);
        }
        let (vl, n, mut o) = (to.div_ceil(16) * 16 - from, to - from, &data[..nb]);
        for g in 0..NKV {
            let (at, sc) = (((slot * NKV + g) * self.cap + from) * HD, ((slot * NKV + g) * self.cap + from) * 4);
            let (k, rest) = o.split_at(n * HD);
            let (v, rest) = rest.split_at(vl * HD);
            let (ks, rest) = rest.split_at(n * 4);
            let (vs, rest) = rest.split_at(n * 4);
            self.kc.write(at, k)?;
            self.vc.write(at, v)?;
            self.ks.write(sc, ks)?;
            self.vs.write(sc, vs)?;
            o = rest;
        }
        Ok(Some(nb))
    }

    /// Unmap slot `slot`'s cache from position `from` on (0: all of it, with its scales)
    pub fn unmap(&mut self, slot: usize, from: usize) -> Result<()> {
        let from = from.div_ceil(self.page_pos()) * self.page_pos();
        for g in 0..NKV {
            let at = (slot * NKV + g) * self.cap * HD;
            self.kc.unmap(at + from * HD, at + self.cap * HD)?;
            self.vc.unmap(at + from * HD, at + self.cap * HD)?;
        }
        if from == 0 {
            let sc = slot * NKV * self.cap * 4;
            self.ks.unmap(sc, sc + NKV * self.cap * 4)?;
            self.vs.unmap(sc, sc + NKV * self.cap * 4)?;
        }
        self.mapped[slot] = self.mapped[slot].min(from);
        Ok(())
    }

    /// Row tiles of the decode attention (attn_mq1..3) for the step's longest short segment
    fn row_tiles(k: &Kernels, st: &Step) -> usize {
        (st.dmax * k.group).div_ceil(16)
    }
    /// Key splits of the decode tokens (the grid's z): at least the most any token takes (nsplits in the kernels: chunks
    /// of at least KPS keys, at most DSPLITS of them), a power of two (few CUDA graphs). A token's splits never depend on
    /// the step, so drafts give the logits of one token per step.
    fn splits(st: &Step) -> usize {
        if st.ndec == 0 {
            return 0;
        }
        (st.dec_max_pos + 1).div_ceil(KPS).next_power_of_two().min(DSPLITS)
    }

    /// Key splits of the prompt tiles (the grid's z): enough splits of at least TKPS * ntile keys for the longest
    /// context, a power of two, at most TSPLITS and TPAIRS / ntile.
    fn tile_splits(st: &Step) -> usize {
        if st.ntile == 0 {
            return 0;
        }
        let most = 1 << TSPLITS.min(TPAIRS / st.ntile).max(1).ilog2();
        (st.tile_max_pos + 1).div_ceil(TKPS * st.ntile).next_power_of_two().min(most)
    }

    /// Read the in-projection's partial products `b.part` [split][n][attn_in], write `b.mix` [n][mix] (the kernels
    /// write `b.o`, then to_xin).
    pub fn run(&self, c: &Ctx, split: usize) -> Result<()> {
        let (s, st, b, k, sc) = (c.s, c.st, c.b, &c.k.attn, &c.b.attn);
        let (nh, group) = (c.d.nh, c.d.nh / NKV);
        let (sp, n, cap) = (split as i32, st.n as i32, self.cap as i32);
        let qscale = std::f32::consts::LOG2_E / (HD as f32).sqrt();
        let (kc, vc, ks, vs) = (self.kc.ptr(), self.vc.ptr(), self.ks.ptr(), self.vs.ptr());
        unsafe {
            s.launch_builder(&k.prep).arg(&b.part).arg(&sp).arg(&n).arg(&self.q_norm).arg(&self.k_norm)
                .arg(&self.inv_freq).arg(&qscale).arg(&sc.q).arg(&b.o).arg(&kc).arg(&vc)
                .arg(&ks).arg(&vs).arg(&st.tok_slot).arg(&st.tok_pos).arg(&cap).launch(cfg((st.n, 1, 1), 32 * (nh + 2 * NKV)))?;
            if st.ndec > 0 {
                let z = Self::splits(st);
                let zi = z as i32;
                let rt = Self::row_tiles(k, st);
                let warps = MQ_WARPS[rt - 1];
                s.launch_builder(&k.decode[rt - 1]).arg(&sc.q).arg(&kc).arg(&vc).arg(&ks).arg(&vs).arg(&st.dseg)
                    .arg(&st.dtok).arg(&st.tok_slot).arg(&st.tok_pos).arg(&cap).arg(&zi).arg(&sc.ml).arg(&sc.acc).arg(&b.o)
                    .launch(LaunchConfig { shared_mem_bytes: mq_smem(rt) as u32, ..cfg((st.ndseg, NKV, z), 32 * warps) })?;
                if z > 1 {
                    s.launch_builder(&k.combine).arg(&sc.ml).arg(&sc.acc).arg(&st.dtok).arg(&st.dkeys).arg(&zi)
                        .arg(&b.o).launch(cfg((st.ndec, nh, 1), HD))?;
                }
            }
            if st.ntile > 0 {
                let z = Self::tile_splits(st);
                let (zi, kps) = (z as i32, (TKPS * st.ntile) as i32);
                let mut lc = cfg((st.ntile, NKV, z), 32 * 4);
                lc.shared_mem_bytes = prefill_smem() as u32;
                s.launch_builder(&k.prefill).arg(&sc.q).arg(&kc).arg(&vc).arg(&ks).arg(&vs).arg(&st.ptile).arg(&st.tok_slot)
                    .arg(&st.tok_pos).arg(&cap).arg(&zi).arg(&kps).arg(&sc.ml).arg(&sc.acc).arg(&b.o).launch(lc)?;
                if z > 1 {
                    s.launch_builder(&k.combine_tiles).arg(&sc.ml).arg(&sc.acc).arg(&st.ptile).arg(&st.tok_pos)
                        .arg(&zi).arg(&kps).arg(&b.o).launch(cfg((st.ntile * (64 / group), nh, 1), HD))?;
                }
            }
            let len = (st.n * c.d.mix()) as i32;
            s.launch_builder(&c.k.to_xin).arg(&b.o).arg(&b.mix).arg(&len).launch(cfg(((st.n * c.d.mix()).div_ceil(1024).min(1024), 1, 1), 256))?;
        }
        Ok(())
    }

    /// Whatever about a step changes the launched kernels beyond its shape (part of the CUDA graph key): the key
    /// splits of the decode tokens and of the prompt tiles.
    pub fn graph_key(_k: &Kernels, st: &Step) -> (usize, usize, usize, usize) {
        (Self::splits(st), Self::tile_splits(st), st.ndseg, st.dmax)
    }
}
