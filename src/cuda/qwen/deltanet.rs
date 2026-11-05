//! Gated DeltaNet of Qwen3.5 (3 of every 4 layers): causal conv (kernel 4, SiLU) over q, k, v, l2-normed q and k,
//! the gated delta rule with a recurrent state per slot and value head, RMSNorm gated by SiLU(z).
//! Three kernels per step (kernels/qwen_deltanet.cuh): delta_conv, delta_rule, delta_norm. All state decisions (new
//! sequence, continue, restart after rejected drafts) are taken on the device from the segments and a few ints of
//! metadata per slot, so a step is a fixed sequence of launches (CUDA graphs). A step may restart a slot up to KEEP
//! positions back inside its previous step (rejected drafts); see the kernel file.
//! Device memory per slot and layer (27B: 6.3 MB of recurrent state in two buffers, 1 MB of replay banks, 1.3 MB of
//! conv ring; padded to whole pages) is mapped while the slot holds a sequence (`map`, `unmap`).
use super::{CONV_K, Ctx, Dims, LK_HEADS, LKD, LVD};
use crate::checkpoint::Checkpoint;
use crate::cuda::core::{SPEC_TOK, cfg};
use crate::cuda::vmm::{Pool, Region};
use anyhow::{Result, ensure};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, PushKernelArg};
use std::sync::Arc;

/// Must match the kernels: conv ring positions, rollback reach (replay bank entries), value columns per delta_rule
/// block and lanes per column, ints of metadata per slot.
const CR: usize = 32;
const KEEP: usize = 16;
const CPB: usize = 16;
const LPC: usize = 4;
const META: usize = 4;
// a speculative step's drafts must be within the rollback reach
const _: () = assert!(KEEP + 1 >= SPEC_TOK);

/// The DeltaNet kernels (from the model's module)
pub struct Kernels {
    conv: CudaFunction,
    rule: CudaFunction,
    norm: CudaFunction,
}

impl Kernels {
    pub fn load(m: &Arc<CudaModule>) -> Result<Self> {
        Ok(Kernels {
            conv: m.load_function("delta_conv")?,
            rule: m.load_function("delta_rule")?,
            norm: m.load_function("delta_norm")?,
        })
    }
}

/// delta_conv's output row (the scratch `Bufs::cv` holds one per token): conv + SiLU of q, k, v (q and k l2-normed),
/// then exp(g) and beta per value head
pub fn cv_width(d: &Dims) -> usize {
    d.lqkv() + 2 * d.lv_heads
}

/// One Gated DeltaNet layer: conv weights, gates, norm and the per-slot conv and recurrent states.
pub struct DeltaNet {
    /// conv1d weight [lqkv][4]
    conv_w: CudaSlice<f32>,
    a_log: CudaSlice<f32>,
    dt_bias: CudaSlice<f32>,
    /// gated RMSNorm weight [LVD] (used as is, not 1 + w)
    norm_w: CudaSlice<f32>,
    /// per slot (`ss` floats each, whole pages; the kernels' SLOT_*): the recurrent states [2][lv_heads][LVD][LKD] (the
    /// end of the slot's last step and its checkpoint), per buffer the delta rule inputs from its checkpoint on
    /// [2][KEEP][k, v, exp(g), beta], the conv inputs at the last CR positions [CR][lqkv]
    mem: Region,
    ss: usize,
    /// floats of the parts: a state buffer, a bank, the ring
    parts: (usize, usize, usize),
    /// per slot: its memory mapped
    mapped: Vec<bool>,
    /// per slot: end of the last step (-1: none), checkpoint position, buffer holding the end state, unused
    meta: CudaSlice<i32>,
}

impl DeltaNet {
    /// `prefix`: e.g. "model.language_model.layers.0.linear_attn"
    pub fn new(
        s: &Arc<CudaStream>,
        pool: &Arc<Pool>,
        w: &Checkpoint,
        d: &Dims,
        prefix: &str,
        slots: usize,
    ) -> Result<Self> {
        let (lv_heads, lqkv) = (d.lv_heads, d.lqkv());
        // replay entry per position: k (per key head), v, exp(g), beta
        let rep = LK_HEADS * LKD + lv_heads * LVD + 2 * lv_heads;
        let f = |n: &str| Ok::<_, anyhow::Error>(w.floats(&format!("{prefix}.{n}"))?.0);
        let (conv_w, a_log, dt_bias, norm_w) = (f("conv1d.weight")?, f("A_log")?, f("dt_bias")?, f("norm.weight")?);
        ensure!(
            conv_w.len() == lqkv * 4 && a_log.len() == lv_heads && dt_bias.len() == lv_heads && norm_w.len() == LVD,
            "{prefix}: unexpected DeltaNet weight shapes"
        );
        let meta: Vec<i32> = (0..slots * META).map(|i| if i % META == 0 { -1 } else { 0 }).collect();
        let parts = (lv_heads * LVD * LKD, KEEP * rep, CR * lqkv);
        let page = pool.page() / 4;
        let ss = (2 * parts.0 + 2 * parts.1 + parts.2).div_ceil(page) * page;
        Ok(DeltaNet {
            conv_w: s.clone_htod(&conv_w)?,
            a_log: s.clone_htod(&a_log)?,
            dt_bias: s.clone_htod(&dt_bias)?,
            norm_w: s.clone_htod(&norm_w)?,
            mem: Region::reserve(s, pool, slots * ss * 4)?,
            ss,
            parts,
            mapped: vec![false; slots],
            meta: s.clone_htod(&meta)?,
        })
    }

    /// Read the in-projection's partial products `b.part` [split][n][delta_in] and those of [in_proj_b; in_proj_a] `b.ba`
    /// [ba_split][n][2 lv_heads], write `b.mix` [n][mix] (`b.o`: the raw output before the gated norm).
    pub fn run(&self, c: &Ctx, split: usize, ba_split: usize) -> Result<()> {
        let (s, st, b, cv) = (c.s, c.st, c.b, &c.b.cv);
        let k = &c.k.delta;
        let (lv_heads, lqkv) = (c.d.lv_heads, c.d.lqkv());
        let (sp, n, segs, bsp) = (split as i32, st.n as i32, st.segs as i32, ba_split as i32);
        let (mem, ss) = (self.mem.ptr(), self.ss as i64);
        unsafe {
            s.launch_builder(&k.conv)
                .arg(&b.part)
                .arg(&sp)
                .arg(&n)
                .arg(&b.ba)
                .arg(&bsp)
                .arg(&self.conv_w)
                .arg(&self.a_log)
                .arg(&self.dt_bias)
                .arg(&mem)
                .arg(&ss)
                .arg(cv)
                .arg(&st.tok_slot)
                .arg(&st.tok_pos)
                .launch(cfg((lqkv / 128, st.n.div_ceil(4), 1), 128))?;
            s.launch_builder(&k.rule)
                .arg(&b.part)
                .arg(&sp)
                .arg(&n)
                .arg(cv)
                .arg(&mem)
                .arg(&ss)
                .arg(&self.meta)
                .arg(&b.o)
                .arg(&st.seg_start)
                .arg(&st.seg_len)
                .arg(&st.seg_slot)
                .arg(&st.seg_pos0)
                .launch(cfg((LVD / CPB, lv_heads, st.segs), CPB * LPC))?;
            s.launch_builder(&k.norm)
                .arg(&b.part)
                .arg(&sp)
                .arg(&n)
                .arg(&self.norm_w)
                .arg(&b.o)
                .arg(&b.mix)
                .arg(&self.meta)
                .arg(&st.seg_len)
                .arg(&st.seg_slot)
                .arg(&st.seg_pos0)
                .arg(&segs)
                .launch(cfg((st.n, lv_heads / 16, 1), 16 * 32))?;
        }
        Ok(())
    }

    /// Device bytes a slot maps (its state buffers, replay banks and ring, in whole pages)
    pub fn slot_bytes(&self) -> usize {
        self.ss * 4
    }

    /// Map slot `slot`'s memory (zeroed when new); false when the pages ran out
    pub fn map(&mut self, slot: usize) -> Result<bool> {
        if !self.mapped[slot] {
            self.mapped[slot] = self.mem.map(slot * self.ss * 4, (slot + 1) * self.ss * 4)?;
        }
        Ok(self.mapped[slot])
    }

    /// Unmap slot `slot`'s memory
    pub fn unmap(&mut self, slot: usize) -> Result<()> {
        self.mapped[slot] = false;
        self.mem.unmap(slot * self.ss * 4, (slot + 1) * self.ss * 4)
    }

    /// Bytes of a slot's saved state (save, load): its end state and the conv ring's last CONV_K - 1 positions (all
    /// the next step reads of it)
    pub fn saved_bytes(&self) -> usize {
        (self.parts.0 + (CONV_K - 1) * self.parts.2 / CR) * 4
    }

    /// Append slot `slot`'s state after its last step (mapped) to `out`: the end state buffer and the conv ring's
    /// last positions; the position it is at
    pub fn save(&self, s: &Arc<CudaStream>, slot: usize, out: &mut Vec<u8>) -> Result<i32> {
        // (a copy from pages not mapped would end the process in the driver, not fail)
        ensure!(self.mapped[slot], "slot {slot}: no DeltaNet state (its memory is not mapped)");
        let meta = s.clone_dtoh(&self.meta.slice(slot * META..slot * META + META))?;
        let (at, n, row) = (out.len(), self.saved_bytes(), self.parts.2 / CR * 4);
        out.resize(at + n, 0);
        let (base, ring) = (slot * self.ss * 4, (2 * self.parts.0 + 2 * self.parts.1) * 4);
        let (state, rows) = out[at..].split_at_mut(self.parts.0 * 4);
        self.mem.read(base + meta[2] as usize * self.parts.0 * 4, state)?;
        for (j, r) in rows.chunks_exact_mut(row).enumerate() {
            let q = meta[0] - (CONV_K as i32 - 1) + j as i32;
            if q >= 0 {
                self.mem.read(base + ring + (q as usize & (CR - 1)) * row, r)?;
            }
        }
        Ok(meta[0])
    }

    /// Set slot `slot` (mapped) to a state from save at position `pos`: the next step continues there
    pub fn load(&mut self, s: &Arc<CudaStream>, slot: usize, data: &[u8], pos: i32) -> Result<()> {
        ensure!(
            data.len() == self.saved_bytes(),
            "a DeltaNet state of {} bytes, not {}",
            data.len(),
            self.saved_bytes()
        );
        let (base, ring, row) = (slot * self.ss * 4, (2 * self.parts.0 + 2 * self.parts.1) * 4, self.parts.2 / CR * 4);
        let (state, rows) = data.split_at(self.parts.0 * 4);
        self.mem.write(base, state)?;
        for (j, r) in rows.chunks_exact(row).enumerate() {
            let q = pos - (CONV_K as i32 - 1) + j as i32;
            if q >= 0 {
                self.mem.write(base + ring + (q as usize & (CR - 1)) * row, r)?;
            }
        }
        s.memcpy_htod(&[pos, pos, 0, 0], &mut self.meta.slice_mut(slot * META..slot * META + META))?;
        Ok(())
    }
}
