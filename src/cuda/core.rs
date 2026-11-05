//! What the CUDA models share: weight matrices packed for the tensor-core kernels (kernels/gemm.cuh) and their
//! launcher, a step's tokens, segments and logit rows on the device (`Step`), sampling on the GPU
//! (kernels/sampling.cuh) and the CUDA graph cache. The model (qwen/) adds its layers, caches and the kernels of its
//! own module, which instantiates the shared ones.
use crate::engine::Tok;
use crate::sample;
use crate::checkpoint::{Checkpoint, Quant};
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaContext, CudaFunction, CudaGraph, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg, sys};
use cudarc::nvrtc::Ptx;
use std::{cell::RefCell, collections::HashMap, hash::Hash, sync::Arc};

/// Most tokens in one step (prompt chunks and decode tokens).
pub const MAX_TOK: usize = 2048;
/// Most rows of logits in one step.
pub const MAX_OUT: usize = 64;
/// Sampled drafts' distributions kept per slot: draft indices, tokens each (kernels/sampling.cuh QD, QCAP)
const QD: usize = 16;
const QCAP: usize = 1024;
/// Tokens per step with speculative decoding at most (decode tokens plus drafts).
pub const SPEC_TOK: usize = 16;
/// Token tiles of the decode matrix kernels (gemm_dec); steps of up to the largest are replayed as CUDA graphs.
pub const DEC_TILES: [usize; 3] = [8, 16, 32];
pub const GRAPH_TOK: usize = 32;
/// Quantization group along a row (one f32 scale, for int4 also a zero point, per group); must match the kernels.
pub const QG: usize = 128;
/// Warps per block of the decode and prompt matrix kernels (one 16-row tile each); must match WARPS, PWARPS.
pub const WARPS: usize = 4;
const PWARPS: usize = 8;
/// prompt kernels: n8 token tiles per block (PNTT in the kernels); their dynamic shared memory: two chunks of X, 8 PNTT
/// tokens of 128 + 8 halves
const PNTT: usize = 8;
const PROMPT_SMEM: usize = 2 * 8 * PNTT * 136 * 2;
/// Decode matrix products stage at most this many columns of X per block (8-token tile; KD * 8 / tile in general);
/// must match KD in the kernels.
const KD: usize = 1024;

/// The GPU's stream (not the legacy default stream, which can't be captured into a graph) and a model's kernels
pub fn open(ptx: &str) -> Result<(Arc<CudaStream>, Arc<CudaModule>)> {
    let ctx = CudaContext::new(0)?;
    unsafe { ctx.disable_event_tracking() };
    Ok((ctx.new_stream()?, ctx.load_module(Ptx::from_src(ptx))?))
}

pub fn cfg(grid: (usize, usize, usize), block: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid.0 as u32, grid.1 as u32, grid.2 as u32),
        block_dim: (block as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Epilogue of a matrix product: plain (split-K partial products), SwiGLU of interleaved gate/up tiles, lm_head with
/// the argmax.
#[derive(Clone, Copy, PartialEq)]
pub enum Epi {
    Plain,
    Silu,
    Argmax,
}

/// How a matrix is stored for the kernels (the weight bits of their GEMM_SET)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Wt {
    F16,
    /// stored q + 128, an f32 scale per row and 128 columns
    I8,
    /// symmetric int4 (zero point 8) in Marlin's layout (pack_marlin; kernels/marlin.cuh), for modules with f16
    /// inputs that instantiate it (Qwen): every product of a step size
    Marlin,
    /// int4 with zero points (AWQ's, asymmetric) in Marlin's layout: the scales followed by (8 - z) * scale
    MarlinZ,
}

impl Wt {
    pub fn bits(self) -> usize {
        match self {
            Wt::F16 => 16,
            Wt::I8 => 8,
            Wt::Marlin | Wt::MarlinZ => 4,
        }
    }

    /// Columns a decode-kernel stage reads (8 loads of 16 bytes per lane: DEC_LOADS in the kernels): split-K ranges
    /// are multiples of it
    fn stage_cols(self) -> usize {
        2048 / self.bits()
    }
}

/// A weight matrix packed for the tensor-core kernels: tiles of 16 rows in mma A-fragment order (see Wt)
pub struct Mat {
    pub w: CudaSlice<u8>,
    pub s: CudaSlice<f32>,
    pub tiles: usize,
    pub cols: usize,
    pub wt: Wt,
    /// split-K factor for decode-sized steps, per DEC_TILES entry (Marlin: 1, its blocks add up their parts)
    pub split: [usize; 3],
    /// rows that count (argmax, draft sampling): all, or those filled so far of a draft head (Mat::marlin_rows)
    pub valid: usize,
}

/// Rows of a stacked matrix in the order they are packed: `silu`: [w1; w3] (gate; up), so that each tile holds 8 rows
/// of each for the SwiGLU epilogue
pub fn row_order(rows: usize, silu: bool) -> Vec<usize> {
    if silu {
        (0..rows).map(|i| (i / 16) * 8 + i % 8 + if i % 16 >= 8 { rows / 2 } else { 0 }).collect()
    } else {
        (0..rows).collect()
    }
}

impl Mat {
    fn upload(s: &Arc<CudaStream>, w: &[u8], sc: &[f32], rows: usize, cols: usize, wt: Wt) -> Result<Self> {
        let tiles = rows / 16;
        let split = match wt {
            Wt::Marlin | Wt::MarlinZ => [1; 3],
            _ => DEC_TILES.map(|t| split_for(tiles, cols, wt, KD * 8 / t)),
        };
        Ok(Mat { w: s.clone_htod(w)?, s: s.clone_htod(sc)?, tiles, cols, wt, split, valid: rows })
    }

    /// Pack `src` [rows][cols] (bf16 bits) as int8 or f16; `silu`: see row_order.
    pub fn new(s: &Arc<CudaStream>, src: &[u16], cols: usize, silu: bool, int8: bool) -> Result<Self> {
        let rows = src.len() / cols;
        ensure!(rows % 16 == 0 && cols % 256 == 0 && rows * cols == src.len(), "matrix of {rows} x {cols}");
        let (w, sc) = pack(src, cols, &row_order(rows, silu), int8);
        Mat::upload(s, &w, &sc, rows, cols, if int8 { Wt::I8 } else { Wt::F16 })
    }

    /// Pack `src` [rows][cols] (f32) as f16, rows in order (a LoRA's A)
    pub fn f16(s: &Arc<CudaStream>, src: &[f32], cols: usize) -> Result<Self> {
        let rows = src.len() / cols;
        ensure!(rows % 16 == 0 && cols % 256 == 0 && rows * cols == src.len(), "matrix of {rows} x {cols}");
        let (w, sc) = pack(src, cols, &row_order(rows, false), false);
        Mat::upload(s, &w, &sc, rows, cols, Wt::F16)
    }

    /// Rows `rows` (whole tiles) of a stored int4 matrix that Mat::marlin takes, in a matrix of `cap` rows (a multiple
    /// of 256): the rest are room for later rows (Gemm::put_block), not `valid`, and their outputs do not count
    /// (`zeros`: its zero points, Quant::zero_points)
    pub fn marlin_rows(s: &Arc<CudaStream>, q: &Quant, zeros: &[u8], rows: &[usize], cap: usize) -> Result<Self> {
        ensure!(q.bits == 4, "Marlin packs int4 matrices, not {}-bit ones", q.bits);
        ensure!(rows.len() % 16 == 0 && cap % 256 == 0 && rows.len() <= cap && !rows.is_empty(),
                "{} rows (room for {cap})", rows.len());
        let sym = zeros.iter().all(|&z| z == 8);
        let order: Vec<usize> = rows.iter().copied().chain(std::iter::repeat(rows[0])).take(cap).collect();
        let (w, sc) = pack_marlin(q, &order, (!sym).then_some(zeros));
        let mut m = Mat::upload(s, &w, &sc, cap, q.cols, if sym { Wt::Marlin } else { Wt::MarlinZ })?;
        m.valid = rows.len();
        Ok(m)
    }

    /// A stored int4 matrix in Marlin's layout, if Marlin takes it: integer zero points (8 everywhere, compressed-tensors'
    /// symmetric int4: Wt::Marlin; else AWQ's and GPTQ's asymmetric ones: Wt::MarlinZ), groups of a multiple of 128
    /// columns, rows a multiple of 256 (Marlin's 256-output tiles of steps over 16 tokens), columns of 128; `silu`: see
    /// row_order.
    pub fn marlin(s: &Arc<CudaStream>, q: &Quant, silu: bool) -> Result<Option<Self>> {
        let fits = q.bits == 4 && q.group % QG == 0 && q.rows % 256 == 0 && q.cols % 128 == 0;
        let Some(zeros) = fits.then(|| q.zero_points()).flatten() else { return Ok(None) };
        let sym = zeros.iter().all(|&z| z == 8);
        let (w, sc) = pack_marlin(q, &row_order(q.rows, silu), (!sym).then_some(&zeros[..]));
        Ok(Some(Mat::upload(s, &w, &sc, q.rows, q.cols, if sym { Wt::Marlin } else { Wt::MarlinZ })?))
    }

    /// A stored symmetric int8 matrix as it is (compressed-tensors: value + 128, a scale per row and group of a multiple
    /// of 128 columns; the int8 kernels' format); `silu`: see row_order.
    pub fn int8(s: &Arc<CudaStream>, q: &Quant, silu: bool) -> Result<Option<Self>> {
        let fits = q.bits == 8 && q.group % QG == 0 && q.rows % 16 == 0 && q.cols % Wt::I8.stage_cols() == 0;
        if !fits || q.zero_points().is_none_or(|z| z.iter().any(|&z| z != 128)) {
            return Ok(None);
        }
        let (order, cols, groups, qgroups) = (row_order(q.rows, silu), q.cols, q.cols / QG, q.cols / q.group);
        let value = |row: usize, c: usize| (q.q[row * cols / 4 + c / 4] >> ((c % 4) * 8)) as u8;
        let tiles = q.rows / 16;
        let mut w = vec![0u8; tiles * 16 * cols];
        let mut sc = vec![0f32; tiles * groups * 16];
        for tile in 0..tiles {
            let rows = &order[tile * 16..][..16];
            for (r, &row) in rows.iter().enumerate() {
                for gi in 0..groups {
                    sc[(tile * groups + gi) * 16 + r] = q.scales[row * qgroups + gi * QG / q.group];
                }
            }
            let mut o = tile * 16 * cols;
            for k32 in 0..cols / 32 {
                for lane in 0..32 {
                    for ks in 0..2 {
                        for (r, c) in frag(k32 * 32 + ks * 16, lane) {
                            w[o] = value(rows[r], c);
                            o += 1;
                        }
                    }
                }
            }
        }
        Ok(Some(Mat::upload(s, &w, &sc, q.rows, q.cols, Wt::I8)?))
    }

    /// Matrices `names` (`.weight`) of the checkpoint with the same column count, stacked by rows: stored int4 in
    /// Marlin's layout (Mat::marlin; an error where it does not fit), stored int8 as it is; else dense (dequantized
    /// when stored quantized), int8 or f16. The 16-bit matrices of a pre-quantized checkpoint stay f16: its maker kept
    /// them precise (RedHatAI's Qwen3.5 w4a16: the DeltaNet projections; requantized to int8 they cost as much as all
    /// the int4 matrices). `silu`: see row_order.
    pub fn stack(s: &Arc<CudaStream>, ck: &Checkpoint, names: &[String], silu: bool, int8: bool) -> Result<Self> {
        let bases: Vec<&str> = names.iter().filter_map(|n| n.strip_suffix(".weight")).collect();
        if bases.len() == names.len() && bases.iter().all(|b| ck.is_quant(b)) {
            let parts = bases.iter().map(|b| ck.quant_matrix(b)).collect::<Result<Vec<_>>>()?;
            let q = Quant::stack(parts)?;
            if q.bits == 4 {
                return Mat::marlin(s, &q, silu)?.with_context(|| format!(
                    "{}: int4 of {} x {} in groups of {}: Marlin needs rows a multiple of 256, columns of 128 and groups \
                     of a multiple of 128 with integer zero points", names[0], q.rows, q.cols, q.group));
            }
            if let Some(m) = Mat::int8(s, &q, silu).with_context(|| names[0].clone())? {
                return Ok(m);
            }
        }
        let (mut v, mut cols) = (Vec::new(), 0);
        for n in names {
            let (w, shape) = ck.bf16(n)?;
            v.extend(w);
            cols = shape[1];
        }
        Mat::new(s, &v, cols, silu, int8 && !ck.prequantized()).with_context(|| names[0].clone())
    }

}

/// a0..a7 of the mma A fragment of lane (g, t) for the k16 step at column k0: (row, column) in the tile
fn frag(k0: usize, lane: usize) -> [(usize, usize); 8] {
    let (g, t) = (lane / 4, lane % 4);
    let c = k0 + 2 * t;
    [(g, c), (g, c + 1), (g + 8, c), (g + 8, c + 1), (g, c + 8), (g, c + 9), (g + 8, c + 8), (g + 8, c + 9)]
}

/// Pack a stored int4 matrix's rows `order` (Marlin's outputs n) for Marlin, as Marlin's `Layer.pack`
/// (with `_get_perms`) lays out B = W^T [k][n]: per k16 step, per 64 outputs, lane l holds 4 words, one per 16-output
/// block j, with the mma B fragments of outputs 16 j + g (n8 tile 0) and 16 j + 8 + g (tile 1), g = l / 4, at columns
/// 2t, 2t + 1, 2t + 8, 2t + 9 (t = l % 4). In a word, nibble x (the stored q, i.e. q + 8 of the signed value) is
/// tile (x >> 1) & 1, column 2t + 8 (x & 1) + (x >> 2): dequant takes nibbles 0, 4 then 1, 5 (tile 0) and the rest
/// shifted by 8 (tile 1). Scales f16 [k / 128][n] (pairs in the f32 slots), per 64 outputs output c at position
/// 8 (c % 8) + c / 8: each lane's int4 of 8 scales covers its 4 blocks' two n8 tiles (marlin::fetch_to_registers).
/// With `zeros` (a zero point per row and group, as the scales), the scales are followed by (8 - z) * scale in the
/// same layout (exact in f16: |8 - z| has at most 3 significant bits; marlin::scale_z).
fn pack_marlin(q: &Quant, order: &[usize], zeros: Option<&[u8]>) -> (Vec<u8>, Vec<f32>) {
    let (n, k, groups) = (order.len(), q.cols, q.cols / QG);
    let (qgroups, words) = (q.cols / q.group, q.cols / 8);
    let value = |row: usize, c: usize| q.q[row * words + c / 8] >> ((c % 8) * 4) & 15;
    let mut sc = vec![0u16; groups * n * if zeros.is_some() { 2 } else { 1 }];
    for (o, &row) in order.iter().enumerate() {
        let (c64, c) = (o / 64, o % 64);
        for gi in 0..groups {
            let (at, g) = (c64 * 64 + 8 * (c % 8) + c / 8, row * qgroups + gi * QG / q.group);
            sc[gi * n + at] = half::f16::from_f32(q.scales[g]).to_bits();
            if let Some(z) = zeros {
                sc[(groups + gi) * n + at] = half::f16::from_f32((8.0 - z[g] as f32) * q.scales[g]).to_bits();
            }
        }
    }
    let sc = sc.chunks(2).map(|p| f32::from_bits(p[0] as u32 | (p[1] as u32) << 16)).collect();
    // one k16 step: 2 n words
    let step_bytes = 2 * n * 4;
    let mut w = vec![0u8; k / 16 * step_bytes];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let per = (k / 16).div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (i, wc) in w.chunks_mut(per * step_bytes).enumerate() {
            scope.spawn(move || {
                for (j, ws) in wc.chunks_mut(step_bytes).enumerate() {
                    let k0 = (i * per + j) * 16;
                    let mut o = 0;
                    for c64 in 0..n / 64 {
                        for lane in 0..32 {
                            let (g, t) = (lane / 4, lane % 4);
                            for blk in 0..4 {
                                let mut word = 0u32;
                                for x in 0..8 {
                                    let out = c64 * 64 + blk * 16 + g + 8 * ((x >> 1) & 1);
                                    word |= value(order[out], k0 + 2 * t + 8 * (x & 1) + (x >> 2)) << (4 * x);
                                }
                                ws[o..o + 4].copy_from_slice(&word.to_le_bytes());
                                o += 4;
                            }
                        }
                    }
                }
            });
        }
    });
    (w, sc)
}

/// Floats of the partial products the plain products of `mats` write at most: split K for decode-sized steps, one
/// product for prompt chunks of up to MAX_TOK tokens
pub fn part_len<'a>(mats: impl IntoIterator<Item = &'a Mat>) -> usize {
    let rows = |m: &Mat| DEC_TILES.iter().zip(m.split).map(|(&n, split)| split * n).max().unwrap_or(0).max(MAX_TOK);
    mats.into_iter().map(|m| rows(m) * m.tiles * 16).max().unwrap_or(0)
}

/// A matrix element as stored on the host: bf16 bits or f32
trait Val: Copy + Sync {
    fn f(self) -> f32;
}

impl Val for u16 {
    fn f(self) -> f32 {
        f32::from_bits((self as u32) << 16)
    }
}

impl Val for f32 {
    fn f(self) -> f32 {
        self
    }
}

/// Pack the rows `order` (16 per tile) of `src` [rows][cols] (bf16 or f32).
fn pack<V: Val>(src: &[V], cols: usize, order: &[usize], int8: bool) -> (Vec<u8>, Vec<f32>) {
    let tiles = order.len() / 16;
    let groups = cols / QG;
    let tile_bytes = 16 * cols * if int8 { 1 } else { 2 };
    let mut w = vec![0u8; tiles * tile_bytes];
    let mut s = vec![0f32; tiles * groups * 16];
    let f = |b: V| b.f();
    let work = |tile: usize, wt: &mut [u8], st: &mut [f32]| {
        let rows: Vec<&[V]> = (0..16).map(|r| &src[order[tile * 16 + r] * cols..][..cols]).collect();
        // int8: q[r][c] + 128 with a scale per row and group
        let mut q = vec![0u8; if int8 { 16 * cols } else { 0 }];
        if int8 {
            for (r, row) in rows.iter().enumerate() {
                for gi in 0..groups {
                    let g = &row[gi * QG..(gi + 1) * QG];
                    let max = g.iter().fold(0f32, |m, &x| m.max(f(x).abs()));
                    let scale = if max > 0.0 { max / 127.0 } else { 1.0 };
                    st[gi * 16 + r] = scale;
                    for (c, &x) in g.iter().enumerate() {
                        q[r * cols + gi * QG + c] = ((f(x) / scale).round().clamp(-127.0, 127.0) as i32 + 128) as u8;
                    }
                }
            }
        }
        let mut o = 0;
        if int8 {
            for k32 in 0..cols / 32 {
                for lane in 0..32 {
                    for ks in 0..2 {
                        for (r, c) in frag(k32 * 32 + ks * 16, lane) {
                            wt[o] = q[r * cols + c];
                            o += 1;
                        }
                    }
                }
            }
        } else {
            for k16 in 0..cols / 16 {
                for lane in 0..32 {
                    for (r, c) in frag(k16 * 16, lane) {
                        let h = half::f16::from_f32(f(rows[r][c])).to_bits().to_le_bytes();
                        wt[o..o + 2].copy_from_slice(&h);
                        o += 2;
                    }
                }
            }
        }
    };
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let per = tiles.div_ceil(threads);
    std::thread::scope(|sc| {
        for (i, (wc, scc)) in w.chunks_mut(per * tile_bytes).zip(s.chunks_mut(per * groups * 16)).enumerate() {
            sc.spawn(move || {
                for (j, (wt, st)) in wc.chunks_mut(tile_bytes).zip(scc.chunks_mut(groups * 16)).enumerate() {
                    work(i * per + j, wt, st);
                }
            });
        }
    });
    (w, s)
}

/// Split-K factor for decode-sized steps: enough (tile, split) units to fill the GPU, splits a whole number of the
/// kernel's column chunks and at most `kd` columns (what a block stages).
fn split_for(tiles: usize, cols: usize, wt: Wt, kd: usize) -> usize {
    const UNITS: usize = 1024;
    let chunks = cols / wt.stage_cols();
    let want = UNITS.div_ceil(tiles).max(cols.div_ceil(kd));
    (1..=chunks).find(|d| chunks % d == 0 && *d >= want).unwrap_or(chunks)
}

/// Marlin's tile configurations (kernels/marlin.cuh MARLIN_SET): 16-token, 16-output and 16-column blocks. Steps of
/// up to 16 tokens take the first, larger ones the others, one per 16-token blocks of a problem (1 to 4).
const MARLIN_CFG: [(usize, usize, usize); 5] = [(1, 8, 8), (1, 16, 4), (2, 16, 4), (3, 16, 4), (4, 16, 4)];
/// Floats of a Marlin scratch slot (SLOT in the kernels)
const MARLIN_SLOT: usize = 64 * 256;

/// Dynamic shared memory of a Marlin configuration: 4 pipeline stages of the A tile (16 mb tokens x 16 kb columns
/// f16), the B tile (16 kb x 16 nb int4) and a row of scales (16 nb f16), in 16-byte units (the in-block reduction
/// reuses the first 32 KB)
fn marlin_smem((mb, nb, kb): (usize, usize, usize), zp: bool) -> usize {
    4 * 16 * (32 * kb * mb + 8 * nb * kb + 2 * nb * if zp { 2 } else { 1 })
}

/// Marlin's kernels [configuration][plain, silu], the grid (a block per SM) and the slots and locks with which the
/// blocks that share an output slice add up their parts (one per block, see marlin::global_reduce)
struct Marlin {
    /// per configuration: plain, silu, argmax, then the same with zero points
    f: Vec<CudaFunction>,
    sms: usize,
    red: CudaSlice<f32>,
    locks: CudaSlice<i32>,
    /// an argmax product's rows of X, contiguous (Marlin reads its A rows so): [MAX_OUT][max_cols] f16
    rows: CudaSlice<u16>,
    gather: CudaFunction,
    put: CudaFunction,
}

/// The matrix kernels of a model's module (GEMM_SET in its .cu): [weights: f16, i8, i4][token tile: t8, t16, t32,
/// t64][epilogue]; and Marlin's, if the module has them (MARLIN_SET: Qwen)
pub struct Gemm(Vec<CudaFunction>, Option<Marlin>);

impl Gemm {
    /// `marlin`: the module has Marlin's kernels (matrices Wt::Marlin, Wt::MarlinZ), for products of at most
    /// `max_cols` columns
    pub fn load(s: &Arc<CudaStream>, m: &Arc<CudaModule>, marlin: bool, max_cols: usize) -> Result<Self> {
        let marlin = match marlin {
            true => {
                let mut f = Vec::new();
                for cfg in MARLIN_CFG {
                    for (epi, zp) in [("plain", false), ("silu", false), ("argmax", false), ("plain_z", true),
                                      ("silu_z", true), ("argmax_z", true)] {
                        let k = m.load_function(&format!("marlin_m{}_n{}_{epi}", cfg.0, cfg.1))?;
                        k.set_attribute(sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                        marlin_smem(cfg, zp) as i32)?;
                        f.push(k);
                    }
                }
                let sms = s.context().attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as usize;
                Some(Marlin { f, sms, red: s.alloc_zeros(sms * MARLIN_SLOT)?, locks: s.alloc_zeros(sms)?,
                              rows: s.alloc_zeros(MAX_OUT * max_cols)?, gather: m.load_function("marlin_gather")?,
                              put: m.load_function("marlin_put_block")? })
            }
            false => None,
        };
        let mut f = Vec::new();
        for wt in ["f16", "i8"] {
            for tt in ["t8", "t16", "t32", "t64"] {
                for epi in ["plain", "silu", "argmax"] {
                    let k = m.load_function(&format!("gemm_{wt}_{tt}_{epi}"))?;
                    if tt == "t64" {
                        k.set_attribute(sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                        PROMPT_SMEM as i32)?;
                    }
                    f.push(k);
                }
            }
        }
        Ok(Gemm(f, marlin))
    }

    /// y = w . x for the step's tokens (Argmax: its logit rows, keeping the running argmax in `st.keys`); split-K
    /// partial products for decode-sized steps. Returns the split factor.
    pub fn run<X: DeviceRepr, Y: DeviceRepr>(&self, s: &Arc<CudaStream>, st: &Step, w: &Mat, x: &CudaSlice<X>,
                                            y: &CudaSlice<Y>, epi: Epi) -> Result<usize> {
        self.run_split(s, st, w, x, y, epi, false)
    }

    /// run; `whole`: a plain product without split K, accumulated as the SwiGLU epilogue's (the same values)
    #[allow(clippy::too_many_arguments)]
    pub fn run_split<X: DeviceRepr, Y: DeviceRepr>(&self, s: &Arc<CudaStream>, st: &Step, w: &Mat, x: &CudaSlice<X>,
                                                  y: &CudaSlice<Y>, epi: Epi, whole: bool) -> Result<usize> {
        if matches!(w.wt, Wt::Marlin | Wt::MarlinZ) {
            if epi == Epi::Argmax {
                // the logit rows of the step's tokens, gathered: Marlin reads A rows contiguously
                let mk = self.1.as_ref().context("a Marlin matrix in a module without Marlin's kernels")?;
                let (n, k8) = (st.n_out, (w.cols / 8) as i32);
                ensure!(n * w.cols <= mk.rows.len(), "{n} rows of {} columns to gather", w.cols);
                if n == 0 {
                    return Ok(1);
                }
                unsafe {
                    s.launch_builder(&mk.gather).arg(x).arg(&st.out_rows).arg(&mk.rows).arg(&k8)
                        .launch(cfg((n, 1, 1), 128))?;
                }
                self.marlin(s, n, w, &mk.rows, y, epi, &st.keys)?;
            } else {
                self.marlin(s, st.n, w, x, y, epi, &st.keys)?;
            }
            return Ok(1);
        }
        let lm = epi == Epi::Argmax;
        let n = if lm { st.n_out } else { st.n };
        let tile = DEC_TILES.iter().position(|&t| n <= t);
        let split = match tile {
            Some(v) if epi == Epi::Plain && !whole => w.split[v],
            _ => 1,
        };
        let variant = tile.unwrap_or(DEC_TILES.len());
        let f = &self.0[(w.wt as usize * 4 + variant) * 3 + epi as usize];
        let ldy = match epi {
            Epi::Silu => w.tiles * 8,
            _ => w.tiles * 16,
        } as i32;
        let (cols, tiles, ni, kper) = (w.cols as i32, w.tiles as i32, n as i32, (w.cols / split) as i32);
        let null = 0u64;
        let mut l = s.launch_builder(f);
        l.arg(&w.w).arg(&w.s).arg(x);
        if lm {
            l.arg(&st.out_rows);
        } else {
            l.arg(&null);
        }
        l.arg(&cols).arg(&tiles).arg(&ni).arg(&kper).arg(y).arg(&ldy).arg(&st.keys);
        let grid_z = if tile.is_some() { 1 } else { n.div_ceil(8 * PNTT) };
        let warps = if tile.is_some() { WARPS } else { PWARPS };
        let lc = cfg((w.tiles.div_ceil(warps), split, grid_z), warps * 32);
        let lc = if tile.is_some() { lc } else { LaunchConfig { shared_mem_bytes: PROMPT_SMEM as u32, ..lc } };
        unsafe { l.launch(lc) }?;
        Ok(split)
    }

    /// Y = w . X for `n` tokens with Marlin, as Marlin's launcher splits it: runs of 64-token problems side by side in
    /// one kernel (`parallel` in the kernel; Marlin's own launcher caps them at 16, one launch serves MAX_TOK here),
    /// the rest (a partial 16-token block in it) in a second; steps of up to 16 tokens with the 128 x 128 tiles.
    /// Plain: y f32 [n][rows] (one part); Silu: f16 [n][rows / 2].
    /// Argmax: y f32 [n][rows] and the running argmax over the `valid` rows in `keys`.
    fn marlin<X: DeviceRepr, Y: DeviceRepr>(&self, s: &Arc<CudaStream>, n: usize, w: &Mat, x: &CudaSlice<X>,
                                           y: &CudaSlice<Y>, epi: Epi, keys: &CudaSlice<u64>) -> Result<()> {
        let mk = self.1.as_ref().context("a Marlin matrix in a module without Marlin's kernels")?;
        let (blocks, pad) = (n.div_ceil(16), n.div_ceil(16) * 16 - n);
        let (rows, cols) = (w.tiles as i32 * 16, w.cols as i32);
        let mut i = 0;
        while i < blocks {
            let (mut mb, mut m, mut par) = (blocks - i, n - 16 * i, 1);
            if mb > 4 {
                // full 64-token problems only (Marlin's `parallel` takes no partial block)
                par = (16 * mb - pad) / 64;
                (m, mb) = (64 * par, 4);
            }
            let c = if n <= 16 { 0 } else { mb };
            let (m, row0) = (m as i32, 16 * i as i32);
            let (zp, valid) = (w.wt == Wt::MarlinZ, w.valid as i32);
            let mut l = s.launch_builder(&mk.f[6 * c + 3 * zp as usize + epi as usize]);
            l.arg(x).arg(&w.w).arg(y).arg(&w.s).arg(&m).arg(&rows).arg(&cols).arg(&row0).arg(&mk.red).arg(&mk.locks)
                .arg(keys).arg(&valid);
            let lc = LaunchConfig { shared_mem_bytes: marlin_smem(MARLIN_CFG[c], zp) as u32, ..cfg((mk.sms, 1, 1), 256) };
            unsafe { l.launch(lc) }?;
            i += 4 * par;
        }
        Ok(())
    }
}

impl Gemm {
    /// Write the stored rows `rows` (16 of them) of `q` as the 16-row block `block` of a Marlin matrix from
    /// Mat::marlin_rows (a draft head learning a row; its quantization, `zeros` its zero points), counting it as valid.
    /// `stage`: device buffers of block_stage's sizes the block goes through (allocated once: no device memory is
    /// taken while serving)
    #[allow(clippy::too_many_arguments)]
    pub fn put_block(&self, s: &Arc<CudaStream>, m: &mut Mat, q: &Quant, zeros: &[u8], block: usize, rows: &[usize],
                     stage: &mut (CudaSlice<u32>, CudaSlice<u16>)) -> Result<()> {
        let mk = self.1.as_ref().context("a Marlin matrix in a module without Marlin's kernels")?;
        let (n, k, groups) = (m.tiles * 16, q.cols, q.cols / QG);
        ensure!(rows.len() == 16 && (block + 1) * 16 <= n && k == m.cols, "block {block} of a matrix of {n} rows");
        let zp = m.wt == Wt::MarlinZ;
        ensure!(zp || rows.iter().all(|&r| (0..q.cols / q.group).all(|g| zeros[r * (q.cols / q.group) + g] == 8)),
                "rows with zero points for a symmetric matrix");
        let (qgroups, words) = (q.cols / q.group, q.cols / 8);
        let value = |row: usize, c: usize| q.q[row * words + c / 8] >> ((c % 8) * 4) & 15;
        let blk = block % 4;
        let mut w = vec![0u32; k / 16 * 32];
        for k16 in 0..k / 16 {
            for lane in 0..32 {
                let (g, t) = (lane / 4, lane % 4);
                for x in 0..8 {
                    let r = g + 8 * ((x >> 1) & 1);
                    w[k16 * 32 + lane] |= value(rows[r], k16 * 16 + 2 * t + 8 * (x & 1) + (x >> 2)) << (4 * x);
                }
            }
        }
        let tables = if zp { 2 } else { 1 };
        let mut sc = vec![0u16; tables * groups * 16];
        for (r, &row) in rows.iter().enumerate() {
            for gi in 0..groups {
                let g = row * qgroups + gi * QG / q.group;
                sc[gi * 16 + r] = half::f16::from_f32(q.scales[g]).to_bits();
                if zp {
                    sc[(groups + gi) * 16 + r] = half::f16::from_f32((8.0 - zeros[g] as f32) * q.scales[g]).to_bits();
                }
            }
        }
        ensure!(stage.0.len() >= w.len() && stage.1.len() >= sc.len(), "a staging buffer too small for a block");
        s.memcpy_htod(&w, &mut stage.0.slice_mut(0..w.len()))?;
        s.memcpy_htod(&sc, &mut stage.1.slice_mut(0..sc.len()))?;
        let (wd, sd) = (&stage.0, &stage.1);
        let (ni, c64, bi, gi, ti) = (n as i32, (block / 4) as i32, blk as i32, groups as i32, tables as i32);
        unsafe {
            s.launch_builder(&mk.put).arg(&m.w).arg(&m.s).arg(wd).arg(sd).arg(&ni).arg(&c64).arg(&bi).arg(&gi).arg(&ti)
                .launch(cfg((k / 16, 1, 1), 32))?;
        }
        m.valid = m.valid.max((block + 1) * 16);
        Ok(())
    }

    /// The staging buffers put_block needs for blocks of matrix `m` (packed words, scales and zero points)
    pub fn block_stage(s: &Arc<CudaStream>, m: &Mat) -> Result<(CudaSlice<u32>, CudaSlice<u16>)> {
        let tables = if m.wt == Wt::MarlinZ { 2 } else { 1 };
        Ok((s.alloc_zeros(m.cols / 16 * 32)?, s.alloc_zeros(tables * m.cols / QG * 16)?))
    }
}

/// How a model's attention takes a step's segments: segments of at most `short` tokens go to the decode attention
/// (each token on its own, at most `max_dec` tokens per step), longer ones to the prompt attention in tiles of
/// `ptile` tokens.
#[derive(Clone, Copy)]
pub struct Segments {
    pub short: usize,
    pub max_dec: usize,
    pub ptile: usize,
    /// a short segment's tokens share `pos / cut` (the decode attention's key splits depend on the position alone and
    /// change at multiples of `cut` keys; usize::MAX: no cuts)
    pub cut: usize,
}

/// A step's tokens, segments (the tokens of one sequence, contiguous and in order) and logit rows, on the device for
/// the kernels and as counts on the host; and the sampler's state: logits, argmax keys, the tokens seen per slot.
pub struct Step {
    /// tokens, logit rows, segments, decode-attention tokens, prompt-attention tiles, longest segment
    pub n: usize,
    pub n_out: usize,
    pub segs: usize,
    pub ndec: usize,
    pub ntile: usize,
    pub max_seg: usize,
    /// Smallest and largest position among the decode-attention tokens; largest among the prompt-tile tokens
    pub dec_min_pos: usize,
    pub dec_max_pos: usize,
    pub tile_max_pos: usize,
    pub ids: CudaSlice<i32>,
    /// per token: its slot and position
    pub tok_slot: CudaSlice<i32>,
    pub tok_pos: CudaSlice<i32>,
    /// per segment: first token, count, slot, position of the first token
    pub seg_start: CudaSlice<i32>,
    pub seg_len: CudaSlice<i32>,
    pub seg_slot: CudaSlice<i32>,
    pub seg_pos0: CudaSlice<i32>,
    /// tokens of short segments (decode attention)
    pub dtok: CudaSlice<i32>,
    /// short segments: (first index into dtok, tokens); per dtok entry the keys of its segment's last token
    pub dseg: CudaSlice<i32>,
    pub dkeys: CudaSlice<i32>,
    pub ndseg: usize,
    /// tokens of the longest short segment
    pub dmax: usize,
    /// prompt attention tiles: (first token, count)
    pub ptile: CudaSlice<i32>,
    /// tokens whose logits the step computes
    pub out_rows: CudaSlice<i32>,
    /// [out rows][vocab]
    pub logits: CudaSlice<f32>,
    /// Per output row (ordered logit << 32 | !index), maximized by lm_head: the greedy next token.
    pub keys: CudaSlice<u64>,
    /// Per slot, a bit per vocabulary entry: tokens fed so far (repetition penalty)
    seen: CudaSlice<u32>,
    /// Per slot, a bit per vocabulary entry: tokens its request generated (presence penalty; set by the host)
    outs: CudaSlice<u32>,
    /// Tokens to add to the seen sets, per segment mark[mark_at[i] .. mark_at[i + 1]]
    mark: CudaSlice<i32>,
    mark_at: CudaSlice<i32>,
    /// Per output row: the first token of its segment
    row_first: CudaSlice<i32>,
    /// Per output row: temperature, top_k, top_p, repetition penalty, two uniform draws, draft token, presence penalty
    sparams: CudaSlice<f32>,
    sampled: CudaSlice<i32>,
    accepted: CudaSlice<i32>,
    /// Sampled drafts' distributions per slot and draft index (kernels/sampling.cuh QD, QCAP): the token drawn, the
    /// count of kept tokens, their ids (ascending) and probabilities
    qtok: CudaSlice<i32>,
    qn: CudaSlice<i32>,
    qid: CudaSlice<i32>,
    qp: CudaSlice<f32>,
    dparams: CudaSlice<f32>,
    drafts: CudaSlice<i32>,
    mark_seen: CudaFunction,
    sample_rows: CudaFunction,
    draft_sample: CudaFunction,
    pub vocab: usize,
    pub slots: usize,
    pub max_ctx: usize,
    layout: Segments,
    /// the step has tokens of slots that use a LoRA adapter (set by the model after upload; cuda::qwen::lora)
    pub lora: bool,
}

impl Step {
    /// The sampler kernels come from the model's module (sampling.cuh, built for its vocabulary).
    pub fn new(s: &Arc<CudaStream>, m: &Arc<CudaModule>, slots: usize, max_ctx: usize, vocab: usize, layout: Segments)
    -> Result<Self> {
        let zi = |n: usize| s.alloc_zeros::<i32>(n);
        Ok(Step {
            n: 0, n_out: 0, segs: 0, ndec: 0, ntile: 0, max_seg: 0, dec_min_pos: 0, dec_max_pos: 0, tile_max_pos: 0, ndseg: 0, dmax: 0,
            ids: zi(MAX_TOK)?,
            tok_slot: zi(MAX_TOK)?,
            tok_pos: zi(MAX_TOK)?,
            seg_start: zi(MAX_TOK)?,
            seg_len: zi(MAX_TOK)?,
            seg_slot: zi(MAX_TOK)?,
            seg_pos0: zi(MAX_TOK)?,
            dtok: zi(MAX_TOK)?,
            dseg: zi(2 * MAX_TOK)?,
            dkeys: zi(MAX_TOK)?,
            ptile: zi(2 * MAX_TOK)?,
            out_rows: zi(MAX_OUT)?,
            logits: s.alloc_zeros(MAX_OUT * vocab)?,
            keys: s.alloc_zeros(MAX_OUT)?,
            seen: s.alloc_zeros(slots * vocab / 32)?,
            outs: s.alloc_zeros(slots * vocab / 32)?,
            mark: zi(MAX_TOK + 8 * MAX_OUT)?,
            mark_at: zi(MAX_TOK + 1)?,
            row_first: zi(MAX_OUT)?,
            sparams: s.alloc_zeros(MAX_OUT * 8)?,
            sampled: zi(MAX_OUT)?,
            accepted: zi(MAX_OUT)?,
            qtok: s.clone_htod(&vec![-1i32; slots * QD])?,
            qn: zi(slots * QD)?,
            qid: zi(slots * QD * QCAP)?,
            qp: s.alloc_zeros(slots * QD * QCAP)?,
            dparams: s.alloc_zeros(MAX_OUT * 8)?,
            drafts: zi(MAX_OUT)?,
            mark_seen: m.load_function("mark_seen")?,
            sample_rows: m.load_function("sample_rows")?,
            draft_sample: m.load_function("draft_sample")?,
            vocab,
            slots,
            max_ctx,
            layout,
            lora: false,
        })
    }

    /// The host-side shape of the step: what changes the launched kernels (part of the graph keys)
    pub fn shape(&self) -> [usize; 6] {
        [self.n, self.n_out, self.segs, self.ndec, self.ntile, self.max_seg]
    }

    /// Copy a step's tokens, segments and logit rows to the device. Tokens of one sequence must be contiguous and in
    /// order; `out`: indices into `toks` whose logits are wanted; `marks`: per segment, the tokens to add to its slot's
    /// seen set (None: all of the segment's tokens).
    pub fn upload(&mut self, s: &Arc<CudaStream>, toks: &[Tok], out: &[usize], marks: Option<&[Vec<u32>]>) -> Result<()> {
        let n = toks.len();
        ensure!((1..=MAX_TOK).contains(&n), "step of {n} tokens");
        ensure!(out.len() <= MAX_OUT && out.iter().all(|&o| o < n), "logit rows {out:?}");
        let mut seg = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for (i, t) in toks.iter().enumerate() {
            ensure!((t.slot as usize) < self.slots, "slot {}", t.slot);
            ensure!((t.pos as usize) < self.max_ctx, "position {} beyond the context", t.pos);
            if i > 0 && toks[i - 1].slot == t.slot {
                ensure!(toks[i - 1].pos + 1 == t.pos, "positions of slot {} not consecutive", t.slot);
                *seg[1].last_mut().unwrap() += 1;
            } else {
                ensure!(!seg[2].contains(&(t.slot as i32)), "slot {} in two segments", t.slot);
                seg[0].push(i as i32);
                seg[1].push(1);
                seg[2].push(t.slot as i32);
                seg[3].push(t.pos as i32);
            }
        }
        // short segments: decode attention; longer ones: prompt attention in tiles of `ptile` tokens
        let Segments { short, max_dec, ptile: tile, cut } = self.layout;
        let (mut dtok, mut ptile, mut dseg, mut dkeys) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (&st, &len) in seg[0].iter().zip(&seg[1]) {
            if len as usize <= short && dtok.len() + len as usize <= max_dec {
                let mut a = st;
                while a < st + len {
                    let part = toks[a as usize].pos as usize / cut;
                    let mut e = a + 1;
                    while e < st + len && toks[e as usize].pos as usize / cut == part {
                        e += 1;
                    }
                    dseg.extend([dtok.len() as i32, e - a]);
                    dtok.extend(a..e);
                    dkeys.extend(std::iter::repeat_n(toks[(e - 1) as usize].pos as i32 + 1, (e - a) as usize));
                    a = e;
                }
            } else {
                for o in (0..len).step_by(tile) {
                    ptile.extend([st + o, (len - o).min(tile as i32)]);
                }
            }
        }
        let ns = seg[0].len();
        let (mut mark, mut mark_at) = (Vec::new(), vec![0i32]);
        for i in 0..ns {
            match marks {
                Some(m) => mark.extend(m[i].iter().map(|&t| t as i32)),
                None => mark.extend((seg[0][i]..seg[0][i] + seg[1][i]).map(|t| toks[t as usize].id as i32)),
            }
            mark_at.push(mark.len() as i32);
        }
        ensure!(marks.is_none_or(|m| m.len() == ns) && mark.len() <= MAX_TOK + 8 * MAX_OUT, "marks per segment");
        let row_first: Vec<i32> =
            out.iter().map(|&o| *seg[0].iter().rev().find(|&&st| st as usize <= o).unwrap()).collect();
        let col = |f: fn(&Tok) -> u32| toks.iter().map(|t| f(t) as i32).collect::<Vec<_>>();
        s.memcpy_htod(&col(|t| t.id), &mut self.ids.slice_mut(0..n))?;
        s.memcpy_htod(&col(|t| t.slot), &mut self.tok_slot.slice_mut(0..n))?;
        s.memcpy_htod(&col(|t| t.pos), &mut self.tok_pos.slice_mut(0..n))?;
        for (v, dst) in seg.iter().zip([&mut self.seg_start, &mut self.seg_len, &mut self.seg_slot, &mut self.seg_pos0]) {
            s.memcpy_htod(v, &mut dst.slice_mut(0..ns))?;
        }
        if !mark.is_empty() {
            s.memcpy_htod(&mark, &mut self.mark.slice_mut(0..mark.len()))?;
        }
        s.memcpy_htod(&mark_at, &mut self.mark_at.slice_mut(0..ns + 1))?;
        if !out.is_empty() {
            let o: Vec<i32> = out.iter().map(|&o| o as i32).collect();
            s.memcpy_htod(&o, &mut self.out_rows.slice_mut(0..o.len()))?;
            s.memcpy_htod(&row_first, &mut self.row_first.slice_mut(0..o.len()))?;
        }
        if !dtok.is_empty() {
            s.memcpy_htod(&dtok, &mut self.dtok.slice_mut(0..dtok.len()))?;
            s.memcpy_htod(&dseg, &mut self.dseg.slice_mut(0..dseg.len()))?;
            s.memcpy_htod(&dkeys, &mut self.dkeys.slice_mut(0..dkeys.len()))?;
        }
        if !ptile.is_empty() {
            s.memcpy_htod(&ptile, &mut self.ptile.slice_mut(0..ptile.len()))?;
        }
        let pos = |ts: &mut dyn Iterator<Item = i32>| ts.map(|t| toks[t as usize].pos as usize).collect::<Vec<_>>();
        let dpos = pos(&mut dtok.iter().copied());
        let tpos = pos(&mut ptile.chunks(2).map(|c| c[0] + c[1] - 1));
        (self.n, self.n_out, self.segs, self.ndec, self.ntile) = (n, out.len(), ns, dtok.len(), ptile.len() / 2);
        self.ndseg = dseg.len() / 2;
        self.dmax = dseg.chunks(2).map(|d| d[1] as usize).max().unwrap_or(0);
        self.max_seg = seg[1].iter().copied().max().unwrap_or(0) as usize;
        self.dec_max_pos = dpos.iter().copied().max().unwrap_or(0);
        self.dec_min_pos = dpos.iter().copied().min().unwrap_or(0);
        self.tile_max_pos = tpos.into_iter().max().unwrap_or(0);
        Ok(())
    }

    /// The seen set of slot `slot` (repetition penalty), as words of 32 tokens
    pub fn seen(&self, s: &Arc<CudaStream>, slot: usize) -> Result<Vec<u32>> {
        let w = self.vocab / 32;
        Ok(s.clone_dtoh(&self.seen.slice(slot * w..(slot + 1) * w))?)
    }

    /// Set the seen set of slot `slot` (from seen)
    pub fn set_seen(&mut self, s: &Arc<CudaStream>, slot: usize, words: &[u32]) -> Result<()> {
        let w = self.vocab / 32;
        ensure!(words.len() == w, "a seen set of {} words, not {w}", words.len());
        Ok(s.memcpy_htod(words, &mut self.seen.slice_mut(slot * w..(slot + 1) * w))?)
    }

    /// Set the generated tokens of slot `slot` (presence penalty), as words of 32 tokens
    pub fn set_outputs(&mut self, s: &Arc<CudaStream>, slot: usize, words: &[u32]) -> Result<()> {
        let w = self.vocab / 32;
        ensure!(slot < self.slots && words.len() == w, "outputs of slot {slot}: {} words, not {w}", words.len());
        Ok(s.memcpy_htod(words, &mut self.outs.slice_mut(slot * w..(slot + 1) * w))?)
    }

    /// Add the step's committed tokens to the seen sets of their slots (first kernel of a step)
    pub fn mark_seen(&self, s: &Arc<CudaStream>) -> Result<()> {
        unsafe {
            s.launch_builder(&self.mark_seen).arg(&self.seen).arg(&self.mark).arg(&self.mark_at).arg(&self.seg_slot)
                .arg(&self.seg_pos0).launch(cfg((self.segs, 1, 1), 256))?;
        }
        Ok(())
    }

    /// Logits of the last step, [out rows][vocab].
    pub fn logits(&self, s: &Arc<CudaStream>) -> Result<Vec<f32>> {
        Ok(s.clone_dtoh(&self.logits.slice(0..self.n_out * self.vocab))?)
    }

    /// Logits of output row `j` of the last step.
    pub fn logits_row(&self, s: &Arc<CudaStream>, j: usize) -> Result<Vec<f32>> {
        Ok(s.clone_dtoh(&self.logits.slice(j * self.vocab..(j + 1) * self.vocab))?)
    }

    /// Sample the next token of each output row of the last step on the GPU (a row with more candidates than a block
    /// sorts: on the host, from the same state). Per row: temperature, top_k, top_p,
    /// repetition penalty, two uniform draws in [0, 1), a draft token (-1: none) and the presence penalty (the slot's
    /// outputs, set_outputs). Returns per row the token and, with a draft, whether the draft was accepted (the token
    /// is then the draft).
    pub fn sample(&mut self, s: &Arc<CudaStream>, params: &[[f32; 8]]) -> Result<Vec<(u32, bool)>> {
        let n = params.len();
        ensure!(n == self.n_out, "{n} sampling rows for {} logit rows", self.n_out);
        let flat: Vec<f32> = params.iter().flatten().copied().collect();
        s.memcpy_htod(&flat, &mut self.sparams.slice_mut(0..n * 8))?;
        unsafe {
            s.launch_builder(&self.sample_rows).arg(&self.logits).arg(&self.out_rows).arg(&self.tok_slot).arg(&self.ids)
                .arg(&self.row_first).arg(&self.seen).arg(&self.outs).arg(&self.sparams).arg(&self.qtok).arg(&self.qn)
                .arg(&self.qid).arg(&self.qp).arg(&self.sampled).arg(&self.accepted)
                .launch(cfg((n, 1, 1), 1024))?;
        }
        let toks = s.clone_dtoh(&self.sampled.slice(0..n))?;
        let acc = s.clone_dtoh(&self.accepted.slice(0..n))?;
        (0..n).map(|j| if acc[j] < 0 { self.sample_on_host(s, j, &params[j]) } else { Ok((toks[j] as u32, acc[j] != 0)) }).collect()
    }

    /// `sample_rows` for row `j` on the host, from the logits and penalty sets on the device
    fn sample_on_host(&self, s: &Arc<CudaStream>, j: usize, row: &[f32; 8]) -> Result<(u32, bool)> {
        let one = |b: &CudaSlice<i32>, i: usize| -> Result<usize> { Ok(s.clone_dtoh(&b.slice(i..i + 1))?[0] as usize) };
        let (tok, first) = (one(&self.out_rows, j)?, one(&self.row_first, j)?);
        let slot = one(&self.tok_slot, tok)?;
        let step: Vec<u32> = s.clone_dtoh(&self.ids.slice(first + 1..tok + 1))?.into_iter().map(|t| t as u32).collect();
        let w = self.vocab / 32;
        let (seen, outs) = (self.seen(s, slot)?, s.clone_dtoh(&self.outs.slice(slot * w..(slot + 1) * w))?);
        let has = |set: &[u32], t: u32| step.contains(&t) || set[t as usize / 32] >> (t % 32) & 1 == 1;
        let (seen, generated) = (|t| has(&seen, t), |t| has(&outs, t));
        let logits = self.logits_row(s, j)?;
        let mut row = *row;
        if row[6] <= -2.0 {
            row[6] = -row[6] - 2.0;
            let t = tok - first;
            if t < QD {
                let (q, drawn) = self.draft_q(s, slot, t)?;
                if drawn == row[6] as u32 && !q.is_empty() {
                    return Ok(sample::sample_row_q(&logits, seen, generated, &row, &q));
                }
            }
        }
        Ok(sample::sample_row(&logits, seen, generated, &row))
    }

    /// Sampled drafts: per output row of the last step (a draft step whose logits are `width` wide in rows of `ld`,
    /// column i the token head_ids[i], or i), a token drawn on the device from the row's sampling of them (read back
    /// with chain_drafts); `params` as for sample, with u1 the draw and u2 the draft's index t in its chain. Its
    /// distribution q is kept for (slot, t), for the sampler's rows whose draft is marked sampled (params[6] = -d - 2).
    pub fn draft_sample(&mut self, s: &Arc<CudaStream>, width: usize, ld: usize, head_ids: Option<&CudaSlice<i32>>,
                        params: &[[f32; 8]]) -> Result<()> {
        let n = params.len();
        ensure!(n == self.n_out && params.iter().all(|p| (p[5] as usize) < QD && p[0] > 0.0),
                "{n} draft sampling rows for {} logit rows", self.n_out);
        let flat: Vec<f32> = params.iter().flatten().copied().collect();
        s.memcpy_htod(&flat, &mut self.dparams.slice_mut(0..n * 8))?;
        let (w, ldi, null) = (width as i32, ld as i32, 0u64);
        unsafe {
            let mut l = s.launch_builder(&self.draft_sample);
            l.arg(&self.logits).arg(&w).arg(&ldi);
            match head_ids {
                Some(h) => l.arg(h),
                None => l.arg(&null),
            };
            l.arg(&self.out_rows).arg(&self.tok_slot).arg(&self.seen).arg(&self.outs).arg(&self.dparams).arg(&self.drafts)
                .arg(&self.qtok).arg(&self.qn).arg(&self.qid).arg(&self.qp).launch(cfg((n, 1, 1), 1024))?;
        }
        Ok(())
    }

    /// The tokens of the next step from the last draft_sample (its rows in order; the step uploaded with any ids):
    /// the next draft step continues each chain on the device
    pub fn ids_from_drafts(&mut self, s: &Arc<CudaStream>) -> Result<()> {
        let n = self.n;
        Ok(s.memcpy_dtod(&self.drafts.slice(0..n), &mut self.ids.slice_mut(0..n))?)
    }

    /// The first `k` drafts of the chains of slots `slots` (draft_sample)
    pub fn chain_drafts(&self, s: &Arc<CudaStream>, slots: &[u32], k: usize) -> Result<Vec<Vec<u32>>> {
        let all = s.clone_dtoh(&self.qtok)?;
        Ok(slots.iter().map(|&sl| all[sl as usize * QD..][..k].iter().map(|&t| t as u32).collect()).collect())
    }

    /// The distribution kept for slot `slot`'s draft at index `t` (draft_sample) and the token drawn from it
    pub fn draft_q(&self, s: &Arc<CudaStream>, slot: usize, t: usize) -> Result<(Vec<(u32, f32)>, u32)> {
        ensure!(slot < self.slots && t < QD, "draft {t} of slot {slot}");
        let q = slot * QD + t;
        let (n, tok) = (s.clone_dtoh(&self.qn.slice(q..q + 1))?[0] as usize, s.clone_dtoh(&self.qtok.slice(q..q + 1))?[0]);
        let ids = s.clone_dtoh(&self.qid.slice(q * QCAP..q * QCAP + n))?;
        let ps = s.clone_dtoh(&self.qp.slice(q * QCAP..q * QCAP + n))?;
        Ok((ids.into_iter().map(|i| i as u32).zip(ps).collect(), tok as u32))
    }

    /// Greedy next token per output row of the last step.
    pub fn argmax(&self, s: &Arc<CudaStream>) -> Result<Vec<u32>> {
        let keys = s.clone_dtoh(&self.keys.slice(0..self.n_out))?;
        Ok(keys.into_iter().map(|k| u32::MAX - k as u32).collect())
    }
}

/// Device memory outside the cache pool (vmm::Pool) that the CUDA graphs leave free: the driver's own needs (page
/// tables of the pages mapped, a kernel's first local memory) come from there too. Below half of it, free pages of
/// the pool go back to the driver.
const GRAPH_FLOOR: usize = 128 << 20;

/// CUDA graphs of decode-sized steps (up to GRAPH_TOK tokens), one per key. Their memory comes from what the cache
/// pool leaves (qwen::PAGE_RESERVE), and the keys are many (step shapes, key splits, the MTP's steps). The driver
/// keeps the memory of a graph dropped for the next ones (the free memory does not grow back), so the cache holds at
/// most `cap` graphs: unbounded until less than GRAPH_FLOOR is free when a new one is captured, then the graphs kept
/// at that point, lowered whenever the free memory falls further; the least recently used go. A step whose graph does not fit even
/// then launches its kernels directly: the same kernels, the same results.
pub struct Graphs<K> {
    cache: RefCell<GraphCache<K>>,
    pool: Option<Arc<crate::cuda::vmm::Pool>>,
}

struct GraphCache<K> {
    /// per key: the graph and when it last ran
    graphs: HashMap<K, (CudaGraph, u64)>,
    tick: u64,
    cap: usize,
    /// the free device memory when the cap was last set
    low: usize,
}

/// Whether a CUDA error is the device running out of memory
fn oom(e: &anyhow::Error) -> bool {
    e.downcast_ref::<cudarc::driver::DriverError>().is_some_and(|d| d.0 == sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)
}

impl<K: Hash + Eq + Clone> Graphs<K> {
    /// `pool`: the cache pool whose free pages go back to the driver when device memory outside it runs short
    pub fn new(pool: Option<Arc<crate::cuda::vmm::Pool>>) -> Self {
        Graphs { cache: RefCell::new(GraphCache { graphs: HashMap::new(), tick: 0, cap: usize::MAX, low: usize::MAX }), pool }
    }

    /// Drop the captured graphs whose keys `keep` refuses (their kernels' arguments changed)
    pub fn retain(&self, keep: impl Fn(&K) -> bool) {
        self.cache.borrow_mut().graphs.retain(|k, _| keep(k));
    }

    /// Run a step of `n` tokens: replay its graph (captured from `launch` on first use), or launch a larger one directly.
    pub fn run(&self, s: &Arc<CudaStream>, n: usize, key: K, launch: impl Fn() -> Result<()>) -> Result<()> {
        if n > GRAPH_TOK {
            return launch();
        }
        let mut c = self.cache.borrow_mut();
        c.tick += 1;
        let tick = c.tick;
        if let Some(g) = c.graphs.get_mut(&key) {
            g.1 = tick;
            return Ok(g.0.launch().context("replaying a CUDA graph")?);
        }
        self.make_room(s, &mut c)?;
        let captured = (|| -> Result<CudaGraph> {
            s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
            let launched = launch();
            let graph = s.end_capture(sys::CUgraphInstantiate_flags(0));
            launched?;
            let graph = graph?.context("empty graph")?;
            graph.launch()?;
            Ok(graph)
        })();
        match captured {
            Ok(g) => {
                c.graphs.insert(key, (g, tick));
                Ok(())
            }
            // no memory for the graph (nothing of the step ran): fewer graphs from now on, this step without one
            Err(e) if oom(&e) => {
                let cap = c.graphs.len() * 3 / 4;
                eprintln!("graphs: no device memory for a new CUDA graph ({} MiB free): {cap} kept of {}, a step of {n} \
                           tokens runs without one", free_mem()? >> 20, c.graphs.len());
                c.cap = cap;
                Self::evict(s, &mut c)?;
                launch()
            }
            Err(e) => Err(e.context(format!("capturing the CUDA graph of a step of {n} tokens"))),
        }
    }

    /// Before a new graph: room for it within the cap (lowered when device memory is short), the least recently used
    /// going; free pages of the pool back to the driver when it is shorter still
    fn make_room(&self, s: &Arc<CudaStream>, c: &mut GraphCache<K>) -> Result<()> {
        let free = free_mem()?;
        if free < GRAPH_FLOOR.min(c.low) && !c.graphs.is_empty() {
            let n = c.cap.min(c.graphs.len());
            let cap = if c.cap == usize::MAX { n } else { n - n / 16 };
            c.low = free;
            eprintln!("graphs: {} MiB of device memory free outside the cache: at most {cap} CUDA graphs kept (of {})",
                      free >> 20, c.graphs.len());
            c.cap = cap;
        }
        if c.graphs.len() >= c.cap {
            c.cap = c.cap.max(1);
            Self::evict(s, c)?;
        }
        if free < GRAPH_FLOOR / 2 && let Some(p) = &self.pool {
            let pages = p.release((GRAPH_FLOOR - free).div_ceil(p.page()))?;
            if pages > 0 {
                eprintln!("cache: {pages} free pages of {} MiB back to the driver ({} MiB of device memory free outside \
                           the cache)", p.page() >> 20, free >> 20);
            }
        }
        Ok(())
    }

    /// The least recently used graphs go until fewer than `cap` are kept
    fn evict(s: &Arc<CudaStream>, c: &mut GraphCache<K>) -> Result<()> {
        if c.graphs.len() < c.cap {
            return Ok(());
        }
        // (they may still run)
        s.synchronize()?;
        let mut by_age: Vec<(u64, K)> = c.graphs.iter().map(|(k, g)| (g.1, k.clone())).collect();
        by_age.sort_unstable_by_key(|a| a.0);
        let n = c.graphs.len() + 1 - c.cap.max(1);
        for (_, k) in by_age.into_iter().take(n) {
            c.graphs.remove(&k);
        }
        Ok(())
    }
}

/// Free device memory now
fn free_mem() -> Result<usize> {
    Ok(cudarc::driver::result::mem_get_info()?.0)
}
