//! A LoRA adapter served unmerged, per request: the base matrices stay as they are stored (int4, int8, 16-bit) and
//! every adapted product adds alpha / r * B (A x) for the tokens of the slots that use the adapter (Model::set_adapter;
//! the other tokens get exactly what the base model gives them). The products that read one input share one stacked
//! A, a 16-bit matrix product of the step (gemm.cuh, f32 accumulation; its split-K partial products in `Bufs::lpart`);
//! kernels/lora.cuh adds B's part to the product's partial products, or, for [gate; up], applies SwiGLU with the
//! delta added first (the product then runs plain, not with the SwiGLU epilogue). Steps without adapter tokens skip
//! all of it. A and B are f16 on the device: f16 holds a bf16 adapter's values exactly (in f16's normal range) and an
//! f32 one to 11 bits, as the f16 weights of a 16-bit model; the products accumulate in f32.
use super::Stack;
use crate::cuda::core::{Mat, cfg, row_order};
use crate::lora::Adapter;
use anyhow::{Context, Result, ensure};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, PushKernelArg};
use std::{collections::HashSet, sync::Arc};

/// Tokens per block of the kernels (LORA_TT), columns of xa at most (LORA_MAXM)
const TT: usize = 8;
const MAXM: usize = 256;
/// Ranks with kernels (LORA_SET)
const RANKS: [usize; 4] = [8, 16, 32, 64];

pub struct Kernels {
    add: Vec<CudaFunction>,
    silu: Vec<CudaFunction>,
}

impl Kernels {
    pub fn load(m: &Arc<CudaModule>) -> Result<Self> {
        let f = |p: &str| RANKS.iter().map(|r| m.load_function(&format!("{p}_r{r}"))).collect::<Result<Vec<_>, _>>();
        Ok(Kernels { add: f("lora_add")?, silu: f("lora_silu")? })
    }
}

/// The adapted modules of the products that read one input: A of each stacked, [modules * r (padded to 16)][in]
pub struct LoraA {
    pub a: Mat,
    r: usize,
}

/// B of one stacked product, its rows in the product's order (silu: row_order), f16 [rows][r]; `tab`: per module the
/// end of its rows, then its first column of xa (-1: not adapted; lora.cuh)
pub struct LoraB {
    b: CudaSlice<u16>,
    tab: CudaSlice<i32>,
    rows: usize,
}

/// The modules of one stacked product: names (`.weight`) and rows, and whether it is packed for SwiGLU
pub struct Stacked {
    pub names: Vec<String>,
    pub rows: Vec<usize>,
    pub silu: bool,
}

/// A's of the stacked products `stacks` that read one input of `cols` columns, and each product's B; None when the
/// adapter changes none of them. `used`: the adapter's matrices taken (all must be: Model::load).
pub fn load(s: &Arc<CudaStream>, ad: &Adapter, stacks: &[Stacked], cols: usize, used: &mut HashSet<String>)
-> Result<Option<(LoraA, Vec<Option<LoraB>>)>> {
    let (mut a, mut r) = (Vec::new(), 0);
    // per stack, per module: its column in xa
    let mut offs: Vec<Vec<i32>> = Vec::new();
    for st in stacks {
        ensure!(st.names.len() <= 4 && st.names.len() == st.rows.len() && (!st.silu || st.names.len() == 2),
                "a stack of {} modules", st.names.len());
        let mut o = Vec::new();
        for n in &st.names {
            match ad.get(n) {
                Some((ma, _, mr)) => {
                    ensure!(ma.len() == mr * cols, "{n}: lora_A of {} values for rank {mr} and {cols} inputs", ma.len());
                    ensure!(r == 0 || r == mr, "{n}: rank {mr} beside {r}");
                    r = mr;
                    o.push((a.len() / cols) as i32);
                    a.extend_from_slice(ma);
                    used.insert(n.clone());
                }
                None => o.push(-1),
            }
        }
        offs.push(o);
    }
    if a.is_empty() {
        return Ok(None);
    }
    ensure!(RANKS.contains(&r), "LoRA rank {r}: served are ranks {RANKS:?}");
    let m = (a.len() / cols).next_multiple_of(16);
    ensure!(m <= MAXM, "{m} LoRA rows on one input (at most {MAXM})");
    a.resize(m * cols, 0.0);
    let a = Mat::new(s, &a, cols, false, false).context("LoRA A")?;
    let mut bs = Vec::new();
    for (st, o) in stacks.iter().zip(&offs) {
        if o.iter().all(|&x| x < 0) {
            bs.push(None);
            continue;
        }
        let rows: usize = st.rows.iter().sum();
        let mut b = vec![0f32; rows * r];
        let mut at = 0;
        for (n, &mr) in st.names.iter().zip(&st.rows) {
            if let Some((_, mb, _)) = ad.get(n) {
                ensure!(mb.len() == mr * r, "{n}: lora_B of {} values for {mr} rows of rank {r}", mb.len());
                b[at * r..(at + mr) * r].copy_from_slice(mb);
            }
            at += mr;
        }
        // in the product's row order
        let order = row_order(rows, st.silu);
        let packed: Vec<u16> = order.iter().flat_map(|&o| b[o * r..(o + 1) * r].iter().map(|&v| half::f16::from_f32(v).to_bits()))
            .collect();
        let mut tab = [i32::MAX; 8];
        let mut end = 0;
        for (i, &mr) in st.rows.iter().enumerate() {
            end += mr;
            if !st.silu && i < 3 {
                tab[i] = end as i32;
            }
        }
        for (i, &x) in o.iter().enumerate() {
            tab[4 + i] = x;
        }
        for x in tab[4 + o.len()..].iter_mut() {
            *x = -1;
        }
        bs.push(Some(LoraB { b: s.clone_htod(&packed)?, tab: s.clone_htod(&tab)?, rows }));
    }
    Ok(Some((LoraA { a, r }, bs)))
}

impl LoraA {
    /// The kernel for this rank of `f`, on the plain product y (split 0 or 1, [n][rows]), xa = A x in `lpart` of `split`
    /// parts (s of the slots that use the adapter); and mid when it is written. A block per 256 rows it writes, of y's or
    /// of mid's half of them.
    fn launch(&self, f: &[CudaFunction], x: &Stack, split: usize, b: &LoraB, y: &CudaSlice<f32>, mid: Option<&CudaSlice<u16>>)
    -> Result<()> {
        let (spi, m, rows, n) = (split as i32, (self.a.tiles * 16) as i32, b.rows as i32, x.st.n as i32);
        let per = if mid.is_some() { 2 * 256 } else { 256 };
        let rank = RANKS.iter().position(|&r| r == self.r).unwrap();
        unsafe {
            let mut l = x.s.launch_builder(&f[rank]);
            l.arg(&x.b.lpart).arg(&spi).arg(&m).arg(&b.b).arg(&rows).arg(&b.tab).arg(&x.st.tok_slot).arg(&x.scales).arg(&n)
                .arg(y).arg(&rows);
            if let Some(mid) = mid {
                l.arg(mid);
            }
            l.launch(cfg((b.rows.div_ceil(per), x.st.n.div_ceil(TT), 1), 256))?;
        }
        Ok(())
    }

    /// y (a plain product's split 0, [n][rows]) += s B xa for the tokens of adapter slots; xa in `lpart`, of `split` parts
    pub fn add(&self, x: &Stack, split: usize, b: &LoraB, y: &CudaSlice<f32>) -> Result<()> {
        self.launch(&x.lk.add, x, split, b, y, None)
    }

    /// mid = SwiGLU of the plain [gate; up] product y (split 1, [n][rows]) with s B xa added for the tokens of adapter
    /// slots, as the SwiGLU epilogue writes it
    pub fn silu(&self, x: &Stack, split: usize, b: &LoraB, y: &CudaSlice<f32>, mid: &CudaSlice<u16>) -> Result<()> {
        self.launch(&x.lk.silu, x, split, b, y, Some(mid))
    }
}

/// A layer's LoRA: the in-projections (in_proj and DeltaNet's in_ba read the same input), out_proj, [gate; up], down
pub struct LayerLora {
    pub inp: Option<(LoraA, Option<LoraB>, Option<LoraB>)>,
    pub out: Option<(LoraA, LoraB)>,
    pub gate_up: Option<(LoraA, LoraB)>,
    pub down: Option<(LoraA, LoraB)>,
}

impl LayerLora {
    /// The A matrices (for the size of their partial products)
    pub fn mats(&self) -> impl Iterator<Item = &Mat> {
        let a = [self.inp.as_ref().map(|x| &x.0), self.out.as_ref().map(|x| &x.0), self.gate_up.as_ref().map(|x| &x.0),
                 self.down.as_ref().map(|x| &x.0)];
        a.into_iter().flatten().map(|a| &a.a)
    }
}

/// One stacked product's A and B
pub fn single(s: &Arc<CudaStream>, ad: &Adapter, st: Stacked, cols: usize, used: &mut HashSet<String>)
-> Result<Option<(LoraA, LoraB)>> {
    Ok(load(s, ad, &[st], cols, used)?.map(|(a, mut b)| (a, b.pop().flatten().unwrap())))
}
