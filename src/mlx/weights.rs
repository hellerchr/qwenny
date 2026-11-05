//! Qwen3.5 / Qwen3.8 weights and the residual stack as MLX operations. Matrix products use f16 inputs (dense or the
//! checkpoint's affine quantization), with f32 residual streams, norms, mixer state and element-wise gates. The
//! backend-specific cache operations are supplied by `Mix`.
use crate::checkpoint::{Checkpoint, Matrix, Quant};
use crate::draft_vocab::{Adaptive, draft_ids};
use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use mlx_rs::{Array, Dtype, error::Exception, fast, nn, ops};
use std::{fs, path::Path};

pub type MResult<T> = std::result::Result<T, Exception>;

pub const NKV: i32 = 4;
pub const HD: i32 = 256;
pub const LK_HEADS: i32 = 16;
pub const LKD: i32 = 128;
pub const LVD: i32 = 128;
pub const CONV_K: i32 = 4;
pub const EPS: f32 = 1e-6;
pub const ROT: i32 = 64;
pub const THETA: f64 = 1e7;
/// qwenny's CUDA backend quantizes dense weights at load in groups of 128.
const QGROUP: i32 = 128;

#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub nh: i32,
    pub lv_heads: i32,
    pub vocab: usize,
}

impl Dims {
    pub fn of(config: &serde_json::Value) -> Result<Self> {
        let tc = config.get("text_config").unwrap_or(config);
        let get = |k: &str| tc[k].as_u64().map(|v| v as i32);
        for (key, want) in [
            ("num_key_value_heads", NKV),
            ("head_dim", HD),
            ("linear_num_key_heads", LK_HEADS),
            ("linear_key_head_dim", LKD),
            ("linear_value_head_dim", LVD),
            ("linear_conv_kernel_dim", CONV_K),
        ] {
            ensure!(get(key) == Some(want), "config {key} = {}, expected {want}", tc[key]);
        }
        ensure!(
            tc["rms_norm_eps"].as_f64().is_some_and(|e| (e - EPS as f64).abs() < 1e-12),
            "rms_norm_eps {}",
            tc["rms_norm_eps"]
        );
        ensure!(
            tc["attn_output_gate"].as_bool() == Some(true) && tc["attention_bias"].as_bool() == Some(false),
            "attention without the output gate or with biases"
        );
        ensure!(
            tc.get("num_experts").and_then(|x| x.as_u64()).unwrap_or(0) == 0,
            "MoE Qwen checkpoints are not supported"
        );
        let rope = &tc["rope_parameters"];
        ensure!(
            rope["rope_type"] == "default"
                && rope["rope_theta"].as_f64() == Some(THETA)
                && rope["partial_rotary_factor"].as_f64() == Some(ROT as f64 / HD as f64),
            "rope {rope}"
        );
        Ok(Dims {
            nh: get("num_attention_heads").context("num_attention_heads")?,
            lv_heads: get("linear_num_value_heads").context("linear_num_value_heads")?,
            vocab: tc["vocab_size"].as_u64().context("vocab_size")? as usize,
        })
    }

    pub fn q_width(self) -> i32 {
        self.nh * HD
    }
    pub fn kv_width(self) -> i32 {
        NKV * HD
    }
    pub fn key_width(self) -> i32 {
        LK_HEADS * LKD
    }
    pub fn value_width(self) -> i32 {
        self.lv_heads * LVD
    }
    pub fn conv_width(self) -> i32 {
        2 * self.key_width() + self.value_width()
    }
}

/// x W^T for W [out][in], dense in f16 or in MLX's affine quantized layout.
#[derive(Clone)]
pub enum Linear {
    Dense(Array),
    Quant { w: Array, scales: Array, biases: Array, group: i32, bits: i32 },
}

impl Linear {
    /// MLX's quantized products take groups of 32, 64 or 128: a larger group that 128 divides (per channel, 256, ...)
    /// is the same matrix in groups of 128 with its scale and offset repeated
    fn stored(q: &Quant) -> Result<Self> {
        let (scales, offsets, group) = if [32, 64, 128].contains(&q.group) {
            (q.scales.clone(), q.offsets.clone(), q.group)
        } else {
            ensure!(q.group.is_multiple_of(128), "groups of {}: MLX takes 32, 64, 128 or multiples of 128", q.group);
            let rep = |v: &[f32]| v.iter().flat_map(|&x| std::iter::repeat_n(x, q.group / 128)).collect::<Vec<_>>();
            (rep(&q.scales), rep(&q.offsets), 128)
        };
        let (rows, groups) = (q.rows as i32, (q.cols / group) as i32);
        Ok(Linear::Quant {
            w: Array::from_slice(&q.q, &[rows, (q.cols * q.bits / 32) as i32]),
            scales: Array::from_slice(&scales, &[rows, groups]).as_dtype(Dtype::Float16)?,
            biases: Array::from_slice(&offsets, &[rows, groups]).as_dtype(Dtype::Float16)?,
            group: group as i32,
            bits: q.bits as i32,
        })
    }

    pub(super) fn arrays(&self) -> Vec<&Array> {
        match self {
            Linear::Dense(w) => vec![w],
            Linear::Quant { w, scales, biases, .. } => vec![w, scales, biases],
        }
    }

    pub fn spec(&self) -> Spec {
        match self {
            Linear::Dense(_) => Spec::Dense,
            Linear::Quant { group, bits, .. } => Spec::Quant { group: *group, bits: *bits, draft: false },
        }
    }

    /// The arrays Spec::apply takes
    fn owned(&self) -> Vec<Array> {
        self.arrays().into_iter().cloned().collect()
    }

    pub fn apply(&self, x: &Array) -> MResult<Array> {
        self.spec().apply(&self.owned(), x)
    }

    pub fn apply_draft(&self, x: &Array) -> MResult<Array> {
        self.spec().for_draft().apply(&self.owned(), x)
    }

    fn select_rows(&self, ids: &[u32]) -> MResult<Self> {
        let idx = Array::from_slice(ids, &[ids.len() as i32]).as_dtype(Dtype::Int32)?;
        Ok(match self {
            Linear::Dense(w) => Linear::Dense(w.take_axis(&idx, 0)?),
            Linear::Quant { w, scales, biases, group, bits } => Linear::Quant {
                w: w.take_axis(&idx, 0)?,
                scales: scales.take_axis(&idx, 0)?,
                biases: biases.take_axis(&idx, 0)?,
                group: *group,
                bits: *bits,
            },
        })
    }

    fn append(&mut self, tail: &Self) -> MResult<()> {
        match (self, tail) {
            (Linear::Dense(a), Linear::Dense(b)) => *a = ops::concatenate(&[&*a, b], 0)?,
            (Linear::Quant { w, scales, biases, .. }, Linear::Quant { w: tw, scales: ts, biases: tb, .. }) => {
                *w = ops::concatenate(&[&*w, tw], 0)?;
                *scales = ops::concatenate(&[&*scales, ts], 0)?;
                *biases = ops::concatenate(&[&*biases, tb], 0)?;
            }
            _ => unreachable!("draft rows keep the source matrix's format"),
        }
        Ok(())
    }

    fn replace_tile(&mut self, at: i32, tile: &Self) -> MResult<()> {
        use mlx_rs::ops::indexing::TryIndexMutOp;
        match (self, tile) {
            (Linear::Dense(a), Linear::Dense(b)) => a.try_index_mut((at..at + 16, ..), b)?,
            (Linear::Quant { w, scales, biases, .. }, Linear::Quant { w: tw, scales: ts, biases: tb, .. }) => {
                w.try_index_mut((at..at + 16, ..), tw)?;
                scales.try_index_mut((at..at + 16, ..), ts)?;
                biases.try_index_mut((at..at + 16, ..), tb)?;
            }
            _ => unreachable!("draft rows keep the source matrix's format"),
        }
        Ok(())
    }
}

/// How a Linear computes, without its arrays: what a compiled stage captures
#[derive(Clone, Copy, Debug)]
pub enum Spec {
    Dense,
    Quant {
        group: i32,
        bits: i32,
        #[cfg_attr(not(has_metal), expect(dead_code))]
        draft: bool,
    },
}

impl Spec {
    fn for_draft(self) -> Self {
        match self {
            Spec::Dense => self,
            Spec::Quant { group, bits, .. } => Spec::Quant { group, bits, draft: true },
        }
    }
    fn n(self) -> usize {
        match self {
            Spec::Dense => 1,
            Spec::Quant { .. } => 3,
        }
    }

    /// x W^T from the matrix's arrays (Linear::owned) at the front of `a`, as f32: products enter the f32
    /// mixer/residual path without another low-precision element-wise pass. Returns the rest of `a`.
    fn apply_take<'a>(self, a: &'a [Array], x: &Array) -> MResult<(Array, &'a [Array])> {
        let y = match self {
            Spec::Dense => ops::matmul(x, ops::transpose(&a[0])?),
            // Prefill (large M): MLX's quantized_matmul is weight-bandwidth-bound and barely amortizes the int4
            // weight read across the step's tokens, so a step of 2,048 is only ~3x a single decode. Dequantize the
            // weight to f16 once and run a dense (compute-bound) matmul instead; the f16 weight is transient, read
            // once and reused across all M rows. Decode (small M) keeps quantized_matmul: there the weight read is
            // the work and dequantizing first would only add a pass.
            Spec::Quant { group, bits, .. } if x.shape()[0] > COMPILE_MAX => {
                let w = ops::dequantize(&a[0], &a[1], Some(&a[2]), group, bits)?;
                ops::matmul(x, ops::transpose(&w)?)
            }
            // Measured on an M2 Pro only; the stock product may be better on chips with MLX's newer kernels.
            #[cfg(has_metal)]
            Spec::Quant { group, bits: 4, draft: true }
                if x.shape()[0] == 1 && x.shape()[1] % 512 == 0 && group % 16 == 0 =>
            {
                super::metal::qmv(x, a)
            }
            #[cfg(has_metal)]
            Spec::Quant { group, bits: bits @ (4 | 8), .. }
                if (1..=16).contains(&x.shape()[0])
                    && (x.shape()[0] >= 4 || bits == 4)
                    && group % 64 == 0
                    && x.shape()[1] % 64 == 0
                    && a[0].shape()[0] >= 1024 =>
            {
                super::metal::verify(x, a)
            }
            Spec::Quant { group, bits, .. } => ops::quantized_matmul(x, &a[0], &a[1], &a[2], true, group, bits),
        }?;
        Ok((y.as_dtype(Dtype::Float32)?, &a[self.n()..]))
    }

    pub fn apply(self, a: &[Array], x: &Array) -> MResult<Array> {
        Ok(self.apply_take(a, x)?.0)
    }
}

/// A step above this many tokens is a prompt chunk. Graphs are compiled per shape, so only the shapes that recur are:
/// steps of up to this many tokens outside a prompt's start, and the full chunk of FULL_CHUNK tokens; a prompt's
/// arbitrary lengths run uncompiled instead of each tracing a graph.
pub const COMPILE_MAX: i32 = 64;
const FULL_CHUNK: i32 = super::model::MAX_TOK as i32;

/// A function of arrays, also compiled (mlx's compile: one graph, element-wise operations fused; on Metal in its safe
/// math mode, through NVRTC without fast math: the operations stay exact, rounding may differ by fused multiply-adds).
type ArrayFn = dyn Fn(&[Array]) -> MResult<Vec<Array>>;
type ArrayFnMut = dyn FnMut(&[Array]) -> MResult<Vec<Array>>;

pub struct Fused {
    plain: std::rc::Rc<ArrayFn>,
    compiled: Box<ArrayFnMut>,
}

impl Fused {
    pub fn new(f: impl Fn(&[Array]) -> MResult<Vec<Array>> + 'static) -> Self {
        let plain: std::rc::Rc<ArrayFn> = std::rc::Rc::new(f);
        let p = plain.clone();
        let mut c = mlx_rs::transforms::compile::compile(move |a: &[Array]| p(a), false);
        Fused { plain, compiled: Box::new(move |a: &[Array]| c(a)) }
    }

    /// compiled when `small`
    pub fn call(&mut self, small: bool, a: &[Array]) -> MResult<Vec<Array>> {
        if small { (self.compiled)(a) } else { (self.plain)(a) }
    }
}

/// A layer's compiled stages: `pre` from the residual stream to the mixer's inputs, `tail` from the mixer's output
/// through out_proj, the residual and the MLP; the weights they read
pub struct Stages {
    pre: Fused,
    pre_args: Vec<Array>,
    tail: Fused,
    tail_args: Vec<Array>,
}

impl Stages {
    fn of(l: &Layer, d: Dims, draft: bool) -> MResult<Self> {
        let spec = |s: Spec| if draft { s.for_draft() } else { s };
        let (sp_in, sout, sgu, sdown) =
            (spec(l.in_proj.spec()), spec(l.out_proj.spec()), spec(l.gate_up.spec()), spec(l.down.spec()));
        let mut pre_args = vec![l.input_norm.clone()];
        pre_args.extend(l.in_proj.owned());
        let pre = match l.kind {
            Kind::Attn => {
                pre_args.extend([l.q_norm.clone().unwrap(), l.k_norm.clone().unwrap()]);
                // [h, cos, sin, input_norm, in_proj.., q_norm, k_norm] -> [q, gate, k int8, k scales, v int8, v scales]
                Fused::new(move |a: &[Array]| {
                    let (h, cos, sin) = (&a[0], &a[1], &a[2]);
                    let t = h.shape()[0];
                    let xn = rms(h, &a[3])?.as_dtype(Dtype::Float16)?;
                    let (projected, rest) = sp_in.apply_take(&a[4..], &xn)?;
                    let s = ops::split_at_indices(&projected, &[2 * d.q_width(), 2 * d.q_width() + d.kv_width()], -1)?;
                    let qg = ops::split_equal(&s[0].reshape(&[t, d.nh, 2, HD])?, 2, 2)?;
                    let q = rms(&qg[0].squeeze_axes(&[2])?, &rest[0])?;
                    let k = rms(&s[1].reshape(&[t, NKV, HD])?, &rest[1])?;
                    let (q, k) = (rope(&q, cos, sin)?, rope(&k, cos, sin)?);
                    let (kq, ks) = quantize_kv(&k)?;
                    let (vq, vs) = quantize_kv(&s[2].reshape(&[t, NKV, HD])?)?;
                    Ok(vec![q, qg[1].squeeze_axes(&[2])?, kq, ks, vq, vs])
                })
            }
            Kind::Delta => {
                let sba = l.in_ba.as_ref().unwrap().spec();
                pre_args.extend(l.in_ba.as_ref().unwrap().owned());
                pre_args.extend([ops::negative(ops::exp(l.a_log.as_ref().unwrap())?)?, l.dt_bias.clone().unwrap()]);
                // [h, input_norm, in_proj.., in_ba.., -exp(A_log), dt_bias] -> [qkv, z, beta, log decay, decay]
                Fused::new(move |a: &[Array]| {
                    let xn = rms(&a[0], &a[1])?.as_dtype(Dtype::Float16)?;
                    let (projected, rest) = sp_in.apply_take(&a[2..], &xn)?;
                    let (ba, rest) = sba.apply_take(rest, &xn)?;
                    let qkvz = ops::split_at_indices(&projected, &[d.conv_width()], -1)?;
                    let ba = ops::split_equal(&ba, 2, -1)?;
                    let beta = nn::sigmoid(&ba[0])?;
                    let lg = ops::multiply(&rest[0], nn::softplus(ops::add(&ba[1], &rest[1])?)?)?;
                    let e = ops::exp(&lg)?;
                    Ok(vec![qkvz[0].clone(), qkvz[1].clone(), beta, lg, e])
                })
            }
        };
        let gated = l.kind == Kind::Attn;
        let mut tail_args = l.out_proj.owned();
        tail_args.push(l.post_norm.clone());
        tail_args.extend(l.gate_up.owned());
        tail_args.extend(l.down.owned());
        // [h, y, (gate), out_proj.., post_norm, gate_up.., down..] -> [h after the mixer and the MLP]
        let tail = Fused::new(move |a: &[Array]| {
            let h = &a[0];
            let t = h.shape()[0];
            let (y, rest) = if gated {
                (gate(&a[1], &a[2])?.reshape(&[t, d.q_width()])?, &a[3..])
            } else {
                (a[1].clone(), &a[2..])
            };
            let (op, rest) = sout.apply_take(rest, &y.as_dtype(Dtype::Float16)?)?;
            let h = ops::add(h, &op)?;
            let xn = rms(&h, &rest[0])?.as_dtype(Dtype::Float16)?;
            let (gu, rest) = sgu.apply_take(&rest[1..], &xn)?;
            let ab = ops::split_equal(&gu, 2, -1)?;
            let f = sdown.apply(rest, &swiglu(&ab[0], &ab[1])?)?;
            Ok(vec![ops::add(&h, &f)?])
        });
        Ok(Stages { pre, pre_args, tail, tail_args })
    }

    fn arrays(&self) -> impl Iterator<Item = &Array> {
        self.pre_args.iter().chain(&self.tail_args)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Delta,
    Attn,
}

pub struct Layer {
    pub kind: Kind,
    /// Attention: [q_proj; k_proj; v_proj], q_proj interleaving query and gate rows per head.
    /// DeltaNet: [in_proj_qkv; in_proj_z].
    pub in_proj: Linear,
    /// DeltaNet: [in_proj_b; in_proj_a].
    pub in_ba: Option<Linear>,
    pub out_proj: Linear,
    pub input_norm: Array,
    pub post_norm: Array,
    pub gate_up: Linear,
    pub down: Linear,
    pub q_norm: Option<Array>,
    pub k_norm: Option<Array>,
    /// DeltaNet convolution [C][4], gates [Hv], and its plain (not zero-centered) output norm [Dv].
    pub conv_w: Option<Array>,
    pub a_log: Option<Array>,
    pub dt_bias: Option<Array>,
    pub delta_norm: Option<Array>,
}

pub struct Weights {
    pub d: Dims,
    /// [V][H], used for lookup and for a dense tied output layer.
    pub embed: Array,
    /// None: a dense tied output layer (`embed`); otherwise the checkpoint's separate or quantized output matrix.
    pub head: Option<Linear>,
    pub final_norm: Array,
    pub layers: Vec<Layer>,
    stages: Vec<Stages>,
    pub mtp: Option<Mtp>,
}

pub struct Mtp {
    pub pre_e: Array,
    pub pre_h: Array,
    pub fc: Linear,
    pub norm: Array,
    pub head: Option<Linear>,
    pub ids: Vec<u32>,
    pub adaptive: Option<Adaptive>,
    layer: Layer,
    stages: Stages,
}

impl Mtp {
    pub fn forward(&mut self, m: &mut impl Mix, h: &Array, ai: usize) -> MResult<Array> {
        let t = h.shape()[0];
        let compiled = t <= 16 || t == FULL_CHUNK;
        run_layer(&self.layer, &mut self.stages, At { li: 0, ai, di: 0 }, m, h, compiled)
    }
}

/// CUDA's dense-to-int8 quantization (core.rs's pack), in the backend-neutral Quant layout.
fn symmetric_int8(m: Matrix) -> Result<Quant> {
    let Matrix::Dense { w, rows, cols } = m else { bail!("quantizing an already quantized matrix") };
    let group = QGROUP as usize;
    ensure!(cols % group == 0, "int8 quantization: {cols} columns, not groups of {group}");
    let (groups, words) = (cols / group, cols / 4);
    let mut q = vec![0u32; rows * words];
    let mut scales = vec![0f32; rows * groups];
    let mut offsets = vec![0f32; rows * groups];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let per = rows.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (i, ((qc, sc_out), oc)) in q
            .chunks_mut(per * words)
            .zip(scales.chunks_mut(per * groups))
            .zip(offsets.chunks_mut(per * groups))
            .enumerate()
        {
            let w = &w;
            sc.spawn(move || {
                for r in 0..sc_out.len() / groups {
                    let row = (i * per + r) * cols;
                    for g in 0..groups {
                        let at = row + g * group;
                        let max =
                            w[at..at + group].iter().fold(0f32, |m, &x| m.max(f32::from_bits((x as u32) << 16).abs()));
                        let scale = if max > 0.0 { max / 127.0 } else { 1.0 };
                        sc_out[r * groups + g] = scale;
                        oc[r * groups + g] = -128.0 * scale;
                        for (c, &x) in w[at..at + group].iter().enumerate() {
                            let x = f32::from_bits((x as u32) << 16);
                            let v = (x / scale).round().clamp(-127.0, 127.0) as i32 + 128;
                            let col = g * group + c;
                            qc[r * words + col / 4] |= (v as u32) << ((col % 4) * 8);
                        }
                    }
                }
            });
        }
    });
    Ok(Quant { bits: 8, group, rows, cols, q, scales, offsets })
}

fn array(bits: &[u16], shape: &[i32]) -> MResult<Array> {
    let v: Vec<bf16> = bits.iter().map(|&b| bf16::from_bits(b)).collect();
    Array::from_slice(&v, shape).as_dtype(Dtype::Float16)
}

impl Weights {
    pub fn load(dir: &Path, int8: bool) -> Result<Self> {
        let config: serde_json::Value = serde_json::from_str(&fs::read_to_string(dir.join("config.json"))?)?;
        let tc = config.get("text_config").unwrap_or(&config);
        let d = Dims::of(&config)?;
        let ck = Checkpoint::open(dir)?;
        // Original HF: model.language_model.*. An mlx-lm conversion is language_model.model.* on disk, which
        // Checkpoint's prefix removal exposes as model.*.
        let text = if ck.has("model.language_model.embed_tokens.weight") {
            "model.language_model"
        } else if ck.has("model.embed_tokens.weight") {
            "model"
        } else {
            bail!("no text embedding table")
        };
        let types = tc["layer_types"].as_array().context("layer_types")?;
        ensure!(!types.is_empty(), "no layers");
        // mlx-lm's sanitizer moves conv [C][1][4] to [C][4][1] and adds one to the zero-centered norms. Detect
        // that exact conversion rather than adding one twice to an MLX checkpoint.
        let first_delta = types.iter().position(|x| x == "linear_attention").context("no DeltaNet layers")?;
        let conv_name = format!("{text}.layers.{first_delta}.linear_attn.conv1d.weight");
        let (_, conv_shape) = ck.floats(&conv_name)?;
        let zero_centered = conv_shape.last() != Some(&1);
        let quantize = int8 && !ck.prequantized();

        let mats = |names: &[String]| -> Result<Vec<Matrix>> {
            names.iter().map(|n| ck.matrix(n.strip_suffix(".weight").unwrap_or(n))).collect()
        };
        let dense = |names: &[String], mats: Vec<Matrix>| -> Result<Array> {
            let (mut all, mut cols) = (Vec::new(), 0);
            for (n, m) in names.iter().zip(mats) {
                let (w, c) = match m {
                    Matrix::Dense { w, rows, cols } => {
                        ensure!(w.len() == rows * cols, "{n}: {} values for {rows} x {cols}", w.len());
                        (w, cols)
                    }
                    Matrix::Quant(q) => (q.to_bf16(), q.cols),
                };
                ensure!(cols == 0 || cols == c, "stacking matrices with different column counts");
                cols = c;
                all.extend(w);
            }
            Ok(array(&all, &[(all.len() / cols) as i32, cols as i32])?)
        };
        let lin = |names: &[String]| -> Result<Linear> {
            let ms = mats(names)?;
            if ms.iter().all(|m| matches!(m, Matrix::Quant(_))) {
                let qs = ms.into_iter().filter_map(|m| if let Matrix::Quant(q) = m { Some(q) } else { None }).collect();
                return Linear::stored(&Quant::stack(qs)?);
            }
            if quantize {
                // CUDA's pack(), not MLX's affine min/max quantization: symmetric q in [-127,127], stored +128,
                // an f32 max/127 scale per row and group of 128, and round/clamp in exactly that order.
                let qs = ms.into_iter().map(symmetric_int8).collect::<Result<Vec<_>>>()?;
                return Linear::stored(&Quant::stack(qs)?);
            }
            Ok(Linear::Dense(dense(names, ms)?))
        };
        let vec = |n: &str| -> Result<Array> {
            let (w, shape) = ck.floats(n)?;
            Ok(Array::from_slice(&w, &shape.iter().map(|&d| d as i32).collect::<Vec<_>>()))
        };
        let norm = |n: &str| -> Result<Array> {
            let (w, _) = ck.floats(n)?;
            let w: Vec<f32> = if zero_centered { w.into_iter().map(|x| 1.0 + x).collect() } else { w };
            Ok(Array::from_slice(&w, &[w.len() as i32]))
        };
        let load_layer = |p: &str, kind: Kind| -> Result<Layer> {
            let mixer = if kind == Kind::Delta { "linear_attn" } else { "self_attn" };
            let a = format!("{p}.{mixer}");
            let in_names: Vec<String> = if kind == Kind::Delta {
                ["in_proj_qkv", "in_proj_z"].map(|n| format!("{a}.{n}.weight")).into()
            } else {
                ["q_proj", "k_proj", "v_proj"].map(|n| format!("{a}.{n}.weight")).into()
            };
            let in_ba = if kind == Kind::Delta {
                Some(lin(&["in_proj_b", "in_proj_a"].map(|n| format!("{a}.{n}.weight")))?)
            } else {
                None
            };
            let attn_norm = |n: &str| -> Result<Option<Array>> {
                if kind == Kind::Attn { Ok(Some(norm(&format!("{a}.{n}.weight"))?)) } else { Ok(None) }
            };
            let delta_vec = |n: &str| -> Result<Option<Array>> {
                if kind == Kind::Delta { Ok(Some(vec(&format!("{a}.{n}"))?)) } else { Ok(None) }
            };
            Ok(Layer {
                kind,
                in_proj: lin(&in_names)?,
                in_ba,
                out_proj: lin(&[format!("{a}.{}.weight", if kind == Kind::Delta { "out_proj" } else { "o_proj" })])?,
                input_norm: norm(&format!("{p}.input_layernorm.weight"))?,
                post_norm: norm(&format!("{p}.post_attention_layernorm.weight"))?,
                gate_up: lin(&["gate_proj", "up_proj"].map(|n| format!("{p}.mlp.{n}.weight")))?,
                down: lin(&[format!("{p}.mlp.down_proj.weight")])?,
                q_norm: attn_norm("q_norm")?,
                k_norm: attn_norm("k_norm")?,
                conv_w: if kind == Kind::Delta {
                    let (w, shape) = ck.floats(&format!("{a}.conv1d.weight"))?;
                    ensure!(
                        shape.iter().product::<usize>() == d.conv_width() as usize * CONV_K as usize,
                        "{a}.conv1d.weight: shape {shape:?}"
                    );
                    Some(Array::from_slice(&w, &[d.conv_width(), CONV_K]))
                } else {
                    None
                },
                a_log: delta_vec("A_log")?,
                dt_bias: delta_vec("dt_bias")?,
                delta_norm: delta_vec("norm.weight")?,
            })
        };
        let layers = types
            .iter()
            .enumerate()
            .map(|(i, kind)| {
                let kind = match kind.as_str() {
                    Some("linear_attention") => Kind::Delta,
                    Some("full_attention") => Kind::Attn,
                    other => bail!("layer {i}: unknown type {other:?}"),
                };
                load_layer(&format!("{text}.layers.{i}"), kind)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut mtp = if ck.has("mtp.fc.weight") {
            let layer = load_layer("mtp.layers.0", Kind::Attn)?;
            let stages = Stages::of(&layer, d, true)?;
            let has_head = ck.has("mtp.draft_lm_head.weight");
            let ids = if has_head {
                draft_ids(dir, d.vocab)?.context("mtp.draft_lm_head without its draft vocabulary")?
            } else {
                (0..d.vocab as u32).collect()
            };
            Some(Mtp {
                pre_e: norm("mtp.pre_fc_norm_embedding.weight")?,
                pre_h: norm("mtp.pre_fc_norm_hidden.weight")?,
                fc: lin(&["mtp.fc.weight".into()])?,
                norm: norm("mtp.norm.weight")?,
                head: if has_head { Some(lin(&["mtp.draft_lm_head.weight".into()])?) } else { None },
                ids,
                adaptive: None,
                layer,
                stages,
            })
        } else {
            None
        };
        let embed_base = format!("{text}.embed_tokens");
        let embed = dense(&[format!("{embed_base}.weight")], mats(&[format!("{embed_base}.weight")])?)?;
        let head_base = if ck.has("lm_head.weight") { "lm_head".to_string() } else { embed_base.clone() };
        let head = if ck.is_quant(&head_base) || head_base != embed_base {
            Some(lin(&[format!("{head_base}.weight")])?)
        } else {
            None
        };
        #[allow(clippy::collapsible_if)]
        // draft_ids reads files and may fail: only when MTP has a 4-bit head to draw from
        if let (Some(m), Some(source @ Linear::Quant { bits: 4, .. })) = (&mut mtp, &head) {
            if let Some(seed) = draft_ids(dir, d.vocab)? {
                let adaptive = Adaptive::load(dir, d.vocab, &seed)?;
                m.ids = adaptive.ids().to_vec();
                m.head = Some(source.select_rows(&m.ids)?);
                eprintln!(
                    "MLX adaptive draft vocabulary: {} rows (up to {})",
                    m.ids.len(),
                    crate::draft_vocab::MAX_ROWS
                );
                m.adaptive = Some(adaptive);
            }
        }
        let stages = layers.iter().map(|l| Stages::of(l, d, false)).collect::<MResult<Vec<_>>>()?;
        let w = Weights { d, embed, head, final_norm: norm(&format!("{text}.norm.weight"))?, layers, stages, mtp };
        w.eval()?;
        Ok(w)
    }

    fn eval(&self) -> MResult<()> {
        let mut all: Vec<&Array> = vec![&self.embed, &self.final_norm];
        all.extend(self.head.iter().flat_map(Linear::arrays));
        for l in &self.layers {
            for x in [&l.in_proj, &l.out_proj, &l.gate_up, &l.down] {
                all.extend(x.arrays());
            }
            all.extend(l.in_ba.iter().flat_map(Linear::arrays));
            all.extend([&l.input_norm, &l.post_norm]);
            all.extend(
                l.q_norm
                    .iter()
                    .chain(&l.k_norm)
                    .chain(&l.conv_w)
                    .chain(&l.a_log)
                    .chain(&l.dt_bias)
                    .chain(&l.delta_norm),
            );
        }
        all.extend(self.stages.iter().flat_map(Stages::arrays));
        if let Some(m) = &self.mtp {
            all.extend([&m.pre_e, &m.pre_h, &m.norm]);
            all.extend(m.fc.arrays());
            all.extend(m.head.iter().flat_map(Linear::arrays));
            all.extend(m.stages.arrays());
        }
        mlx_rs::transforms::eval(all)
    }

    /// Learn only committed ids. Draft distributions already drawn retain their token ids and probabilities.
    pub fn learn_drafts(&mut self, ids: &[u32]) -> Result<()> {
        let (Some(m), Some(source)) = (&mut self.mtp, &self.head) else { return Ok(()) };
        let Some(adaptive) = &mut m.adaptive else { return Ok(()) };
        let (head, row_ids) = (&mut m.head, &mut m.ids);
        adaptive.learn(ids, |rows, tiles| {
            // Keep the old head live until every packed-row write succeeds (no partial publication on OOM).
            let old = row_ids.len();
            let mut new = head.as_ref().unwrap().clone();
            if rows.len() > old {
                new.append(&source.select_rows(&rows[old..])?)?;
            }
            for at in tiles.iter().map(|t| t * 16).filter(|&at| at < old) {
                new.replace_tile(at as i32, &source.select_rows(&rows[at..at + 16])?)?;
            }
            mlx_rs::transforms::eval(new.arrays())?;
            *head = Some(new);
            *row_ids = rows.to_vec();
            Ok(())
        })
    }

    pub fn counts(&self) -> (usize, usize) {
        let n_attn = self.layers.iter().filter(|l| l.kind == Kind::Attn).count();
        (n_attn, self.layers.len() - n_attn)
    }

    pub fn positions(&self) -> Vec<At> {
        let (mut ai, mut di) = (0, 0);
        self.layers
            .iter()
            .enumerate()
            .map(|(li, l)| {
                let at = At { li, ai, di };
                if l.kind == Kind::Delta {
                    di += 1
                } else {
                    ai += 1
                }
                at
            })
            .collect()
    }

    pub fn layers(&mut self, m: &mut impl Mix, mut h: Array, prompt: bool) -> MResult<Array> {
        let prefill = h.shape()[0] > COMPILE_MAX;
        // Short prompts vary in length: tracing each length causes seconds of cold compilation.
        // Keep decode shapes compiled, but run short prompt stages directly.
        let small = if prefill { h.shape()[0] == FULL_CHUNK } else { !prompt };
        let mut pending = std::collections::VecDeque::<Array>::new();
        for at in self.positions() {
            if pending.len() == 2 {
                pending.pop_front().unwrap().eval()?;
            }
            h = run_layer(&self.layers[at.li], &mut self.stages[at.li], at, m, &h, small)?;
            if prefill {
                m.submit(&h)?;
                pending.push_back(h.clone());
            }
        }
        Ok(h)
    }
}

fn run_layer(l: &Layer, st: &mut Stages, at: At, m: &mut impl Mix, h: &Array, small: bool) -> MResult<Array> {
    let with =
        |front: Vec<Array>, rest: &[Array]| -> Vec<Array> { front.into_iter().chain(rest.iter().cloned()).collect() };
    match l.kind {
        Kind::Attn => {
            let (cos, sin) = m.rope();
            let r = st.pre.call(small, &with(vec![h.clone(), cos, sin], &st.pre_args))?;
            let o = m.attend(at.ai, &r[0], &r[2], &r[3], &r[4], &r[5])?;
            Ok(st.tail.call(small, &with(vec![h.clone(), o, r[1].clone()], &st.tail_args))?.remove(0))
        }
        Kind::Delta => {
            let r = st.pre.call(small, &with(vec![h.clone()], &st.pre_args))?;
            let y = m.delta(at.di, l, small, &r[0], &r[1], &r[2], &r[3], &r[4])?;
            Ok(st.tail.call(small, &with(vec![h.clone(), y], &st.tail_args))?.remove(0))
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct At {
    pub li: usize,
    pub ai: usize,
    pub di: usize,
}

/// The cache-dependent parts of a step. Arrays have no batch dimension: each segment is one sequence.
pub trait Mix {
    /// Start computing `h` and the cache writes so far (the step's end waits for them)
    fn submit(&self, h: &Array) -> MResult<()>;
    /// Split-half RoPE tables [T][1][ROT], f32.
    fn rope(&self) -> (Array, Array);
    /// q [T][Nq][256] f32; the new keys and values quantize_kv'd: int8 [T][Nkv][256], f32 scales [T][Nkv][1];
    /// result [T][Nq][256] f32.
    fn attend(&mut self, ai: usize, q: &Array, kq: &Array, ks: &Array, vq: &Array, vs: &Array) -> MResult<Array>;
    /// qkv [T][C], z [T][Hv*Dv], beta, log decay and decay [T][Hv]; result [T][Hv*Dv]. `small`: compiled stages.
    #[allow(clippy::too_many_arguments)]
    fn delta(
        &mut self,
        di: usize,
        layer: &Layer,
        small: bool,
        qkv: &Array,
        z: &Array,
        beta: &Array,
        lg: &Array,
        e: &Array,
    ) -> MResult<Array>;
}

pub fn rms(x: &Array, w: &Array) -> MResult<Array> {
    fast::rms_norm(x, Some(w), EPS)
}

/// Split-half RoPE on the first 64 dimensions; the remaining 192 pass through unchanged.
pub fn rope(x: &Array, cos: &Array, sin: &Array) -> MResult<Array> {
    let dtype = x.dtype();
    let x = x.as_dtype(Dtype::Float32)?;
    let parts = ops::split_at_indices(&x, &[ROT], -1)?;
    let halves = ops::split_equal(&parts[0], 2, -1)?;
    let rot = ops::concatenate(&[ops::negative(&halves[1])?, halves[0].clone()], -1)?;
    let rotated = ops::add(ops::multiply(&parts[0], cos)?, ops::multiply(&rot, sin)?)?;
    ops::concatenate(&[rotated, parts[1].clone()], -1)?.as_dtype(dtype)
}

/// CUDA's cache quantization per row (position and head) of x [T][NKV][HD]: int8 q = round(x * (127 / max|x|)) (0 for a
/// zero row) and the scale max|x| / 127 [T][NKV][1]
pub fn quantize_kv(x: &Array) -> MResult<(Array, Array)> {
    let mx = ops::max_axis(ops::abs(x)?, -1, true)?;
    let scale = ops::divide(&mx, Array::from_f32(127.0))?;
    let inv = ops::select(
        ops::gt(&mx, Array::from_f32(0.0))?,
        ops::divide(Array::from_f32(127.0), &mx)?,
        Array::from_f32(0.0),
    )?;
    let q = ops::round(ops::multiply(x, inv)?, None)?.as_dtype(Dtype::Int8)?;
    Ok((q, scale))
}

/// o times the per-channel sigmoid gate, in f32.
pub fn gate(o: &Array, g: &Array) -> MResult<Array> {
    ops::multiply(o, nn::sigmoid(g.as_dtype(Dtype::Float32)?)?)
}

/// silu(a) * b in f32, rounded to f16 (the next matrix's input).
pub fn swiglu(a: &Array, b: &Array) -> MResult<Array> {
    ops::multiply(nn::silu(a.as_dtype(Dtype::Float32)?)?, b.as_dtype(Dtype::Float32)?)?.as_dtype(Dtype::Float16)
}

/// Once per process, before MLX's first operation. Changing prompt shapes must not trip the graph-cache warning, and freed buffers
/// must not accumulate without a bound.
pub fn init() -> MResult<()> {
    static ONCE: std::sync::Once = std::sync::Once::new();
    let mut r = Ok(());
    ONCE.call_once(|| {
        for (k, v) in [("MLX_ENABLE_CACHE_THRASHING_CHECK", "0"), ("MLX_USE_CUDA_GRAPHS", "0")] {
            if std::env::var_os(k).is_none() {
                // SAFETY: startup, before MLX and before any thread that reads these variables.
                unsafe { std::env::set_var(k, v) };
            }
        }
        let cache = if cfg!(has_metal) { 256 << 20 } else { 2 << 30 };
        r = mlx_rs::memory::set_cache_limit(cache).map(|_| ());
    });
    r
}

/// An array's values on the host as f32, in row-major order.
pub fn host(a: &Array) -> Result<Vec<f32>> {
    let a = a.as_dtype(Dtype::Float32)?.contiguous()?;
    a.eval()?;
    Ok(a.try_as_slice::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?.to_vec())
}
