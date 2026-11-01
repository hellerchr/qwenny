//! Qwen3.5 (text; Qwen3.8 has the same architecture) on one CUDA GPU: 3 of every 4 layers Gated DeltaNet
//! (deltanet.rs), every 4th gated full attention (attn.rs), each followed by a SwiGLU MLP; pre-norm residuals with
//! RMSNorm scaling by 1 + weight. Both mixers read the split-K partial products of their in-projection from
//! `Bufs::part` and write MIX columns per token to `Bufs::attn`; the out-projection, the residual and the MLP run
//! here. Kernels: kernels/qwen.cu, built per model size.
mod attn;
mod deltanet;
mod draft;
pub mod train;

use super::core::{self, Epi, Gemm, Graphs, MAX_OUT, MAX_TOK, Mat, SPEC_TOK, Step, cfg};
use crate::engine::{DraftSampling, Engine, Tok};
use crate::checkpoint::Checkpoint;
use crate::lora::Adapter;
use anyhow::{Context, Result, bail, ensure};
use attn::Attn;
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, PushKernelArg};
use deltanet::DeltaNet;
use draft::DraftHead;
use crate::cuda::vmm::{HostMapped, Pool};

/// Device memory left to other allocations (CUDA graphs, the driver) when the per-slot caches take the rest
const PAGE_RESERVE: usize = 512 << 20;
use std::{fs, path::Path, sync::Arc};

/// Full attention: key/value heads and head dim; Gated DeltaNet: key heads, key and value head dims (all sizes)
const NKV: usize = 4;
const HD: usize = 256;
const LK_HEADS: usize = 16;
const LKD: usize = 128;
const LVD: usize = 128;
pub const CONV_K: usize = 4;
/// RMSNorm epsilon; must match EPS in kernels/qwen.cu (attn_prep and the DeltaNet kernels; rmsnorm takes it)
const EPS: f32 = 1e-6;

/// What differs between the sizes (config.json); the kernels are built for each (kernels/qwen.cu).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Dims {
    pub h: usize,
    pub ff: usize,
    /// full attention: query heads
    pub nh: usize,
    /// Gated DeltaNet: value heads
    pub lv_heads: usize,
    pub vocab: usize,
}

impl Dims {
    /// The sizes with kernels: name, dims, PTX
    const SIZES: [(&str, Dims, &str); 3] = [
        ("Qwen3.5-4B", Dims { h: 2560, ff: 9216, nh: 16, lv_heads: 32, vocab: 248320 },
         include_str!(concat!(env!("OUT_DIR"), "/qwen4b.ptx"))),
        ("Qwen3.5-9B", Dims { h: 4096, ff: 12288, nh: 16, lv_heads: 32, vocab: 248320 },
         include_str!(concat!(env!("OUT_DIR"), "/qwen9b.ptx"))),
        ("Qwen3.8-27B", Dims { h: 5120, ff: 17408, nh: 24, lv_heads: 48, vocab: 248320 },
         include_str!(concat!(env!("OUT_DIR"), "/qwen27b.ptx"))),
    ];

    /// In-projection rows: [q_proj (per head q, gate); k_proj; v_proj]
    fn attn_in(&self) -> usize {
        2 * self.nh * HD + 2 * NKV * HD
    }

    /// Conv channels of the Gated DeltaNet: q, k, v
    fn lqkv(&self) -> usize {
        2 * LK_HEADS * LKD + self.lv_heads * LVD
    }

    /// In-projection rows: [in_proj_qkv (q, k, v); in_proj_z] (in_proj_b, in_proj_a are a product of their own:
    /// checkpoints keep them 16-bit when the rest is quantized)
    fn delta_in(&self) -> usize {
        self.lqkv() + self.lv_heads * LVD
    }

    /// Mixer output width (both kinds): o_proj / out_proj input
    fn mix(&self) -> usize {
        self.nh * HD
    }

    /// The size of a text config (`text_config` of the checkpoint's config.json) and its inference kernels
    fn of(tc: &serde_json::Value) -> Result<(Dims, &'static str)> {
        let d = Self::parse(tc)?;
        match Self::SIZES.iter().find(|(_, s, _)| *s == d) {
            Some(&(_, _, ptx)) => Ok((d, ptx)),
            None => bail!("a Qwen3.5 of hidden size {}, {} query and {} DeltaNet value heads: kernels are built for {}",
                          d.h, d.nh, d.lv_heads, Self::SIZES.map(|s| s.0).join(", ")),
        }
    }

    /// The size of a text config, checked against what the kernels fix (any size)
    fn parse(tc: &serde_json::Value) -> Result<Dims> {
        let get = |k: &str| tc[k].as_u64().map(|v| v as usize);
        let fixed = [("num_key_value_heads", NKV), ("head_dim", HD), ("linear_num_key_heads", LK_HEADS),
                     ("linear_key_head_dim", LKD), ("linear_value_head_dim", LVD), ("linear_conv_kernel_dim", CONV_K)];
        for (key, want) in fixed {
            ensure!(get(key) == Some(want), "config {key} = {}, expected {want}", tc[key]);
        }
        ensure!(tc["rms_norm_eps"].as_f64().is_some_and(|e| (e - EPS as f64).abs() < 1e-12), "rms_norm_eps {}", tc["rms_norm_eps"]);
        ensure!(tc["attn_output_gate"].as_bool() == Some(true) && tc["attention_bias"].as_bool() == Some(false),
                "attention without the output gate or with biases");
        let rope = &tc["rope_parameters"];
        ensure!(rope["rope_type"] == "default" && rope["rope_theta"].as_f64() == Some(attn::THETA)
                && rope["partial_rotary_factor"].as_f64() == Some(attn::ROT as f64 / HD as f64), "rope {rope}");
        let d = Dims { h: get("hidden_size").context("hidden_size")?, ff: get("intermediate_size").context("intermediate_size")?,
                       nh: get("num_attention_heads").context("num_attention_heads")?,
                       lv_heads: get("linear_num_value_heads").context("linear_num_value_heads")?,
                       vocab: get("vocab_size").context("vocab_size")? };
        Ok(d)
    }
}

/// The weight of a Qwen3.5 RMSNorm as the kernels take it: 1 + w (the model scales by 1 + weight)
fn norm(w: &Checkpoint, name: &str) -> Result<Vec<f32>> {
    Ok(w.floats(name)?.0.into_iter().map(|w| 1.0 + w).collect())
}

struct Kernels {
    embed: CudaFunction,
    embed_q8: CudaFunction,
    rmsnorm: CudaFunction,
    add_rmsnorm: CudaFunction,
    to_xin: CudaFunction,
    mtp_in: CudaFunction,
    sum_parts: CudaFunction,
    attn: attn::Kernels,
    delta: deltanet::Kernels,
}

enum Mixer {
    Delta(DeltaNet),
    Attn(Attn),
}

struct Layer {
    mixer: Mixer,
    /// [q_proj; k_proj; v_proj] or [in_proj_qkv; in_proj_z]
    in_proj: Mat,
    /// DeltaNet: [in_proj_b; in_proj_a]
    in_ba: Option<Mat>,
    /// o_proj or out_proj
    out_proj: Mat,
    input_norm: CudaSlice<f32>,
    post_norm: CudaSlice<f32>,
    /// [gate_proj; up_proj], packed for the SwiGLU epilogue
    gate_up: Mat,
    down: Mat,
}

impl Layer {
    /// The layer whose tensors start with `p` (model.language_model.layers.{i}, or mtp.layers.0), of kind `kind`
    /// (config.json layer_types)
    #[allow(clippy::too_many_arguments)]
    fn load(s: &Arc<CudaStream>, pool: &Arc<Pool>, w: &Checkpoint, d: &Dims, p: &str, kind: &str, slots: usize,
            max_ctx: usize, int8: bool, adapter: Option<&Adapter>) -> Result<Self> {
        let mat = |a: &str, ns: &[&str], silu: bool| {
            Mat::stack(s, w, &ns.iter().map(|n| format!("{p}.{a}.{n}.weight")).collect::<Vec<_>>(), silu, int8, adapter)
        };
        let in_ba = match kind {
            "linear_attention" => Some(mat("linear_attn", &["in_proj_b", "in_proj_a"], false)?),
            _ => None,
        };
        let (mixer, in_proj, out_proj, rows) = match kind {
            "linear_attention" => (
                Mixer::Delta(DeltaNet::new(s, pool, w, d, &format!("{p}.linear_attn"), slots)?),
                mat("linear_attn", &["in_proj_qkv", "in_proj_z"], false)?,
                mat("linear_attn", &["out_proj"], false)?,
                d.delta_in(),
            ),
            "full_attention" => (
                Mixer::Attn(Attn::new(s, pool, w, &format!("{p}.self_attn"), slots, max_ctx)?),
                mat("self_attn", &["q_proj", "k_proj", "v_proj"], false)?,
                mat("self_attn", &["o_proj"], false)?,
                d.attn_in(),
            ),
            other => bail!("{p}: unknown type {other:?}"),
        };
        ensure!(in_proj.tiles * 16 == rows && out_proj.cols == d.mix(), "{p}: in-projection of {} rows", in_proj.tiles * 16);
        let fl = |n: &str| -> Result<CudaSlice<f32>> { Ok(s.clone_htod(&norm(w, &format!("{p}.{n}.weight"))?)?) };
        Ok(Layer {
            mixer,
            in_proj,
            in_ba,
            out_proj,
            input_norm: fl("input_layernorm")?,
            post_norm: fl("post_attention_layernorm")?,
            gate_up: mat("mlp", &["gate_proj", "up_proj"], true)?,
            down: mat("mlp", &["down_proj"], false)?,
        })
    }
}

/// Step activations and the mixers' scratch, sized for MAX_TOK tokens. The matrix products' inputs (xn, mix, mid)
/// are f16 (XIN in the kernels).
pub struct Bufs {
    h: CudaSlice<f32>,
    xn: CudaSlice<u16>,
    /// split-K partial products of the plain matrix products, [split][n][rows]: the mixers read their in-projection's
    pub part: CudaSlice<f32>,
    /// the partial products of DeltaNet's [in_proj_b; in_proj_a]
    pub ba: CudaSlice<f32>,
    /// the mixer's raw output [n][mix] (DeltaNet: before its gated norm)
    pub o: CudaSlice<f32>,
    /// mixer output [n][mix], the out-projection's input
    pub mix: CudaSlice<u16>,
    mid: CudaSlice<u16>,
    attn: attn::Scratch,
    /// delta_conv output [MAX_TOK][deltanet::cv_width]
    cv: CudaSlice<f32>,
}

/// What a mixer reads besides its own weights and state
pub struct Ctx<'a> {
    s: &'a Arc<CudaStream>,
    st: &'a Step,
    b: &'a Bufs,
    k: &'a Kernels,
    d: &'a Dims,
}

/// The layers with what a step over them needs: the model without its embedding and lm_head, or one layer
/// (layercheck).
struct Stack {
    s: Arc<CudaStream>,
    /// the pages of the per-slot caches (KV, DeltaNet states)
    pool: Arc<Pool>,
    k: Kernels,
    mm: Gemm,
    /// the step's tokens and the sampler
    st: Step,
    d: Dims,
    layers: Vec<Layer>,
    b: Bufs,
}

impl Stack {
    /// `int8`: int8 weights (else f16); `only`: just this layer; `adapter`: a LoRA merged into the weights. Also
    /// returns the checkpoint.
    fn load(dir: &Path, max_ctx: usize, slots: usize, int8: bool, only: Option<usize>, adapter: Option<&Adapter>)
    -> Result<(Self, Checkpoint)> {
        let config: serde_json::Value = serde_json::from_str(&fs::read_to_string(dir.join("config.json"))?)?;
        let tc = config.get("text_config").unwrap_or(&config);
        let (d, ptx) = Dims::of(tc)?;
        ensure!(max_ctx.is_multiple_of(64) && max_ctx > 0, "context {max_ctx}: a multiple of 64");
        let (s, m) = core::open(ptx)?;
        let f = |n: &str| m.load_function(n);
        let k = Kernels { embed: f("embed")?, embed_q8: f("embed_q8")?, rmsnorm: f("rmsnorm")?, add_rmsnorm: f("add_rmsnorm")?,
                          to_xin: f("to_xin")?, mtp_in: f("mtp_in")?, sum_parts: f("sum_parts")?,
                          attn: attn::Kernels::load(&s, &m, &d)?, delta: deltanet::Kernels::load(&m)? };
        let w = Checkpoint::open(dir)?;
        let pool = Pool::new(&s)?;
        let types = tc["layer_types"].as_array().context("layer_types")?;
        let layers = types.iter().enumerate().filter(|(i, _)| only.is_none_or(|o| o == *i))
            .map(|(i, kind)| Layer::load(&s, &pool, &w, &d, &format!("model.language_model.layers.{i}"), kind.as_str().unwrap_or("?"), slots,
                                max_ctx, int8, adapter))
            .collect::<Result<Vec<_>>>()?;
        ensure!(!layers.is_empty(), "no layer {only:?}");
        let z = |n: usize| s.alloc_zeros::<f32>(n);
        let zh = |n: usize| s.alloc_zeros::<u16>(n);
        let b = Bufs {
            h: z(MAX_TOK * d.h)?,
            xn: zh(MAX_TOK * d.h)?,
            part: z(core::part_len(layers.iter().flat_map(|l| [&l.in_proj, &l.out_proj, &l.down])))?,
            ba: z(core::part_len(layers.iter().filter_map(|l| l.in_ba.as_ref())).max(1))?,
            o: z(MAX_TOK * d.mix())?,
            mix: zh(MAX_TOK * d.mix())?,
            mid: zh(MAX_TOK * d.ff)?,
            attn: attn::Scratch::new(&s, &d)?,
            cv: z(MAX_TOK * deltanet::cv_width(&d))?,
        };
        let st = Step::new(&s, &m, slots, max_ctx, d.vocab, attn::segments(d.nh / NKV))?;
        let x = Stack { mm: Gemm::load(&s, &m, true, d.h)?, k, st, d, layers, b, s, pool };
        x.set_budget()?;
        Ok((x, w))
    }

    /// Give the per-slot caches what device memory is free now, less PAGE_RESERVE
    fn set_budget(&self) -> Result<()> {
        self.s.synchronize()?;
        let (free, _) = cudarc::driver::result::mem_get_info()?;
        self.pool.set_budget(free.saturating_sub(PAGE_RESERVE) / self.pool.page() + self.pool.usage().0);
        Ok(())
    }

    /// Map the per-slot memory a step over `toks` writes: each slot's DeltaNet states and its KV up to the
    /// segment's last position (in `extra` too: the MTP's layer)
    fn map_toks(&mut self, toks: &[Tok], extra: Option<&mut Layer>) -> Result<()> {
        let mut end: Vec<(usize, usize)> = Vec::new();
        for t in toks {
            match end.iter_mut().find(|e| e.0 == t.slot as usize) {
                Some(e) => e.1 = e.1.max(t.pos as usize + 1),
                None => end.push((t.slot as usize, t.pos as usize + 1)),
            }
        }
        let pool = self.pool.clone();
        for mx in self.mixers(extra) {
            for &(slot, len) in &end {
                let ok = match mx {
                    Mixer::Attn(a) => a.map(slot, len)?,
                    Mixer::Delta(d) => d.map(slot)?,
                };
                let (used, budget) = pool.usage();
                ensure!(ok, "the cache is full: {used} of {budget} pages of {} MiB in use (slot {slot} at {len})",
                        pool.page() >> 20);
            }
        }
        Ok(())
    }

    /// The layers' per-slot caches (and those of `extra`: the MTP's layer)
    fn mixers<'a>(&'a mut self, extra: Option<&'a mut Layer>) -> impl Iterator<Item = &'a mut Mixer> {
        self.layers.iter_mut().chain(extra).map(|l| &mut l.mixer)
    }

    fn gemm<Y: cudarc::driver::DeviceRepr>(&self, w: &Mat, x: &CudaSlice<u16>, y: &CudaSlice<Y>, epi: Epi) -> Result<usize> {
        self.mm.run(&self.s, &self.st, w, x, y, epi)
    }

    /// h += the partial products in `part`; xn = w_next * h / rms(h)
    fn add_rmsnorm(&self, split: usize, w_next: &CudaSlice<f32>) -> Result<()> {
        let (b, n) = (&self.b, self.st.n);
        let (split, ni, eps) = (split as i32, n as i32, EPS);
        unsafe {
            self.s.launch_builder(&self.k.add_rmsnorm).arg(&b.h).arg(&b.part).arg(&split).arg(&ni).arg(w_next)
                .arg(&b.xn).arg(&eps).launch(cfg((n, 1, 1), 512))?;
        }
        Ok(())
    }

    /// The mixer of layer `l` and its projections: xn -> part (in_proj) -> mix (mixer) -> part (out_proj).
    /// Returns the out_proj split factor.
    fn mix(&self, l: &Layer) -> Result<usize> {
        let b = &self.b;
        let sp = self.gemm(&l.in_proj, &b.xn, &b.part, Epi::Plain)?;
        let c = Ctx { s: &self.s, st: &self.st, b, k: &self.k, d: &self.d };
        match &l.mixer {
            Mixer::Delta(d) => {
                let bsp = self.gemm(l.in_ba.as_ref().context("DeltaNet layer without in_proj_b/a")?, &b.xn, &b.ba, Epi::Plain)?;
                d.run(&c, sp, bsp)?
            }
            Mixer::Attn(a) => a.run(&c, sp)?,
        }
        self.gemm(&l.out_proj, &b.mix, &b.part, Epi::Plain)
    }
}

/// The embedding table: bf16, or int8 as a checkpoint stores it (value + 128, a scale per row and 128 columns; in host
/// memory the device reads)
enum Embed {
    Bf16(CudaSlice<u16>),
    Q8(HostMapped, HostMapped),
}

impl Embed {
    fn load(s: &Arc<CudaStream>, w: &Checkpoint, name: &str) -> Result<Self> {
        let base = name.strip_suffix(".weight").unwrap_or(name);
        if w.is_quant(base) {
            let q = w.quant_matrix(base)?;
            if q.bits == 8 && q.group == 128 && q.zero_points().is_some_and(|z| z.iter().all(|&z| z == 128)) {
                // in host memory: a step reads a few rows (1.3 GB of device memory more for the caches: 27B)
                let bytes: Vec<u8> = q.q.iter().flat_map(|w| w.to_le_bytes()).collect();
                let sc: Vec<u8> = q.scales.iter().flat_map(|f| f.to_le_bytes()).collect();
                return Ok(Embed::Q8(HostMapped::new(s, &bytes)?, HostMapped::new(s, &sc)?));
            }
        }
        Ok(Embed::Bf16(s.clone_htod(&w.bf16(name)?.0)?))
    }

    /// out[t] = the embedding of ids[t] for n tokens (f32 [n][H]); clears the first nk argmax keys
    fn launch(&self, s: &Arc<CudaStream>, k: &Kernels, ids: &CudaSlice<i32>, out: &CudaSlice<f32>,
              keys: &CudaSlice<u64>, nk: i32, n: usize) -> Result<()> {
        unsafe {
            match self {
                Embed::Bf16(e) => s.launch_builder(&k.embed).arg(e).arg(ids).arg(out).arg(keys).arg(&nk).launch(cfg((n, 1, 1), 256))?,
                Embed::Q8(e, sc) => s.launch_builder(&k.embed_q8).arg(&e.ptr()).arg(&sc.ptr()).arg(ids).arg(out).arg(keys)
                    .arg(&nk).launch(cfg((n, 1, 1), 256))?,
            };
        }
        Ok(())
    }
}

/// CUDA graph key: the step's shape and the attention's key splits (decode, prompt tiles)
type Key = ([usize; 6], (usize, usize, usize, usize));

/// The MTP layer (multi-token prediction; the draft model of speculative decoding), as vLLM's
/// Qwen3_5MultiTokenPredictor: x = fc([pre_e(embed(t_{q+1})); pre_h(h_q)]), one full-attention layer over the MTP's own
/// positions (its own KV cache), norm, the target's output layer: the logits of t_{q+2}. h_q is the target's final
/// hidden state at position q (after its final norm), or for further drafts the MTP's own output state.
struct Mtp {
    layer: Layer,
    /// [H][2 H]
    fc: Mat,
    pre_e: CudaSlice<f32>,
    pre_h: CudaSlice<f32>,
    norm: CudaSlice<f32>,
    /// fc's input [MAX_TOK][2 H] (XIN)
    xin: CudaSlice<u16>,
    /// per MTP token: the row of the residual stream (of the step before) its hidden state comes from
    rows: CudaSlice<i32>,
    /// keyed by the step's shape and whether the hidden states are the MTP's own (the norm before pre_h)
    graphs: Graphs<(Key, bool)>,
    /// the draft head, if the checkpoint names a draft vocabulary; else the drafts read the whole output layer
    head: Option<DraftHead>,
}

pub struct Model {
    x: Stack,
    /// the embedding table
    embed: Embed,
    /// tied to the embedding (4B) or its own matrix (27B)
    lm_head: Mat,
    final_norm: CudaSlice<f32>,
    graphs: Graphs<Key>,
    mtp: Option<Mtp>,
}

impl Model {
    /// `int8`: int8 weights (else f16); `adapter`: a LoRA merged into the weights (stored int4 matrices it changes
    /// are dequantized).
    pub fn load(dir: &Path, max_ctx: usize, slots: usize, int8: bool, adapter: Option<&Adapter>)
    -> Result<Self> {
        let (x, w) = Stack::load(dir, max_ctx, slots, int8, None, adapter)?;
        let e = "model.language_model.embed_tokens.weight".to_string();
        let head = if w.has("lm_head.weight") { "lm_head.weight".to_string() } else { e.clone() };
        let mtp = match w.has("mtp.fc.weight") {
            true => {
                let (s, d) = (&x.s, &x.d);
                let fl = |n: &str| -> Result<CudaSlice<f32>> { Ok(s.clone_htod(&norm(&w, &format!("mtp.{n}.weight"))?)?) };
                Some(Mtp {
                    layer: Layer::load(s, &x.pool, &w, d, "mtp.layers.0", "full_attention", x.st.slots, x.st.max_ctx, int8, None)?,
                    fc: Mat::stack(s, &w, &["mtp.fc.weight".to_string()], false, int8, None)?,
                    pre_e: fl("pre_fc_norm_embedding")?,
                    pre_h: fl("pre_fc_norm_hidden")?,
                    norm: fl("norm")?,
                    xin: s.alloc_zeros(MAX_TOK * 2 * d.h)?,
                    rows: s.alloc_zeros(MAX_TOK)?,
                    graphs: Graphs::new(),
                    head: DraftHead::load(s, &w, dir, d.vocab)?,
                })
            }
            false => None,
        };
        let embed = Embed::load(&x.s, &w, &e)?;
        let lm_head = Mat::stack(&x.s, &w, &[head], false, int8, None)?;
        let final_norm = x.s.clone_htod(&norm(&w, "model.language_model.norm.weight")?)?;
        let model = Model { embed, lm_head, final_norm, graphs: Graphs::new(), x, mtp };
        // the whole model loaded: the caches' pages made at once (Pool::fill)
        model.x.set_budget()?;
        model.x.pool.fill()?;
        let (page, (_, budget)) = (model.x.pool.page(), model.x.pool.usage());
        eprintln!("cache: {:.1} GiB of device memory for the slots' KV and DeltaNet states ({} tokens of KV)",
                  (budget * page) as f64 / 1073741824.0, budget * page / model.kv_bytes_per_token());
        Ok(model)
    }

    /// The kernels of one step over the uploaded tokens.
    fn launch_step(&self) -> Result<()> {
        let x = &self.x;
        let (s, k, b, st) = (&x.s, &x.k, &x.b, &x.st);
        let (eps, nk) = (EPS, st.n_out as i32);
        st.mark_seen(s)?;
        unsafe {
            self.embed.launch(s, k, &st.ids, &b.h, &st.keys, nk, st.n)?;
            s.launch_builder(&k.rmsnorm).arg(&b.h).arg(&x.layers[0].input_norm).arg(&b.xn).arg(&eps)
                .launch(cfg((st.n, 1, 1), 256))?;
        }
        for (i, l) in x.layers.iter().enumerate() {
            let sp = x.mix(l)?;
            x.add_rmsnorm(sp, &l.post_norm)?;
            x.gemm(&l.gate_up, &b.xn, &b.mid, Epi::Silu)?;
            let sp = x.gemm(&l.down, &b.mid, &b.part, Epi::Plain)?;
            x.add_rmsnorm(sp, x.layers.get(i + 1).map_or(&self.final_norm, |l| &l.input_norm))?;
        }
        if st.n_out > 0 {
            x.gemm(&self.lm_head, &b.xn, &st.logits, Epi::Argmax)?;
        }
        Ok(())
    }

    /// The kernels of one MTP step over the uploaded tokens, their hidden states at `m.rows` of the residual stream
    /// `b.h`: the target's (`own` false: its final norm) or the MTP's own of the step before (`own`: the MTP's norm).
    fn launch_mtp(&self, m: &Mtp, own: bool) -> Result<()> {
        let x = &self.x;
        let (s, k, b, st, d) = (&x.s, &x.k, &x.b, &x.st, &x.d);
        let (eps, nk, n, h) = (EPS, st.n_out as i32, st.n as i32, d.h as i32);
        let fw = if own { &m.norm } else { &self.final_norm };
        // the tokens' embeddings into b.o (scratch here), then fc's input
        self.embed.launch(s, k, &st.ids, &b.o, &st.keys, nk, st.n)?;
        unsafe {
            s.launch_builder(&k.mtp_in).arg(&b.o).arg(&b.h).arg(&m.rows).arg(fw).arg(&m.pre_e)
                .arg(&m.pre_h).arg(&m.xin).arg(&eps).launch(cfg((st.n, 1, 1), 256))?;
        }
        // the MTP's residual stream starts at fc's output: h = the sum of its partial products, xn = norm(h)
        let sp = x.gemm(&m.fc, &m.xin, &b.part, Epi::Plain)? as i32;
        unsafe {
            s.launch_builder(&k.sum_parts).arg(&b.part).arg(&sp).arg(&n).arg(&h).arg(&b.h).launch(cfg((st.n, 1, 1), 256))?;
            s.launch_builder(&k.rmsnorm).arg(&b.h).arg(&m.layer.input_norm).arg(&b.xn).arg(&eps)
                .launch(cfg((st.n, 1, 1), 256))?;
        }
        let l = &m.layer;
        let sp = x.mix(l)?;
        x.add_rmsnorm(sp, &l.post_norm)?;
        x.gemm(&l.gate_up, &b.xn, &b.mid, Epi::Silu)?;
        let sp = x.gemm(&l.down, &b.mid, &b.part, Epi::Plain)?;
        x.add_rmsnorm(sp, &m.norm)?;
        if st.n_out > 0 {
            x.gemm(m.head.as_ref().map_or(&self.lm_head, |h| &h.mat), &b.xn, &st.logits, Epi::Argmax)?;
        }
        Ok(())
    }

    /// Bytes of KV cache per position of a slot (keys, values and their scales, every attention layer)
    fn kv_bytes_per_token(&self) -> usize {
        let attn = self.x.layers.iter().chain(self.mtp.as_ref().map(|m| &m.layer)).filter(|l| matches!(l.mixer, Mixer::Attn(_)));
        attn.count() * NKV * (2 * HD + 8)
    }

    /// Map the per-slot memory a step over `toks` writes (in the MTP's layer too)
    fn map_step(&mut self, toks: &[Tok]) -> Result<()> {
        self.x.map_toks(toks, self.mtp.as_mut().map(|m| &mut m.layer))
    }

    /// MTP drafts drawn on the device (Engine::draft): each step's drafts from its logits (the draft head's
    /// or the output layer's) under the rows' sampling (argmax rows: top_k 1, the argmax), fed to the next step on the
    /// device; the chains come back once at the end
    fn draft_chains(&mut self, toks: &[Tok], rows: &[usize], out: &[usize], k: usize, sampling: &[Option<DraftSampling>])
    -> Result<Vec<Vec<u32>>> {
        if k == 0 || out.is_empty() {
            self.mtp_launch(toks, rows, &[], false, false)?;
            return Ok(vec![Vec::new(); out.len()]);
        }
        self.mtp_launch(toks, rows, out, false, false)?;
        self.launch_drafts(sampling, 0)?;
        let mut cur: Vec<Tok> = out.iter().map(|&o| toks[o]).collect();
        let mut at: Vec<usize> = out.to_vec();
        let all: Vec<usize> = (0..cur.len()).collect();
        for t in 1..k {
            // the next draft: the MTP on its own state and the draft just made (its id on the device), one position on
            for c in cur.iter_mut() {
                *c = Tok { id: 0, slot: c.slot, pos: c.pos + 1 };
            }
            self.mtp_launch(&cur, &at, &all, true, true)?;
            self.launch_drafts(sampling, t)?;
            at = all.clone();
        }
        let slots: Vec<u32> = cur.iter().map(|c| c.slot).collect();
        self.x.st.chain_drafts(&self.x.s, &slots, k)
    }

    /// draft_sample over the last MTP step's rows; `t`: the drafts' index in their chains
    fn launch_drafts(&mut self, sampling: &[Option<DraftSampling>], t: usize) -> Result<()> {
        let params: Vec<[f32; 8]> = sampling.iter().map(|s| match s {
            Some(s) => {
                let mut r = s.row;
                (r[4], r[5]) = (s.u.get(t).copied().unwrap_or(0.5), t as f32);
                r
            }
            None => crate::sample::argmax_draft_row(t),
        }).collect();
        let m = self.mtp.as_ref().context("no MTP layer in this model")?;
        let (width, ld, ids) = match &m.head {
            Some(h) => (h.mat.valid, h.mat.tiles * 16, Some(&h.ids)),
            None => (self.x.d.vocab, self.x.d.vocab, None),
        };
        self.x.st.draft_sample(&self.x.s, width, ld, ids, &params)
    }

    /// One MTP step: tokens `toks`, their hidden states at `rows` of the step before, logits at `out`; `device_ids`: its
    /// tokens are the last draft_sample's
    fn mtp_launch(&mut self, toks: &[Tok], rows: &[usize], out: &[usize], own: bool, device_ids: bool) -> Result<()> {
        self.map_step(toks)?;
        let r: Vec<i32> = rows.iter().map(|&r| r as i32).collect();
        let m = self.mtp.as_mut().context("no MTP layer in this model")?;
        self.x.s.memcpy_htod(&r, &mut m.rows.slice_mut(0..r.len()))?;
        let x = &mut self.x;
        x.st.upload(&x.s, toks, out, None)?;
        if device_ids {
            x.st.ids_from_drafts(&x.s)?;
        }
        let key = ((x.st.shape(), Attn::graph_key(&x.k.attn, &x.st)), own);
        let m = self.mtp.as_ref().unwrap();
        m.graphs.run(&self.x.s, self.x.st.n, key, || self.launch_mtp(m, own))
    }

}

/// One layer on its own (layercheck): its mixer and projections on given inputs.
pub struct LayerCheck(Stack);

impl LayerCheck {
    pub fn load(dir: &Path, layer: usize, int8: bool) -> Result<Self> {
        Ok(LayerCheck(Stack::load(dir, 4096, 4, int8, Some(layer), None)?.0))
    }

    /// The mixer output (after out_proj) for the step `toks` whose mixer inputs (input_layernorm output) are `xn`
    /// [n][h]; returns [n][h].
    pub fn run(&mut self, toks: &[Tok], xn: &[f32]) -> Result<Vec<f32>> {
        let x = &mut self.0;
        let (n, h) = (toks.len(), x.d.h);
        ensure!(xn.len() == n * h, "{h} inputs per token");
        x.map_toks(toks, None)?;
        x.st.upload(&x.s, toks, &[], None)?;
        let xn: Vec<u16> = xn.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        x.s.memcpy_htod(&xn, &mut x.b.xn.slice_mut(0..n * h))?;
        let x = &self.0;
        let sp = x.mix(&x.layers[0])?;
        let (spi, ni, ld) = (sp as i32, n as i32, h as i32);
        unsafe {
            x.s.launch_builder(&x.k.sum_parts).arg(&x.b.part).arg(&spi).arg(&ni).arg(&ld).arg(&x.b.h)
                .launch(cfg((n, 1, 1), 256))?;
        }
        Ok(x.s.clone_dtoh(&x.b.h.slice(0..n * h))?)
    }

    /// Most tokens in one step
    pub fn max_tok(&self) -> usize {
        MAX_TOK
    }
}

impl Engine for Model {
    fn slots(&self) -> usize {
        self.x.st.slots
    }

    fn can_save(&self) -> bool {
        true
    }

    fn cache_room(&self) -> Option<usize> {
        Some(self.x.pool.available() * self.x.pool.page())
    }

    fn save_state(&mut self, slot: u32) -> Result<Vec<u8>> {
        let (s, slot) = (self.x.s.clone(), slot as usize);
        s.synchronize()?;
        let mut data = crate::sessions::host_buf(self.image_bytes(0));
        data.clear();
        let mut at = None;
        for mx in self.x.mixers(None) {
            if let Mixer::Delta(d) = mx {
                let p = d.save(&s, slot, &mut data)?;
                ensure!(at.is_none_or(|a| a == p), "slot {slot}: DeltaNet layers at {at:?} and {p}");
                at = Some(p);
            }
        }
        for w in self.x.st.seen(&s, slot)? {
            data.extend_from_slice(&w.to_le_bytes());
        }
        Ok(data)
    }

    fn load_state(&mut self, slot: u32, data: &[u8], pos: usize) -> Result<bool> {
        let (s, slot) = (self.x.s.clone(), slot as usize);
        let mut o = data;
        for mx in self.x.mixers(None) {
            if let Mixer::Delta(d) = mx {
                let n = d.saved_bytes();
                ensure!(o.len() >= n, "slot {slot}: a state of {} bytes", data.len());
                if !d.map(slot)? {
                    return Ok(false);
                }
                d.load(&s, slot, &o[..n], pos as i32)?;
                o = &o[n..];
            }
        }
        let words: Vec<u32> = o.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        self.x.st.set_seen(&s, slot, &words)?;
        Ok(true)
    }

    fn save_kv(&mut self, slot: u32, from: usize, to: usize) -> Result<Vec<u8>> {
        self.x.s.synchronize()?;
        let mut attn: Vec<&Attn> = Vec::new();
        for l in self.x.layers.iter().chain(self.mtp.as_ref().map(|m| &m.layer)) {
            if let Mixer::Attn(a) = &l.mixer {
                attn.push(a);
            }
        }
        // one allocation (no copy as it would grow), each layer's part copied into it
        let n = Attn::saved_bytes(from, to);
        let mut data = crate::sessions::host_buf(attn.len() * n);
        for (a, out) in attn.iter().zip(data.chunks_exact_mut(n.max(1))) {
            a.save(slot as usize, from, to, out)?;
        }
        Ok(data)
    }

    fn load_kv(&mut self, slot: u32, from: usize, to: usize, data: &[u8]) -> Result<bool> {
        let mut o = data;
        for mx in self.x.mixers(self.mtp.as_mut().map(|m| &mut m.layer)) {
            if let Mixer::Attn(a) = mx {
                match a.load(slot as usize, from, to, o)? {
                    Some(n) => o = &o[n..],
                    None => return Ok(false),
                }
            }
        }
        ensure!(o.is_empty(), "a KV cache of {} bytes too many", o.len());
        Ok(true)
    }

    fn image_bytes(&self, len: usize) -> usize {
        let layers = self.x.layers.iter().chain(self.mtp.as_ref().map(|m| &m.layer));
        layers.map(|l| match &l.mixer {
            Mixer::Attn(_) => Attn::saved_bytes(0, len),
            Mixer::Delta(d) => d.saved_bytes(),
        }).sum::<usize>() + self.x.st.vocab / 8
    }

    fn truncate(&mut self, slot: u32, from: usize) -> Result<()> {
        for mx in self.x.mixers(self.mtp.as_mut().map(|m| &mut m.layer)) {
            if let Mixer::Attn(a) = mx {
                a.unmap(slot as usize, from)?;
            }
        }
        Ok(())
    }

    fn slot_bytes(&self, len: usize) -> usize {
        self.x.layers.iter().chain(self.mtp.as_ref().map(|m| &m.layer)).map(|l| match &l.mixer {
            Mixer::Attn(a) => a.slot_bytes(len),
            Mixer::Delta(d) => d.slot_bytes(),
        }).sum::<usize>()
    }

    fn release(&mut self, slot: u32) -> Result<()> {
        for mx in self.x.mixers(self.mtp.as_mut().map(|m| &mut m.layer)) {
            match mx {
                Mixer::Attn(a) => a.unmap(slot as usize, 0)?,
                Mixer::Delta(d) => d.unmap(slot as usize)?,
            }
        }
        Ok(())
    }

    fn max_ctx(&self) -> usize {
        self.x.st.max_ctx
    }

    fn limit_ctx(&mut self, ctx: usize) {
        self.x.st.max_ctx = self.x.st.max_ctx.min(ctx.max(64));
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
        self.x.d.vocab
    }

    fn step(&mut self, toks: &[Tok], out: &[usize], marks: Option<&[Vec<u32>]>) -> Result<()> {
        self.map_step(toks)?;
        // the draft head takes in the tokens committed since the last step (the drafts are not)
        if let Some(m) = self.mtp.as_mut()
            && let Some(h) = m.head.as_mut()
        {
            let ids: Vec<u32> = match marks {
                Some(mk) => mk.concat(),
                None => toks.iter().map(|t| t.id).collect(),
            };
            if h.learn(&self.x.s, &self.x.mm, &ids)? {
                m.graphs.clear();
            }
        }
        self.x.st.upload(&self.x.s, toks, out, marks)?;
        let key = (self.x.st.shape(), Attn::graph_key(&self.x.k.attn, &self.x.st));
        self.graphs.run(&self.x.s, self.x.st.n, key, || self.launch_step())
    }

    fn logits(&self) -> Result<Vec<f32>> {
        self.x.st.logits(&self.x.s)
    }

    fn logits_row(&self, j: usize) -> Result<Vec<f32>> {
        self.x.st.logits_row(&self.x.s, j)
    }

    fn argmax(&self) -> Result<Vec<u32>> {
        self.x.st.argmax(&self.x.s)
    }

    fn sample(&mut self, params: &[[f32; 8]]) -> Result<Vec<(u32, bool)>> {
        self.x.st.sample(&self.x.s, params)
    }

    fn set_outputs(&mut self, slot: u32, words: &[u32]) -> Result<()> {
        self.x.st.set_outputs(&self.x.s, slot as usize, words)
    }

    fn sync(&self) -> Result<()> {
        Ok(self.x.s.synchronize()?)
    }

    fn draft_q(&self, slot: u32, t: usize) -> Result<(Vec<(u32, f32)>, u32)> {
        self.x.st.draft_q(&self.x.s, slot as usize, t)
    }

    fn has_draft(&self) -> bool {
        self.mtp.is_some()
    }

    /// 4: a step of 5 tokens is still two tiles of the 27B's decode attention, which reads the cache once for all of
    /// them
    fn draft_tokens(&self) -> usize {
        4
    }

    /// xn after a step holds the final norm of every token of the step (f16)
    fn draft(&mut self, toks: &[Tok], rows: &[usize], out: &[usize], k: usize, sampling: &[Option<DraftSampling>])
    -> Result<Vec<Vec<u32>>> {
        ensure!(sampling.is_empty() || sampling.len() == out.len(), "{} draft samplings for {} rows",
                sampling.len(), out.len());
        let argmax = vec![None; out.len()];
        self.draft_chains(toks, rows, out, k, if sampling.is_empty() { &argmax } else { sampling })
    }
}
