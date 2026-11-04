//! A PEFT LoRA adapter: served unmerged (src/cuda/qwen/lora.rs: y = W x + alpha / r * B A x, per request), and
//! `qwenny merge`, which writes the model with it merged into the bf16 base weights (W += alpha / r * B A).
use anyhow::{Context, Result, ensure};
use safetensors::{Dtype, SafeTensors};
use serde_json::Value as Json;
use std::{collections::HashMap, fs, path::Path};

pub struct Adapter {
    scale: f32,
    /// base_model_name_or_path of adapter_config.json
    base: String,
    /// lora_A [r][in] and lora_B [out][r] by base weight name ("model.language_model.layers.3.self_attn.q_proj.weight")
    pairs: HashMap<String, (Vec<f32>, Vec<f32>, usize)>,
}

fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn floats(st: &SafeTensors, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let t = st.tensor(name)?;
    let v = match t.dtype() {
        Dtype::F32 => f32s(t.data()),
        Dtype::BF16 => bf16_to_f32(t.data()),
        d => anyhow::bail!("{name}: dtype {d:?}"),
    };
    Ok((v, t.shape().to_vec()))
}

impl Adapter {
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg: Json = serde_json::from_str(&fs::read_to_string(dir.join("adapter_config.json")).context("adapter_config.json")?)?;
        let r = cfg["r"].as_f64().context("r")?;
        let alpha = cfg["lora_alpha"].as_f64().context("lora_alpha")?;
        ensure!(cfg["use_rslora"].as_bool() != Some(true) && cfg["use_dora"].as_bool() != Some(true), "rsLoRA/DoRA adapter");
        let empty = |k: &str| cfg[k].is_null() || cfg[k].as_object().is_some_and(|o| o.is_empty());
        ensure!(empty("rank_pattern") && empty("alpha_pattern"), "an adapter with a rank_pattern or alpha_pattern");
        let bytes = fs::read(dir.join("adapter_model.safetensors"))?;
        let st = SafeTensors::deserialize(&bytes)?;
        let mut pairs = HashMap::new();
        for name in st.names() {
            let Some(stem) = name.strip_suffix(".lora_A.weight") else { continue };
            let (a, sa) = floats(&st, name)?;
            let (b, sb) = floats(&st, &format!("{stem}.lora_B.weight"))?;
            ensure!(sa.len() == 2 && sb.len() == 2 && sa[0] == sb[1] && sa[0] as f64 == r, "{stem}: shapes {sa:?} {sb:?} (r {r})");
            let base = stem.strip_prefix("base_model.model.").unwrap_or(stem).to_string() + ".weight";
            pairs.insert(base, (a, b, sa[0]));
        }
        ensure!(!pairs.is_empty(), "no LoRA tensors in {}", dir.display());
        let base = cfg["base_model_name_or_path"].as_str().unwrap_or("?").to_string();
        Ok(Adapter { scale: (alpha / r) as f32, base, pairs })
    }

    /// That every matrix the adapter changes is in the model (else it is an adapter of another model, which would
    /// change nothing)
    pub fn check(&self, ck: &crate::checkpoint::Checkpoint) -> Result<()> {
        let missing: Vec<&String> = self.pairs.keys().filter(|n| !ck.has(n)).collect();
        ensure!(missing.is_empty(), "the adapter does not fit this model: {} of its {} matrices are not in it (e.g. {}); \
                 it was trained on {}", missing.len(), self.pairs.len(), missing[0], self.base);
        Ok(())
    }

    /// Matrices the adapter changes
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// alpha / r
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// The names of the base weights it changes
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.pairs.keys()
    }

    /// lora_A [r][in], lora_B [out][r] and r of base weight `name`, if the adapter changes it
    pub fn get(&self, name: &str) -> Option<(&[f32], &[f32], usize)> {
        self.pairs.get(name).map(|(a, b, r)| (&a[..], &b[..], *r))
    }

    /// Fold the adapter into weight `name` (bf16 [rows][cols], rows = len / cols) if it targets it.
    pub fn apply(&self, name: &str, w: &mut [u16]) -> bool {
        let Some((a, b, r)) = self.pairs.get(name) else { return false };
        let r = *r;
        let cols = a.len() / r;
        let scale = self.scale;
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
        let per = (w.len() / cols).div_ceil(threads) * cols;
        std::thread::scope(|s| {
            for (ci, chunk) in w.chunks_mut(per).enumerate() {
                s.spawn(move || {
                    let row0 = ci * per / cols;
                    let mut acc = vec![0f32; cols];
                    for (ri, row) in chunk.chunks_mut(cols).enumerate() {
                        acc.iter_mut().for_each(|x| *x = 0.0);
                        let br = &b[(row0 + ri) * r..(row0 + ri + 1) * r];
                        for (k, &bk) in br.iter().enumerate() {
                            for (x, &ak) in acc.iter_mut().zip(&a[k * cols..(k + 1) * cols]) {
                                *x += bk * ak;
                            }
                        }
                        for (wv, &d) in row.iter_mut().zip(&acc) {
                            let v = f32::from_bits((*wv as u32) << 16) + scale * d;
                            *wv = half::bf16::from_f32(v).to_bits();
                        }
                    }
                });
            }
        });
        true
    }
}

/// The residual-stream WRITER submodules of a Qwen-family transformer layer: every matrix whose output is added
/// back into the residual stream. Abliteration edits each as W' = W - alpha * r (rᵀW), a rank-1 LoRA (B = -r,
/// A = rᵀW). A Qwen layer writes the residual through exactly: the mixer out-projection (DeltaNet
/// `linear_attn.out_proj` OR attention `self_attn.o_proj`, whichever that layer has) and the MLP down-projection
/// (dense `mlp.down_proj`, or every expert `mlp.experts.{e}.down_proj` when the layer is MoE). THIS LIST IS THE ONLY
/// ARCHITECTURE-SPECIFIC KNOWLEDGE IN THE BAKE: a new Qwen variant with a new residual-writer submodule is a one-line
/// addition here (and, if it is an expert set, to the MoE probe in `writers_of`).
const WRITER_SUBMODULES: &[&str] = &["linear_attn.out_proj", "self_attn.o_proj", "mlp.down_proj"];

/// Enumerate the residual-writer matrices of the model STRUCTURALLY from the checkpoint (never a hardcoded layer
/// count or writer count): scan `model.language_model.layers.{i}` while it exists, and for each take whichever
/// `WRITER_SUBMODULES` are present; `mlp.down_proj` expands to `mlp.experts.{e}.down_proj` for a MoE layer. Returns
/// (base weight name without `.weight`, a short kind label for reporting).
fn writers_of(ck: &crate::checkpoint::Checkpoint) -> Vec<(String, &'static str)> {
    let present = |base: &str| ck.is_quant(base) || ck.has(&format!("{base}.weight"));
    let mut out = Vec::new();
    let mut i = 0usize;
    loop {
        let lp = format!("model.language_model.layers.{i}");
        if !ck.has(&format!("{lp}.input_layernorm.weight")) {
            break;
        }
        for sub in WRITER_SUBMODULES {
            let base = format!("{lp}.{sub}");
            if *sub == "mlp.down_proj" {
                if present(&base) {
                    out.push((base, "mlp.down_proj"));
                } else {
                    let mut e = 0usize; // MoE: per-expert down-projections
                    loop {
                        let eb = format!("{lp}.mlp.experts.{e}.down_proj");
                        if present(&eb) { out.push((eb, "mlp.experts.down_proj")); e += 1; } else { break; }
                    }
                }
            } else if present(&base) {
                out.push((base, sub));
            }
        }
        i += 1;
    }
    out
}

/// A = rᵀW [cols] for writer `base`, dequantized through the checkpoint's OWN dequant path (int4/int8/dense alike),
/// threaded over rows. Also returns (rows, cols) = W's output/input dims.
fn writer_a(ck: &crate::checkpoint::Checkpoint, base: &str, r: &[f32]) -> Result<(usize, usize, Vec<f32>)> {
    use crate::checkpoint::Matrix;
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let reduce = |rows: usize, cols: usize, row_f32: &(dyn Fn(usize, &mut [f32]) + Sync)| -> Vec<f32> {
        let per = rows.div_ceil(threads).max(1);
        let partials: Vec<Vec<f32>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..rows).step_by(per).map(|r0| {
                s.spawn(move || {
                    let (mut acc, mut buf) = (vec![0f32; cols], vec![0f32; cols]);
                    for row in r0..(r0 + per).min(rows) {
                        row_f32(row, &mut buf);
                        let rr = r[row];
                        for (a, &b) in acc.iter_mut().zip(&buf) { *a += rr * b; }
                    }
                    acc
                })
            }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut a = vec![0f32; cols];
        for p in partials { for (x, &y) in a.iter_mut().zip(&p) { *x += y; } }
        a
    };
    match ck.matrix(base)? {
        Matrix::Quant(q) => {
            let (rows, cols) = (q.rows, q.cols);
            let a = reduce(rows, cols, &|row, buf| q.row_into(row, buf));
            Ok((rows, cols, a))
        }
        Matrix::Dense { w, rows, cols } => {
            let a = reduce(rows, cols, &|row, buf| {
                for (c, o) in buf.iter_mut().enumerate() {
                    *o = f32::from_bits((w[row * cols + c] as u32) << 16);
                }
            });
            Ok((rows, cols, a))
        }
    }
}

/// `qwenny bake-direction`: build the abliteration adapter from a direction `r` [hidden]. For every residual-writer
/// matrix W in the model, emit a rank-1 LoRA A = rᵀW (from the served int4/int8 dequant), B = -r, so the served
/// unmerged edit is W x + alpha*B(A x) = (W - alpha * r rᵀ W) x. `alpha` is the strength knob (lora_alpha; 1.0 = full).
pub fn bake_direction(model: &Path, dir_file: &Path, out: &Path, alpha: f32) -> Result<()> {
    let bytes = fs::read(dir_file).with_context(|| format!("reading {}", dir_file.display()))?;
    let st = SafeTensors::deserialize(&bytes)?;
    let (r, rshape) = floats(&st, "r")?;
    ensure!(rshape.len() == 1, "direction 'r' must be a vector, got shape {rshape:?}");
    let ck = crate::checkpoint::Checkpoint::open(model)?;
    let writers = writers_of(&ck);
    ensure!(!writers.is_empty(), "no residual-writer matrices under model.language_model.layers.*");

    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    let mut leaves: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut out_vec: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    // The serving kernel serves LoRA ranks [8,16,32,64], so pad rank 1 -> 8 with zero rows/cols (exact same edit).
    const RANK: usize = 8;
    let mut b_bytes: Vec<u8> = Vec::with_capacity(r.len() * RANK * 4); // B = -r in col 0, zeros in cols 1..8, [rows][8]
    for &v in &r {
        b_bytes.extend_from_slice(&(-v).to_le_bytes());
        b_bytes.extend(std::iter::repeat(0u8).take((RANK - 1) * 4));
    }
    let (mut hidden, mut per_layer) = (0usize, std::collections::BTreeMap::<usize, usize>::new());
    for (base, kind) in &writers {
        let (rows, cols, a) = writer_a(&ck, base, &r).with_context(|| format!("writer {base}"))?;
        ensure!(rows == r.len(), "{base}: {rows} output rows != direction length {}", r.len());
        hidden = rows;
        *counts.entry(kind).or_default() += 1;
        if let Some(seg) = base.split(".layers.").nth(1).and_then(|s| s.split('.').next()).and_then(|s| s.parse().ok()) {
            *per_layer.entry(seg).or_default() += 1;
        }
        leaves.insert(base.rsplit('.').next().unwrap_or(base).to_string());
        let stem = format!("base_model.model.{base}");
        let mut a_bytes: Vec<u8> = Vec::with_capacity(RANK * cols * 4); // A = rᵀW in row 0, zeros in rows 1..8, [8][cols]
        a_bytes.extend(a.iter().flat_map(|x| x.to_le_bytes()));
        a_bytes.extend(std::iter::repeat(0u8).take((RANK - 1) * cols * 4));
        out_vec.push((format!("{stem}.lora_A.weight"), Dtype::F32, vec![RANK, cols], a_bytes));
        out_vec.push((format!("{stem}.lora_B.weight"), Dtype::F32, vec![rows, RANK], b_bytes.clone()));
    }

    fs::create_dir_all(out)?;
    let views: Vec<_> = out_vec.iter()
        .map(|(n, d, sh, b)| Ok((n.clone(), safetensors::tensor::TensorView::new(*d, sh.clone(), b)?)))
        .collect::<Result<_>>()?;
    let meta: HashMap<String, String> = [("format".to_string(), "pt".to_string())].into();
    safetensors::serialize_to_file(views, Some(meta), &out.join("adapter_model.safetensors"))?;
    let targets: Vec<String> = leaves.into_iter().collect();
    let cfg = serde_json::json!({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": RANK, "lora_alpha": alpha * RANK as f32, "lora_dropout": 0.0,
        "bias": "none", "fan_in_fan_out": false, "use_rslora": false, "use_dora": false,
        "rank_pattern": {}, "alpha_pattern": {}, "target_modules": targets,
        "base_model_name_or_path": model.to_string_lossy(),
    });
    fs::write(out.join("adapter_config.json"), serde_json::to_string_pretty(&cfg)?)?;

    let per_vals: Vec<usize> = per_layer.values().copied().collect();
    let (lo, hi) = (per_vals.iter().min().copied().unwrap_or(0), per_vals.iter().max().copied().unwrap_or(0));
    println!("baked {} residual-writer matrices across {} layers (writers/layer {}..{}), hidden {hidden}, rank {RANK}, alpha {alpha}:",
             writers.len(), per_layer.len(), lo, hi);
    for (k, c) in &counts {
        println!("  {k}: {c}");
    }
    println!("wrote {}/adapter_model.safetensors + adapter_config.json", out.display());
    Ok(())
}

/// Copy the model directory to `out` with the adapter merged into model.safetensors.
pub fn merge(base: &Path, adapter: &Path, out: &Path) -> Result<()> {
    let ad = Adapter::load(adapter)?;
    fs::create_dir_all(out)?;
    for e in fs::read_dir(base)? {
        let e = e?;
        let name = e.file_name();
        if e.file_type()?.is_file() && name != "model.safetensors" {
            fs::copy(e.path(), out.join(&name))?;
        }
    }
    let file = fs::File::open(base.join("model.safetensors"))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let mut data: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut merged = 0;
    for (name, t) in st.tensors() {
        let mut bytes = t.data().to_vec();
        if t.dtype() == Dtype::BF16 {
            let mut w: Vec<u16> = bytes.as_chunks().0.iter().map(|&c| u16::from_le_bytes(c)).collect();
            if ad.apply(&name, &mut w) {
                merged += 1;
                bytes = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            }
        }
        data.push((name, t.dtype(), t.shape().to_vec(), bytes));
    }
    ensure!(merged == ad.len(), "merged {merged} of the adapter's {} matrices", ad.len());
    let views: Vec<_> = data.iter().map(|(n, d, s, b)| (n.clone(), safetensors::tensor::TensorView::new(*d, s.clone(), b).unwrap())).collect();
    let meta: HashMap<String, String> = [("format".to_string(), "pt".to_string())].into();
    safetensors::serialize_to_file(views, Some(meta), &out.join("model.safetensors"))?;
    println!("merged {merged} matrices into {}", out.display());
    Ok(())
}
