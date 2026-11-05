//! A PEFT LoRA adapter: served unmerged (src/cuda/qwen/lora.rs: y = W x + alpha / r * B A x, per request), and
//! `qwenny merge`, which writes the model with it merged into the bf16 base weights (W += alpha / r * B A).
use anyhow::{Context, Result, ensure};
use safetensors::{Dtype, SafeTensors};
use serde_json::{Value as Json, json};
use std::{collections::HashMap, fs, path::{Path, PathBuf}};

/// A matrix of a composed adapter: name, output rows, input cols, the adapters (index + kept slots) that touch it
type PlanEntry<'a> = (&'a String, usize, usize, Vec<(usize, Vec<usize>)>);

/// LoRA ranks the serving kernel has (src/cuda/qwen/lora.rs RANKS); a composed adapter pads to one of these.
const RANKS: [usize; 4] = [8, 16, 32, 64];

pub struct Adapter {
    scale: f32,
    /// base_model_name_or_path of adapter_config.json
    base: String,
    /// lora_A [r][in] and lora_B [out][r] by base weight name ("model.language_model.layers.3.self_attn.q_proj.weight")
    pairs: HashMap<String, (Vec<f32>, Vec<f32>, usize)>,
}

/// little-endian f32 bytes as floats
pub fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks().0.iter().map(|&c| f32::from_le_bytes(c)).collect()
}

/// An f32 or bf16 tensor as floats, with its shape
pub fn floats(st: &SafeTensors, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let t = st.tensor(name)?;
    let v = match t.dtype() {
        Dtype::F32 => f32s(t.data()),
        Dtype::BF16 => t.data().as_chunks().0.iter().map(|&c| f32::from_bits((u16::from_le_bytes(c) as u32) << 16)).collect(),
        d => anyhow::bail!("{name}: dtype {d:?}"),
    };
    Ok((v, t.shape().to_vec()))
}

/// What an adapter_config.json says that serving, merging and training take from it
pub struct Config {
    pub r: usize,
    pub alpha: f64,
    /// base_model_name_or_path
    pub base: String,
}

/// adapter_config.json of a PEFT adapter directory: an error for anything but a plain additive LoRA
pub fn read_config(dir: &Path) -> Result<Config> {
    let path = dir.join("adapter_config.json");
    let cfg: Json = serde_json::from_str(&fs::read_to_string(&path).with_context(|| path.display().to_string())?)?;
    ensure!(cfg["peft_type"].as_str().is_none_or(|t| t == "LORA"), "{} is not a LoRA", dir.display());
    ensure!(cfg["use_rslora"].as_bool() != Some(true) && cfg["use_dora"].as_bool() != Some(true), "rsLoRA/DoRA adapter");
    let empty = |k: &str| cfg[k].is_null() || cfg[k].as_object().is_some_and(|o| o.is_empty());
    ensure!(empty("rank_pattern") && empty("alpha_pattern"), "an adapter with a rank_pattern or alpha_pattern");
    // modules_to_save replaces whole weights, not an additive BA delta: it cannot be served unmerged or composed.
    let empty_list = |k: &str| empty(k) || cfg[k].as_array().is_some_and(|a| a.is_empty());
    ensure!(empty_list("modules_to_save"), "an adapter with modules_to_save (a full-weight replacement, not a \
            stackable additive delta)");
    ensure!(cfg["lora_bias"].as_bool() != Some(true), "a LoRA adapter with lora_bias (bias terms are not served)");
    Ok(Config {
        r: cfg["r"].as_u64().context("r")? as usize,
        alpha: cfg["lora_alpha"].as_f64().context("lora_alpha")?,
        base: cfg["base_model_name_or_path"].as_str().unwrap_or("?").to_string(),
    })
}

/// adapter_config.json in PEFT's format into `out`; its target modules are those of the lora_A tensors in `names`
pub fn write_config<'a>(out: &Path, base: &str, rank: usize, alpha: f32, dropout: f32, names: impl Iterator<Item = &'a str>)
    -> Result<()> {
    // "...layers.3.self_attn.q_proj.lora_A.weight" -> "q_proj"
    let mut targets: Vec<&str> = names.filter_map(|n| n.strip_suffix(".lora_A.weight")?.rsplit('.').next()).collect();
    targets.sort();
    targets.dedup();
    let cfg = json!({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base,
        "r": rank, "lora_alpha": alpha, "lora_dropout": dropout, "target_modules": targets,
        "bias": "none", "fan_in_fan_out": false, "inference_mode": true, "init_lora_weights": true, "use_dora": false,
        "use_rslora": false, "modules_to_save": null, "rank_pattern": {}, "alpha_pattern": {},
    });
    fs::write(out.join("adapter_config.json"), serde_json::to_string_pretty(&cfg)?)?;
    Ok(())
}

/// `tensors` (name, dtype, shape, little-endian bytes) as a safetensors file, with PyTorch's metadata
pub fn save_tensors<'a>(path: &Path, tensors: impl IntoIterator<Item = (String, Dtype, Vec<usize>, &'a [u8])>) -> Result<()> {
    let views = tensors.into_iter().map(|(n, d, s, b)| Ok((n, safetensors::tensor::TensorView::new(d, s, b)?)))
        .collect::<Result<Vec<_>>>()?;
    let meta: HashMap<String, String> = [("format".to_string(), "pt".to_string())].into();
    safetensors::serialize_to_file(views, Some(meta), path)?;
    Ok(())
}

impl Adapter {
    pub fn load(dir: &Path) -> Result<Self> {
        let Config { r, alpha, base } = read_config(dir)?;
        let bytes = fs::read(dir.join("adapter_model.safetensors"))?;
        let st = SafeTensors::deserialize(&bytes)?;
        ensure!(!st.names().iter().any(|n| n.contains(".lora_") && n.ends_with(".bias")),
                "a LoRA adapter with bias tensors (bias terms are not served)");
        let mut pairs = HashMap::new();
        for name in st.names() {
            let Some(stem) = name.strip_suffix(".lora_A.weight") else { continue };
            let (a, sa) = floats(&st, name)?;
            let (b, sb) = floats(&st, &format!("{stem}.lora_B.weight"))?;
            ensure!(sa.len() == 2 && sb.len() == 2 && sa[0] == sb[1] && sa[0] == r, "{stem}: shapes {sa:?} {sb:?} (r {r})");
            let base = stem.strip_prefix("base_model.model.").unwrap_or(stem).to_string() + ".weight";
            pairs.insert(base, (a, b, sa[0]));
        }
        ensure!(!pairs.is_empty(), "no LoRA tensors in {}", dir.display());
        Ok(Adapter { scale: (alpha / r as f64) as f32, base, pairs })
    }

    /// Load and compose adapter directories into one served adapter. A single directory is loaded exactly as `load`
    /// (byte-identical serving: it does NOT go through `compose`); two or more are composed with `compose`.
    pub fn compose_dirs(dirs: &[PathBuf]) -> Result<Adapter> {
        ensure!(!dirs.is_empty(), "compose: no adapter directories");
        if dirs.len() == 1 {
            return Adapter::load(&dirs[0]);
        }
        let parts = dirs.iter().map(|d| Adapter::load(d)).collect::<Result<Vec<_>>>()?;
        Adapter::compose(&parts)
    }

    /// Compose loaded adapters into one additive low-rank update per target matrix. LoRA deltas on the same base add:
    /// W + Σ (alpha_i/r_i) Bᵢ Aᵢ. For each matrix we concatenate the adapters' factors — A rows stacked
    /// vertically, B columns stacked horizontally — and fold each adapter's own scale alpha_i/r_i INTO its B block
    /// (in f32) before concatenation; the composed adapter stores r = alpha = R_pad so `scale()` is exactly 1 and the
    /// serving kernel's apply-scale is 1 (every stacked adapter keeps its own scale). Each adapter's own zero-padding
    /// is stripped to its TRUE rank first (a rank slot whose A row or B column is all-zero contributes nothing), the
    /// true ranks are summed, and the sum is padded ONCE to the next served rank {8,16,32,64}; a sum above 64 is an
    /// error. Every target matrix is padded to the same R_pad with zero A rows AND zero B columns (both sides, so a
    /// matrix a given adapter does not touch contributes an exact zero block, never a 0·inf). Handling is per concrete
    /// matrix name, so the kernel's grouped stacks (q/k/v, gate/up) and single matrices (o_proj/out_proj/down_proj)
    /// all come out at the same uniform R_pad, which is exactly what the grouped loader requires.
    ///
    /// This is exact for LoRAs trained on the SAME base — including a future capability LoRA trained ON the
    /// abliterated base, which is the practice we follow so additive composition equals the model actually trained.
    /// A capability LoRA trained on the UN-abliterated base would instead want a load-time (I − r rᵀ) projection of
    /// its B onto the complement of the refusal direction r to stay orthogonal to the abliteration; that projection
    /// (and SVD re-compression when Σr > 64) is deliberately NOT built here — this comment is the guard note.
    fn compose(parts: &[Adapter]) -> Result<Adapter> {
        ensure!(parts.len() >= 2, "compose expects at least two adapters");
        let base = parts[0].base.clone();
        for p in parts {
            ensure!(p.base == base, "composing adapters of different base models: {base:?} and {:?}", p.base);
        }
        // the rank slots an adapter actually contributes for a matrix: outer product Bᵢ[:,k]·Aᵢ[k,:] is nonzero
        let kept = |a: &[f32], b: &[f32], r: usize| -> Vec<usize> {
            let (cols, rows) = (a.len() / r, b.len() / r);
            (0..r).filter(|&k| {
                a[k * cols..(k + 1) * cols].iter().any(|&x| x != 0.0) && (0..rows).any(|o| b[o * r + k] != 0.0)
            }).collect()
        };
        // every target matrix, deterministic order
        let mut names: Vec<&String> = parts.iter().flat_map(|p| p.pairs.keys()).collect();
        names.sort();
        names.dedup();
        // per matrix: output rows, input cols, the adapters (index + kept slots) that touch it; and the largest sum
        let mut plan: Vec<PlanEntry<'_>> = Vec::new();
        let mut max_sum = 0usize;
        for name in names {
            let (mut rows, mut cols, mut blocks, mut sum) = (None, None, Vec::new(), 0usize);
            for (pi, p) in parts.iter().enumerate() {
                let Some((a, b, r)) = p.pairs.get(name).map(|(a, b, r)| (a.as_slice(), b.as_slice(), *r)) else { continue };
                let (c, ro) = (a.len() / r, b.len() / r);
                if let Some(pc) = cols { ensure!(pc == c, "{name}: input dim {c} vs {pc} across stacked adapters"); }
                if let Some(pr) = rows { ensure!(pr == ro, "{name}: output dim {ro} vs {pr} across stacked adapters"); }
                cols = Some(c);
                rows = Some(ro);
                let ks = kept(a, b, r);
                sum += ks.len();
                if !ks.is_empty() { blocks.push((pi, ks)); }
            }
            if blocks.is_empty() { continue; }
            max_sum = max_sum.max(sum);
            plan.push((name, rows.unwrap(), cols.unwrap(), blocks));
        }
        ensure!(max_sum > 0, "the composed adapters change nothing (all factors zero)");
        let r_pad = *RANKS.iter().find(|&&r| r >= max_sum)
            .with_context(|| format!("combined LoRA rank {max_sum} exceeds the largest served rank {}", RANKS[RANKS.len() - 1]))?;
        let mut pairs = HashMap::new();
        for (name, rows, cols, blocks) in plan {
            let mut a = vec![0f32; r_pad * cols]; // [R_pad][in], real rows packed first, rest zero
            let mut b = vec![0f32; rows * r_pad]; // [out][R_pad], real cols packed first, rest zero
            let mut off = 0usize;
            for (pi, ks) in blocks {
                let (pa, pb, pr) = parts[pi].pairs.get(name).map(|(a, b, r)| (a.as_slice(), b.as_slice(), *r)).unwrap();
                let scale = parts[pi].scale;
                for (j, &k) in ks.iter().enumerate() {
                    let dst = off + j;
                    a[dst * cols..(dst + 1) * cols].copy_from_slice(&pa[k * cols..(k + 1) * cols]);
                    for o in 0..rows {
                        b[o * r_pad + dst] = scale * pb[o * pr + k]; // fold this adapter's scale into B, in f32
                    }
                }
                off += ks.len();
            }
            pairs.insert(name.clone(), (a, b, r_pad));
        }
        // r = alpha = R_pad -> scale() == 1: the per-adapter scales are already folded into B
        Ok(Adapter { scale: 1.0, base, pairs })
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
    pub fn apply(&self, name: &str, w: &mut [u16]) -> Result<bool> {
        let Some((a, b, r)) = self.pairs.get(name) else { return Ok(false) };
        let r = *r;
        let cols = a.len() / r;
        ensure!(a.len() % r == 0 && b.len() % r == 0 && (b.len() / r) * cols == w.len(),
                "{name}: the adapter's shape does not fit the model: is it for another one?");
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
        Ok(true)
    }
}

/// The residual-stream WRITER submodules of a Qwen layer: every matrix whose output is added back into the residual
/// stream. Abliteration edits each as W' = W - alpha * r (rᵀW), a rank-1 LoRA (B = -r, A = rᵀW). A layer writes the
/// residual through exactly: the mixer out-projection (DeltaNet `linear_attn.out_proj` OR attention `self_attn.o_proj`,
/// whichever that layer has) and the MLP down-projection. THIS LIST IS THE ONLY ARCHITECTURE-SPECIFIC KNOWLEDGE IN THE
/// BAKE: a new residual-writer submodule is a one-line addition here.
const WRITER_SUBMODULES: &[&str] = &["linear_attn.out_proj", "self_attn.o_proj", "mlp.down_proj"];

/// The residual-writer matrices of the model, found structurally from the checkpoint (never a hardcoded layer count):
/// scan `model.language_model.layers.{i}` while it exists and take whichever `WRITER_SUBMODULES` are present. The base
/// weight names without `.weight`.
fn writers_of(ck: &crate::checkpoint::Checkpoint) -> Vec<String> {
    let mut out = Vec::new();
    for i in 0.. {
        let layer = format!("model.language_model.layers.{i}");
        if !ck.has(&format!("{layer}.input_layernorm.weight")) {
            break;
        }
        for sub in WRITER_SUBMODULES {
            let base = format!("{layer}.{sub}");
            if ck.is_quant(&base) || ck.has(&format!("{base}.weight")) {
                out.push(base);
            }
        }
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
                    let end = (r0 + per).min(rows);
                    for (row, &rr) in (r0..end).zip(&r[r0..end]) {
                        row_f32(row, &mut buf);
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
    bake(model, &r, out, alpha)
}

/// The abliteration adapter of the unit direction `r` (bake_direction)
pub fn bake(model: &Path, r: &[f32], out: &Path, alpha: f32) -> Result<()> {
    let ck = crate::checkpoint::Checkpoint::open(model)?;
    let writers = writers_of(&ck);
    ensure!(!writers.is_empty(), "no residual-writer matrices under model.language_model.layers.*");

    // The serving kernel serves LoRA ranks [8,16,32,64], so pad rank 1 -> 8 with zero rows/cols (exact same edit).
    const RANK: usize = 8;
    let mut b_bytes: Vec<u8> = Vec::with_capacity(r.len() * RANK * 4); // B = -r in col 0, zeros in cols 1..8, [rows][8]
    for &v in r {
        b_bytes.extend_from_slice(&(-v).to_le_bytes());
        b_bytes.extend(std::iter::repeat_n(0u8, (RANK - 1) * 4));
    }
    let mut tensors: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    for base in &writers {
        let (rows, cols, a) = writer_a(&ck, base, r).with_context(|| format!("writer {base}"))?;
        ensure!(rows == r.len(), "{base}: {rows} output rows != direction length {}", r.len());
        let stem = format!("base_model.model.{base}");
        let mut a_bytes: Vec<u8> = Vec::with_capacity(RANK * cols * 4); // A = rᵀW in row 0, zeros in rows 1..8, [8][cols]
        a_bytes.extend(a.iter().flat_map(|x| x.to_le_bytes()));
        a_bytes.extend(std::iter::repeat_n(0u8, (RANK - 1) * cols * 4));
        tensors.push((format!("{stem}.lora_A.weight"), Dtype::F32, vec![RANK, cols], a_bytes));
        tensors.push((format!("{stem}.lora_B.weight"), Dtype::F32, vec![rows, RANK], b_bytes.clone()));
    }

    fs::create_dir_all(out)?;
    save_tensors(&out.join("adapter_model.safetensors"), tensors.iter().map(|(n, d, s, b)| (n.clone(), *d, s.clone(), &b[..])))?;
    write_config(out, &crate::hub::name(model), RANK, alpha * RANK as f32, 0.0, tensors.iter().map(|t| t.0.as_str()))?;
    println!("baked {} residual-writer matrices (hidden {}, rank {RANK}, alpha {alpha}) into {}", writers.len(), r.len(),
             out.display());
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
            if ad.apply(&name, &mut w)? {
                merged += 1;
                bytes = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            }
        }
        data.push((name, t.dtype(), t.shape().to_vec(), bytes));
    }
    ensure!(merged == ad.len(), "merged {merged} of the adapter's {} matrices", ad.len());
    save_tensors(&out.join("model.safetensors"), data.iter().map(|(n, d, s, b)| (n.clone(), *d, s.clone(), &b[..])))?;
    println!("merged {merged} matrices into {}", out.display());
    Ok(())
}
