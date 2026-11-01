//! A PEFT LoRA adapter: served unmerged (src/cuda/qwen/lora.rs: y = W x + alpha / r * B A x, per request), and
//! `qwenny merge`, which writes the model with it merged into the bf16 base weights (W += alpha / r * B A).
use crate::training::{bf16_to_f32, f32s};
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
