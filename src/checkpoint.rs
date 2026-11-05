//! A model's weights as stored: Hugging Face safetensors (one file or shards) in bf16, f16 or f32, or pre-quantized as
//! MLX's affine format (`quantization` in config.json: tensors `.weight`, `.scales`, `.biases`), or
//! compressed-tensors `pack-quantized` (`quantization_config`, what llm-compressor, AutoRound and GPTQModel write
//!   for vLLM, e.g. RedHatAI's w4a16 and HyperQwen's AutoRound int4: tensors `.weight_packed` (int32, the same packing
//!   along each row), `.weight_scale`, `.weight_shape`, `.weight_zero_point` when asymmetric; symmetric values are
//!   stored + 2^(bits-1)). Its layers listed in `ignore` stay dense. Symmetry is a matrix's own (a zero point stored
//!   or not): config groups may differ in it (Swift 1.5's HyperQwen INT4 heads: an asymmetric AWQ body, symmetric
//!   GPTQ output layer and MTP head).
//! They are read into one form, `Quant`. Tensors keep their Hugging Face names; a checkpoint that prefixes them
//! (`language_model.`) is read without it.
//!
//! `bf16` returns any tensor dense, dequantizing quantized matrices; `matrix` / `quant_matrix` return a quantized matrix
//! as stored, for backends that compute with quantized weights.
use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors, tensor::Metadata};
use serde_json::Value as Json;
use std::{collections::HashMap, fs, path::Path};

/// A quantized matrix [rows][cols], affine: w = q * scale + offset per group of a row
pub struct Quant {
    pub bits: usize,
    pub group: usize,
    pub rows: usize,
    pub cols: usize,
    /// [rows][cols * bits / 32]: value c of a row in word c * bits / 32, bits (c % (32 / bits)) * bits up
    pub q: Vec<u32>,
    /// [rows][cols / group]
    pub scales: Vec<f32>,
    pub offsets: Vec<f32>,
}

impl Quant {
    fn words(&self) -> usize {
        self.cols * self.bits / 32
    }

    /// Row `r` dequantized into `out` [cols]
    pub fn row_into(&self, r: usize, out: &mut [f32]) {
        let (per, mask, groups) = (32 / self.bits, (1u32 << self.bits) - 1, self.cols / self.group);
        let words = &self.q[r * self.words()..][..self.words()];
        for (c, o) in out.iter_mut().enumerate() {
            let v = words[c / per] >> ((c % per) * self.bits) & mask;
            let g = r * groups + c / self.group;
            *o = v as f32 * self.scales[g] + self.offsets[g];
        }
    }

    /// All rows dequantized, as bf16 bits (rounded to nearest even)
    pub fn to_bf16(&self) -> Vec<u16> {
        let mut out = vec![0u16; self.rows * self.cols];
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let per = self.rows.div_ceil(threads).max(1);
        std::thread::scope(|sc| {
            for (i, chunk) in out.chunks_mut(per * self.cols).enumerate() {
                sc.spawn(move || {
                    let mut row = vec![0f32; self.cols];
                    for (j, o) in chunk.chunks_mut(self.cols).enumerate() {
                        self.row_into(i * per + j, &mut row);
                        for (d, &x) in o.iter_mut().zip(&row) {
                            *d = half::bf16::from_f32(x).to_bits();
                        }
                    }
                });
            }
        });
        out
    }

    /// The integer zero point of each group (w = (q - z) * scale), if every offset is -z * scale for an integer z
    /// in the value range: symmetric and integer-zero-point formats (compressed-tensors, GPTQ, AWQ)
    pub fn zero_points(&self) -> Option<Vec<u8>> {
        let max = ((1u32 << self.bits) - 1) as f32;
        self.scales.iter().zip(&self.offsets).map(|(&s, &o)| {
            let z = if s == 0.0 { 0.0 } else { -o / s };
            (z.round() >= 0.0 && z.round() <= max && (z - z.round()).abs() < 1e-3).then(|| z.round() as u8)
        }).collect()
    }

    /// Matrices with the same columns and quantization, stacked by rows
    pub fn stack(parts: Vec<Quant>) -> Result<Quant> {
        let mut it = parts.into_iter();
        let mut out = it.next().context("nothing to stack")?;
        for p in it {
            ensure!((p.bits, p.group, p.cols) == (out.bits, out.group, out.cols), "stacking differently quantized matrices");
            out.rows += p.rows;
            out.q.extend(p.q);
            out.scales.extend(p.scales);
            out.offsets.extend(p.offsets);
        }
        Ok(out)
    }
}

/// A weight matrix as stored
pub enum Matrix {
    /// bf16 bits [rows][cols]
    Dense { w: Vec<u16>, rows: usize, cols: usize },
    Quant(Quant),
}

struct File {
    mmap: Mmap,
    header: usize,
    meta: Metadata,
}

pub struct Checkpoint {
    files: Vec<File>,
    /// tensor name (without the prefix) -> file
    index: HashMap<String, usize>,
    prefix: String,
    quant: Option<Scheme>,
}

/// How a checkpoint's quantized matrices are stored
#[derive(Clone, Copy, Debug)]
enum Scheme {
    Mlx { bits: usize, group: usize },
    /// compressed-tensors pack-quantized; `group` 0: one group per row (strategy "channel"); a matrix is
    /// symmetric unless it stores `weight_zero_point`
    Packed { bits: usize, group: usize },
}

/// compressed-tensors' quantization_config: its weight scheme (int, 4 or 8 bits, grouped or per channel, no activation
/// quantization). Several config groups may differ in their bits (HyperQwen's AutoRound: int4 for the linear
/// layers, int8 for the embedding, output layer and MTP head) and symmetry (Swift 1.5 INT4 heads); a matrix's own bits
/// follow from its packed and its real shape (`weight_shape`), its symmetry from a stored zero point; `bits` is the
/// first group's.
fn packed_scheme(qc: &Json) -> Result<Scheme> {
    ensure!(qc["format"] == "pack-quantized", "compressed-tensors format {}: pack-quantized only", qc["format"]);
    let groups = qc["config_groups"].as_object().context("config_groups")?;
    let mut schemes = Vec::new();
    for g in groups.values() {
        ensure!(g["input_activations"].is_null(), "activation quantization (w{}a{}) is not supported: weights only",
                g["weights"]["num_bits"], g["input_activations"]["num_bits"]);
        let w = &g["weights"];
        ensure!(w["type"] == "int", "compressed-tensors weights of type {}: int only", w["type"]);
        let bits = w["num_bits"].as_u64().context("num_bits")? as usize;
        ensure!([4, 8].contains(&bits), "{bits}-bit weights: 4 or 8");
        let group = match w["strategy"].as_str() {
            Some("group") => w["group_size"].as_u64().context("group_size")? as usize,
            Some("channel") => 0,
            other => bail!("compressed-tensors strategy {other:?}: group or channel"),
        };
        ensure!(w["actorder"].is_null() || w["actorder"] == "static" || w["actorder"] == "weight" || w["actorder"] == false,
                "actorder {}: activation-ordered groups (g_idx) are not supported", w["actorder"]);
        schemes.push((bits, group));
    }
    let &(bits, group) = schemes.first().context("no compressed-tensors config groups")?;
    ensure!(schemes.iter().all(|s| s.1 == group) && (schemes.len() == 1 || group > 0),
            "compressed-tensors config groups of different group sizes: {schemes:?}");
    Ok(Scheme::Packed { bits, group })
}

/// Bytes per value of a tensor stored as floats, and the value at the front of a slice as f32
fn float_format(dtype: Dtype) -> Option<(usize, fn(&[u8]) -> f32)> {
    match dtype {
        Dtype::BF16 => Some((2, |c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))),
        Dtype::F16 => Some((2, |c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())),
        Dtype::F32 => Some((4, |c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))),
        _ => None,
    }
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Result<Self> {
        let config: Json = serde_json::from_str(&fs::read_to_string(dir.join("config.json"))?)?;
        let names: Vec<String> = match fs::read_to_string(dir.join("model.safetensors.index.json")) {
            Ok(text) => {
                let idx: Json = serde_json::from_str(&text)?;
                let mut files: Vec<String> = idx["weight_map"].as_object().context("weight_map")?.values()
                    .filter_map(|v| v.as_str().map(String::from)).collect();
                files.sort();
                files.dedup();
                files
            }
            Err(_) => vec!["model.safetensors".into()],
        };
        let mut files = Vec::new();
        let mut index = HashMap::new();
        for (i, name) in names.iter().enumerate() {
            let path = dir.join(name);
            let file = fs::File::open(&path).with_context(|| format!("{}", path.display()))?;
            let mmap = unsafe { Mmap::map(&file)? };
            let (header, meta) = SafeTensors::read_metadata(&mmap).with_context(|| format!("{}", path.display()))?;
            for t in meta.tensors().keys() {
                index.insert(t.clone(), i);
            }
            files.push(File { mmap, header, meta });
        }
        // Qwen3.5 keeps its text model under model.language_model. (beside the vision tower and MTP): names as stored
        // (the embedding may be stored packed: HyperQwen's AutoRound keeps it int8)
        let embed = |n: &str| index.contains_key(&format!("{n}.weight")) || index.contains_key(&format!("{n}.weight_packed"));
        let prefix = if embed("model.language_model.embed_tokens") {
            String::new()
        } else {
            ["", "language_model."].into_iter()
                .find(|p| embed(&format!("{p}model.embed_tokens")))
                .context("no model.embed_tokens.weight: not a Qwen3.5-style checkpoint")?.to_string()
        };
        let index = index.into_iter().filter_map(|(k, v)| Some((k.strip_prefix(&prefix)?.to_string(), v))).collect();
        let (q, qc) = (&config["quantization"], &config["quantization_config"]);
        let quant = if q.is_object() {
            ensure!(q["mode"].as_str().unwrap_or("affine") == "affine", "quantization mode {}: only MLX's affine one", q["mode"]);
            let (bits, group) = (q["bits"].as_u64().context("quantization bits")? as usize,
                                 q["group_size"].as_u64().context("quantization group_size")? as usize);
            ensure!([4, 8].contains(&bits), "{bits}-bit weights: 4 or 8");
            Some(Scheme::Mlx { bits, group })
        } else if qc.is_object() {
            ensure!(qc["quant_method"] == "compressed-tensors",
                    "quantization_config {}: compressed-tensors (llm-compressor, AutoRound, GPTQModel for vLLM) and MLX's \
                     format are supported", qc["quant_method"]);
            Some(packed_scheme(qc)?)
        } else {
            None
        };
        Ok(Checkpoint { files, index, prefix, quant })
    }

    fn raw(&self, name: &str) -> Option<(Dtype, &[usize], &[u8])> {
        let f = &self.files[*self.index.get(name)?];
        let info = f.meta.info(&format!("{}{name}", self.prefix))?;
        let (a, b) = info.data_offsets;
        Some((info.dtype, &info.shape, &f.mmap[8 + f.header + a..8 + f.header + b]))
    }

    /// Whether the checkpoint stores some matrices quantized (a pre-quantized checkpoint)
    pub fn prequantized(&self) -> bool {
        self.quant.is_some()
    }

    /// Whether the checkpoint has this tensor; `{base}.weight` also when it is stored quantized as `weight_packed`
    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name) || name.strip_suffix(".weight").is_some_and(|b| self.packed(b))
    }

    /// Whether `{base}` is stored compressed-tensors packed
    fn packed(&self, base: &str) -> bool {
        self.index.contains_key(&format!("{base}.weight_packed"))
    }

    /// Whether the packed `{base}` stores zero points (asymmetric)
    fn has_zero_point(&self, base: &str) -> bool {
        self.index.contains_key(&format!("{base}.weight_zero_point"))
    }

    /// Whether `{base}` is stored quantized (either format)
    pub fn is_quant(&self, base: &str) -> bool {
        self.packed(base) || self.index.contains_key(&format!("{base}.scales"))
    }

    /// A tensor stored as floats (bf16, f16 or f32), as f32, and its shape
    pub fn floats(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (dtype, shape, data) = self.raw(name).with_context(|| format!("tensor {name}"))?;
        let (width, value) = float_format(dtype).with_context(|| format!("{name} is {dtype:?}, expected floats"))?;
        Ok((data.chunks_exact(width).map(value).collect(), shape.to_vec()))
    }

    /// Rows `ids` of the matrix `{base}.weight` as f32 [ids][cols], read from the file without the rest (an embedding
    /// lookup): stored as floats or compressed-tensors packed (symmetric)
    pub fn rows(&self, base: &str, ids: &[u32]) -> Result<Vec<f32>> {
        let format = |dtype: Dtype| float_format(dtype).with_context(|| format!("{base}: {dtype:?} values"));
        if !self.packed(base) {
            ensure!(!self.is_quant(base), "{base}: row reads of MLX-quantized matrices are not supported");
            let (dtype, shape, data) = self.raw(&format!("{base}.weight")).with_context(|| format!("tensor {base}.weight"))?;
            let (cols, (width, value)) = (shape[1], format(dtype)?);
            let mut out = Vec::with_capacity(ids.len() * cols);
            for &r in ids {
                ensure!((r as usize) < shape[0], "{base}: row {r} of {}", shape[0]);
                out.extend(data[r as usize * cols * width..][..cols * width].chunks_exact(width).map(value));
            }
            return Ok(out);
        }
        let Some(Scheme::Packed { group, .. }) = self.quant.filter(|_| !self.has_zero_point(base)) else {
            bail!("{base}: row reads of asymmetric or MLX-quantized matrices are not supported");
        };
        let (_, qshape, q) = self.raw(&format!("{base}.weight_packed")).context("weight_packed")?;
        let (sdtype, _, sc) = self.raw(&format!("{base}.weight_scale")).context("weight_scale")?;
        let (rows, words) = (qshape[0], qshape[1]);
        let cols = self.packed_cols(base, rows)?.with_context(|| format!("{base}: no weight_shape"))?;
        let bits = words * 32 / cols;
        let (swidth, scale) = format(sdtype)?;
        let (per, mask, group) = (32 / bits, (1u32 << bits) - 1, if group == 0 { cols } else { group });
        let zero = (1u32 << (bits - 1)) as f32;
        let mut out = Vec::with_capacity(ids.len() * cols);
        for &r in ids {
            let r = r as usize;
            ensure!(r < rows, "{base}: row {r} of {rows}");
            for c in 0..cols {
                let w = u32::from_le_bytes(q[4 * (r * words + c / per)..][..4].try_into().unwrap());
                let v = (w >> ((c % per) * bits) & mask) as f32;
                out.push((v - zero) * scale(&sc[(r * (cols / group) + c / group) * swidth..]));
            }
        }
        Ok(out)
    }

    /// Any tensor dense as bf16 bits, and its shape [rows][cols] for matrices: quantized matrices are dequantized
    pub fn bf16(&self, name: &str) -> Result<(Vec<u16>, Vec<usize>)> {
        if let Some(base) = name.strip_suffix(".weight").filter(|b| self.is_quant(b)) {
            let q = self.quant_matrix(base)?;
            return Ok((q.to_bf16(), vec![q.rows, q.cols]));
        }
        let (dtype, shape, data) = self.raw(name).with_context(|| format!("tensor {name}"))?;
        let v = match dtype {
            Dtype::BF16 => data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
            _ => self.floats(name)?.0.into_iter().map(|x| half::bf16::from_f32(x).to_bits()).collect(),
        };
        Ok((v, shape.to_vec()))
    }

    /// The matrix `{base}.weight` as stored
    pub fn matrix(&self, base: &str) -> Result<Matrix> {
        if self.is_quant(base) {
            return Ok(Matrix::Quant(self.quant_matrix(base)?));
        }
        let (w, shape) = self.bf16(&format!("{base}.weight"))?;
        ensure!(shape.len() == 2, "{base}.weight has shape {shape:?}, expected a matrix");
        Ok(Matrix::Dense { w, rows: shape[0], cols: shape[1] })
    }

    /// The quantized matrix `{base}.weight` as stored (`is_quant`)
    pub fn quant_matrix(&self, base: &str) -> Result<Quant> {
        match self.quant.with_context(|| format!("{base} is stored quantized but config.json has no quantization"))? {
            Scheme::Mlx { bits, group } => self.mlx_matrix(base, bits, group),
            Scheme::Packed { bits, group } => self.packed_matrix(base, bits, group),
        }
    }

    fn words(&self, name: &str) -> Result<(Vec<u32>, Vec<usize>)> {
        let (dtype, shape, data) = self.raw(name).with_context(|| format!("tensor {name}"))?;
        ensure!(matches!(dtype, Dtype::U32 | Dtype::I32) && shape.len() == 2, "{name} is {dtype:?} {shape:?}, expected packed 32-bit words");
        Ok((data.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect(), shape.to_vec()))
    }

    fn mlx_matrix(&self, base: &str, bits: usize, group: usize) -> Result<Quant> {
        let (q, shape) = self.words(&format!("{base}.weight"))?;
        let (rows, cols) = (shape[0], shape[1] * 32 / bits);
        let (scales, s_shape) = self.floats(&format!("{base}.scales"))?;
        let (offsets, _) = self.floats(&format!("{base}.biases"))?;
        ensure!(s_shape == [rows, cols / group] && offsets.len() == scales.len(),
                "{base}: scales {s_shape:?} for {rows} x {cols} in groups of {group}");
        Ok(Quant { bits, group, rows, cols, q, scales, offsets })
    }

    /// The columns `{base}.weight_shape` gives a packed matrix of `rows` rows, if it is stored
    fn packed_cols(&self, base: &str, rows: usize) -> Result<Option<usize>> {
        let Some((Dtype::I64, _, d)) = self.raw(&format!("{base}.weight_shape")) else { return Ok(None) };
        let s: Vec<i64> = d.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
        ensure!(s.len() == 2 && s[0] as usize == rows, "{base}.weight_shape {s:?} for {rows} packed rows");
        Ok(Some(s[1] as usize))
    }

    /// compressed-tensors: w = (q - zero) * scale, zero 2^(bits-1) when symmetric (no zero point stored; q stored
    /// unsigned), else the stored zero point
    fn packed_matrix(&self, base: &str, bits: usize, group: usize) -> Result<Quant> {
        let (q, shape) = self.words(&format!("{base}.weight_packed"))?;
        let rows = shape[0];
        let cols = self.packed_cols(base, rows)?.unwrap_or(shape[1] * 32 / bits);
        // the matrix's own bits (config groups may differ in them)
        let bits = if (shape[1] * 32).is_multiple_of(cols) { shape[1] * 32 / cols } else { bits };
        ensure!([4, 8].contains(&bits), "{base}: {} words per row for {cols} columns", shape[1]);
        ensure!(cols == shape[1] * 32 / bits, "{base}: {cols} columns in {} words of {bits}-bit values (padded rows are not supported)", shape[1]);
        let group = if group == 0 { cols } else { group };
        ensure!(cols % group == 0, "{base}: {cols} columns in groups of {group}");
        let groups = cols / group;
        let (scales, s_shape) = self.floats(&format!("{base}.weight_scale"))?;
        ensure!(s_shape.iter().product::<usize>() == rows * groups, "{base}: weight_scale {s_shape:?} for {rows} rows of {groups} groups");
        let zeros: Vec<f32> = if !self.has_zero_point(base) {
            vec![(1 << (bits - 1)) as f32; rows * groups]
        } else {
            // packed along the rows (compressed-tensors' packed_dim 0), or along the groups of a row
            let (z, z_shape) = self.words(&format!("{base}.weight_zero_point"))?;
            let (per, mask) = (32 / bits, (1u32 << bits) - 1);
            let along_rows = z_shape == [rows.div_ceil(per), groups];
            ensure!(along_rows || z_shape == [rows, groups.div_ceil(per)],
                    "{base}: weight_zero_point {z_shape:?} for {rows} x {groups} groups");
            (0..rows * groups).map(|i| {
                let (r, g) = (i / groups, i % groups);
                let v = if along_rows { z[(r / per) * groups + g] >> ((r % per) * bits) } else { z[r * z_shape[1] + g / per] >> ((g % per) * bits) };
                (v & mask) as f32
            }).collect()
        };
        let offsets = scales.iter().zip(&zeros).map(|(s, z)| -s * z).collect();
        Ok(Quant { bits, group, rows, cols, q, scales, offsets })
    }
}
