//! LoRA training: data (chat template, loss masks, packing), the training loop (schedule, accumulation, evaluation),
//! adapters in PEFT's format, and the checks against PEFT and Unsloth's masks. The forward and backward passes run in
//! the CUDA trainer (`crate::cuda::qwen::train`).
use crate::cuda::learn::mem_gib;
use crate::cuda::qwen::train::Trainer;
use crate::chat::Chat;
use anyhow::{Context, Result, bail, ensure};
use safetensors::{Dtype, SafeTensors};
use serde_json::{Value as Json, json};
use std::{collections::HashMap, fs, path::Path, time::Instant};

#[derive(Clone)]
pub struct Opts {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    /// Most tokens in one forward/backward pass; a step of more is split and the gradients accumulate.
    pub micro_tokens: usize,
    /// f32 activations and weights (the check build of the kernels) instead of bf16
    pub f32: bool,
    /// f16 activations and weights, the big products accumulated in f16 (twice the tensor-core rate on GeForce)
    pub f16: bool,
    /// Only the model's first layers (0: all), for checks against a reference of a cut model
    pub layers: usize,
}

#[derive(Default)]
pub struct Micro {
    pub ids: Vec<i32>,
    pub pos: Vec<i32>,
    pub dstart: Vec<i32>,
    pub dend: Vec<i32>,
    pub dlen: Vec<i32>,
    /// (first token, length) of each document
    pub docs: Vec<(usize, usize)>,
    /// positions whose next token is in the loss, and that token
    pub rows: Vec<i32>,
    pub targets: Vec<i32>,
}

impl Micro {
    pub fn push(&mut self, ids: &[u32], mask: &[u8]) {
        let s = self.ids.len();
        let l = ids.len();
        for (i, &id) in ids.iter().enumerate() {
            self.ids.push(id as i32);
            self.pos.push(i as i32);
            self.dstart.push(s as i32);
            self.dend.push((s + l) as i32);
            self.dlen.push(l as i32);
            if i + 1 < l && mask[i + 1] != 0 {
                self.rows.push((s + i) as i32);
                self.targets.push(ids[i + 1] as i32);
            }
        }
        self.docs.push((s, l));
    }

    pub fn tokens(&self) -> usize {
        self.ids.len()
    }
}

pub struct Example {
    pub ids: Vec<u32>,
    pub mask: Vec<u8>,
}

impl Example {
    fn loss_tokens(&self) -> usize {
        self.mask[1..].iter().filter(|&&m| m != 0).count()
    }
}

/// The non-empty lines of a JSONL file with their line numbers.
fn read_rows(path: &Path) -> Result<Vec<(usize, String)>> {
    let text = fs::read_to_string(path).with_context(|| format!("{}", path.display()))?;
    Ok(text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()).map(|(i, l)| (i + 1, l.to_string())).collect())
}

/// Rows `{"messages": [...]}` rendered without thinking (a row's `chat_template_kwargs` override that, e.g. a
/// thinking row with `{"enable_thinking": true, "reasoning_effort": "low"}`), at most `max_len` tokens; `name` names the
/// file in messages. A row that is not JSON, has no messages or no assistant reply is an error.
fn encode_rows(chat: &Chat, rows: &[(usize, String)], max_len: usize, name: &str) -> Result<Vec<Example>> {
    let kwargs = json!({"enable_thinking": false}).as_object().unwrap().clone();
    let (mut out, mut no_reply, mut cut, mut lost) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (line, text) in rows {
        let row: Json = serde_json::from_str(text).with_context(|| format!("{name}:{line}: not a JSON object"))?;
        let Some(messages) = row["messages"].as_array() else {
            bail!("{name}:{line}: no \"messages\" list; a row is {{\"messages\": [{{\"role\": \"user\", \"content\": ...}}, \
                   {{\"role\": \"assistant\", \"content\": ...}}]}}");
        };
        let replies = |m: &&Json| m["role"] == "assistant" && (m["content"].as_str().is_some_and(|c| !c.trim().is_empty())
            || m["content"].as_array().is_some_and(|c| !c.is_empty()) || m["tool_calls"].as_array().is_some_and(|c| !c.is_empty()));
        if !messages.iter().any(|m| replies(&m)) {
            no_reply.push(*line);
            continue;
        }
        // a row's own chat_template_kwargs (e.g. {"enable_thinking": true, "reasoning_effort": "low"}: the system
        // prompt then carries the effort's instruction, as the server renders such a request) over the default
        let mut kw = kwargs.clone();
        match &row["chat_template_kwargs"] {
            Json::Null => {}
            Json::Object(o) => kw.extend(o.iter().map(|(k, v)| (k.clone(), v.clone()))),
            _ => bail!("{name}:{line}: chat_template_kwargs is not an object"),
        }
        let (mut ids, mut mask) = chat.encode_train(&row["messages"], &kw).with_context(|| format!("{name}:{line}"))?;
        if ids.len() > max_len {
            cut.push(*line);
            ids.truncate(max_len);
            mask.truncate(max_len);
        }
        let e = Example { ids, mask };
        if e.loss_tokens() == 0 {
            lost.push(*line);
            continue;
        }
        out.push(e);
    }
    let lines = |v: &[usize]| format!("line{} {:?}{}", if v.len() > 1 { "s" } else { "" }, &v[..v.len().min(10)],
                                      if v.len() > 10 { " ..." } else { "" });
    ensure!(no_reply.is_empty(), "{name}: {} without an assistant reply to learn from ({})",
            if no_reply.len() == 1 { "a row".to_string() } else { format!("{} rows", no_reply.len()) }, lines(&no_reply));
    // a cut example trains on a reply without its end (and without <|im_end|>)
    if !cut.is_empty() {
        eprintln!("warning: {name}: {} of {} examples longer than --max-len={max_len} are cut ({}){}", cut.len(), rows.len(),
                  lines(&cut), if lost.is_empty() { String::new() } else { format!(", {} of them lose the whole reply and are left out", lost.len()) });
    }
    ensure!(!out.is_empty(), "{name}: no examples");
    Ok(out)
}

/// The examples of a JSONL file (see `encode_rows`).
pub fn load_examples(chat: &Chat, path: &Path, max_len: usize) -> Result<Vec<Example>> {
    encode_rows(chat, &read_rows(path)?, max_len, &path.display().to_string())
}

/// Takes a dev set out of `rows` when there is no dev.jsonl: 5 % (at least 30 or a tenth, at most 200), chosen with a
/// fixed seed so that every run of the same data holds out the same rows; nothing below 100 rows.
fn hold_out(rows: &mut Vec<(usize, String)>) -> Vec<(usize, String)> {
    let n = rows.len();
    if n < 100 {
        return Vec::new();
    }
    let n_dev = (n / 20).max((n / 10).min(30)).min(200);
    let order = shuffled(n, &mut 0x5EED);
    let mut dev_idx = order[..n_dev].to_vec();
    dev_idx.sort();
    let dev: Vec<_> = dev_idx.iter().map(|&i| rows[i].clone()).collect();
    for &i in dev_idx.iter().rev() {
        rows.remove(i);
    }
    dev
}

/// runs/<name of the data>-<first free number>
pub fn run_dir(data: &Path) -> std::path::PathBuf {
    let name = data.file_stem().map_or("run".into(), |s| s.to_string_lossy().into_owned());
    (1..).map(|i| Path::new("runs").join(format!("{name}-{i}"))).find(|d| !d.exists()).unwrap()
}

pub fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// 0..n in a random order (Fisher-Yates)
pub fn shuffled(n: usize, rng: &mut u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, (splitmix(rng) % (i as u64 + 1)) as usize);
    }
    order
}

/// little-endian f32 bytes as floats
pub fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks().0.iter().map(|&c| f32::from_le_bytes(c)).collect()
}

pub fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

/// adapter_model.safetensors (f32) and adapter_config.json in PEFT's format
pub fn save_adapter(tr: &Trainer, out: &Path, base: &Path) -> Result<()> {
    fs::create_dir_all(out)?;
    let params = tr.read_params(false)?;
    let tensors = tr.tensors();
    let mut views = Vec::new();
    for (n, r, c) in &tensors {
        views.push((n.clone(), safetensors::tensor::TensorView::new(Dtype::F32, vec![*r, *c], f32_bytes(&params[n]))?));
    }
    safetensors::serialize_to_file(views, None, &out.join("adapter_model.safetensors"))?;
    // "...layers.3.self_attn.q_proj.lora_A.weight" -> "q_proj"
    let mut targets: Vec<&str> = tensors.iter().filter_map(|(n, ..)| n.strip_suffix(".lora_A.weight")?.rsplit('.').next()).collect();
    targets.sort();
    targets.dedup();
    let o = tr.opts();
    let cfg = json!({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": crate::hub::name(base),
        "r": o.rank, "lora_alpha": o.alpha, "lora_dropout": o.dropout, "target_modules": targets,
        "bias": "none", "fan_in_fan_out": false, "inference_mode": true, "init_lora_weights": true, "use_dora": false,
        "use_rslora": false, "modules_to_save": null, "rank_pattern": {}, "alpha_pattern": {},
    });
    fs::write(out.join("adapter_config.json"), serde_json::to_string_pretty(&cfg)?)?;
    Ok(())
}

pub fn pack<'a>(examples: impl Iterator<Item = &'a Example>, limit: usize) -> Vec<Micro> {
    let mut out: Vec<Micro> = vec![Micro::default()];
    for e in examples {
        if out.last().unwrap().tokens() + e.ids.len() > limit && out.last().unwrap().tokens() > 0 {
            out.push(Micro::default());
        }
        out.last_mut().unwrap().push(&e.ids, &e.mask);
    }
    out
}

pub struct TrainOpts {
    /// at most: with a dev set the adapter of the lowest dev loss is kept and training stops early (PATIENCE)
    pub epochs: usize,
    /// examples per optimizer step
    pub batch: usize,
    pub lr: f32,
    /// longer examples are cut
    pub max_len: usize,
}

/// The rest of the recipe: warmup share of the steps, gradient clipping, AdamW weight decay, the seed of the shuffle
/// and of LoRA's init, and how many dev measurements (every half epoch) without a new low stop training
const WARMUP: f32 = 0.05;
const MAX_GRAD_NORM: f32 = 1.0;
const WEIGHT_DECAY: f32 = 0.0;
const SEED: u64 = 42;
const PATIENCE: usize = 2;

impl TrainOpts {
    /// Learning rate of `step` of `total`: HF's cosine schedule with linear warmup, evaluated before the step (the
    /// first step has lr 0)
    pub fn lr_at(&self, step: usize, total: usize) -> f32 {
        let warm = (WARMUP * total as f32).ceil() as usize;
        if step < warm {
            self.lr * step as f32 / warm as f32
        } else {
            self.lr * 0.5 * (1.0 + (std::f32::consts::PI * (step - warm) as f32 / (total - warm).max(1) as f32).cos())
        }
    }
}

/// Mean loss and token accuracy over `examples`, HF Trainer's eval_loss (mean of per-batch token means over batches
/// of 8 in order, weighted by batch size), and the mean of per-example token means (HF's eval_loss with an eval batch
/// of 1, as ft-kit / Unsloth report it).
fn evaluate(t: &mut Trainer, examples: &[Example]) -> Result<(f64, f64, f64, f64)> {
    let (mut sum, mut n, mut correct, mut hf, mut hf_n, mut per_ex) = (0.0, 0, 0, 0.0, 0, 0.0);
    for batch in examples.chunks(8) {
        let (mut bs, mut bn) = (0.0, 0);
        for mb in pack(batch.iter(), t.tmax()) {
            let (s, c) = t.pass(&mb, false, 0.0, 0, 0)?;
            let rows = t.row_losses(mb.rows.len())?;
            for &(s0, len) in &mb.docs {
                let doc: Vec<f32> = mb.rows.iter().zip(&rows).filter(|(r, _)| (s0..s0 + len).contains(&(**r as usize)))
                    .map(|(_, &l)| l).collect();
                per_ex += doc.iter().map(|&l| l as f64).sum::<f64>() / doc.len().max(1) as f64;
            }
            bs += s;
            bn += mb.rows.len();
            correct += c;
        }
        sum += bs;
        n += bn;
        hf += bs / bn.max(1) as f64 * batch.len() as f64;
        hf_n += batch.len();
    }
    Ok((sum / n as f64, correct as f64 / n as f64, hf / hf_n as f64, per_ex / examples.len() as f64))
}

/// With a dev set: the dev loss is measured every half epoch, the adapter is saved whenever it is the lowest so far,
/// and training stops after PATIENCE measurements without a new low, so `adapter/` holds the best adapter and `epochs`
/// is an upper bound. Without one (fewer than 100 examples and no dev.jsonl): the adapter after `epochs`.
pub fn train(model: &Path, data: &Path, out: Option<&Path>, o: Opts, to: TrainOpts) -> Result<()> {
    let t0 = Instant::now();
    // data: a directory with train.jsonl (and dev.jsonl), or one JSONL file
    let (train_path, dev_path) = if data.is_dir() {
        let t = data.join("train.jsonl");
        ensure!(t.exists(), "{} has no train.jsonl (one row {{\"messages\": [...]}} per line; dev.jsonl is optional)",
                data.display());
        (t, Some(data.join("dev.jsonl")).filter(|d| d.exists()))
    } else {
        ensure!(data.exists(), "{}: no such file or directory", data.display());
        (data.to_path_buf(), None)
    };
    let out = out.map_or_else(|| run_dir(data), Path::to_path_buf);
    let adapter = out.join("adapter");
    ensure!(!adapter.exists(), "{} already holds an adapter: give another out dir (or leave it out for runs/<data>-N)",
            out.display());
    let chat = Chat::load(model)?;
    let tname = train_path.display().to_string();
    let mut rows = read_rows(&train_path)?;
    let (dev_rows, dev_name, held_out) = match &dev_path {
        Some(d) => (read_rows(d)?, d.display().to_string(), false),
        None => (hold_out(&mut rows), tname.clone(), true),
    };
    let train_ex = encode_rows(&chat, &rows, to.max_len, &tname)?;
    let dev_ex = if dev_rows.is_empty() { Vec::new() } else { encode_rows(&chat, &dev_rows, to.max_len, &dev_name)? };
    fs::create_dir_all(&out).with_context(|| format!("{}", out.display()))?;
    if held_out && !dev_rows.is_empty() {
        fs::write(out.join("dev.jsonl"), dev_rows.iter().map(|(_, l)| format!("{l}\n")).collect::<String>())?;
    }

    let lmax = train_ex.iter().chain(&dev_ex).map(|e| e.ids.len()).max().context("no examples")?;
    let tokens: usize = train_ex.iter().map(|e| e.ids.len()).sum();
    println!("{} training examples, {tokens} tokens ({} in the loss), {} dev examples; longest {lmax}",
             train_ex.len(), train_ex.iter().map(|e| e.loss_tokens()).sum::<usize>(), dev_ex.len());
    if held_out && !dev_ex.is_empty() {
        println!("dev: {} examples held out of {tname} (the same ones every run), written to {}; a dev.jsonl next to \
                  train.jsonl chooses them yourself", dev_ex.len(), out.join("dev.jsonl").display());
    } else if dev_ex.is_empty() {
        println!("warning: fewer than 100 examples and no dev.jsonl: no dev set, so no early stopping and no best adapter; \
                  training all {} epochs", to.epochs);
    }
    let seen: std::collections::HashSet<&[u32]> = train_ex.iter().map(|e| e.ids.as_slice()).collect();
    let leaked = dev_ex.iter().filter(|e| seen.contains(e.ids.as_slice())).count();
    if leaked > 0 {
        println!("warning: {leaked} of {} dev examples are also in the training data: the dev loss looks better than it is \
                  and early stopping stops late", dev_ex.len());
    }
    if train_ex.len() < 500 {
        println!("note: {} training examples; 500 to 5,000 are the usual advice", train_ex.len());
    }
    // short of GPU memory: a smaller micro-batch (the same gradients, accumulated over more passes) until it fits:
    // half, but not below the longest example (rounded up to 256 tokens: a micro-batch holds at least one whole row)
    let mut o = o;
    let floor = lmax.max(512).div_ceil(256) * 256;
    let mut tr = loop {
        match Trainer::new(model, o.clone(), o.micro_tokens.max(lmax), lmax, SEED) {
            Ok(t) => break t,
            Err(e) if crate::engine::out_of_memory(&e) => {
                let (free, all) = mem_gib();
                ensure!(o.micro_tokens > floor, "GPU out of memory ({free:.1} of {all:.1} GiB free) with micro-batches \
                        of {} tokens, the least that holds the longest example ({lmax} tokens): a lower --max-len, or \
                        free the GPU (nvidia-smi lists what holds it)", o.micro_tokens);
                o.micro_tokens = (o.micro_tokens / 2).max(floor);
                println!("GPU memory short ({free:.1} GiB free): --micro-tokens={} (same result, a little slower)", o.micro_tokens);
            }
            Err(e) => return Err(e),
        }
    };
    let (free, all) = mem_gib();
    println!("loaded in {:.1}s, {} LoRA parameters, {free:.1} GiB free of {all:.1}", t0.elapsed().as_secs_f64(),
             tr.tensors().iter().map(|t| t.1 * t.2).sum::<usize>());

    let steps_per_epoch = train_ex.len().div_ceil(to.batch);
    let total = steps_per_epoch * to.epochs;
    let eval_every = if dev_ex.is_empty() { 0 } else { steps_per_epoch.div_ceil(2) };
    if eval_every > 0 {
        println!("up to {} epochs ({total} steps); dev loss every {eval_every} steps, the adapter of the lowest kept, \
                  stopping after {PATIENCE} without a new low", to.epochs);
    }
    let mut rng = SEED;
    let mut log = String::new();
    let t1 = Instant::now();
    let (mut step, mut trained, mut eval_s) = (0, 0, 0.0);
    // (dev loss, HF eval_loss, token accuracy, step) of the saved adapter; measurements since it
    let (mut best, mut since): (Option<(f64, f64, f64, usize)>, usize) = (None, 0);
    'train: for epoch in 0..to.epochs {
        for chunk in shuffled(train_ex.len(), &mut rng).chunks(to.batch) {
            let ts = Instant::now();
            let lr = to.lr_at(step, total);
            let n_loss: usize = chunk.iter().map(|&i| train_ex[i].loss_tokens()).sum();
            tr.zero_grads()?;
            let (mut loss, mut toks) = (0.0, 0);
            for (mi, mb) in pack(chunk.iter().map(|&i| &train_ex[i]), tr.tmax()).iter().enumerate() {
                loss += tr.pass(mb, true, 1.0 / n_loss as f32, step as u64, mi)?.0;
                toks += mb.tokens();
            }
            let norm = tr.optimizer_step(lr, MAX_GRAD_NORM, WEIGHT_DECAY)?;
            let dt = ts.elapsed().as_secs_f64();
            let line = json!({"step": step + 1, "epoch": epoch, "loss": loss / n_loss as f64, "grad_norm": norm, "lr": lr,
                              "tokens": toks, "seconds": dt});
            log.push_str(&format!("{line}\n"));
            if step % 10 == 0 || step + 1 == total {
                println!("step {:4}/{total}  loss {:.4}  grad norm {:.3}  lr {:.2e}  {:.0} tok/s", step + 1,
                         loss / n_loss as f64, norm, lr, toks as f64 / dt);
            }
            step += 1;
            trained += toks;
            if eval_every > 0 && (step % eval_every == 0 || step == total) {
                let te = Instant::now();
                let (l, acc, hf, ex) = evaluate(&mut tr, &dev_ex)?;
                let new_low = best.is_none_or(|b| l < b.0);
                if new_low {
                    save_adapter(&tr, &adapter, model)?;
                    best = Some((l, hf, acc, step));
                    since = 0;
                } else {
                    since += 1;
                }
                eval_s += te.elapsed().as_secs_f64();
                log.push_str(&format!("{}\n", json!({"step": step, "dev_loss": l, "eval_loss": hf, "eval_loss_per_example": ex,
                                                      "token_accuracy": acc})));
                println!("  dev loss {l:.4} (HF eval_loss {hf:.4}, per example {ex:.4}), token accuracy {:.2}%{}", acc * 100.0,
                         if new_low { "  (best, saved)".to_string() } else { format!("  ({since} without a new low)") });
                if since >= PATIENCE && step < total {
                    println!("stopping at step {step}/{total}: no new low in {since} measurements");
                    break 'train;
                }
            }
        }
    }
    let train_s = t1.elapsed().as_secs_f64();
    let free = mem_gib().0;
    println!("TRAINED in {train_s:.1}s ({:.0} tok/s), {:.1} GiB free at the end", trained as f64 / (train_s - eval_s), free);
    let mut summary = json!({"train_seconds": train_s, "steps": step, "max_steps": total, "tokens": trained});
    if let Some((l, hf, acc, s)) = best {
        println!("EVAL best adapter (step {s}): dev loss {l:.4} (HF eval_loss {hf:.4}), token accuracy {:.2}%", acc * 100.0);
        summary["best_step"] = json!(s);
        summary["dev_loss"] = json!(l);
        summary["eval_loss"] = json!(hf);
        summary["token_accuracy"] = json!(acc);
    } else {
        save_adapter(&tr, &adapter, model)?;
    }
    fs::write(out.join("train_log.jsonl"), log)?;
    fs::write(out.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;
    let name = crate::hub::name(model);
    println!("adapter in {}, {:.1}s in all\nserve it: qwenny serve {}--adapter={}", adapter.display(), t0.elapsed().as_secs_f64(),
             if name == crate::hub::DEFAULT_MODEL { String::new() } else { format!("--model={name} ") }, adapter.display());
    Ok(())
}

fn ref_f32(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name).with_context(|| format!("reference tensor {name}"))?;
    ensure!(t.dtype() == Dtype::F32, "{name} is {:?}", t.dtype());
    Ok(f32s(t.data()))
}

fn ref_i32(st: &SafeTensors, name: &str) -> Result<Vec<i32>> {
    let t = st.tensor(name).with_context(|| format!("reference tensor {name}"))?;
    Ok(t.data().as_chunks().0.iter().map(|&c| i32::from_le_bytes(c)).collect())
}

/// Reference tensors with the given prefix, by PEFT name without the adapter name (".default")
fn ref_params(st: &SafeTensors, prefix: &str) -> Result<HashMap<String, Vec<f32>>> {
    let mut out = HashMap::new();
    for name in st.names() {
        if let Some(rest) = name.strip_prefix(prefix) {
            out.insert(rest.replace(".default", ""), ref_f32(st, name)?);
        }
    }
    Ok(out)
}

/// |a - b| / |b| and the cosine of a and b
fn compare(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut d, mut nb, mut na, mut dot) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += ((x - y) as f64).powi(2);
        nb += (y as f64).powi(2);
        na += (x as f64).powi(2);
        dot += x as f64 * y as f64;
    }
    ((d / nb.max(1e-30)).sqrt(), dot / (na * nb).sqrt().max(1e-30))
}

/// Loss, per-token NLL, LoRA gradients and one AdamW step against make_train_reference.py (dropout off).
pub fn traincheck(model: &Path, reference: &Path, o: Opts, packed: bool) -> Result<()> {
    let bytes = fs::read(reference)?;
    let r = SafeTensors::deserialize(&bytes)?;
    let mut docs = Vec::new();
    while let Ok(ids) = ref_i32(&r, &format!("ids_{}", docs.len())) {
        let d = docs.len();
        let mask: Vec<u8> = ref_i32(&r, &format!("mask_{d}"))?.into_iter().map(|m| m as u8).collect();
        docs.push((Example { ids: ids.into_iter().map(|x| x as u32).collect(), mask }, ref_f32(&r, &format!("nll_{d}"))?));
    }
    let lmax = docs.iter().map(|d| d.0.ids.len()).max().context("no documents in the reference")?;
    let total: usize = docs.iter().map(|d| d.0.loss_tokens()).sum();
    let tmax = if packed { docs.iter().map(|d| d.0.ids.len()).sum() } else { lmax };
    let mut o = o;
    o.dropout = 0.0;
    o.layers = ref_i32(&r, "layers").map_or(0, |l| l[0] as usize);
    let mode = if o.f32 { "f32" } else if o.f16 { "f16" } else { "bf16" };
    let mut tr = Trainer::new(model, o, tmax, lmax, 0)?;
    tr.write_params(&ref_params(&r, "init.")?)?;
    tr.zero_grads()?;
    let micros = if packed { pack(docs.iter().map(|d| &d.0), tmax) } else { docs.iter().map(|d| pack(std::iter::once(&d.0), tmax).pop().unwrap()).collect() };
    let (mut loss, mut worst_nll) = (0.0, 0f32);
    let mut nll_err = 0f64;
    let mut di = 0;
    for (mi, mb) in micros.iter().enumerate() {
        loss += tr.pass(mb, true, 1.0 / total as f32, 0, mi)?.0;
        let ours = tr.row_losses(mb.rows.len())?;
        let mut k = 0;
        for &(s0, len) in &mb.docs {
            let reff = &docs[di].1;
            for (j, &row) in mb.rows.iter().enumerate() {
                let row = row as usize;
                if row >= s0 && row < s0 + len {
                    let d = (ours[j] - reff[row - s0]).abs();
                    worst_nll = worst_nll.max(d);
                    nll_err += d as f64;
                    k += 1;
                }
            }
            di += 1;
        }
        let _ = k;
    }
    let ref_loss = ref_f32(&r, "loss")?[0] as f64;
    println!("{} documents ({}), {total} loss tokens, {} build", docs.len(), if packed { "packed in one pass" } else { "one pass each" },
             mode);
    println!("loss {:.6} vs reference {ref_loss:.6} (diff {:.2e}); per-token NLL mean |diff| {:.2e}, max {:.2e}",
             loss / total as f64, (loss / total as f64 - ref_loss).abs(), nll_err / total as f64, worst_nll);

    let ours = tr.read_params(true)?;
    let reff = ref_params(&r, "grad.")?;
    let (mut worst, mut worst_name, mut min_cos) = (0f64, String::new(), 1f64);
    let mut by_kind: HashMap<String, (f64, usize)> = HashMap::new();
    for (name, g) in &reff {
        let (rel, cos) = compare(&ours[name], g);
        if rel > worst {
            worst = rel;
            worst_name = name.clone();
        }
        min_cos = min_cos.min(cos);
        let kind = name.split('.').rev().nth(2).unwrap_or("?").to_string() + "." + name.split('.').rev().nth(1).unwrap_or("?");
        let e = by_kind.entry(kind).or_default();
        e.0 = e.0.max(rel);
        e.1 += 1;
    }
    println!("gradients of {} tensors: worst relative error {worst:.2e} ({worst_name}), lowest cosine {min_cos:.6}", reff.len());
    let mut kinds: Vec<_> = by_kind.into_iter().collect();
    kinds.sort_by(|a, b| a.0.cmp(&b.0));
    for (k, (e, n)) in kinds {
        println!("  {k:<18} worst relative error {e:.2e} ({n} tensors)");
    }
    let norm = tr.optimizer_step(2e-4, 1.0, 0.0)?;
    let ref_norm = ref_f32(&r, "grad_norm")?[0];
    let after = tr.read_params(false)?;
    let ref_after = ref_params(&r, "step1.")?;
    let init = ref_params(&r, "init.")?;
    let mut worst_step = 0f64;
    for (name, p) in &ref_after {
        let ours_d: Vec<f32> = after[name].iter().zip(&init[name]).map(|(a, b)| a - b).collect();
        let ref_d: Vec<f32> = p.iter().zip(&init[name]).map(|(a, b)| a - b).collect();
        worst_step = worst_step.max(compare(&ours_d, &ref_d).0);
    }
    println!("grad norm {norm:.5} vs reference {ref_norm:.5}; AdamW step: worst relative error of the update {worst_step:.2e}");
    Ok(())
}

/// Token ids and loss masks against make_train_reference.py's `data` reference (TRL's tokenization).
pub fn datacheck(model: &Path, data: &Path, reference: &Path) -> Result<()> {
    let chat = Chat::load(model)?;
    let bytes = fs::read(reference)?;
    let r = SafeTensors::deserialize(&bytes)?;
    for split in ["train", "dev"] {
        let ex = load_examples(&chat, &data.join(format!("{split}.jsonl")), usize::MAX)?;
        let (ids, mask, starts) = (ref_i32(&r, &format!("{split}_ids"))?, ref_i32(&r, &format!("{split}_mask"))?,
                                   ref_i32(&r, &format!("{split}_starts"))?);
        ensure!(starts.len() == ex.len() + 1, "{split}: {} examples, reference {}", ex.len(), starts.len() - 1);
        let (mut bad_ids, mut bad_mask) = (0, 0);
        for (i, e) in ex.iter().enumerate() {
            let (a, b) = (starts[i] as usize, starts[i + 1] as usize);
            let same_ids = e.ids.len() == b - a && e.ids.iter().zip(&ids[a..b]).all(|(x, y)| *x as i32 == *y);
            let same_mask = e.mask.len() == b - a && e.mask.iter().zip(&mask[a..b]).all(|(x, y)| *x as i32 == *y);
            if !same_ids && bad_ids < 3 {
                println!("  {split} {i}: ids differ ({} vs {} tokens)", e.ids.len(), b - a);
            } else if same_ids && !same_mask && bad_mask < 3 {
                let at = e.mask.iter().zip(&mask[a..b]).position(|(x, y)| *x as i32 != *y).unwrap();
                println!("  {split} {i}: mask differs first at token {at} (ours {}, TRL {})", e.mask[at], mask[a + at]);
            }
            bad_ids += !same_ids as usize;
            bad_mask += (same_ids && !same_mask) as usize;
        }
        println!("{split}: {} examples: ids identical for {}, masks identical for {}", ex.len(), ex.len() - bad_ids,
                 ex.len() - bad_ids - bad_mask);
    }
    Ok(())
}
