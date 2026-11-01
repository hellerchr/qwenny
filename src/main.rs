mod auth;
mod chat;
mod checkpoint;
mod checks;
mod cuda;
mod decide;
mod engine;
mod hub;
mod lora;
mod sample;
mod server;
mod sessions;
mod training;

use anyhow::{Context, Result, bail, ensure};
use engine::Engine;
use std::{env, path::Path};

const USAGE: &str = "usage: qwenny <command> [--model=<dir or Hugging Face id>] ...  (default Qwen/Qwen3.5-4B, downloaded once;
                      Qwen3.5-4B, Qwen3.5-9B and Qwen3.8-27B, also compressed-tensors int4/int8 checkpoints)
  qwenny serve                                  OpenAI-compatible server (--ui: also the chat and decision page)
  qwenny train <data> [out dir]                 LoRA fine-tune (data: a dir with train.jsonl and optionally dev.jsonl,
                                                or one .jsonl; out: runs/<data>-N)
  qwenny merge <adapter dir> <out dir>          write the model with a LoRA merged in
  qwenny bench [tokens]                         decode (1 and more streams), verify steps and prompt speed
  qwenny password                               set the web page's password (asked for; it is also chosen on the
                                                page's first visit)
 checks (tools/checks.sh runs them against their references):
  qwenny check <reference.safetensors> [chunk]  logits vs. Transformers (tools/make_reference_qwen.py) or vLLM's top-k
                                                log-probabilities (tools/make_reference_vllm.py)
  qwenny layercheck <reference.safetensors>     one layer's mixer vs. the reference (--layer=0: DeltaNet, 3: attention)
  qwenny mtpcheck <reference.safetensors>       the draft layer (MTP) vs. tools/make_reference_mtp.py
  qwenny adaptercheck <reference.safetensors>   a LoRA (--adapter) and the base model in the same steps vs. each
                                                alone and vs. tools/make_reference_lora.py
  qwenny tokcheck <reference.safetensors> <dev.jsonl>   prompt tokens vs. the reference
  qwenny chatcheck <tokref.json>                chat template and tokenizer vs. Transformers (tools/make_tokref.py)
  qwenny samplecheck                            GPU sampler vs. the host reference
  qwenny speccheck                              steps with drafts vs. one token per step
  qwenny longcheck                              long context: decode steps vs. prompt chunks
  qwenny cachecheck, swapcheck                  slots saved, loaded and swapped to host memory continue identically
  qwenny traincheck <grads reference>           loss, gradients, AdamW step vs. PEFT (--precision, --packed)
  qwenny datacheck <data dir> <data reference>  token ids and loss masks vs. Unsloth's
options: --adapter=[<name>=]<dir> (a LoRA, served unmerged: serve answers \"model\": <name> with it and --name without
         it; name default: the directory's, or its parent's for runs/<data>-N/adapter; the checks and bench run it),
         --bf16 (16-bit weights instead of int8),
         bench: --pos=0 (context before decoding) --slots=1 --ctx=4096,
         serve: --ui (the web page at /), --host=0.0.0.0 --port=18023 --name=qwenny, --ctx=<tokens> (default: the
                model's native context, or what one session can hold on the GPU if less), --slots=16 (sessions at
                most; they share GPU and host memory by what they hold), --spec=7 (draft tokens per step; 0: off),
                --decisions=<presets.json>[,more.json] (named decision tasks for /v1/decisions and the Decide tab),
                --db=<file> (the password and API keys; default ~/.local/share/qwenny/qwenny.db)
         train: the defaults are the tested recipe: --epochs=3 (at most: the adapter of the lowest dev loss is kept,
                training stops after 2 measurements without a new low) --batch=8 --lr=2e-4 --rank=16 --alpha=32
                --dropout=0.05 --max-len=4096 --precision=f16 (the big products accumulated in f16; bf16; f32: the
                check build)
         --key value works like --key=value";

/// Command-line options: `--flag` or `--key=value`.
struct Opts {
    int8: bool,
    /// training precision: f16 (default), bf16, f32 (the check build)
    precision: String,
    values: Vec<(String, String)>,
}

impl Opts {
    /// The model with --ctx, --slots and --adapter (every slot with the adapter)
    fn load(&self, dir: &str) -> Result<Box<dyn Engine>> {
        let adapter = self.adapter()?;
        let slots = self.get("slots", 1)?;
        let mut m = engine::load(Path::new(dir), self.get("ctx", 4096)?, slots, self.int8, adapter.as_ref().map(|a| &a.1))?;
        if adapter.is_some() {
            for s in 0..slots as u32 {
                m.set_adapter(s, true)?;
            }
        }
        Ok(m)
    }

    /// The model with a given context and slots, without an adapter
    fn load_at(&self, dir: &str, ctx: usize, slots: usize) -> Result<Box<dyn Engine>> {
        engine::load(Path::new(dir), ctx, slots, self.int8, None)
    }

    /// The same with --adapter loaded (the slots on the base model until set_adapter)
    fn load_at_adapter(&self, dir: &str, ctx: usize, slots: usize) -> Result<Box<dyn Engine>> {
        engine::load(Path::new(dir), ctx, slots, self.int8, self.adapter()?.as_ref().map(|a| &a.1))
    }

    fn path(&self, key: &str) -> Option<std::path::PathBuf> {
        self.values.iter().find(|(k, _)| k == key).map(|(_, v)| v.into())
    }

    /// --adapter=[<name>=]<dir>: a LoRA (served unmerged) and the name it is served under: by default its directory's
    /// name, or its parent's for qwenny train's `runs/<data>-N/adapter`
    fn adapter(&self) -> Result<Option<(String, lora::Adapter)>> {
        let Some(v) = self.values.iter().find(|(k, _)| k == "adapter").map(|(_, v)| v.as_str()) else { return Ok(None) };
        let (name, dir) = match v.split_once('=') {
            Some((n, d)) if !n.is_empty() && !n.contains('/') && !Path::new(v).exists() => (Some(n.to_string()), Path::new(d)),
            _ => (None, Path::new(v)),
        };
        let dir = std::fs::canonicalize(dir).with_context(|| format!("--adapter: {}", dir.display()))?;
        let name = name.unwrap_or_else(|| {
            let last = |p: &Path| p.file_name().map(|f| f.to_string_lossy().into_owned());
            match last(&dir) {
                Some(n) if n == "adapter" => dir.parent().and_then(last).unwrap_or(n),
                n => n.unwrap_or_else(|| "adapter".into()),
            }
        });
        Ok(Some((name, lora::Adapter::load(&dir)?)))
    }

    fn lora(&self) -> Result<training::Opts> {
        Ok(training::Opts {
            rank: self.get("rank", 16)?, alpha: self.get("alpha", 32.0)?, dropout: self.get("dropout", 0.05)?,
            micro_tokens: 4096, f32: self.precision == "f32", f16: self.precision == "f16", layers: 0,
        })
    }

    fn train_opts(&self) -> Result<training::TrainOpts> {
        Ok(training::TrainOpts {
            epochs: self.get("epochs", 3)?, batch: self.get("batch", 8)?, lr: self.get("lr", 2e-4)?,
            max_len: self.get("max-len", 4096)?,
        })
    }

    fn get<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T> {
        match self.values.iter().find(|(k, _)| k == key) {
            Some((_, v)) => v.parse().map_err(|_| anyhow::anyhow!("bad value for --{key}: {v}")),
            None => Ok(default),
        }
    }
}

/// Options that take no value; every other `--key` takes one (`--key=value` or `--key value`).
const SWITCHES: [&str; 3] = ["bf16", "packed", "ui"];

/// Options, switches and positional arguments (after the command) of every command, checked before anything runs.
/// Every command takes --model (a directory or a Hugging Face id; default Qwen/Qwen3.5-4B).
fn strict(cmd: &str) -> Option<(Vec<&'static str>, Vec<&'static str>, std::ops::RangeInclusive<usize>)> {
    const LOAD: [&str; 4] = ["model", "adapter", "ctx", "slots"];
    let bf16 = vec!["bf16"];
    let lora = ["model", "rank", "alpha", "dropout", "precision"];
    Some(match cmd {
        "serve" => ([&LOAD[..], &["host", "port", "name", "spec", "decisions", "db"]].concat(), vec!["bf16", "ui"], 0..=0),
        "password" => (vec!["db"], vec![], 0..=0),
        "train" => ([&lora[..], &["epochs", "batch", "lr", "max-len"]].concat(), vec![], 1..=2),
        "merge" => (vec!["model"], vec![], 2..=2),
        "bench" => ([&LOAD[..], &["pos"]].concat(), bf16, 0..=1),
        "check" => (LOAD.to_vec(), bf16, 1..=2),
        "layercheck" => (vec!["model", "layer"], bf16, 1..=1),
        "mtpcheck" => (vec!["model"], bf16, 1..=1),
        "adaptercheck" => (vec!["model", "adapter"], bf16, 1..=1),
        "tokcheck" => (vec!["model"], vec![], 2..=2),
        "chatcheck" => (vec!["model"], vec![], 1..=1),
        "speccheck" => (vec!["model"], bf16, 0..=0),
        "samplecheck" | "longcheck" | "cachecheck" => (vec!["model"], bf16, 0..=0),
        "swapcheck" => (vec!["model", "len"], bf16, 0..=0),
        "traincheck" => (lora.to_vec(), vec!["packed"], 1..=1),
        "datacheck" => (vec!["model"], vec![], 2..=2),
        _ => return None,
    })
}

/// " (did you mean --x?)" for the closest name within two edits, else ""
fn did_you_mean<'a>(k: &str, names: impl Iterator<Item = &'a &'static str>) -> String {
    let dist = |a: &str, b: &str| {
        let b: Vec<char> = b.chars().collect();
        let mut row: Vec<usize> = (0..=b.len()).collect();
        for (i, ca) in a.chars().enumerate() {
            let mut prev = row[0];
            row[0] = i + 1;
            for j in 0..b.len() {
                let cur = row[j + 1];
                row[j + 1] = (prev + (ca != b[j]) as usize).min(row[j] + 1).min(cur + 1);
                prev = cur;
            }
        }
        row[b.len()]
    };
    names.map(|n| (dist(k, n), n)).filter(|(d, _)| *d <= 2).min().map_or(String::new(), |(_, n)| format!(" (did you mean --{n}?)"))
}

fn main() -> Result<()> {
    let all: Vec<String> = env::args().skip(1).collect();
    let (mut args, mut flags, mut values) = (Vec::new(), Vec::new(), Vec::new());
    let mut it = all.iter();
    while let Some(a) = it.next() {
        let Some(f) = a.strip_prefix("--") else {
            args.push(a);
            continue;
        };
        match f.split_once('=') {
            // `--key=~/x`: the shell leaves the ~ alone after `=`
            Some((k, v)) => values.push((k.to_string(), match (v.strip_prefix("~/"), env::var("HOME")) {
                (Some(rest), Ok(home)) => format!("{home}/{rest}"),
                _ => v.to_string(),
            })),
            None if SWITCHES.contains(&f) => flags.push(a),
            None => match it.clone().next() {
                Some(v) if !v.starts_with("--") => values.push((f.to_string(), it.next().unwrap().clone())),
                _ => flags.push(a),
            },
        }
    }
    let Some(cmd) = args.first().map(|s| s.as_str()) else { bail!("{USAGE}") };
    let Some((keys, switches, positional)) = strict(cmd) else { bail!("no command {cmd}\n{USAGE}") };
    for (k, _) in &values {
        ensure!(keys.contains(&k.as_str()), "{cmd} has no option --{k}{}", did_you_mean(k, keys.iter().chain(switches.iter())));
    }
    for f in &flags {
        ensure!(!keys.contains(&&f[2..]), "{f} needs a value: {f}=<value>");
        ensure!(switches.contains(&&f[2..]), "{cmd} has no option {f}{}", did_you_mean(&f[2..], keys.iter().chain(switches.iter())));
    }
    let a: Vec<&str> = args[1..].iter().map(|s| s.as_str()).collect();
    ensure!(cmd != "train" || !a.is_empty(), "train: which data? qwenny train <data dir or .jsonl>\n{USAGE}");
    ensure!(a.len() >= *positional.start(), "{cmd}: missing arguments\n{USAGE}");
    if let Some(dir) = a.iter().find(|x| Path::new(x).join("config.json").exists()) {
        ensure!(a.len() <= *positional.end(), "{cmd}: {dir} is a model: give it as --model={dir}");
    }
    ensure!(a.len() <= *positional.end(), "{cmd}: unexpected arguments {:?}\n{USAGE}", &a[*positional.end()..]);
    if cmd == "train" {
        ensure!(Path::new(a[0]).exists(), "{}: no such file or directory", a[0]);
    }
    let mut opts = Opts { int8: !flags.iter().any(|f| *f == "--bf16"), precision: String::new(), values };
    opts.precision = opts.get("precision", "f16".to_string())?;
    ensure!(["f16", "bf16", "f32"].contains(&opts.precision.as_str()), "--precision is f16, bf16 or f32");
    if cmd == "password" {
        return password(&opts);
    }
    // the model: a directory or a Hugging Face id, downloaded once
    let model = hub::resolve(&opts.get("model", hub::DEFAULT_MODEL.to_string())?)?;
    let (m, p) = (model.to_str().context("model path is not UTF-8")?, |i: usize| Path::new(a[i]));
    match cmd {
        "serve" => {
            let chat = chat::Chat::load(&model)?;
            // no --ctx: the model's native context, lowered to what one session can hold on the device (server.rs)
            let native = native_ctx(&model).unwrap_or(32768);
            let (ctx, slots, int8) = (opts.get("ctx", native)?, opts.get("slots", 16)?, opts.int8);
            let so = server::ServeOpts { host: opts.get("host", "0.0.0.0".to_string())?, port: opts.get("port", 18023)?,
                                         name: opts.get("name", "qwenny".to_string())?, adapter: None, spec: opts.get("spec", 7)?,
                                         sessions: opts.path("slots").map(|_| slots), ui: flags.iter().any(|f| *f == "--ui"),
                                         presets: match opts.get("decisions", String::new())? {
                                             // several files, comma-separated: their presets one after the other
                                             s if s.is_empty() => serde_json::json!([]),
                                             s => serde_json::Value::Array(s.split(',').map(|p| -> Result<Vec<serde_json::Value>> {
                                                 Ok(serde_json::from_str(&std::fs::read_to_string(p)?)?)
                                             }).collect::<Result<Vec<_>>>()?.concat()),
                                         },
                                         db: opts.path("db").unwrap_or_else(auth::default_path),
                                         fit_ctx: opts.path("ctx").is_none() };
            let (adapter, name) = match opts.adapter()? {
                Some((n, a)) => (Some(a), Some(n)),
                None => (None, None),
            };
            ensure!(name.as_ref().is_none_or(|n| *n != so.name), "the adapter's name is the base model's (--name): give it \
                     another with --adapter=<name>=<dir>");
            let so = server::ServeOpts { adapter: name, ..so };
            server::serve(move || engine::load(&model, ctx, slots, int8, adapter.as_ref()), chat, so)
        }
        "train" => training::train(&model, p(0), a.get(1).map(Path::new), opts.lora()?, opts.train_opts()?),
        "merge" => lora::merge(&model, p(0), p(1)),
        "bench" => checks::bench(&opts, m, a.first().map_or(Ok(256), |c| c.parse())?),
        "check" => checks::check(&opts, m, a[0], a.get(1).map_or(Ok(1), |c| c.parse())?),
        "layercheck" => checks::layercheck(&opts, m, a[0], opts.get("layer", 0)?),
        "mtpcheck" => checks::mtpcheck(&opts, m, a[0]),
        "adaptercheck" => checks::adaptercheck(&opts, m, a[0]),
        "tokcheck" => checks::tokcheck(m, a[0], a[1]),
        "chatcheck" => {
            ensure!(Path::new(a[0]).is_file(), "chatcheck: {} is not a file; make one with tools/make_tokref.py", a[0]);
            ensure!(chat::Chat::load(&model)?.check(Path::new(a[0]))?, "chatcheck: the prompts differ from Transformers'");
            Ok(())
        }
        "samplecheck" => checks::samplecheck(&opts, m),
        "speccheck" => checks::speccheck(&opts, m),
        "longcheck" => checks::longcheck(&opts, m),
        "cachecheck" => checks::cachecheck(&opts, m),
        "swapcheck" => checks::swapcheck(&opts, m),
        "traincheck" => training::traincheck(&model, p(0), opts.lora()?, flags.iter().any(|f| *f == "--packed")),
        "datacheck" => training::datacheck(&model, p(0), p(1)),
        _ => unreachable!("{cmd}: in `strict`, not here"),
    }
}

/// `qwenny password`: set the web page's password (asked for twice on a terminal, without echo; else one line of stdin)
fn password(opts: &Opts) -> Result<()> {
    use std::io::{IsTerminal, Write};
    let a = auth::Auth::open(&opts.path("db").unwrap_or_else(auth::default_path))?;
    let tty = std::io::stdin().is_terminal();
    let stty = |arg: &str| { let _ = std::process::Command::new("stty").arg(arg).stdin(std::process::Stdio::inherit()).status(); };
    let ask = |prompt: &str| -> Result<String> {
        if tty {
            print!("{prompt}");
            std::io::stdout().flush()?;
            stty("-echo");
        }
        let mut l = String::new();
        let read = std::io::stdin().read_line(&mut l);
        if tty {
            stty("echo");
            println!();
        }
        read?;
        Ok(l.trim_end_matches(['\r', '\n']).to_string())
    };
    let pw = ask("new password: ")?;
    if tty {
        ensure!(ask("again: ")? == pw, "the passwords differ");
    }
    a.set_password(&pw)?;
    println!("password set (every login ended)");
    Ok(())
}

/// The model's native context (`max_position_embeddings` of its config, a multiple of 64)
fn native_ctx(dir: &Path) -> Option<usize> {
    let c: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).ok()?).ok()?;
    let n = c.get("text_config").unwrap_or(&c)["max_position_embeddings"].as_u64()? as usize;
    (n >= 64).then_some(n / 64 * 64)
}
