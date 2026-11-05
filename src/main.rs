mod auth;
mod chat;
mod checkpoint;
mod checks;
#[cfg(has_cuda)]
mod cuda;
mod decide;
mod draft_vocab;
mod engine;
mod hub;
mod lora;
mod sample;
mod server;
mod sessions;
#[cfg(has_cuda)]
mod training;
#[cfg(has_mlx)]
mod mlx;
#[cfg(has_cuda)]
mod preference;

use anyhow::{Context, Result, bail, ensure};
use engine::Engine;
use std::{env, path::Path};

const HELP: &str = "qwenny: serve Qwen models and fine-tune them with LoRA.

  qwenny serve --model=Qwen/Qwen3.5-4B    OpenAI-compatible server and chat page, http://localhost:18023/
  qwenny train <data> [out dir]           LoRA fine-tune (data: a dir with train.jsonl and optionally dev.jsonl, or one
                                          .jsonl; out: runs/<data>-N); rows with \"chosen\" and \"rejected\":
                                          preference training (DPO + NLL, RPO)
  qwenny merge <adapter dir> <out dir>    write the model with a LoRA merged in
  qwenny bench [tokens]                   decode (1 and more streams), verify steps and prompt speed
  qwenny password                         set the web page's password (asked for; it is also chosen on the page's first
                                          visit)
  qwenny help checks                      the checks and research commands

--model is a directory or a Hugging Face id, downloaded once (default Qwen/Qwen3.5-4B; also Qwen3.5-9B, Qwen3.8-27B
and compressed-tensors int4/int8 checkpoints).
options of serve, bench and the checks that load a model:
  --adapter=[<name>=]<dir>[,<dir>...]  a LoRA, served unmerged. Without <name>=: applied to the one served model (--name),
                                       no separate base. With <name>=: serve answers \"model\": <name> with it and --name
                                       without it (the base). Several comma-separated dirs compose (stack) into one
                                       adapter: their deltas add, each keeping its own scale, combined rank <= 64
  --bf16                               16-bit weights instead of int8
  --backend=cuda|mlx                   default: CUDA when built in, else MLX
  --cache-mem=<GiB>                    device memory the slots' caches take at most (default: all that is free)
serve: --host=0.0.0.0 --port=18023 --name=qwenny
  --ctx=<tokens>        default: the model's native context, or what one session can hold on the GPU if less
  --slots=16            sessions at most; they share GPU and host memory by what they hold
  --drafts=<n>          speculative drafts per step, 0: off (default: the draft layer's own, 7 for n-gram chains)
  --decisions=<presets.json>[,more.json]   named decision tasks for /v1/decisions and the Decide tab
  --db=<file>           the password and API keys (default ~/.local/share/qwenny/qwenny.db)
  --stats               log tokens per step and acceptance per draft position for each request
bench: --pos=0 (context before decoding) --slots=1 --ctx=<pos + 4096>
train: the defaults are the tested recipe: --epochs=3 (at most: the adapter of the lowest dev loss is kept, training
  stops after 2 measurements without a new low) --batch=8 --lr=2e-4 --rank=16 --alpha=32 --dropout=0.05 --max-len=4096
  --precision=f16 (the big products accumulated in f16; bf16; f32: the check build)
  --init-adapter=<dir> (start from a PEFT adapter, its rank and alpha)
  preference rows: --beta=0.1 --nll=1.0 (the chosen response's NLL weight) --lr=5e-6 --dropout=0
--key value works like --key=value";

const HELP_CHECKS: &str = "checks: tools/checks.sh runs them against their references (tools/make_*.py make those).
Each ends in one summary line and exits non-zero when it fails. They take --model=, and --bf16 where it applies.
  qwenny check <reference.safetensors> [chunk]  logits vs. Transformers (tools/make_reference.py) or vLLM's top-k
                                                log-probabilities (tools/make_reference_vllm.py); --limit=<max mean
                                                |diff|>; --write=<file> writes this engine's logits as a reference
  qwenny layercheck <reference.safetensors>     one layer's mixer vs. the reference (--layer=N; Qwen: 0 DeltaNet, 3
                                                attention; --limit=<max |diff|>)
  qwenny mtpcheck <reference.safetensors>       the draft layer (MTP) vs. tools/make_reference_mtp.py (reported only)
  qwenny adaptercheck <reference.safetensors>   a LoRA (--adapter) and the base model in the same steps vs. each alone
                                                and vs. tools/make_reference_lora.py (--limit=<max mean |diff|>)
  qwenny tokcheck <reference.safetensors> <dev.jsonl>   prompt tokens vs. the reference
  qwenny chatcheck <tokref.json>                chat template and tokenizer vs. Transformers (tools/make_tokref.py)
  qwenny samplecheck                            GPU sampler vs. the host reference
  qwenny speccheck                              steps with drafts vs. one token per step
  qwenny longcheck                              long context: decode steps vs. prompt chunks (--limit=<max |diff|>)
  qwenny cachecheck, swapcheck                  slots saved, loaded and swapped to host memory continue identically
                                                (swapcheck: --len=<tokens>)
  qwenny traincheck <grads reference>           loss, gradients, AdamW step vs. PEFT (--precision, --packed)
  qwenny datacheck <data dir> <data reference>  token ids and loss masks vs. Unsloth's
  qwenny dpocheck <dpo reference>               preference training: RPO loss, log-probabilities, gradients vs. PyTorch
                                                + PEFT (tools/make_dpo_reference.py; --precision, --packed)
research commands (not checks):
  qwenny capture <prompts.jsonl> <out.safetensors>   residual activations per prompt and layer (--last=K suffix rows)
  qwenny bake-direction <direction.safetensors> <out_adapter_dir>   abliteration adapter from a direction (--alpha=1.0)";

/// --stats: log speculative decoding per request
static SPEC_STATS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// --cache-mem: GiB of device memory the slots' caches take at most
#[cfg(has_cuda)]
static CACHE_MEM_GIB: std::sync::OnceLock<f64> = std::sync::OnceLock::new();

/// Command-line options: `--flag` or `--key=value`.
struct Opts {
    int8: bool,
    /// training precision: f16 (default), bf16, f32 (the check build)
    precision: String,
    values: Vec<(String, String)>,
    switches: Vec<String>,
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

    /// The model with a given context and slots and --adapter loaded (the slots on the base model until set_adapter)
    fn load_at(&self, dir: &str, ctx: usize, slots: usize) -> Result<Box<dyn Engine>> {
        engine::load(Path::new(dir), ctx, slots, self.int8, self.adapter()?.as_ref().map(|a| &a.1))
    }

    fn has(&self, key: &str) -> bool {
        self.values.iter().any(|(k, _)| k == key)
    }

    fn path(&self, key: &str) -> Option<std::path::PathBuf> {
        self.values.iter().find(|(k, _)| k == key).map(|(_, v)| v.into())
    }

    fn switch(&self, name: &str) -> bool {
        self.switches.iter().any(|s| s == name)
    }

    /// --adapter=[<name>=]<dir>[,<dir>...]: a LoRA (served unmerged). Several comma-separated dirs COMPOSE (stack)
    /// into one adapter (lora::Adapter::compose). WITHOUT <name>= the adapter(s) are applied to the one served model
    /// (the --name default) with no separate pristine base; WITH <name>= the base is served under --name and the
    /// adapted model additionally under <name>, both selectable. Returns the explicit name (None: applied to the
    /// default model) and the adapter. Repeating --adapter for several DISTINCT served adapters is not supported yet.
    fn adapter(&self) -> Result<Option<(Option<String>, lora::Adapter)>> {
        let vals: Vec<&str> = self.values.iter().filter(|(k, _)| k == "adapter").map(|(_, v)| v.as_str()).collect();
        let v = match vals.as_slice() {
            [] => return Ok(None),
            [v] => *v,
            _ => bail!("serving several distinct named adapters at once is not supported yet; compose several into \
                        one with --adapter=<name>=<dir1>,<dir2>,..."),
        };
        // [<name>=]<dir>[,<dir>...]: a leading <name>= only when it is not itself a path or a comma-list
        let (name, rest) = match v.split_once('=') {
            Some((n, d)) if !n.is_empty() && !n.contains('/') && !n.contains(',') && !Path::new(v).exists() => (Some(n.to_string()), d),
            _ => (None, v),
        };
        let dirs: Vec<std::path::PathBuf> = rest.split(',').filter(|s| !s.is_empty())
            .map(|d| std::fs::canonicalize(tilde(d)).with_context(|| format!("--adapter: {d}")))
            .collect::<Result<_>>()?;
        ensure!(!dirs.is_empty(), "--adapter: no directory given");
        Ok(Some((name, lora::Adapter::compose_dirs(&dirs)?)))
    }

    #[cfg(has_cuda)]
    fn lora(&self) -> Result<training::Opts> {
        Ok(training::Opts {
            rank: self.get("rank", 16)?, alpha: self.get("alpha", 32.0)?, dropout: self.get("dropout", 0.05)?,
            micro_tokens: 4096, f32: self.precision == "f32", f16: self.precision == "f16", layers: 0,
        })
    }

    #[cfg(has_cuda)]
    fn train_opts(&self) -> Result<training::TrainOpts> {
        Ok(training::TrainOpts {
            epochs: self.get("epochs", 3)?, batch: self.get("batch", 8)?, lr: self.get("lr", 2e-4)?,
            max_len: self.get("max-len", 4096)?, init: self.path("init-adapter"),
        })
    }

    fn get<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T> {
        Ok(self.opt(key)?.unwrap_or(default))
    }

    /// The value of --key, None when it is not given
    fn opt<T: std::str::FromStr>(&self, key: &str) -> Result<Option<T>> {
        match self.values.iter().find(|(k, _)| k == key) {
            Some((_, v)) => v.parse().map(Some).map_err(|_| anyhow::anyhow!("bad value for --{key}: {v}")),
            None => Ok(None),
        }
    }

    /// --limit=<x>: the pass criterion of a check, which fails when its measure is not below it; the checks' limits
    /// live in tools/checks.sh
    fn limit(&self) -> Result<f32> {
        self.opt("limit")?.context("--limit=<number> is required: the check fails when its measure is not below it")
    }
}

/// `~/x` as `$HOME/x` (the shell leaves a ~ alone after `=`)
fn tilde(p: &str) -> String {
    match (p.strip_prefix("~/"), env::var("HOME")) {
        (Some(rest), Ok(home)) => format!("{home}/{rest}"),
        _ => p.to_string(),
    }
}

/// Options that take no value; every other `--key` takes one (`--key=value` or `--key value`).
const SWITCHES: [&str; 3] = ["bf16", "packed", "stats"];

/// Options, switches and positional arguments (after the command) of every command, checked before anything runs.
/// Every command takes --model (a directory or a Hugging Face id; default Qwen/Qwen3.5-4B).
fn strict(cmd: &str) -> Option<(Vec<&'static str>, Vec<&'static str>, std::ops::RangeInclusive<usize>)> {
    const GPU: [&str; 2] = ["model", "cache-mem"];
    const LOAD: [&str; 5] = ["model", "cache-mem", "adapter", "ctx", "slots"];
    let (bf16, with) = (vec!["bf16"], |base: &[&'static str], more: &[&'static str]| [base, more].concat());
    let lora = ["model", "rank", "alpha", "dropout", "precision"];
    Some(match cmd {
        "serve" => (with(&LOAD, &["host", "port", "name", "drafts", "decisions", "db"]), vec!["bf16", "stats"], 0..=0),
        "password" => (vec!["db"], vec![], 0..=0),
        "train" => (with(&lora, &["epochs", "batch", "lr", "max-len", "init-adapter", "beta", "nll"]), vec![], 1..=2),
        "merge" => (vec!["model"], vec![], 2..=2),
        "bench" => (with(&LOAD, &["pos"]), bf16, 0..=1),
        "check" => (with(&LOAD, &["write", "limit"]), bf16, 1..=2),
        "layercheck" => (vec!["model", "layer", "limit"], bf16, 1..=1),
        "capture" => (with(&LOAD, &["last"]), bf16, 2..=2),
        "bake-direction" => (vec!["model", "alpha"], vec![], 2..=2),
        "mtpcheck" => (with(&GPU, &[]), bf16, 1..=1),
        "adaptercheck" => (with(&GPU, &["adapter", "limit"]), bf16, 1..=1),
        "tokcheck" | "datacheck" => (vec!["model"], vec![], 2..=2),
        "chatcheck" => (vec!["model"], vec![], 1..=1),
        "speccheck" | "samplecheck" | "cachecheck" => (with(&GPU, &[]), bf16, 0..=0),
        "longcheck" => (with(&GPU, &["limit"]), bf16, 0..=0),
        "swapcheck" => (with(&GPU, &["len"]), bf16, 0..=0),
        "traincheck" | "dpocheck" => (lora.to_vec(), vec!["packed"], 1..=1),
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

const SEE_HELP: &str = "qwenny --help lists the commands and options";

fn main() -> Result<()> {
    let all: Vec<String> = env::args().skip(1).collect();
    if all.is_empty() || all.iter().any(|a| a == "--help" || a == "-h") || all[0] == "help" {
        // not println!: `qwenny --help | head` must not panic on the closed pipe
        use std::io::Write;
        let _ = writeln!(std::io::stdout(), "{}", if all.get(1).is_some_and(|a| a == "checks") { HELP_CHECKS } else { HELP });
        return Ok(());
    }
    let (mut args, mut flags, mut values) = (Vec::new(), Vec::new(), Vec::new());
    let mut it = all.iter();
    while let Some(a) = it.next() {
        let Some(f) = a.strip_prefix("--") else {
            args.push(a);
            continue;
        };
        match f.split_once('=') {
            Some((k, v)) => values.push((k.to_string(), tilde(v))),
            None if SWITCHES.contains(&f) => flags.push(a),
            None => match it.clone().next() {
                Some(v) if !v.starts_with("--") => values.push((f.to_string(), it.next().unwrap().clone())),
                _ => flags.push(a),
            },
        }
    }
    let Some(cmd) = args.first().map(|s| s.as_str()) else { bail!("no command given: {SEE_HELP}") };
    let Some((mut keys, switches, positional)) = strict(cmd) else { bail!("no command {cmd}: {SEE_HELP}") };
    keys.push("backend");
    for (k, _) in &values {
        ensure!(keys.contains(&k.as_str()), "{cmd} has no option --{k}{}", did_you_mean(k, keys.iter().chain(switches.iter())));
    }
    for f in &flags {
        ensure!(!keys.contains(&&f[2..]), "{f} needs a value: {f}=<value>");
        ensure!(switches.contains(&&f[2..]), "{cmd} has no option {f}{}", did_you_mean(&f[2..], keys.iter().chain(switches.iter())));
    }
    let a: Vec<&str> = args[1..].iter().map(|s| s.as_str()).collect();
    ensure!(a.len() >= *positional.start(), "{cmd}: missing arguments ({SEE_HELP})");
    if let Some(dir) = a.iter().find(|x| Path::new(x).join("config.json").exists()) {
        ensure!(a.len() <= *positional.end(), "{cmd}: {dir} is a model: give it as --model={dir}");
    }
    ensure!(a.len() <= *positional.end(), "{cmd}: unexpected arguments {:?} ({SEE_HELP})", &a[*positional.end()..]);
    if cmd == "train" {
        ensure!(Path::new(a[0]).exists(), "{}: no such file or directory", a[0]);
    }
    let mut opts = Opts {
        int8: !flags.iter().any(|f| *f == "--bf16"), precision: String::new(), values,
        switches: flags.iter().map(|f| f[2..].to_string()).collect(),
    };
    opts.precision = opts.get("precision", "f16".to_string())?;
    ensure!(["f16", "bf16", "f32"].contains(&opts.precision.as_str()), "--precision is f16, bf16 or f32");
    SPEC_STATS.store(opts.switch("stats"), std::sync::atomic::Ordering::Relaxed);
    if let Some(gib) = opts.opt::<f64>("cache-mem")? {
        ensure!(gib > 0.0, "--cache-mem=<GiB of device memory>, e.g. 4");
        #[cfg(has_cuda)]
        CACHE_MEM_GIB.set(gib).ok();
    }
    // MLX would run f32 matrix products in TF32 (10-bit mantissas): off, they stay f32
    #[cfg(has_mlx)]
    // SAFETY: startup, before any thread
    unsafe {
        env::set_var("MLX_ENABLE_TF32", "0")
    };
    // Do not call engine::backend() eagerly as opts.get's default: that initializes the OnceLock before set_backend.
    let selected = match opts.values.iter().find(|(k, _)| k == "backend") {
        Some((_, b)) => b.clone(),
        None => engine::backend().name().to_string(),
    };
    engine::set_backend(engine::Backend::parse(&selected)?);
    if cmd == "password" {
        return password(&opts);
    }
    // the model: a directory or a Hugging Face id, downloaded once
    let model = hub::resolve(&opts.get("model", hub::DEFAULT_MODEL.to_string())?)?;
    let (m, p) = (model.to_str().context("model path is not UTF-8")?, |i: usize| Path::new(a[i]));
    match cmd {
        "serve" => {
            let (adapter, name) = match opts.adapter()? {
                Some((n, a)) => (Some(a), n),
                None => (None, None),
            };
            let base = opts.get("name", "qwenny".to_string())?;
            ensure!(name.as_ref().is_none_or(|n| *n != base), "the adapter's name is the base model's (--name): give it \
                     another with --adapter=<name>=<dir>");
            // no explicit name= : apply the adapter to the one served model (no separate pristine base)
            let adapter_use = match name {
                Some(n) => server::AdapterUse::Named(n),
                None if adapter.is_some() => server::AdapterUse::Always,
                None => server::AdapterUse::None,
            };
            let chat = chat::Chat::load(&model)?;
            // no --ctx: the model's native context, lowered to what one session can hold on the device (server.rs)
            let native = native_ctx(&model).unwrap_or(32768);
            let (ctx, slots, int8) = (opts.get("ctx", native)?, opts.get("slots", 16)?, opts.int8);
            // several files, comma-separated: their presets one after the other
            let presets = opts.get("decisions", String::new())?.split(',').filter(|f| !f.is_empty())
                .map(|f| Ok(serde_json::from_str::<Vec<serde_json::Value>>(&std::fs::read_to_string(f).with_context(|| f.to_string())?)?))
                .collect::<Result<Vec<_>>>()?.concat();
            let so = server::ServeOpts {
                host: opts.get("host", "0.0.0.0".to_string())?, port: opts.get("port", 18023)?, name: base,
                adapter: adapter_use,
                drafts: opts.opt("drafts")?, sessions: opts.opt("slots")?,
                presets: serde_json::Value::Array(presets), db: opts.path("db").unwrap_or_else(auth::default_path),
                fit_ctx: !opts.has("ctx"),
            };
            server::serve(move || engine::load(&model, ctx, slots, int8, adapter.as_ref()), chat, so)
        }
        #[cfg(has_cuda)]
        "train" => {
            let (mut lo, mut to) = (opts.lora()?, opts.train_opts()?);
            if preference::is_preference(p(0))? {
                // preference defaults: dropout off (TRL's DPOConfig.disable_dropout: the implicit reward compares the
                // policy with the reference, both without dropout), lr 5e-6 (alignment-handbook's QLoRA DPO recipe)
                if !opts.has("dropout") {
                    lo.dropout = 0.0;
                }
                if !opts.has("lr") {
                    to.lr = 5e-6;
                }
                let po = preference::PrefOpts { beta: opts.get("beta", 0.1)?, nll: opts.get("nll", 1.0)? };
                preference::train(&model, p(0), a.get(1).map(Path::new), lo, to, po)
            } else {
                ensure!(!opts.has("beta") && !opts.has("nll"), "--beta and --nll are options of \
                         preference training (rows with \"chosen\" and \"rejected\")");
                training::train(&model, p(0), a.get(1).map(Path::new), lo, to)
            }
        }
        "merge" => lora::merge(&model, p(0), p(1)),
        "bench" => checks::bench(&opts, m, a.first().map_or(Ok(256), |c| c.parse())?),
        "check" => checks::check(&opts, m, a[0], a.get(1).map_or(Ok(1), |c| c.parse())?),
        #[cfg(has_cuda)]
        "layercheck" => checks::layercheck(&opts, m, a[0], opts.get("layer", 0)?),
        #[cfg(has_cuda)]
        "capture" => checks::capture(&opts, m, a[0], a[1]),
        "bake-direction" => lora::bake_direction(&model, p(0), p(1), opts.get("alpha", 1.0f32)?),
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
        #[cfg(has_cuda)]
        "traincheck" => training::traincheck(&model, p(0), opts.lora()?, opts.switch("packed")),
        #[cfg(has_cuda)]
        "datacheck" => training::datacheck(&model, p(0), p(1)),
        #[cfg(has_cuda)]
        "dpocheck" => preference::dpocheck(&model, p(0), opts.lora()?, opts.switch("packed")),
        #[cfg(not(has_cuda))]
        "train" | "traincheck" | "datacheck" | "dpocheck" | "layercheck" | "capture" => bail!("{cmd}: CUDA backend only so far"),
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
