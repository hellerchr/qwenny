#[cfg(has_cuda)]
mod abliterate;
mod auth;
mod chat;
mod checkpoint;
mod checks;
mod cli;
#[cfg(has_cuda)]
mod cuda;
mod decide;
mod download;
mod draft_vocab;
mod engine;
mod hub;
mod lora;
#[cfg(has_mlx)]
mod mlx;
#[cfg(has_cuda)]
mod preference;
mod sample;
mod server;
mod sessions;
#[cfg(has_cuda)]
mod training;

use anyhow::{Context, Result, ensure};
use cli::{App, Command, Group, Opt, Opts, arg, opt, optional, switch};
use engine::Engine;
use std::path::{Path, PathBuf};

/// --stats: log speculative decoding per request
static SPEC_STATS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// --cache-mem: GiB of device memory the slots' caches take at most
static CACHE_MEM_GIB: std::sync::OnceLock<f64> = std::sync::OnceLock::new();

/// A command's `run` on CUDA builds; elsewhere an error, the command having no other backend yet
macro_rules! cuda {
    ($run:expr) => {{
        #[cfg(has_cuda)]
        let run: cli::Run = $run;
        #[cfg(not(has_cuda))]
        let run: cli::Run = |o| anyhow::bail!("{}: CUDA backend only so far", o.cmd.name);
        run
    }};
}

const MODEL: Opt = opt(
    "model",
    "<dir or id>",
    hub::DEFAULT_MODEL,
    "a model directory or a Hugging Face id, downloaded once to the Hugging Face cache (HF_HOME, HF_TOKEN and \
     HF_HUB_OFFLINE apply): Qwen3.5-4B, Qwen3.5-9B, Qwen3.8-27B and their compressed-tensors int4/int8 checkpoints",
);
const BF16: Opt = switch("bf16", "16-bit weights instead of int8 (pre-quantized checkpoints keep their format)");
const BACKEND: Opt = opt("backend", "cuda|mlx", "", "default: CUDA when built in, else MLX");
const CACHE_MEM: Opt =
    opt("cache-mem", "<GiB>", "", "device memory the slots' caches take at most (default: all that is free)");
const ADAPTER: Opt = opt(
    "adapter",
    "[<name>=]<dir>[,<dir>...]",
    "",
    "a PEFT LoRA, applied unmerged. Several comma-separated dirs compose into one: their deltas add, each with its own \
     scale, combined rank <= 64. serve with <name>= answers \"model\": <name> with it and --name without it; without \
     <name>= every request gets it. The default: none",
);
const CTX: Opt = opt("ctx", "<tokens>", "4096", "context of each slot");
const SLOTS: Opt = opt("slots", "<n>", "1", "sequences held at once");
/// What every check on a loaded engine takes
const GPU: &[Opt] = &[MODEL, BF16, BACKEND, CACHE_MEM];
/// ... and those that load it with an adapter and a context of their choosing
const LOAD: &[Opt] = &[MODEL, BF16, BACKEND, CACHE_MEM, ADAPTER, CTX, SLOTS];
const LORA: &[Opt] = &[
    MODEL,
    opt("rank", "<r>", "16", "LoRA rank"),
    opt("alpha", "<a>", "32", "LoRA alpha: the update is scaled by alpha / rank"),
    opt("dropout", "<p>", "0.05", "dropout on the LoRA input; preference training: 0 unless given"),
    opt("precision", "f16|bf16|f32", "f16", "what the big products accumulate in; f32: the check build"),
];
const PACKED: Opt = switch("packed", "every sequence in one micro-batch; without it, one sequence per micro-batch");
const fn limit(measure: &'static str) -> Opt {
    opt("limit", "<x>", "", measure)
}

static APP: App = App {
    name: "qwenny",
    about: "qwenny: serve Qwen models and fine-tune them with LoRA",
    notes: "qwenny help <command> (or qwenny <command> --help): its arguments, options and defaults.\nOptions are \
            --key=value or --key value. --model, a directory or a Hugging Face id, is downloaded once.",
    checks: "checks: tools/checks.sh runs them against their references (tools/make_*.py make those). Each ends in \
             one summary line and exits non-zero when it fails. qwenny help <check>: its arguments and options.",
    commands: &[
        Command {
            name: "serve",
            group: Group::Main,
            summary: "OpenAI-compatible API and chat page, on http://localhost:18023/",
            about: "The API: /v1/chat/completions, /v1/models, and /v1/decisions (an answer chosen among options; \
                    --decisions names tasks for it). The page talks to the base model or, with --adapter=<name>=<dir>, \
                    to either.",
            args: &[],
            opts: &[
                &[MODEL, BF16, BACKEND, CACHE_MEM, ADAPTER],
                &[
                    opt("host", "<address>", "0.0.0.0", "the address to listen on"),
                    opt("port", "<port>", "18023", "the port to listen on"),
                    opt("name", "<name>", "qwenny", "the model's name (\"model\" in requests and /v1/models)"),
                    opt(
                        "ctx",
                        "<tokens>",
                        "",
                        "a session's context at most (default: the model's native context, or what one session can \
                         hold on the GPU if less)",
                    ),
                    opt("slots", "<n>", "16", "sessions at most; they share GPU and host memory by what they hold"),
                    opt(
                        "drafts",
                        "<n>",
                        "",
                        "speculative drafts per step, 0: off (default: the draft layer's own, 7 for n-gram chains)",
                    ),
                    opt(
                        "decisions",
                        "<presets.json>[,<more.json>]",
                        "",
                        "named decision tasks for /v1/decisions and the Decide tab; without it, none",
                    ),
                    opt("db", "<file>", "", "the password and API keys (default: ~/.local/share/qwenny/qwenny.db)"),
                    switch("stats", "log tokens per step and acceptance per draft position for each request"),
                ],
            ],
            run: serve,
        },
        Command {
            name: "train",
            group: Group::Main,
            summary: "LoRA fine-tune; rows with \"chosen\" and \"rejected\": preference training (DPO + NLL, RPO)",
            about: "A row is {\"messages\": [{\"role\": \"user\", \"content\": ...}, {\"role\": \"assistant\", \
                    \"content\": ...}]}, the loss on the assistant's replies; a preference row is {\"messages\": \
                    [<prompt>], \"chosen\": {<assistant message>}, \"rejected\": {<assistant message>}}. Without \
                    dev.jsonl, a dev set is held out of the rows (the same one every run).\nThe out dir gets adapter/ \
                    (the lowest dev loss), dev.jsonl, train_log.jsonl and summary.json. The defaults are the tested \
                    recipe.",
            args: &[
                arg("data", "a directory with train.jsonl and optionally dev.jsonl, or one .jsonl file"),
                optional("out dir", "where the run goes (default: runs/<data>-N)"),
            ],
            opts: &[
                LORA,
                &[
                    opt("epochs", "<n>", "3", "at most: training stops after 2 dev measurements without a new low"),
                    opt("batch", "<rows>", "8", "rows per optimizer step"),
                    opt("lr", "<rate>", "2e-4", "learning rate; preference training: 5e-6 unless given"),
                    opt("max-len", "<tokens>", "4096", "tokens per row at most; longer rows are cut"),
                    opt(
                        "init-adapter",
                        "<dir>",
                        "",
                        "a PEFT adapter to start from, with its rank and alpha; without it, a new one",
                    ),
                    opt("beta", "<b>", "0.1", "preference training: DPO's beta"),
                    opt("nll", "<w>", "1.0", "preference training: the weight of the chosen reply's NLL"),
                ],
            ],
            run: cuda!(train),
        },
        Command {
            name: "merge",
            group: Group::Main,
            summary: "write the model with a LoRA merged in",
            about: "",
            args: &[
                arg("adapter dir", "a PEFT LoRA (adapter_config.json and adapter_model.safetensors)"),
                arg("out dir", "the model's directory, copied, with the adapter merged into model.safetensors"),
            ],
            opts: &[&[MODEL]],
            run: |o| lora::merge(&o.model()?, o.arg_path(0), o.arg_path(1)),
        },
        Command {
            name: "bench",
            group: Group::Main,
            summary: "decode speed with 1 and more streams, verify steps and prompt speed",
            about: "",
            args: &[optional("tokens", "tokens decoded per measurement (default: 256)")],
            opts: &[
                &[MODEL, BF16, BACKEND, CACHE_MEM, ADAPTER],
                &[
                    opt("pos", "<tokens>", "0", "context before decoding"),
                    opt("ctx", "<tokens>", "", "context of each slot (default: --pos + 4096)"),
                    opt("slots", "<n>", "1", "streams at most"),
                ],
            ],
            run: |o| checks::bench(o, &o.model_str()?, o.args.first().map_or(Ok(256), |c| c.parse())?),
        },
        Command {
            name: "abliterate",
            group: Group::Main,
            summary: "find what makes the model refuse and write a LoRA that removes it, keeping what it knows",
            about: "The search is automatic:\n\
                    1. Both sets are kept to what the base model refuses (harmful) and answers (harmless). One in \
                    four prompts of each is held out (by its hash: the same ones every run); 32 + 8 per set must be \
                    left, so give 50 or more.\n\
                    2. Each middle layer gets its own refusal direction (the difference of the means at the last \
                    prompt token), taken out of what the layer writes (a rank-1 LoRA on out_proj, o_proj and \
                    down_proj). The candidates take it out whole, or without the 8 or 32 directions the harmless \
                    prompts spread along most (their topics, so what the model knows of them stays).\n\
                    3. The candidates answer 32 held-out harmful prompts; the two best are tried again 1.5 times \
                    stronger; the two best of all answer every held-out prompt (with --thinking also with thinking).\n\
                    An edit passes when at least 0.8 of its harmful answers have no refusal phrase, it answers at most \
                    0.02 fewer harmless prompts than the base model (the base model judges what was answered, on \
                    whole answers), and its KL on the base model's own harmless answers is at most 0.1 nats per \
                    token. The best passing one is written to <out dir>/adapter: serve it with \
                    --adapter=abliterated=<out dir>/adapter. When none passes it is an error, and no adapter is \
                    written.\n\
                    <out dir> also gets report.txt, the base model's answers (base-*.jsonl) and every candidate's \
                    adapter and answers (candidates/). About half an hour for the 27B; longer with --thinking.",
            args: &[
                arg("harmful.jsonl", "prompts the model refuses and should answer: one {\"text\": ...} per line"),
                arg("harmless.jsonl", "prompts it answers, the contrast and what the edit must not change"),
                arg("out dir", "the adapter, the report and every answer"),
            ],
            opts: &[
                &[MODEL, BF16, CACHE_MEM],
                &[
                    opt("alpha", "<x>", "1.0", "the edit's strength: 1 takes the direction out whole"),
                    opt(
                        "evaluate",
                        "<adapter dir>",
                        "",
                        "measure this adapter as the finalists are, instead of searching (its answers go to evaluated/ in \
                         the out dir); without it, the search",
                    ),
                    switch(
                        "thinking",
                        "also answer every held-out harmful prompt with thinking on, and rank the finalists by both \
                         (long: the thoughts are generated); for a model served with thinking",
                    ),
                ],
            ],
            run: cuda!(|o| {
                let settings = abliterate::Settings {
                    alpha: o.get("alpha")?,
                    int8: o.int8(),
                    thinking: o.switch("thinking"),
                    evaluate: o.path("evaluate"),
                };
                abliterate::run(&o.model()?, o.arg_path(0), o.arg_path(1), o.arg_path(2), &settings)
            }),
        },
        Command {
            name: "password",
            group: Group::Main,
            summary: "set the web page's password (it is also chosen on the page's first visit)",
            about: "Asked for twice on a terminal, without echo; else read as one line of stdin. Every login ends.",
            args: &[],
            opts: &[&[opt("db", "<file>", "", "the password and API keys (default: ~/.local/share/qwenny/qwenny.db)")]],
            run: password,
        },
        Command {
            name: "capture",
            group: Group::Research,
            summary: "residual activations per prompt and layer, at the prompts' last tokens",
            about: "The prompts are rendered as one user message without thinking, and their last --last tokens must \
                    be the same (the chat template's). The file holds p<i> [layers, last, hidden] per prompt, \
                    group_ids, tail_ids and n_tokens, and the groups' names in its metadata.",
            args: &[
                arg("prompts.jsonl", "one {\"group\": ..., \"text\": ...} per line"),
                arg("out.safetensors", "the activations"),
            ],
            opts: &[
                &[MODEL, BF16, CACHE_MEM, ADAPTER, CTX],
                &[opt("last", "<k>", "4", "the rows per prompt: its last k tokens")],
            ],
            run: cuda!(|o| checks::capture(o, &o.model_str()?, o.arg(0), o.arg(1))),
        },
        Command {
            name: "bake-direction",
            group: Group::Research,
            summary: "an abliteration adapter from a direction: it is taken out of every layer's residual writes",
            about: "For every matrix W writing to the residual stream, a rank-1 LoRA A = rᵀW, B = -r: served, W x + \
                    alpha B A x = (W - alpha r rᵀ W) x.",
            args: &[
                arg("direction.safetensors", "the direction: a vector \"r\" of the model's hidden size"),
                arg("out adapter dir", "the PEFT adapter"),
            ],
            opts: &[&[MODEL, opt("alpha", "<x>", "1.0", "the strength: 1 takes the direction out whole")]],
            run: |o| lora::bake_direction(&o.model()?, o.arg_path(0), o.arg_path(1), o.get("alpha")?),
        },
        Command {
            name: "check",
            group: Group::Check,
            summary: "logits vs. Transformers' or vLLM's",
            about: "",
            args: &[
                arg(
                    "reference.safetensors",
                    "from tools/make_reference.py (Transformers' logits) or tools/make_reference_vllm.py (vLLM's \
                     top-k log-probabilities)",
                ),
                optional("chunk", "prompt tokens per step (default: 1)"),
            ],
            opts: &[
                LOAD,
                &[
                    opt(
                        "write",
                        "<file>",
                        "",
                        "write this engine's logits there as a reference instead; without it, the comparison",
                    ),
                    limit("required without --write: the check fails when the mean |diff| is not below it"),
                ],
            ],
            run: |o| checks::check(o, &o.model_str()?, o.arg(0), o.args.get(1).map_or(Ok(1), |c| c.parse())?),
        },
        Command {
            name: "layercheck",
            group: Group::Check,
            summary: "one layer's mixer vs. the reference",
            about: "",
            args: &[arg("reference.safetensors", "from tools/make_reference.py")],
            opts: &[
                &[MODEL, BF16],
                &[
                    opt("layer", "<n>", "0", "the layer; Qwen: 0 is DeltaNet, 3 attention"),
                    limit("required: the check fails when the largest |diff| is not below it"),
                ],
            ],
            run: cuda!(|o| checks::layercheck(o, &o.model_str()?, o.arg(0), o.get("layer")?)),
        },
        Command {
            name: "mtpcheck",
            group: Group::Check,
            summary: "the draft layer (MTP) vs. tools/make_reference_mtp.py (reported only)",
            about: "",
            args: &[arg("reference.safetensors", "from tools/make_reference_mtp.py")],
            opts: &[GPU],
            run: |o| checks::mtpcheck(o, &o.model_str()?, o.arg(0)),
        },
        Command {
            name: "adaptercheck",
            group: Group::Check,
            summary: "a LoRA and the base model in the same steps vs. each alone and vs. the reference",
            about: "",
            args: &[arg("reference.safetensors", "from tools/make_reference_lora.py")],
            opts: &[
                GPU,
                &[
                    opt("adapter", "<dir>", "", "required: the reference's adapter"),
                    limit("required: the check fails when the mean |diff| is not below it"),
                ],
            ],
            run: |o| checks::adaptercheck(o, &o.model_str()?, o.arg(0)),
        },
        Command {
            name: "tokcheck",
            group: Group::Check,
            summary: "prompt tokens vs. the reference",
            about: "",
            args: &[
                arg("reference.safetensors", "from tools/make_reference.py"),
                arg("dev.jsonl", "the rows the reference tokenized"),
            ],
            opts: &[&[MODEL]],
            run: |o| checks::tokcheck(&o.model_str()?, o.arg(0), o.arg(1)),
        },
        Command {
            name: "chatcheck",
            group: Group::Check,
            summary: "chat template and tokenizer vs. Transformers",
            about: "",
            args: &[arg("tokref.json", "from tools/make_tokref.py")],
            opts: &[&[MODEL]],
            run: |o| {
                ensure!(o.arg_path(0).is_file(), "{} is not a file; make one with tools/make_tokref.py", o.arg(0));
                ensure!(chat::Chat::load(&o.model()?)?.check(o.arg_path(0))?, "the prompts differ from Transformers'");
                Ok(())
            },
        },
        Command {
            name: "samplecheck",
            group: Group::Check,
            summary: "the GPU sampler vs. the host reference",
            about: "",
            args: &[],
            opts: &[GPU],
            run: |o| checks::samplecheck(o, &o.model_str()?),
        },
        Command {
            name: "speccheck",
            group: Group::Check,
            summary: "steps with drafts vs. one token per step",
            about: "",
            args: &[],
            opts: &[GPU],
            run: |o| checks::speccheck(o, &o.model_str()?),
        },
        Command {
            name: "longcheck",
            group: Group::Check,
            summary: "long context: decode steps vs. prompt chunks",
            about: "",
            args: &[],
            opts: &[GPU, &[limit("required: the check fails when the largest |diff| is not below it")]],
            run: |o| checks::longcheck(o, &o.model_str()?),
        },
        Command {
            name: "cachecheck",
            group: Group::Check,
            summary: "slots saved and loaded continue identically",
            about: "",
            args: &[],
            opts: &[GPU],
            run: |o| checks::cachecheck(o, &o.model_str()?),
        },
        Command {
            name: "swapcheck",
            group: Group::Check,
            summary: "slots swapped to host memory and back continue identically",
            about: "",
            args: &[],
            opts: &[GPU, &[opt("len", "<tokens>", "131072", "the swapped sessions' length")]],
            run: |o| checks::swapcheck(o, &o.model_str()?),
        },
        Command {
            name: "traincheck",
            group: Group::Check,
            summary: "loss, per-token NLL, LoRA gradients and one AdamW step vs. PEFT",
            about: "",
            args: &[arg("grads reference", "from tools/make_train_reference.py grads")],
            opts: &[LORA, &[PACKED]],
            run: cuda!(|o| training::traincheck(&o.model()?, o.arg_path(0), o.lora()?, o.switch("packed"))),
        },
        Command {
            name: "datacheck",
            group: Group::Check,
            summary: "token ids and loss masks vs. Unsloth's",
            about: "",
            args: &[
                arg("data dir", "the training data the reference was made from"),
                arg("data reference", "from tools/make_train_reference.py data"),
            ],
            opts: &[&[MODEL]],
            run: cuda!(|o| training::datacheck(&o.model()?, o.arg_path(0), o.arg_path(1))),
        },
        Command {
            name: "dpocheck",
            group: Group::Check,
            summary: "preference training: RPO loss, log-probabilities and gradients vs. PyTorch + PEFT",
            about: "",
            args: &[arg("dpo reference", "from tools/make_dpo_reference.py")],
            opts: &[LORA, &[PACKED]],
            run: cuda!(|o| preference::dpocheck(&o.model()?, o.arg_path(0), o.lora()?, o.switch("packed"))),
        },
    ],
};

impl Opts {
    /// --model: a directory or a Hugging Face id, downloaded once
    fn model(&self) -> Result<PathBuf> {
        hub::resolve(&self.get::<String>("model")?)
    }

    fn model_str(&self) -> Result<String> {
        Ok(self.model()?.to_str().context("the model's path is not UTF-8")?.to_string())
    }

    fn int8(&self) -> bool {
        !self.switch("bf16")
    }

    /// The model with --ctx, --slots and --adapter (every slot with the adapter)
    fn load(&self, dir: &str) -> Result<Box<dyn Engine>> {
        let adapter = self.adapter()?;
        let slots = self.get("slots")?;
        let mut m = engine::load(Path::new(dir), self.get("ctx")?, slots, self.int8(), adapter.as_ref().map(|a| &a.1))?;
        if adapter.is_some() {
            for s in 0..slots as u32 {
                m.set_adapter(s, true)?;
            }
        }
        Ok(m)
    }

    /// The model with a given context and slots and --adapter loaded (the slots on the base model until set_adapter)
    fn load_at(&self, dir: &str, ctx: usize, slots: usize) -> Result<Box<dyn Engine>> {
        engine::load(Path::new(dir), ctx, slots, self.int8(), self.adapter()?.as_ref().map(|a| &a.1))
    }

    /// --adapter=[<name>=]<dir>[,<dir>...]: the name it is served under (None: applied to the one served model) and
    /// the adapter, its dirs composed (lora::Adapter::compose)
    fn adapter(&self) -> Result<Option<(Option<String>, lora::Adapter)>> {
        let Some((name, dirs)) = self.adapter_dirs()? else { return Ok(None) };
        Ok(Some((name, lora::Adapter::compose_dirs(&dirs)?)))
    }

    /// --adapter: its name if given, and its directories
    fn adapter_dirs(&self) -> Result<Option<(Option<String>, Vec<PathBuf>)>> {
        let Some(v) = self.given("adapter") else { return Ok(None) };
        // a leading <name>= only when it is not itself a path or a comma-list
        let (name, rest) = match v.split_once('=') {
            Some((n, d)) if !n.is_empty() && !n.contains('/') && !n.contains(',') && !Path::new(v).exists() => {
                (Some(n.to_string()), d)
            }
            _ => (None, v),
        };
        let dirs: Vec<PathBuf> = rest
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|d| std::fs::canonicalize(cli::tilde(d)).with_context(|| format!("--adapter: {d}")))
            .collect::<Result<_>>()?;
        ensure!(!dirs.is_empty(), "--adapter: no directory given");
        Ok(Some((name, dirs)))
    }

    #[cfg(has_cuda)]
    fn lora(&self) -> Result<training::Opts> {
        let precision: String = self.get("precision")?;
        Ok(training::Opts {
            rank: self.get("rank")?,
            alpha: self.get("alpha")?,
            dropout: self.get("dropout")?,
            micro_tokens: 4096,
            f32: precision == "f32",
            f16: precision == "f16",
            layers: 0,
        })
    }
}

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let o = match cli::parse(&APP, &argv)? {
        cli::Parsed::Help(text) => {
            // not println!: `qwenny --help | head` must not panic on the closed pipe
            use std::io::Write;
            let _ = writeln!(std::io::stdout(), "{text}");
            return Ok(());
        }
        cli::Parsed::Run(o) => o,
    };
    if let Some(gib) = o.opt::<f64>("cache-mem")? {
        ensure!(gib > 0.0, "--cache-mem=<GiB of device memory>, e.g. 4");
        CACHE_MEM_GIB.set(gib).ok();
    }
    // MLX would run f32 matrix products in TF32 (10-bit mantissas): off, they stay f32
    #[cfg(has_mlx)]
    // SAFETY: startup, before any thread
    unsafe {
        std::env::set_var("MLX_ENABLE_TF32", "0")
    };
    if let Some(b) = o.given("backend") {
        engine::set_backend(engine::Backend::parse(b)?);
    }
    (o.cmd.run)(&o)
}

fn serve(o: &Opts) -> Result<()> {
    SPEC_STATS.store(o.switch("stats"), std::sync::atomic::Ordering::Relaxed);
    let model = o.model()?;
    let (adapter, name) = match o.adapter()? {
        Some((n, a)) => (Some(a), n),
        None => (None, None),
    };
    let base: String = o.get("name")?;
    ensure!(
        name.as_ref().is_none_or(|n| *n != base),
        "the adapter's name is the base model's (--name): give it another with --adapter=<name>=<dir>"
    );
    // no explicit name= : apply the adapter to the one served model (no separate pristine base)
    let adapter_use = match name {
        Some(n) => server::AdapterUse::Named(n),
        None if adapter.is_some() => server::AdapterUse::Always,
        None => server::AdapterUse::None,
    };
    let chat = chat::Chat::load(&model)?;
    // no --ctx: the model's native context, lowered to what one session can hold on the device (server.rs)
    let ctx = o.opt("ctx")?.unwrap_or(native_ctx(&model).unwrap_or(32768));
    let (slots, int8) = (o.get("slots")?, o.int8());
    // several files, comma-separated: their presets one after the other
    let presets = o
        .given("decisions")
        .unwrap_or("")
        .split(',')
        .filter(|f| !f.is_empty())
        .map(|f| {
            Ok(serde_json::from_str::<Vec<serde_json::Value>>(
                &std::fs::read_to_string(f).with_context(|| f.to_string())?,
            )?)
        })
        .collect::<Result<Vec<_>>>()?
        .concat();
    let so = server::ServeOpts {
        host: o.get("host")?,
        port: o.get("port")?,
        name: base.clone(),
        adapter: adapter_use,
        drafts: o.opt("drafts")?,
        sessions: o.opt("slots")?,
        presets: serde_json::Value::Array(presets),
        db: o.path("db").unwrap_or_else(auth::default_path),
        fit_ctx: !o.has("ctx"),
        downloads: std::iter::once(download::Dir::new(&base, &model))
            .chain(o.adapter_dirs()?.into_iter().flat_map(|(n, dirs)| {
                let n = n.unwrap_or_else(|| format!("{base}'s adapter"));
                dirs.into_iter().map(move |d| download::Dir::new(&n, &d))
            }))
            .collect(),
    };
    server::serve(move || engine::load(&model, ctx, slots, int8, adapter.as_ref()), chat, so)
}

#[cfg(has_cuda)]
fn train(o: &Opts) -> Result<()> {
    let data = o.arg_path(0);
    ensure!(data.exists(), "{}: no such file or directory", data.display());
    let mut lo = o.lora()?;
    let mut to = training::TrainOpts {
        epochs: o.get("epochs")?,
        batch: o.get("batch")?,
        lr: o.get("lr")?,
        max_len: o.get("max-len")?,
        init: o.path("init-adapter"),
    };
    let (model, out) = (o.model()?, o.args.get(1).map(Path::new));
    if preference::is_preference(data)? {
        // preference defaults: dropout off (TRL's DPOConfig.disable_dropout: the implicit reward compares the
        // policy with the reference, both without dropout), lr 5e-6 (alignment-handbook's QLoRA DPO recipe)
        if !o.has("dropout") {
            lo.dropout = 0.0;
        }
        if !o.has("lr") {
            to.lr = 5e-6;
        }
        let po = preference::PrefOpts { beta: o.get("beta")?, nll: o.get("nll")? };
        preference::train(&model, data, out, lo, to, po)
    } else {
        ensure!(
            !o.has("beta") && !o.has("nll"),
            "--beta and --nll are options of preference training (rows with \"chosen\" and \"rejected\")"
        );
        training::train(&model, data, out, lo, to)
    }
}

/// `qwenny password`: set the web page's password (asked for twice on a terminal, without echo; else one line of stdin)
fn password(opts: &Opts) -> Result<()> {
    use std::io::{IsTerminal, Write};
    let a = auth::Auth::open(&opts.path("db").unwrap_or_else(auth::default_path))?;
    let tty = std::io::stdin().is_terminal();
    let stty = |arg: &str| {
        let _ = std::process::Command::new("stty").arg(arg).stdin(std::process::Stdio::inherit()).status();
    };
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
