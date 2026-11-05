//! What the server and the checks ask of the model (`Engine`, implemented by `cuda::qwen::Model` and the server's test
//! doubles), and which models it runs.
use crate::lora::Adapter;
use anyhow::{Context, Result, bail, ensure};
use std::{path::Path, sync::OnceLock};

/// The compute backend: hand-written CUDA kernels, or MLX (Metal on Apple silicon, MLX's CUDA backend on Linux).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    Cuda,
    Mlx,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self { Backend::Cuda => "cuda", Backend::Mlx => "mlx" }
    }

    pub fn parse(s: &str) -> Result<Self> {
        let b = match s {
            "cuda" => Backend::Cuda,
            "mlx" => Backend::Mlx,
            _ => bail!("--backend is cuda or mlx"),
        };
        let compiled = match b { Backend::Cuda => cfg!(has_cuda), Backend::Mlx => cfg!(has_mlx) };
        ensure!(compiled, "this build has no {s} backend (cargo feature `{s}`)");
        Ok(b)
    }
}

static CURRENT: OnceLock<Backend> = OnceLock::new();

/// Choose the backend for this process (once, at startup).
pub fn set_backend(b: Backend) {
    let _ = CURRENT.set(b);
}

/// The chosen backend; CUDA when built with it, else MLX.
pub fn backend() -> Backend {
    *CURRENT.get_or_init(|| if cfg!(has_cuda) { Backend::Cuda } else { Backend::Mlx })
}

static DRAFTER: OnceLock<std::path::PathBuf> = OnceLock::new();

/// A DFlash2 drafter's directory for the models this process loads (--dflash2, once, at startup)
pub fn set_drafter(dir: std::path::PathBuf) {
    let _ = DRAFTER.set(dir);
}

/// The drafter's directory, if one was given
pub fn drafter() -> Option<&'static Path> {
    DRAFTER.get().map(|p| p.as_path())
}

/// The model families this engine runs, by config.json's `model_type`
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Family {
    /// Qwen3.5 / Qwen3.8 (`qwen3_5`): Gated DeltaNet and gated attention; inference, adapters and training
    Qwen,
    /// Granite (`granite`, e.g. TwIL-8B): plain attention in every layer; inference on CUDA only
    Granite,
}

impl Family {
    /// The family of a parsed config.json
    pub fn of(config: &serde_json::Value) -> Result<Self> {
        match config["model_type"].as_str() {
            Some("qwen3_5") => Ok(Family::Qwen),
            Some("granite") => Ok(Family::Granite),
            other => bail!("a {} model; this engine runs Qwen3.5 / Qwen3.8 (model_type qwen3_5, e.g. Qwen/Qwen3.5-4B) and \
                            Granite (model_type granite, e.g. webAI-Official/TwIL-8B)", other.unwrap_or("?")),
        }
    }

    /// The family of a model directory
    pub fn of_dir(dir: &Path) -> Result<Self> {
        let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)
            .with_context(|| format!("{}/config.json", dir.display()))?;
        Self::of(&config).with_context(|| format!("{}", dir.display()))
    }
}

/// The model in `dir` on the selected backend: `int8` weights (else 16-bit), `adapter` served unmerged for the slots
/// that use it (Engine::set_adapter; CUDA only).
#[allow(unused_variables)]
pub fn load(dir: &Path, max_ctx: usize, slots: usize, int8: bool, adapter: Option<&Adapter>) -> Result<Box<dyn Engine>> {
    let family = Family::of_dir(dir)?;
    ensure!(adapter.is_none() || family == Family::Qwen, "LoRA adapters: Qwen models only so far");
    ensure!(drafter().is_none() || (family == Family::Granite && backend() == Backend::Cuda),
            "--dflash2: Granite models on CUDA only so far");
    if let Some(a) = adapter {
        a.check(&crate::checkpoint::Checkpoint::open(dir)?)?;
    }
    match backend() {
        #[cfg(has_cuda)]
        Backend::Cuda => Ok(Box::new(crate::cuda::qwen::Model::load(dir, max_ctx, slots, int8, adapter)?)),
        #[cfg(has_mlx)]
        Backend::Mlx => {
            ensure!(family == Family::Qwen, "--backend=mlx: Qwen models only so far");
            ensure!(adapter.is_none(), "LoRA adapters: CUDA backend only so far");
            Ok(Box::new(crate::mlx::model::Model::load(dir, max_ctx, slots, int8)?))
        }
        #[allow(unreachable_patterns)]
        b => bail!("no {} backend in this build", b.name()),
    }
}

/// One token of a step: its id, the sequence slot it belongs to and its position in that sequence.
#[derive(Clone, Copy, Debug)]
pub struct Tok {
    pub id: u32,
    pub slot: u32,
    pub pos: u32,
}

/// How the drafts of one output row are drawn (Engine::draft): the request's sampler row
/// (sample::Params::draft_row) and a uniform draw per draft
#[derive(Clone, Debug)]
pub struct DraftSampling {
    pub row: [f32; 8],
    pub u: Vec<f32>,
}

/// Inference: steps over any mix of sequences (slots), logits for chosen rows, sampling.
pub trait Engine {
    fn slots(&self) -> usize;
    fn max_ctx(&self) -> usize;
    /// Most tokens in one step, most logit rows in one step, most tokens per step with speculative decoding
    fn max_tok(&self) -> usize;
    fn max_out(&self) -> usize;
    fn spec_tok(&self) -> usize;
    /// Vocabulary size: the length of a logits row
    fn vocab(&self) -> usize;
    /// Run one step; afterwards the logits of the tokens at `out` (indices into `toks`) are available. `marks`: per
    /// segment of the step, the tokens committed since the last step, for its slot's repetition-penalty set and the
    /// draft vocabulary (None: the step's own tokens).
    fn step(&mut self, toks: &[Tok], out: &[usize], marks: Option<&[Vec<u32>]>) -> Result<()>;
    /// `step` with every token of the step committed
    fn run(&mut self, toks: &[Tok], out: &[usize]) -> Result<()> {
        self.step(toks, out, None)
    }
    /// Logits of the last step's output rows, [rows][vocab()]
    fn logits(&self) -> Result<Vec<f32>>;
    fn logits_row(&self, j: usize) -> Result<Vec<f32>>;
    /// Greedy token per output row
    fn argmax(&self) -> Result<Vec<u32>>;
    /// Per output row the sampler's parameters (crate::sample::Params::row); the token and whether its draft was
    /// accepted
    fn sample(&mut self, params: &[[f32; 8]]) -> Result<Vec<(u32, bool)>>;
    /// The tokens slot `slot`'s request generated so far, a bit per vocabulary entry in words of 32 (the presence
    /// penalty of the next `sample`; rows whose presence penalty is 0 do not read it)
    fn set_outputs(&mut self, _slot: u32, _words: &[u32]) -> Result<()> {
        bail!("presence penalty: not supported by this backend")
    }
    /// Compile/prime common decode shapes before the server advertises readiness. No-op for backends that
    /// do not need this; it must not learn tokens or leave a user sequence in a slot.
    fn warmup(&mut self, _draft_depth: usize) -> Result<()> { Ok(()) }
    /// Wait for the device (timing)
    fn sync(&self) -> Result<()>;
    /// Whether a LoRA adapter is loaded (served unmerged beside the base model: set_adapter)
    fn has_adapter(&self) -> bool {
        false
    }
    /// Whether slot `slot`'s next steps use the adapter (else the base model); a slot's cached state is the model's
    /// it was computed with (sessions.rs keeps them apart)
    fn set_adapter(&mut self, _slot: u32, _on: bool) -> Result<()> {
        bail!("no adapter loaded")
    }
    /// Whether the model has a draft layer for speculative decoding (Qwen's MTP, a DFlash2 drafter)
    fn has_draft(&self) -> bool {
        false
    }
    /// Whether `draft` draws from a distribution it keeps for the sampler (MTP); else its drafts are fixed proposals,
    /// accepted with p(token)
    fn samples_drafts(&self) -> bool {
        true
    }
    /// Drafts per step from the model's draft layer, unless --mtp-drafts asks for another depth
    fn draft_tokens(&self) -> usize {
        4
    }
    /// Most drafts the model's draft layer can return per step
    fn draft_capacity(&self) -> usize {
        self.draft_tokens()
    }
    /// Speculative decoding with the draft layer, after a step: `toks` are the tokens that followed the step's tokens
    /// at `rows` (indices into the step's tokens), each at the position of its row's token (the draft layer's cache
    /// is filled for them); then `k` drafts per output row `out` (indices into `toks`), each continuing its row. A row
    /// with a `sampling` (none, or an empty slice: none for any) draws its drafts from the draft layer's distribution
    /// under it, kept for the target's sampler (sample::mark_sampled); the others take the argmax.
    fn draft(&mut self, _toks: &[Tok], _rows: &[usize], _out: &[usize], _k: usize, _sampling: &[Option<DraftSampling>])
    -> Result<Vec<Vec<u32>>> {
        bail!("speculative decoding with a draft layer: not supported by this backend and model")
    }
    /// The distribution a sampled draft of slot `slot` at index `t` of its chain was drawn from (token, probability;
    /// ascending ids) and the draft (samplecheck)
    fn draft_q(&self, _slot: u32, _t: usize) -> Result<(Vec<(u32, f32)>, u32)> {
        bail!("sampled drafts: not supported by this backend")
    }
    /// Bytes of device memory the slots' caches can still take (None: fixed per slot)
    fn cache_room(&self) -> Option<usize> {
        None
    }

    /// Whether slots can be saved and loaded (save_kv, load_kv, save_state, load_state; sessions::Image)
    fn can_save(&self) -> bool {
        false
    }

    /// The recurrent state of slot `slot` after its last step and the tokens seen (not the KV cache): with the KV of
    /// the positions before, enough to continue the slot from there later (load_state)
    fn save_state(&mut self, _slot: u32) -> Result<Vec<u8>> {
        anyhow::bail!("this engine cannot save slots")
    }

    /// Set slot `slot` to a state from save_state taken at position `pos` (its KV cache before `pos` there): its next
    /// step continues at `pos`; false when the device memory ran out
    fn load_state(&mut self, _slot: u32, _data: &[u8], _pos: usize) -> Result<bool> {
        anyhow::bail!("this engine cannot save slots")
    }

    /// Slot `slot`'s KV cache of positions [from, to) (`from` a multiple of 16), in the engine's layout
    fn save_kv(&mut self, _slot: u32, _from: usize, _to: usize) -> Result<Vec<u8>> {
        anyhow::bail!("this engine cannot save slots")
    }

    /// Put a KV cache from save_kv into slot `slot` at positions [from, to); false when the device memory ran out
    fn load_kv(&mut self, _slot: u32, _from: usize, _to: usize, _data: &[u8]) -> Result<bool> {
        anyhow::bail!("this engine cannot save slots")
    }

    /// Host bytes an image of a slot at `len` positions takes (sessions::Image: save_kv of [0, len) and save_state)
    fn image_bytes(&self, _len: usize) -> usize {
        0
    }

    /// Device bytes a slot holding `len` positions maps (cache_room counts them)
    fn slot_bytes(&self, len: usize) -> usize {
        self.image_bytes(len)
    }

    /// Lower the context to `ctx` positions (after loading, from what the device holds; never above the loaded one)
    fn limit_ctx(&mut self, _ctx: usize) {}

    /// Slot `slot` holds no sequence any more: its per-slot memory may go back (the next sequence starts at 0)
    fn release(&mut self, _slot: u32) -> Result<()> {
        Ok(())
    }

    /// Slot `slot`'s KV cache from position `from` (a multiple of 8,192) on goes back; the rest stays
    fn truncate(&mut self, _slot: u32, _from: usize) -> Result<()> {
        anyhow::bail!("this engine cannot save slots")
    }
}

/// Whether an error is the device running out of memory (CUDA_ERROR_OUT_OF_MEMORY)
pub fn out_of_memory(e: &anyhow::Error) -> bool {
    let s = format!("{e:?}");
    s.contains("OUT_OF_MEMORY") || s.contains("out of memory")
}
