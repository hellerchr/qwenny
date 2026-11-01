//! What the server and the checks ask of the model (`Engine`, implemented by `cuda::qwen::Model` and the server's test
//! doubles), and which models it runs.
use crate::lora::Adapter;
use anyhow::{Context, Result, bail};
use std::path::Path;

/// That a parsed config.json is a model this engine runs: Qwen3.5 / Qwen3.8 (`model_type` qwen3_5: Gated DeltaNet and
/// gated attention)
pub fn check_config(config: &serde_json::Value) -> Result<()> {
    match config["model_type"].as_str() {
        Some("qwen3_5") => Ok(()),
        other => bail!("a {} model; this engine runs Qwen3.5 / Qwen3.8 (model_type qwen3_5, e.g. Qwen/Qwen3.5-4B)",
                       other.unwrap_or("?")),
    }
}

/// `check_config` of a model directory
pub fn check_model(dir: &Path) -> Result<()> {
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)
        .with_context(|| format!("{}/config.json", dir.display()))?;
    check_config(&config).with_context(|| format!("{}", dir.display()))
}

/// The model in `dir` on the GPU: `int8` weights (else 16-bit), `adapter` served unmerged for the slots that use it
/// (Engine::set_adapter)
pub fn load(dir: &Path, max_ctx: usize, slots: usize, int8: bool, adapter: Option<&Adapter>) -> Result<Box<dyn Engine>> {
    check_model(dir)?;
    if let Some(a) = adapter {
        a.check(&crate::checkpoint::Checkpoint::open(dir)?)?;
    }
    Ok(Box::new(crate::cuda::qwen::Model::load(dir, max_ctx, slots, int8, adapter)?))
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
    /// segment of the step, tokens to add to its slot's repetition-penalty set.
    fn step(&mut self, toks: &[Tok], out: &[usize], marks: Option<&[Vec<u32>]>) -> Result<()>;
    /// `step` without repetition-penalty marks
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
    /// Whether the model has a draft layer for speculative decoding (Qwen's MTP)
    fn has_draft(&self) -> bool {
        false
    }
    /// Most drafts the model's draft layer returns per step
    fn draft_tokens(&self) -> usize {
        4
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
