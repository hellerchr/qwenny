//! The MLX backend: Qwen3.5 / Qwen3.8 as an MLX graph, with Metal on Apple silicon and MLX's CUDA backend on Linux.
//! The server, chat handling and host sampler are shared with CUDA. Inference with MTP drafts and an adaptive
//! draft vocabulary; no adapters, training or saved session images yet.
// on Metal, metal.rs has the delta rule's kernels and only its tests use this one
#[cfg_attr(has_metal, allow(dead_code))]
mod delta;
pub mod model;
mod weights;
#[cfg(has_metal)]
mod metal;
