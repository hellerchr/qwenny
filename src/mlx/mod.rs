//! The MLX backend: Qwen3.5 / Qwen3.8 as an MLX graph, with Metal on Apple silicon and MLX's CUDA backend on Linux.
//! The server, chat handling and host sampler are shared with the CUDA backend. Inference only: no MTP, adapters,
//! training or saved session images yet.
pub mod model;
mod weights;
