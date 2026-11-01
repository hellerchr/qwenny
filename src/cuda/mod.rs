//! The CUDA engine: hand-written kernels (kernels/qwen.cu for inference, kernels/qwen_train.cu for training, shared
//! headers in kernels/*.cuh) and cuBLAS. core.rs holds the shared parts (matrices, steps, sampling, graphs), qwen/ the
//! model, learn.rs what the trainer needs of cuBLAS and the LoRA arena.
pub mod core;
pub mod learn;
pub mod qwen;
pub mod vmm;
