// Compiles kernels/qwen.cu (per Qwen size) to PTX, and kernels/qwen_train.cu (per size) three
// times each (bf16, f32 and f16 activations) (compute_86: JIT-compiled by the driver on the 3090 and newer GPUs).
use std::{env, path::PathBuf, process::Command};

fn ptx(src: &str, out: &str, defines: &[&str]) {
    println!("cargo:rerun-if-changed={src}");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join(out);
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    let ok = Command::new(&nvcc)
        .args(["--ptx", "-O3", "-arch=compute_86", "-lineinfo"])
        .args(defines)
        .arg(src)
        .arg("-o")
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("running {nvcc}: {e}"))
        .success();
    assert!(ok, "nvcc failed on {src}");
}

fn main() {
    println!("cargo:rerun-if-env-changed=NVCC");
    // backends: CUDA when its feature is on (never on macOS), MLX always on macOS and with the `mlx` feature
    println!("cargo::rustc-check-cfg=cfg(has_cuda, has_mlx, has_metal)");
    let macos = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");
    if env::var_os("CARGO_FEATURE_MLX").is_some() || macos {
        println!("cargo:rustc-cfg=has_mlx");
    }
    if macos {
        println!("cargo:rustc-cfg=has_metal");
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_none() || macos {
        return;
    }
    println!("cargo:rustc-cfg=has_cuda");
    println!("cargo:rerun-if-changed=kernels");
    ptx("kernels/qwen.cu", "qwen4b.ptx", &[]);
    ptx("kernels/qwen.cu", "qwen27b.ptx", &["-DQWEN_27B"]);
    ptx("kernels/qwen.cu", "qwen9b.ptx", &["-DQWEN_9B"]);
    ptx("kernels/qwen_train.cu", "qwen4b_train_bf16.ptx", &[]);
    ptx("kernels/qwen_train.cu", "qwen4b_train_f32.ptx", &["-DACT_F32"]);
    ptx("kernels/qwen_train.cu", "qwen4b_train_f16.ptx", &["-DACT_F16"]);
    ptx("kernels/qwen_train.cu", "qwen27b_train_bf16.ptx", &["-DQWEN_27B"]);
    ptx("kernels/qwen_train.cu", "qwen27b_train_f32.ptx", &["-DQWEN_27B", "-DACT_F32"]);
    ptx("kernels/qwen_train.cu", "qwen27b_train_f16.ptx", &["-DQWEN_27B", "-DACT_F16"]);
    ptx("kernels/qwen_train.cu", "qwen9b_train_bf16.ptx", &["-DQWEN_9B"]);
    ptx("kernels/qwen_train.cu", "qwen9b_train_f32.ptx", &["-DQWEN_9B", "-DACT_F32"]);
    ptx("kernels/qwen_train.cu", "qwen9b_train_f16.ptx", &["-DQWEN_9B", "-DACT_F16"]);
}
