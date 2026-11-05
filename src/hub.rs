//! Models by Hugging Face id, downloaded once into the Hugging Face cache (the one Python uses: HF_HOME, HF_TOKEN,
//! HF_HUB_OFFLINE apply).

use anyhow::{Context, Result, bail, ensure};
use hf_hub::{Cache, Repo, RepoType, api::sync::ApiBuilder};
use std::path::{Path, PathBuf};

pub const DEFAULT_MODEL: &str = "Qwen/Qwen3.5-4B";

/// What the engine reads of a model directory
const NEEDED: [&str; 3] = ["config.json", "tokenizer.json", "tokenizer_config.json"];

/// A local model directory as given, or the directory of a Hugging Face id (`Qwen/Qwen3.5-4B`) in the cache,
/// downloaded first when missing (the repository's top-level files but GGUF ones, all of one commit: a model repository
/// may also hold other variants in folders and GGUF builds, tens of GB the engine does not read). Either must be a model the engine runs
/// (`engine::Family`).
pub fn resolve(model: &str) -> Result<PathBuf> {
    resolve_files(model, &NEEDED, true)
}

fn resolve_files(model: &str, needed: &[&str], target_arch: bool) -> Result<PathBuf> {
    let p = Path::new(model);
    let dir = if p.exists() {
        ensure!(p.is_dir(), "{model} is a file; the model is a directory or a Hugging Face id like {DEFAULT_MODEL}");
        p.to_path_buf()
    } else {
        ensure!(model.split('/').count() == 2 && !model.starts_with(['.', '/', '~']),
                "{model}: no such directory, and not a Hugging Face id like {DEFAULT_MODEL}");
        download(model, needed)?
    };
    let missing: Vec<_> = needed.iter().filter(|f| !dir.join(f).exists()).collect();
    ensure!(missing.is_empty(), "{} is not a model directory: no {:?}", dir.display(), missing);
    ensure!(dir.join("model.safetensors").exists() || dir.join("model.safetensors.index.json").exists(),
            "{} has no model.safetensors(.index.json): not a model directory", dir.display());
    if target_arch {
        crate::engine::Family::of_dir(&dir)?;
    }
    Ok(dir)
}

/// The snapshot directory of a Hugging Face id in the cache, if it has the files the engine reads; no network.
#[cfg(test)]
pub fn cached(id: &str) -> Option<PathBuf> {
    cached_files(id, &NEEDED)
}

fn cached_files(id: &str, needed: &[&str]) -> Option<PathBuf> {
    let repo = Cache::from_env().model(id.to_string());
    let paths: Option<Vec<PathBuf>> = needed.iter().map(|f| repo.get(f)).collect();
    paths.and_then(|p| p[0].parent().map(Path::to_path_buf)).filter(|d| needed.iter().all(|f| d.join(f).exists()))
}

/// Whether the engine has a use for a repository's file: the top level, not GGUF
fn engine_reads(file: &str) -> bool {
    !file.contains('/') && !file.ends_with(".gguf")
}

fn download(id: &str, needed: &[&str]) -> Result<PathBuf> {
    if std::env::var("HF_HUB_OFFLINE").is_ok_and(|v| v == "1") {
        return cached_files(id, needed).with_context(|| format!("{id} is not in the Hugging Face cache and HF_HUB_OFFLINE=1"));
    }
    let api = ApiBuilder::from_env().with_progress(true).build()?;
    let info = match api.model(id.to_string()).info() {
        Ok(info) => info,
        Err(e) => match cached_files(id, needed) {
            Some(dir) => {
                eprintln!("note: huggingface.co not reached ({e}); using the cached {id}");
                return Ok(dir);
            }
            None => bail!("{id}: not in the Hugging Face cache and huggingface.co not reached: {e} \
                           (a private or gated model needs HF_TOKEN)"),
        },
    };
    // every file of one commit, so that they land in one snapshot directory
    let repo = api.repo(Repo::with_revision(id.to_string(), RepoType::Model, info.sha.clone()));
    let files: Vec<&str> = info.siblings.iter().map(|s| s.rfilename.as_str()).filter(|f| engine_reads(f)).collect();
    for f in needed {
        ensure!(files.contains(&f), "{id} has no {f}: not a model this engine can load");
    }
    let have = Cache::from_env().repo(Repo::with_revision(id.to_string(), RepoType::Model, info.sha.clone()));
    if files.iter().any(|f| have.get(f).is_none()) {
        eprintln!("downloading {id} (commit {}) into {} (once)", &info.sha[..info.sha.len().min(8)],
                  Cache::from_env().path().display());
    }
    let mut dir = None;
    for f in &files {
        let path = repo.get(f).with_context(|| format!("downloading {id}/{f}"))?;
        if *f == "config.json" {
            dir = path.parent().map(Path::to_path_buf);
        }
    }
    dir.context("no config.json")
}

/// The Hugging Face id of a directory in the cache (`.../models--Qwen--Qwen3.5-4B/snapshots/<sha>`), else the path
pub fn name(dir: &Path) -> String {
    let repo = dir.parent().and_then(Path::parent).and_then(Path::file_name).and_then(|n| n.to_str());
    match repo.and_then(|n| n.strip_prefix("models--")) {
        Some(id) if dir.parent().and_then(Path::file_name).is_some_and(|n| n == "snapshots") => id.replacen("--", "/", 1),
        _ => dir.display().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::engine_reads;

    #[test]
    fn downloads_the_top_level_without_gguf() {
        for f in ["config.json", "model-balanced.safetensors", "model.safetensors.index.json", "mtp_draft_vocab_ids.pt"] {
            assert!(engine_reads(f), "{f}");
        }
        for f in ["TwIL-8B-F16-verbose.gguf", "verbose/model-verbose.safetensors", "assets/logo.png"] {
            assert!(!engine_reads(f), "{f}");
        }
    }
}
