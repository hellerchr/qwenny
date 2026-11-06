//! The served model's files for the page's download: each directory (the model, an adapter's) as one `.tar`, streamed
//! from disk with its length known up front, or file by file. Only the top level's regular files are offered (symlinks
//! followed, as in the Hugging Face cache), not hidden ones or GGUF builds: the model directory as the engine reads it.

use anyhow::{Context, Result, ensure};
use axum::body::{Body, Bytes};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value as Json, json};
use std::io::Read;
use std::path::{Path, PathBuf};

/// One directory offered for download
pub struct Dir {
    /// what the page shows: the served name of the model or adapter
    pub label: String,
    /// the archive's name and its top folder
    pub stem: String,
    /// the model's Hugging Face id when it came from the cache
    pub hub: Option<String>,
    pub path: PathBuf,
}

impl Dir {
    pub fn new(label: &str, path: &Path) -> Self {
        let id = crate::hub::name(path);
        let hub = (id != path.display().to_string()).then_some(id.clone());
        let base = Path::new(&id).file_name().and_then(|n| n.to_str()).unwrap_or(label);
        let stem: String =
            base.chars().map(|c| if c.is_ascii_alphanumeric() || "-_.".contains(c) { c } else { '_' }).collect();
        Self { label: label.to_string(), stem, hub, path: path.to_path_buf() }
    }

    /// The files offered (name, bytes), by name
    fn files(&self) -> Result<Vec<(String, u64)>> {
        let mut out = vec![];
        for e in std::fs::read_dir(&self.path).with_context(|| self.path.display().to_string())? {
            let e = e?;
            let Ok(name) = e.file_name().into_string() else { continue };
            let Ok(meta) = std::fs::metadata(e.path()) else { continue };
            if meta.is_file() && !name.starts_with('.') && !name.ends_with(".gguf") {
                out.push((name, meta.len()));
            }
        }
        out.sort();
        Ok(out)
    }
}

/// `GET /download`: the directories and their files, `[{label, stem, hub, files: [{name, size}]}]`
pub fn list(dirs: &[Dir]) -> Response {
    let out: Result<Vec<Json>> = dirs
        .iter()
        .map(|d| {
            let files: Vec<Json> = d.files()?.into_iter().map(|(n, s)| json!({"name": n, "size": s})).collect();
            Ok(json!({"label": d.label, "stem": d.stem, "hub": d.hub, "files": files}))
        })
        .collect();
    match out {
        Ok(v) => axum::Json(Json::Array(v)).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response(),
    }
}

/// `GET /download/{i}`: directory `i` as `<stem>.tar`; `GET /download/{i}/{file}`: one of its files
pub fn get(dirs: &[Dir], i: usize, file: Option<&str>) -> Response {
    let Some(d) = dirs.get(i) else { return StatusCode::NOT_FOUND.into_response() };
    let files = match d.files() {
        Ok(f) => f,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")).into_response(),
    };
    // a name from the list only: nothing outside the directory
    let (parts, name) = match file {
        Some(f) => match files.into_iter().find(|(n, _)| n == f) {
            Some((n, s)) => (vec![Part::File(d.path.join(&n), s)], n),
            None => return StatusCode::NOT_FOUND.into_response(),
        },
        None => (tar_parts(&d.stem, &d.path, files), format!("{}.tar", d.stem)),
    };
    let len: u64 = parts.iter().map(Part::len).sum();
    let (tx, rx) = tokio::sync::mpsc::channel::<std::io::Result<Bytes>>(4);
    std::thread::spawn(move || {
        if let Err(e) = send(parts, &tx) {
            // the client is gone, or a file changed: the response ends short of its length, a failed download
            let _ = tx.blocking_send(Err(e));
        }
    });
    let mime = if file.is_some() { "application/octet-stream" } else { "application/x-tar" };
    (
        [
            (header::CONTENT_TYPE, mime.to_string()),
            (header::CONTENT_LENGTH, len.to_string()),
            (header::CONTENT_DISPOSITION, format!("attachment; filename=\"{name}\"")),
        ],
        Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx)),
    )
        .into_response()
}

/// A piece of the response: bytes as they are, or a file's `n` bytes
enum Part {
    Bytes(Vec<u8>),
    File(PathBuf, u64),
}

impl Part {
    fn len(&self) -> u64 {
        match self {
            Part::Bytes(b) => b.len() as u64,
            Part::File(_, n) => *n,
        }
    }
}

const CHUNK: usize = 1 << 20;

fn send(parts: Vec<Part>, tx: &tokio::sync::mpsc::Sender<std::io::Result<Bytes>>) -> std::io::Result<()> {
    let gone = || std::io::Error::other("the client is gone");
    for p in parts {
        match p {
            Part::Bytes(b) => tx.blocking_send(Ok(b.into())).map_err(|_| gone())?,
            Part::File(path, n) => {
                let mut f = std::fs::File::open(&path)?.take(n);
                let mut left = n;
                while left > 0 {
                    let mut buf = vec![0; CHUNK.min(left as usize)];
                    f.read_exact(&mut buf)?;
                    left -= buf.len() as u64;
                    tx.blocking_send(Ok(buf.into())).map_err(|_| gone())?;
                }
            }
        }
    }
    Ok(())
}

/// A tar of `files` in `dir` under the folder `stem/`: a header, the file and its padding to 512 bytes each, then two
/// empty blocks
fn tar_parts(stem: &str, dir: &Path, files: Vec<(String, u64)>) -> Vec<Part> {
    let mut parts = vec![];
    for (name, size) in files {
        let Ok(h) = header(&format!("{stem}/{name}"), size) else { continue };
        parts.push(Part::Bytes(h.to_vec()));
        parts.push(Part::File(dir.join(&name), size));
        let pad = (512 - size % 512) % 512;
        if pad > 0 {
            parts.push(Part::Bytes(vec![0; pad as usize]));
        }
    }
    parts.push(Part::Bytes(vec![0; 1024]));
    parts
}

/// A tar header (GNU format) for a regular file; sizes from 8 GiB on in base-256, as GNU tar writes them
fn header(path: &str, size: u64) -> Result<[u8; 512]> {
    ensure!(path.len() < 100, "{path}: a name of 100 bytes or more");
    let mut h = [0u8; 512];
    let octal = |h: &mut [u8; 512], at: usize, width: usize, v: u64| {
        let s = format!("{v:0w$o}\0", w = width - 1);
        h[at..at + width].copy_from_slice(s.as_bytes());
    };
    h[..path.len()].copy_from_slice(path.as_bytes());
    octal(&mut h, 100, 8, 0o644);
    octal(&mut h, 108, 8, 0);
    octal(&mut h, 116, 8, 0);
    if size < 1 << 33 {
        octal(&mut h, 124, 12, size);
    } else {
        h[124] = 0x80;
        h[128..136].copy_from_slice(&size.to_be_bytes());
    }
    let mtime = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    octal(&mut h, 136, 12, mtime);
    h[156] = b'0';
    h[257..265].copy_from_slice(b"ustar  \0");
    // the checksum counts its own field as spaces
    h[148..156].fill(b' ');
    let sum: u32 = h.iter().map(|&b| b as u32).sum();
    let s = format!("{sum:06o}\0 ");
    h[148..156].copy_from_slice(s.as_bytes());
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tar that Python's tarfile and GNU tar read, small and over 8 GiB (base-256 size)
    #[test]
    fn tar_headers() {
        let h = header("Qwen3.5-4B/config.json", 1234).unwrap();
        assert_eq!(&h[124..136], b"00000002322\0");
        assert_eq!(&h[257..265], b"ustar  \0");
        let sum: u32 = h.iter().enumerate().map(|(i, &b)| if (148..156).contains(&i) { 32 } else { b as u32 }).sum();
        assert_eq!(std::str::from_utf8(&h[148..154]).unwrap(), format!("{sum:06o}"));
        let big = header("m/model.safetensors", 9 << 30).unwrap();
        assert_eq!(big[124], 0x80);
        assert_eq!(u64::from_be_bytes(big[128..136].try_into().unwrap()), 9 << 30);
        assert!(header(&"x".repeat(100), 1).is_err());
    }
}
