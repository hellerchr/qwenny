//! Backend-neutral adaptive MTP vocabulary: committed-token counts, 16-row tiles, decay and persistence.
use anyhow::{Context, Result, bail, ensure};
use std::{fs, path::{Path, PathBuf}, time::{Duration, Instant}};

pub const MAX_ROWS: usize = 65536;
const DECAY: usize = 20_000;
const NONE: u32 = u32::MAX;

#[derive(Clone, Debug)]
pub struct Rows {
    pub ids: Vec<u32>,
    pub score: Vec<f32>,
    row_of: Vec<u32>,
    pending: Vec<u32>,
    waiting: Vec<bool>,
    since: usize,
    cap: usize,
}

impl Rows {
    pub fn new(vocab: usize, ids: Vec<u32>, score: Vec<f32>, cap: usize) -> Result<Self> {
        ensure!(ids.len() % 16 == 0 && ids.len() <= cap && cap % 16 == 0, "{} rows (at most {cap})", ids.len());
        let mut row_of = vec![NONE; vocab];
        for (r, &t) in ids.iter().enumerate() {
            ensure!((t as usize) < vocab && row_of[t as usize] == NONE, "token {t} twice or beyond the vocabulary");
            row_of[t as usize] = r as u32;
        }
        let score = if score.len() == ids.len() { score } else { vec![0.0; ids.len()] };
        Ok(Self { ids, score, row_of, pending: Vec::new(), waiting: vec![false; vocab], since: 0, cap })
    }

    /// Snapshot only steps that can publish a row change, not ordinary count-only decode steps.
    pub fn may_change(&self, toks: &[u32]) -> bool {
        let mut new = Vec::new();
        for &t in toks {
            if self.row_of.get(t as usize) == Some(&NONE) && !self.waiting[t as usize] && !new.contains(&t) {
                new.push(t);
                if self.pending.len() + new.len() >= 16 { return true; }
            }
        }
        false
    }

    /// Changed tiles. New tokens wait for sixteen distinct ids; then grow, or replace cold rows at capacity.
    pub fn take(&mut self, toks: &[u32]) -> Vec<usize> {
        for &t in toks {
            match self.row_of.get(t as usize) {
                Some(&NONE) if !self.waiting[t as usize] => {
                    self.waiting[t as usize] = true;
                    self.pending.push(t);
                }
                Some(&r) if r != NONE => self.score[r as usize] += 1.0,
                _ => {}
            }
        }
        self.since += toks.len();
        if self.since >= DECAY {
            self.since = 0;
            self.score.iter_mut().for_each(|s| *s *= 0.5);
        }
        let n = self.pending.len() / 16 * 16;
        if n == 0 { return Vec::new(); }
        let new: Vec<u32> = self.pending.drain(..n).collect();
        new.iter().for_each(|&t| self.waiting[t as usize] = false);
        let room = (self.cap - self.ids.len()).min(n);
        let mut tiles: Vec<usize> = (self.ids.len() / 16..(self.ids.len() + room) / 16).collect();
        for &t in &new[..room] {
            self.row_of[t as usize] = self.ids.len() as u32;
            self.ids.push(t);
            self.score.push(1.0);
        }
        let rest = &new[room..];
        let mut cold: Vec<usize> = (0..self.ids.len()).filter(|&r| self.score[r] < 1.0).collect();
        let k = rest.len().min(cold.len());
        if k == 0 { return tiles; }
        if k < cold.len() { cold.select_nth_unstable_by(k, |&a, &b| self.score[a].total_cmp(&self.score[b])); }
        cold.truncate(k);
        for (&r, &t) in cold.iter().zip(rest) {
            self.row_of[self.ids[r] as usize] = NONE;
            (self.ids[r], self.score[r], self.row_of[t as usize]) = (t, 1.0, r as u32);
        }
        tiles.extend(cold.iter().map(|&r| r / 16));
        tiles.sort_unstable();
        tiles.dedup();
        tiles
    }
}

/// Same model-keyed cache as CUDA: ids and scores, never weights.
pub fn saved_path(dir: &Path) -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    let dir = fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    let name = dir.to_string_lossy().trim_matches('/').replace('/', "_");
    Some(base.join("qwenny/draft-heads").join(format!("{name}.json")))
}

pub fn read_saved(p: &Path, vocab: usize) -> Option<(Vec<u32>, Vec<f32>)> {
    let v: serde_json::Value = serde_json::from_str(&fs::read_to_string(p).ok()?).ok()?;
    let ids: Vec<u32> = serde_json::from_value(v["ids"].clone()).ok()?;
    let score: Vec<f32> = serde_json::from_value(v["score"].clone()).ok()?;
    let mut seen = vec![false; vocab];
    let ok = ids.len() == score.len() && ids.len() <= MAX_ROWS
        && ids.iter().all(|&t| (t as usize) < vocab && !std::mem::replace(&mut seen[t as usize], true));
    if !ok { eprintln!("draft head: {} does not fit this model, starting from the draft vocabulary", p.display()); }
    ok.then_some((ids, score))
}

pub struct Adaptive {
    pub rows: Rows,
    path: Option<PathBuf>,
    dirty: bool,
    saved: Instant,
}

impl Adaptive {
    pub fn load(dir: &Path, vocab: usize, seed: &[u32]) -> Result<Self> {
        let path = saved_path(dir);
        let (mut ids, mut score) = path.as_deref().and_then(|p| read_saved(p, vocab)).unwrap_or_default();
        let mut have = vec![false; vocab];
        ids.iter().for_each(|&t| have[t as usize] = true);
        for &t in seed {
            ensure!((t as usize) < vocab, "draft token {t} beyond the vocabulary");
            if ids.len() < MAX_ROWS && !have[t as usize] {
                have[t as usize] = true;
                ids.push(t);
                score.push(0.0);
            }
        }
        let whole = ids.len() / 16 * 16;
        ids.truncate(whole);
        score.truncate(whole);
        ensure!(!ids.is_empty(), "empty adaptive draft vocabulary");
        Ok(Self { rows: Rows::new(vocab, ids, score, MAX_ROWS)?, path, dirty: false, saved: Instant::now() })
    }

    pub fn changed(&mut self) { self.dirty = true; }

    pub fn maybe_save(&mut self) {
        if self.dirty && self.saved.elapsed() >= Duration::from_secs(60) { self.save(); }
    }

    fn save(&mut self) {
        let Some(path) = &self.path else { return };
        let json = serde_json::json!({"ids": self.rows.ids, "score": self.rows.score});
        let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
        let done = path.parent().map_or(Ok(()), fs::create_dir_all)
            .and_then(|_| fs::write(&tmp, json.to_string())).and_then(|_| fs::rename(&tmp, path));
        match done {
            Ok(()) => self.dirty = false,
            Err(e) => eprintln!("draft head: saving {}: {e}", path.display()),
        }
        self.saved = Instant::now();
    }
}

impl Drop for Adaptive {
    fn drop(&mut self) { if self.dirty { self.save(); } }
}

/// A stored PyTorch int64 tensor: central-directory sizes locate */data/0.
pub fn draft_ids(dir: &Path, vocab: usize) -> Result<Vec<u32>> {
    let path = dir.join("mtp_draft_vocab_ids.pt");
    let b = fs::read(&path).with_context(|| format!("{}: draft vocabulary", path.display()))?;
    let at = |o: usize, n: usize| -> Result<usize> {
        let v = b.get(o..o + n).context("truncated draft vocabulary zip")?;
        Ok(v.iter().rev().fold(0, |a, &x| a << 8 | x as usize))
    };
    let end = b.len().checked_sub(22).filter(|&e| b[e..e + 4] == *b"PK\x05\x06")
        .context("draft vocabulary: not a zip archive")?;
    let mut c = at(end + 16, 4)?;
    for _ in 0..at(end + 10, 2)? {
        let (method, size, nl, el, cl, off) = (at(c + 10, 2)?, at(c + 20, 4)?, at(c + 28, 2)?, at(c + 30, 2)?,
                                             at(c + 32, 2)?, at(c + 42, 4)?);
        if b.get(c + 46..c + 46 + nl).is_some_and(|n| n.ends_with(b"/data/0")) {
            ensure!(method == 0 && size % 8 == 0, "draft vocabulary: not stored int64");
            let data = off + 30 + at(off + 26, 2)? + at(off + 28, 2)?;
            let data = b.get(data..data + size).context("truncated draft vocabulary tensor")?;
            let mut seen = vec![false; vocab];
            let mut ids = Vec::with_capacity(size / 8);
            for c in data.chunks_exact(8) {
                let id = i64::from_le_bytes(c.try_into().unwrap());
                ensure!(id >= 0 && (id as usize) < vocab && !std::mem::replace(&mut seen[id as usize], true),
                        "invalid or repeated draft token {id}");
                ids.push(id as u32);
            }
            ensure!(!ids.is_empty(), "empty draft vocabulary");
            return Ok(ids);
        }
        c += 46 + nl + el + cl;
    }
    bail!("draft vocabulary: no tensor data")
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rows(cap: usize) -> Rows { Rows::new(1000, (0..32).collect(), vec![0.0;32], cap).unwrap() }

    #[test]
    fn new_tokens_wait_for_a_tile_then_append() {
        let mut r = rows(64);
        assert!(r.take(&(100..115).collect::<Vec<_>>()).is_empty());
        assert_eq!(r.take(&[100,5,115]), [2]);
        assert_eq!(r.ids.len(),48);
        assert_eq!(r.row_of[115],47);
        assert_eq!(r.score[5],1.0);
    }
    #[test]
    fn full_head_replaces_cold_rows() {
        let mut r = rows(32);
        r.take(&(0..16).collect::<Vec<_>>());
        assert_eq!(r.take(&(100..116).collect::<Vec<_>>()),[1]);
        for t in 0..16 { assert_eq!(r.row_of[t],t as u32); }
        for t in 16..32 { assert_eq!(r.row_of[t],NONE); }
        for t in 100..116 { assert_eq!(r.ids[r.row_of[t] as usize],t as u32); }
    }
    #[test]
    fn hot_rows_stay_until_decay() {
        let mut r = rows(32);
        r.take(&(0..32).collect::<Vec<_>>());
        assert!(r.take(&(100..116).collect::<Vec<_>>()).is_empty());
        r.since = DECAY-1;
        r.take(&[0]);
        assert_eq!(r.score[0],1.0);
        assert!(r.score[1] < 1.0);
        assert!(!r.take(&(200..216).collect::<Vec<_>>()).is_empty());
    }
}
