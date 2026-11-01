//! The MTP's draft head: the output layer for part of the vocabulary, in Marlin's layout. Its drafts come from those rows
//! only (argmax or drawn: sampling.cuh draft_sample), a fraction of the output layer's bytes per draft; the target still
//! checks every draft against the whole vocabulary, so the output is the same and only fewer drafts may be accepted.
//!
//! It starts with the checkpoint's draft vocabulary (`mtp_draft_vocab_ids.pt`, HyperQwen's 27B: the 40,960 tokens
//! the model emitted most over Danish, English and code, which leaves out much of other languages) and the rows a
//! server of this model learned before (saved under ~/.cache/qwenny/draft-heads). Every token a step commits
//! (prompts and accepted replies) adds 1 to the score of its row, scores halve every DECAY tokens; a token without a
//! row waits until 16 do, then gets a new tile while there is room (DRAFT_ROWS) and else the rows of the coldest
//! tokens (those of score below 1, rewritten in place).
use crate::checkpoint::{Checkpoint, Quant};
use crate::cuda::core::{Gemm, Mat};
use anyhow::{Context, Result, bail, ensure};
use cudarc::driver::{CudaSlice, CudaStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::fs;

/// Rows of the draft head at most (HyperQwen's draft vocabulary: 40,960)
const DRAFT_ROWS: usize = 65536;
/// Committed tokens between two halvings of the scores
const DECAY: usize = 20_000;
/// Time between two saves of a changed head at least
const SAVE_EVERY: Duration = Duration::from_secs(60);
const NONE: u32 = u32::MAX;

/// Which token has which row (pure bookkeeping, the GPU's part is DraftHead)
#[derive(Debug)]
struct Rows {
    /// the token of each row
    ids: Vec<u32>,
    /// the score of each row: its token's committed count, halved every DECAY tokens
    score: Vec<f32>,
    /// per vocabulary id: its row, or NONE
    row_of: Vec<u32>,
    /// ids without a row in the order they came, and which ids are in it
    pending: Vec<u32>,
    waiting: Vec<bool>,
    /// committed tokens since the last halving
    since: usize,
    cap: usize,
}

impl Rows {
    /// Rows for `ids` (with their scores; none: 0), a multiple of 16 of them at most `cap`
    fn new(vocab: usize, ids: Vec<u32>, score: Vec<f32>, cap: usize) -> Result<Self> {
        ensure!(ids.len() % 16 == 0 && ids.len() <= cap && cap % 16 == 0, "{} rows (at most {cap})", ids.len());
        let mut row_of = vec![NONE; vocab];
        for (r, &t) in ids.iter().enumerate() {
            ensure!((t as usize) < vocab && row_of[t as usize] == NONE, "token {t} twice or beyond the vocabulary");
            row_of[t as usize] = r as u32;
        }
        let score = if score.len() == ids.len() { score } else { vec![0.0; ids.len()] };
        Ok(Rows { ids, score, row_of, pending: Vec::new(), waiting: vec![false; vocab], since: 0, cap })
    }

    /// Take in committed tokens; the tiles (of 16 rows) whose tokens changed, in order
    fn take(&mut self, toks: &[u32]) -> Vec<usize> {
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
        if n == 0 {
            return Vec::new();
        }
        let new: Vec<u32> = self.pending.drain(..n).collect();
        new.iter().for_each(|&t| self.waiting[t as usize] = false);
        // new tiles while there is room, then the coldest rows (score below 1: unused since the last halvings)
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
        if k == 0 {
            return tiles;
        }
        if k < cold.len() {
            cold.select_nth_unstable_by(k, |&a, &b| self.score[a].total_cmp(&self.score[b]));
        }
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

/// The draft head on the GPU, its rows and where they are saved
pub struct DraftHead {
    pub mat: Mat,
    /// the token of each row on the device (sampled drafts map the head's columns to tokens with it)
    pub ids: CudaSlice<i32>,
    rows: Rows,
    /// the output layer as stored (the rows come from it) and its zero points
    q: Quant,
    zeros: Vec<u8>,
    path: Option<PathBuf>,
    dirty: bool,
    saved: Instant,
}

impl DraftHead {
    /// From the output layer `lm_head` (stored int4) and the draft vocabulary in `dir`, if both are there; with the
    /// rows saved for this model, if any (and room left for the vocabulary's)
    pub fn load(s: &Arc<CudaStream>, w: &Checkpoint, dir: &Path, vocab: usize) -> Result<Option<Self>> {
        if !dir.join("mtp_draft_vocab_ids.pt").exists() || !w.is_quant("lm_head") {
            return Ok(None);
        }
        let q = w.quant_matrix("lm_head")?;
        let Some(zeros) = q.zero_points() else { return Ok(None) };
        let path = saved_path(dir);
        let (mut ids, mut score) = path.as_deref().and_then(|p| read_saved(p, vocab)).unwrap_or_default();
        let mut have = vec![false; vocab];
        ids.iter().for_each(|&t| have[t as usize] = true);
        for t in draft_ids(dir, vocab)? {
            if ids.len() < DRAFT_ROWS && !have[t as usize] {
                have[t as usize] = true;
                ids.push(t);
                score.push(0.0);
            }
        }
        let whole = ids.len() / 16 * 16;
        (ids.truncate(whole), score.truncate(whole));
        let rows = Rows::new(vocab, ids, score, DRAFT_ROWS)?;
        let order: Vec<usize> = rows.ids.iter().map(|&t| t as usize).collect();
        let mat = Mat::marlin_rows(s, &q, &zeros, &order, DRAFT_ROWS)?;
        let mut ids = s.alloc_zeros(DRAFT_ROWS)?;
        s.memcpy_htod(&rows.ids.iter().map(|&t| t as i32).collect::<Vec<_>>(), &mut ids.slice_mut(0..rows.ids.len()))?;
        Ok(Some(DraftHead { mat, ids, rows, q, zeros, path, dirty: false, saved: Instant::now() }))
    }

    /// Take in the tokens a step commits; whether the head grew (the MTP's graphs launch it with its row count)
    pub fn learn(&mut self, s: &Arc<CudaStream>, mm: &Gemm, toks: &[u32]) -> Result<bool> {
        let valid = self.mat.valid;
        for t in self.rows.take(toks) {
            let order: Vec<usize> = self.rows.ids[16 * t..16 * t + 16].iter().map(|&t| t as usize).collect();
            mm.put_block(s, &mut self.mat, &self.q, &self.zeros, t, &order)?;
            let ids: Vec<i32> = order.iter().map(|&t| t as i32).collect();
            s.memcpy_htod(&ids, &mut self.ids.slice_mut(16 * t..16 * t + 16))?;
            self.dirty = true;
        }
        if self.dirty && self.saved.elapsed() >= SAVE_EVERY {
            self.save();
        }
        Ok(self.mat.valid != valid)
    }

    /// Save the rows learned (a failure costs only the next server's warm-up: reported, not fatal)
    fn save(&mut self) {
        (self.dirty, self.saved) = (false, Instant::now());
        let Some(p) = &self.path else { return };
        let r = &self.rows;
        let json = serde_json::json!({ "ids": r.ids, "score": r.score });
        let tmp = p.with_extension("tmp");
        let done = p.parent().map_or(Ok(()), fs::create_dir_all)
            .and_then(|_| fs::write(&tmp, json.to_string()))
            .and_then(|_| fs::rename(&tmp, p));
        if let Err(e) = done {
            eprintln!("draft head: saving {}: {e}", p.display());
        }
    }
}

impl Drop for DraftHead {
    fn drop(&mut self) {
        if self.dirty {
            self.save();
        }
    }
}

/// Where the rows learned for the model in `dir` are kept: $XDG_CACHE_HOME (else ~/.cache)/qwenny/draft-heads/<the
/// model directory, / as _>.json
fn saved_path(dir: &Path) -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    let dir = fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    let name = dir.to_string_lossy().trim_matches('/').replace('/', "_");
    Some(base.join("qwenny/draft-heads").join(format!("{name}.json")))
}

/// Rows saved before: their tokens and scores (unreadable, or not of this vocabulary: none)
fn read_saved(p: &Path, vocab: usize) -> Option<(Vec<u32>, Vec<f32>)> {
    let v: serde_json::Value = serde_json::from_str(&fs::read_to_string(p).ok()?).ok()?;
    let ids: Vec<u32> = serde_json::from_value(v["ids"].clone()).ok()?;
    let score: Vec<f32> = serde_json::from_value(v["score"].clone()).ok()?;
    let mut seen = vec![false; vocab];
    let ok = ids.len() == score.len() && ids.len() <= DRAFT_ROWS
        && ids.iter().all(|&t| (t as usize) < vocab && !std::mem::replace(&mut seen[t as usize], true));
    if !ok {
        eprintln!("draft head: {} does not fit this model, starting from the draft vocabulary", p.display());
    }
    ok.then_some((ids, score))
}

/// The token ids of the draft vocabulary: `mtp_draft_vocab_ids.pt` in the model directory, a PyTorch int64 tensor
/// (a zip archive, stored uncompressed: the entry `*/data/0` holds the values; PyTorch puts the sizes in the central
/// directory only)
fn draft_ids(dir: &Path, vocab: usize) -> Result<Vec<u32>> {
    let path = dir.join("mtp_draft_vocab_ids.pt");
    let b = fs::read(&path).with_context(|| format!("{}: the draft vocabulary", path.display()))?;
    let at = |o: usize, n: usize| -> Result<usize> {
        let v = b.get(o..o + n).with_context(|| format!("{}: not a zip archive", path.display()))?;
        Ok(v.iter().rev().fold(0, |a, &x| a << 8 | x as usize))
    };
    // the end of central directory record (no archive comment), then the central directory's entries
    let end = b.len().checked_sub(22).filter(|&e| b[e..e + 4] == *b"PK\x05\x06")
        .with_context(|| format!("{}: not a zip archive", path.display()))?;
    let mut c = at(end + 16, 4)?;
    for _ in 0..at(end + 10, 2)? {
        let (method, size, nl, el, cl, off) = (at(c + 10, 2)?, at(c + 20, 4)?, at(c + 28, 2)?, at(c + 30, 2)?,
                                               at(c + 32, 2)?, at(c + 42, 4)?);
        if b.get(c + 46..c + 46 + nl).is_some_and(|n| n.ends_with(b"/data/0")) {
            ensure!(method == 0 && size % 8 == 0, "{}: {size} bytes (method {method}), not int64", path.display());
            let data = off + 30 + at(off + 26, 2)? + at(off + 28, 2)?;
            let v = b.get(data..data + size).with_context(|| format!("{}: a truncated entry", path.display()))?;
            let ids: Vec<u32> = v.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as u32).collect();
            ensure!(ids.iter().all(|&t| (t as usize) < vocab), "{}: a token id beyond the vocabulary", path.display());
            return Ok(ids);
        }
        c += 46 + nl + el + cl;
    }
    bail!("{}: no tensor data (*/data/0)", path.display())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(cap: usize) -> Rows {
        Rows::new(1000, (0..32).collect(), vec![0.0; 32], cap).unwrap()
    }

    #[test]
    fn new_tokens_wait_for_a_tile_then_append() {
        let mut r = rows(64);
        assert!(r.take(&(100..115).collect::<Vec<_>>()).is_empty());
        assert_eq!(r.take(&[100, 5, 115]), [2]);  // 100 again waits once; 115 is the 16th: a third tile
        assert_eq!(r.ids.len(), 48);
        assert_eq!(r.row_of[115], 47);
        assert_eq!(r.score[5], 1.0);
    }

    #[test]
    fn full_head_replaces_the_coldest_rows() {
        let mut r = rows(32);
        let hot: Vec<u32> = (0..16).collect();
        r.take(&hot);
        // tokens 16..31 (score 0) give way, tokens 0..15 (score 1) stay
        assert_eq!(r.take(&(200..216).collect::<Vec<_>>()), [1]);
        assert!((0..16).all(|t| r.row_of[t] == t as u32));
        assert!((16..32).all(|t| r.row_of[t] == NONE));
        assert!((200..216).all(|t| r.row_of[t] != NONE && r.ids[r.row_of[t] as usize] as usize == t));
    }

    #[test]
    fn hot_rows_are_kept_and_decay() {
        let mut r = rows(32);
        let all: Vec<u32> = (0..32).collect();
        r.take(&all);
        assert!(r.take(&(200..216).collect::<Vec<_>>()).is_empty());  // every row in use: the new ones get none
        assert!(r.pending.is_empty() && (200..216).all(|t| r.row_of[t] == NONE && !r.waiting[t]));
        r.since = DECAY - 1;
        r.take(&[0]);  // the halving: rows 1..31 fall below 1
        assert_eq!(r.score[0], 1.0);
        assert!(!r.take(&(200..216).collect::<Vec<_>>()).is_empty());
        assert_eq!(r.row_of[0], 0);
        assert_eq!((0..32).filter(|&t| r.row_of[t] != NONE).count(), 16);
        assert!((200..216).all(|t| r.ids[r.row_of[t] as usize] as usize == t));
    }
}
