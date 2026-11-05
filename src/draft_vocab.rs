//! The tokens of the MTP's draft head, for both backends. It starts with the checkpoint's draft vocabulary
//! (`mtp_draft_vocab_ids.pt`, HyperQwen's 27B: the 40,960 tokens the model emitted most over Danish, English and code,
//! which leaves out much of other languages) and the rows a server of this model learned before (saved under
//! ~/.cache/qwenny/draft-heads). Every token a step commits (prompts and accepted replies) adds 1 to the score of its
//! row, scores halve every DECAY tokens; a token without a row waits until 16 do, then gets a new tile while there is
//! room (MAX_ROWS) and else the rows of the coldest tokens (those of score below 1, rewritten in place).
use anyhow::{Context, Result, bail, ensure};
use std::{fs, path::{Path, PathBuf}, time::{Duration, Instant}};

/// Rows of the draft head at most (HyperQwen's draft vocabulary: 40,960)
pub const MAX_ROWS: usize = 65536;
/// Committed tokens between two halvings of the scores
const DECAY: usize = 20_000;
/// Time between two saves of a changed head at least
const SAVE_EVERY: Duration = Duration::from_secs(60);
const NONE: u32 = u32::MAX;

/// Which token has which row (pure bookkeeping: the head's weights are the backend's)
#[derive(Clone, Debug)]
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
        ensure!(ids.len().is_multiple_of(16) && ids.len() <= cap && cap.is_multiple_of(16), "{} rows (at most {cap})", ids.len());
        let mut row_of = vec![NONE; vocab];
        for (r, &t) in ids.iter().enumerate() {
            ensure!((t as usize) < vocab && row_of[t as usize] == NONE, "token {t} twice or beyond the vocabulary");
            row_of[t as usize] = r as u32;
        }
        let score = if score.len() == ids.len() { score } else { vec![0.0; ids.len()] };
        Ok(Rows { ids, score, row_of, pending: Vec::new(), waiting: vec![false; vocab], since: 0, cap })
    }

    /// Whether `take(toks)` can change a tile: most steps only count, and need no snapshot to roll back to
    fn may_change(&self, toks: &[u32]) -> bool {
        let mut new = Vec::new();
        for &t in toks {
            if self.row_of.get(t as usize) == Some(&NONE) && !self.waiting[t as usize] && !new.contains(&t) {
                new.push(t);
                if self.pending.len() + new.len() >= 16 {
                    return true;
                }
            }
        }
        false
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
    let ok = ids.len() == score.len() && ids.len() <= MAX_ROWS
        && ids.iter().all(|&t| (t as usize) < vocab && !std::mem::replace(&mut seen[t as usize], true));
    if !ok {
        eprintln!("draft head: {} does not fit this model, starting from the draft vocabulary", p.display());
    }
    ok.then_some((ids, score))
}

/// The rows of a draft head as it learns, and where they are saved
pub struct Adaptive {
    rows: Rows,
    path: Option<PathBuf>,
    dirty: bool,
    saved: Instant,
}

impl Adaptive {
    /// The rows saved for the model in `dir`, if any, then those of `seed` that are new while there is room; whole tiles
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
        ensure!(!ids.is_empty(), "a draft vocabulary of less than one tile");
        Ok(Adaptive { rows: Rows::new(vocab, ids, score, MAX_ROWS)?, path, dirty: false, saved: Instant::now() })
    }

    /// The token of each row
    pub fn ids(&self) -> &[u32] {
        &self.rows.ids
    }

    /// Take in the tokens a step commits. When tiles change, `publish(ids, tiles)` puts them into the backend's head;
    /// if it fails the rows are as before, and the head must be too (or at worst hold rows the target verifies anyway).
    pub fn learn(&mut self, toks: &[u32], publish: impl FnOnce(&[u32], &[usize]) -> Result<()>) -> Result<()> {
        let before = self.rows.may_change(toks).then(|| self.rows.clone());
        let tiles = self.rows.take(toks);
        if !tiles.is_empty() {
            if let Err(e) = publish(&self.rows.ids, &tiles) {
                self.rows = before.expect("a change of tiles has its snapshot");
                return Err(e);
            }
            self.dirty = true;
        }
        if self.dirty && self.saved.elapsed() >= SAVE_EVERY {
            self.save();
        }
        Ok(())
    }

    /// Save the rows learned, through a file of this process's own (a failure costs only the next server's warm-up:
    /// reported, not fatal, and tried again)
    fn save(&mut self) {
        self.saved = Instant::now();
        let Some(path) = &self.path else { return };
        let json = serde_json::json!({ "ids": self.rows.ids, "score": self.rows.score });
        let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
        let done = path.parent().map_or(Ok(()), fs::create_dir_all)
            .and_then(|_| fs::write(&tmp, json.to_string()))
            .and_then(|_| fs::rename(&tmp, path));
        match done {
            Ok(()) => self.dirty = false,
            Err(e) => eprintln!("draft head: saving {}: {e}", path.display()),
        }
    }
}

impl Drop for Adaptive {
    fn drop(&mut self) {
        if self.dirty {
            self.save();
        }
    }
}

/// The token ids of the draft vocabulary, repeats dropped: `mtp_draft_vocab_ids.pt` in the model directory, a PyTorch
/// int64 tensor (a zip archive, stored uncompressed: the entry `*/data/0` holds the values; PyTorch puts the sizes in
/// the central directory only). None: the model has no such file.
pub fn draft_ids(dir: &Path, vocab: usize) -> Result<Option<Vec<u32>>> {
    let path = dir.join("mtp_draft_vocab_ids.pt");
    if !path.exists() {
        return Ok(None);
    }
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
            let mut seen = vec![false; vocab];
            let mut ids = Vec::with_capacity(size / 8);
            for c in v.as_chunks::<8>().0 {
                let id = i64::from_le_bytes(*c);
                ensure!(id >= 0 && (id as usize) < vocab, "{}: token id {id} beyond the vocabulary", path.display());
                if !std::mem::replace(&mut seen[id as usize], true) {
                    ids.push(id as u32);
                }
            }
            return Ok(Some(ids));
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

    #[test]
    fn may_change_exactly_when_take_can() {
        // repeats count once; a token waiting already is no new one; tokens with rows and beyond the vocabulary are none
        let mut r = rows(64);
        assert!(!r.may_change(&(100..115).chain(100..115).chain(0..32).chain([5000]).collect::<Vec<_>>()));
        assert!(r.may_change(&(100..116).collect::<Vec<_>>()));
        assert!(r.take(&(100..115).collect::<Vec<_>>()).is_empty());
        assert!(!r.may_change(&(100..115).collect::<Vec<_>>()));
        assert!(r.may_change(&[115]));
        assert!(!r.take(&[115]).is_empty());
    }

    fn adaptive() -> Adaptive {
        Adaptive { rows: rows(64), path: None, dirty: false, saved: Instant::now() }
    }

    #[test]
    fn learn_publishes_changed_tiles_and_marks_only_them_dirty() {
        let mut a = adaptive();
        // counting alone publishes nothing and leaves nothing to save
        a.learn(&[1, 2, 3], |_, _| panic!("no tile changed")).unwrap();
        assert!(!a.dirty);
        let mut got = None;
        a.learn(&(100..116).collect::<Vec<_>>(), |ids, tiles| {
            got = Some((ids.len(), tiles.to_vec()));
            Ok(())
        }).unwrap();
        assert_eq!(got, Some((48, vec![2])));
        assert!(a.dirty);
        assert_eq!(a.ids()[32..48], (100..116).collect::<Vec<_>>());
    }

    #[test]
    fn a_failed_publish_rolls_the_rows_back() {
        let mut a = adaptive();
        a.learn(&[1], |_, _| Ok(())).unwrap();
        let (score, ids) = (a.rows.score.clone(), a.ids().to_vec());
        let new: Vec<u32> = (100..116).collect();
        assert!(a.learn(&new, |_, _| bail!("no device memory")).is_err());
        assert!(!a.dirty);
        assert_eq!((a.ids(), &a.rows.score), (&ids[..], &score));
        assert!(new.iter().all(|&t| a.rows.row_of[t as usize] == NONE && !a.rows.waiting[t as usize]));
        // the same tokens again: nothing was lost
        a.learn(&new, |_, _| Ok(())).unwrap();
        assert_eq!(a.ids().len(), 48);
    }

    /// A zip archive as PyTorch writes one tensor: a stored entry `archive/data/0` with `ids` as int64
    fn pt_file(ids: &[i64]) -> Vec<u8> {
        let (name, data): (&[u8], Vec<u8>) = (b"archive/data/0", ids.iter().flat_map(|i| i.to_le_bytes()).collect());
        let mut local = [0u8; 30];
        local[..4].copy_from_slice(b"PK\x03\x04");
        local[26..28].copy_from_slice(&(name.len() as u16).to_le_bytes());
        let mut central = vec![0u8; 46];
        central[..4].copy_from_slice(b"PK\x01\x02");
        central[20..24].copy_from_slice(&(data.len() as u32).to_le_bytes());
        central[28..30].copy_from_slice(&(name.len() as u16).to_le_bytes());
        let cd = (local.len() + name.len() + data.len()) as u32;
        let mut end = vec![0u8; 22];
        end[..4].copy_from_slice(b"PK\x05\x06");
        end[10..12].copy_from_slice(&1u16.to_le_bytes());
        end[16..20].copy_from_slice(&cd.to_le_bytes());
        [&local[..], name, &data, &central, name, &end].concat()
    }

    #[test]
    fn draft_ids_drop_repeats_and_name_the_file_in_errors() {
        let dir = std::env::temp_dir().join(format!("qwenny-draft-ids-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        assert_eq!(draft_ids(&dir, 100).unwrap(), None);
        let path = dir.join("mtp_draft_vocab_ids.pt");
        fs::write(&path, pt_file(&[7, 3, 7, 9, 3])).unwrap();
        assert_eq!(draft_ids(&dir, 100).unwrap(), Some(vec![7, 3, 9]));
        for (bytes, what) in [(pt_file(&[7, 100]), "beyond the vocabulary"), (pt_file(&[-1]), "beyond the vocabulary"),
                              (b"not a zip".repeat(4), "not a zip archive")] {
            fs::write(&path, bytes).unwrap();
            let e = draft_ids(&dir, 100).unwrap_err().to_string();
            assert!(e.contains(what) && e.contains("mtp_draft_vocab_ids.pt"), "{e}");
        }
        fs::remove_dir_all(&dir).unwrap();
    }
}
