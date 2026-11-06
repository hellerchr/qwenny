//! OpenAI-compatible HTTP server. One engine thread owns the GPU: every step it feeds the last token of each decoding
//! request, plus draft tokens when few requests decode (speculative decoding: from the model's draft layer, MTP or
//! run after each step, or an n-gram chain), plus prompt chunks of new requests, up to max_tok tokens, and
//! samples the next tokens on the GPU, accepting drafts so the output follows the model's distribution exactly. A
//! decision request (crate::decide) is prefilled like a chat request, then reads the logits at its last position.
use crate::auth::{self, Auth};
use crate::chat::{Chat, OutputParser, Piece, ToolCall, split_stray_think};
use crate::download;
use crate::engine::{DraftSampling, Engine, Tok};
use crate::sample::{self, Params, Rng};
use crate::sessions::{self, Held, Image, SNAP_FROM, Store};
use anyhow::{Context, Result, bail};
use axum::{
    Json as AxJson, Router,
    extract::{Path, State},
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response, sse::Event, sse::Sse},
    routing::{get, post},
};
use serde_json::{Value as Json, json};
use std::{
    collections::VecDeque,
    convert::Infallible,
    hash::{DefaultHasher, Hash, Hasher},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio_stream::{StreamExt, wrappers::UnboundedReceiverStream};

pub struct Job {
    pub prompt: Vec<u32>,
    pub params: Params,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    /// keep going past the end-of-sequence token (benchmarks)
    pub ignore_eos: bool,
    /// a decision: no generation, the logits of these candidate tokens at the prompt's last position
    pub decide: Option<Vec<u32>>,
    /// run with the LoRA adapter (App::model), else the base model
    pub adapter: bool,
    /// what a request that restarts has already generated and sent; `prompt` then holds it too
    pub progress: Option<Box<Progress>>,
    pub tx: UnboundedSender<Msg>,
}

/// The output state of a request that gave way without room for its image in host memory
pub struct Progress {
    reply_at: usize,
    outs: Vec<u32>,
    pending: String,
    from: usize,
    read: usize,
    rng: Rng,
}

pub enum Msg {
    Text(String),
    Done {
        finish: &'static str,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    /// candidate logits and the share of the full next-token distribution on the candidates
    Decision {
        logits: Vec<f32>,
        label_mass: f64,
        prompt_tokens: usize,
    },
    Error(String),
}

struct Seq {
    job: Job,
    slot: u32,
    /// prompt tokens fed so far
    fed: usize,
    /// prompt + reply
    history: Vec<u32>,
    /// where the reply starts in `history` (the prompt's end, unless the request restarted: Progress)
    reply_at: usize,
    /// the reply as a bit per vocabulary entry, words of 32 (presence penalty: Engine::set_outputs)
    outs: Vec<u32>,
    /// decoded text held back: its end may be the start of a stop string (what was sent cannot hold one's start)
    pending: String,
    /// incremental decoding: the text so far ends with the text of reply[from..read] (from: the start of the window
    /// before the last one decoded, a character boundary; read: the tokens decoded)
    from: usize,
    read: usize,
    rng: Rng,
    finish: Option<&'static str>,
    /// history tokens already in the slot's seen set (repetition penalty)
    marked: usize,
    /// the draft layer's drafts after the last step (speculative decoding), and whether they were sampled from its
    /// distribution (kept on the device; else argmaxes)
    drafts: Vec<u32>,
    sampled: bool,
    /// the next step's drafts come from the request's own context (a chain: see CHAIN_MIN), not the draft layer
    chain: bool,
    /// history tokens whose KV the slot holds (fed and accepted); the slot's state ran past them (rejected drafts)
    valid: usize,
    dirty: bool,
    /// saved recurrent states (sessions::Held::snaps), and where the next one is saved (the start of the prompt's
    /// last turn: the reply's header, which the next turn renders differently, e.g. without an empty think block)
    snaps: Vec<(usize, Vec<u8>)>,
    snap_at: Vec<usize>,
    /// admission order (the newest request gives way when device memory runs out)
    born: u64,
    /// what its speculative decoding did (SpecStats)
    spec: SpecStats,
}

/// Per request, with --stats, on the server's log: tokens per step and acceptance per draft position
#[derive(Default)]
struct SpecStats {
    tokens: usize,
    steps: usize,
    /// per draft position: drafts offered, accepted
    offered: [usize; 16],
    accepted: [usize; 16],
}

impl SpecStats {
    /// A step that offered `drafts` tokens, the first `accepted` of them taken, and made `tokens` tokens
    fn step(&mut self, drafts: usize, accepted: usize, tokens: usize) {
        self.steps += 1;
        self.tokens += tokens;
        for j in 0..drafts.min(self.offered.len()) {
            self.offered[j] += 1;
            self.accepted[j] += (j < accepted) as usize;
        }
    }

    fn line(&self) -> String {
        let rates: Vec<String> = self
            .offered
            .iter()
            .zip(&self.accepted)
            .take_while(|(o, _)| **o > 0)
            .map(|(o, a)| format!("{:.2}", *a as f64 / *o as f64))
            .collect();
        format!(
            "SPECSTAT tokens={} steps={} per_step={:.2} accepted=[{}]",
            self.tokens,
            self.steps,
            self.tokens as f64 / self.steps.max(1) as f64,
            rates.join(" ")
        )
    }
}

/// N-gram chains next to a draft layer: a deterministic request (temperature 0 or top_k 1) whose last CHAIN_MIN or more
/// tokens (up to CHAIN_MAX compared) occurred before in its prompt or reply takes its next step's drafts from what
/// followed there, up to --spec of them, and the draft layer only takes in the step's positions (its cache stays
/// whole) without proposing; the first step without such a match goes back to the draft layer. A draft's token is
/// checked as any other, so the output stays the target's. As HyperQwen's n-gram chains: deterministic requests
/// only (under sampling a lookup draft is accepted with p(token) and would displace the draft layer's better ones).
const CHAIN_MIN: usize = 8;
const CHAIN_MAX: usize = 32;

/// The latest earlier occurrence of the longest suffix of `h` of CHAIN_MIN to CHAIN_MAX tokens with a token after it: the
/// index after that occurrence (where its continuation starts)
fn suffix_match(h: &[u32]) -> Option<usize> {
    let n = h.len();
    if n <= CHAIN_MIN {
        return None;
    }
    let mut best: Option<(usize, usize)> = None;
    // j: the index after a candidate occurrence, before the end so that a continuation follows
    for j in (CHAIN_MIN..n).rev() {
        let len = (1..=CHAIN_MAX.min(j)).take_while(|&l| h[j - l] == h[n - l]).count();
        if len >= CHAIN_MIN && best.is_none_or(|b| len > b.0) {
            best = Some((len, j));
            if len == CHAIN_MAX {
                break;
            }
        }
    }
    best.map(|b| b.1)
}

/// A chain's drafts: up to `k` tokens after the latest longest suffix match of `h` (suffix_match)
fn chain_draft(h: &[u32], k: usize) -> &[u32] {
    match suffix_match(h) {
        Some(j) if k > 0 => &h[j..(j + k).min(h.len())],
        _ => &[],
    }
}

impl Seq {
    fn reply(&self) -> &[u32] {
        &self.history[self.reply_at..]
    }

    /// The text held back goes to the client: the reply ended, and that text is not a stop string
    fn flush(&mut self) {
        if !self.pending.is_empty() {
            let _ = self.job.tx.send(Msg::Text(std::mem::take(&mut self.pending)));
        }
    }

    /// Append a sampled token, send the new text, and decide whether the request is done.
    fn accept(&mut self, tok: u32, chat: &Chat, max_ctx: usize) {
        if chat.is_eos(tok) && !self.job.ignore_eos {
            self.finish = Some("stop");
            self.flush();
            return;
        }
        self.history.push(tok);
        if let Some(w) = self.outs.get_mut(tok as usize / 32) {
            *w |= 1 << (tok % 32);
        }
        // the new text: that of the tokens from `from` on, less that of from..read (decoding the whole reply per token
        // cost the GPU up to 1 ms per step)
        let reply = self.reply();
        let (prev, cur) = (chat.decode(&reply[self.from..self.read]), chat.decode(&reply[self.from..]));
        // an incomplete UTF-8 sequence decodes to U+FFFD: wait for the next token
        if !cur.ends_with('\u{FFFD}') && cur.len() > prev.len() && cur.starts_with(&prev) {
            (self.from, self.read) = (self.read, reply.len());
            self.pending.push_str(&cur[prev.len()..]);
            // what was sent holds no start of a stop string, since the text that could be one waits here
            if let Some(end) = self.job.stop.iter().filter_map(|s| self.pending.find(s.as_str())).min() {
                self.finish = Some("stop");
                self.pending.truncate(end);
                self.flush();
            } else {
                let hold = self.job.stop.iter().map(|s| crate::chat::partial_tag(&self.pending, s)).max().unwrap_or(0);
                let held = self.pending.split_off(self.pending.len() - hold);
                self.flush();
                self.pending = held;
            }
        }
        if self.finish.is_none() && (self.reply().len() >= self.job.max_tokens || self.history.len() + 1 >= max_ctx) {
            self.finish = Some("length");
            self.flush();
        }
    }
}

/// Device memory a request needs free beyond its own to start (or resume) while others run: they grow too
const ROOM_MARGIN: usize = 1 << 30;

/// A running request gives way: what its slot holds (settled at its accepted end) to the host, the slot released
fn give_way(m: &mut Box<dyn Engine>, s: &mut Seq, shadow: Option<Image>) -> Result<Image> {
    sessions::settle(m, s.slot, &s.history[..s.valid], s.job.adapter, &mut s.dirty)?;
    let img = sessions::save_image(m, s.slot, s.valid, shadow)?;
    m.release(s.slot)?;
    Ok(img)
}

/// Slot `slot`'s next steps with the adapter or the base model (an engine without an adapter: the base)
fn use_adapter(m: &mut Box<dyn Engine>, slot: u32, adapter: bool) -> Result<()> {
    if m.has_adapter() {
        m.set_adapter(slot, adapter)
    } else if adapter {
        bail!("no adapter loaded")
    } else {
        Ok(())
    }
}

/// N-gram chains' drafts per step, unless --drafts says otherwise
const DEFAULT_DRAFTS: usize = 7;

/// Drafts per step from the draft layer: --drafts, else the layer's own default, at most what the backend can do
fn mtp_depth(has_draft: bool, default: usize, capacity: usize, drafts: Option<usize>) -> usize {
    if has_draft { drafts.unwrap_or(default).min(capacity) } else { 0 }
}

/// The engine thread's state between steps
struct Sched {
    m: Box<dyn Engine>,
    store: Option<Store>,
    chat: Arc<Chat>,
    /// draft tokens per step at most: from an n-gram chain (0: none), from the model's draft layer
    spec: usize,
    mtp: usize,
    spec_stats: bool,
    waiting: VecDeque<Job>,
    active: Vec<Seq>,
    free: Vec<u32>,
    /// the requests that gave way when device memory ran out, with what their slots held
    parked: VecDeque<(Seq, u64)>,
    /// requests that gave way before their first step (they start again when the device has room for their prompt)
    held: VecDeque<Job>,
    born: u64,
}

/// One step, as build_step made it
struct Step {
    toks: Vec<Tok>,
    /// (request, logit row of the step, draft token checked against that row), in the order of the step's logit rows
    want: Vec<(usize, usize, Option<u32>)>,
    /// per segment, the tokens committed since the request's last step
    marks: Vec<Vec<u32>>,
    /// per request in the step: (request, first index in the step, first position, tokens)
    segs: Vec<(usize, usize, usize, usize)>,
    /// each request's (fed, marked) before the step, restored when it cannot run (device memory)
    undo: Vec<(usize, usize)>,
    /// a chain's drafts per request at most in this step
    per_chain: usize,
}

fn engine(
    m: Box<dyn Engine>,
    store: Option<Store>,
    chat: Arc<Chat>,
    mut rx: UnboundedReceiver<Job>,
    spec: usize,
    mtp: usize,
) {
    let free = (0..m.slots() as u32).rev().collect();
    let mut e = Sched {
        m,
        store,
        chat,
        spec,
        mtp,
        spec_stats: crate::SPEC_STATS.load(std::sync::atomic::Ordering::Relaxed),
        free,
        waiting: VecDeque::new(),
        active: Vec::new(),
        parked: VecDeque::new(),
        held: VecDeque::new(),
        born: 0,
    };
    loop {
        if e.active.is_empty() && e.waiting.is_empty() && e.parked.is_empty() && e.held.is_empty() {
            match rx.blocking_recv() {
                Some(j) => e.waiting.push_back(j),
                None => return,
            }
        }
        while let Ok(j) = rx.try_recv() {
            e.waiting.push_back(j);
        }
        e.admit();
        if e.active.is_empty() {
            continue;
        }
        let step = e.build_step();
        match e.run_step(&step) {
            Ok(next) => {
                e.read_decisions(&step);
                let accepted = e.commit(&step, &next);
                e.draft_next(&step, &accepted);
                e.end_finished();
            }
            Err(err) => e.step_failed(&step, err),
        }
    }
}

impl Sched {
    /// Whether a request may start (or one that gave way resume) next to the running ones: the device has room for it
    fn room(&self, need: usize) -> bool {
        self.m.cache_room().is_none_or(|r| r >= need + ROOM_MARGIN)
    }

    /// Requests into slots: those that gave way first, then those that wait
    fn admit(&mut self) {
        if let Some(st) = &mut self.store {
            st.trim();
        }
        self.resume_parked();
        if self.parked.is_empty()
            && let Some(job) = self.held.front()
            && (self.active.is_empty() || self.room(self.m.slot_bytes(job.prompt.len())))
        {
            self.waiting.push_front(self.held.pop_front().unwrap());
        }
        self.start_waiting();
    }

    /// The requests that gave way come back (from the store: loaded, or swapped in for an idle sequence)
    fn resume_parked(&mut self) {
        while let Some(&(ref seq, id)) = self.parked.front() {
            // when its whole prompt fits (else it would give way again at once), or nothing else runs
            if !(self.active.is_empty() || self.room(self.m.slot_bytes(seq.job.prompt.len().max(seq.valid + 1)))) {
                break;
            }
            let Some(st) = &mut self.store else { break };
            match st.bring_back(&mut self.m, id, &mut self.free) {
                Ok(Some(slot)) => {
                    let (mut seq, _) = self.parked.pop_front().unwrap();
                    if let Err(e) = use_adapter(&mut self.m, slot, seq.job.adapter) {
                        let _ = seq.job.tx.send(Msg::Error(format!("engine: {e}")));
                        self.free.push(slot);
                        continue;
                    }
                    eprintln!("cache: a request resumes in slot {slot} at {} tokens", seq.valid);
                    (seq.slot, seq.dirty) = (slot, false);
                    seq.drafts.clear();
                    seq.chain = false;
                    self.active.push(seq);
                }
                // nothing runs that would make room: an idle sequence goes for good (does not happen while the
                // sessions fit as Store::new counted)
                Ok(None) if self.active.is_empty() => match st.park_one(&mut self.m, None, true) {
                    Ok(Some(slot)) => self.free.push(slot),
                    _ => {
                        let (seq, _) = self.parked.pop_front().unwrap();
                        let _ = seq.job.tx.send(Msg::Error("engine: no room to resume the request".into()));
                    }
                },
                Ok(None) => break,
                Err(e) => {
                    let (seq, _) = self.parked.pop_front().unwrap();
                    eprintln!("engine: a request that gave way could not resume: {e:#}");
                    let _ = seq.job.tx.send(Msg::Error(format!("engine: {e:#}")));
                }
            }
        }
    }

    fn start_waiting(&mut self) {
        while self.parked.is_empty() && self.held.is_empty() && !self.waiting.is_empty() {
            let job = self.waiting.front().unwrap();
            // a request that continues a cached session takes it in by itself (Store::take: from its slot, or
            // swapped in from host memory); a new one starts when the device has room, idle sequences moving to
            // host memory as it holds them
            let continues = self.store.as_ref().is_some_and(|st| st.continues(&job.prompt, job.adapter));
            if !(continues || self.active.is_empty() || self.room(self.m.slot_bytes(job.prompt.len()))) {
                let keep = self.store.as_ref().and_then(|st| st.slot_for(&job.prompt, job.adapter));
                match self.store.as_mut().map(|st| st.park_one(&mut self.m, keep, false)) {
                    Some(Ok(Some(slot))) => {
                        self.free.push(slot);
                        continue;
                    }
                    _ => break,
                }
            }
            if job.prompt.is_empty() || job.prompt.len() + 1 >= self.m.max_ctx() {
                let job = self.waiting.pop_front().unwrap();
                let msg =
                    format!("prompt of {} tokens does not fit the context of {}", job.prompt.len(), self.m.max_ctx());
                let _ = job.tx.send(Msg::Error(msg));
                continue;
            }
            let got = match &mut self.store {
                Some(st) => st.take(&mut self.m, &job.prompt, job.adapter, &mut self.free, self.active.len()),
                None => Ok(self.free.pop().map(|s| (s, 0, Vec::new()))),
            };
            // the slot's next steps under the request's model (its cached state, if any, is that model's)
            let got = got.and_then(|g| {
                if let Some((slot, ..)) = &g {
                    use_adapter(&mut self.m, *slot, job.adapter)?;
                }
                Ok(g)
            });
            let (slot, at, snaps) = match got {
                Ok(Some(x)) => x,
                Ok(None) => break,
                Err(e) => {
                    let job = self.waiting.pop_front().unwrap();
                    eprintln!("engine: a request ({} tokens) could not start: {e:#}", job.prompt.len());
                    let _ = job.tx.send(Msg::Error(format!("engine: {e:#}")));
                    continue;
                }
            };
            let job = self.waiting.pop_front().unwrap();
            self.start(job, slot, at, snaps);
        }
    }

    /// `job` runs in `slot`, which holds its first `at` tokens
    fn start(&mut self, mut job: Job, slot: u32, at: usize, snaps: Vec<(usize, Vec<u8>)>) {
        let history = job.prompt.clone();
        self.born += 1;
        // states saved while the prompt is read: where its first message ends (other conversations with the same
        // system prompt and tools share up to there) and where its last turn begins
        let mut snap_at = Vec::new();
        if let Some(t) = self
            .chat
            .turn_token()
            .filter(|_| self.store.is_some() && job.decide.is_none() && job.prompt.len() >= SNAP_FROM)
        {
            let starts: Vec<usize> = job.prompt.iter().enumerate().filter(|(_, x)| **x == t).map(|(i, _)| i).collect();
            snap_at.extend(starts.get(1).filter(|&&p| p >= SNAP_FROM));
            snap_at.extend(starts.last().filter(|&&p| p > 0));
            snap_at.sort_unstable();
            snap_at.dedup();
            snap_at.retain(|&p| p > at);
        }
        let Progress { reply_at, outs, pending, from, read, rng } = match job.progress.take() {
            Some(p) => *p,
            None => Progress {
                reply_at: job.prompt.len(),
                outs: vec![0u32; if job.params.presence_penalty != 0.0 { self.m.vocab() / 32 } else { 0 }],
                pending: String::new(),
                from: 0,
                read: 0,
                rng: Rng::new(job.params.seed),
            },
        };
        self.active.push(Seq {
            job,
            slot,
            fed: at,
            history,
            reply_at,
            outs,
            pending,
            from,
            read,
            rng,
            finish: None,
            marked: at,
            drafts: Vec::new(),
            sampled: false,
            chain: false,
            valid: at,
            dirty: false,
            snaps,
            snap_at,
            born: self.born,
            spec: SpecStats::default(),
        });
    }

    /// The next step: decode tokens (with drafts) first, then prompt chunks. Drafts for the decoding requests also while
    /// others read prompts (agents: one reads a tool's output while the others write): next to a prompt chunk they cost
    /// a few small draft steps
    fn build_step(&mut self) -> Step {
        let m = &self.m;
        let decoding = self.active.iter().filter(|s| s.fed == s.job.prompt.len()).count();
        let (max_tok, max_out) = (m.max_tok(), m.max_out());
        let room_each = m.spec_tok().saturating_sub(decoding).checked_div(decoding).unwrap_or(0);
        let (per, per_chain) = (room_each.min(self.mtp), room_each.min(self.spec));
        let undo = self.active.iter().map(|s| (s.fed, s.marked)).collect();
        let (mut toks, mut want, mut marks, mut segs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, s) in self.active.iter_mut().enumerate() {
            if s.fed == s.job.prompt.len() {
                let room = s
                    .job
                    .max_tokens
                    .saturating_sub(s.history.len() - s.reply_at + 1)
                    .min(m.max_ctx().saturating_sub(s.history.len() + 1));
                let d: Vec<u32> = if s.chain {
                    chain_draft(&s.history, per_chain.min(room)).to_vec()
                } else {
                    s.drafts.iter().take(per.min(room)).copied().collect()
                };
                let pos = s.history.len() - 1;
                segs.push((i, toks.len(), pos, d.len() + 1));
                for (j, &id) in std::iter::once(s.history.last().unwrap()).chain(&d).enumerate() {
                    want.push((i, toks.len(), d.get(j).copied()));
                    toks.push(Tok { id, slot: s.slot, pos: (pos + j) as u32 });
                }
                marks.push(s.history[s.marked..].to_vec());
                s.marked = s.history.len();
            }
        }
        for (i, s) in self.active.iter_mut().enumerate() {
            let left = s.job.prompt.len() - s.fed;
            if left > 0 && toks.len() < max_tok && want.len() < max_out {
                let mut n = (max_tok - toks.len()).min(left);
                // the chunk ends where a state is saved
                if let Some(&h) = s.snap_at.iter().find(|&&h| h > s.fed && h < s.fed + n) {
                    n = h - s.fed;
                }
                segs.push((i, toks.len(), s.fed, n));
                for j in s.fed..s.fed + n {
                    toks.push(Tok { id: s.job.prompt[j], slot: s.slot, pos: j as u32 });
                }
                marks.push(s.job.prompt[s.fed..s.fed + n].to_vec());
                s.fed += n;
                s.marked = s.fed;
                if s.fed == s.job.prompt.len() {
                    want.push((i, toks.len() - 1, None));
                }
            }
        }
        Step { toks, want, marks, segs, undo, per_chain }
    }

    /// The step on the device, and per logit row the token and whether its draft was accepted
    fn run_step(&mut self, step: &Step) -> Result<Vec<(u32, bool)>> {
        let rows: Vec<usize> = step.want.iter().map(|w| w.1).collect();
        self.m.step(&step.toks, &rows, Some(&step.marks))?;
        if step.want.iter().all(|&(i, ..)| self.active[i].job.params.greedy()) {
            return Ok(self.m.argmax()?.into_iter().zip(&step.want).map(|(t, w)| (t, w.2 == Some(t))).collect());
        }
        // the presence penalty's set: the request's outputs, uploaded per step (the sampler also counts the step's drafts
        // before a row); a row that ends a prompt has none, and its segment's tokens are the prompt's: no penalty there
        let mut up: Vec<usize> = step.want.iter().map(|w| w.0).collect();
        up.dedup();
        for &i in &up {
            let s = &self.active[i];
            if s.job.params.presence_penalty != 0.0 && !s.reply().is_empty() {
                self.m.set_outputs(s.slot, &s.outs)?;
            }
        }
        let p: Vec<[f32; 8]> = step
            .want
            .iter()
            .map(|&(i, _, d)| {
                let s = &mut self.active[i];
                let mut row = s.job.params.row(&mut s.rng, d);
                if s.reply().is_empty() {
                    row[7] = 0.0;
                }
                if s.sampled && !s.chain {
                    sample::mark_sampled(&mut row);
                }
                row
            })
            .collect();
        self.m.sample(&p)
    }

    /// Decisions read their row before anything else touches the logits
    fn read_decisions(&mut self, step: &Step) {
        for (j, &(i, ..)) in step.want.iter().enumerate() {
            let s = &mut self.active[i];
            let Some(cands) = &s.job.decide else { continue };
            let msg = match self.m.logits_row(j) {
                Ok(row) => {
                    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let lse = mx as f64 + row.iter().map(|&x| ((x - mx) as f64).exp()).sum::<f64>().ln();
                    let logits: Vec<f32> = cands.iter().map(|&c| row[c as usize]).collect();
                    let label_mass = logits.iter().map(|&x| (x as f64 - lse).exp()).sum();
                    Msg::Decision { logits, label_mass, prompt_tokens: s.job.prompt.len() }
                }
                Err(e) => Msg::Error(format!("engine: {e:#}")),
            };
            let _ = s.job.tx.send(msg);
            s.finish = Some("decision");
        }
    }

    /// The step did not run: device memory is made (the step is built again), or its requests end
    fn step_failed(&mut self, step: &Step, e: anyhow::Error) {
        if self.store.is_some() && e.to_string().contains("the cache is full") {
            for (s, &(fed, marked)) in self.active.iter_mut().zip(&step.undo) {
                (s.fed, s.marked) = (fed, marked);
            }
            self.make_device_room(e);
        } else {
            eprintln!(
                "engine: a step of {} tokens failed, its {} requests end: {e:#}",
                step.toks.len(),
                self.active.len()
            );
            for s in self.active.drain(..) {
                let _ = s.job.tx.send(Msg::Error(format!("engine: {e:#}")));
                let _ = self.m.release(s.slot);
                self.free.push(s.slot);
            }
        }
    }

    /// Device memory: sequences waiting in slots go to the host, else the newest request gives way
    fn make_device_room(&mut self, e: anyhow::Error) {
        let st = self.store.as_mut().unwrap();
        // a sequence in a slot to the host, or, without room there, dropped (read again when continued)
        let moved = match st.park_one(&mut self.m, None, false) {
            Ok(None) => st.park_one(&mut self.m, None, true),
            p => p,
        };
        match moved {
            Ok(Some(slot)) => self.free.push(slot),
            Ok(None) if self.active.len() > 1 => self.give_way_newest(),
            Ok(None) => {
                let s = self.active.pop().unwrap();
                eprintln!("engine: a request ({} tokens) ends: {e:#}", s.valid);
                let _ = s.job.tx.send(Msg::Error(format!("engine: {e:#}")));
                let _ = self.m.release(s.slot);
                self.free.push(s.slot);
            }
            Err(e) => eprintln!("cache: {e}"),
        }
    }

    /// The newest request gives way. With nothing in its slot yet (its memory perhaps mapped in part), or no host memory
    /// for its image (room made first: the copy would otherwise go past the budget), it starts again when its prompt fits
    fn give_way_newest(&mut self) {
        let st = self.store.as_mut().unwrap();
        let i = (0..self.active.len()).max_by_key(|&i| self.active[i].born).unwrap();
        let mut seq = self.active.swap_remove(i);
        // its shadow becomes part of its image (the others' go first when host memory is short)
        let own = st.reclaim(seq.slot);
        let fits = seq.valid > 0 && st.make_room(self.m.image_bytes(seq.valid), None);
        if !fits {
            eprintln!(
                "cache: a request ({} tokens, {} of them read) waits for device memory",
                seq.job.prompt.len(),
                seq.valid
            );
            let _ = self.m.release(seq.slot);
            self.free.push(seq.slot);
            // it reads its history again and goes on from there: what it sent stays sent
            if seq.history.len() > seq.reply_at {
                seq.job.prompt = std::mem::take(&mut seq.history);
                seq.job.progress = Some(Box::new(Progress {
                    reply_at: seq.reply_at,
                    outs: seq.outs,
                    pending: seq.pending,
                    from: seq.from,
                    read: seq.read,
                    rng: seq.rng,
                }));
            }
            self.held.push_back(seq.job);
            return;
        }
        match give_way(&mut self.m, &mut seq, own) {
            Ok(img) => {
                eprintln!("cache: a request ({} tokens) gives way in slot {}", seq.valid, seq.slot);
                self.free.push(seq.slot);
                let id = st.keep_parked(seq.history[..seq.valid].to_vec(), seq.job.adapter, img);
                self.parked.push_back((seq, id));
            }
            Err(e) => {
                eprintln!("engine: a request ({} tokens) could not give way: {e:#}", seq.valid);
                let _ = seq.job.tx.send(Msg::Error(format!("engine: {e:#}")));
                let _ = self.m.release(seq.slot);
                self.free.push(seq.slot);
            }
        }
    }

    /// The step's tokens into the requests: per request accepted drafts, then the token sampled where the first draft
    /// failed (or after the last); then what each slot holds. The drafts accepted per request.
    fn commit(&mut self, step: &Step, next: &[(u32, bool)]) -> Vec<usize> {
        let max_ctx = self.m.max_ctx();
        let mut acc = vec![0usize; self.active.len()];
        let want = &step.want;
        let mut j = 0;
        while j < want.len() {
            let (i, first) = (want[j].0, j);
            let s = &mut self.active[i];
            let (mut done, mut made) = (false, 0);
            while j < want.len() && want[j].0 == i {
                if !done && s.finish.is_none() && s.job.decide.is_none() {
                    let (tok, accepted) = next[j];
                    s.accept(tok, &self.chat, max_ctx);
                    done = !accepted;
                    acc[i] += accepted as usize;
                    made += 1;
                }
                j += 1;
            }
            if made > 0 {
                s.spec.step(j - first - 1, acc[i], made);
            }
        }
        // what each slot holds now: a decode segment's first token and its accepted drafts, a prompt chunk; the state at
        // the end of a long prompt
        for &(i, _, p0, n) in &step.segs {
            let s = &mut self.active[i];
            if p0 >= s.job.prompt.len() {
                // the slot's state ends after the segment; the history may end before: rejected drafts, an accepted
                // end-of-sequence draft (not in the history)
                s.valid = (p0 + 1 + acc[i]).min(s.history.len());
                s.dirty = s.valid != p0 + n;
            } else {
                (s.valid, s.dirty) = (p0 + n, false);
                if s.snap_at.first() == Some(&s.fed) {
                    s.snap_at.remove(0);
                    match self.m.save_state(s.slot) {
                        Ok(st) => s.snaps.push((s.fed, st)),
                        Err(e) => eprintln!("cache: {e}"),
                    }
                    sessions::keep_snaps(&mut s.snaps);
                }
            }
        }
        acc
    }

    /// The draft layer over the step's accepted positions (each with the token that followed it), then the drafts for the
    /// next step after each decoding request's last token
    fn draft_next(&mut self, step: &Step, acc: &[usize]) {
        let (mtp, per_chain) = (self.mtp, step.per_chain);
        if mtp == 0 && per_chain == 0 {
            return;
        }
        let (mut dt, mut rows, mut out, mut who) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &(i, at, p0, n) in &step.segs {
            let s = &mut self.active[i];
            s.drafts.clear();
            s.sampled = false;
            let was_chain = s.chain;
            s.chain = false;
            if s.finish.is_some() || s.job.decide.is_some() {
                continue;
            }
            for q in p0..p0 + n {
                if q + 1 >= s.history.len() {
                    break;
                }
                dt.push(Tok { id: s.history[q + 1], slot: s.slot, pos: q as u32 });
                rows.push(at + q - p0);
            }
            let last = dt.last().is_some_and(|t: &Tok| t.slot == s.slot && t.pos as usize + 2 == s.history.len());
            if s.fed == s.job.prompt.len() && last && s.history.len() + mtp < self.m.max_ctx() {
                let p = &s.job.params;
                // A full chain continues; its first miss ends it, and this normal step refreshes the evidence.
                s.chain = if was_chain {
                    n > 1 && acc[i] + 1 == n
                } else {
                    per_chain > 0 && (p.temperature <= 0.0 || p.top_k == 1) && suffix_match(&s.history).is_some()
                };
                if !s.chain {
                    out.push(dt.len() - 1);
                    who.push(i);
                }
            }
        }
        if mtp == 0 || dt.is_empty() {
            return;
        }
        // a sampled reply's drafts are drawn from the draft layer's distribution under its sampling (the presence
        // penalty's set as of now)
        let mut sampling = Vec::with_capacity(who.len());
        for &i in &who {
            let s = &mut self.active[i];
            let replying = !s.reply().is_empty();
            let p = &s.job.params;
            sampling.push(p.sampling().then(|| {
                let u = (0..mtp).map(|_| s.rng.next_f32()).collect();
                DraftSampling { row: p.draft_row(replying), u }
            }));
            if p.sampling() && p.presence_penalty != 0.0 && replying {
                let _ = self.m.set_outputs(s.slot, &s.outs);
            }
        }
        match self.m.draft(&dt, &rows, &out, mtp, &sampling) {
            Ok(d) => {
                for ((d, &i), q) in d.into_iter().zip(&who).zip(&sampling) {
                    self.active[i].drafts = d;
                    self.active[i].sampled = q.is_some();
                }
            }
            Err(e) => eprintln!("draft layer: {e}"),
        }
    }

    /// Requests that are done or whose client left: what their slots hold stays cached, or the slots go back
    fn end_finished(&mut self) {
        let mut i = 0;
        while i < self.active.len() {
            let s = &self.active[i];
            if s.finish.is_none() && !s.job.tx.is_closed() {
                i += 1;
                continue;
            }
            let s = self.active.swap_remove(i);
            if self.spec_stats {
                eprintln!("{}", s.spec.line());
            }
            match &mut self.store {
                Some(st) if s.job.decide.is_none() && s.valid > 0 => {
                    let snaps = s.snaps.into_iter().filter(|p| p.0 <= s.valid).collect();
                    st.put(
                        s.slot,
                        Held { tokens: s.history[..s.valid].to_vec(), adapter: s.job.adapter, dirty: s.dirty, snaps },
                    );
                }
                _ => {
                    if let Err(e) = self.m.release(s.slot) {
                        eprintln!("engine: releasing slot {}: {e}", s.slot);
                    }
                    self.free.push(s.slot);
                }
            }
            if let Some(finish) = s.finish.filter(|f| *f != "decision") {
                let _ = s.job.tx.send(Msg::Done {
                    finish,
                    prompt_tokens: s.reply_at,
                    completion_tokens: s.history.len() - s.reply_at,
                });
            }
        }
    }
}

/// How a loaded LoRA adapter is served
#[derive(Clone)]
pub enum AdapterUse {
    /// no adapter is loaded
    None,
    /// under this name (`"model"` of a request), beside the base model
    Named(String),
    /// always: the one served model is the base with the adapter applied, whatever a request names
    Always,
}

pub struct App {
    pub chat: Arc<Chat>,
    pub jobs: UnboundedSender<Job>,
    pub name: String,
    pub max_ctx: usize,
    pub adapter: AdapterUse,
    /// named decision tasks (question, options, threshold, ...) for the page and for requests with "preset"
    pub presets: Json,
    /// pages from the chat (`POST /v1/artifacts`), newest last
    artifacts: Mutex<VecDeque<(String, String)>>,
    /// the admin, API keys and API access mode (auth.rs)
    pub auth: Arc<Auth>,
    /// the served model's directory and the adapter's, for the page's download (download.rs)
    pub downloads: Vec<crate::download::Dir>,
}

/// Artifacts kept in memory (the page posts each one it opens; they are gone after a restart).
const ARTIFACTS: usize = 64;
/// No allow-same-origin: the page gets an opaque origin, so it cannot read the chats the UI keeps in this origin.
const SANDBOX: &str = "sandbox allow-scripts allow-forms allow-modals allow-popups allow-pointer-lock allow-downloads";

/// Keep a page (the body, HTML) under the hash of its content; the reply has its URL.
async fn put_artifact(State(app): State<Arc<App>>, html: String) -> Response {
    let mut h = DefaultHasher::new();
    html.hash(&mut h);
    let id = format!("{:016x}", h.finish());
    let mut a = app.artifacts.lock().unwrap();
    a.retain(|(k, _)| *k != id);
    a.push_back((id.clone(), html));
    if a.len() > ARTIFACTS {
        a.pop_front();
    }
    AxJson(json!({"id": id, "url": format!("/artifacts/{id}")})).into_response()
}

async fn get_artifact(State(app): State<Arc<App>>, Path(id): Path<String>) -> Response {
    let a = app.artifacts.lock().unwrap();
    match a.iter().find(|(k, _)| *k == id) {
        Some((_, html)) => (
            [
                (header::CONTENT_TYPE, "text/html; charset=utf-8"),
                (header::CONTENT_SECURITY_POLICY, SANDBOX),
                (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
            ],
            html.clone(),
        )
            .into_response(),
        None => {
            (StatusCode::NOT_FOUND, format!("No such page: the server keeps the latest {ARTIFACTS} until it restarts."))
                .into_response()
        }
    }
}

impl App {
    /// The model a request names (`"model"`): whether to use the adapter, and the name to answer with. Without an adapter
    /// or with one always applied any name is the one served model; a named adapter makes a name that is neither it nor
    /// the base an error.
    pub fn model(&self, body: &Json) -> Result<(bool, String)> {
        let asked = body.get("model").and_then(Json::as_str);
        match (&self.adapter, asked) {
            (AdapterUse::Always, _) => Ok((true, self.name.clone())),
            (AdapterUse::Named(a), Some(m)) if m == a => Ok((true, a.clone())),
            (AdapterUse::Named(a), Some(m)) if m != self.name => {
                bail!("no model {m:?}: this server serves {:?} and {a:?}", self.name)
            }
            _ => Ok((false, self.name.clone())),
        }
    }
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// An error in the form OpenAI's API has
fn api_error(status: StatusCode, kind: &str, msg: &str) -> Response {
    (status, AxJson(json!({"error": {"message": msg, "type": kind}}))).into_response()
}

pub fn bad_request(msg: String) -> Response {
    api_error(StatusCode::BAD_REQUEST, "invalid_request_error", &msg)
}

fn unauthorized(msg: &str) -> Response {
    api_error(StatusCode::UNAUTHORIZED, "invalid_api_key", msg)
}

fn db_error(e: anyhow::Error) -> Response {
    api_error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", &format!("{e:#}"))
}

/// A reply's token counts
fn usage(prompt_tokens: usize, completion_tokens: usize) -> Json {
    json!({"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens})
}

/// Sampling parameters the request leaves out: the model card's general ones, which depend on thinking (the prompt opened
/// a think block), with its presence penalty of 1.5 (against endless thinking: the card's advice for both modes).
fn sampling_defaults(thinking: bool) -> Params {
    let (temperature, top_p, top_k) = if thinking { (1.0, 0.95, 20) } else { (0.7, 0.8, 20) };
    Params { temperature, top_p, top_k, repetition_penalty: 1.0, presence_penalty: 1.5, seed: 0 }
}

/// A rendered, tokenized request; `thinking`: the prompt opened a think block, so output starts as reasoning.
struct Prepared {
    job: Job,
    /// the model's name (the base or the adapter)
    model: String,
    rx: UnboundedReceiver<Msg>,
    parser: OutputParser,
    thinking: bool,
    stream: bool,
    include_usage: bool,
}

/// The chat template's variables of a request: its chat_template_kwargs, plus what OpenAI-style clients send at the top
/// level when the kwargs leave it out: `enable_thinking`, and `reasoning_effort` (or `reasoning.effort`) in OpenAI's
/// values: none and minimal turn thinking off, high is the template's xhigh (Qwen3.8's has xhigh, medium and low and
/// raises on others), the rest goes to the template as it is.
fn template_kwargs(body: &Json) -> serde_json::Map<String, Json> {
    let mut kwargs = body.get("chat_template_kwargs").and_then(Json::as_object).cloned().unwrap_or_default();
    if let Some(t) = body.get("enable_thinking").filter(|t| t.is_boolean()) {
        kwargs.entry("enable_thinking").or_insert(t.clone());
    }
    let effort = body.get("reasoning_effort").or(body.pointer("/reasoning/effort")).and_then(Json::as_str);
    match effort {
        Some("none" | "minimal") => {
            kwargs.entry("enable_thinking").or_insert(json!(false));
        }
        Some(e) => {
            kwargs.entry("reasoning_effort").or_insert(json!(if e == "high" { "xhigh" } else { e }));
        }
        None => {}
    }
    kwargs
}

fn prepare(app: &App, body: &Json) -> Result<Prepared> {
    let (adapter, model) = app.model(body)?;
    let messages = body.get("messages").filter(|m| m.is_array()).context("messages must be an array")?;
    let tools = body.get("tools").filter(|t| t.as_array().is_some_and(|a| !a.is_empty()));
    let kwargs = template_kwargs(body);
    let prompt_text = app.chat.render(messages, tools, &kwargs)?;
    let prompt = app.chat.encode(&prompt_text)?;
    let thinking = prompt_text.trim_end().ends_with("<think>");
    let d = sampling_defaults(thinking);
    let f = |k: &str| body.get(k).and_then(Json::as_f64);
    let params = Params {
        temperature: f("temperature").map_or(d.temperature, |t| t as f32),
        top_p: f("top_p").map_or(d.top_p, |p| p as f32),
        top_k: f("top_k").map_or(d.top_k, |k| k.max(0.0) as usize),
        repetition_penalty: f("repetition_penalty").or(f("repeat_penalty")).map_or(d.repetition_penalty, |r| r as f32),
        presence_penalty: f("presence_penalty").map_or(d.presence_penalty, |p| p as f32),
        seed: body
            .get("seed")
            .and_then(Json::as_u64)
            .unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)),
    };
    let room = app.max_ctx.saturating_sub(prompt.len() + 1);
    let max_tokens = f("max_tokens").or(f("max_completion_tokens")).map_or(room, |m| (m as usize).min(room));
    let stop = match body.get("stop") {
        Some(Json::String(s)) => vec![s.clone()],
        Some(Json::Array(a)) => a.iter().filter_map(|s| s.as_str().map(String::from)).collect(),
        None | Some(Json::Null) => vec![],
        Some(_) => bail!("stop must be a string or a list of strings"),
    };
    let (tx, rx) = unbounded_channel();
    Ok(Prepared {
        job: Job {
            prompt,
            params,
            max_tokens,
            stop,
            progress: None,
            ignore_eos: body.get("ignore_eos").and_then(Json::as_bool).unwrap_or(false),
            decide: None,
            adapter,
            tx,
        },
        model,
        rx,
        parser: OutputParser::new(thinking, tools),
        thinking,
        stream: body.get("stream").and_then(Json::as_bool).unwrap_or(false),
        include_usage: body.pointer("/stream_options/include_usage").and_then(Json::as_bool).unwrap_or(false),
    })
}

fn tool_calls_json(calls: &[ToolCall], id: &str) -> Json {
    Json::Array(
        calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                json!({
                    "index": i, "id": format!("call_{id}_{i}"), "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments.to_string()}
                })
            })
            .collect(),
    )
}

async fn chat_completions(State(app): State<Arc<App>>, AxJson(body): AxJson<Json>) -> Response {
    let p = match prepare(&app, &body) {
        Ok(p) => p,
        Err(e) => return bad_request(format!("{e:#}")),
    };
    let Prepared { job, model, mut rx, mut parser, thinking, stream, include_usage } = p;
    if app.jobs.send(job).is_err() {
        return (StatusCode::SERVICE_UNAVAILABLE, "engine stopped").into_response();
    }
    let id = format!("{:x}", SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    let created = now();

    if !stream {
        let mut pieces = Vec::new();
        let (finish, counts) = loop {
            match rx.recv().await {
                Some(Msg::Text(t)) => parser.push(&t, &mut pieces),
                Some(Msg::Done { finish, prompt_tokens, completion_tokens }) => {
                    break (finish, usage(prompt_tokens, completion_tokens));
                }
                Some(Msg::Error(e)) => return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
                None | Some(Msg::Decision { .. }) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, "engine dropped the request").into_response();
                }
            }
        };
        let calls = parser.finish(&mut pieces);
        let (mut reasoning, mut content) = (String::new(), String::new());
        for p in pieces {
            match p {
                Piece::Reasoning(s) => reasoning.push_str(&s),
                Piece::Content(s) => content.push_str(&s),
            }
        }
        let mut content = content.trim_end().to_string();
        // with thinking off the model may still write `</think>` itself: what comes before it is reasoning
        if !thinking {
            let (r, c) = split_stray_think(&content);
            if !r.is_empty() {
                (reasoning, content) = (r, c);
            }
        }
        let mut message = json!({"role": "assistant", "content": content});
        if thinking || !reasoning.is_empty() {
            message["reasoning_content"] = json!(reasoning.trim());
        }
        let finish = if calls.is_empty() { finish } else { "tool_calls" };
        if !calls.is_empty() {
            message["tool_calls"] = tool_calls_json(&calls, &id);
        }
        return AxJson(json!({
            "id": format!("chatcmpl-{id}"), "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": counts
        }))
        .into_response();
    }

    let (etx, erx) = unbounded_channel::<Event>();
    tokio::spawn(async move {
        let chunk = |delta: Json, finish: Option<&str>| {
            Event::default().data(json!({
                "id": format!("chatcmpl-{id}"), "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
            }).to_string())
        };
        let send_pieces = |pieces: &mut Vec<Piece>| -> bool {
            pieces.drain(..).all(|p| {
                let delta = match p {
                    Piece::Reasoning(s) => json!({"reasoning_content": s}),
                    Piece::Content(s) => json!({"content": s}),
                };
                etx.send(chunk(delta, None)).is_ok()
            })
        };
        if etx.send(chunk(json!({"role": "assistant", "content": ""}), None)).is_err() {
            return;
        }
        let mut pieces = Vec::new();
        while let Some(msg) = rx.recv().await {
            match msg {
                Msg::Text(t) => {
                    parser.push(&t, &mut pieces);
                    if !send_pieces(&mut pieces) {
                        return; // client gone: dropping rx stops the request
                    }
                }
                Msg::Done { finish, prompt_tokens, completion_tokens } => {
                    let calls = parser.finish(&mut pieces);
                    send_pieces(&mut pieces);
                    let finish = if calls.is_empty() {
                        finish
                    } else {
                        let _ = etx.send(chunk(json!({"tool_calls": tool_calls_json(&calls, &id)}), None));
                        "tool_calls"
                    };
                    let _ = etx.send(chunk(json!({}), Some(finish)));
                    if include_usage {
                        let _ = etx.send(Event::default().data(json!({
                            "id": format!("chatcmpl-{id}"), "object": "chat.completion.chunk", "created": created,
                            "model": model, "choices": [], "usage": usage(prompt_tokens, completion_tokens)
                        }).to_string()));
                    }
                    break;
                }
                Msg::Error(e) => {
                    let _ = etx.send(Event::default().data(json!({"error": {"message": e}}).to_string()));
                    break;
                }
                Msg::Decision { .. } => break,
            }
        }
        let _ = etx.send(Event::default().data("[DONE]"));
    });
    Sse::new(UnboundedReceiverStream::new(erx).map(Ok::<_, Infallible>)).into_response()
}

pub struct ServeOpts {
    pub host: String,
    pub port: u16,
    pub name: String,
    pub adapter: AdapterUse,
    /// Drafts per step at most (speculative decoding; 0: off). None: the draft layer's default, DEFAULT_DRAFTS for chains
    pub drafts: Option<usize>,
    /// the sessions (--slots) asked for, each promised the whole context; None: as many as fit
    pub sessions: Option<usize>,
    /// decision presets (a JSON array of {"name", "question", "options", ...})
    pub presets: Json,
    /// where the page's password and the API keys are kept (auth.rs)
    pub db: std::path::PathBuf,
    /// no --ctx: the engine is loaded at the model's native context and lowered to what one session can hold
    pub fit_ctx: bool,
    /// what the page offers for download: the model's directory, then the adapter's
    pub downloads: Vec<crate::download::Dir>,
}

/// The value of cookie `name` in a request's headers
fn cookie<'a>(h: &'a axum::http::HeaderMap, name: &str) -> Option<&'a str> {
    h.get_all(header::COOKIE)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(';'))
        .find_map(|c| c.trim().strip_prefix(name)?.strip_prefix('='))
}

/// The key of `Authorization: Bearer <key>`
fn bearer(h: &axum::http::HeaderMap) -> Option<&str> {
    h.get(header::AUTHORIZATION)?.to_str().ok()?.strip_prefix("Bearer ").map(str::trim)
}

fn logged_in(app: &App, h: &axum::http::HeaderMap) -> bool {
    cookie(h, auth::COOKIE).is_some_and(|t| app.auth.session(t).unwrap_or(false))
}

/// Who may do what (auth.rs). Without a password the web page is open; with one, the page and /auth/* need its login.
/// The API (/v1/...) takes the page, and anyone else unless it requires a key (`Authorization: Bearer <key>`).
/// /health, the login page and logging in stay open.
async fn gate(State(app): State<Arc<App>>, req: axum::extract::Request, next: axum::middleware::Next) -> Response {
    let path = req.uri().path();
    if matches!(path, "/health" | "/login" | "/auth/login" | "/auth/state") || page_allowed(&app, req.headers()) {
        return next.run(req).await;
    }
    if path.starts_with("/v1/") && path != "/v1/artifacts" {
        let ok = match app.auth.require_key() {
            Ok(false) => true,
            Ok(true) => bearer(req.headers()).is_some_and(|k| app.auth.check_key(k).unwrap_or(false)),
            Err(e) => return db_error(e),
        };
        return match ok {
            true => next.run(req).await,
            false => unauthorized("this server requires an API key: Authorization: Bearer <key>"),
        };
    }
    if req.method() == axum::http::Method::GET && path == "/" {
        return axum::response::Redirect::to("/login").into_response();
    }
    unauthorized("log in first")
}

/// Whether a request may use the page: no password set, or logged in
fn page_allowed(app: &App, h: &axum::http::HeaderMap) -> bool {
    !app.auth.has_password().unwrap_or(true) || logged_in(app, h)
}

fn session_cookie(token: &str, max_age: i64) -> String {
    format!("{}={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age={max_age}", auth::COOKIE)
}

fn logged_in_reply(token: String) -> Response {
    ([(header::SET_COOKIE, session_cookie(&token, auth::SESSION_SECS))], AxJson(json!({"ok": true}))).into_response()
}

/// Run `f` on the database off the async threads: password hashes take a while
async fn with_auth<T: Send + 'static>(app: &App, f: impl FnOnce(&Auth) -> Result<T> + Send + 'static) -> Result<T> {
    let a = app.auth.clone();
    tokio::task::spawn_blocking(move || f(&a)).await?
}

/// POST /auth/login {"password"}: the session cookie (a wrong password waits a second: no fast guessing)
async fn auth_login(State(app): State<Arc<App>>, AxJson(b): AxJson<Json>) -> Response {
    let pw = b["password"].as_str().unwrap_or("").to_string();
    match with_auth(&app, move |a| a.login(&pw)).await {
        Ok(Some(token)) => logged_in_reply(token),
        Ok(None) => {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            unauthorized("wrong password")
        }
        Err(e) => db_error(e),
    }
}

/// POST /auth/setup {"password"}: set the first password (while none is set); logged in with it
async fn auth_setup(State(app): State<Arc<App>>, AxJson(b): AxJson<Json>) -> Response {
    let pw = b["password"].as_str().unwrap_or("").to_string();
    match with_auth(&app, move |a| a.setup(&pw)).await {
        Ok(Some(token)) => logged_in_reply(token),
        Ok(None) => unauthorized("a password is set already: log in"),
        Err(e) => bad_request(format!("{e:#}")),
    }
}

async fn auth_logout(State(app): State<Arc<App>>, h: axum::http::HeaderMap) -> Response {
    if let Some(t) = cookie(&h, auth::COOKIE) {
        let _ = app.auth.logout(t);
    }
    ([(header::SET_COOKIE, session_cookie("", 0))], AxJson(json!({"ok": true}))).into_response()
}

/// GET /auth/state: whether a password is set and this browser is logged in; logged in also whether the API requires
/// a key, and the keys
async fn auth_state(State(app): State<Arc<App>>, h: axum::http::HeaderMap) -> Response {
    let a = &app.auth;
    let state = || -> Result<Json> {
        let (set, logged) = (a.has_password()?, logged_in(&app, &h));
        let mut s = json!({"password_set": set, "logged_in": logged});
        if logged || !set {
            s["require_key"] = json!(a.require_key()?);
            s["keys"] = json!(a.keys()?);
        }
        Ok(s)
    };
    match state() {
        Ok(s) => AxJson(s).into_response(),
        Err(e) => db_error(e),
    }
}

/// POST /auth/require-key {"on": bool}
async fn auth_require_key(State(app): State<Arc<App>>, AxJson(b): AxJson<Json>) -> Response {
    let on = b["on"].as_bool().unwrap_or(false);
    if on && !app.auth.has_password().unwrap_or(false) {
        return bad_request("set a password first: without one, anyone who can open this page can make a key".into());
    }
    match app.auth.set_require_key(on) {
        Ok(()) => AxJson(json!({"ok": true})).into_response(),
        Err(e) => db_error(e),
    }
}

/// POST /auth/keys {"name"}: a new API key, returned once
async fn auth_new_key(State(app): State<Arc<App>>, AxJson(b): AxJson<Json>) -> Response {
    match app.auth.new_key(b["name"].as_str().unwrap_or("")) {
        Ok((id, key)) => AxJson(json!({"id": id, "key": key})).into_response(),
        Err(e) => bad_request(format!("{e:#}")),
    }
}

/// DELETE /auth/keys/{id}
async fn auth_revoke(State(app): State<Arc<App>>, Path(id): Path<i64>) -> Response {
    match app.auth.revoke(id) {
        Ok(true) => AxJson(json!({"ok": true})).into_response(),
        Ok(false) => api_error(StatusCode::NOT_FOUND, "not_found", "no such key"),
        Err(e) => db_error(e),
    }
}

/// POST /auth/password-remove {"old"}: the page is open again
async fn auth_password_remove(State(app): State<Arc<App>>, AxJson(b): AxJson<Json>) -> Response {
    let old = b["old"].as_str().unwrap_or("").to_string();
    match with_auth(&app, move |a| a.remove_password(&old)).await {
        Ok(true) => ([(header::SET_COOKIE, session_cookie("", 0))], AxJson(json!({"ok": true}))).into_response(),
        Ok(false) => unauthorized("the password is wrong"),
        Err(e) => db_error(e),
    }
}

/// POST /auth/password {"old", "new"}: other logins end, this one stays
async fn auth_password(State(app): State<Arc<App>>, h: axum::http::HeaderMap, AxJson(b): AxJson<Json>) -> Response {
    let (old, new) = (b["old"].as_str().unwrap_or("").to_string(), b["new"].as_str().unwrap_or("").to_string());
    let keep = cookie(&h, auth::COOKIE).unwrap_or("").to_string();
    match with_auth(&app, move |a| a.change_password(&old, &new, &keep)).await {
        Ok(true) => AxJson(json!({"ok": true})).into_response(),
        Ok(false) => unauthorized("the current password is wrong"),
        Err(e) => bad_request(format!("{e:#}")),
    }
}

/// Start an engine thread for `load`; its job queue and context once the model is loaded.
/// `sessions`: the slots asked for (None: as many as fit at the whole context)
/// `fit_ctx`: lower the engine's context to what one session can hold on the device (no --ctx given)
fn start_engine(
    load: impl FnOnce() -> Result<Box<dyn Engine>> + Send + 'static,
    chat: Arc<Chat>,
    drafts: Option<usize>,
    sessions: Option<usize>,
    fit_ctx: bool,
) -> Result<(UnboundedSender<Job>, usize, std::thread::JoinHandle<()>)> {
    let (jobs, rx) = unbounded_channel();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let thread = std::thread::spawn(move || {
        // finished sequences kept for the requests that continue them (engines that can save slots)
        let ready = load().and_then(|mut m| {
            if fit_ctx && m.cache_room().is_some() {
                let c = sessions::fitting_ctx(&*m, m.max_ctx());
                m.limit_ctx(c);
            }
            let capacity = m.draft_capacity().min(m.spec_tok().saturating_sub(1)).min(m.max_out().saturating_sub(1));
            let mtp = mtp_depth(m.has_draft(), m.draft_tokens(), capacity, drafts);
            m.warmup(mtp)?;
            let store = if m.can_save() { Some(Store::new(&*m, sessions)?) } else { None };
            Ok((m, store, mtp))
        });
        match ready {
            Ok((m, store, mtp)) => {
                let _ = ready_tx.send(Ok(m.max_ctx()));
                engine(m, store, chat, rx, drafts.unwrap_or(DEFAULT_DRAFTS), mtp);
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        }
    });
    let max_ctx = ready_rx.recv()??;
    Ok((jobs, max_ctx, thread))
}

/// A model answering in this process, without HTTP (qwenny abliterate): the base model is "qwenny"
pub struct Local {
    app: Arc<App>,
    engine: std::thread::JoinHandle<()>,
    db: std::path::PathBuf,
}

impl Local {
    pub fn start(
        load: impl FnOnce() -> Result<Box<dyn Engine>> + Send + 'static,
        chat: Arc<Chat>,
        adapter: AdapterUse,
        sessions: usize,
    ) -> Result<Self> {
        let (jobs, max_ctx, engine) = start_engine(load, chat.clone(), None, Some(sessions), false)?;
        let db = std::env::temp_dir().join(format!("qwenny-local-{}.db", std::process::id()));
        let app = Arc::new(App {
            chat,
            jobs,
            name: "qwenny".into(),
            max_ctx,
            adapter,
            presets: json!([]),
            artifacts: Mutex::default(),
            auth: Arc::new(Auth::open(&db)?),
            downloads: vec![],
        });
        Ok(Self { app, engine, db })
    }

    /// Every request at once; the replies in order, an error if any fails
    fn all<F: Future<Output = Response> + Send + 'static>(
        &self,
        bodies: Vec<Json>,
        f: impl Fn(State<Arc<App>>, AxJson<Json>) -> F,
    ) -> Result<Vec<Json>> {
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
        rt.block_on(async {
            let tasks: Vec<_> =
                bodies.into_iter().map(|b| tokio::spawn(f(State(self.app.clone()), AxJson(b)))).collect();
            let mut out = Vec::with_capacity(tasks.len());
            for t in tasks {
                let r = t.await?;
                let status = r.status();
                let bytes = axum::body::to_bytes(r.into_body(), usize::MAX).await?;
                anyhow::ensure!(status.is_success(), "{status}: {}", String::from_utf8_lossy(&bytes));
                out.push(serde_json::from_slice(&bytes)?);
            }
            Ok(out)
        })
    }

    pub fn chats(&self, bodies: Vec<Json>) -> Result<Vec<Json>> {
        self.all(bodies, chat_completions)
    }

    pub fn decisions(&self, bodies: Vec<Json>) -> Result<Vec<Json>> {
        self.all(bodies, crate::decide::decisions)
    }

    /// Free the device: the engine ends once nothing can send it requests
    pub fn stop(self) {
        drop(self.app);
        let _ = self.engine.join();
        let _ = std::fs::remove_file(self.db);
    }
}

/// Load the model on the engine thread, then serve until the process ends.
pub fn serve(
    load: impl FnOnce() -> Result<Box<dyn Engine>> + Send + 'static,
    chat: Chat,
    opts: ServeOpts,
) -> Result<()> {
    let auth = Arc::new(Auth::open(&opts.db)?);
    let chat = Arc::new(chat);
    let (jobs, max_ctx, _) = start_engine(load, chat.clone(), opts.drafts, opts.sessions, opts.fit_ctx)?;
    let app = Arc::new(App {
        chat,
        jobs,
        name: opts.name.clone(),
        max_ctx,
        adapter: opts.adapter.clone(),
        presets: opts.presets.clone(),
        artifacts: Mutex::default(),
        auth: auth.clone(),
        downloads: opts.downloads,
    });
    // the base model first (the page shows the first), then the adapter
    let mut names = vec![opts.name.clone()];
    if let AdapterUse::Named(a) = &opts.adapter {
        names.push(a.clone());
    }
    let router =
        Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/decisions", post(crate::decide::decisions))
            .route(
                "/v1/decisions/presets",
                get(|State(app): State<Arc<App>>| async move { AxJson(app.presets.clone()) }),
            )
            .route(
                "/v1/models",
                get(move || {
                    let data: Vec<Json> =
                        names.iter().map(|n| json!({"id": n, "object": "model", "owned_by": "qwenny"})).collect();
                    async move { AxJson(json!({"object": "list", "data": data})) }
                }),
            )
            .route("/health", get(|| async { "ok" }))
            .route("/auth/login", post(auth_login))
            .route("/auth/logout", post(auth_logout))
            .route("/auth/state", get(auth_state))
            .route("/auth/setup", post(auth_setup))
            .route("/auth/require-key", post(auth_require_key))
            .route("/auth/keys", post(auth_new_key))
            .route("/auth/keys/{id}", axum::routing::delete(auth_revoke))
            .route("/auth/password", post(auth_password))
            .route("/auth/password-remove", post(auth_password_remove))
            .route("/", get(|| async { Html(include_str!("ui.html")) }))
            .route("/login", get(|| async { Html(include_str!("login.html")) }))
            .route("/download", get(|State(app): State<Arc<App>>| async move { download::list(&app.downloads) }))
            .route(
                "/download/{i}",
                get(|State(app): State<Arc<App>>, Path(i): Path<usize>| async move {
                    download::get(&app.downloads, i, None)
                }),
            )
            .route(
                "/download/{i}/{file}",
                get(|State(app): State<Arc<App>>, Path((i, f)): Path<(usize, String)>| async move {
                    download::get(&app.downloads, i, Some(&f))
                }),
            )
            .route("/v1/artifacts", post(put_artifact))
            .route("/artifacts/{id}", get(get_artifact));
    let router = router.layer(axum::middleware::from_fn_with_state(app.clone(), gate)).with_state(app);
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind((opts.host.as_str(), opts.port)).await?;
        println!(
            "serving {}{} on http://{}:{} (context {max_ctx}, web page at /{})",
            opts.name,
            match &opts.adapter {
                AdapterUse::None => String::new(),
                AdapterUse::Named(a) => format!(" and its adapter {a}"),
                AdapterUse::Always => " (adapter applied)".to_string(),
            },
            opts.host,
            opts.port,
            if auth.require_key()? { ", the API requires a key" } else { "" }
        );
        axum::serve(listener, router).await?;
        Ok::<_, anyhow::Error>(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::test_models;

    #[test]
    fn mtp_depth_follows_drafts_within_capacity() {
        assert_eq!(mtp_depth(true, 4, 15, None), 4);
        assert_eq!(mtp_depth(true, 4, 15, Some(7)), 7);
        assert_eq!(mtp_depth(true, 4, 4, Some(7)), 4);
        assert_eq!(mtp_depth(true, 4, 15, Some(2)), 2);
        assert_eq!(mtp_depth(true, 4, 15, Some(0)), 0);
        assert_eq!(mtp_depth(false, 4, 4, Some(7)), 0);
    }

    #[test]
    fn sampling_defaults_follow_thinking() {
        let q = sampling_defaults;
        assert!(
            (q(true).temperature, q(true).top_p, q(true).top_k, q(true).repetition_penalty) == (1.0, 0.95, 20, 1.0)
        );
        assert!(
            (q(false).temperature, q(false).top_p, q(false).top_k, q(false).repetition_penalty) == (0.7, 0.8, 20, 1.0)
        );
        assert!((q(true).presence_penalty, q(false).presence_penalty) == (1.5, 1.5));
    }

    #[test]
    fn chains_continue_the_longest_latest_match() {
        // no earlier occurrence of 8 tokens: no chain
        let h: Vec<u32> = (0..40).collect();
        assert_eq!(suffix_match(&h), None);
        // the last 10 tokens occurred at 5..15: continue at 15
        let mut h: Vec<u32> = (100..140).collect();
        h.extend(105..115);
        assert_eq!(suffix_match(&h), Some(15));
        assert_eq!(chain_draft(&h, 4), &[115, 116, 117, 118]);
        assert_eq!(chain_draft(&h, 0), &[] as &[u32]);
        // 7 tokens are too few
        let mut h: Vec<u32> = (100..140).collect();
        h.extend(105..112);
        assert_eq!(chain_draft(&h, 4), &[] as &[u32]);
        // a longer match wins over a later shorter one; among equal ones the latest
        let a: Vec<u32> = (1..=12).collect();
        let mut h = a.clone();
        h.push(50);
        h.extend(&a[4..]);
        h.push(60);
        h.extend(&a);
        assert_eq!(suffix_match(&h), Some(12));
        let mut h = a.clone();
        h.push(50);
        h.extend(&a);
        h.push(60);
        h.extend(&a);
        assert_eq!(suffix_match(&h), Some(25));
        // a match ending at the last token has no continuation and is not one; the drafts stop at the history's end
        let h = vec![7u32; 9];
        assert_eq!(suffix_match(&h), Some(8));
        assert_eq!(chain_draft(&h, 4), &[7]);
    }

    #[derive(Default)]
    struct DraftStats {
        normal: std::sync::atomic::AtomicUsize,
        context: std::sync::atomic::AtomicUsize,
    }

    /// An engine that answers with a fixed token script at each output position.
    struct Scripted {
        script: Vec<u32>,
        prompt: std::cell::Cell<usize>,
        rows: Vec<usize>,
        drafts: Option<Arc<DraftStats>>,
        slots: usize,
        /// the cache is full once, when two slots decode; set then
        full_once: Option<Arc<std::sync::atomic::AtomicBool>>,
    }

    impl Engine for Scripted {
        fn slots(&self) -> usize {
            self.slots
        }
        // no host memory for an image; the slots take no device memory
        fn image_bytes(&self, _: usize) -> usize {
            1 << 60
        }
        fn slot_bytes(&self, _: usize) -> usize {
            0
        }
        fn max_ctx(&self) -> usize {
            4096
        }
        fn max_tok(&self) -> usize {
            512
        }
        fn max_out(&self) -> usize {
            8
        }
        fn spec_tok(&self) -> usize {
            8
        }
        fn vocab(&self) -> usize {
            248320
        }
        fn step(&mut self, toks: &[Tok], out: &[usize], _: Option<&[Vec<u32>]>) -> Result<()> {
            if self.prompt.get() == 0 {
                self.prompt.set(toks.last().map_or(0, |t| t.pos as usize + 1));
            }
            let decoding = |slot: u32| toks.iter().any(|t| t.slot == slot && t.pos as usize > self.prompt.get());
            if let Some(full) = &self.full_once
                && decoding(0)
                && decoding(1)
                && !full.swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                bail!("the cache is full");
            }
            self.rows = out.iter().map(|&r| (toks[r].pos as usize + 1).saturating_sub(self.prompt.get())).collect();
            Ok(())
        }
        fn logits(&self) -> Result<Vec<f32>> {
            bail!("no logits")
        }
        fn logits_row(&self, _: usize) -> Result<Vec<f32>> {
            bail!("no logits")
        }
        fn argmax(&self) -> Result<Vec<u32>> {
            Ok(self.rows.iter().map(|&i| self.script[i.min(self.script.len() - 1)]).collect())
        }
        fn sample(&mut self, _: &[[f32; 8]]) -> Result<Vec<(u32, bool)>> {
            bail!("greedy only")
        }
        fn sync(&self) -> Result<()> {
            Ok(())
        }
        fn has_draft(&self) -> bool {
            self.drafts.is_some()
        }
        fn draft(
            &mut self,
            _: &[Tok],
            _: &[usize],
            out: &[usize],
            k: usize,
            _: &[Option<DraftSampling>],
        ) -> Result<Vec<Vec<u32>>> {
            let stats = self.drafts.as_ref().unwrap();
            let count = if out.is_empty() { &stats.context } else { &stats.normal };
            count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(out.iter().map(|_| vec![u32::MAX; k]).collect())
        }
    }

    /// Send `body` to a server whose model (the one in `dir`) answers with `reply` and then `end`; the response body.
    fn serve_once(dir: &std::path::Path, reply: &str, end: u32, body: Json) -> String {
        serve_once_drafter(dir, reply, end, body, 0, None)
    }

    fn serve_once_drafter(
        dir: &std::path::Path,
        reply: &str,
        end: u32,
        mut body: Json,
        spec: usize,
        drafts: Option<Arc<DraftStats>>,
    ) -> String {
        let chat = Arc::new(Chat::load(dir).unwrap());
        let mut script = chat.encode(reply).unwrap();
        script.push(end);
        let (jobs, rx) = unbounded_channel();
        let engine_chat = chat.clone();
        std::thread::spawn(move || {
            let m =
                Scripted { script, prompt: Default::default(), rows: Vec::new(), drafts, slots: 1, full_once: None };
            let mtp = mtp_depth(m.has_draft(), m.draft_tokens(), m.draft_capacity(), Some(spec));
            engine(Box::new(m), None, engine_chat, rx, spec, mtp)
        });
        let db = std::env::temp_dir().join(format!(
            "qwenny-test-{}-{:?}.db",
            std::process::id(),
            std::thread::current().id()
        ));
        let app = Arc::new(App {
            chat,
            jobs,
            name: "t".into(),
            max_ctx: 4096,
            adapter: AdapterUse::None,
            presets: json!([]),
            artifacts: Mutex::default(),
            auth: Arc::new(Auth::open(&db).unwrap()),
            downloads: vec![],
        });
        // the scripted engine only takes the argmax: greedy, and no presence penalty (Qwen's default would sample)
        body["temperature"] = json!(0);
        body["presence_penalty"] = json!(0);
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        rt.block_on(async {
            let r = chat_completions(State(app), AxJson(body)).await;
            let bytes = axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
            String::from_utf8(bytes.to_vec()).unwrap()
        })
    }

    /// A request that gives way with no host memory for its image reads its history again and goes on: the client gets
    /// the reply once and whole, and the usage counts the prompt once.
    #[test]
    fn giving_way_without_host_room_keeps_what_was_sent() {
        let Some(dir) = test_models::dir(test_models::QWEN) else { return };
        let chat = Arc::new(Chat::load(&dir).unwrap());
        let reply = "One two three four five six seven eight nine ten eleven twelve.";
        let mut script = chat.encode(reply).unwrap();
        script.push(248046);
        let (jobs, rx) = unbounded_channel();
        let full = Arc::new(std::sync::atomic::AtomicBool::new(false));
        // the engine starts when both requests wait, so that both run
        let (start, started) = std::sync::mpsc::channel();
        let (engine_chat, full_once) = (chat.clone(), Some(full.clone()));
        std::thread::spawn(move || {
            started.recv().unwrap();
            let m: Box<dyn Engine> = Box::new(Scripted {
                script,
                prompt: Default::default(),
                rows: Vec::new(),
                drafts: None,
                slots: 2,
                full_once,
            });
            let store = Store::new(&*m, Some(2)).unwrap();
            engine(m, Some(store), engine_chat, rx, 0, 0)
        });
        let db = std::env::temp_dir().join(format!(
            "qwenny-test-{}-{:?}.db",
            std::process::id(),
            std::thread::current().id()
        ));
        let app = Arc::new(App {
            chat,
            jobs,
            name: "t".into(),
            max_ctx: 4096,
            adapter: AdapterUse::None,
            presets: json!([]),
            artifacts: Mutex::default(),
            auth: Arc::new(Auth::open(&db).unwrap()),
            downloads: vec![],
        });
        let body = json!({"messages": [{"role": "user", "content": "Count."}], "temperature": 0, "presence_penalty": 0,
                          "chat_template_kwargs": {"enable_thinking": false}});
        let ask = || async {
            let r = chat_completions(State(app.clone()), AxJson(body.clone())).await;
            serde_json::from_slice::<Json>(&axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap()).unwrap()
        };
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let (a, b, _) = rt.block_on(async { tokio::join!(ask(), ask(), async { start.send(()).unwrap() }) });
        assert!(full.load(std::sync::atomic::Ordering::Relaxed), "the step that fails did not run");
        for r in [&a, &b] {
            assert_eq!(r["choices"][0]["message"]["content"], reply, "{r}");
        }
        assert_eq!(a["usage"], b["usage"]);
    }

    /// A context chain accepts the same reply as the target script, updates the drafter's context, and skips its
    /// proposals while every chain draft hits. Its wrong normal drafts exercise the return to the drafter too.
    #[test]
    fn ngram_chains_work_beside_the_drafter() {
        let Some(dir) = test_models::dir(test_models::QWEN) else { return };
        let code = "fn report(&self) -> String {\n    self.items.iter().map(|item| item.name.clone()).collect()\n}\n"
            .repeat(2);
        let body = json!({"messages": [{"role": "user", "content": format!("Copy this code exactly:\n```rust\n{code}```")}],
                          "chat_template_kwargs": {"enable_thinking": false}});
        let plain: Json = serde_json::from_str(&serve_once(&dir, &code, 248046, body.clone())).unwrap();
        let stats = Arc::new(DraftStats::default());
        let chained: Json =
            serde_json::from_str(&serve_once_drafter(&dir, &code, 248046, body, 7, Some(stats.clone()))).unwrap();
        assert_eq!(chained["choices"], plain["choices"]);
        assert_eq!(chained["usage"], plain["usage"]);
        assert!(stats.normal.load(std::sync::atomic::Ordering::Relaxed) > 0);
        assert!(stats.context.load(std::sync::atomic::Ordering::Relaxed) > 0);
    }

    /// Qwen replies through the whole server path (prompt, parser, end tokens, response), with a scripted engine.
    #[test]
    fn stop_strings_never_leak() {
        let Some(dir) = test_models::dir(test_models::QWEN) else { return };
        let ask = |stop: &str| {
            json!({"messages": [{"role": "user", "content": "Hi"}], "stop": [stop],
                                      "chat_template_kwargs": {"enable_thinking": false}})
        };
        let content = |r: String| serde_json::from_str::<Json>(&r).unwrap()["choices"][0]["message"]["content"].clone();
        // a stop string over several tokens: none of it reaches the client
        assert_eq!(content(serve_once(&dir, "Hello there.\n\nUser: more", 248046, ask("\n\nUser:"))), "Hello there.");
        // its start, without the rest: the reply's own text
        assert_eq!(content(serve_once(&dir, "Hello there.\n\nUs", 248046, ask("\n\nUser:"))), "Hello there.\n\nUs");
    }

    #[test]
    fn qwen_replies() {
        let Some(dir) = test_models::dir(test_models::QWEN) else { return };
        let tools = json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}]);
        let ask = json!({"messages": [{"role": "user", "content": "Wetter in 80331?"}], "tools": tools});
        // thinking by default; <|im_end|> ends the reply
        let r: Json =
            serde_json::from_str(&serve_once(&dir, "Zip code.\n</think>\n\nDas Wetter ist gut.", 248046, ask.clone()))
                .unwrap();
        let m = &r["choices"][0]["message"];
        assert_eq!(
            (m["reasoning_content"].as_str(), m["content"].as_str()),
            (Some("Zip code."), Some("Das Wetter ist gut.")),
            "{r}"
        );
        assert_eq!(r["choices"][0]["finish_reason"], "stop");
        // a tool call, typed by the schema (a numeric city stays a string); <|endoftext|> ends the reply
        let call = "x\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\n80331\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        let r: Json = serde_json::from_str(&serve_once(&dir, call, 248044, ask.clone())).unwrap();
        assert_eq!(r["choices"][0]["finish_reason"], "tool_calls", "{r}");
        let f = &r["choices"][0]["message"]["tool_calls"][0]["function"];
        assert_eq!(f["name"], "get_weather");
        assert_eq!(
            serde_json::from_str::<Json>(f["arguments"].as_str().unwrap()).unwrap(),
            json!({"city": "80331", "days": 3})
        );
        // thinking off, a stray </think> in the reply
        let mut off = ask.clone();
        off["chat_template_kwargs"] = json!({"enable_thinking": false});
        let r: Json =
            serde_json::from_str(&serve_once(&dir, "oops\n</think>\n\nAntwort", 248046, off.clone())).unwrap();
        let m = &r["choices"][0]["message"];
        assert_eq!((m["reasoning_content"].as_str(), m["content"].as_str()), (Some("oops"), Some("Antwort")), "{r}");
        // streamed: reasoning, then content, then the end
        off["chat_template_kwargs"] = json!({});
        off["stream"] = json!(true);
        let sse = serve_once(&dir, "Zip.\n</think>\n\nAntwort", 248046, off);
        assert!(
            sse.contains("\"reasoning_content\":\"Zip\"")
                && sse.contains("Antwort")
                && sse.contains("\"finish_reason\":\"stop\"")
                && sse.ends_with("[DONE]\n\n"),
            "{sse}"
        );
    }

    #[test]
    fn openai_thinking_fields_reach_the_template() {
        let k = |b: Json| Json::Object(template_kwargs(&b));
        assert_eq!(k(json!({})), json!({}));
        assert_eq!(k(json!({"reasoning_effort": "low"})), json!({"reasoning_effort": "low"}));
        assert_eq!(k(json!({"reasoning_effort": "high"})), json!({"reasoning_effort": "xhigh"}));
        assert_eq!(k(json!({"reasoning": {"effort": "medium"}})), json!({"reasoning_effort": "medium"}));
        assert_eq!(k(json!({"reasoning_effort": "none"})), json!({"enable_thinking": false}));
        assert_eq!(
            k(json!({"reasoning_effort": "minimal", "enable_thinking": true})),
            json!({"enable_thinking": true})
        );
        assert_eq!(k(json!({"enable_thinking": false})), json!({"enable_thinking": false}));
        // the template's own variables win
        let b = json!({"reasoning_effort": "low", "enable_thinking": false,
                       "chat_template_kwargs": {"reasoning_effort": "xhigh", "enable_thinking": true}});
        assert_eq!(k(b), json!({"reasoning_effort": "xhigh", "enable_thinking": true}));
    }
}
