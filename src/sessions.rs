//! The server's cache of sequences. What a finished request leaves in its slot stays there for a request that
//! continues it (the next turn of a conversation, an agent's next call): that request resumes at its end, or at a
//! saved recurrent state on the prefix it shares, instead of reading the whole conversation again. When another
//! request needs the slot, or its device memory, the sequence moves to host memory (an Image: ~35 KB per position
//! and 0.16 GB of recurrent state for the 27B; PCIe 4.0 x4 on the 3090: ~6.6 GB/s) and comes back when it is
//! continued. A sequence loaded from the host keeps that copy (its shadow, below where it resumed): moved to the host
//! again, it copies only what it added since. Within a budget of host memory the shadows go first, then the oldest
//! saved states, then the sequences least recently used (read again when continued). Sessions share the memory by
//! what they hold, not by the context they may reach: the server keeps up to `max` of them (running or cached), one
//! of the whole context fits on the device (Store::new checks it), and a new conversation beyond `max` replaces the
//! session least recently used. A request never fails for memory: idle sequences go to the host or are dropped,
//! running ones give way (server.rs).
use crate::engine::{Engine, Tok};
use anyhow::{Result, ensure};
use std::time::Instant;

/// Positions per chunk of an Image's KV cache: whole chunks below a sequence's resume point stay valid
pub const CHUNK: usize = 8192;

/// `n` zeroed bytes for an Image, in huge pages where the system gives them on request (the copies from the device
/// fill fresh pages: on the 3090's VM 2.5 GB/s in 4 KB pages, 8 GB/s in 2 MB ones)
pub fn host_buf(n: usize) -> Vec<u8> {
    unsafe extern "C" {
        fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
    }
    const HUGE: usize = 2 << 20;
    // large enough to be its own mapping, not touched yet
    let mut v = vec![0u8; n];
    let a = (v.as_mut_ptr() as usize).next_multiple_of(HUGE);
    let len = (v.as_ptr() as usize + n).saturating_sub(a) / HUGE * HUGE;
    if len > 0 {
        unsafe { madvise(a as *mut _, len, 14 /* MADV_HUGEPAGE */) };
    }
    v
}

/// What a slot holds after a step at its end `len`: the KV cache of [0, len) in chunks of CHUNK positions (the
/// engine's layout, Engine::save_kv) and the recurrent state at `len` (Engine::save_state)
pub struct Image {
    pub len: usize,
    chunks: Vec<(usize, usize, Vec<u8>)>,
    state: Vec<u8>,
}

impl Image {
    pub fn bytes(&self) -> usize {
        self.chunks.iter().map(|c| c.2.len()).sum::<usize>() + self.state.len()
    }

    /// The whole chunks below `upto` (what stays valid when the slot continues from `upto`)
    fn below(mut self, upto: usize) -> Self {
        self.chunks.retain(|c| c.1 <= upto && c.1 - c.0 == CHUNK);
        self.len = self.chunks.last().map_or(0, |c| c.1);
        self.state.clear();
        self
    }
}

/// Slot `slot` after its last step, which ended at `len`, as an Image: the chunks of `base` (a copy of what the slot
/// holds below `base.len`) reused, the rest copied
pub fn save_image(m: &mut Box<dyn Engine>, slot: u32, len: usize, base: Option<Image>) -> Result<Image> {
    let mut chunks = base.map_or_else(Vec::new, |b| b.below(len).chunks);
    let mut from = chunks.last().map_or(0, |c| c.1);
    while from < len {
        let to = (from + CHUNK).min(len);
        chunks.push((from, to, m.save_kv(slot, from, to)?));
        from = to;
    }
    Ok(Image { len, chunks, state: m.save_state(slot)? })
}

/// Put an Image into slot `slot` (what it held is gone): its next step continues at `img.len`; false when the device
/// memory ran out (the slot then holds nothing)
pub fn load_image(m: &mut Box<dyn Engine>, slot: u32, img: &Image) -> Result<bool> {
    let whole = img.chunks.iter().try_fold(0, |at, c| (c.0 == at && c.1 > c.0).then_some(c.1)) == Some(img.len);
    ensure!(whole && !img.state.is_empty(), "an incomplete image");
    m.release(slot)?;
    for (a, b, d) in &img.chunks {
        if !m.load_kv(slot, *a, *b, d)? {
            m.release(slot)?;
            return Ok(false);
        }
    }
    if !m.load_state(slot, &img.state, img.len)? {
        m.release(slot)?;
        return Ok(false);
    }
    Ok(true)
}

/// What a slot holds after a request
pub struct Held {
    /// the tokens of positions [0, len): their KV cache, and the recurrent state after them
    pub tokens: Vec<u32>,
    /// the slot's state ran past the end (its last step's rejected drafts): settled before it is saved
    pub dirty: bool,
    /// recurrent states at earlier positions (Engine::save_state), oldest first, at most SNAPS: where the
    /// request's turn began (an edited last message continues there), where its reply's turn begins (the next turn
    /// renders that differently: Qwen's template drops an empty or any think block; a regenerated reply), and those
    /// of earlier turns
    pub snaps: Vec<(usize, Vec<u8>)>,
}

/// Recurrent states kept per sequence (sessions::Held::snaps), ~0.16 GB each for the 27B
pub const SNAPS: usize = 3;
/// Prompts this long get states saved (where their turn begins, where their reply's turn begins)
pub const SNAP_FROM: usize = 1024;

enum Place {
    /// in a slot, with its shadow (the host copy of its chunks below where it resumed), if it came from the host
    Slot(u32, Option<Image>),
    Host(Image),
}

struct Entry {
    id: u64,
    held: Held,
    at: Place,
    used: u64,
    /// the image of a running request that gave way (Store::keep_parked): no other prompt continues it, and it is
    /// not dropped for memory (the request has answered in part)
    parked: bool,
}

impl Entry {
    fn host_bytes(&self) -> usize {
        self.held.snaps.iter().map(|p| p.1.len()).sum::<usize>() + match &self.at {
            Place::Host(img) | Place::Slot(_, Some(img)) => img.bytes(),
            Place::Slot(_, None) => 0,
        }
    }
}

/// Where `prompt` can continue `held`: at its end (all of it starts the prompt, which goes on), else at the latest
/// saved state the prompt shares the tokens before (and goes on from)
fn resume_at(held: &Held, prompt: &[u32]) -> Option<usize> {
    let m = held.tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count();
    if m == held.tokens.len() && prompt.len() > m {
        return Some(m);
    }
    held.snaps.iter().map(|p| p.0).filter(|&p| p > 0 && p <= m && prompt.len() > p).max()
}

/// How `prompt` can use `held`: the position it starts at, and whether it takes `held` over or copies its prefix.
/// The next turn (at its end) and an edited or regenerated last turn (at its latest saved state) take it over; a
/// prompt that shares only an earlier part (another conversation with the same system prompt and tools: a state
/// saved where the first message ends) copies that much, and `held` stays cached.
fn reuse(held: &Held, prompt: &[u32]) -> Option<(usize, bool)> {
    let at = resume_at(held, prompt)?;
    let latest = held.snaps.iter().map(|p| p.0).max().unwrap_or(0);
    Some((at, at < held.tokens.len() && at < latest))
}

/// Keep at most SNAPS saved states, the first (where the first message ends: shared by other conversations) always
pub fn keep_snaps(snaps: &mut Vec<(usize, Vec<u8>)>) {
    while snaps.len() > SNAPS {
        snaps.remove(1);
    }
}

/// Store::bring_in: in a slot, waiting (no slot or device memory now), or gone (its entry dropped while a slot was
/// made: the slot is free)
enum Bring {
    In(u32),
    Wait,
    Gone(u32),
}

/// A request's slot, the position it starts at and the saved states it keeps (Held::snaps); its shadow (Place::Slot)
/// stays with the store while it runs
pub type Resume = (u32, usize, Vec<(usize, Vec<u8>)>);

pub struct Store {
    entries: Vec<Entry>,
    tick: u64,
    /// bytes of host memory the cache may take (MemAvailable at the start, less a reserve)
    budget: usize,
    /// the shadows of the running requests, by slot (back to their sequences when they finish; the first to go when
    /// host memory runs short)
    running: Vec<(u32, Image)>,
    /// most sessions (running requests and cached sequences)
    max: usize,
}

/// The longest context (a multiple of 1,024, at most `max`) of which one session fits in the engine's cache room
pub fn fitting_ctx(m: &Box<dyn Engine>, max: usize) -> usize {
    let room = m.cache_room().unwrap_or(0);
    (1..=max / 1024).rev().map(|k| k * 1024).find(|&c| m.slot_bytes(c) <= room).unwrap_or(0)
}

/// The system's MemAvailable in bytes
fn mem_available() -> Option<usize> {
    std::fs::read_to_string("/proc/meminfo").ok().and_then(|m| {
        m.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse::<usize>().ok()
    }).map(|kb| kb << 10)
}

/// Host memory left to everything else when the cache takes the rest (1.5 GiB was too little on the 3090's VM of
/// 12 GB: with the budget full, memory pressure stalled copies for seconds and the system swapped)
const HOST_RESERVE: usize = 3 << 30;
/// What the system must still have free for a copy to the host go ahead (against other processes taking memory
/// since the start; the budget plans with HOST_RESERVE)
const HOST_FLOOR: usize = 1 << 30;

impl Store {
    /// The cache of `m`'s sessions: `want` of them at most (None: one per slot), sharing device and host memory by what
    /// they hold; an error when one session of the engine's whole context does not fit on the device
    pub fn new(m: &Box<dyn Engine>, want: Option<usize>) -> Result<Self> {
        let budget = mem_available().map_or(0, |b| b.saturating_sub(HOST_RESERVE));
        let (gib, ctx, room) = (|b: usize| b as f64 / 1073741824.0, m.max_ctx(), m.cache_room().unwrap_or(0));
        ensure!(m.slot_bytes(ctx) <= room, "a session of {ctx} tokens does not fit: {:.1} GiB in a slot, the device has {:.1} \
                 GiB: --ctx={} at most (or leave --ctx out)", gib(m.slot_bytes(ctx)), gib(room), fitting_ctx(m, ctx));
        let max = want.unwrap_or(m.slots()).min(m.slots());
        eprintln!("cache: up to {max} sessions sharing {:.1} GiB of device memory and {:.1} GiB of host memory; one session \
                   up to {ctx} tokens", gib(room), gib(budget));
        Ok(Store { entries: Vec::new(), tick: 0, budget, running: Vec::new(), max })
    }

    fn touch(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    /// The shadow of the request running in slot `slot`, back from the store
    pub fn reclaim(&mut self, slot: u32) -> Option<Image> {
        let i = self.running.iter().position(|r| r.0 == slot)?;
        Some(self.running.swap_remove(i).1)
    }

    /// Slot `slot` holds `held` after a request (with the request's shadow)
    pub fn put(&mut self, slot: u32, held: Held) {
        let shadow = self.reclaim(slot);
        let used = self.touch();
        self.entries.push(Entry { id: used, held, at: Place::Slot(slot, shadow), used, parked: false });
        self.trim();
    }

    /// Whether `extra` more bytes would put the host memory over the budget: by the store's count, or by what the
    /// system has left (the count misses the rest of the process and other processes; past it the system swaps, and
    /// copies to the host crawl)
    fn over_by(&self, extra: usize) -> bool {
        self.host_used() + extra > self.budget
            || mem_available().is_some_and(|a| a < HOST_FLOOR + extra)
    }

    /// Keep the host memory within the budget: drop the shadows of the least recently used, then their oldest saved
    /// states, then the least recently used sequences on the host
    pub fn trim(&mut self) {
        self.make_room(0, None);
    }

    /// Room in host memory for `extra` more bytes (an image of a request that gives way); whether there is
    pub fn room_for(&mut self, extra: usize) -> bool {
        self.make_room(extra, None)
    }

    /// Trim until `extra` more bytes fit, sparing the entry `spare`; whether they fit
    fn make_room(&mut self, extra: usize, spare: Option<u64>) -> bool {
        while self.over_by(extra) {
            if let Some(i) = (0..self.running.len()).max_by_key(|&i| self.running[i].1.bytes()) {
                self.running.swap_remove(i);
                continue;
            }
            let others = self.entries.iter_mut().filter(|e| Some(e.id) != spare);
            if let Some(e) = others.filter(|e| matches!(e.at, Place::Slot(_, Some(_)))).min_by_key(|e| e.used) {
                if let Place::Slot(_, shadow) = &mut e.at {
                    *shadow = None;
                }
                continue;
            }
            let others = self.entries.iter_mut().filter(|e| Some(e.id) != spare);
            if let Some(e) = others.filter(|e| !e.held.snaps.is_empty()).min_by_key(|e| e.used) {
                // the first state (where the first message ends, shared by other conversations) goes last
                let k = if e.held.snaps.len() > 1 { 1 } else { 0 };
                e.held.snaps.remove(k);
                continue;
            }
            let host = self.entries.iter().enumerate()
                .filter(|(_, e)| Some(e.id) != spare && !e.parked && matches!(e.at, Place::Host(_)))
                .min_by_key(|(_, e)| e.used);
            match host {
                Some((i, _)) => {
                    eprintln!("cache: host memory full: a session of {} tokens dropped (read again when continued)",
                              self.entries[i].held.tokens.len());
                    drop(self.entries.swap_remove(i))
                }
                None => return false,
            }
        }
        true
    }

    /// Settle a dirty slot: its last token again, so the slot's state ends exactly at its sequence's end
    fn settle(m: &mut Box<dyn Engine>, slot: u32, held: &mut Held) -> Result<()> {
        if held.dirty && !held.tokens.is_empty() {
            let pos = held.tokens.len() - 1;
            m.run(&[Tok { id: held.tokens[pos], slot, pos: pos as u32 }], &[])?;
            held.dirty = false;
        }
        Ok(())
    }

    /// Whether `prompt` continues a cached sequence (takes it over; a copy needs room as a new session does)
    pub fn continues(&self, prompt: &[u32]) -> bool {
        self.entries.iter().any(|e| !e.parked && reuse(&e.held, prompt).is_some_and(|(_, copy)| !copy))
    }

    /// Move the least recently used sequence in a slot (other than `keep`) to the host; its slot, if any. `must`: when
    /// host memory has no room for it, the sequence goes (read again when continued); else it stays (None)
    pub fn park_one(&mut self, m: &mut Box<dyn Engine>, keep: Option<u32>, must: bool) -> Result<Option<u32>> {
        let Some(i) = self.entries.iter().enumerate()
            .filter(|(_, e)| matches!(e.at, Place::Slot(s, _) if Some(s) != keep))
            .min_by_key(|(_, e)| e.used).map(|(i, _)| i) else { return Ok(None) };
        let t = Instant::now();
        let (id, n) = (self.entries[i].id, self.entries[i].held.tokens.len());
        let Place::Slot(_, shadow) = &self.entries[i].at else { unreachable!() };
        let reused = shadow.as_ref().map_or(0, |s| s.chunks.iter().filter(|c| c.1 <= n && c.1 - c.0 == CHUNK).count() * CHUNK);
        // room for the image before it is copied (the host memory would otherwise peak at the budget and the image);
        // what does not fit is read again when continued
        let fits = self.make_room(m.image_bytes(n).saturating_sub(m.image_bytes(reused)), Some(id));
        if !fits && !must {
            return Ok(None);
        }
        let i = self.find(id).unwrap();
        let e = &mut self.entries[i];
        let Place::Slot(slot, shadow) = &mut e.at else { unreachable!() };
        let (slot, shadow) = (*slot, shadow.take());
        if !fits {
            eprintln!("cache: no host memory for slot {slot} ({n} tokens): its sequence is read again when continued");
        }
        let img = match m.can_save() && fits {
            true => Self::settle(m, slot, &mut e.held).and_then(|_| save_image(m, slot, n, shadow))
                .map_err(|err| eprintln!("cache: slot {slot} not saved ({err}): its sequence is read again when continued"))
                .ok(),
            false => None,
        };
        m.release(slot)?;
        match img {
            Some(img) => {
                eprintln!("cache: slot {slot} ({n} tokens, {reused} of them on the host already) to the host: {:.0} MB in {:.2} s",
                          img.bytes() as f64 / 1e6, t.elapsed().as_secs_f64());
                e.at = Place::Host(img);
                self.trim();
            }
            None => drop(self.entries.swap_remove(i)),
        }
        Ok(Some(slot))
    }

    /// Host memory the cache holds (the running requests' copies too)
    fn host_used(&self) -> usize {
        self.entries.iter().map(Entry::host_bytes).sum::<usize>()
            + self.running.iter().map(|r| r.1.bytes()).sum::<usize>()
    }

    /// Load the image of entry `id` (on the host) into the empty slot `s` while the device is short of memory: the
    /// least recently used other sequence in a slot moves to the host meanwhile, its chunks from its end down as the
    /// incoming chunks need their pages. Host memory then holds about one image at a time, not both: the incoming
    /// chunks already loaded give way to the outgoing ones (the highest first; those left are the incoming sequence's
    /// shadow). The slot the outgoing sequence leaves, or None when there is none to move.
    fn swap_in(&mut self, m: &mut Box<dyn Engine>, s: u32, id: u64) -> Result<Option<u32>> {
        let t = Instant::now();
        let Some(bi) = self.entries.iter().enumerate()
            .filter(|(_, e)| e.id != id && matches!(e.at, Place::Slot(x, _) if x != s))
            .min_by_key(|(_, e)| e.used).map(|(i, _)| i) else { return Ok(None) };
        let bid = self.entries[bi].id;
        // the outgoing sequence: settled, its state, the whole chunks its shadow holds already
        let Place::Slot(bslot, bshadow) = &mut self.entries[bi].at else { unreachable!() };
        let (bslot, bshadow) = (*bslot, bshadow.take());
        Self::settle(m, bslot, &mut self.entries[bi].held)?;
        let nb = self.entries[bi].held.tokens.len();
        let bstate = m.save_state(bslot)?;
        let mut bchunks = bshadow.map_or_else(Vec::new, |b| b.below(nb).chunks);
        let base = bchunks.last().map_or(0, |c| c.1);
        // the incoming image, its entry in slot s from now on
        let ai = self.find(id).unwrap();
        let Place::Host(mut a) = std::mem::replace(&mut self.entries[ai].at, Place::Slot(s, None)) else { unreachable!() };
        m.release(s)?;
        let local = |a: &Image, bc: &[(usize, usize, Vec<u8>)], out: &[(usize, usize, Vec<u8>)]| {
            a.bytes() + bc.iter().chain(out).map(|c| c.2.len()).sum::<usize>() + bstate.len()
        };
        let (mut top, mut out, mut i, mut b_out, mut lost) = (nb, Vec::new(), 0, false, false);
        // the outgoing sequence's highest chunk not on the host yet to the host (room made by dropping the incoming
        // chunks already loaded, the highest first); false: no room, the sequence is lost
        let copy_top = |st: &mut Self, m: &mut Box<dyn Engine>, a: &mut Image, i: usize, top: usize,
                            out: &mut Vec<(usize, usize, Vec<u8>)>| -> Result<bool> {
            let from = ((top - 1) / CHUNK * CHUNK).max(base);
            let need = (m.image_bytes(top - from) - m.image_bytes(0)).max(1);
            let fits = |st: &Self, a: &Image, out: &[(usize, usize, Vec<u8>)]| {
                st.host_used() + local(a, &bchunks, out) + need <= st.budget
                    && mem_available().is_none_or(|f| f >= HOST_FLOOR + need)
            };
            while !fits(st, a, out) && let Some(k) = (0..i).rev().find(|&k| !a.chunks[k].2.is_empty()) {
                a.chunks[k].2 = Vec::new();
            }
            // then the other copies that only save work (shadows, saved states; both sequences are in slots now)
            if !fits(st, a, out) {
                st.make_room(local(a, &bchunks, out) + need, None);
            }
            if !fits(st, a, out) {
                eprintln!("cache: host memory short of what the sessions were promised: slot {bslot}'s sequence ({nb} \
                           tokens) is read again when continued");
                return Ok(false);
            }
            out.push((from, top, m.save_kv(bslot, from, top)?));
            Ok(true)
        };
        loop {
            if i < a.chunks.len() {
                let (lo, hi) = (a.chunks[i].0, a.chunks[i].1);
                if m.load_kv(s, lo, hi, &a.chunks[i].2)? {
                    i += 1;
                    continue;
                }
            } else if m.load_state(s, &a.state, a.len)? {
                break;
            }
            // no device memory: the outgoing sequence's highest chunk to the host, its pages back
            if top > 0 {
                if top > base && !lost {
                    lost = !copy_top(self, m, &mut a, i, top, &mut out)?;
                }
                let from = (top - 1) / CHUNK * CHUNK;
                m.truncate(bslot, from)?;
                top = from;
            } else if !b_out {
                m.release(bslot)?;
                b_out = true;
            } else {
                m.release(s)?;
                anyhow::bail!("no device memory for a sequence of {} tokens in slot {s}", a.len);
            }
        }
        // the rest of the outgoing sequence (what is still in its slot and not on the host) too
        while top > base && !lost {
            let n = a.chunks.len();
            lost = !copy_top(self, m, &mut a, n, top, &mut out)?;
            top = ((top - 1) / CHUNK * CHUNK).max(base);
        }
        m.release(bslot)?;
        // the outgoing image: the shadow's chunks, then those copied (from the top down)
        out.reverse();
        bchunks.extend(out);
        let bi = self.find(bid).unwrap();
        if lost {
            drop(self.entries.swap_remove(bi));
        } else {
            self.entries[bi].at = Place::Host(Image { len: nb, chunks: bchunks, state: bstate });
        }
        // what is left of the incoming copy, from the start: its shadow
        let kept = a.chunks.iter().take_while(|c| !c.2.is_empty()).count();
        a.chunks.truncate(kept);
        a.state.clear();
        let alen = a.len;
        a.len = a.chunks.last().map_or(0, |c| c.1);
        let ai = self.find(id).unwrap();
        self.entries[ai].at = Place::Slot(s, Some(a));
        eprintln!("cache: {alen} tokens from the host into slot {s} as slot {bslot}'s {nb} go to the host ({} of them \
                   copied now): {:.2} s", nb - base, t.elapsed().as_secs_f64());
        self.trim();
        Ok(Some(bslot))
    }

    /// Entry `id` (on the host) into a slot: a free one or one an idle sequence leaves; loaded at once when the
    /// device has room, else swapped in as another sequence in a slot moves out (Store::swap_in)
    fn bring_in(&mut self, m: &mut Box<dyn Engine>, id: u64, free: &mut Vec<u32>) -> Result<Bring> {
        let s = match free.pop() {
            Some(s) => s,
            None => match self.park_one(m, None, false)? {
                Some(s) => s,
                None => return Ok(Bring::Wait),
            },
        };
        let t = Instant::now();
        let Some(i) = self.find(id) else { return Ok(Bring::Gone(s)) };
        let Place::Host(img) = &self.entries[i].at else { unreachable!() };
        if m.cache_room().is_none_or(|r| r >= m.slot_bytes(img.len)) && load_image(m, s, img)? {
            eprintln!("cache: {} tokens from the host into slot {s}: {:.0} MB in {:.2} s", img.len,
                      img.bytes() as f64 / 1e6, t.elapsed().as_secs_f64());
            let Place::Host(img) = std::mem::replace(&mut self.entries[i].at, Place::Slot(s, None)) else { unreachable!() };
            self.entries[i].at = Place::Slot(s, Some(img));
            return Ok(Bring::In(s));
        }
        // no device memory: another sequence in a slot goes to the host as this one comes, else it waits
        match self.swap_in(m, s, id)? {
            Some(b) => {
                free.push(b);
                Ok(Bring::In(s))
            }
            None => {
                free.push(s);
                Ok(Bring::Wait)
            }
        }
    }

    /// A request that gives way: what its slot held (`tokens`, `img`) kept as an entry; the id it comes back by
    pub fn keep_parked(&mut self, tokens: Vec<u32>, img: Image) -> u64 {
        let used = self.touch();
        let held = Held { tokens, dirty: false, snaps: Vec::new() };
        self.entries.push(Entry { id: used, held, at: Place::Host(img), used, parked: true });
        used
    }

    /// The request parked as `id` back into a slot (its shadow with the store while it runs); None: it waits
    pub fn bring_back(&mut self, m: &mut Box<dyn Engine>, id: u64, free: &mut Vec<u32>) -> Result<Option<u32>> {
        let s = match self.bring_in(m, id, free)? {
            Bring::In(s) => s,
            Bring::Wait => return Ok(None),
            Bring::Gone(s) => {
                free.push(s);
                anyhow::bail!("the image of a request that gave way is gone");
            }
        };
        let e = self.entries.swap_remove(self.find(id).unwrap());
        self.reclaim(s);
        if let Place::Slot(_, Some(sh)) = e.at {
            self.running.push((s, sh));
        }
        Ok(Some(s))
    }

    /// A slot for a request: a free one, else the slot of the least recently used sequence (moved to the host)
    pub fn slot(&mut self, m: &mut Box<dyn Engine>, free: &mut Vec<u32>) -> Result<Option<u32>> {
        if let Some(s) = free.pop() {
            return Ok(Some(s));
        }
        self.park_one(m, None, true)
    }

    fn find(&self, id: u64) -> Option<usize> {
        self.entries.iter().position(|e| e.id == id)
    }

    /// The request for `prompt`: the sequence it continues (in its slot, or loaded from the host), else a fresh slot
    /// at 0. None: no slot, or no device memory for the sequence (the request waits)
    /// `running`: the sessions of the running requests (and of those that gave way)
    pub fn take(&mut self, m: &mut Box<dyn Engine>, prompt: &[u32], free: &mut Vec<u32>, running: usize)
                -> Result<Option<Resume>> {
        let best = self.entries.iter().filter(|e| !e.parked)
            .filter_map(|e| reuse(&e.held, prompt).map(|(at, copy)| (e.id, at, copy, matches!(e.at, Place::Slot(..)))))
            .max_by_key(|&(_, at, _, in_slot)| (at, in_slot));
        let Some((id, at, copy, in_slot)) = best else {
            return self.fresh(m, free, running, None);
        };
        if copy {
            return self.share(m, free, running, id, at);
        }
        if !in_slot {
            match self.bring_in(m, id, free)? {
                Bring::In(_) => {}
                Bring::Wait => return Ok(None),
                // getting the slot dropped this one (host memory): then a fresh start
                Bring::Gone(s) => {
                    self.reclaim(s);
                    return Ok(Some((s, 0, Vec::new())));
                }
            }
        }
        let mut e = self.entries.swap_remove(self.find(id).unwrap());
        let Place::Slot(slot, shadow) = e.at else { unreachable!() };
        let n = e.held.tokens.len();
        if at < n {
            let state = &e.held.snaps.iter().find(|p| p.0 == at).unwrap().1;
            ensure!(m.load_state(slot, state, at)?, "no device memory for slot {slot}'s state");
            eprintln!("cache: slot {slot} resumes at a saved state ({at} of {n} tokens)");
        } else {
            eprintln!("cache: slot {slot} continues its {at} tokens");
        }
        // the states the new request keeps: those on the prefix it shares, and where its turn begins
        let mut snaps: Vec<(usize, Vec<u8>)> = e.held.snaps.drain(..).filter(|p| p.0 <= at).collect();
        if at == n && at >= SNAP_FROM {
            Self::settle(m, slot, &mut e.held)?;
            snaps.push((at, m.save_state(slot)?));
        }
        keep_snaps(&mut snaps);
        self.reclaim(slot);
        if let Some(sh) = shadow.map(|s| s.below(at)) {
            self.running.push((slot, sh));
        }
        Ok(Some((slot, at, snaps)))
    }

    /// A new session in a fresh slot at 0 (beyond `max` it replaces the least recently used other than `spare`; none
    /// cached: None, it waits)
    fn fresh(&mut self, m: &mut Box<dyn Engine>, free: &mut Vec<u32>, running: usize, spare: Option<u64>)
             -> Result<Option<Resume>> {
        while self.entries.len() + running >= self.max {
            let Some(i) = self.entries.iter().enumerate().filter(|(_, e)| !e.parked && Some(e.id) != spare)
                .min_by_key(|(_, e)| e.used).map(|(i, _)| i) else { return Ok(None) };
            if let Place::Slot(s, _) = self.entries.swap_remove(i).at {
                m.release(s)?;
                free.push(s);
            }
        }
        Ok(self.slot(m, free)?.map(|s| {
            self.reclaim(s);
            (s, 0, Vec::new())
        }))
    }

    /// A new session that shares the first `at` tokens of entry `id`: a fresh slot with a copy of their KV cache and
    /// the state saved at `at`; the entry stays cached. Without device memory for the copy: a fresh start at 0.
    fn share(&mut self, m: &mut Box<dyn Engine>, free: &mut Vec<u32>, running: usize, id: u64, at: usize)
             -> Result<Option<Resume>> {
        let t = Instant::now();
        let Some((s, ..)) = self.fresh(m, free, running, Some(id))? else { return Ok(None) };
        // getting the slot may have moved the entry to the host, or dropped it
        let Some(i) = self.find(id) else { return Ok(Some((s, 0, Vec::new()))) };
        let used = self.touch();
        let e = &mut self.entries[i];
        e.used = used;
        let snaps: Vec<(usize, Vec<u8>)> = e.held.snaps.iter().filter(|p| p.0 <= at).cloned().collect();
        let state = &snaps.iter().find(|p| p.0 == at).expect("a saved state at the shared end").1;
        let copied = match &e.at {
            Place::Slot(src, _) => {
                let mut ok = true;
                let mut from = 0;
                while ok && from < at {
                    let to = (from + CHUNK).min(at);
                    let kv = m.save_kv(*src, from, to)?;
                    ok = m.load_kv(s, from, to, &kv)?;
                    from = to;
                }
                ok
            }
            // the chunks that hold [0, at): positions past `at` in the last one are written anew from there on
            Place::Host(img) => {
                let mut ok = true;
                for (a, b, d) in img.chunks.iter().filter(|c| c.0 < at) {
                    ok = ok && m.load_kv(s, *a, *b, d)?;
                }
                ok
            }
        };
        if !(copied && m.load_state(s, state, at)?) {
            m.release(s)?;
            eprintln!("cache: no device memory to share {at} tokens with a cached session: read again");
            return Ok(Some((s, 0, Vec::new())));
        }
        eprintln!("cache: slot {s} shares the first {at} tokens of a cached session (copied in {:.2} s)", t.elapsed().as_secs_f64());
        Ok(Some((s, at, snaps)))
    }

    /// The slot holding the sequence `prompt` would continue, if one does
    pub fn slot_for(&self, prompt: &[u32]) -> Option<u32> {
        self.entries.iter().filter(|e| !e.parked).filter_map(|e| resume_at(&e.held, prompt).map(|at| (at, &e.at)))
            .max_by_key(|&(at, p)| (at, matches!(p, Place::Slot(..))))
            .and_then(|(_, p)| match p { Place::Slot(s, _) => Some(*s), Place::Host(_) => None })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn held(tokens: &[u32], pe: Option<usize>) -> Held {
        Held { tokens: tokens.to_vec(), dirty: false, snaps: pe.map(|p| (p, Vec::new())).into_iter().collect() }
    }

    #[test]
    fn continues_at_the_end_or_a_saved_state() {
        let h = held(&[1, 2, 3, 4, 5, 6], Some(3));
        assert_eq!(resume_at(&h, &[1, 2, 3, 4, 5, 6, 7, 8]), Some(6));  // the next turn
        assert_eq!(resume_at(&h, &[1, 2, 3, 9, 9]), Some(3));  // the last reply regenerated or edited
        assert_eq!(resume_at(&h, &[1, 2, 3]), None);  // nothing left to read: the last token's logits are not kept
        assert_eq!(resume_at(&h, &[1, 2, 9]), None);  // the prompt itself changed
        assert_eq!(resume_at(&h, &[1, 2, 3, 4, 5, 6]), Some(3));  // the same as held: back to the saved state
        assert_eq!(resume_at(&held(&[1, 2], None), &[1, 3]), None);
        let h = Held { tokens: vec![1, 2, 3, 4, 5, 6], dirty: false, snaps: vec![(2, vec![]), (4, vec![])] };
        assert_eq!(resume_at(&h, &[1, 2, 3, 4, 9]), Some(4));  // the latest state on the shared prefix
        assert_eq!(resume_at(&h, &[1, 2, 3, 9]), Some(2));
        assert_eq!(resume_at(&h, &[1, 2, 3, 4]), Some(2));  // a state must leave a token to read
    }

    #[test]
    fn takes_over_its_own_turns_and_copies_shared_beginnings() {
        // states where the first message ends (3) and where the last turn begins (7)
        let h = Held { tokens: (1..=10).collect(), dirty: false, snaps: vec![(3, vec![]), (7, vec![])] };
        assert_eq!(reuse(&h, &(1..=12).collect::<Vec<_>>()), Some((10, false)));  // the next turn
        assert_eq!(reuse(&h, &[1, 2, 3, 4, 5, 6, 7, 99]), Some((7, false)));  // the last turn edited
        assert_eq!(reuse(&h, &[1, 2, 3, 50, 51]), Some((3, true)));  // another conversation, same system prompt
        assert_eq!(reuse(&h, &[1, 2, 50]), None);
        let mut snaps: Vec<(usize, Vec<u8>)> = (1..=5).map(|p| (p, vec![])).collect();
        keep_snaps(&mut snaps);
        assert_eq!(snaps.iter().map(|p| p.0).collect::<Vec<_>>(), vec![1, 4, 5]);  // the first one stays
    }

    #[test]
    fn shadows_keep_whole_chunks_below_the_resume_point() {
        let c = |a: usize, b: usize| (a, b, vec![0u8; 4]);
        let img = |n| Image { len: n, chunks: vec![c(0, 8192), c(8192, 16384), c(16384, n)], state: vec![1] };
        let s = img(20000).below(17000);
        assert_eq!((s.len, s.chunks.len(), s.state.len()), (16384, 2, 0));
        assert_eq!(img(20000).below(9000).len, 8192);
        assert_eq!(img(20000).below(100).len, 0);
    }
}
