//! The commands that check the engine against references (tools/checks.sh runs them) and `bench`.
use crate::engine::{DraftSampling, Engine, Tok};
use crate::{Opts, chat, sample, sessions};
use anyhow::{Context, Result, ensure};
use safetensors::{Dtype, SafeTensors};
use std::{path::Path, time::Instant};

fn read(path: &str) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("reading {path}"))
}

/// One step of a pass: (slot, first position, count) per segment
type Step = Vec<(u32, usize, usize)>;

fn ints(st: &SafeTensors, name: &str) -> Result<Vec<u32>> {
    Ok(st.tensor(name)?.data().as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c) as u32).collect())
}

fn floats(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    Ok(st.tensor(name)?.data().as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

/// The prompts of a reference: those with an `ids_<p>` tensor
fn prompts<'a>(r: &'a SafeTensors<'a>) -> impl Iterator<Item = usize> + 'a {
    (0..).take_while(|p| r.tensor(&format!("ids_{p}")).is_ok())
}

/// The largest of `a` and `b`; a NaN wins, so that it fails a limit
fn worse(a: f32, b: f32) -> f32 {
    if b.is_nan() || b > a { b } else { a }
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0., worse)
}

/// The index of the largest value (the first of equals)
fn argmax(l: &[f32]) -> u32 {
    l.iter().enumerate().fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m }).0 as u32
}

/// The tokens `ids[from..to]` at their own positions in `slot`
fn toks(ids: &[u32], from: usize, to: usize, slot: u32) -> Vec<Tok> {
    (from..to).map(|i| Tok { id: ids[i], slot, pos: i as u32 }).collect()
}

/// `ids[from..to]` through `slot` in steps of 64 tokens; the logits of the last token
fn feed(m: &mut dyn Engine, ids: &[u32], from: usize, to: usize, slot: u32) -> Result<Vec<f32>> {
    for c in (from..to).step_by(64) {
        let step = toks(ids, c, (c + 64).min(to), slot);
        m.run(&step, &[step.len() - 1])?;
    }
    m.logits()
}

/// The words of a bit set over the vocabulary with `tokens` in it (what `set_outputs` takes)
fn bitset(vocab: usize, tokens: impl IntoIterator<Item = u32>) -> Vec<u32> {
    let mut words = vec![0u32; vocab / 32];
    tokens.into_iter().for_each(|t| words[t as usize / 32] |= 1 << (t % 32));
    words
}

/// The seconds `f` takes on the engine, the device synchronized before and after
fn timed(m: &mut dyn Engine, f: impl FnOnce(&mut dyn Engine) -> Result<()>) -> Result<f64> {
    m.sync()?;
    let t = Instant::now();
    f(m)?;
    m.sync()?;
    Ok(t.elapsed().as_secs_f64())
}

/// The worst (largest) of `values`, which the check passes only when it is below `limit`; there must be some
fn below(what: &str, values: impl IntoIterator<Item = f32>, limit: f32) -> Result<f32> {
    let worst = values.into_iter().reduce(worse).context(format!("{what}: nothing was compared"))?;
    ensure!(worst < limit, "{what}: {worst} is not below the limit {limit}");
    Ok(worst)
}

/// A safetensors file from (name, type, shape, little-endian bytes)
fn write_tensors(path: &Path, tensors: &[(String, Dtype, Vec<usize>, Vec<u8>)]) -> Result<()> {
    let views: Vec<_> = tensors
        .iter()
        .map(|(n, d, sh, b)| Ok((n.clone(), safetensors::tensor::TensorView::new(*d, sh.clone(), b)?)))
        .collect::<Result<_>>()?;
    Ok(safetensors::serialize_to_file(views, None, path)?)
}

/// Feed each reference prompt in chunks, compare logits and argmax at every position, then decode greedily.
/// --write=<file>: also write this engine's logits, argmax and greedy tokens as a reference of the same prompts and
/// positions (the MLX area compares MLX with CUDA through one; no --limit then). Else it fails when the mean |diff| of
/// a prompt is not below --limit.
pub fn check(opts: &Opts, dir: &str, reference: &str, chunk: usize) -> Result<()> {
    let bytes = read(reference)?;
    let r = SafeTensors::deserialize(&bytes)?;
    let write: Option<String> = opts.opt("write")?;
    let limit = if write.is_some() { None } else { Some(opts.limit()?) };
    let mut m = opts.load(dir)?;
    let (chunk, vocab) = (chunk.clamp(1, m.max_out()), m.vocab());
    // name, shape, little-endian bytes (i32 or f32)
    let mut out: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    let le_i32 = |v: &[u32]| v.iter().flat_map(|&x| (x as i32).to_le_bytes()).collect::<Vec<u8>>();
    let mut means = Vec::new();
    for p in prompts(&r) {
        let ids = ints(&r, &format!("ids_{p}"))?;
        // a reference with the top-k log-probabilities at every position (tools/make_reference_vllm.py) instead of
        // full logits at some: compared with our log-softmax on those tokens
        let topk = r.tensor(&format!("topk_ids_{p}")).is_ok();
        let (pos, ref_logits, k) = if topk {
            let k = r.tensor(&format!("topk_ids_{p}"))?.shape()[1];
            ((0..ids.len() as u32).collect(), floats(&r, &format!("topk_lp_{p}"))?, k)
        } else {
            (ints(&r, &format!("pos_{p}"))?, floats(&r, &format!("logits_{p}"))?, 0)
        };
        let topk_ids = if topk { ints(&r, &format!("topk_ids_{p}"))? } else { Vec::new() };
        let ref_argmax = ints(&r, &format!("argmax_{p}"))?;
        let ref_gen = ints(&r, &format!("gen_{p}"))?;
        let (mut agree, mut max_diff, mut sum_diff, mut compared) = (0, 0f32, 0f64, 0);
        // where the largest difference is: position, token, our logit, the reference's
        let mut worst = (0, 0, 0f32, 0f32);
        let mut next = 0;
        let (mut our_argmax, mut our_logits) = (Vec::new(), Vec::new());
        for start in (0..ids.len()).step_by(chunk) {
            let step = toks(&ids, start, (start + chunk).min(ids.len()), 0);
            m.run(&step, &(0..step.len()).collect::<Vec<_>>())?;
            let am = m.argmax()?;
            let lg = m.logits()?;
            for j in 0..step.len() {
                let at = start + j;
                agree += (am[j] == ref_argmax[at]) as usize;
                let ours = &lg[j * vocab..(j + 1) * vocab];
                if write.is_some() {
                    our_argmax.push(am[j]);
                    if !topk && pos.contains(&(at as u32)) {
                        our_logits.extend_from_slice(ours);
                    }
                }
                if topk {
                    let mx = ours.iter().fold(f32::MIN, |a, &b| a.max(b));
                    let lse = mx as f64 + ours.iter().map(|&x| ((x - mx) as f64).exp()).sum::<f64>().ln();
                    for e in 0..k {
                        let t = topk_ids[at * k + e] as usize;
                        let (a, b) = ((ours[t] as f64 - lse) as f32, ref_logits[at * k + e]);
                        let d = (a - b).abs();
                        if d > max_diff {
                            (max_diff, worst) = (d, (at, t, a, b));
                        }
                        sum_diff += d as f64;
                    }
                    compared += 1;
                } else if let Some(row) = pos.iter().position(|&q| q as usize == at) {
                    let theirs = &ref_logits[row * vocab..(row + 1) * vocab];
                    for (t, (a, b)) in ours.iter().zip(theirs).enumerate() {
                        let d = (a - b).abs();
                        if d > max_diff {
                            (max_diff, worst) = (d, (at, t, *a, *b));
                        }
                        sum_diff += d as f64;
                    }
                    compared += 1;
                }
            }
            next = am[step.len() - 1];
        }
        let mut generated = Vec::new();
        for g in 0..ref_gen.len() {
            generated.push(next);
            m.run(&[Tok { id: next, slot: 0, pos: (ids.len() + g) as u32 }], &[0])?;
            next = m.argmax()?[0];
        }
        let gen_agree = generated.iter().zip(&ref_gen).take_while(|(a, b)| a == b).count();
        if write.is_some() {
            ensure!(!topk, "--write: a reference with full logits");
            out.push((format!("ids_{p}"), Dtype::I32, vec![ids.len()], le_i32(&ids)));
            out.push((format!("pos_{p}"), Dtype::I32, vec![pos.len()], le_i32(&pos)));
            out.push((
                format!("logits_{p}"),
                Dtype::F32,
                vec![pos.len(), vocab],
                our_logits.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ));
            out.push((format!("argmax_{p}"), Dtype::I32, vec![our_argmax.len()], le_i32(&our_argmax)));
            out.push((format!("gen_{p}"), Dtype::I32, vec![generated.len()], le_i32(&generated)));
        }
        let mean = sum_diff / (compared * if topk { k } else { vocab }) as f64;
        means.push(mean as f32);
        println!(
            "prompt {p}: {} tokens, argmax agrees at {agree}/{}, {} at {compared} positions: max |diff| {max_diff:.4} \
             (position {}, token {}: {:.3} vs. {:.3}), mean {mean:.5}; greedy matches the first {gen_agree}/{} tokens",
            ids.len(),
            ids.len(),
            if topk { format!("log-probabilities of the top {k}") } else { "logits".into() },
            worst.0,
            worst.1,
            worst.2,
            worst.3,
            ref_gen.len()
        );
    }
    if let Some(path) = write {
        write_tensors(Path::new(&path), &out)?;
        println!("wrote {path}");
    }
    if let Some(limit) = limit {
        let worst = below("check: mean |diff| per prompt", means, limit)?;
        println!("mean |logit diff| at most {worst:.5} per prompt (< {limit})");
    }
    Ok(())
}

/// The draft layer (Qwen's MTP) vs. tools/make_reference_mtp.py, teacher-forced: the target over chunks of 64 tokens,
/// after each the MTP over the chunk's positions (each with the next token): its argmax at every position and its logits
/// at the reference's positions; then the second draft (the MTP on its own state and first draft) at 24 positions.
pub fn mtpcheck(opts: &Opts, dir: &str, reference: &str) -> Result<()> {
    let bytes = read(reference)?;
    let r = SafeTensors::deserialize(&bytes)?;
    let mut m = opts.load(dir)?;
    ensure!(m.has_draft(), "{dir}: no draft layer (MTP)");
    let vocab = m.vocab();
    // the target and the MTP over ids[..end], chunk by chunk; returns the MTP's argmax per position (and logits per
    // position when `logits`)
    let pass = |m: &mut Box<dyn Engine>, ids: &[u32], end: usize, logits: bool| -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
        let (mut am, mut lg) = (Vec::new(), Vec::new());
        for c0 in (0..end).step_by(64) {
            let c1 = (c0 + 64).min(end);
            m.run(&toks(ids, c0, c1, 0), &[])?;
            let next: Vec<Tok> = (c0..c1)
                .filter(|&q| q + 1 < ids.len())
                .map(|q| Tok { id: ids[q + 1], slot: 0, pos: q as u32 })
                .collect();
            let rows: Vec<usize> = (0..next.len()).collect();
            let d = m.draft(&next, &rows, &rows, 1, &[])?;
            am.extend(d.iter().map(|d| d[0]));
            if logits {
                let l = m.logits()?;
                lg.extend(l.chunks(vocab).map(|c| c.to_vec()));
            }
        }
        Ok((am, lg))
    };
    for p in prompts(&r) {
        let ids = ints(&r, &format!("ids_{p}"))?;
        let (target, want, pos, ref_lg, chain) = (
            ints(&r, &format!("target_{p}"))?,
            ints(&r, &format!("mtp_{p}"))?,
            ints(&r, &format!("pos_{p}"))?,
            floats(&r, &format!("logits_{p}"))?,
            ints(&r, &format!("chain_{p}"))?,
        );
        let (am, lg) = pass(&mut m, &ids, ids.len(), true)?;
        let agree = am.iter().zip(&want).filter(|(a, b)| a == b).count();
        let accept = am.iter().zip(&target[1..]).filter(|(a, b)| a == b).count();
        let (mut max_diff, mut sum) = (0f32, 0f64);
        for (row, &q) in pos.iter().enumerate() {
            for (a, b) in lg[q as usize].iter().zip(&ref_lg[row * vocab..(row + 1) * vocab]) {
                max_diff = max_diff.max((a - b).abs());
                sum += (a - b).abs() as f64;
            }
        }
        // the second draft at the first positions of the reference's chain
        let mut chain_agree = 0;
        for (j, &q) in pos.iter().take(chain.len()).enumerate() {
            let q = q as usize;
            pass(&mut m, &ids, q, false)?;
            m.run(&[Tok { id: ids[q], slot: 0, pos: q as u32 }], &[])?;
            let d = m.draft(&[Tok { id: ids[q + 1], slot: 0, pos: q as u32 }], &[0], &[0], 2, &[])?;
            chain_agree += (d[0][1] == chain[j]) as usize;
        }
        println!(
            "prompt {p}: {} positions, MTP argmax agrees with the reference at {agree}/{} (draft 1 = the target's \
                  next argmax at {:.1} %), logits at {} positions: max |diff| {max_diff:.4}, mean {:.5}; second draft \
                  agrees at {chain_agree}/{}",
            am.len(),
            am.len(),
            100.0 * accept as f64 / am.len() as f64,
            pos.len(),
            sum / (pos.len() * vocab) as f64,
            chain.len()
        );
    }
    Ok(())
}

/// layercheck: a slot bit marking draft tokens that the next step rejects
#[cfg(has_cuda)]
const DRAFT: u32 = 0x100;

/// One layer's mixer (in_proj, mixer, out_proj) on the reference's mixer inputs, compared with its outputs, fed in
/// several ways: (a) the prompt as MAX_TOK-token chunks then the last 4 tokens as single-token decode steps, in slot
/// 0; (b) chunks of 5 tokens (short segments) in slot 1; (c) two sequences per step, chunks of 64, in slots 2 and 3;
/// (d) a 3-token step in slot 0 while slot 1 decodes, after both had their prompts; (e) speculative decoding with
/// rejected drafts in slot 1. Fails when the largest |diff| of a way is not below --limit.
#[cfg(has_cuda)]
pub fn layercheck(opts: &Opts, dir: &str, reference: &str, layer: usize) -> Result<()> {
    let bytes = read(reference)?;
    let r = SafeTensors::deserialize(&bytes)?;
    let limit = opts.limit()?;
    let mut m = crate::cuda::qwen::LayerCheck::load(Path::new(dir), layer, opts.int8)?;
    let mut worsts = Vec::new();
    for p in prompts(&r) {
        let n = ints(&r, &format!("ids_{p}"))?.len();
        let xn = floats(&r, &format!("xn_{p}_{layer}")).with_context(|| format!("reference has no xn_{p}_{layer}"))?;
        let want = floats(&r, &format!("mix_{p}_{layer}"))?;
        let scale = want.iter().fold(0f32, |a, &b| a.max(b.abs()));
        let (h, max_tok) = (xn.len() / n, m.max_tok());
        // steps: lists of (slot, first position, count)
        let mut passes: Vec<(&str, Vec<Step>)> = Vec::new();
        let tail = 4.min(n - 1);
        let mut a: Vec<_> = (0..n - tail).step_by(max_tok).map(|s| vec![(0, s, max_tok.min(n - tail - s))]).collect();
        a.extend((n - tail..n).map(|s| vec![(0, s, 1)]));
        passes.push(("chunks + decode", a));
        passes.push(("5-token chunks", (0..n).step_by(5).map(|s| vec![(1, s, 5.min(n - s))]).collect()));
        passes.push((
            "two per step",
            (0..n).step_by(64).map(|s| vec![(2, s, 64.min(n - s)), (3, s, 64.min(n - s))]).collect(),
        ));
        if n > 8 {
            let k = n - 4;
            // the prompts in steps of at most max_tok tokens
            let mut mixed: Vec<_> = (0..k).step_by(max_tok).map(|s| vec![(0, s, max_tok.min(k - s))]).collect();
            mixed.extend((0..k + 1).step_by(max_tok).map(|s| vec![(1, s, max_tok.min(k + 1 - s))]));
            mixed.push(vec![(0, k, 3), (1, k + 1, 1)]);
            passes.push(("mixed step", mixed));
        }
        if n > 8 {
            // speculative decoding in slot 1: each step ends in drafts (slot | DRAFT: fed another position's input,
            // not compared) that are rejected, the next step restarts at the first draft. A 40-token first step,
            // then mostly 2 accepted + 4 drafts, every 7th step 1 + 15, every 50th 64 + 16 (restart at its edge).
            let l = n.min(40);
            let mut rb = vec![vec![(1, 0, l - 4), (1 | DRAFT, l - 4, 4)]];
            let mut s = l - 4;
            while s < n {
                let (good, drafts) = match rb.len() {
                    i if i % 50 == 0 => (64, 16),
                    i if i % 7 == 0 => (1, 15),
                    _ => (2, 4),
                };
                let good = good.min(n - s);
                let drafts = drafts.min(n - s - good);
                rb.push(if drafts > 0 {
                    vec![(1, s, good), (1 | DRAFT, s + good, drafts)]
                } else {
                    vec![(1, s, good)]
                });
                s += good;
            }
            passes.push(("rollback", rb));
        }
        for (name, steps) in passes {
            let (mut worst, mut sum, mut cnt) = (0f32, 0f64, 0usize);
            for step in steps {
                let toks: Vec<Tok> = step
                    .iter()
                    .flat_map(|&(slot, s, c)| {
                        (s..s + c).map(move |q| Tok { id: 0, slot: slot & !DRAFT, pos: q as u32 })
                    })
                    .collect();
                // the input row of each token: its own position, or another one for drafts
                let src: Vec<usize> = step
                    .iter()
                    .flat_map(|&(slot, s, c)| {
                        (s..s + c).map(move |q| if slot & DRAFT != 0 { (q * 7 + 3) % n } else { q })
                    })
                    .collect();
                let x: Vec<f32> = src.iter().flat_map(|&q| xn[q * h..][..h].iter().copied()).collect();
                let got = m.run(&toks, &x)?;
                // not the drafts: one whose input happens to be its own position's still follows other drafts
                let drafts = step.iter().flat_map(|&(slot, _, c)| std::iter::repeat_n(slot & DRAFT != 0, c));
                for ((i, t), _) in toks.iter().enumerate().zip(drafts).filter(|(_, draft)| !draft) {
                    for (g, w) in got[i * h..][..h].iter().zip(&want[t.pos as usize * h..][..h]) {
                        let d = (g - w).abs();
                        worst = worse(worst, d);
                        sum += d as f64;
                        cnt += 1;
                    }
                }
            }
            println!(
                "prompt {p} ({n} tokens), layer {layer}, {name}: max |diff| {worst:.4} (output max {scale:.2}), \
                      mean {:.5}",
                sum / cnt.max(1) as f64
            );
            worsts.push(worst);
        }
    }
    let worst = below(&format!("layercheck {layer}: max |diff|"), worsts.iter().copied(), limit)?;
    println!("layer {layer}: max |diff| {worst:.4} (< {limit}) in {} ways of feeding it", worsts.len());
    Ok(())
}

/// Decode `n` tokens one at a time after a short prompt; no host round trip per token. With --adapter: all of it for
/// the base model's requests (the adapter loaded), then for the adapter's.
pub fn bench(opts: &Opts, dir: &str, n: usize) -> Result<()> {
    let slots = opts.get("slots", 1)?;
    let mut m = opts.load_at(dir, opts.opt("ctx")?.unwrap_or(opts.get("pos", 0usize)? + 4096), slots)?;
    for adapter in if m.has_adapter() { vec![false, true] } else { vec![false] } {
        if m.has_adapter() {
            println!("-- {} requests (an adapter loaded)", if adapter { "adapter" } else { "base model" });
        }
        for s in 0..slots as u32 {
            m.release(s)?;
            if m.has_adapter() {
                m.set_adapter(s, adapter)?;
            }
        }
        bench_steps(opts, m.as_mut(), n)?;
    }
    Ok(())
}

/// `len` tokens (`id` of a position) from position 0 of `slot` in steps of the largest size; the last row's logits
/// when `last`
fn fill(m: &mut dyn Engine, slot: u32, len: usize, id: impl Fn(usize) -> u32, last: bool) -> Result<()> {
    for c in (0..len).step_by(m.max_tok()) {
        let step: Vec<Tok> = (c..len.min(c + m.max_tok())).map(|j| Tok { id: id(j), slot, pos: j as u32 }).collect();
        m.run(&step, &[step.len() - 1][..last as usize])?;
    }
    Ok(())
}

fn bench_steps(opts: &Opts, m: &mut dyn Engine, n: usize) -> Result<()> {
    // --pos=P: decode after a P-token context (fed in steps), not after 8 tokens
    let p0 = opts.get("pos", 0u32)?;
    if p0 > 0 {
        let dt = timed(m, |m| fill(m, 0, p0 as usize, |i| 1000 + i as u32 % 1000, false))?;
        println!("context of {p0}: {:.0} tok/s ({dt:.1} s)", p0 as f64 / dt);
    }
    let prompt: Vec<Tok> = (0..8).map(|i| Tok { id: 1000 + i, slot: 0, pos: p0 + i }).collect();
    m.run(&prompt, &[7])?;
    for warm in 0..8 {
        m.run(&[Tok { id: 42, slot: 0, pos: p0 + 8 + warm }], &[0])?;
    }
    let dt = timed(m, |m| {
        for i in 0..n {
            m.run(&[Tok { id: 42, slot: 0, pos: p0 + 16 + i as u32 }], &[0])?;
        }
        Ok(())
    })?;
    println!("decode: {n} tokens in {dt:.3}s = {:.1} tok/s ({:.3} ms/token)", n as f64 / dt, dt * 1e3 / n as f64);
    // as the server does it: wait for each token and feed it back
    let (t, mut launch, mut next) = (Instant::now(), 0f64, 42);
    for i in 0..n {
        let l = Instant::now();
        m.run(&[Tok { id: next, slot: 0, pos: p0 + (16 + n + i) as u32 }], &[0])?;
        launch += l.elapsed().as_secs_f64();
        next = m.argmax()?[0];
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "round trip: {:.1} tok/s ({:.3} ms/token, {:.3} ms of it in step())",
        n as f64 / dt,
        dt * 1e3 / n as f64,
        launch * 1e3 / n as f64
    );
    let dt = timed(m, |m| {
        for i in 0..n {
            m.run(&[Tok { id: next, slot: 0, pos: p0 + (16 + 2 * n + i) as u32 }], &[0])?;
            let _ = m.argmax()?;
            let _ = m.logits_row(0)?;
        }
        Ok(())
    })?;
    println!("with logits: {:.1} tok/s ({:.3} ms/token)", n as f64 / dt, dt * 1e3 / n as f64);
    // speculative decoding's steps: the last token and k drafts of one sequence, every row's logits (each run continues
    // where the last one ended: engines that roll back only inside a step refuse an earlier restart)
    let mut p = p0 + (16 + 3 * n) as u32;
    let drafts = |p: u32, k: usize| (0..=k as u32).map(|j| Tok { id: 42, slot: 0, pos: p + j }).collect::<Vec<_>>();
    for k in [2usize, 4] {
        let dt = timed(m, |m| {
            for _ in 0..n / 2 {
                m.run(&drafts(p, k), &(0..=k).collect::<Vec<_>>())?;
                p += k as u32 + 1;
            }
            Ok(())
        })?;
        println!("steps of 1 + {k} drafts: {:.3} ms/step", dt * 1e3 / (n / 2) as f64);
    }
    // a round of speculative decoding with the model's draft layer: the target over the last token and its drafts, every
    // row's logits, the drafter on the last token
    if m.has_draft() {
        let k = m.draft_tokens();
        let dt = timed(m, |m| {
            for _ in 0..n / 2 {
                m.run(&drafts(p, k), &(0..=k).collect::<Vec<_>>())?;
                let _ = m.draft(&[Tok { id: 42, slot: 0, pos: p }], &[0], &[0], k, &[])?;
                p += 1;
            }
            Ok(())
        })?;
        println!("rounds of 1 + {k} drafts with the draft layer: {:.3} ms/round", dt * 1e3 / (n / 2) as f64);
    }
    // batched decode: one token for each of `slots` sequences per step
    for b in [4, 8, 12, 16] {
        if m.slots() < b {
            continue;
        }
        let rows: Vec<usize> = (0..b).collect();
        // every slot holds a sequence of p0 + 16 tokens first (not timed)
        for s in 0..b as u32 {
            fill(m, s, p0 as usize + 16, |_| 42, false)?;
        }
        let at = |i: usize| -> Vec<Tok> {
            (0..b as u32).map(|s| Tok { id: 42, slot: s, pos: p0 + 16 + i as u32 }).collect()
        };
        m.run(&at(0), &rows)?;
        let dt = timed(m, |m| {
            for i in 1..=n / 4 {
                m.run(&at(i), &rows)?;
            }
            Ok(())
        })?;
        println!("decode x{b}: {:.1} tok/s ({:.3} ms/step)", (n / 4 * b) as f64 / dt, dt * 1e3 / (n / 4) as f64);
    }
    // prompt: chunks of m.max_tok() tokens in a fresh slot
    let slot = m.slots() as u32 - 1;
    for len in [m.max_tok(), 2048] {
        let dt = timed(m, |m| fill(m, slot, len, |j| 1000 + j as u32 % 1000, true))?;
        println!("prompt of {len}: {:.0} tok/s ({:.1} ms)", len as f64 / dt, dt * 1e3);
    }
    Ok(())
}

/// capture: residual-stream activations for representation engineering (contrastive activation steering).
/// For each prompt in `prompts` (JSONL, one `{"group": "...", "text": "..."}` per line) render it as a single user
/// turn with the serving chat template (enable_thinking false, generation prompt appended), run one ungraphed forward,
/// and store the last K positions' raw residual rows at every layer (K from --last, default 4). Writes a safetensors
/// with one f32 tensor `p{idx}` [n_layers, K, hidden] per prompt, an int32 `group_ids` tensor, and metadata (groups,
/// counts, dims) for the offline direction/gate math in numpy. State is reset (`release`) between prompts.
#[cfg(has_cuda)]
pub fn capture(opts: &Opts, dir: &str, prompts: &str, out: &str) -> Result<()> {
    use std::collections::HashMap;
    let k: usize = opts.get("last", 4)?;
    ensure!(k >= 1, "--last must be >= 1");
    let ctx = opts.get("ctx", 4096)?;
    let chat = chat::Chat::load(Path::new(dir))?;
    let kwargs = serde_json::json!({"enable_thinking": false}).as_object().unwrap().clone();
    // prompts.jsonl is project input (not raw/ untrusted), parsed as plain JSON per line
    let text = std::fs::read_to_string(prompts).with_context(|| format!("reading {prompts}"))?;
    let rows: Vec<(String, String)> = text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            let v: serde_json::Value = serde_json::from_str(l).with_context(|| format!("{prompts} line {}", i + 1))?;
            let g = v["group"].as_str().with_context(|| format!("{prompts} line {}: no \"group\" string", i + 1))?;
            let t = v["text"].as_str().with_context(|| format!("{prompts} line {}: no \"text\" string", i + 1))?;
            Ok((g.to_string(), t.to_string()))
        })
        .collect::<Result<_>>()?;
    ensure!(!rows.is_empty(), "no prompts in {prompts}");

    let adapter = opts.adapter()?;
    let mut m = crate::cuda::qwen::Model::load(Path::new(dir), ctx, 1, opts.int8, adapter.as_ref().map(|a| &a.1))?;
    if adapter.is_some() {
        m.set_adapter(0, true)?;
    }
    let (nl, h) = (m.n_layers(), m.hidden());

    let mut group_names: Vec<String> = Vec::new();
    let mut group_ids: Vec<u32> = Vec::new();
    let mut tensors: Vec<(String, Vec<f32>)> = Vec::new();
    let mut tail_ids: Vec<i32> = Vec::new();
    let mut n_tokens: Vec<i32> = Vec::new();
    let mut tail0: Option<Vec<i32>> = None;

    for (idx, (group, txt)) in rows.iter().enumerate() {
        let msgs = serde_json::json!([{"role": "user", "content": txt}]);
        let ids = chat.encode(&chat.render(&msgs, None, &kwargs)?)?;
        ensure!(ids.len() > k, "prompt {idx} (group {group}) has {} tokens <= --last={k}", ids.len());
        // the last K ids are the template suffix we tap: assert it is identical across prompts (position alignment)
        let tail: Vec<i32> = ids[ids.len() - k..].iter().map(|&x| x as i32).collect();
        match &tail0 {
            None => tail0 = Some(tail.clone()),
            Some(t0) => ensure!(
                &tail == t0,
                "prompt {idx} (group {group}) suffix ids {tail:?} differ from prompt 0's                                  {t0:?}: the tapped positions are not aligned across prompts"
            ),
        }
        tail_ids.extend_from_slice(&tail);
        n_tokens.push(ids.len() as i32);
        let toks: Vec<Tok> = ids.iter().enumerate().map(|(p, &id)| Tok { id, slot: 0, pos: p as u32 }).collect();
        let caps = m
            .capture_forward(&toks, k)
            .with_context(|| format!("prompt {idx} (group {group}, {} tokens)", ids.len()))?;
        ensure!(caps.len() == nl * k * h, "capture returned {} values, expected {}", caps.len(), nl * k * h);
        let gid = group_names.iter().position(|g| g == group).unwrap_or_else(|| {
            group_names.push(group.clone());
            group_names.len() - 1
        });
        group_ids.push(gid as u32);
        m.release(0)?;
        tensors.push((format!("p{idx}"), caps));
    }

    let mut out_vec: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    for (name, data) in &tensors {
        out_vec.push((name.clone(), Dtype::F32, vec![nl, k, h], data.iter().flat_map(|x| x.to_le_bytes()).collect()));
    }
    out_vec.push((
        "group_ids".to_string(),
        Dtype::I32,
        vec![group_ids.len()],
        group_ids.iter().flat_map(|&g| (g as i32).to_le_bytes()).collect(),
    ));
    out_vec.push((
        "tail_ids".to_string(),
        Dtype::I32,
        vec![rows.len(), k],
        tail_ids.iter().flat_map(|&x| x.to_le_bytes()).collect(),
    ));
    out_vec.push((
        "n_tokens".to_string(),
        Dtype::I32,
        vec![rows.len()],
        n_tokens.iter().flat_map(|&x| x.to_le_bytes()).collect(),
    ));
    // FNV-1a 64-bit of a canonical rendered prompt: a stable fingerprint of the chat template + kwargs + specials,
    // so the offline direction is only ever applied to the exact config it was measured on.
    let probe = chat.render(&serde_json::json!([{"role": "user", "content": "QWENNY_CALIB_PROBE"}]), None, &kwargs)?;
    let template_hash =
        probe.bytes().fold(0xcbf29ce484222325u64, |hsh, b| (hsh ^ b as u64).wrapping_mul(0x100000001b3));
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("groups".into(), group_names.join(","));
    meta.insert("n_prompts".into(), rows.len().to_string());
    meta.insert("n_layers".into(), nl.to_string());
    meta.insert("k".into(), k.to_string());
    meta.insert("hidden".into(), h.to_string());
    if let Some((name, ad)) = &adapter {
        meta.insert("adapter".into(), name.clone().unwrap_or_else(|| "adapter".into()));
        meta.insert("adapter_matrices".into(), ad.len().to_string());
    }
    meta.insert("model_dir".into(), dir.to_string());
    meta.insert("int8".into(), opts.int8.to_string());
    meta.insert("template_hash".into(), format!("{template_hash:016x}"));
    let views: Vec<_> = out_vec
        .iter()
        .map(|(n, d, sh, b)| Ok((n.clone(), safetensors::tensor::TensorView::new(*d, sh.clone(), b)?)))
        .collect::<Result<_>>()?;
    safetensors::serialize_to_file(views, Some(meta), Path::new(out))?;

    println!("wrote {out}: {} prompts x [{nl}][{k}][{h}] f32", rows.len());
    Ok(())
}

/// Render the reference prompts with the chat template and compare the token ids with Transformers': fails when one
/// differs.
pub fn tokcheck(dir: &str, reference: &str, tests: &str) -> Result<()> {
    let chat = chat::Chat::load(Path::new(dir))?;
    let bytes = read(reference)?;
    let r = SafeTensors::deserialize(&bytes)?;
    let ticket: serde_json::Value = serde_json::from_str(std::fs::read_to_string(tests)?.lines().next().unwrap_or(""))?;
    let ticket_msgs: Vec<_> =
        ticket["messages"].as_array().unwrap().iter().filter(|m| m["role"] != "assistant").cloned().collect();
    let prompts = [
        serde_json::json!([{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]),
        serde_json::Value::Array(ticket_msgs),
    ];
    let kwargs = serde_json::json!({"enable_thinking": false}).as_object().unwrap().clone();
    let mut differ = 0;
    for (i, msgs) in prompts.iter().enumerate() {
        let ours = chat.encode(&chat.render(msgs, None, &kwargs)?)?;
        let theirs = ints(&r, &format!("ids_{i}"))?;
        let same = ours == theirs;
        println!(
            "prompt {i}: {} tokens (reference {}): {}",
            ours.len(),
            theirs.len(),
            if same { "identical" } else { "DIFFERENT" }
        );
        if !same {
            differ += 1;
            let at = ours.iter().zip(&theirs).position(|(a, b)| a != b).unwrap_or(ours.len().min(theirs.len()));
            println!(
                "  first difference at {at}: ours {:?} reference {:?}",
                &ours[at..(at + 5).min(ours.len())],
                &theirs[at..(at + 5).min(theirs.len())]
            );
        }
    }
    ensure!(differ == 0, "tokcheck: the prompt tokens of {differ} of {} prompts differ", prompts.len());
    println!("{} prompts: tokens identical to Transformers'", prompts.len());
    Ok(())
}

/// Sample on the GPU and with the host reference from the same logits, many parameter sets and draws.
pub fn samplecheck(opts: &Opts, dir: &str) -> Result<()> {
    let mut m = opts.load_at(dir, 4096, 2)?;
    // two sequences with different histories; repeated tokens so the penalty matters
    let hist: [Vec<u32>; 2] =
        [(0..60).map(|i| 1000 + (i * 7) % 23).collect(), (0..40).map(|i| 5000 + (i * 13) % 37).collect()];
    let both: Vec<Tok> = hist.iter().enumerate().flat_map(|(s, h)| toks(h, 0, h.len(), s as u32)).collect();
    let rows = [hist[0].len() - 1, both.len() - 1];
    // the presence penalty's set: the last 20 tokens of each history count as generated
    let gened: Vec<std::collections::HashSet<u32>> =
        hist.iter().map(|h| h[h.len() - 20..].iter().copied().collect()).collect();
    for (s, g) in gened.iter().enumerate() {
        let words = bitset(m.vocab(), g.iter().copied());
        m.set_outputs(s as u32, &words)?;
    }
    // every other draw from a step of each sequence's last token alone (its rows read the penalties' sets only; the
    // whole histories' steps count their tokens after the first as the step's drafts)
    let last: Vec<Tok> = hist.iter().enumerate().flat_map(|(s, h)| toks(h, h.len() - 1, h.len(), s as u32)).collect();
    let (mut same, mut total) = (0, 0);
    let mut rng = sample::Rng::new(7);
    for &(temp, k, p, rep, pres) in &[
        (0.0, 50, 1.0, 1.1, 0.0),
        (0.1, 50, 1.0, 1.1, 0.0),
        (0.7, 20, 0.8, 1.0, 0.0),
        (1.0, 0, 0.95, 1.1, 0.0),
        (1.5, 0, 1.0, 1.0, 0.0),
        (0.9, 5, 0.5, 1.3, 0.0),
        (0.0, 20, 1.0, 1.0, 1.5),
        (1.0, 20, 0.95, 1.0, 1.5),
        (0.7, 20, 0.8, 1.1, 2.0),
    ] {
        let params = sample::Params {
            temperature: temp,
            top_k: k,
            top_p: p,
            repetition_penalty: rep,
            presence_penalty: pres,
            seed: 0,
        };
        for it in 0..50 {
            let whole = it % 2 == 0;
            let rowp: Vec<[f32; 8]> = (0..2).map(|_| params.row(&mut rng, None)).collect();
            if whole {
                m.run(&both, &rows)?
            } else {
                m.run(&last, &[0, 1])?
            }
            let gpu = m.sample(&rowp)?;
            for j in 0..2 {
                let lg = m.logits_row(j)?;
                let seen: std::collections::HashSet<u32> = hist[j].iter().copied().collect();
                let in_step = |t: u32| whole && hist[j][1..].contains(&t);
                let want =
                    sample::sample_row(&lg, |t| seen.contains(&t), |t| gened[j].contains(&t) || in_step(t), &rowp[j]).0;
                same += (want == gpu[j].0) as usize;
                total += 1;
            }
        }
        println!(
            "temperature {temp} top_k {k} top_p {p} penalty {rep} presence {pres}: {same}/{total} draws agree so far"
        );
    }
    let mut disagree = total - same;
    // sampled drafts (the draft layer's): drawn on the GPU from its distribution q under the row's sampling, then
    // accepted with min(1, p / q) or resampled from max(0, p - q): the GPU's tokens must be the host reference's
    // (sample::sample_row_q, whose output follows p: the unit test) for the q the GPU kept, draw for draw
    if m.has_draft() {
        let h = &hist[0];
        let n = h.len();
        let (mut same, mut total, mut acc) = (0, 0, 0);
        for &(temp, k, p, rep, pres) in &[(1.0, 20, 0.95, 1.0, 1.5), (0.7, 0, 0.8, 1.1, 0.0), (1.3, 50, 1.0, 1.0, 0.0)]
        {
            let params = sample::Params {
                temperature: temp,
                top_k: k,
                top_p: p,
                repetition_penalty: rep,
                presence_penalty: pres,
                seed: 0,
            };
            for it in 0..20 {
                let next = 3000 + it as u32;
                m.run(&toks(h, 0, n, 0), &[n - 1])?;
                let ds = DraftSampling { row: params.draft_row(true), u: (0..3).map(|_| rng.next_f32()).collect() };
                let d = m.draft(&[Tok { id: next, slot: 0, pos: (n - 1) as u32 }], &[n - 1], &[0], 3, &[Some(ds)])?;
                ensure!(d[0].len() == 3, "sampled drafts: {d:?}");
                let d = &d[0];
                let mut step = vec![Tok { id: next, slot: 0, pos: n as u32 }];
                step.extend(d.iter().enumerate().map(|(j, &id)| Tok { id, slot: 0, pos: (n + 1 + j) as u32 }));
                m.run(&step, &[0, 1, 2, 3])?;
                let rowp: Vec<[f32; 8]> = (0..4)
                    .map(|j| {
                        let mut r = params.row(&mut rng, d.get(j).copied());
                        if j < 3 {
                            sample::mark_sampled(&mut r);
                        }
                        r
                    })
                    .collect();
                let words = bitset(m.vocab(), gened[0].iter().copied());
                m.set_outputs(0, &words)?;
                let gpu = m.sample(&rowp)?;
                // the step marks all its tokens seen (run: no marks), its drafts included
                let seen: std::collections::HashSet<u32> =
                    h.iter().copied().chain(std::iter::once(next)).chain(d.iter().copied()).collect();
                for j in 0..4 {
                    let lg = m.logits_row(j)?;
                    let before = &d[..j];
                    let gn = |t: u32| gened[0].contains(&t) || before.contains(&t);
                    let sn = |t: u32| seen.contains(&t);
                    let want = if j < 3 {
                        let (q, qt) = m.draft_q(0, j)?;
                        ensure!(qt == d[j] && !q.is_empty(), "draft {j}: kept {qt}, drafted {}", d[j]);
                        let mut r = rowp[j];
                        r[6] = d[j] as f32;
                        sample::sample_row_q(&lg, sn, gn, &r, &q)
                    } else {
                        sample::sample_row(&lg, sn, gn, &rowp[j])
                    };
                    same += (want == gpu[j]) as usize;
                    total += 1;
                    acc += (j < 3 && gpu[j].1) as usize;
                    if !gpu[j].1 {
                        break;
                    }
                }
            }
            println!(
                "sampled drafts at temperature {temp} top_k {k} top_p {p} penalty {rep} presence {pres}: \
                      {same}/{total} draws agree so far ({acc} drafts accepted)"
            );
        }
        disagree += total - same;
    } else {
        println!("no draft layer: no sampled drafts to check");
    }
    // with a draft: the accept-or-resample result must follow the same distribution as plain sampling
    m.run(&both, &rows)?;
    let lg = m.logits_row(0)?;
    let mut max_tv = 0f64;
    for &(temp, k, p, rep, pres) in
        &[(1.0, 1024, 1.0, 1.0, 0.0), (0.8, 20, 0.9, 1.1, 0.0), (1.5, 50, 1.0, 1.3, 0.0), (1.0, 20, 0.95, 1.0, 1.5)]
    {
        let params = sample::Params {
            temperature: temp,
            top_k: k,
            top_p: p,
            repetition_penalty: rep,
            presence_penalty: pres,
            seed: 0,
        };
        // target: the host reference's distribution, estimated with many draws
        let mut want: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
        let seen: std::collections::HashSet<u32> = hist[0].iter().copied().collect();
        // the uniform grid's draws, on all cores
        let draws: Vec<u32> = std::thread::scope(|sc| {
            let parts: Vec<_> = (0..16)
                .map(|c| {
                    let (lg, seen, g) = (&lg, &seen, &gened[0]);
                    sc.spawn(move || {
                        (c * 1250..(c + 1) * 1250)
                            .map(|i| {
                                let row = [temp, k as f32, p, rep, (i as f32 + 0.5) / 20000.0, 0.0, -1.0, pres];
                                sample::sample_row(lg, |t| seen.contains(&t), |t| g.contains(&t), &row).0
                            })
                            .collect::<Vec<u32>>()
                    })
                })
                .collect();
            parts.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        for t in draws {
            *want.entry(t).or_insert(0.0) += 1.0 / 20000.0;
        }
        let top = *want.iter().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        let second = want.iter().filter(|e| *e.0 != top).max_by(|a, b| a.1.total_cmp(b.1)).map_or(top, |e| *e.0);
        for d in [None, Some(top), Some(second)] {
            let mut got: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
            let draws = 8000;
            for _ in 0..draws / 2 {
                let rowp: Vec<[f32; 8]> = (0..2).map(|_| params.row(&mut rng, d)).collect();
                for (t, _) in m.sample(&rowp)?.into_iter().take(1) {
                    *got.entry(t).or_insert(0.0) += 2.0 / draws as f64;
                }
            }
            let keys: std::collections::HashSet<u32> = want.keys().chain(got.keys()).copied().collect();
            let tv: f64 = keys
                .iter()
                .map(|t| (want.get(t).copied().unwrap_or(0.0) - got.get(t).copied().unwrap_or(0.0)).abs())
                .sum::<f64>()
                / 2.0;
            max_tv = max_tv.max(tv);
            let pd = d.map_or(0.0, |d| want.get(&d).copied().unwrap_or(0.0));
            println!(
                "draft {d:?} at temperature {temp} top_k {k} top_p {p} penalty {rep} presence {pres}: total variation {tv:.3} over {} draws (p(draft) {pd:.3})",
                draws / 2
            );
        }
    }
    ensure!(disagree == 0, "samplecheck: {disagree} draws differ from the host reference");
    println!(
        "every draw agrees with the host reference; max total variation of the accept-or-resample result {max_tv:.3}"
    );
    Ok(())
}

/// Greedy decoding one token per step vs. steps that feed drafts (some right, then a wrong one): the same tokens, or
/// it fails.
pub fn speccheck(opts: &Opts, dir: &str) -> Result<()> {
    let mut m = opts.load_at(dir, 4096, 2)?;
    let prompt: Vec<u32> = (0..300).map(|i| 1000 + (i * 7919) % 5000).collect();
    // greedy tokens in `slot` with `k` drafts per step: the first `good` of them from `reference`, then wrong ones
    let run = |m: &mut dyn Engine, slot: u32, k: usize, reference: &[u32]| -> Result<Vec<u32>> {
        m.run(&toks(&prompt, 0, prompt.len(), slot), &[prompt.len() - 1])?;
        let mut out = vec![m.argmax()?[0]];
        let mut step = 0;
        while out.len() < 120 {
            let pos = prompt.len() + out.len() - 1;
            let good = step % (k + 1);
            step += 1;
            let d: Vec<u32> = (0..k)
                .map(|j| if j < good { reference.get(out.len() + j).copied().unwrap_or(7) } else { 7 + j as u32 })
                .collect();
            let toks: Vec<Tok> = std::iter::once(*out.last().unwrap())
                .chain(d.iter().copied())
                .enumerate()
                .map(|(j, id)| Tok { id, slot, pos: (pos + j) as u32 })
                .collect();
            m.run(&toks, &(0..toks.len()).collect::<Vec<_>>())?;
            let am = m.argmax()?;
            for (j, &t) in am.iter().enumerate() {
                out.push(t);
                if j >= d.len() || d[j] != t {
                    break;
                }
            }
        }
        out.truncate(120);
        Ok(out)
    };
    let reference = run(&mut *m, 0, 0, &[])?;
    let mut wrong = Vec::new();
    for k in [1, 2, 3, 4, 7] {
        let got = run(&mut *m, 1, k, &reference)?;
        let same = got.iter().zip(&reference).take_while(|(a, b)| a == b).count();
        println!("drafts of {k}: first {same}/{} tokens as one token per step", reference.len());
        if same != reference.len() {
            wrong.push(k);
        }
    }
    ensure!(wrong.is_empty(), "speccheck: drafts of {wrong:?} give other tokens than one token per step");
    println!("drafts of 1, 2, 3, 4 and 7 give the same {} tokens as one token per step", reference.len());
    Ok(())
}

/// The session cache under device pressure (sessions::Store): sequences a and b finish in slots, a moves to the host,
/// c takes the device's room, a comes back while b moves out chunk by chunk (Store::swap_in), then b comes back: both
/// continue with exactly the logits they have when they never leave a slot, or it fails
pub fn swapcheck(opts: &Opts, dir: &str) -> Result<()> {
    let len: usize = opts.get("len", 131072)?;
    ensure!(len.is_multiple_of(64), "--len: a multiple of 64");
    let tail = 256;
    let mut m = opts.load_at(dir, len + 1024, 3)?;
    let seq =
        |k: usize| -> Vec<u32> { (0..len + tail).map(|i| 1000 + ((i * 7919 + k * 104729) % 20000) as u32).collect() };
    let (a, b, c) = (seq(1), seq(2), seq(3));
    let held =
        |ids: &[u32]| sessions::Held { tokens: ids[..len].to_vec(), adapter: false, dirty: false, snaps: Vec::new() };
    // each alone in a slot: its prompt, then its tail
    let mut refs = Vec::new();
    for ids in [&a, &b] {
        feed(m.as_mut(), ids, 0, len, 0)?;
        refs.push(feed(m.as_mut(), ids, len, len + tail, 0)?);
        m.release(0)?;
    }
    let mut st = sessions::Store::new(&*m, None)?;
    feed(m.as_mut(), &a, 0, len, 0)?;
    st.put(0, held(&a));
    feed(m.as_mut(), &b, 0, len, 1)?;
    st.put(1, held(&b));
    // a to the host, c in its slot: then the device holds b and c, and not a third
    ensure!(st.park_one(&mut m, None, true)? == Some(0), "a not moved to the host");
    feed(m.as_mut(), &c, 0, len, 0)?;
    st.put(0, held(&c));
    let mut free = vec![2];
    let mut all = true;
    for (name, ids, r) in [("a", &a, &refs[0]), ("b", &b, &refs[1])] {
        let (slot, at, _) = st.take(&mut m, &ids[..len + tail], false, &mut free, 0)?.context("not taken")?;
        ensure!(at == len, "{name} resumes at {at}, not {len}");
        let l = feed(m.as_mut(), ids, len, len + tail, slot)?;
        let d = max_diff(&l, r);
        println!("{name} back in slot {slot} after {len} positions: max |logit diff| {d} vs. never leaving a slot");
        all &= d == 0.0;
        m.release(slot)?;
        free.push(slot);
    }
    ensure!(
        all,
        "swapcheck: MISMATCH, the sequences do not continue identically after moving between slots and host memory"
    );
    println!("sequences continue identically after moving between slots and host memory");
    Ok(())
}

/// A slot saved and put into another slot continues as if it had never left: slot 0 runs a prompt in chunks of 64
/// and then decodes; slot 1 runs its first half, is saved, then overwritten by another sequence, and the image goes
/// into slot 2, which runs the rest and half the decode; saved again, slot 1 decodes the rest. The logits after the
/// prompt and the tokens decoded must be identical (the same steps, the same state), or it fails.
pub fn cachecheck(opts: &Opts, dir: &str) -> Result<()> {
    let mut m = opts.load_at(dir, 32768, 3)?;
    let v = m.vocab();
    let decode = |m: &mut Box<dyn Engine>, first: u32, pos: usize, n: usize, slot: u32| -> Result<Vec<u32>> {
        let mut out = vec![first];
        for i in 0..n {
            m.run(&[Tok { id: *out.last().unwrap(), slot, pos: (pos + i) as u32 }], &[0])?;
            out.push(m.argmax()?[0]);
        }
        Ok(out)
    };
    let mut all = true;
    // 20,000: the second image reuses the first's whole chunk of 8,192 positions (sessions::CHUNK)
    for len in [1500usize, 4000, 20000] {
        let ids: Vec<u32> = (0..len).map(|i| 1000 + ((i * 7919) % 20000) as u32).collect();
        let cut = len / 2 / 64 * 64;
        let la = feed(m.as_mut(), &ids, 0, len, 0)?;
        let ta = decode(&mut m, argmax(&la), len, 12, 0)?;
        m.release(0)?;
        feed(m.as_mut(), &ids, 0, cut, 1)?;
        let t = std::time::Instant::now();
        let img = sessions::save_image(&mut m, 1, cut, None)?;
        let save_ms = t.elapsed().as_secs_f64() * 1e3;
        m.release(1)?;
        let junk: Vec<u32> = (0..700).map(|i| 3000 + ((i * 104729) % 30000) as u32).collect();
        feed(m.as_mut(), &junk, 0, junk.len(), 1)?;
        let t = std::time::Instant::now();
        ensure!(sessions::load_image(&mut m, 2, &img)?, "no device memory to load the slot");
        let load_ms = t.elapsed().as_secs_f64() * 1e3;
        let lb = feed(m.as_mut(), &ids, cut, len, 2)?;
        let mut tb = decode(&mut m, argmax(&lb), len, 6, 2)?;
        // saved again on top of the first image (its whole chunks reused, as the session cache does)
        let img2 = sessions::save_image(&mut m, 2, len + 6, Some(img))?;
        m.release(2)?;
        ensure!(sessions::load_image(&mut m, 1, &img2)?, "no device memory to load the slot");
        tb.extend(decode(&mut m, *tb.last().unwrap(), len + 6, 6, 1)?.into_iter().skip(1));
        m.release(1)?;
        let l2 = feed(m.as_mut(), &ids, 0, len, 1)?;
        m.release(1)?;
        let base = max_diff(&la, &l2);
        println!("context {len}: the same prompt again in another slot (after): max |logit diff| {base}");
        let maxd = max_diff(&la, &lb);
        let same = ta.iter().zip(&tb).filter(|(a, b)| a == b).count();
        all &= maxd == 0.0 && same == ta.len();
        println!(
            "context {len}, saved at {cut}: max |logit diff| {maxd} over {v} logits; decoded tokens identical at \
                  {same}/{}; image {:.1} MB, save {save_ms:.0} ms, load {load_ms:.0} ms",
            ta.len(),
            img2.bytes() as f64 / 1e6
        );
    }
    ensure!(all, "cachecheck: MISMATCH, the slots do not continue identically after save and load");
    println!("slots continue identically after save and load");
    Ok(())
}

/// A long sequence fed as prompt chunks (tile attention) vs. one token per step (decode attention): logits of the
/// last 64 positions should agree up to rounding: it fails when the largest |logit diff| is not below --limit (an
/// argmax may flip where two logits nearly tie: the 4B has one at 1,500 tokens, 0.026 apart).
pub fn longcheck(opts: &Opts, dir: &str) -> Result<()> {
    let limit = opts.limit()?;
    let mut m = opts.load_at(dir, 8192, 2)?;
    let (mut diffs, mut agrees) = (Vec::new(), Vec::new());
    for len in [600usize, 1500, 4000] {
        let ids: Vec<u32> = (0..len).map(|i| 1000 + ((i * 7919) % 20000) as u32).collect();
        let tail = len - 64;
        // slot 0: chunks of up to 64 tokens with all logits of the tail
        let mut a = Vec::new();
        for c in (0..len).step_by(64) {
            let step = toks(&ids, c, (c + 64).min(len), 0);
            let rows: Vec<usize> = (0..step.len()).filter(|j| c + j >= tail).collect();
            m.run(&step, &rows)?;
            if !rows.is_empty() {
                a.extend(m.logits()?);
            }
        }
        // slot 1: one token per step
        let mut b = Vec::new();
        for (i, &id) in ids.iter().enumerate() {
            m.run(&[Tok { id, slot: 1, pos: i as u32 }], &[0])?;
            if i >= tail {
                b.extend(m.logits()?);
            }
        }
        let (mut maxd, mut agree) = (0f32, 0);
        for r in 0..64 {
            let v = m.vocab();
            let (x, y) = (&a[r * v..(r + 1) * v], &b[r * v..(r + 1) * v]);
            agree += (argmax(x) == argmax(y)) as usize;
            maxd = worse(maxd, max_diff(x, y));
        }
        println!("context {len}: argmax agrees at {agree}/64 positions, max |logit diff| {maxd:.3}");
        diffs.push(maxd);
        agrees.push(format!("{agree}/64"));
    }
    let worst = below("longcheck: max |logit diff|", diffs, limit)?;
    println!("max |logit diff| {worst:.3} (< {limit}) at every context length, argmax agrees at {}", agrees.join(" "));
    Ok(())
}

/// A LoRA adapter served unmerged beside its base model (tools/make_reference_lora.py; --adapter): two requests per
/// step, the reference's prompts 1 and 2 in slots 0 and 1 (a first step of 2,048 tokens: both prompts' first chunks),
/// then 6 decode steps of 4 tokens each (the reference's greedy tokens fed back, as accepted drafts). Four runs: both
/// slots on the base model, both on the adapter, and the two mixed ones. Every row of a mixed run must be identical
/// to the same row of the run with both slots on its slot's model (the other model in the step changes nothing),
/// and the logits at the reference's positions are compared with Transformers + PEFT (adapter) or Transformers (base):
/// it fails when a row of a mixed run differs, or a mean |logit diff| is not below --limit.
pub fn adaptercheck(opts: &Opts, dir: &str, reference: &str) -> Result<()> {
    let limit = opts.limit()?;
    let bytes = read(reference)?;
    let r = SafeTensors::deserialize(&bytes)?;
    ensure!(opts.path("adapter").is_some(), "adaptercheck: --adapter=<dir> (the reference's adapter)");
    let mut m = opts.load_at(dir, 8192, 2)?;
    let (vocab, max_tok, max_out) = (m.vocab(), m.max_tok(), m.max_out());
    // per slot: the prompt, its reference positions, and per model (base, adapter) its logits there and greedy tokens
    let prompts = [1, 2];
    let mut ids = Vec::new();
    let mut pos = Vec::new();
    let mut refs = Vec::new();
    for p in prompts {
        ids.push(ints(&r, &format!("ids_{p}"))?);
        pos.push(ints(&r, &format!("pos_{p}"))?);
        let one = |pre: &str| -> Result<(Vec<f32>, Vec<u32>)> {
            Ok((floats(&r, &format!("{pre}logits_{p}"))?, ints(&r, &format!("{pre}gen_{p}"))?))
        };
        refs.push([one("base_")?, one("")?]);
    }
    // the steps: (slot, first position, tokens), logit rows wanted per slot at most
    let n0 = ids[0].len();
    ensure!(n0 < max_tok && ids[1].len() > max_tok - n0, "prompts of {} and {} tokens", n0, ids[1].len());
    let mut steps: Vec<Vec<(usize, usize, usize)>> = vec![vec![(0, 0, n0), (1, 0, max_tok - n0)]];
    let mut at = max_tok - n0;
    while at < ids[1].len() {
        let c = (ids[1].len() - at).min(max_tok);
        steps.push(vec![(1, at, c)]);
        at += c;
    }
    const DEC: usize = 6;
    for k in 0..DEC {
        steps.push((0..2).map(|s| (s, ids[s].len() + 4 * k, 4)).collect());
    }
    // one run: per slot its model; per step the logit rows (slot, position) and their logits
    type Rows = Vec<(Vec<(usize, usize)>, Vec<f32>)>;
    let run = |m: &mut Box<dyn Engine>, models: [bool; 2]| -> Result<Rows> {
        for (s, &a) in models.iter().enumerate() {
            m.release(s as u32)?;
            m.set_adapter(s as u32, a)?;
        }
        let mut out = Vec::new();
        for step in &steps {
            let (mut toks, mut rows, mut which) = (Vec::new(), Vec::new(), Vec::new());
            for &(s, p0, c) in step {
                let hist: Vec<u32> =
                    ids[s].iter().copied().chain(refs[s][models[s] as usize].1.iter().copied()).collect();
                // prompt chunks: the reference's positions in the chunk (half the rows each); decode: every token
                let mut want: Vec<usize> = if p0 >= ids[s].len() {
                    (p0..p0 + c).collect()
                } else {
                    pos[s].iter().map(|&q| q as usize).filter(|&q| q >= p0 && q < p0 + c).take(max_out / 2).collect()
                };
                want.truncate(max_out / 2);
                for (q, &id) in (p0..).zip(&hist[p0..p0 + c]) {
                    if want.contains(&q) {
                        rows.push(toks.len());
                        which.push((s, q));
                    }
                    toks.push(Tok { id, slot: s as u32, pos: q as u32 });
                }
            }
            m.run(&toks, &rows)?;
            out.push((which, m.logits()?));
        }
        Ok(out)
    };
    let names = ["base", "adapter"];
    let uniform = [run(&mut m, [false, false])?, run(&mut m, [true, true])?];
    let (mut same, mut rows, mut worst, mut means) = (0usize, 0usize, 0f32, Vec::new());
    for models in [[false, true], [true, false]] {
        let got = run(&mut m, models)?;
        // per slot: the sum of |diff| over the reference's logits, values compared; argmax vs. its greedy tokens
        let (mut sum, mut cnt, mut agree, mut dec) = ([0f64; 2], [0usize; 2], [0usize; 2], [0usize; 2]);
        for (k, (which, lg)) in got.iter().enumerate() {
            for (j, &(s, q)) in which.iter().enumerate() {
                let a = models[s] as usize;
                let row = &lg[j * vocab..(j + 1) * vocab];
                let (uw, ul) = &uniform[a][k];
                let ui = uw.iter().position(|&w| w == (s, q)).context("a row missing in the uniform run")?;
                let d = max_diff(row, &ul[ui * vocab..(ui + 1) * vocab]);
                same += (d == 0.0) as usize;
                rows += 1;
                worst = worse(worst, d);
                let (rl, greedy) = &refs[s][a];
                if let Some(i) = pos[s].iter().position(|&p| p as usize == q) {
                    for (x, y) in row.iter().zip(&rl[i * vocab..(i + 1) * vocab]) {
                        sum[s] += (x - y).abs() as f64;
                    }
                    cnt[s] += vocab;
                }
                if q + 1 >= ids[s].len() && q + 1 - ids[s].len() < greedy.len() {
                    agree[s] += (argmax(row) == greedy[q + 1 - ids[s].len()]) as usize;
                    dec[s] += 1;
                }
            }
        }
        for s in 0..2 {
            let mean = sum[s] / cnt[s].max(1) as f64;
            means.push(mean as f32);
            println!(
                "mixed run, slot {s} ({}, prompt {}): mean |logit diff| {mean:.5} vs. the reference at {} positions; \
                      greedy agrees with the reference's at {}/{} decode positions",
                names[models[s] as usize],
                prompts[s],
                cnt[s] / vocab,
                agree[s],
                dec[s]
            );
        }
    }
    println!("mixed steps: {same}/{rows} logit rows identical to the steps with one model (max |diff| {worst})");
    ensure!(
        same == rows,
        "adaptercheck: only {same}/{rows} rows of the mixed steps are identical to the steps with one model"
    );
    let worst = below("adaptercheck: mean |logit diff|", means, limit)?;
    println!(
        "mixed base + adapter steps: {rows} rows identical to one-model steps, mean |logit diff| at most {worst:.5} (< {limit})"
    );
    Ok(())
}
