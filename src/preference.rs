//! Preference training: DPO with an NLL term on the chosen response (RPO; Pang et al. 2024, "Iterative Reasoning
//! Preference Optimization"), on the LoRA trainer of `crate::training`.
//!
//! A row is `{"messages": [<prompt>], "chosen": {<assistant message>}, "rejected": {<assistant message>}}` (and an
//! optional "chat_template_kwargs" as SFT rows have). Each response is rendered as `messages + [response]` and only it
//! is in the loss, masked as SFT masks a reply. The reference policy is the starting point, frozen: the LoRA of
//! --init-adapter, or the base model (a fresh LoRA, B = 0). Its log-probabilities of every response are computed once
//! before training. Per pair, with summed response log-probabilities `lp` (policy) and `rf` (reference):
//!
//!   margin = beta * ((lp_c - rf_c) - (lp_r - rf_r)),   loss = -log sigmoid(margin) + alpha * NLL_c
//!
//! NLL_c the chosen response's mean per-token negative log-likelihood; the step's loss is the mean over its pairs.
//! The gradient reaches each loss row through a weight (`Trainer::pass_weighted`): a chosen token
//! (beta * sigmoid(-margin) + alpha / n_c) / B, a rejected one -beta * sigmoid(-margin) / B, B the pairs of the step.
//! Both responses of a pair go into the same micro-batch when they fit; a pair split over two micro-batches gets the
//! later half's log-probability from a forward pass first.
use crate::chat::Chat;
use crate::cuda::qwen::train::Trainer;
use crate::training::{self, Dev, Example, Micro, Opts, Run, Step, TrainOpts};
use anyhow::{Context, Result, bail, ensure};
use safetensors::SafeTensors;
use serde_json::{Value as Json, json};
use std::{fs, path::Path, time::Instant};

/// The RPO loss's parameters
#[derive(Clone, Copy)]
pub struct PrefOpts {
    /// DPO's beta: the strength of the tie to the reference policy
    pub beta: f32,
    /// alpha: the weight of the chosen response's NLL
    pub nll: f32,
}

pub struct Pair {
    pub chosen: Example,
    pub rejected: Example,
}

/// Whether a JSONL file (or a data dir's train.jsonl) holds preference rows: its first row has "chosen"
pub fn is_preference(data: &Path) -> Result<bool> {
    let (train, _) = training::data_paths(data)?;
    let rows = training::read_rows(&train)?;
    let Some((line, first)) = rows.first() else { return Ok(false) };
    let row: Json = serde_json::from_str(first).with_context(|| format!("{}:{line}: not a JSON object", train.display()))?;
    Ok(row.get("chosen").is_some() || row.get("rejected").is_some())
}

/// The pairs of preference rows, at most `max_len` tokens per response (with its prompt); `name` names the file.
fn encode_pairs(chat: &Chat, rows: &[(usize, String)], max_len: usize, name: &str) -> Result<Vec<Pair>> {
    let defaults = json!({"enable_thinking": false}).as_object().unwrap().clone();
    let (mut out, mut cut, mut lost, mut same) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (line, text) in rows {
        let at = format!("{name}:{line}");
        let row: Json = serde_json::from_str(text).with_context(|| format!("{at}: not a JSON object"))?;
        let Some(messages) = row["messages"].as_array().filter(|m| !m.is_empty()) else {
            bail!("{at}: no \"messages\" list (the prompt); a preference row is {{\"messages\": [{{\"role\": \"user\", \
                   \"content\": ...}}], \"chosen\": {{\"role\": \"assistant\", \"content\": ...}}, \"rejected\": {{...}}}}");
        };
        ensure!(messages.last().unwrap()["role"] != "assistant",
                "{at}: the prompt ends with an assistant message; \"chosen\" and \"rejected\" are the replies to it");
        let kw = training::row_kwargs(&defaults, &row, name, *line)?;
        let mut encode = |key: &str| -> Result<Option<Example>> {
            let Some(r) = row[key].as_object() else { bail!("{at}: no \"{key}\" object (an assistant message)") };
            ensure!(r.get("role").is_none_or(|x| x == "assistant"), "{at}: \"{key}\" has role {}, not assistant", r["role"]);
            let has = |k: &str| r.get(k).is_some_and(|v| v.as_str().is_some_and(|c| !c.trim().is_empty())
                || v.as_array().is_some_and(|c| !c.is_empty()));
            ensure!(has("content") || has("reasoning_content") || has("reasoning") || has("tool_calls"), "{at}: \"{key}\" is empty");
            let mut reply = r.clone();
            reply.insert("role".into(), json!("assistant"));
            let mut msgs = messages.clone();
            msgs.push(Json::Object(reply));
            let (mut ids, mut mask) = chat.encode_last_reply(&Json::Array(msgs), &kw).with_context(|| at.clone())?;
            if ids.len() > max_len {
                cut.push(*line);
                ids.truncate(max_len);
                mask.truncate(max_len);
            }
            let e = Example { ids, mask };
            Ok((e.loss_tokens() > 0).then_some(e))
        };
        let (chosen, rejected) = (encode("chosen")?, encode("rejected")?);
        let (Some(chosen), Some(rejected)) = (chosen, rejected) else {
            lost.push(*line);
            continue;
        };
        if chosen.ids == rejected.ids {
            same.push(*line);
            continue;
        }
        out.push(Pair { chosen, rejected });
    }
    let lines = |v: &mut Vec<usize>| {
        v.dedup();
        format!("line{} {:?}{}", if v.len() > 1 { "s" } else { "" }, &v[..v.len().min(10)], if v.len() > 10 { " ..." } else { "" })
    };
    if !cut.is_empty() {
        eprintln!("warning: {name}: responses longer than --max-len={max_len} (with the prompt) are cut ({}){}", lines(&mut cut),
                  if lost.is_empty() { String::new() } else { format!("; {} pairs lose a whole response and are left out ({})",
                                                                       lost.len(), lines(&mut lost)) });
    }
    if !same.is_empty() {
        eprintln!("warning: {name}: {} pairs whose chosen and rejected are the same are left out ({})", same.len(), lines(&mut same));
    }
    ensure!(!out.is_empty(), "{name}: no pairs");
    Ok(out)
}

/// The two sequences of each pair in order: chosen 0, rejected 0, chosen 1, ...
fn seqs<'a>(pairs: &[&'a Pair]) -> Vec<&'a Example> {
    pairs.iter().flat_map(|p| [&p.chosen, &p.rejected]).collect()
}

/// The summed loss (NLL) of each document of a micro-batch from its rows' losses
fn doc_sums(mb: &Micro, rows: &[f32]) -> Vec<f64> {
    mb.doc_rows.iter().map(|r| training::nll_sum(&rows[r.clone()])).collect()
}

/// The summed log-probability of each pair's responses (chosen, rejected) under the trainer's adapter: forward passes
pub fn logps(tr: &Trainer, pairs: &[&Pair]) -> Result<Vec<(f64, f64)>> {
    let mut lp = Vec::new();
    for mb in training::pack(seqs(pairs).into_iter(), tr.tmax) {
        let (rows, _) = tr.pass(&mb, false, 0.0, 0, 0)?;
        lp.extend(doc_sums(&mb, &rows).into_iter().map(|s| -s));
    }
    Ok(lp.chunks(2).map(|c| (c[0], c[1])).collect())
}

/// The sequences in micro-batches of at most `tmax` tokens, a pair's two together when they fit in one; and where
/// each sequence is (micro-batch, document in it)
fn pack_pairs(seqs: &[&Example], tmax: usize) -> (Vec<Micro>, Vec<(usize, usize)>) {
    let mut out = vec![Micro::default()];
    let mut place = Vec::new();
    for (i, e) in seqs.iter().enumerate() {
        let cur = out.last().unwrap().tokens();
        // a chosen response opens a new micro-batch unless its pair fits (or nothing is in it yet)
        let need = if i % 2 == 0 { e.ids.len() + seqs[i + 1].ids.len() } else { e.ids.len() };
        if cur > 0 && cur + need > tmax {
            out.push(Micro::default());
        }
        let n = out.len() - 1;
        let mb = out.last_mut().unwrap();
        place.push((n, mb.docs.len()));
        mb.push(&e.ids, &e.mask);
    }
    (out, place)
}

/// One pair's terms: (margin, -log sigmoid(margin), the chosen response's mean NLL)
pub fn pair_terms(po: &PrefOpts, lp: (f64, f64), rf: (f64, f64), n_c: usize) -> (f64, f64, f64) {
    let m = po.beta as f64 * ((lp.0 - rf.0) - (lp.1 - rf.1));
    let dpo = if m > 0.0 { (-m).exp().ln_1p() } else { -m + m.exp().ln_1p() };
    (m, dpo, -lp.0 / n_c as f64)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// d(mean loss over `b` pairs) / d(NLL of one response token) for a pair of margin `m`: a chosen token
/// (beta * sigmoid(-m) + alpha / n_c) / b, a rejected one -beta * sigmoid(-m) / b
fn token_weight(po: &PrefOpts, m: f64, n_c: usize, chosen: bool, b: f64) -> f64 {
    let g = po.beta as f64 * sigmoid(-m);
    if chosen { (g + po.nll as f64 / n_c as f64) / b } else { -g / b }
}

/// The means over pairs: loss, DPO part, NLL part (unweighted), margin, share of positive margins, the chosen and the
/// rejected response's reward (beta * (lp - rf))
#[derive(Clone, Copy, Debug, Default)]
pub struct Terms {
    pub loss: f64,
    pub dpo: f64,
    pub nll: f64,
    pub margin: f64,
    pub accuracy: f64,
    pub chosen: f64,
    pub rejected: f64,
}

impl Terms {
    /// As a dev measurement
    fn dev(&self) -> Dev {
        Dev {
            loss: self.loss,
            log: json!({"dev_loss": self.loss, "dev_dpo": self.dpo, "dev_nll": self.nll, "dev_margin": self.margin,
                        "dev_accuracy": self.accuracy}),
            text: format!("loss {:.4} (DPO {:.4}, NLL {:.4}), margin {:+.4}, positive margin {:.1}%", self.loss, self.dpo,
                          self.nll, self.margin, self.accuracy * 100.0),
        }
    }
}

pub fn terms(po: &PrefOpts, pairs: &[&Pair], lp: &[(f64, f64)], rf: &[(f64, f64)]) -> Terms {
    let mut t = Terms::default();
    for ((p, &l), &r) in pairs.iter().zip(lp).zip(rf) {
        let (m, dpo, nll) = pair_terms(po, l, r, p.chosen.loss_tokens());
        t.loss += dpo + po.nll as f64 * nll;
        t.dpo += dpo;
        t.nll += nll;
        t.margin += m;
        t.accuracy += (m > 0.0) as u8 as f64;
        t.chosen += po.beta as f64 * (l.0 - r.0);
        t.rejected += po.beta as f64 * (l.1 - r.1);
    }
    let n = pairs.len() as f64;
    for v in [&mut t.loss, &mut t.dpo, &mut t.nll, &mut t.margin, &mut t.accuracy, &mut t.chosen, &mut t.rejected] {
        *v /= n;
    }
    t
}

/// One optimizer step's passes over `pairs` (reference log-probabilities `rf`): adds the gradients of the mean RPO
/// loss over the pairs to the arena. Returns the policy's log-probabilities (chosen, rejected) of each pair.
pub fn step_passes(tr: &Trainer, po: &PrefOpts, pairs: &[&Pair], rf: &[(f64, f64)], step: u64) -> Result<Vec<(f64, f64)>> {
    let seqs = seqs(pairs);
    let (micros, place) = pack_pairs(&seqs, tr.tmax);
    // a pair over two micro-batches: the rejected response's log-probability before the chosen one's backward pass
    let mut lp: Vec<Option<f64>> = vec![None; seqs.len()];
    let mut early: Vec<Option<f64>> = vec![None; seqs.len()];
    let split: Vec<usize> = (0..pairs.len()).filter(|&i| place[2 * i].0 != place[2 * i + 1].0).collect();
    let mut pre: Vec<usize> = split.iter().map(|&i| place[2 * i + 1].0).collect();
    pre.dedup();
    for &m in &pre {
        let (rows, _) = tr.pass(&micros[m], false, 0.0, 0, 0)?;
        let sums = doc_sums(&micros[m], &rows);
        for &i in &split {
            if place[2 * i + 1].0 == m {
                early[2 * i + 1] = Some(-sums[place[2 * i + 1].1]);
            }
        }
    }
    let b = pairs.len() as f64;
    for (m, mb) in micros.iter().enumerate() {
        let in_mb: Vec<usize> = (0..seqs.len()).filter(|&s| place[s].0 == m).collect();
        let mut weigh = |rows: &[f32]| -> Result<Vec<f32>> {
            let sums = doc_sums(mb, rows);
            for &s in &in_mb {
                lp[s] = Some(-sums[place[s].1]);
            }
            let mut doc_w = vec![0f32; mb.docs.len()];
            for &s in &in_mb {
                let i = s / 2;
                let get = |k: usize| lp[k].or(early[k]).context("a pair's log-probability is missing");
                let (m_, _, _) = pair_terms(po, (get(2 * i)?, get(2 * i + 1)?), rf[i], pairs[i].chosen.loss_tokens());
                doc_w[place[s].1] = token_weight(po, m_, pairs[i].chosen.loss_tokens(), s % 2 == 0, b) as f32;
            }
            // each loss row's weight: its document's
            Ok(mb.doc_rows.iter().zip(&doc_w).flat_map(|(r, &w)| std::iter::repeat_n(w, r.len())).collect())
        };
        tr.pass_weighted(mb, step, m, &mut weigh)?;
    }
    lp.chunks(2).map(|c| Ok((c[0].context("no log-probability")?, c[1].context("no log-probability")?))).collect()
}

/// Preference training (see the module's description), on `training::Run`
pub fn train(model: &Path, data: &Path, out: Option<&Path>, o: Opts, to: TrainOpts, po: PrefOpts) -> Result<()> {
    let run = Run::load(model, data, out, o, &to, "pairs", encode_pairs)?;
    let (train, dev) = (&run.train, &run.dev);
    let lmax = train.iter().chain(dev).flat_map(|p| [&p.chosen, &p.rejected]).map(|e| e.ids.len()).max().context("no pairs")?;
    let tokens = |pairs: &[&Pair]| -> usize { pairs.iter().map(|p| p.chosen.ids.len() + p.rejected.ids.len()).sum() };
    let train_refs: Vec<&Pair> = train.iter().collect();
    let dev_refs: Vec<&Pair> = dev.iter().collect();
    println!("preference training (RPO: DPO beta {} + {} x NLL of the chosen response): {} pairs, {} tokens, {} dev \
              pairs; longest response with its prompt {lmax} tokens", po.beta, po.nll, train.len(), tokens(&train_refs),
             dev.len());
    let mut tr = run.trainer(lmax)?;

    // the reference policy: the starting point, its log-probabilities once
    let tr0 = Instant::now();
    let rf_train = logps(&tr, &train_refs)?;
    let rf_dev = logps(&tr, &dev_refs)?;
    println!("reference log-probabilities of {} responses in {:.1}s ({})", 2 * (train.len() + dev.len()),
             tr0.elapsed().as_secs_f64(), if run.init.is_some() { "the --init-adapter, frozen" } else { "the base model" });
    // at the start the policy is the reference: margin 0
    let start = (!dev.is_empty()).then(|| terms(&po, &dev_refs, &rf_dev, &rf_dev).dev());
    run.fit(&mut tr, &to, json!({"beta": po.beta, "nll_weight": po.nll}), start, |tr, chunk, step| {
        let pairs: Vec<&Pair> = chunk.iter().map(|&i| &train[i]).collect();
        let rf: Vec<(f64, f64)> = chunk.iter().map(|&i| rf_train[i]).collect();
        let lp = step_passes(tr, &po, &pairs, &rf, step as u64)?;
        let t = terms(&po, &pairs, &lp, &rf);
        Ok(Step {
            tokens: tokens(&pairs),
            log: json!({"loss": t.loss, "dpo": t.dpo, "nll": t.nll, "margin": t.margin, "accuracy": t.accuracy,
                        "reward_chosen": t.chosen, "reward_rejected": t.rejected}),
            text: format!("loss {:.4}  dpo {:.4}  nll {:.4}  margin {:+.4}  acc {:.2}", t.loss, t.dpo, t.nll, t.margin, t.accuracy),
        })
    }, |tr| Ok(terms(&po, &dev_refs, &logps(tr, &dev_refs)?, &rf_dev).dev()))
}

/// `qwenny dpocheck`: against tools/make_dpo_reference.py: the pairs' token ids and masks (the rows are in the
/// reference's metadata), the reference policy's log-probabilities (its "ref." adapter, loaded as --init-adapter
/// loads one), then from the policy ("init.") the RPO loss, its parts, the log-probabilities, the LoRA gradients and
/// one AdamW step. `packed`: all pairs in one micro-batch; else one sequence per micro-batch (every pair split).
pub fn dpocheck(model: &Path, reference: &Path, o: Opts, packed: bool) -> Result<()> {
    let bytes = fs::read(reference)?;
    let (_, meta) = SafeTensors::read_metadata(&bytes)?;
    let meta = meta.metadata().clone().context("the reference has no metadata (pairs)")?;
    let r = SafeTensors::deserialize(&bytes)?;
    let po = PrefOpts { beta: training::ref_f32(&r, "beta")?[0], nll: training::ref_f32(&r, "alpha")?[0] };
    let chat = Chat::load(model)?;
    let rows: Vec<(usize, String)> = meta.get("pairs").context("no pairs in the reference's metadata")?.lines().enumerate().map(|(i, l)| (i + 1, l.to_string())).collect();
    let pairs = encode_pairs(&chat, &rows, usize::MAX, "pairs")?;
    let (mut ids_ok, mut mask_ok) = (0, 0);
    for (i, p) in pairs.iter().enumerate() {
        for (k, e) in [("c", &p.chosen), ("r", &p.rejected)] {
            let ids = training::ref_i32(&r, &format!("ids_{k}_{i}"))?;
            let mask = training::ref_i32(&r, &format!("mask_{k}_{i}"))?;
            let same_ids = e.ids.len() == ids.len() && e.ids.iter().zip(&ids).all(|(a, b)| *a as i32 == *b);
            ids_ok += same_ids as usize;
            mask_ok += (same_ids && e.mask.iter().zip(&mask).all(|(a, b)| *a as i32 == *b)) as usize;
        }
    }
    println!("{} responses: ids identical for {ids_ok}, masks identical for {mask_ok}", 2 * pairs.len());
    ensure!(ids_ok == 2 * pairs.len() && mask_ok == ids_ok, "the pairs' tokens differ from the reference's");

    let refs: Vec<&Pair> = pairs.iter().collect();
    let lmax = seqs(&refs).iter().map(|e| e.ids.len()).max().unwrap();
    let tmax = if packed { seqs(&refs).iter().map(|e| e.ids.len()).sum() } else { lmax };
    let mut o = o;
    o.dropout = 0.0;
    o.layers = training::ref_i32(&r, "layers").map_or(0, |l| l[0] as usize);
    let mode = if o.f32 { "f32" } else if o.f16 { "f16" } else { "bf16" };
    let mut tr = Trainer::new(model, o, tmax, lmax, 0)?;
    let named = |prefix: &str| -> Result<std::collections::HashMap<String, Vec<f32>>> {
        Ok(training::ref_params(&r, prefix)?.into_iter().map(|(n, v)| (training::trainer_name(&n), v)).collect())
    };
    training::apply_init(&mut tr, &named("ref.")?)?;
    let rf = logps(&tr, &refs)?;
    let want_rf = training::ref_f32(&r, "ref_logp")?;
    let want_lp = training::ref_f32(&r, "logp")?;
    let worst = |got: &[(f64, f64)], want: &[f32]| got.iter().flat_map(|&(c, r)| [c, r]).zip(want)
        .map(|(a, &b)| (a - b as f64).abs() / (b as f64).abs().max(1.0)).fold(0f64, f64::max);
    println!("{} pairs ({}), {mode} build, beta {} alpha {}", pairs.len(),
             if packed { "packed in one pass" } else { "one sequence per pass: every pair split" }, po.beta, po.nll);
    println!("reference log-probabilities: worst relative error {:.2e}", worst(&rf, &want_rf));

    training::apply_init(&mut tr, &named("init.")?)?;
    tr.l.zero_grads()?;
    let lp = step_passes(&tr, &po, &refs, &rf, 0)?;
    let t = terms(&po, &refs, &lp, &rf);
    let g = |k: &str| -> Result<f64> { Ok(training::ref_f32(&r, k)?[0] as f64) };
    let (want_loss, want_dpo, want_nll, want_m) = (g("loss")?, g("dpo")?, g("nll")?, g("margin")?);
    println!("policy log-probabilities: worst relative error {:.2e}", worst(&lp, &want_lp));
    println!("loss {:.6} vs reference {want_loss:.6} (diff {:.2e}); DPO {:.6} vs {want_dpo:.6}, NLL {:.6} vs {want_nll:.6}, \
              margin {:+.6} vs {want_m:+.6}", t.loss, (t.loss - want_loss).abs(), t.dpo, t.nll, t.margin);
    if let Ok(trl) = g("trl_loss") {
        println!("TRL's DPOTrainer (sigmoid + sft) on the same pairs: loss {trl:.6}");
    }
    training::grads_and_step(&mut tr, &r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(n: usize) -> Example {
        Example { ids: vec![1; n], mask: (0..n).map(|i| (i >= n / 2) as u8).collect() }
    }

    #[test]
    fn a_pair_stays_in_one_micro_batch_when_it_fits() {
        let e = [ex(30), ex(40), ex(50), ex(20), ex(90), ex(80)];
        let s: Vec<&Example> = e.iter().collect();
        let (micros, place) = pack_pairs(&s, 100);
        // pair 0 (70) in micro 0; pair 1 (70) does not fit next to it: micro 1; pair 2 (170) is split: 90 in micro 2,
        // 80 in micro 3
        assert_eq!(place, vec![(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)]);
        assert_eq!(micros.iter().map(|m| m.tokens()).collect::<Vec<_>>(), vec![70, 70, 90, 80]);
    }

    #[test]
    fn doc_sums_add_each_documents_rows() {
        let mut mb = Micro::default();
        mb.push(&[1, 2, 3, 4], &[0, 0, 1, 1]);
        mb.push(&[5, 6, 7], &[0, 1, 1]);
        assert_eq!(mb.rows, vec![1, 2, 4, 5]);
        assert_eq!(doc_sums(&mb, &[1.0, 2.0, 4.0, 8.0]), vec![3.0, 12.0]);
    }

    /// The row weights are the loss's derivatives by each token's NLL (= minus by the response's log-probability)
    #[test]
    fn token_weights_are_the_losss_gradient() {
        let po = PrefOpts { beta: 0.1, nll: 1.0 };
        let (rf, n_c, b) = ((-120.0, -95.0), 37, 3.0);
        let loss = |lp: (f64, f64)| {
            let (_, dpo, nll) = pair_terms(&po, lp, rf, n_c);
            (dpo + po.nll as f64 * nll) / b
        };
        for lp in [(-110.0, -99.0), (-130.0, -80.0), (-120.0, -95.0)] {
            let (m, ..) = pair_terms(&po, lp, rf, n_c);
            let h = 1e-4;
            let dc = (loss((lp.0 + h, lp.1)) - loss((lp.0 - h, lp.1))) / (2.0 * h);
            let dr = (loss((lp.0, lp.1 + h)) - loss((lp.0, lp.1 - h))) / (2.0 * h);
            assert!((token_weight(&po, m, n_c, true, b) + dc).abs() < 1e-8, "{lp:?}");
            assert!((token_weight(&po, m, n_c, false, b) + dr).abs() < 1e-8, "{lp:?}");
        }
        // margin 0: -log sigmoid(0) = ln 2
        assert!((pair_terms(&po, rf, rf, n_c).1 - std::f64::consts::LN_2).abs() < 1e-12);
    }

    #[test]
    fn large_margins_stay_finite() {
        let po = PrefOpts { beta: 1.0, nll: 0.0 };
        let (_, dpo, _) = pair_terms(&po, (0.0, -2000.0), (0.0, 0.0), 1);
        assert!(dpo >= 0.0 && dpo < 1e-300);
        let (_, dpo, _) = pair_terms(&po, (-2000.0, 0.0), (0.0, 0.0), 1);
        assert!((dpo - 2000.0).abs() < 1e-9);
    }
}
