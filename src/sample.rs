//! Sampling parameters and the per-request random numbers; the sampling itself runs on the GPU (Model::sample) in
//! vLLM's order: repetition penalty, presence penalty, temperature, top-k, top-p.

#[derive(Clone, Debug)]
pub struct Params {
    pub temperature: f32,
    pub top_p: f32,
    /// 0: no top-k (the 1024 most likely tokens are still the candidates); at most 1024
    pub top_k: usize,
    pub repetition_penalty: f32,
    /// subtracted from the logit of every token the request generated so far (OpenAI's and vLLM's presence_penalty;
    /// 0: off)
    pub presence_penalty: f32,
    pub seed: u64,
}

impl Params {
    /// The greedy token can come from the step's argmax: no penalty changes the logits and no sampling.
    pub fn greedy(&self) -> bool {
        (self.temperature <= 0.0 || self.top_k == 1) && self.repetition_penalty == 1.0 && self.presence_penalty == 0.0
    }

    /// The GPU sampler's row: temperature, top_k, top_p, repetition penalty, two draws from `rng` (only when
    /// sampling; the second only with a draft), the draft token (-1: none) and the presence penalty
    pub fn row(&self, rng: &mut Rng, draft: Option<u32>) -> [f32; 8] {
        let sampling = self.temperature > 0.0 && self.top_k != 1;
        let u1 = if sampling { rng.next_f32() } else { 0.0 };
        let u2 = if sampling && draft.is_some() { rng.next_f32() } else { 0.0 };
        let d = draft.map_or(-1.0, |d| d as f32);
        [self.temperature, self.top_k.min(1024) as f32, self.top_p, self.repetition_penalty, u1, u2, d, self.presence_penalty]
    }
}

impl Params {
    /// Whether the reply is sampled (else the argmax: drafts stay argmaxes)
    pub fn sampling(&self) -> bool {
        self.temperature > 0.0 && self.top_k != 1
    }

    /// The draft sampler's row (Engine::draft): this row's sampling without draws or draft; `presence`:
    /// whether the presence penalty applies yet (the request generated a token)
    pub fn draft_row(&self, presence: bool) -> [f32; 8] {
        [self.temperature, self.top_k.min(1024) as f32, self.top_p, self.repetition_penalty, 0.0, 0.0, -1.0,
         if presence { self.presence_penalty } else { 0.0 }]
    }
}

/// The draft sampler's row of a greedy reply (Engine::draft): top_k 1 takes the argmax; `t` the draft's index in its
/// chain
pub fn argmax_draft_row(t: usize) -> [f32; 8] {
    [1.0, 1.0, 1.0, 1.0, 0.5, t as f32, -1.0, 0.0]
}

/// Mark a sampler row's draft as sampled from the draft layer's distribution (kept on the device): -d - 2
pub fn mark_sampled(row: &mut [f32; 8]) {
    if row[6] >= 0.0 {
        row[6] = -row[6] - 2.0;
    }
}

/// xorshift64*
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }

    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// One row of the sampler on the host, exactly as the CUDA kernel `sample_rows` does it (`samplecheck` compares with
/// it): repetition penalty for the tokens `seen` reports, presence penalty for those `generated` reports, temperature,
/// top-k (0: the 1024 most likely; ties to the lower index), top-p, then the draw with the row's uniform numbers. With a draft d: accepted with probability p(d), else a draw
/// from p without d. Returns (token, draft accepted).
pub fn sample_row(logits: &[f32], seen: impl Fn(u32) -> bool, generated: impl Fn(u32) -> bool, row: &[f32; 8]) -> (u32, bool) {
    let (temp, top_k, top_p, rep, u1, u2, pres) = (row[0], row[1] as usize, row[2], row[3], row[4], row[5], row[7]);
    let draft = (row[6] >= 0.0).then_some(row[6] as u32);
    let vals: Vec<f32> = logits.iter().enumerate().map(|(i, &l)| {
        let mut v = l;
        if rep != 1.0 && seen(i as u32) {
            v = if v > 0.0 { v / rep } else { v * rep };
        }
        if pres != 0.0 && generated(i as u32) {
            v -= pres;
        }
        v
    }).collect();
    let val = |i: usize| vals[i];
    let greedy = temp <= 0.0 || top_k == 1;
    let k = if greedy { 1 } else if top_k > 0 { top_k.min(logits.len()) } else { 1024 };
    // the k largest by value, ties to the lower index
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let cmp = |a: &u32, b: &u32| val(*b as usize).total_cmp(&val(*a as usize)).then(a.cmp(b));
    idx.select_nth_unstable_by(k - 1, cmp);
    idx.truncate(k);
    idx.sort_unstable_by(cmp);
    if greedy {
        return (idx[0], draft == Some(idx[0]));
    }
    let vmax = val(idx[0] as usize);
    let p: Vec<f32> = idx.iter().map(|&i| ((val(i as usize) - vmax) / temp).exp()).collect();
    let cum: Vec<f32> = p.iter().scan(0.0, |c, &x| { *c += x; Some(*c) }).collect();
    let total = cum[k - 1];
    // the smallest prefix with mass >= top_p (at least one token)
    let keep = if top_p < 1.0 { cum.iter().position(|&c| c / total >= top_p).map_or(k, |i| i + 1) } else { k };
    let mass = cum[keep - 1];
    let at_d = draft.and_then(|d| idx[..keep].iter().position(|&t| t == d));
    let pd = at_d.map_or(0.0, |a| p[a] / mass);
    if let Some(d) = draft
        && u1 < pd
    {
        return (d, true);
    }
    // draw from the kept tokens without the rejected draft
    let target = if draft.is_some() { u2 * (mass - pd * mass) } else { u1 * mass };
    let mut r = (0..keep)
        .find(|&i| Some(i) != at_d && cum[i] - if at_d.is_some_and(|a| i >= a) { pd * mass } else { 0.0 } >= target)
        .unwrap_or(keep - 1);
    if Some(r) == at_d {
        r = if r + 1 < keep { r + 1 } else { r.saturating_sub(1) };
    }
    (idx[r], false)
}

/// sample_row for a draft d drawn from `q` (token, probability; ascending ids), exactly as `sample_rows` does it for
/// a sampled draft: accepted with probability min(1, p(d) / q(d)) (u1), else a draw from max(0, p - q) normalized
/// (u2). Returns (token, draft accepted).
pub fn sample_row_q(logits: &[f32], seen: impl Fn(u32) -> bool, generated: impl Fn(u32) -> bool, row: &[f32; 8],
                    q: &[(u32, f32)]) -> (u32, bool) {
    let (p, idx) = kept_dist(logits, &seen, &generated, row);
    let d = row[6] as u32;
    let qof = |t: u32| q.binary_search_by_key(&t, |e| e.0).map_or(0.0, |i| q[i].1);
    let pd = idx.iter().position(|&t| t == d).map_or(0.0, |a| p[a]);
    if row[4] * qof(d) < pd {
        return (d, true);
    }
    let r: Vec<f32> = idx.iter().zip(&p).map(|(&t, &pt)| (pt - qof(t)).max(0.0)).collect();
    let cum: Vec<f32> = r.iter().scan(0.0, |c, &x| { *c += x; Some(*c) }).collect();
    let total = *cum.last().unwrap_or(&0.0);
    if total <= 0.0 {
        return (idx[0], false);
    }
    let target = row[5] * total;
    let at = (0..r.len()).find(|&i| r[i] > 0.0 && cum[i] >= target).or_else(|| (0..r.len()).rev().find(|&i| r[i] > 0.0));
    (idx[at.unwrap_or(0)], false)
}

/// The kept tokens of a sampling row (penalties, temperature, top-k, top-p as sample_row) and their probabilities
pub fn kept_dist(logits: &[f32], seen: &impl Fn(u32) -> bool, generated: &impl Fn(u32) -> bool, row: &[f32; 8])
-> (Vec<f32>, Vec<u32>) {
    let (temp, top_k, top_p, rep, pres) = (row[0], row[1] as usize, row[2], row[3], row[7]);
    let vals: Vec<f32> = logits.iter().enumerate().map(|(i, &l)| {
        let mut v = l;
        if rep != 1.0 && seen(i as u32) {
            v = if v > 0.0 { v / rep } else { v * rep };
        }
        if pres != 0.0 && generated(i as u32) {
            v -= pres;
        }
        v
    }).collect();
    let k = if top_k > 0 { top_k.min(logits.len()) } else { 1024 };
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let cmp = |a: &u32, b: &u32| vals[*b as usize].total_cmp(&vals[*a as usize]).then(a.cmp(b));
    idx.select_nth_unstable_by(k - 1, cmp);
    idx.truncate(k);
    idx.sort_unstable_by(cmp);
    let vmax = vals[idx[0] as usize];
    let p: Vec<f32> = idx.iter().map(|&i| ((vals[i as usize] - vmax) / temp).exp()).collect();
    let cum: Vec<f32> = p.iter().scan(0.0, |c, &x| { *c += x; Some(*c) }).collect();
    let total = cum[k - 1];
    let keep = if top_p < 1.0 { cum.iter().position(|&c| c / total >= top_p).map_or(k, |i| i + 1) } else { k };
    let mass = cum[keep - 1];
    (p[..keep].iter().map(|&x| x / mass).collect(), idx[..keep].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sampled draft d ~ q, accepted or resampled by sample_row_q, gives tokens distributed as p (the row's own
    /// sampling): the distribution over a grid of draws, for a q unlike p
    #[test]
    fn sampled_drafts_follow_p() {
        let logits: Vec<f32> = (0..64).map(|i| ((i * 37) % 11) as f32 * 0.4 - (i as f32) * 0.02).collect();
        let row = [0.9, 20.0, 0.9, 1.0, 0.0, 0.0, -1.0, 0.0];
        let (p, idx) = kept_dist(&logits, &|_| false, &|_| false, &row);
        // q: a different distribution over part of p's tokens and others
        let qlog: Vec<f32> = (0..64).map(|i| ((i * 13) % 7) as f32 * 0.5).collect();
        let (qp, qidx) = kept_dist(&qlog, &|_| false, &|_| false, &[1.2, 30.0, 1.0, 1.0, 0.0, 0.0, -1.0, 0.0]);
        let mut q: Vec<(u32, f32)> = qidx.iter().copied().zip(qp.iter().copied()).collect();
        q.sort_by_key(|e| e.0);
        let mut got = vec![0f64; 64];
        let n = 400;
        for (&d, &qd) in qidx.iter().zip(&qp) {
            for a in 0..n {
                for b in 0..n {
                    let mut r = row;
                    (r[4], r[5], r[6]) = ((a as f32 + 0.5) / n as f32, (b as f32 + 0.5) / n as f32, d as f32);
                    let (t, _) = sample_row_q(&logits, |_| false, |_| false, &r, &q);
                    got[t as usize] += qd as f64 / (n * n) as f64;
                }
            }
        }
        let mut want = vec![0f64; 64];
        idx.iter().zip(&p).for_each(|(&t, &x)| want[t as usize] = x as f64);
        let tv: f64 = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).sum::<f64>() / 2.0;
        assert!(tv < 0.01, "total variation {tv}");
    }
}
