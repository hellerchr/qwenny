//! The gated delta rule of one DeltaNet layer and sequence segment, per value head h (state S [Dv][Dk], f32):
//!   S_t = e_t S_{t-1} + delta_t k_t^T,  delta_t = b_t (v_t - e_t S_{t-1} k_t),  o_t = S_t q_t
//! where e_t = exp(lg_t) is the decay and b_t = beta. Two exact forms:
//! - `token`: one token, as CUDA's delta_rule computes it (o_t = e_t S_{t-1} q_t + delta_t (k_t . q_t)); decode steps,
//!   drafts and the rollback replay, so that a replayed state is bit for bit the state the step reached.
//! - `chunked`: the same recurrence over chunks of CH tokens (the WY form of FLA's chunked gated delta rule): within a
//!   chunk the deltas solve a unit lower triangular system that does not depend on the state, so all chunks of a
//!   prompt are set up at once with matrix products and only CH-token steps between chunks remain sequential. All in
//!   f32 (TF32 is off: weights::init); decays enter only as exp of differences of cumulative log decays (<= 0).
use super::weights::{LK_HEADS, LKD, MResult};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, ops};

/// Tokens per chunk of `chunked`
pub const CH: i32 = 64;
/// Leaf blocks of the triangular inverse
const LEAF: i32 = 16;

/// Inputs of one token: per key head kq [Lk][Dk][2] (k and q side by side) and k . q [Lk]; per value head v [Hv][Dv],
/// e and b [Hv]. The value heads of a key head are adjacent (Hv = Lk * G).
pub struct Tok<'a> {
    pub kq: &'a Array,
    pub v: &'a Array,
    pub e: &'a Array,
    pub b: &'a Array,
    pub dot: &'a Array,
}

/// q or k [len][Lk][Dk] for each value head [len][Hv][Dk]: every value head reads its key head (adjacent groups)
pub fn repeat(x: &Array, lv: i32) -> MResult<Array> {
    let len = x.shape()[0];
    ops::broadcast_to(ops::expand_dims(x, 2)?, &[len, LK_HEADS, lv / LK_HEADS, LKD])?.reshape(&[len, lv, LKD])
}

/// `front` zero rows in front and `back` behind of x [len][..]
pub fn pad_rows(x: &Array, front: i32, back: i32) -> MResult<Array> {
    if front == 0 && back == 0 {
        return Ok(x.clone());
    }
    let mut sh = x.shape().to_vec();
    let mut parts = Vec::with_capacity(3);
    if front > 0 {
        sh[0] = front;
        parts.push(ops::zeros::<f32>(&sh)?);
    }
    parts.push(x.clone());
    if back > 0 {
        sh[0] = back;
        parts.push(ops::zeros::<f32>(&sh)?);
    }
    ops::concatenate(&parts, 0)
}

/// Row i of x (a view: integer indices are gathers in mlx-rs)
pub fn row(x: &Array, i: i32) -> MResult<Array> {
    x.index(i..i + 1).squeeze_axes(&[0])
}

/// Index i of x's last axis (a view)
fn last(x: &Array, i: i32) -> MResult<Array> {
    x.index((std::ops::RangeFull, std::ops::RangeFull, std::ops::RangeFull, i..i + 1)).squeeze_axes(&[-1])
}

/// One token: (the new state, o [Hv][Dv]). The key heads' vectors broadcast over their value heads.
pub fn token(s: &Array, t: Tok) -> MResult<(Array, Array)> {
    let (hv, dv, dk) = (s.shape()[0], s.shape()[1], s.shape()[2]);
    let lk = t.kq.shape()[0];
    let g = hv / lk;
    let s4 = s.reshape(&[lk, g, dv, dk])?;
    let m = ops::matmul(&s4, ops::expand_dims(t.kq, 1)?)?; // [Lk][G][Dv][2]: S k, S q
    let (dkk, dq) = (last(&m, 0)?, last(&m, 1)?);
    let e = t.e.reshape(&[lk, g, 1])?;
    let delta = ops::multiply(ops::subtract(t.v.reshape(&[lk, g, dv])?, ops::multiply(&e, &dkk)?)?, t.b.reshape(&[lk, g, 1])?)?;
    let o = ops::add(ops::multiply(&e, &dq)?, ops::multiply(&delta, t.dot.reshape(&[lk, 1, 1])?)?)?;
    let k = t.kq.index((.., .., 0..1)).reshape(&[lk, 1, 1, dk])?;
    let s = ops::add(ops::multiply(e.reshape(&[lk, g, 1, 1])?, &s4)?, ops::multiply(ops::expand_dims(&delta, -1)?, &k)?)?;
    Ok((s.reshape(&[hv, dv, dk])?, o.reshape(&[hv, dv])?))
}

/// n tokens (a multiple of CH) from state s0: q, k [n][Hv][Dk], v [n][Hv][Dv], lg (log decay), b [n][Hv]. Returns o
/// [n][Hv][Dv], the state after the last token and the state after `at` tokens (`at` a multiple of CH, <= n). Zero
/// tokens (k = v = b = lg = 0) leave the state as it is: padding.
pub fn chunked(s0: &Array, q: &Array, k: &Array, v: &Array, lg: &Array, b: &Array, at: i32)
               -> MResult<(Array, Array, Array)> {
    let (n, hv, dk, dv) = (q.shape()[0], q.shape()[1], q.shape()[2], v.shape()[2]);
    let nc = n / CH;
    // heads first: [Hv][nc][CH][d], [Hv][nc][CH]
    let heads = |x: &Array, d: i32| ops::transpose_axes(x.reshape(&[nc, CH, hv, d])?, &[2, 0, 1, 3]);
    let heads3 = |x: &Array| ops::transpose_axes(x.reshape(&[nc, CH, hv])?, &[2, 0, 1]);
    let (q, k, v, lg, b) = (heads(q, dk)?, heads(k, dk)?, heads(v, dv)?, heads3(lg)?, heads3(b)?);
    let g = ops::cumsum(&lg, -1, None, true)?; // G_i, cumulative within the chunk
    let gam = ops::exp(&g)?;
    let diff = ops::subtract(ops::expand_dims(&g, -1)?, ops::expand_dims(&g, -2)?)?; // [i][j]: G_i - G_j
    let strict = ops::tri::<bool>(CH, None, Some(-1))?;
    let lower = ops::tri::<bool>(CH, None, Some(0))?;
    let zero = Array::from_f32(0.0);
    let decay = |mask: &Array| ops::exp(ops::select(mask, &diff, &zero)?);
    let kt = ops::swap_axes(&k, -1, -2)?;
    let col = |x: &Array| ops::expand_dims(x, -1);
    let kb = ops::multiply(&k, col(&b)?)?;
    // A_ij = b_i (k_i . k_j) exp(G_i - G_j), j < i; deltas: (I + A) D = diag(b) (V - diag(gam) K S0^T)
    let a = ops::select(&strict, ops::multiply(ops::matmul(&kb, &kt)?, decay(&strict)?)?, &zero)?;
    let t = inv_unit_lower(&a)?;
    let u = ops::matmul(&t, ops::multiply(&v, col(&b)?)?)?; // D = U - W S0^T
    let w = ops::matmul(&t, ops::multiply(&kb, col(&gam)?)?)?;
    // o_i = gam_i S0 q_i + sum_{j <= i} (q_i . k_j) exp(G_i - G_j) d_j
    let p = ops::select(&lower, ops::multiply(ops::matmul(&q, &kt)?, decay(&lower)?)?, &zero)?;
    let qg = ops::multiply(&q, col(&gam)?)?;
    let last = g.index((.., .., CH - 1..CH)); // [Hv][nc][1]
    let kd = ops::multiply(&k, col(&ops::exp(ops::subtract(&last, &g)?)?)?)?; // k_j exp(G_C - G_j)
    let gc = ops::exp(&last)?;
    let mut s = s0.clone();
    let mut s_at = if at == 0 { Some(s0.clone()) } else { None };
    let mut outs = Vec::with_capacity(nc as usize);
    for c in 0..nc {
        let st = ops::swap_axes(&s, -1, -2)?; // [Hv][Dk][Dv]
        let at_c = |x: &Array| x.index((.., c..c + 1)).squeeze_axes(&[1]);
        let d = ops::subtract(at_c(&u)?, ops::matmul(at_c(&w)?, &st)?)?; // [Hv][CH][Dv]
        outs.push(ops::add(ops::matmul(at_c(&qg)?, &st)?, ops::matmul(at_c(&p)?, &d)?)?);
        let dt = ops::swap_axes(&d, -1, -2)?; // [Hv][Dv][CH]
        s = ops::add(ops::multiply(at_c(&gc)?.reshape(&[hv, 1, 1])?, &s)?, ops::matmul(&dt, at_c(&kd)?)?)?;
        if (c + 1) * CH == at {
            s_at = Some(s.clone());
        }
    }
    let o = ops::transpose_axes(ops::stack(&outs, 1)?, &[1, 2, 0, 3])?.reshape(&[n, hv, dv])?;
    Ok((o, s, s_at.expect("`at` a multiple of CH within the tokens")))
}

/// (I + A)^-1 for strictly lower triangular A [..][C][C]: forward substitution on LEAF x LEAF diagonal blocks (T = I - A T,
/// row i final after i iterations; the rows not yet final multiply exact zeros of A), then merged pairwise:
/// [[T1, 0], [A21, T2]] -> [[T1, 0], [-T2 A21 T1, T2]].
fn inv_unit_lower(a: &Array) -> MResult<Array> {
    let sh = a.shape().to_vec();
    let (r, c) = (sh.len() - 2, sh[sh.len() - 1]);
    let leaf = LEAF.min(c);
    let nb = c / leaf;
    let block = |x: &Array, rows: std::ops::Range<i32>, cols: std::ops::Range<i32>| x.index((.., .., rows, cols));
    debug_assert!(r == 2);
    let leaves: Vec<Array> = (0..nb).map(|i| block(a, i * leaf..(i + 1) * leaf, i * leaf..(i + 1) * leaf)).collect();
    let al = ops::stack(&leaves, 2)?; // [..][nb][leaf][leaf]
    let eye = ops::eye::<f32>(leaf, None, None)?;
    let mut t = ops::broadcast_to(&eye, al.shape())?;
    for _ in 1..leaf {
        t = ops::subtract(&eye, ops::matmul(&al, &t)?)?;
    }
    let mut s = leaf;
    while s < c {
        let pairs = c / (2 * s);
        let tt = t.reshape(&[sh[0], sh[1], pairs, 2, s, s])?;
        let half = |i: i32| tt.index((.., .., .., i..i + 1)).squeeze_axes(&[3]);
        let (t1, t2) = (half(0)?, half(1)?);
        let a21: Vec<Array> = (0..pairs).map(|p| {
            let o = p * 2 * s;
            block(a, o + s..o + 2 * s, o..o + s)
        }).collect();
        let a21 = ops::stack(&a21, 2)?;
        let t21 = ops::negative(ops::matmul(ops::matmul(&t2, &a21)?, &t1)?)?;
        let top = ops::concatenate(&[&t1, &ops::zeros_like(&t1)?], -1)?;
        let bottom = ops::concatenate(&[&t21, &t2], -1)?;
        t = ops::concatenate(&[top, bottom], -2)?; // [..][pairs][2s][2s]
        s *= 2;
    }
    t.reshape(&sh)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// uniform [lo, hi) values from a fixed seed (mlx_rs::random does not link on Linux)
    fn values(shape: &[i32], lo: f32, hi: f32, seed: &mut u64) -> Array {
        let n: i32 = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + (hi - lo) * ((*seed >> 40) as f32 / (1u64 << 24) as f32)
        }).collect();
        Array::from_slice(&v, shape)
    }

    /// `chunked` and `token` compute the same recurrence (f32 rounding apart), and the triangular inverse inverts
    #[test]
    fn chunked_matches_tokens() {
        super::super::weights::init().unwrap();
        // the CPU device: unit tests run without a GPU
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        let (n, lk, hv, d) = (2 * CH, 2, 4, 16);
        let unit = |x: Array| -> Array {
            let ss = ops::sum_axis(ops::square(&x).unwrap(), -1, true).unwrap();
            ops::multiply(&x, ops::rsqrt(ss).unwrap()).unwrap()
        };
        let mut seed = 7;
        let q1 = unit(values(&[n, lk, d], -1.0, 1.0, &mut seed));
        let k1 = unit(values(&[n, lk, d], -1.0, 1.0, &mut seed));
        let rep = |x: &Array| ops::broadcast_to(ops::expand_dims(x, 2).unwrap(), &[n, lk, hv / lk, d]).unwrap().reshape(&[n, hv, d]).unwrap();
        let (q, k) = (rep(&q1), rep(&k1));
        let v = values(&[n, hv, d], -1.0, 1.0, &mut seed);
        let lg = values(&[n, hv], -0.5, 0.0, &mut seed);
        let b = values(&[n, hv], 0.0, 1.0, &mut seed);
        let s0 = values(&[hv, d, d], -1.0, 1.0, &mut seed);
        let (o, s, s_at) = chunked(&s0, &q, &k, &v, &lg, &b, CH).unwrap();
        let e = ops::exp(&lg).unwrap();
        let dot = ops::sum_axis(ops::multiply(&q1, &k1).unwrap(), -1, false).unwrap();
        let kq = ops::stack(&[&k1, &q1], -1).unwrap();
        let mut st = s0.clone();
        let mut worst = 0f32;
        let max = |a: &Array, b: &Array| ops::max(ops::abs(ops::subtract(a, b).unwrap()).unwrap(), None).unwrap().item_exact::<f32>();
        for i in 0..n {
            let r = |x: &Array| row(x, i).unwrap();
            let (s2, oi) = token(&st, Tok { kq: &r(&kq), v: &r(&v), e: &r(&e), b: &r(&b), dot: &r(&dot) }).unwrap();
            st = s2;
            worst = worst.max(max(&oi, &r(&o)));
            if i + 1 == CH {
                assert!(max(&st, &s_at) < 1e-4, "state at CH");
            }
        }
        eprintln!("chunked vs. token by token: max abs output diff {worst}, end state {}", max(&st, &s));
        assert!(worst < 1e-4, "outputs differ by {worst}");
        assert!(max(&st, &s) < 1e-4, "end states differ");
    }
}
