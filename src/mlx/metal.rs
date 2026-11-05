//! Metal custom operations for skinny verify products. All reductions accumulate in f32.
use super::weights::MResult;
use mlx_rs::{Array, Dtype, Stream, error::Exception};
use mlx_sys as sys;
use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::{CStr, CString},
};

fn check(status: i32, op: &str) -> MResult<()> {
    if status == 0 { Ok(()) } else { Err(Exception::custom(format!("{op}: MLX status {status}"))) }
}

struct Kernel(sys::mlx_fast_metal_kernel);
impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { sys::mlx_fast_metal_kernel_free(self.0) };
    }
}
impl Kernel {
    fn new_named(name: &CStr, inputs: &[&CStr], outputs: &[&CStr], source: &str) -> Self {
        let source = CString::new(source).unwrap();
        let mut names: Vec<_> = inputs.iter().map(|n| n.as_ptr()).collect();
        let mut out: Vec<_> = outputs.iter().map(|n| n.as_ptr()).collect();
        // SAFETY: the C API copies the strings and returns an owning kernel handle.
        unsafe {
            let ins = sys::mlx_vector_string_new_data(names.as_mut_ptr(), names.len());
            let outs = sys::mlx_vector_string_new_data(out.as_mut_ptr(), out.len());
            let k = sys::mlx_fast_metal_kernel_new(
                name.as_ptr(),
                ins,
                outs,
                source.as_ptr(),
                c"#include <metal_simdgroup_matrix>\n".as_ptr(),
                true,
                false,
            );
            sys::mlx_vector_string_free(ins);
            sys::mlx_vector_string_free(outs);
            Kernel(k)
        }
    }
    fn apply_many(
        &self,
        a: &[Array],
        shapes: &[&[i32]],
        grid: [i32; 3],
        group: [i32; 3],
        templates: &[(&CStr, i32)],
        dtype: Dtype,
    ) -> MResult<Vec<Array>> {
        let raw: Vec<_> = a.iter().map(Array::as_ptr).collect();
        let stream = Stream::thread_local_or_default();
        // SAFETY: input handles remain alive through the call. The result array takes ownership of the
        // independent handle returned by vector_array_get. Every temporary C handle is freed on failure too.
        unsafe {
            let cfg = sys::mlx_fast_metal_kernel_config_new();
            let ins = sys::mlx_vector_array_new_data(raw.as_ptr(), raw.len());
            let mut outs = sys::mlx_vector_array_new();
            let result = (|| {
                for shape in shapes {
                    check(
                        sys::mlx_fast_metal_kernel_config_add_output_arg(
                            cfg,
                            shape.as_ptr(),
                            shape.len(),
                            dtype.into(),
                        ),
                        "output",
                    )?;
                }
                check(sys::mlx_fast_metal_kernel_config_set_grid(cfg, grid[0], grid[1], grid[2]), "grid")?;
                check(
                    sys::mlx_fast_metal_kernel_config_set_thread_group(cfg, group[0], group[1], group[2]),
                    "threadgroup",
                )?;
                for &(name, v) in templates {
                    check(sys::mlx_fast_metal_kernel_config_add_template_arg_int(cfg, name.as_ptr(), v), "template")?;
                }
                check(sys::mlx_fast_metal_kernel_apply(&mut outs, self.0, ins, cfg, stream.as_ptr()), "verify kernel")?;
                let mut results = Vec::with_capacity(shapes.len());
                for i in 0..shapes.len() {
                    let mut out = sys::mlx_array_new();
                    let status = sys::mlx_vector_array_get(&mut out, outs, i);
                    if status != 0 {
                        sys::mlx_array_free(out);
                        check(status, "kernel output")?;
                    }
                    results.push(Array::from_ptr(out));
                }
                Ok(results)
            })();
            sys::mlx_vector_array_free(outs);
            sys::mlx_vector_array_free(ins);
            sys::mlx_fast_metal_kernel_config_free(cfg);
            result
        }
    }
}

thread_local! { static VERIFY: RefCell<HashMap<bool, Kernel>> = RefCell::new(HashMap::new()); }

pub fn verify(x: &Array, a: &[Array]) -> MResult<Array> {
    let (m, n, cols) = (x.shape()[0], a[0].shape()[0], x.shape()[1]);
    let bits = 32 * a[0].shape()[1] / cols;
    let sg8 = m <= 8 && bits == 4 && a[1].shape()[1] % 8 == 0 && n % 16 == 0;
    // the fastest tile counts and threadgroup splits measured on an M2 Pro
    let (nt, s) = if sg8 {
        (
            4,
            if n >= 65536 {
                4
            } else if cols >= 16384 {
                2
            } else if (16384..32768).contains(&n) {
                4
            } else {
                8
            },
        )
    } else {
        (if bits == 8 { 2 } else { 4 }, if m <= 8 && bits == 4 { 8 } else { 4 })
    };
    verify_geometry(x, a, sg8, nt, s)
}

fn verify_geometry(x: &Array, a: &[Array], sg8: bool, nt: i32, s: i32) -> MResult<Array> {
    let (m, n) = (x.shape()[0], a[0].shape()[0]);
    let rt = (m + 7) / 8;
    let bits = 32 * a[0].shape()[1] / x.shape()[1];
    VERIFY.with(|ks| {
        let mut k = ks.borrow_mut();
        let source = if sg8 { include_str!("verify_sg8.metal") } else { include_str!("verify.metal") };
        let name = if sg8 { c"qwenny_verify_sg8" } else { c"qwenny_verify" };
        let k =
            k.entry(sg8).or_insert_with(|| Kernel::new_named(name, &[c"X", c"Wq", c"SC", c"BI"], &[c"OUT"], source));
        let shape = [m, n];
        let mut results = k.apply_many(
            &[x.clone(), a[0].clone(), a[1].clone(), a[2].clone()],
            &[&shape],
            [(n + 8 * nt - 1) / (8 * nt) * s * 32, (m + 8 * rt - 1) / (8 * rt), 1],
            [s * 32, 1, 1],
            &[(c"RT", rt), (c"NT", nt), (c"S", s), (c"B", bits)],
            Dtype::Float32,
        )?;
        Ok(results.remove(0))
    })
}

thread_local! { static QMV: RefCell<Option<Kernel>> = const { RefCell::new(None) }; }

pub fn qmv(x: &Array, a: &[Array]) -> MResult<Array> {
    let n = a[0].shape()[0];
    QMV.with(|cell| {
        let mut cell = cell.borrow_mut();
        let k = cell.get_or_insert_with(|| {
            Kernel::new_named(c"qwenny_draft_qmv", &[c"X", c"Wq", c"SC", c"BI"], &[c"OUT"], include_str!("qmv.metal"))
        });
        let mut out = k.apply_many(
            &[x.clone(), a[0].clone(), a[1].clone(), a[2].clone()],
            &[&[1, n]],
            [((n + 7) / 8) * 64, 1, 1],
            [64, 1, 1],
            &[],
            Dtype::Float32,
        )?;
        Ok(out.remove(0))
    })
}

thread_local! { static DELTA: RefCell<HashMap<bool, Kernel>> = RefCell::new(HashMap::new()); }

pub fn delta(
    s: &Array,
    kq: &Array,
    v: &Array,
    e: &Array,
    beta: &Array,
    dot: &Array,
    at: i32,
) -> MResult<(Array, Array, Array)> {
    delta_with(s, kq, v, e, beta, dot, at, s.shape()[2] == 128)
}

#[allow(clippy::too_many_arguments)]
fn delta_with(
    s: &Array,
    kq: &Array,
    v: &Array,
    e: &Array,
    beta: &Array,
    dot: &Array,
    at: i32,
    eight: bool,
) -> MResult<(Array, Array, Array)> {
    let hv = s.shape()[0];
    let dv = s.shape()[1];
    let len = e.shape()[0];
    DELTA.with(|cell| {
        let mut cell = cell.borrow_mut();
        let name = if eight { c"qwenny_delta8" } else { c"qwenny_delta" };
        let source = if eight { include_str!("delta8.metal") } else { include_str!("delta.metal") };
        let k = cell.entry(eight).or_insert_with(|| {
            Kernel::new_named(
                name,
                &[c"ST", c"KQ", c"V", c"E", c"BETA", c"DOT", c"AT"],
                &[c"SF", c"CK", c"OUT"],
                source,
            )
        });
        let at = Array::from_slice(&[at], &[1]);
        let mut r = k.apply_many(
            &[s.clone(), kq.clone(), v.clone(), e.clone(), beta.clone(), dot.clone(), at],
            &[s.shape(), s.shape(), &[len, hv, dv]],
            [if eight { ((dv + 15) / 16) * 128 } else { dv * 32 }, hv, 1],
            [128, 1, 1],
            &[],
            Dtype::Float32,
        )?;
        let o = r.pop().unwrap();
        let ck = r.pop().unwrap();
        Ok((r.pop().unwrap(), ck, o))
    })
}

thread_local! { static REPLAY: RefCell<Option<Kernel>> = const { RefCell::new(None) }; }

pub fn delta_replay(s: &Array, old: [&Array; 4], replay: i32, current: [&Array; 5]) -> MResult<(Array, Array, Array)> {
    let (hv, dv, len) = (s.shape()[0], s.shape()[1], current[2].shape()[0]);
    REPLAY.with(|cell| {
        let mut cell = cell.borrow_mut();
        let k = cell.get_or_insert_with(|| {
            Kernel::new_named(
                c"qwenny_delta_replay",
                &[c"ST", c"RKQ", c"RV", c"RE", c"RBETA", c"KQ", c"V", c"E", c"BETA", c"DOT", c"RP"],
                &[c"SF", c"CK", c"OUT"],
                include_str!("delta_replay.metal"),
            )
        });
        let rp = Array::from_slice(&[replay], &[1]);
        let inputs: Vec<Array> = [s].into_iter().chain(old).chain(current).cloned().chain([rp]).collect();
        let mut r = k.apply_many(
            &inputs,
            &[s.shape(), s.shape(), &[len, hv, dv]],
            [((dv + 15) / 16) * 128, hv, 1],
            [128, 1, 1],
            &[],
            Dtype::Float32,
        )?;
        let out = r.pop().unwrap();
        let ck = r.pop().unwrap();
        Ok((r.pop().unwrap(), ck, out))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn verify_matches_affine_reference() -> anyhow::Result<()> {
        mlx_rs::with_device(mlx_rs::Device::gpu(), verify_reference)
    }

    fn verify_reference() -> anyhow::Result<()> {
        super::super::weights::init()?;
        for (m, n, k, group, bits) in [
            (3, 35, 128, 64, 4),
            (5, 64, 256, 128, 4),
            (8, 33, 128, 128, 4),
            (16, 32, 64, 64, 4),
            (8, 35, 256, 128, 8),
            (5, 64, 1024, 128, 4),
            (5, 48, 1024, 128, 4),
            (1, 35, 1024, 128, 4),
        ] {
            let (m, n, k, group) = (m as usize, n as usize, k as usize, group as usize);
            let pack = 32 / bits;
            let xv: Vec<f16> = (0..m * k).map(|i| f16::from_f32(((i * 17 % 101) as f32 - 50.0) / 53.0)).collect();
            let q: Vec<u32> =
                (0..n * k / pack).map(|i| (i as u32).wrapping_mul(7919).wrapping_add(0x1949abcd)).collect();
            let sc: Vec<f16> = (0..n * k / group).map(|i| f16::from_f32((i % 11 + 1) as f32 / 97.0)).collect();
            let bi: Vec<f16> = sc.iter().map(|s| f16::from_f32(-8.0 * s.to_f32())).collect();
            let x = Array::from_slice(&xv, &[m as i32, k as i32]);
            let a = [
                Array::from_slice(&q, &[n as i32, (k / pack) as i32]),
                Array::from_slice(&sc, &[n as i32, (k / group) as i32]),
                Array::from_slice(&bi, &[n as i32, (k / group) as i32]),
            ];
            let sg8 = if bits == 4 && k / group % 8 == 0 { Some(verify_geometry(&x, &a, true, 4, 8)?) } else { None };
            let got = verify(&x, &a)?;
            let qmv = if m == 1 && k % 512 == 0 && bits == 4 { Some(super::qmv(&x, &a)?) } else { None };
            mlx_rs::transforms::eval([&got].into_iter().chain(sg8.iter()).chain(qmv.iter()))?;
            let got = got.as_slice::<f32>();
            for r in 0..m {
                for c in 0..n {
                    let mut want = 0f64;
                    for d in 0..k {
                        let qi = ((q[c * k / pack + d / pack] >> (bits * (d % pack))) & ((1 << bits) - 1)) as f64;
                        let g = c * k / group + d / group;
                        want += xv[r * k + d].to_f32() as f64 * (qi * sc[g].to_f32() as f64 + bi[g].to_f32() as f64);
                    }
                    if let Some(qmv) = &qmv {
                        let v = qmv.as_slice::<f32>()[r * n + c] as f64;
                        assert!((v - want).abs() < 2e-5 * (1.0 + want.abs()), "qmv {v} vs {want}");
                    }
                    if let Some(sg8) = &sg8 {
                        let v = sg8.as_slice::<f32>()[r * n + c] as f64;
                        assert!((v - want).abs() < 1e-4 * (1.0 + want.abs()), "sg8 {v} vs {want}");
                    }
                    assert!(
                        (got[r * n + c] as f64 - want).abs() < 2e-5 * (1.0 + want.abs()),
                        "M={m} N={n} K={k} group={group} row={r} col={c}: {} vs {want}",
                        got[r * n + c]
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn delta_matches_reference_and_partition() -> anyhow::Result<()> {
        mlx_rs::with_device(mlx_rs::Device::gpu(), || {
            use mlx_rs::{ops, ops::indexing::IndexOp};
            super::super::weights::init()?;
            let (t, lk, hv, dv, dk) = (17, 2, 4, 8, 128);
            let vals = |shape: &[i32], scale: f32| {
                let n: i32 = shape.iter().product();
                let v: Vec<f32> = (0..n).map(|i| ((i * 37 % 103) as f32 - 51.0) * scale).collect();
                Array::from_slice(&v, shape)
            };
            let s = vals(&[hv, dv, dk], 0.01);
            let kq = vals(&[t, lk, dk, 2], 0.0005);
            let v = vals(&[t, hv, dv], 0.01);
            let e = ops::full::<f32>(&[t, hv], Array::from_f32(0.95))?;
            let b = ops::full::<f32>(&[t, hv], Array::from_f32(0.7))?;
            let dot = ops::sum_axis(ops::multiply(kq.index((.., .., .., 0)), kq.index((.., .., .., 1)))?, -1, false)?;
            let (end, ck, out) = delta(&s, &kq, &v, &e, &b, &dot, 7)?;
            let (prefix, _, _) =
                delta(&s, &kq.index(..7), &v.index(..7), &e.index(..7), &b.index(..7), &dot.index(..7), 0)?;
            let (part, _, tail) =
                delta(&prefix, &kq.index(7..), &v.index(7..), &e.index(7..), &b.index(7..), &dot.index(7..), 0)?;
            let (fused, fused_ck, fused_out) = delta_replay(
                &s,
                [&kq, &v, &e, &b],
                7,
                [&kq.index(7..), &v.index(7..), &e.index(7..), &b.index(7..), &dot.index(7..)],
            )?;
            let (end8, ck8, out8) = delta_with(&s, &kq, &v, &e, &b, &dot, 7, true)?;
            let fused_tail_reference = out8.index(7..).contiguous()?;
            mlx_rs::transforms::eval([
                &end,
                &ck,
                &out,
                &prefix,
                &part,
                &tail,
                &fused,
                &fused_ck,
                &fused_out,
                &end8,
                &ck8,
                &fused_tail_reference,
            ])?;
            assert_eq!(end8.as_slice::<f32>(), fused.as_slice::<f32>(), "fused replay state differs");
            assert_eq!(ck8.as_slice::<f32>(), fused_ck.as_slice::<f32>(), "fused replay checkpoint differs");
            assert_eq!(
                fused_tail_reference.as_slice::<f32>(),
                fused_out.as_slice::<f32>(),
                "fused replay outputs differ"
            );
            assert_eq!(end.as_slice::<f32>(), part.as_slice::<f32>(), "state depends on segment partition");
            assert_eq!(ck.as_slice::<f32>(), prefix.as_slice::<f32>(), "checkpoint differs");
            let yo = out.index(7..).contiguous()?;
            yo.eval()?;
            assert_eq!(yo.as_slice::<f32>(), tail.as_slice::<f32>(), "outputs depend on segment partition");
            let mut st = s;
            let mut reference = Vec::new();
            for i in 0..t {
                let row = |a: &Array| super::super::delta::row(a, i);
                let (s2, y) = super::super::delta::token(
                    &st,
                    super::super::delta::Tok {
                        kq: &row(&kq)?,
                        v: &row(&v)?,
                        e: &row(&e)?,
                        b: &row(&b)?,
                        dot: &row(&dot)?,
                    },
                )?;
                st = s2;
                reference.push(y);
            }
            let reference = ops::stack(&reference, 0)?;
            let err = ops::max(ops::abs(ops::subtract(&reference, &out)?)?, None)?;
            err.eval()?;
            assert!(err.item_exact::<f32>() < 1e-5, "DeltaNet reference mismatch");
            Ok(())
        })
    }
}
