//! Metal custom operations for skinny verify products, the draft head and DeltaNet; on M5 also tensor-unit verify
//! products. All reductions accumulate in f32.
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
        Self::with_header(name, inputs, outputs, source, c"#include <metal_simdgroup_matrix>\n")
    }
    fn with_header(name: &CStr, inputs: &[&CStr], outputs: &[&CStr], source: &str, header: &CStr) -> Self {
        let source = CString::new(source).unwrap();
        let mut names: Vec<_> = inputs.iter().map(|n| n.as_ptr()).collect();
        let mut out: Vec<_> = outputs.iter().map(|n| n.as_ptr()).collect();
        // SAFETY: the C API copies the strings and returns an owning kernel handle.
        unsafe {
            let ins = sys::mlx_vector_string_new_data(names.as_mut_ptr(), names.len());
            let outs = sys::mlx_vector_string_new_data(out.as_mut_ptr(), out.len());
            let k =
                sys::mlx_fast_metal_kernel_new(name.as_ptr(), ins, outs, source.as_ptr(), header.as_ptr(), true, false);
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

/// The GPU is an M5's (Metal 4 tensor units): only then are the tensor-unit kernels compiled
pub fn m5() -> bool {
    static M5: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *M5.get_or_init(|| {
        // SAFETY: the device name is copied before the owning handles are freed.
        unsafe {
            let dev = sys::mlx_device_new_type(sys::mlx_device_type__MLX_GPU, 0);
            let mut info = sys::mlx_device_info_new();
            let mut name = std::ptr::null();
            let ok = sys::mlx_device_info_get(&mut info, dev) == 0
                && sys::mlx_device_info_get_string(&mut name, info, c"device_name".as_ptr()) == 0
                && !name.is_null()
                && CStr::from_ptr(name).to_bytes().starts_with(b"Apple M5");
            sys::mlx_device_info_free(info);
            sys::mlx_device_free(dev);
            ok
        }
    })
}

/// The GPU's recommended working set (less than the machine's unified memory)
pub fn recommended_memory() -> MResult<usize> {
    // SAFETY: the size is copied out; the owning handles are freed on success and on error.
    unsafe {
        let dev = sys::mlx_device_new_type(sys::mlx_device_type__MLX_GPU, 0);
        let mut info = sys::mlx_device_info_new();
        let mut size = 0;
        let r = check(sys::mlx_device_info_get(&mut info, dev), "device info").and_then(|_| {
            check(
                sys::mlx_device_info_get_size(&mut size, info, c"max_recommended_working_set_size".as_ptr()),
                "device memory",
            )
        });
        sys::mlx_device_info_free(info);
        sys::mlx_device_free(dev);
        r.map(|_| size)
    }
}

/// int4 codes [N][K/8] and their scales and biases [N][K/group] in verify_nax's layout: tiles of 128 rows, in each the
/// codes of a 64-wide block of K for all its rows together, and per group the 128 rows' scales (biases) together. A
/// permutation: no stored value changes.
pub fn pack_verify(a: &[Array]) -> MResult<[Array; 3]> {
    let (n, k, groups) = (a[0].shape()[0], a[0].shape()[1] * 8, a[1].shape()[1]);
    let affine = |v: &Array| v.reshape(&[n / 128, 128, groups])?.transpose_axes(&[0, 2, 1])?.contiguous();
    Ok([
        a[0].reshape(&[n / 128, 128, k / 64, 8])?.transpose_axes(&[0, 2, 1, 3])?.contiguous()?,
        affine(&a[1])?,
        affine(&a[2])?,
    ])
}

/// pack_verify's inverse: the row-major arrays (for products of more rows than verify_nax takes)
pub fn unpack_verify(a: &[Array]) -> MResult<[Array; 3]> {
    let (tiles, blocks, groups) = (a[0].shape()[0], a[0].shape()[1], a[1].shape()[1]);
    let affine = |v: &Array| v.transpose_axes(&[0, 2, 1])?.reshape(&[tiles * 128, groups]);
    Ok([a[0].transpose_axes(&[0, 2, 1, 3])?.reshape(&[tiles * 128, blocks * 8])?, affine(&a[1])?, affine(&a[2])?])
}

/// Rows `ids` of a pack_verify'd matrix, row-major (the draft head's rows), without unpacking the rest
pub fn packed_rows(a: &[Array], ids: &[u32]) -> MResult<[Array; 3]> {
    let (blocks, groups) = (a[0].shape()[1] as u32, a[1].shape()[1] as u32);
    let at = |count: u32| {
        let v: Vec<u32> =
            ids.iter().flat_map(|&id| (0..count).map(move |g| ((id / 128) * count + g) * 128 + id % 128)).collect();
        Array::from_slice(&v, &[v.len() as i32])
    };
    let (codes, affine, rows) = (at(blocks), at(groups), ids.len() as i32);
    Ok([
        a[0].reshape(&[-1, 8])?.take_axis(&codes, 0)?.reshape(&[rows, blocks as i32 * 8])?,
        a[1].reshape(&[-1])?.take_axis(&affine, 0)?.reshape(&[rows, groups as i32])?,
        a[2].reshape(&[-1])?.take_axis(&affine, 0)?.reshape(&[rows, groups as i32])?,
    ])
}

thread_local! { static NAX: RefCell<Option<Kernel>> = const { RefCell::new(None) }; }

/// x W^T for 1 to 8 rows of x [M][K] f16 on an M5's tensor units, W pack_verify'd int4 with the checkpoint's affine
/// scales and biases (groups of 64 or 128): f16 products of the codes, accumulated in f32, scaled in f32. Out f32.
pub fn verify_nax(x: &Array, a: &[Array], group: i32) -> MResult<Array> {
    let (m, k, n) = (x.shape()[0], x.shape()[1], a[1].shape()[0] * 128);
    let (tiles, groups) = (n / 128, k / group);
    // K splits so that about 512 threadgroups run, summed after; two halves of a split's groups in small matrices
    let split = (1..=groups).rev().find(|d| groups % d == 0 && tiles * d <= 512).unwrap_or(1);
    let parts = if tiles < 256 && (groups / split) % 2 == 0 { 2 } else { 1 };
    NAX.with(|cell| {
        let mut cell = cell.borrow_mut();
        let kernel = cell.get_or_insert_with(|| {
            Kernel::with_header(
                c"qwenny_verify_nax",
                &[c"X", c"W", c"SC", c"BI"],
                &[c"OUT"],
                include_str!("verify_nax.metal"),
                c"#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n",
            )
        });
        let mut out = kernel.apply_many(
            &[x.clone(), a[0].clone(), a[1].clone(), a[2].clone()],
            &[&[split, m, n]],
            [256 * parts * tiles, split, 1],
            [256 * parts, 1, 1],
            &[(c"K", k), (c"N", n), (c"GS", group), (c"KSPLIT", split), (c"PARTS", parts)],
            Dtype::Float32,
        )?;
        let y = out.remove(0);
        if split == 1 { y.reshape(&[m, n]) } else { mlx_rs::ops::sum_axis(y, 0, false) }
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
    #[ignore = "M5 only (Metal 4 tensor units): verify_nax and the packed layout"]
    fn verify_nax_matches_affine_reference() -> anyhow::Result<()> {
        mlx_rs::with_device(mlx_rs::Device::gpu(), || {
            use mlx_rs::ops;
            super::super::weights::init()?;
            anyhow::ensure!(m5(), "not an M5");
            let shapes: [(usize, usize, usize, usize); 7] = [
                (1, 128, 512, 64),
                (2, 128, 512, 64),
                (5, 256, 1024, 128),
                (8, 384, 5120, 128),
                (5, 34816, 5120, 128),
                (5, 5120, 17408, 128),
                (5, 248320, 5120, 128),
            ];
            for (m, n, k, group) in shapes {
                let xv: Vec<f16> = (0..m * k).map(|i| f16::from_f32(((i * 17 % 101) as f32 - 50.0) / 53.0)).collect();
                let q: Vec<u32> =
                    (0..n * k / 8).map(|i| (i as u32).wrapping_mul(7919).wrapping_add(0x1949abcd)).collect();
                let sc: Vec<f16> = (0..n * k / group).map(|i| f16::from_f32((i % 11 + 1) as f32 / 97.0)).collect();
                let bi: Vec<f16> = sc.iter().map(|s| f16::from_f32(-8.0 * s.to_f32())).collect();
                let x = Array::from_slice(&xv, &[m as i32, k as i32]);
                let a = [
                    Array::from_slice(&q, &[n as i32, (k / 8) as i32]),
                    Array::from_slice(&sc, &[n as i32, (k / group) as i32]),
                    Array::from_slice(&bi, &[n as i32, (k / group) as i32]),
                ];
                let packed = pack_verify(&a)?;
                let ids = [0, 127, n as u32 - 1, 0];
                let idx = Array::from_slice(&ids, &[4]);
                let same = |x: &Array, y: &Array| -> anyhow::Result<bool> {
                    let e = ops::all(ops::eq(x, y)?, None)?;
                    e.eval()?;
                    Ok(e.item_exact::<bool>())
                };
                for ((src, rows), back) in a.iter().zip(packed_rows(&packed, &ids)?).zip(unpack_verify(&packed)?) {
                    anyhow::ensure!(same(&src.take_axis(&idx, 0)?, &rows)?, "packed rows changed stored values");
                    anyhow::ensure!(same(src, &back)?, "the packed layout changed stored values");
                }
                let w = ops::dequantize(
                    &a[0],
                    &a[1].as_dtype(Dtype::Float32)?,
                    Some(&a[2].as_dtype(Dtype::Float32)?),
                    group as i32,
                    4,
                )?;
                let want = ops::matmul(x.as_dtype(Dtype::Float32)?, w.transpose()?)?;
                let err = ops::max(ops::abs(ops::subtract(&verify_nax(&x, &packed, group as i32)?, &want)?)?, None)?;
                err.eval()?;
                let err = err.item_exact::<f32>();
                anyhow::ensure!(err < 2e-4, "verify_nax {m}x{n}x{k} group {group}: max error {err}");
            }
            Ok(())
        })
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
