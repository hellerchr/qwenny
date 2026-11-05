//! The trainer's (qwen/train.rs) general parts: device buffers, kernel launches, the cuBLAS products, the LoRA arena
//! (f32 values, gradients, Adam moments and the act copy the products read) with its module groups, LoRA dropout,
//! clipping and AdamW. Its kernels are in kernels/train_common.cuh, which kernels/qwen_train.cu includes.
use crate::cuda::core::cfg;
use crate::training::{Opts, f32_bytes, splitmix};
use anyhow::{Context, Result, ensure};
use cudarc::cublas::{CudaBlas, result as blas, sys as bsys};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg, result as drv};
use std::{collections::HashMap, ffi::c_void, sync::Arc};

/// Loss scale of the f16 build (halved after a step whose gradients overflowed)
pub const F16_LOSS_SCALE: f32 = 1024.0;

/// The kernels of train_common.cuh
const KERNELS: [&str; 7] = ["dropout", "dropout_add_grad", "dropout_add_act", "sum_squares", "adamw", "to_act_copy", "gather_rows"];

pub struct Buf {
    _mem: CudaSlice<u8>,
    pub p: u64,
}

pub enum Arg {
    P(u64),
    I(i32),
    F(f32),
}
use Arg::{F, P};

/// One LoRA module of a group: output width, first output column in the group's product, offset of B in the arena.
pub struct Mod {
    pub out: usize,
    pub col: usize,
    pub b: usize,
}

/// LoRA modules that read the same input: their A matrices are stacked rows at `a` ([mods * rank][inp]), so one
/// product gives U for all of them, and they share one dropout mask.
pub struct Group {
    pub inp: usize,
    pub a: usize,
    pub mods: Vec<Mod>,
}

/// The arena's layout while a trainer adds its modules: (PEFT name, rows, cols, offset) per tensor.
pub struct Layout {
    rank: usize,
    pub tensors: Vec<(String, usize, usize, usize)>,
    pub n: usize,
}

impl Layout {
    pub fn new(rank: usize) -> Self {
        Layout { rank, tensors: Vec::new(), n: 0 }
    }

    /// The modules `mods` (name, output width) under `prefix`, all reading an input of width `inp`
    pub fn group(&mut self, prefix: &str, inp: usize, mods: &[(&str, usize)]) -> Group {
        let (r, a) = (self.rank, self.n);
        for (name, _) in mods {
            self.tensors.push((format!("{prefix}.{name}.lora_A.weight"), r, inp, self.n));
            self.n += r * inp;
        }
        let mut col = 0;
        let mut ms = Vec::new();
        for (name, out) in mods {
            self.tensors.push((format!("{prefix}.{name}.lora_B.weight"), *out, r, self.n));
            ms.push(Mod { out: *out, col, b: self.n });
            self.n += out * r;
            col += out;
        }
        Group { inp, a, mods: ms }
    }

    /// PEFT's init: A kaiming-uniform (bound 1 / sqrt(fan_in)), B zero (everything else zero too)
    pub fn init(&self, seed: u64) -> Vec<f32> {
        let mut init = vec![0f32; self.n];
        let mut rng = seed;
        for (name, rows, cols, off) in &self.tensors {
            if name.contains("lora_A") {
                let bound = 1.0 / (*cols as f32).sqrt();
                for x in &mut init[*off..off + rows * cols] {
                    *x = ((splitmix(&mut rng) >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * bound;
                }
            }
        }
        init
    }
}

/// Sizes of the LoRA scratch: most tokens, widest module input, most rank columns of a group (U's row stride)
pub struct Scratch {
    pub tmax: usize,
    pub inp: usize,
    pub ldu: usize,
}

pub struct Learn {
    pub s: Arc<CudaStream>,
    blas: CudaBlas,
    f: HashMap<&'static str, CudaFunction>,
    pub o: Opts,
    /// bytes per activation / weight element (2: bf16 or f16, 4: f32)
    pub es: usize,
    /// row stride of the saved LoRA products U
    pub ldu: usize,
    lscale: f32,
    pub n_params: usize,
    pub tensors: Vec<(String, usize, usize, usize)>,
    p32: Buf,
    pg: Buf,
    pm: Buf,
    pv: Buf,
    pa: Buf,
    adam_t: i32,
    /// gradients are computed for loss_scale * loss (f16 build)
    pub loss_scale: f32,
    sumsq: Buf,
    /// dropout(x), the rank-space gradient, and the LoRA path's f32 input gradient
    xd: Buf,
    vbuf: Buf,
    pub tmp32: Buf,
}

impl Learn {
    /// `names`: the trainer's own kernels in `module` (besides train_common.cuh's); `init`: the arena's values
    pub fn new(s: Arc<CudaStream>, module: &Arc<CudaModule>, names: &[&'static str], o: Opts, layout: Layout,
               init: &[f32], sc: Scratch) -> Result<Self> {
        let blas = CudaBlas::new(s.clone())?;
        let mut f = HashMap::new();
        for &name in KERNELS.iter().chain(names) {
            f.insert(name, module.load_function(name).with_context(|| format!("kernel {name}"))?);
        }
        let es = if o.f32 { 4 } else { 2 };
        let n = layout.n;
        let alloc = |bytes: usize| alloc(&s, bytes);
        let l = Learn {
            p32: upload(&s, f32_bytes(init))?,
            pg: alloc(n * 4)?,
            pm: alloc(n * 4)?,
            pv: alloc(n * 4)?,
            pa: alloc(n * es)?,
            sumsq: alloc(4)?,
            xd: alloc(sc.tmax * sc.inp * es)?,
            vbuf: alloc(sc.tmax * sc.ldu * es)?,
            tmp32: alloc(sc.tmax * sc.inp * 4)?,
            adam_t: 0,
            loss_scale: if o.f16 { F16_LOSS_SCALE } else { 1.0 },
            lscale: o.alpha / o.rank as f32,
            ldu: sc.ldu,
            n_params: n,
            tensors: layout.tensors,
            blas,
            f,
            o,
            es,
            s,
        };
        l.sync_act()?;
        Ok(l)
    }

    pub fn launch_cfg(&self, name: &str, cfg: LaunchConfig, args: &[Arg]) -> Result<()> {
        let mut b = self.s.launch_builder(&self.f[name]);
        for a in args {
            match a {
                Arg::P(x) => b.arg(x),
                Arg::I(x) => b.arg(x),
                Arg::F(x) => b.arg(x),
            };
        }
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    pub fn launch(&self, name: &str, grid: usize, block: usize, args: &[Arg]) -> Result<()> {
        self.launch_cfg(name, cfg((grid, 1, 1), block), args)
    }

    pub fn elementwise(&self, name: &str, n: usize, args: &[Arg]) -> Result<()> {
        self.launch(name, n.div_ceil(256), 256, args)
    }

    fn act_type(&self) -> bsys::cudaDataType {
        if self.o.f32 {
            bsys::cudaDataType::CUDA_R_32F
        } else if self.o.f16 {
            bsys::cudaDataType::CUDA_R_16F
        } else {
            bsys::cudaDataType::CUDA_R_16BF
        }
    }

    /// Whether dX of the products (the norms' input gradients) is f32 (else f16)
    pub fn g32(&self) -> bool {
        !self.o.f16
    }

    /// Row-major C[m][n] = alpha * op(A) op(B) + beta * C, with op(A) [m][k] (A stored [k][m] if ta) and op(B)
    /// [k][n] (B stored [n][k] if tb); ld* are row strides in elements; A and B act, C f32 if c32 else act.
    #[allow(clippy::too_many_arguments)]
    pub fn mm(&self, ta: bool, tb: bool, m: usize, n: usize, k: usize, alpha: f32, a: u64, lda: usize, b: u64, ldb: usize,
              beta: f32, c: u64, ldc: usize, c32: bool) -> Result<()> {
        use bsys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
        let op = |t: bool| if t { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let at = self.act_type();
        let ct = if c32 { bsys::cudaDataType::CUDA_R_32F } else { at };
        use bsys::cublasComputeType_t::{CUBLAS_COMPUTE_16F, CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F_PEDANTIC};
        // f16 accumulation needs f16 C and f16 alpha, beta
        let acc16 = self.o.f16 && !c32;
        let compute = if self.o.f32 { CUBLAS_COMPUTE_32F_PEDANTIC } else if acc16 { CUBLAS_COMPUTE_16F } else { CUBLAS_COMPUTE_32F };
        let (alpha16, beta16) = (half::f16::from_f32(alpha), half::f16::from_f32(beta));
        let (pa, pb) = if acc16 {
            (&alpha16 as *const half::f16 as *const c_void, &beta16 as *const half::f16 as *const c_void)
        } else {
            (&alpha as *const f32 as *const c_void, &beta as *const f32 as *const c_void)
        };
        // column-major: C^T = op(B)^T op(A)^T; a batch of one (strides 0)
        unsafe {
            blas::gemm_strided_batched_ex(
                *self.blas.handle(), op(tb), op(ta), n as i32, m as i32, k as i32,
                pa, b as *const c_void, at, ldb as i32, 0, a as *const c_void, at, lda as i32, 0, pb,
                c as *mut c_void, ct, ldc as i32, 0, 1, compute, bsys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )?;
        }
        Ok(())
    }

    pub fn zero(&self, p: u64, bytes: usize) -> Result<()> {
        unsafe { drv::memset_d8_async(p, 0, bytes, self.s.cu_stream()) }?;
        Ok(())
    }

    /// Device-to-device copy
    pub fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
        unsafe { drv::memcpy_dtod_async(dst, src, bytes, self.s.cu_stream()) }?;
        Ok(())
    }

    pub fn htod<T: cudarc::driver::DeviceRepr>(&self, p: u64, v: &[T]) -> Result<()> {
        unsafe { drv::memcpy_htod_async(p, v, self.s.cu_stream()) }?;
        Ok(())
    }

    pub fn dtoh<T: cudarc::driver::DeviceRepr + Default + Clone>(&self, p: u64, n: usize) -> Result<Vec<T>> {
        let mut v = vec![T::default(); n];
        unsafe { drv::memcpy_dtoh_async(&mut v, p, self.s.cu_stream()) }?;
        self.s.synchronize()?;
        Ok(v)
    }

    /// The act copy of the LoRA weights, after loading or changing the f32 values
    pub fn sync_act(&self) -> Result<()> {
        self.elementwise("to_act_copy", self.n_params, &[P(self.p32.p), P(self.pa.p), P(self.n_params as u64)])
    }

    /// Device address of arena element `off`: act copy, gradient
    pub fn pa(&self, off: usize) -> u64 {
        self.pa.p + (off * self.es) as u64
    }

    pub fn pgr(&self, off: usize) -> u64 {
        self.pg.p + (off * 4) as u64
    }

    /// dropout(x) for the LoRA path, or x itself
    fn lora_input(&self, x: u64, n: usize, train: bool, seed: u64) -> Result<u64> {
        if !train || self.o.dropout == 0.0 {
            return Ok(x);
        }
        self.launch("dropout", n.div_ceil(8 * 256), 256, &[P(x), P(self.xd.p), P(n as u64), F(self.o.dropout), P(seed)])?;
        Ok(self.xd.p)
    }

    /// y[:, col..] += scale * dropout(x) A^T B^T for each module; U = dropout(x) A^T saved at u (row stride ldu)
    #[allow(clippy::too_many_arguments)]
    pub fn lora_fwd(&self, g: &Group, x: u64, t: usize, y: u64, ldy: usize, u: u64, train: bool, seed: u64) -> Result<()> {
        let (r, es, ldu) = (self.o.rank, self.es as u64, self.ldu);
        self.lora_u(g, x, t, u, train, seed)?;
        for (i, m) in g.mods.iter().enumerate() {
            self.mm(false, true, t, m.out, r, self.lscale, u + (i * r) as u64 * es, ldu, self.pa(m.b), r, 1.0,
                    y + m.col as u64 * es, ldy, false)?;
        }
        Ok(())
    }

    /// Only U = dropout(x) A^T of `lora_fwd` (a recomputed forward whose output is not needed)
    pub fn lora_u(&self, g: &Group, x: u64, t: usize, u: u64, train: bool, seed: u64) -> Result<()> {
        let xd = self.lora_input(x, t * g.inp, train, seed)?;
        self.mm(false, true, t, g.mods.len() * self.o.rank, g.inp, 1.0, xd, g.inp, self.pa(g.a), g.inp, 0.0, u, self.ldu, false)
    }

    /// Gradients of A and B from dy (the module outputs' gradient), and dx += the LoRA path's input gradient (dx f32
    /// if `dx_grad` and g32(), else act).
    #[allow(clippy::too_many_arguments)]
    pub fn lora_bwd(&self, g: &Group, dy: u64, ldy: usize, x: u64, t: usize, u: u64, dx: u64, dx_grad: bool, seed: u64)
    -> Result<()> {
        let (r, es, ldu) = (self.o.rank, self.es as u64, self.ldu);
        let mr = g.mods.len() * r;
        let xd = self.lora_input(x, t * g.inp, true, seed)?;
        for (i, m) in g.mods.iter().enumerate() {
            let dyi = dy + m.col as u64 * es;
            let ui = u + (i * r) as u64 * es;
            self.mm(false, false, t, r, m.out, 1.0, dyi, ldy, self.pa(m.b), r, 0.0, self.vbuf.p + (i * r) as u64 * es, mr, false)?;
            self.mm(true, false, m.out, r, t, self.lscale, dyi, ldy, ui, ldu, 1.0, self.pgr(m.b), r, true)?;
        }
        self.mm(true, false, mr, g.inp, t, self.lscale, self.vbuf.p, mr, xd, g.inp, 1.0, self.pgr(g.a), g.inp, true)?;
        if self.o.dropout == 0.0 {
            self.mm(false, false, t, g.inp, mr, self.lscale, self.vbuf.p, mr, self.pa(g.a), g.inp, 1.0, dx, g.inp, dx_grad && self.g32())?;
        } else {
            self.mm(false, false, t, g.inp, mr, self.lscale, self.vbuf.p, mr, self.pa(g.a), g.inp, 0.0, self.tmp32.p, g.inp, true)?;
            let n = t * g.inp;
            let k = if dx_grad { "dropout_add_grad" } else { "dropout_add_act" };
            self.launch(k, n.div_ceil(8 * 256), 256, &[P(dx), P(self.tmp32.p), P(n as u64), F(self.o.dropout), P(seed)])?;
        }
        Ok(())
    }

    pub fn zero_grads(&self) -> Result<()> {
        self.zero(self.pg.p, self.n_params * 4)
    }

    /// clip_grad_norm_(max_norm) and one AdamW step; returns the gradient norm before clipping. With f16 gradients
    /// that overflowed the step is skipped (NaN returned) and the loss scale halved.
    pub fn optimizer_step(&mut self, lr: f32, max_norm: f32, wd: f32) -> Result<f32> {
        self.zero(self.sumsq.p, 4)?;
        self.launch("sum_squares", 1024, 256, &[P(self.pg.p), P(self.n_params as u64), P(self.sumsq.p)])?;
        let ss: Vec<f32> = self.dtoh(self.sumsq.p, 1)?;
        let norm = ss[0].sqrt() / self.loss_scale;
        if !norm.is_finite() {
            ensure!(self.loss_scale > 1.0, "gradient norm {norm}");
            self.loss_scale /= 2.0;
            return Ok(f32::NAN);
        }
        self.adam_t += 1;
        let (b1, b2) = (0.9f32, 0.999f32);
        let (bc1, bc2) = (1.0 - b1.powi(self.adam_t), 1.0 - b2.powi(self.adam_t));
        self.elementwise("adamw", self.n_params, &[P(self.p32.p), P(self.pg.p), P(self.pm.p), P(self.pv.p), P(self.pa.p),
            P(self.n_params as u64), P(self.sumsq.p), F(max_norm), F(lr), F(b1), F(b2), F(1e-8), F(wd), F(bc1), F(bc2),
            F(1.0 / self.loss_scale)])?;
        Ok(norm)
    }

    /// LoRA tensors by PEFT name (without the adapter name)
    pub fn read_params(&self, grads: bool) -> Result<HashMap<String, Vec<f32>>> {
        let mut all: Vec<f32> = self.dtoh(if grads { self.pg.p } else { self.p32.p }, self.n_params)?;
        if grads {
            all.iter_mut().for_each(|g| *g /= self.loss_scale);
        }
        Ok(self.tensors.iter().map(|(n, r, c, off)| (n.clone(), all[*off..off + r * c].to_vec())).collect())
    }

    /// Set the tensors in `values`
    pub fn write_params(&self, values: &HashMap<String, Vec<f32>>) -> Result<()> {
        let mut all: Vec<f32> = self.dtoh(self.p32.p, self.n_params)?;
        for (n, r, c, off) in &self.tensors {
            let v = values.get(n).with_context(|| format!("no tensor {n}"))?;
            ensure!(v.len() == r * c, "{n}: {} values, expected {}", v.len(), r * c);
            all[*off..off + r * c].copy_from_slice(v);
        }
        self.htod(self.p32.p, &all)?;
        self.sync_act()?;
        self.s.synchronize()?;
        Ok(())
    }

    /// LoRA tensors: PEFT name, rows, columns
    pub fn tensor_shapes(&self) -> Vec<(String, usize, usize)> {
        self.tensors.iter().map(|(n, r, c, _)| (n.clone(), *r, *c)).collect()
    }

    /// Seed of a LoRA dropout mask: the same in forward and backward
    pub fn seed(step: u64, micro: usize, layer: usize, group: usize) -> u64 {
        let mut x = step << 32 ^ (micro as u64) << 16 ^ (layer as u64) << 4 ^ group as u64;
        splitmix(&mut x)
    }
}

pub fn alloc(s: &Arc<CudaStream>, bytes: usize) -> Result<Buf> {
    let mem = s.alloc_zeros::<u8>(bytes.max(1))?;
    let p = mem.device_ptr(s).0;
    Ok(Buf { _mem: mem, p })
}

pub fn upload(s: &Arc<CudaStream>, bytes: &[u8]) -> Result<Buf> {
    let mem = s.clone_htod(bytes)?;
    let p = mem.device_ptr(s).0;
    Ok(Buf { _mem: mem, p })
}

/// Free and total device memory in GiB
pub fn mem_gib() -> (f64, f64) {
    drv::mem_get_info().map(|(f, t)| (f as f64 / 1073741824.0, t as f64 / 1073741824.0)).unwrap_or((0.0, 0.0))
}
