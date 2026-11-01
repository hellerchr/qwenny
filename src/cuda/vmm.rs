//! Device memory mapped on demand (CUDA's virtual memory management): a Region reserves a range of addresses once
//! and maps physical pages into it only where data lives; the pages come from a Pool shared by all regions, within a
//! budget. Kernels and CUDA graphs see fixed addresses, so a per-slot cache can grow, shrink and move to the host
//! without any kernel knowing: a slot's part of the KV cache costs memory only for the positions it holds.
use anyhow::{Result, bail, ensure};
use cudarc::driver::sys::{self, CUdeviceptr, CUmemGenericAllocationHandle};
use cudarc::driver::CudaStream;
use std::sync::{Arc, Mutex};

fn check(r: sys::CUresult, what: &str) -> Result<()> {
    ensure!(r == sys::CUresult::CUDA_SUCCESS, "{what}: {r:?}");
    Ok(())
}

/// Physical pages for the regions: made when first needed, at most `budget` of them, reused when unmapped
pub struct Pool {
    page: usize,
    dev: i32,
    inner: Mutex<Pages>,
}

struct Pages {
    free: Vec<CUmemGenericAllocationHandle>,
    made: usize,
    budget: usize,
}

impl Pool {
    pub fn new(s: &Arc<CudaStream>) -> Result<Arc<Self>> {
        s.context().bind_to_thread()?;
        let dev = s.context().ordinal() as i32;
        let mut page = 0usize;
        check(unsafe { sys::cuMemGetAllocationGranularity(&mut page, &prop(dev),
              sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM) }, "page size")?;
        Ok(Arc::new(Pool { page, dev, inner: Mutex::new(Pages { free: Vec::new(), made: 0, budget: usize::MAX }) }))
    }

    /// Bytes of a page (2 MiB on current GPUs)
    pub fn page(&self) -> usize {
        self.page
    }

    /// Allow at most `pages` pages (those made already stay)
    pub fn set_budget(&self, pages: usize) {
        self.inner.lock().unwrap().budget = pages;
    }

    /// Make the budget's pages now (those the device cannot give lower it): the regions own that memory from the
    /// start, and what is allocated later (CUDA graphs, workspaces) cannot take it from under them
    pub fn fill(&self) -> Result<()> {
        let mut p = self.inner.lock().unwrap();
        while p.made < p.budget {
            let mut h = 0;
            let r = unsafe { sys::cuMemCreate(&mut h, self.page, &prop(self.dev), 0) };
            if r == sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
                p.budget = p.made;
                break;
            }
            check(r, "cuMemCreate")?;
            p.made += 1;
            p.free.push(h);
        }
        Ok(())
    }

    /// Pages in use (mapped) and the budget
    pub fn usage(&self) -> (usize, usize) {
        let p = self.inner.lock().unwrap();
        (p.made - p.free.len(), p.budget)
    }

    /// Pages that can still be mapped
    pub fn available(&self) -> usize {
        let p = self.inner.lock().unwrap();
        p.free.len() + p.budget.saturating_sub(p.made)
    }

    fn take(&self) -> Result<Option<CUmemGenericAllocationHandle>> {
        let mut p = self.inner.lock().unwrap();
        if let Some(h) = p.free.pop() {
            return Ok(Some(h));
        }
        if p.made >= p.budget {
            return Ok(None);
        }
        let mut h = 0;
        let r = unsafe { sys::cuMemCreate(&mut h, self.page, &prop(self.dev), 0) };
        if r == sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
            p.budget = p.made;
            return Ok(None);
        }
        check(r, "cuMemCreate")?;
        p.made += 1;
        Ok(Some(h))
    }

    fn give(&self, h: CUmemGenericAllocationHandle) {
        self.inner.lock().unwrap().free.push(h);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        for h in self.inner.get_mut().unwrap().free.drain(..) {
            unsafe { sys::cuMemRelease(h) };
        }
    }
}

/// Host memory the device reads directly (pinned, mapped into its address space): tables a step reads a few rows of,
/// off the device's memory
pub struct HostMapped {
    host: *mut u8,
    dev: CUdeviceptr,
}

unsafe impl Send for HostMapped {}

impl HostMapped {
    pub fn new(s: &Arc<CudaStream>, data: &[u8]) -> Result<Self> {
        s.context().bind_to_thread()?;
        let mut host = std::ptr::null_mut();
        let flags = sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP;
        check(unsafe { sys::cuMemHostAlloc(&mut host, data.len().max(1), flags) }, "cuMemHostAlloc")?;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), host as *mut u8, data.len()) };
        let mut dev = 0;
        check(unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, host, 0) }, "cuMemHostGetDevicePointer")?;
        Ok(HostMapped { host: host as *mut u8, dev })
    }

    /// The device address of byte 0 (a kernel argument)
    pub fn ptr(&self) -> CUdeviceptr {
        self.dev
    }
}

impl Drop for HostMapped {
    fn drop(&mut self) {
        unsafe { sys::cuMemFreeHost(self.host as *mut _) };
    }
}

fn prop(dev: i32) -> sys::CUmemAllocationProp {
    let mut p: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    p.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.__bindgen_anon_1.id = dev;
    p
}

/// A reserved range of device addresses with pages mapped where data lives
pub struct Region {
    base: CUdeviceptr,
    size: usize,
    /// per page: its physical page, if mapped
    pages: Vec<Option<CUmemGenericAllocationHandle>>,
    pool: Arc<Pool>,
    s: Arc<CudaStream>,
}

impl Region {
    /// Reserve `size` bytes (rounded up to whole pages); nothing is mapped yet
    pub fn reserve(s: &Arc<CudaStream>, pool: &Arc<Pool>, size: usize) -> Result<Self> {
        s.context().bind_to_thread()?;
        let size = size.div_ceil(pool.page) * pool.page;
        let mut base = 0;
        check(unsafe { sys::cuMemAddressReserve(&mut base, size, pool.page, 0, 0) }, "cuMemAddressReserve")?;
        Ok(Region { base, size, pages: vec![None; size / pool.page], pool: pool.clone(), s: s.clone() })
    }

    /// Bytes of a page
    pub fn page_size(&self) -> usize {
        self.pool.page
    }

    /// The device address of byte 0 (a kernel argument)
    pub fn ptr(&self) -> CUdeviceptr {
        self.base
    }

    /// Map the pages covering bytes [from, to), zeroed; false (and nothing more mapped) when the pool has too few
    pub fn map(&mut self, from: usize, to: usize) -> Result<bool> {
        let (p0, p1) = (from / self.pool.page, to.min(self.size).div_ceil(self.pool.page));
        let need = (p0..p1).filter(|&i| self.pages[i].is_none()).count();
        if need == 0 {
            return Ok(true);
        }
        if self.pool.available() < need {
            return Ok(false);
        }
        self.s.context().bind_to_thread()?;
        let access = sys::CUmemAccessDesc {
            location: prop(self.pool.dev).location,
            flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        for i in p0..p1 {
            if self.pages[i].is_some() {
                continue;
            }
            let Some(h) = self.pool.take()? else { bail!("device memory: no page left (pool of {:?})", self.pool.usage()) };
            let at = self.base + (i * self.pool.page) as u64;
            check(unsafe { sys::cuMemMap(at, self.pool.page, 0, h, 0) }, "cuMemMap")?;
            check(unsafe { sys::cuMemSetAccess(at, self.pool.page, &access, 1) }, "cuMemSetAccess")?;
            check(unsafe { sys::cuMemsetD8Async(at, 0, self.pool.page, self.s.cu_stream()) }, "cuMemsetD8Async")?;
            self.pages[i] = Some(h);
        }
        Ok(true)
    }

    /// Unmap the pages that lie wholly inside bytes [from, to) (after the stream's work on them)
    pub fn unmap(&mut self, from: usize, to: usize) -> Result<()> {
        let (p0, p1) = (from.div_ceil(self.pool.page), to.min(self.size) / self.pool.page);
        if (p0..p1).all(|i| self.pages[i].is_none()) {
            return Ok(());
        }
        self.s.synchronize()?;
        for i in p0..p1 {
            if let Some(h) = self.pages[i].take() {
                check(unsafe { sys::cuMemUnmap(self.base + (i * self.pool.page) as u64, self.pool.page) }, "cuMemUnmap")?;
                self.pool.give(h);
            }
        }
        Ok(())
    }

    /// Copy bytes [from, from + out.len()) to the host (mapped: the caller maps what it reads)
    pub fn read(&self, from: usize, out: &mut [u8]) -> Result<()> {
        ensure!(from + out.len() <= self.size, "read beyond the region");
        self.s.synchronize()?;
        check(unsafe { sys::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), self.base + from as u64, out.len()) }, "cuMemcpyDtoH")
    }

    /// Copy `data` to bytes [from, from + data.len()) (mapped)
    pub fn write(&self, from: usize, data: &[u8]) -> Result<()> {
        ensure!(from + data.len() <= self.size, "write beyond the region");
        // on the stream, waited for: a copy from pageable memory may return before it lands, and the stream's next
        // kernels would not wait for the legacy stream's copy
        check(unsafe { sys::cuMemcpyHtoDAsync_v2(self.base + from as u64, data.as_ptr().cast(), data.len(), self.s.cu_stream()) }, "cuMemcpyHtoDAsync")?;
        Ok(self.s.synchronize()?)
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        let _ = self.s.synchronize();
        for (i, p) in self.pages.iter_mut().enumerate() {
            if let Some(h) = p.take() {
                unsafe { sys::cuMemUnmap(self.base + (i * self.pool.page) as u64, self.pool.page) };
                self.pool.give(h);
            }
        }
        unsafe { sys::cuMemAddressFree(self.base, self.size) };
    }
}
