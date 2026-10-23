//! Steps 2+3 on a CUDA GPU (`--features cuda`).
//!
//! CPU worker threads run the step-1 DFS and pack endpoints into per-class
//! batches (one bitvector per launch); a single GPU thread streams batches
//! through a few CUDA streams and records kernel time per class.  Endpoints
//! whose `f` lies beyond the last bitvector are finished on the CPU, as in
//! the reference program.

use crate::arith::POW3;
use crate::lookahead::Lookahead;
use crate::tail::{process_endpoint, TailStats};
use crate::tree::{dfs, Node, Sink, TreeStats};
use crate::verify::{VerifyConfig, VerifyReport};
use anyhow::{bail, ensure, Result};
use rayon::prelude::*;
use serde::Serialize;
use std::ffi::c_void;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy)]
struct KParams {
    b_mask: u64,
    fam_stride: u64,
    m: u32,
    words32: u32,
    m_limb: u32,
    m_shift: u32,
    cutoff_hi: u32,
    tail_oee: u32,
    flag_cap: u32,
    pad: u32,
}

impl KParams {
    /// Parameters for launches on class `class`, whose family-1 array sits
    /// `fam_stride` words after its family-0 array.
    fn new(la: &Lookahead, class: usize, cutoff_bits: u32, tail_oee: bool, fam_stride: u64) -> Self {
        KParams {
            b_mask: (1u64 << la.bs[class]) - 1,
            fam_stride,
            m: la.m,
            words32: (1u32 << la.a) / 32,
            m_limb: la.m / 32,
            m_shift: la.m % 32,
            cutoff_hi: 1u32 << (cutoff_bits - 96),
            tail_oee: tail_oee as u32,
            flag_cap: 0,
            pad: 0,
        }
    }
}

extern "C" {
    fn cv_gpu_device_name(buf: *mut u8, len: i32) -> i32;
    fn cv_gpu_init(
        bv: *const u32,
        bv_words: usize,
        bv_stride: usize,
        mod9: *const u32,
        mod9_words: usize,
        nslots: i32,
        cap: usize,
    ) -> *mut c_void;
    fn cv_gpu_host_alloc(bytes: usize) -> *mut u32;
    fn cv_gpu_upload_mix(ctx: *mut c_void, flat: *const u64, words: usize, offsets: *const u64) -> i32;
    fn cv_gpu_alloc_bv(ctx: *mut c_void, words: usize) -> i32;
    fn cv_gpu_build_array(
        ctx: *mut c_void,
        offset: usize,
        b: u32,
        a: u32,
        magic: u64,
        cutoff: i32,
        start_evens: u32,
        pm_inf: u32,
        mix_j: u32,
        ms: *mut f32,
    ) -> i32;
    fn cv_gpu_download_bv(ctx: *mut c_void, start: usize, host: *mut u32, words: usize) -> i32;
    fn cv_gpu_mem_info(free: *mut usize, total: *mut usize) -> i32;
    fn cv_gpu_host_free(p: *mut u32);
    fn cv_gpu_launch(
        ctx: *mut c_void,
        slot: i32,
        h_pairs: *const u32,
        n: u32,
        bv_offset: usize,
        magic: u64,
        tp: *const u32,
        p: *const KParams,
    ) -> i32;
    fn cv_gpu_wait(
        ctx: *mut c_void,
        slot: i32,
        ms: *mut f32,
        cases: *mut u64,
        iters: *mut u64,
        hash: *mut u32,
        nflags: *mut u32,
        flags_out: *mut u32,
        cap: u32,
    ) -> i32;
    fn cv_gpu_free(ctx: *mut c_void);
}

pub fn device_name() -> String {
    let mut buf = [0u8; 256];
    if unsafe { cv_gpu_device_name(buf.as_mut_ptr(), 256) } != 0 {
        return "unknown".into();
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(0);
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// u32 words per array: 2^B bits plus 2^A bits of wraparound (as u64 words x 2).
fn array_stride(la: &Lookahead) -> usize {
    2 * la.words
}

/// Upload the mixed-word tables if the window rules use them.
fn upload_mix(ctx: *mut c_void, la: &Lookahead) -> Result<()> {
    if let Some(t) = &la.window.mix {
        let (flat, off) = t.flat();
        if unsafe { cv_gpu_upload_mix(ctx, flat.as_ptr(), flat.len(), off.as_ptr()) } != 0 {
            bail!("mix table upload failed");
        }
    }
    Ok(())
}

/// Build array `(family, class)` of `la` at device word `offset`.
fn build_array(ctx: *mut c_void, la: &Lookahead, offset: usize, family: usize, class: usize) -> Result<f64> {
    let mut ms = 0f32;
    let rc = unsafe {
        cv_gpu_build_array(
            ctx,
            offset,
            la.bs[class],
            la.a,
            la.magic64[class],
            la.cutoffs[class],
            la.mode.start(family),
            la.window.pm_inf as u32,
            la.window.mix.as_ref().map_or(0, |t| t.j),
            &mut ms,
        )
    };
    if rc != 0 {
        bail!("device bitvector build failed (out of GPU memory?)");
    }
    Ok(ms as f64 / 1e3)
}

/// Build every array of `la` on the device (`device_layout`); returns seconds.
fn build_all(ctx: *mut c_void, la: &Lookahead) -> Result<f64> {
    upload_mix(ctx, la)?;
    let (off, total) = la.device_layout();
    if unsafe { cv_gpu_alloc_bv(ctx, total) } != 0 {
        bail!("cannot allocate {:.1} GiB of bitvectors", (total * 4) as f64 / (1u64 << 30) as f64);
    }
    let mut secs = 0.0;
    for fam in 0..la.families {
        for i in 0..la.num_bv {
            secs += build_array(ctx, la, off[fam * la.num_bv + i], fam, i)?;
        }
    }
    Ok(secs)
}

/// Per-class launch parameters and family-0 offsets for the device layout.
fn class_params(la: &Lookahead, cutoff_bits: u32, tail_oee: bool) -> Vec<(usize, KParams)> {
    let (off, _) = la.device_layout();
    (0..la.num_bv)
        .map(|i| {
            let fam_stride = if la.families == 2 { (off[la.num_bv + i] - off[i]) as u64 } else { 0 };
            (off[i], KParams::new(la, i, cutoff_bits, tail_oee, fam_stride))
        })
        .collect()
}

/// Build all arrays on the device and compare with host-built `la` word for
/// word.  Returns (mismatched u32 words, total words, build seconds).
pub fn check_device_build(la: &Lookahead) -> Result<(u64, u64, f64)> {
    ensure!(la.has_arrays(), "need host arrays to compare against");
    let mod9 = la.mod9_u32();
    let ctx = unsafe { cv_gpu_init(std::ptr::null(), 0, 0, mod9.as_ptr(), mod9.len(), 1, 1024) };
    ensure!(!ctx.is_null(), "cv_gpu_init failed");
    let secs = build_all(ctx, la)?;
    let host = la.bv_u32();
    let mut dev = vec![0u32; host.len()];
    unsafe { cv_gpu_download_bv(ctx, 0, dev.as_mut_ptr(), dev.len()) };
    unsafe { cv_gpu_free(ctx) };
    let bad = host.iter().zip(&dev).filter(|(a, b)| a != b).count() as u64;
    Ok((bad, host.len() as u64, secs))
}

/// Fail early, with the sizes, if the bitvectors and launch buffers cannot fit.
fn check_vram(la: &Lookahead, slots: usize) -> Result<()> {
    let (mut free, mut total) = (0usize, 0usize);
    if unsafe { cv_gpu_mem_info(&mut free, &mut total) } != 0 {
        bail!("cannot query GPU memory");
    }
    let arrays = la.device_layout().1 * 4;
    let mix = la.window.mix.as_ref().map_or(0, |t| t.bytes());
    let need = arrays + mix + slots * BATCH * WORDS * 4 + (256 << 20);
    if need > free {
        let gib = |x: usize| x as f64 / (1u64 << 30) as f64;
        bail!(
            "need {:.2} GiB on the GPU ({:.2} GiB of bitvectors) but only {:.2} of {:.2} GiB are free; \
             use smaller windows (--b-hi / --class-b) or --bv-shared",
            gib(need),
            gib(arrays),
            gib(free),
            gib(total)
        );
    }
    Ok(())
}

/// Endpoints per kernel launch.
const BATCH: usize = 1 << 20;
/// Endpoints buffered per worker per class before touching shared state.
const CHUNK: usize = 1 << 12;
/// Concurrent CUDA streams (override with CV_GPU_SLOTS; 1 = serialized,
/// which makes per-class kernel times exact).
const SLOTS_DEFAULT: usize = 3;

fn slots() -> usize {
    std::env::var("CV_GPU_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(SLOTS_DEFAULT).clamp(1, 8)
}
/// u32 words per endpoint on the wire (n0 limbs, OEE state in the top bits).
const WORDS: usize = 4;

/// A pinned host buffer of `BATCH` endpoints.
struct Pinned {
    ptr: *mut u32,
    len: usize,
}
unsafe impl Send for Pinned {}

impl Pinned {
    fn alloc() -> Self {
        let ptr = unsafe { cv_gpu_host_alloc(BATCH * WORDS * 4) };
        assert!(!ptr.is_null(), "cudaHostAlloc failed");
        Pinned { ptr, len: 0 }
    }
    fn room(&self) -> usize {
        BATCH * WORDS - self.len
    }
    fn append(&mut self, src: &[u32]) {
        assert!(src.len() <= self.room());
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.add(self.len), src.len()) };
        self.len += src.len();
    }
}

struct Pool {
    free: Mutex<Vec<Pinned>>,
    ready: Condvar,
    /// Set when the GPU thread has failed: producers stop instead of waiting.
    closed: AtomicBool,
}

impl Pool {
    fn take(&self) -> Option<Pinned> {
        let mut g = self.free.lock().unwrap();
        loop {
            if self.closed.load(Ordering::Acquire) {
                return None;
            }
            if let Some(mut b) = g.pop() {
                b.len = 0;
                return Some(b);
            }
            g = self.ready.wait(g).unwrap();
        }
    }
    fn give(&self, b: Pinned) {
        self.free.lock().unwrap().push(b);
        self.ready.notify_one();
    }
    fn close(&self) {
        self.closed.store(true, Ordering::Release);
        let _g = self.free.lock().unwrap();
        self.ready.notify_all();
    }
}

struct Batch {
    idx: usize,
    buf: Pinned,
}

/// Per-class GPU accounting.
#[derive(Clone, Debug, Default, Serialize)]
pub struct GpuClassStats {
    pub launches: u64,
    pub endpoints: u64,
    pub cases: u64,
    pub iters: u64,
    pub kernel_seconds: f64,
}

#[derive(Debug, Default, Serialize)]
pub struct GpuStats {
    pub device: String,
    pub batch: usize,
    /// Seconds spent building bitvectors on the device (B > 30).
    pub build_seconds: f64,
    pub classes: Vec<GpuClassStats>,
    pub kernel_seconds: f64,
    pub hash: u32,
    pub flags: u64,
    #[serde(skip)]
    pub flagged: Vec<u128>,
}

struct Shared<'a> {
    acc: Vec<Mutex<Option<Pinned>>>,
    pool: &'a Pool,
    tx: SyncSender<Batch>,
    la: &'a Lookahead,
    cfg: &'a VerifyConfig,
    totals: Mutex<(TreeStats, TailStats)>,
}

impl Shared<'_> {
    fn push(&self, idx: usize, mut chunk: &[u32]) {
        while !chunk.is_empty() {
            if self.pool.closed.load(Ordering::Acquire) {
                return; // the GPU thread failed; its error is reported by verify_gpu
            }
            let full = {
                let mut slot = self.acc[idx].lock().unwrap();
                if slot.is_none() {
                    match self.pool.take() {
                        Some(b) => *slot = Some(b),
                        None => return,
                    }
                }
                let buf = slot.as_mut().unwrap();
                let n = chunk.len().min(buf.room());
                buf.append(&chunk[..n]);
                chunk = &chunk[n..];
                if buf.room() == 0 { slot.take() } else { None }
            };
            if let Some(buf) = full {
                if self.tx.send(Batch { idx, buf }).is_err() {
                    self.pool.close();
                    return;
                }
            }
        }
    }
}

struct GpuSink<'s, 'a> {
    sh: &'s Shared<'a>,
    chunks: Vec<Vec<u32>>,
    tree: TreeStats,
    cpu: TailStats,
}

impl Sink for GpuSink<'_, '_> {
    #[inline]
    fn endpoint(&mut self, nd: &Node) {
        if self.sh.pool.closed.load(Ordering::Relaxed) {
            return;
        }
        let Some(idx) = self.sh.la.bigindex(nd.f) else {
            process_endpoint(self.sh.la, &self.sh.cfg.tail, nd, &mut self.cpu);
            return;
        };
        let c = &mut self.chunks[idx];
        c.extend_from_slice(&[
            nd.n0 as u32,
            (nd.n0 >> 32) as u32,
            (nd.n0 >> 64) as u32,
            ((nd.n0 >> 96) as u32) | ((nd.oee as u32) << 30) | ((self.sh.la.family(nd.ne) as u32) << 29),
        ]);
        if c.len() >= WORDS * CHUNK {
            self.sh.push(idx, c);
            c.clear();
        }
    }
}

impl Drop for GpuSink<'_, '_> {
    fn drop(&mut self) {
        for idx in 0..self.chunks.len() {
            if !self.chunks[idx].is_empty() {
                self.sh.push(idx, &self.chunks[idx]);
            }
        }
        let mut t = self.sh.totals.lock().unwrap();
        t.0.merge(&self.tree);
        t.1.merge(&self.cpu);
    }
}

fn gpu_thread(la: &Lookahead, cfg: &VerifyConfig, pool: &Pool, rx: Receiver<Batch>) -> Result<GpuStats> {
    let bv = la.bv_u32();
    let mod9 = la.mod9_u32();
    let stride = array_stride(la);
    let nslots = slots();
    let ctx = unsafe { cv_gpu_init(bv.as_ptr(), bv.len(), stride, mod9.as_ptr(), mod9.len(), nslots as i32, BATCH) };
    if ctx.is_null() {
        bail!("cv_gpu_init failed");
    }
    let build_seconds = if la.has_arrays() { 0.0 } else { build_all(ctx, la)? };
    let params = class_params(la, cfg.tail.cutoff_bits, cfg.tail.tail_oee);
    let mut st = GpuStats {
        device: device_name(),
        batch: BATCH,
        classes: vec![Default::default(); la.num_bv],
        build_seconds,
        ..Default::default()
    };
    let mut running: Vec<Option<(usize, Pinned)>> = (0..nslots).map(|_| None).collect();
    let mut flagbuf = vec![0u32; 4 << 16];
    let mut collect = |slot: usize, running: &mut Vec<Option<(usize, Pinned)>>, st: &mut GpuStats| -> Result<()> {
        let Some((idx, buf)) = running[slot].take() else { return Ok(()) };
        let (mut ms, mut cases, mut iters, mut hash, mut nf) = (0f32, 0u64, 0u64, 0u32, 0u32);
        let rc = unsafe {
            cv_gpu_wait(ctx, slot as i32, &mut ms, &mut cases, &mut iters, &mut hash, &mut nf, flagbuf.as_mut_ptr(), 1 << 16)
        };
        if rc != 0 {
            bail!("kernel failed");
        }
        let c = &mut st.classes[idx];
        c.launches += 1;
        c.endpoints += (buf.len / WORDS) as u64;
        c.cases += cases;
        c.iters += iters;
        c.kernel_seconds += ms as f64 / 1e3;
        st.kernel_seconds += ms as f64 / 1e3;
        st.hash ^= hash;
        if nf > 1 << 16 {
            bail!("{nf} trajectories reached 2^{} in one launch (the flag buffer holds 65536); raise --cutoff-bits", cfg.tail.cutoff_bits);
        }
        st.flags += nf as u64;
        for i in 0..(nf as usize) {
            let w = &flagbuf[4 * i..4 * i + 4];
            st.flagged.push(w[0] as u128 | (w[1] as u128) << 32 | (w[2] as u128) << 64 | (w[3] as u128) << 96);
        }
        pool.give(buf);
        Ok(())
    };
    let mut next = 0usize;
    for batch in rx {
        let slot = next;
        next = (next + 1) % nslots;
        collect(slot, &mut running, &mut st)?;
        let f = la.offset + batch.idx as u32;
        let p3 = POW3[f as usize];
        let tp = [p3 as u32, (p3 >> 32) as u32, (p3 >> 64) as u32, (p3 >> 96) as u32];
        let n = (batch.buf.len / WORDS) as u32;
        let rc = unsafe {
            let (offset, p) = &params[batch.idx];
            cv_gpu_launch(ctx, slot as i32, batch.buf.ptr, n, *offset, la.magic64[batch.idx], tp.as_ptr(), p)
        };
        if rc != 0 {
            bail!("launch failed");
        }
        running[slot] = Some((batch.idx, batch.buf));
    }
    for slot in 0..nslots {
        collect(slot, &mut running, &mut st)?;
    }
    unsafe { cv_gpu_free(ctx) };
    Ok(st)
}

/// Verify n < 2^N with steps 2+3 on the GPU.
pub fn verify_gpu(cfg: &VerifyConfig) -> Result<VerifyReport> {
    let t0 = Instant::now();
    let m = cfg.m();
    if cfg.tail.cutoff_bits < 97 {
        bail!("the GPU path needs --cutoff-bits >= 97");
    }
    let top_f = cfg.m() * crate::arith::NUM / crate::arith::DEN + cfg.num_bv as u32;
    // the largest GPU start value is below 3^top_f 2^A; keep it under 2^120 so
    // the first odd run cannot wrap the 128-bit limbs
    if top_f as f64 * 3f64.log2() + cfg.a as f64 >= 120.0 {
        bail!("N = {} with {} bitvector classes exceeds the GPU's 128-bit headroom", cfg.n, cfg.num_bv);
    }
    let la = cfg.lookahead();
    let mut top = TreeStats::new(m);
    let units = cfg.work_units(&mut top);
    check_vram(&la, slots())?;
    let pool = Pool {
        free: Mutex::new((0..la.num_bv + slots() + 3).map(|_| Pinned::alloc()).collect()),
        ready: Condvar::new(),
        closed: AtomicBool::new(false),
    };
    let (tx, rx) = sync_channel::<Batch>(slots() + 2);
    let shared = Shared {
        acc: (0..la.num_bv).map(|_| Mutex::new(None)).collect(),
        pool: &pool,
        tx,
        la: &la,
        cfg,
        totals: Mutex::new((TreeStats::new(m), TailStats::new(la.num_bv))),
    };
    let (gpu, cpu) = std::thread::scope(|s| -> Result<(GpuStats, TailStats)> {
        let (la_ref, pool_ref) = (&la, &pool);
        let h = s.spawn(move || {
            let r = gpu_thread(la_ref, cfg, pool_ref, rx);
            if r.is_err() {
                pool_ref.close();
            }
            r
        });
        units.par_iter().for_each_init(
            || GpuSink {
                sh: &shared,
                chunks: vec![Vec::new(); la.num_bv],
                tree: TreeStats::new(m),
                cpu: TailStats::new(la.num_bv),
            },
            |sink, u| {
                let mut tree = std::mem::take(&mut sink.tree);
                dfs::<_, true>(u, m, &cfg.rules, sink, &mut tree);
                sink.tree = tree;
            },
        );
        // Every sink has flushed (dropped); send partial batches, close the channel.
        let Shared { acc, tx, totals, .. } = shared;
        for (idx, a) in acc.into_iter().enumerate() {
            if let Some(buf) = a.into_inner().unwrap() {
                if buf.len > 0 && !pool.closed.load(Ordering::Acquire) {
                    let _ = tx.send(Batch { idx, buf });
                } else {
                    pool.give(buf);
                }
            }
        }
        drop(tx);
        let (tree, cpu) = totals.into_inner().unwrap();
        top.merge(&tree);
        let g = h.join().expect("gpu thread panicked")?;
        Ok((g, cpu))
    })?;
    let mut tail = cpu;
    for (i, c) in gpu.classes.iter().enumerate() {
        tail.endpoints[i] += c.endpoints;
        tail.cases[i] += c.cases;
        tail.iters[i] += c.iters;
    }
    tail.hash ^= gpu.hash;
    tail.flagged.extend_from_slice(&gpu.flagged);
    for b in pool.free.into_inner().unwrap() {
        unsafe { cv_gpu_host_free(b.ptr) };
    }
    let seconds = t0.elapsed().as_secs_f64();
    let mut rep = VerifyReport::new(cfg, "gpu", units.len(), top, tail, seconds);
    rep.gpu = Some(serde_json::to_value(&gpu)?);
    Ok(rep)
}

/// Serialized kernel timing on caller-supplied endpoints (for `cv measure`).
/// With host arrays they are uploaded once; otherwise the two family arrays
/// of the class being timed are built on the device (B up to 39, memory
/// permitting: 2 x 2^B bits).
pub struct GpuBench {
    ctx: *mut c_void,
    buf: Pinned,
    la: Lookahead,
    cutoff_bits: u32,
    tail_oee: bool,
    per_class: bool,
    resident: Option<usize>,
    pub build_seconds: f64,
    flagbuf: Vec<u32>,
}

impl GpuBench {
    pub fn new(la: &Lookahead, cutoff_bits: u32, tail_oee: bool) -> Result<Self> {
        let bv = la.bv_u32();
        let mod9 = la.mod9_u32();
        let stride = array_stride(la);
        let ctx = unsafe { cv_gpu_init(bv.as_ptr(), bv.len(), stride, mod9.as_ptr(), mod9.len(), 1, BATCH) };
        if ctx.is_null() {
            bail!("cv_gpu_init failed");
        }
        let per_class = !la.has_arrays();
        if per_class {
            upload_mix(ctx, la)?;
        }
        // per-class mode needs one class's arrays at a time
        let mut one = la.clone();
        one.bv = Vec::new();
        if per_class {
            let widest = (0..la.num_bv).max_by_key(|&i| la.bs[i]).unwrap_or(0);
            one.bs = vec![la.bs[widest]; 1];
            one.num_bv = 1;
        }
        check_vram(&one, 1)?;
        let _ = stride;
        let mut meta = la.clone();
        meta.bv = Vec::new();
        Ok(GpuBench {
            ctx,
            buf: Pinned::alloc(),
            la: meta,
            cutoff_bits,
            tail_oee,
            per_class,
            resident: None,
            build_seconds: 0.0,
            flagbuf: vec![0u32; 4 << 16],
        })
    }

    /// Median of `reps` timed runs (after one untimed warm-up run).
    pub fn run_median(&mut self, idx: usize, nodes: &[Node], reps: usize) -> Result<GpuClassStats> {
        self.run(idx, nodes)?;
        let mut runs: Vec<GpuClassStats> = (0..reps.max(1)).map(|_| self.run(idx, nodes)).collect::<Result<_>>()?;
        runs.sort_by(|a, b| a.kernel_seconds.total_cmp(&b.kernel_seconds));
        Ok(runs.swap_remove(runs.len() / 2))
    }

    /// Run `nodes` (all of bitvector class `idx`) in full batches, one at a time.
    pub fn run(&mut self, idx: usize, nodes: &[Node]) -> Result<GpuClassStats> {
        let la = &self.la;
        // Per-class mode: only this class's arrays are resident, at offset 0.
        let (offset, params) = if self.per_class {
            let words = la.array_words(idx);
            if self.resident != Some(idx) {
                if unsafe { cv_gpu_alloc_bv(self.ctx, la.families * words) } != 0 {
                    bail!("cannot allocate {} x 2^{} bits of bitvectors on the device", la.families, la.bs[idx]);
                }
                for fam in 0..la.families {
                    self.build_seconds += build_array(self.ctx, la, fam * words, fam, idx)?;
                }
                self.resident = Some(idx);
            }
            let fam_stride = if la.families == 2 { words as u64 } else { 0 };
            (0, KParams::new(la, idx, self.cutoff_bits, self.tail_oee, fam_stride))
        } else {
            class_params(la, self.cutoff_bits, self.tail_oee).swap_remove(idx)
        };
        let mut st = GpuClassStats::default();
        let f = la.offset + idx as u32;
        let p3 = POW3[f as usize];
        let tp = [p3 as u32, (p3 >> 32) as u32, (p3 >> 64) as u32, (p3 >> 96) as u32];
        for part in nodes.chunks(BATCH) {
            self.buf.len = 0;
            for nd in part {
                debug_assert_eq!(nd.f, f);
                self.buf.append(&[
                    nd.n0 as u32,
                    (nd.n0 >> 32) as u32,
                    (nd.n0 >> 64) as u32,
                    ((nd.n0 >> 96) as u32) | ((nd.oee as u32) << 30) | ((la.family(nd.ne) as u32) << 29),
                ]);
            }
            let (mut ms, mut cases, mut iters, mut hash, mut nf) = (0f32, 0u64, 0u64, 0u32, 0u32);
            unsafe {
                if cv_gpu_launch(self.ctx, 0, self.buf.ptr, part.len() as u32, offset, la.magic64[idx], tp.as_ptr(), &params)
                    != 0
                    || cv_gpu_wait(self.ctx, 0, &mut ms, &mut cases, &mut iters, &mut hash, &mut nf, self.flagbuf.as_mut_ptr(), 1 << 16)
                        != 0
                {
                    bail!("kernel failed");
                }
            }
            st.launches += 1;
            st.endpoints += part.len() as u64;
            st.cases += cases;
            st.iters += iters;
            st.kernel_seconds += ms as f64 / 1e3;
        }
        Ok(st)
    }
}

impl Drop for GpuBench {
    fn drop(&mut self) {
        unsafe {
            cv_gpu_host_free(self.buf.ptr);
            cv_gpu_free(self.ctx);
        }
    }
}
