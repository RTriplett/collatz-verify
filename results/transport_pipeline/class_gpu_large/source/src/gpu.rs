//! Steps 2+3 on a CUDA GPU (`--features cuda`).
//!
//! CPU worker threads run the step-1 DFS and pack endpoints into per-class
//! batches (one bitvector per launch); a single GPU thread streams batches
//! through a few CUDA streams and records kernel time per class.  Endpoints
//! whose `f` lies beyond the last bitvector are finished on the CPU, as in
//! the reference program.

use crate::arith::POW3;
use crate::expansion::{DfsExpander, Expander};
use crate::lookahead::Lookahead;
use crate::tail::{process_endpoint, TailStats};
use crate::tree::{expand_terminal_group, Node, Sink, TerminalWord, TreeStats};
use crate::verify::{VerifyConfig, VerifyReport};
use anyhow::{bail, ensure, Result};
use rayon::prelude::*;
use serde::Serialize;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
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
    fn new(
        la: &Lookahead,
        class: usize,
        cutoff_bits: u32,
        tail_oee: bool,
        fam_stride: u64,
    ) -> Self {
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
    fn cv_gpu_upload_mix(
        ctx: *mut c_void,
        flat: *const u64,
        words: usize,
        offsets: *const u64,
    ) -> i32;
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
    fn cv_gpu_enable_experimental(ctx: *mut c_void, cap: usize) -> i32;
    fn cv_gpu_upload_terminal(
        ctx: *mut c_void,
        words: *const TerminalWord,
        record_count: usize,
    ) -> i32;
    fn cv_gpu_launch_ex(
        ctx: *mut c_void,
        slot: i32,
        input: *const u32,
        endpoints: u32,
        records: u32,
        mode: u32,
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
    fn cudaDeviceSynchronize() -> i32;
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
fn build_array(
    ctx: *mut c_void,
    la: &Lookahead,
    offset: usize,
    family: usize,
    class: usize,
) -> Result<f64> {
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
        bail!(
            "cannot allocate {:.1} GiB of bitvectors",
            (total * 4) as f64 / (1u64 << 30) as f64
        );
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
            let fam_stride = if la.families == 2 {
                (off[la.num_bv + i] - off[i]) as u64
            } else {
                0
            };
            (
                off[i],
                KParams::new(la, i, cutoff_bits, tail_oee, fam_stride),
            )
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
    check_vram_transport(la, slots, false)
}

fn check_vram_transport(la: &Lookahead, slots: usize, experimental: bool) -> Result<()> {
    let (mut free, mut total) = (0usize, 0usize);
    if unsafe { cv_gpu_mem_info(&mut free, &mut total) } != 0 {
        bail!("cannot query GPU memory");
    }
    let arrays = la.device_layout().1 * 4;
    let mix = la.window.mix.as_ref().map_or(0, |t| t.bytes());
    // Experimental slots reserve both maximum-size parent input and expanded
    // affine records. The regular path retains its original allocation.
    let launch_bytes = if experimental { 64 + 32 } else { WORDS * 4 };
    let need = arrays + mix + slots * BATCH * launch_bytes + (256 << 20);
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
    std::env::var("CV_GPU_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(SLOTS_DEFAULT)
        .clamp(1, 8)
}
/// u32 words per endpoint on the wire (n0 limbs, OEE state in the top bits).
const WORDS: usize = 4;

/// Experimental representations of the same logical GPU endpoint stream.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum GpuTransport {
    Replay16 = 0,
    Packed8 = 1,
    Affine32 = 2,
    Parents = 3,
}

impl GpuTransport {
    pub fn name(self) -> &'static str {
        match self {
            Self::Replay16 => "replay16",
            Self::Packed8 => "packed8",
            Self::Affine32 => "affine32",
            Self::Parents => "parents",
        }
    }

    fn record_words(self) -> usize {
        match self {
            Self::Replay16 => 4,
            Self::Packed8 => 2,
            Self::Affine32 => 8,
            Self::Parents => 16,
        }
    }
}

/// A pinned host buffer of `BATCH` endpoints.
struct Pinned {
    ptr: *mut u32,
    len: usize,
    capacity: usize,
    endpoints: usize,
    records: usize,
}
unsafe impl Send for Pinned {}

impl Pinned {
    fn alloc() -> Self {
        Self::try_alloc().expect("cudaHostAlloc failed")
    }
    fn try_alloc() -> Result<Self> {
        Self::try_alloc_words(WORDS)
    }
    fn try_alloc_words(words: usize) -> Result<Self> {
        let capacity = BATCH * words;
        let ptr = unsafe { cv_gpu_host_alloc(capacity * 4) };
        ensure!(!ptr.is_null(), "cudaHostAlloc failed");
        Ok(Pinned {
            ptr,
            len: 0,
            capacity,
            endpoints: 0,
            records: 0,
        })
    }
    fn room(&self) -> usize {
        self.capacity - self.len
    }
    fn append(&mut self, src: &[u32]) {
        assert!(src.len() <= self.room());
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.add(self.len), src.len()) };
        self.len += src.len();
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        unsafe { cv_gpu_host_free(self.ptr) };
    }
}

struct Pool {
    free: Mutex<Vec<Pinned>>,
    ready: Condvar,
    /// Set when the GPU thread has failed: producers stop instead of waiting.
    closed: AtomicBool,
    /// Sum across producer threads, not elapsed wall time.
    wait_ns: AtomicU64,
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
                b.endpoints = 0;
                b.records = 0;
                return Some(b);
            }
            let t0 = Instant::now();
            g = self.ready.wait(g).unwrap();
            self.wait_ns
                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
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
    /// One-time session setup, reported separately from a warm run.
    pub setup_seconds: f64,
    pub setup_included_in_report_seconds: bool,
    pub pipeline_seconds: f64,
    /// Includes producer computation, queue waits and sink flushing.
    pub producer_wall_seconds: f64,
    /// Producer-thread seconds blocked waiting for a free pinned buffer.
    pub producer_pool_wait_seconds: f64,
    /// Producer-thread seconds inside bounded-queue sends (including overhead).
    pub producer_queue_send_seconds: f64,
    pub streams: usize,
    pub transfer_bytes: u64,
    pub transport: &'static str,
    /// Fixed bytes per endpoint, absent for grouped parent descriptors.
    pub payload_bytes_per_endpoint: Option<usize>,
    pub input_records: u64,
    pub classes: Vec<GpuClassStats>,
    pub kernel_seconds: f64,
    /// Multiple streams overlap: kernel_seconds is not GPU busy wall time.
    pub kernel_seconds_kind: &'static str,
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
    transport: GpuTransport,
    totals: Mutex<(TreeStats, TailStats)>,
    send_ns: AtomicU64,
}

impl Shared<'_> {
    fn push(&self, idx: usize, mut chunk: &[u32]) {
        let words = self.transport.record_words();
        assert_ne!(self.transport, GpuTransport::Parents);
        assert_eq!(chunk.len() % words, 0);
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
                let n = chunk.len().min((BATCH - buf.endpoints) * words);
                buf.append(&chunk[..n]);
                buf.endpoints += n / words;
                buf.records += n / words;
                chunk = &chunk[n..];
                if buf.endpoints == BATCH {
                    slot.take()
                } else {
                    None
                }
            };
            if let Some(buf) = full {
                let t0 = Instant::now();
                let result = self.tx.send(Batch { idx, buf });
                self.send_ns
                    .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                if result.is_err() {
                    self.pool.close();
                    return;
                }
            }
        }
    }

    fn push_parents(&self, idx: usize, chunk: &[u32]) {
        assert_eq!(self.transport, GpuTransport::Parents);
        assert_eq!(chunk.len() % 16, 0);
        for descriptor in chunk.chunks_exact(16) {
            let mut record: [u32; 16] = descriptor.try_into().unwrap();
            let mut remaining = record[12] as usize;
            while remaining > 0 {
                if self.pool.closed.load(Ordering::Acquire) {
                    return;
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
                    let take = remaining.min(BATCH - buf.endpoints);
                    record[12] = take as u32;
                    record[13] = buf.endpoints as u32;
                    buf.append(&record);
                    buf.records += 1;
                    buf.endpoints += take;
                    record[11] += take as u32;
                    remaining -= take;
                    if buf.endpoints == BATCH {
                        slot.take()
                    } else {
                        None
                    }
                };
                if let Some(buf) = full {
                    let t0 = Instant::now();
                    let result = self.tx.send(Batch { idx, buf });
                    self.send_ns
                        .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    if result.is_err() {
                        self.pool.close();
                        return;
                    }
                }
            }
        }
    }

    fn push_chunk(&self, idx: usize, chunk: &[u32]) {
        if self.transport == GpuTransport::Parents {
            self.push_parents(idx, chunk);
        } else {
            self.push(idx, chunk);
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
        self.pack(idx, nd.n0, nd.oee, nd.ne, Some(nd.m));
    }

    #[inline]
    fn compact_endpoint(&mut self, n0: u64, k: u32, f: u32, oee: u8, ne: u8) {
        if self.sh.pool.closed.load(Ordering::Relaxed) {
            return;
        }
        if let Some(idx) = self.sh.la.bigindex(f) {
            debug_assert_eq!(k, self.sh.la.m);
            let m = (self.sh.transport == GpuTransport::Affine32)
                .then(|| Node::from_compact(n0, k, f, oee, ne).m);
            self.pack(idx, n0 as u128, oee, ne, m);
        } else {
            // Only the very rare out-of-range F classes need a full iterate.
            let node = Node::from_compact(n0, k, f, oee, ne);
            process_endpoint(self.sh.la, &self.sh.cfg.tail, &node, &mut self.cpu);
        }
    }

    fn terminal_group(
        &mut self,
        parent: &Node,
        entry_start: u32,
        entries: &[TerminalWord],
        target: u32,
    ) {
        if self.sh.pool.closed.load(Ordering::Relaxed) || entries.is_empty() {
            return;
        }
        if self.sh.transport != GpuTransport::Parents {
            expand_terminal_group(self, parent, entries, target);
            return;
        }
        assert_eq!(target, self.sh.la.m);
        let width = target
            .checked_sub(parent.k)
            .expect("terminal parent exceeds target");
        assert!((1..=12).contains(&width) && target <= 64);
        let f = entries[0].f;
        debug_assert!(
            entries.iter().all(|word| word.f == f),
            "parent records need one F class"
        );
        let Some(idx) = self.sh.la.bigindex(f) else {
            expand_terminal_group(self, parent, entries, target);
            return;
        };
        assert!(parent.n0 < (1u128 << parent.k));
        let p3 = POW3[parent.f as usize];
        let max_x = p3
            .checked_mul((1u128 << width) - 1)
            .and_then(|x| x.checked_add(parent.m))
            .expect("parent expansion exceeds exact 128-bit arithmetic");
        (max_x >> width)
            .checked_mul(entries[0].pow3 as u128)
            .and_then(|x| x.checked_add(u32::MAX as u128))
            .expect("terminal affine iterate exceeds exact 128-bit arithmetic");
        let descriptor = [
            parent.n0 as u32,
            (parent.n0 >> 32) as u32,
            parent.m as u32,
            (parent.m >> 32) as u32,
            (parent.m >> 64) as u32,
            (parent.m >> 96) as u32,
            p3 as u32,
            (p3 >> 32) as u32,
            (p3 >> 64) as u32,
            (p3 >> 96) as u32,
            crate::tree::terminal_delta(parent, width),
            entry_start,
            entries.len() as u32,
            0,
            width,
            parent.k,
        ];
        let chunk = &mut self.chunks[idx];
        chunk.extend_from_slice(&descriptor);
        if chunk.len() >= 16 * CHUNK {
            self.sh.push_parents(idx, chunk);
            chunk.clear();
        }
    }
}

impl GpuSink<'_, '_> {
    #[inline]
    fn pack(&mut self, idx: usize, n0: u128, oee: u8, ne: u8, m: Option<u128>) {
        let c = &mut self.chunks[idx];
        if self.sh.transport == GpuTransport::Packed8 {
            assert!(n0 < (1u128 << 62) && oee < 3);
            let packed = n0 as u64 | ((oee as u64) << 62);
            c.extend_from_slice(&[packed as u32, (packed >> 32) as u32]);
        } else {
            assert_ne!(
                self.sh.transport,
                GpuTransport::Parents,
                "parent transport requires terminal groups"
            );
            c.extend_from_slice(&[
                n0 as u32,
                (n0 >> 32) as u32,
                (n0 >> 64) as u32,
                ((n0 >> 96) as u32) | ((oee as u32) << 30) | ((self.sh.la.family(ne) as u32) << 29),
            ]);
            if self.sh.transport == GpuTransport::Affine32 {
                let m = m.expect("affine transport needs exact iterate");
                c.extend_from_slice(&[
                    m as u32,
                    (m >> 32) as u32,
                    (m >> 64) as u32,
                    (m >> 96) as u32,
                ]);
            }
        }
        if c.len() >= self.sh.transport.record_words() * CHUNK {
            self.sh.push(idx, c);
            c.clear();
        }
    }
}

impl Drop for GpuSink<'_, '_> {
    fn drop(&mut self) {
        for idx in 0..self.chunks.len() {
            if !self.chunks[idx].is_empty() {
                self.sh.push_chunk(idx, &self.chunks[idx]);
            }
        }
        let mut t = self.sh.totals.lock().unwrap();
        t.0.merge(&self.tree);
        t.1.merge(&self.cpu);
    }
}

fn gpu_run(
    ctx: *mut c_void,
    la: &Lookahead,
    tail: crate::tail::TailParams,
    pool: &Pool,
    rx: Receiver<Batch>,
    nslots: usize,
    transport: GpuTransport,
) -> Result<GpuStats> {
    let params = class_params(la, tail.cutoff_bits, tail.tail_oee);
    let mut st = GpuStats {
        device: device_name(),
        batch: BATCH,
        classes: vec![Default::default(); la.num_bv],
        streams: nslots,
        transport: transport.name(),
        payload_bytes_per_endpoint: (transport != GpuTransport::Parents)
            .then_some(transport.record_words() * 4),
        kernel_seconds_kind: "sum of CUDA stream event intervals; concurrent streams may overlap",
        ..Default::default()
    };
    let mut running: Vec<Option<(usize, Pinned)>> = (0..nslots).map(|_| None).collect();
    let mut flagbuf = vec![0u32; 4 << 16];
    let mut collect = |slot: usize,
                       running: &mut Vec<Option<(usize, Pinned)>>,
                       st: &mut GpuStats|
     -> Result<()> {
        let Some((idx, buf)) = running[slot].take() else {
            return Ok(());
        };
        let (mut ms, mut cases, mut iters, mut hash, mut nf) = (0f32, 0u64, 0u64, 0u32, 0u32);
        let rc = unsafe {
            cv_gpu_wait(
                ctx,
                slot as i32,
                &mut ms,
                &mut cases,
                &mut iters,
                &mut hash,
                &mut nf,
                flagbuf.as_mut_ptr(),
                1 << 16,
            )
        };
        if rc != 0 {
            // Keep pinned storage alive until all asynchronous access stops.
            unsafe { cudaDeviceSynchronize() };
            bail!("kernel failed");
        }
        let c = &mut st.classes[idx];
        c.launches += 1;
        c.endpoints += buf.endpoints as u64;
        c.cases += cases;
        c.iters += iters;
        c.kernel_seconds += ms as f64 / 1e3;
        st.kernel_seconds += ms as f64 / 1e3;
        st.transfer_bytes += (buf.len * 4) as u64;
        st.input_records += buf.records as u64;
        st.hash ^= hash;
        if nf > 1 << 16 {
            bail!("{nf} trajectories reached 2^{} in one launch (the flag buffer holds 65536); raise --cutoff-bits", tail.cutoff_bits);
        }
        st.flags += nf as u64;
        for i in 0..(nf as usize) {
            let w = &flagbuf[4 * i..4 * i + 4];
            st.flagged.push(
                w[0] as u128 | (w[1] as u128) << 32 | (w[2] as u128) << 64 | (w[3] as u128) << 96,
            );
        }
        pool.give(buf);
        Ok(())
    };
    let result = (|| -> Result<()> {
        let mut next = 0usize;
        for batch in rx {
            let slot = next;
            next = (next + 1) % nslots;
            collect(slot, &mut running, &mut st)?;
            let idx = batch.idx;
            let f = la.offset + idx as u32;
            let p3 = POW3[f as usize];
            let tp = [
                p3 as u32,
                (p3 >> 32) as u32,
                (p3 >> 64) as u32,
                (p3 >> 96) as u32,
            ];
            let n = batch.buf.endpoints as u32;
            let records = batch.buf.records as u32;
            let ptr = batch.buf.ptr;
            // Record ownership before launch: even a failed launch can have
            // enqueued an asynchronous host-to-device copy.
            running[slot] = Some((idx, batch.buf));
            let rc = unsafe {
                let (offset, p) = &params[idx];
                if transport == GpuTransport::Replay16 {
                    cv_gpu_launch(
                        ctx,
                        slot as i32,
                        ptr,
                        n,
                        *offset,
                        la.magic64[idx],
                        tp.as_ptr(),
                        p,
                    )
                } else {
                    cv_gpu_launch_ex(
                        ctx,
                        slot as i32,
                        ptr,
                        n,
                        records,
                        transport as u32,
                        *offset,
                        la.magic64[idx],
                        tp.as_ptr(),
                        p,
                    )
                }
            };
            ensure!(rc == 0, "launch failed");
        }
        for slot in 0..nslots {
            collect(slot, &mut running, &mut st)?;
        }
        Ok(())
    })();
    if result.is_err() {
        pool.close();
        unsafe { cudaDeviceSynchronize() };
    }
    result?;
    Ok(st)
}

fn validate_gpu_config(cfg: &VerifyConfig) -> Result<()> {
    ensure!(
        (97..=123).contains(&cfg.tail.cutoff_bits),
        "the GPU path needs cutoff-bits in 97..=123"
    );
    ensure!(
        cfg.tail.cutoff_bits >= cfg.n,
        "cutoff-bits must be at least N"
    );
    let top_f = cfg.m() * crate::arith::NUM / crate::arith::DEN + cfg.num_bv as u32;
    // the largest GPU start value is below 3^top_f 2^A; keep it under 2^120 so
    // the first odd run cannot wrap the 128-bit limbs
    if top_f as f64 * 3f64.log2() + cfg.a as f64 >= 120.0 {
        bail!(
            "N = {} with {} bitvector classes exceeds the GPU's 128-bit headroom",
            cfg.n,
            cfg.num_bv
        );
    }
    Ok(())
}

/// The arrays depend on these fields; tree policy, units and tail options may
/// change between runs without rebuilding them.
#[derive(PartialEq, Eq)]
struct SessionShape {
    n: u32,
    a: u32,
    b: u32,
    num_bv: usize,
    mode: crate::lookahead::BvMode,
    class_b: Option<Vec<u32>>,
    window: String,
}
impl SessionShape {
    fn of(cfg: &VerifyConfig) -> Self {
        Self {
            n: cfg.n,
            a: cfg.a,
            b: cfg.b,
            num_bv: cfg.num_bv,
            mode: cfg.bv_mode,
            class_b: cfg.class_b.clone(),
            window: cfg.window.label(),
        }
    }
}

struct Context(*mut c_void);
impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            cudaDeviceSynchronize();
            cv_gpu_free(self.0);
        }
    }
}

struct RunCommand {
    tail: crate::tail::TailParams,
    batches: Receiver<Batch>,
    reply: SyncSender<Result<GpuStats>>,
    transport: GpuTransport,
}

enum Command {
    Run(RunCommand),
    UploadTerminal {
        words: Vec<TerminalWord>,
        reply: SyncSender<Result<()>>,
    },
}

/// Reusable production GPU pipeline. One worker owns the CUDA context,
/// streams and arrays for its entire lifetime; repeated runs reuse the
/// pinned host pool as well. `run` reports warm pipeline wall time, while
/// setup and device-array construction are recorded separately.
pub struct GpuSession {
    la: Arc<Lookahead>,
    shape: SessionShape,
    pool: Arc<Pool>,
    commands: Option<SyncSender<Command>>,
    worker: Option<std::thread::JoinHandle<()>>,
    nslots: usize,
    experimental: bool,
    terminal_words: Vec<TerminalWord>,
    pub setup_seconds: f64,
    pub build_seconds: f64,
}

impl GpuSession {
    pub fn new(cfg: &VerifyConfig) -> Result<Self> {
        Self::create(cfg, false)
    }

    /// Larger streaming buffers for controlled transport comparisons. The
    /// production constructor continues to reserve sixteen bytes per endpoint.
    pub fn new_experimental(cfg: &VerifyConfig) -> Result<Self> {
        Self::create(cfg, true)
    }

    fn create(cfg: &VerifyConfig, experimental: bool) -> Result<Self> {
        let t0 = Instant::now();
        validate_gpu_config(cfg)?;
        let la = Arc::new(cfg.lookahead());
        let nslots = slots();
        check_vram_transport(&la, nslots, experimental)?;
        let buffers = (0..la.num_bv + nslots + 3)
            .map(|_| Pinned::try_alloc_words(if experimental { 16 } else { WORDS }))
            .collect::<Result<Vec<_>>>()?;
        let pool = Arc::new(Pool {
            free: Mutex::new(buffers),
            ready: Condvar::new(),
            closed: AtomicBool::new(false),
            wait_ns: AtomicU64::new(0),
        });
        let (commands, requests) = sync_channel::<Command>(1);
        let (ready, initialized) = sync_channel::<Result<f64>>(1);
        let worker_la = Arc::clone(&la);
        let worker_pool = Arc::clone(&pool);
        let worker = std::thread::spawn(move || {
            let init = (|| -> Result<(Context, f64)> {
                let bv = worker_la.bv_u32();
                let mod9 = worker_la.mod9_u32();
                let raw = unsafe {
                    cv_gpu_init(
                        bv.as_ptr(),
                        bv.len(),
                        array_stride(&worker_la),
                        mod9.as_ptr(),
                        mod9.len(),
                        nslots as i32,
                        BATCH,
                    )
                };
                ensure!(!raw.is_null(), "cv_gpu_init failed");
                let ctx = Context(raw);
                if experimental {
                    ensure!(
                        unsafe { cv_gpu_enable_experimental(ctx.0, BATCH) } == 0,
                        "experimental GPU buffer allocation failed"
                    );
                }
                let build = if worker_la.has_arrays() {
                    0.0
                } else {
                    build_all(ctx.0, &worker_la)?
                };
                Ok((ctx, build))
            })();
            let ctx = match init {
                Ok((ctx, build)) => {
                    if ready.send(Ok(build)).is_err() {
                        return;
                    }
                    ctx
                }
                Err(err) => {
                    let _ = ready.send(Err(err));
                    return;
                }
            };
            for command in requests {
                let failed = match command {
                    Command::Run(command) => {
                        let result = gpu_run(
                            ctx.0,
                            &worker_la,
                            command.tail,
                            &worker_pool,
                            command.batches,
                            nslots,
                            command.transport,
                        );
                        let failed = result.is_err();
                        let _ = command.reply.send(result);
                        failed
                    }
                    Command::UploadTerminal { words, reply } => {
                        let result = if unsafe {
                            cv_gpu_upload_terminal(ctx.0, words.as_ptr(), words.len())
                        } == 0
                        {
                            Ok(())
                        } else {
                            Err(anyhow::anyhow!("terminal table upload failed"))
                        };
                        let failed = result.is_err();
                        let _ = reply.send(result);
                        failed
                    }
                };
                if failed {
                    worker_pool.close();
                    break;
                }
            }
        });
        let build_seconds = match initialized.recv() {
            Ok(Ok(value)) => value,
            result => {
                drop(commands);
                let _ = worker.join();
                return match result {
                    Ok(Err(err)) => Err(err),
                    Err(_) => Err(anyhow::anyhow!("GPU worker stopped during initialization")),
                    _ => unreachable!(),
                };
            }
        };
        Ok(Self {
            la,
            shape: SessionShape::of(cfg),
            pool,
            commands: Some(commands),
            worker: Some(worker),
            nslots,
            experimental,
            terminal_words: Vec::new(),
            setup_seconds: t0.elapsed().as_secs_f64(),
            build_seconds,
        })
    }

    pub fn run<E: Expander>(&mut self, cfg: &VerifyConfig, expander: &E) -> Result<VerifyReport> {
        self.run_transport(cfg, expander, GpuTransport::Replay16, &[])
    }

    /// Upload a terminal table once, outside warm pipeline timing.
    pub fn set_terminal_table(&mut self, words: &[TerminalWord]) -> Result<()> {
        ensure!(
            self.experimental,
            "terminal tables require an experimental GPU session"
        );
        ensure!(
            !words.is_empty() && words.len() <= u32::MAX as usize,
            "invalid terminal table length"
        );
        let (reply, result) = sync_channel(1);
        self.commands
            .as_ref()
            .unwrap()
            .send(Command::UploadTerminal {
                words: words.to_vec(),
                reply,
            })
            .map_err(|_| anyhow::anyhow!("GPU worker has stopped"))?;
        result
            .recv()
            .map_err(|_| anyhow::anyhow!("GPU worker stopped during terminal table upload"))??;
        self.terminal_words = words.to_vec();
        Ok(())
    }

    pub fn run_transport<E: Expander>(
        &mut self,
        cfg: &VerifyConfig,
        expander: &E,
        transport: GpuTransport,
        terminal_words: &[TerminalWord],
    ) -> Result<VerifyReport> {
        validate_gpu_config(cfg)?;
        ensure!(
            self.experimental || transport == GpuTransport::Replay16,
            "non-default transport requires an experimental GPU session"
        );
        if transport == GpuTransport::Packed8 {
            ensure!(
                cfg.m() <= 62 && self.la.families == 1,
                "packed8 needs M <= 62 and shared bitvectors"
            );
        }
        if transport == GpuTransport::Parents {
            ensure!(cfg.m() <= 64, "parent transport needs M <= 64");
            ensure!(
                !terminal_words.is_empty() && self.terminal_words == terminal_words,
                "install the matching terminal table before running parent transport"
            );
        }
        ensure!(
            self.shape == SessionShape::of(cfg),
            "GPU session configuration differs: create a new session for different N/A/windows"
        );
        ensure!(
            !self.pool.closed.load(Ordering::Acquire),
            "GPU session is closed after a previous error"
        );
        let t0 = Instant::now();
        let m = cfg.m();
        let la = &self.la;
        let pool = &self.pool;
        pool.wait_ns.store(0, Ordering::Relaxed);
        let mut top = TreeStats::new(m);
        let units = cfg.work_units(&mut top);
        let (tx, rx) = sync_channel::<Batch>(self.nslots + 2);
        let (reply, result) = sync_channel::<Result<GpuStats>>(1);
        self.commands
            .as_ref()
            .unwrap()
            .send(Command::Run(RunCommand {
                tail: cfg.tail,
                batches: rx,
                reply,
                transport,
            }))
            .map_err(|_| anyhow::anyhow!("GPU worker has stopped"))?;
        let shared = Shared {
            acc: (0..la.num_bv).map(|_| Mutex::new(None)).collect(),
            pool,
            tx,
            la,
            cfg,
            transport,
            totals: Mutex::new((TreeStats::new(m), TailStats::new(la.num_bv))),
            send_ns: AtomicU64::new(0),
        };
        let producer_start = Instant::now();
        units.par_iter().for_each_init(
            || GpuSink {
                sh: &shared,
                chunks: vec![Vec::new(); la.num_bv],
                tree: TreeStats::new(m),
                cpu: TailStats::new(la.num_bv),
            },
            |sink, u| {
                let mut tree = std::mem::take(&mut sink.tree);
                expander.expand::<_, true>(u, m, &cfg.rules, sink, &mut tree);
                sink.tree = tree;
            },
        );
        // Every sink has flushed (dropped); send partial batches, close the channel.
        let Shared {
            acc,
            tx,
            totals,
            send_ns,
            ..
        } = shared;
        let mut send_ns = send_ns.into_inner();
        for (idx, a) in acc.into_iter().enumerate() {
            if let Some(buf) = a.into_inner().unwrap() {
                if buf.len > 0 && !pool.closed.load(Ordering::Acquire) {
                    let send_start = Instant::now();
                    let sent = tx.send(Batch { idx, buf });
                    send_ns += send_start.elapsed().as_nanos() as u64;
                    if sent.is_err() {
                        pool.close();
                    }
                } else {
                    pool.give(buf);
                }
            }
        }
        drop(tx);
        let producer_wall_seconds = producer_start.elapsed().as_secs_f64();
        let (tree, cpu) = totals.into_inner().unwrap();
        top.merge(&tree);
        let mut gpu = result
            .recv()
            .map_err(|_| anyhow::anyhow!("GPU worker stopped during verification"))??;
        let mut tail = cpu;
        for (i, c) in gpu.classes.iter().enumerate() {
            tail.endpoints[i] += c.endpoints;
            tail.cases[i] += c.cases;
            tail.iters[i] += c.iters;
        }
        tail.hash ^= gpu.hash;
        tail.flagged.extend_from_slice(&gpu.flagged);
        let seconds = t0.elapsed().as_secs_f64();
        gpu.build_seconds = self.build_seconds;
        gpu.setup_seconds = self.setup_seconds;
        gpu.pipeline_seconds = seconds;
        gpu.producer_wall_seconds = producer_wall_seconds;
        gpu.producer_pool_wait_seconds = pool.wait_ns.load(Ordering::Relaxed) as f64 / 1e9;
        gpu.producer_queue_send_seconds = send_ns as f64 / 1e9;
        let mut rep = VerifyReport::new(cfg, "gpu", units.len(), top, tail, seconds);
        rep.gpu = Some(serde_json::to_value(&gpu)?);
        Ok(rep)
    }
}

impl Drop for GpuSession {
    fn drop(&mut self) {
        drop(self.commands.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Verify n < 2^N with steps 2+3 on the GPU, including cold setup/teardown.
pub fn verify_gpu(cfg: &VerifyConfig) -> Result<VerifyReport> {
    let t0 = Instant::now();
    let mut session = GpuSession::new(cfg)?;
    let mut rep = session.run(cfg, &DfsExpander)?;
    drop(session);
    rep.seconds = t0.elapsed().as_secs_f64();
    if let Some(gpu) = rep.gpu.as_mut() {
        gpu["setup_included_in_report_seconds"] = serde_json::json!(true);
    }
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
    /// Work checksum and overflow count from the most recent isolated run.
    pub last_hash: u32,
    pub last_flags: u64,
    flagbuf: Vec<u32>,
}

impl GpuBench {
    pub fn new(la: &Lookahead, cutoff_bits: u32, tail_oee: bool) -> Result<Self> {
        let bv = la.bv_u32();
        let mod9 = la.mod9_u32();
        let stride = array_stride(la);
        let ctx = unsafe {
            cv_gpu_init(
                bv.as_ptr(),
                bv.len(),
                stride,
                mod9.as_ptr(),
                mod9.len(),
                1,
                BATCH,
            )
        };
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
            last_hash: 0,
            last_flags: 0,
            flagbuf: vec![0u32; 4 << 16],
        })
    }

    /// Median of `reps` timed runs (after one untimed warm-up run).
    pub fn run_median(&mut self, idx: usize, nodes: &[Node], reps: usize) -> Result<GpuClassStats> {
        self.run(idx, nodes)?;
        let mut runs: Vec<GpuClassStats> = (0..reps.max(1))
            .map(|_| self.run(idx, nodes))
            .collect::<Result<_>>()?;
        runs.sort_by(|a, b| a.kernel_seconds.total_cmp(&b.kernel_seconds));
        Ok(runs.swap_remove(runs.len() / 2))
    }

    /// Run `nodes` (all of bitvector class `idx`) in full batches, one at a time.
    pub fn run(&mut self, idx: usize, nodes: &[Node]) -> Result<GpuClassStats> {
        self.last_hash = 0;
        self.last_flags = 0;
        let la = &self.la;
        // Per-class mode: only this class's arrays are resident, at offset 0.
        let (offset, params) = if self.per_class {
            let words = la.array_words(idx);
            if self.resident != Some(idx) {
                if unsafe { cv_gpu_alloc_bv(self.ctx, la.families * words) } != 0 {
                    bail!(
                        "cannot allocate {} x 2^{} bits of bitvectors on the device",
                        la.families,
                        la.bs[idx]
                    );
                }
                for fam in 0..la.families {
                    self.build_seconds += build_array(self.ctx, la, fam * words, fam, idx)?;
                }
                self.resident = Some(idx);
            }
            let fam_stride = if la.families == 2 { words as u64 } else { 0 };
            (
                0,
                KParams::new(la, idx, self.cutoff_bits, self.tail_oee, fam_stride),
            )
        } else {
            class_params(la, self.cutoff_bits, self.tail_oee).swap_remove(idx)
        };
        let mut st = GpuClassStats::default();
        let f = la.offset + idx as u32;
        let p3 = POW3[f as usize];
        let tp = [
            p3 as u32,
            (p3 >> 32) as u32,
            (p3 >> 64) as u32,
            (p3 >> 96) as u32,
        ];
        for part in nodes.chunks(BATCH) {
            self.buf.len = 0;
            for nd in part {
                debug_assert_eq!(nd.f, f);
                self.buf.append(&[
                    nd.n0 as u32,
                    (nd.n0 >> 32) as u32,
                    (nd.n0 >> 64) as u32,
                    ((nd.n0 >> 96) as u32)
                        | ((nd.oee as u32) << 30)
                        | ((la.family(nd.ne) as u32) << 29),
                ]);
            }
            let (mut ms, mut cases, mut iters, mut hash, mut nf) = (0f32, 0u64, 0u64, 0u32, 0u32);
            unsafe {
                if cv_gpu_launch(
                    self.ctx,
                    0,
                    self.buf.ptr,
                    part.len() as u32,
                    offset,
                    la.magic64[idx],
                    tp.as_ptr(),
                    &params,
                ) != 0
                    || cv_gpu_wait(
                        self.ctx,
                        0,
                        &mut ms,
                        &mut cases,
                        &mut iters,
                        &mut hash,
                        &mut nf,
                        self.flagbuf.as_mut_ptr(),
                        1 << 16,
                    ) != 0
                {
                    bail!("kernel failed");
                }
            }
            st.launches += 1;
            self.last_hash ^= hash;
            self.last_flags += nf as u64;
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
            cv_gpu_free(self.ctx);
        }
    }
}
