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
use crate::tree::{expand_chained_group, expand_terminal_group, ChainLeaf, ChainProgram, ChainRoute, ChainTables, Node, Sink, TerminalWord, TreeStats};
use crate::verify::{VerifyConfig, VerifyReport};
use anyhow::{bail, ensure, Result};
use rayon::prelude::*;
use serde::Serialize;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

#[path = "gpu_deferred.rs"]
mod deferred;
use deferred::DeferredWorkers;
#[path = "gpu_early.rs"]
mod early;
use early::EarlyWorkers;

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
    tail_budget: u32,
    tail_cap: u32,
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
            tail_budget: 0,
            tail_cap: 0,
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
    fn cv_gpu_current_context_id() -> u64;
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
    // c3-hyg: per-context dip tables (cuda/jdip.cuh jd_upload), measurement
    // opt-in, context identity, warp-queue layout kinds, jd traits.
    fn cv_jd_upload(ctx: *mut c_void, wd: u32, dq: u32, desc: u32, quanta: *const u16, n: u64) -> i32;
    fn cv_jd_present(ctx: *mut c_void, wd: u32, dq: u32, desc: u32) -> i32;
    fn cv_gpu_allow_measurement(ctx: *mut c_void, allow: i32) -> i32;
    fn cv_gpu_context_ok(ctx: *mut c_void) -> i32;
    fn cv_wq_layout_kind(v: u32) -> i32;
    fn cv_tp_layout_kind(v: u32) -> i32;
    fn cv_jd_code_traits(tv: u32, out: *mut i32) -> i32;
    fn cv_gpu_set_tail_jump(ctx: *mut c_void, width: u32) -> i32;
    fn cv_gpu_set_tail_capacity(ctx: *mut c_void, capacity: u32) -> i32;
    fn cv_gpu_wait_deferred(ctx: *mut c_void, slot: i32, attempts: *mut u32,
                            records: *mut u32, capacity: usize) -> i32;
    fn cv_gpu_prepare_screen(ctx: *mut c_void, words32: u32) -> i32;
    fn cv_gpu_launch_screen_ex(
        ctx: *mut c_void, slot: i32, input: *const u32, endpoints: u32,
        records: u32, mode: u32, bv_offset: usize, magic: u64,
        tp: *const u32, p: *const KParams,
    ) -> i32;
    fn cv_gpu_wait_screen(
        ctx: *mut c_void, slot: i32, kernel_ms: *mut f32, d2h_ms: *mut f32,
        affine: *mut *const u32, masks: *mut *const u32,
        endpoints: *mut u32, words32: *mut u32,
    ) -> i32;
    fn cv_gpu_upload_terminal(
        ctx: *mut c_void,
        words: *const TerminalWord,
        record_count: usize,
    ) -> i32;
    fn cv_gpu_upload_chain(
        ctx: *mut c_void,
        programs: *const ChainProgram, program_count: usize,
        leaves: *const ChainLeaf, leaf_count: usize,
        words: *const TerminalWord, word_count: usize,
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
    fn cv_gpu_last_dbg(ctx: *mut c_void, slot: i32, out6: *mut u64) -> i32;
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
    fn cv_gpu_launch_scr(
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
        spec: *const ScreenSpec,
    ) -> i32;
    fn cv_gpu_screen_words(
        ctx: *mut c_void,
        slot: i32,
        input: *const u32,
        endpoints: u32,
        bv_offset: usize,
        magic: u64,
        tp: *const u32,
        p: *const KParams,
        spec: *const ScreenSpec,
        out: *mut u32,
    ) -> i32;
    fn cv_gpu_upload_rows(
        ctx: *mut c_void,
        bits: *const u32,
        nwords: usize,
        rowmap: *const u16,
        kbits: u32,
        row_words: u32,
        d: u32,
        minv: u32,
        limb_c: *const u32,
        a: u32,
    ) -> i32;
}

/// c-screen: the CUDA `ScreenSpec` (cuda/screen.cuh).  All zero is the
/// ordinary launch.  `diag` selects a diagnostic flavour that is not a
/// verification (4 load-only, 8 screen-only, 16 screen + case setup).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ScreenSpec {
    pub rows: u32,
    pub w_first: u32,
    pub w_end: u32,
    pub diag: u32,
}

/// jd FD3 codes with screened per-thread kernels (cuda/jdscreen.cuh JDS_CODES).
pub const JDS_CODES: [u32; 7] = [90, 91, 93, 95, 96, 103, 111];
/// Tail codes the warp-queue scheduler can run with a screen (cuda/wq_jd.cuh WQS_JD).
pub const WQS_CODES: [u32; 3] = [91, 92, 95];
/// jd FD3 codes the warp-queue scheduler runs (cuda/wq_jd.cuh wq_jd_code).
pub const WQ_JD_CODES: [u32; 4] = [91, 92, 94, 95];

/// Can a launch of this tail code carry a non-default `ScreenSpec`?
pub fn screen_code_ok(code: u32) -> bool {
    (49..=53).contains(&code) || JDS_CODES.contains(&code)
}
pub fn wq_screen_code_ok(code: u32) -> bool {
    WQS_CODES.contains(&code)
}

pub const SCREEN_DIAG_LOAD: u32 = 4;
pub const SCREEN_DIAG_SCAN: u32 = 8;
pub const SCREEN_DIAG_SETUP: u32 = 16;
/// Scan options (work-preserving): mod-9 word prefetch (no rows) and lazy
/// row-word fetch (rows); carried in the same `diag` field.
pub const SCREEN_OPT_M9PF: u32 = 32;
pub const SCREEN_OPT_LAZY: u32 = 64;
/// c-final: rows flavours of the jd screen kernels compiled with the
/// 48-register launch bound (cuda/jdscreen.cuh SCR_LB5); work-preserving.
pub const SCREEN_OPT_LB5: u32 = 128;

impl ScreenSpec {
    /// Device request for launches of bitvector class `class` in a run with
    /// `screen` on `la` (u32 completion words; per-class row gating).
    pub fn of(screen: &crate::tail::Screen, la: &Lookahead, class: usize, diag: u32) -> Result<Self> {
        ensure!(matches!(diag, 0 | SCREEN_DIAG_LOAD | SCREEN_DIAG_SCAN | SCREEN_DIAG_SETUP | SCREEN_OPT_M9PF | SCREEN_OPT_LAZY
                         | SCREEN_OPT_LB5),
            "invalid screen diagnostic/option {diag}");
        let (lo64, hi64) = screen.word_range(la.a);
        let words32 = (1u32 << la.a) / 32;
        let (w_first, w_end) = ((2 * lo64) as u32, (2 * hi64) as u32);
        ensure!(!screen.rows || la.rows.is_some(), "screen.rows requested but the lookahead carries no row table");
        let rows = screen.rows_for(Some(class));
        // Options apply where they are defined (M9PF without rows, LAZY with rows).
        let diag = match diag {
            SCREEN_OPT_M9PF if rows => 0,
            SCREEN_OPT_LAZY if !rows => 0,
            SCREEN_OPT_LB5 if !rows => 0,
            d => d,
        };
        Ok(ScreenSpec {
            rows: rows as u32,
            w_first,
            w_end: if w_end == words32 { 0 } else { w_end },
            diag,
        })
    }
    pub fn is_default(&self) -> bool {
        *self == ScreenSpec::default()
    }
}

/// Upload `la`'s row table (if any) into a context.
fn upload_rows(ctx: *mut c_void, la: &Lookahead) -> Result<()> {
    let Some(r) = &la.rows else { return Ok(()) };
    let rc = unsafe {
        cv_gpu_upload_rows(ctx, r.bits.as_ptr(), r.bits.len(), r.rowmap.as_ptr(), r.k, r.row_words as u32,
                           r.d, r.minv32, r.limb_c.as_ptr(), r.a)
    };
    ensure!(rc == 0, "row table upload failed");
    Ok(())
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
// Shared by every final-F program for a compact parent.
const CHAIN_INV3: [u32; 65] = {
    let mut values = [1u32; 65];
    let mut i = 1;
    while i < values.len() {
        values[i] = values[i - 1].wrapping_mul(0xaaaa_aaab);
        i += 1;
    }
    values
};

const BATCH: usize = 1 << 20;

/// c3-a: endpoints per pipeline launch (CV_GPU_BATCH: a power of two in [2^14, BATCH], default BATCH).
/// Buffers keep BATCH capacity; only the point at which a class buffer is considered full moves.
fn batch_fill() -> usize {
    static FILL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *FILL.get_or_init(|| match std::env::var("CV_GPU_BATCH") {
        Ok(v) => {
            let b: usize = v.parse().expect("CV_GPU_BATCH");
            assert!(b.is_power_of_two() && ((1 << 14)..=BATCH).contains(&b),
                "CV_GPU_BATCH must be a power of two in [16384, {BATCH}]");
            b
        }
        Err(_) => BATCH,
    })
}
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
    CompactChain = 4,
}

impl GpuTransport {
    pub fn name(self) -> &'static str {
        match self {
            Self::Replay16 => "replay16",
            Self::Packed8 => "packed8",
            Self::Affine32 => "affine32",
            Self::Parents => "parents",
            Self::CompactChain => "compact_chain",
        }
    }

    fn is_grouped(self) -> bool {
        matches!(self, Self::Parents | Self::CompactChain)
    }

    fn record_words(self) -> usize {
        match self {
            Self::Replay16 => 4,
            Self::Packed8 => 2,
            Self::Affine32 => 8,
            Self::Parents | Self::CompactChain => 16,
        }
    }
}

/// Opt-in kernel and host dispatch experiments. Defaults retain current behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GpuTuning {
    pub block_threads: u32,
    pub overflow_vote: bool,
    pub readonly_powers: bool,
    pub oee_mask: bool,
    /// Experimental low-bit affine tail table; zero retains ordinary stepping.
    #[serde(skip_serializing_if = "zero_jump_bits")]
    pub tail_jump_bits: u32,
    /// Per-producer grouped descriptor flush threshold; omitted means 4096.
    /// This changes host scheduling only, never CUDA endpoint batch capacity.
    #[serde(skip_serializing_if = "default_group_chunk_records")]
    pub group_chunk_records: u32,
    /// Completed double iterations per candidate before offering CPU handoff.
    #[serde(skip_serializing_if = "zero_jump_bits")]
    pub tail_budget: u32,
    /// Continuation records per GPU launch; a full queue continues on the GPU.
    #[serde(skip_serializing_if = "default_tail_capacity")]
    pub tail_queue_capacity: u32,
    #[serde(skip_serializing_if = "default_tail_workers")]
    pub tail_cpu_workers: u32,
    /// Screen every Nth whole launch on the GPU and finish its survivors on
    /// the CPU. Zero disables this experiment; the first launch is index zero.
    #[serde(skip_serializing_if = "zero_jump_bits")]
    pub early_every: u32,
    /// 0 scalar, 1 bucketed scalar, 2 table, 3 initially grouped table,
    /// 4 regroup active states after every shared table transition.
    #[serde(skip_serializing_if = "zero_jump_bits")]
    pub early_mode: u32,
    #[serde(skip_serializing_if = "default_early_bits")]
    pub early_group_bits: u32,
    #[serde(skip_serializing_if = "default_tail_workers")]
    pub early_cpu_workers: u32,
    /// Experimental tail variant per bitvector class (cuda/xtail.cuh codes;
    /// 0 keeps the original loop). Requires the no-vote early-OEE kernel.
    /// (32 entries since c-screen: the forced-overflow fixture has 24 classes.)
    #[serde(skip_serializing_if = "no_xtail")]
    pub xtail: [u8; 32],
    /// c-screen diagnostic kernel flavour (0 = verify; see `ScreenSpec`).
    #[serde(skip_serializing_if = "zero_jump_bits")]
    pub screen_diag: u32,
    /// Warp-queue scheduler bits per class (cuda/wq_jump.cuh: bit 22 set, K in
    /// bits 3-7, log2 fetch in 10-12, blocks/SM cap in 13-15, layout in 16-20
    /// plus bit 23).  Zero keeps the per-thread kernel. Requires a jump code in
    /// `xtail`.  (32 entries, as `xtail`.)
    #[serde(skip_serializing_if = "no_wq")]
    pub wq: [u32; 32],
    /// c3-a: CUDA streams used by this run (0 = every session slot); at most the session's slot count.
    #[serde(skip_serializing_if = "zero_jump_bits")]
    pub run_slots: u32,
    /// c3-a: endpoints per launch for this run (0 = CV_GPU_BATCH or BATCH); a power of two in [2^14, BATCH].
    #[serde(skip_serializing_if = "zero_jump_bits")]
    pub run_batch: u32,
}

fn no_xtail(x: &[u8; 32]) -> bool { x.iter().all(|&c| c == 0) }
fn no_wq(x: &[u32; 32]) -> bool { x.iter().all(|&c| c == 0) }

/// Allowed warp-queue fields of `GpuTuning::wq` (bit 21, the timing-only
/// capped test tail, is deliberately excluded from verification runs).
pub use crate::runspec::WQ_FIELDS;
/// c3-final (c3-2ph): two-phase kernel fields and codes (shared with CPU runs).
pub use crate::runspec::{is_two_phase, TP_CODES, TP_FIELDS};

/// Warp-queue bits from a spec `v<layout>[.k<K>][.c<log2 fetch>][.o<blocks/SM>]`
/// (fetch defaults to 2 endpoints); `-` or `0` keeps the per-thread kernel.
/// (c3-hyg: the parser lives in `crate::runspec`, shared with CPU runs.)
pub fn parse_wq(spec: &str) -> Result<u32> {
    crate::runspec::parse_wq_spec(spec)
}

/// Is a screen diagnostic measurement-only (load-only / screen-only /
/// screen + setup kernels run no tails, or no screening)?
pub fn measurement_diag(diag: u32) -> bool {
    matches!(diag, SCREEN_DIAG_LOAD | SCREEN_DIAG_SCAN | SCREEN_DIAG_SETUP)
}

/// c3-hyg: the CUDA traits of a jd code (valid, w, wd, dq, ka, desc, floor,
/// lay), for the consistency check against `jump::jd_kind`.  Host-only call;
/// creates no CUDA context.
pub fn jd_code_traits(code: u32) -> [i32; 8] {
    let mut out = [0i32; 8];
    unsafe { cv_jd_code_traits(code, out.as_mut_ptr()) };
    out
}

/// c3-hyg: warp-queue layout kind (bit 0 valid, bit 1 per-case digest
/// layout, bit 2 instrumented counters).  Host-only call.
pub fn wq_layout_kind(layout: u32) -> i32 {
    unsafe { cv_wq_layout_kind(layout) }
}

/// c3-final: two-phase layout kind (cuda/twophase.cuh TpCfg<V>): bit 0 valid,
/// bit 1 per-case digest (DIG), bit 2 instrumented counters (INSTR).
pub fn tp_layout_kind(layout: u32) -> i32 {
    unsafe { cv_tp_layout_kind(layout) }
}

/// Layout V of warp-queue pad bits (bits 16-20 plus bit 23).
pub fn wq_layout_of(wq: u32) -> u32 {
    ((wq >> 16) & 31) | (((wq >> 23) & 1) << 5)
}

/// c3-hyg: accumulate `KOut.dbg` of one launch chunk into `acc`.  Digest
/// layouts (sum, XOR, count) XOR word 1 and sum the others; every other
/// layout's counters are sums.  (Round 2 summed the XOR digest word too.)
pub fn accumulate_dbg(acc: &mut [u64; 6], chunk: &[u64; 6], wq: u32) {
    // c3-final: two-phase digest layouts (c3-2ph's DIG copies) XOR word 1 too.
    let kind = if is_two_phase(wq) { tp_layout_kind((wq >> 16) & 31) } else { wq_layout_kind(wq_layout_of(wq)) };
    let digest = wq != 0 && (kind & 2) != 0;
    for i in 0..6 {
        acc[i] = if digest && i == 1 { acc[i] ^ chunk[i] } else { acc[i].wrapping_add(chunk[i]) };
    }
}

/// Host dip quanta of a jd key (cached process-wide: pure CPU data, built
/// sequentially because this can run on the pipeline's GPU thread while every
/// rayon worker waits for it).
fn jd_host_quanta(wd: u32, dq: u32, desc: bool) -> Arc<Vec<u16>> {
    static CACHE: Mutex<Vec<((u32, u32, bool), Arc<Vec<u16>>)>> = Mutex::new(Vec::new());
    if let Some((_, q)) = CACHE.lock().unwrap().iter().find(|(k, _)| *k == (wd, dq, desc)) {
        return Arc::clone(q);
    }
    let t = crate::jdip::DipTable::build_seq(wd, crate::jump::production_window_rules(), dq, desc);
    let q = Arc::new(t.quanta());
    let mut cache = CACHE.lock().unwrap();
    if let Some((_, existing)) = cache.iter().find(|(k, _)| *k == (wd, dq, desc)) {
        return Arc::clone(existing);
    }
    cache.push(((wd, dq, desc), Arc::clone(&q)));
    q
}

/// c3-hyg: upload the dip tables of `codes` (cuda/jdip.cuh codes >= 64) into
/// one context, unless that context already holds them.  The tables are per
/// context (cuda/devtables.cuh): a new context, or one created after a device
/// reset, uploads its own.
fn upload_jd_ctx(ctx: *mut c_void, codes: &[u32]) -> Result<()> {
    for &code in codes {
        if code < 64 {
            continue;
        }
        let Some(crate::jump::TailKind::JumpDip { wd, dq, desc, .. }) = crate::jump::jd_kind(code) else {
            bail!("not a jd code: {code}");
        };
        if unsafe { cv_jd_present(ctx, wd, dq, desc as u32) } == 1 {
            continue;
        }
        let q = jd_host_quanta(wd, dq, desc);
        let rc = unsafe { cv_jd_upload(ctx, wd, dq, desc as u32, q.as_ptr(), q.len() as u64) };
        ensure!(rc == 0, "cv_jd_upload failed for code {code} (wd {wd}, dq {dq}, desc {desc}): {rc}");
    }
    Ok(())
}

fn zero_jump_bits(bits: &u32) -> bool { *bits == 0 }
fn default_group_chunk_records(records: &u32) -> bool { *records == CHUNK as u32 }
fn default_tail_capacity(capacity: &u32) -> bool { *capacity == 65_536 }
fn default_tail_workers(workers: &u32) -> bool { *workers == 1 }
fn default_early_bits(bits: &u32) -> bool { *bits == 10 }

impl Default for GpuTuning {
    fn default() -> Self {
        Self {
            block_threads: 256,
            overflow_vote: true,
            readonly_powers: false,
            oee_mask: false,
            tail_jump_bits: 0,
            group_chunk_records: CHUNK as u32,
            tail_budget: 0,
            tail_queue_capacity: 65_536,
            tail_cpu_workers: 1,
            early_every: 0,
            early_mode: 0,
            early_group_bits: 10,
            early_cpu_workers: 1,
            xtail: [0; 32],
            screen_diag: 0,
            wq: [0; 32],
            run_slots: 0,
            run_batch: 0,
        }
    }
}

impl GpuTuning {
    fn bits(self) -> Result<u32> {
        ensure!(self.early_mode <= 4, "early CPU mode must be in 0..=4");
        ensure!(matches!(self.early_group_bits, 8 | 10 | 12 | 14),
            "early CPU grouping width must be 8, 10, 12 or 14");
        ensure!((1..=8).contains(&self.early_cpu_workers), "early CPU workers must be in 1..=8");
        ensure!(self.early_every == 0 || (self.tail_budget == 0 && self.tail_jump_bits == 0
            && !self.overflow_vote && !self.readonly_powers && self.oee_mask),
            "early CPU screening requires no-vote early-OEE without tail jumps or late offload");
        ensure!((1..=1_048_576).contains(&self.tail_queue_capacity),
            "tail continuation capacity must be in 1..=1048576");
        ensure!((1..=8).contains(&self.tail_cpu_workers), "tail CPU workers must be in 1..=8");
        ensure!(self.tail_budget == 0 || (!self.overflow_vote && !self.readonly_powers
            && self.oee_mask && self.tail_jump_bits == 0),
            "CPU tail continuation requires the no-vote, early-OEE kernel without jumps");
        ensure!(self.group_chunk_records > 0 && self.group_chunk_records <= CHUNK as u32,
            "grouped descriptor chunk limit must be in 1..=4096");
        ensure!(matches!(self.tail_jump_bits, 0 | 8 | 10 | 12 | 14),
            "tail jump width must be 0, 8, 10, 12 or 14");
        ensure!(self.run_batch == 0 || (self.run_batch.is_power_of_two()
            && ((1u32 << 14)..=BATCH as u32).contains(&self.run_batch)),
            "run_batch must be 0 or a power of two in [16384, {BATCH}]");
        ensure!(self.run_slots <= 8, "run_slots must be in 0..=8");
        ensure!(self.tail_jump_bits == 0 || (!self.overflow_vote && !self.readonly_powers && self.oee_mask),
            "tail jumps require the no-vote, early-OEE kernel");
        let block = match self.block_threads {
            256 => 0,
            128 => 1,
            512 => 2,
            other => bail!("unsupported GPU block size {other}; use 128, 256 or 512"),
        };
        Ok(((!self.overflow_vote) as u32)
            | ((self.readonly_powers as u32) << 1)
            | ((self.oee_mask as u32) << 2)
            | (block << 8)
            | (self.tail_jump_bits << 16))
    }
}

/// A pinned host buffer of `BATCH` endpoints.
struct Pinned {
    ptr: *mut u32,
    len: usize,
    capacity: usize,
    endpoints: usize,
    records: usize,
    /// c3-hyg: CUDA context the buffer was allocated in (0: ids unavailable).
    /// A reset context has already released it; freeing it again could free
    /// a newer allocation at the same address, so `drop` then leaks the handle.
    ctx_id: u64,
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
            ctx_id: unsafe { cv_gpu_current_context_id() },
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
        if self.ctx_id == 0 || unsafe { cv_gpu_current_context_id() } == self.ctx_id {
            unsafe { cv_gpu_host_free(self.ptr) };
        }
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
    pub tuning: GpuTuning,
    /// Fixed bytes per endpoint, absent for grouped parent descriptors.
    pub payload_bytes_per_endpoint: Option<usize>,
    pub input_records: u64,
    pub classes: Vec<GpuClassStats>,
    pub kernel_seconds: f64,
    /// Multiple streams overlap: kernel_seconds is not GPU busy wall time.
    pub kernel_seconds_kind: &'static str,
    pub hash: u32,
    pub flags: u64,
    /// Logical GPU-origin work above includes completed CPU continuations.
    /// Subtract deferred_iterations/flags and XOR deferred_hash to recover
    /// device-only work.
    pub deferred_attempts: u64,
    pub deferred_cases: u64,
    pub deferred_rejected: u64,
    pub deferred_iterations: u64,
    pub deferred_hash: u32,
    pub deferred_flags: u64,
    /// Device-to-host continuation records; separate from input transfer_bytes.
    pub deferred_transfer_bytes: u64,
    pub deferred_cpu_seconds: f64,
    pub deferred_queue_wait_seconds: f64,
    pub deferred_drain_seconds: f64,
    /// Ordinary fused GPU launches, separate from early screen-only batches.
    pub regular_batches: u64,
    pub early_batches: u64,
    pub early_endpoints: u64,
    /// Whole-launch selection can change class mix when producers interleave.
    pub early_class_endpoints: Vec<u64>,
    pub early_class_cases: Vec<u64>,
    pub early_class_iterations: Vec<u64>,
    pub early_cases: u64,
    pub early_iterations: u64,
    pub early_hash: u32,
    pub early_flags: u64,
    /// Full affine context and survivor masks copied device-to-host.
    pub early_transfer_bytes: u64,
    /// Sum of stream event intervals around early device-to-host copies.
    pub early_copy_seconds: f64,
    /// CPU thread time including reconstruction, grouping and verification.
    pub early_cpu_seconds: f64,
    pub early_decode_seconds: f64,
    /// Host thread time copying compact exports into owned worker jobs.
    pub early_host_copy_seconds: f64,
    pub early_group_seconds: f64,
    pub early_verification_seconds: f64,
    pub early_table_setup_seconds: f64,
    pub early_queue_wait_seconds: f64,
    pub early_drain_seconds: f64,
    pub early_buckets: u64,
    pub early_jumps: u64,
    pub early_jumped_iterations: u64,
    pub early_scalar_iterations: u64,
    pub early_table_lookups: u64,
    /// Diagnostic raw-state export; nonzero runs are not performance samples.
    pub early_capture_cases: u64,
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
    group_chunk_records: usize,
    totals: Mutex<(TreeStats, TailStats)>,
    send_ns: AtomicU64,
    /// c3-a: endpoints per launch for this run.
    fill: usize,
}

impl Shared<'_> {
    fn push(&self, idx: usize, mut chunk: &[u32]) {
        let words = self.transport.record_words();
        assert!(!self.transport.is_grouped());
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
                let fill = self.fill;
                let n = chunk.len().min((fill - buf.endpoints) * words);
                buf.append(&chunk[..n]);
                buf.endpoints += n / words;
                buf.records += n / words;
                chunk = &chunk[n..];
                if buf.endpoints == fill {
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
        assert!(self.transport.is_grouped());
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
                    let fill = self.fill;
                    let take = remaining.min(fill - buf.endpoints);
                    record[12] = take as u32;
                    record[13] = buf.endpoints as u32;
                    buf.append(&record);
                    buf.records += 1;
                    buf.endpoints += take;
                    if self.transport == GpuTransport::CompactChain {
                        record[14] += take as u32;
                    } else {
                        record[11] += take as u32;
                    }
                    remaining -= take;
                    if buf.endpoints == fill {
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
        if self.transport.is_grouped() {
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

    fn chained_group(&mut self, parent: &Node, program_index: u32,
                     program: &ChainProgram, routes: &[ChainRoute],
                     words: &[TerminalWord], target: u32) {
        if self.sh.pool.closed.load(Ordering::Relaxed) || program.endpoints == 0 { return; }
        if self.sh.transport != GpuTransport::CompactChain {
            expand_chained_group(self, parent, program, routes, words, target);
            return;
        }
        assert_eq!(target, self.sh.la.m);
        let first_route = routes.first().expect("nonempty chain program");
        let final_f = words[first_route.second_start as usize].f;
        let Some(idx) = self.sh.la.bigindex(final_f) else {
            expand_chained_group(self, parent, program, routes, words, target);
            return;
        };
        let width = target.checked_sub(parent.k).expect("chain parent exceeds target");
        assert_eq!(width, program.first_width + program.second_width);
        assert!((1..=6).contains(&program.first_width) && (1..=12).contains(&program.second_width));
        assert!(target <= 64 && parent.n0 < (1u128 << parent.k));
        let p3 = POW3[parent.f as usize];
        // Both affine stages are exact within the same full-width extension
        // bound; checking the total width also covers the first-stage input.
        let max_x = p3.checked_mul((1u128 << width) - 1)
            .and_then(|x| x.checked_add(parent.m))
            .expect("compact chain exceeds exact 128-bit input arithmetic");
        let max_first = ((max_x >> program.first_width) + 1)
            .checked_mul(POW3[program.first_width as usize])
            .expect("compact chain first stage exceeds exact arithmetic");
        ((max_first >> program.second_width) + 1)
            .checked_mul(POW3[program.second_width as usize])
            .expect("compact chain second stage exceeds exact arithmetic");
        let descriptor = [
            parent.n0 as u32, (parent.n0 >> 32) as u32,
            parent.m as u32, (parent.m >> 32) as u32,
            (parent.m >> 64) as u32, (parent.m >> 96) as u32,
            p3 as u32, (p3 >> 32) as u32, (p3 >> 64) as u32, (p3 >> 96) as u32,
            CHAIN_INV3[parent.f as usize], program_index,
            program.endpoints, 0, 0, parent.k,
        ];
        let chunk = &mut self.chunks[idx];
        chunk.extend_from_slice(&descriptor);
        if chunk.len() >= 16 * self.sh.group_chunk_records {
            self.sh.push_parents(idx, chunk);
            chunk.clear();
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
        if chunk.len() >= 16 * self.sh.group_chunk_records {
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
            assert!(!self.sh.transport.is_grouped(), "group transport requires program descriptors");
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

/// c3-hyg: one configuration source.  The per-class codes and warp-queue bits
/// a GPU run launches must be the run's own `TailParams` values (which the
/// CPU reference, the CPU-only classes and the report use); measurement-only
/// configurations need the session's explicit opt-in.
pub fn check_gpu_config(tail: &crate::tail::TailParams, tuning: &GpuTuning, num_bv: usize, measurement_ok: bool)
    -> Result<()> {
    ensure!(tuning.xtail == tail.codes.gpu_xtail(),
        "GPU tail codes {:?} differ from the run's tail codes {:?}: set GpuTuning::xtail from TailParams::codes \
         (the single source of the per-class semantics)",
        &tuning.xtail[..num_bv.min(32)], tail.codes.per_class(num_bv.min(32)));
    ensure!(tuning.wq == tail.schedule.bits(),
        "GPU warp-queue bits differ from the run's schedule: set GpuTuning::wq from TailParams::schedule");
    tail.codes.check(num_bv, tail.cutoff_bits)?;
    tail.schedule.check(&tail.codes, num_bv)?;
    let measurement = tail.codes.is_measurement() || measurement_diag(tuning.screen_diag);
    ensure!(!measurement || measurement_ok,
        "measurement-only configuration refused (jd floor code or screen diagnostic {}): not a verification; \
         benchmarks opt in with allow_measurement_only()", tuning.screen_diag);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gpu_run(
    ctx: *mut c_void,
    la: &Lookahead,
    tail: crate::tail::TailParams,
    pool: &Pool,
    rx: Receiver<Batch>,
    nslots: usize,
    transport: GpuTransport,
    tuning: GpuTuning,
    measurement_ok: bool,
) -> Result<GpuStats> {
    check_gpu_config(&tail, &tuning, la.num_bv, measurement_ok)?;
    // c3-a: a run may use fewer streams than the session holds (work-preserving).
    let nslots = if tuning.run_slots > 0 { nslots.min(tuning.run_slots as usize) } else { nslots };
    let mut params = class_params(la, tail.cutoff_bits, tail.tail_oee);
    let tuning_bits = tuning.bits()?;
    for (_, param) in &mut params {
        param.pad = tuning_bits;
        param.tail_budget = tuning.tail_budget;
    }
    for (class, (_, param)) in params.iter_mut().enumerate() {
        let code = tuning.xtail.get(class).copied().unwrap_or(0) as u32;
        if code != 0 {
            ensure!(tuning_bits == 5 && tuning.tail_budget == 0 && tuning.early_every == 0,
                "experimental tails require the plain no-vote early-OEE kernel");
            ensure!(matches!(crate::jump::xv_kind(code), Some(crate::jump::TailKind::Jump { skip_window: false, .. })
                | Some(crate::jump::TailKind::JumpPrefix { .. }) | Some(crate::jump::TailKind::Base)
                | Some(crate::jump::TailKind::JumpDip { .. })),
                "pipeline xtail must not use window-dependent descent skipping");
            param.pad |= code << 24;
            if code >= 64 {
                // jd tails: H0 = 485 F - 306 M of this class (cuda/jdip.cuh);
                // c3-hyg: this context's dip tables (a no-op when prepared).
                upload_jd_ctx(ctx, &[code])?;
                param.tail_budget = la.cutoffs[class] as u32;
            }
        }
        let wq = tuning.wq.get(class).copied().unwrap_or(0);
        if wq != 0 && is_two_phase(wq) {
            ensure!(wq & !TP_FIELDS == 0, "two-phase bits {wq:#x} outside the verification fields");
            ensure!(TP_CODES.contains(&code),
                "the two-phase kernel runs jd FD3 codes {TP_CODES:?} (class {class} has {code})");
            ensure!(la.a <= 13, "the two-phase kernel needs A <= 13");
            param.pad |= wq;
        } else if wq != 0 {
            ensure!(wq & !WQ_FIELDS == 0 && wq & (1 << 22) != 0,
                "warp-queue bits {wq:#x} outside the verification fields");
            ensure!((49..=53).contains(&code) || WQ_JD_CODES.contains(&code),
                "the warp-queue scheduler runs jump codes 49..53 or jd FD3 codes {WQ_JD_CODES:?} (class {class} has {code})");
            param.pad |= wq;
        }
    }
    // c-screen: explicit completion screening for this run (row table, h range,
    // diagnostics). It runs only on the jump screen kernels, never silently on
    // a kernel that would ignore it.
    let specs: Vec<ScreenSpec> = (0..la.num_bv).map(|c| ScreenSpec::of(&tail.screen, la, c, tuning.screen_diag))
        .collect::<Result<_>>()?;
    if specs.iter().any(|s| !s.is_default()) {
        ensure!(tuning.early_every == 0 && tuning.tail_budget == 0,
            "screening cannot combine with early CPU screening or tail deferral");
        ensure!(matches!(transport, GpuTransport::Affine32 | GpuTransport::Parents | GpuTransport::CompactChain),
            "screening requires an exact-iterate GPU transport");
        for class in 0..la.num_bv {
            if specs[class].is_default() {
                continue;   // this class runs the ordinary (unscreened) launch
            }
            let code = tuning.xtail.get(class).copied().unwrap_or(0) as u32;
            let wq = tuning.wq.get(class).copied().unwrap_or(0);
            if is_two_phase(wq) {
                ensure!(TP_CODES.contains(&code) && tuning.screen_diag == 0,
                    "class {class}: the two-phase kernel screens codes {TP_CODES:?} without diagnostics (has {code})");
            } else if wq == 0 {
                ensure!(screen_code_ok(code),
                    "screening requires a jump code 49..53 or a jd code {JDS_CODES:?} (class {class} has {code})");
            } else {
                ensure!(wq_screen_code_ok(code),
                    "class {class}: the warp-queue scheduler screens only codes {WQS_CODES:?} (has {code})");
            }
        }
    }
    let mut deferred = (tuning.tail_budget > 0)
        .then(|| DeferredWorkers::new(la.num_bv, tuning.tail_cpu_workers as usize, tail.cutoff_bits));
    let mut early = if tuning.early_every > 0 {
        Some(EarlyWorkers::new(la.num_bv, tuning.early_cpu_workers as usize,
            tail.cutoff_bits, tuning.early_group_bits, tuning.early_mode)?)
    } else { None };
    let mut st = GpuStats {
        device: device_name(),
        batch: if tuning.run_batch > 0 { tuning.run_batch as usize } else { batch_fill() },
        classes: vec![Default::default(); la.num_bv],
        early_class_endpoints: vec![0; la.num_bv],
        early_class_cases: vec![0; la.num_bv],
        early_class_iterations: vec![0; la.num_bv],
        streams: nslots,
        transport: transport.name(),
        tuning,
        payload_bytes_per_endpoint: (!transport.is_grouped())
            .then_some(transport.record_words() * 4),
        kernel_seconds_kind: "sum of CUDA stream event intervals; concurrent streams may overlap",
        ..Default::default()
    };
    let mut running: Vec<Option<(usize, Pinned, bool)>> = (0..nslots).map(|_| None).collect();
    let mut flagbuf = vec![0u32; 4 << 16];
    let mut collect = |slot: usize,
                       running: &mut Vec<Option<(usize, Pinned, bool)>>,
                       st: &mut GpuStats|
     -> Result<()> {
        let Some((idx, buf, screened)) = running[slot].take() else {
            return Ok(());
        };
        if screened {
            let (mut ms, mut d2h_ms) = (0f32, 0f32);
            let (mut affine, mut masks) = (std::ptr::null(), std::ptr::null());
            let (mut endpoints, mut words32) = (0u32, 0u32);
            let rc = unsafe { cv_gpu_wait_screen(ctx, slot as i32, &mut ms, &mut d2h_ms,
                &mut affine, &mut masks, &mut endpoints, &mut words32) };
            if rc != 0 {
                unsafe { cudaDeviceSynchronize() };
                bail!("early screening or return copy failed");
            }
            ensure!(endpoints as usize == buf.endpoints && words32 == (1u32 << la.a) / 32
                && !affine.is_null() && !masks.is_null(), "invalid early screening export");
            // These pinned pointers belong to this slot. Every submitted job
            // owns copies of its masks/context before the next slot launch.
            let affine = unsafe { std::slice::from_raw_parts(affine, endpoints as usize * 8) };
            let masks = unsafe { std::slice::from_raw_parts(masks, endpoints as usize * words32 as usize) };
            // Queue at most one compact shard per worker. Reconstruction is
            // chunked inside workers, so dispatch can resume GPU work without
            // waiting for a launch's millions of CPU trajectories to finish.
            let shard_endpoints = (endpoints as usize).div_ceil(tuning.early_cpu_workers as usize);
            for start in (0..endpoints as usize).step_by(shard_endpoints.max(1)) {
                let end = (start + shard_endpoints).min(endpoints as usize);
                early.as_mut().unwrap().submit(idx, la.m, POW3[(la.offset + idx as u32) as usize],
                    words32 as usize, &affine[start * 8..end * 8],
                    &masks[start * words32 as usize..end * words32 as usize])?;
            }
            let c = &mut st.classes[idx];
            c.launches += 1;
            c.endpoints += endpoints as u64;
            c.kernel_seconds += ms as f64 / 1e3;
            st.kernel_seconds += ms as f64 / 1e3;
            st.transfer_bytes += (buf.len * 4) as u64;
            st.input_records += buf.records as u64;
            st.early_batches += 1;
            st.early_endpoints += endpoints as u64;
            st.early_class_endpoints[idx] += endpoints as u64;
            st.early_transfer_bytes += endpoints as u64 * (8 + words32 as u64) * 4;
            st.early_copy_seconds += d2h_ms as f64 / 1e3;
            pool.give(buf);
            return Ok(());
        }
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
        st.regular_batches += 1;
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
        if let Some(workers) = deferred.as_mut() {
            let mut attempts = 0u32;
            ensure!(unsafe { cv_gpu_wait_deferred(ctx, slot as i32, &mut attempts,
                std::ptr::null_mut(), 0) } == 0, "continuation count read failed");
            let kept = attempts.min(tuning.tail_queue_capacity) as usize;
            st.deferred_attempts += attempts as u64;
            st.deferred_rejected += attempts as u64 - kept as u64;
            if kept > 0 {
                let mut records = Vec::<crate::tail::TailContinuation>::with_capacity(kept);
                let mut repeated_attempts = 0;
                ensure!(unsafe { cv_gpu_wait_deferred(ctx, slot as i32, &mut repeated_attempts,
                    records.as_mut_ptr().cast(), kept) } == 0, "continuation records read failed");
                ensure!(repeated_attempts == attempts, "continuation count changed during read");
                // CUDA wrote exactly kept initialized 32-byte records.
                unsafe { records.set_len(kept) };
                st.deferred_cases += kept as u64;
                st.deferred_transfer_bytes += kept as u64 * 32;
                workers.submit(idx, records)?;
            }
        }
        pool.give(buf);
        Ok(())
    };
    let result = (|| -> Result<()> {
        let mut next = 0usize;
        for (launch_index, batch) in rx.into_iter().enumerate() {
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
            let screened = tuning.early_every > 0 && launch_index as u64 % tuning.early_every as u64 == 0;
            // Record ownership before launch: even a failed launch can have
            // enqueued an asynchronous host-to-device copy.
            running[slot] = Some((idx, batch.buf, screened));
            let rc = unsafe {
                let (offset, p) = &params[idx];
                if screened {
                    cv_gpu_launch_screen_ex(ctx, slot as i32, ptr, n, records,
                        transport as u32, *offset, la.magic64[idx], tp.as_ptr(), p)
                } else if !specs[idx].is_default() {
                    cv_gpu_launch_scr(ctx, slot as i32, ptr, n, records, transport as u32, *offset,
                        la.magic64[idx], tp.as_ptr(), p, &specs[idx])
                } else if transport == GpuTransport::Replay16 {
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
    drop(collect);
    if let Some(workers) = deferred {
        workers.finish(&mut st)?;
    }
    if let Some(workers) = early {
        workers.finish(&mut st)?;
    }
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
    tuning: GpuTuning,
    measurement_ok: bool,
}

enum Command {
    Run(RunCommand),
    UploadTerminal {
        words: Vec<TerminalWord>,
        reply: SyncSender<Result<()>>,
    },
    PrepareTailJump { width: u32, reply: SyncSender<Result<()>> },
    PrepareTailOffload { capacity: u32, reply: SyncSender<Result<()>> },
    PrepareEarlyScreen { reply: SyncSender<Result<()>> },
    UploadChain { tables: ChainTables, reply: SyncSender<Result<()>> },
    /// c3-hyg: upload jd dip tables into the worker's context.
    PrepareJd { codes: Vec<u32>, reply: SyncSender<Result<()>> },
    /// c3-hyg: let the worker's context run measurement-only kernels.
    AllowMeasurement { reply: SyncSender<Result<()>> },
    /// c3-hyg: is the worker's context still current (not reset)?
    CheckContext { reply: SyncSender<Result<()>> },
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
    chain_tables: Option<ChainTables>,
    tail_capacity: u32,
    early_prepared: bool,
    /// c3-hyg: set by `allow_measurement_only` (benchmarks only).
    measurement_ok: bool,
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
                upload_rows(ctx.0, &worker_la)?;
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
                            command.tuning,
                            command.measurement_ok,
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
                    Command::UploadChain { tables, reply } => {
                        let result = if unsafe { cv_gpu_upload_chain(ctx.0,
                            tables.programs.as_ptr(), tables.programs.len(),
                            tables.leaves.as_ptr(), tables.leaves.len(),
                            tables.words.as_ptr(), tables.words.len()) } == 0 {
                            Ok(())
                        } else { Err(anyhow::anyhow!("compact chain upload failed")) };
                        let failed = result.is_err();
                        let _ = reply.send(result);
                        failed
                    }
                    Command::PrepareTailJump { width, reply } => {
                        let result = if unsafe { cv_gpu_set_tail_jump(ctx.0, width) } == 0 {
                            Ok(())
                        } else {
                            Err(anyhow::anyhow!("tail jump table preparation failed"))
                        };
                        let failed = result.is_err();
                        let _ = reply.send(result);
                        failed
                    }
                    Command::PrepareTailOffload { capacity, reply } => {
                        let result = if unsafe { cv_gpu_set_tail_capacity(ctx.0, capacity) } == 0 {
                            Ok(())
                        } else { Err(anyhow::anyhow!("tail continuation buffer preparation failed")) };
                        let failed = result.is_err();
                        let _ = reply.send(result);
                        failed
                    }
                    Command::PrepareEarlyScreen { reply } => {
                        let result = if unsafe { cv_gpu_prepare_screen(ctx.0, (1u32 << worker_la.a) / 32) } == 0 {
                            Ok(())
                        } else { Err(anyhow::anyhow!("early screen buffer preparation failed")) };
                        let failed = result.is_err();
                        let _ = reply.send(result);
                        failed
                    }
                    Command::PrepareJd { codes, reply } => {
                        let result = upload_jd_ctx(ctx.0, &codes);
                        let failed = result.is_err();
                        let _ = reply.send(result);
                        failed
                    }
                    Command::AllowMeasurement { reply } => {
                        let result = if unsafe { cv_gpu_allow_measurement(ctx.0, 1) } == 0 {
                            Ok(())
                        } else { Err(anyhow::anyhow!("measurement opt-in failed")) };
                        let failed = result.is_err();
                        let _ = reply.send(result);
                        failed
                    }
                    Command::CheckContext { reply } => {
                        let result = if unsafe { cv_gpu_context_ok(ctx.0) } == 1 {
                            Ok(())
                        } else {
                            Err(anyhow::anyhow!("GPU context is not current (the CUDA context was reset): refused"))
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
            chain_tables: None,
            tail_capacity: 0,
            early_prepared: false,
            measurement_ok: false,
            setup_seconds: t0.elapsed().as_secs_f64(),
            build_seconds,
        })
    }

    /// c3-hyg: upload the dip tables of these jd codes (>= 64; others are
    /// ignored) into this session's context, outside timed runs.  `gpu_run`
    /// uploads missing tables itself, so this only moves the cost.
    pub fn prepare_jd_tables(&mut self, codes: &[u32]) -> Result<()> {
        let codes: Vec<u32> = codes.iter().copied().filter(|&c| c >= 64).collect();
        if codes.is_empty() {
            return Ok(());
        }
        let (reply, result) = sync_channel(1);
        self.commands.as_ref().unwrap().send(Command::PrepareJd { codes, reply })
            .map_err(|_| anyhow::anyhow!("GPU worker has stopped"))?;
        result.recv().map_err(|_| anyhow::anyhow!("GPU worker stopped during jd table upload"))??;
        Ok(())
    }

    /// c3-hyg: explicit opt-in for measurement-only kernels (jd floor codes
    /// 125..127, screen diagnostics `_dload/_dscan/_dsetup`).  Benchmark
    /// binaries only; every report of such a run is marked
    /// `measurement_only` and is not a verification.
    pub fn allow_measurement_only(&mut self) -> Result<()> {
        if self.measurement_ok {
            return Ok(());
        }
        let (reply, result) = sync_channel(1);
        self.commands.as_ref().unwrap().send(Command::AllowMeasurement { reply })
            .map_err(|_| anyhow::anyhow!("GPU worker has stopped"))?;
        result.recv().map_err(|_| anyhow::anyhow!("GPU worker stopped during measurement opt-in"))??;
        self.measurement_ok = true;
        Ok(())
    }

    pub fn run<E: Expander>(&mut self, cfg: &VerifyConfig, expander: &E) -> Result<VerifyReport> {
        self.run_transport(cfg, expander, GpuTransport::Replay16, &[])
    }

    /// Prepare an experimental tail table outside measured pipeline runs.
    pub fn set_tail_jump(&mut self, width: u32) -> Result<()> {
        ensure!(self.experimental, "tail jumps require an experimental GPU session");
        ensure!(matches!(width, 0 | 8 | 10 | 12 | 14), "unsupported tail jump width");
        let (reply, result) = sync_channel(1);
        self.commands.as_ref().unwrap().send(Command::PrepareTailJump { width, reply })
            .map_err(|_| anyhow::anyhow!("GPU worker has stopped"))?;
        result.recv().map_err(|_| anyhow::anyhow!("GPU worker stopped during tail table preparation"))??;
        Ok(())
    }

    /// Allocate bounded continuation buffers outside timed pipeline work.
    pub fn prepare_tail_offload(&mut self, capacity: u32) -> Result<()> {
        ensure!(self.experimental, "tail offload requires an experimental session");
        ensure!((1..=1_048_576).contains(&capacity), "invalid tail continuation capacity");
        if self.tail_capacity == capacity { return Ok(()); }
        let (reply, result) = sync_channel(1);
        self.commands.as_ref().unwrap().send(Command::PrepareTailOffload { capacity, reply })
            .map_err(|_| anyhow::anyhow!("GPU worker has stopped"))?;
        result.recv().map_err(|_| anyhow::anyhow!("GPU worker stopped during offload preparation"))??;
        self.tail_capacity = capacity;
        Ok(())
    }

    /// Allocate screen masks and affine return buffers outside timed runs.
    pub fn prepare_early_screen(&mut self) -> Result<()> {
        ensure!(self.experimental, "early screening requires an experimental session");
        if self.early_prepared { return Ok(()); }
        let (reply, result) = sync_channel(1);
        self.commands.as_ref().unwrap().send(Command::PrepareEarlyScreen { reply })
            .map_err(|_| anyhow::anyhow!("GPU worker has stopped"))?;
        result.recv().map_err(|_| anyhow::anyhow!("GPU worker stopped during early screening preparation"))??;
        self.early_prepared = true;
        Ok(())
    }

    /// Install the resident short words and chain references outside timing.
    pub fn set_chain_table(&mut self, tables: &ChainTables) -> Result<()> {
        ensure!(self.experimental, "compact chains require an experimental session");
        ensure!(!tables.programs.is_empty() && !tables.leaves.is_empty() && !tables.words.is_empty(),
                "empty compact chain tables");
        if self.chain_tables.as_ref() == Some(tables) { return Ok(()); }
        let (reply, result) = sync_channel(1);
        self.commands.as_ref().unwrap().send(Command::UploadChain { tables: tables.clone(), reply })
            .map_err(|_| anyhow::anyhow!("GPU worker has stopped"))?;
        result.recv().map_err(|_| anyhow::anyhow!("GPU worker stopped during chain upload"))??;
        self.chain_tables = Some(tables.clone());
        Ok(())
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
        self.run_transport_tuned(
            cfg,
            expander,
            transport,
            terminal_words,
            GpuTuning::default(),
        )
    }

    /// Dispatch compact groups only with their matching resident program graph.
    pub fn run_chain_tuned<E: Expander>(&mut self, cfg: &VerifyConfig, expander: &E,
                                      tables: &ChainTables, tuning: GpuTuning) -> Result<VerifyReport> {
        ensure!(self.chain_tables.as_ref() == Some(tables),
                "install the matching compact chain table before running");
        self.run_transport_tuned_inner(cfg, expander, GpuTransport::CompactChain, &[], tuning)
    }

    /// Run the same endpoint stream with explicit kernel tuning. Arrays and
    /// the persistent worker remain resident between differently tuned runs.
    pub fn run_transport_tuned<E: Expander>(
        &mut self,
        cfg: &VerifyConfig,
        expander: &E,
        transport: GpuTransport,
        terminal_words: &[TerminalWord],
        tuning: GpuTuning,
    ) -> Result<VerifyReport> {
        ensure!(transport != GpuTransport::CompactChain,
                "compact chain dispatch requires run_chain_tuned and matching tables");
        self.run_transport_tuned_inner(cfg, expander, transport, terminal_words, tuning)
    }

    fn run_transport_tuned_inner<E: Expander>(
        &mut self, cfg: &VerifyConfig, expander: &E, transport: GpuTransport,
        terminal_words: &[TerminalWord], tuning: GpuTuning,
    ) -> Result<VerifyReport> {
        tuning.bits()?;
        // c3-hyg: the configuration's own limits and the single code source.
        cfg.validate()?;
        crate::runspec::reject_legacy_env()?;
        check_gpu_config(&cfg.tail, &tuning, cfg.num_bv, self.measurement_ok)?;
        validate_gpu_config(cfg)?;
        ensure!(!cfg.tail.screen.rows || self.la.rows.is_some(),
            "this run screens with the row table, but the session was created without one \
             (create it from a configuration whose tail.screen.rows is set)");
        if tuning.early_every > 0 {
            ensure!(self.experimental && self.early_prepared,
                "prepare early screen buffers before timing offload");
            ensure!(matches!(transport, GpuTransport::Affine32 | GpuTransport::Parents | GpuTransport::CompactChain),
                "early CPU screening requires exact-iterate GPU transport");
        }
        if tuning.tail_budget > 0 {
            ensure!(self.experimental && self.tail_capacity == tuning.tail_queue_capacity,
                "prepare matching continuation capacity before timing offload");
            ensure!(matches!(transport, GpuTransport::Affine32 | GpuTransport::Parents | GpuTransport::CompactChain),
                "CPU tail continuation requires exact-iterate GPU transport");
        }
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
        ensure!(transport != GpuTransport::CompactChain || self.chain_tables.is_some(),
                "install a compact chain table before dispatch");
        ensure!(
            self.shape == SessionShape::of(cfg),
            "GPU session configuration differs: create a new session for different N/A/windows"
        );
        ensure!(
            !self.pool.closed.load(Ordering::Acquire),
            "GPU session is closed after a previous error"
        );
        // c3-hyg: a reset context has also lost its pinned buffers, which the
        // producers below write; refuse before they start.
        {
            let (reply, result) = sync_channel(1);
            self.commands.as_ref().unwrap().send(Command::CheckContext { reply })
                .map_err(|_| anyhow::anyhow!("GPU worker has stopped"))?;
            result.recv().map_err(|_| anyhow::anyhow!("GPU worker stopped during the context check"))??;
        }
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
                tuning,
                measurement_ok: self.measurement_ok,
            }))
            .map_err(|_| anyhow::anyhow!("GPU worker has stopped"))?;
        let shared = Shared {
            acc: (0..la.num_bv).map(|_| Mutex::new(None)).collect(),
            pool,
            tx,
            la,
            cfg,
            transport,
            group_chunk_records: tuning.group_chunk_records as usize,
            totals: Mutex::new((TreeStats::new(m), TailStats::new(la.num_bv))),
            send_ns: AtomicU64::new(0),
            fill: if tuning.run_batch > 0 { tuning.run_batch as usize } else { batch_fill() },
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
        if measurement_diag(tuning.screen_diag) {
            rep.mark_measurement_only();
        }
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

/// `verify_gpu` with explicit kernel tuning (e.g. jump tails in the warp-queue
/// scheduler: `cv verify --gpu --tail-codes ... --gpu-wq ...`).  c3-hyg: the
/// tuning's per-class codes and warp-queue bits must equal `cfg.tail.codes`
/// and `cfg.tail.schedule`, which a CPU run of the same configuration uses.
pub fn verify_gpu_tuned(cfg: &VerifyConfig, tuning: GpuTuning) -> Result<VerifyReport> {
    let t0 = Instant::now();
    // c-final: screened runs (row table / h range) need exact iterates (mode 2):
    // the host computes T^M(n0) (Affine32 transport, experimental buffers).
    let screened = !cfg.tail.screen.is_default();
    let mut session = if screened { GpuSession::new_experimental(cfg)? } else { GpuSession::new(cfg)? };
    let transport = if screened { GpuTransport::Affine32 } else { GpuTransport::Replay16 };
    let mut rep = session.run_transport_tuned(cfg, &DfsExpander, transport, &[], tuning)?;
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
    /// Warp-queue instrumentation counters summed over the most recent run_x
    /// (trips, active lane-steps, pop rounds, materializations, screens, U).
    pub last_dbg: [u64; 6],
    /// Sorted flagged starts of the most recent run_x (exact multiset).
    pub last_flag_list: Vec<u128>,
    flagbuf: Vec<u32>,
    buf2: Option<Pinned>,
    /// c3-hyg: endpoints per launch (BATCH by default; smaller values let
    /// tests exercise the per-chunk accumulation of totals and digests).
    launch_chunk: usize,
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
        upload_rows(ctx, la)?;
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
            last_dbg: [0; 6],
            last_flag_list: Vec::new(),
            buf2: None,
            launch_chunk: BATCH,
        })
    }

    /// c3-hyg: explicit opt-in for measurement-only kernels in this bench's
    /// context (jd floor codes, screen diagnostics, the capped test tail).
    pub fn allow_measurement_only(&mut self) -> Result<()> {
        ensure!(unsafe { cv_gpu_allow_measurement(self.ctx, 1) } == 0, "measurement opt-in failed");
        Ok(())
    }

    /// c3-hyg: endpoints per launch for run_x / run_xs_wq (1..=BATCH).
    pub fn set_launch_chunk(&mut self, endpoints: usize) -> Result<()> {
        ensure!((1..=BATCH).contains(&endpoints), "launch chunk must be in 1..={BATCH}");
        self.launch_chunk = endpoints;
        Ok(())
    }

    /// c3-hyg: does this bench's context still match the calling thread's
    /// current device and CUDA context?
    pub fn context_ok(&self) -> bool {
        unsafe { cv_gpu_context_ok(self.ctx) == 1 }
    }

    /// c3-hyg: refuse to run on a stale context.  A device reset also frees
    /// the context's pinned host buffers, so this must come before any
    /// endpoint is packed (writing them would fault).
    fn check_context(&self) -> Result<()> {
        ensure!(self.context_ok(),
            "GPU context is not current (other device, or the CUDA context was reset): refused");
        Ok(())
    }

    /// c3-hyg: upload jd dip tables into this bench's context (run_x and
    /// run_xs_wq also do it on demand).
    pub fn prepare_jd_tables(&mut self, codes: &[u32]) -> Result<()> {
        upload_jd_ctx(self.ctx, codes)
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
    /// `CV_PAD` (tuning bits, e.g. 5) and `CV_XV` (experimental tail variant)
    /// override the original kernel for experiments; `CV_TAIL` must name the
    /// matching CPU reference (see `crate::jump`).
    pub fn run(&mut self, idx: usize, nodes: &[Node]) -> Result<GpuClassStats> {
        let pad: u32 = std::env::var("CV_PAD").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let xv: u32 = std::env::var("CV_XV").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        if pad == 0 && xv == 0 {
            return self.run_tuned(idx, nodes, GpuTuning::default());
        }
        // c3-hyg: no skipped descent tests (CV_TAIL, which used to choose
        // them, is refused); `cv measure` compares the GPU's cases and
        // iterations with the original-loop CPU reference and fails loudly
        // when CV_XV selects different semantics.
        let mode2 = std::env::var("CV_MODE2").map_or(false, |v| v == "1");
        self.run_x(idx, nodes, pad | (xv << 24), 0, mode2)
    }

    /// Experimental timing: raw pad bits (tuning | variant << 24), the jump
    /// variant's skipped descent checks, and optionally exact-iterate (mode 2)
    /// records as produced by production parent expansion.
    pub fn run_x(&mut self, idx: usize, nodes: &[Node], pad: u32, skip: u32, mode2: bool) -> Result<GpuClassStats> {
        self.check_context()?;
        self.last_hash = 0;
        self.last_flags = 0;
        self.last_dbg = [0; 6];
        self.last_flag_list.clear();
        if mode2 && self.buf2.is_none() {
            ensure!(unsafe { cv_gpu_enable_experimental(self.ctx, BATCH) } == 0, "cv_gpu_enable_experimental failed");
            self.buf2 = Some(Pinned::try_alloc_words(8)?);
        }
        let la = &self.la;
        let (offset, mut params) = if self.per_class {
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
        params.pad = pad;
        params.tail_budget = if pad >> 24 != 0 { skip } else { 0 };
        if pad >> 24 >= 64 {
            upload_jd_ctx(self.ctx, &[pad >> 24])?;   // c3-hyg: this bench's context
            params.tail_budget = la.cutoffs[idx] as u32;
        }
        let wq_bits = if pad & (1 << 22) != 0 { pad & (WQ_FIELDS | (1 << 21)) } else { 0 };
        let mut st = GpuClassStats::default();
        let f = la.offset + idx as u32;
        let p3 = POW3[f as usize];
        let tp = [p3 as u32, (p3 >> 32) as u32, (p3 >> 64) as u32, (p3 >> 96) as u32];
        let m = la.m;
        for part in nodes.chunks(self.launch_chunk) {
            let buf = if mode2 { self.buf2.as_mut().unwrap() } else { &mut self.buf };
            buf.len = 0;
            for nd in part {
                debug_assert_eq!(nd.f, f);
                buf.append(&[
                    nd.n0 as u32,
                    (nd.n0 >> 32) as u32,
                    (nd.n0 >> 64) as u32,
                    ((nd.n0 >> 96) as u32) | ((nd.oee as u32) << 30) | ((la.family(nd.ne) as u32) << 29),
                ]);
                if mode2 {
                    debug_assert_eq!(crate::arith::t_iter(nd.n0, m).0, nd.m);
                    buf.append(&[nd.m as u32, (nd.m >> 32) as u32, (nd.m >> 64) as u32, (nd.m >> 96) as u32]);
                }
            }
            let ptr = buf.ptr;
            let (mut ms, mut cases, mut iters, mut hash, mut nf) = (0f32, 0u64, 0u64, 0u32, 0u32);
            unsafe {
                if cv_gpu_launch_ex(self.ctx, 0, ptr, part.len() as u32, part.len() as u32, if mode2 { 2 } else { 0 },
                                    offset, la.magic64[idx], tp.as_ptr(), &params) != 0
                    || cv_gpu_wait(self.ctx, 0, &mut ms, &mut cases, &mut iters, &mut hash, &mut nf,
                                   self.flagbuf.as_mut_ptr(), 1 << 16) != 0
                {
                    bail!("kernel failed (pad {pad:#x}, mode2 {mode2})");
                }
                let mut dbg = [0u64; 6];
                if cv_gpu_last_dbg(self.ctx, 0, dbg.as_mut_ptr()) == 0 {
                    accumulate_dbg(&mut self.last_dbg, &dbg, wq_bits);   // c3-hyg: XOR the digest word
                }
            }
            ensure!(nf <= 1 << 16, "flag buffer overflow ({nf} flags in one launch)");
            for i in 0..nf as usize {
                let w = &self.flagbuf[4 * i..4 * i + 4];
                self.last_flag_list.push(w[0] as u128 | (w[1] as u128) << 32 | (w[2] as u128) << 64 | (w[3] as u128) << 96);
            }
            st.launches += 1;
            self.last_hash ^= hash;
            self.last_flags += nf as u64;
            st.endpoints += part.len() as u64;
            st.cases += cases;
            st.iters += iters;
            st.kernel_seconds += ms as f64 / 1e3;
        }
        self.last_flag_list.sort_unstable();
        Ok(st)
    }

    /// c-screen: exact-iterate (MODE 2) timing of a jump code (49..53) with an
    /// explicit screen and optional diagnostic flavour.  The CPU reference
    /// must use the same `screen` with this bench's lookahead (which carries
    /// the row table it uploaded), so GPU and CPU always screen alike.
    pub fn run_xs(&mut self, idx: usize, nodes: &[Node], code: u32, screen: &crate::tail::Screen, diag: u32)
        -> Result<GpuClassStats> {
        self.run_xs_wq(idx, nodes, code, 0, screen, diag)
    }

    /// c-final: as `run_xs`, for jump codes 49..53 and the screened jd codes
    /// (`JDS_CODES`, H0 = 485 F - 306 M in tail_budget), optionally through the
    /// warp-queue scheduler (`wq` = scheduler pad bits, 0 = per-thread kernel).
    /// A default screen uses the ordinary launch (cv_kernel_x / cv_kernel_jd /
    /// the warp queue without screening).
    pub fn run_xs_wq(&mut self, idx: usize, nodes: &[Node], code: u32, wq: u32, screen: &crate::tail::Screen, diag: u32)
        -> Result<GpuClassStats> {
        self.check_context()?;
        let spec = ScreenSpec::of(screen, &self.la, idx, diag)?;
        if is_two_phase(wq) {
            ensure!(TP_CODES.contains(&code) && diag == 0, "two-phase kernels take jd codes {TP_CODES:?} without diagnostics");
            ensure!(wq & !TP_FIELDS == 0, "two-phase bits {wq:#x}");
        } else if wq == 0 {
            ensure!(spec.is_default() || screen_code_ok(code), "screen kernels take jump codes 49..53 or jd codes {JDS_CODES:?}");
        } else {
            ensure!(spec.is_default() || wq_screen_code_ok(code), "warp-queue screens take codes {WQS_CODES:?}");
        }
        ensure!(wq & !(WQ_FIELDS | (1 << 21)) == 0, "warp-queue bits {wq:#x}");
        self.last_hash = 0;
        self.last_flags = 0;
        self.last_dbg = [0; 6];
        self.last_flag_list.clear();
        if self.buf2.is_none() {
            ensure!(unsafe { cv_gpu_enable_experimental(self.ctx, BATCH) } == 0, "cv_gpu_enable_experimental failed");
            self.buf2 = Some(Pinned::try_alloc_words(8)?);
        }
        let la = &self.la;
        let (offset, mut params) = if self.per_class {
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
        params.pad = 5 | (code << 24) | wq;
        params.tail_budget = 0;
        if code >= 64 {
            upload_jd_ctx(self.ctx, &[code])?;   // c3-hyg: this bench's context
            params.tail_budget = la.cutoffs[idx] as u32;
        }
        let mut st = GpuClassStats::default();
        let f = la.offset + idx as u32;
        let p3 = POW3[f as usize];
        let tp = [p3 as u32, (p3 >> 32) as u32, (p3 >> 64) as u32, (p3 >> 96) as u32];
        for part in nodes.chunks(self.launch_chunk) {
            let buf = self.buf2.as_mut().unwrap();
            buf.len = 0;
            for nd in part {
                debug_assert_eq!(nd.f, f);
                buf.append(&[
                    nd.n0 as u32,
                    (nd.n0 >> 32) as u32,
                    (nd.n0 >> 64) as u32,
                    ((nd.n0 >> 96) as u32) | ((nd.oee as u32) << 30) | ((la.family(nd.ne) as u32) << 29),
                    nd.m as u32, (nd.m >> 32) as u32, (nd.m >> 64) as u32, (nd.m >> 96) as u32,
                ]);
            }
            let ptr = buf.ptr;
            let (mut ms, mut cases, mut iters, mut hash, mut nf) = (0f32, 0u64, 0u64, 0u32, 0u32);
            unsafe {
                if cv_gpu_launch_scr(self.ctx, 0, ptr, part.len() as u32, part.len() as u32, 2,
                                     offset, la.magic64[idx], tp.as_ptr(), &params, &spec) != 0
                    || cv_gpu_wait(self.ctx, 0, &mut ms, &mut cases, &mut iters, &mut hash, &mut nf,
                                   self.flagbuf.as_mut_ptr(), 1 << 16) != 0
                {
                    bail!("screen kernel failed (code {code}, wq {wq:#x}, spec {spec:?})");
                }
                let mut dbg = [0u64; 6];
                if cv_gpu_last_dbg(self.ctx, 0, dbg.as_mut_ptr()) == 0 {
                    accumulate_dbg(&mut self.last_dbg, &dbg, wq);   // c3-hyg: XOR the digest word
                }
            }
            ensure!(nf <= 1 << 16, "flag buffer overflow ({nf} flags in one launch)");
            for i in 0..nf as usize {
                let w = &self.flagbuf[4 * i..4 * i + 4];
                self.last_flag_list.push(w[0] as u128 | (w[1] as u128) << 32 | (w[2] as u128) << 64 | (w[3] as u128) << 96);
            }
            st.launches += 1;
            self.last_hash ^= hash;
            self.last_flags += nf as u64;
            st.endpoints += part.len() as u64;
            st.cases += cases;
            st.iters += iters;
            st.kernel_seconds += ms as f64 / 1e3;
        }
        self.last_flag_list.sort_unstable();
        Ok(st)
    }

    /// c-screen: completion words of `nodes` from the cooperative per-lane-word
    /// routine (cuda/screen.cuh `scr_word`, one warp per endpoint), as a
    /// warp-queue scheduler would form them; `words32` u32 words per endpoint.
    pub fn screen_words(&mut self, idx: usize, nodes: &[Node], screen: &crate::tail::Screen) -> Result<Vec<u32>> {
        self.check_context()?;
        let spec = ScreenSpec::of(screen, &self.la, idx, 0)?;
        if self.buf2.is_none() {
            ensure!(unsafe { cv_gpu_enable_experimental(self.ctx, BATCH) } == 0, "cv_gpu_enable_experimental failed");
            self.buf2 = Some(Pinned::try_alloc_words(8)?);
        }
        let la = &self.la;
        let words32 = (1usize << la.a) / 32;
        // The words reuse the slot's 32-byte-per-endpoint affine buffer.
        let chunk = (BATCH * 8 / words32).max(1);
        let (offset, params) = if self.per_class {
            let words = la.array_words(idx);
            if self.resident != Some(idx) {
                if unsafe { cv_gpu_alloc_bv(self.ctx, la.families * words) } != 0 {
                    bail!("cannot allocate the class arrays");
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
        let f = la.offset + idx as u32;
        let p3 = POW3[f as usize];
        let tp = [p3 as u32, (p3 >> 32) as u32, (p3 >> 64) as u32, (p3 >> 96) as u32];
        let mut out = vec![0u32; nodes.len() * words32];
        for (ci, part) in nodes.chunks(chunk).enumerate() {
            let buf = self.buf2.as_mut().unwrap();
            buf.len = 0;
            for nd in part {
                buf.append(&[
                    nd.n0 as u32, (nd.n0 >> 32) as u32, (nd.n0 >> 64) as u32,
                    ((nd.n0 >> 96) as u32) | ((nd.oee as u32) << 30) | ((la.family(nd.ne) as u32) << 29),
                    nd.m as u32, (nd.m >> 32) as u32, (nd.m >> 64) as u32, (nd.m >> 96) as u32,
                ]);
            }
            let dst = &mut out[ci * chunk * words32..(ci * chunk + part.len()) * words32];
            ensure!(unsafe { cv_gpu_screen_words(self.ctx, 0, buf.ptr, part.len() as u32, offset, la.magic64[idx],
                tp.as_ptr(), &params, &spec, dst.as_mut_ptr()) } == 0, "screen words failed");
        }
        Ok(out)
    }

    /// Replace one class's window (forces that class's arrays to be rebuilt
    /// on the next per-class launch).  For window sweeps.
    pub fn set_class_window(&mut self, idx: usize, b: u32) -> Result<()> {
        ensure!(self.per_class, "window changes need per-class device arrays");
        ensure!(b >= self.la.a && b <= crate::lookahead::MAX_B, "window {b} out of range");
        self.la.bs[idx] = b;
        if self.resident == Some(idx) {
            self.resident = None;
        }
        Ok(())
    }

    pub fn lookahead(&self) -> &Lookahead {
        &self.la
    }

    /// Serialized timing with the same opt-in choices as the pipeline.
    pub fn run_tuned(
        &mut self,
        idx: usize,
        nodes: &[Node],
        tuning: GpuTuning,
    ) -> Result<GpuClassStats> {
        self.check_context()?;
        let tuning_bits = tuning.bits()?;
        ensure!(tuning.tail_budget == 0, "CPU continuation requires the full GPU pipeline");
        ensure!(tuning.early_every == 0, "early CPU screening requires the full GPU pipeline");
        self.last_hash = 0;
        self.last_flags = 0;
        let la = &self.la;
        // Per-class mode: only this class's arrays are resident, at offset 0.
        let (offset, mut params) = if self.per_class {
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
        params.pad = tuning_bits;
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
            // c3-hyg: the flag buffer holds 65536 starts; more is an error, never a silent drop.
            ensure!(nf <= 1 << 16, "flag buffer overflow ({nf} flags in one launch)");
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

#[cfg(test)]
mod tuning_tests {
    use super::GpuTuning;

    #[test]
    fn grouped_chunk_tuning_is_explicit_and_does_not_change_kernel_bits() {
        let ordinary = GpuTuning::default();
        let small = GpuTuning { group_chunk_records: 1024, ..ordinary };
        assert_eq!(ordinary.group_chunk_records, 4096);
        assert_eq!(small.bits().unwrap(), ordinary.bits().unwrap());
        assert!(serde_json::to_value(ordinary).unwrap().get("group_chunk_records").is_none());
        assert_eq!(serde_json::to_value(small).unwrap()["group_chunk_records"], 1024);
        for records in [0, 4097, u32::MAX] {
            assert!(GpuTuning { group_chunk_records: records, ..ordinary }.bits().is_err());
        }
    }

    #[test]
    fn tuning_encoding_keeps_defaults_and_rejects_unsupported_blocks() {
        assert_eq!(GpuTuning::default().bits().unwrap(), 0);
        let tuned = GpuTuning {
            block_threads: 128,
            overflow_vote: false,
            readonly_powers: true,
            oee_mask: true,
            tail_jump_bits: 0,
            group_chunk_records: super::CHUNK as u32,
            ..GpuTuning::default()
        };
        assert_eq!(tuned.bits().unwrap(), 0x107);
        assert_eq!(
            GpuTuning { block_threads: 512, ..tuned }.bits().unwrap(),
            0x207
        );
        assert!(GpuTuning { block_threads: 192, ..tuned }.bits().is_err());
    }

    #[test]
    fn tail_jump_encoding_matches_cuda_width_and_block_fields() {
        for width in [8, 10, 12, 14] {
            for (threads, block_bits) in [(256, 0), (128, 0x100), (512, 0x200)] {
                let tuned = GpuTuning {
                    block_threads: threads,
                    overflow_vote: false,
                    readonly_powers: false,
                    oee_mask: true,
                    tail_jump_bits: width,
                    group_chunk_records: super::CHUNK as u32,
                    ..GpuTuning::default()
                };
                let encoded = tuned.bits().unwrap();
                assert_eq!(encoded, (width << 16) | block_bits | 5);
                assert_eq!((encoded >> 16) & 31, width);
                assert_eq!(encoded & !0x1f0307, 0);
            }
        }
    }

    #[test]
    fn tail_jumps_reject_unsupported_widths_and_kernel_combinations() {
        let supported = GpuTuning {
            overflow_vote: false,
            oee_mask: true,
            tail_jump_bits: 12,
            ..GpuTuning::default()
        };
        for width in [1, 6, 7, 9, 11, 13, 15, 16, 31, 32, u32::MAX] {
            assert!(GpuTuning { tail_jump_bits: width, ..supported }.bits().is_err());
        }
        for overflow_vote in [false, true] {
            for readonly_powers in [false, true] {
                for oee_mask in [false, true] {
                    let tuned = GpuTuning {
                        overflow_vote,
                        readonly_powers,
                        oee_mask,
                        ..supported
                    };
                    assert_eq!(tuned.bits().is_ok(), !overflow_vote && !readonly_powers && oee_mask);
                    // The restriction must not change preexisting, non-jump tunings.
                    assert!(GpuTuning { tail_jump_bits: 0, ..tuned }.bits().is_ok());
                }
            }
        }
        assert!(GpuTuning { block_threads: 192, ..supported }.bits().is_err());
    }

    #[test]
    fn tail_jump_serialization_preserves_the_previous_nonjump_shape() {
        assert_eq!(serde_json::to_value(GpuTuning::default()).unwrap(), serde_json::json!({
            "block_threads": 256,
            "overflow_vote": true,
            "readonly_powers": false,
            "oee_mask": false,
        }));
        let ordinary = GpuTuning {
            overflow_vote: false,
            oee_mask: true,
            ..GpuTuning::default()
        };
        let expected = serde_json::json!({
            "block_threads": 256,
            "overflow_vote": false,
            "readonly_powers": false,
            "oee_mask": true,
        });
        assert_eq!(serde_json::to_value(ordinary).unwrap(), expected);
        for width in [8, 10, 12, 14] {
            let mut jump_expected = expected.clone();
            jump_expected["tail_jump_bits"] = width.into();
            assert_eq!(serde_json::to_value(GpuTuning {
                tail_jump_bits: width,
                ..ordinary
            }).unwrap(), jump_expected);
        }
    }

    #[test]
    fn periodic_oee_mask_matches_candidate_filter_for_every_low_residue() {
        for number0 in 0u32..4 {
            for tp in [1u32, 3] {
                for oee in 0..3 {
                    let mask = match oee {
                        2 => {
                            if number0 & 1 != 0 { 0xAAAAAAAAu32 } else { 0x55555555u32 }
                        }
                        1 => {
                            !(0x11111111u32 << (1u32.wrapping_sub(number0).wrapping_mul(tp) & 3))
                        }
                        _ => u32::MAX,
                    };
                    for aval in 0..1024 {
                        let num = number0 + aval * tp;
                        let allowed = !(oee == 2 && num & 1 != 0 || oee == 1 && num & 3 == 1);
                        assert_eq!(mask & (1 << (aval % 32)) != 0, allowed);
                    }
                }
            }
        }
    }

    #[test]
    fn tail_offload_preserves_kernel_bits_and_rejects_incompatible_modes() {
        assert_eq!(std::mem::size_of::<super::KParams>(), 56);
        let tuned = GpuTuning {
            overflow_vote: false, oee_mask: true, tail_budget: 128,
            ..GpuTuning::default()
        };
        assert_eq!(tuned.bits().unwrap(), 5);
        assert_eq!(GpuTuning { tail_queue_capacity: 262_144, tail_cpu_workers: 4,
            ..tuned }.bits().unwrap(), 5);
        assert_eq!(serde_json::to_value(tuned).unwrap()["tail_budget"], 128);
        assert!(serde_json::to_value(GpuTuning::default()).unwrap().get("tail_budget").is_none());
        for invalid in [
            GpuTuning { overflow_vote: true, ..tuned },
            GpuTuning { readonly_powers: true, ..tuned },
            GpuTuning { oee_mask: false, ..tuned },
            GpuTuning { tail_jump_bits: 8, ..tuned },
            GpuTuning { tail_queue_capacity: 0, ..tuned },
            GpuTuning { tail_queue_capacity: 1_048_577, ..tuned },
            GpuTuning { tail_cpu_workers: 0, ..tuned },
            GpuTuning { tail_cpu_workers: 9, ..tuned },
        ] { assert!(invalid.bits().is_err()); }
    }

    #[test]
    fn early_screening_preserves_fused_kernel_bits_and_checks_configuration() {
        let tuned = GpuTuning { overflow_vote: false, oee_mask: true, early_every: 64,
            ..GpuTuning::default() };
        assert_eq!(tuned.bits().unwrap(), 5);
        for mode in 0..=4 {
            for width in [8, 10, 12, 14] {
                assert_eq!(GpuTuning { early_mode: mode, early_group_bits: width,
                    early_cpu_workers: 4, ..tuned }.bits().unwrap(), 5);
            }
        }
        assert!(serde_json::to_value(GpuTuning::default()).unwrap().get("early_every").is_none());
        for invalid in [
            GpuTuning { overflow_vote: true, ..tuned },
            GpuTuning { readonly_powers: true, ..tuned },
            GpuTuning { oee_mask: false, ..tuned },
            GpuTuning { tail_budget: 16, ..tuned },
            GpuTuning { tail_jump_bits: 10, ..tuned },
            GpuTuning { early_mode: 5, ..tuned },
            GpuTuning { early_group_bits: 9, ..tuned },
            GpuTuning { early_cpu_workers: 0, ..tuned },
            GpuTuning { early_cpu_workers: 9, ..tuned },
        ] { assert!(invalid.bits().is_err()); }
    }
}

#[cfg(test)]
mod wq_spec_tests {
    use super::*;

    #[test]
    fn warp_queue_specs_encode_the_pad_fields() {
        assert_eq!(parse_wq("-").unwrap(), 0);
        assert_eq!(parse_wq("0").unwrap(), 0);
        assert_eq!(parse_wq("v17").unwrap(), (1 << 22) | (17 << 16) | (1 << 10));
        assert_eq!(parse_wq("v2.k16.c3.o1").unwrap(), (1 << 22) | (2 << 16) | (1 << 13) | (3 << 10) | (16 << 3));
        assert!(parse_wq("k8").is_err());
        assert!(parse_wq("v64").is_err());
        assert_eq!(parse_wq("v40").unwrap(), (1 << 23) | (1 << 22) | (8 << 16) | (1 << 10));
        assert!(parse_wq("v2.x1").is_err());
        assert_eq!(parse_wq("t1").unwrap(), (1 << 21) | (1 << 16) | (1 << 10));
        assert_eq!(parse_wq("t16.k4.c2.o3").unwrap(), (1 << 21) | (16 << 16) | (3 << 13) | (2 << 10) | (4 << 3));
        assert!(parse_wq("t32").is_err());
        assert!(is_two_phase(parse_wq("t4").unwrap()) && !is_two_phase(parse_wq("v43.k8").unwrap()));
        assert_eq!(parse_wq("v31.k8").unwrap() & !WQ_FIELDS, 0);
    }
}

#[cfg(test)]
mod hygiene_tests {
    //! c3-hyg host-side checks (no CUDA context: these call host-only
    //! functions of the CUDA library).
    use super::*;
    use crate::jump::{jd_kind, TailKind};

    #[test]
    fn cuda_jd_traits_match_the_cpu_semantics_table() {
        for code in 0..=255u32 {
            let t = jd_code_traits(code);
            match jd_kind(code) {
                Some(TailKind::JumpDip { w, wd, dq, ka, desc }) => {
                    assert_eq!(t[0], 1, "code {code} known to the CPU but not to CUDA");
                    assert_eq!((t[1], t[2], t[3], t[4], t[5] != 0), (w as i32, wd as i32, dq as i32, ka as i32, desc),
                        "code {code}: CUDA traits differ from cv::jump::jd_kind");
                    assert_eq!(t[6] != 0, crate::runspec::is_floor_code(code), "code {code}: floor flag");
                    assert_eq!(t[7] == 1, crate::jdip::fd3_key(wd, dq) && t[6] == 0, "code {code}: FD3 layout");
                }
                _ => assert_eq!(t[0], 0, "code {code} known to CUDA but not to the CPU"),
            }
        }
    }

    #[test]
    fn digest_word_is_xored_across_chunks_only_for_digest_layouts() {
        assert_eq!(wq_layout_kind(52) & 3, 3, "layout 52 is a digest layout");
        assert_eq!(wq_layout_kind(43) & 3, 1, "layout 43 (production) is not");
        assert_eq!(wq_layout_kind(24) & 5, 5, "layout 24 is instrumented");
        assert_eq!(wq_layout_kind(63), 0);
        let dig = parse_wq("v52.k8").unwrap();
        let mut acc = [0u64; 6];
        for chunk in [[5u64, 0xF0, 2, 0, 0, 0], [7, 0x0F, 3, 0, 0, 0], [1, 0xFF, 1, 0, 0, 0]] {
            accumulate_dbg(&mut acc, &chunk, dig);
        }
        assert_eq!(acc[..3], [13, 0x00, 6], "sum, XOR, count");
        let mut acc = [0u64; 6];
        for chunk in [[5u64, 0xF0, 2, 1, 1, 4], [7, 0x0F, 3, 1, 1, 4]] {
            accumulate_dbg(&mut acc, &chunk, parse_wq("v24").unwrap());
        }
        assert_eq!(acc, [12, 0xFF, 5, 2, 2, 8]);
    }

    #[test]
    fn gpu_config_must_equal_the_run_tail_params() {
        let mut tail = crate::tail::TailParams::default();
        tail.codes = crate::runspec::TailCodes::parse("93,91,93", 3).unwrap();
        tail.schedule = crate::runspec::WqSchedule::parse("-,v43.k8,-", 3).unwrap();
        let ok = GpuTuning { xtail: tail.codes.gpu_xtail(), wq: tail.schedule.bits(), ..GpuTuning::default() };
        assert!(check_gpu_config(&tail, &ok, 3, false).is_ok());
        let mut bad = ok;
        bad.xtail[2] = 0;
        assert!(check_gpu_config(&tail, &bad, 3, false).is_err(), "class 2 would run the original loop");
        let mut bad = ok;
        bad.xtail[3] = 93;
        assert!(check_gpu_config(&tail, &bad, 3, false).is_err(), "a code beyond the run's classes");
        let mut bad = ok;
        bad.wq[1] = 0;
        assert!(check_gpu_config(&tail, &bad, 3, false).is_err());
        assert!(check_gpu_config(&crate::tail::TailParams::default(), &GpuTuning::default(), 16, false).is_ok());
        let diag = GpuTuning { screen_diag: SCREEN_DIAG_SCAN, ..ok };
        assert!(check_gpu_config(&tail, &diag, 3, false).is_err());
        assert!(check_gpu_config(&tail, &diag, 3, true).is_ok());
        let mut floor = tail;
        floor.codes = crate::runspec::TailCodes::measurement(&[127], 3).unwrap();
        floor.schedule = crate::runspec::WqSchedule::none();
        let ft = GpuTuning { xtail: floor.codes.gpu_xtail(), ..GpuTuning::default() };
        assert!(check_gpu_config(&floor, &ft, 3, false).is_err());
        assert!(check_gpu_config(&floor, &ft, 3, true).is_ok());
    }
}
