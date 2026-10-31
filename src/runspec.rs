//! c3-hyg: one configuration source for the per-class semantics of a run, and
//! the configuration record every verification report carries.
//!
//! * [`TailCodes`]: the tail code of every bitvector class (cuda/xtail.cuh,
//!   cuda/jdip.cuh; 0 = the original loop).  It lives in
//!   [`crate::tail::TailParams`], so the CPU reference (`tail::endpoint_work`)
//!   and the GPU launches (`gpu::gpu_run`, which refuses a `GpuTuning::xtail`
//!   that differs from it) read the same value.  A list must name every class
//!   (one code for all, or exactly one per class); there is no fallback to the
//!   original loop for classes beyond a short list.  The CPU-only classes
//!   (F beyond the last bitvector) always run the original loop, recorded as
//!   such.  `CV_XTAIL_CPU` (the round-2 side channel) is refused.
//! * [`WqSchedule`]: per-class warp-queue scheduler bits.  Scheduling only (no
//!   effect on the work tuple); recorded identically by CPU and GPU runs, and
//!   `GpuTuning::wq` must equal it.  c3-final: the same per-class word also
//!   selects c3-2ph's two-phase kernel (`t<layout>` specs, pad bit 21 with
//!   bit 22 clear; cuda/twophase.cuh), equally scheduling only.
//! * [`RunRecord`]: what a report must say for its work tuple to be
//!   reproducible and its scope unambiguous: h range, screen, codes, windows,
//!   bitvector rules, warp-queue specs, cutoff, and the version tag of the
//!   iteration / checksum / flag definitions.

use crate::jump::{growth_bits, xv_kind, TailKind};
use crate::lookahead::BvMode;
use crate::tail::Screen;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};

/// Version tag of the work-tuple definitions (cases, iterations, checksum,
/// flags) a report's numbers follow.  `cvwork-1` was the original repo's
/// implicit definition (original loop everywhere); `cvwork-2` adds per-class
/// tail codes whose definitions are listed in `RunRecord::work_definition`.
pub const WORK_DEFINITION_VERSION: &str = "cvwork-2";
/// Schema of the verification report (`VerifyReport`, `config` block).
pub const REPORT_SCHEMA: u32 = 2;
/// Bitvector classes a per-class code list can name (`GpuTuning::xtail`).
pub const MAX_CODED_CLASSES: usize = 32;

/// Measurement-only jd floor codes (cuda/jdip.cuh JD_FLOOR / JD_FLOORW:
/// `JDTraits<code>::floor != 0`): they count cases and run no tail.
pub fn is_floor_code(code: u32) -> bool {
    (125..=127).contains(&code)
}

/// Jump codes the warp-queue scheduler runs (cuda/wq_jump.cuh, cuda/wq_jd.cuh).
pub fn wq_code_ok(code: u32) -> bool {
    (49..=53).contains(&code) || matches!(code, 91 | 92 | 94 | 95)
}

/// c3-final (c3-2ph's kernel): jd FD3 codes the two-phase kernel runs
/// (cuda/twophase.cuh TP_CODES).
pub const TP_CODES: [u32; 6] = [90, 91, 92, 93, 95, 111];
pub fn tp_code_ok(code: u32) -> bool {
    TP_CODES.contains(&code)
}

/// The per-case semantics of a tail code usable in a verification, or why not.
/// These are exactly the codes `gpu_run` accepts.
pub fn verification_kind(code: u32) -> Result<TailKind> {
    ensure!(!is_floor_code(code),
        "tail code {code} is a measurement-only floor (counts cases, runs no tail): not a verification");
    let kind = xv_kind(code).with_context(|| format!("unknown tail code {code}"))?;
    match kind {
        TailKind::Base | TailKind::JumpPrefix { .. } | TailKind::JumpDip { .. }
        | TailKind::Jump { skip_window: false, .. } => Ok(kind),
        TailKind::Jump { skip_window: true, .. } =>
            bail!("tail code {code} skips descent tests by the class window; not supported for verification"),
        TailKind::Cap(_) => bail!("tail code {code} (odd runs capped at 5) is not supported by the GPU pipeline"),
        TailKind::JumpCap { .. } => bail!("tail code {code}: the capped test tail is timing-only"),
    }
}

/// Smallest cutoff a code's flag guard supports (96 + K with 2^K >= 1.5^W).
pub fn min_cutoff(code: u32) -> u32 {
    match xv_kind(code) {
        Some(TailKind::Jump { w, .. } | TailKind::JumpPrefix { w, .. } | TailKind::JumpDip { w, .. }
             | TailKind::JumpCap { w, .. }) => 96 + growth_bits(w),
        _ => 0,
    }
}

/// Per-class tail codes (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TailCodes {
    per_class: [u8; MAX_CODED_CLASSES],
    /// Bitvector classes the list covers; 0 = the original loop everywhere.
    classes: u8,
    /// A measurement-only floor code is present (benchmarks only; never a verification).
    measurement: bool,
}

impl TailCodes {
    /// The original loop on every class.
    pub fn original() -> Self {
        Self::default()
    }

    /// Comma list: one code for every class, or exactly `num_bv` codes.
    pub fn parse(spec: &str, num_bv: usize) -> Result<Self> {
        let codes = parse_code_list(spec)?;
        Self::from_codes(&codes, num_bv)
    }

    /// Verification codes only (floor codes are refused).
    pub fn from_codes(codes: &[u32], num_bv: usize) -> Result<Self> {
        Self::build(codes, num_bv, false)
    }

    /// Benchmarks: as `from_codes`, but measurement-only floor codes are
    /// allowed; a run with one is marked measurement-only (not a
    /// verification) and needs an explicit opt-in on the GPU session.
    pub fn measurement(codes: &[u32], num_bv: usize) -> Result<Self> {
        Self::build(codes, num_bv, true)
    }

    fn build(codes: &[u32], num_bv: usize, allow_floor: bool) -> Result<Self> {
        ensure!((1..=MAX_CODED_CLASSES).contains(&num_bv),
            "per-class tail codes need 1..={MAX_CODED_CLASSES} bitvector classes (the run has {num_bv})");
        ensure!(codes.len() == 1 || codes.len() == num_bv,
            "tail codes: give one code for every class or exactly one per bitvector class ({num_bv}); got {} \
             (no class may fall back to the original loop silently)", codes.len());
        let mut per_class = [0u8; MAX_CODED_CLASSES];
        let mut floor = false;
        for (i, slot) in per_class.iter_mut().enumerate().take(num_bv) {
            let code = if codes.len() == 1 { codes[0] } else { codes[i] };
            if allow_floor && is_floor_code(code) {
                floor = true;
            } else {
                verification_kind(code).with_context(|| format!("bitvector class {i}"))?;
            }
            *slot = u8::try_from(code).context("tail codes are below 256")?;
        }
        Ok(TailCodes { per_class, classes: num_bv as u8, measurement: floor })
    }

    /// The code of bitvector class `class` (None: a CPU-only class, always 0).
    #[inline]
    pub fn code(&self, class: Option<usize>) -> u32 {
        match class {
            Some(c) if self.classes != 0 => {
                assert!(c < self.classes as usize,
                    "bitvector class {c} has no tail code (codes were given for {} classes): configuration mismatch",
                    self.classes);
                self.per_class[c] as u32
            }
            _ => 0,
        }
    }

    pub fn is_original(&self) -> bool {
        self.classes == 0 || self.per_class.iter().all(|&c| c == 0)
    }
    pub fn is_measurement(&self) -> bool {
        self.measurement
    }
    /// Classes the list was given for (0 = original loop everywhere).
    pub fn classes(&self) -> usize {
        self.classes as usize
    }
    /// `GpuTuning::xtail` image: the per-class codes, zero beyond the list.
    pub fn gpu_xtail(&self) -> [u8; MAX_CODED_CLASSES] {
        self.per_class
    }
    /// Codes of classes `0..num_bv` (0 everywhere for the original loop).
    pub fn per_class(&self, num_bv: usize) -> Vec<u32> {
        (0..num_bv).map(|c| if self.classes == 0 { 0 } else if c < self.classes as usize { self.per_class[c] as u32 } else { 0 })
            .collect()
    }

    /// The list must cover exactly the run's bitvector classes, and each
    /// code's flag guard needs cutoff >= 96 + K.
    pub fn check(&self, num_bv: usize, cutoff_bits: u32) -> Result<()> {
        ensure!(self.classes == 0 || self.classes as usize == num_bv,
            "tail codes were given for {} classes but the run has {num_bv} bitvector classes", self.classes);
        for (c, code) in self.per_class(num_bv).into_iter().enumerate() {
            let need = min_cutoff(code);
            ensure!(cutoff_bits >= need,
                "class {c}: tail code {code} needs cutoff-bits >= {need} (96 + K; got {cutoff_bits})");
        }
        Ok(())
    }
}

pub fn parse_code_list(spec: &str) -> Result<Vec<u32>> {
    spec.split(',').map(|c| c.trim().parse::<u32>().with_context(|| format!("bad tail code {c:?} in {spec:?}")))
        .collect()
}

/// Warp-queue pad fields a verification may set (cuda/wq_jump.cuh: bit 22,
/// K bits 3-7, log2 fetch 10-12, blocks/SM 13-15, layout 16-20 + bit 23).
/// Bit 21 (the timing-only capped test tail) is excluded.
pub const WQ_FIELDS: u32 = (1 << 23) | (1 << 22) | (31 << 16) | (7 << 13) | (7 << 10) | (31 << 3);

/// c3-final (c3-2ph): fields of the two-phase kernel (cuda/twophase.cuh
/// TP_FIELD_MASK): bit 21 (bit 22 clear), layout bits 16-20, blocks/SM cap
/// 13-15, log2 fetch 10-12, K 3-7.
pub const TP_FIELDS: u32 = (1 << 21) | (31 << 16) | (7 << 13) | (7 << 10) | (31 << 3);

/// Is this per-class scheduler word a two-phase kernel request?
pub fn is_two_phase(bits: u32) -> bool {
    bits & (1 << 21) != 0 && bits & (1 << 22) == 0
}

/// Warp-queue bits from a spec `v<layout>[.k<K>][.c<log2 fetch>][.o<blocks/SM>]`
/// (fetch defaults to 2 endpoints); `-` or `0` keeps the per-thread kernel.
/// c3-final (c3-2ph): `t<layout>[.k<K>][.c<log2 fetch>][.o<blocks/SM>]`
/// selects the two-phase kernel (layout 1..31; same field meanings).
pub fn parse_wq_spec(spec: &str) -> Result<u32> {
    let spec = spec.trim();
    if spec == "-" || spec == "0" {
        return Ok(0);
    }
    let (mut v, mut k, mut c, mut o) = (None, 0u32, 1u32, 0u32);
    let mut tp = false;
    for part in spec.split('.') {
        ensure!(!part.is_empty(), "bad warp-queue spec {spec}");
        let (key, value) = part.split_at(1);
        let value: u32 = value.parse().map_err(|_| anyhow::anyhow!("bad warp-queue field {part} in {spec}"))?;
        match key {
            "t" => { ensure!(v.is_none(), "bad warp-queue spec {spec}"); v = Some(value); tp = true; }
            "v" => { ensure!(v.is_none(), "bad warp-queue spec {spec}"); v = Some(value) }
            "k" => k = value,
            "c" => c = value,
            "o" => o = value,
            _ => bail!("bad warp-queue field {part} in {spec}"),
        }
    }
    let v = v.ok_or_else(|| anyhow::anyhow!("warp-queue spec {spec} needs a layout v<N> or t<N>"))?;
    ensure!(v < 64 && k < 32 && c < 8 && o < 8, "warp-queue spec {spec} out of range");
    if tp {
        ensure!(v < 32 && v > 0, "two-phase spec {spec}: layout 1..31");
        return Ok((1 << 21) | (v << 16) | (o << 13) | (c << 10) | (k << 3));
    }
    Ok(((v >> 5) << 23) | (1 << 22) | ((v & 31) << 16) | (o << 13) | (c << 10) | (k << 3))
}

/// The canonical spec of warp-queue bits (inverse of `parse_wq_spec`).
pub fn wq_spec_string(bits: u32) -> String {
    if bits == 0 {
        return "-".into();
    }
    let tp = is_two_phase(bits);
    let v = if tp { (bits >> 16) & 31 } else { ((bits >> 16) & 31) | (((bits >> 23) & 1) << 5) };
    let (k, c, o) = ((bits >> 3) & 31, (bits >> 10) & 7, (bits >> 13) & 7);
    let mut s = if tp { format!("t{v}") } else { format!("v{v}") };
    if k != 0 { s += &format!(".k{k}"); }
    if c != 1 { s += &format!(".c{c}"); }
    if o != 0 { s += &format!(".o{o}"); }
    s
}

/// Per-class warp-queue scheduling (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct WqSchedule {
    bits: [u32; MAX_CODED_CLASSES],
}

impl WqSchedule {
    pub fn none() -> Self {
        Self::default()
    }
    /// One spec for every class or exactly `num_bv`.
    pub fn parse(spec: &str, num_bv: usize) -> Result<Self> {
        ensure!((1..=MAX_CODED_CLASSES).contains(&num_bv), "warp-queue specs need 1..={MAX_CODED_CLASSES} classes");
        let specs: Vec<&str> = spec.split(',').collect();
        ensure!(specs.len() == 1 || specs.len() == num_bv,
            "warp-queue specs: one for every class or exactly {num_bv}; got {}", specs.len());
        let mut bits = [0u32; MAX_CODED_CLASSES];
        for (i, b) in bits.iter_mut().enumerate().take(num_bv) {
            *b = parse_wq_spec(specs[if specs.len() == 1 { 0 } else { i }])?;
        }
        Ok(WqSchedule { bits })
    }
    pub fn from_bits(bits: [u32; MAX_CODED_CLASSES]) -> Result<Self> {
        for (i, &b) in bits.iter().enumerate() {
            ensure!(b == 0 || (b & !WQ_FIELDS == 0 && b & (1 << 22) != 0)
                    || (is_two_phase(b) && b & !TP_FIELDS == 0 && (b >> 16) & 31 != 0),
                "class {i}: warp-queue bits {b:#x} outside the verification fields");
        }
        Ok(WqSchedule { bits })
    }
    pub fn bits(&self) -> [u32; MAX_CODED_CLASSES] {
        self.bits
    }
    pub fn is_none(&self) -> bool {
        self.bits.iter().all(|&b| b == 0)
    }
    pub fn specs(&self, num_bv: usize) -> Vec<String> {
        (0..num_bv).map(|c| wq_spec_string(self.bits.get(c).copied().unwrap_or(0))).collect()
    }
    /// Classes beyond the run carry no bits, and every scheduled class has a
    /// tail code the warp queue runs.
    pub fn check(&self, codes: &TailCodes, num_bv: usize) -> Result<()> {
        for (c, &b) in self.bits.iter().enumerate() {
            if b == 0 {
                continue;
            }
            ensure!(c < num_bv, "warp-queue spec for class {c}, but the run has {num_bv} bitvector classes");
            let code = codes.code(Some(c));
            if is_two_phase(b) {
                ensure!(b & !TP_FIELDS == 0 && (b >> 16) & 31 != 0,
                    "class {c}: two-phase bits {b:#x} outside the verification fields");
                ensure!(tp_code_ok(code),
                    "class {c}: the two-phase kernel runs jd FD3 codes {TP_CODES:?} (class has {code})");
                continue;
            }
            ensure!(b & !WQ_FIELDS == 0 && b & (1 << 22) != 0,
                "class {c}: warp-queue bits {b:#x} outside the verification fields");
            ensure!(wq_code_ok(code),
                "class {c}: the warp-queue scheduler runs jump codes 49..53 or jd codes 91/92/94/95 (class has {code})");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Report record.

/// Completion range of a run (`Screen::h_lo`, `h_hi`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct HRange {
    /// `full`, `top_half`, `bottom_half` or `explicit`.
    pub kind: String,
    /// Completions n = n0 + h 2^M with h in [h_lo, h_hi).
    pub h_lo: u32,
    pub h_hi: u32,
    /// 2^A completions per endpoint.
    pub completions: u32,
    /// What a successful run establishes.
    pub covers: String,
}

pub fn h_range(screen: &Screen, n: u32, a: u32) -> HRange {
    let total = 1u32 << a;
    let hi = if screen.h_hi == 0 { total } else { screen.h_hi };
    let lo = screen.h_lo;
    let half = total / 2;
    let (kind, covers) = if lo == 0 && hi == total {
        ("full", format!("every n < 2^{n}"))
    } else if lo == half && hi == total {
        ("top_half", format!("every n in [2^{}, 2^{n}); sound as a verification of n < 2^{n} only together with \
                              a verification of every n < 2^{} (strong induction)", n - 1, n - 1))
    } else if lo == 0 && hi == half {
        ("bottom_half", format!("every n < 2^{}", n - 1))
    } else {
        ("explicit", format!("completions n = n0 + h 2^M with h in [{lo}, {hi}) of every endpoint; sound for that \
                              set only together with a verification of every smaller n"))
    };
    HRange { kind: kind.into(), h_lo: lo, h_hi: hi, completions: total, covers }
}

/// Work units of a run.
#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct UnitsRecord {
    /// `all` (the whole tree) or `selected` (a work package / unit list).
    pub selection: String,
    /// Selected units: how many, and a 64-bit FNV-1a digest of their (n0, k)
    /// in order (identity check, not cryptographic).
    pub count: Option<usize>,
    pub digest: Option<String>,
    /// `cv verify --plan/--package`: which plan and packages.
    pub plan: Option<PlanRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanRecord {
    pub file: String,
    /// FNV-1a 64 of the plan file's bytes.
    pub file_digest: String,
    /// Package id or range ("7" or "7..12").
    pub packages: String,
    /// The plan's own metadata (what it was made for).
    pub plan_n: u32,
    pub plan_a: u32,
    pub plan_b: u32,
    pub plan_rules: String,
    /// Slices per package (`--slices`), and the single slice run (`--slice`), if any.
    pub slices: usize,
    pub slice: Option<usize>,
}

/// c3-hyg: the work tuple of one part (package, slice) of a plan run, for spot
/// checks: re-run the same configuration with `--package <package> --slices
/// <slices> --slice <slice>` (CPU or GPU) and compare.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PartRecord {
    pub package: usize,
    pub slice: usize,
    pub slices: usize,
    pub units: usize,
    pub units_digest: String,
    pub tree_nodes: u64,
    pub endpoints: u64,
    pub cases: u64,
    pub iters: u64,
    pub hash: String,
    pub flagged: usize,
    /// FNV-1a 64 of the sorted flagged starts (decimal, comma separated).
    pub flagged_digest: String,
}

impl PartRecord {
    pub fn of(package: usize, slice: usize, slices: usize, nodes: &[crate::tree::Node],
              r: &crate::verify::VerifyReport) -> Self {
        let mut fl: Vec<u128> = r.flagged.iter().map(|x| x.parse().unwrap()).collect();
        fl.sort_unstable();
        let text = fl.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        PartRecord {
            package,
            slice,
            slices,
            units: nodes.len(),
            units_digest: units_digest(nodes),
            tree_nodes: r.tree_nodes,
            endpoints: r.endpoints,
            cases: r.cases,
            iters: r.iters,
            hash: r.hash.clone(),
            flagged: fl.len(),
            flagged_digest: format!("{:016x}", fnv1a64(text.into_bytes())),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScreenRecord {
    /// `mod9` everywhere, or the (12, 9) row table for classes >= rows_from
    /// and the CPU-only classes.
    pub rows: bool,
    pub rows_from: u32,
    pub label: String,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct BitvectorRecord {
    pub mode: BvMode,
    pub b: u32,
    pub num_bv: usize,
    /// Merge credits in the window (`--bv-pm-inf`, `--bv-mix-j`).
    pub pm_inf: bool,
    pub mix_j: u32,
    pub label: String,
    /// Window of every bitvector class (`--class-b`, padded with `--b`).
    pub class_windows: Vec<u32>,
}

/// Definitions of one tail code's work tuple.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct CodeDefinition {
    pub code: u32,
    /// Bitvector classes running it; `cpu_only` = the classes beyond the last bitvector.
    pub classes: Vec<usize>,
    pub cpu_only: bool,
    pub tag: String,
    pub iterations: String,
    pub hash: String,
    pub stop: String,
    pub flag: String,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct WorkDefinition {
    pub version: &'static str,
    pub cases: String,
    pub codes: Vec<CodeDefinition>,
}

pub fn code_definition(code: u32) -> CodeDefinition {
    let jump_iter = "exact W-step jumps x -> T^W(x) on x = T^j(n), from x = T^M(n)".to_string();
    let jump_hash = "XOR of x mod 2^32 after every jump".to_string();
    let flag = |w: u32| format!("x >= 2^(cutoff - {}) at a jump boundary before the stop; flagged starts are \
                                 finished by the 256-bit CPU recheck", growth_bits(w));
    let (tag, iterations, hash, stop, flag) = match xv_kind(code) {
        Some(TailKind::Base) => (
            "orig".to_string(),
            "double iterations (odd run + even run, twice) of the alternating loop on y = T^j(n) + 1".to_string(),
            "XOR of y mod 2^32 after every double iteration".to_string(),
            "y < n after a double iteration".to_string(),
            "an odd-run end reaches 2^cutoff (the case then ends); flagged starts are finished by the 256-bit \
             CPU recheck".to_string()),
        Some(TailKind::Jump { w, coarse, .. }) => (
            format!("jump-W{w}{}", if coarse { "-coarse" } else { "" }),
            jump_iter, jump_hash,
            if coarse { "after a jump: (x >> 32) < (n >> 32) for 2^32 <= n < 2^96, else x < n".to_string() }
            else { "x < n after a jump".to_string() },
            flag(w)),
        Some(TailKind::JumpPrefix { w, prefix }) => (
            format!("jump-W{w}-prefix{prefix}"),
            jump_iter, jump_hash,
            format!("after jump {prefix} and every later jump: (x >> 32) < (n >> 32) for 2^32 <= n < 2^96, else x < n"),
            flag(w)),
        Some(TailKind::JumpDip { w, wd, dq, ka, desc }) => (
            format!("jd-W{w}-WD{wd}-Q{dq}-KA{ka}{}", if desc { "-desc" } else { "" }),
            jump_iter, jump_hash,
            format!("for 2^32 <= n < 2^96: at a boundary, dip[x_(j+{ka}) mod 2^{wd}] > H_(j+{ka}) with H = 485 F' - 306 M' \
                     (dips stored as {dq} floor(dip / {dq}), {} window rules); otherwise x < n after a jump",
                    if desc { "descent-only" } else { "production (pm_inf + mix(3^13))" }),
            flag(w)),
        Some(TailKind::Cap(c)) => (format!("orig-cap{c}"), "double iterations with odd runs capped".into(),
            "XOR of y mod 2^32 after every double iteration".into(), "y < n".into(), "odd-run end >= 2^cutoff".into()),
        Some(TailKind::JumpCap { w, cap }) => (format!("jump-W{w}-cap{cap} (timing only)"), "jumps".into(),
            "XOR".into(), "after at most cap jumps (not a certificate)".into(), flag(w)),
        None => (format!("unknown code {code}"), String::new(), String::new(), String::new(), String::new()),
    };
    let (tag, iterations, stop) = if is_floor_code(code) {
        (format!("floor-{code} (measurement only)"), "none (no tail)".to_string(), "none: NOT A VERIFICATION".to_string())
    } else {
        (tag, iterations, stop)
    };
    CodeDefinition { code, classes: Vec::new(), cpu_only: false, tag, iterations, hash, stop, flag }
}

/// The configuration record of a run (identical for the CPU and GPU paths).
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct RunRecord {
    pub schema: u32,
    pub n: u32,
    pub a: u32,
    pub m: u32,
    pub rules: String,
    pub h_range: HRange,
    pub units: UnitsRecord,
    pub screen: ScreenRecord,
    pub bitvector: BitvectorRecord,
    /// Tail code of every bitvector class; the CPU-only classes run code 0.
    pub tail_codes: Vec<u32>,
    pub cpu_only_classes_code: u32,
    /// GPU scheduling only (no effect on the work tuple): per-class spec or `-`.
    pub warp_queue: Vec<String>,
    pub cutoff_bits: u32,
    pub tail_oee: bool,
    pub work_definition: WorkDefinition,
    /// True for benchmark runs with measurement-only kernels: not a verification.
    pub measurement_only: bool,
}

/// FNV-1a 64 (identity digests in reports; not cryptographic).
pub fn fnv1a64(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn units_digest(units: &[crate::tree::Node]) -> String {
    let d = fnv1a64(units.iter().flat_map(|u| u.n0.to_le_bytes().into_iter().chain(u.k.to_le_bytes())));
    format!("{d:016x}")
}

impl RunRecord {
    pub fn of(cfg: &crate::verify::VerifyConfig) -> Self {
        let tp = &cfg.tail;
        let codes = tp.codes.per_class(cfg.num_bv);
        let mut defs: Vec<CodeDefinition> = Vec::new();
        let mut add = |code: u32, class: Option<usize>| {
            let i = match defs.iter().position(|d| d.code == code) {
                Some(i) => i,
                None => {
                    defs.push(code_definition(code));
                    defs.len() - 1
                }
            };
            match class {
                Some(c) => defs[i].classes.push(c),
                None => defs[i].cpu_only = true,
            }
        };
        for (c, &code) in codes.iter().enumerate() {
            add(code, Some(c));
        }
        add(tp.codes.code(None), None);
        let class_windows: Vec<u32> = (0..cfg.num_bv)
            .map(|i| cfg.class_b.as_ref().and_then(|w| w.get(i).copied()).unwrap_or(cfg.b))
            .collect();
        RunRecord {
            schema: REPORT_SCHEMA,
            n: cfg.n,
            a: cfg.a,
            m: cfg.n.saturating_sub(cfg.a),
            rules: cfg.rules.label(),
            h_range: h_range(&tp.screen, cfg.n, cfg.a),
            units: match &cfg.units {
                None => UnitsRecord { selection: "all".into(), ..Default::default() },
                Some(u) => UnitsRecord { selection: "selected".into(), count: Some(u.len()),
                                         digest: Some(units_digest(u)), plan: None },
            },
            screen: ScreenRecord { rows: tp.screen.rows, rows_from: tp.screen.rows_from,
                                   label: tp.screen.label() },
            bitvector: BitvectorRecord {
                mode: cfg.bv_mode,
                b: cfg.b,
                num_bv: cfg.num_bv,
                pm_inf: cfg.window.pm_inf,
                mix_j: cfg.window.mix.as_ref().map_or(0, |t| t.j),
                label: cfg.window.label(),
                class_windows,
            },
            tail_codes: codes,
            cpu_only_classes_code: tp.codes.code(None),
            warp_queue: tp.schedule.specs(cfg.num_bv),
            cutoff_bits: tp.cutoff_bits,
            tail_oee: tp.tail_oee,
            work_definition: WorkDefinition {
                version: WORK_DEFINITION_VERSION,
                cases: "completions n = n0 + h 2^M (h in h_range) of every depth-M endpoint kept by its class's \
                        bitvector window (bitvector.class_windows, bitvector rules), the mask (mod 9, or the \
                        (12, 9) row table for classes >= screen.rows_from and the CPU-only classes) and the \
                        carried OEE state (tail_oee)".into(),
                codes: defs,
            },
            measurement_only: tp.codes.is_measurement(),
        }
    }

    /// `verification_scope` of the report (top level).
    pub fn scope_label(&self) -> String {
        if self.measurement_only {
            return "measurement_only_not_a_verification".into();
        }
        let range = match self.h_range.kind.as_str() {
            "full" => None,
            "explicit" => Some(format!("h_range[{},{})", self.h_range.h_lo, self.h_range.h_hi)),
            k => Some(k.to_string()),
        };
        match (self.units.selection == "selected", range) {
            (false, None) => "full_range".into(),
            (true, None) => "selected_work_units".into(),
            (false, Some(r)) => r,
            (true, Some(r)) => format!("selected_work_units+{r}"),
        }
    }
}

/// The round-2 side channels for the CPU semantics are gone; refuse them
/// instead of silently ignoring them.
pub fn reject_legacy_env() -> Result<()> {
    for var in ["CV_XTAIL_CPU", "CV_TAIL"] {
        ensure!(std::env::var_os(var).is_none(),
            "{var} is no longer read: the CPU reference takes its per-class tail codes from the run's \
             configuration (TailParams::codes; cv verify --tail-codes). Unset {var}.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_need_every_class_and_refuse_measurement_floors() {
        let full = TailCodes::parse("93,91,93,90,90,90,90,111,111,111,111,111,111,111,111,111", 16).unwrap();
        assert_eq!(full.code(Some(0)), 93);
        assert_eq!(full.code(Some(1)), 91);
        assert_eq!(full.code(Some(15)), 111);
        assert_eq!(full.code(None), 0, "CPU-only classes run the original loop");
        assert_eq!(TailCodes::parse("93", 16).unwrap().per_class(16), vec![93; 16]);
        // A short list is refused (round 2 silently ran the original loop beyond it on the CPU).
        assert!(TailCodes::parse("93,91", 16).is_err());
        assert!(TailCodes::parse("93,91,93", 4).is_err());
        for floor in [125, 126, 127] {
            let e = format!("{:#}", TailCodes::parse(&floor.to_string(), 16).unwrap_err());
            assert!(e.contains("measurement-only"), "{e}");
            assert!(TailCodes::measurement(&[floor], 16).unwrap().is_measurement());
        }
        assert!(!TailCodes::measurement(&[93], 16).unwrap().is_measurement());
        for bad in [6, 7, 18, 25, 26, 63, 74, 128, 300] {
            assert!(TailCodes::parse(&bad.to_string(), 16).is_err(), "code {bad}");
        }
        assert!(TailCodes::parse("93", 33).is_err(), "at most 32 coded classes");
        assert!(TailCodes::original().is_original());
        assert_eq!(TailCodes::original().per_class(3), vec![0, 0, 0]);
    }

    #[test]
    fn code_list_must_match_the_class_count_and_cutoff() {
        let codes = TailCodes::parse("93,91", 2).unwrap();
        assert!(codes.check(2, 120).is_ok());
        assert!(codes.check(3, 120).is_err());
        // W12 (code 91) needs 96 + 8, W11 (93) needs 96 + 7.
        assert!(codes.check(2, 104).is_ok());
        assert!(codes.check(2, 103).is_err());
        assert!(TailCodes::parse("93", 2).unwrap().check(2, 103).is_ok());
        assert_eq!(min_cutoff(0), 0);
        assert_eq!(min_cutoff(111), 104);
    }

    #[test]
    #[should_panic(expected = "configuration mismatch")]
    fn a_class_beyond_the_list_panics_instead_of_falling_back() {
        TailCodes::parse("93,91", 2).unwrap().code(Some(2));
    }

    #[test]
    fn warp_queue_specs_round_trip_and_need_queue_codes() {
        for spec in ["-", "v43.k8", "v17", "v52.k8.c3.o1", "v2.k16"] {
            assert_eq!(wq_spec_string(parse_wq_spec(spec).unwrap()), spec);
        }
        let codes = TailCodes::parse("93,91,93", 3).unwrap();
        let s = WqSchedule::parse("-,v43.k8,-", 3).unwrap();
        assert!(s.check(&codes, 3).is_ok());
        assert_eq!(s.specs(3), vec!["-", "v43.k8", "-"]);
        assert!(WqSchedule::parse("v43.k8,-,-", 3).unwrap().check(&codes, 3).is_err(), "code 93 has no warp queue");
        assert!(WqSchedule::parse("-,v43", 3).is_err());
        assert!(WqSchedule::from_bits([1 << 21; 32]).is_err(), "capped test tail bit");
    }

    #[test]
    fn two_phase_specs_round_trip_and_need_two_phase_codes() {
        // c3-final: c3-2ph's two-phase kernel shares the per-class scheduler word (bit 21, bit 22 clear).
        for spec in ["t3", "t5", "t29", "t30", "t31", "t3.k4.c2.o1"] {
            let b = parse_wq_spec(spec).unwrap();
            assert!(is_two_phase(b) && b & !TP_FIELDS == 0, "{spec}");
            assert_eq!(wq_spec_string(b), spec);
        }
        assert!(!is_two_phase(parse_wq_spec("v43.k8").unwrap()));
        assert!(parse_wq_spec("t0").is_err() && parse_wq_spec("t32").is_err() && parse_wq_spec("t3.v4").is_err());
        let codes = TailCodes::parse("91,91,93,111", 4).unwrap();
        let s = WqSchedule::parse("t3,t3,t5,t5", 4).unwrap();
        assert!(s.check(&codes, 4).is_ok());
        assert_eq!(s.specs(4), vec!["t3", "t3", "t5", "t5"]);
        assert_eq!(WqSchedule::from_bits(s.bits()).unwrap(), s);
        // jump codes and warp-queue-only jd codes have no two-phase kernel
        for bad in ["49,91,91,91", "94,91,91,91", "0,91,91,91"] {
            assert!(s.check(&TailCodes::parse(bad, 4).unwrap(), 4).is_err(), "{bad}");
        }
        // the two-phase bit without a layout, and bits outside its fields, are refused
        let mut bits = [0u32; MAX_CODED_CLASSES];
        bits[0] = 1 << 21;
        assert!(WqSchedule::from_bits(bits).is_err());
        bits[0] = parse_wq_spec("t3").unwrap() | (1 << 9);
        assert!(WqSchedule::from_bits(bits).is_err());
    }

    #[test]
    fn h_ranges_and_scopes_are_named() {
        let s = |sc: Screen| h_range(&sc, 40, 10);
        assert_eq!(s(Screen::default()).kind, "full");
        assert_eq!(s(Screen::top(10)).kind, "top_half");
        assert_eq!(s(Screen::bottom(10)).kind, "bottom_half");
        let e = s(Screen { h_lo: 64, h_hi: 128, ..Screen::default() });
        assert_eq!((e.kind.as_str(), e.h_lo, e.h_hi, e.completions), ("explicit", 64, 128, 1024));
        assert_eq!(s(Screen::top(10)).h_lo, 512);
        assert_eq!(s(Screen::top(10)).h_hi, 1024);
    }

    #[test]
    fn every_jd_code_has_a_definition_tag() {
        for code in 64..=127 {
            if xv_kind(code).is_some() {
                let d = code_definition(code);
                assert!(!d.tag.is_empty() && !d.tag.starts_with("unknown"), "code {code}");
            }
        }
        assert_eq!(code_definition(93).tag, "jd-W11-WD11-Q32-KA0");
        assert_eq!(code_definition(91).tag, "jd-W12-WD12-Q128-KA1");
        assert_eq!(code_definition(0).tag, "orig");
        assert!(code_definition(127).stop.contains("NOT A VERIFICATION"));
    }
}
