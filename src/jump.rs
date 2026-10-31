//! Experimental exact W-step jump tail: CPU reference for the CUDA jump kernels.
//!
//! For `x = 2^W q + r` with `0 <= r < 2^W`, the first `W` parities of `x` are
//! those of `r`, so `T^W(x) = 3^f(r) q + T^W(r)` exactly, where `f(r)` is the
//! number of odd steps among the first `W` steps of `r`.  A table of
//! `(3^f(r), T^W(r))` advances the actual trajectory value `x = T^j(n)` (not
//! `x + 1`) by `W` steps per lookup.
//!
//! Semantics (identical on CPU and GPU):
//! * a case starts at `x = T^M(n)`;
//! * before every jump, `x >= 2^(cutoff - K)` flags the case and stops it,
//!   where `2^K >= 1.5^W`.  Because `T(z) + 1 <= 1.5 (z + 1)`, every value
//!   inside an unflagged jump is `< 2^cutoff`, so every trajectory that reaches
//!   `2^cutoff` before stopping is flagged (a conservative superset), and the
//!   128-bit arithmetic never wraps;
//! * after every jump the low 32 bits of `x` are XORed into the hash and the
//!   jump counter is incremented;
//! * after jump `j > skip`, `x < n` stops the case: a genuine descent.
//!   Skipping checks only delays stopping; it never accepts a non-descent.
//!
//! The baseline loop (`crate::tail::step_until_below`) stays the default.

use crate::arith::POW3;
use crate::lookahead::Lookahead;
use crate::tail::{step_until_below, TailParams};
use crate::tree::Node;
use std::sync::OnceLock;

pub struct JumpTable {
    pub w: u32,
    pub mult: Vec<u32>,
    pub add: Vec<u32>,
    pub odd: Vec<u8>,
}

impl JumpTable {
    pub fn new(w: u32) -> Self {
        assert!((1..=20).contains(&w));
        let size = 1usize << w;
        let (mut mult, mut add, mut odd) = (Vec::with_capacity(size), Vec::with_capacity(size), Vec::with_capacity(size));
        for r in 0..size as u64 {
            let mut x = r;
            let mut f = 0u32;
            for _ in 0..w {
                if x & 1 == 1 {
                    x = (3 * x + 1) / 2;
                    f += 1;
                } else {
                    x /= 2;
                }
            }
            assert!(x < POW3[f as usize] as u64 || (r == 0 && x == 0), "T^W(r) < 3^f(r)");
            mult.push(POW3[f as usize] as u32);
            add.push(x as u32);
            odd.push(f as u8);
        }
        JumpTable { w, mult, add, odd }
    }

    #[inline(always)]
    pub fn jump(&self, x: u128) -> u128 {
        let r = (x as usize) & ((1usize << self.w) - 1);
        (self.mult[r] as u128) * (x >> self.w) + self.add[r] as u128
    }
}

/// Smallest K with 2^K >= 1.5^W, i.e. 2^(K+W) >= 3^W.
pub fn growth_bits(w: u32) -> u32 {
    let p = POW3[w as usize];
    (0..64).find(|&k| (1u128 << (k + w)) >= p).unwrap()
}

/// Returns (hash, jumps, flagged).
#[inline]
pub fn jump_until_below(base: u128, x: u128, tab: &JumpTable, flag_at: u128, skip: u32) -> (u32, u64, bool) {
    jump_until(base, x, tab, flag_at, skip, false)
}

#[inline]
pub fn jump_until(base: u128, mut x: u128, tab: &JumpTable, flag_at: u128, skip: u32, coarse: bool) -> (u32, u64, bool) {
    let mut hash = 0u32;
    let mut jumps = 0u64;
    loop {
        if x >= flag_at {
            return (hash, jumps, true);
        }
        x = tab.jump(x);
        hash ^= x as u32;
        jumps += 1;
        if jumps > skip as u64 && if coarse { descends_coarse(x, base) } else { x < base } {
            return (hash, jumps, false);
        }
    }
}

/// Baseline loop with the odd run capped at `cap` (<= 8) instead of 8.
#[inline]
pub fn step_until_below_cap(base: u128, mut number: u128, cutoff: u128, cap: u32) -> (u32, u64, bool) {
    let mut hash = 0u32;
    let mut iters = 0u64;
    let mut flagged = false;
    loop {
        for _ in 0..2 {
            let a = ((number as u32) | (1 << cap)).trailing_zeros();
            number = POW3[a as usize] * (number >> a);
            if number >= cutoff {
                flagged = true;
                number = 2;
            }
            number -= 1;
            let b = ((number as u32) | (1 << 31)).trailing_zeros();
            number = (number >> b) + 1;
        }
        hash ^= number as u32;
        iters += 1;
        if number < base {
            return (hash, iters, flagged);
        }
    }
}

/// Per-case semantics of a tail code (CPU reference of the CUDA variants).
/// c3-hyg: selected per class by `TailParams::codes` (`crate::runspec`); the
/// round-2 environment selectors `CV_TAIL` / `CV_XTAIL_CPU` are refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TailKind {
    /// Original double-iteration loop (any GPU code in `CV_GPU_VARIANT` that
    /// preserves it, e.g. shared-memory powers or a 3-limb fast path).
    Base,
    /// Original loop with the odd run capped at `cap`.
    Cap(u32),
    /// Exact W-step jumps; `skip_window` skips descent checks for the first
    /// floor(B/W) jumps of each case (B = that class's window).
    Jump { w: u32, skip_window: bool, coarse: bool },
    /// Coarse-test jumps whose descent tests start after a fixed `prefix` jumps.
    JumpPrefix { w: u32, prefix: u32 },
    /// TIMING-ONLY test tail of the warp-queue scheduler (pad bit 21): the
    /// coarse jump tail, but every case also stops after `cap` jumps.  Not a
    /// certificate; never used for verification.
    JumpCap { w: u32, cap: u32 },
    /// c-tail: jumps with the window-dip certificate (width `wd`, quantum `dq`,
    /// descent-only when `desc`) tested `ka` windows ahead (cv::jdip).
    JumpDip { w: u32, wd: u32, dq: u32, ka: u32, desc: bool },
}

/// Semantics of a cuda/jdip.cuh code (64..=127), mirroring its JD_DEF table.
pub fn jd_kind(xv: u32) -> Option<TailKind> {
    let k = |w, wd, dq, ka, desc| Some(TailKind::JumpDip { w, wd, dq, ka, desc });
    match xv {
        64 => k(12, 12, 16, 0, false),
        65 => k(12, 12, 16, 1, false),
        66 => k(12, 12, 16, 2, false),
        67 => k(12, 12, 1, 2, false),
        68 => k(12, 16, 32, 2, false),
        69 => k(12, 20, 32, 2, false),
        70 => k(12, 20, 32, 1, false),
        71 => k(12, 16, 32, 1, false),
        72 => k(12, 24, 32, 1, false),
        73 => k(12, 12, 16, 2, false),
        76 => k(12, 12, 16, 2, false),
        77 => k(12, 16, 32, 2, false),
        78 => k(12, 20, 32, 2, false),
        80 => k(11, 12, 16, 2, false),
        81 => k(11, 16, 32, 2, false),
        82 => k(11, 20, 32, 2, false),
        84 => k(10, 12, 16, 2, false),
        85 => k(10, 16, 32, 2, false),
        86 => k(10, 20, 32, 2, false),
        88 => k(12, 12, 16, 2, true),
        89 => k(12, 20, 32, 2, true),
        // FD3 embedded dips (Wd = W, quantum 128/32/8 for W = 12/11/10).
        90 => k(12, 12, 128, 0, false),
        91 => k(12, 12, 128, 1, false),
        92 => k(12, 12, 128, 2, false),
        93 => k(11, 11, 32, 0, false),
        94 => k(11, 11, 32, 1, false),
        95 => k(11, 11, 32, 2, false),
        96 => k(10, 10, 8, 0, false),
        97 => k(10, 10, 8, 1, false),
        98 => k(10, 10, 8, 2, false),
        99 => k(12, 12, 128, 1, false),
        100 => k(12, 12, 128, 2, false),
        101 => k(12, 12, 128, 1, true),
        102 => k(12, 12, 128, 2, true),
        103 => k(12, 12, 128, 0, false),
        104 => k(11, 11, 32, 0, false),
        105 => k(12, 12, 128, 0, true),
        // speculative-order FD3 (same semantics as 91/92/94/95/98/99/100)
        106 | 111 => k(12, 12, 128, 1, false),
        107 | 112 => k(12, 12, 128, 2, false),
        108 => k(11, 11, 32, 1, false),
        109 => k(11, 11, 32, 2, false),
        110 => k(10, 10, 8, 2, false),
        // Measurement-only floors (no tail; not a verification): the CPU side
        // only checks the case count, see jumpbench.
        126 | 127 => k(12, 12, 16, 0, false),
        125 => k(10, 12, 16, 0, false),
        _ => None,
    }
}

/// Coarse descent test of the optimized jump kernels: for 2^32 <= n < 2^96,
/// (x >> 32) < (n >> 32); otherwise x < n.  Either implies x < n.
#[inline(always)]
pub fn descends_coarse(x: u128, base: u128) -> bool {
    if base >= 1u128 << 32 && base < 1u128 << 96 {
        (x >> 32) < (base >> 32)
    } else {
        x < base
    }
}

pub struct TailConfig {
    pub kind: TailKind,
    pub table: Option<JumpTable>,
    pub dip: Option<crate::jdip::DipTable>,
}

/// Production window rules (pm_inf + mix(3^13)), as the pipeline's bitvectors.
pub fn production_window_rules() -> &'static crate::lookahead::WindowRules {
    static WR: OnceLock<crate::lookahead::WindowRules> = OnceLock::new();
    WR.get_or_init(|| crate::lookahead::WindowRules {
        pm_inf: true,
        mix: Some(std::sync::Arc::new(crate::rules::MixTables::build(13))),
    })
}

impl TailConfig {
    pub fn new(kind: TailKind) -> Self {
        Self::make(kind, true)
    }

    /// As `new`, building any dip table on the calling thread only (no rayon).
    pub fn new_seq(kind: TailKind) -> Self {
        Self::make(kind, false)
    }

    fn make(kind: TailKind, parallel: bool) -> Self {
        let table = match kind {
            TailKind::Jump { w, .. } | TailKind::JumpPrefix { w, .. } | TailKind::JumpCap { w, .. }
            | TailKind::JumpDip { w, .. } => Some(JumpTable::new(w)),
            _ => None,
        };
        let dip = match kind {
            TailKind::JumpDip { wd, dq, desc, .. } => Some(if parallel {
                crate::jdip::DipTable::build(wd, production_window_rules(), dq, desc)
            } else {
                crate::jdip::DipTable::build_seq(wd, production_window_rules(), dq, desc)
            }),
            _ => None,
        };
        TailConfig { kind, table, dip }
    }
}

/// CPU semantics of each experimental CUDA variant code (cuda/xtail.cuh).
pub fn xv_kind(xv: u32) -> Option<TailKind> {
    let j = |w, s| Some(TailKind::Jump { w, skip_window: s, coarse: false });
    let o = |w, s| Some(TailKind::Jump { w, skip_window: s, coarse: true });
    match xv {
        0..=5 => Some(TailKind::Base),
        6 | 7 => Some(TailKind::Cap(5)),
        16 | 17 | 29 => j(10, false),
        18 => j(10, true),
        19 => j(8, false),
        20 => j(11, false),
        21 | 24 => j(12, false),
        22 | 30 => j(13, false),
        23 => j(14, false),
        25 => j(13, true),
        26 | 31 => j(12, true),
        27 => j(11, true),
        28 => j(9, false),
        32 | 41 | 44 => o(10, false),
        33 | 42 | 47 => o(11, false),
        34 | 35 | 43 => o(12, false),
        36 => o(10, true),
        37 => o(11, true),
        38 | 39 => o(12, true),
        40 | 45 => o(13, false),
        46 => o(13, true),
        48 => Some(TailKind::Base),
        49 => o(10, false),
        50 | 55 => o(11, false),
        51 | 52 | 53 => o(12, false),
        54 => o(13, false),
        56 | 57 | 60 | 62 => Some(TailKind::JumpPrefix { w: 12, prefix: 2 }),
        58 | 61 => Some(TailKind::JumpPrefix { w: 11, prefix: 2 }),
        59 => Some(TailKind::JumpPrefix { w: 10, prefix: 2 }),
        64..=127 => jd_kind(xv),
        _ => None,
    }
}

/// CPU semantics of tail code `code` (cached per code; built sequentially,
/// so first use never nests a rayon job inside a lock).  c3-hyg: this is the
/// only selector `tail::endpoint_work` uses, with the code taken from the
/// run's `TailParams::codes`; there is no environment fallback.
pub fn tail_config_code(code: u32) -> &'static TailConfig {
    static CFG: [OnceLock<TailConfig>; 128] = [const { OnceLock::new() }; 128];
    let slot = CFG.get(code as usize).unwrap_or_else(|| panic!("tail code {code} out of range"));
    slot.get_or_init(|| TailConfig::new_seq(xv_kind(code).unwrap_or_else(|| panic!("unknown tail code {code}"))))
}

/// The original loop (code 0).
pub fn tail_config() -> &'static TailConfig {
    tail_config_code(0)
}

impl TailConfig {
    /// Descent checks skipped at the start of each case for a class window `b`.
    pub fn skip(&self, b: u32) -> u32 {
        match self.kind {
            TailKind::Jump { w, skip_window: true, .. } => b / w,
            TailKind::JumpPrefix { prefix, .. } => prefix,
            _ => 0,
        }
    }

    /// One case: `x = T^M(n)`. Returns (hash, iters, flagged) in this
    /// variant's own counting convention.
    #[inline]
    pub fn run(&self, base: u128, x: u128, cutoff_bits: u32, b: u32) -> (u32, u64, bool) {
        self.run_h(base, x, cutoff_bits, b, i32::MIN)
    }

    /// As `run`, with the case's bitvector cutoff `h0 = 485 F - 306 M` (needed by JumpDip).
    #[inline]
    pub fn run_h(&self, base: u128, x: u128, cutoff_bits: u32, b: u32, h0: i32) -> (u32, u64, bool) {
        match self.kind {
            TailKind::JumpDip { w, ka, .. } => {
                assert!(h0 != i32::MIN, "JumpDip needs the class cutoff H0");
                let k = growth_bits(w);
                assert!(cutoff_bits >= 96 + k && cutoff_bits <= 128);
                crate::jdip::jump_dip_until_k(base, x, self.table.as_ref().unwrap(), self.dip.as_ref().unwrap(), h0,
                                              1u128 << (cutoff_bits - k), ka)
            }
            TailKind::Base => step_until_below(base, x + 1, 1u128 << cutoff_bits),
            TailKind::Cap(cap) => step_until_below_cap(base, x + 1, 1u128 << cutoff_bits, cap),
            TailKind::Jump { w, coarse, .. } => {
                let k = growth_bits(w);
                assert!(cutoff_bits >= 96 + k && cutoff_bits <= 128);
                jump_until(base, x, self.table.as_ref().unwrap(), 1u128 << (cutoff_bits - k), self.skip(b), coarse)
            }
            TailKind::JumpPrefix { w, prefix } => {
                let k = growth_bits(w);
                assert!(cutoff_bits >= 96 + k && cutoff_bits <= 128);
                jump_until(base, x, self.table.as_ref().unwrap(), 1u128 << (cutoff_bits - k), prefix, true)
            }
            TailKind::JumpCap { w, cap } => {
                let k = growth_bits(w);
                assert!(cutoff_bits >= 96 + k && cutoff_bits <= 128);
                jump_until_cap(base, x, self.table.as_ref().unwrap(), 1u128 << (cutoff_bits - k), cap)
            }
        }
    }
}

/// Strong 64-bit mixer (splitmix64 finalizer), shared with cuda/wq_jump.cuh.
#[inline]
pub fn mix64(mut a: u64) -> u64 {
    a ^= a >> 30;
    a = a.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    a ^= a >> 27;
    a = a.wrapping_mul(0x94D0_49BB_1331_11EB);
    a ^= a >> 31;
    a
}

/// Per-case audit digest of (n, the case's jumps, the case's XOR hash), as the
/// warp-queue DIG layouts compute it on the GPU.  Summed (mod 2^64) and XORed
/// over cases, it detects dropped, duplicated or exchanged cases that the
/// order-independent totals and the XOR hash cannot.
#[inline]
pub fn case_digest(base: u128, jumps: u32, hash: u32) -> u64 {
    let lo = base as u64;
    let hi = (base >> 64) as u64;
    mix64(lo ^ mix64(hi ^ mix64(((jumps as u64) << 32) | hash as u64)))
}

/// CPU reference of the warp-queue capped test tail (timing only): the coarse
/// jump tail of `jump_until`, except that a case also stops after `cap`
/// jumps.  The guard is evaluated after each jump for the next one (the same
/// condition `jump_until` tests at the top of its loop), before the cap.
pub fn jump_until_cap(base: u128, mut x: u128, tab: &JumpTable, flag_at: u128, cap: u32) -> (u32, u64, bool) {
    if x >= flag_at {
        return (0, 0, true);
    }
    let mut hash = 0u32;
    let mut jumps = 0u64;
    if cap == 0 {
        return (0, 0, false);
    }
    loop {
        x = tab.jump(x);
        hash ^= x as u32;
        jumps += 1;
        if descends_coarse(x, base) {
            return (hash, jumps, false);
        }
        if x >= flag_at {
            return (hash, jumps, true);
        }
        if jumps == cap as u64 {
            return (hash, jumps, false);
        }
    }
}

/// Every case of an endpoint exactly as `tail::endpoint_work` enumerates them:
/// `f(base, x)` with `x = T^M(base)`; starts that do not fit are reported via
/// `wide(base)` instead.
#[inline]
pub fn for_each_case(la: &Lookahead, tp: &TailParams, nd: &Node, mut f: impl FnMut(u128, u128), mut wide: impl FnMut(u128)) {
    let direct = !la.has_arrays();
    let arr = if direct { None } else { la.array(nd.f, nd.ne) };
    let start = arr.map_or(0, |i| la.start(i, nd.m));
    let r9 = crate::arith::mod_pow3(nd.n0, 2) as usize;
    let p3 = POW3[nd.f as usize];
    let (w_lo, w_hi) = tp.screen.word_range(la.a);
    let rows = la.screen_rows(&tp.screen, la.bigindex(nd.f));
    let (row, phase) = rows.map_or((0, 0), |rs| (rs.row_index(nd.n0), rs.phase(nd.n0)));
    for word in w_lo..w_hi {
        let mut t = match rows {
            Some(rs) => rs.word64(row, phase, word) & la.bv_word(arr, start, word),
            None => la.completion_word(arr, start, r9, word),
        };
        while t != 0 {
            let h = (64 * word) as u128 + t.trailing_zeros() as u128;
            t &= t - 1;
            let Some(x) = p3.checked_mul(h).and_then(|v| v.checked_add(nd.m)).filter(|&x| x < 1u128 << 123) else {
                wide(nd.n0 + (h << la.m));
                continue;
            };
            if direct && !la.keeps(nd.f, nd.ne, (x + 1) as u64) {
                continue;
            }
            if tp.tail_oee {
                if nd.oee == 2 && x & 1 == 0 {
                    continue;
                }
                if nd.oee == 1 && x & 3 == 0 {
                    continue;
                }
            }
            f(nd.n0 + (h << la.m), x);
        }
    }
}

/// Per-case trajectory structure (task 1 statistics).
#[derive(Clone, Debug, Default)]
pub struct CaseProfile {
    /// T-steps from T^M(n) until the first value < n.
    pub first_desc: u32,
    /// Double iterations and T-steps executed by the baseline loop.
    pub base_iters: u32,
    pub base_steps: u32,
    /// Bit length of the start x = T^M(n), of n, and of the trajectory maximum
    /// up to the baseline's stopping point.
    pub start_bits: u32,
    pub base_bits: u32,
    pub max_bits: u32,
    /// T-step index (from T^M) of that maximum.
    pub argmax: u32,
    /// The baseline would flag this case (some odd-run end >= 2^120).
    pub over: bool,
}

#[inline]
fn bitlen(x: u128) -> u32 {
    128 - x.leading_zeros()
}

pub fn profile_case(base: u128, x0: u128) -> CaseProfile {
    // Baseline loop, counting T-steps: an odd run of a steps, then b even steps.
    let mut p = CaseProfile { start_bits: bitlen(x0), base_bits: bitlen(base), ..Default::default() };
    let mut number = x0 + 1;
    let mut over = false;
    let mut steps = 0u32;
    let mut iters = 0u32;
    'outer: loop {
        for _ in 0..2 {
            let a = ((number as u32) | (1 << 8)).trailing_zeros();
            number = POW3[a as usize] * (number >> a);
            steps += a;
            if number >= 1u128 << 120 {
                over = true;
            }
            number -= 1;
            let b = ((number as u32) | (1 << 31)).trailing_zeros();
            number = (number >> b) + 1;
            steps += b;
        }
        iters += 1;
        if number < base || over || iters > 1_000_000 {
            break 'outer;
        }
    }
    p.base_iters = iters;
    p.base_steps = steps;
    p.over = over;
    if over {
        return p;
    }
    // Exact step-by-step trajectory up to the baseline's stopping point.
    let mut x = x0;
    let mut mx = x0;
    let mut first = None;
    for s in 1..=steps {
        x = if x & 1 == 0 { x / 2 } else { x + x / 2 + 1 };
        if x > mx {
            mx = x;
            p.argmax = s;
        }
        if first.is_none() && x < base {
            first = Some(s);
        }
    }
    p.first_desc = first.unwrap_or(steps);
    p.max_bits = bitlen(mx);
    p
}

/// Jumps executed by the W-jump loop (no flag guard; statistics only).
pub fn jumps_until_below(base: u128, mut x: u128, tab: &JumpTable, skip: u32) -> u32 {
    let mut j = 0u32;
    loop {
        x = tab.jump(x);
        j += 1;
        if j > skip && x < base {
            return j;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::t_step;

    #[test]
    fn table_matches_direct_iteration() {
        let mut rng = 0x9E37_79B9_7F4A_7C15_D1B5_4A32_D192_ED03u128;
        for w in [8, 10, 11, 12, 13, 14, 16] {
            let tab = JumpTable::new(w);
            for i in 0..20_000u32 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let x = match i % 4 {
                    0 => rng >> 8,
                    1 => rng >> 40,
                    2 => (rng >> 60) << 20,
                    _ => (i as u128) % (1u128 << w),
                };
                let mut y = x;
                for _ in 0..w {
                    y = t_step(y);
                }
                assert_eq!(tab.jump(x), y, "w={w} x={x}");
            }
        }
    }

    #[test]
    fn growth_bound() {
        for w in 4..=16 {
            let k = growth_bits(w);
            assert!((1u128 << (k + w)) >= POW3[w as usize]);
            assert!((1u128 << (k - 1 + w)) < POW3[w as usize]);
            // Worst case: all-odd residues grow by exactly (3/2)^W on z + 1.
            let tab = JumpTable::new(w);
            let lim = 1u128 << (100 - k);
            let x = lim - 1;
            assert!(tab.jump(x) < 1u128 << 100);
            let mut y = x;
            for _ in 0..w {
                y = t_step(y);
                assert!(y < 1u128 << 100);
            }
        }
    }

    #[test]
    fn jump_reference_terminates_below_and_flags_superset() {
        let tab = JumpTable::new(10);
        let mut rng = 0x6182_da73_51ee_bbf4_9037_8ac1_909d_733bu128;
        let mut cases = 0;
        for _ in 0..20_000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let base = (1u128 << 71) | (rng & ((1u128 << 71) - 1)) | 1;
            let mut x = base;
            for _ in 0..62 {
                x = t_step(x);
            }
            if x + 1 < base {
                continue;
            }
            cases += 1;
            for cutoff in [97u32 + 6, 110, 120] {
                let (_, jumps, flagged) = jump_until_below(base, x, &tab, 1u128 << (cutoff - growth_bits(10)), 0);
                // Independent check: replay the jumps step by step.
                let mut y = x;
                let mut peak = x;
                for _ in 0..jumps * 10 {
                    y = t_step(y);
                    peak = peak.max(y);
                }
                if !flagged {
                    assert!(y < base);
                    assert!(peak < 1u128 << cutoff);
                }
                let (_, _, bflag) = step_until_below(base, x + 1, 1u128 << cutoff);
                if bflag {
                    assert!(flagged, "baseline flag must be covered");
                }
            }
        }
        assert!(cases > 300);
    }
}

#[cfg(test)]
mod wq_tests {
    use super::*;
    use crate::arith::t_step;

    #[test]
    fn capped_reference_is_jump_until_up_to_the_cap() {
        let tab = JumpTable::new(12);
        let flag_at = 1u128 << (120 - growth_bits(12));
        let mut rng = 0x2545_F491_4F6C_DD1D_8E21_3D2F_9B1A_7C55u128;
        for _ in 0..5_000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let base = (1u128 << 71) | (rng & ((1u128 << 71) - 1)) | 1;
            let mut x = base;
            for _ in 0..62 {
                x = t_step(x);
            }
            let full = jump_until(base, x, &tab, flag_at, 0, true);
            for cap in [1u32, 2, 3, 5, 1000] {
                let c = jump_until_cap(base, x, &tab, flag_at, cap);
                if full.1 <= cap as u64 {
                    assert_eq!(c, full, "cap {cap} at or above the case length");
                } else {
                    assert_eq!(c.1, cap as u64);
                    assert!(!c.2);
                }
            }
        }
    }

    #[test]
    fn case_digest_separates_fields() {
        let a = case_digest(5, 1, 2);
        assert_ne!(a, case_digest(5, 2, 1));
        assert_ne!(a, case_digest(6, 1, 2));
        assert_ne!(a, case_digest(5 + (1u128 << 64), 1, 2));
        assert_eq!(mix64(0), 0);
    }
}
