//! Steps 2+3 on the CPU: enumerate the surviving completions of an endpoint
//! and iterate each until it falls below its start.
//!
//! The stepping and the checksum replicate the CUDA kernels bit-for-bit, so
//! a CPU run and a GPU run over the same cases produce the same hash.  With
//! the default `TailParams::codes` that is the reference kernel's loop
//! (Barina's alternating odd-run / even-run loop on `T(n) + 1`, XOR of the low
//! 32 bits after every double iteration); a per-class tail code selects the
//! jump / jd tails of `crate::jump` and `crate::jdip` instead (c3-hyg: the
//! codes come from the run's own `TailParams`, the same value the GPU
//! launches use; see `crate::runspec`).

use crate::arith::{mod_pow3, POW3};
use crate::lookahead::Lookahead;
use crate::tree::{Node, Sink};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug)]
pub struct TailParams {
    /// Trajectories reaching 2^cutoff_bits are flagged for big-integer
    /// re-checking and abandoned (as in the reference code).
    pub cutoff_bits: u32,
    /// Apply the OEE state carried into the window (two cheap parity tests;
    /// Angeltveit's CPU `search2`).  Sound; removes ~0.3% of cases.
    pub tail_oee: bool,
    /// Experimental completion screening (c-screen); the default keeps the
    /// original mod-9 mask over every completion.
    pub screen: Screen,
    /// c3-hyg: per-class tail codes, the single source of the per-case
    /// semantics for the CPU reference and the GPU (the default is the
    /// original loop on every class).
    pub codes: crate::runspec::TailCodes,
    /// c3-hyg: per-class warp-queue scheduling (GPU only, no effect on the
    /// work tuple); recorded in reports, and `GpuTuning::wq` must equal it.
    pub schedule: crate::runspec::WqSchedule,
}

impl Default for TailParams {
    fn default() -> Self {
        TailParams {
            cutoff_bits: 120,
            tail_oee: true,
            screen: Screen::default(),
            codes: crate::runspec::TailCodes::original(),
            schedule: crate::runspec::WqSchedule::none(),
        }
    }
}

/// Explicit per-run completion screening, applied identically by the CPU
/// reference (`endpoint_work`, `jump::for_each_case`) and the GPU screen
/// kernels (`cuda/screen.cuh`).
///
/// * `rows`: replace the mod-9 mask by the (12, 9) row table
///   (`crate::rowsieve`); the run's `Lookahead` must carry the table.
/// * `[h_lo, h_hi)`: verify only completions `n = n0 + h 2^M` with `h` in this
///   range (multiples of 64; `h_hi = 0` means `2^A`).  `h_lo = 2^(A-1)` is the
///   top half, `n >= 2^(N-1)`.  A restricted run is a sound verification of
///   that range only when every smaller `n` is covered by another run or a
///   published result (strong induction).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Screen {
    pub rows: bool,
    /// With `rows`: apply the table only to bitvector classes >= `rows_from`
    /// and to the CPU-only classes beyond the last bitvector (per-class
    /// gating; light classes are faster with the mod-9 mask on the GPU).
    pub rows_from: u32,
    pub h_lo: u32,
    pub h_hi: u32,
}

impl Screen {
    pub fn is_default(&self) -> bool {
        *self == Screen::default()
    }
    /// The top half `h >= 2^(A-1)`, i.e. `n >= 2^(N-1)`.
    pub fn top(a: u32) -> Self {
        Screen { h_lo: 1 << (a - 1), ..Screen::default() }
    }
    /// The bottom half `h < 2^(A-1)`.
    pub fn bottom(a: u32) -> Self {
        Screen { h_hi: 1 << (a - 1), ..Screen::default() }
    }
    pub fn with_rows(mut self, rows: bool) -> Self {
        self.rows = rows;
        self
    }
    /// Does an endpoint of bitvector class `class` (None: CPU-only class) use the rows?
    #[inline]
    pub fn rows_for(&self, class: Option<usize>) -> bool {
        self.rows && class.map_or(true, |c| c >= self.rows_from as usize)
    }
    /// c3-hyg: the screen is usable with `2^a` completions per endpoint
    /// (the same conditions `word_range` asserts, as an error).
    pub fn check(&self, a: u32) -> anyhow::Result<()> {
        anyhow::ensure!((6..=31).contains(&a), "A = {a} outside 6..=31");
        let total = 1u32 << a;
        let hi = if self.h_hi == 0 { total } else { self.h_hi };
        anyhow::ensure!(self.h_lo % 64 == 0 && hi % 64 == 0 && self.h_lo < hi && hi <= total,
            "completion range [{}, {}) must be 64-aligned and non-empty within 2^A = {total} \
             (the top/bottom halves need A >= 7)", self.h_lo, hi);
        Ok(())
    }
    /// Range of 64-completion words `[lo, hi)` for `2^a` completions.
    pub fn word_range(&self, a: u32) -> (usize, usize) {
        let total = 1u32 << a;
        let hi = if self.h_hi == 0 { total } else { self.h_hi };
        assert!(self.h_lo % 64 == 0 && hi % 64 == 0 && self.h_lo < hi && hi <= total,
                "screen range [{}, {}) must be 64-aligned within 2^A = {total}", self.h_lo, hi);
        ((self.h_lo / 64) as usize, (hi / 64) as usize)
    }
    pub fn label(&self) -> String {
        let mut v = Vec::new();
        if self.rows { v.push(if self.rows_from > 0 { format!("rows(12,9)@c>={}", self.rows_from) } else { "rows(12,9)".to_string() }); }
        if self.h_lo != 0 || self.h_hi != 0 { v.push(format!("h[{},{})", self.h_lo, if self.h_hi == 0 { "2^A".to_string() } else { self.h_hi.to_string() })); }
        if v.is_empty() { "mod9".into() } else { v.join("+") }
    }
}

/// Work counters, kept per bitvector class (index `num_bv` = no bitvector).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TailStats {
    pub endpoints: Vec<u64>,
    pub cases: Vec<u64>,
    /// Iterations summed over cases, in each class's tail-code unit (code 0:
    /// double iterations, one odd run + one even run, twice; jump / jd codes:
    /// W-step jumps).  See `runspec::code_definition`.
    pub iters: Vec<u64>,
    pub hash: u32,
    #[serde(skip)]
    pub flagged: Vec<u128>,
}

impl TailStats {
    pub fn new(num_bv: usize) -> Self {
        TailStats {
            endpoints: vec![0; num_bv + 1],
            cases: vec![0; num_bv + 1],
            iters: vec![0; num_bv + 1],
            hash: 0,
            flagged: Vec::new(),
        }
    }
    pub fn merge(&mut self, o: &TailStats) {
        for (a, b) in self.endpoints.iter_mut().zip(&o.endpoints) {
            *a += b;
        }
        for (a, b) in self.cases.iter_mut().zip(&o.cases) {
            *a += b;
        }
        for (a, b) in self.iters.iter_mut().zip(&o.iters) {
            *a += b;
        }
        self.hash ^= o.hash;
        self.flagged.extend_from_slice(&o.flagged);
    }
    pub fn total_cases(&self) -> u64 {
        self.cases.iter().sum()
    }
    pub fn total_iters(&self) -> u64 {
        self.iters.iter().sum()
    }
    pub fn total_endpoints(&self) -> u64 {
        self.endpoints.iter().sum()
    }
}

/// Iterate `base` (with `number = T^M(base) + 1`) until `T^j(base) + 1 < base`.
/// Returns (hash, double iterations, flagged).
#[inline]
pub fn step_until_below(base: u128, mut number: u128, cutoff: u128) -> (u32, u64, bool) {
    let mut hash = 0u32;
    let mut iters = 0u64;
    let mut flagged = false;
    loop {
        for _ in 0..2 {
            let a = ((number as u32) | (1 << 8)).trailing_zeros();
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

/// A GPU tail paused after a complete double iteration, including its checksum
/// update and descent test. Values use little-endian 32-bit limbs to match the
/// CUDA queue ABI. The GPU has already counted the case and every iteration up
/// to this boundary; continuing it adds only the remaining hash and iterations.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TailContinuation {
    pub limbs: [u32; 8],
}

impl TailContinuation {
    pub fn from_values(base: u128, number: u128) -> Self {
        let mut limbs = [0; 8];
        for i in 0..4 {
            limbs[i] = (base >> (32 * i)) as u32;
            limbs[4 + i] = (number >> (32 * i)) as u32;
        }
        Self { limbs }
    }

    pub fn base(&self) -> u128 {
        self.limbs[..4].iter().rev().fold(0, |value, &limb| (value << 32) | limb as u128)
    }

    /// The shifted trajectory state `T^j(base) + 1` at the handoff boundary.
    pub fn number(&self) -> u128 {
        self.limbs[4..].iter().rev().fold(0, |value, &limb| (value << 32) | limb as u128)
    }
}

/// Finish a GPU continuation with the same cutoff and checksum convention as
/// the uninterrupted GPU tail. A flagged result requires the usual wide
/// recheck of `record.base()`; it is not an additional case.
///
/// The upper bound on `number` leaves enough headroom for an eight-step odd
/// run in 128-bit arithmetic. Reject completed or malformed records instead of
/// silently performing an extra iteration after descent.
pub fn finish_continuation(record: TailContinuation, cutoff_bits: u32) -> Result<(u32, u64, bool), &'static str> {
    if !(97..=123).contains(&cutoff_bits) {
        return Err("continuation cutoff must be between 97 and 123 bits");
    }
    let base = record.base();
    let number = record.number();
    if base < 3 {
        return Err("continuation base must be at least 3");
    }
    if number < base {
        return Err("continuation has already descended below its base");
    }
    if number >= 1u128 << 123 {
        return Err("continuation state exceeds the GPU arithmetic bound");
    }
    Ok(step_until_below(base, number, 1u128 << cutoff_bits))
}

/// Finish a flagged trajectory with 256-bit arithmetic: iterate `n` until it
/// falls below its start.  Returns the number of steps, or None if the value
/// would exceed 2^256 (never observed; such an `n` would be a new record by
/// a factor of ~2^100).
pub fn recheck_wide(n: u128) -> Option<u64> {
    let (mut hi, mut lo) = (0u128, n);
    let mut steps = 0u64;
    loop {
        if hi == 0 && lo < n {
            return Some(steps);
        }
        if lo & 1 == 0 {
            lo = (lo >> 1) | (hi << 127);
            hi >>= 1;
        } else {
            // x + x/2 + 1 on (hi, lo)
            let h_lo = (lo >> 1) | (hi << 127);
            let h_hi = hi >> 1;
            let (s1, c1) = lo.overflowing_add(h_lo);
            let (s2, c2) = s1.overflowing_add(1);
            hi = hi.checked_add(h_hi)?.checked_add(c1 as u128 + c2 as u128)?;
            lo = s2;
        }
        steps += 1;
    }
}

/// Work done by steps 2+3 for one endpoint.
#[derive(Clone, Copy, Debug, Default)]
pub struct EndpointWork {
    pub cases: u64,
    pub iters: u64,
    pub hash: u32,
    pub flagged: u32,
}

/// Steps 2+3 for one endpoint; `on_flag` receives starts that reached the cutoff.
#[inline]
pub fn endpoint_work(la: &Lookahead, tp: &TailParams, nd: &Node, mut on_flag: impl FnMut(u128)) -> EndpointWork {
    // Without stored arrays, evaluate each completion's window directly.
    let direct = !la.has_arrays();
    let arr = if direct { None } else { la.array(nd.f, nd.ne) };
    let start = arr.map_or(0, |i| la.start(i, nd.m));
    let r9 = mod_pow3(nd.n0, 2) as usize;
    let p3 = POW3[nd.f as usize];
    let cutoff = 1u128 << tp.cutoff_bits;
    // c3-hyg: this class's code from the run's own configuration (the GPU
    // launches use the same TailParams::codes; CPU-only classes: code 0).
    let class = la.bigindex(nd.f);
    let xtail = crate::jump::tail_config_code(tp.codes.code(class));
    let xb = class.map_or(la.b, |i| la.bs[i]);
    let (w_lo, w_hi) = tp.screen.word_range(la.a);
    let rows = la.screen_rows(&tp.screen, la.bigindex(nd.f));
    let (row, phase) = rows.map_or((0, 0), |rs| (rs.row_index(nd.n0), rs.phase(nd.n0)));
    let mut w = EndpointWork::default();
    for word in w_lo..w_hi {
        let mut t = match rows {
            Some(rs) => rs.word64(row, phase, word) & la.bv_word(arr, start, word),
            None => la.completion_word(arr, start, r9, word),
        };
        while t != 0 {
            let h = (64 * word) as u128 + t.trailing_zeros() as u128;
            t &= t - 1;
            // x = T^M(n).  Starts below 2^123 are stepped exactly as on the GPU
            // (the first odd run flags them if they reach the cutoff); larger
            // ones, possible only for rare high-f endpoints finished on the
            // CPU at large N, would wrap the 128-bit arithmetic: flag them.
            let Some(x) = p3.checked_mul(h).and_then(|v| v.checked_add(nd.m)).filter(|&x| x < 1u128 << 123) else {
                let base = nd.n0 + (h << la.m);
                w.cases += 1;
                w.flagged += 1;
                on_flag(base);
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
            let base = nd.n0 + (h << la.m);
            let (hash, iters, flagged) = if xtail.kind == crate::jump::TailKind::Base {
                step_until_below(base, x + 1, cutoff)
            } else {
                xtail.run_h(base, x, tp.cutoff_bits, xb, 485 * nd.f as i32 - 306 * la.m as i32)
            };
            w.hash ^= hash;
            w.cases += 1;
            w.iters += iters;
            if flagged {
                w.flagged += 1;
                on_flag(base);
            }
        }
    }
    w
}

/// Steps 2+3 for one endpoint, accumulated into `st`.
#[inline]
pub fn process_endpoint(la: &Lookahead, tp: &TailParams, nd: &Node, st: &mut TailStats) {
    let slot = la.bigindex(nd.f).unwrap_or(la.num_bv);
    let flagged = &mut st.flagged;
    let w = endpoint_work(la, tp, nd, |b| flagged.push(b));
    st.endpoints[slot] += 1;
    st.cases[slot] += w.cases;
    st.iters[slot] += w.iters;
    st.hash ^= w.hash;
}

/// Sink running steps 2+3 on the CPU.
pub struct CpuTailSink<'a> {
    pub la: &'a Lookahead,
    pub tp: TailParams,
    pub st: TailStats,
}

impl<'a> CpuTailSink<'a> {
    pub fn new(la: &'a Lookahead, tp: TailParams) -> Self {
        CpuTailSink { la, tp, st: TailStats::new(la.num_bv) }
    }
}

impl Sink for CpuTailSink<'_> {
    #[inline]
    fn endpoint(&mut self, node: &Node) {
        process_endpoint(self.la, &self.tp, node, &mut self.st);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent scalar prefix: use individual multiply/divide operations
    // instead of the production table, bit scans, and compressed odd run.
    // Return a continuation only after the checksum and descent test.
    fn scalar_prefix(base: u128, mut number: u128, cutoff_bits: u32, budget: u64)
        -> (u32, u64, bool, Option<TailContinuation>)
    {
        let cutoff = 1u128 << cutoff_bits;
        let mut hash = 0;
        let mut flagged = false;
        for iters in 1..=budget {
            for _ in 0..2 {
                for _ in 0..8 {
                    if number % 2 != 0 { break; }
                    number = (number / 2).checked_mul(3).expect("scalar odd-run overflow");
                }
                if number >= cutoff {
                    flagged = true;
                    number = 2;
                }
                number -= 1;
                for _ in 0..31 {
                    if number % 2 != 0 { break; }
                    number /= 2;
                }
                number += 1;
            }
            hash ^= number as u32;
            if number < base {
                return (hash, iters, flagged, None);
            }
        }
        (hash, budget, flagged, Some(TailContinuation::from_values(base, number)))
    }

    fn check_split(base: u128, number: u128, cutoff_bits: u32, budget: u64) -> bool {
        let expected = step_until_below(base, number, 1u128 << cutoff_bits);
        let scalar = scalar_prefix(base, number, cutoff_bits, expected.1 + 1);
        assert_eq!((scalar.0, scalar.1, scalar.2), expected);
        assert_eq!(scalar.3, None);
        let (hash, iters, flagged, continuation) = scalar_prefix(base, number, cutoff_bits, budget);
        if let Some(record) = continuation {
            assert!(!flagged, "flagged trajectories must complete before handoff");
            let remaining = finish_continuation(record, cutoff_bits).unwrap();
            assert_eq!((hash ^ remaining.0, iters + remaining.1, remaining.2), expected,
                "base={base}, number={number}, budget={budget}, cutoff={cutoff_bits}");
            true
        } else {
            assert_eq!((hash, iters, flagged), expected);
            false
        }
    }

    #[test]
    fn continuation_matches_random_n72_trajectories() {
        let budgets = [1, 2, 4, 8, 16, 32, 64, 128];
        let mut rng = 0x6182_da73_51ee_bbf4_9037_8ac1_909d_733bu128;
        let mut cases = 0;
        let mut handed_off = [0; 8];
        for _ in 0..16_384 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let base = (1u128 << 71) | (rng & ((1u128 << 71) - 1)) | 1;
            let mut x = base;
            // The actual shortcut orbit through an M=62 endpoint. Retain
            // only states still eligible for a tail rather than inventing
            // unrelated starting values and current states.
            for _ in 0..62 {
                x = if x & 1 == 0 { x / 2 } else { (3 * x + 1) / 2 };
            }
            if x + 1 < base { continue; }
            cases += 1;
            for (i, &budget) in budgets.iter().enumerate() {
                handed_off[i] += check_split(base, x + 1, 120, budget) as usize;
            }
        }
        assert!(cases >= 300, "fixture must exercise enough surviving N72 tails");
        assert!(handed_off[0] >= 150);
        assert!(handed_off[3] > 0);
    }

    #[test]
    fn continuation_preserves_forced_overflow_and_wide_states() {
        // Consecutive odd runs give a legitimate initial trajectory that
        // crosses the 97-bit cutoff after more than one double iteration.
        let base = (1u128 << 72) - 1;
        let number = base + 1;
        assert!(step_until_below(base, number, 1u128 << 97).2);
        assert!(check_split(base, number, 97, 1));
        assert!(check_split(base, number, 97, 2));
        for budget in [4, 8, 16, 32, 64, 128] {
            check_split(base, number, 97, budget);
        }

        // Both the start and carried state require more than three limbs.
        // The first double iteration succeeds; the continuation flags later.
        let wide_base = (1u128 << 112) - 1;
        assert!(check_split(wide_base, wide_base + 1, 123, 1));
        let (_, _, _, record) = scalar_prefix(wide_base, wide_base + 1, 123, 1);
        let record = record.unwrap();
        assert!(record.base() > u64::MAX as u128);
        assert!(record.number() > 1u128 << 120);
        assert!(finish_continuation(record, 123).unwrap().2);
        for budget in [1, 2, 4, 8, 16, 32, 64, 128] {
            check_split((1u128 << 100) + 3, (1u128 << 120) + 5, 123, budget);
        }
    }

    #[test]
    fn continuation_limb_abi_and_validation() {
        assert_eq!(std::mem::size_of::<TailContinuation>(), 32);
        assert_eq!(std::mem::align_of::<TailContinuation>(), 4);
        let base = 0x0011_2233_4455_6677_8899_aabb_ccdd_eeff;
        let number = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210;
        let record = TailContinuation::from_values(base, number);
        assert_eq!(record.limbs, [0xccdd_eeff, 0x8899_aabb, 0x4455_6677, 0x0011_2233,
            0x7654_3210, 0xfedc_ba98, 0x89ab_cdef, 0x0123_4567]);
        assert_eq!(record.base(), base);
        assert_eq!(record.number(), number);
        for cutoff in [0, 96, 124, 128, u32::MAX] {
            assert!(finish_continuation(record, cutoff).is_err());
        }
        assert!(finish_continuation(TailContinuation::from_values(2, 4), 120).is_err());
        assert!(finish_continuation(TailContinuation::from_values(27, 26), 120).is_err());
        assert!(finish_continuation(TailContinuation::from_values(27, 1u128 << 123), 123).is_err());
        assert!(finish_continuation(TailContinuation::from_values(27, 27), 120).is_ok());
    }

    #[test]
    fn wide_recheck_matches_narrow() {
        for n in [27u128, 703, 77671, 1_410_123_943, 8_528_817_511] {
            let mut x = n;
            let mut steps = 0;
            while x >= n {
                x = crate::arith::t_step(x);
                steps += 1;
            }
            assert_eq!(recheck_wide(n), Some(steps));
        }
    }

    #[test]
    fn low_cutoff_flags_are_rechecked() {
        // With a 40-bit cutoff, path-record starts get flagged; all finish in 256 bits.
        let (_, _, flagged) = step_until_below(77671, 77671 + 1, 1u128 << 20);
        assert!(flagged);
        assert!(recheck_wide(77671).is_some());
    }
}
