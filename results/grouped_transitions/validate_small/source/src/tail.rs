//! Steps 2+3 on the CPU: enumerate the surviving completions of an endpoint
//! and iterate each until it falls below its start.
//!
//! The stepping and the checksum replicate the reference CUDA kernel
//! bit-for-bit (Barina's alternating odd-run / even-run loop on `T(n) + 1`,
//! XOR of the low 32 bits after every double iteration), so a CPU run and a
//! GPU run over the same cases produce the same hash.

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
}

impl Default for TailParams {
    fn default() -> Self {
        TailParams { cutoff_bits: 120, tail_oee: true }
    }
}

/// Work counters, kept per bitvector class (index `num_bv` = no bitvector).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TailStats {
    pub endpoints: Vec<u64>,
    pub cases: Vec<u64>,
    /// Double iterations (one odd run + one even run, twice) summed over cases.
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
    let mut w = EndpointWork::default();
    for word in 0..la.mod9_words {
        let mut t = la.completion_word(arr, start, r9, word);
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
            let (hash, iters, flagged) = step_until_below(base, x + 1, cutoff);
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
