//! c-tail: exact W-step jump tails with a window-dip certificate tested at
//! every jump boundary (CPU reference for cuda/jdip.cuh).
//!
//! Convention.  The jump state is the actual value `x = T^j(n)` (not `x + 1`).
//! At a jump boundary the lane is a member of the step-1 class at depth
//! `M' = M + (steps so far)` with `F'` odd steps, and the per-lane integer
//! `H = 485 F' - 306 M'` is that depth's bitvector cutoff (`H0 = 485 F - 306 M`
//! for the endpoint's class).  The next `Wd` parity steps of `x` are fixed by
//! `x mod 2^Wd`, equivalently by the "+1" pattern `(x + 1) mod 2^Wd` used by
//! the bitvector construction, so
//!     `dip[r] = window_dip_ext((r + 1) mod 2^Wd, Wd, START_UNKNOWN, rules)`
//! (the project's own window construction: descent, pm1, OEE, pm_inf, mix13)
//! is the dip of the window that starts at the boundary, and `dip > H` is the
//! bitvector exclusion criterion applied at depth `M'` (ALGORITHM.md section
//! 4).  It certifies that `n` either falls below `n` or merges with a positive
//! partner `p < n` within `Wd` steps.  A jump by `W` steps adds
//! `485 f(r) - 306 W` to `H`, where `f(r)` is the jump's odd-step count.
//!
//! Precondition of the dip theorem: every odd iterate before the witness is
//! >= 99781, or the trajectory has already descended.  It holds for
//! `n >= 99781`.  The GPU and this reference apply the dip test only when
//! `2^32 <= n < 2^96` (the jump kernels' "small base" split); other cases use
//! the plain jump loop with the exact descent test.
//!
//! Semantics of one case (`x = T^M(n)` on entry), identical on CPU and GPU:
//! ```text
//! loop at each boundary:
//!     if x >= 2^(cutoff - K):  flag the case and stop      (2^K >= 1.5^W)
//!     if dip[x mod 2^Wd] > H:  stop (certified)
//!     H += 485 f(x mod 2^W) - 306 W;  x = T^W(x);  hash ^= x mod 2^32;  jumps += 1
//! ```
//! Iterations are jumps; the hash is the XOR of the low 32 bits after every
//! jump, as in `cv::jump::jump_until`.

use crate::jump::{growth_bits, jump_until, JumpTable};
use crate::lookahead::{window_dip_ext, WindowRules, START_UNKNOWN};
use rayon::prelude::*;

/// Window dips indexed by `x mod 2^wd` (jump convention).
pub struct DipTable {
    pub wd: u32,
    /// Quantum: stored dips are `q * min(cap, floor(dip / q))` (q = 1: exact),
    /// `cap = quanta_cap(wd, q)` the device format's capacity.
    pub q: u32,
    pub dip: Vec<u16>,
}

/// FD3 keys: the dip of the jump's own window embedded in the jump entry
/// (cuda/jdip.cuh JD_EMB / JD_EMBS, codes 90..112): (W, Q) = (12, 128),
/// (11, 32), (10, 8).
pub fn fd3_key(wd: u32, q: u32) -> bool {
    matches!((wd, q), (12, 128) | (11, 32) | (10, 8))
}

/// Largest storable quantum `floor(dip / q)` of a dip table (identical to
/// cuda/jdip.cuh `jd_quanta_cap`): the FD3 field of 25 - DB bits (31 / 127 /
/// 511 quanta for W = 12 / 11 / 10), a u8 image (255) for the other q > 1
/// tables, u16 for exact tables.  c3-hyg: the W10 FD3 tables (codes 96..98,
/// 110) used to be clamped at the u8 image's 255 quanta although their field
/// holds 511; dips above 2040 are now stored as q floor(dip / q).
pub fn quanta_cap(wd: u32, q: u32) -> u32 {
    if fd3_key(wd, q) {
        let db = match wd { 12 => 20, 11 => 18, _ => 16 };
        (1 << (25 - db)) - 1
    } else if q == 1 {
        u16::MAX as u32
    } else {
        255
    }
}

/// Pure descent dip of the `wd` steps of `x` (no merge credits): the largest
/// `306 b - 179 a` over prefixes with `a` odd and `b` even steps.
pub fn descent_dip(r: u64, wd: u32) -> i32 {
    let (mut x, mut cur, mut max) = (r, 0i32, 0i32);
    for _ in 0..wd {
        if x & 1 == 1 {
            cur -= 179;
            x = x + x / 2 + 1;
        } else {
            cur += 306;
            x /= 2;
        }
        max = max.max(cur);
    }
    max
}

impl DipTable {
    /// Production window dips (`desc = false`) or pure descent dips, stored as
    /// `q * min(quanta_cap(wd, q), floor(dip / q))` for `q > 1` and exactly
    /// for `q = 1`.
    pub fn build(wd: u32, wr: &WindowRules, q: u32, desc: bool) -> Self {
        assert!((1..=26).contains(&wd) && q >= 1);
        let mask = (1u64 << wd) - 1;
        let cap = quanta_cap(wd, q);
        let dip: Vec<u16> = (0..(1u64 << wd))
            .into_par_iter()
            .map(|r| {
                let d = if desc { descent_dip(r, wd) } else { window_dip_ext((r + 1) & mask, wd, START_UNKNOWN, wr) };
                assert!((0..65536).contains(&d), "dip out of range");
                let d = if q == 1 { d as u32 } else { (d as u32 / q).min(cap) * q };
                d as u16
            })
            .collect();
        DipTable { wd, q, dip }
    }

    pub fn new(wd: u32, wr: &WindowRules, q: u32) -> Self {
        Self::build(wd, wr, q, false)
    }

    /// As `build`, on the calling thread only (no rayon).  Used by the GPU
    /// upload path, which can run on the pipeline's GPU thread while the rayon
    /// workers are blocked feeding it (a parallel build there deadlocks).
    pub fn build_seq(wd: u32, wr: &WindowRules, q: u32, desc: bool) -> Self {
        assert!((1..=26).contains(&wd) && q >= 1);
        let mask = (1u64 << wd) - 1;
        let cap = quanta_cap(wd, q);
        let dip: Vec<u16> = (0..(1u64 << wd))
            .map(|r| {
                let d = if desc { descent_dip(r, wd) } else { window_dip_ext((r + 1) & mask, wd, START_UNKNOWN, wr) };
                assert!((0..65536).contains(&d), "dip out of range");
                let d = if q == 1 { d as u32 } else { (d as u32 / q).min(cap) * q };
                d as u16
            })
            .collect();
        DipTable { wd, q, dip }
    }

    /// Device upload image (cuda/jdip.cuh `cv_jd_upload`): `dip / q` per
    /// entry as u16, at most `quanta_cap(wd, q)`.
    pub fn quanta(&self) -> Vec<u16> {
        self.dip.iter().map(|&d| (d as u32 / self.q) as u16).collect()
    }

    #[inline(always)]
    pub fn get(&self, x: u128) -> i32 {
        self.dip[(x as usize) & ((1usize << self.wd) - 1)] as i32
    }

    pub fn max(&self) -> u32 {
        self.dip.iter().map(|&d| d as u32).max().unwrap_or(0)
    }
}

/// `485 f - 306 W` for each jump residue.
pub fn h_increments(jt: &JumpTable) -> Vec<i32> {
    jt.odd.iter().map(|&f| 485 * f as i32 - 306 * jt.w as i32).collect()
}

/// Is the dip test allowed for this base?  (Matches the kernels' small-base split.)
#[inline(always)]
pub fn dip_eligible(base: u128) -> bool {
    base >= 1u128 << 32 && base < 1u128 << 96
}

/// One case with the per-boundary dip test.  Returns (hash, jumps, flagged).
#[inline]
pub fn jump_dip_until(base: u128, mut x: u128, jt: &JumpTable, dt: &DipTable, h0: i32, flag_at: u128) -> (u32, u64, bool) {
    if !dip_eligible(base) {
        return jump_until(base, x, jt, flag_at, 0, true);
    }
    let mut hash = 0u32;
    let mut jumps = 0u64;
    let mut h = h0;
    let mj = (1usize << jt.w) - 1;
    loop {
        if x >= flag_at {
            return (hash, jumps, true);
        }
        if dt.get(x) > h {
            return (hash, jumps, false);
        }
        let r = (x as usize) & mj;
        h += 485 * jt.odd[r] as i32 - 306 * jt.w as i32;
        x = jt.jump(x);
        hash ^= x as u32;
        jumps += 1;
    }
}

/// Same, also returning the stop state `(x, H, jumps)` for audits.
pub fn jump_dip_trace(base: u128, mut x: u128, jt: &JumpTable, dt: &DipTable, h0: i32, flag_at: u128)
    -> (u32, u64, bool, u128, i32) {
    let mut hash = 0u32;
    let mut jumps = 0u64;
    let mut h = h0;
    let mj = (1usize << jt.w) - 1;
    loop {
        if x >= flag_at {
            return (hash, jumps, true, x, h);
        }
        if dt.get(x) > h {
            return (hash, jumps, false, x, h);
        }
        let r = (x as usize) & mj;
        h += 485 * jt.odd[r] as i32 - 306 * jt.w as i32;
        x = jt.jump(x);
        hash ^= x as u32;
        jumps += 1;
    }
}

/// One case with the window `ka` jumps ahead tested at every boundary
/// (bit-exact with cuda/jdip.cuh).  `ka = 0` is `jump_dip_until`.  The index
/// of the window `ka` jumps ahead is the exact low part of `T^(ka W)(x)`,
/// computed here with wrapping 64-bit mini jumps (exact mod 2^(64 - ka W)).
#[inline]
pub fn jump_dip_until_k(base: u128, mut x: u128, jt: &JumpTable, dt: &DipTable, h0: i32, flag_at: u128, ka: u32)
    -> (u32, u64, bool) {
    if !dip_eligible(base) {
        return jump_until(base, x, jt, flag_at, 0, true);
    }
    assert!(ka <= 2 && dt.wd + ka * jt.w <= 64);
    let mut hash = 0u32;
    let mut jumps = 0u64;
    let mut h = h0;
    let w = jt.w;
    let mj = (1u64 << w) - 1;
    let g = |r: u64| 485 * jt.odd[r as usize] as i32 - 306 * w as i32;
    loop {
        if x >= flag_at {
            return (hash, jumps, true);
        }
        let mut y = x as u64;
        let mut hh = h;
        for _ in 0..ka {
            let r = y & mj;
            hh += g(r);
            y = (jt.mult[r as usize] as u64).wrapping_mul(y >> w).wrapping_add(jt.add[r as usize] as u64);
        }
        if dt.get(y as u128) > hh {
            return (hash, jumps, false);
        }
        h += g(x as u64 & mj);
        x = jt.jump(x);
        hash ^= x as u32;
        jumps += 1;
    }
}

/// Reference for a variant: `cutoff_bits` must be >= 96 + K.
pub fn run_case(base: u128, x: u128, jt: &JumpTable, dt: &DipTable, h0: i32, cutoff_bits: u32) -> (u32, u64, bool) {
    let k = growth_bits(jt.w);
    assert!(cutoff_bits >= 96 + k && cutoff_bits <= 128);
    jump_dip_until(base, x, jt, dt, h0, 1u128 << (cutoff_bits - k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::t_step;
    use crate::rules::MixTables;
    use std::sync::Arc;

    #[test]
    fn fd3_tables_use_the_field_width_not_a_u8_image() {
        // c3-hyg regression: W10 FD3 (codes 96..98, 110; q = 8) holds 511
        // quanta.  Round 2 clamped at 255 quanta (2040 units).
        let wr = WindowRules { pm_inf: true, mix: Some(Arc::new(MixTables::build(13))) };
        assert_eq!((quanta_cap(12, 128), quanta_cap(11, 32), quanta_cap(10, 8)), (31, 127, 511));
        assert_eq!((quanta_cap(12, 16), quanta_cap(20, 32), quanta_cap(12, 1)), (255, 255, 65535));
        for (wd, q) in [(10u32, 8u32), (11, 32), (12, 128)] {
            let t = DipTable::build_seq(wd, &wr, q, false);
            let mut above = 0;
            for r in 0..(1u64 << wd) {
                let exact = window_dip_ext((r + 1) & ((1 << wd) - 1), wd, START_UNKNOWN, &wr) as u32;
                assert!(exact / q <= quanta_cap(wd, q), "W{wd}: dip {exact} exceeds the FD3 field");
                assert_eq!(t.dip[r as usize] as u32, q * (exact / q), "W{wd} r {r}: stored dip is q floor(dip/q)");
                above += (exact / q > 255) as u32;
            }
            assert!(t.quanta().iter().all(|&x| x as u32 <= quanta_cap(wd, q)));
            if wd == 10 {
                assert!(above > 0, "fixture: some W10 dips exceed 255 quanta");
                assert!(t.max() > 2040);
            }
            assert_eq!(DipTable::build(wd, &wr, q, false).dip, t.dip, "parallel and sequential builds agree");
        }
    }

    #[test]
    fn dip_table_is_the_plus_one_window_dip() {
        let wr = WindowRules { pm_inf: true, mix: Some(Arc::new(MixTables::build(13))) };
        let dt = DipTable::new(10, &wr, 1);
        for r in 0..1024u64 {
            let p = (r + 1) & 1023;
            assert_eq!(dt.dip[r as usize] as i32, window_dip_ext(p, 10, START_UNKNOWN, &wr));
        }
    }

    /// Some p with 1 <= p < n and T^s(p) = y (s >= 1, at most `omax` odd steps),
    /// pruning branches that cannot reach below n (as cv's jdaudit).
    fn partner(y: u128, n: u128, omax: u32, smax: u32) -> bool {
        let mut stack = vec![(y, 0u32, 0u32)];
        while let Some((v, s, o)) = stack.pop() {
            if s >= 1 && v < n && v >= 1 {
                return true;
            }
            if s == smax {
                continue;
            }
            let r = (omax - o).min(smax - s);
            match (v.checked_mul(1u128 << r.min(100)), crate::arith::POW3[r as usize].checked_mul(n + 1)) {
                (Some(l), Some(rr)) if l >= rr => continue,
                (None, _) => continue,
                _ => {}
            }
            if let Some(e) = v.checked_mul(2) {
                stack.push((e, s + 1, o));
            }
            if o < omax && v % 3 == 2 {
                stack.push(((2 * v - 1) / 3, s + 1, o + 1));
            }
        }
        false
    }

    #[test]
    fn dip_stop_has_a_descent_or_partner_witness() {
        // Default rules: descent + pm1 + OEE credits.  Every stop must have a
        // value below n, or a partner p < n, within Wd steps of the stop state.
        let wr = WindowRules::default();
        let jt = JumpTable::new(12);
        let dt = DipTable::new(12, &wr, 1);
        let mut rng = 0x6182_da73_51ee_bbf4_9037_8ac1_909d_733bu128;
        let mut cases = 0;
        for _ in 0..20_000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let base = (1u128 << 71) | (rng & ((1u128 << 71) - 1)) | 1;
            let mut x = base;
            let mut f = 0;
            for _ in 0..62 {
                if x & 1 == 1 {
                    f += 1;
                }
                x = t_step(x);
            }
            if x + 1 < base {
                continue;
            }
            cases += 1;
            let h0 = 485 * f - 306 * 62;
            let (_, _, flagged, xs, _) = jump_dip_trace(base, x, &jt, &dt, h0, 1u128 << 113);
            assert!(!flagged);
            let mut y = xs;
            let mut ok = y < base || partner(y, base, 13, 26);
            for _ in 0..12 {
                y = t_step(y);
                ok |= y < base || partner(y, base, 13, 26);
            }
            assert!(ok, "no witness within the window after a dip stop: base {base} x {x} stop {xs}");
        }
        assert!(cases > 300);
    }
}
