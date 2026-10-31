//! Experimental 3-adic predecessor sieve on completions (replaces the mod-9 mask).
//!
//! A completion `n` is skipped when some strictly smaller positive integer `p`
//! has a trajectory through `n` (root words) or through `T(n)` (universal
//! siblings). Root: an inverse word with `o` odd and `s` total steps,
//! `p = (2^s n - b)/3^o`, exists exactly on residues `n mod 3^o`; `2^s < 3^o`
//! gives `p < n` for every `n >= 1` because `b > 0`.
//! Siblings: every completion is odd, and `v_k = 4^k n + (4^k-1)/3` satisfies
//! `T^(2k+1)(v_k) = T(n)`. A predecessor `p` of `v_k` with `3^o > 2^(s+2k)`
//! satisfies `p < n` for all `n >= 2^s (4^k-1)/3`, which is below 2^40 here.
//! Completions with `n < 2^40` therefore never use sibling exclusions.
//!
//! `J = 2` without siblings reproduces the mod-9 mask exactly.
//!
//! Layout: bit `h` of an endpoint's mask is `Q[phase + h]`, where
//! `phase = n0 * (2^M)^-1 mod 3^J` and `Q[i] = keep(i * 2^M mod 3^J)`, stored
//! periodically for `3^J + 2^A + 128` bits. One funnel shift per word, like the
//! bitvectors.

use crate::arith::POW3;
use rayon::prelude::*;

pub struct Sieve3 {
    pub j: u32,
    pub d: u64,
    pub sib: u32,
    pub m: u32,
    pub a: u32,
    /// (2^M)^-1 mod D, and 2^64 * that mod D.
    pub ginv: u64,
    pub ginv64: u64,
    /// floor(2^64 / D) for the device Barrett reduction.
    pub minv: u64,
    /// keep[r] for r = n mod D (unpacked, for audits and statistics).
    pub keep: Vec<bool>,
    /// Q bits as u64 words.
    pub bits: Vec<u64>,
    /// Kept residues out of D.
    pub kept: u64,
}

/// Completions below this never use sibling exclusions.
pub const SIB_MIN_N: u128 = 1u128 << 40;

/// Best inverse word per residue mod 3^e as (o, s) packed `o | s << 5`, 0 = none
/// with ratio 3^o / 2^s > 1. Returns the table for level `e = j`.
pub fn best_words(j: u32) -> Vec<u16> {
    let mut prev: Vec<u16> = vec![0u16; 1];
    for e in 1..=j {
        let d = 3u64.pow(e);
        let dprev = 3u64.pow(e - 1);
        let mut cur = vec![0u16; d as usize];
        for y in 0..d {
            let r = y % 3;
            if r == 0 {
                continue;
            }
            let mut best: u16 = 0;
            let mut k = if r == 2 { 0u32 } else { 1u32 };
            loop {
                let ub = 1.584962500721156 - (k + 1) as f64 + 0.584962500721156 * (e - 1) as f64;
                if ub <= 0.0 {
                    break;
                }
                let t = (((y as u128) << (k + 1)) + d as u128 - 1) % d as u128;
                debug_assert!(t % 3 == 0);
                let sub = prev[((t / 3) as u64 % dprev) as usize];
                let (o, s) = ((sub & 31) as u32 + 1, (sub >> 5) as u32 + k + 1);
                if POW3[o as usize] > (1u128 << s) {
                    let code = (o | (s << 5)) as u16;
                    if best == 0 || ratio_gt(code, best) {
                        best = code;
                    }
                }
                k += 2;
            }
            cur[y as usize] = best;
        }
        prev = cur;
    }
    prev
}

#[inline]
fn ratio_gt(a: u16, b: u16) -> bool {
    let (oa, sa) = ((a & 31) as usize, (a >> 5) as u32);
    let (ob, sb) = ((b & 31) as usize, (b >> 5) as u32);
    (POW3[oa] << sb) > (POW3[ob] << sa)
}

fn inv_mod(a: u64, m: u64) -> u64 {
    let (mut t, mut nt) = (0i128, 1i128);
    let (mut r, mut nr) = (m as i128, a as i128);
    while nr != 0 {
        let q = r / nr;
        (t, nt) = (nt, t - q * nt);
        (r, nr) = (nr, r - q * nr);
    }
    assert_eq!(r, 1);
    (if t < 0 { t + m as i128 } else { t }) as u64
}

impl Sieve3 {
    pub fn new(j: u32, sib: u32, m: u32, a: u32) -> Self {
        assert!((2..=19).contains(&j), "sieve J must be in 2..=19");
        assert!(sib <= 6);
        // Sibling exclusions are proven only for n >= 2^40; with M >= 40 every
        // completion except h = 0 satisfies that, and h = 0 is guarded.
        assert!(sib == 0 || m >= 40, "sibling sieve requires M >= 40");
        let d = 3u64.pow(j);
        let best = best_words(j);
        let mut keep = vec![false; d as usize];
        let mut kept = 0u64;
        for r in 0..d {
            let mut excluded = best[r as usize] != 0;
            let mut pk = 1u128;
            for k in 1..=sib {
                if excluded {
                    break;
                }
                pk *= 4;
                let c = (pk - 1) / 3;
                let v = ((pk % d as u128) * r as u128 + c) % d as u128;
                let code = best[v as usize];
                if code != 0 {
                    let (o, s) = ((code & 31) as usize, (code >> 5) as u32);
                    assert!(s < 40 && ((1u128 << s) * c) < SIB_MIN_N, "sibling bound");
                    if POW3[o] > (1u128 << (s + 2 * k)) {
                        excluded = true;
                    }
                }
            }
            keep[r as usize] = !excluded;
            kept += !excluded as u64;
        }
        let g = ((1u128 << m) % d as u128) as u64;
        let ginv = inv_mod(g, d);
        let ginv64 = (((1u128 << 64) % d as u128) * ginv as u128 % d as u128) as u64;
        let minv = u64::MAX / d;
        let nbits = d as usize + (1usize << a) + 128;
        let mut bits = vec![0u64; nbits.div_ceil(64)];
        for i in 0..nbits {
            let r = ((i as u128 % d as u128) * g as u128 % d as u128) as usize;
            if keep[r] {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }
        Sieve3 { j, d, sib, m, a, ginv, ginv64, minv, keep, bits, kept }
    }

    /// Phase of the endpoint `n0`: bit `h` of its mask is `Q[phase + h]`.
    #[inline]
    pub fn phase(&self, n0: u128) -> u64 {
        ((n0 % self.d as u128) * self.ginv as u128 % self.d as u128) as u64
    }

    /// The same phase computed as the GPU does it (two halves, Barrett).
    pub fn phase_device_model(&self, n0: u128) -> u64 {
        let red = |x: u64| -> u64 {
            let q = ((x as u128 * self.minv as u128) >> 64) as u64;
            let mut r = x - q * self.d;
            if r >= self.d {
                r -= self.d;
            }
            r
        };
        let lo = n0 as u64;
        let hi = (n0 >> 64) as u64;
        let x = red(lo) * self.ginv + red(hi) * self.ginv64;
        red(x)
    }

    /// 64 completion bits for word `w` (h = 64w .. 64w+63).
    #[inline(always)]
    pub fn word(&self, phase: u64, w: usize, n0: u128) -> u64 {
        let i = phase as usize + 64 * w;
        let (q, off) = (i / 64, (i % 64) as u32);
        let lo = self.bits[q];
        let mut v = if off == 0 { lo } else { (lo >> off) | (self.bits[q + 1] << (64 - off)) };
        if w == 0 && self.sib > 0 && n0 < SIB_MIN_N {
            v |= 1;
        }
        v
    }

    /// Does the sieve keep completion n? (Same decision as the mask bit.)
    pub fn keeps(&self, n: u128) -> bool {
        if self.sib > 0 && n < SIB_MIN_N {
            return true;
        }
        self.keep[(n % self.d as u128) as usize]
    }

    /// Bits as little-endian u32 words (device layout).
    pub fn bits_u32(&self) -> &[u32] {
        unsafe { std::slice::from_raw_parts(self.bits.as_ptr() as *const u32, self.bits.len() * 2) }
    }
}

/// Independent certificate search for audits (no tables): find an inverse word
/// from `v` with at most `omax` odd steps and `3^o > 2^(s + extra)`. Returns the
/// partner p and (s, o). Depth-first over inverse even/odd steps on exact u128.
pub fn find_predecessor(v: u128, omax: u32, extra: u32) -> Option<(u128, u32, u32)> {
    // state: value, s, o
    let mut stack = vec![(v, 0u32, 0u32)];
    let l23 = 1.584962500721156f64;
    while let Some((x, s, o)) = stack.pop() {
        if o > 0 && POW3[o as usize] > (1u128 << (s + extra)) {
            return Some((x, s, o));
        }
        // Best possible completion uses all remaining odd steps and no evens:
        // final value <= omax (l23 - 1) + o - s - extra. Prune when <= 0.
        let r = omax - o;
        let room = |s: u32| omax as f64 * (l23 - 1.0) + o as f64 - (s + extra) as f64;
        if room(s) <= 0.0 {
            continue;
        }
        if r > 0 && x % 3 == 2 {
            stack.push(((2 * x - 1) / 3, s + 1, o + 1));
        }
        if room(s + 1) > 0.0 && x < (1u128 << 126) {
            stack.push((2 * x, s + 1, o));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::t_iter;

    #[test]
    fn j2_reproduces_mod9() {
        let s = Sieve3::new(2, 0, 62, 10);
        let kept: Vec<u64> = (0..9).filter(|&r| s.keep[r as usize]).collect();
        assert_eq!(kept, vec![0, 1, 3, 6, 7]);
    }

    #[test]
    fn survival_counts_match_independent_dp() {
        // From scratchpad/sieve3/dp/sieve_dp.rs (separate code) and the
        // orchestrator's DFS: J=4: 44/81, J=7: 1174/2187, J=9: 10538/19683.
        assert_eq!(Sieve3::new(4, 0, 62, 10).kept, 44);
        assert_eq!(Sieve3::new(7, 0, 62, 10).kept, 1174);
        assert_eq!(Sieve3::new(9, 0, 62, 10).kept, 10538);
        assert_eq!(Sieve3::new(12, 0, 62, 10).kept, 283552);
        assert_eq!(Sieve3::new(12, 3, 62, 10).kept, 277650);
    }

    #[test]
    fn phase_layout_and_device_model() {
        let s = Sieve3::new(9, 3, 62, 10);
        let mut x = 0x1234_5678_9abc_def1u128;
        for _ in 0..20000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) & ((1u128 << 62) - 1);
            let n0 = x | 3;
            let ph = s.phase(n0);
            assert_eq!(ph, s.phase_device_model(n0));
            for h in [0u128, 1, 2, 63, 64, 500, 1023] {
                let n = n0 + (h << 62);
                let bit = (s.word(ph, (h / 64) as usize, n0) >> (h % 64)) & 1;
                assert_eq!(bit == 1, s.keeps(n), "n0={n0} h={h}");
            }
        }
    }

    #[test]
    fn excluded_residues_have_certificates() {
        let s = Sieve3::new(8, 3, 62, 10);
        let mut x = 0xdead_beefu128;
        let mut checked = 0;
        for _ in 0..20000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) & ((1u128 << 72) - 1);
            let n = x | (1u128 << 71) | 3;
            if s.keeps(n) {
                continue;
            }
            checked += 1;
            if let Some((p, st, o)) = find_predecessor(n, 8, 0) {
                assert!(p < n);
                assert_eq!(t_iter(p, st), (n, o));
                continue;
            }
            let mut ok = false;
            let mut pk = 1u128;
            for k in 1..=3u32 {
                pk *= 4;
                let v = pk * n + (pk - 1) / 3;
                if let Some((p, st, o)) = find_predecessor(v, 8, 2 * k) {
                    assert!(p < n);
                    assert_eq!(t_iter(p, st), (v, o));
                    assert_eq!(t_iter(v, 2 * k + 1).0, t_iter(n, 1).0);
                    ok = true;
                    break;
                }
            }
            assert!(ok, "no certificate for excluded n={n}");
        }
        assert!(checked > 5000);
    }
}

/// Best inverse word per residue mod 3^e for every e <= omax (code o | s << 5).
pub fn best_words_all(omax: u32) -> Vec<Vec<u16>> {
    let mut levels: Vec<Vec<u16>> = vec![vec![0u16; 1]];
    for e in 1..=omax {
        let d = 3u64.pow(e);
        let dprev = 3u64.pow(e - 1);
        let prev = &levels[(e - 1) as usize];
        let mut cur = vec![0u16; d as usize];
        for y in 0..d {
            let r = y % 3;
            if r == 0 {
                continue;
            }
            let mut best: u16 = 0;
            let mut k = if r == 2 { 0u32 } else { 1u32 };
            loop {
                let ub = 1.584962500721156 - (k + 1) as f64 + 0.584962500721156 * (e - 1) as f64;
                if ub <= 0.0 {
                    break;
                }
                let t = (((y as u128) << (k + 1)) + d as u128 - 1) % d as u128;
                let sub = prev[((t / 3) as u64 % dprev) as usize];
                let (o, s) = ((sub & 31) as u32 + 1, (sub >> 5) as u32 + k + 1);
                if POW3[o as usize] > (1u128 << s) {
                    let code = (o | (s << 5)) as u16;
                    if best == 0 || ratio_gt(code, best) {
                        best = code;
                    }
                }
                k += 2;
            }
            cur[y as usize] = best;
        }
        levels.push(cur);
    }
    levels
}

/// Experimental (K, J) table: completion n = n0 + h 2^M is skipped when some
/// p < n reaches T^j(n) (j <= K) through an inverse word with at most
/// f_j + J odd steps (f_j = odd steps among n's first j steps, capped at
/// `omax`). Validity of the word depends only on (n mod 2^K, n mod 3^J);
/// the size condition is proven for every n >= 2^M in the class, so h = 0
/// (n = n0 < 2^M) is always kept. Rows are indexed by n0 mod 2^K and use the
/// same phase layout as `Sieve3`.
pub struct SieveK {
    pub j: u32,
    pub k: u32,
    pub d: u64,
    pub m: u32,
    pub a: u32,
    pub ginv: u64,
    pub ginv64: u64,
    pub minv: u64,
    pub row_words64: usize,
    pub bits: Vec<u64>,
    /// excluded (row, residue) pairs over rows that survive the baseline prefix rules
    pub excluded: u64,
    pub rows_used: u64,
}

impl SieveK {
    pub fn new(j: u32, k: u32, m: u32, a: u32, omax: u32) -> Self {
        assert!(k >= 2 && k <= 20 && k <= m && j >= 2 && j <= 12 && omax <= 18);
        let d = 3u64.pow(j);
        let tables = best_words_all(omax);
        let g = ((1u128 << m) % d as u128) as u64;
        let ginv = inv_mod(g, d);
        let ginv64 = (((1u128 << 64) % d as u128) * ginv as u128 % d as u128) as u64;
        let minv = u64::MAX / d;
        let nbits = d as usize + (1usize << a) + 128;
        let row_words64 = nbits.div_ceil(64) + 1;
        let nrows = 1usize << k;
        // inverse of 2^K mod D and of D mod 2^K for CRT representatives
        let p2k = 1u128 << k;
        let inv2k = inv_mod((p2k % d as u128) as u64, d) as u128;
        let rows: Vec<(Vec<u64>, u64, bool)> = (0..nrows)
            .into_par_iter()
            .map(|r| {
                let mut keep = vec![true; d as usize];
                let mut used = false;
                if r & 3 == 3 {
                    used = true;
                    for t in 0..d {
                        // representative n = r + 2^K * u with n = t (mod D)
                        let u = ((t as u128 + d as u128 - (r as u128 % d as u128)) % d as u128) * inv2k % d as u128;
                        let n = r as u128 + p2k * u + (p2k * d as u128) * (1u128 << 50);
                        let (mut y, mut f) = (n, 0u32);
                        for jj in 0..=k {
                            let ob = (f + j).min(omax);
                            let code = tables[ob as usize][(y % POW3[ob as usize]) as usize];
                            if code != 0 {
                                let (o, s) = ((code & 31) as u32, (code >> 5) as u32);
                                // y_j = (3^f n + dj)/2^j: need n (2^j 3^o - 2^s 3^f) >= 2^s dj for all n >= 2^M
                                let dj = (y << jj) - POW3[f as usize] * n;
                                let lhs = (POW3[o as usize] << jj) as i128 - (POW3[f as usize] << s) as i128;
                                if lhs >= 1 {
                                    let need = dj << s; // < 2^110
                                    if (lhs as u128).checked_mul(1u128 << m).map_or(true, |v| v >= need) {
                                        keep[t as usize] = false;
                                        break;
                                    }
                                }
                            }
                            if jj == k {
                                break;
                            }
                            if y & 1 == 1 {
                                y = y + (y >> 1) + 1;
                                f += 1;
                            } else {
                                y >>= 1;
                            }
                        }
                    }
                }
                let mut rb = vec![0u64; row_words64];
                let mut ex = 0u64;
                for i in 0..nbits {
                    let res = ((i as u128 % d as u128) * g as u128 % d as u128) as usize;
                    if keep[res] {
                        rb[i / 64] |= 1u64 << (i % 64);
                    }
                }
                if used {
                    ex = keep.iter().filter(|&&x| !x).count() as u64;
                }
                (rb, ex, used)
            })
            .collect();
        let mut bits = Vec::with_capacity(nrows * row_words64);
        let (mut excluded, mut rows_used) = (0u64, 0u64);
        for (rb, ex, used) in rows {
            bits.extend_from_slice(&rb);
            excluded += ex;
            rows_used += used as u64;
        }
        SieveK { j, k, d, m, a, ginv, ginv64, minv, row_words64, bits, excluded, rows_used }
    }

    #[inline]
    pub fn phase(&self, n0: u128) -> u64 {
        ((n0 % self.d as u128) * self.ginv as u128 % self.d as u128) as u64
    }

    #[inline(always)]
    pub fn word(&self, phase: u64, w: usize, n0: u128) -> u64 {
        let row = (n0 & ((1u128 << self.k) - 1)) as usize;
        let base = row * self.row_words64;
        let i = phase as usize + 64 * w;
        let (q, off) = (base + i / 64, (i % 64) as u32);
        let lo = self.bits[q];
        let mut v = if off == 0 { lo } else { (lo >> off) | (self.bits[q + 1] << (64 - off)) };
        if w == 0 {
            v |= 1; // h = 0: n = n0 < 2^M is always kept
        }
        v
    }

    pub fn keeps(&self, n: u128) -> bool {
        if n < (1u128 << self.m) {
            return true;
        }
        let n0 = n & ((1u128 << self.m) - 1);
        let h = (n >> self.m) as usize;
        (self.word(self.phase(n0), h / 64, n0) >> (h % 64)) & 1 == 1
    }

    pub fn bits_u32(&self) -> &[u32] {
        unsafe { std::slice::from_raw_parts(self.bits.as_ptr() as *const u32, self.bits.len() * 2) }
    }
}
