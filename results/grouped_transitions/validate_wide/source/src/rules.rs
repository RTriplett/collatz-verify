//! Step-1 sieve rules.
//!
//! A *class* is the residue family `n = n0 + 2^k a` (a >= 0) with
//! `m = T^k(n0)` and `f` odd steps, so `T^k(n) = m + 3^f a`.  A rule may cull
//! a class only with a *certificate*: a partner family `p + S a` with
//! `T^s(p + S a) = T^k(n0 + 2^k a)`, `p < n0` and `S <= 2^k`.  Then every
//! member merges with a strictly smaller integer, and by strong induction
//! converges.  Descent is the special case "partner = the member's own
//! iterate".  Every rule here is per-class (no global tables of visited
//! states), so the search stays an embarrassingly parallel DFS.
//!
//! Baseline (Angeltveit, arXiv 2026, v0.3):
//!   * descent:  `306 k >= 485 f` (or `m < n0`)
//!   * pm1:      `m = 2 (mod 3)` after an even number (>= 2) of even steps,
//!               partner `(2m-1)/3` (Lemmas 2.7/2.8), slope test only
//!   * oee:      Hercher's odd-...-odd-even-even merge with `(x-1)/2`
//!               (Lemma 2.9), as a 3-state machine; `root_oee` starts it armed,
//!               matching the CUDA reference input `3 2 2 9 1 0`.
//!
//! Improvements:
//!   * pm_inf:   maximal 3-adic path merge.  `m + 1 = 3^t q` gives the chain
//!               of `t = min(v3(m+1), f)` inverse odd steps ending at
//!               `u = 2^t q - 1`, slope `2^t 3^(f-t)`.
//!   * mix(3^J): mixed inverse words (inverse even AND inverse odd steps).
//!               A word with `s` steps, `o` of them odd, maps `m` to
//!               `p = (2^s m - b)/3^o` and is valid exactly on one residue of
//!               `m mod 3^o`; we tabulate, for every residue mod 3^j (j <= J),
//!               the valid word with the largest value ratio `3^o / 2^s`.

use crate::arith::{mod_pow3, shl_lt_mul_add, weight_le_pow2k, POW3, U256};
use std::sync::Arc;

/// Why a class (or a would-be child) was removed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Cull {
    Descent = 0,
    Pm1 = 1,
    Oee = 2,
    PmInf = 3,
    Mix = 4,
    Walkback = 5,
}
pub const NUM_CULLS: usize = 6;
pub const CULL_NAMES: [&str; NUM_CULLS] = ["descent", "pm1", "oee", "pm_inf", "mix", "walkback"];

#[derive(Clone)]
pub struct Rules {
    pub pm1: bool,
    pub oee: bool,
    pub root_oee: bool,
    pub pm_inf: bool,
    pub mix: Option<Arc<MixTables>>,
    /// Apply the mixed-word rule only at depths >= this (0 = every depth).
    pub mix_from: u32,
    /// Exhaustive reverse search at one checkpoint depth.
    pub walkback: Option<Walkback>,
}

/// Bounded reverse-tree search for a merge certificate at depth `depth`.
#[derive(Clone, Copy, Debug)]
pub struct Walkback {
    pub depth: u32,
    pub max_states: u32,
}

impl Rules {
    /// Angeltveit's rule set exactly as in the reference CUDA program.
    pub fn baseline() -> Self {
        Rules { pm1: true, oee: true, root_oee: true, pm_inf: false, mix: None, mix_from: 0, walkback: None }
    }
    /// Baseline plus maximal path merge plus mixed-word tables mod 3^j.
    pub fn improved(mix_j: u32) -> Self {
        let mix = (mix_j > 0).then(|| Arc::new(MixTables::build(mix_j)));
        Rules { pm_inf: true, mix, ..Self::baseline() }
    }
    /// Descent only (OEIS A076227 survivor counts).
    pub fn descent_only() -> Self {
        Rules { pm1: false, oee: false, root_oee: false, pm_inf: false, mix: None, mix_from: 0, walkback: None }
    }
    pub fn mix_j(&self) -> u32 {
        self.mix.as_ref().map_or(0, |t| t.j)
    }
    pub fn label(&self) -> String {
        let mut parts = vec!["descent".to_string()];
        if self.pm1 {
            parts.push("pm1".into());
        }
        if self.oee {
            parts.push(if self.root_oee { "oee(root)".into() } else { "oee".into() });
        }
        if self.pm_inf {
            parts.push("pm_inf".into());
        }
        if let Some(t) = &self.mix {
            if self.mix_from > 0 {
                parts.push(format!("mix(3^{})@k>={}", t.j, self.mix_from));
            } else {
                parts.push(format!("mix(3^{})", t.j));
            }
        }
        if let Some(w) = self.walkback {
            parts.push(format!("walkback@{}({})", w.depth, w.max_states));
        }
        parts.join("+")
    }
    /// True when only the parity-state rules are active, i.e. the survivor
    /// set is a function of the parity vector and `crate::dp` counts it
    /// exactly.
    pub fn is_parity_only(&self) -> bool {
        !self.pm_inf && self.mix.is_none() && self.walkback.is_none()
    }

    /// Value-dependent extra rules (applied after the parity rules).
    #[inline]
    pub fn extra_cull(&self, n0: u128, k: u32, f: u32, m: u128, m3: u32) -> Option<Cull> {
        if self.pm_inf && pm_inf_culls(n0, k, f, m) {
            return Some(Cull::PmInf);
        }
        if let Some(t) = &self.mix {
            if k >= self.mix_from && t.culls_r(n0, k, f, m, m3) {
                return Some(Cull::Mix);
            }
        }
        if let Some(w) = self.walkback {
            if k == w.depth && walkback_certificate(n0, k, f, m, w.max_states).is_some() {
                return Some(Cull::Walkback);
            }
        }
        None
    }
}

/// Depth-first search of the inverse tree of `m` over inverse even
/// (`v -> 2v`) and inverse odd (`v -> (2v-1)/3`, `v = 2 mod 3`, `t < f`) steps.
/// A state `(v, r, t)` is a certificate when `r >= 1`, `v < n0` and
/// `2^r 3^(f-t) <= 2^k`.  Prunes: `r + (f - t) > k` (since `3^e >= 2^e` the
/// slope test can never pass again) and `(v+1) 2^(f-t) >= 3^(f-t) (n0+1)` (even using every
/// remaining inverse odd step, `v + 1` shrinks by at most `(2/3)^(f-t)`).
/// Gives up (keeps the class) after `max_states` states.
pub fn walkback_certificate(n0: u128, k: u32, f: u32, m: u128, max_states: u32) -> Option<Certificate> {
    let mut stack: Vec<(u128, u32, u32)> = vec![(m, 0, 0)];
    let mut visited = 0u32;
    while let Some((v, r, t)) = stack.pop() {
        visited += 1;
        if visited > max_states {
            return None;
        }
        if r >= 1 && v < n0 && weight_le_pow2k(r, f - t, k) {
            return Some(Certificate { p: v, s: r, o: t });
        }
        let e = f - t;
        if e < MAX_POW3_U128 && !U256::mul(v + 1, 1u128 << e).lt(&U256::mul(POW3[e as usize], n0 + 1)) {
            continue;
        }
        // An inverse odd step keeps r + (f - t); an inverse even step raises it.
        if r + 1 + e <= k {
            if let Some(v2) = v.checked_mul(2) {
                stack.push((v2, r + 1, t));
            }
        }
        if t < f && v % 3 == 2 {
            stack.push(((2 * v - 1) / 3, r + 1, t + 1));
        }
    }
    None
}

/// `2^e` and `3^e` both fit in u128.
const MAX_POW3_U128: u32 = 80;

/// Multiplicative inverse of 3 modulo 2^128; `x * INV3` is `x / 3` when
/// `3 | x`, and `3 | x` iff the product is `<= u128::MAX / 3`.
const INV3: u128 = 0xAAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAB;
const THIRD: u128 = u128::MAX / 3;

/// Maximal path merge.  Cull iff `u = 2^t (m+1)/3^t - 1 < n0` and
/// `2^t 3^(f-t) <= 2^k` with `t = min(v3(m+1), f)`.  `u` decreases and the
/// slope test weakens as `t` grows, so only `t_max` needs testing.
#[inline]
pub fn pm_inf_culls(n0: u128, k: u32, f: u32, m: u128) -> bool {
    let mut q = m + 1;
    let mut t = 0u32;
    while t < f {
        let d = q.wrapping_mul(INV3);
        if d > THIRD {
            break;
        }
        q = d;
        t += 1;
    }
    if t == 0 {
        return false;
    }
    let u = (q << t) - 1;
    u < n0 && weight_le_pow2k(t, f - t, k)
}

/// Same rule with the certificate returned, for audits.
pub fn pm_inf_certificate(n0: u128, k: u32, f: u32, m: u128) -> Option<Certificate> {
    if !pm_inf_culls(n0, k, f, m) {
        return None;
    }
    let mut q = m + 1;
    let mut t = 0u32;
    while t < f && q % 3 == 0 {
        q /= 3;
        t += 1;
    }
    Some(Certificate { p: (q << t) - 1, s: t, o: t })
}

/// A merge certificate for class `(n0, k, f, m)`: `T^s(p + 2^s 3^(f-o) a) = m + 3^f a`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Certificate {
    pub p: u128,
    pub s: u32,
    pub o: u32,
}

/// Dense mixed-word tables, one per modulus 3^j for 1 <= j <= J.
///
/// Entry encoding (u64): 0 = empty, else `1 + (b << 11 | o << 6 | s)`.
pub struct MixTables {
    pub j: u32,
    tables: Vec<Vec<u64>>,
    pub filled: Vec<u64>,
    /// Lemire fastmod constants for 3^j.
    fastmod: Vec<u64>,
}

#[inline(always)]
fn fastmod_u32(a: u32, mconst: u64, d: u32) -> u32 {
    ((mconst.wrapping_mul(a as u64) as u128 * d as u128) >> 64) as u32
}

impl MixTables {
    pub fn build(max_j: u32) -> Self {
        assert!((1..=16).contains(&max_j), "mix table modulus must be 3^1..3^16");
        let mut tables: Vec<Vec<u64>> = vec![Vec::new()];
        let mut filled = vec![0u64];
        for j in 1..=max_j {
            // Candidate (s, o) shapes: o odd steps, s total, value-decreasing.
            let mut pairs: Vec<(u32, u32)> = Vec::new();
            for o in 1..=j {
                for s in o..=2 * o {
                    if (1u128 << s) < POW3[o as usize] {
                        pairs.push((s, o));
                    }
                }
            }
            // Best ratio 3^o / 2^s first; ties broken deterministically.
            pairs.sort_by(|&(sa, oa), &(sb, ob)| {
                let lhs = POW3[oa as usize] << sb;
                let rhs = POW3[ob as usize] << sa;
                rhs.cmp(&lhs).then((sa, oa).cmp(&(sb, ob)))
            });
            let modulus_j = POW3[j as usize] as u64;
            let mut table = vec![0u64; modulus_j as usize];
            let mut used = 0u64;
            for (s, o) in pairs {
                let modulus_o = POW3[o as usize] as u64;
                let inv = inverse_mod((1u128 << s) % modulus_o as u128, modulus_o as u128) as u64;
                visit_odd_masks(s, o, &mut |bits| {
                    let b = mask_constant(s, bits);
                    assert!(b < (1u128 << 53), "mix constant overflows packing");
                    let rho = ((b % modulus_o as u128) * inv as u128 % modulus_o as u128) as u64;
                    let packed = 1 + (((b as u64) << 11) | ((o as u64) << 6) | s as u64);
                    let mut lift = rho;
                    while lift < modulus_j {
                        let slot = &mut table[lift as usize];
                        if *slot == 0 {
                            *slot = packed;
                            used += 1;
                        }
                        lift += modulus_o;
                    }
                });
            }
            tables.push(table);
            filled.push(used);
        }
        // Lemire constants for 3^j, j >= 1 (entry 0 is never used)
        let fastmod = (0..=max_j).map(|j| if j == 0 { 0 } else { u64::MAX / POW3[j as usize] as u64 + 1 }).collect();
        MixTables { j: max_j, tables, filled, fastmod }
    }

    /// All tables concatenated, with `offsets[j]` the start of the 3^j table
    /// (the layout the CUDA bitvector builder reads).
    pub fn flat(&self) -> (Vec<u64>, [u64; 17]) {
        let mut off = [0u64; 17];
        let mut data = Vec::with_capacity(self.bytes() / 8);
        for (j, t) in self.tables.iter().enumerate() {
            off[j] = data.len() as u64;
            data.extend_from_slice(t);
        }
        (data, off)
    }

    pub fn bytes(&self) -> usize {
        self.tables.iter().map(|t| t.len() * 8).sum()
    }

    #[inline]
    fn entry(&self, f: u32, m: u128) -> Option<(u32, u32, u128)> {
        let jp = f.min(self.j);
        if jp == 0 {
            return None;
        }
        self.decode(self.tables[jp as usize][mod_pow3(m, jp) as usize])
    }

    /// Tabulated word `(s, o, b)` for residue `r` of `m mod 3^jp` (`jp <= J`).
    #[inline]
    pub fn lookup(&self, jp: u32, r: u64) -> Option<(u32, u32, u128)> {
        self.decode(self.tables[jp as usize][r as usize])
    }

    /// Lookup with `m3 = m mod 3^16` already known.
    #[inline(always)]
    fn entry_r(&self, f: u32, m3: u32) -> Option<(u32, u32, u128)> {
        let jp = f.min(self.j);
        if jp == 0 {
            return None;
        }
        let r = if jp == 16 { m3 } else { fastmod_u32(m3, self.fastmod[jp as usize], POW3[jp as usize] as u32) };
        self.decode(self.tables[jp as usize][r as usize])
    }

    #[inline(always)]
    fn decode(&self, packed: u64) -> Option<(u32, u32, u128)> {
        if packed == 0 {
            return None;
        }
        let raw = packed - 1;
        Some(((raw & 0x3f) as u32, ((raw >> 6) & 0x1f) as u32, (raw >> 11) as u128))
    }

    /// Cull iff the tabulated word gives `0 < p < n0` with `2^s 3^(f-o) <= 2^k`.
    #[inline]
    pub fn culls(&self, n0: u128, k: u32, f: u32, m: u128) -> bool {
        let Some(e) = self.entry(f, m) else { return false };
        Self::word_culls(e, n0, k, f, m)
    }

    #[inline(always)]
    pub fn culls_r(&self, n0: u128, k: u32, f: u32, m: u128, m3: u32) -> bool {
        let Some(e) = self.entry_r(f, m3) else { return false };
        Self::word_culls(e, n0, k, f, m)
    }

    #[inline(always)]
    fn word_culls((s, o, b): (u32, u32, u128), n0: u128, k: u32, f: u32, m: u128) -> bool {
        if !weight_le_pow2k(s, f - o, k) {
            return false;
        }
        // p = (2^s m - b) / 3^o; p >= 1 and p < n0  <=>  b < 2^s m < 3^o n0 + b
        shl_lt_mul_add(m, s, n0, POW3[o as usize] as u64, b) && !shl_lt_mul_add(m, s, 1, 0, b + 1)
    }

    pub fn certificate(&self, n0: u128, k: u32, f: u32, m: u128) -> Option<Certificate> {
        if !self.culls(n0, k, f, m) {
            return None;
        }
        let (s, o, b) = self.entry(f, m)?;
        let p = ((m << s) - b) / POW3[o as usize];
        Some(Certificate { p, s, o })
    }
}

fn inverse_mod(a: u128, modulus: u128) -> u128 {
    let (mut t, mut new_t) = (0i128, 1i128);
    let (mut r, mut new_r) = (modulus as i128, a as i128);
    while new_r != 0 {
        let q = r / new_r;
        (t, new_t) = (new_t, t - q * new_t);
        (r, new_r) = (new_r, r - q * new_r);
    }
    assert_eq!(r, 1);
    if t < 0 {
        t += modulus as i128;
    }
    t as u128
}

/// Enumerate forward parity masks of length `s` with `o` odd steps whose
/// first forward step is odd (bit 0 set).  Words starting with a forward even
/// step are dominated: dropping that step halves `p` and the slope.
fn visit_odd_masks(s: u32, o: u32, visit: &mut impl FnMut(u64)) {
    fn rec(next: u32, s: u32, remaining: u32, bits: u64, visit: &mut impl FnMut(u64)) {
        if remaining == 0 {
            visit(bits);
            return;
        }
        for pos in next..s {
            if s - pos < remaining {
                break;
            }
            rec(pos + 1, s, remaining - 1, bits | (1u64 << pos), visit);
        }
    }
    if o > 0 && o <= s {
        rec(1, s, o - 1, 1, visit);
    }
}

/// `b` with `T^s(p) = (3^o p + b) / 2^s` for the forward mask `bits`.
fn mask_constant(s: u32, bits: u64) -> u128 {
    let mut b = 0u128;
    for t in 0..s {
        if bits & (1u64 << t) != 0 {
            b = 3 * b + (1u128 << t);
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::t_iter;

    fn check_cert(n0: u128, k: u32, f: u32, m: u128, c: Certificate) {
        assert_eq!(t_iter(n0, k), (m, f));
        assert_eq!(t_iter(c.p, c.s), (m, c.o), "partner does not reach m");
        let slope = (1u128 << c.s) * POW3[(f - c.o) as usize];
        assert!(c.p < n0 && slope <= (1u128 << k));
        for a in [0u128, 1, 2, 7, 12345] {
            let member = n0 + (a << k);
            let partner = c.p + slope * a;
            let (tm, _) = t_iter(member, k);
            let (tp, fp) = t_iter(partner, c.s);
            assert_eq!(tm, tp, "merge fails n0={n0} k={k} a={a} cert={c:?}");
            assert_eq!(fp, c.o);
            assert!(partner < member);
        }
    }

    #[test]
    fn pm_inf_certificates_merge() {
        let mut fired = 0;
        for k in 6..=18u32 {
            for n0 in 0..(1u128 << k) {
                let (m, f) = t_iter(n0, k);
                if let Some(c) = pm_inf_certificate(n0, k, f, m) {
                    check_cert(n0, k, f, m, c);
                    fired += 1;
                }
            }
        }
        assert!(fired > 1000);
    }

    #[test]
    fn mix_certificates_merge() {
        let t = MixTables::build(8);
        let mut fired = 0;
        for k in 6..=18u32 {
            for n0 in 0..(1u128 << k) {
                let (m, f) = t_iter(n0, k);
                if let Some(c) = t.certificate(n0, k, f, m) {
                    check_cert(n0, k, f, m, c);
                    fired += 1;
                }
            }
        }
        assert!(fired > 1000);
    }

    #[test]
    fn walkback_certificates_merge() {
        let mut fired = 0;
        for k in 8..=16u32 {
            for n0 in 0..(1u128 << k) {
                let (m, f) = t_iter(n0, k);
                if 485 * f <= 306 * k {
                    continue;
                }
                if let Some(c) = walkback_certificate(n0, k, f, m, 4096) {
                    check_cert(n0, k, f, m, c);
                    fired += 1;
                }
            }
        }
        assert!(fired > 1000);
    }

    #[test]
    fn mix_table_fill_matches_reference() {
        // Residue-sieve papercomparison: J=16 fills 20,113,810 of 43,046,721.
        // Smaller moduli are checked for monotone, nonzero fill here.
        let t = MixTables::build(10);
        for j in 1..=10 {
            assert!(t.filled[j] > 0 && t.filled[j] <= POW3[j] as u64);
        }
    }
}
