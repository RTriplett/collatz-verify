//! Shallow h-dependent merge row table on completions (c-screen port of the
//! sieve3 (K, J) = (12, 9) table), built only for the rows the tree can reach.
//!
//! **Certificate.**  For a completion `n = n0 + h 2^M` (`h >= 1`, so
//! `n >= 2^M`) and a depth `j <= K`, `2^j T^j(n) = 3^f n + d_j` where `f`
//! (odd steps among the first `j`) and `d_j` depend only on `n mod 2^j`.  An
//! inverse word with `s` steps and `o <= f + J` inverse-odd steps is valid on
//! one residue of `T^j(n) mod 3^o`, and `T^j(n) mod 3^(f+J)` is fixed by
//! `(n mod 2^K, n mod 3^J)`.  If the best word for that residue satisfies
//! `lhs = 2^j 3^o - 2^s 3^f >= 1` and `2^M lhs >= 2^s d_j`, then
//! `2^s T^j(n) <= 3^o n` for every `n >= 2^M` of the class, so the word's
//! partner `p = (2^s T^j(n) - c) / 3^o` (`c >= 1`) is a positive integer with
//! `p < n` and `T^s(p) = T^j(n)`: `n` merges with a smaller number (strong
//! induction).  `h = 0` (`n = n0 < 2^M`) is always kept.  Exactness and the
//! odd-step cap `omax = 18` are exactly sieve3's (`cv::sieve3::SieveK` in its
//! tree); `rowsieve_audit` checks bit equality against that code.
//!
//! **Reachable rows.**  Rows are indexed by `n0 mod 2^K`.  Every depth-`M`
//! endpoint descends from a depth-`K` tree survivor, so only those residues
//! need a row: 144 of 4096 at K = 12 under the baseline rules (any rule set
//! that includes the baseline culls reaches a subset).  Every other residue
//! maps to row 0, which keeps every completion: reachability affects only
//! the saving, never soundness.
//!
//! **Layout.**  Bit `i` of a row is `keep[(i mod 3^J) 2^M mod 3^J]`, so the
//! completions of an endpoint are the consecutive bits starting at
//! `phase = n0 (2^M)^-1 mod 3^J`, read with the same funnel shift as the
//! bitvectors.  Rows are `row_words` u32 words, long enough for any phase
//! plus `2^A + 64` bits.
//!
//! **Build.**  The best-word DP of sieve3 (`best_words_all`) is kept dense
//! up to level 16 (129 MB) and evaluated on demand above it, so a build takes
//! seconds instead of 35 s and 1.2 GB.

use crate::arith::POW3;
use crate::rules::Rules;
use crate::tree::{frontier, TreeStats};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Dense DP levels; levels above are evaluated on demand.
const DENSE_MAX: u32 = 16;

pub struct RowSieve {
    pub k: u32,
    pub j: u32,
    pub m: u32,
    pub a: u32,
    pub omax: u32,
    /// D = 3^J.
    pub d: u32,
    /// g = 2^M mod D and ginv = g^-1 mod D.
    pub g: u32,
    pub ginv: u32,
    /// `limb_c[i] = 2^(32 i) ginv mod D` (device phase from u32 limbs).
    pub limb_c: [u32; 4],
    /// floor(2^32 / D) (device Barrett reduction).
    pub minv32: u32,
    /// u32 words per row.
    pub row_words: usize,
    /// Reachable residues `n0 mod 2^K`, sorted; compact row `i + 1`.
    pub rows: Vec<u32>,
    /// 2^K entries: compact row index (0 = all-ones row, keep everything).
    pub rowmap: Vec<u16>,
    /// `(rows.len() + 1) * row_words` u32 words; row 0 is all ones.
    pub bits: Vec<u32>,
    /// Excluded (row, n mod D) pairs summed over reachable rows.
    pub excluded: u64,
    pub build_seconds: f64,
}

#[inline]
fn ratio_gt(a: u16, b: u16) -> bool {
    let (oa, sa) = ((a & 31) as usize, (a >> 5) as u32);
    let (ob, sb) = ((b & 31) as usize, (b >> 5) as u32);
    (POW3[oa] << sb) > (POW3[ob] << sa)
}

/// sieve3's best-word recurrence for one residue `y mod 3^e`, given the level
/// below as a function (code `o | s << 5`, 0 = no word with 3^o > 2^s).
#[inline]
fn best_from(y: u64, e: u32, prev: impl Fn(u64) -> u16) -> u16 {
    let d = POW3[e as usize] as u64;
    let dprev = d / 3;
    let r = y % 3;
    if r == 0 {
        return 0;
    }
    let mut best: u16 = 0;
    let mut k = if r == 2 { 0u32 } else { 1u32 };
    loop {
        // Identical floating-point bound to sieve3::best_words_all.
        let ub = 1.584962500721156 - (k + 1) as f64 + 0.584962500721156 * (e - 1) as f64;
        if ub <= 0.0 {
            break;
        }
        // y 2^(k+1) - 1 mod D; y < 3^18 < 2^29 and k + 1 <= 11, so u64 suffices.
        let t = ((y << (k + 1)) + d - 1) % d;
        debug_assert!(t % 3 == 0);
        let sub = prev((t / 3) % dprev);
        let (o, s) = ((sub & 31) as u32 + 1, (sub >> 5) as u32 + k + 1);
        if POW3[o as usize] > (1u128 << s) {
            let code = (o | (s << 5)) as u16;
            if best == 0 || ratio_gt(code, best) {
                best = code;
            }
        }
        k += 2;
    }
    best
}

/// Best-word tables, dense through `DENSE_MAX`, on demand above.
struct BestWords {
    dense: Vec<Vec<u16>>,
    omax: u32,
}

impl BestWords {
    fn new(omax: u32) -> Self {
        let top = omax.min(DENSE_MAX);
        let mut dense: Vec<Vec<u16>> = vec![vec![0u16; 1]];
        for e in 1..=top {
            let d = POW3[e as usize] as usize;
            let prev = &dense[(e - 1) as usize];
            let mut cur = vec![0u16; d];
            cur.par_chunks_mut(1 << 16).enumerate().for_each(|(ci, chunk)| {
                for (i, slot) in chunk.iter_mut().enumerate() {
                    let y = ((ci << 16) + i) as u64;
                    *slot = best_from(y, e, |x| prev[x as usize]);
                }
            });
            dense.push(cur);
        }
        BestWords { dense, omax }
    }

    #[inline]
    fn get(&self, e: u32, y: u64) -> u16 {
        debug_assert!(e <= self.omax && y < POW3[e as usize] as u64);
        if (e as usize) < self.dense.len() {
            self.dense[e as usize][y as usize]
        } else {
            best_from(y, e, |x| self.get(e - 1, x))
        }
    }
}

fn inv_mod(a: u64, m: u64) -> u64 {
    let (mut t, mut nt) = (0i128, 1i128);
    let (mut r, mut nr) = (m as i128, a as i128);
    while nr != 0 {
        let q = r / nr;
        (t, nt) = (nt, t - q * nt);
        (r, nr) = (nr, r - q * nr);
    }
    assert_eq!(r, 1, "not invertible");
    (if t < 0 { t + m as i128 } else { t }) as u64
}

/// Residues `n0 mod 2^k` of the depth-`k` survivors of `rules`.
pub fn reachable_rows(k: u32, rules: &Rules) -> Vec<u32> {
    let mut st = TreeStats::new(k);
    let mut rows: Vec<u32> = frontier(k, rules, &mut st).iter().map(|nd| nd.n0 as u32).collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// Keep decisions of one row over all residues `n mod 3^J` (sieve3's loop).
fn row_keep(r: u32, k: u32, j: u32, m: u32, omax: u32, bw: &BestWords) -> Vec<bool> {
    let d = POW3[j as usize];
    let p2k = 1u128 << k;
    let inv2k = inv_mod((p2k % d) as u64, d as u64) as u128;
    let mut keep = vec![true; d as usize];
    for t in 0..d {
        // Representative n = r (mod 2^K), n = t (mod 3^J) (sieve3's choice;
        // every quantity used below is class-invariant).
        let u = ((t + d - (r as u128 % d)) % d) * inv2k % d;
        let n = r as u128 + p2k * u + (p2k * d) * (1u128 << 50);
        let (mut y, mut f) = (n, 0u32);
        for jj in 0..=k {
            let ob = (f + j).min(omax);
            let code = bw.get(ob, (y % POW3[ob as usize]) as u64);
            if code != 0 {
                let (o, s) = ((code & 31) as u32, (code >> 5) as u32);
                // y_j = (3^f n + d_j) / 2^j: need n (2^j 3^o - 2^s 3^f) >= 2^s d_j for all n >= 2^M.
                let dj = (y << jj) - POW3[f as usize] * n;
                let lhs = (POW3[o as usize] << jj) as i128 - (POW3[f as usize] << s) as i128;
                if lhs >= 1 {
                    let need = dj << s;
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
    keep
}

impl RowSieve {
    /// Build the (K, J) table for endpoints at depth `m` with `2^a` completions,
    /// rows for the depth-K survivors of `rules`, odd-step cap `omax`.
    pub fn build(k: u32, j: u32, m: u32, a: u32, omax: u32, rules: &Rules) -> Self {
        let t0 = std::time::Instant::now();
        assert!((2..=14).contains(&k) && k <= m && (2..=12).contains(&j) && omax <= 18 && omax >= j);
        assert!((6..=16).contains(&a) && m < 127);
        // Representative bound: 2^K 3^J 2^50 3^K < 2^128.
        assert!(k as f64 + 1.585 * (j + k) as f64 + 50.0 < 126.0);
        let d = POW3[j as usize] as u64;
        let g = ((1u128 << m) % d as u128) as u64;
        let ginv = inv_mod(g, d);
        let limb_c = [0u32, 1, 2, 3].map(|i| (((1u128 << (32 * i)) % d as u128) * ginv as u128 % d as u128) as u32);
        let minv32 = ((1u64 << 32) / d) as u32;
        let words32 = (1usize << a) / 32;
        let row_words = (d as usize).div_ceil(32) + words32 + 2;
        let rows = reachable_rows(k, rules);
        assert!(rows.len() < u16::MAX as usize);
        let mut rowmap = vec![0u16; 1 << k];
        for (i, &r) in rows.iter().enumerate() {
            rowmap[r as usize] = (i + 1) as u16;
        }
        let bw = BestWords::new(omax);
        // perm[i] = i g mod D: bit i of a row is keep[perm[i mod D]].
        let perm: Vec<u32> = (0..d).map(|i| ((i as u128 * g as u128) % d as u128) as u32).collect();
        let built: Vec<(Vec<u32>, u64)> = rows
            .par_iter()
            .map(|&r| {
                let keep = row_keep(r, k, j, m, omax, &bw);
                let mut words = vec![0u32; row_words];
                for i in 0..row_words * 32 {
                    if keep[perm[i % d as usize] as usize] {
                        words[i / 32] |= 1u32 << (i % 32);
                    }
                }
                (words, keep.iter().filter(|&&x| !x).count() as u64)
            })
            .collect();
        let mut bits = vec![u32::MAX; row_words];
        let mut excluded = 0;
        for (w, ex) in built {
            bits.extend_from_slice(&w);
            excluded += ex;
        }
        RowSieve {
            k, j, m, a, omax, d: d as u32, g: g as u32, ginv: ginv as u32, limb_c, minv32, row_words,
            rows, rowmap, bits, excluded, build_seconds: t0.elapsed().as_secs_f64(),
        }
    }

    /// The production table (K, J, omax) = (12, 9, 18) for `(m, a)`, built once
    /// per process from the baseline rules (a pure function of its inputs).
    pub fn standard(m: u32, a: u32) -> Arc<RowSieve> {
        static CACHE: OnceLock<Mutex<HashMap<(u32, u32), Arc<RowSieve>>>> = OnceLock::new();
        let cache = CACHE.get_or_init(Default::default);
        if let Some(t) = cache.lock().unwrap().get(&(m, a)) {
            return Arc::clone(t);
        }
        let t = Arc::new(RowSieve::build(12, 9, m, a, 18, &Rules::baseline()));
        cache.lock().unwrap().entry((m, a)).or_insert(t).clone()
    }

    /// `n0 (2^M)^-1 mod D`: bit `h` of the endpoint's mask is row bit `phase + h`.
    #[inline]
    pub fn phase(&self, n0: u128) -> u32 {
        ((n0 % self.d as u128) * self.ginv as u128 % self.d as u128) as u32
    }

    /// The same phase as the CUDA kernel computes it (u32 limbs, Barrett).
    pub fn phase_device_model(&self, n0: u128) -> u32 {
        let red = |x: u32| -> u32 {
            let q = ((x as u64 * self.minv32 as u64) >> 32) as u32;
            let r = x.wrapping_sub(q.wrapping_mul(self.d));
            if r >= self.d { r - self.d } else { r }
        };
        let mut s = 0u32;
        for i in 0..4 {
            s += red((n0 >> (32 * i)) as u32) * self.limb_c[i];
        }
        red(s)
    }

    /// Compact row of an endpoint (0: unreachable residue, keep everything).
    #[inline]
    pub fn row_index(&self, n0: u128) -> usize {
        self.rowmap[(n0 as usize) & ((1usize << self.k) - 1)] as usize
    }

    /// 64 completion bits `h = 64 w .. 64 w + 63` of an endpoint (`h = 0` kept).
    #[inline(always)]
    pub fn word64(&self, row: usize, phase: u32, w: usize) -> u64 {
        let i = phase as usize + 64 * w;
        let q = row * self.row_words + i / 32;
        let off = (i % 32) as u32;
        let b = &self.bits;
        let f = |lo: u32, hi: u32| -> u32 { if off == 0 { lo } else { (lo >> off) | (hi << (32 - off)) } };
        let v = f(b[q], b[q + 1]) as u64 | (f(b[q + 1], b[q + 2]) as u64) << 32;
        if w == 0 { v | 1 } else { v }
    }

    /// Does the table keep completion `n`?  (Same decision as the mask bit.)
    pub fn keeps(&self, n: u128) -> bool {
        let n0 = n & ((1u128 << self.m) - 1);
        let h = (n >> self.m) as usize;
        if h == 0 {
            return true;
        }
        assert!(h < 1usize << self.a, "n beyond the 2^(M+A) range");
        (self.word64(self.row_index(n0), self.phase(n0), h / 64) >> (h % 64)) & 1 == 1
    }

    pub fn device_bytes(&self) -> usize {
        self.bits.len() * 4 + self.rowmap.len() * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reachable_rows_at_k12_are_the_144_baseline_survivors() {
        let rows = reachable_rows(12, &Rules::baseline());
        assert_eq!(rows.len(), 144);
        assert!(rows.iter().all(|r| r & 3 == 3));
        // Improved rules reach a subset.
        let imp = reachable_rows(12, &Rules::improved(8));
        assert!(imp.iter().all(|r| rows.binary_search(r).is_ok()));
    }

    #[test]
    fn phase_layout_and_device_model_agree() {
        // A small table keeps the test fast: K = 8, J = 5.
        let s = RowSieve::build(8, 5, 62, 10, 12, &Rules::baseline());
        let bw = BestWords::new(s.omax);
        let keeps: Vec<Vec<bool>> = s.rows.iter().map(|&r| row_keep(r, s.k, s.j, s.m, s.omax, &bw)).collect();
        let mut x = 0x1234_5678_9abc_def1u128;
        for _ in 0..20000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) & ((1u128 << 62) - 1);
            let n0 = x | 3;
            assert_eq!(s.phase(n0), s.phase_device_model(n0));
            let (row, ph) = (s.row_index(n0), s.phase(n0));
            for h in [1usize, 2, 63, 64, 65, 500, 1023] {
                let bit = (s.word64(row, ph, h / 64) >> (h % 64)) & 1 == 1;
                // Direct decision: keep[n mod D] of this row.
                let n = n0 + ((h as u128) << 62);
                let want = row == 0 || keeps[row - 1][(n % s.d as u128) as usize];
                assert_eq!(bit, want, "n0={n0} h={h}");
                assert_eq!(s.keeps(n), want);
            }
            assert!(s.keeps(n0));
        }
    }

    #[test]
    fn excluded_classes_have_forward_certificates() {
        // Every exclusion of a small table, at the smallest member >= 2^M, is
        // re-derived by an independent inverse search and forward iteration.
        use crate::arith::t_step;
        let m = 40;
        let s = RowSieve::build(8, 5, m, 10, 12, &Rules::baseline());
        let d = s.d as u128;
        let l = 256u128 * d;
        let mut checked = 0;
        for &r in s.rows.iter().take(6) {
            for t in 0..d {
                let x0 = (0..d).map(|u| r as u128 + 256 * u).find(|x| x % d == t).unwrap();
                let n = x0 + l * ((1u128 << m) - x0).div_ceil(l);
                if s.keeps(n) {
                    continue;
                }
                checked += 1;
                // search: inverse words from T^j(n), j <= 8, odd budget f_j + 5 (cap 12)
                let (mut y, mut f) = (n, 0u32);
                let mut found = false;
                'depth: for _j in 0..=8 {
                    let budget = (f + 5).min(12);
                    let mut stack = vec![(y, 0u32, 0u32)];
                    while let Some((v, st, o)) = stack.pop() {
                        if st >= 1 && v >= 1 && v < n {
                            let mut q = v;
                            for _ in 0..st { q = t_step(q); }
                            assert_eq!(q, y);
                            found = true;
                            break 'depth;
                        }
                        // Any descendant q has q + 1 >= (v + 1)(2/3)^r with r odd steps left.
                        let r = budget - o;
                        if (v + 1) << r >= POW3[r as usize] * (n + 1) { continue; }
                        stack.push((2 * v, st + 1, o));
                        if r > 0 && v % 3 == 2 { stack.push(((2 * v - 1) / 3, st + 1, o + 1)); }
                    }
                    if y & 1 == 1 { f += 1; }
                    y = t_step(y);
                }
                assert!(found, "no certificate for excluded n={n}");
            }
        }
        assert!(checked > 100);
    }
}
