//! Step 2: B-step lookahead bitvectors and the mod-9 preimage sieve
//! (Angeltveit §4).
//!
//! For an endpoint `(n0, f, m)` at depth `M = N - A`, the `2^A` completions
//! `n = n0 + h 2^M` have `T^M(n) = m + 3^f h`, so the next `B` parity steps of
//! completion `h` are fixed by `(m + 3^f h) mod 2^B`.  `BV_i` (for
//! `f = offset + i`) has a bit set for every `B`-bit pattern whose path does
//! *not* dip below the start within the window (descent, mod-3 path merge
//! and OEE, all in units of `log2 / 306`).  Bits are stored at position
//! `pattern * 3^-f mod 2^B`, which makes the `2^A` completions of one
//! endpoint consecutive: one funnel shift plus AND per 64 completions.
//!
//! Patterns are in the "+1" representation `T^M(n) + 1` used by the
//! reference code: `x` even means the next step is odd.
//!
//! **Soundness fix.**  The mod-3 merge credit uses Lemma 2.8 (`y = 2 mod 3`
//! iff an even number of even steps follow the last odd step).  While the
//! window is still in its leading even run, that count includes the
//! endpoint's trailing evens *before* the window, which a bitvector indexed
//! only by `f` cannot see.  The reference program starts the count at zero,
//! so for endpoints with an odd trailing-even count it credits merges that do
//! not exist (e.g. n = 55,795,739 at N = 26: 386 unjustified exclusions out
//! of 857,601; see tests/bv_audit.rs).  We build two families of arrays,
//! selected by the parity of the endpoint's trailing evens, which is exact
//! (`BvMode::Split`).  `BvMode::Shared` keeps one array per class and gives
//! no mod-3 credit before the window's first odd step: sound for every
//! endpoint at half the memory.  `BvMode::Legacy` reproduces the reference.

use crate::arith::{DEN, NUM, POW3};
use crate::rules::MixTables;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Extra merge credits baked into the bitvectors (zero runtime cost).
///
/// Inside the window, `y = T^(M+j)(n)` satisfies `y = T^j(x0 - 1) (mod 3^t)`
/// whenever the window has taken `f_w >= t` odd steps (the `3^(f) z` term of
/// `T^j(z) = (3^f z + c)/2^j` vanishes mod 3^t), so the pattern alone fixes
/// `y mod 3^min(f_w, J)`.  With that residue we credit
///   * `pm_inf`: the inverse-odd chain of length `t = min(v3(y+1), f_w)`,
///     dip `+179 t` (Angeltveit's mod-3 credit is `t = 1`), and
///   * `mix`: the tabulated mixed word `(s, o)`, dip `+485 o - 306 s`.
/// Partners are compared with `n` exactly as in Angeltveit's Theorem 4.4:
/// a strict dip excess of one unit (1/306 bit) absorbs the `+1` terms.
#[derive(Clone, Default)]
pub struct WindowRules {
    pub pm_inf: bool,
    pub mix: Option<Arc<MixTables>>,
}

impl WindowRules {
    pub fn label(&self) -> String {
        let mut v = vec!["descent+pm1+oee".to_string()];
        if self.pm_inf {
            v.push("pm_inf".into());
        }
        if let Some(t) = &self.mix {
            v.push(format!("mix(3^{})", t.j));
        }
        v.join("+")
    }
}

#[inline]
fn v3(mut x: u64) -> u32 {
    let mut t = 0;
    while x != 0 && x % 3 == 0 {
        x /= 3;
        t += 1;
    }
    t
}

/// Inverse of 3 in Z/2^64.
const ONETHIRD64: u64 = 0xAAAA_AAAA_AAAA_AAAB;

/// `mod9[r]` bit `h` set iff `(r + h 2^M) mod 9` is not 2, 4, 5 or 8.
fn mod9_masks(m: u32, a: u32, words: usize) -> Vec<u64> {
    let mut mod9 = vec![0u64; 9 * words];
    let step = ((1u128 << m) % 9) as usize;
    for r in 0..9 {
        let mut red = r;
        for h in 0..(1usize << a) {
            if matches!(red, 0 | 1 | 3 | 6 | 7) {
                mod9[r * words + h / 64] |= 1 << (h % 64);
            }
            red = (red + step) % 9;
        }
    }
    mod9
}

/// How the trailing-even state of an endpoint is handled (see module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum BvMode {
    /// Reference construction: one array per class, count starts at 0 (unsound).
    Legacy,
    /// Two arrays per class, selected by the endpoint's trailing-even parity (exact).
    Split,
    /// One array per class with no mod-3 credit before the window's first odd
    /// step: sound for every endpoint, half the memory of `Split`.
    Shared,
}

/// `start_evens` value meaning "parity unknown": no pm1 credit until the
/// window has taken an odd step.
pub const START_UNKNOWN: u32 = 2;

impl BvMode {
    pub fn families(self) -> usize {
        if self == BvMode::Split {
            2
        } else {
            1
        }
    }
    /// `start_evens` for the array of family `fam`.
    pub fn start(self, fam: usize) -> u32 {
        match self {
            BvMode::Legacy => 0,
            BvMode::Split => fam as u32,
            BvMode::Shared => START_UNKNOWN,
        }
    }
}

#[derive(Clone)]
pub struct Lookahead {
    pub m: u32,
    pub a: u32,
    /// Base lookahead window; `bs[i]` is the window of class `i` (larger for
    /// the heavy low-f classes with `with_class_windows`).
    pub b: u32,
    pub bs: Vec<u32>,
    pub num_bv: usize,
    pub mode: BvMode,
    /// Arrays per class: 2 for `Split`, else 1.
    pub families: usize,
    /// Smallest possible `f` at depth `M` for a survivor: floor(306 M / 485) + 1.
    pub offset: u32,
    /// `3^-(offset+i) mod 2^32` and `mod 2^64` (the 64-bit form indexes B > 32).
    pub magic: Vec<u32>,
    pub magic64: Vec<u64>,
    /// Array `i` keeps patterns with `dip <= cutoffs[i]`.
    pub cutoffs: Vec<i32>,
    pub window: WindowRules,
    /// `families * num_bv` arrays (index `family * num_bv + i`) of `words`
    /// u64 each (2^B bits + 2^A bits of wraparound).
    pub bv: Vec<u64>,
    pub words: usize,
    /// `mod9[r]` bit `h` set iff `(r + h 2^M) mod 9` is not 2, 4, 5 or 8.
    pub mod9: Vec<u64>,
    pub mod9_words: usize,
    /// Per array: fraction of set bits, and set-bit fractions restricted to
    /// patterns allowed under OEE states 2 and 1 (see `crate::tail`).
    pub density: Vec<f64>,
    pub density_oee2: Vec<f64>,
    pub density_oee1: Vec<f64>,
}

/// Does the window starting at `i` hold an odd run followed by two evens?
#[inline]
fn odd_run_then_two_evens(updown: u64, i: u32, b: u32) -> bool {
    let a = ((updown >> i) + 1).trailing_zeros();
    if i + a + 2 >= b {
        return false;
    }
    (updown >> (i + a)) & 0b11 == 0
}

/// Maximal dip (in 1/306 bits) reachable inside the window for pattern `x0`,
/// given the parity of the even steps already taken since the last odd step.
#[inline]
pub fn window_dip(x0: u64, b: u32, start_evens: u32) -> i32 {
    window_dip_ext(x0, b, start_evens, &WindowRules::default())
}

/// Largest window: the iterate `x < 2^B (3/2)^B` must fit in a u64.
pub const MAX_B: u32 = 39;

/// `window_dip` with the extra credits of `wr`.
#[inline]
pub fn window_dip_ext(x0: u64, b: u32, start_evens: u32, wr: &WindowRules) -> i32 {
    debug_assert!(b <= MAX_B);
    // Pattern 0 stands for T^M(n)+1 = 0 (mod 2^B); iterate the representative
    // 2^B so that x - 1 >= 0 carries the true residues mod 3^fw.
    let x0 = if x0 == 0 { 1u64 << b } else { x0 };
    let mut updown = 0u64;
    let mut x = x0;
    for i in 0..b {
        if x & 1 == 0 {
            x = x + x / 2;
            updown |= 1 << i;
        } else {
            x = x / 2 + 1;
        }
    }
    let (num, den) = (NUM as i32, DEN as i32);
    let extra = wr.pm_inf || wr.mix.is_some();
    let known = start_evens != START_UNKNOWN;
    let (mut cur, mut max, mut evens, mut fw) = (0i32, 0i32, if known { start_evens } else { 0 }, 0u32);
    let mut x = x0;
    for i in 0..b {
        if updown & (1u64 << i) != 0 {
            if cur + num > max && odd_run_then_two_evens(updown, i, b) {
                max = cur + num;
            }
            cur -= den - num;
            evens = 0;
            fw += 1;
            x = x + x / 2;
        } else {
            cur += num;
            evens += 1;
            if evens % 2 == 0 && (known || fw > 0) && cur + (den - num) > max {
                max = cur + (den - num);
            }
            x = x / 2 + 1;
        }
        max = max.max(cur);
        if extra && fw > 0 {
            // x = y + 1 (mod 3^fw) for the true iterate y
            if wr.pm_inf {
                let t = v3(x).min(fw) as i32;
                max = max.max(cur + (den - num) * t);
            }
            if let Some(tab) = &wr.mix {
                let jp = fw.min(tab.j);
                let r = (x - 1) % POW3[jp as usize] as u64;
                if let Some((s, o, _)) = tab.lookup(jp, r) {
                    max = max.max(cur + den * o as i32 - num * s as i32);
                }
            }
        }
    }
    max
}

impl Lookahead {
    pub fn new(m: u32, a: u32, b: u32, num_bv: usize) -> Self {
        Self::build(m, a, b, num_bv, BvMode::Split)
    }

    /// Angeltveit's original single-family construction (see module docs).
    pub fn legacy(m: u32, a: u32, b: u32, num_bv: usize) -> Self {
        Self::build(m, a, b, num_bv, BvMode::Legacy)
    }

    pub fn build(m: u32, a: u32, b: u32, num_bv: usize, mode: BvMode) -> Self {
        Self::build_ext(m, a, b, num_bv, mode, &WindowRules::default())
    }

    /// Parameters, mod-9 masks and cutoffs without the arrays: completions
    /// are then tested by evaluating their window directly (any B <= 39).
    pub fn meta(m: u32, a: u32, b: u32, num_bv: usize, mode: BvMode, wr: &WindowRules) -> Self {
        Self::make(m, a, b, num_bv, mode, wr, false)
    }

    pub fn build_ext(m: u32, a: u32, b: u32, num_bv: usize, mode: BvMode, wr: &WindowRules) -> Self {
        Self::make(m, a, b, num_bv, mode, wr, true)
    }

    fn make(m: u32, a: u32, b: u32, num_bv: usize, mode: BvMode, wr: &WindowRules, arrays: bool) -> Self {
        assert!(a >= 6 && a <= b && b <= MAX_B, "need 6 <= A <= B <= {MAX_B}");
        assert!(!arrays || b <= 30, "host-built arrays are limited to B <= 30");
        let families = mode.families();
        let offset = m * NUM / DEN + 1;
        let first_cutoff = (DEN * offset - NUM * m) as i32;
        let mut magic64 = Vec::with_capacity(num_bv);
        let mut mg = 1u64;
        for i in 1..(offset + num_bv as u32) {
            mg = mg.wrapping_mul(ONETHIRD64);
            if i >= offset {
                magic64.push(mg);
            }
        }
        let magic: Vec<u32> = magic64.iter().map(|&x| x as u32).collect();
        let cutoffs: Vec<i32> = (0..num_bv).map(|i| first_cutoff + i as i32 * DEN as i32).collect();
        let tail_words = ((1usize << a) + 63) / 64;
        let mod9 = mod9_masks(m, a, tail_words);
        if !arrays {
            return Lookahead {
                m,
                a,
                b,
                bs: vec![b; num_bv],
                num_bv,
                mode,
                families,
                offset,
                magic,
                magic64,
                cutoffs,
                window: wr.clone(),
                bv: Vec::new(),
                words: (1usize << (b - 6)) + tail_words,
                mod9,
                mod9_words: tail_words,
                density: Vec::new(),
                density_oee2: Vec::new(),
                density_oee1: Vec::new(),
            };
        }
        let words = (1usize << (b - 6)) + tail_words;
        let mask = (1u64 << b) - 1;
        let narr = families * num_bv;
        let bv: Vec<AtomicU64> = (0..words * narr).map(|_| AtomicU64::new(0)).collect();
        let nb = num_bv;
        // set-bit counts: total, OEE-state-2 allowed, OEE-state-1 allowed (f even/odd)
        let counts: Vec<[u64; 4]> = (0..(1u64 << b))
            .into_par_iter()
            .fold(
                || vec![[0u64; 4]; narr],
                |mut acc, x| {
                  for fam in 0..families {
                    let dip = window_dip_ext(x, b, mode.start(fam), wr);
                    for i in 0..nb {
                        if dip > cutoffs[i] {
                            continue;
                        }
                        let arr = fam * nb + i;
                        let index = (x.wrapping_mul(magic[i] as u64)) & mask;
                        let base = arr * words;
                        bv[base + (index / 64) as usize].fetch_or(1 << (index % 64), Ordering::Relaxed);
                        if ((index / 64) as usize) < tail_words {
                            bv[base + (index / 64) as usize + (1 << (b - 6))]
                                .fetch_or(1 << (index % 64), Ordering::Relaxed);
                        }
                        acc[arr][0] += 1;
                        // OEE state 2: the next step must be odd (x even).
                        if x & 1 == 0 {
                            acc[arr][1] += 1;
                        }
                        // OEE state 1: the next two steps must not both be even,
                        // i.e. T^M(n) != 0 (mod 4) <=> x != 1 (mod 4).
                        if x & 3 != 1 {
                            acc[arr][2] += 1;
                        }
                    }
                  }
                    acc
                },
            )
            .reduce(
                || vec![[0u64; 4]; narr],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(&b) {
                        for j in 0..4 {
                            x[j] += y[j];
                        }
                    }
                    a
                },
            );
        let total = (1u64 << b) as f64;
        let density = counts.iter().map(|c| c[0] as f64 / total).collect();
        let density_oee2 = counts.iter().map(|c| c[1] as f64 / total).collect();
        let density_oee1 = counts.iter().map(|c| c[2] as f64 / total).collect();
        let bv = bv.into_iter().map(|w| w.into_inner()).collect();

        Lookahead {
            m,
            a,
            b,
            bs: vec![b; num_bv],
            num_bv,
            mode,
            families,
            offset,
            magic,
            magic64,
            cutoffs,
            window: wr.clone(),
            bv,
            words,
            mod9,
            mod9_words: tail_words,
            density,
            density_oee2,
            density_oee1,
        }
    }

    pub fn has_arrays(&self) -> bool {
        !self.bv.is_empty()
    }

    /// Per-class windows: class `i < windows.len()` gets `windows[i]` (the
    /// low-f classes carry nearly all GPU time).  Arrays are then built on
    /// the device.
    pub fn with_class_windows(mut self, windows: &[u32]) -> Self {
        assert!(!self.has_arrays(), "per-class windows need device-built arrays");
        for (b, &w) in self.bs.iter_mut().zip(windows) {
            assert!(w >= self.a && w <= MAX_B, "class window {w} must be in [A, {MAX_B}]");
            *b = w;
        }
        self
    }

    /// u32 words of one array of class `i`: 2^B_i bits plus 2^A wraparound bits.
    pub fn array_words(&self, class: usize) -> usize {
        (1usize << (self.bs[class] - 5)) + 2 * self.mod9_words
    }

    /// Device layout: u32-word offset of array `(family, class)` at index
    /// `family * num_bv + class`, and the total.  For a uniform window this is
    /// exactly the host layout.
    pub fn device_layout(&self) -> (Vec<usize>, usize) {
        let mut off = Vec::with_capacity(self.families * self.num_bv);
        let mut total = 0;
        for _fam in 0..self.families {
            for i in 0..self.num_bv {
                off.push(total);
                total += self.array_words(i);
            }
        }
        (off, total)
    }

    /// Would array `(f, ne)` keep the completion whose window pattern (low B
    /// bits of `T^M(n) + 1`) is `x`?  Same decision as the stored bit.
    #[inline]
    pub fn keeps(&self, f: u32, ne: u8, x: u64) -> bool {
        let Some(i) = self.bigindex(f) else { return true };
        let b = self.bs[i];
        window_dip_ext(x & ((1u64 << b) - 1), b, self.mode.start(self.family(ne)), &self.window) <= self.cutoffs[i]
    }

    /// Bitvector class for endpoint `f` (None: no array, only mod 9 applies).
    #[inline(always)]
    pub fn bigindex(&self, f: u32) -> Option<usize> {
        let i = f.checked_sub(self.offset)? as usize;
        (i < self.num_bv).then_some(i)
    }

    /// Family of an endpoint with trailing-even state `ne` (0 in legacy mode).
    #[inline(always)]
    pub fn family(&self, ne: u8) -> usize {
        if self.families == 2 {
            (ne & 1) as usize
        } else {
            0
        }
    }

    /// Array used by an endpoint `(f, ne)`.
    #[inline(always)]
    pub fn array(&self, f: u32, ne: u8) -> Option<usize> {
        Some(self.family(ne) * self.num_bv + self.bigindex(f)?)
    }

    /// 64 completion bits starting at word `w`, for an endpoint using array `arr`.
    #[inline(always)]
    pub fn completion_word(&self, arr: Option<usize>, start: u64, r9: usize, w: usize) -> u64 {
        let mut t = self.mod9[r9 * self.mod9_words + w];
        if let Some(i) = arr {
            let base = i * self.words + (start / 64) as usize + w;
            let off = (start % 64) as u32;
            let lo = self.bv[base];
            let v = if off == 0 { lo } else { (lo >> off) | (self.bv[base + 1] << (64 - off)) };
            t &= v;
        }
        t
    }

    /// Permuted start position of an endpoint's completions in array `arr`.
    #[inline(always)]
    pub fn start(&self, arr: usize, m: u128) -> u64 {
        let x = (m + 1) as u32;
        (self.magic[arr % self.num_bv].wrapping_mul(x) as u64) & ((1u64 << self.b) - 1)
    }

    /// The arrays as little-endian u32 words, the layout the CUDA kernel reads.
    pub fn bv_u32(&self) -> &[u32] {
        // u64 little-endian words viewed as consecutive u32 pairs
        unsafe { std::slice::from_raw_parts(self.bv.as_ptr() as *const u32, self.bv.len() * 2) }
    }
    pub fn mod9_u32(&self) -> &[u32] {
        unsafe { std::slice::from_raw_parts(self.mod9.as_ptr() as *const u32, self.mod9.len() * 2) }
    }
}
