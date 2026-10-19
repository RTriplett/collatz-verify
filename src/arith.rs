//! Exact integer helpers shared by every stage.
//!
//! The shortcut map is `T(n) = n/2` (n even) and `T(n) = (3n+1)/2` (n odd).
//! For a residue class `n = n0 + 2^k a` with `m = T^k(n0)` and `f` odd steps,
//! `T^k(n) = m + 3^f a` (Lemma 2.3 of Angeltveit).  All sieve decisions are
//! statements about such affine families.

/// `NUM / DEN = 306 / 485` is a rational under-approximation of `log 2 / log 3`
/// (`485/306 > log2 3` by 4.8e-6).  `DEN*f <= NUM*k` therefore implies
/// `3^f < 2^k`, which is Angeltveit's "induction" form of the descent sieve.
pub const NUM: u32 = 306;
pub const DEN: u32 = 485;

pub const MAX_POW3: usize = 81;

/// `POW3[i] = 3^i` for `i <= 80` (3^80 < 2^127).
pub const POW3: [u128; MAX_POW3] = {
    let mut t = [1u128; MAX_POW3];
    let mut i = 1;
    while i < MAX_POW3 {
        t[i] = t[i - 1] * 3;
        i += 1;
    }
    t
};

/// Angeltveit's induction sieve: `306 k >= 485 f` implies every member of the
/// class has descended below its start after `k` steps (Theorem 4.1, valid
/// once all intermediate values are >= 99781).
#[inline(always)]
pub fn slope_descends(k: u32, f: u32) -> bool {
    NUM * k >= DEN * f
}

/// Exact test `2^s * 3^e <= 2^k`: the slope of a partner family is no larger
/// than the slope `2^k` of the member family.
#[inline]
pub fn weight_le_pow2k(s: u32, e: u32, k: u32) -> bool {
    if s > k {
        return false;
    }
    let room = k - s;
    if room >= 127 {
        return true;
    }
    (e as usize) < MAX_POW3 && POW3[e as usize] <= (1u128 << room)
}

/// `x mod 3^j` for `j <= 40` using 64-bit arithmetic only.
#[inline]
pub fn mod_pow3(x: u128, j: u32) -> u64 {
    let p = POW3[j as usize] as u64;
    let hi = (x >> 64) as u64;
    let lo = x as u64;
    // 2^64 mod p
    let r64 = ((1u128 << 64) % p as u128) as u64;
    let a = ((hi % p) as u128 * r64 as u128 % p as u128) as u64;
    ((a as u128 + (lo % p) as u128) % p as u128) as u64
}

/// Compare `x * 2^s` against `y * c + d` without overflow (all inputs < 2^128,
/// `s < 128`, `c < 2^64`).  Returns `x*2^s < y*c + d`.
#[inline]
pub fn shl_lt_mul_add(x: u128, s: u32, y: u128, c: u64, d: u128) -> bool {
    let lhs = U256::from(x).shl(s);
    let rhs = U256::mul_u64(y, c).add(U256::from(d));
    lhs.lt(&rhs)
}

/// Minimal unsigned 256-bit value (hi, lo).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct U256 {
    pub hi: u128,
    pub lo: u128,
}

impl U256 {
    #[inline]
    pub fn from(x: u128) -> Self {
        U256 { hi: 0, lo: x }
    }
    #[inline]
    pub fn shl(self, s: u32) -> Self {
        if s == 0 {
            self
        } else if s < 128 {
            U256 { hi: (self.hi << s) | (self.lo >> (128 - s)), lo: self.lo << s }
        } else {
            U256 { hi: self.lo << (s - 128), lo: 0 }
        }
    }
    #[inline]
    pub fn mul_u64(y: u128, c: u64) -> Self {
        let y0 = y as u64 as u128;
        let y1 = (y >> 64) as u128;
        let c = c as u128;
        let p0 = y0 * c;
        let p1 = y1 * c + (p0 >> 64);
        U256 { hi: p1 >> 64, lo: (p1 << 64) | (p0 as u64 as u128) }
    }
    /// Full 128x128 -> 256-bit product.
    #[inline]
    pub fn mul(a: u128, b: u128) -> Self {
        let (a0, a1) = (a as u64 as u128, a >> 64);
        let (b0, b1) = (b as u64 as u128, b >> 64);
        let p00 = a0 * b0;
        let p01 = a0 * b1;
        let p10 = a1 * b0;
        let p11 = a1 * b1;
        let mid = (p00 >> 64) + (p01 as u64 as u128) + (p10 as u64 as u128);
        let lo = (p00 as u64 as u128) | (mid << 64);
        let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
        U256 { hi, lo }
    }
    #[inline]
    pub fn add(self, o: U256) -> Self {
        let (lo, c) = self.lo.overflowing_add(o.lo);
        U256 { hi: self.hi + o.hi + c as u128, lo }
    }
    #[inline]
    pub fn lt(&self, o: &U256) -> bool {
        (self.hi, self.lo) < (o.hi, o.lo)
    }
}

/// One shortcut step.
#[inline(always)]
pub fn t_step(n: u128) -> u128 {
    if n & 1 == 0 {
        n >> 1
    } else {
        n + (n >> 1) + 1
    }
}

/// Reference iterate: `(T^k(n), odd steps)`; panics on overflow.
pub fn t_iter(mut n: u128, k: u32) -> (u128, u32) {
    let mut f = 0;
    for _ in 0..k {
        if n & 1 == 1 {
            n = n.checked_mul(3).and_then(|x| x.checked_add(1)).expect("overflow") / 2;
            f += 1;
        } else {
            n >>= 1;
        }
    }
    (n, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approx_is_conservative() {
        // 485 f <= 306 k  =>  3^f < 2^k for every k the code can reach.
        for k in 1..=120u32 {
            for f in 0..=k.min(80) {
                if !slope_descends(k, f) {
                    continue;
                }
                assert!((f as f64) * 3f64.log2() < k as f64, "k={k} f={f}");
            }
        }
    }

    #[test]
    fn mod_pow3_matches() {
        let xs = [0u128, 1, 2, 3, 12345678901234567890123456789, u128::MAX, 3u128.pow(70) - 1];
        for &x in &xs {
            for j in 1..=40 {
                assert_eq!(mod_pow3(x, j) as u128, x % POW3[j as usize]);
            }
        }
    }

    #[test]
    fn u256_mul() {
        let a = 0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFFu128;
        let p = U256::mul(a, a); // (2^128-1)^2 = 2^256 - 2^129 + 1
        assert_eq!(p.lo, 1);
        assert_eq!(p.hi, u128::MAX - 1);
        let q = U256::mul(3u128.pow(70), 1u128 << 60);
        assert_eq!(q, U256::from(3u128.pow(70)).shl(60));
    }

    #[test]
    fn u256_compare() {
        let x = 3u128.pow(70);
        assert!(shl_lt_mul_add(x, 10, x, 1025, 0));
        assert!(!shl_lt_mul_add(x, 10, x, 1024, 0));
        assert!(shl_lt_mul_add(x, 10, x, 1024, 1));
        assert!(!shl_lt_mul_add(u128::MAX, 100, u128::MAX, u64::MAX, u128::MAX));
    }
}
