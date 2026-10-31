//! v3-sound: independent checker, extended from v2-sound's v2check for c3-final
//! (two-phase digest layouts t2/t12/t14/t16-21/t24/t31, forced launch chunks,
//! synthetic-start delivery endpoints counted separately).  Original header:
//!
//! v2-sound: independent checker of the c-final configuration.
//!
//! SHARES NO CODE WITH THE cv CRATE (no `cv::` import anywhere in this file):
//! plain T steps, its own window-dip rule (descent, pm1, OEE, pm_inf and the
//! mixed-word credit, implemented from the definitions in docs/ALGORITHM.md
//! section 4), its own mixed-word credit table (enumerated forward parity
//! masks), its own stop rule of the jd tails (H = 485 F' - 306 M' recomputed
//! from scratch, the tested window's low bits from exact iteration), and a
//! generic witness search (descent; pm1 / pm_inf / OEE special forms; bounded
//! inverse-tree DFS) whose partners are re-verified by forward iteration.
//!
//! Input: v2dump's PREFIX.eps (endpoints) and PREFIX.jsonl (GPU results of the
//! real kernels).  For every endpoint it recomputes T^M(n0) and F, the screen
//! (bitvector window dip <= 485 F - 306 M, mod 9 or the dumped row-table keep
//! words, carried OEE mask, h range), every case's stop (jumps, hash, flag),
//! and compares per endpoint (single-endpoint launches) and per batch
//! (including the warp-queue per-case digest) with the GPU.  At every stop and
//! (on audited endpoints) every exclusion it demands an explicit witness.
//!
//! v-n73-sound (N73 extension): a variant's completion range is selected by the n INTERVAL its
//! range code names at the dump's N (0 full [0, 2^N), 1 top [2^(N-1), 2^N), 2 bottom [0, 2^(N-1)),
//! 3 u2 [2^(N-2), 2^N), 4 l2 [0, 2^(N-2)), 5 m2 [2^(N-2), 2^(N-1))), tested on n = n0 + h 2^M itself, not
//! through h bounds; the h set this implies is derived per endpoint and must be one contiguous range,
//! the same for every endpoint (reported).  Exclusion audits whose trajectory would overflow u128 are
//! counted (excl_skipped) instead of being skipped silently; every endpoint must have n0 < 2^M.
//!
//! v2check PREFIX [--audit-eps N] [--excl-eps N] [--hbias D] [--span-cut S] [--json OUT]
//!   --audit-eps N  witness-audit the stops of the first N endpoints per class (default all)
//!   --excl-eps N   witness-audit every exclusion of the first N endpoints per class (default 50)
//!   --hbias D      NEGATIVE CONTROL: the checker's own rule uses H - D (expect witness failures)
//!   --span-cut S   NEGATIVE CONTROL: witnesses searched only up to the window end minus S
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, Read};
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Arithmetic.

const P3: [u128; 81] = {
    let mut t = [0u128; 81];
    let mut i = 0;
    let mut p = 1u128;
    while i < 81 {
        t[i] = p;
        if i < 80 {
            p *= 3;
        }
        i += 1;
    }
    t
};

/// T(y) = y/2 or (3y+1)/2, None on u128 overflow.
#[inline(always)]
fn tstep(y: u128) -> Option<u128> {
    if y & 1 == 1 {
        y.checked_add(y >> 1)?.checked_add(1)
    } else {
        Some(y >> 1)
    }
}

#[inline]
fn titer(mut y: u128, k: u32) -> Option<(u128, u32)> {
    let mut f = 0;
    for _ in 0..k {
        if y & 1 == 1 {
            f += 1;
        }
        y = tstep(y)?;
    }
    Some((y, f))
}

fn v3(mut x: u128) -> u32 {
    let mut t = 0;
    while x != 0 && x % 3 == 0 {
        x /= 3;
        t += 1;
    }
    t
}

fn inv_mod(a: u128, m: u128) -> u128 {
    // extended Euclid on i128
    let (mut t, mut nt) = (0i128, 1i128);
    let (mut r, mut nr) = (m as i128, (a % m) as i128);
    while nr != 0 {
        let q = r / nr;
        (t, nt) = (nt, t - q * nt);
        (r, nr) = (nr, r - q * nr);
    }
    assert_eq!(r, 1);
    (if t < 0 { t + m as i128 } else { t }) as u128
}

/// 256-bit T steps for flagged rechecks: (lo, hi) little-endian u128 halves.
fn recheck_wide(n: u128, max_steps: u64) -> Option<u64> {
    let (mut lo, mut hi) = (n, 0u128);
    for s in 1..=max_steps {
        if lo & 1 == 1 {
            // (3y + 1) / 2 = y + (y >> 1) + 1
            let sh_lo = (lo >> 1) | (hi << 127);
            let sh_hi = hi >> 1;
            let (a, c1) = lo.overflowing_add(sh_lo);
            let (b, c2) = a.overflowing_add(1);
            hi = hi.checked_add(sh_hi)?.checked_add(c1 as u128 + c2 as u128)?;
            lo = b;
        } else {
            lo = (lo >> 1) | (hi << 127);
            hi >>= 1;
        }
        if hi == 0 && lo < n {
            return Some(s);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Mixed-word credits (own construction).  A forward word of s steps, o of them
// odd, maps p to (3^o p + b)/2^s; its inverse p = (2^s y - b)/3^o is integral
// (and then automatically has that parity pattern) iff y = b 2^-s (mod 3^o).
// best[o][r] = max over such words valid for y = r (mod 3^o), with 2^s < 3^o,
// of the credit 485 o - 306 s.  credit(jp, y) = max over o <= jp.

const NONE: i16 = i16::MIN;

struct Mix {
    omax: u32,
    best: Vec<Vec<i16>>,
}

impl Mix {
    fn build(omax: u32) -> Self {
        let mut best: Vec<Vec<i16>> = vec![Vec::new()];
        for o in 1..=omax {
            let mo = P3[o as usize];
            let mut tab = vec![NONE; mo as usize];
            let mut s = o;
            while (1u128 << s) < P3[o as usize] {
                let credit = (485 * o as i32 - 306 * s as i32) as i16;
                let inv2s = inv_mod((1u128 << s) % mo, mo);
                // all masks: bit 0 set (first step odd), o - 1 more odd steps among 1..s
                let mut stack: Vec<(u32, u32, u128)> = vec![(1, o - 1, 1)]; // (next position, odd left, b after step 0)
                // b after step 0 (odd): b = 3*0 + 2^0 = 1
                while let Some((pos, left, b)) = stack.pop() {
                    if pos == s {
                        if left == 0 {
                            let rho = (b % mo) * inv2s % mo;
                            let e = &mut tab[rho as usize];
                            if *e < credit {
                                *e = credit;
                            }
                        }
                        continue;
                    }
                    if s - pos < left {
                        continue;
                    }
                    // even step at pos: b unchanged
                    stack.push((pos + 1, left, b));
                    if left > 0 {
                        stack.push((pos + 1, left - 1, 3 * b + (1u128 << pos)));
                    }
                }
                s += 1;
            }
            best.push(tab);
        }
        Mix { omax, best }
    }

    #[inline]
    fn credit(&self, jp: u32, y: u128) -> i32 {
        let jp = jp.min(self.omax);
        let r13 = y % P3[jp as usize];
        let mut c = NONE as i32;
        for o in 1..=jp {
            let e = self.best[o as usize][(r13 % P3[o as usize]) as usize];
            if e != NONE && (e as i32) > c {
                c = e as i32;
            }
        }
        c
    }

    /// The same credit by an inverse-word DFS on the actual value (self test).
    fn credit_dfs(&self, jp: u32, y: u128) -> i32 {
        let jp = jp.min(self.omax);
        let mut best = NONE as i32;
        let mut stack = vec![(y, 0u32, 0u32)];
        while let Some((v, s, o)) = stack.pop() {
            if o >= 1 && (1u128 << s) < P3[o as usize] {
                best = best.max(485 * o as i32 - 306 * s as i32);
            }
            if s >= 21 {
                continue;
            }
            if let Some(e) = v.checked_mul(2) {
                stack.push((e, s + 1, o));
            }
            if o < jp && v % 3 == 2 {
                stack.push(((2 * v - 1) / 3, s + 1, o + 1));
            }
        }
        best
    }
}

// ---------------------------------------------------------------------------
// Window dip (START_UNKNOWN: no knowledge of the steps before the window).
// ys[0..=w] are the actual trajectory values from the window start.  Units of
// 1/306 bit: +306 per even step, -179 per odd step; credits: pm1 (+179 after
// an even number >= 2 of even steps since an odd step inside the window), OEE
// (+306 before an odd step that starts an odd run followed by two even steps,
// all strictly inside the window: i + a + 2 < w), pm_inf (+179 t, t =
// min(v3(y + 1), f_w)), mixed word (credit(min(f_w, 13), y)).

fn window_dip(ys: &[u128], w: usize, mix: &Mix) -> i32 {
    let odd = |i: usize| ys[i] & 1 == 1;
    let (mut cur, mut best, mut evens, mut fw) = (0i32, 0i32, 0u32, 0u32);
    for i in 0..w {
        if odd(i) {
            let mut a = 0;
            while i + a < w && odd(i + a) {
                a += 1;
            }
            if i + a + 2 < w && !odd(i + a) && !odd(i + a + 1) {
                best = best.max(cur + 306);
            }
            cur -= 179;
            evens = 0;
            fw += 1;
        } else {
            cur += 306;
            evens += 1;
            if evens % 2 == 0 && fw > 0 {
                best = best.max(cur + 179);
            }
        }
        best = best.max(cur);
        if fw > 0 {
            let y = ys[i + 1];
            let t = v3(y + 1).min(fw) as i32;
            best = best.max(cur + 179 * t);
            let c = mix.credit(fw.min(13), y);
            if c != NONE as i32 {
                best = best.max(cur + c);
            }
        }
    }
    best
}

fn traj(z: u128, w: usize) -> Option<Vec<u128>> {
    let mut v = Vec::with_capacity(w + 1);
    v.push(z);
    for i in 0..w {
        v.push(tstep(v[i])?);
    }
    Some(v)
}

// ---------------------------------------------------------------------------
// The jd tail rule (documented semantics of codes 90..111).

#[derive(Clone, Copy, Debug)]
struct Code {
    w: u32,
    ka: u32,
    q: i32,
}

fn code_params(code: u32) -> Option<Code> {
    Some(match code {
        90 | 103 => Code { w: 12, ka: 0, q: 128 },
        91 | 99 | 106 | 111 => Code { w: 12, ka: 1, q: 128 },
        92 | 100 | 107 | 112 => Code { w: 12, ka: 2, q: 128 },
        93 | 104 => Code { w: 11, ka: 0, q: 32 },
        94 | 108 => Code { w: 11, ka: 1, q: 32 },
        95 | 109 => Code { w: 11, ka: 2, q: 32 },
        96 => Code { w: 10, ka: 0, q: 8 },
        97 => Code { w: 10, ka: 1, q: 8 },
        98 | 110 => Code { w: 10, ka: 2, q: 8 },
        _ => return None,
    })
}

/// Smallest K with 2^(K + W) >= 3^W.
fn growth_bits(w: u32) -> u32 {
    let mut k = 0;
    while (1u128 << (k + w)) < P3[w as usize] {
        k += 1;
    }
    k
}

#[derive(Clone, Copy, Debug, Default)]
struct CaseOut {
    jumps: u32,
    hash: u32,
    flagged: bool,
    exact: bool,    // small-base exact-descent path
    /// first trajectory position of the certified window (dip stops), or the
    /// descent position (exact path)
    win_lo: u32,
    win_hi: u32,
    overflow: bool, // checker arithmetic overflow (must never happen)
}

/// One case: n, x = T^M(n), H0 = 485 F - 306 M, at depth M.
/// NEGATIVE CONTROL model of a mutated GPU table: (delta quanta, modulus) for one code.
#[derive(Clone, Copy, Debug, Default)]
struct Mutation {
    delta: i32,
    modulus: u128,
}

fn run_case(n: u128, x0: u128, h0: i32, m: u32, c: Code, cutoff: u32, mix: &Mix, hbias: i32) -> CaseOut {
    run_case_m(n, x0, h0, m, c, cutoff, mix, hbias, Mutation::default())
}

fn run_case_m(n: u128, x0: u128, h0: i32, m: u32, c: Code, cutoff: u32, mix: &Mix, hbias: i32, mu: Mutation) -> CaseOut {
    let k = growth_bits(c.w);
    let flag_at = 1u128 << (cutoff - k);
    let w = c.w as usize;
    let mut o = CaseOut::default();
    let mut x = x0;
    if !(n >= 1u128 << 32 && n < 1u128 << 96) {
        o.exact = true;
        loop {
            if x >= flag_at {
                o.flagged = true;
                return o;
            }
            match titer(x, c.w) {
                Some((nx, _)) => x = nx,
                None => {
                    o.overflow = true;
                    return o;
                }
            }
            o.hash ^= x as u32;
            o.jumps += 1;
            if x < n {
                o.win_lo = m + o.jumps * c.w;
                o.win_hi = o.win_lo;
                return o;
            }
        }
    }
    let mut h = h0;
    loop {
        if x >= flag_at {
            o.flagged = true;
            return o;
        }
        // the tested window starts KA jumps ahead
        let (mut z, mut hz) = (x, h);
        for _ in 0..c.ka {
            let Some((nz, f)) = titer(z, c.w) else { o.overflow = true; return o; };
            hz += 485 * f as i32 - 306 * c.w as i32;
            z = nz;
        }
        let Some(ys) = traj(z, w) else { o.overflow = true; return o; };
        let dip = window_dip(&ys, w, mix);
        let mut dq = (dip / c.q) * c.q;
        if mu.delta != 0 && (z & ((1u128 << c.w) - 1)) % mu.modulus == 0 {
            // FD3 field limit: 31/127/511 quanta for W = 12/11/10
            let fmax = (1i32 << (25 - if c.w == 12 { 20 } else if c.w == 11 { 18 } else { 16 })) - 1;
            dq = (dq / c.q + mu.delta).min(fmax) * c.q;
        }
        if dq > hz - hbias {
            o.win_lo = m + (o.jumps + c.ka) * c.w;
            o.win_hi = o.win_lo + c.w;
            return o;
        }
        let Some((nx, f)) = titer(x, c.w) else { o.overflow = true; return o; };
        h += 485 * f as i32 - 306 * c.w as i32;
        x = nx;
        o.hash ^= x as u32;
        o.jumps += 1;
    }
}

// ---------------------------------------------------------------------------
// Witness search.  A witness for n at trajectory position s (y = T^s(n)) is
// either y < n, or a partner 1 <= p < n with T^a(p) = y (a >= 1), found by
// the special forms or a bounded inverse-tree DFS, re-verified forward.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum WKind {
    Descent,
    Pm1,
    PmInf,
    Oee,
    Dfs,
}

fn forward_hits(p: u128, a: u32, y: u128) -> bool {
    matches!(titer(p, a), Some((v, _)) if v == y)
}

fn dfs_partner(y: u128, n: u128, omax: u32, smax: u32) -> Option<(u128, u32)> {
    let mut stack: Vec<(u128, u32, u32)> = vec![(y, 0, 0)];
    while let Some((v, s, o)) = stack.pop() {
        if s >= 1 && v >= 1 && v < n {
            return Some((v, s));
        }
        if s == smax {
            continue;
        }
        // prune: with r more inverse odd steps, p + 1 >= (v + 1) (2/3)^r
        let r = omax - o;
        match ((v + 1).checked_mul(1u128 << r), P3[r as usize].checked_mul(n + 1)) {
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
    None
}

/// Witness at some position in [lo, hi] of the trajectory ys (ys[i] = T^i(n)).
/// Cheap forms at every position first, then the DFS.  Returns (kind, position).
fn find_witness(ys: &[u128], n: u128, lo: usize, hi: usize, omax: u32, smax: u32) -> Option<(WKind, usize)> {
    let hi = hi.min(ys.len() - 1);
    for s in lo..=hi {
        let y = ys[s];
        if y < n {
            return Some((WKind::Descent, s));
        }
        if y % 3 == 2 {
            let p = (2 * y - 1) / 3;
            if p >= 1 && p < n && forward_hits(p, 1, y) {
                return Some((WKind::Pm1, s));
            }
        }
        let t = v3(y + 1);
        if t >= 1 && t <= 80 {
            let u = (y + 1) / P3[t as usize];
            if let Some(u) = u.checked_shl(t).filter(|&v| v >> t == u) {
                let u = u - 1;
                if u >= 1 && u < n && forward_hits(u, t, y) {
                    return Some((WKind::PmInf, s));
                }
            }
        }
        // OEE (Hercher): y odd starting O^a E E -> merges with (y - 1)/2 at position s + a + 2
        if y & 1 == 1 {
            let mut a = 0;
            while s + a < ys.len() && ys[s + a] & 1 == 1 {
                a += 1;
            }
            let e = s + a + 2;
            if e <= hi && ys[s + a] & 1 == 0 && ys[s + a + 1] & 1 == 0 {
                let p = (y - 1) / 2;
                if p >= 1 && p < n {
                    // T^(a+1)(p) must equal T^(a+2)(y)
                    if forward_hits(p, a as u32 + 1, ys[e]) {
                        return Some((WKind::Oee, s));
                    }
                }
            }
        }
    }
    for s in lo..=hi {
        if let Some((p, a)) = dfs_partner(ys[s], n, omax, smax) {
            if p >= 1 && p < n && forward_hits(p, a, ys[s]) {
                return Some((WKind::Dfs, s));
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Data.

#[derive(Clone, Debug)]
struct Ep {
    n0: u128,
    m: u128,
    f: u32,
    class: u32,
    oee: u8,
    rk: Vec<u64>,
}

#[derive(Clone, Debug)]
struct Var {
    name: String,
    codes: Vec<u32>,
    rows: Vec<bool>,
    range: Vec<u8>,
    wq: Vec<String>,
}

#[derive(Default, Clone, Debug)]
struct Agg {
    cases: u64,
    iters: u64,
    hash: u32,
    flags: Vec<u128>,
    dsum: u64,
    dxor: u64,
    dcount: u64,
}

fn mix64(mut a: u64) -> u64 {
    a ^= a >> 30;
    a = a.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    a ^= a >> 27;
    a = a.wrapping_mul(0x94D0_49BB_1331_11EB);
    a ^= a >> 31;
    a
}

fn digest(n: u128, jumps: u32, hash: u32) -> u64 {
    mix64((n as u64) ^ mix64(((n >> 64) as u64) ^ mix64(((jumps as u64) << 32) | hash as u64)))
}

#[derive(Default)]
struct Counters {
    stops: AtomicU64,
    stops_audited: AtomicU64,
    flagged: AtomicU64,
    exact: AtomicU64,
    w_kind: [AtomicU64; 5],
    w_fail: AtomicU64,
    w_late: AtomicU64,
    recheck_fail: AtomicU64,
    overflow: AtomicU64,
    m_mismatch: AtomicU64,
    synthetic: AtomicU64,
    excl: [AtomicU64; 4],      // bitvector, mod9, rows, oee exclusions audited
    excl_fail: [AtomicU64; 4],
    flag_cover_fail: AtomicU64,
    excl_skipped: AtomicU64,
    n0_range_fail: AtomicU64,
}

/// Scheduler specs whose kernels return the per-case digest (sum, XOR, count of
/// mix64(n, case jumps, case hash)) in KOut.dbg[0..2]: c-sched's warp-queue
/// audit layouts 26 and 52, and c3-2ph / c3-final's two-phase DIG copies
/// (cuda/twophase.cuh TpCfg<V>::DIG: 2, 12, 14, 16-21, 24, 31).  Hard-coded here
/// from the layout tables (the checker takes nothing from cv); a wrong claim
/// can only produce a mismatch, never a false pass of the core comparison.
fn is_digest_layout(spec: &str) -> bool {
    let layout = |p: char| -> Option<u32> {
        let rest = spec.strip_prefix(p)?;
        rest.split('.').next()?.parse().ok()
    };
    if let Some(v) = layout('v') {
        return matches!(v, 26 | 52);
    }
    if let Some(t) = layout('t') {
        return matches!(t, 2 | 12 | 14 | 16..=21 | 24 | 31);
    }
    false
}

/// v-n73-sound: the n interval [lo, hi) of a range code at N (see the header).
fn n_interval(range: u8, nn: u32) -> (u128, u128) {
    let p = |e: u32| 1u128 << e;
    match range {
        0 => (0, p(nn)),
        1 => (p(nn - 1), p(nn)),
        2 => (0, p(nn - 1)),
        3 => (p(nn - 2), p(nn)),
        4 => (0, p(nn - 2)),
        5 => (p(nn - 2), p(nn - 1)),
        _ => panic!("unknown range code {range}"),
    }
}

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("selftest") {
        selftest();
        return;
    }
    if args.get(1).map(|s| s.as_str()) == Some("tables") {
        // Exhaustive: every residue r < 2^W of the uploaded tables vs this checker's
        // window dip of the representative x = r (any x = r mod 2^W gives the same dip).
        let mix = Mix::build(13);
        let v: Value = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
        for t in v.as_array().unwrap() {
            let w = t["w"].as_u64().unwrap() as usize;
            let q = t["q"].as_u64().unwrap() as i32;
            let dip: Vec<i64> = t["dip"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
            let exact: Vec<i64> = t["exact"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
            let cap = t["cap"].as_i64().map(|c| c as i32);
            let (mut bad_q, mut bad_e, mut maxd, mut clamped, mut bad_capped) = (0, 0, 0i32, 0, 0);
            for r in 0..(1usize << w) {
                // also a large representative: r + 2^W * (odd big multiplier)
                let mut mine = Vec::new();
                for rep in [r as u128, r as u128 + (0x9E37_79B9_7F4A_7C15u128 << w), r as u128 + (3u128 << 80)] {
                    let ys = traj(rep, w).unwrap();
                    mine.push(window_dip(&ys, w, &mix));
                }
                assert!(mine.iter().all(|&d| d == mine[0]), "dip depends on the representative at r={r}");
                let d = mine[0];
                maxd = maxd.max(d);
                if (d / q) * q != dip[r] as i32 { bad_q += 1; }
                if d != exact[r] as i32 { bad_e += 1; }
                // v3: c3-hyg's stored value q * min(cap, floor(dip / q)) (cap = FD3 field width)
                if let Some(c) = cap {
                    let want = q * (d / q).min(c);
                    if want != dip[r] as i32 { bad_capped += 1; }
                    if d / q > c { clamped += 1; }
                }
            }
            let db = if w == 12 { 20 } else if w == 11 { 18 } else { 16 };
            let fmax = ((1i32 << (25 - db)) - 1) * q;
            eprintln!("W={w}: {} residues; exact-dip mismatches {bad_e}, quantized (q={q}) mismatches {bad_q}; max dip {maxd} (FD3 field holds up to {fmax}): {}",
                1usize << w, if maxd <= fmax + q - 1 { "fits" } else { "OVERFLOW" });
            eprintln!("W={w}: stored = q*min(cap {:?}, floor(dip/q)) mismatches {bad_capped}; entries above the cap (clamped, conservative) {clamped}", cap);
        }
        return;
    }
    let prefix = &args[1];
    let audit_eps: usize = arg(&args, "--audit-eps", usize::MAX);
    let excl_eps: usize = arg(&args, "--excl-eps", 50);
    let hbias: i32 = arg(&args, "--hbias", 0);
    let span_cut: usize = arg(&args, "--span-cut", 0);
    // v-n73-sound NEGATIVE CONTROL: shift the lower n bound of every range with a nonzero lower bound
    // up by K completions (K 2^M), i.e. drop the first K h values of u2 / top / m2 (expect mismatches).
    let neg_range_shift: u128 = arg(&args, "--neg-range-shift", 0u64) as u128;
    let json_out: String = arg(&args, "--json", format!("{prefix}.check.json"));
    // --mutate code:delta:modulus (model of v2dump's V2_MUTATE negative control)
    let mutate: String = arg(&args, "--mutate", String::new());
    let (mu_code, mu) = if mutate.is_empty() { (0u32, Mutation::default()) } else {
        let v: Vec<i64> = mutate.split(':').map(|x| x.parse().unwrap()).collect();
        (v[0] as u32, Mutation { delta: v[1] as i32, modulus: v[2] as u128 })
    };
    let t0 = std::time::Instant::now();

    // --- read config, variants, results
    let f = std::fs::File::open(format!("{prefix}.jsonl")).expect("jsonl");
    let mut cfg = Value::Null;
    let mut vars: Vec<Var> = Vec::new();
    let mut g1: HashMap<(String, u32, usize), Value> = HashMap::new();
    let mut gb: Vec<Value> = Vec::new();
    let mut cpu: Vec<Value> = Vec::new();
    for line in std::io::BufReader::new(f).lines() {
        let v: Value = serde_json::from_str(&line.unwrap()).unwrap();
        match v["t"].as_str().unwrap() {
            "cfg" => cfg = v,
            "var" => vars.push(Var {
                name: v["name"].as_str().unwrap().to_string(),
                codes: v["codes"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect(),
                rows: v["rows"].as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect(),
                range: v["range"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u8).collect(),
                wq: v["wq"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect(),
            }),
            "g1" => {
                g1.insert((v["v"].as_str().unwrap().to_string(), v["c"].as_u64().unwrap() as u32, v["i"].as_u64().unwrap() as usize), v);
            }
            "gb" => gb.push(v),
            "cpu" => cpu.push(v),
            _ => {}
        }
    }
    let nn = cfg["n"].as_u64().unwrap() as u32;
    let a = cfg["a"].as_u64().unwrap() as u32;
    let m = cfg["m"].as_u64().unwrap() as u32;
    let offset = cfg["offset"].as_u64().unwrap() as u32;
    let windows: Vec<u32> = cfg["windows"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let cutoff = cfg["cutoff"].as_u64().unwrap() as u32;
    let words64 = cfg["words64"].as_u64().unwrap() as usize;
    let rec_bytes = cfg["rec_bytes"].as_u64().unwrap() as usize;
    assert_eq!(nn, m + a);
    // own offset: smallest f with 485 f > 306 M
    let my_offset = (306 * m) / 485 + 1;
    assert_eq!(my_offset, offset, "class offset");

    // --- endpoints
    let mut raw = Vec::new();
    std::fs::File::open(format!("{prefix}.eps")).expect("eps").read_to_end(&mut raw).unwrap();
    assert_eq!(raw.len() % rec_bytes, 0);
    let mut by_class: HashMap<u32, Vec<Ep>> = HashMap::new();
    for rec in raw.chunks(rec_bytes) {
        let n0 = u128::from_le_bytes(rec[0..16].try_into().unwrap());
        let mm = u128::from_le_bytes(rec[16..32].try_into().unwrap());
        let f = u32::from_le_bytes(rec[32..36].try_into().unwrap());
        let class = u32::from_le_bytes(rec[36..40].try_into().unwrap());
        let oee = rec[40];
        let rk = (0..words64).map(|w| u64::from_le_bytes(rec[44 + 8 * w..52 + 8 * w].try_into().unwrap())).collect();
        by_class.entry(class).or_default().push(Ep { n0, m: mm, f, class, oee, rk });
    }
    let mix = Mix::build(13);
    let cnt = Counters::default();
    let mut report = Vec::new();
    let mut mismatches = Vec::new();
    let mut classes: Vec<u32> = by_class.keys().copied().collect();
    classes.sort();

    for &class in &classes {
        let eps = &by_class[&class];
        let b = windows[class as usize] as usize;
        let f_class = offset + class;
        let h0 = 485 * f_class as i32 - 306 * m as i32;
        // --- per endpoint: own screen (bitvector, mod 9, oee) and per-case results per code
        let codes: Vec<u32> = {
            let mut c: Vec<u32> = vars.iter().map(|v| v.codes[class as usize]).collect();
            c.sort();
            c.dedup();
            c
        };
        struct EpRes {
            aggs: Vec<(Agg, (usize, usize))>,          // per variant (v-n73-sound: with the derived h range)
            excl: [u64; 4],          // completions excluded by bitvector, mod 9, rows, oee
        }
        let results: Vec<EpRes> = eps.par_iter().enumerate().map(|(ei, ep)| {
            // own T^M(n0) and F
            if ep.n0 >> m != 0 {
                cnt.n0_range_fail.fetch_add(1, Ordering::Relaxed);
            }
            let (mm, ff) = titer(ep.n0, m).expect("T^M(n0)");
            let synthetic = mm != ep.m && (ep.m >> 110) == 1 && ff == ep.f && ff == f_class;
            if synthetic {
                cnt.synthetic.fetch_add(1, Ordering::Relaxed);
            } else if mm != ep.m || ff != ep.f || ff != f_class {
                cnt.m_mismatch.fetch_add(1, Ordering::Relaxed);
            }
            let comps = 1usize << a;
            let cutoff_f = 485 * ep.f as i32 - 306 * m as i32;
            let mut keep_bv = vec![false; comps];
            let mut keep_m9 = vec![false; comps];
            let mut keep_rows = vec![false; comps];
            let mut keep_oee = vec![false; comps];
            for h in 0..comps {
                let n = ep.n0 + ((h as u128) << m);
                let x = ep.m + P3[ep.f as usize] * h as u128;
                let ys = traj(x, b).expect("window");
                keep_bv[h] = window_dip(&ys, b, &mix) <= cutoff_f;
                keep_m9[h] = !matches!(n % 9, 2 | 4 | 5 | 8);
                keep_rows[h] = (ep.rk[h / 64] >> (h % 64)) & 1 == 1;
                keep_oee[h] = match ep.oee {
                    2 => x & 1 == 1,
                    1 => x & 3 != 0,
                    _ => true,
                };
            }
            let audit_stops = ei < audit_eps;
            let mut per_code: Vec<Vec<CaseOut>> = Vec::new();
            for &code in &codes {
                let cp = code_params(code).expect("jd code");
                let mut outs = vec![CaseOut::default(); comps];
                for h in 0..comps {
                    if !(keep_bv[h] && keep_oee[h] && (keep_m9[h] || keep_rows[h])) {
                        continue; // not a case in any variant
                    }
                    let n = ep.n0 + ((h as u128) << m);
                    let x = ep.m + P3[ep.f as usize] * h as u128;
                    let co = if code == mu_code { run_case_m(n, x, h0, m, cp, cutoff, &mix, hbias, mu) } else { run_case(n, x, h0, m, cp, cutoff, &mix, hbias) };
                    outs[h] = co;
                    if co.overflow {
                        cnt.overflow.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    if !audit_stops {
                        continue;
                    }
                    cnt.stops.fetch_add(1, Ordering::Relaxed);
                    if co.flagged {
                        cnt.flagged.fetch_add(1, Ordering::Relaxed);
                        // the guard must hold at the stop boundary, and the case must finish (256-bit)
                        let (xb, _) = titer(x, co.jumps * cp.w).unwrap();
                        if xb < 1u128 << (cutoff - growth_bits(cp.w)) {
                            cnt.recheck_fail.fetch_add(1, Ordering::Relaxed);
                        }
                        if recheck_wide(n, 1_000_000).is_none() {
                            cnt.recheck_fail.fetch_add(1, Ordering::Relaxed);
                        }
                        continue;
                    }
                    // flag coverage: every value up to the stop boundary is < 2^cutoff
                    {
                        let mut y = x;
                        let mut ok = y < 1u128 << cutoff;
                        for _ in 0..co.jumps * cp.w {
                            y = tstep(y).unwrap();
                            ok &= y < 1u128 << cutoff;
                        }
                        if !ok {
                            cnt.flag_cover_fail.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    if co.exact {
                        cnt.exact.fetch_add(1, Ordering::Relaxed);
                    }
                    // witness within the certified window [win_lo, win_hi - span_cut]
                    let lo = co.win_lo as usize;
                    let hi = (co.win_hi as usize).saturating_sub(span_cut);
                    let Some(ys) = traj(n, co.win_hi as usize + 2) else { cnt.overflow.fetch_add(1, Ordering::Relaxed); continue; };
                    cnt.stops_audited.fetch_add(1, Ordering::Relaxed);
                    let found = if hi >= lo { find_witness(&ys, n, lo, hi, 13, 21) } else { None };
                    match found {
                        Some((k, _)) => {
                            cnt.w_kind[k as usize].fetch_add(1, Ordering::Relaxed);
                        }
                        None => {
                            // any witness at all up to the window end (outside the claimed window)?
                            if find_witness(&ys, n, 0, hi.max(lo), 13, 26).is_some() {
                                cnt.w_late.fetch_add(1, Ordering::Relaxed);
                            } else {
                                cnt.w_fail.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
                per_code.push(outs);
            }
            // exclusion audits
            if ei < excl_eps && !synthetic {
                for h in 0..comps {
                    let n = ep.n0 + ((h as u128) << m);
                    let need = [!keep_bv[h], !keep_m9[h], !keep_rows[h], !keep_oee[h]];
                    if !need.iter().any(|&z| z) {
                        continue;
                    }
                    let Some(ys) = traj(n, m as usize + b + 2) else { cnt.excl_skipped.fetch_add(1, Ordering::Relaxed); continue; };
                    // 0: bitvector window [M, M + B]
                    if need[0] {
                        cnt.excl[0].fetch_add(1, Ordering::Relaxed);
                        if find_witness(&ys, n, m as usize, m as usize + b, 13, 21).is_none() {
                            cnt.excl_fail[0].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    // 1: mod 9 at position 0
                    if need[1] {
                        cnt.excl[1].fetch_add(1, Ordering::Relaxed);
                        if find_witness(&ys, n, 0, 0, 3, 6).is_none() {
                            cnt.excl_fail[1].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    // 2: rows: depth <= 12, words with <= 18 odd steps; h = 0 must never be excluded
                    if need[2] {
                        cnt.excl[2].fetch_add(1, Ordering::Relaxed);
                        if h == 0 || find_witness(&ys, n, 0, 12, 18, 30).is_none() {
                            cnt.excl_fail[2].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    // 3: carried OEE: the odd run before M completes with E E by position M + 2
                    if need[3] {
                        cnt.excl[3].fetch_add(1, Ordering::Relaxed);
                        if find_witness(&ys, n, 0, m as usize + 2, 13, 21).is_none() {
                            cnt.excl_fail[3].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }
            // per-variant aggregates of this endpoint
            let mut aggs: Vec<(Agg, (usize, usize))> = Vec::with_capacity(vars.len());
            for var in &vars {
                let code = var.codes[class as usize];
                let ci = codes.iter().position(|&c| c == code).unwrap();
                let rows = var.rows[class as usize];
                // v-n73-sound: select completions by the n interval of the range code.
                let (mut nlo, nhi) = n_interval(var.range[class as usize], nn);
                if nlo > 0 {
                    nlo += neg_range_shift << m;
                }
                let mut g = Agg::default();
                let (mut hmin, mut hmax, mut hcount) = (usize::MAX, 0usize, 0usize);
                for h in 0..comps {
                    let nh = ep.n0 + ((h as u128) << m);
                    if !(nh >= nlo && nh < nhi) {
                        continue;
                    }
                    hmin = hmin.min(h);
                    hmax = hmax.max(h);
                    hcount += 1;
                    let mask = if rows { keep_rows[h] } else { keep_m9[h] };
                    if !(keep_bv[h] && keep_oee[h] && mask) {
                        continue;
                    }
                    let co = per_code[ci][h];
                    let n = ep.n0 + ((h as u128) << m);
                    g.cases += 1;
                    g.iters += co.jumps as u64;
                    g.hash ^= co.hash;
                    if co.flagged {
                        g.flags.push(n);
                    }
                    let d = digest(n, co.jumps, co.hash);
                    g.dsum = g.dsum.wrapping_add(d);
                    g.dxor ^= d;
                    g.dcount += 1;
                }
                g.flags.sort_unstable();
                // the selected h set must be one contiguous range
                let hr = if hcount == 0 { (0, 0) } else { assert_eq!(hmax + 1 - hmin, hcount, "non-contiguous h set"); (hmin, hmax + 1) };
                aggs.push((g, hr));
            }
            let count = |v: &Vec<bool>| v.iter().filter(|&&z| !z).count() as u64;
            EpRes { aggs, excl: [count(&keep_bv), count(&keep_m9), count(&keep_rows), count(&keep_oee)] }
        }).collect();

        // --- per variant: compare with the GPU
        for (vi, var) in vars.iter().enumerate() {
            let code = var.codes[class as usize];
            let rows = var.rows[class as usize];
            // v-n73-sound: the h range derived from the n interval, identical for every endpoint
            let (hl, hh) = results.first().map_or((0, 0), |r| r.aggs[vi].1);
            let h_uniform = results.iter().all(|r| r.aggs[vi].1 == (hl, hh));
            assert!(h_uniform || neg_range_shift > 0, "derived h range differs between endpoints (var {})", var.name);
            let (nlo, nhi) = n_interval(var.range[class as usize], nn);
            let per_ep: Vec<&Agg> = results.iter().map(|r| &r.aggs[vi].0).collect();
            // single-endpoint launches
            let (mut s_ok, mut s_bad, mut s_n) = (0u64, 0u64, 0u64);
            for (i, g) in per_ep.iter().enumerate() {
                let Some(r) = g1.get(&(var.name.clone(), class, i)) else { continue };
                s_n += 1;
                let flags: Vec<u128> = r["flags"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().parse().unwrap()).collect();
                let ok = r["cases"].as_u64().unwrap() == g.cases && r["iters"].as_u64().unwrap() == g.iters
                    && r["hash"].as_u64().unwrap() as u32 == g.hash && flags == g.flags;
                if ok {
                    s_ok += 1;
                } else {
                    s_bad += 1;
                    if mismatches.len() < 40 {
                        mismatches.push(json!({"kind": "single", "var": var.name, "class": class, "i": i,
                            "gpu": [r["cases"], r["iters"], r["hash"], flags.len()],
                            "mine": [g.cases, g.iters, g.hash, g.flags.len()], "n0": eps[i].n0.to_string()}));
                    }
                }
            }
            // batches (prefixes of the class's endpoint list)
            let mut batch_rows = Vec::new();
            for r in gb.iter().filter(|r| r["v"].as_str() == Some(&var.name) && r["c"].as_u64() == Some(class as u64)) {
                let k = r["k"].as_u64().unwrap() as usize;
                let mut tot = Agg::default();
                for g in &per_ep[..k] {
                    tot.cases += g.cases;
                    tot.iters += g.iters;
                    tot.hash ^= g.hash;
                    tot.flags.extend(g.flags.iter().copied());
                    tot.dsum = tot.dsum.wrapping_add(g.dsum);
                    tot.dxor ^= g.dxor;
                    tot.dcount += g.dcount;
                }
                tot.flags.sort_unstable();
                let flags: Vec<u128> = r["flags"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().parse().unwrap()).collect();
                let dbg: Vec<u64> = r["dbg"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
                let dig_layout = is_digest_layout(&var.wq[class as usize]);
                let ok_core = r["cases"].as_u64().unwrap() == tot.cases && r["iters"].as_u64().unwrap() == tot.iters
                    && r["hash"].as_u64().unwrap() as u32 == tot.hash && flags == tot.flags;
                let ok_dig = !dig_layout || (dbg[0] == tot.dsum && dbg[1] == tot.dxor && dbg[2] == tot.dcount);
                if !(ok_core && ok_dig) && mismatches.len() < 80 {
                    mismatches.push(json!({"kind": "batch", "var": var.name, "class": class, "k": k, "chunk": r["chunk"],
                        "gpu": [r["cases"], r["iters"], r["hash"], flags.len(), dbg], "mine": [tot.cases, tot.iters, tot.hash, tot.flags.len(), [tot.dsum, tot.dxor, tot.dcount]]}));
                }
                batch_rows.push(json!({"k": k, "chunk": r["chunk"], "launches": r["launches"], "ok": ok_core, "digest_checked": dig_layout, "digest_ok": ok_dig,
                    "cases": tot.cases, "iters": tot.iters, "flags": tot.flags.len()}));
            }
            // project CPU reference rows (triangulation only)
            let cpu_ok: Vec<bool> = cpu.iter().filter(|r| r["v"].as_str() == Some(&var.name) && r["c"].as_u64() == Some(class as u64)).map(|r| {
                let k = r["k"].as_u64().unwrap() as usize;
                let (c, it, h) = per_ep[..k].iter().fold((0u64, 0u64, 0u32), |s, g| (s.0 + g.cases, s.1 + g.iters, s.2 ^ g.hash));
                r["cases"].as_u64().unwrap() == c && r["iters"].as_u64().unwrap() == it && r["hash"].as_u64().unwrap() as u32 == h
            }).collect();
            let tc: u64 = per_ep.iter().map(|g| g.cases).sum();
            let ti: u64 = per_ep.iter().map(|g| g.iters).sum();
            let tf: usize = per_ep.iter().map(|g| g.flags.len()).sum();
            eprintln!("class {class:2} var {:8} code {code} rows {rows} range [{hl},{hh}) wq {:8}: eps {} cases {tc} iters {ti} flags {tf} | single {s_ok}/{s_n} ok | batches {}",
                var.name, var.wq[class as usize], eps.len(), batch_rows.iter().map(|b| if b["ok"].as_bool().unwrap() && b["digest_ok"].as_bool().unwrap() { "ok".to_string() } else { "BAD".to_string() }).collect::<Vec<_>>().join(","));
            report.push(json!({"class": class, "var": var.name, "code": code, "rows": rows, "range": [hl, hh],
                "n_interval": [nlo.to_string(), nhi.to_string()], "range_code": var.range[class as usize], "wq": var.wq[class as usize],
                "endpoints": eps.len(), "cases": tc, "iters": ti, "flags": tf,
                "single_ok": s_ok, "single_bad": s_bad, "single_n": s_n, "batches": batch_rows, "cpu_ref_ok": cpu_ok}));
        }
        // screen statistics for this class
        let ex = results.iter().fold([0u64; 4], |mut acc, r| { for i in 0..4 { acc[i] += r.excl[i]; } acc });
        eprintln!("class {class:2}: excluded by bitvector {}, mod9 {}, rows {}, oee {} (of {})", ex[0], ex[1], ex[2], ex[3], eps.len() << a);
    }
    let ld = |x: &AtomicU64| x.load(Ordering::Relaxed);
    let summary = json!({
        "prefix": prefix, "seconds": t0.elapsed().as_secs_f64(), "hbias": hbias, "span_cut": span_cut, "neg_range_shift": neg_range_shift as u64,
        "endpoints": classes.iter().map(|c| by_class[c].len()).sum::<usize>(),
        "stops": ld(&cnt.stops), "stops_witness_audited": ld(&cnt.stops_audited), "flagged": ld(&cnt.flagged), "exact_path": ld(&cnt.exact),
        "witness_kinds": {"descent": ld(&cnt.w_kind[0]), "pm1": ld(&cnt.w_kind[1]), "pm_inf": ld(&cnt.w_kind[2]), "oee": ld(&cnt.w_kind[3]), "dfs": ld(&cnt.w_kind[4])},
        "witness_fail": ld(&cnt.w_fail), "witness_only_outside_window": ld(&cnt.w_late),
        "flag_cover_fail": ld(&cnt.flag_cover_fail), "recheck_fail": ld(&cnt.recheck_fail),
        "excl_skipped": ld(&cnt.excl_skipped), "n0_range_fail": ld(&cnt.n0_range_fail), "n": nn, "a": a, "m": m,
        "checker_overflow": ld(&cnt.overflow), "tm_mismatch": ld(&cnt.m_mismatch), "synthetic_endpoints": ld(&cnt.synthetic),
        "exclusions_audited": {"bitvector": ld(&cnt.excl[0]), "mod9": ld(&cnt.excl[1]), "rows": ld(&cnt.excl[2]), "oee": ld(&cnt.excl[3])},
        "exclusion_fail": {"bitvector": ld(&cnt.excl_fail[0]), "mod9": ld(&cnt.excl_fail[1]), "rows": ld(&cnt.excl_fail[2]), "oee": ld(&cnt.excl_fail[3])},
        "mismatches": mismatches, "rows": report,
    });
    std::fs::write(&json_out, serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    eprintln!("{}", serde_json::to_string(&json!({"stops": summary["stops"], "audited": summary["stops_witness_audited"],
        "witness_fail": summary["witness_fail"], "late": summary["witness_only_outside_window"], "flag_cover_fail": summary["flag_cover_fail"],
        "recheck_fail": summary["recheck_fail"], "overflow": summary["checker_overflow"], "excl_skipped": summary["excl_skipped"], "n0_range_fail": summary["n0_range_fail"], "tm_mismatch": summary["tm_mismatch"], "synthetic": summary["synthetic_endpoints"],
        "excl": summary["exclusions_audited"], "excl_fail": summary["exclusion_fail"], "mismatches": summary["mismatches"].as_array().unwrap().len(),
        "seconds": summary["seconds"]})).unwrap());
}

/// Self tests of the checker's own pieces (no GPU, no cv code).
fn selftest() {
    let mix = Mix::build(13);
    // (1) table credit == DFS credit on random values and levels
    let mut x = 0x1234_5678_9abc_def1_2345_6789u128;
    let mut bad = 0;
    for i in 0..200_000 {
        x = x.wrapping_mul(0x2545_F491_4F6C_DD1D_9E37_79B9_7F4A_7C15).wrapping_add(1442695040888963407);
        let y = (x >> 20) | (1 << 60);
        let jp = 1 + (i % 13) as u32;
        let a = mix.credit(jp, y);
        let b = mix.credit_dfs(jp, y);
        if a != b {
            bad += 1;
            if bad < 5 {
                eprintln!("mix mismatch y={y} jp={jp} table={a} dfs={b}");
            }
        }
    }
    eprintln!("selftest mix table vs dfs: {bad} mismatches of 200000");
    // (2) every exclusion-type certificate found by window_dip has a witness:
    // random windows at random depths with large n.
    let mut fails = 0u64;
    let mut fired = 0u64;
    for i in 0..200_000u64 {
        x = x.wrapping_mul(0x2545_F491_4F6C_DD1D_9E37_79B9_7F4A_7C15).wrapping_add(i as u128);
        let n = (x >> 60) | (1u128 << 66);
        // walk n for a random depth d, compute H, test windows of width 12
        let d = 40 + (i % 40) as usize;
        let ys = traj(n, d + 14).unwrap();
        let f = (0..d).filter(|&k| ys[k] & 1 == 1).count() as i32;
        let h = 485 * f - 306 * d as i32;
        let dip = window_dip(&ys[d..], 12, &mix);
        if dip > h {
            fired += 1;
            if find_witness(&ys, n, d, d + 12, 13, 21).is_none() && find_witness(&ys, n, 0, d + 12, 13, 21).is_none() {
                fails += 1;
            }
        }
    }
    eprintln!("selftest random windows: {fired} fired, {fails} without witness");
    // (3) recheck sanity
    assert_eq!(recheck_wide(27, 1000).is_some(), true);
    eprintln!("selftest growth bits W10/11/12: {} {} {}", growth_bits(10), growth_bits(11), growth_bits(12));
}
