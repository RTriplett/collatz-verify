//! v-n73-sound: r3pr-gpu's r3check (copied) with two N73 changes, nothing else:
//!   (1) the quarter range codes 3 u2 / 4 l2 / 5 m2 of v-n73-sound's v3dump, selected by the n interval
//!       [2^(N-2), 2^N) / [0, 2^(N-2)) / [2^(N-2), 2^(N-1)) (with 1 top, 2 bottom, 0 full likewise);
//!   (2) the small-n check (violation 16) applies to the h = 0 slice n < 2^M of any N with M <= 63
//!       (was: N72 and n < 2^62).
//!
//! r3pr-gpu: round-3 path-record bound audit on the real two-phase kernels'
//! decisions (pr-audit's prcheck, extended).  Changes from prcheck:
//!   * v3-sound's v3check addition: the two-phase digest audit layouts
//!     (t2/t12/t14/t16-21/t24/t31) for the per-case digest of batches (every
//!     batch row of the dump, whatever its launch chunk, is compared);
//!   * three more ceilings (violations 14-16): every post-stop value < n 2^24
//!     (v-pr-sound's table-independent ceiling); every value of a bitvector
//!     exclusion's window, T^(M+j)(n) + 1 <= 3^F (n + 64) 1.5^j / 2^M (the
//!     class-specific screen bound of pr-proof section 3); and, for n < 2^62
//!     (N72), no unflagged stop or exclusion with a pre-witness value >= 2^c;
//!   * negative controls: --neg-tested D, --neg-post24 D, --neg-screen D
//!     (bounds lowered by D bits), --neg-span S (stop witnesses searched only up
//!     to the window end minus S); PR_EXTRA_RECORDS as before.
//!
//! pr-audit: path-record bound audit on the real kernels' decisions.
//!
//! Derived from v2-sound's independent checker (src/bin/v2check.rs, copied
//! verbatim where marked "v2check"): it SHARES NO CODE WITH THE cv CRATE (no
//! `cv::` import), recomputes every case's stop from n0 with plain T steps
//! (screen: bitvector window dip <= 485 F - 306 M, mod 9 or the dumped row-table
//! keep words, carried OEE mask; tail: jd stop rule with H = 485 F' - 306 M'),
//! and compares per endpoint / per batch with the GPU results in PREFIX.jsonl.
//!
//! NEW in pr-audit: for every non-flagged GPU stop and every exclusion it
//! computes the maximum of T^i(n) for 0 <= i <= s, where s is
//!   * s_w: the merge step of the checker's independently found witness
//!          (descent, pm1, pm_inf, OEE (merge at s + a + 2), inverse DFS),
//!   * s_c: the earliest merge step the certificate itself certifies (the
//!          first credit position of the window dip with value > H),
//!   * s_e: the end of the claimed span (window end; exclusion span end),
//! and checks it against the derived bounds:
//!   * prefix  [0, M]:      T^i(n) + 1 <= 1.5^i (n + 1)
//!   * tested  (M, P]:      y < L * max_t min(1.5^t, 2^(W - t))   (L = 2^(cutoff - K), P = last flag-tested boundary)
//!   * post    (P, s_e]:    y + 1 <= 1.5^t (x_P + 1)              (the "naive" (KA+1) W steps of growth)
//!                          y < L * 1.5^((KA+1) W)                (orchestrator's bound)
//!                          log2(y / n) < 2W - 1 (KA 1), 485 W / 306 (KA 0)   (H-based: window fires only if H_Z < 306 W)
//! and against the path-record table (records #88-#97, shortcut maxima).
//!
//! prcheck PREFIX OUT.json [--audit-eps N] [--excl-eps N] [--trace FILE] [--label L]
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, Read};
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Arithmetic (v2check).

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

#[inline]
fn lg(x: u128) -> f64 {
    (x as f64).log2()
}

// ---------------------------------------------------------------------------
// Mixed-word credits (v2check).

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
                let mut stack: Vec<(u32, u32, u128)> = vec![(1, o - 1, 1)];
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
}

// ---------------------------------------------------------------------------
// Window dip (v2check) and the certified position (new).

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

/// Credit kinds of the certified position.
const CK_NAMES: [&str; 6] = ["descent", "pm1", "pm_inf", "oee", "mixed", "start"];

/// Earliest MERGE step (relative to ys[0], the window start) of a credit of the
/// window dip whose value exceeds `hz` (the same credits as window_dip; OEE
/// merges at i + a + 2, every other credit at i + 1; position 0 if hz < 0).
fn window_cert(ys: &[u128], w: usize, mix: &Mix, hz: i32) -> Option<(usize, u8)> {
    if 0 > hz {
        return Some((0, 5));
    }
    let odd = |i: usize| ys[i] & 1 == 1;
    let mut best: Option<(usize, u8)> = None;
    fn upd(best: &mut Option<(usize, u8)>, s: usize, k: u8) {
        if best.map_or(true, |(b, _)| s < b) {
            *best = Some((s, k));
        }
    }
    let (mut cur, mut evens, mut fw) = (0i32, 0u32, 0u32);
    for i in 0..w {
        if let Some((b, _)) = best {
            if b <= i + 1 {
                break;
            }
        }
        if odd(i) {
            let mut a = 0;
            while i + a < w && odd(i + a) {
                a += 1;
            }
            if i + a + 2 < w && !odd(i + a) && !odd(i + a + 1) && cur + 306 > hz {
                upd(&mut best, i + a + 2, 3);
            }
            cur -= 179;
            evens = 0;
            fw += 1;
        } else {
            cur += 306;
            evens += 1;
            if evens % 2 == 0 && fw > 0 && cur + 179 > hz {
                upd(&mut best, i + 1, 1);
            }
        }
        if cur > hz {
            upd(&mut best, i + 1, 0);
        }
        if fw > 0 {
            let y = ys[i + 1];
            let t = v3(y + 1).min(fw) as i32;
            if cur + 179 * t > hz {
                upd(&mut best, i + 1, 2);
            }
            let c = mix.credit(fw.min(13), y);
            if c != NONE as i32 && cur + c > hz {
                upd(&mut best, i + 1, 4);
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
// jd tail rule (v2check).

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
    exact: bool,
    win_lo: u32,
    win_hi: u32,
    overflow: bool,
}

fn run_case(n: u128, x0: u128, h0: i32, m: u32, c: Code, cutoff: u32, mix: &Mix) -> CaseOut {
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
        let (mut z, mut hz) = (x, h);
        for _ in 0..c.ka {
            let Some((nz, f)) = titer(z, c.w) else { o.overflow = true; return o; };
            hz += 485 * f as i32 - 306 * c.w as i32;
            z = nz;
        }
        let Some(ys) = traj(z, w) else { o.overflow = true; return o; };
        let dip = window_dip(&ys, w, mix);
        let dq = (dip / c.q) * c.q;
        if dq > hz {
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
// Witness search (v2check), returning the merge step as well.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum WKind {
    Descent,
    Pm1,
    PmInf,
    Oee,
    Dfs,
}
const WK_NAMES: [&str; 5] = ["descent", "pm1", "pm_inf", "oee", "dfs"];

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

/// Witness at some position in [lo, hi]: (kind, position, merge step).  The merge
/// step is the position for every kind except OEE (position + a + 2 <= hi).
fn find_witness(ys: &[u128], n: u128, lo: usize, hi: usize, omax: u32, smax: u32) -> Option<(WKind, usize, usize)> {
    let hi = hi.min(ys.len() - 1);
    for s in lo..=hi {
        let y = ys[s];
        if y < n {
            return Some((WKind::Descent, s, s));
        }
        if y % 3 == 2 {
            let p = (2 * y - 1) / 3;
            if p >= 1 && p < n && forward_hits(p, 1, y) {
                return Some((WKind::Pm1, s, s));
            }
        }
        let t = v3(y + 1);
        if t >= 1 && t <= 80 {
            let u = (y + 1) / P3[t as usize];
            if let Some(u) = u.checked_shl(t).filter(|&v| v >> t == u) {
                let u = u - 1;
                if u >= 1 && u < n && forward_hits(u, t, y) {
                    return Some((WKind::PmInf, s, s));
                }
            }
        }
        if y & 1 == 1 {
            let mut a = 0;
            while s + a < ys.len() && ys[s + a] & 1 == 1 {
                a += 1;
            }
            let e = s + a + 2;
            if e <= hi && ys[s + a] & 1 == 0 && ys[s + a + 1] & 1 == 0 {
                let p = (y - 1) / 2;
                if p >= 1 && p < n && forward_hits(p, a as u32 + 1, ys[e]) {
                    return Some((WKind::Oee, s, e));
                }
            }
        }
    }
    for s in lo..=hi {
        if let Some((p, a)) = dfs_partner(ys[s], n, omax, smax) {
            if p >= 1 && p < n && forward_hits(p, a, ys[s]) {
                return Some((WKind::Dfs, s, s));
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Path records (Barina's table, shortcut maxima; #88-#97 replayed exactly).

const RECORDS: [(u128, &str); 10] = [
    (1980976057694848447, "32012333661096566765082938647132369010"),
    (35136221158664800255, "92102545196486820634779928066830214436"),
    (48503373501652785087, "296696710908147364747230298439288489642"),
    (55247846101001863167, "482192631091346876742345874501396316228"),
    (71149323674102624415, "4527691962113372170289733115168874698466"),
    (274133054632352106267, "56649062372194325899121269007146717645316"),
    (1378299700343633691495, "73161949456862481446077319071744298810624"),
    (1735519168865914451271, "88638514583197544506368462565103733234728"),
    (1765856170146672440559, "394170168671074358667552016954030019514164"),
    (2358909599867980429759, "494114680693632162318569565806744468898092"),
];

/// log2 of a decimal string (may exceed u128).
fn lg_dec(s: &str) -> f64 {
    let d = s.len();
    let head: f64 = s[..d.min(17)].parse().unwrap();
    head.log2() + ((d - d.min(17)) as f64) * 10f64.log2()
}

struct Rec {
    id: Vec<i64>,
    start: Vec<u128>,
    max_u: Vec<Option<u128>>,
    max_lg: Vec<f64>,
}

impl Rec {
    fn new() -> Self {
        // NEGATIVE CONTROL (env PR_EXTRA_RECORDS="start:max,..."): extra (earlier) records prepended to the
        // table, so that R(n) is also known (as a lower bound) below #88 and missed records there are detected.
        // entries "id:start:max"
        let mut recs: Vec<(u128, String, i64)> = RECORDS.iter().enumerate().map(|(i, r)| (r.0, r.1.to_string(), 88 + i as i64)).collect();
        if let Ok(extra) = std::env::var("PR_EXTRA_RECORDS") {
            for e in extra.split(',') {
                let v: Vec<&str> = e.trim().split(':').collect();
                recs.push((v[1].parse().unwrap(), v[2].to_string(), v[0].parse().unwrap()));
            }
            recs.sort();
        }
        let id = recs.iter().map(|r| r.2).collect();
        let start = recs.iter().map(|r| r.0).collect();
        let max_u = recs.iter().map(|r| r.1.parse::<u128>().ok()).collect();
        let max_lg = recs.iter().map(|r| lg_dec(&r.1)).collect();
        Rec { id, start, max_u, max_lg }
    }
    /// Index k such that R(n) = max of record k (n in (start_k, start_{k+1}]),
    /// or the last record for n > #97 (a lower bound of R(n)); None for n <= #88.
    fn idx(&self, n: u128) -> Option<usize> {
        if n <= self.start[0] {
            return None;
        }
        let mut k = 0;
        while k + 1 < self.start.len() && n > self.start[k + 1] {
            k += 1;
        }
        Some(k)
    }
    /// true iff v >= R(n) (a potential record), for n > #88.
    fn reaches(&self, k: usize, v: u128) -> bool {
        match self.max_u[k] {
            Some(r) => v >= r,
            None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics.

#[derive(Clone, Copy, Debug)]
struct Ext {
    v: f64,
    n: u128,
    aux: i64,
}
impl Default for Ext {
    fn default() -> Self {
        Ext { v: f64::NEG_INFINITY, n: 0, aux: 0 }
    }
}
impl Ext {
    #[inline]
    fn up(&mut self, v: f64, n: u128, aux: i64) {
        if v > self.v {
            *self = Ext { v, n, aux };
        }
    }
    fn merge(&mut self, o: &Ext) {
        if o.v > self.v {
            *self = *o;
        }
    }
    fn js(&self) -> Value {
        if self.v == f64::NEG_INFINITY {
            Value::Null
        } else {
            json!({"v": (self.v * 1e6).round() / 1e6, "n": self.n.to_string(), "aux": self.aux})
        }
    }
}

const NV: usize = 17;
const V_NAMES: [&str; NV] = [
    "prefix_bound",      // 0: T^i(n)+1 > 1.5^i (n+1) in [0, M]
    "tested_bound",      // 1: tested-region value >= L * max_t min(1.5^t, 2^(W-t))
    "post_rel_bound",    // 2: post value y+1 > 1.5^t (x_P + 1)
    "post_naive_bound",  // 3: post value >= L * 1.5^((KA+1)W)
    "post_h_bound",      // 4: log2(post/n) >= 2W-1 (KA1) / 485W/306 (KA0)
    "witness_none_in_span", // 5: no checker witness in the claimed window
    "witness_only_outside", // 6: a witness exists only outside the claimed window
    "cert_none",         // 7: the certificate fired but no credit position exceeds H (model error)
    "cert_after_span",   // 8: certified merge step beyond the span end
    "pmw_ge_R",          // 9: max up to the witness >= R(n), n > #88 (potential missed record)
    "pme_ge_R",          // 10: max up to the span end >= R(n), n > #88
    "pmw_gt_t88",        // 11: max up to the witness > t(#88)
    "pme_gt_t88",        // 12: max up to the span end > t(#88)
    "traj_overflow",     // 13: u128 overflow (not audited)
    "post_24_bound",     // 14 (r3pr-gpu): log2(post/n) >= 24 in (P, span end]
    "screen_class_bound", // 15 (r3pr-gpu): bitvector window value T^(M+j)(n)+1 > 3^F (n+64) 1.5^j / 2^M
    "pmw_ge_2c_small_n", // 16 (r3pr-gpu): n < 2^62 (N72): unflagged pre-witness value >= 2^c
];

#[derive(Clone)]
struct Stat {
    count: u64,
    n_gt88: u64,
    pm_w: Ext,
    pm_c: Ext,
    pm_e: Ext,
    pm_w_n: Ext,     // log2(PM_w / n)
    pm_e_n: Ext,
    prefix: Ext,     // log2 max over [0, M]
    tested_l: Ext,   // log2(max over tested region / L)
    bnd_l: Ext,      // log2(max tested boundary / L)
    post_rel_w: Ext, // log2(max over (P, s_w] / x_P)
    pmw_xp: Ext,     // log2(PM_w / x_P): the overall pre-witness maximum over the last flag-tested boundary value
    post_rel_e: Ext, // log2(max over (P, s_e] / x_P)
    post_n_e: Ext,   // log2(max over (P, s_e] / n)
    post_abs_e: Ext, // log2 max over (P, s_e]
    wstep_rel: Ext,  // s_w - P (steps past the last tested boundary)
    cstep_rel: Ext,  // s_c - P
    min_margin_w: Ext, // -(log2 R(n) - log2 PM_w): stored negated so that "up" = worst
    min_margin_e: Ext,
    le88_pm_e: Ext,  // log2 PM_e for n <= #88 (not covered by the record argument)
    screen_slack: Ext, // r3pr-gpu: max over the window of log2((T^(M+j)(n)+1) / (3^F (n+64) 1.5^j / 2^M)) (<= 0)
    where_w: [u64; 3],
    wkind: [u64; 5],
    ckind: [u64; 6],
    viol: [u64; NV],
    viol_ex: Vec<String>,
    hist_pmw: [u64; 80],    // log2 PM_w in [60, 140)
    hist_tested: [u64; 64], // log2(tested/L) in quarter bits [-12, 4)
}

impl Default for Stat {
    fn default() -> Self {
        Stat {
            count: 0, n_gt88: 0, pm_w: Ext::default(), pm_c: Ext::default(), pm_e: Ext::default(),
            pm_w_n: Ext::default(), pm_e_n: Ext::default(), prefix: Ext::default(), tested_l: Ext::default(),
            bnd_l: Ext::default(), post_rel_w: Ext::default(), pmw_xp: Ext::default(), post_rel_e: Ext::default(), post_n_e: Ext::default(),
            post_abs_e: Ext::default(), wstep_rel: Ext::default(), cstep_rel: Ext::default(),
            min_margin_w: Ext::default(), min_margin_e: Ext::default(), le88_pm_e: Ext::default(), screen_slack: Ext::default(),
            where_w: [0; 3], wkind: [0; 5], ckind: [0; 6], viol: [0; NV], viol_ex: Vec::new(),
            hist_pmw: [0; 80], hist_tested: [0; 64],
        }
    }
}

impl Stat {
    fn merge(&mut self, o: &Stat) {
        self.count += o.count;
        self.n_gt88 += o.n_gt88;
        for (a, b) in [
            (&mut self.pm_w, &o.pm_w), (&mut self.pm_c, &o.pm_c), (&mut self.pm_e, &o.pm_e),
            (&mut self.pm_w_n, &o.pm_w_n), (&mut self.pm_e_n, &o.pm_e_n), (&mut self.prefix, &o.prefix),
            (&mut self.tested_l, &o.tested_l), (&mut self.bnd_l, &o.bnd_l), (&mut self.post_rel_w, &o.post_rel_w), (&mut self.pmw_xp, &o.pmw_xp),
            (&mut self.post_rel_e, &o.post_rel_e), (&mut self.post_n_e, &o.post_n_e), (&mut self.post_abs_e, &o.post_abs_e),
            (&mut self.wstep_rel, &o.wstep_rel), (&mut self.cstep_rel, &o.cstep_rel),
            (&mut self.min_margin_w, &o.min_margin_w), (&mut self.min_margin_e, &o.min_margin_e),
            (&mut self.le88_pm_e, &o.le88_pm_e), (&mut self.screen_slack, &o.screen_slack),
        ] {
            a.merge(b);
        }
        for i in 0..3 { self.where_w[i] += o.where_w[i]; }
        for i in 0..5 { self.wkind[i] += o.wkind[i]; }
        for i in 0..6 { self.ckind[i] += o.ckind[i]; }
        for i in 0..NV { self.viol[i] += o.viol[i]; }
        for e in &o.viol_ex {
            if self.viol_ex.len() < 20 {
                self.viol_ex.push(e.clone());
            }
        }
        for i in 0..80 { self.hist_pmw[i] += o.hist_pmw[i]; }
        for i in 0..64 { self.hist_tested[i] += o.hist_tested[i]; }
    }
    fn violate(&mut self, k: usize, n: u128, info: &str) {
        self.viol[k] += 1;
        if self.viol_ex.len() < 20 {
            self.viol_ex.push(format!("{} n={} {}", V_NAMES[k], n, info));
        }
    }
    fn js(&self) -> Value {
        let neg = |e: &Ext| if e.v == f64::NEG_INFINITY { Value::Null } else { json!({"v": ((-e.v) * 1e6).round() / 1e6, "n": e.n.to_string(), "aux": e.aux}) };
        json!({
            "count": self.count, "n_gt_88": self.n_gt88,
            "log2_pm_witness": self.pm_w.js(), "log2_pm_cert": self.pm_c.js(), "log2_pm_span_end": self.pm_e.js(),
            "log2_pm_witness_over_n": self.pm_w_n.js(), "log2_pm_span_end_over_n": self.pm_e_n.js(),
            "log2_prefix_max": self.prefix.js(), "log2_tested_over_L": self.tested_l.js(), "log2_boundary_over_L": self.bnd_l.js(),
            "log2_post_witness_over_xP": self.post_rel_w.js(), "log2_pm_witness_over_xP": self.pmw_xp.js(), "log2_post_span_end_over_xP": self.post_rel_e.js(),
            "log2_post_span_end_over_n": self.post_n_e.js(), "log2_post_span_end": self.post_abs_e.js(),
            "max_witness_steps_after_P": self.wstep_rel.js(), "max_cert_steps_after_P": self.cstep_rel.js(),
            "min_margin_bits_witness": neg(&self.min_margin_w), "min_margin_bits_span_end": neg(&self.min_margin_e),
            "log2_pm_span_end_n_le_88": self.le88_pm_e.js(),
            "log2_screen_bound_slack": self.screen_slack.js(),
            "argmax_region_witness": {"prefix": self.where_w[0], "tested": self.where_w[1], "post": self.where_w[2]},
            "witness_kinds": WK_NAMES.iter().zip(self.wkind.iter()).map(|(k, v)| (k.to_string(), json!(v))).collect::<serde_json::Map<_, _>>(),
            "cert_kinds": CK_NAMES.iter().zip(self.ckind.iter()).map(|(k, v)| (k.to_string(), json!(v))).collect::<serde_json::Map<_, _>>(),
            "violations": V_NAMES.iter().zip(self.viol.iter()).map(|(k, v)| (k.to_string(), json!(v))).collect::<serde_json::Map<_, _>>(),
            "violation_examples": self.viol_ex,
            "hist_log2_pm_witness_60_140": self.hist_pmw.to_vec(),
            "hist_log2_tested_over_L_quarter_bits_from_-12": self.hist_tested.to_vec(),
        })
    }
}

type Key = (String, u32, u32); // (kind, code, class)

fn bump(map: &mut HashMap<Key, Stat>, key: Key) -> &mut Stat {
    map.entry(key).or_default()
}

// ---------------------------------------------------------------------------
// Data (v2check).

#[derive(Clone, Debug)]
struct Ep {
    n0: u128,
    m: u128,
    f: u32,
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

/// v3-sound's v3check: scheduler specs whose kernels return the per-case digest (sum, XOR,
/// count of mix64(n, case jumps, case hash)) in KOut.dbg[0..2]: c-sched's warp-queue audit
/// layouts 26 and 52, and the two-phase DIG copies (cuda/twophase.cuh TpCfg<V>::DIG: 2, 12,
/// 14, 16-21, 24, 31).  Hard-coded (the checker takes nothing from cv).
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

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// Shared per-run constants.
struct Ctx<'a> {
    m: u32,
    cutoff: u32,
    mix: &'a Mix,
    rec: &'a Rec,
    t88: u128,
    /// r3pr-gpu: N (the small-n check applies to N72 dumps only) and negative-control offsets (bits / steps).
    nn: u32,
    neg_tested: f64,
    neg_post24: f64,
    neg_screen: f64,
    neg_span: usize,
}

/// The pre-witness accounting of one non-flagged stop.  `ys[i] = T^i(n)`, `p` = last flag-tested
/// boundary, `e` = span end, `kz` = window start (dip stops) or None (exact path).
fn account_stop(cx: &Ctx, st: &mut Stat, n: u128, ys: &[u128], cp: Code, jumps: u32, exact: bool,
                s_w: Option<(WKind, usize, usize)>, late: bool) {
    let m = cx.m as usize;
    let w = cp.w as usize;
    let k = growth_bits(cp.w);
    let lbits = (cx.cutoff - k) as f64;
    let (p, e, z) = if exact {
        (m + (jumps as usize - 1) * w, m + jumps as usize * w, None)
    } else {
        let p = m + jumps as usize * w;
        (p, p + (cp.ka as usize + 1) * w, Some(p + cp.ka as usize * w))
    };
    st.count += 1;
    // regions
    let mut prefix_max = 0u128;
    let ln1 = lg(n + 1);
    let l15 = 1.5f64.log2();
    let mut bad_prefix = false;
    for i in 0..=m {
        prefix_max = prefix_max.max(ys[i]);
        if lg(ys[i] + 1) > ln1 + i as f64 * l15 + 1e-9 {
            bad_prefix = true;
        }
    }
    if bad_prefix {
        st.violate(0, n, "");
    }
    st.prefix.up(lg(prefix_max), n, jumps as i64);
    // tested region (M, P]: boundaries at M + k W, k <= jumps (exact: <= jumps - 1)
    let inj = {
        let mut best = 0f64;
        let mut t = 1usize;
        while t < w {
            let v = (t as f64 * l15).min((w - t) as f64);
            best = best.max(v);
            t += 1;
        }
        best
    };
    let mut tested_max = 0u128;
    let mut bnd_max = 0u128;
    let mut kb = m;
    while kb <= p {
        bnd_max = bnd_max.max(ys[kb]);
        kb += w;
    }
    for i in (m + 1)..=p {
        tested_max = tested_max.max(ys[i]);
    }
    st.bnd_l.up(lg(bnd_max) - lbits, n, jumps as i64);
    if p > m {
        let tl = lg(tested_max) - lbits;
        st.tested_l.up(tl, n, jumps as i64);
        let b = ((tl + 12.0) * 4.0).floor();
        if b >= 0.0 && (b as usize) < 64 {
            st.hist_tested[b as usize] += 1;
        }
        if tl > inj - cx.neg_tested + 1e-9 {
            st.violate(1, n, &format!("log2(tested/L)={tl:.4} bound {:.4}", inj - cx.neg_tested));
        }
    }
    // post region (P, e]
    let xp = ys[p];
    let lxp1 = lg(xp + 1);
    let mut post_max = 0u128;
    let mut post_n = f64::NEG_INFINITY;
    let naive = lbits + ((cp.ka + 1) * cp.w) as f64 * l15;
    // H-based post bound: the window fires only if H_Z < 306 W (all-even dip); going back from Z
    // each step adds at most 306, going forward at most 179: KA >= 1 -> (KA+1) W - 1 bits, KA 0 -> 485 W / 306.
    let hb = if cp.ka >= 1 { ((cp.ka + 1) * cp.w - 1) as f64 } else { 485.0 * cp.w as f64 / 306.0 };
    let lnn = lg(n);
    for t in 1..=(e - p) {
        let y = ys[p + t];
        post_max = post_max.max(y);
        if lg(y + 1) > lxp1 + t as f64 * l15 + 1e-9 {
            st.violate(2, n, &format!("t={t}"));
        }
        let ln = lg(y) - lnn;
        post_n = post_n.max(ln);
    }
    if e > p {
        st.post_rel_e.up(lg(post_max) - lg(xp), n, jumps as i64);
        st.post_n_e.up(post_n, n, jumps as i64);
        st.post_abs_e.up(lg(post_max), n, jumps as i64);
        if lg(post_max) > naive + 1e-9 {
            st.violate(3, n, &format!("log2 post={:.4} naive {naive:.4}", lg(post_max)));
        }
        if !exact && post_n > hb + 1e-6 {
            st.violate(4, n, &format!("log2(post/n)={post_n:.4} bound {hb:.4}"));
        }
        if post_n >= 24.0 - cx.neg_post24 {
            st.violate(14, n, &format!("log2(post/n)={post_n:.4} ceiling {:.4}", 24.0 - cx.neg_post24));
        }
    }
    // span-end maximum
    let pm_e = ys[..=e].iter().copied().max().unwrap();
    st.pm_e.up(lg(pm_e), n, jumps as i64);
    st.pm_e_n.up(lg(pm_e) - lnn, n, jumps as i64);
    // the certified position (dip stops): earliest credit > H at the window start
    let s_c = match z {
        None => Some(e),
        Some(zs) => {
            let fz = ys[..zs].iter().filter(|&&v| v & 1 == 1).count() as i32;
            let hz = 485 * fz - 306 * zs as i32;
            match window_cert(&ys[zs..], w, cx.mix, hz) {
                Some((rel, kind)) => {
                    st.ckind[kind as usize] += 1;
                    Some(zs + rel)
                }
                None => {
                    st.violate(7, n, "");
                    None
                }
            }
        }
    };
    if let Some(sc) = s_c {
        if sc > e {
            st.violate(8, n, &format!("s_c={sc} e={e}"));
        }
        let pm_c = ys[..=sc.min(e)].iter().copied().max().unwrap();
        st.pm_c.up(lg(pm_c), n, jumps as i64);
        st.cstep_rel.up(sc as f64 - p as f64, n, jumps as i64);
    }
    // the checker's witness
    match s_w {
        Some((kind, _pos, merge)) => {
            st.wkind[kind as usize] += 1;
            let pm_w = ys[..=merge].iter().copied().max().unwrap();
            let am = ys[..=merge].iter().position(|&v| v == pm_w).unwrap();
            st.where_w[if am <= m { 0 } else if am <= p { 1 } else { 2 }] += 1;
            st.pm_w.up(lg(pm_w), n, merge as i64);
            st.pm_w_n.up(lg(pm_w) - lnn, n, merge as i64);
            st.pmw_xp.up(lg(pm_w) - lg(xp), n, merge as i64);
            if merge > p {
                let pw = ys[p + 1..=merge].iter().copied().max().unwrap();
                st.post_rel_w.up(lg(pw) - lg(xp), n, jumps as i64);
            }
            st.wstep_rel.up(merge as f64 - p as f64, n, jumps as i64);
            let b = lg(pm_w).floor() - 60.0;
            if b >= 0.0 && (b as usize) < 80 {
                st.hist_pmw[b as usize] += 1;
            }
            if pm_w > cx.t88 {
                st.violate(11, n, &format!("log2 pm_w={:.4}", lg(pm_w)));
            }
            if pm_e > cx.t88 {
                st.violate(12, n, &format!("log2 pm_e={:.4}", lg(pm_e)));
            }
            if cx.m <= 63 && n < 1u128 << cx.m && lg(pm_w) >= cx.cutoff as f64 {
                st.violate(16, n, &format!("log2 pm_w={:.4} c={}", lg(pm_w), cx.cutoff));
            }
            match cx.rec.idx(n) {
                Some(ri) => {
                    st.n_gt88 += 1;
                    let mw = cx.rec.max_lg[ri] - lg(pm_w);
                    let me = cx.rec.max_lg[ri] - lg(pm_e);
                    st.min_margin_w.up(-mw, n, cx.rec.id[ri]);
                    st.min_margin_e.up(-me, n, cx.rec.id[ri]);
                    if cx.rec.reaches(ri, pm_w) {
                        st.violate(9, n, &format!("record interval #{}", cx.rec.id[ri]));
                    }
                    if cx.rec.reaches(ri, pm_e) {
                        st.violate(10, n, &format!("record interval #{}", cx.rec.id[ri]));
                    }
                }
                None => st.le88_pm_e.up(lg(pm_e), n, jumps as i64),
            }
        }
        None => {
            st.violate(if late { 6 } else { 5 }, n, "");
        }
    }
}

/// Accounting of one exclusion: witness search in [lo, hi] (span end hi), certified merge step `s_c`.
fn account_excl(cx: &Ctx, st: &mut Stat, n: u128, ys: &[u128], lo: usize, hi: usize, omax: u32, smax: u32,
                s_c: Option<(usize, u8)>) {
    st.count += 1;
    let lnn = lg(n);
    let ln1 = lg(n + 1);
    let l15 = 1.5f64.log2();
    for i in 0..=hi {
        if lg(ys[i] + 1) > ln1 + i as f64 * l15 + 1e-9 {
            st.violate(0, n, "");
            break;
        }
    }
    let pm_e = ys[..=hi].iter().copied().max().unwrap();
    st.pm_e.up(lg(pm_e), n, hi as i64);
    st.pm_e_n.up(lg(pm_e) - lnn, n, hi as i64);
    if let Some((sc, kind)) = s_c {
        st.ckind[kind as usize] += 1;
        if sc > hi {
            st.violate(8, n, &format!("s_c={sc} hi={hi}"));
        }
        let pm_c = ys[..=sc.min(hi)].iter().copied().max().unwrap();
        st.pm_c.up(lg(pm_c), n, sc as i64);
    }
    match find_witness(ys, n, lo, hi, omax, smax) {
        Some((kind, _pos, merge)) => {
            st.wkind[kind as usize] += 1;
            let pm_w = ys[..=merge].iter().copied().max().unwrap();
            st.pm_w.up(lg(pm_w), n, merge as i64);
            st.pm_w_n.up(lg(pm_w) - lnn, n, merge as i64);
            st.wstep_rel.up(merge as f64, n, 0);
            let b = lg(pm_w).floor() - 60.0;
            if b >= 0.0 && (b as usize) < 80 {
                st.hist_pmw[b as usize] += 1;
            }
            if pm_w > cx.t88 {
                st.violate(11, n, &format!("log2 pm_w={:.4}", lg(pm_w)));
            }
            if pm_e > cx.t88 {
                st.violate(12, n, &format!("log2 pm_e={:.4}", lg(pm_e)));
            }
            if cx.m <= 63 && n < 1u128 << cx.m && lg(pm_w) >= cx.cutoff as f64 {
                st.violate(16, n, &format!("log2 pm_w={:.4} c={}", lg(pm_w), cx.cutoff));
            }
            match cx.rec.idx(n) {
                Some(ri) => {
                    st.n_gt88 += 1;
                    st.min_margin_w.up(-(cx.rec.max_lg[ri] - lg(pm_w)), n, cx.rec.id[ri]);
                    st.min_margin_e.up(-(cx.rec.max_lg[ri] - lg(pm_e)), n, cx.rec.id[ri]);
                    if cx.rec.reaches(ri, pm_w) {
                        st.violate(9, n, &format!("record interval #{}", cx.rec.id[ri]));
                    }
                    if cx.rec.reaches(ri, pm_e) {
                        st.violate(10, n, &format!("record interval #{}", cx.rec.id[ri]));
                    }
                }
                None => st.le88_pm_e.up(lg(pm_e), n, 0),
            }
        }
        None => st.violate(5, n, ""),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(args.len() >= 3, "usage: r3check PREFIX OUT.json [--audit-eps N] [--excl-eps N] [--trace FILE] [--label L] [--neg-tested D] [--neg-post24 D] [--neg-screen D] [--neg-span S]");
    let prefix = &args[1];
    let json_out = &args[2];
    let neg_tested: f64 = arg(&args, "--neg-tested", 0.0);
    let neg_post24: f64 = arg(&args, "--neg-post24", 0.0);
    let neg_screen: f64 = arg(&args, "--neg-screen", 0.0);
    let neg_span: usize = arg(&args, "--neg-span", 0usize);
    let audit_eps: usize = arg(&args, "--audit-eps", usize::MAX);
    let excl_eps: usize = arg(&args, "--excl-eps", usize::MAX);
    let label: String = arg(&args, "--label", prefix.rsplit('/').next().unwrap().to_string());
    let trace_file: String = arg(&args, "--trace", String::new());
    let trace: Vec<u128> = if trace_file.is_empty() { Vec::new() } else {
        std::fs::read_to_string(&trace_file).unwrap().split_whitespace().map(|s| s.parse().unwrap()).collect()
    };
    let t0 = std::time::Instant::now();

    // --- config, variants, GPU results (v2check)
    let f = std::fs::File::open(format!("{prefix}.jsonl")).expect("jsonl");
    let mut cfg = Value::Null;
    let mut vars: Vec<Var> = Vec::new();
    let mut g1: HashMap<(String, u32, usize), Value> = HashMap::new();
    let mut gb: Vec<Value> = Vec::new();
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
            "mutate" => panic!("mutated-table dump: not a production run"),
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
    assert_eq!((306 * m) / 485 + 1, offset, "class offset");

    // --- endpoints (v2check)
    let mut raw = Vec::new();
    std::fs::File::open(format!("{prefix}.eps")).expect("eps").read_to_end(&mut raw).unwrap();
    assert_eq!(raw.len() % rec_bytes, 0);
    let mut by_class: BTreeMap<u32, Vec<Ep>> = BTreeMap::new();
    for rec in raw.chunks(rec_bytes) {
        let n0 = u128::from_le_bytes(rec[0..16].try_into().unwrap());
        let mm = u128::from_le_bytes(rec[16..32].try_into().unwrap());
        let f = u32::from_le_bytes(rec[32..36].try_into().unwrap());
        let class = u32::from_le_bytes(rec[36..40].try_into().unwrap());
        let oee = rec[40];
        let rk = (0..words64).map(|w| u64::from_le_bytes(rec[44 + 8 * w..52 + 8 * w].try_into().unwrap())).collect();
        by_class.entry(class).or_default().push(Ep { n0, m: mm, f, oee, rk });
    }
    let mix = Mix::build(13);
    let rec = Rec::new();
    let t88: u128 = RECORDS[0].1.parse().unwrap();
    let cx = Ctx { m, cutoff, mix: &mix, rec: &rec, t88, nn, neg_tested, neg_post24, neg_screen, neg_span };
    let mut stats: HashMap<Key, Stat> = HashMap::new();
    let mut gpu_cmp = json!({});
    let (mut single_ok, mut single_n, mut batch_ok, mut batch_n, mut dig_ok, mut dig_n) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let mut tm_mismatch = 0u64;
    let traces = Mutex::new(Vec::<Value>::new());

    for (&class, eps) in &by_class {
        let b = windows[class as usize] as usize;
        let f_class = offset + class;
        let h0 = 485 * f_class as i32 - 306 * m as i32;
        let codes: Vec<u32> = {
            let mut c: Vec<u32> = vars.iter().map(|v| v.codes[class as usize]).collect();
            c.sort();
            c.dedup();
            c
        };
        struct EpRes {
            aggs: Vec<Agg>,
            tm_bad: bool,
        }
        // per-endpoint statistics are merged into one map per class as each endpoint finishes
        // (keeping them per endpoint costs ~20 KB each: 4 GB at complete N36)
        let class_stats: Mutex<HashMap<Key, Stat>> = Mutex::new(HashMap::new());
        let results: Vec<EpRes> = eps.par_iter().enumerate().map(|(ei, ep)| {
            let mut sm: HashMap<Key, Stat> = HashMap::new();
            let (mm, ff) = titer(ep.n0, m).expect("T^M(n0)");
            let tm_bad = mm != ep.m || ff != ep.f || ff != f_class;
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
            // synthetic endpoints (m != T^M(n0), v2-sound's scheduler delivery test) are compared with
            // the GPU but carry no trajectory: they are excluded from the bound accounting.
            let audit = ei < audit_eps && !tm_bad;
            let mut per_code: Vec<Vec<CaseOut>> = Vec::new();
            for &code in &codes {
                let cp = code_params(code).expect("jd code");
                let mut outs = vec![CaseOut::default(); comps];
                for h in 0..comps {
                    if !(keep_bv[h] && keep_oee[h] && (keep_m9[h] || keep_rows[h])) {
                        continue;
                    }
                    let n = ep.n0 + ((h as u128) << m);
                    let x = ep.m + P3[ep.f as usize] * h as u128;
                    let co = run_case(n, x, h0, m, cp, cutoff, &mix);
                    outs[h] = co;
                    let is_trace = trace.contains(&n);
                    if co.overflow {
                        bump(&mut sm, ("overflow".into(), code, class)).violate(13, n, "run_case");
                        continue;
                    }
                    if (!audit && !is_trace) || tm_bad {
                        continue;
                    }
                    if co.flagged {
                        let st = bump(&mut sm, ("flag".into(), code, class));
                        st.count += 1;
                        // flag boundary: the first boundary >= L
                        let k = growth_bits(cp.w);
                        let (xb, _) = titer(x, co.jumps * cp.w).unwrap();
                        st.bnd_l.up(lg(xb) - (cutoff - k) as f64, n, co.jumps as i64);
                        if is_trace {
                            traces.lock().unwrap().push(json!({"n": n.to_string(), "class": class, "h": h, "code": code,
                                "status": "flagged", "jumps": co.jumps, "flag_step": m + co.jumps * cp.w,
                                "log2_flag_boundary": lg(xb), "log2_L": (cutoff - k) as f64}));
                        }
                        continue;
                    }
                    let kind = if co.exact { "exact" } else { "dip" };
                    let Some(ys) = traj(n, co.win_hi as usize + 2) else {
                        bump(&mut sm, (kind.into(), code, class)).violate(13, n, "traj");
                        continue;
                    };
                    let lo = co.win_lo as usize;
                    let hi = co.win_hi as usize;
                    let found = find_witness(&ys, n, lo, hi.saturating_sub(cx.neg_span).max(lo), 13, 21);
                    let late = found.is_none() && find_witness(&ys, n, 0, hi, 13, 26).is_some();
                    let st = bump(&mut sm, (kind.into(), code, class));
                    account_stop(&cx, st, n, &ys, cp, co.jumps, co.exact, found, late);
                    if is_trace {
                        let pm = ys[..=hi].iter().copied().max().unwrap();
                        traces.lock().unwrap().push(json!({"n": n.to_string(), "class": class, "h": h, "code": code,
                            "status": kind, "jumps": co.jumps, "win": [lo, hi], "witness": found.map(|w| json!([WK_NAMES[w.0 as usize], w.1, w.2])),
                            "log2_pm_span_end": lg(pm)}));
                    }
                }
                per_code.push(outs);
            }
            // exclusions
            if (ei < excl_eps || !trace.is_empty()) && !tm_bad {
                for h in 0..comps {
                    let n = ep.n0 + ((h as u128) << m);
                    let need = [!keep_bv[h], !keep_m9[h], !keep_rows[h], !keep_oee[h]];
                    if !need.iter().any(|&z| z) {
                        continue;
                    }
                    let is_trace = trace.contains(&n);
                    if !(ei < excl_eps) && !is_trace {
                        continue;
                    }
                    let Some(ys) = traj(n, m as usize + b + 2) else {
                        bump(&mut sm, ("x_overflow".into(), 0, class)).violate(13, n, "excl traj");
                        continue;
                    };
                    if need[0] {
                        // r3pr-gpu: class-specific screen bound on the window values (pr-proof section 3):
                        // T^(M+j)(n) + 1 <= 3^F (n + 64) 1.5^j / 2^M, j = 0..=B.
                        {
                            let st = bump(&mut sm, ("x_bitvector".into(), 0, class));
                            let l3f = (ep.f as f64) * 3f64.log2();
                            let ln64 = lg(n + 64);
                            let l15 = 1.5f64.log2();
                            let mut worst = f64::NEG_INFINITY;
                            for j in 0..=b {
                                let lhs = lg(ys[m as usize + j] + 1);
                                let rhs = l3f + ln64 + j as f64 * l15 - m as f64;
                                worst = worst.max(lhs - rhs);
                            }
                            st.screen_slack.up(worst, n, ep.f as i64);
                            if worst > -cx.neg_screen + 1e-9 {
                                st.violate(15, n, &format!("log2 excess {worst:.4} over the screen bound"));
                            }
                        }
                        let hm = cutoff_f; // H at depth M
                        let sc = window_cert(&ys[m as usize..], b, &mix, hm).map(|(r, k)| (m as usize + r, k));
                        let st = bump(&mut sm, ("x_bitvector".into(), 0, class));
                        if sc.is_none() {
                            st.violate(7, n, "bitvector exclusion without a credit position > H_M");
                        }
                        account_excl(&cx, st, n, &ys, m as usize, m as usize + b, 13, 21, sc);
                    }
                    if need[1] {
                        account_excl(&cx, bump(&mut sm, ("x_mod9".into(), 0, class)), n, &ys, 0, 0, 3, 6, Some((0, 5)));
                    }
                    if need[2] {
                        let st = bump(&mut sm, ("x_rows".into(), 0, class));
                        if h == 0 {
                            st.violate(5, n, "h = 0 excluded by rows");
                        }
                        account_excl(&cx, st, n, &ys, 0, 12, 18, 30, None);
                    }
                    if need[3] {
                        let sc = if ep.oee == 2 { m as usize + 1 } else { m as usize + 2 };
                        account_excl(&cx, bump(&mut sm, ("x_oee".into(), 0, class)), n, &ys, 0, m as usize + 2, 13, 21, Some((sc, 3)));
                    }
                    if is_trace {
                        traces.lock().unwrap().push(json!({"n": n.to_string(), "class": class, "h": h, "status": "excluded",
                            "by": {"bitvector": need[0], "mod9": need[1], "rows": need[2], "oee": need[3]}}));
                    }
                }
            }
            // per-variant aggregates (v2check)
            let mut aggs = Vec::with_capacity(vars.len());
            for var in &vars {
                let code = var.codes[class as usize];
                let ci = codes.iter().position(|&c| c == code).unwrap();
                let rows = var.rows[class as usize];
                // v-n73-sound: the n interval of the range code (as v-n73-sound's v3check)
                let pw = |e: u32| 1u128 << e;
                let (nlo, nhi) = match var.range[class as usize] {
                    0 => (0, pw(nn)),
                    1 => (pw(nn - 1), pw(nn)),
                    2 => (0, pw(nn - 1)),
                    3 => (pw(nn - 2), pw(nn)),
                    4 => (0, pw(nn - 2)),
                    5 => (pw(nn - 2), pw(nn - 1)),
                    r => panic!("unknown range code {r}"),
                };
                let mut g = Agg::default();
                for h in 0..(1usize << a) {
                    let nh = ep.n0 + ((h as u128) << m);
                    if !(nh >= nlo && nh < nhi) {
                        continue;
                    }
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
                aggs.push(g);
            }
            {
                let mut g = class_stats.lock().unwrap();
                for (k, v) in sm {
                    g.entry(k).or_default().merge(&v);
                }
            }
            EpRes { aggs, tm_bad }
        }).collect();

        // --- GPU comparison (v2check): every endpoint of the class is covered by the whole-class batch
        let mut cls = json!({});
        for (vi, var) in vars.iter().enumerate() {
            let per_ep: Vec<&Agg> = results.iter().map(|r| &r.aggs[vi]).collect();
            let (mut s_ok, mut s_n) = (0u64, 0u64);
            for (i, g) in per_ep.iter().enumerate() {
                let Some(r) = g1.get(&(var.name.clone(), class, i)) else { continue };
                s_n += 1;
                let flags: Vec<u128> = r["flags"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().parse().unwrap()).collect();
                if r["cases"].as_u64().unwrap() == g.cases && r["iters"].as_u64().unwrap() == g.iters
                    && r["hash"].as_u64().unwrap() as u32 == g.hash && flags == g.flags {
                    s_ok += 1;
                }
            }
            let (mut b_ok, mut b_n, mut d_ok, mut d_n, mut whole) = (0u64, 0u64, 0u64, 0u64, false);
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
                b_n += 1;
                if ok_core && ok_dig {
                    b_ok += 1;
                }
                if dig_layout {
                    d_n += 1;
                    if ok_dig {
                        d_ok += 1;
                    }
                }
                if k == eps.len() && ok_core && ok_dig {
                    whole = true;
                }
            }
            single_ok += s_ok;
            single_n += s_n;
            batch_ok += b_ok;
            batch_n += b_n;
            dig_ok += d_ok;
            dig_n += d_n;
            cls[&var.name] = json!({"code": var.codes[class as usize], "single": [s_ok, s_n], "batches": [b_ok, b_n],
                "digests": [d_ok, d_n], "whole_class_batch_matches": whole});
        }
        gpu_cmp[class.to_string()] = cls;
        for r in &results {
            if r.tm_bad {
                tm_mismatch += 1;
            }
        }
        for (k, v) in class_stats.into_inner().unwrap() {
            stats.entry(k).or_default().merge(&v);
        }
        eprintln!("class {class:2}: {} endpoints done at {:.1}s", eps.len(), t0.elapsed().as_secs_f64());
    }

    // --- aggregate by (kind, code) and by kind
    let mut by_kc: BTreeMap<(String, u32), Stat> = BTreeMap::new();
    let mut by_k: BTreeMap<String, Stat> = BTreeMap::new();
    for ((k, c, _cl), v) in &stats {
        by_kc.entry((k.clone(), *c)).or_default().merge(v);
        by_k.entry(k.clone()).or_default().merge(v);
    }
    let mut per_class = serde_json::Map::new();
    let mut keys: Vec<&Key> = stats.keys().collect();
    keys.sort();
    for k in keys {
        per_class.insert(format!("{}|{}|{}", k.0, k.1, k.2), stats[k].js());
    }
    let total_viol: u64 = by_k.values().map(|s| s.viol.iter().sum::<u64>()).sum();
    let out = json!({
        "label": label, "prefix": prefix, "seconds": t0.elapsed().as_secs_f64(),
        "cfg": cfg, "codes_by_class": by_class.keys().map(|&c| (c.to_string(), json!(vars.iter().map(|v| v.codes[c as usize]).collect::<Vec<_>>()))).collect::<serde_json::Map<_, _>>(),
        "gpu_match": {"single": [single_ok, single_n], "batches": [batch_ok, batch_n], "digest_batches": [dig_ok, dig_n], "tm_mismatch": tm_mismatch, "per_class": gpu_cmp},
        "total_violations": total_viol,
        "by_kind_code": by_kc.iter().map(|((k, c), v)| (format!("{k}|{c}"), v.js())).collect::<serde_json::Map<_, _>>(),
        "by_kind": by_k.iter().map(|(k, v)| (k.clone(), v.js())).collect::<serde_json::Map<_, _>>(),
        "per_class": per_class,
        "traces": traces.into_inner().unwrap(),
    });
    std::fs::write(json_out, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    for ((k, c), v) in &by_kc {
        eprintln!("{label} {k:12} code {c:3}: n={:10} pm_w {:8.3} pm_e {:8.3} tested/L {:8.3} post_e/xP {:7.3} post_e/n {:7.3} viol {:?}",
            v.count, v.pm_w.v, v.pm_e.v, v.tested_l.v, v.post_rel_e.v, v.post_n_e.v,
            V_NAMES.iter().zip(v.viol.iter()).filter(|(_, &x)| x > 0).collect::<Vec<_>>());
    }
    eprintln!("{label}: gpu single {single_ok}/{single_n} batches {batch_ok}/{batch_n} digests {dig_ok}/{dig_n} tm_mismatch {tm_mismatch}; total violations {total_viol}; {:.1}s",
        t0.elapsed().as_secs_f64());
}
