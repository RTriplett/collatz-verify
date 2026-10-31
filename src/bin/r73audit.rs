//! n73-port: r3pr-proof's r3audit.rs (copied) plus N73 configurations (cfg_of), nothing else changed.
//! r3pr-proof: CPU audits of the path-record completeness lemma for the ROUND-3
//! configurations (no GPU).  Adapted from pr-proof's praudit.rs (round 2, fxr2).
//!
//! Semantics (CPU reference = what the round-3 GPU kernels match, c3-final section 5 and
//! v3-sound): tree rules improved(13) plus the K38 walkback (records mode), shared
//! bitvectors with pm_inf + mix13 credits, the (12, 9) row table on EVERY class and on the
//! CPU-only classes (rows from class 0), the carried OEE state, code 91 (W12, KA1, K = 8)
//! on every bitvector class, and the original Base loop on the CPU-fallback classes.
//!
//!   --config full : N72, A = 10, M = 62, F24 windows, GPU F = 40..55, fallback F = 56..62, h in [0, 1024)
//!   --config top  : N72, A = 11, M = 61, W11 windows, GPU F = 39..54, fallback F = 55..61, h in [1024, 2048)
//!   --config a11  : as top, every h in [0, 2048) (records mode: the known records lie below 2^71)
//!
//! r3audit records FILE CUTOFFS --config C
//!     Traces every record "k n t" through the tree, screens and tail at each cutoff.
//! r3audit bounds --config C --per-class E --seed S --cutoff C [--classes L] [--fallback E2]
//!     Sampled audit (uniform baseline endpoints per class, every completion of the range):
//!       * bitvector exclusions: first firing window position (mirror of window_dip_ext),
//!         certificate re-verified forward, largest value up to the merge point vs the class
//!         bound (3^F / 2^M)(n + 64) 1.5^B;
//!       * tail cases (code 91, bit-exact with cv::jdip::jump_dip_until_k): every examined value
//!         < 2^c; the in-jump overshoot between two tested boundaries <= L 1.5^7 (L = 2^(c-8));
//!         after the stop boundary b: the jump (b, b+12] < L 1.5^12 and <= n 2^23.0 (H argument),
//!         the fired window (b+12, witness] <= 3^12 n (table-free); the earliest explicit witness
//!         (descent, or a partner p < n found by inverse search and re-verified forward) within
//!         24 steps of b; the H lemma T^i(n) <= n 2^(H_i/306) for every i <= min(witness, d(n));
//!       * CPU-fallback classes (Base loop): every unflagged case's pre-descent maximum <= 2^c - 2.
use cv::arith::POW3;
use cv::jdip::{dip_eligible, jump_dip_until_k, DipTable};
use cv::jump::{descends_coarse, growth_bits, jd_kind, production_window_rules, JumpTable, TailKind};
use cv::lookahead::{BvMode, Lookahead, WindowRules, START_UNKNOWN};
use cv::rowsieve::RowSieve;
use cv::rules::{Rules, Walkback};
use cv::sample::Sampler;
use cv::tree::Node;
use rayon::prelude::*;

const CODE: u32 = 91;
const F24: [u32; 16] = [28, 35, 35, 35, 35, 34, 33, 33, 27, 27, 27, 27, 27, 27, 27, 27];
const W11: [u32; 16] = [28, 34, 35, 36, 34, 34, 33, 33, 29, 26, 25, 26, 27, 27, 27, 27];

#[derive(Clone)]
struct Cfg {
    name: String,
    m: u32,
    a: u32,
    windows: [u32; 16],
    h_lo: u128,
    h_hi: u128,
    fb_lo: u32,
    fb_hi: u32,
}

fn cfg_of(name: &str) -> Cfg {
    match name {
        "full" => Cfg { name: name.into(), m: 62, a: 10, windows: F24, h_lo: 0, h_hi: 1024, fb_lo: 56, fb_hi: 62 },
        "top" => Cfg { name: name.into(), m: 61, a: 11, windows: W11, h_lo: 1024, h_hi: 2048, fb_lo: 55, fb_hi: 61 },
        "a11" => Cfg { name: name.into(), m: 61, a: 11, windows: W11, h_lo: 0, h_hi: 2048, fb_lo: 55, fb_hi: 61 },
        // n73-port: N73 configurations (M = 73 - A; GPU F = offset..offset+15, fallback above, as lookahead.rs).
        "n73a10" => Cfg { name: name.into(), m: 63, a: 10, windows: F24, h_lo: 0, h_hi: 1024, fb_lo: 56, fb_hi: 63 },
        "n73a10ge71" => Cfg { name: name.into(), m: 63, a: 10, windows: F24, h_lo: 256, h_hi: 1024, fb_lo: 56, fb_hi: 63 },
        "n73a10top" => Cfg { name: name.into(), m: 63, a: 10, windows: F24, h_lo: 512, h_hi: 1024, fb_lo: 56, fb_hi: 63 },
        "n73a11" => Cfg { name: name.into(), m: 62, a: 11, windows: F24, h_lo: 0, h_hi: 2048, fb_lo: 56, fb_hi: 62 },
        "n73a11ge71" => Cfg { name: name.into(), m: 62, a: 11, windows: F24, h_lo: 512, h_hi: 2048, fb_lo: 56, fb_hi: 62 },
        "n73a11w" => Cfg { name: name.into(), m: 62, a: 11, windows: W11, h_lo: 0, h_hi: 2048, fb_lo: 56, fb_hi: 62 },
        "n73a11top" => Cfg { name: name.into(), m: 62, a: 11, windows: F24, h_lo: 1024, h_hi: 2048, fb_lo: 56, fb_hi: 62 },
        "n73a12" => Cfg { name: name.into(), m: 61, a: 12, windows: W11, h_lo: 0, h_hi: 4096, fb_lo: 55, fb_hi: 61 },
        "n73a12ge71" => Cfg { name: name.into(), m: 61, a: 12, windows: W11, h_lo: 1024, h_hi: 4096, fb_lo: 55, fb_hi: 61 },
        "n73a12top" => Cfg { name: name.into(), m: 61, a: 12, windows: W11, h_lo: 2048, h_hi: 4096, fb_lo: 55, fb_hi: 61 },
        "n73a9" => Cfg { name: name.into(), m: 64, a: 9, windows: F24, h_lo: 0, h_hi: 512, fb_lo: 57, fb_hi: 64 },
        _ => panic!("--config full|top|a11|n73a10|n73a10ge71|n73a10top|n73a11|n73a11ge71|n73a11w|n73a11top|n73a12|n73a12ge71|n73a12top|n73a9"),
    }
}

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(default)
}

#[inline]
fn t_step(y: u128) -> Option<u128> {
    if y & 1 == 1 { y.checked_mul(3)?.checked_add(1).map(|v| v / 2) } else { Some(y / 2) }
}

#[inline]
fn lg(x: u128) -> f64 {
    if x == 0 { return f64::NEG_INFINITY; }
    let b = 128 - x.leading_zeros();
    if b <= 52 { (x as f64).log2() } else { (b - 52) as f64 + ((x >> (b - 52)) as f64).log2() }
}

/// 256-bit replay from n: (log2 max before first descent, peak step, first descent step).
fn premax_wide(n: u128) -> (f64, u64, u64) {
    let (mut hi, mut lo) = (0u128, n);
    let (mut mhi, mut mlo) = (0u128, n);
    let (mut step, mut pk) = (0u64, 0u64);
    loop {
        if hi == 0 && lo < n {
            let l = if mhi > 0 { 128.0 + (mhi as f64 + mlo as f64 / 2f64.powi(128)).log2() } else { lg(mlo) };
            return (l, pk, step);
        }
        if lo & 1 == 0 {
            lo = (lo >> 1) | (hi << 127);
            hi >>= 1;
        } else {
            let h_lo = (lo >> 1) | (hi << 127);
            let h_hi = hi >> 1;
            let (s1, c1) = lo.overflowing_add(h_lo);
            let (s2, c2) = s1.overflowing_add(1);
            hi = hi + h_hi + c1 as u128 + c2 as u128;
            lo = s2;
        }
        step += 1;
        if (hi, lo) > (mhi, mlo) {
            mhi = hi;
            mlo = lo;
            pk = step;
        }
    }
}

/// Generic inverse search: some p with 1 <= p < n and T^s(p) = y, s in 1..=smax,
/// at most omax inverse odd steps.
fn find_partner(y: u128, n: u128, omax: u32, smax: u32) -> Option<(u128, u32, u32)> {
    let mut stack: Vec<(u128, u32, u32)> = vec![(y, 0, 0)];
    while let Some((v, s, o)) = stack.pop() {
        if s >= 1 && v < n && v >= 1 {
            return Some((v, s, o));
        }
        if s == smax {
            continue;
        }
        let r = (omax - o).min(smax - s);
        let lhs = v.checked_mul(1u128 << r.min(100));
        let rhs = POW3.get(r as usize).and_then(|p| p.checked_mul(n + 1));
        match (lhs, rhs) {
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

fn forward(mut p: u128, s: u32) -> Option<u128> {
    for _ in 0..s {
        p = t_step(p)?;
    }
    Some(p)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind { Desc, Pm1, Oee(u32), PmInf(u32), Mix(u32, u32, u128) }

fn v3(mut x: u64) -> u32 {
    let mut t = 0;
    while x != 0 && x % 3 == 0 { x /= 3; t += 1; }
    t
}

/// Mirror of lookahead::window_dip_ext returning the FIRST firing window step.
fn first_cert(x0: u64, b: u32, start_evens: u32, wr: &WindowRules, cutoff: i32) -> Option<(Kind, u32, u32)> {
    let x0 = if x0 == 0 { 1u64 << b } else { x0 };
    let mut updown = 0u64;
    let mut x = x0;
    for i in 0..b {
        if x & 1 == 0 { x = x + x / 2; updown |= 1 << i; } else { x = x / 2 + 1; }
    }
    let (num, den) = (306i32, 485i32);
    let extra = wr.pm_inf || wr.mix.is_some();
    let known = start_evens != START_UNKNOWN;
    let (mut cur, mut evens, mut fw) = (0i32, if known { start_evens } else { 0 }, 0u32);
    let mut x = x0;
    for i in 0..b {
        if updown & (1u64 << i) != 0 {
            let a = ((updown >> i) + 1).trailing_zeros();
            let oee_ok = i + a + 2 < b && (updown >> (i + a)) & 0b11 == 0;
            if cur + num > cutoff && oee_ok {
                return Some((Kind::Oee(a), i, i + a + 2));
            }
            cur -= den - num;
            evens = 0;
            fw += 1;
            x = x + x / 2;
        } else {
            cur += num;
            evens += 1;
            if evens % 2 == 0 && (known || fw > 0) && cur + (den - num) > cutoff {
                return Some((Kind::Pm1, i, i + 1));
            }
            x = x / 2 + 1;
        }
        if cur > cutoff {
            return Some((Kind::Desc, i, i + 1));
        }
        if extra && fw > 0 {
            if wr.pm_inf {
                let t = v3(x).min(fw);
                if cur + (den - num) * t as i32 > cutoff {
                    return Some((Kind::PmInf(t), i, i + 1));
                }
            }
            if let Some(tab) = &wr.mix {
                let jp = fw.min(tab.j);
                let r = (x - 1) % POW3[jp as usize] as u64;
                if let Some((s, o, bb)) = tab.lookup(jp, r) {
                    if cur + den * o as i32 - num * s as i32 > cutoff {
                        return Some((Kind::Mix(s, o, bb), i, i + 1));
                    }
                }
            }
        }
    }
    None
}

fn verify_cert(kind: Kind, i: u32, vals: &[u128], n: u128) -> bool {
    let i = i as usize;
    match kind {
        Kind::Desc => vals[i + 1] < n,
        Kind::Pm1 => {
            let y = vals[i + 1];
            if y % 3 != 2 { return false; }
            let q = (2 * y - 1) / 3;
            q >= 1 && q < n && t_step(q) == Some(y)
        }
        Kind::Oee(a) => {
            let xi = vals[i];
            if xi < 3 || xi & 1 == 0 { return false; }
            let p = (xi - 1) / 2;
            p >= 1 && p < n && forward(p, a + 1) == Some(vals[i + a as usize + 2])
        }
        Kind::PmInf(t) => {
            let y = vals[i + 1];
            if (y + 1) % POW3[t as usize] != 0 { return false; }
            let u = (((y + 1) / POW3[t as usize]) << t) - 1;
            u >= 1 && u < n && forward(u, t) == Some(y)
        }
        Kind::Mix(s, o, bb) => {
            let y = vals[i + 1];
            let inv3: u128 = 0xAAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAB;
            let mut inv = 1u128;
            for _ in 0..o { inv = inv.wrapping_mul(inv3); }
            let p = y.wrapping_shl(s).wrapping_sub(bb).wrapping_mul(inv);
            p >= 1 && p < n && forward(p, s) == Some(y)
        }
    }
}

struct Ctx {
    cfg: Cfg,
    /// negative controls (0 = off)
    span_cut: u32,
    bound_delta: f64,
    exam_bits: u32,
    la: Lookahead,
    rows: std::sync::Arc<RowSieve>,
    jt: JumpTable,
    dt: DipTable,
    w: u32,
    wd: u32,
    ka: u32,
    wr: WindowRules,
}

impl Ctx {
    fn new(cfg: Cfg) -> Self {
        let wr = production_window_rules().clone();
        let la = Lookahead::meta(cfg.m, cfg.a, 25, 16, BvMode::Shared, &wr).with_class_windows(&cfg.windows);
        let rows = RowSieve::standard(cfg.m, cfg.a);
        let Some(TailKind::JumpDip { w, wd, dq, ka, desc }) = jd_kind(CODE) else { panic!() };
        assert_eq!((w, wd, dq, ka, desc), (12, 12, 128, 1, false));
        Ctx { cfg, span_cut: 0, bound_delta: 0.0, exam_bits: 0, la, rows, jt: JumpTable::new(w), dt: DipTable::build(wd, &wr, dq, desc), w, wd, ka, wr }
    }
}

/// jd reference with the stop boundary: (flagged, jumps, state at stop, small base).
fn jd_trace(base: u128, x0: u128, jt: &JumpTable, dt: &DipTable, h0: i32, flag_at: u128, ka: u32) -> (bool, u64, u128, bool) {
    if !dip_eligible(base) {
        let (mut x, mut jumps) = (x0, 0u64);
        loop {
            if x >= flag_at { return (true, jumps, x, true); }
            x = jt.jump(x);
            jumps += 1;
            if descends_coarse(x, base) { return (false, jumps, x, true); }
        }
    }
    let w = jt.w;
    let mj = (1u64 << w) - 1;
    let g = |r: u64| 485 * jt.odd[r as usize] as i32 - 306 * w as i32;
    let (mut x, mut h, mut jumps) = (x0, h0, 0u64);
    loop {
        if x >= flag_at { return (true, jumps, x, false); }
        let mut y = x as u64;
        let mut hh = h;
        for _ in 0..ka {
            let r = y & mj;
            hh += g(r);
            y = (jt.mult[r as usize] as u64).wrapping_mul(y >> w).wrapping_add(jt.add[r as usize] as u64);
        }
        if dt.get(y as u128) > hh { return (false, jumps, x, false); }
        h += g(x as u64 & mj);
        x = jt.jump(x);
        jumps += 1;
    }
}

fn base_flag(base: u128, x: u128, cutoff_bits: u32) -> bool {
    cv::tail::step_until_below(base, x + 1, 1u128 << cutoff_bits).2
}

fn records(args: &[String]) {
    let file = &args[2];
    let cutoffs: Vec<u32> = args[3].split(',').map(|s| s.parse().unwrap()).collect();
    let cfg = cfg_of(&arg(args, "--config", "full".to_string()));
    let ctx = Ctx::new(cfg.clone());
    let m = cfg.m;
    let mut rules = Rules::improved(13);
    rules.walkback = Some(Walkback { depth: 38, max_states: 65_536 });
    println!("CONFIG records {} M {} A {} code {CODE} windows {:?} rows from class 0 cutoffs {cutoffs:?}", cfg.name, cfg.m, cfg.a, cfg.windows);
    for line in std::fs::read_to_string(file).unwrap().lines() {
        let v: Vec<&str> = line.split_whitespace().collect();
        if v.len() < 3 { continue; }
        let (k, n, t): (u32, u128, String) = (v[0].parse().unwrap(), v[1].parse().unwrap(), v[2].to_string());
        let (lpm, pk, desc) = premax_wide(n);
        let mut nd = Node::root(&rules);
        let mut cull = None;
        if n & 3 != 3 { cull = Some((2, "root(0,1,2 mod 4)".to_string())); }
        while cull.is_none() && nd.k < m {
            let kids = nd.children(&rules);
            let bit = ((n >> nd.k) & 1) as usize;
            match kids[bit] {
                None => { cull = Some((nd.k + 1, "oee".into())); }
                Some(c) => {
                    if let Some(why) = c.cull(&rules) { cull = Some((c.k, format!("{why:?}"))); }
                    nd = c;
                }
            }
        }
        let mut line_out = format!("#{k} log2n={:.3} premax_log2={lpm:.3} peak={pk} desc={desc} published_t_digits={} ", lg(n), t.len());
        if let Some((d, why)) = cull {
            println!("{line_out}TREE-CULLED at depth {d} by {why}  <-- would contradict record status");
            continue;
        }
        assert_eq!(nd.n0, n & ((1u128 << m) - 1));
        let f = nd.f;
        let h = n >> m;
        let x = nd.m + POW3[f as usize] * h;
        assert_eq!(cv::arith::t_iter(n, m), (x, f));
        let class = f.checked_sub(ctx.la.offset).map(|c| c as usize).filter(|&c| c < 16);
        let in_range = h >= cfg.h_lo && h < cfg.h_hi;
        let rows_ok = ctx.rows.keeps(n);
        let bv_ok = ctx.la.keeps(f, nd.ne, (x + 1) as u64);
        let oee_ok = !((nd.oee == 2 && x & 1 == 0) || (nd.oee == 1 && x & 3 == 0));
        line_out += &format!("F={f} class={class:?} h={h} in_range={in_range} log2TM={:.2} oee={} rows_ok={rows_ok} bv_keep={bv_ok} oee_keep={oee_ok} |",
                             lg(x), nd.oee);
        if !(rows_ok && bv_ok && oee_ok) {
            println!("{line_out} EXCLUDED  <-- would contradict record status");
            continue;
        }
        for &c in &cutoffs {
            match class {
                Some(_) => {
                    let kk = growth_bits(ctx.w);
                    let h0 = 485 * f as i32 - 306 * m as i32;
                    let flag_at = 1u128 << (c - kk);
                    let (fl, jumps, xs, small) = jd_trace(n, x, &ctx.jt, &ctx.dt, h0, flag_at, ctx.ka);
                    let r = jump_dip_until_k(n, x, &ctx.jt, &ctx.dt, h0, flag_at, ctx.ka);
                    assert_eq!((r.1, r.2), (jumps, fl), "trace != cv::jdip reference");
                    line_out += &format!(" c{c}:{}@j{jumps}(x=2^{:.2}{})", if fl { "FLAG" } else { "stop" }, lg(xs), if small { ",small" } else { "" });
                }
                None => {
                    let fl = base_flag(n, x, c);
                    line_out += &format!(" c{c}:{}(base)", if fl { "FLAG" } else { "stop" });
                }
            }
        }
        println!("{line_out}");
    }
}

#[derive(Default, Clone)]
struct St {
    endpoints: u64,
    completions: u64,
    rows_ex: u64,
    oee_ex: u64,
    bv_ex: u64,
    bv_kind: [u64; 5],
    bv_bad: u64,
    bv_mismatch: u64,
    bv_max_log2: f64,
    bv_max_ratio: f64,
    bv_max_merge_off: u32,
    bv_bound_viol: u64,
    bv_max_vs_class_bound: f64,
    tree_max_ratio: f64,
    cases: u64,
    flagged: u64,
    stops: u64,
    small: u64,
    ref_mismatch: u64,
    exam_max_log2: f64,
    exam_viol: u64,
    injump_max_over_l: f64,
    injump_viol: u64,
    pj_max_over_n: f64,
    pj_max_over_l: f64,
    pj_viol: u64,
    fw_max_over_n: f64,
    fw_viol: u64,
    post_max_log2: f64,
    post_max_off: u32,
    pre_cert_ge_cutoff: u64,
    no_witness: u64,
    hlem_checked: u64,
    hlem_max_excess: f64,
    hlem_viol: u64,
    pmw_max_log2: f64,
    fb_cases: u64,
    fb_flagged: u64,
    fb_max_log2: f64,
    fb_viol: u64,
}

impl St {
    /// Identity element: maxima of quantities that can be negative start at -inf.
    fn fresh() -> St {
        St { injump_max_over_l: f64::NEG_INFINITY, pj_max_over_l: f64::NEG_INFINITY, pj_max_over_n: f64::NEG_INFINITY,
             fw_max_over_n: f64::NEG_INFINITY, hlem_max_excess: f64::NEG_INFINITY, bv_max_vs_class_bound: f64::NEG_INFINITY,
             ..Default::default() }
    }
    fn add(&mut self, o: &St) {
        self.endpoints += o.endpoints; self.completions += o.completions;
        self.rows_ex += o.rows_ex; self.oee_ex += o.oee_ex;
        self.bv_ex += o.bv_ex; for i in 0..5 { self.bv_kind[i] += o.bv_kind[i]; }
        self.bv_bad += o.bv_bad; self.bv_mismatch += o.bv_mismatch;
        self.bv_max_log2 = self.bv_max_log2.max(o.bv_max_log2); self.bv_max_ratio = self.bv_max_ratio.max(o.bv_max_ratio);
        self.bv_max_merge_off = self.bv_max_merge_off.max(o.bv_max_merge_off); self.bv_bound_viol += o.bv_bound_viol;
        self.bv_max_vs_class_bound = self.bv_max_vs_class_bound.max(o.bv_max_vs_class_bound);
        self.tree_max_ratio = self.tree_max_ratio.max(o.tree_max_ratio);
        self.cases += o.cases; self.flagged += o.flagged; self.stops += o.stops; self.small += o.small;
        self.ref_mismatch += o.ref_mismatch;
        self.exam_max_log2 = self.exam_max_log2.max(o.exam_max_log2); self.exam_viol += o.exam_viol;
        self.injump_max_over_l = self.injump_max_over_l.max(o.injump_max_over_l); self.injump_viol += o.injump_viol;
        self.pj_max_over_n = self.pj_max_over_n.max(o.pj_max_over_n); self.pj_max_over_l = self.pj_max_over_l.max(o.pj_max_over_l);
        self.pj_viol += o.pj_viol;
        self.fw_max_over_n = self.fw_max_over_n.max(o.fw_max_over_n); self.fw_viol += o.fw_viol;
        self.post_max_log2 = self.post_max_log2.max(o.post_max_log2);
        self.post_max_off = self.post_max_off.max(o.post_max_off);
        self.pre_cert_ge_cutoff += o.pre_cert_ge_cutoff; self.no_witness += o.no_witness;
        self.hlem_checked += o.hlem_checked; self.hlem_max_excess = self.hlem_max_excess.max(o.hlem_max_excess);
        self.hlem_viol += o.hlem_viol;
        self.pmw_max_log2 = self.pmw_max_log2.max(o.pmw_max_log2);
        self.fb_cases += o.fb_cases; self.fb_flagged += o.fb_flagged; self.fb_max_log2 = self.fb_max_log2.max(o.fb_max_log2);
        self.fb_viol += o.fb_viol;
    }
    fn violations(&self) -> u64 {
        self.bv_bad + self.bv_mismatch + self.bv_bound_viol + self.ref_mismatch + self.exam_viol + self.injump_viol
            + self.pj_viol + self.fw_viol + self.pre_cert_ge_cutoff + self.no_witness + self.hlem_viol + self.fb_viol
    }
    fn line(&self) -> String {
        format!("endpoints {} completions {} | rows-ex {} oee-ex {} | bv-ex {} kinds[desc,pm1,oee,pminf,mix] {:?} bad-cert {} mirror-mismatch {} max-merge-off {} max log2(value<=merge) {:.3} max log2(value/n) {:.3} max log2(value/classbound) {:.3} bound-viol {} | tree max log2(T^i/n, i<=M) {:.3} | cases {} flagged {} stops {} small {} ref-mismatch {} | examined max log2 {:.3} viol {} | in-jump max log2(y/L) {:.4} viol {} | post-jump max log2(y/n) {:.3} max log2(y/L) {:.3} viol {} | fired-window max log2(y/n) {:.3} viol {} | post max log2 {:.3} max witness off {} pre-cert>=2^c {} no-witness {} | H-lemma values {} max excess {:.3e} viol {} | max log2 pre-witness (unflagged) {:.3} | fallback cases {} flagged {} max log2(pre-descent, unflagged) {:.3} viol {} | VIOLATIONS {}",
            self.endpoints, self.completions, self.rows_ex, self.oee_ex, self.bv_ex, self.bv_kind,
            self.bv_bad, self.bv_mismatch, self.bv_max_merge_off, self.bv_max_log2, self.bv_max_ratio, self.bv_max_vs_class_bound, self.bv_bound_viol,
            self.tree_max_ratio, self.cases, self.flagged, self.stops, self.small, self.ref_mismatch, self.exam_max_log2, self.exam_viol,
            self.injump_max_over_l, self.injump_viol, self.pj_max_over_n, self.pj_max_over_l, self.pj_viol, self.fw_max_over_n, self.fw_viol,
            self.post_max_log2, self.post_max_off, self.pre_cert_ge_cutoff, self.no_witness,
            self.hlem_checked, self.hlem_max_excess, self.hlem_viol, self.pmw_max_log2,
            self.fb_cases, self.fb_flagged, self.fb_max_log2, self.fb_viol, self.violations())
    }
}

/// H lemma check on a trajectory slice: values[i] = T^(start+i)(n), with (F, step) at slice start.
/// Checks log2(T^i(n)/n) <= H_i/306 for every value before the first descent.  Returns (checked, max excess).
fn h_lemma(n: u128, vals: &[u128], f0: u32, i0: u32) -> (u64, f64, bool) {
    let (mut f, mut i) = (f0, i0);
    let mut checked = 0u64;
    let mut worst = f64::NEG_INFINITY;
    for (j, &y) in vals.iter().enumerate() {
        if j > 0 {
            // the step from vals[j-1] to vals[j]
            if vals[j - 1] & 1 == 1 { f += 1; }
            i += 1;
        }
        if i > 0 && y < n { break; }
        let h = 485 * f as i64 - 306 * i as i64;
        let ex = (lg(y) - lg(n)) - h as f64 / 306.0;
        if ex > 1e-9 { return (checked + 1, ex, true); }
        if i > 0 { worst = worst.max(ex); }
        checked += 1;
    }
    (checked, worst, worst > 1e-9)
}

fn audit_endpoint(ctx: &Ctx, nd: &Node, cutoff: u32, st: &mut St) {
    let (m, w, wd, ka) = (ctx.cfg.m, ctx.w, ctx.wd, ctx.ka);
    st.endpoints += 1;
    let f = nd.f;
    let class = f.checked_sub(ctx.la.offset).map(|c| c as usize).filter(|&c| c < 16);
    let p3 = POW3[f as usize];
    let h0 = 485 * f as i32 - 306 * m as i32;
    let kk = growth_bits(w);
    let l = 1u128 << (cutoff - kk);
    for h in ctx.cfg.h_lo..ctx.cfg.h_hi {
        st.completions += 1;
        let n = nd.n0 + (h << m);
        let x = nd.m + p3 * h;
        if h % 97 == 0 {
            let mut y = n;
            let mut mx = n;
            for _ in 0..m { y = t_step(y).unwrap(); mx = mx.max(y); }
            debug_assert_eq!(y, x);
            st.tree_max_ratio = st.tree_max_ratio.max(lg(mx) - lg(n));
        }
        // (12, 9) rows on every class (rows from class 0) and on the CPU-only classes
        if !ctx.rows.keeps(n) {
            st.rows_ex += 1;
            continue;
        }
        if let Some(ci) = class {
            let b = ctx.cfg.windows[ci];
            let keep = ctx.la.keeps(f, nd.ne, (x + 1) as u64);
            let fc = first_cert(((x + 1) as u64) & ((1u64 << b) - 1), b, START_UNKNOWN, &ctx.wr, ctx.la.cutoffs[ci]);
            if keep != fc.is_none() { st.bv_mismatch += 1; }
            if let Some((kind, i, off)) = fc {
                st.bv_ex += 1;
                let ki = match kind { Kind::Desc => 0, Kind::Pm1 => 1, Kind::Oee(_) => 2, Kind::PmInf(_) => 3, Kind::Mix(..) => 4 };
                st.bv_kind[ki] += 1;
                let mut vals = Vec::with_capacity(off as usize + 1);
                let mut y = x;
                vals.push(y);
                for _ in 0..off { y = t_step(y).unwrap(); vals.push(y); }
                if !verify_cert(kind, i, &vals, n) { st.bv_bad += 1; }
                let mx = *vals.iter().max().unwrap();
                st.bv_max_log2 = st.bv_max_log2.max(lg(mx));
                st.bv_max_ratio = st.bv_max_ratio.max(lg(mx) - lg(n));
                st.bv_max_merge_off = st.bv_max_merge_off.max(off);
                let class_bound = f as f64 * 3f64.log2() - m as f64 + lg(n + 64) + b as f64 * 1.5f64.log2();
                st.bv_max_vs_class_bound = st.bv_max_vs_class_bound.max(lg(mx) - class_bound);
                if lg(mx) > class_bound + 1e-9 || off > b { st.bv_bound_viol += 1; }
                continue;
            }
        }
        if (nd.oee == 2 && x & 1 == 0) || (nd.oee == 1 && x & 3 == 0) {
            st.oee_ex += 1;
            continue;
        }
        match class {
            None => {
                st.fb_cases += 1;
                if base_flag(n, x, cutoff) {
                    st.fb_flagged += 1;
                } else {
                    let (mut y, mut mx) = (x, x);
                    while y >= n { y = t_step(y).unwrap(); mx = mx.max(y); }
                    st.fb_max_log2 = st.fb_max_log2.max(lg(mx));
                    if mx > (1u128 << cutoff) - 2 { st.fb_viol += 1; }
                }
            }
            Some(_) => {
                st.cases += 1;
                let (fl, jumps, xs, small) = jd_trace(n, x, &ctx.jt, &ctx.dt, h0, l, ka);
                let r = jump_dip_until_k(n, x, &ctx.jt, &ctx.dt, h0, l, ka);
                if (r.1, r.2) != (jumps, fl) { st.ref_mismatch += 1; }
                if fl { st.flagged += 1; continue; }
                if small { st.small += 1; }
                st.stops += 1;
                // examined span [M, stop boundary]: all values, and the overshoot inside jumps
                let mut exam = Vec::with_capacity((jumps * w as u64 + 1) as usize);
                let mut y = x;
                exam.push(y);
                for _ in 0..(jumps * w as u64) { y = t_step(y).unwrap(); exam.push(y); }
                assert_eq!(y, xs);
                let emx = *exam.iter().max().unwrap();
                st.exam_max_log2 = st.exam_max_log2.max(lg(emx));
                let eb = if ctx.exam_bits > 0 { ctx.exam_bits } else { cutoff };
                if emx >= 1u128 << eb { st.exam_viol += 1; }
                for (j, &v) in exam.iter().enumerate() {
                    if j % w as usize != 0 {
                        let r = lg(v) - lg(l);
                        st.injump_max_over_l = st.injump_max_over_l.max(r);
                        if r > 7.0 * 1.5f64.log2() + 1e-9 { st.injump_viol += 1; }
                    }
                }
                // the prefix [0, M] for the H lemma and the pre-witness maximum
                let mut pre = Vec::with_capacity(m as usize + 1);
                let mut z = n;
                pre.push(z);
                for _ in 0..m { z = t_step(z).unwrap(); pre.push(z); }
                let pmx_pre = *pre.iter().max().unwrap();
                if small {
                    // exact descent at a boundary: no post span (examined values checked above)
                    st.pmw_max_log2 = st.pmw_max_log2.max(lg(pmx_pre.max(emx)));
                    continue;
                }
                // after the stop boundary b: the jump (b, b+12] and the fired window (b+12, b+24]
                let span = (ka * w + wd - ctx.span_cut.min(ka * w + wd)) as usize;
                let mut vals = Vec::with_capacity(span + 1);
                let mut q = xs;
                vals.push(q);
                for _ in 0..span { q = t_step(q).unwrap(); vals.push(q); }
                let mut wpos = None;
                for (j, &v) in vals.iter().enumerate() {
                    if v < n { wpos = Some(j); break; }
                    if let Some((p, s, _o)) = find_partner(v, n, 13, 26) {
                        if p >= 1 && p < n && forward(p, s) == Some(v) { wpos = Some(j); break; }
                    }
                }
                let Some(j) = wpos else { st.no_witness += 1; continue; };
                let pmx = *vals[..=j].iter().max().unwrap();
                st.post_max_log2 = st.post_max_log2.max(lg(pmx));
                st.post_max_off = st.post_max_off.max(j as u32);
                for (t, &v) in vals[..=j].iter().enumerate().skip(1) {
                    if t <= w as usize {
                        // post jump: < L 1.5^12 (guard) and <= n 2^((3671 + 306*11)/306) (H argument)
                        let rn = lg(v) - lg(n);
                        let rl = lg(v) - lg(l);
                        st.pj_max_over_n = st.pj_max_over_n.max(rn);
                        st.pj_max_over_l = st.pj_max_over_l.max(rl);
                        if rl > 12.0 * 1.5f64.log2() + 1e-9 || rn > (3671.0 + 306.0 * 11.0) / 306.0 + 1e-9 - ctx.bound_delta { st.pj_viol += 1; }
                    } else {
                        // fired window: <= 3^12 n (table-free: partner with <= 12 odd steps, <= 12 halvings)
                        let rn = lg(v) - lg(n);
                        st.fw_max_over_n = st.fw_max_over_n.max(rn);
                        // exact (3^12 n < 2^92 for n < 2^72); the negative control lowers it by bound_delta bits
                        let viol = if ctx.bound_delta == 0.0 { v >= 531441 * n }
                                   else { rn > 12.0 * 3f64.log2() - ctx.bound_delta };
                        if viol { st.fw_viol += 1; }
                    }
                }
                // H lemma over [0, b + j]: prefix, examined span, post span up to the witness
                let mut traj: Vec<u128> = pre.clone();
                traj.extend_from_slice(&exam[1..]);
                traj.extend_from_slice(&vals[1..=j]);
                let (ck, ex, bad) = h_lemma(n, &traj, 0, 0);
                st.hlem_checked += ck;
                st.hlem_max_excess = st.hlem_max_excess.max(ex);
                if bad { st.hlem_viol += 1; }
                let pmw = pmx_pre.max(emx).max(pmx);
                st.pmw_max_log2 = st.pmw_max_log2.max(lg(pmw));
                if pmx.max(emx) >= 1u128 << cutoff { st.pre_cert_ge_cutoff += 1; }
            }
        }
    }
}

fn bounds(args: &[String]) {
    let cfg = cfg_of(&arg(args, "--config", "full".to_string()));
    let per_class: usize = arg(args, "--per-class", 200);
    let seed: u64 = arg(args, "--seed", 20261464);
    let cutoff: u32 = arg(args, "--cutoff", 104);
    let classes_s: String = arg(args, "--classes", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15".to_string());
    let fallback: usize = arg(args, "--fallback", 0);
    let classes: Vec<usize> = classes_s.split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect();
    let mut ctx = Ctx::new(cfg.clone());
    ctx.span_cut = arg(args, "--span-cut", 0u32);
    ctx.bound_delta = arg(args, "--bound-delta", 0f64);
    ctx.exam_bits = arg(args, "--exam-bits", 0u32);
    let baseline = Rules::baseline();
    let t0 = std::time::Instant::now();
    println!("NEGATIVE-CONTROLS span_cut {} bound_delta {} exam_bits {}", ctx.span_cut, ctx.bound_delta, ctx.exam_bits);
    println!("CONFIG {} bounds M {} A {} h [{}, {}) per-class {per_class} seed {seed} cutoff {cutoff} classes {classes:?} fallback {fallback} (F {}..{}) code {CODE} everywhere windows {:?} rows from class 0",
             cfg.name, cfg.m, cfg.a, cfg.h_lo, cfg.h_hi, cfg.fb_lo, cfg.fb_hi, cfg.windows);
    let mut total = St::fresh();
    let mut groups: Vec<(String, u32, u32, usize, u64)> = classes.iter().map(|&c| {
        let f = ctx.la.offset + c as u32;
        (format!("class {c:2} F={f} B={} code {CODE}", cfg.windows[c]), f, f, per_class, seed ^ (c as u64 + 101).wrapping_mul(0x2545_F491_4F6C_DD1D))
    }).collect();
    if fallback > 0 {
        groups.push((format!("fallback F={}..{} (Base)", cfg.fb_lo, cfg.fb_hi), cfg.fb_lo, cfg.fb_hi, fallback, seed ^ 0x5151_5151));
    }
    for (name, flo, fhi, count, s) in groups {
        let sampler = Sampler::new(&baseline, cfg.m, Some((flo, fhi)));
        let draws = sampler.draws(count, s, None);
        let st = draws.par_iter().fold(St::fresh, |mut st, d| { audit_endpoint(&ctx, &d.node, cutoff, &mut st); st })
            .reduce(St::fresh, |mut a, b| { a.add(&b); a });
        println!("{name}: {} | {:.1}s", st.line(), t0.elapsed().as_secs_f64());
        total.add(&st);
    }
    println!("TOTAL {} cutoff {cutoff}: {} | {:.1}s", cfg.name, total.line(), t0.elapsed().as_secs_f64());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("records") => records(&args),
        Some("bounds") => bounds(&args),
        _ => eprintln!("usage: r3audit records FILE CUTOFFS --config C | r3audit bounds --config C --per-class E --seed S --cutoff C [--classes L] [--fallback E2]"),
    }
}
