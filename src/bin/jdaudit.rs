//! c-tail: independent audit of the jd tails (window-dip certificate tested KA
//! jumps ahead at every jump boundary, cuda/jdip.cuh / cv::jdip).
//!
//! For every case (uniform N-bit endpoints from the cv sampler with fresh
//! seeds, or a complete small-N enumeration with --exhaustive B):
//!   1. the CPU reference cv::jdip::jump_dip_until_k (what the GPU matches),
//!   2. an independent re-implementation that uses no jump table: plain T steps
//!      give every boundary state x_j = T^(jW)(x) and its odd count, H is
//!      recomputed from scratch as 485 F' - 306 M', the tested window's low
//!      bits come from exact iteration (not 64-bit mini jumps), and the rule
//!      (dip > H at the window KA jumps ahead, flag if x_j >= 2^(cutoff-K))
//!      is applied; (hash, jumps, flagged) must equal the reference;
//!   3. at every unflagged stop, an explicit witness within the tested span
//!      (KA W + Wd steps from the stop state): a value below n, or a partner
//!      1 <= p < n with T^s(p) = T^(M'+i)(n) found by a generic bounded
//!      inverse-tree search (s <= SMAX, o <= OMAX) and re-verified forward.
//!      The witness search uses no window, mix or dip code.
//!
//! jdaudit --code C [--n 72 --a 10 --per-class 2000 --seed S --class-b LIST --classes LIST
//!          --cutoff-bits 120 --omax 13 --smax 26 --exhaustive B --dip-min-bits 32 --verbose]
use cv::arith::{mod_pow3, POW3};
use cv::jdip::{jump_dip_until_k, DipTable};
use cv::jump::{growth_bits, jd_kind, jump_until, production_window_rules, JumpTable, TailKind};
use cv::lookahead::{BvMode, Lookahead};
use cv::rules::Rules;
use cv::sample::Sampler;
use cv::tree::{dfs, frontier, Node, Sink, TreeStats};
use rayon::prelude::*;

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(default)
}

#[inline]
fn t_step(y: u128) -> Option<u128> {
    if y & 1 == 1 { y.checked_mul(3)?.checked_add(1).map(|v| v / 2) } else { Some(y / 2) }
}

/// Generic inverse search (vm_audit): some p with 1 <= p < n and T^s(p) = y,
/// s in 1..=smax, at most omax inverse odd steps.  Prunes only branches that
/// cannot reach below n: after r more inverse steps any value is > (2/3)^r v - 1.
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

#[derive(Default, Clone)]
struct St {
    endpoints: u64,
    cases: u64,
    dip_cases: u64,
    stops: u64,
    flagged: u64,
    descent_w: u64,
    merge_w: u64,
    small_exact: u64,
    fail: u64,
    ref_mismatch: u64,
    h_mismatch: u64,
    unverifiable: u64,
    jumps: u64,
    jx_jumps: u64,
    later_than_jx: u64,
    jx_flags: u64,
    flags_not_in_jx: u64,
    jx_flags_lost: u64,
    orig_flags: u64,
    orig_lost: u64,
    not_in_orig: u64,
    wpos: [u64; 8],
}

impl St {
    fn add(&mut self, o: &St) {
        self.endpoints += o.endpoints;
        self.cases += o.cases;
        self.dip_cases += o.dip_cases;
        self.stops += o.stops;
        self.flagged += o.flagged;
        self.descent_w += o.descent_w;
        self.merge_w += o.merge_w;
        self.small_exact += o.small_exact;
        self.fail += o.fail;
        self.ref_mismatch += o.ref_mismatch;
        self.h_mismatch += o.h_mismatch;
        self.unverifiable += o.unverifiable;
        self.jumps += o.jumps;
        self.jx_jumps += o.jx_jumps;
        self.later_than_jx += o.later_than_jx;
        self.jx_flags += o.jx_flags;
        self.flags_not_in_jx += o.flags_not_in_jx;
        self.jx_flags_lost += o.jx_flags_lost;
        self.orig_flags += o.orig_flags;
        self.orig_lost += o.orig_lost;
        self.not_in_orig += o.not_in_orig;
        for i in 0..8 {
            self.wpos[i] += o.wpos[i];
        }
    }
    fn line(&self) -> String {
        format!("endpoints {} cases {} dip-cases {} stops {} flagged {} | witnesses descent {} merge {} small-exact {} | FAIL {} ref_mismatch {} h_mismatch {} unverifiable {} | jumps {} vs jx {} ({:+.2}%) later-than-jx {} | flags jx {} new {} lost {} | orig flags {} lost {} not-in-orig {} | witness pos/12 {:?}",
            self.endpoints, self.cases, self.dip_cases, self.stops, self.flagged, self.descent_w, self.merge_w, self.small_exact,
            self.fail, self.ref_mismatch, self.h_mismatch, self.unverifiable, self.jumps, self.jx_jumps,
            100.0 * (self.jumps as f64 / self.jx_jumps.max(1) as f64 - 1.0), self.later_than_jx,
            self.jx_flags, self.flags_not_in_jx, self.jx_flags_lost, self.orig_flags, self.orig_lost, self.not_in_orig, self.wpos)
    }
}

struct Cfg {
    jt: JumpTable,
    dt: DipTable,
    w: u32,
    wd: u32,
    ka: u32,
    m: u32,
    cutoff_bits: u32,
    omax: u32,
    smax: u32,
    dip_min_bits: u32,
    verbose: bool,
    /// Negative controls: bias subtracted from H in the rule (fires too early),
    /// and steps removed from the witness span.
    h_bias: i64,
    span_cut: u32,
}

/// Independent rule: plain steps, H from scratch.  Returns (hash, jumps, flagged, stop state, H at stop, H mismatch).
fn independent(n: u128, x: u128, f: u32, c: &Cfg) -> Option<(u32, u64, bool, u128)> {
    let k = growth_bits(c.w);
    let flag_at = 1u128 << (c.cutoff_bits - k);
    let (mut y, mut odd, mut depth) = (x, f as i64, c.m as i64);
    let (mut hash, mut jumps) = (0u32, 0u64);
    loop {
        if y >= flag_at {
            return Some((hash, jumps, true, y));
        }
        // window KA jumps ahead: exact iteration
        let (mut z, mut zo) = (y, odd);
        for _ in 0..(c.ka * c.w) {
            if z & 1 == 1 { zo += 1; }
            z = t_step(z)?;
        }
        let hz = 485 * zo - 306 * (depth + (c.ka * c.w) as i64) - c.h_bias;
        if c.dt.get(z) as i64 > hz {
            return Some((hash, jumps, false, y));
        }
        for _ in 0..c.w {
            if y & 1 == 1 { odd += 1; }
            y = t_step(y)?;
        }
        depth += c.w as i64;
        hash ^= y as u32;
        jumps += 1;
        if jumps > 100_000 {
            return None;
        }
    }
}

fn audit_case(n: u128, x: u128, f: u32, c: &Cfg, st: &mut St) {
    st.cases += 1;
    let k = growth_bits(c.w);
    let flag_at = 1u128 << (c.cutoff_bits - k);
    let h0 = 485 * f as i32 - 306 * c.m as i32;
    let eligible = n >= 1u128 << c.dip_min_bits && n < 1u128 << 96;
    // jx baseline (coarse jump-only, same W)
    let (_, jxj, jxf) = jump_until(n, x, &c.jt, flag_at, 0, true);
    st.jx_jumps += jxj;
    st.jx_flags += jxf as u64;
    // The GPU contract applies the dip test for 2^32 <= n < 2^96 (reference =
    // cv::jdip).  With --dip-min-bits below 32 the rule is also applied to
    // smaller n (an audit of the theorem itself, outside the GPU contract); its
    // reference is then the independent implementation.
    let reference = if c.dip_min_bits == 32 && c.h_bias == 0 {
        jump_dip_until_k(n, x, &c.jt, &c.dt, h0, flag_at, c.ka)
    } else if eligible {
        independent(n, x, f, c).map(|(a, b, d, _)| (a, b, d)).unwrap_or((0, 0, false))
    } else {
        jump_until(n, x, &c.jt, flag_at, 0, true)
    };
    st.jumps += reference.1;
    if reference.1 > jxj {
        st.later_than_jx += 1;
    }
    if reference.2 && !jxf {
        st.flags_not_in_jx += 1;
    }
    if jxf && !reference.2 {
        st.jx_flags_lost += 1;
    }
    // original kernel (double iterations on T(n)+1) at the same cutoff
    let (_, _, of) = cv::tail::step_until_below(n, x + 1, 1u128 << c.cutoff_bits);
    st.orig_flags += of as u64;
    if of && !reference.2 {
        st.orig_lost += 1;
    }
    if reference.2 && !of {
        st.not_in_orig += 1;
    }
    if !eligible {
        // exact-descent path: stop state must be below n
        st.small_exact += 1;
        if reference.2 {
            st.flagged += 1;
            return;
        }
        st.stops += 1;
        let mut y = x;
        for _ in 0..reference.1 * c.w as u64 {
            y = t_step(y).unwrap();
        }
        if y >= n {
            st.fail += 1;
            eprintln!("SMALL FAIL n={n}");
        }
        return;
    }
    st.dip_cases += 1;
    let Some((hash, jumps, flagged, ys)) = independent(n, x, f, c) else {
        st.unverifiable += 1;
        return;
    };
    if (hash, jumps, flagged) != reference {
        st.ref_mismatch += 1;
        if c.verbose || st.ref_mismatch <= 3 {
            eprintln!("REF MISMATCH n={n}: indep {:?} ref {:?}", (hash, jumps, flagged), reference);
        }
    }
    // H bookkeeping check against the reference's incremental H
    {
        let mut y = x;
        let mut h = h0;
        let mj = (1usize << c.w) - 1;
        let mut odd = 0i64;
        for _ in 0..jumps {
            h += 485 * c.jt.odd[(y as usize) & mj] as i32 - 306 * c.w as i32;
            for _ in 0..c.w {
                if y & 1 == 1 { odd += 1; }
                y = t_step(y).unwrap();
            }
        }
        let hs = 485 * (f as i64 + odd) - 306 * (c.m as i64 + (jumps * c.w as u64) as i64);
        if hs != h as i64 || y != ys {
            st.h_mismatch += 1;
        }
    }
    if flagged {
        st.flagged += 1;
        return; // wide recheck of n, not the dip certificate
    }
    st.stops += 1;
    let span = (c.ka * c.w + c.wd - c.span_cut) as usize;
    let mut vals = Vec::with_capacity(span + 1);
    let mut y = ys;
    vals.push(y);
    let mut overflow = false;
    for _ in 0..span {
        match t_step(y) {
            Some(z) => { y = z; vals.push(y); }
            None => { overflow = true; break; }
        }
    }
    if let Some(j) = vals.iter().position(|&v| v < n) {
        st.descent_w += 1;
        st.wpos[(j / 12).min(7)] += 1;
        return;
    }
    for (j, &yj) in vals.iter().enumerate() {
        if let Some((p, s, o)) = find_partner(yj, n, c.omax, c.smax) {
            let mut z = p;
            let mut ok = p >= 1 && p < n;
            let mut odd_seen = 0;
            for _ in 0..s {
                if z & 1 == 1 { odd_seen += 1; }
                z = t_step(z).unwrap();
            }
            ok &= z == yj && odd_seen == o;
            if ok {
                st.merge_w += 1;
                st.wpos[(j / 12).min(7)] += 1;
                return;
            }
            eprintln!("BAD PARTNER n={n} p={p} s={s} o={o}");
        }
    }
    if overflow {
        st.unverifiable += 1;
        return;
    }
    st.fail += 1;
    if c.verbose || st.fail <= 5 {
        eprintln!("AUDIT FAIL n={n} f={f} x={x} stop={ys} jumps={jumps}");
    }
}

struct ExSink<'a> {
    la: &'a Lookahead,
    c: &'a Cfg,
    st: St,
}

impl Sink for ExSink<'_> {
    fn endpoint(&mut self, nd: &Node) {
        let la = self.la;
        self.st.endpoints += 1;
        let Some(arr) = la.array(nd.f, nd.ne) else { return };
        let start = la.start(arr, nd.m);
        let r9 = mod_pow3(nd.n0, 2) as usize;
        let p3 = POW3[nd.f as usize];
        for word in 0..la.mod9_words {
            let mut t = la.completion_word(Some(arr), start, r9, word);
            while t != 0 {
                let h = (64 * word) as u128 + t.trailing_zeros() as u128;
                t &= t - 1;
                let x = p3 * h + nd.m;
                if nd.oee == 2 && x & 1 == 0 { continue; }
                if nd.oee == 1 && x & 3 == 0 { continue; }
                let n = nd.n0 + (h << la.m);
                audit_case(n, x, nd.f, self.c, &mut self.st);
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let code: u32 = arg(&args, "--code", 91);
    let n_bits: u32 = arg(&args, "--n", 72);
    let a: u32 = arg(&args, "--a", 10);
    let per_class: usize = arg(&args, "--per-class", 2000);
    let seed: u64 = arg(&args, "--seed", 20261205);
    let cutoff_bits: u32 = arg(&args, "--cutoff-bits", 120);
    let omax: u32 = arg(&args, "--omax", 13);
    let smax: u32 = arg(&args, "--smax", 26);
    let dip_min_bits: u32 = arg(&args, "--dip-min-bits", 32);
    let windows_s: String = arg(&args, "--class-b", "32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27".to_string());
    let classes_s: String = arg(&args, "--classes", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15".to_string());
    let ex_b: u32 = arg(&args, "--exhaustive", 0);
    let verbose = args.iter().any(|a| a == "--verbose");
    let Some(TailKind::JumpDip { w, wd, dq, ka, desc }) = jd_kind(code) else { panic!("not a jd code") };
    let wr = production_window_rules();
    let c = Cfg { jt: JumpTable::new(w), dt: DipTable::build(wd, wr, dq, desc), w, wd, ka, m: n_bits - a, cutoff_bits,
                  omax, smax, dip_min_bits, verbose, h_bias: arg(&args, "--h-bias", 0i64), span_cut: arg(&args, "--span-cut", 0u32) };
    let m = n_bits - a;
    let nbv = 16usize;
    let mut windows: Vec<u32> = windows_s.split(',').map(|s| s.parse().unwrap()).collect();
    while windows.len() < nbv { windows.push(*windows.last().unwrap()); }
    let classes: Vec<usize> = classes_s.split(',').map(|s| s.parse().unwrap()).collect();
    let baseline = Rules::baseline();
    let t0 = std::time::Instant::now();
    println!("CONFIG code {code} W {w} Wd {wd} q {dq} KA {ka} desc {desc} N {n_bits} A {a} M {m} cutoff {cutoff_bits} omax {omax} smax {smax} dip_min_bits {dip_min_bits} exhaustive {ex_b} seed {seed}");
    if ex_b > 0 {
        let lax = Lookahead::build_ext(m, a, ex_b, nbv, BvMode::Shared, wr);
        let mut top = TreeStats::new(m);
        let units = frontier(m.min(16), &baseline, &mut top);
        let st = units
            .par_iter()
            .fold(St::default, |mut acc, u| {
                let mut sink = ExSink { la: &lax, c: &c, st: St::default() };
                let mut ts = TreeStats::new(m);
                dfs::<_, false>(u, m, &baseline, &mut sink, &mut ts);
                acc.add(&sink.st);
                acc
            })
            .reduce(St::default, |mut x, y| { x.add(&y); x });
        println!("EXHAUSTIVE N={n_bits} A={a} B={ex_b}: {} | {:.1}s", st.line(), t0.elapsed().as_secs_f64());
        return;
    }
    let wr2 = wr.clone();
    let la = Lookahead::meta(m, a, *windows.iter().min().unwrap(), nbv, BvMode::Shared, &wr2).with_class_windows(&windows);
    let mut total = St::default();
    for &class in &classes {
        let f = la.offset + class as u32;
        let sampler = Sampler::new(&baseline, m, Some((f, f)));
        let draws = sampler.draws(per_class, seed ^ (class as u64 + 11).wrapping_mul(0x2545_F491_4F6C_DD1D), None);
        let st = draws
            .par_iter()
            .fold(St::default, |mut st, d| {
                let nd = d.node;
                st.endpoints += 1;
                let r9 = mod_pow3(nd.n0, 2) as usize;
                let p3 = POW3[nd.f as usize];
                for word in 0..la.mod9_words {
                    let mut t = la.completion_word(None, 0, r9, word);
                    while t != 0 {
                        let h = (64 * word) as u128 + t.trailing_zeros() as u128;
                        t &= t - 1;
                        let x = p3 * h + nd.m;
                        if !la.keeps(nd.f, nd.ne, (x + 1) as u64) { continue; }
                        if nd.oee == 2 && x & 1 == 0 { continue; }
                        if nd.oee == 1 && x & 3 == 0 { continue; }
                        let n = nd.n0 + (h << m);
                        audit_case(n, x, nd.f, &c, &mut st);
                    }
                }
                st
            })
            .reduce(St::default, |mut x, y| { x.add(&y); x });
        println!("class {class:2} F={f} B={}: {}", windows[class], st.line());
        total.add(&st);
    }
    println!("TOTAL code {code} N={n_bits} cutoff={cutoff_bits}: {} | {:.1}s", total.line(), t0.elapsed().as_secs_f64());
}
