//! c-screen audits of the reachable-row table (src/rowsieve.rs).
//!
//!   rowsieve_audit compare M A          bit equality with sieve3's SieveK (its own code,
//!                                       sieve3_ref.rs) on every (reachable row, n mod 3^9)
//!                                       class, plus random completions; build times
//!   rowsieve_audit exhaustive M A EVERY  every excluded class of every reachable row: an
//!                                       independent inverse search (verify-sieve3's vaudit
//!                                       search, no tables) must certify its smallest member
//!                                       >= 2^M, then forward iteration re-checks T^s(p) =
//!                                       T^j(n), 1 <= p < n.  Emits every EVERY-th
//!                                       certificate as JSON for the Python checker.
//!   rowsieve_audit members M A COUNT SEED  random excluded classes: smallest, random and
//!                                       largest member below 2^(M+A) certified the same way
//!
//! Output: JSON lines (type=summary / cert).
mod arith {
    pub use cv::arith::*;
}
#[allow(dead_code)]
mod sieve3_ref;

use cv::arith::POW3;
use cv::rowsieve::RowSieve;
use cv::rules::Rules;
use cv::sample::Rng;
use rayon::prelude::*;

#[inline]
fn t(x: u128) -> u128 {
    if x & 1 == 1 { x + (x >> 1) + 1 } else { x >> 1 }
}

/// verify-sieve3's search (written independently of both table builders):
/// inverse words from y with at most `budget` inverse-odd steps and s >= 1,
/// reaching some 1 <= p < n.  Exhaustive up to `cap` states; Ok(None) = none.
fn search(y: u128, n: u128, budget: u32, cap: u64) -> Result<Option<(u128, u32, u32)>, ()> {
    let mut st: Vec<(u128, u32, u32)> = vec![(y, 0, 0)];
    let mut vis = 0u64;
    while let Some((v, s, o)) = st.pop() {
        vis += 1;
        if vis > cap {
            return Err(());
        }
        if s >= 1 && v >= 1 && v < n {
            return Ok(Some((v, s, o)));
        }
        let r = budget - o;
        assert!(r < 80);
        // any descendant q has q+1 >= (v+1)(2/3)^r; prune when that is >= n+1 (exact 256-bit compare)
        if !cv::arith::U256::from(v + 1).shl(r).lt(&cv::arith::U256::mul(POW3[r as usize], n + 1)) {
            continue;
        }
        st.push((v.checked_mul(2).expect("value overflow"), s + 1, o));
        if r > 0 && v % 3 == 2 {
            st.push(((2 * v - 1) / 3, s + 1, o + 1));
        }
    }
    Ok(None)
}

/// Certificate within the table's family: depth j <= kmax, odd budget min(f_j + jb, omax).
fn cert(n: u128, kmax: u32, jb: u32, omax: u32, cap: u64) -> (&'static str, Option<(u32, u128, u32, u32)>) {
    let (mut y, mut f) = (n, 0u32);
    let mut capped = false;
    for j in 0..=kmax {
        match search(y, n, (f + jb).min(omax), cap) {
            Ok(Some((p, s, o))) => return ("ok", Some((j, p, s, o))),
            Ok(None) => {}
            Err(()) => capped = true,
        }
        if y & 1 == 1 {
            f += 1;
        }
        y = t(y);
    }
    (if capped { "capped" } else { "none" }, None)
}

fn forward_ok(n: u128, j: u32, p: u128, s: u32) -> bool {
    let mut y = n;
    for _ in 0..j {
        y = t(y);
    }
    let mut q = p;
    for _ in 0..s {
        q = t(q);
    }
    q == y && p >= 1 && p < n && s >= 1
}

/// Smallest member >= lo of the class (r mod 2^k, tt mod d).
fn class_min(r: u128, tt: u128, k: u32, d: u128, lo: u128) -> u128 {
    let p2k = 1u128 << k;
    let l = p2k * d;
    let x0 = (0..d).map(|u| r + p2k * u).find(|x| x % d == tt).unwrap();
    x0 + l * lo.saturating_sub(x0).div_ceil(l)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m: u32 = args[2].parse().unwrap();
    let a: u32 = args[3].parse().unwrap();
    let cap = 50_000_000u64;
    match args[1].as_str() {
        "compare" => {
            let t0 = std::time::Instant::now();
            let mine = RowSieve::build(12, 9, m, a, 18, &Rules::baseline());
            let t_mine = t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            let theirs = sieve3_ref::SieveK::new(9, 12, m, a, 18);
            let t_theirs = t1.elapsed().as_secs_f64();
            let d = POW3[9];
            let lo = 1u128 << m;
            let hi = 1u128 << (m + a);
            // Every class of every reachable row, at its smallest member >= 2^M (h >= 1).
            let res: Vec<(u64, u64, u64)> = mine.rows.par_iter().map(|&r| {
                let (mut same, mut diff, mut excl) = (0u64, 0u64, 0u64);
                for tt in 0..d {
                    let n = class_min(r as u128, tt, 12, d, lo);
                    if n >= hi { continue; }
                    let (a1, a2) = (mine.keeps(n), theirs.keeps(n));
                    if a1 == a2 { same += 1 } else { diff += 1 }
                    excl += (!a1) as u64;
                }
                (same, diff, excl)
            }).collect();
            let (same, diff, excl) = res.iter().fold((0, 0, 0), |x, y| (x.0 + y.0, x.1 + y.1, x.2 + y.2));
            // Random completions (all rows): unreachable rows must be kept by mine.
            let mut rng = Rng(0xC5C5_2026);
            let (mut rs, mut rd, mut unreach_kept, mut unreach) = (0u64, 0u64, 0u64, 0u64);
            for _ in 0..2_000_000 {
                let n = lo + rng.below(hi - lo);
                let row = (n & 4095) as u32;
                if mine.rows.binary_search(&row).is_ok() {
                    if mine.keeps(n) == theirs.keeps(n) { rs += 1 } else { rd += 1 }
                } else {
                    unreach += 1;
                    unreach_kept += mine.keeps(n) as u64;
                }
            }
            // Phase: CPU exact vs device model.
            let mut pm = 0u64;
            for _ in 0..1_000_000 {
                let n0 = rng.below(lo);
                pm += (mine.phase(n0) != mine.phase_device_model(n0)) as u64;
            }
            println!("{}", serde_json::json!({"type":"summary","mode":"compare","m":m,"a":a,
                "reachable_rows":mine.rows.len(),"row_words":mine.row_words,"device_bytes":mine.device_bytes(),
                "excluded_pairs":mine.excluded,"build_seconds_mine":t_mine,"build_seconds_sieve3":t_theirs,
                "classes_compared":same+diff,"classes_equal":same,"classes_differ":diff,"classes_excluded":excl,
                "random_reachable_equal":rs,"random_reachable_differ":rd,
                "random_unreachable":unreach,"random_unreachable_kept":unreach_kept,
                "phase_model_mismatches":pm,
                "threads":rayon::current_num_threads()}));
            assert_eq!(diff, 0, "tables differ");
            assert_eq!(rd, 0, "tables differ on random completions");
            assert_eq!(unreach, unreach_kept, "unreachable rows must keep everything");
            assert_eq!(pm, 0);
        }
        "exhaustive" => {
            let every: u64 = args[4].parse().unwrap();
            let t0 = std::time::Instant::now();
            let sk = RowSieve::build(12, 9, m, a, 18, &Rules::baseline());
            let build = t0.elapsed().as_secs_f64();
            let d = POW3[9];
            let lo = 1u128 << m;
            let hi = 1u128 << (m + a);
            let res: Vec<(u64, u64, u64, u64, u64, Vec<String>)> = sk.rows.par_iter().map(|&r| {
                let (mut excl, mut ok, mut capped, mut none, mut bad) = (0u64, 0u64, 0u64, 0u64, 0u64);
                let mut lines = Vec::new();
                for tt in 0..d {
                    let n = class_min(r as u128, tt, 12, d, lo);
                    if n >= hi || sk.keeps(n) { continue; }
                    excl += 1;
                    let (st, c) = cert(n, 12, 9, 18, cap);
                    match c {
                        Some((j, p, s, o)) => {
                            if forward_ok(n, j, p, s) { ok += 1 } else { bad += 1 }
                            if excl % every == 0 {
                                lines.push(format!("{{\"type\":\"cert\",\"src\":\"exh\",\"status\":\"ok\",\"row\":{r},\"t\":{tt},\"n\":\"{n}\",\"j\":{j},\"p\":\"{p}\",\"s\":{s},\"o\":{o}}}"));
                            }
                        }
                        None => {
                            if st == "capped" { capped += 1 } else { none += 1 }
                            lines.push(format!("{{\"type\":\"cert\",\"src\":\"exh\",\"status\":\"{st}\",\"row\":{r},\"t\":{tt},\"n\":\"{n}\",\"j\":0,\"p\":\"0\",\"s\":0,\"o\":0}}"));
                        }
                    }
                }
                (excl, ok, capped, none, bad, lines)
            }).collect();
            let (mut e, mut o, mut c, mut nn, mut b) = (0, 0, 0, 0, 0);
            for (excl, ok, capped, none, bad, lines) in res {
                e += excl; o += ok; c += capped; nn += none; b += bad;
                for l in lines { println!("{l}"); }
            }
            println!("{}", serde_json::json!({"type":"summary","mode":"exhaustive","m":m,"a":a,
                "rows":sk.rows.len(),"excluded_classes":e,"certified":o,"capped":c,"none":nn,
                "forward_failures":b,"table_build_seconds":build,"seconds":t0.elapsed().as_secs_f64()}));
            assert!(c == 0 && nn == 0 && b == 0 && o == e, "audit failures");
        }
        "members" => {
            let count: usize = args[4].parse().unwrap();
            let seed: u64 = args[5].parse().unwrap();
            let sk = RowSieve::build(12, 9, m, a, 18, &Rules::baseline());
            let d = POW3[9];
            let lo = 1u128 << m;
            let hi = 1u128 << (m + a);
            let l = 4096u128 * d;
            let mut rng = Rng(seed);
            let mut picks = Vec::new();
            while picks.len() < count {
                let r = sk.rows[rng.below(sk.rows.len() as u128) as usize] as u128;
                let tt = rng.below(d);
                let nmin = class_min(r, tt, 12, d, lo);
                if nmin >= hi || sk.keeps(nmin) { continue; }
                let nmax = nmin + l * ((hi - 1 - nmin) / l);
                let nr = nmin + l * rng.below((nmax - nmin) / l + 1);
                picks.push([nmin, nr, nmax]);
            }
            let res: Vec<(u64, u64, u64, Vec<String>)> = picks.par_iter().map(|mem| {
                let (mut ok, mut fail, mut layout) = (0u64, 0u64, 0u64);
                let mut lines = Vec::new();
                for (mi, &n) in mem.iter().enumerate() {
                    layout += sk.keeps(n) as u64; // every member must be excluded by the table
                    let (st, c) = cert(n, 12, 9, 18, cap);
                    match c {
                        Some((j, p, s, o)) if forward_ok(n, j, p, s) => {
                            ok += 1;
                            lines.push(format!("{{\"type\":\"cert\",\"src\":\"member{mi}\",\"status\":\"ok\",\"n\":\"{n}\",\"j\":{j},\"p\":\"{p}\",\"s\":{s},\"o\":{o}}}"));
                        }
                        _ => {
                            fail += 1;
                            lines.push(format!("{{\"type\":\"cert\",\"src\":\"member{mi}\",\"status\":\"{st}\",\"n\":\"{n}\",\"j\":0,\"p\":\"0\",\"s\":0,\"o\":0}}"));
                        }
                    }
                }
                (ok, fail, layout, lines)
            }).collect();
            let (mut o, mut f, mut lay) = (0, 0, 0);
            for (ok, fail, layout, lines) in res {
                o += ok; f += fail; lay += layout;
                for l in lines { println!("{l}"); }
            }
            println!("{}", serde_json::json!({"type":"summary","mode":"members","m":m,"a":a,"classes":count,
                "members_certified":o,"members_failed":f,"members_kept_by_table":lay}));
            assert!(f == 0 && lay == 0);
        }
        _ => panic!("usage: rowsieve_audit compare|exhaustive|members M A ..."),
    }
}
