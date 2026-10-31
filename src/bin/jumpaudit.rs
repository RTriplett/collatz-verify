//! Independent audit of the jump-tail reference on real sampled cases.
//!
//! For every case of uniformly drawn baseline endpoints (N, A, class windows),
//! every jump variant is replayed with plain single T-steps (arith::t_step):
//!  * each W-step jump equals W plain steps (table and limb-free arithmetic);
//!  * an unflagged case stops at a value strictly below its start n;
//!  * every intermediate value of an unflagged case is < 2^cutoff;
//!  * every case the original loop flags is flagged by the jump variant;
//!  * the coarse descent test never stops where x >= n.
//! Runs at the production cutoff and at low cutoffs that force many flags.
//!
//! jumpaudit N A DRAWS SEED CLASS_B

use cv::arith::t_step;
use cv::jump::{descends_coarse, for_each_case, growth_bits, jump_until, JumpTable};
use cv::lookahead::{BvMode, Lookahead, WindowRules};
use cv::rules::{MixTables, Rules};
use cv::sample::Sampler;
use cv::tail::{step_until_below, TailParams};
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Default, Clone)]
struct Tally {
    cases: u64,
    jumps: u64,
    flagged: u64,
    base_flagged: u64,
    coarse_extra_jumps: u64,
    steps_replayed: u64,
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: u32 = a[1].parse().unwrap();
    let aa: u32 = a[2].parse().unwrap();
    let draws: usize = a[3].parse().unwrap();
    let seed: u64 = a[4].parse().unwrap();
    let mut cb: Vec<u32> = a[5].split(',').map(|x| x.parse().unwrap()).collect();
    while cb.len() < 16 {
        cb.push(*cb.last().unwrap());
    }
    let m = n - aa;
    let wr = WindowRules { pm_inf: true, mix: Some(Arc::new(MixTables::build(13))) };
    let la = Lookahead::meta(m, aa, *cb.iter().min().unwrap(), 16, BvMode::Shared, &wr).with_class_windows(&cb);
    let tp = TailParams::default();
    let ws = [10u32, 11, 12, 13];
    let tabs: Vec<JumpTable> = ws.iter().map(|&w| JumpTable::new(w)).collect();
    let cutoffs = [120u32, 108, 104];
    let baseline = Rules::baseline();
    let mut grand = vec![vec![Tally::default(); 4]; ws.len() * cutoffs.len()];
    for class in 0..16usize {
        let f = la.offset + class as u32;
        let b = la.bs[class];
        let s = Sampler::new(&baseline, m, Some((f, f)));
        let ds = s.draws(draws, seed ^ (class as u64 + 1).wrapping_mul(0x9E37_79B9), None);
        let res: Vec<Vec<Vec<Tally>>> = ds
            .par_iter()
            .map(|d| {
                let mut t = vec![vec![Tally::default(); 4]; ws.len() * cutoffs.len()];
                for_each_case(&la, &tp, &d.node, |base, x0| {
                    for (ci, &cutoff) in cutoffs.iter().enumerate() {
                        let (_, _, bflag) = step_until_below(base, x0 + 1, 1u128 << cutoff);
                        for (wi, tab) in tabs.iter().enumerate() {
                            let w = tab.w;
                            let k = growth_bits(w);
                            if cutoff < 96 + k {
                                continue;
                            }
                            let flag_at = 1u128 << (cutoff - k);
                            // variants: exact/no skip, exact/skip, coarse/no skip, coarse/skip
                            let mut exact_jumps = 0;
                            for v in 0..4 {
                                let skip = if v & 1 == 1 { b / w } else { 0 };
                                let coarse = v >= 2;
                                let (_, jumps, flagged) = jump_until(base, x0, tab, flag_at, skip, coarse);
                                // Replay with plain steps.
                                let mut y = x0;
                                for j in 0..jumps {
                                    let before = y;
                                    for _ in 0..w {
                                        y = t_step(y);
                                        if !flagged {
                                            assert!(y < 1u128 << cutoff, "intermediate reached cutoff");
                                        }
                                    }
                                    assert_eq!(tab.jump(before), y, "jump != W steps");
                                    // No earlier stop was possible under this variant's rule.
                                    if j + 1 < jumps && j + 1 > skip as u64 {
                                        let stop = if coarse { descends_coarse(y, base) } else { y < base };
                                        assert!(!stop, "missed an earlier stop");
                                    }
                                }
                                let tt = &mut t[ci * ws.len() + wi][v];
                                tt.cases += 1;
                                tt.jumps += jumps;
                                tt.steps_replayed += jumps * w as u64;
                                if flagged {
                                    tt.flagged += 1;
                                    assert!(y >= flag_at, "flag only at the guard");
                                } else {
                                    assert!(y < base, "unflagged case must end below its start");
                                }
                                if bflag {
                                    tt.base_flagged += 1;
                                    assert!(flagged, "baseline flag not covered (cutoff {cutoff}, W {w}, v {v})");
                                }
                                if v == 0 {
                                    exact_jumps = jumps;
                                }
                                if v == 2 && !flagged {
                                    tt.coarse_extra_jumps += jumps - exact_jumps;
                                }
                            }
                        }
                    }
                }, |_| panic!("wide start"));
                t
            })
            .collect();
        for r in res {
            for (g, x) in grand.iter_mut().zip(r) {
                for (gg, xx) in g.iter_mut().zip(x) {
                    gg.cases += xx.cases;
                    gg.jumps += xx.jumps;
                    gg.flagged += xx.flagged;
                    gg.base_flagged += xx.base_flagged;
                    gg.coarse_extra_jumps += xx.coarse_extra_jumps;
                    gg.steps_replayed += xx.steps_replayed;
                }
            }
        }
        eprintln!("class {class} audited");
    }
    println!("{{\"n\":{n},\"a\":{aa},\"draws_per_class\":{draws},\"seed\":{seed},\"class_b\":{cb:?},\"results\":[");
    let mut rows = Vec::new();
    for (ci, &cutoff) in cutoffs.iter().enumerate() {
        for (wi, &w) in ws.iter().enumerate() {
            for v in 0..4 {
                let t = &grand[ci * ws.len() + wi][v];
                if t.cases == 0 {
                    continue;
                }
                rows.push(format!(
                    "{{\"cutoff\":{cutoff},\"w\":{w},\"skip\":{},\"coarse\":{},\"cases\":{},\"jumps\":{},\"flagged\":{},\"baseline_flagged\":{},\"coarse_extra_jumps\":{},\"plain_steps_replayed\":{}}}",
                    v & 1 == 1, v >= 2, t.cases, t.jumps, t.flagged, t.base_flagged, t.coarse_extra_jumps, t.steps_replayed));
            }
        }
    }
    println!("{}\n]}}", rows.join(",\n"));
}
