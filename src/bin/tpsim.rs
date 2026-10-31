//! c3-2ph: CPU model of SIMT schedules for the verification kernel (no GPU).
//!
//! tpsim N A PER_CLASS SEED CLASS_B CODES ROWS_FROM [CLASSES]
//!
//! For parent-order endpoints of each class (screenbench's sampler), every
//! case is run through the exact CPU reference of its jd code (jumps per
//! case), and three warp schedules are scored in "steps" (one warp-wide
//! instruction group):
//!   PT   per-thread kernel as compiled (cv_kernel_jds SASS): lane i owns one
//!        endpoint; each case is a reconvergent round = max over active lanes
//!        of (scan words to its next bit) + setup + max(jumps); the warp
//!        waits for its longest endpoint (BSYNC per endpoint).
//!   TPR  two-phase, round-based: the cases of a warp's endpoints are pooled;
//!        rounds of 32 cases, each round costs setup + max(jumps).
//!   TPF  two-phase, flattened (per-lane pop when idle, every step): lane-steps
//!        = sum(jumps) + pops, warp steps = ceil per warp of the busiest lane.
//! Output: one JSON line per class with the step counts and the jumps-per-case
//! histogram.  This is a model (no memory latency), for design only.
use cv::jump::{for_each_case, xv_kind, TailConfig};
use cv::lookahead::{BvMode, Lookahead, WindowRules};
use cv::rowsieve::RowSieve;
use cv::rules::{MixTables, Rules};
use cv::sample::Sampler;
use cv::tail::{Screen, TailParams};
use cv::tree::{dfs, Node, Sink, TreeStats};
use rayon::prelude::*;
use serde_json::json;
use std::sync::Arc;

struct Collect(Vec<Node>);
impl Sink for Collect {
    fn endpoint(&mut self, node: &Node) {
        self.0.push(*node);
    }
}

fn parent_endpoints(m: u32, la: &Lookahead, classes: &[usize], per_class: usize, seed: u64) -> Vec<Vec<Node>> {
    let entry = m - 10;
    let cutoff = m - 12;
    let base = Rules::baseline();
    let improved = Rules::improved(13);
    let sampler = Sampler::new(&base, entry, None);
    let mut out: Vec<Vec<Node>> = vec![Vec::new(); classes.len()];
    let mut round = 0u64;
    while out.iter().any(|v| v.len() < per_class) {
        let draws = sampler.draws(65536, seed ^ round.wrapping_mul(0x9E37_79B9_7F4A_7C15), Some(&improved));
        round += 1;
        let roots: Vec<Node> = draws.into_iter().filter(|d| d.culled.map_or(true, |(k, _)| k > cutoff)).map(|d| d.node).collect();
        let expanded: Vec<Vec<Node>> = roots.par_iter().map(|r| {
            let mut sink = Collect(Vec::new());
            let mut st = TreeStats::new(m);
            dfs::<_, false>(r, m, &base, &mut sink, &mut st);
            sink.0
        }).collect();
        for eps in expanded {
            for nd in eps {
                if let Some(c) = la.bigindex(nd.f) {
                    if let Some(i) = classes.iter().position(|&x| x == c) {
                        if out[i].len() < per_class {
                            out[i].push(nd);
                        }
                    }
                }
            }
        }
        assert!(round < 4096, "sampling did not fill every class");
    }
    out
}

/// One endpoint's cases: (completion word index (32-bit words), jumps).
fn endpoint_cases(la: &Lookahead, tp: &TailParams, nd: &Node, cfg: &TailConfig, b: u32) -> Vec<(u32, u32)> {
    let mut v = Vec::new();
    let h0 = 485 * nd.f as i32 - 306 * la.m as i32;
    for_each_case(la, tp, nd, |base, x| {
        let h = ((base >> la.m) as u32) & ((1u32 << la.a) - 1);
        let (_hash, iters, _fl) = cfg.run_h(base, x, tp.cutoff_bits, b, h0);
        v.push((h / 32, iters as u32));
    }, |_| panic!("wide"));
    v
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: u32 = a[1].parse().unwrap();
    let aa: u32 = a[2].parse().unwrap();
    let per_class: usize = a[3].parse().unwrap();
    let seed: u64 = a[4].parse().unwrap();
    let mut cb: Vec<u32> = a[5].split(',').map(|x| x.parse().unwrap()).collect();
    while cb.len() < 16 { cb.push(*cb.last().unwrap()); }
    let mut codes: Vec<u32> = a[6].split(',').map(|x| x.parse().unwrap()).collect();
    while codes.len() < 16 { codes.push(*codes.last().unwrap()); }
    let rows_from: u32 = a[7].parse().unwrap();
    let classes: Vec<usize> = a.get(8).map_or((0..8).collect(), |s| s.split(',').map(|x| x.parse().unwrap()).collect());
    let m = n - aa;
    let wr = WindowRules { pm_inf: true, mix: Some(Arc::new(MixTables::build(13))) };
    let base_b = *cb.iter().min().unwrap();
    let la = Lookahead::meta(m, aa, base_b, 16, BvMode::Shared, &wr).with_class_windows(&cb)
        .with_rows(Some(RowSieve::standard(m, aa)));
    let sets = parent_endpoints(m, &la, &classes, per_class, seed);
    let words32 = (1u32 << aa) / 32;
    for (ci, &class) in classes.iter().enumerate() {
        let mut tp = TailParams::default();
        tp.screen = Screen { rows: true, rows_from, ..Screen::default() };
        let cfg = TailConfig::new(xv_kind(codes[class]).unwrap());
        let b = la.bs[class];
        let eps: Vec<Vec<(u32, u32)>> = sets[ci].par_iter().map(|nd| endpoint_cases(&la, &tp, nd, &cfg, b)).collect();
        let mut hist = vec![0u64; 65];
        let (mut cases, mut jumps) = (0u64, 0u64);
        for e in &eps {
            for &(_, j) in e {
                hist[(j as usize).min(64)] += 1;
                cases += 1;
                jumps += j as u64;
            }
        }
        // PT model, per warp of 32 consecutive endpoints.
        let (mut pt_rounds, mut pt_scan, mut pt_jump, mut pt_ep_rounds) = (0u64, 0u64, 0u64, 0u64);
        let mut pt_case_lane_rounds = 0u64; // sum over rounds of active lanes (for setup utilization)
        // TPR: pooled cases per warp batch of 32 endpoints, rounds of 32.
        let (mut tpr_rounds, mut tpr_jump) = (0u64, 0u64);
        // TPF: flattened; warp steps = max over lanes of lane steps when cases are dealt greedily
        // (list scheduling: each lane takes the next case when idle) -> simulate.
        let mut tpf_steps = 0u64;
        for w in eps.chunks(32) {
            let maxc = w.iter().map(|e| e.len()).max().unwrap_or(0);
            // per-lane round costs; a lane after its last case scans to the end once.
            let nr = maxc + 1;
            pt_ep_rounds += nr as u64;
            for r in 0..nr {
                let (mut ms, mut mj, mut act) = (0u32, 0u32, 0u32);
                for e in w {
                    if r < e.len() {
                        let prev = if r == 0 { 0 } else { e[r - 1].0 };
                        let skip = e[r].0 - prev; // words advanced to reach this case's word
                        ms = ms.max(skip);
                        mj = mj.max(e[r].1);
                        act += 1;
                    } else if r == e.len() {
                        let prev = if r == 0 { 0 } else { e[r - 1].0 };
                        ms = ms.max(words32 - prev);
                    }
                }
                pt_scan += ms as u64;
                pt_jump += mj as u64;
                if act > 0 { pt_rounds += 1; }
                pt_case_lane_rounds += act as u64;
            }
            let pool: Vec<u32> = w.iter().flat_map(|e| e.iter().map(|c| c.1)).collect();
            for r in pool.chunks(32) {
                tpr_rounds += 1;
                tpr_jump += *r.iter().max().unwrap() as u64;
            }
            // greedy list scheduling of the pool on 32 lanes; a case costs jumps + 1 (pop/setup slot).
            let mut lanes = [0u64; 32];
            for &j in &pool {
                let (k, _) = lanes.iter().enumerate().min_by_key(|(_, &t)| t).unwrap();
                lanes[k] += j as u64 + 1;
            }
            tpf_steps += *lanes.iter().max().unwrap();
        }
        let nw = eps.len().div_ceil(32) as u64;
        println!("{}", json!({"class":class,"code":codes[class],"b":b,"endpoints":eps.len(),"cases":cases,"jumps":jumps,
            "cases_per_ep":cases as f64 / eps.len() as f64,"jumps_per_case":jumps as f64 / cases as f64,
            "pt":{"rounds":pt_rounds,"scan_word_steps":pt_scan,"jump_steps":pt_jump,"round_slots":pt_ep_rounds,
                  "setup_util":pt_case_lane_rounds as f64 / (32.0 * pt_rounds as f64),
                  "jump_util":jumps as f64 / (32.0 * pt_jump as f64)},
            "tpr":{"rounds":tpr_rounds,"jump_steps":tpr_jump,"jump_util":jumps as f64 / (32.0 * tpr_jump as f64)},
            "tpf":{"steps":tpf_steps,"util":(jumps + cases) as f64 / (32.0 * tpf_steps as f64)},
            "warps":nw,"hist":hist}));
    }
}
