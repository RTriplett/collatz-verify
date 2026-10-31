//! c-tail simulator (CPU only): full jumps per case for
//!   * jump-only tails (coarse descent test, as the jx kernels), and
//!   * jump tails with a window-dip certificate of width Wd tested k windows
//!     ahead at every jump boundary (k = 0: the window starting at the
//!     boundary; k = 1, 2: the window starting 1 or 2 jumps later, whose index
//!     comes from low-bit mini jumps),
//! for jump widths Wj in {10, 11, 12}; plus the potential of reusing the
//! class bitvectors as one deep lookahead at boundary J.
//!
//! jdsim N A PER_CLASS SEED CLASS_B [CLASSES]
use cv::jdip::{dip_eligible, DipTable};
use cv::jump::{for_each_case, growth_bits, jump_until, JumpTable};
use cv::lookahead::{window_dip_ext, BvMode, Lookahead, WindowRules, START_UNKNOWN};
use cv::rules::{MixTables, Rules};
use cv::sample::Sampler;
use cv::tail::TailParams;
use rayon::prelude::*;
use std::sync::Arc;

const WJS: [u32; 3] = [10, 11, 12];
const WDS: [(u32, u32); 9] = [(11, 64), (11, 1), (12, 1), (12, 16), (12, 32), (12, 64), (12, 128), (16, 1), (20, 1)];
const KS: usize = 3;
const NB: usize = 48;

#[derive(Clone)]
struct Acc {
    cases: u64,
    eligible: u64,
    base_j: [u64; 3],
    // [wj][wd][k]: full jumps
    dip_j: Vec<u64>,
    // first fire boundary histogram for (12, 12, 0)
    hist: [u64; 16],
    bvre_fire: [u64; 6],
    bvre_j: [u64; 6],
}

impl Acc {
    fn new() -> Self {
        Acc { cases: 0, eligible: 0, base_j: [0; 3], dip_j: vec![0; 3 * WDS.len() * KS], hist: [0; 16], bvre_fire: [0; 6], bvre_j: [0; 6] }
    }
    fn add(&mut self, o: &Acc) {
        self.cases += o.cases;
        self.eligible += o.eligible;
        for i in 0..3 { self.base_j[i] += o.base_j[i]; }
        for i in 0..self.dip_j.len() { self.dip_j[i] += o.dip_j[i]; }
        for i in 0..16 { self.hist[i] += o.hist[i]; }
        for i in 0..6 { self.bvre_fire[i] += o.bvre_fire[i]; self.bvre_j[i] += o.bvre_j[i]; }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: u32 = a[1].parse().unwrap();
    let aa: u32 = a[2].parse().unwrap();
    let per_class: usize = a[3].parse().unwrap();
    let seed: u64 = a[4].parse().unwrap();
    let mut cb: Vec<u32> = a[5].split(',').map(|x| x.parse().unwrap()).collect();
    while cb.len() < 16 { cb.push(*cb.last().unwrap()); }
    let classes: Vec<usize> = a.get(6).map_or((0..16).collect(), |s| s.split(',').map(|x| x.parse().unwrap()).collect());
    let m = n - aa;
    let wr = WindowRules { pm_inf: true, mix: Some(Arc::new(MixTables::build(13))) };
    let la = Lookahead::meta(m, aa, *cb.iter().min().unwrap(), 16, BvMode::Shared, &wr).with_class_windows(&cb);
    let tp = TailParams::default();
    let jts: Vec<JumpTable> = WJS.iter().map(|&w| JumpTable::new(w)).collect();
    let t0 = std::time::Instant::now();
    let dts: Vec<DipTable> = WDS.iter().map(|&(wd, q)| DipTable::new(wd, &wr, q)).collect();
    eprintln!("dip tables built in {:.1}s", t0.elapsed().as_secs_f64());
    let cutoff_bits = tp.cutoff_bits;
    let baseline = Rules::baseline();
    println!("{{\"type\":\"config\",\"n\":{n},\"a\":{aa},\"per_class\":{per_class},\"seed\":{seed},\"class_b\":{cb:?},\"wj\":{:?},\"wd\":{:?},\"k\":{KS}}}",
             WJS, WDS.iter().map(|p| vec![p.0, p.1]).collect::<Vec<_>>());
    for &class in &classes {
        let f = la.offset + class as u32;
        let h0 = la.cutoffs[class];
        let sampler = Sampler::new(&baseline, m, Some((f, f)));
        let draws = sampler.draws(per_class, seed ^ (class as u64 + 1).wrapping_mul(0x9E37_79B9), None);
        let acc = draws.par_iter().fold(Acc::new, |mut acc, d| {
            let nd = d.node;
            for_each_case(&la, &tp, &nd, |base, x| {
                acc.cases += 1;
                for (i, jt) in jts.iter().enumerate() {
                    let k = growth_bits(jt.w);
                    let (_, j, _) = jump_until(base, x, jt, 1u128 << (cutoff_bits - k), 0, true);
                    acc.base_j[i] += j;
                }
                if !dip_eligible(base) { return; }
                acc.eligible += 1;
                for (wi, jt) in jts.iter().enumerate() {
                    let mask = (1usize << jt.w) - 1;
                    let mut xs = [0u128; NB];
                    let mut hs = [0i32; NB];
                    let (mut y, mut h) = (x, h0);
                    for j in 0..NB {
                        xs[j] = y;
                        hs[j] = h;
                        h += 485 * jt.odd[(y as usize) & mask] as i32 - 306 * jt.w as i32;
                        y = jt.jump(y);
                    }
                    for (di, dt) in dts.iter().enumerate() {
                        let fire: Vec<bool> = (0..NB).map(|j| dt.get(xs[j]) > hs[j]).collect();
                        for k in 0..KS {
                            let first = (k..NB).find(|&j| fire[j]).unwrap_or(NB - 1);
                            acc.dip_j[(wi * WDS.len() + di) * KS + k] += (first - k) as u64;
                        }
                        if jt.w == 12 && WDS[di] == (12, 1) {
                            let first = (0..NB).find(|&j| fire[j]).unwrap_or(NB - 1);
                            acc.hist[first.min(15)] += 1;
                            // Class bitvector reuse at boundary J (conservative class: cutoff >= H).
                            for jb in 0..6usize {
                                let hj = hs[jb];
                                let mut fired = false;
                                if jb < first {
                                    let i = if hj <= la.cutoffs[0] { 0 } else { ((hj - la.cutoffs[0]) + 484) / 485 };
                                    if (i as usize) < 16 {
                                        let i = i as usize;
                                        let b = la.bs[i];
                                        let p = ((xs[jb] + 1) as u64) & ((1u64 << b) - 1);
                                        fired = window_dip_ext(p, b, START_UNKNOWN, &wr) > la.cutoffs[i];
                                    }
                                }
                                acc.bvre_fire[jb] += fired as u64;
                                acc.bvre_j[jb] += if fired { jb as u64 } else { first as u64 };
                            }
                        }
                    }
                }
            }, |_| panic!("wide start"));
            acc
        }).reduce(Acc::new, |mut a, b| { a.add(&b); a });
        let c = acc.cases as f64;
        let e = acc.eligible.max(1) as f64;
        let mut dip = Vec::new();
        for (wi, &wj) in WJS.iter().enumerate() {
            for (di, &(wd, q)) in WDS.iter().enumerate() {
                let v: Vec<f64> = (0..KS).map(|k| ((acc.dip_j[(wi * WDS.len() + di) * KS + k] as f64 / e) * 1e4).round() / 1e4).collect();
                dip.push(format!("{{\"wj\":{wj},\"wd\":{wd},\"q\":{q},\"jumps_k\":{v:?}}}"));
            }
        }
        println!("{{\"type\":\"class\",\"class\":{class},\"f\":{f},\"b\":{},\"h0\":{h0},\"endpoints\":{},\"cases\":{},\"eligible\":{},\"base_jumps\":[{:.4},{:.4},{:.4}],\"dip\":[{}],\"hist_12_12\":{:?},\"bvre_fire_frac\":{:?},\"bvre_jumps\":{:?}}}",
            la.bs[class], draws.len(), acc.cases, acc.eligible,
            acc.base_j[0] as f64 / c, acc.base_j[1] as f64 / c, acc.base_j[2] as f64 / c, dip.join(","), acc.hist,
            acc.bvre_fire.iter().map(|&j| (j as f64 / e * 1e4).round() / 1e4).collect::<Vec<_>>(),
            acc.bvre_j.iter().map(|&j| (j as f64 / e * 1e4).round() / 1e4).collect::<Vec<_>>());
        eprintln!("class {class} done ({:.1}s)", t0.elapsed().as_secs_f64());
    }
}
