//! Task-1 statistics: per-case trajectory structure on uniformly sampled
//! baseline endpoints at the target N (CPU only).
//!
//! jumpstats N A DRAWS SEED B0,B1,...  (class windows; shorter lists pad with B0)

use cv::jump::{for_each_case, growth_bits, jumps_until_below, profile_case, JumpTable};
use cv::lookahead::{BvMode, Lookahead, WindowRules};
use cv::rules::{MixTables, Rules};
use cv::sample::Sampler;
use cv::tail::TailParams;
use rayon::prelude::*;
use std::sync::Arc;

const WS: [u32; 7] = [8, 10, 11, 12, 13, 14, 16];

#[derive(Clone, Default)]
struct Acc {
    endpoints: u64,
    cases: u64,
    wide: u64,
    over: u64,
    base_iters: u64,
    base_steps: u64,
    first_desc: u64,
    in_window: u64,
    after_window: u64,
    desc_in_window: u64,
    hist: Vec<u64>,      // first_desc histogram, bins of 4 steps up to 400
    growth: Vec<u64>,    // max_bits - start_bits histogram (0..=64)
    maxbits: Vec<u64>,   // max_bits histogram (0..=128)
    startbits: Vec<u64>, // start_bits histogram
    jumps: Vec<u64>,     // per W: total jumps (no skip)
    jumps_skip: Vec<u64>,// per W: total jumps with skip floor(B/W)
    above3: Vec<u64>,    // per W: cases whose max >= 2^(96-K)
}

impl Acc {
    fn new() -> Self {
        Acc {
            hist: vec![0; 101],
            growth: vec![0; 65],
            maxbits: vec![0; 129],
            startbits: vec![0; 129],
            jumps: vec![0; WS.len()],
            jumps_skip: vec![0; WS.len()],
            above3: vec![0; WS.len()],
            ..Default::default()
        }
    }
    fn merge(mut self, o: Acc) -> Acc {
        self.endpoints += o.endpoints;
        self.cases += o.cases;
        self.wide += o.wide;
        self.over += o.over;
        self.base_iters += o.base_iters;
        self.base_steps += o.base_steps;
        self.first_desc += o.first_desc;
        self.in_window += o.in_window;
        self.after_window += o.after_window;
        self.desc_in_window += o.desc_in_window;
        for (a, b) in self.hist.iter_mut().zip(&o.hist) { *a += b; }
        for (a, b) in self.growth.iter_mut().zip(&o.growth) { *a += b; }
        for (a, b) in self.maxbits.iter_mut().zip(&o.maxbits) { *a += b; }
        for (a, b) in self.startbits.iter_mut().zip(&o.startbits) { *a += b; }
        for (a, b) in self.jumps.iter_mut().zip(&o.jumps) { *a += b; }
        for (a, b) in self.jumps_skip.iter_mut().zip(&o.jumps_skip) { *a += b; }
        for (a, b) in self.above3.iter_mut().zip(&o.above3) { *a += b; }
        self
    }
}

fn quantile(hist: &[u64], q: f64, bin: f64) -> f64 {
    let total: u64 = hist.iter().sum();
    let target = (q * total as f64).ceil() as u64;
    let mut c = 0;
    for (i, &h) in hist.iter().enumerate() {
        c += h;
        if c >= target.max(1) {
            return i as f64 * bin;
        }
    }
    (hist.len() - 1) as f64 * bin
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: u32 = args.get(1).map_or(72, |s| s.parse().unwrap());
    let a: u32 = args.get(2).map_or(10, |s| s.parse().unwrap());
    let draws: usize = args.get(3).map_or(2000, |s| s.parse().unwrap());
    let seed: u64 = args.get(4).map_or(1, |s| s.parse().unwrap());
    let windows: Vec<u32> = args.get(5).map_or(vec![28], |s| s.split(',').map(|x| x.parse().unwrap()).collect());
    let m = n - a;
    let nbv = 16;
    let mut cb = windows.clone();
    while cb.len() < nbv {
        cb.push(windows[0]);
    }
    let base_b = *cb.iter().min().unwrap();
    let wr = WindowRules { pm_inf: true, mix: Some(Arc::new(MixTables::build(13))) };
    let la = Lookahead::meta(m, a, base_b, nbv, BvMode::Shared, &wr).with_class_windows(&cb);
    let tp = TailParams::default();
    let baseline = Rules::baseline();
    let tables: Vec<JumpTable> = WS.iter().map(|&w| JumpTable::new(w)).collect();
    println!("{{\"n\":{n},\"a\":{a},\"draws_per_class\":{draws},\"seed\":{seed},\"class_b\":{:?},\"classes\":[", cb);
    for class in 0..nbv {
        let f = la.offset + class as u32;
        let b = la.bs[class];
        let sampler = Sampler::new(&baseline, m, Some((f, f)));
        let ds = sampler.draws(draws, seed ^ (class as u64 + 1).wrapping_mul(0x9E37_79B9), None);
        let acc = ds
            .par_iter()
            .fold(Acc::new, |mut acc, d| {
                acc.endpoints += 1;
                let mut wide = 0;
                for_each_case(&la, &tp, &d.node, |base, x| {
                    acc.cases += 1;
                    let p = profile_case(base, x);
                    if p.over {
                        acc.over += 1;
                        return;
                    }
                    acc.base_iters += p.base_iters as u64;
                    acc.base_steps += p.base_steps as u64;
                    acc.first_desc += p.first_desc as u64;
                    acc.in_window += p.first_desc.min(b) as u64;
                    acc.after_window += p.first_desc.saturating_sub(b) as u64;
                    acc.desc_in_window += (p.first_desc <= b) as u64;
                    acc.hist[((p.first_desc / 4) as usize).min(100)] += 1;
                    acc.growth[(p.max_bits.saturating_sub(p.start_bits) as usize).min(64)] += 1;
                    acc.maxbits[p.max_bits as usize] += 1;
                    acc.startbits[p.start_bits as usize] += 1;
                    for (i, t) in tables.iter().enumerate() {
                        acc.jumps[i] += jumps_until_below(base, x, t, 0) as u64;
                        acc.jumps_skip[i] += jumps_until_below(base, x, t, b / t.w) as u64;
                        if p.max_bits > 96 - growth_bits(t.w) {
                            acc.above3[i] += 1;
                        }
                    }
                }, |_| wide += 1);
                acc.wide += wide;
                acc
            })
            .reduce(Acc::new, Acc::merge);
        let c = acc.cases.max(1) as f64;
        let ok = (acc.cases - acc.over).max(1) as f64;
        let jumps: Vec<String> = WS
            .iter()
            .enumerate()
            .map(|(i, w)| format!("\"{w}\":[{:.4},{:.4},{:.3e}]", acc.jumps[i] as f64 / ok, acc.jumps_skip[i] as f64 / ok, acc.above3[i] as f64 / c))
            .collect();
        println!(
            "{{\"class\":{class},\"f\":{f},\"b\":{b},\"endpoints\":{},\"cases_per_ep\":{:.4},\"over\":{},\"wide\":{},\
\"base_iters\":{:.4},\"base_steps\":{:.4},\"first_desc\":{:.4},\"in_window\":{:.4},\"after_window\":{:.4},\
\"frac_desc_in_window\":{:.5},\"first_desc_q\":[{},{},{},{},{}],\"growth_q\":[{},{},{},{},{}],\"max_growth\":{},\
\"maxbits_q\":[{},{},{},{}],\"max_maxbits\":{},\"startbits_q\":[{},{},{}],\"jumps\":{{{}}}}}{}",
            acc.endpoints,
            acc.cases as f64 / acc.endpoints as f64,
            acc.over,
            acc.wide,
            acc.base_iters as f64 / ok,
            acc.base_steps as f64 / ok,
            acc.first_desc as f64 / ok,
            acc.in_window as f64 / ok,
            acc.after_window as f64 / ok,
            acc.desc_in_window as f64 / ok,
            quantile(&acc.hist, 0.1, 4.0), quantile(&acc.hist, 0.5, 4.0), quantile(&acc.hist, 0.9, 4.0),
            quantile(&acc.hist, 0.99, 4.0), quantile(&acc.hist, 0.9999, 4.0),
            quantile(&acc.growth, 0.5, 1.0), quantile(&acc.growth, 0.9, 1.0), quantile(&acc.growth, 0.99, 1.0),
            quantile(&acc.growth, 0.9999, 1.0), quantile(&acc.growth, 0.999999, 1.0),
            acc.growth.iter().rposition(|&x| x > 0).unwrap_or(0),
            quantile(&acc.maxbits, 0.5, 1.0), quantile(&acc.maxbits, 0.99, 1.0), quantile(&acc.maxbits, 0.9999, 1.0),
            quantile(&acc.maxbits, 0.999999, 1.0),
            acc.maxbits.iter().rposition(|&x| x > 0).unwrap_or(0),
            quantile(&acc.startbits, 0.01, 1.0), quantile(&acc.startbits, 0.5, 1.0), quantile(&acc.startbits, 0.99, 1.0),
            jumps.join(","),
            if class + 1 < nbv { "," } else { "" }
        );
    }
    println!("]}}");
}
