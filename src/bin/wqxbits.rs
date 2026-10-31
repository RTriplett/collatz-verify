//! CPU statistics for the warp-queue step mode: distribution of the bit length
//! of x at jump boundaries (before each jump) for real N72 cases, and the
//! probability that all 32 lanes of a warp are below a threshold.
//! wqxbits N A CLASSES DRAWS SEED CLASS_B W
use cv::jump::{for_each_case, JumpTable, descends_coarse};
use cv::lookahead::{BvMode, Lookahead, WindowRules};
use cv::rules::{MixTables, Rules};
use cv::sample::Sampler;
use cv::tail::TailParams;
use rayon::prelude::*;
use std::sync::Arc;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: u32 = a[1].parse().unwrap();
    let aa: u32 = a[2].parse().unwrap();
    let classes: Vec<usize> = a[3].split(',').map(|x| x.parse().unwrap()).collect();
    let draws: usize = a[4].parse().unwrap();
    let seed: u64 = a[5].parse().unwrap();
    let mut cb: Vec<u32> = a[6].split(',').map(|x| x.parse().unwrap()).collect();
    while cb.len() < 16 { cb.push(*cb.last().unwrap()); }
    let w: u32 = a[7].parse().unwrap();
    let m = n - aa;
    let wr = WindowRules { pm_inf: true, mix: Some(Arc::new(MixTables::build(13))) };
    let la = Lookahead::meta(m, aa, *cb.iter().min().unwrap(), 16, BvMode::Shared, &wr).with_class_windows(&cb);
    let tp = TailParams::default();
    let tab = JumpTable::new(w);
    for &class in &classes {
        let f = la.offset + class as u32;
        let sampler = Sampler::new(&Rules::baseline(), m, Some((f, f)));
        let nodes: Vec<_> = sampler.draws(draws, seed ^ (class as u64 + 1).wrapping_mul(0x9E37_79B9), None).iter().map(|d| d.node).collect();
        // histogram of bit length of x before each jump
        let hist = nodes.par_iter().map(|nd| {
            let mut h = vec![0u64; 129];
            for_each_case(&la, &tp, nd, |base, x0| {
                let mut x = x0;
                loop {
                    h[(128 - x.leading_zeros()) as usize] += 1;
                    x = tab.jump(x);
                    if descends_coarse(x, base) { break; }
                }
            }, |_| {});
            h
        }).reduce(|| vec![0u64; 129], |mut s, h| { for i in 0..129 { s[i] += h[i]; } s });
        let tot: u64 = hist.iter().sum();
        let above = |b: usize| hist[b + 1..].iter().sum::<u64>() as f64 / tot as f64;
        let mut line = format!("{{\"class\":{class},\"boundaries\":{tot}");
        for b in [72usize, 76, 80, 84, 88] {
            let p = above(b);
            line += &format!(",\"p_ge_2^{b}\":{p:.6},\"p_warp32_all_below_2^{b}\":{:.4}", (1.0 - p).powi(32));
        }
        line += "}";
        println!("{line}");
    }
}
