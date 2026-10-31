//! CPU cost of wide rechecks for cases flagged by the conservative jump guard.
//! flagcost N A DRAWS SEED W CLASSES
use cv::jump::{for_each_case, TailConfig, TailKind};
use cv::lookahead::{BvMode, Lookahead, WindowRules};
use cv::rules::{MixTables, Rules};
use cv::sample::Sampler;
use cv::tail::{recheck_wide, step_until_below, TailParams};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (n, aa, draws, seed, w): (u32, u32, usize, u64, u32) =
        (a[1].parse().unwrap(), a[2].parse().unwrap(), a[3].parse().unwrap(), a[4].parse().unwrap(), a[5].parse().unwrap());
    let classes: Vec<usize> = a[6].split(',').map(|x| x.parse().unwrap()).collect();
    let cb = vec![32, 35, 35, 35, 35, 34, 32, 32, 27, 27, 27, 27, 27, 27, 27, 27];
    let m = n - aa;
    let wr = WindowRules { pm_inf: true, mix: Some(Arc::new(MixTables::build(13))) };
    let la = Lookahead::meta(m, aa, 27, 16, BvMode::Shared, &wr).with_class_windows(&cb);
    let tp = TailParams::default();
    let cfg = TailConfig::new(TailKind::Jump { w, skip_window: false, coarse: true });
    for &class in &classes {
        let f = la.offset + class as u32;
        let s = Sampler::new(&Rules::baseline(), m, Some((f, f)));
        let ds = s.draws(draws, seed ^ (class as u64 + 1).wrapping_mul(0x9E37_79B9), None);
        let flagged: Vec<(u128, bool)> = ds.par_iter().flat_map_iter(|d| {
            let mut v = Vec::new();
            for_each_case(&la, &tp, &d.node, |base, x| {
                let (_, _, fl) = cfg.run(base, x, 120, la.bs[class]);
                if fl {
                    let (_, _, bf) = step_until_below(base, x + 1, 1u128 << 120);
                    v.push((base, bf));
                }
            }, |_| {});
            v
        }).collect();
        let t = Instant::now();
        let mut steps = 0u64;
        for &(b, _) in &flagged {
            steps += recheck_wide(b).expect("recheck must finish");
        }
        let secs = t.elapsed().as_secs_f64();
        let base_flags = flagged.iter().filter(|x| x.1).count();
        println!("{{\"class\":{class},\"endpoints\":{draws},\"jump_flags\":{},\"of_which_baseline_flags\":{base_flags},\"recheck_seconds\":{secs:.6},\"us_per_recheck\":{:.3},\"mean_steps\":{:.1}}}",
                 flagged.len(), 1e6 * secs / flagged.len().max(1) as f64, steps as f64 / flagged.len().max(1) as f64);
    }
}
