use cv::lookahead::{BvMode, Lookahead, WindowRules};
use cv::rules::{walkback_certificate, Rules};
use cv::sample::Sampler;
use cv::tail::{endpoint_work, TailParams};
use cv::tree::Node;
use rayon::prelude::*;
use serde_json::json;
use std::time::Instant;

const KS: [u32; 9] = [30, 34, 38, 42, 46, 50, 54, 58, 62];
fn main() {
    let draws: usize = std::env::args()
        .nth(1)
        .unwrap_or("30000".into())
        .parse()
        .unwrap();
    let seed: u64 = std::env::args()
        .nth(2)
        .unwrap_or("17391".into())
        .parse()
        .unwrap();
    let model_path = std::env::args()
        .nth(3)
        .unwrap_or("results/walkback_reuse/model.json".into());
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .unwrap();
    let model: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(model_path).unwrap()).unwrap();
    let windows: Vec<u32> = model["windows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let pc = model["fit"]["per_case"].as_f64().unwrap();
    let pi = model["fit"]["per_iter"].as_f64().unwrap();
    let overheads: Vec<f64> = windows
        .iter()
        .map(|b| {
            let mut choices: Vec<(u32, f64)> = model["fit"]["overhead"]
                .as_object()
                .unwrap()
                .iter()
                .filter_map(|(key, v)| {
                    key.strip_prefix("10,")
                        .map(|s| (s.parse::<u32>().unwrap(), v.as_f64().unwrap()))
                })
                .filter(|(bb, _)| bb <= b)
                .collect();
            choices.sort_by_key(|x| x.0);
            choices.last().unwrap().1
        })
        .collect();
    let baseline_seconds = model["baseline_seconds"].as_f64().unwrap();
    let base = Rules::baseline();
    let imp = Rules::improved(13);
    let wr = WindowRules {
        pm_inf: true,
        mix: imp.mix.clone(),
    };
    let la = Lookahead::meta(62, 10, 27, 16, BvMode::Shared, &wr).with_class_windows(&windows);
    let tp = TailParams::default();
    let mut total_saved = [0.0; 9];
    let mut total_var = [0.0; 9];
    let mut total_eps = [0.0; 9];
    let mut total_hits = [0u64; 9];
    let t0 = Instant::now();
    println!(
        "{}",
        json!({"kind":"config","n":72,"a":10,"windows":windows,"depths":KS,"cap":65536,"draws_per_class":draws,"seed":seed,"baseline_model_seconds":baseline_seconds,"gpu_timing":false,"cpu_threads":8})
    );
    for cls in 0..16 {
        let sampler = Sampler::new(
            &base,
            62,
            Some((la.offset + cls as u32, la.offset + cls as u32)),
        );
        let population = sampler.population() as f64;
        let ds = sampler.draws(
            draws,
            seed ^ (cls as u64 + 1).wrapping_mul(0x9E3779B9),
            Some(&imp),
        );
        // All checkpoint alternatives are evaluated on the same target draws.
        // For full improved survivors every ancestor survives the cheap rules.
        let obs: Vec<(u16, f64)> = ds
            .par_iter()
            .map(|d| {
                if d.culled.is_some() {
                    return (0, 0.0);
                }
                let mut nd = Node::root(&base);
                let mut mask = 0u16;
                while nd.k < 62 {
                    let bit = ((d.node.n0 >> nd.k) & 1) as usize;
                    nd = nd.children(&base)[bit].unwrap();
                    if let Some(j) = KS.iter().position(|&k| k == nd.k) {
                        if walkback_certificate(nd.n0, nd.k, nd.f, nd.m, 65536).is_some() {
                            mask |= 1 << j;
                        }
                    }
                }
                assert_eq!(nd.n0, d.node.n0);
                assert_eq!(nd.m, d.node.m);
                if mask == 0 {
                    return (0, 0.0);
                }
                // Exact direct-window and tail operation counts only for removed
                // endpoints; zero contributions need no expensive tail evaluation.
                let work = endpoint_work(&la, &tp, &nd, |_| {});
                (
                    mask,
                    overheads[cls] + pc * work.cases as f64 + pi * work.iters as f64,
                )
            })
            .collect();
        let mut sums = [0.0; 9];
        let mut sumsq = [0.0; 9];
        let mut hits = [0u64; 9];
        for (mask, cost) in obs {
            for j in 0..9 {
                if mask & (1 << j) != 0 {
                    sums[j] += cost;
                    sumsq[j] += cost * cost;
                    hits[j] += 1;
                }
            }
        }
        for j in 0..9 {
            total_saved[j] += population * sums[j] / draws as f64;
            let variance =
                ((sumsq[j] - sums[j] * sums[j] / draws as f64) / (draws - 1) as f64).max(0.0);
            total_var[j] += population * population * variance / draws as f64;
            total_eps[j] += population * hits[j] as f64 / draws as f64;
            total_hits[j] += hits[j];
        }
        println!(
            "{}",
            json!({"kind":"class","class":cls,"population":population,"cheap_survived_draws":ds.iter().filter(|d|d.culled.is_none()).count(),"hits":hits,"saved_model_seconds":sums.map(|s|s*population/draws as f64),"elapsed_seconds":t0.elapsed().as_secs_f64()})
        );
    }
    for j in 0..9 {
        println!(
            "{}",
            json!({"kind":"result","k":KS[j],"sample_hits":total_hits[j],"estimated_removed_endpoints":total_eps[j],"saved_model_seconds":total_saved[j],"saved_model_seconds_se":total_var[j].sqrt(),"saved_gpu_hours":total_saved[j]/3600.0,"saved_fraction":total_saved[j]/baseline_seconds,"saved_fraction_se":total_var[j].sqrt()/baseline_seconds})
        );
    }
}
