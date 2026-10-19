use cv::{
    dp::ForwardDp,
    rules::{walkback_certificate, Rules},
    sample::{Rng, Sampler},
};
use serde_json::json;
use std::{hint::black_box, time::Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let count: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(10000);
    let depths: Vec<u32> = if args.len() > 2 {
        args[2..].iter().map(|s| s.parse().unwrap()).collect()
    } else {
        vec![30, 34, 38, 42, 46, 50, 54, 58]
    };
    let base = Rules::baseline();
    let imp = Rules::improved(13);
    let fw = ForwardDp::from_root(&base, 62);
    let cap = 65536u32;
    for k in depths {
        let s = Sampler::new(&base, k, None);
        let seed = 0x64a7_42b1_9c8e_3000 + k as u64;
        let mut rng = Rng(seed);
        let start = Instant::now();
        let candidates: Vec<_> = (0..count)
            .map(|_| s.draw(&mut rng, Some(&imp)))
            .filter(|d| d.culled.is_none())
            .map(|d| d.node)
            .collect();
        let sampling_seconds = start.elapsed().as_secs_f64();
        let mut times = Vec::new();
        let mut culls = 0;
        let mut certificate_examples = Vec::new();
        for rep in 0..5 {
            let st = Instant::now();
            let mut got = 0;
            for nd in &candidates {
                if let Some(c) = black_box(walkback_certificate(
                    black_box(nd.n0),
                    nd.k,
                    nd.f,
                    nd.m,
                    cap,
                )) {
                    got += 1;
                    if rep == 0 && certificate_examples.len() < 3 {
                        certificate_examples.push(json!({"n0":nd.n0.to_string(), "k":nd.k,"f":nd.f,"m":nd.m.to_string(),"p":c.p.to_string(),"s":c.s,"o":c.o}));
                    }
                }
            }
            times.push(st.elapsed().as_secs_f64());
            if rep == 0 {
                culls = got;
            } else {
                assert_eq!(culls, got);
            }
            if times[0] > 0.5 {
                break;
            }
        }
        let mut sorted = times.clone();
        sorted.sort_by(f64::total_cmp);
        let seconds = sorted[sorted.len() / 2];
        let improved_population = s.population() as f64 * candidates.len() as f64 / count as f64;
        let hit_population = s.population() as f64 * culls as f64 / count as f64;
        let us = 1e6 * seconds / candidates.len() as f64;
        println!(
            "{}",
            json!({
                "depth":k,"draws":count,"seed":seed,"max_states":cap,
                "baseline_population":s.population().to_string(),"improved_candidates":candidates.len(),
                "improved_population_estimate":improved_population,"walkback_culls":culls,
                "additional_cull_fraction_of_improved":culls as f64/candidates.len() as f64,
                "additional_cull_population_estimate":hit_population,
                "sampling_seconds":sampling_seconds,"search_seconds_repetitions":times,
                "search_seconds_median":seconds,"search_us_per_improved_candidate":us,
                "search_total_single_core_hours_estimate":improved_population*us/3.6e9,
                "minimal_improved_bitmap_bytes_estimate":(improved_population/8.0).ceil(),
                "baseline_ordinal_bitmap_bytes":s.population().div_ceil(8).to_string(),
                "sparse_cull_n0_8byte_bytes_estimate":8.0*hit_population,
                "sparse_cull_certificate_32byte_bytes_estimate":32.0*hit_population,
                "baseline_tree_generated_through_k":fw.generated_through(k).to_string(),
                "baseline_tree_generated_through_60":fw.generated_through(60).to_string(),
                "baseline_tree_generated_through_62":fw.generated_through(62).to_string(),
                "baseline_tree_fraction_at_or_above_k_vs_60":fw.generated_through(k) as f64/fw.generated_through(60) as f64,
                "baseline_tree_fraction_at_or_above_k_vs_62":fw.generated_through(k) as f64/fw.generated_through(62) as f64,
                "certificate_examples":certificate_examples
            })
        );
    }
}
