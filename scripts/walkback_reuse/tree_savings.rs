use cv::{
    dp::ForwardDp,
    rules::{walkback_certificate, Rules},
    sample::{Rng, Sampler},
    tree::Node,
};
use serde_json::json;

fn main() {
    let count: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(10000);
    let checkpoints = [30u32, 34, 38, 42, 46, 50];
    let base = Rules::baseline();
    let imp = Rules::improved(13);
    let fw = ForwardDp::from_root(&base, 62);
    let mut total_saved = [0.0f64; 6];
    let mut total_variance = [0.0f64; 6];
    let mut total_improved = fw.generated_through(30) as f64;
    let mut total_delta = 0.0;
    let mut total_delta_variance = 0.0;
    for parent_depth in 30..=61u32 {
        let sampler = Sampler::new(&base, parent_depth, None);
        let seed = 0x79ba_1038_44a0_0000 + parent_depth as u64;
        let mut rng = Rng(seed);
        let mut saved = [0u64; 6];
        let mut saved_sq = [0u64; 6];
        let mut improved = 0u64;
        let mut delta_sum = 0i64;
        let mut delta_sq = 0u64;
        let mut hit_draws = [0u64; 6];
        for _ in 0..count {
            let draw = sampler.draw(&mut rng, Some(&imp));
            if draw.culled.is_some() {
                continue;
            }
            let nchildren = draw
                .node
                .children(&base)
                .iter()
                .filter(|c| c.is_some())
                .count() as u64;
            improved += nchildren;
            let mut nd = Node::root(&base);
            let mut hit = [false; 6];
            let mut next_checkpoint = 0;
            while nd.k < parent_depth.min(50) {
                let bit = ((draw.node.n0 >> nd.k) & 1) as usize;
                nd = nd.children(&base)[bit].expect("baseline surviving ancestry");
                if nd.k == checkpoints[next_checkpoint] {
                    if walkback_certificate(nd.n0, nd.k, nd.f, nd.m, 65536).is_some() {
                        hit[next_checkpoint] = true;
                        hit_draws[next_checkpoint] += 1;
                        saved[next_checkpoint] += nchildren;
                        saved_sq[next_checkpoint] += nchildren * nchildren;
                    }
                    next_checkpoint += 1;
                    if next_checkpoint == checkpoints.len() {
                        break;
                    }
                }
            }
            let delta = (hit[3] as i64 - hit[2] as i64) * nchildren as i64;
            delta_sum += delta;
            delta_sq += (delta * delta) as u64;
        }
        let population = sampler.population() as f64;
        let scale = population / count as f64;
        let mean_improved = improved as f64 / count as f64;
        let mut savings = Vec::new();
        for i in 0..6 {
            let estimate = scale * saved[i] as f64;
            let mean = saved[i] as f64 / count as f64;
            let variance =
                population * population * (saved_sq[i] as f64 / count as f64 - mean * mean)
                    / (count - 1) as f64;
            total_saved[i] += estimate;
            total_variance[i] += variance;
            savings.push(json!({"checkpoint":checkpoints[i],"hit_draws":hit_draws[i],"saved_child_count_sample":saved[i],"estimated_saved_generated":estimate,"standard_error":variance.sqrt()}));
        }
        let delta_mean = delta_sum as f64 / count as f64;
        let delta_variance =
            population * population * (delta_sq as f64 / count as f64 - delta_mean * delta_mean)
                / (count - 1) as f64;
        total_delta += population * delta_mean;
        total_delta_variance += delta_variance;
        total_improved += population * mean_improved;
        println!(
            "{}",
            json!({"type":"level","parent_depth":parent_depth,"generated_depth":parent_depth+1,"draws":count,"seed":seed,"baseline_parent_population":sampler.population().to_string(),"estimated_improved_generated":population*mean_improved,"checkpoints":savings})
        );
    }
    let seconds_per_generated_node = 32.93e-9 / 2.0;
    let summary:Vec<_>=(0..6).map(|i|json!({"checkpoint":checkpoints[i],"saved_generated_nodes":total_saved[i],"saved_generated_nodes_standard_error":total_variance[i].sqrt(),"saved_fraction_of_improved_generated":total_saved[i]/total_improved,"core_hours_saved":total_saved[i]*seconds_per_generated_node/3600.0,"core_hours_saved_standard_error":total_variance[i].sqrt()*seconds_per_generated_node/3600.0})).collect();
    println!(
        "{}",
        json!({"type":"summary","draws_per_parent_depth":count,"ns_per_thread_node":32.93,"threads_per_core":2,"generated_depths_sampled":[31,62],"improved_generated_total_estimate":total_improved,"baseline_generated_total":fw.generated_through(62).to_string(),"checkpoints":summary,"k42_minus_k38_generated_saved":total_delta,"k42_minus_k38_core_hours_saved":total_delta*seconds_per_generated_node/3600.0,"k42_minus_k38_core_hours_saved_standard_error":total_delta_variance.sqrt()*seconds_per_generated_node/3600.0,"caveat":"Total improved node estimate treats depths<=30 as baseline; those depths are negligible. Core conversion follows repository 2-threads-per-core model, not independently measured scaling."})
    );
}
