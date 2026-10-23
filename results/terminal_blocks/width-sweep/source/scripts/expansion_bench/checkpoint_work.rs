//! Paired work estimates for discontinuing CPU-only extra checks after K.
//! The shared lookahead still includes pm_inf and mix13 window credits.
//! This does not time a GPU or enumerate the complete M=62 frontier.
use cv::lookahead::{BvMode, Lookahead, WindowRules};
use cv::rules::{pm_inf_culls, walkback_certificate, Rules};
use cv::sample::Sampler;
use cv::tail::{endpoint_work, TailParams};
use cv::tree::Node;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::time::Instant;

const STOPS: [u32; 6] = [38, 46, 50, 54, 58, 62];
const SCHEDULES: usize = STOPS.len() + 1;
const METRICS: [&str; 5] = ["endpoints", "cases", "iters", "flagged", "model_seconds"];
const CHECKPOINT: u32 = 38;
const WALKBACK_CAP: u32 = 65_536;
const TARGET: u32 = 62;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Checkpoint {
    ExtraReject,
    WalkbackReject,
    Kept,
}

#[derive(Clone, Copy, Debug)]
struct Classification {
    checkpoint: Checkpoint,
    first_extra: Option<u32>,
    first_pm_after: Option<u32>,
}

impl Classification {
    fn full_keep(self) -> bool {
        self.checkpoint == Checkpoint::Kept && self.first_extra.is_none()
    }

    fn keep_stop(self, stop: u32) -> bool {
        self.checkpoint == Checkpoint::Kept && self.first_extra.is_none_or(|k| k > stop)
    }

    fn keep_pm(self) -> bool {
        self.checkpoint == Checkpoint::Kept && self.first_pm_after.is_none()
    }

    fn additional_mask(self) -> u16 {
        if self.full_keep() {
            return 0;
        }
        let mut mask = 0;
        for (j, &stop) in STOPS.iter().enumerate() {
            if self.keep_stop(stop) {
                mask |= 1 << j;
            }
        }
        if self.keep_pm() {
            mask |= 1 << STOPS.len();
        }
        mask
    }
}

/// Reconstruct a sampled baseline path and keep the first rejection of each
/// independent rule schedule. A rejection stops that schedule, never the
/// baseline reconstruction needed to classify the other schedules.
fn classify(
    target: &Node,
    baseline: &Rules,
    improved: &Rules,
    checkpoint: u32,
    cap: u32,
) -> Classification {
    assert!(checkpoint > 2 && checkpoint <= target.k);
    let mut nd = Node::root(baseline);
    let mut out = Classification {
        checkpoint: Checkpoint::Kept,
        first_extra: None,
        first_pm_after: None,
    };
    while nd.k < target.k {
        let bit = ((target.n0 >> nd.k) & 1) as usize;
        nd = nd.children(baseline)[bit].expect("sampled baseline path exists");
        debug_assert!(nd.cull(baseline).is_none());
        let pm = improved.pm_inf && pm_inf_culls(nd.n0, nd.k, nd.f, nd.m);
        if pm && nd.k > checkpoint && out.first_pm_after.is_none() {
            out.first_pm_after = Some(nd.k);
        }
        if out.first_extra.is_none()
            && (pm
                || improved.mix.as_ref().is_some_and(|mix| {
                    nd.k >= improved.mix_from && mix.culls_r(nd.n0, nd.k, nd.f, nd.m, nd.m3)
                }))
        {
            out.first_extra = Some(nd.k);
        }
        if nd.k == checkpoint {
            if out.first_extra.is_some() {
                out.checkpoint = Checkpoint::ExtraReject;
                return out;
            }
            if walkback_certificate(nd.n0, nd.k, nd.f, nd.m, cap).is_some() {
                out.checkpoint = Checkpoint::WalkbackReject;
                return out;
            }
        }
    }
    assert_eq!(nd, *target);
    out
}

#[derive(Clone, Copy, Default)]
struct Moments {
    sum: [f64; 5],
    sumsq: [f64; 5],
}

impl Moments {
    fn add(&mut self, values: [f64; 5]) {
        for (j, value) in values.into_iter().enumerate() {
            self.sum[j] += value;
            self.sumsq[j] += value * value;
        }
    }

    fn estimates(self, population: f64, draws: usize, work_measured: bool) -> Value {
        let mut out = serde_json::Map::new();
        for (j, metric) in METRICS.iter().enumerate() {
            if j > 0 && !work_measured {
                out.insert((*metric).into(), Value::Null);
                continue;
            }
            out.insert(
                (*metric).into(),
                json!({
                    "estimate": population * self.sum[j] / draws as f64,
                    "se": (population * population * sample_variance(self, draws, j)
                        / draws as f64).sqrt()
                }),
            );
        }
        Value::Object(out)
    }
}

fn sample_variance(m: Moments, draws: usize, metric: usize) -> f64 {
    ((m.sumsq[metric] - m.sum[metric] * m.sum[metric] / draws as f64) / (draws - 1) as f64).max(0.0)
}

#[derive(Clone, Copy, Default)]
struct Total {
    estimate: [f64; 5],
    variance: [f64; 5],
}

impl Total {
    fn add_class(&mut self, moments: Moments, population: f64, draws: usize) {
        for j in 0..5 {
            self.estimate[j] += population * moments.sum[j] / draws as f64;
            self.variance[j] +=
                population * population * sample_variance(moments, draws, j) / draws as f64;
        }
    }

    fn json(self, work_measured: bool) -> Value {
        let mut out = serde_json::Map::new();
        for (j, metric) in METRICS.iter().enumerate() {
            out.insert(
                (*metric).into(),
                if j == 0 || work_measured {
                    json!({"estimate":self.estimate[j], "se":self.variance[j].sqrt()})
                } else {
                    Value::Null
                },
            );
        }
        Value::Object(out)
    }
}

fn schedule(j: usize) -> String {
    if j < STOPS.len() {
        format!("stop_extra_after_{}", STOPS[j])
    } else {
        "pm_only_after_38".into()
    }
}

fn overhead_for(model: &Value, window: u32) -> f64 {
    model["fit"]["overhead"]
        .as_object()
        .unwrap()
        .iter()
        .filter_map(|(key, value)| {
            key.strip_prefix("10,")
                .map(|s| (s.parse::<u32>().unwrap(), value.as_f64().unwrap()))
        })
        .filter(|&(b, _)| b <= window)
        .max_by_key(|&(b, _)| b)
        .expect("fitted A=10 endpoint overhead exists")
        .1
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let draws: usize = args.get(1).map_or("30000", String::as_str).parse().unwrap();
    let seed: u64 = args.get(2).map_or("17391", String::as_str).parse().unwrap();
    let model_path = args
        .get(3)
        .map_or("results/walkback_reuse/model.json", String::as_str);
    let threads: usize = args.get(4).map_or("8", String::as_str).parse().unwrap();
    let mode = args.get(5).map_or("delta", String::as_str);
    assert!(draws >= 2 && threads > 0);
    assert!(matches!(mode, "delta" | "all"));
    let measure_full = mode == "all";
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let model_text = std::fs::read_to_string(model_path).unwrap();
    let model: Value = serde_json::from_str(&model_text).unwrap();
    let windows: Vec<u32> = model["windows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(windows.len(), 16);
    let pc = model["fit"]["per_case"].as_f64().unwrap();
    let pi = model["fit"]["per_iter"].as_f64().unwrap();
    let reference_seconds = model["baseline_seconds"].as_f64().unwrap();
    let base = Rules::baseline();
    let improved = Rules::improved(13);
    let window_rules = WindowRules {
        pm_inf: true,
        mix: improved.mix.clone(),
    };
    let la = Lookahead::meta(TARGET, 10, 27, 16, BvMode::Shared, &window_rules)
        .with_class_windows(&windows);
    let tail = TailParams::default();
    let omitted_population =
        Sampler::new(&base, TARGET, Some((la.offset + 16, TARGET))).population();
    let mut added = [Total::default(); SCHEDULES];
    let mut full = Total::default();
    let mut checkpoint_total = Total::default();
    let mut covariance = [[0.0; 5]; SCHEDULES];
    let mut hits = [0u64; SCHEDULES];
    let started = Instant::now();
    println!(
        "{}",
        json!({"kind":"config", "n":72,"a":10,"m":TARGET,
            "checkpoint":CHECKPOINT,"walkback_cap":WALKBACK_CAP,"mix_j":13,
            "cpu_schedules":(0..SCHEDULES).map(schedule).collect::<Vec<_>>(),
            "window_rules":"shared, pm_inf plus mix13 retained for every CPU schedule",
            "windows":windows,"draws_per_class":draws,"seed":seed,"cpu_threads":threads,
            "sampling":"independent draws with replacement within exact baseline F strata; paired schedules",
            "f_min":la.offset,"f_max":la.offset+15,
            "omitted_baseline_population_f_ge_56":omitted_population.to_string(),
            "tail_evaluation":mode,"gpu_timing":false,"gpu_model":"existing fitted RTX 4090, including per-endpoint overhead",
            "model_path":model_path,"model":model,
            "reference_model_seconds_without_checkpoint_walkback":reference_seconds,
            "uncertainty":"stratified sampling standard errors only; GPU fit uncertainty is not included"})
    );
    for cls in 0..16 {
        let f = la.offset + cls as u32;
        let sampler = Sampler::new(&base, TARGET, Some((f, f)));
        let population_exact = sampler.population();
        let population = population_exact as f64;
        let sampled = sampler.draws(
            draws,
            seed ^ (cls as u64 + 1).wrapping_mul(0x9E37_79B9),
            None,
        );
        let overhead = overhead_for(&model, windows[cls]);
        let observations: Vec<_> = sampled
            .par_iter()
            .map(|draw| {
                let c = classify(&draw.node, &base, &improved, CHECKPOINT, WALKBACK_CAP);
                let mask = c.additional_mask();
                let values = if mask != 0 || (measure_full && c.full_keep()) {
                    let work = endpoint_work(&la, &tail, &draw.node, |_| {});
                    [
                        1.0,
                        work.cases as f64,
                        work.iters as f64,
                        work.flagged as f64,
                        overhead + pc * work.cases as f64 + pi * work.iters as f64,
                    ]
                } else {
                    [1.0, 0.0, 0.0, 0.0, 0.0]
                };
                (c, mask, values)
            })
            .collect();
        let mut extra_moments = [Moments::default(); SCHEDULES];
        let mut full_moments = Moments::default();
        let mut checkpoint_moments = Moments::default();
        let mut checkpoint_counts = [0u64; 3];
        let mut first_extra_hist = [0u64; 63];
        for (classification, mask, values) in observations {
            let ck = match classification.checkpoint {
                Checkpoint::ExtraReject => 0,
                Checkpoint::WalkbackReject => 1,
                Checkpoint::Kept => 2,
            };
            checkpoint_counts[ck] += 1;
            if classification.checkpoint == Checkpoint::Kept {
                checkpoint_moments.add([1.0, 0.0, 0.0, 0.0, 0.0]);
                if let Some(k) = classification.first_extra {
                    first_extra_hist[k as usize] += 1;
                }
            }
            if classification.full_keep() {
                full_moments.add(values);
            }
            for j in 0..SCHEDULES {
                if mask & (1 << j) != 0 {
                    hits[j] += 1;
                    extra_moments[j].add(values);
                }
            }
        }
        full.add_class(full_moments, population, draws);
        checkpoint_total.add_class(checkpoint_moments, population, draws);
        for j in 0..SCHEDULES {
            added[j].add_class(extra_moments[j], population, draws);
            for metric in 0..5 {
                // Added and full-check surviving endpoints are disjoint, so
                // the cross-product sum is exactly zero on the paired draws.
                covariance[j][metric] -= population
                    * population
                    * extra_moments[j].sum[metric]
                    * full_moments.sum[metric]
                    / (draws * draws * (draws - 1)) as f64;
            }
        }
        println!(
            "{}",
            json!({"kind":"class","class":cls,"f":f,"baseline_population_exact":population_exact.to_string(),
                "draws":draws,"checkpoint_extra_rejects":checkpoint_counts[0],
                "checkpoint_walkback_rejects":checkpoint_counts[1],"checkpoint_survivors":checkpoint_counts[2],
                "first_extra_failure_after_checkpoint":first_extra_hist.to_vec(),
                "full":full_moments.estimates(population,draws,measure_full),
                "additional":(0..SCHEDULES).map(|j|json!({"schedule":schedule(j),
                    "sample_hits":extra_moments[j].sum[0] as u64,
                    "work":extra_moments[j].estimates(population,draws,true)})).collect::<Vec<_>>(),
                "elapsed_cpu_probe_seconds":started.elapsed().as_secs_f64()})
        );
    }
    println!(
        "{}",
        json!({"kind":"reference","full_check":full.json(measure_full),
        "checkpoint_only_endpoints":checkpoint_total.json(false)})
    );
    for j in 0..SCHEDULES {
        let mut ratios = serde_json::Map::new();
        for (metric, name) in METRICS.iter().enumerate() {
            let y = full.estimate[metric];
            if (metric > 0 && !measure_full) || y == 0.0 {
                ratios.insert((*name).into(), Value::Null);
            } else {
                let x = added[j].estimate[metric];
                let ratio = x / y;
                let variance = (added[j].variance[metric] + ratio * ratio * full.variance[metric]
                    - 2.0 * ratio * covariance[j][metric])
                    / (y * y);
                ratios.insert(
                    (*name).into(),
                    json!({"estimate":ratio,"se":variance.max(0.0).sqrt()}),
                );
            }
        }
        println!(
            "{}",
            json!({"kind":"result","schedule":schedule(j),"sample_added_hits":hits[j],
            "additional":added[j].json(true),"additional_fraction_of_paired_full":ratios,
            "additional_fraction_of_original_model_seconds":added[j].estimate[4]/reference_seconds,
            "additional_fraction_of_original_model_seconds_se":added[j].variance[4].sqrt()/reference_seconds,
            "additional_gpu_model_hours":added[j].estimate[4]/3600.0})
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv::sample::Rng;
    use cv::tree::{dfs, Sink, TreeStats};

    #[derive(Default)]
    struct Collect(Vec<Node>);
    impl Sink for Collect {
        fn endpoint(&mut self, node: &Node) {
            self.0.push(*node);
        }
    }

    /// Independent traversal applies its actual chosen schedule at every
    /// generated prefix. Comparing complete sets catches prefix omissions.
    fn scheduled_dfs(
        node: &Node,
        target: u32,
        checkpoint: u32,
        cap: u32,
        stop: u32,
        pm_only: bool,
        rules: &Rules,
        out: &mut Vec<Node>,
    ) {
        if node.k == target {
            out.push(*node);
            return;
        }
        for child in node.children(rules).into_iter().flatten() {
            let mut chosen = Rules::baseline();
            if child.k <= stop {
                chosen = rules.clone();
            } else if pm_only {
                chosen.pm_inf = true;
            }
            if child.cull(&chosen).is_some() {
                continue;
            }
            if child.k == checkpoint
                && walkback_certificate(child.n0, child.k, child.f, child.m, cap).is_some()
            {
                continue;
            }
            scheduled_dfs(&child, target, checkpoint, cap, stop, pm_only, rules, out);
        }
    }

    #[test]
    fn classification_matches_independent_schedule_enumeration() {
        let base = Rules::baseline();
        let improved = Rules::improved(5);
        let checkpoint = 14;
        let target = 22;
        let cap = 128;
        let mut collected = Collect::default();
        dfs::<_, false>(
            &Node::root(&base),
            target,
            &base,
            &mut collected,
            &mut TreeStats::default(),
        );
        let classified: Vec<_> = collected
            .0
            .iter()
            .map(|node| (*node, classify(node, &base, &improved, checkpoint, cap)))
            .collect();
        for (stop, pm_only) in [
            (14, false),
            (16, false),
            (18, false),
            (20, false),
            (22, false),
            (14, true),
        ] {
            let actual: Vec<_> = classified
                .iter()
                .filter_map(|(node, classification)| {
                    let keep = if pm_only {
                        classification.keep_pm()
                    } else {
                        classification.keep_stop(stop)
                    };
                    keep.then_some(*node)
                })
                .collect();
            let mut expected = Vec::new();
            scheduled_dfs(
                &Node::root(&base),
                target,
                checkpoint,
                cap,
                stop,
                pm_only,
                &improved,
                &mut expected,
            );
            assert_eq!(actual, expected, "stop={stop}, pm_only={pm_only}");
        }
    }

    #[test]
    fn first_failure_matches_sampler_and_masks_are_nested_at_target() {
        let base = Rules::baseline();
        let improved = Rules::improved(13);
        let sampler = Sampler::new(&base, TARGET, None);
        let mut rng = Rng(97_413);
        for _ in 0..2048 {
            let draw = sampler.draw(&mut rng, Some(&improved));
            let c = classify(&draw.node, &base, &improved, CHECKPOINT, WALKBACK_CAP);
            if c.checkpoint == Checkpoint::Kept {
                assert_eq!(c.first_extra, draw.culled.map(|(k, _)| k));
                assert_eq!(c.full_keep(), c.keep_stop(TARGET));
                for stops in STOPS.windows(2) {
                    assert!(!c.keep_stop(stops[1]) || c.keep_stop(stops[0]));
                }
                assert!(!c.full_keep() || c.keep_pm());
                assert_eq!(c.additional_mask() & (1 << (STOPS.len() - 1)), 0);
            }
        }
    }

    #[test]
    fn paired_disjoint_ratio_variance() {
        // Four equally weighted observations: extra-only twice, full-only
        // once, neither once. Population four makes estimated totals 2 and 1.
        let mut x = Moments::default();
        let mut y = Moments::default();
        x.add([1.0, 0.0, 0.0, 0.0, 0.0]);
        x.add([1.0, 0.0, 0.0, 0.0, 0.0]);
        y.add([1.0, 0.0, 0.0, 0.0, 0.0]);
        let mut tx = Total::default();
        let mut ty = Total::default();
        tx.add_class(x, 4.0, 4);
        ty.add_class(y, 4.0, 4);
        assert_eq!(tx.estimate[0], 2.0);
        assert_eq!(ty.estimate[0], 1.0);
        assert!((tx.variance[0] - 4.0 / 3.0).abs() < 1e-12);
        assert!((ty.variance[0] - 1.0).abs() < 1e-12);
        let cov = -16.0 * x.sum[0] * y.sum[0] / (4 * 4 * 3) as f64;
        assert!((cov + 2.0 / 3.0).abs() < 1e-12);
    }
}
