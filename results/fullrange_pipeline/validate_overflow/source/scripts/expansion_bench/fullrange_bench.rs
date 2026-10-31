//! Fixed-draw checkpoint cohorts for extrapolating the live N72 GPU pipeline.
//!
//! Unlike transport_bench, sample size counts baseline draws, including rejected
//! draws and duplicates. Every baseline K38 residue has the same inclusion
//! probability. A rejected draw contributes zero downstream work; an accepted
//! duplicate is dispatched again. Timing repetitions measure noise, while the
//! independent cohorts measure sampling uncertainty.
#[allow(dead_code)]
mod affine;
#[allow(dead_code)]
mod combined;
#[allow(dead_code)]
mod fast_tail;
#[cfg(test)]
#[allow(dead_code)]
mod tail_policy;
#[allow(dead_code)]
mod terminal;
#[allow(dead_code)]
mod terminal_walk;
#[allow(dead_code)]
mod ternary;

use affine::Affine;
use combined::{Combined, Mode};
use cv::dp::{BackwardDp, PState, ParityRules};
use cv::gpu::{GpuSession, GpuTransport, GpuTuning};
use cv::lookahead::{BvMode, WindowRules};
use cv::rules::{walkback_certificate, Rules};
use cv::sample::{Rng, Sampler};
use cv::tail::TailParams;
use cv::tree::{Node, TreeStats};
use cv::verify::{verify_cpu_with_expander, VerifyConfig, VerifyReport};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::time::Instant;

const CHECKPOINT: u32 = 38;
const WALKBACK_CAP: u32 = 65_536;

fn cpu_seconds() -> f64 {
    #[repr(C)]
    struct Timespec {
        seconds: std::os::raw::c_long,
        nanoseconds: std::os::raw::c_long,
    }
    unsafe extern "C" {
        fn clock_gettime(clock: i32, time: *mut Timespec) -> i32;
    }
    let mut ts = Timespec {
        seconds: 0,
        nanoseconds: 0,
    };
    assert_eq!(unsafe { clock_gettime(2, &mut ts) }, 0);
    ts.seconds as f64 + ts.nanoseconds as f64 * 1e-9
}

fn env_number(name: &str, default: usize) -> usize {
    std::env::var(name)
        .map(|value| value.parse().unwrap_or_else(|_| panic!("invalid {name}")))
        .unwrap_or(default)
}

fn config(n: u32, a: u32, b: u32, num_bv: usize) -> VerifyConfig {
    let rules = Rules::improved(13);
    VerifyConfig {
        n,
        a,
        b,
        num_bv,
        bv_mode: BvMode::Shared,
        window: WindowRules {
            pm_inf: true,
            mix: rules.mix.clone(),
        },
        class_b: None,
        split: 10,
        rules,
        tail: TailParams::default(),
        units: None,
    }
}

fn signature(report: &VerifyReport) -> Value {
    let mut flagged = report.flagged.clone();
    flagged.sort_unstable();
    json!({"tree":report.tree,"tail":report.tail,"hash":report.hash,
        "flagged":flagged,"flagged_rechecked":report.flagged_rechecked})
}

struct Cohort {
    roots: Vec<Node>,
    metadata: Value,
}

fn cohort(
    index: usize,
    count: usize,
    seed: u64,
    sampler: &Sampler,
    descendants: &BackwardDp,
    rules: &Rules,
    walkback_cap: u32,
) -> Cohort {
    let started = Instant::now();
    let cpu = cpu_seconds();
    let draws = sampler.draws(count, seed, Some(rules));
    // This control-variate diagnostic includes ALL baseline draws, including
    // those subsequently rejected. Its exact population mean is independently
    // known from the baseline backward DP.
    let baseline_descendants: u128 = draws
        .iter()
        .map(|draw| descendants.ways(PState::of(&draw.node)))
        .sum();
    let classified: Vec<_> = draws
        .into_par_iter()
        .map(|draw| {
            if draw.culled.is_some() {
                return (draw.node, 1u8);
            }
            let rejected = walkback_certificate(
                draw.node.n0,
                draw.node.k,
                draw.node.f,
                draw.node.m,
                walkback_cap,
            )
            .is_some();
            (draw.node, if rejected { 2 } else { 0 })
        })
        .collect();
    let cheap_rejected = classified.iter().filter(|(_, why)| *why == 1).count();
    let walk_rejected = classified.iter().filter(|(_, why)| *why == 2).count();
    let roots: Vec<_> = classified
        .into_iter()
        .filter_map(|(node, why)| (why == 0).then_some(node))
        .collect();
    assert_eq!(count, cheap_rejected + walk_rejected + roots.len());
    let unique = roots
        .iter()
        .map(|node| node.n0)
        .collect::<HashSet<_>>()
        .len();
    let metadata = json!({
        "type":"cohort","cohort":index,"seed":seed,"drawn":count,
        "cheap_rejected":cheap_rejected,"walk_rejected":walk_rejected,
        "accepted":roots.len(),"duplicates_retained":roots.len()-unique,
        "baseline_descendants":baseline_descendants.to_string(),
        "roots":roots,"sampling_seconds":started.elapsed().as_secs_f64(),
        "sampling_process_cpu_seconds":cpu_seconds()-cpu,
    });
    Cohort { roots, metadata }
}

fn engines(cfg: &VerifyConfig, cutoff: u32, entry: u32) -> (Combined, Affine) {
    (
        Combined::new(
            cfg.m(),
            &cfg.rules,
            Mode::Terminal {
                cutoff,
                width: cfg.m() - entry,
            },
        ),
        Affine::new(cfg.m(), cutoff, entry, &cfg.rules),
    )
}

fn run_variant(
    gpu: &mut GpuSession,
    cfg: &VerifyConfig,
    compact: &Combined,
    affine: &Affine,
    name: &str,
) -> VerifyReport {
    let words = affine.table_words();
    let mut tuning = GpuTuning::default();
    tuning.overflow_vote = !matches!(
        name,
        "novote" | "both" | "both_oee" | "block128" | "block512"
    );
    tuning.readonly_powers = matches!(
        name,
        "readonly" | "both" | "both_oee" | "block128" | "block512"
    );
    tuning.oee_mask = matches!(name, "both_oee" | "block128" | "block512");
    tuning.block_threads = match name {
        "block128" => 128,
        "block512" => 512,
        _ => 256,
    };
    match name {
        "replay16" => gpu.run_transport(cfg, compact, GpuTransport::Replay16, words),
        "parents" | "novote" | "readonly" | "both" | "both_oee" | "block128" | "block512" => {
            gpu.run_transport_tuned(cfg, affine, GpuTransport::Parents, words, tuning)
        }
        _ => panic!("unknown variant {name}"),
    }
    .expect("GPU full-range cohort run")
}

fn validate(wide: bool, a: u32, names: &[&str]) {
    let mut cfg = config(if wide { 72 } else { 32 }, a, 12, 2);
    cfg.class_b = Some(vec![12, 13]);
    let (cutoff, entry) = if wide { (50, 52) } else { (16, 18) };
    if wide {
        let sampler = Sampler::new(&Rules::baseline(), entry, None);
        let mut rng = Rng(31_419);
        let mut roots = Vec::new();
        while roots.len() < 16 {
            let draw = sampler.draw(&mut rng, Some(&cfg.rules));
            if draw.culled.is_none() {
                roots.push(draw.node);
            }
        }
        cfg.units = Some(roots);
    } else {
        cfg.rules.walkback = Some(cv::rules::Walkback {
            depth: 12,
            max_states: 256,
        });
    }
    let (compact, affine) = engines(&cfg, cutoff, entry);
    let cpu = verify_cpu_with_expander(&cfg, &compact);
    let affine_cpu = verify_cpu_with_expander(&cfg, &affine);
    assert_eq!(signature(&cpu), signature(&affine_cpu));
    let mut gpu = GpuSession::new_experimental(&cfg).expect("GPU initialization");
    gpu.set_terminal_table(affine.table_words())
        .expect("table upload");
    for name in names {
        let report = run_variant(&mut gpu, &cfg, &compact, &affine, name);
        assert_eq!(signature(&cpu), signature(&report), "validation {name}");
        assert!(report.flagged.is_empty());
        assert!(report.tail.endpoints[cfg.num_bv] > 0);
        println!(
            "{}",
            json!({"type":"validation","variant":name,
            "scope":if wide {"sixteen K52 packages, N72 with CPU fallback"} else {"complete N32 with CPU fallback"},
            "cpu_gpu_equal":true,"roots":cfg.units,"report":report})
        );
    }
}

fn validate_overflow(names: &[&str]) {
    let mut cfg = config(72, 10, 12, 24);
    cfg.class_b = Some((0..cfg.num_bv).map(|i| 12 + (i % 2) as u32).collect());
    cfg.tail.cutoff_bits = 97;
    let baseline = Rules::baseline();
    let mut high = Node::root(&baseline);
    while high.k < 52 {
        high = high.children(&baseline)[1].expect("all-one residue remains odd");
        assert!(high.cull(&baseline).is_none());
    }
    assert_eq!(high.n0, (1u128 << 52) - 1);
    assert_eq!(high.f, 52);
    let mut roots = vec![high];
    let sampler = Sampler::new(&baseline, 52, None);
    let mut rng = Rng(31_419);
    while roots.len() < 17 {
        let draw = sampler.draw(&mut rng, Some(&cfg.rules));
        if draw.culled.is_none() {
            roots.push(draw.node);
        }
    }
    cfg.units = Some(roots);
    let (compact, affine) = engines(&cfg, 50, 52);
    let cpu = verify_cpu_with_expander(&cfg, &compact);
    let affine_cpu = verify_cpu_with_expander(&cfg, &affine);
    assert_eq!(signature(&cpu), signature(&affine_cpu));
    assert!(
        !cpu.flagged.is_empty(),
        "overflow audit must exercise flag path"
    );
    assert_eq!(cpu.flagged.len(), cpu.flagged_rechecked);
    let mut gpu = GpuSession::new_experimental(&cfg).expect("GPU initialization");
    gpu.set_terminal_table(affine.table_words())
        .expect("table upload");
    for name in names {
        let report = run_variant(&mut gpu, &cfg, &compact, &affine, name);
        assert_eq!(
            signature(&cpu),
            signature(&report),
            "overflow validation {name}"
        );
        let gpu_flags = report.gpu.as_ref().unwrap()["flags"].as_u64().unwrap();
        assert!(
            gpu_flags > 0,
            "overflow audit must exercise actual GPU flag path"
        );
        assert_eq!(report.flagged.len(), report.flagged_rechecked);
        println!(
            "{}",
            json!({"type":"validation","variant":name,
            "scope":"N72 forced GPU overflow, cutoff97, high-F all-one K52 root plus sixteen random K52 roots",
            "cpu_gpu_equal":true,"flagged_origins_equal":true,"unresolved_flags":0,
            "cutoff_bits":cfg.tail.cutoff_bits,"roots":cfg.units,"report":report})
        );
    }
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let profile = args.first().map(String::as_str).unwrap_or("validate");
    let number =
        |i: usize, default: usize| args.get(i).map(|s| s.parse().unwrap()).unwrap_or(default);
    let threads = number(3, 2);
    let a = number(7, 10) as u32;
    let names: Vec<_> = args
        .get(5)
        .map(String::as_str)
        .unwrap_or("parents")
        .split(',')
        .collect();
    assert!((8..=10).contains(&a));
    assert!(threads > 0 && !names.is_empty());
    assert_eq!(names.iter().collect::<HashSet<_>>().len(), names.len());
    assert!(names.iter().all(|name| matches!(
        *name,
        "parents"
            | "replay16"
            | "novote"
            | "readonly"
            | "both"
            | "both_oee"
            | "block128"
            | "block512"
    )));
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    if profile == "validate-device" {
        validate_overflow(&names);
        return;
    }
    if matches!(profile, "validate" | "validate-wide") {
        validate(profile == "validate-wide", a, &names);
        return;
    }
    assert_eq!(profile, "target");
    let drawn = number(1, 512);
    let repetitions = number(2, 2);
    let seed = number(4, 20261026) as u64;
    let count = env_number("CV_FULLRANGE_COHORTS", 16);
    let cutoff = env_number("CV_FULLRANGE_CUTOFF", 50) as u32;
    let entry = env_number("CV_FULLRANGE_ENTRY", 52) as u32;
    let aggregate_repetitions = env_number("CV_FULLRANGE_AGGREGATE", 1);
    assert!(drawn > 0 && repetitions > 0 && count > 0);
    assert!(CHECKPOINT <= cutoff && cutoff <= entry);
    let mut cfg = config(72, a, 27, 16);
    cfg.class_b = Some(
        args.get(6)
            .map(String::as_str)
            .unwrap_or("32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27")
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect(),
    );
    assert_eq!(cfg.class_b.as_ref().unwrap().len(), cfg.num_bv);
    let started = Instant::now();
    let (compact, affine) = engines(&cfg, cutoff, entry);
    let engine_setup_seconds = started.elapsed().as_secs_f64();
    let baseline = Rules::baseline();
    let sampler = Sampler::new(&baseline, CHECKPOINT, None);
    let descendants = BackwardDp::run(ParityRules::of(&baseline), 2, cfg.m(), None);
    let baseline_endpoint_population = descendants.ways(PState::of(&Node::root(&baseline)));
    println!(
        "{}",
        json!({
            "type":"config","experiment":"fullrange","n":cfg.n,"a":a,"m":cfg.m(),
            "checkpoint":CHECKPOINT,"cutoff":cutoff,"terminal_entry":entry,
            "terminal_width":cfg.m()-entry,"walkback_cap":WALKBACK_CAP,
            "sampling_design":"fixed_baseline_draws_with_replacement",
            "baseline_population":sampler.population().to_string(),
            "baseline_endpoint_population":baseline_endpoint_population.to_string(),
            "cohort_count":count,"draws_per_cohort":drawn,"seed":seed,
            "repetitions":repetitions,"aggregate_repetitions":aggregate_repetitions,
            "threads":threads,"num_bv":cfg.num_bv,"class_b":cfg.class_b,
            "window":cfg.window.label(),"rules":cfg.rules.label(),"variants":names,
            "engine_setup_seconds":engine_setup_seconds,
            "terminal_table_bytes":affine.table_bytes(),
            "terminal_word_count":affine.table_words().len(),
            "timed_scope":"live expansion after K38, packing, queues, transfers, device expansion, verification and fallback; resident arrays and suffix tables",
            "excluded_scope":"K38 frontier construction and walkback preprocessing, sampling, resident-array construction, checkpoint storage and distributed coordination",
            "kernel_seconds_meaning":"overlapping CUDA event sums, not GPU busy wall time",
            "sampling_protocol":"SplitMix64 cohort seeds followed by Sampler::draws deterministic blocks; duplicates retained; improved13 ancestry and bounded K38 walkback rejections contribute zero downstream work"
        })
    );
    let mut seeder = Rng(seed);
    let mut cohorts = Vec::with_capacity(count);
    for index in 0..count {
        let sampled = cohort(
            index,
            drawn,
            seeder.next_u64(),
            &sampler,
            &descendants,
            &cfg.rules,
            WALKBACK_CAP,
        );
        println!("{}", sampled.metadata);
        cohorts.push(sampled);
    }
    let started = Instant::now();
    let cpu = cpu_seconds();
    let mut gpu = GpuSession::new_experimental(&cfg).expect("GPU initialization");
    gpu.set_terminal_table(affine.table_words())
        .expect("table upload");
    println!(
        "{}",
        json!({"type":"gpu_setup","wall_seconds":started.elapsed().as_secs_f64(),
        "process_cpu_seconds":cpu_seconds()-cpu})
    );
    cfg.units = Some(
        cohorts
            .iter()
            .flat_map(|c| c.roots.iter())
            .take(32)
            .copied()
            .collect(),
    );
    let mut warmup_signature = None;
    for name in &names {
        let report = run_variant(&mut gpu, &cfg, &compact, &affine, name);
        assert!(report.flagged.is_empty());
        if let Some(want) = &warmup_signature {
            assert_eq!(*want, signature(&report), "warmup {name}");
        } else {
            warmup_signature = Some(signature(&report));
        }
        println!(
            "{}",
            json!({"type":"warmup","variant":name,"report":report})
        );
    }
    let mut expected: Vec<Option<Value>> = vec![None; count];
    // Rotate cohort and variant order between repetitions to avoid binding a
    // particular cohort or variant to warmup, clock drift, or desktop activity.
    for repetition in 0..repetitions {
        for cohort_position in 0..count {
            let index = (cohort_position + repetition) % count;
            cfg.units = Some(cohorts[index].roots.clone());
            for order_position in 0..names.len() {
                let name = names[(order_position + index + repetition) % names.len()];
                let cpu = cpu_seconds();
                let started = Instant::now();
                let report = run_variant(&mut gpu, &cfg, &compact, &affine, name);
                let wall = started.elapsed().as_secs_f64();
                let cpu = cpu_seconds() - cpu;
                assert!(report.flagged.is_empty());
                if let Some(want) = &expected[index] {
                    assert_eq!(*want, signature(&report), "cohort {index} variant {name}");
                } else {
                    expected[index] = Some(signature(&report));
                }
                println!(
                    "{}",
                    json!({"type":"trial","cohort":index,"variant":name,
                    "repetition":repetition,"cohort_order_position":cohort_position,
                    "order_position":order_position,"drawn":drawn,
                    "wall_seconds":wall,"process_cpu_seconds":cpu,
                    "average_busy_cpu_cores":cpu/wall,"report":report})
                );
            }
        }
    }
    if aggregate_repetitions > 0 {
        let roots: Vec<_> = cohorts
            .iter()
            .flat_map(|c| c.roots.iter())
            .copied()
            .collect();
        println!(
            "{}",
            json!({"type":"aggregate_config","drawn":drawn*count,
            "accepted":roots.len(),"roots":roots,
            "cohort_order":(0..count).collect::<Vec<_>>(),
            "scope":"same accepted draws concatenated; measures flush and partial-batch overhead, not an independent sampling replicate"})
        );
        cfg.units = Some(roots);
        let mut aggregate_tree = TreeStats::new(cfg.m());
        let mut aggregate_tail = cv::tail::TailStats::new(cfg.num_bv);
        for e in &expected {
            let e = e.as_ref().unwrap();
            aggregate_tree.merge(&serde_json::from_value(e["tree"].clone()).unwrap());
            aggregate_tail.merge(&serde_json::from_value(e["tail"].clone()).unwrap());
        }
        let mut aggregate_expected = None;
        for repetition in 0..aggregate_repetitions {
            for order_position in 0..names.len() {
                let name = names[(order_position + repetition) % names.len()];
                let cpu = cpu_seconds();
                let started = Instant::now();
                let report = run_variant(&mut gpu, &cfg, &compact, &affine, name);
                let wall = started.elapsed().as_secs_f64();
                let cpu = cpu_seconds() - cpu;
                assert!(report.flagged.is_empty());
                if let Some(want) = &aggregate_expected {
                    assert_eq!(*want, signature(&report), "aggregate {name}");
                } else {
                    aggregate_expected = Some(signature(&report));
                }
                // Every tree/per-class work counter and XOR hash must equal
                // the sum of the separate cohorts, including duplicate draws.
                assert_eq!(
                    serde_json::to_value(&report.tree).unwrap(),
                    serde_json::to_value(&aggregate_tree).unwrap(),
                    "aggregate tree"
                );
                assert_eq!(
                    serde_json::to_value(&report.tail).unwrap(),
                    serde_json::to_value(&aggregate_tail).unwrap(),
                    "aggregate tail"
                );
                println!(
                    "{}",
                    json!({"type":"aggregate_trial","variant":name,
                    "repetition":repetition,"order_position":order_position,
                    "drawn":drawn*count,"accepted":cfg.units.as_ref().unwrap().len(),
                    "wall_seconds":wall,"process_cpu_seconds":cpu,
                    "average_busy_cpu_cores":cpu/wall,"report":report})
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_draw_cohorts_keep_duplicates_and_rejections_in_denominator() {
        let baseline = Rules::baseline();
        let rules = Rules::improved(5);
        let sampler = Sampler::new(&baseline, 10, None);
        let descendants = BackwardDp::run(ParityRules::of(&baseline), 2, 14, None);
        let sampled = cohort(0, 256, 1234, &sampler, &descendants, &rules, 16);
        let same = cohort(0, 256, 1234, &sampler, &descendants, &rules, 16);
        assert_eq!(sampled.roots, same.roots);
        assert_eq!(sampled.metadata["drawn"], 256);
        assert!(sampled.metadata["duplicates_retained"].as_u64().unwrap() > 0);
        let total: u64 = ["accepted", "cheap_rejected", "walk_rejected"]
            .iter()
            .map(|key| sampled.metadata[*key].as_u64().unwrap())
            .sum();
        assert_eq!(total, 256);
        let draws = sampler.draws(256, 1234, Some(&rules));
        let reference: Vec<_> = draws
            .into_iter()
            .filter(|draw| {
                draw.culled.is_none()
                    && walkback_certificate(draw.node.n0, draw.node.k, draw.node.f, draw.node.m, 16)
                        .is_none()
            })
            .map(|draw| draw.node)
            .collect();
        assert_eq!(sampled.roots, reference);
    }
}
