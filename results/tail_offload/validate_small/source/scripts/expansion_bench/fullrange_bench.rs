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
mod chained;
#[allow(dead_code)]
mod combined;
#[allow(dead_code)]
mod compact_chain;
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
use chained::{Chained, LayerStats};
use combined::{Combined, Mode};
use compact_chain::CompactChain;
use cv::dp::{BackwardDp, PState, ParityRules};
use cv::gpu::{GpuSession, GpuTransport, GpuTuning};
use cv::lookahead::{BvMode, WindowRules};
use cv::rules::{walkback_certificate, Rules};
use cv::sample::{Rng, Sampler};
use cv::tail::{TailParams, TailStats};
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

fn normalized_flags(flags: impl IntoIterator<Item = u128>) -> Vec<String> {
    let mut flags: Vec<_> = flags.into_iter().collect();
    flags.sort_unstable();
    flags.into_iter().map(|n| n.to_string()).collect()
}

fn tail_signature(tail: &TailStats) -> Value {
    let mut value = serde_json::to_value(tail).unwrap();
    // TailStats intentionally skips raw u128 flags in public JSON. Include
    // exact decimal strings in the internal signature, with duplicates kept.
    value["flagged"] = json!(normalized_flags(tail.flagged.iter().copied()));
    value
}

fn signature(report: &VerifyReport) -> Value {
    let flagged = normalized_flags(report.flagged.iter().map(|n| n.parse().unwrap()));
    let tail = tail_signature(&report.tail);
    assert_eq!(
        json!(flagged),
        tail["flagged"],
        "report/tail flagged origins"
    );
    json!({"tree":report.tree,"tail":tail,"hash":report.hash,
        "flagged":flagged,"flagged_rechecked":report.flagged_rechecked})
}

fn require_resolved(report: &VerifyReport) {
    assert_eq!(
        report.flagged.len(),
        report.flagged_rechecked,
        "benchmark contains unresolved flagged starts"
    );
}

fn aggregate_signature(signatures: &[Option<Value>], m: u32, num_bv: usize) -> Value {
    let mut tree = TreeStats::new(m);
    let mut tail = TailStats::new(num_bv);
    let mut flagged = Vec::new();
    let mut flagged_rechecked = 0u64;
    for signature in signatures {
        let signature = signature.as_ref().unwrap();
        tree.merge(&serde_json::from_value(signature["tree"].clone()).unwrap());
        let mut part: TailStats = serde_json::from_value(signature["tail"].clone()).unwrap();
        // serde(skip) means these must be restored explicitly.
        part.flagged = signature["tail"]["flagged"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_str().unwrap().parse().unwrap())
            .collect();
        tail.merge(&part);
        flagged.extend(
            signature["flagged"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_str().unwrap().parse::<u128>().unwrap()),
        );
        flagged_rechecked += signature["flagged_rechecked"].as_u64().unwrap();
    }
    json!({"tree":tree,"tail":tail_signature(&tail),
        "hash":format!("{:08X}", tail.hash),"flagged":normalized_flags(flagged),
        "flagged_rechecked":flagged_rechecked})
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

struct CompactExperiments {
    early: Affine,
    chains: Vec<CompactChain>,
}

fn engines(cfg: &VerifyConfig, cutoff: u32, entry: u32) -> (Combined, Affine, Vec<Chained>, CompactExperiments) {
    let earliest_root = cfg.units.as_ref().and_then(|roots| roots.iter().map(|n| n.k).min()).unwrap_or(cutoff);
    let early_entry = cutoff.max(earliest_root).max(cfg.m().saturating_sub(12));
    let span = cfg.m() - early_entry;
    assert!(span >= 2);
    let compressed = CompactExperiments {
        early: Affine::new(cfg.m(), cutoff, early_entry, &cfg.rules),
        chains: [6.min(span / 2), 4.min(span - 1)].into_iter()
            .map(|width| CompactChain::new(cfg.m(), cutoff, early_entry, width, &cfg.rules)).collect(),
    };
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
        [4, 6]
            .into_iter()
            .flat_map(|width| {
                [false, true]
                    .into_iter()
                    .map(move |grouped| Chained::new(cfg.m(), cutoff, width, &cfg.rules, grouped))
            })
            .collect(),
        compressed,
    )
}

/// Independent structural diagnostics: preserve the fixed-draw cohort design,
/// but cap expansion work at the first sixteen accepted roots of each cohort.
/// These are arithmetic-sharing measurements, not elapsed-time estimates.
fn chain_census(cohorts: &[Cohort], chains: &[Chained], rules: &Rules, names: &[&str]) {
    const ROOT_CAP: usize = 16;
    let cohort_roots: Vec<_> = cohorts
        .iter()
        .enumerate()
        .map(|(index, cohort)| {
            json!({"cohort":index, "accepted":cohort.roots.len(),
               "profiled":cohort.roots.len().min(ROOT_CAP)})
        })
        .collect();
    let profiled_roots: usize = cohorts.iter().map(|c| c.roots.len().min(ROOT_CAP)).sum();
    for name in names {
        let index = match *name {
            "chain4" => 0,
            "group4" | "group4_jump12" => 1,
            "chain6" => 2,
            "group6" => 3,
            _ => panic!("census requires chain4, group4, chain6, or group6 variants"),
        };
        let chain = &chains[index];
        let mut totals: Vec<_> = chain
            .table_boundaries()
            .into_iter()
            .map(|(start, target)| LayerStats {
                start,
                target,
                ..LayerStats::default()
            })
            .collect();
        for cohort in cohorts {
            for root in cohort.roots.iter().take(ROOT_CAP) {
                for layer in chain.census(root, rules) {
                    let total = totals
                        .iter_mut()
                        .find(|t| (t.start, t.target) == (layer.start, layer.target))
                        .expect("census returned an unexpected table boundary");
                    total.parents += layer.parents;
                    total.buckets += layer.buckets;
                    total.parents_in_shared_buckets += layer.parents_in_shared_buckets;
                    total.largest_bucket = total.largest_bucket.max(layer.largest_bucket);
                    total.table_dispatches += layer.table_dispatches;
                    total.word_translations += layer.word_translations;
                    total.output_nodes += layer.output_nodes;
                }
            }
        }
        let ratio = |numerator: u64, denominator: u64| {
            (denominator > 0).then(|| numerator as f64 / denominator as f64)
        };
        for total in totals {
            let mut record = serde_json::to_value(&total).unwrap();
            let fields = record.as_object_mut().unwrap();
            fields.insert("type".into(), json!("chain_census"));
            fields.insert("variant".into(), json!(name));
            fields.insert("engine".into(), json!(chain.name()));
            fields.insert("cohort_roots".into(), json!(cohort_roots));
            fields.insert("profiled_roots".into(), json!(profiled_roots));
            fields.insert("root_cap_per_cohort".into(), json!(ROOT_CAP));
            fields.insert(
                "mean_parents_per_bucket".into(),
                json!(ratio(total.parents, total.buckets)),
            );
            fields.insert(
                "shared_parent_fraction".into(),
                json!(ratio(total.parents_in_shared_buckets, total.parents)),
            );
            fields.insert(
                "dispatch_reduction".into(),
                json!(ratio(total.parents - total.table_dispatches, total.parents)),
            );
            fields.insert(
                "word_translation_reduction".into(),
                json!(ratio(
                    total.output_nodes - total.word_translations,
                    total.output_nodes
                )),
            );
            fields.insert("scope".into(), json!("Untimed CPU structural census; first at most sixteen accepted K38 draws per cohort, duplicates retained; no GPU initialization or tail execution."));
            println!("{}", record);
        }
    }
}

fn jump_width(name: &str) -> u32 {
    match name {
        "jump8" => 8,
        "jump10" => 10,
        "jump12" | "group4_jump12" => 12,
        "jump14" => 14,
        _ => 0,
    }
}

// These fixed labels keep the dispatch settings independently auditable. The
// maximal budget is an instrumentation control: it exercises the budget branch
// without deliberately moving any ordinary trajectory to the CPU.
fn tail_offload(name: &str) -> Option<(u32, u32, u32)> {
    let (budget, capacity, workers) = match name {
        "defer1" => (1, 65_536, 1),
        "defer32" => (32, 65_536, 1),
        "defer64" => (64, 65_536, 1),
        "defer128" | "compact_defer128" => (128, 65_536, 1),
        "defer256" | "compact_defer256" => (256, 65_536, 1),
        "defer512" => (512, 65_536, 1),
        "defer_count" => (u32::MAX, 65_536, 1),
        "defer1_cap1" => (1, 1, 1),
        "defer32_w2" => (32, 65_536, 2),
        "defer64_w2" => (64, 65_536, 2),
        "defer128_w2" => (128, 65_536, 2),
        "defer256_w2" => (256, 65_536, 2),
        _ => return None,
    };
    Some((budget, capacity, workers))
}

fn compact_chain_index(name: &str) -> Option<usize> {
    match name {
        "compact66" | "compact_defer128" | "compact_defer256" => Some(0),
        "compact48" => Some(1),
        _ => None,
    }
}

fn prepare_jumps(gpu: &mut GpuSession, names: &[&str]) {
    for width in [8, 10, 12, 14] {
        if names.iter().any(|name| jump_width(name) == width) {
            gpu.set_tail_jump(width).expect("tail jump table upload");
        }
    }
}

// Keep uploads and table switching outside all warm pipeline timings.
fn prepare_variant(gpu: &mut GpuSession, affine: &Affine, compact: &CompactExperiments, name: &str) {
    let name = name.strip_suffix("_1k").unwrap_or(name);
    if let Some(index) = compact_chain_index(name) {
        gpu.set_chain_table(compact.chains[index].tables()).expect("compact chain upload");
    } else if name == "early_parents" {
        gpu.set_terminal_table(compact.early.table_words()).expect("early table upload");
    } else {
        gpu.set_terminal_table(affine.table_words()).expect("baseline table upload");
    }
    if let Some((_, capacity, _)) = tail_offload(name) {
        gpu.prepare_tail_offload(capacity).expect("tail offload buffer allocation");
    }
}

fn run_variant(
    gpu: &mut GpuSession,
    cfg: &VerifyConfig,
    compact: &Combined,
    affine: &Affine,
    chains: &[Chained],
    compressed: &CompactExperiments,
    name: &str,
) -> VerifyReport {
    let words = affine.table_words();
    let mut tuning = GpuTuning::default();
    if name.ends_with("_1k") {
        tuning.group_chunk_records = 1024;
    }
    let name = name.strip_suffix("_1k").unwrap_or(name);
    tuning.overflow_vote = !matches!(
        name,
        "novote"
            | "both"
            | "both_oee"
            | "block128"
            | "block512"
            | "novote_oee"
            | "novote_oee128"
            | "novote_oee512"
    );
    tuning.readonly_powers = matches!(
        name,
        "readonly" | "both" | "both_oee" | "block128" | "block512"
    );
    tuning.oee_mask = matches!(
        name,
        "both_oee" | "block128" | "block512" | "novote_oee" | "novote_oee128" | "novote_oee512"
    );
    tuning.block_threads = match name {
        "block128" | "novote_oee128" => 128,
        "block512" | "novote_oee512" => 512,
        _ => 256,
    };
    let experiment = matches!(
        name,
        "affine32" | "chain4" | "group4" | "chain6" | "group6" | "group4_jump12"
        | "early_parents" | "compact66" | "compact48"
    ) || jump_width(name) > 0 || tail_offload(name).is_some();
    if experiment {
        tuning.overflow_vote = false;
        tuning.readonly_powers = false;
        tuning.oee_mask = true;
        tuning.tail_jump_bits = jump_width(name);
    }
    if let Some((budget, capacity, workers)) = tail_offload(name) {
        tuning.tail_budget = budget;
        tuning.tail_queue_capacity = capacity;
        tuning.tail_cpu_workers = workers;
    }
    match name {
        "early_parents" => gpu.run_transport_tuned(cfg, &compressed.early, GpuTransport::Parents,
                                                   compressed.early.table_words(), tuning),
        "compact66" | "compact48" | "compact_defer128" | "compact_defer256" => {
            let engine = &compressed.chains[compact_chain_index(name).unwrap()];
            gpu.run_chain_tuned(cfg, engine, engine.tables(), tuning)
        }
        "affine32" => gpu.run_transport_tuned(cfg, affine, GpuTransport::Affine32, &[], tuning),
        "chain4" | "group4" | "chain6" | "group6" | "group4_jump12" => {
            let index = match name {
                "chain4" => 0,
                "group4" | "group4_jump12" => 1,
                "chain6" => 2,
                _ => 3,
            };
            gpu.run_transport_tuned(cfg, &chains[index], GpuTransport::Affine32, &[], tuning)
        }
        "jump8" | "jump10" | "jump12" | "jump14" => {
            gpu.run_transport_tuned(cfg, affine, GpuTransport::Parents, words, tuning)
        }
        "replay16" => gpu.run_transport(cfg, compact, GpuTransport::Replay16, words),
        "parents" | "novote" | "readonly" | "both" | "both_oee" | "block128" | "block512"
        | "novote_oee" | "novote_oee128" | "novote_oee512" => {
            gpu.run_transport_tuned(cfg, affine, GpuTransport::Parents, words, tuning)
        }
        name if tail_offload(name).is_some() => {
            gpu.run_transport_tuned(cfg, affine, GpuTransport::Parents, words, tuning)
        }
        _ => panic!("unknown variant {name}"),
    }
    .expect("GPU full-range cohort run")
}

fn validate_tail_queue(name: &str, report: &VerifyReport) {
    if name == "defer1" || name == "defer1_cap1" {
        let gpu = report.gpu.as_ref().unwrap();
        assert!(gpu["deferred_cases"].as_u64().unwrap() > 0,
                "budget-one fixture must exercise CPU continuation");
    }
    if name == "defer1_cap1" {
        let gpu = report.gpu.as_ref().unwrap();
        assert!(gpu["deferred_rejected"].as_u64().unwrap() > 0,
                "single-record queue fixture must exercise full-queue GPU completion");
    }
    if name == "defer_count" {
        let gpu = report.gpu.as_ref().unwrap();
        assert_eq!(gpu["deferred_attempts"].as_u64().unwrap(), 0,
                   "instrumentation control must retain validation trajectories on GPU");
        assert_eq!(gpu["deferred_cases"].as_u64().unwrap(), 0);
    }
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
    let (compact, affine, chains, compressed) = engines(&cfg, cutoff, entry);
    let cpu = verify_cpu_with_expander(&cfg, &compact);
    let affine_cpu = verify_cpu_with_expander(&cfg, &affine);
    assert_eq!(signature(&cpu), signature(&affine_cpu));
    let mut gpu = GpuSession::new_experimental(&cfg).expect("GPU initialization");
    gpu.set_terminal_table(affine.table_words())
        .expect("table upload");
    prepare_jumps(&mut gpu, &names);
    for name in names {
        prepare_variant(&mut gpu, &affine, &compressed, name);
        let report = run_variant(&mut gpu, &cfg, &compact, &affine, &chains, &compressed, name);
        assert_eq!(signature(&cpu), signature(&report), "validation {name}");
        require_resolved(&report);
        validate_tail_queue(name, &report);
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
    cfg.tail.cutoff_bits = 110;
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
    let (compact, affine, chains, compressed) = engines(&cfg, 50, 52);
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
    prepare_jumps(&mut gpu, &names);
    for name in names {
        prepare_variant(&mut gpu, &affine, &compressed, name);
        let report = run_variant(&mut gpu, &cfg, &compact, &affine, &chains, &compressed, name);
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
        validate_tail_queue(name, &report);
        if *name == "defer1" {
            assert!(report.gpu.as_ref().unwrap()["deferred_flags"].as_u64().unwrap() > 0,
                    "budget-one overflow fixture must exercise flagged CPU continuation");
        }
        println!(
            "{}",
            json!({"type":"validation","variant":name,
            "scope":"N72 forced GPU overflow, cutoff110, high-F all-one K52 root plus sixteen random K52 roots",
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
        .unwrap_or(if profile == "census" {
            "chain4,group4,chain6,group6"
        } else {
            "parents"
        })
        .split(',')
        .collect();
    assert!((8..=10).contains(&a));
    assert!(threads > 0 && !names.is_empty());
    assert_eq!(names.iter().collect::<HashSet<_>>().len(), names.len());
    assert!(names.iter().all(|name| tail_offload(name).is_some() || matches!(
        *name,
        "parents"
            | "replay16"
            | "novote"
            | "readonly"
            | "both"
            | "both_oee"
            | "block128"
            | "block512"
            | "novote_oee"
            | "novote_oee128"
            | "novote_oee512"
            | "affine32"
            | "chain4"
            | "group4"
            | "chain6"
            | "group6"
            | "jump8"
            | "jump10"
            | "jump12"
            | "jump14"
            | "group4_jump12"
            | "early_parents" | "compact66" | "compact48"
            | "early_parents_1k" | "compact66_1k" | "compact48_1k"
            | "novote_oee_1k"
    )));
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    if profile == "validate-jump" {
        unsafe extern "C" {
            fn cv_gpu_tail_jump_self_test() -> i32;
        }
        assert_eq!(
            unsafe { cv_gpu_tail_jump_self_test() },
            0,
            "direct GPU jump validation"
        );
        println!(
            "{}",
            json!({"type":"validation", "scope":"adversarial direct GPU tails", "exact_equal":true})
        );
        return;
    }
    if profile == "validate-device" {
        validate_overflow(&names);
        return;
    }
    if matches!(profile, "validate" | "validate-wide") {
        validate(profile == "validate-wide", a, &names);
        return;
    }
    assert!(matches!(profile, "target" | "census"));
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
    let (compact, affine, chains, compressed) = engines(&cfg, cutoff, entry);
    let engine_setup_seconds = started.elapsed().as_secs_f64();
    let baseline = Rules::baseline();
    let sampler = Sampler::new(&baseline, CHECKPOINT, None);
    let descendants = BackwardDp::run(ParityRules::of(&baseline), 2, cfg.m(), None);
    let baseline_endpoint_population = descendants.ways(PState::of(&Node::root(&baseline)));
    println!(
        "{}",
        json!({
            "type":"config","experiment":"fullrange","profile":profile,"n":cfg.n,"a":a,"m":cfg.m(),
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
            "chained_table_bytes":chains.iter().map(Chained::table_bytes).collect::<Vec<_>>(),
            "compact_engines":compressed.chains.iter().map(|c| json!({
                "name":c.name(),"host_table_bytes":c.table_bytes(),
                "programs":c.tables().programs.len(),"leaves":c.tables().leaves.len(),
                "words":c.tables().words.len(),
                "device_table_bytes":c.tables().programs.len()*32+c.tables().leaves.len()*8+c.tables().words.len()*32
            })).collect::<Vec<_>>(),
            "early_parent_engine":compressed.early.name(),
            "early_parent_table_bytes":compressed.early.table_bytes(),
            "early_parent_word_count":compressed.early.table_words().len(),
            "early_parent_device_table_bytes":compressed.early.table_words().len()*32,
            "timed_scope":if profile == "census" { "none; untimed CPU arithmetic-sharing census" } else { "live expansion after K38, packing, queues, transfers, device expansion, verification and fallback; resident arrays and suffix tables" },
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
    if profile == "census" {
        chain_census(&cohorts, &chains, &cfg.rules, &names);
        return;
    }
    let started = Instant::now();
    let cpu = cpu_seconds();
    let mut gpu = GpuSession::new_experimental(&cfg).expect("GPU initialization");
    gpu.set_terminal_table(affine.table_words())
        .expect("table upload");
    prepare_jumps(&mut gpu, &names);
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
        prepare_variant(&mut gpu, &affine, &compressed, name);
        let report = run_variant(&mut gpu, &cfg, &compact, &affine, &chains, &compressed, name);
        require_resolved(&report);
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
                let preparation = Instant::now();
                prepare_variant(&mut gpu, &affine, &compressed, name);
                let variant_prepare_seconds = preparation.elapsed().as_secs_f64();
                let cpu = cpu_seconds();
                let started = Instant::now();
                let report = run_variant(&mut gpu, &cfg, &compact, &affine, &chains, &compressed, name);
                let wall = started.elapsed().as_secs_f64();
                let cpu = cpu_seconds() - cpu;
                require_resolved(&report);
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
                    "wall_seconds":wall,"process_cpu_seconds":cpu,"variant_prepare_seconds":variant_prepare_seconds,
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
        let combined_signature = aggregate_signature(&expected, cfg.m(), cfg.num_bv);
        let mut aggregate_expected = None;
        for repetition in 0..aggregate_repetitions {
            for order_position in 0..names.len() {
                let name = names[(order_position + repetition) % names.len()];
                let preparation = Instant::now();
                prepare_variant(&mut gpu, &affine, &compressed, name);
                let variant_prepare_seconds = preparation.elapsed().as_secs_f64();
                let cpu = cpu_seconds();
                let started = Instant::now();
                let report = run_variant(&mut gpu, &cfg, &compact, &affine, &chains, &compressed, name);
                let wall = started.elapsed().as_secs_f64();
                let cpu = cpu_seconds() - cpu;
                require_resolved(&report);
                if let Some(want) = &aggregate_expected {
                    assert_eq!(*want, signature(&report), "aggregate {name}");
                } else {
                    aggregate_expected = Some(signature(&report));
                }
                // Every tree/per-class counter, XOR hash, flagged-origin
                // multiset and recheck total must compose across cohorts.
                assert_eq!(
                    signature(&report),
                    combined_signature,
                    "aggregate work and flags"
                );
                println!(
                    "{}",
                    json!({"type":"aggregate_trial","variant":name,
                    "repetition":repetition,"order_position":order_position,
                    "drawn":drawn*count,"accepted":cfg.units.as_ref().unwrap().len(),
                    "wall_seconds":wall,"process_cpu_seconds":cpu,"variant_prepare_seconds":variant_prepare_seconds,
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
    fn aggregate_signatures_preserve_wide_flag_multiplicity_and_rechecks() {
        let mut tree = TreeStats::new(10);
        tree.generated[10] = 7;
        let mut tail = TailStats::new(2);
        tail.endpoints[0] = 3;
        tail.cases[0] = 11;
        tail.iters[0] = 19;
        tail.hash = 0x1234_5678;
        let wide = (1u128 << 100) + 123;
        tail.flagged = vec![wide, 7, wide];
        let part = json!({"tree":tree,"tail":tail_signature(&tail),
            "hash":format!("{:08X}", tail.hash),
            "flagged":normalized_flags(tail.flagged.iter().copied()),
            "flagged_rechecked":3});
        tail.flagged.reverse();
        assert_eq!(tail_signature(&tail), part["tail"]);
        let result = aggregate_signature(&[Some(part.clone()), Some(part)], 10, 2);
        let flags = json!(vec![
            "7".to_string(),
            "7".to_string(),
            wide.to_string(),
            wide.to_string(),
            wide.to_string(),
            wide.to_string()
        ]);
        assert_eq!(result["flagged"], flags);
        assert_eq!(result["tail"]["flagged"], flags);
        assert_eq!(result["flagged_rechecked"], 6);
        assert_eq!(result["tail"]["endpoints"][0], 6);
        assert_eq!(result["tail"]["cases"][0], 22);
        assert_eq!(result["tail"]["iters"][0], 38);
        assert_eq!(result["tree"]["generated"][10], 14);
        assert_eq!(result["hash"], "00000000");
        assert_eq!(result["tail"]["hash"], 0);
    }

    #[test]
    fn resolved_flagged_reports_are_accepted_and_unresolved_reports_rejected() {
        let cfg = config(32, 10, 12, 2);
        let mut tail = TailStats::new(2);
        tail.flagged = vec![7, 3, 7];
        let mut report = VerifyReport::new(&cfg, "cpu", 0, TreeStats::new(cfg.m()), tail, 0.0);
        assert_eq!(report.flagged_rechecked, 3);
        require_resolved(&report);
        let expected = signature(&report);
        report.flagged.reverse();
        report.tail.flagged.rotate_left(1);
        assert_eq!(signature(&report), expected);
        report.flagged_rechecked -= 1;
        assert!(std::panic::catch_unwind(|| require_resolved(&report)).is_err());
    }

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
