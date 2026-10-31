//! Transport and reusable terminal-program experiments on the live GPU pipeline.
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
use cv::expansion::Expander;
use cv::gpu::{GpuSession, GpuTransport};
use cv::lookahead::{BvMode, WindowRules};
use cv::rules::{walkback_certificate, Rules};
use cv::sample::{Rng, Sampler};
use cv::tail::TailParams;
use cv::tree::{Node, Sink, TreeStats};
use cv::verify::{verify_cpu_with_expander, VerifyConfig, VerifyReport};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::time::Instant;
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
    assert_eq!(
        unsafe { clock_gettime(2, &mut ts) },
        0,
        "process CPU clock unavailable"
    );
    ts.seconds as f64 + ts.nanoseconds as f64 * 1e-9
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

// Compare actual work and every tree counter; wall times/backend labels differ.
fn signature(r: &VerifyReport) -> Value {
    json!({"tree":r.tree, "tail":r.tail, "hash":r.hash,
        "flagged":r.flagged, "flagged_rechecked":r.flagged_rechecked})
}

fn corpus(count: usize, seed: u64, rules: &Rules) -> (Vec<Node>, Value) {
    let timer = Instant::now();
    let sampler = Sampler::new(&Rules::baseline(), 38, None);
    let population = sampler.population();
    let mut rng = Rng(seed);
    let mut roots = Vec::with_capacity(count);
    let mut seen = HashSet::new();
    let (mut draws, mut cheap_rejected, mut walk_rejected, mut duplicates) =
        (0u64, 0u64, 0u64, 0u64);
    let mut walk_seconds = 0.0;
    while roots.len() < count {
        draws += 1;
        let d = sampler.draw(&mut rng, Some(rules));
        if d.culled.is_some() {
            cheap_rejected += 1;
            continue;
        }
        let t = Instant::now();
        let rejected =
            walkback_certificate(d.node.n0, d.node.k, d.node.f, d.node.m, 65_536).is_some();
        walk_seconds += t.elapsed().as_secs_f64();
        if rejected {
            walk_rejected += 1;
            continue;
        }
        if !seen.insert(d.node.n0) {
            duplicates += 1;
            continue;
        }
        roots.push(d.node);
    }
    let meta = json!({"seconds":timer.elapsed().as_secs_f64(), "walk_seconds":walk_seconds,
        "baseline_population":population.to_string(), "draws":draws,"cheap_rejected":cheap_rejected,
        "walk_rejected":walk_rejected,"duplicates":duplicates,"walkback_cap":65_536,
        "selection":"Unique uniform baseline K38 draws, rejecting improved13 ancestry and bounded K38 walkback; identical saved roots in all variants"});
    (roots, meta)
}

fn run_variant(
    gpu: &mut GpuSession,
    cfg: &VerifyConfig,
    compact: &Combined,
    affine: &Affine,
    name: &str,
) -> VerifyReport {
    let words = affine.table_words();
    let result = match name {
        "replay16" => gpu.run_transport(cfg, compact, GpuTransport::Replay16, words),
        "packed8" => gpu.run_transport(cfg, compact, GpuTransport::Packed8, words),
        "affine16" => gpu.run_transport(cfg, affine, GpuTransport::Replay16, words),
        "affine32" => gpu.run_transport(cfg, affine, GpuTransport::Affine32, words),
        "parents" => gpu.run_transport(cfg, affine, GpuTransport::Parents, words),
        _ => panic!("unknown transport variant {name}"),
    };
    result.expect("GPU transport run")
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

fn validate(wide: bool, a: u32, split: bool) {
    let mut cfg = config(if wide { 72 } else { 32 }, a, 12, 2);
    if split {
        cfg.bv_mode = BvMode::Split;
    }
    cfg.class_b = Some(vec![12, 13]);
    let (cutoff, entry) = if wide { (50, 52) } else { (16, 18) };
    if wide {
        let sampler = Sampler::new(&Rules::baseline(), entry, None);
        let mut rng = Rng(31_419);
        let mut roots = Vec::new();
        let mut seen = HashSet::new();
        while roots.len() < 16 {
            let draw = sampler.draw(&mut rng, Some(&cfg.rules));
            if draw.culled.is_none() && seen.insert(draw.node.n0) {
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
    let reference = verify_cpu_with_expander(&cfg, &compact);
    let affine_cpu = verify_cpu_with_expander(&cfg, &affine);
    assert_eq!(
        signature(&reference),
        signature(&affine_cpu),
        "affine CPU mismatch"
    );
    struct Coverage {
        top_bit: u128,
        top: u64,
        wide: u64,
        gpu: u64,
        fallback: u64,
        offset: u32,
    }
    impl Sink for Coverage {
        fn endpoint(&mut self, nd: &Node) {
            if nd.f >= self.offset && nd.f < self.offset + 2 {
                self.gpu += 1;
                self.top += u64::from(nd.n0 & self.top_bit != 0);
                self.wide += u64::from(nd.m > u64::MAX as u128);
            } else {
                self.fallback += 1;
            }
        }
    }
    let mut coverage = Coverage {
        top_bit: 1u128 << (cfg.m() - 1),
        top: 0,
        wide: 0,
        gpu: 0,
        fallback: 0,
        offset: cfg.m() * cv::arith::NUM / cv::arith::DEN + 1,
    };
    if wide {
        let mut stats = TreeStats::new(cfg.m());
        for root in cfg.units.as_ref().unwrap() {
            affine.expand::<_, true>(root, cfg.m(), &cfg.rules, &mut coverage, &mut stats);
        }
        assert!(coverage.top > 0 && coverage.wide > 0 && coverage.fallback > 0);
    }
    let mut gpu = GpuSession::new_experimental(&cfg).expect("GPU initialization");
    gpu.set_terminal_table(affine.table_words())
        .expect("table upload");
    for name in ["replay16", "packed8", "affine16", "affine32", "parents"] {
        if name == "packed8" && (cfg.m() > 62 || split) {
            continue;
        }
        let report = run_variant(&mut gpu, &cfg, &compact, &affine, name);
        if wide {
            assert_eq!(
                report.tail.endpoints[..cfg.num_bv].iter().sum::<u64>(),
                coverage.gpu
            );
            assert_eq!(report.tail.endpoints[cfg.num_bv], coverage.fallback);
        }
        assert_eq!(
            signature(&reference),
            signature(&report),
            "CPU/GPU mismatch {name}"
        );
        assert!(
            *report.tail.endpoints.last().unwrap() > 0,
            "validation must cover fallback"
        );
        assert!(report.flagged.is_empty());
        println!(
            "{}",
            json!({"type":"validation","variant":name,"n":cfg.n,"a":a,"m":cfg.m(),
            "scope":if wide {"sixteen sampled K52 packages"}else{"complete N32"},
            "cpu_gpu_equal":true,"same_policy_equal":true,"roots":cfg.units,
            "coverage":{"gpu_endpoints":coverage.gpu,"top_bit_endpoints":coverage.top,
            "wide_iterate_endpoints":coverage.wide,"fallback_endpoints":coverage.fallback},"report":report})
        );
    }
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let profile = args.first().map(String::as_str).unwrap_or("validate");
    let number =
        |i: usize, default: usize| args.get(i).map(|s| s.parse().unwrap()).unwrap_or(default);
    let threads = number(3, 8);
    let a = number(7, 10) as u32;
    assert!((8..=10).contains(&a));
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    if matches!(profile, "validate" | "validate-device" | "validate-wide") {
        validate(
            profile == "validate-wide",
            a,
            args.get(5).is_some_and(|s| s == "split"),
        );
        return;
    }
    assert_eq!(profile, "target");
    let (nroots, reps, seed) = (number(1, 1024), number(2, 3), number(4, 20261023) as u64);
    let names: Vec<_> = args
        .get(5)
        .map(String::as_str)
        .unwrap_or("replay16,packed8,affine16,affine32,parents")
        .split(',')
        .collect();
    assert_eq!(names.iter().collect::<HashSet<_>>().len(), names.len());
    for name in &names {
        assert!(["replay16", "packed8", "affine16", "affine32", "parents"].contains(name));
    }
    assert!(nroots > 0 && reps > 0 && threads > 0);
    let mut cfg = config(72, a, 27, 16);
    if names.contains(&"packed8") {
        assert!(cfg.m() <= 62);
    }
    cfg.class_b = Some(
        args.get(6)
            .map(String::as_str)
            .unwrap_or("32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27")
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect(),
    );
    assert_eq!(cfg.class_b.as_ref().unwrap().len(), 16);
    let t = Instant::now();
    let (compact, affine) = engines(&cfg, 50, 52);
    let engine_seconds = t.elapsed().as_secs_f64();
    let (roots, sampling) = corpus(nroots, seed, &cfg.rules);
    if a == 10 {
        let t = Instant::now();
        let probe = affine.class_probe(&roots, &cfg.rules);
        println!(
            "{}",
            json!({"type":"class_probe","wall_seconds":t.elapsed().as_secs_f64(),"probe":probe})
        );
    }
    println!(
        "{}",
        json!({"type":"config","experiment":"transport","n":72,"a":a,"m":cfg.m(),
        "checkpoint":38,"cutoff":50,"terminal_entry":52,"terminal_width":cfg.m()-52,
        "roots":roots,"seed":seed,"sampling":sampling,"repetitions":reps,"threads":threads,
        "num_bv":cfg.num_bv,"class_b":cfg.class_b,"window":cfg.window.label(),"rules":cfg.rules.label(),
        "variants":names,"engine_setup_seconds":engine_seconds,"terminal_table_bytes":affine.table_bytes(),
        "terminal_word_count":affine.table_words().len(),
        "timed_scope":"live expansion, packing, queues, transfers, device expansion, verification and fallback; resident arrays and suffix tables",
        "kernel_seconds_meaning":"overlapping CUDA event sums, not GPU busy wall time"})
    );
    let t = Instant::now();
    let cpu = cpu_seconds();
    let mut gpu = GpuSession::new_experimental(&cfg).expect("GPU initialization");
    gpu.set_terminal_table(affine.table_words())
        .expect("terminal table upload");
    println!(
        "{}",
        json!({"type":"gpu_setup","wall_seconds":t.elapsed().as_secs_f64(),
        "process_cpu_seconds":cpu_seconds()-cpu})
    );
    cfg.units = Some(roots[..nroots.min(32)].to_vec());
    let mut warmup_signature = None;
    for name in &names {
        let report = run_variant(&mut gpu, &cfg, &compact, &affine, name);
        assert!(report.flagged.is_empty());
        if let Some(want) = &warmup_signature {
            assert_eq!(*want, signature(&report), "warmup work mismatch {name}");
        } else {
            warmup_signature = Some(signature(&report));
        }
        println!(
            "{}",
            json!({"type":"warmup","variant":name,"report":report})
        );
    }
    cfg.units = Some(roots);
    let mut expected = None;
    for rep in 0..reps {
        for pos in 0..names.len() {
            let name = names[(pos + rep) % names.len()];
            let cpu = cpu_seconds();
            let t = Instant::now();
            let report = run_variant(&mut gpu, &cfg, &compact, &affine, name);
            let wall = t.elapsed().as_secs_f64();
            let cpu = cpu_seconds() - cpu;
            assert!(report.flagged.is_empty());
            if let Some(want) = &expected {
                assert_eq!(*want, signature(&report), "work mismatch {name}");
            } else {
                expected = Some(signature(&report));
            }
            println!(
                "{}",
                json!({"type":"trial","variant":name,"repetition":rep,"order_position":pos,
                "wall_seconds":wall,"process_cpu_seconds":cpu,"average_busy_cpu_cores":cpu/wall,"report":report})
            );
        }
    }
}
