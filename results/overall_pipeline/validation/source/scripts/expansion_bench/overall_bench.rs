//! Actual queued CPU→GPU pipeline benchmark, with resident lookahead arrays.
//! overall_bench validate | target [roots reps threads seed variants]
//! Linux process CPU clock includes every producer and the GPU worker.
#[allow(dead_code)]
mod combined;
#[allow(dead_code)]
mod fast_tail;
#[allow(dead_code)]
mod terminal;
#[allow(dead_code)]
mod terminal_walk;
#[allow(dead_code)]
mod ternary;
#[cfg(test)]
#[allow(dead_code)]
mod tail_policy;

use combined::{Combined, Mode};
use cv::gpu::GpuSession;
use cv::lookahead::{BvMode, WindowRules};
use cv::rules::{walkback_certificate, Rules};
use cv::sample::{Rng, Sampler};
use cv::tail::TailParams;
use cv::tree::Node;
use cv::verify::{verify_cpu_with_expander, VerifyConfig, VerifyReport};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

fn cpu_seconds() -> f64 {
    #[repr(C)]
    struct Timespec { seconds: std::os::raw::c_long, nanoseconds: std::os::raw::c_long }
    unsafe extern "C" { fn clock_gettime(clock: i32, time: *mut Timespec) -> i32; }
    let mut ts = Timespec { seconds: 0, nanoseconds: 0 };
    assert_eq!(unsafe { clock_gettime(2, &mut ts) }, 0, "process CPU clock unavailable");
    ts.seconds as f64 + ts.nanoseconds as f64 * 1e-9
}

fn config(n: u32, a: u32, b: u32, num_bv: usize) -> VerifyConfig {
    let rules = Rules::improved(13);
    VerifyConfig {
        n, a, b, num_bv, bv_mode: BvMode::Shared,
        window: WindowRules { pm_inf: true, mix: rules.mix.clone() },
        class_b: None, split: 10, rules,
        tail: TailParams::default(), units: None,
    }
}

// Compare actual work and every tree counter; wall times/backend labels differ.
fn signature(r: &VerifyReport) -> Value {
    json!({"tree":r.tree, "tail":r.tail, "hash":r.hash,
        "flagged":r.flagged, "flagged_rechecked":r.flagged_rechecked})
}

fn validate() {
    // Two classes deliberately force real CPU fallbacks from compact payloads.
    let mut cfg = config(32, 10, 12, 2);
    cfg.rules.walkback = Some(cv::rules::Walkback { depth: 12, max_states: 256 });
    let mut gpu = GpuSession::new(&cfg).expect("GPU initialization");
    let modes = [Mode::Full, Mode::Screened, Mode::ReferenceCutoff(16),
        Mode::CompactCutoff(16), Mode::Terminal { cutoff: 16, width: 4 },
        Mode::Terminal { cutoff: 16, width: 6 }];
    let mut full = None;
    let mut cutoff = None;
    for mode in modes {
        let engine = Combined::new(cfg.m(), &cfg.rules, mode);
        let cpu = verify_cpu_with_expander(&cfg, &engine);
        let device = gpu.run(&cfg, &engine).expect("GPU validation");
        assert_eq!(signature(&cpu), signature(&device), "CPU/GPU mismatch {}", engine.name());
        assert!(*device.tail.endpoints.last().unwrap() > 0, "fallback not exercised");
        assert!(device.flagged.is_empty());
        let expected = if matches!(mode, Mode::Full | Mode::Screened) { &mut full } else { &mut cutoff };
        if let Some(want) = expected {
            assert_eq!(*want, signature(&device), "same-policy mismatch {}", engine.name());
        } else { *expected = Some(signature(&device)); }
        println!("{}", json!({"type":"validation", "variant":engine.name(),
            "complete_range_bits":cfg.n, "cpu_gpu_equal":true,
            "same_policy_equal":true,"report":device}));
    }
}

fn corpus(count: usize, seed: u64, rules: &Rules) -> (Vec<Node>, Value) {
    let timer = Instant::now();
    let sampler = Sampler::new(&Rules::baseline(), 38, None);
    let population = sampler.population();
    let mut rng = Rng(seed);
    let mut roots = Vec::with_capacity(count);
    let mut seen = HashSet::new();
    let (mut draws, mut cheap_rejected, mut walk_rejected, mut duplicates) = (0u64,0u64,0u64,0u64);
    let mut walk_seconds = 0.0;
    while roots.len() < count {
        draws += 1;
        let d = sampler.draw(&mut rng, Some(rules));
        if d.culled.is_some() { cheap_rejected += 1; continue; }
        let t = Instant::now();
        let rejected = walkback_certificate(d.node.n0, d.node.k, d.node.f, d.node.m, 65_536).is_some();
        walk_seconds += t.elapsed().as_secs_f64();
        if rejected { walk_rejected += 1; continue; }
        if !seen.insert(d.node.n0) { duplicates += 1; continue; }
        roots.push(d.node);
    }
    let meta = json!({"seconds":timer.elapsed().as_secs_f64(), "walk_seconds":walk_seconds,
        "baseline_population":population.to_string(), "draws":draws,"cheap_rejected":cheap_rejected,
        "walk_rejected":walk_rejected,"duplicates":duplicates,"walkback_cap":65_536,
        "selection":"Unique uniform baseline K38 draws, rejecting improved13 ancestry and bounded K38 walkback; identical saved roots in all variants"});
    (roots,meta)
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let profile = args.first().map(String::as_str).unwrap_or("validate");
    let number = |i: usize, default: usize| args.get(i).map(|x| x.parse().unwrap()).unwrap_or(default);
    let threads = number(3, 8);
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    if profile == "validate" { validate(); return; }
    assert_eq!(profile, "target", "profile must be validate or target");
    let (nroots, reps, seed) = (number(1, 1024), number(2, 3), number(4, 20261023) as u64);
    assert!(nroots > 0 && reps > 0 && threads > 0);
    let selected = args.get(5).map(String::as_str).unwrap_or("full,screened,compact54,combined54,compact50,combined50");
    let available = [ ("full", Mode::Full), ("screened", Mode::Screened),
        ("reference54", Mode::ReferenceCutoff(54)), ("compact54", Mode::CompactCutoff(54)),
        ("combined54", Mode::Terminal { cutoff:54, width:8 }),
        ("reference50", Mode::ReferenceCutoff(50)), ("compact50", Mode::CompactCutoff(50)),
        ("combined50", Mode::Terminal { cutoff:50, width:10 }) ];
    let selected: Vec<_> = selected.split(',').collect();
    assert_eq!(selected.iter().collect::<HashSet<_>>().len(), selected.len(), "duplicate variant");
    let mut cfg = config(72, 10, 27, 16);
    cfg.class_b = Some(vec![32,35,35,35,35,35,32,32,27,27,27,27,27,27,27,27]);
    let setup = Instant::now();
    let variants: Vec<_> = selected.into_iter().map(|name| {
        let mode = available.iter().find(|(n,_)| *n == name).expect("unknown variant").1;
        (name, mode, Combined::new(cfg.m(), &cfg.rules, mode))
    }).collect();
    let engine_seconds = setup.elapsed().as_secs_f64();
    let (roots, sampling) = corpus(nroots, seed, &cfg.rules);
    println!("{}",json!({"type":"config", "n":72,"a":10,"m":62,"checkpoint":38,
        "num_bv":cfg.num_bv,"class_b":cfg.class_b,"window":cfg.window.label(),
        "rules":cfg.rules.label(),"roots":roots,"seed":seed,"repetitions":reps,"threads":threads,
        "sampling":sampling,"engine_setup_seconds":engine_seconds,
        "variants":variants.iter().map(|(name,_,e)|json!({"name":name,"engine":e.name(),"table_bytes":e.table_bytes()})).collect::<Vec<_>>(),
        "timed_scope":"Live production packing, queues, transfers, GPU kernels and CPU fallback; resident arrays; fixed sampled K38 work packages, not complete 2^72 verification",
        "excluded_from_trials":["checkpoint generation and filtering","archive IO","table and GPU array setup"],
        "kernel_seconds_meaning":"sum of CUDA event durations across overlapping streams, not GPU busy wall"}));
    cfg.units = Some(roots.clone());
    let setup = Instant::now();
    let setup_cpu = cpu_seconds();
    let mut gpu = GpuSession::new(&cfg).expect("GPU initialization");
    println!("{}",json!({"type":"gpu_setup","wall_seconds":setup.elapsed().as_secs_f64(),
        "process_cpu_seconds":cpu_seconds()-setup_cpu}));
    // Warm every code path on the same small prefix; exclude these records from summaries.
    cfg.units = Some(roots[..nroots.min(32)].to_vec());
    for (name,_,engine) in &variants {
        let report = gpu.run(&cfg,engine).expect("GPU warmup");
        assert!(report.flagged.is_empty(),"flagged trajectories in warmup");
        println!("{}",json!({"type":"warmup","variant":name,"report":report}));
    }
    cfg.units = Some(roots);
    let mut expected = BTreeMap::<String,Value>::new();
    let mut timings = vec![Vec::new();variants.len()];
    let mut cpus = timings.clone();
    for rep in 0..reps {
        for pos in 0..variants.len() {
            let i = (pos+rep)%variants.len();
            let (name,mode,engine) = &variants[i];
            let cpu = cpu_seconds();
            let wall = Instant::now();
            let report = gpu.run(&cfg,engine).expect("GPU trial");
            let seconds = wall.elapsed().as_secs_f64();
            let cpu = cpu_seconds()-cpu;
            assert!(report.flagged.is_empty(),"flagged trajectories in trial");
            let policy = match mode {
                Mode::Full | Mode::Screened => "full".to_string(),
                Mode::ReferenceCutoff(k) | Mode::CompactCutoff(k) | Mode::Terminal{cutoff:k,..} => format!("cutoff{k}"),
            };
            if let Some(want) = expected.get(&policy) {
                assert_eq!(*want,signature(&report),"same-policy/repetition mismatch for {name}");
            } else { expected.insert(policy,signature(&report)); }
            timings[i].push(seconds); cpus[i].push(cpu);
            println!("{}",json!({"type":"trial","variant":name,"repetition":rep,"order_position":pos,
                "wall_seconds":seconds,"process_cpu_seconds":cpu,"average_busy_cpu_cores":cpu/seconds,
                "report":report}));
        }
    }
    for (i,(name,_,_)) in variants.iter().enumerate() {
        timings[i].sort_by(f64::total_cmp); cpus[i].sort_by(f64::total_cmp);
        println!("{}",json!({"type":"summary","variant":name,
            "median_wall_seconds":timings[i][reps/2],"minimum_wall_seconds":timings[i][0],
            "maximum_wall_seconds":timings[i][reps-1],"median_process_cpu_seconds":cpus[i][reps/2]}));
    }
}
