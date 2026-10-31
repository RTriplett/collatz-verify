//! Scalar CPU timing allowance for rare N72/A10 endpoints beyond GPU arrays.
//!
//! rare_tail_bench [draws=4096] [cohorts=16] [repetitions=2] [seed=20261030]
//! Run under taskset -c 0 to pin the complete process to one physical core.
//! Samples every baseline F56..62 endpoint with equal probability, retaining
//! duplicates and deliberately ignoring improved ancestry/checkpoint screening.
//! No GPU initialization or lookahead-array construction is performed.

use cv::dp::ForwardDp;
use cv::lookahead::{BvMode, Lookahead, WindowRules};
use cv::rules::Rules;
use cv::sample::{Rng, Sampler};
use cv::tail::{endpoint_work, recheck_wide, TailParams};
use cv::tree::Node;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::time::Instant;

const N: u32 = 72;
const A: u32 = 10;
const M: u32 = N - A;
const F_MIN: u32 = 56;
const F_MAX: u32 = 62;
const CLASSES: usize = (F_MAX - F_MIN + 1) as usize;
const POPULATION: u128 = 49_063_601;

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

#[derive(Clone, Debug, PartialEq, Eq)]
struct Work {
    endpoints: [u64; CLASSES],
    cases: [u64; CLASSES],
    iters: [u64; CLASSES],
    hash: u32,
    flagged: Vec<u128>,
    rechecked: u64,
    recheck_steps: u64,
}

impl Work {
    fn new() -> Self {
        Self {
            endpoints: [0; CLASSES],
            cases: [0; CLASSES],
            iters: [0; CLASSES],
            hash: 0,
            flagged: Vec::new(),
            rechecked: 0,
            recheck_steps: 0,
        }
    }

    fn json(&self) -> Value {
        json!({
            "endpoints":self.endpoints.iter().sum::<u64>(),
            "cases":self.cases.iter().sum::<u64>(),
            "iters":self.iters.iter().sum::<u64>(),
            "hash":format!("{:08X}",self.hash),
            "by_f":(F_MIN..=F_MAX).enumerate().map(|(i,f)| json!({
                "f":f,"endpoints":self.endpoints[i],"cases":self.cases[i],
                "iters":self.iters[i]})).collect::<Vec<_>>(),
            "flagged":self.flagged.iter().map(u128::to_string).collect::<Vec<_>>(),
            "flagged_rechecked":self.rechecked,"unresolved_flags":0,
            "wide_recheck_steps":self.recheck_steps,
        })
    }
}

struct Measured {
    work: Work,
    timing: Value,
}

fn run(nodes: &[Node], lookahead: &Lookahead, tail: &TailParams) -> Measured {
    let cpu_started = cpu_seconds();
    let started = Instant::now();
    let mut work = Work::new();
    for node in nodes {
        let class = (node.f - F_MIN) as usize;
        let flags_before = work.flagged.len();
        let result = endpoint_work(lookahead, tail, node, |n| work.flagged.push(n));
        assert_eq!(work.flagged.len() - flags_before, result.flagged as usize);
        work.endpoints[class] += 1;
        work.cases[class] += result.cases;
        work.iters[class] += result.iters;
        work.hash ^= result.hash;
    }
    let tail_wall_seconds = started.elapsed().as_secs_f64();
    let tail_cpu_seconds = cpu_seconds() - cpu_started;
    let recheck_started = Instant::now();
    let recheck_cpu_started = cpu_seconds();
    for &n in &work.flagged {
        let steps = recheck_wide(n).expect("unresolved rare-tail overflow");
        work.rechecked += 1;
        work.recheck_steps += steps;
    }
    let wide_recheck_wall_seconds = recheck_started.elapsed().as_secs_f64();
    let wide_recheck_cpu_seconds = cpu_seconds() - recheck_cpu_started;
    let wall_seconds = started.elapsed().as_secs_f64();
    let process_cpu_seconds = cpu_seconds() - cpu_started;
    assert_eq!(work.flagged.len() as u64, work.rechecked);
    // Ordering normalization and JSON formatting are outside the timed scope.
    work.flagged.sort_unstable();
    Measured {
        work,
        timing: json!({"wall_seconds":wall_seconds,
            "process_cpu_seconds":process_cpu_seconds,
            "tail_wall_seconds":tail_wall_seconds,"tail_process_cpu_seconds":tail_cpu_seconds,
            "wide_recheck_wall_seconds":wide_recheck_wall_seconds,
            "wide_recheck_process_cpu_seconds":wide_recheck_cpu_seconds,
            "average_busy_cpu_cores":process_cpu_seconds/wall_seconds}),
    }
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let number = |i: usize, default: usize| {
        args.get(i)
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(default)
    };
    let draws = number(0, 4096);
    let count = number(1, 16);
    let repetitions = number(2, 2);
    let seed = number(3, 20261030) as u64;
    assert!(draws > 0 && count >= 2 && repetitions > 0);
    let setup_cpu = cpu_seconds();
    let setup_started = Instant::now();
    let baseline = Rules::baseline();
    let sampler = Sampler::new(&baseline, M, Some((F_MIN, F_MAX)));
    let forward = ForwardDp::from_root(&baseline, M);
    let populations = forward.layer(M).by_f();
    let forward_population: u128 = populations[F_MIN as usize..=F_MAX as usize].iter().sum();
    assert_eq!(sampler.population(), POPULATION);
    assert_eq!(forward_population, POPULATION);
    let lookahead = Lookahead::meta(M, A, 27, 16, BvMode::Shared, &WindowRules::default());
    assert!(!lookahead.has_arrays());
    assert!((F_MIN..=F_MAX).all(|f| lookahead.bigindex(f).is_none()));
    let tail = TailParams::default();
    let setup_seconds = setup_started.elapsed().as_secs_f64();
    let setup_process_cpu_seconds = cpu_seconds() - setup_cpu;
    println!(
        "{}",
        json!({"type":"config","experiment":"rare_cpu_tail",
            "n":N,"a":A,"m":M,"f_min":F_MIN,"f_max":F_MAX,
            "baseline_population":POPULATION.to_string(),
            "baseline_population_by_f":(F_MIN..=F_MAX).map(|f| json!({
                "f":f,"endpoints":populations[f as usize].to_string()})).collect::<Vec<_>>(),
            "baseline_population_forward_backward_equal":true,
            "sampling_design":"fixed_uniform_baseline_endpoints_with_replacement",
            "draws_per_cohort":draws,"cohort_count":count,"repetitions":repetitions,"seed":seed,
            "threads":1,"rules":baseline.label(),"num_bv":16,
            "cutoff_bits":tail.cutoff_bits,"tail_oee":tail.tail_oee,
            "lookahead_arrays_built":false,"gpu_initialized":false,
            "setup_seconds":setup_seconds,"setup_process_cpu_seconds":setup_process_cpu_seconds,
            "timed_scope":"scalar endpoint_work fallback plus every required 256-bit recheck; includes work counters and collecting flagged starts",
            "excluded_scope":"sampling, endpoint expansion, checkpoint screening, lookahead construction, queueing, storage and GPU work",
            "policy":"retain all baseline F56..62 endpoints, without improved13 or K38 walkback screening; conservative CPU workload allowance, not a correction to measured GPU cohorts",
            "affinity_instruction":"run the entire process under taskset on one physical core; record affinity and host metadata externally",
        })
    );
    let mut cohorts = Vec::with_capacity(count);
    let mut seeder = Rng(seed);
    for index in 0..count {
        let cohort_seed = seeder.next_u64();
        let cpu = cpu_seconds();
        let started = Instant::now();
        let mut rng = Rng(cohort_seed);
        let nodes: Vec<_> = (0..draws)
            .map(|_| sampler.draw(&mut rng, None).node)
            .collect();
        assert!(nodes.iter().all(|n| n.k == M
            && (F_MIN..=F_MAX).contains(&n.f)
            && n.cull(&baseline).is_none()
            && lookahead.array(n.f, n.ne).is_none()));
        let unique = nodes.iter().map(|n| n.n0).collect::<HashSet<_>>().len();
        println!(
            "{}",
            json!({"type":"cohort","cohort":index,"seed":cohort_seed,
            "drawn":draws,"accepted":draws,"duplicates_retained":draws-unique,
            "sampling_seconds":started.elapsed().as_secs_f64(),
            "sampling_process_cpu_seconds":cpu_seconds()-cpu,"roots":nodes})
        );
        cohorts.push(nodes);
    }
    let warmup = run(&cohorts[0][..draws.min(64)], &lookahead, &tail);
    println!(
        "{}",
        json!({"type":"warmup","timing":warmup.timing,"work":warmup.work.json()})
    );
    let mut expected: Vec<Option<Work>> = vec![None; count];
    for repetition in 0..repetitions {
        for position in 0..count {
            let index = (position + repetition) % count;
            let measured = run(&cohorts[index], &lookahead, &tail);
            if let Some(want) = &expected[index] {
                assert_eq!(want, &measured.work, "rare-tail cohort {index} differs");
            } else {
                expected[index] = Some(measured.work.clone());
            }
            println!(
                "{}",
                json!({"type":"trial","cohort":index,"repetition":repetition,
                "order_position":position,"drawn":draws,"timing":measured.timing,
                "work":measured.work.json()})
            );
        }
    }
}
