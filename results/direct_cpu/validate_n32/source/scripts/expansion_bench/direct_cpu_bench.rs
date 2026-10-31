//! CPU microprobe for owning complete M62 source families before GPU dispatch.
//! Run independently of GPU timings. Resident arrays and suffix tables are setup.

use cv::arith::POW3;
use cv::direct_cpu::{DirectWork, Plan};
use cv::lookahead::WindowRules;
use cv::rules::Rules;
use cv::sample::{Rng, Sampler};
use cv::tail::{recheck_wide, step_until_below};
use cv::tree::{dfs, Node, Sink, TreeStats, P16};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

const N: u32 = 72;
const SOURCE_M: u32 = 62;

fn cpu_seconds() -> f64 {
    #[repr(C)]
    struct Timespec { seconds: std::os::raw::c_long, nanoseconds: std::os::raw::c_long }
    unsafe extern "C" { fn clock_gettime(clock: i32, time: *mut Timespec) -> i32; }
    let mut ts = Timespec { seconds: 0, nanoseconds: 0 };
    assert_eq!(unsafe { clock_gettime(3, &mut ts) }, 0);
    ts.seconds as f64 + ts.nanoseconds as f64 * 1e-9
}

fn csv(value: &str) -> Vec<u32> {
    let values: Vec<_> = value.split(',').map(|v| v.parse().unwrap()).collect();
    assert!(!values.is_empty() && values.iter().collect::<HashSet<_>>().len() == values.len());
    values
}

struct Cohort { f: u32, nodes: Vec<Node>, metadata: Value }

fn sample(f: u32, count: usize, seed: u64, rules: &Rules) -> Cohort {
    let started = Instant::now();
    let sampler = Sampler::new(&Rules::baseline(), SOURCE_M, Some((f, f)));
    assert!(sampler.population() > 0);
    let mut rng = Rng(seed ^ (f as u64).wrapping_mul(0xd1b5_4a32_d192_ed03));
    let mut nodes = Vec::with_capacity(count);
    let (mut draws, mut rejected) = (0u64, 0u64);
    while nodes.len() < count {
        let draw = sampler.draw(&mut rng, Some(rules));
        draws += 1;
        if draw.culled.is_some_and(|(k, _)| k <= 50) { rejected += 1; continue; }
        nodes.push(draw.node);
    }
    let mut fnv = 0xcbf2_9ce4_8422_2325u64;
    for node in &nodes {
        for byte in node.n0.to_le_bytes() { fnv = (fnv ^ byte as u64).wrapping_mul(0x100_0000_01b3); }
    }
    let unique = nodes.iter().map(|node| node.n0).collect::<HashSet<_>>().len();
    let metadata = json!({
        "source_f": f, "count": count, "draws": draws, "rejected_before_or_at_k50": rejected,
        "duplicates_retained": count - unique, "baseline_population": sampler.population().to_string(),
        "ordered_n0_fnv1a64_le128": format!("{fnv:016X}"),
        "source_n0": nodes.iter().map(|node| node.n0.to_string()).collect::<Vec<_>>(),
        "first_8_sources": &nodes[..nodes.len().min(8)],
        "sampling_wall_seconds": started.elapsed().as_secs_f64(),
    });
    Cohort { f, nodes, metadata }
}

fn work_json(work: &DirectWork) -> Value {
    let mut flags = work.flagged.clone();
    flags.sort_unstable();
    json!({
        "source_endpoints": work.source_endpoints, "endpoints": work.endpoints,
        "cases": work.cases, "iters": work.iters, "hash": format!("{:08X}", work.hash),
        "flagged": flags.iter().map(u128::to_string).collect::<Vec<_>>(),
    })
}

fn run(plan: &Plan, nodes: &[Node]) -> (DirectWork, Value) {
    let mut work = DirectWork::default();
    let cpu_start = cpu_seconds();
    let start = Instant::now();
    for node in nodes { plan.process_into(node, &mut work); }
    let kernel_wall = start.elapsed().as_secs_f64();
    let kernel_cpu = cpu_seconds() - cpu_start;
    let recheck_start = Instant::now();
    let mut recheck_steps = 0;
    for &flag in &work.flagged { recheck_steps += recheck_wide(flag).expect("unresolved wide flag"); }
    let recheck_wall = recheck_start.elapsed().as_secs_f64();
    let wall = start.elapsed().as_secs_f64();
    let cpu = cpu_seconds() - cpu_start;
    (work, json!({
        "wall_seconds": wall, "thread_cpu_seconds": cpu,
        "expansion_screen_tail_wall_seconds": kernel_wall,
        "expansion_screen_tail_thread_cpu_seconds": kernel_cpu,
        "wide_recheck_wall_seconds": recheck_wall, "wide_recheck_steps": recheck_steps,
    }))
}

#[derive(Default)]
struct Nodes(Vec<Node>);
impl Sink for Nodes { fn endpoint(&mut self, node: &Node) { self.0.push(*node); } }

/// Direct ordinary-step oracle for one original start's suffix. It uses exact
/// powers for the baseline inequalities, independent of compiled word emission.
fn exact_suffix(source: &Node, target: u32, q: u128) -> Option<Node> {
    let n0 = source.n0 + (q << source.k);
    let (mut x, mut f, mut oee, mut ne) =
        (source.m + POW3[source.f as usize] * q, source.f, source.oee, source.ne);
    for k in source.k + 1..=target {
        if x % 2 == 0 {
            if oee == 2 { return None; }
            x /= 2;
            oee *= 2;
            ne = if ne < 2 { ne + 1 } else { 5 - ne };
        } else {
            let arm = u8::from(POW3[f as usize] <= 1u128 << k);
            x = (3 * x + 1) / 2;
            f += 1;
            oee = (oee | arm) & 1;
            ne = 0;
        }
        if POW3[f as usize] <= 1u128 << k
            || (ne == 2 && POW3[(f - 1) as usize] <= 1u128 << (k - 1)) {
            return None;
        }
    }
    Some(Node { n0, k: target, f, m: x, m3: (x % P16 as u128) as u32, oee, ne })
}

fn audit(plan: &Plan, sources: &[Node]) -> Value {
    let mut classifications = [0u64; 5]; // prefix, mod9, OEE, B, verified
    let mut expected = DirectWork::default();
    let mut actual = DirectWork::default();
    let mut wide_steps = 0;
    for source in sources {
        let mut compiled = Vec::new();
        plan.for_each_endpoint(source, |node| compiled.push(*node));
        compiled.sort_unstable_by_key(|node| node.n0);
        let mut reference = Nodes::default();
        dfs::<_, false>(source, plan.m(), &Rules::baseline(), &mut reference, &mut TreeStats::default());
        reference.0.sort_unstable_by_key(|node| node.n0);
        assert_eq!(compiled, reference.0, "compiled suffix / DFS mismatch");
        let mut ledger = vec![None; 1 << (N - SOURCE_M)];
        for node in &compiled {
            assert_eq!(cv::arith::t_iter(node.n0, node.k), (node.m, node.f));
            for h in 0..1u128 << plan.a() {
                let base = node.n0 + (h << plan.m());
                let original_h = ((base - source.n0) >> SOURCE_M) as usize;
                assert!(ledger[original_h].is_none(), "duplicate source completion");
                ledger[original_h] = Some(*node);
            }
        }
        expected.source_endpoints += 1;
        expected.endpoints += compiled.len() as u64;
        for (h, descendant) in ledger.into_iter().enumerate() {
            let base = source.n0 + ((h as u128) << SOURCE_M);
            // A separate wide trajectory from the original integer establishes
            // resolution for every audited input, including every skipped input.
            wide_steps += recheck_wide(base).expect("unresolved original start");
            let q = h as u128 & ((1u128 << (plan.m() - SOURCE_M)) - 1);
            let independent = exact_suffix(source, plan.m(), q);
            assert_eq!(descendant, independent, "original-start ledger mismatch");
            let Some(node) = independent else { classifications[0] += 1; continue };
            if matches!(base % 9, 2 | 4 | 5 | 8) { classifications[1] += 1; continue; }
            let (x, f) = cv::arith::t_iter(base, plan.m());
            assert_eq!(f, node.f);
            if (node.oee == 2 && x % 2 == 0) || (node.oee == 1 && x % 4 == 0) {
                classifications[2] += 1;
                continue;
            }
            if plan.b().is_some() && !plan.lookahead().keeps(f, node.ne, (x + 1) as u64) {
                classifications[3] += 1;
                continue;
            }
            classifications[4] += 1;
            expected.cases += 1;
            if x >= 1u128 << 123 { expected.flagged.push(base); continue; }
            let (hash, iters, flagged) = step_until_below(base, x + 1, 1u128 << plan.cutoff_bits());
            expected.hash ^= hash;
            expected.iters += iters;
            if flagged { expected.flagged.push(base); }
        }
        plan.process_into(source, &mut actual);
    }
    expected.flagged.sort_unstable();
    actual.flagged.sort_unstable();
    assert_eq!(actual, expected, "exact-start oracle tail mismatch");
    assert_eq!(classifications.iter().sum::<u64>(), sources.len() as u64 * (1 << (N - SOURCE_M)));
    json!({
        "exact_equal": true, "sources": sources.len(),
        "original_starts": classifications.iter().sum::<u64>(),
        "prefix_culled": classifications[0], "mod9_culled": classifications[1],
        "oee_culled": classifications[2], "b_culled": classifications[3],
        "verified": classifications[4], "all_starts_independent_wide_steps": wide_steps,
        "work": work_json(&actual),
    })
}

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    if v.len() % 2 == 0 { (v[v.len()/2-1] + v[v.len()/2]) * 0.5 } else { v[v.len()/2] }
}

fn main() {
    let mut options = BTreeMap::new();
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(args.len() % 2, 0, "all options take one value");
    for pair in args.chunks_exact(2) {
        assert!(matches!(pair[0].as_str(), "--count" | "--repetitions" | "--seed" | "--classes"
            | "--a" | "--b" | "--setup-threads" | "--audit"));
        assert!(options.insert(pair[0].as_str(), pair[1].as_str()).is_none());
    }
    let opt = |key, default| *options.get(key).unwrap_or(&default);
    let count: usize = opt("--count", "2048").parse().unwrap();
    let repetitions: usize = opt("--repetitions", "5").parse().unwrap();
    let seed: u64 = opt("--seed", "20261103").parse().unwrap();
    let classes = csv(opt("--classes", "40,41,42"));
    let a_values = csv(opt("--a", "10,8,6,4,2,0"));
    let b_values = csv(opt("--b", "0,16,20"));
    let setup_threads: usize = opt("--setup-threads", "8").parse().unwrap();
    let audit_count: usize = opt("--audit", "4").parse().unwrap();
    assert!(count > 0 && repetitions > 0 && setup_threads > 0 && audit_count > 0);
    assert!(classes.iter().all(|&f| (40..=62).contains(&f)));
    rayon::ThreadPoolBuilder::new().num_threads(setup_threads).build_global().unwrap();
    let rules = Rules::improved(13);
    let window = WindowRules { pm_inf: true, mix: rules.mix.clone() };
    let cohorts: Vec<_> = classes.iter().map(|&f| sample(f, count, seed, &rules)).collect();
    println!("{}", json!({
        "type": "config", "experiment": "direct_cpu_complete_source_families",
        "n": N, "source_m": SOURCE_M, "source_a": N-SOURCE_M,
        "count_per_class": count, "repetitions": repetitions, "seed": seed,
        "classes": classes, "a_values": a_values, "b_values": b_values,
        "setup_threads": setup_threads, "timing_threads": 1, "audit_sources_per_class": audit_count,
        "sampling": "Uniform conditional baseline M62/F draws, rejecting improved13 ancestry culls at or before K50; duplicates retained. No K38 walkback filtering. These are conditional endpoint cohorts, not full-range K38 packages.",
        "digest_note": "FNV-1a over ordered 16-byte little-endian source n0 values; deterministic identity, not cryptographic.",
        "timing_scope": "Fused reusable suffix expansion, mod9/OEE, optional resident B arrays, scalar tails and flagged wide rechecks; setup, sampling, validation and JSON excluded.",
        "comparison_scope": "Same original source families for every A/B; native endpoint counts, cases, iterations and checksums may differ across A/B. No GPU savings or full-range estimate is inferred here.",
        "window": window.label(), "bv_mode": "Shared", "cutoff_bits": 120,
        "cohorts": cohorts.iter().map(|cohort| &cohort.metadata).collect::<Vec<_>>(),
    }));
    let mut plans = Vec::new();
    for &a in &a_values {
        for &b in &b_values {
            let started = Instant::now();
            let plan = Plan::new(N, SOURCE_M, a, (b != 0).then_some(b), 16, &window);
            println!("{}", json!({"type":"setup", "a":a, "m":N-a, "b":b,
                "wall_seconds":started.elapsed().as_secs_f64(),
                "table_bytes":plan.table_bytes(), "lookahead_bytes":plan.lookahead_bytes()}));
            for cohort in &cohorts {
                println!("{}", json!({"type":"validation", "a":a,"b":b,"source_f":cohort.f,
                    "audit":audit(&plan, &cohort.nodes[..audit_count.min(count)])}));
            }
            plans.push(plan);
        }
    }
    let combinations: Vec<_> = (0..plans.len()).flat_map(|p| (0..cohorts.len()).map(move |c| (p,c))).collect();
    let mut signatures = Vec::new();
    for &(p,c) in &combinations {
        let (work, timing) = run(&plans[p], &cohorts[c].nodes);
        let signature = work_json(&work);
        println!("{}", json!({"type":"warmup", "a":plans[p].a(),"b":plans[p].b().unwrap_or(0),
            "source_f":cohorts[c].f,"work":signature,"timing":timing}));
        signatures.push(signature);
    }
    let mut times = vec![Vec::new(); combinations.len()];
    for rep in 0..repetitions {
        let mut order: Vec<_> = (0..combinations.len()).collect();
        let length = order.len();
        order.rotate_left((rep * 11) % length);
        if rep % 2 != 0 { order.reverse(); }
        for (position, index) in order.into_iter().enumerate() {
            let (p,c) = combinations[index];
            let (work, timing) = run(&plans[p], &cohorts[c].nodes);
            let signature = work_json(&work);
            assert_eq!(signature, signatures[index], "repeated work signature mismatch");
            times[index].push(timing["wall_seconds"].as_f64().unwrap());
            println!("{}", json!({"type":"trial","repetition":rep,"position":position,
                "a":plans[p].a(),"m":plans[p].m(),"b":plans[p].b().unwrap_or(0),
                "source_f":cohorts[c].f,"work":signature,"timing":timing}));
        }
    }
    for (index,&(p,c)) in combinations.iter().enumerate() {
        let values = &times[index];
        println!("{}", json!({"type":"summary","a":plans[p].a(),"m":plans[p].m(),
            "b":plans[p].b().unwrap_or(0),"source_f":cohorts[c].f,"repetitions":values.len(),
            "median_wall_seconds":median(values),
            "min_wall_seconds":values.iter().copied().reduce(f64::min),
            "max_wall_seconds":values.iter().copied().reduce(f64::max),
            "median_ns_per_source":median(values)*1e9/count as f64,
            "work":signatures[index]}));
    }
    println!("{}", json!({"type":"complete","exact_equal":true,
        "configurations":combinations.len(),"trials":combinations.len()*repetitions}));
}
