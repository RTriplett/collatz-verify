//! Serialized GPU microprobe of deeper expansion for original M62/F40 nodes.
//! Accepts run_overall.py's positional protocol: target [count reps threads seed
//! ignored_variants ignored_windows ignored_a]. The probe always uses A10/9/8
//! and class-indexed windows [32,35], followed by an A10 end control.
//!
//! This conditional endpoint sample is distinct from the saved K38 packages:
//! it applies improved13 ancestry through K50, but no K38 walkback certificate.
use cv::gpu::{GpuBench, GpuClassStats};
use cv::lookahead::{BvMode, Lookahead, WindowRules};
use cv::rules::Rules;
use cv::sample::{Draw, Rng, Sampler};
use cv::tail::{endpoint_work, TailParams};
use cv::tree::{dfs, Node, Sink, TreeStats};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

const SOURCE_M: u32 = 62;
const SOURCE_F: u32 = 40;
const CUTOFF: u32 = 50;
const BLOCK: usize = 4096;
const BLOCK_SEED: u64 = 0xD1B5_4A32_D192_ED03;

fn sample(count: usize, seed: u64, threads: usize, rules: &Rules) -> (Vec<Node>, Value) {
    let timer = Instant::now();
    let sampler = Sampler::new(&Rules::baseline(), SOURCE_M, Some((SOURCE_F, SOURCE_F)));
    let mut nodes = Vec::with_capacity(count);
    let mut seen = HashSet::with_capacity(count);
    let (mut draws, mut rejected, mut duplicates) = (0u64, 0u64, 0u64);
    let mut first_block = 0;
    // Indexed parallel collection preserves a single stream of deterministic
    // blocks, independent of the number of workers or blocks in a round.
    'rounds: loop {
        let blocks: Vec<Vec<Draw>> = (first_block..first_block + threads.max(1) * 4)
            .into_par_iter()
            .map(|block| {
                let mut rng = Rng(seed ^ (block as u64).wrapping_mul(BLOCK_SEED));
                (0..BLOCK)
                    .map(|_| sampler.draw(&mut rng, Some(rules)))
                    .collect()
            })
            .collect();
        first_block += blocks.len();
        for block in blocks {
            for draw in block {
                draws += 1;
                if draw.culled.is_some_and(|(depth, _)| depth <= CUTOFF) {
                    rejected += 1;
                    continue;
                }
                if !seen.insert(draw.node.n0) {
                    duplicates += 1;
                    continue;
                }
                nodes.push(draw.node);
                if nodes.len() % (1 << 19) == 0 {
                    eprintln!("accepted {} of {count} F40 source endpoints", nodes.len());
                }
                if nodes.len() == count {
                    break 'rounds;
                }
            }
        }
    }
    let (mut fnv, mut xor, mut sum) = (0xcbf2_9ce4_8422_2325u64, 0u64, 0u128);
    for node in &nodes {
        let n = node.n0 as u64;
        xor ^= n;
        sum += n as u128;
        for byte in n.to_le_bytes() {
            fnv = (fnv ^ byte as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    let metadata = json!({
        "selection": "Unique uniform baseline M62/F40 draws; reject first improved13 cull at depth <=50. Improved culls after K50 do not reject. No K38 walkback filtering.",
        "different_from_k38_corpus": true,
        "source_m": SOURCE_M, "source_f": SOURCE_F, "cutoff": CUTOFF,
        "accepted": nodes.len(), "draws_consumed": draws,
        "rejected_through_cutoff": rejected, "duplicates": duplicates,
        "baseline_population": sampler.population().to_string(),
        "seed": seed, "deterministic_draw_block_size": BLOCK,
        "block_seed_multiplier": format!("{BLOCK_SEED:016X}"),
        "ordered_n0_fnv1a64_le": format!("{fnv:016X}"),
        "digest_note": "FNV-1a over the ordered eight-byte little-endian source residues; deterministic identifier, not a cryptographic hash.",
        "n0_xor": format!("{xor:016X}"), "n0_sum": sum.to_string(),
        "first_16_sources": &nodes[..nodes.len().min(16)],
        "sampling_wall_seconds": timer.elapsed().as_secs_f64(),
    });
    (nodes, metadata)
}

struct Classes {
    offset: u32,
    nodes: [Vec<Node>; 2],
}

impl Sink for Classes {
    fn endpoint(&mut self, node: &Node) {
        let class = node.f.checked_sub(self.offset).expect("class below offset") as usize;
        assert!(
            class < self.nodes.len(),
            "descendant class lacks a bitvector"
        );
        self.nodes[class].push(*node);
    }
}

fn expand(roots: &[Node], target: u32, offset: u32) -> Classes {
    let mut classes = Classes {
        offset,
        nodes: [Vec::new(), Vec::new()],
    };
    let baseline = Rules::baseline();
    let mut unused = TreeStats::default();
    for root in roots {
        dfs::<_, false>(root, target, &baseline, &mut classes, &mut unused);
    }
    classes
}

#[derive(Clone, Default)]
struct Work {
    endpoints: u64,
    cases: u64,
    iters: u64,
    hash: u32,
    flags: u64,
    launches: u64,
    kernel_seconds: f64,
}

impl Work {
    fn gpu(stats: &GpuClassStats, bench: &GpuBench) -> Self {
        Self {
            endpoints: stats.endpoints,
            cases: stats.cases,
            iters: stats.iters,
            hash: bench.last_hash,
            flags: bench.last_flags,
            launches: stats.launches,
            kernel_seconds: stats.kernel_seconds,
        }
    }

    fn signature(&self) -> (u64, u64, u64, u32, u64) {
        (
            self.endpoints,
            self.cases,
            self.iters,
            self.hash,
            self.flags,
        )
    }

    fn merge(&mut self, other: &Self) {
        self.endpoints += other.endpoints;
        self.cases += other.cases;
        self.iters += other.iters;
        self.hash ^= other.hash;
        self.flags += other.flags;
        self.launches += other.launches;
        self.kernel_seconds += other.kernel_seconds;
    }

    fn json(&self) -> Value {
        json!({"endpoints":self.endpoints,"cases":self.cases,"iters":self.iters,
            "hash":format!("{:08X}",self.hash),"flagged":self.flags,
            "launches":self.launches,"kernel_seconds":self.kernel_seconds})
    }
}

fn cpu_work(nodes: &[Node], la: &Lookahead, tail: &TailParams) -> Work {
    let mut total = Work::default();
    for node in nodes {
        let work = endpoint_work(la, tail, node, |_| {});
        total.endpoints += 1;
        total.cases += work.cases;
        total.iters += work.iters;
        total.hash ^= work.hash;
        total.flags += work.flagged as u64;
    }
    total
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(
        args.first().map(String::as_str).unwrap_or("target"),
        "target"
    );
    let number = |i: usize, default: usize| {
        args.get(i)
            .map(|s| s.parse::<usize>().unwrap())
            .unwrap_or(default)
    };
    let (count, reps, threads, seed) = (
        number(1, 1 << 21),
        number(2, 3),
        number(3, 8),
        number(4, 20261025) as u64,
    );
    assert!(count >= 16 && reps > 0 && threads > 0);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let rules = Rules::improved(13);
    let (sources, corpus) = sample(count, seed, threads, &rules);
    let window = WindowRules {
        pm_inf: true,
        mix: rules.mix.clone(),
    };
    let tail = TailParams::default();
    println!(
        "{}",
        json!({"type":"config","experiment":"f40_serialized_gpu_probe",
        "n":72,"source_m":SOURCE_M,"source_f":SOURCE_F,"count":count,
        "repetitions":reps,"threads":threads,"seed":seed,"corpus":corpus,
        "a_order":[10,9,8,10],"class_b":[32,35],"bv_mode":"Shared",
        "window":window.label(),"cutoff_bits":tail.cutoff_bits,"tail_oee":tail.tail_oee,
        "timing_scope":"Serialized CUDA kernel event seconds, summed over classes. Host expansion, packing, transfer and array construction excluded; no end-to-end mixed-A pipeline is measured.",
        "sum_scope":"For each repetition index, sum the separately timed classes. Classes are measured in contiguous blocks to keep their arrays resident.",
        "validation_scope":"CPU/GPU exact endpoint, case, iteration, hash and flag comparisons on descendants of the first 16 source endpoints at every A."})
    );
    let mut same_a = BTreeMap::new();
    let mut summaries = Vec::new();
    let mut baseline_seconds = None;
    for (block, a) in [10, 9, 8, 10].into_iter().enumerate() {
        let m = 72 - a;
        let label = if block == 3 {
            "a10_end".to_string()
        } else {
            format!("a{a}")
        };
        let la =
            Lookahead::meta(m, a, 32, 2, BvMode::Shared, &window).with_class_windows(&[32, 35]);
        let expansion = Instant::now();
        let classes = expand(&sources, m, la.offset);
        let audit = expand(&sources[..16], m, la.offset);
        let expansion_seconds = expansion.elapsed().as_secs_f64();
        let init = Instant::now();
        let mut bench = GpuBench::new(&la, tail.cutoff_bits, tail.tail_oee)
            .expect("GPU microprobe initialization");
        let init_seconds = init.elapsed().as_secs_f64();
        let mut totals = vec![Work::default(); reps];
        for (idx, nodes) in classes.nodes.iter().enumerate() {
            if nodes.is_empty() {
                continue;
            }
            let f = la.offset + idx as u32;
            let expected = cpu_work(&audit.nodes[idx], &la, &tail);
            let audit_timer = Instant::now();
            let stats = bench
                .run(idx, &audit.nodes[idx])
                .expect("GPU CPU-oracle audit");
            let actual = Work::gpu(&stats, &bench);
            assert_eq!(
                actual.signature(),
                expected.signature(),
                "CPU/GPU mismatch {label} F{f}"
            );
            assert_eq!(actual.flags, 0);
            println!(
                "{}",
                json!({"type":"validation","block":block,"label":label,
                "a":a,"m":m,"class":idx,"f":f,"source_endpoints":16,
                "cpu_gpu_equal":true,"work":actual.json(),
                "wall_seconds_including_initial_array_build":audit_timer.elapsed().as_secs_f64()})
            );
            let warmup = bench.run(idx, nodes).expect("GPU class warmup");
            let warmup = Work::gpu(&warmup, &bench);
            assert_eq!(warmup.flags, 0, "flagged microprobe trajectory");
            println!(
                "{}",
                json!({"type":"warmup","block":block,"label":label,
                "a":a,"m":m,"class":idx,"f":f,"work":warmup.json()})
            );
            for (rep, total) in totals.iter_mut().enumerate() {
                let stats = bench.run(idx, nodes).expect("GPU class timing");
                let work = Work::gpu(&stats, &bench);
                assert_eq!(
                    work.signature(),
                    warmup.signature(),
                    "work changed {label} F{f}"
                );
                total.merge(&work);
                println!(
                    "{}",
                    json!({"type":"class_trial","block":block,"label":label,
                    "a":a,"m":m,"class":idx,"f":f,"b":la.bs[idx],
                    "window_cutoff":la.cutoffs[idx],"repetition":rep,"work":work.json()})
                );
            }
        }
        for total in &totals {
            assert_eq!(total.signature(), totals[0].signature());
        }
        if let Some(previous) = same_a.insert(a, totals[0].signature()) {
            assert_eq!(
                previous,
                totals[0].signature(),
                "same-A control work changed"
            );
        }
        let seconds: Vec<_> = totals.iter().map(|work| work.kernel_seconds).collect();
        let measured = median(&seconds);
        let baseline = *baseline_seconds.get_or_insert(measured);
        let summary = json!({"type":"shape_summary","block":block,"label":label,
            "a":a,"m":m,"offset":la.offset,"class_b":la.bs,"cutoffs":la.cutoffs,
            "source_count":count,"expansion_wall_seconds":expansion_seconds,
            "gpu_initialization_wall_seconds":init_seconds,
            "array_build_seconds":bench.build_seconds,
            "kernel_seconds_by_repetition":seconds,"median_kernel_seconds":measured,
            "kernel_seconds_ratio_to_initial_a10":measured / baseline,
            "endpoint_count_factor":totals[0].endpoints as f64 / count as f64,
            "work":totals[0].json(),"trials":totals.iter().map(Work::json).collect::<Vec<_>>()});
        println!("{summary}");
        summaries.push(summary);
    }
    println!(
        "{}",
        json!({"type":"probe_summary","blocks":summaries,
        "interpretation":"Conditional original-F40 corpus; serialized GPU work only. Ratios do not measure end-to-end selective expansion and are not extrapolated to the saved K38 corpus."})
    );
}
