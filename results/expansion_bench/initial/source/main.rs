//! Exact expansion experiments; no changes to the production verification path.
//! Usage: expansion_bench [roots=8192] [repetitions=7] [threads=1]
//!        [seed=17391] [case=early|late|full] [rules=improved|baseline]
//!        [audit_roots=64] [variants=all|comma,separated,names]
//! All algorithms use the same roots and inexpensive endpoint digest sink.
//! Full sorted Node equality is audited separately, outside timed regions.
mod blocks;
mod gating;

use cv::rules::{walkback_certificate, Rules};
use cv::sample::{Rng, Sampler};
use cv::tree::{dfs, Node, Sink, TreeStats};
use rayon::prelude::*;
use serde_json::json;
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Digest {
    count: u64,
    sum_n: u128,
    sum_m: u128,
    state_xor: u64,
}
impl Sink for Digest {
    #[inline(always)]
    fn endpoint(&mut self, nd: &Node) {
        self.count += 1;
        self.sum_n = self.sum_n.wrapping_add(nd.n0);
        self.sum_m = self.sum_m.wrapping_add(nd.m);
        self.state_xor ^= nd.m3 as u64
            | ((nd.f as u64) << 32)
            | ((nd.k as u64) << 40)
            | ((nd.oee as u64) << 48)
            | ((nd.ne as u64) << 52);
    }
}
impl Digest {
    fn merge(self, b: Self) -> Self {
        Self {
            count: self.count + b.count,
            sum_n: self.sum_n.wrapping_add(b.sum_n),
            sum_m: self.sum_m.wrapping_add(b.sum_m),
            state_xor: self.state_xor ^ b.state_xor,
        }
    }
    fn json(self) -> serde_json::Value {
        json!({"endpoints": self.count, "sum_n":self.sum_n.to_string(),
            "sum_m":self.sum_m.to_string(),"state_xor":self.state_xor})
    }
}
#[derive(Default)]
struct Collect(Vec<Node>);
impl Sink for Collect {
    fn endpoint(&mut self, nd: &Node) {
        self.0.push(*nd);
    }
}

#[derive(Clone, Copy, Debug)]
enum Variant {
    Reference,
    Gated,
    Early,
    Ternary,
    Block4,
    Block6,
}
impl Variant {
    fn name(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Gated => "gated",
            Self::Early => "early",
            Self::Ternary => "ternary",
            Self::Block4 => "block4",
            Self::Block6 => "block6",
        }
    }
}
const VARIANTS: [Variant; 6] = [
    Variant::Reference,
    Variant::Gated,
    Variant::Early,
    Variant::Ternary,
    Variant::Block4,
    Variant::Block6,
];

struct Engines {
    rules: Rules,
    target: u32,
    gates: gating::Gates,
    block4: blocks::Blocks,
    block6: blocks::Blocks,
}
impl Engines {
    fn new(start: u32, target: u32, rules: &Rules) -> Self {
        Self {
            rules: rules.clone(),
            target,
            gates: gating::Gates::new(target, rules.mix_j()),
            block4: blocks::Blocks::new(start, target, 4, rules),
            block6: blocks::Blocks::new(start, target, 6, rules),
        }
    }
    fn run<S: Sink>(&self, v: Variant, root: &Node, sink: &mut S) {
        match v {
            Variant::Reference => dfs::<_, false>(
                root,
                self.target,
                &self.rules,
                sink,
                &mut TreeStats::default(),
            ),
            Variant::Gated => gating::dfs_gated(root, self.target, &self.rules, &self.gates, sink),
            Variant::Early => gating::dfs_early(root, self.target, &self.rules, &self.gates, sink),
            Variant::Ternary => {
                gating::dfs_ternary(root, self.target, &self.rules, &self.gates, sink)
            }
            Variant::Block4 => {
                blocks::dfs_blocks(root, self.target, &self.rules, &self.block4, sink)
            }
            Variant::Block6 => {
                blocks::dfs_blocks(root, self.target, &self.rules, &self.block6, sink)
            }
        }
    }
    fn batch(&self, v: Variant, roots: &[Node], pool: &rayon::ThreadPool) -> Digest {
        pool.install(|| {
            roots
                .par_iter()
                .fold(Digest::default, |mut sink, root| {
                    self.run(v, root, &mut sink);
                    sink
                })
                .reduce(Digest::default, Digest::merge)
        })
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let number =
        |i: usize, default: usize| args.get(i).map(|s| s.parse().unwrap()).unwrap_or(default);
    let nroots = number(0, 8192);
    let reps = number(1, 7);
    let threads = number(2, 1);
    let seed = number(3, 17391) as u64;
    let case = args.get(4).map(String::as_str).unwrap_or("early");
    let (start, target) = match case {
        "early" => (38, 50),
        "late" => (50, 62),
        "full" => (38, 62),
        _ => panic!("invalid case"),
    };
    let rules = match args.get(5).map(String::as_str).unwrap_or("improved") {
        "improved" => Rules::improved(13),
        "baseline" => Rules::baseline(),
        _ => panic!("invalid rules"),
    };
    let audit_roots = number(6, 64).min(nroots);
    let selection = args.get(7).map(String::as_str).unwrap_or("all");
    let variants: Vec<_> = VARIANTS
        .into_iter()
        .filter(|v| selection == "all" || selection.split(',').any(|s| s == v.name()))
        .collect();
    assert!(
        nroots > 0
            && reps > 0
            && threads > 0
            && variants
                .first()
                .is_some_and(|v| matches!(v, Variant::Reference))
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();

    let sample_start = Instant::now();
    let base = Rules::baseline();
    let sampler = Sampler::new(&base, start, None);
    let mut rng = Rng(seed);
    let mut roots = Vec::with_capacity(nroots);
    let mut draws = 0u64;
    let mut cheap_rejected = 0u64;
    let mut walk_rejected = 0u64;
    while roots.len() < nroots {
        draws += 1;
        let draw = sampler.draw(&mut rng, Some(&rules));
        if draw.culled.is_some() {
            cheap_rejected += 1;
            continue;
        }
        let ancestor = if start == 38 {
            draw.node
        } else {
            let mut nd = Node::root(&base);
            while nd.k < 38 {
                nd = nd.children(&base)[((draw.node.n0 >> nd.k) & 1) as usize].unwrap();
            }
            nd
        };
        if walkback_certificate(ancestor.n0, 38, ancestor.f, ancestor.m, 65_536).is_some() {
            walk_rejected += 1;
            continue;
        }
        roots.push(draw.node);
    }
    let mut corpus = Digest::default();
    for nd in &roots {
        corpus.endpoint(nd);
    }
    let sampling_seconds = sample_start.elapsed().as_secs_f64();
    let setup = Instant::now();
    let engines = Engines::new(start, target, &rules);
    let setup_seconds = setup.elapsed().as_secs_f64();
    println!(
        "{}",
        json!({"type":"config","case":case,"start":start,"target":target,
        "roots":nroots,"seed":seed,"threads":threads,"repetitions":reps,"rules":rules.label(),
        "corpus":corpus.json(),"baseline_draws":draws,"cheap_rejected":cheap_rejected,"walk_rejected":walk_rejected,
        "root_selection":"Uniform baseline roots at start, reject cheap-rule ancestry and bounded walkback at ancestor38; not a replayed canonical archive",
        "walkback_depth":38,"walkback_cap":65536,"sampling_seconds":sampling_seconds,"setup_seconds":setup_seconds,
        "block4_bytes":engines.block4.bytes(),"block6_bytes":engines.block6.bytes(),
        "statistics":false,"sink":"count, wrapping128-bit sums of n0 and m, XOR of smallstate",
        "excludes":["archive IO","setup","per-depth accounting","GPU","endpoint transfer"],
        "variants":variants.iter().map(|v|v.name()).collect::<Vec<_>>()})
    );

    // Exact comparisons are bounded to twelve levels to avoid enormous vectors.
    let audit_target = target.min(start + 12);
    let audit = Engines::new(start, audit_target, &rules);
    let mut checked_endpoints = 0u64;
    for root in roots.iter().take(audit_roots) {
        let mut expected = Collect::default();
        audit.run(Variant::Reference, root, &mut expected);
        expected.0.sort_unstable_by_key(|nd| nd.n0);
        checked_endpoints += expected.0.len() as u64;
        for &v in variants.iter().skip(1) {
            let mut got = Collect::default();
            audit.run(v, root, &mut got);
            got.0.sort_unstable_by_key(|nd| nd.n0);
            assert_eq!(
                got.0,
                expected.0,
                "full endpoint mismatch {} at root {}",
                v.name(),
                root.n0
            );
        }
    }
    println!(
        "{}",
        json!({"type":"validation","roots":audit_roots,"start":start,"target":audit_target,
        "reference_endpoints_compared":checked_endpoints,"all_fields":true,"passed":true})
    );
    // One warmup per variant, on the same small subset, before rotating trials.
    for &v in &variants {
        black_box(engines.batch(v, &roots[..roots.len().min(64)], &pool));
    }
    let mut expected = None;
    let mut timings = vec![Vec::<f64>::new(); variants.len()];
    for rep in 0..reps {
        for pos in 0..variants.len() {
            let i = (pos + rep) % variants.len();
            let variant = variants[i];
            let timer = Instant::now();
            let digest = black_box(engines.batch(variant, black_box(&roots), &pool));
            let seconds = timer.elapsed().as_secs_f64();
            if let Some(want) = expected {
                assert_eq!(
                    digest,
                    want,
                    "timed endpoint digest mismatch {}",
                    variant.name()
                );
            } else {
                expected = Some(digest);
            }
            timings[i].push(seconds);
            println!(
                "{}",
                json!({"type":"trial","variant":variant.name(),"repetition":rep,
                "order_position":pos,"seconds":seconds,"endpoints":digest.count})
            );
        }
    }
    let median = |xs: &[f64]| {
        let mut x = xs.to_vec();
        x.sort_by(f64::total_cmp);
        x[x.len() / 2]
    };
    let reference = median(&timings[0]);
    for (i, v) in variants.iter().enumerate() {
        let med = median(&timings[i]);
        println!(
            "{}",
            json!({"type":"summary","variant":v.name(),"median_seconds":med,
            "minimum_seconds":timings[i].iter().copied().fold(f64::INFINITY,f64::min),
            "maximum_seconds":timings[i].iter().copied().fold(0.0,f64::max),
            "speedup_vs_reference":reference/med,"elapsed_reduction":1.0-med/reference,
            "endpoints_per_second":expected.unwrap().count as f64/med,"digest":expected.unwrap().json()})
        );
    }
}
