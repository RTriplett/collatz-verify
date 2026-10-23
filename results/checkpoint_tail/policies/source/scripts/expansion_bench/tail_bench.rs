//! CPU policy comparison below identical improved+walkback K38 roots.
//! tail_bench [roots=128] [repetitions=5] [threads=8] [seed=17395]
//!            [target=62] [variants=all|comma,separated,names]
//! Counters are enabled. Timing excludes sampling, audit, archive IO and GPU.
mod fast_tail;
mod tail_policy;
#[allow(dead_code)]
mod ternary;

use cv::rules::{walkback_certificate, Rules};
use cv::sample::{Rng, Sampler};
use cv::tree::{Node, Sink, TreeStats};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::hint::black_box;
use std::time::Instant;
use tail_policy::{Engines, Policy};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Digest {
    count: u64,
    sum_n: u128,
    state_xor: u64,
    by_f: [u64; 65],
}
impl Default for Digest {
    fn default() -> Self {
        Self {
            count: 0,
            sum_n: 0,
            state_xor: 0,
            by_f: [0; 65],
        }
    }
}
impl fast_tail::Endpoints for Digest {
    #[inline(always)]
    fn endpoint(&mut self, n0: u64, f: u32, oee: u8, ne: u8) {
        self.count += 1;
        self.sum_n = self.sum_n.wrapping_add(n0 as u128);
        self.state_xor ^= f as u64 | ((oee as u64) << 32) | ((ne as u64) << 40);
        self.by_f[f as usize] += 1;
    }
}
impl Sink for Digest {
    #[inline(always)]
    fn endpoint(&mut self, nd: &Node) {
        fast_tail::Endpoints::endpoint(self, nd.n0 as u64, nd.f, nd.oee, nd.ne);
    }
}
impl Digest {
    fn merge(mut self, b: Self) -> Self {
        self.count += b.count;
        self.sum_n = self.sum_n.wrapping_add(b.sum_n);
        self.state_xor ^= b.state_xor;
        for (a, b) in self.by_f.iter_mut().zip(b.by_f) {
            *a += b;
        }
        self
    }
    fn json(&self) -> Value {
        json!({"endpoints":self.count,"sum_n":self.sum_n.to_string(),
            "state_xor":self.state_xor,"by_f":self.by_f.to_vec()})
    }
}

#[derive(Clone, Debug)]
struct Outcome {
    digest: Digest,
    stats: TreeStats,
}
impl Outcome {
    fn new(target: u32) -> Self {
        Self {
            digest: Digest::default(),
            stats: TreeStats::new(target),
        }
    }
    fn merge(mut self, other: Self) -> Self {
        self.digest = self.digest.merge(other.digest);
        self.stats.merge(&other.stats);
        self
    }
}
impl PartialEq for Outcome {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest
            && self.stats.generated == other.stats.generated
            && self.stats.survived == other.stats.survived
            && self.stats.culls == other.stats.culls
    }
}

#[derive(Clone, Copy, Debug)]
enum Variant {
    Policy(Policy),
    FastBaseline,
}
impl Variant {
    fn name(self) -> String {
        match self {
            Self::Policy(p) => p.name(),
            Self::FastBaseline => "fast_baseline".into(),
        }
    }
    fn run<S: Sink + fast_tail::Endpoints>(
        self,
        engines: &Engines,
        base: &Rules,
        target: u32,
        root: &Node,
        sink: &mut S,
        stats: &mut TreeStats,
    ) {
        match self {
            Self::Policy(p) => engines.run::<_, true>(p, root, sink, stats),
            Self::FastBaseline => fast_tail::dfs::<_, true>(root, target, base, sink, stats),
        }
    }
}
fn variants() -> Vec<Variant> {
    use Policy::*;
    let mut v: Vec<_> = [
        Full,
        Screened,
        Baseline,
        PmOnly,
        PmScreened,
        Cutoff(46),
        Cutoff(50),
        Cutoff(54),
        Cutoff(58),
    ]
    .into_iter()
    .map(Variant::Policy)
    .collect();
    v.insert(3, Variant::FastBaseline);
    v
}

fn batch(
    v: Variant,
    engines: &Engines,
    base: &Rules,
    target: u32,
    roots: &[Node],
    pool: &rayon::ThreadPool,
) -> Outcome {
    pool.install(|| {
        roots
            .par_iter()
            .fold(
                || Outcome::new(target),
                |mut out, root| {
                    v.run(engines, base, target, root, &mut out.digest, &mut out.stats);
                    out
                },
            )
            .reduce(|| Outcome::new(target), Outcome::merge)
    })
}

#[derive(Default)]
struct Payloads(Vec<(u64, u32, u8, u8)>);
impl Sink for Payloads {
    fn endpoint(&mut self, nd: &Node) {
        self.0.push((nd.n0 as u64, nd.f, nd.oee, nd.ne));
    }
}
impl fast_tail::Endpoints for Payloads {
    fn endpoint(&mut self, n0: u64, f: u32, oee: u8, ne: u8) {
        self.0.push((n0, f, oee, ne));
    }
}
fn audit(roots: &[Node], target: u32, full: &Rules, base: &Rules) {
    let engines = Engines::new(full.clone(), target);
    let mut compared = 0;
    for root in roots {
        let mut outputs = Vec::new();
        for v in variants() {
            let mut got = Payloads::default();
            v.run(
                &engines,
                base,
                target,
                root,
                &mut got,
                &mut TreeStats::new(target),
            );
            got.0.sort_unstable();
            outputs.push(got.0);
        }
        assert_eq!(outputs[0], outputs[1]); // full == screened
        assert_eq!(outputs[2], outputs[3]); // baseline == fast_baseline
        assert_eq!(outputs[4], outputs[5]); // pm_only == pm_screened
        for output in &outputs {
            assert!(outputs[0].iter().all(|x| output.binary_search(x).is_ok()));
            assert!(output.iter().all(|x| outputs[2].binary_search(x).is_ok()));
        }
        compared += outputs[2].len();
    }
    println!(
        "{}",
        json!({"type":"validation","roots":roots.len(),"target":target,
        "baseline_payloads_compared":compared,"all_policies_audited":true,"passed":true})
    );
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let number =
        |i: usize, default: usize| args.get(i).map(|s| s.parse().unwrap()).unwrap_or(default);
    let nroots = number(0, 128);
    let reps = number(1, 5);
    let threads = number(2, 8);
    let seed = number(3, 17395) as u64;
    let target = number(4, 62) as u32;
    let selection = args.get(5).map(String::as_str).unwrap_or("all");
    let selected: Vec<_> = variants()
        .into_iter()
        .filter(|v| selection == "all" || selection.split(',').any(|s| s == v.name()))
        .collect();
    assert!(nroots > 0 && reps > 0 && threads > 0 && (38..=64).contains(&target));
    assert!(!selected.is_empty());
    if selection != "all" {
        assert_eq!(
            selection.split(',').count(),
            selected.len(),
            "unknown or duplicate variant"
        );
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let full = Rules::improved(13);
    let base = Rules::baseline();
    let sampler = Sampler::new(&base, 38, None);
    let mut rng = Rng(seed);
    let mut roots = Vec::with_capacity(nroots);
    let (mut draws, mut cheap_rejected, mut walk_rejected) = (0u64, 0u64, 0u64);
    let sampling = Instant::now();
    while roots.len() < nroots {
        draws += 1;
        let draw = sampler.draw(&mut rng, Some(&full));
        if draw.culled.is_some() {
            cheap_rejected += 1;
            continue;
        }
        let nd = draw.node;
        if walkback_certificate(nd.n0, nd.k, nd.f, nd.m, 65_536).is_some() {
            walk_rejected += 1;
            continue;
        }
        roots.push(nd);
    }
    let sampling_seconds = sampling.elapsed().as_secs_f64();
    let mut corpus = Digest::default();
    for nd in &roots {
        Sink::endpoint(&mut corpus, nd);
    }
    let engines = Engines::new(full.clone(), target);
    println!(
        "{}",
        json!({"type":"config","checkpoint":38,"target":target,
        "roots":nroots,"repetitions":reps,"threads":threads,"seed":seed,
        "baseline_draws":draws,"cheap_rejected":cheap_rejected,"walk_rejected":walk_rejected,
        "sampling_seconds":sampling_seconds,"corpus":corpus.json(),"statistics":true,
        "root_selection":"Uniform baseline K38 roots rejected by improved13 ancestry and capped K38 walkback; same roots for every tail policy, not a stored archive replay",
        "walkback_cap":65536,"variants":selected.iter().map(|v|v.name()).collect::<Vec<_>>(),
        "sink":"count, wrapping u128 sum of n0, XOR of f/oee/ne, counts by f; identical GPU-consumed fields",
        "excludes":["archive IO","setup","GPU","endpoint transfer"]})
    );
    audit(&roots[..nroots.min(64)], target.min(50), &full, &base);
    for &v in &selected {
        black_box(batch(
            v,
            &engines,
            &base,
            target,
            &roots[..nroots.min(32)],
            &pool,
        ));
    }
    let mut expected: Vec<Option<Outcome>> = vec![None; selected.len()];
    let mut timings = vec![Vec::<f64>::new(); selected.len()];
    for rep in 0..reps {
        for pos in 0..selected.len() {
            let i = (pos + rep) % selected.len();
            let timer = Instant::now();
            let outcome = black_box(batch(
                selected[i],
                &engines,
                &base,
                target,
                black_box(&roots),
                &pool,
            ));
            let seconds = timer.elapsed().as_secs_f64();
            if let Some(ref want) = expected[i] {
                assert_eq!(&outcome, want, "within-policy outcome changed");
            } else {
                expected[i] = Some(outcome.clone());
            }
            timings[i].push(seconds);
            println!(
                "{}",
                json!({"type":"trial","variant":selected[i].name(),"repetition":rep,
                "order_position":pos,"seconds":seconds,"endpoints":outcome.digest.count})
            );
        }
    }
    for (a, b) in [
        ("full", "screened"),
        ("baseline", "fast_baseline"),
        ("pm_only", "pm_screened"),
    ] {
        if let (Some(i), Some(j)) = (
            selected.iter().position(|v| v.name() == a),
            selected.iter().position(|v| v.name() == b),
        ) {
            assert_eq!(
                expected[i], expected[j],
                "equivalent-policy outcomes differ"
            );
        }
    }
    for (i, v) in selected.iter().enumerate() {
        let mut sorted = timings[i].clone();
        sorted.sort_by(f64::total_cmp);
        let out = expected[i].as_ref().unwrap();
        println!(
            "{}",
            json!({"type":"summary","variant":v.name(),"median_seconds":sorted[sorted.len()/2],
            "minimum_seconds":sorted[0],"maximum_seconds":sorted[sorted.len()-1],
            "digest":out.digest.json(),"tree_stats":out.stats})
        );
    }
}
