//! Matched-policy comparison of direct terminal enumeration against DFS.
//! terminal_bench roots reps threads seed target cutoff variants sink
//! Counters enabled; one pruning policy per run, independent of table width.
mod fast_tail;
#[allow(dead_code)]
mod tail_policy;
mod terminal;
#[cfg(test)]
mod terminal_tests;
mod terminal_walk;
#[allow(dead_code)]
mod ternary;

use cv::rules::{walkback_certificate, Rules};
use cv::sample::{Rng, Sampler};
use cv::tree::{Node, Sink, TreeStats};
use fast_tail::Endpoints;
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
impl Endpoints for Digest {
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
    fn endpoint(&mut self, n: &Node) {
        Endpoints::endpoint(self, n.n0 as u64, n.f, n.oee, n.ne);
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

trait BenchSink: Endpoints + Sink + Send {
    fn new() -> Self;
    fn finish(self) -> Digest;
}
impl BenchSink for Digest {
    fn new() -> Self {
        Self::default()
    }
    fn finish(self) -> Digest {
        self
    }
}

// Pack 16-byte payloads by F and expose every completed chunk to black_box.
// This includes stores and bounded buffering, but no locks, GPU or transfer.
const CHUNK: usize = 4096;
struct Packed {
    digest: Digest,
    buffers: Vec<Vec<u32>>,
}
impl BenchSink for Packed {
    fn new() -> Self {
        Self {
            digest: Digest::default(),
            buffers: vec![Vec::new(); 65],
        }
    }
    fn finish(self) -> Digest {
        for buffer in &self.buffers {
            if !buffer.is_empty() {
                black_box(buffer.as_slice());
            }
        }
        self.digest
    }
}
impl Endpoints for Packed {
    #[inline(always)]
    fn endpoint(&mut self, n0: u64, f: u32, oee: u8, ne: u8) {
        Endpoints::endpoint(&mut self.digest, n0, f, oee, ne);
        let buffer = &mut self.buffers[f as usize];
        // Shared lookahead uses family zero. M<=64 leaves upper n0 words zero.
        buffer.extend_from_slice(&[n0 as u32, (n0 >> 32) as u32, 0, (oee as u32) << 30]);
        if buffer.len() == 4 * CHUNK {
            black_box(buffer.as_slice());
            buffer.clear();
        }
    }
}
impl Sink for Packed {
    #[inline(always)]
    fn endpoint(&mut self, n: &Node) {
        Endpoints::endpoint(self, n.n0 as u64, n.f, n.oee, n.ne);
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
    fn merge(mut self, b: Self) -> Self {
        self.digest = self.digest.merge(b.digest);
        self.stats.merge(&b.stats);
        self
    }
}
impl PartialEq for Outcome {
    fn eq(&self, b: &Self) -> bool {
        self.digest == b.digest
            && self.stats.generated == b.stats.generated
            && self.stats.survived == b.stats.survived
            && self.stats.culls == b.stats.culls
    }
}

#[derive(Clone, Copy, Debug)]
enum Variant {
    Production,
    Compact,
    Table(u32, bool),
}
impl Variant {
    fn name(self) -> String {
        match self {
            Self::Production => "production".into(),
            Self::Compact => "compact".into(),
            Self::Table(w, g) => format!("{}{w}", if g { "group" } else { "flat" }),
        }
    }
}
struct Setup {
    target: u32,
    cutoff: u32,
    full: Rules,
    base: Rules,
    engines: Engines,
    tables: Vec<terminal::Table>,
}
impl Setup {
    fn new(target: u32, cutoff: u32, variants: &[Variant]) -> Self {
        let full = Rules::improved(13);
        let base = Rules::baseline();
        let mut widths: Vec<_> = variants
            .iter()
            .filter_map(|v| match v {
                Variant::Table(w, _) => Some(*w),
                _ => None,
            })
            .collect();
        widths.sort_unstable();
        widths.dedup();
        let tables = widths
            .into_iter()
            .map(|w| terminal::Table::new(target - w, target, &base))
            .collect();
        Self {
            target,
            cutoff,
            engines: Engines::new(full.clone(), target),
            full,
            base,
            tables,
        }
    }
    fn run<S: Endpoints + Sink>(
        &self,
        v: Variant,
        root: &Node,
        sink: &mut S,
        stats: &mut TreeStats,
    ) {
        match v {
            Variant::Production => {
                self.engines
                    .run::<_, true>(Policy::Cutoff(self.cutoff), root, sink, stats)
            }
            Variant::Compact => fast_tail::cutoff_dfs::<_, true>(
                root,
                self.target,
                self.cutoff,
                &self.full,
                &self.base,
                sink,
                stats,
            ),
            Variant::Table(w, g) => {
                let table = self
                    .tables
                    .iter()
                    .find(|t| t.target() - t.start() == w)
                    .unwrap();
                if g {
                    terminal_walk::run::<_, true, true>(
                        root,
                        self.cutoff,
                        &self.full,
                        &self.base,
                        table,
                        sink,
                        stats,
                    );
                } else {
                    terminal_walk::run::<_, true, false>(
                        root,
                        self.cutoff,
                        &self.full,
                        &self.base,
                        table,
                        sink,
                        stats,
                    );
                }
            }
        }
    }
    fn batch<S: BenchSink>(&self, v: Variant, roots: &[Node], pool: &rayon::ThreadPool) -> Outcome {
        pool.install(|| {
            roots
                .par_iter()
                .fold(
                    || (S::new(), TreeStats::new(self.target)),
                    |(mut sink, mut stats), root| {
                        self.run(v, root, &mut sink, &mut stats);
                        (sink, stats)
                    },
                )
                .map(|(sink, stats)| Outcome {
                    digest: sink.finish(),
                    stats,
                })
                .reduce(|| Outcome::new(self.target), Outcome::merge)
        })
    }
    fn timed_batch(
        &self,
        v: Variant,
        roots: &[Node],
        pool: &rayon::ThreadPool,
        packed: bool,
    ) -> Outcome {
        if packed {
            self.batch::<Packed>(v, roots, pool)
        } else {
            self.batch::<Digest>(v, roots, pool)
        }
    }
}

#[derive(Default)]
struct Payloads(Vec<(u64, u32, u8, u8)>);
impl Endpoints for Payloads {
    fn endpoint(&mut self, n0: u64, f: u32, oee: u8, ne: u8) {
        self.0.push((n0, f, oee, ne));
    }
}
impl Sink for Payloads {
    fn endpoint(&mut self, n: &Node) {
        Endpoints::endpoint(self, n.n0 as u64, n.f, n.oee, n.ne);
    }
}
fn audit(setup: &Setup, variants: &[Variant], seed: u64) {
    let start = setup.target.saturating_sub(12).max(2);
    let sampler = Sampler::new(&setup.base, start, None);
    let mut rng = Rng(seed ^ 0x76fbc8);
    let mut endpoints = 0;
    for _ in 0..64 {
        let root = loop {
            let d = sampler.draw(&mut rng, Some(&setup.full));
            if d.culled.is_none() {
                break d.node;
            }
        };
        let mut expected = Payloads::default();
        let mut want_stats = TreeStats::new(setup.target);
        setup.run(Variant::Production, &root, &mut expected, &mut want_stats);
        expected.0.sort_unstable();
        endpoints += expected.0.len();
        for &v in variants {
            let mut got = Payloads::default();
            let mut stats = TreeStats::new(setup.target);
            setup.run(v, &root, &mut got, &mut stats);
            got.0.sort_unstable();
            assert_eq!(got.0, expected.0, "payload mismatch {}", v.name());
            assert_eq!(stats.generated, want_stats.generated);
            assert_eq!(stats.survived, want_stats.survived);
            assert_eq!(stats.culls, want_stats.culls);
        }
    }
    println!(
        "{}",
        json!({"type":"validation","roots":64,"start":start,"target":setup.target,
        "exact_payloads_compared":endpoints,"all_counters_equal":true,"passed":true})
    );
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let number = |i: usize, d: usize| args.get(i).map(|s| s.parse().unwrap()).unwrap_or(d);
    let (nroots, reps, threads, seed, target, cutoff) = (
        number(0, 128),
        number(1, 5),
        number(2, 8),
        number(3, 17411) as u64,
        number(4, 62) as u32,
        number(5, 50) as u32,
    );
    let selection = args.get(6).map(String::as_str).unwrap_or("all");
    let sink = args.get(7).map(String::as_str).unwrap_or("digest");
    assert!(nroots > 0 && reps > 0 && threads > 0 && (38..=64).contains(&target));
    assert!(cutoff >= 38 && cutoff <= target && matches!(sink, "digest" | "packed"));
    let mut available = vec![Variant::Production, Variant::Compact];
    for w in [4, 6, 8, 10, 12] {
        if w <= target - cutoff {
            available.push(Variant::Table(w, false));
            available.push(Variant::Table(w, true));
        }
    }
    let variants: Vec<_> = available
        .into_iter()
        .filter(|v| selection == "all" || selection.split(',').any(|s| s == v.name()))
        .collect();
    assert!(
        variants.iter().any(|v| matches!(v, Variant::Compact)),
        "include compact control"
    );
    if selection != "all" {
        assert_eq!(
            selection.split(',').count(),
            variants.len(),
            "invalid variant for cutoff"
        );
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let setup_start = Instant::now();
    let setup = Setup::new(target, cutoff, &variants);
    let setup_seconds = setup_start.elapsed().as_secs_f64();
    let sampling = Instant::now();
    let sampler = Sampler::new(&setup.base, 38, None);
    let mut rng = Rng(seed);
    let mut roots = Vec::with_capacity(nroots);
    let (mut draws, mut cheap_rejected, mut walk_rejected) = (0u64, 0u64, 0u64);
    while roots.len() < nroots {
        draws += 1;
        let d = sampler.draw(&mut rng, Some(&setup.full));
        if d.culled.is_some() {
            cheap_rejected += 1;
            continue;
        }
        let n = d.node;
        if walkback_certificate(n.n0, n.k, n.f, n.m, 65536).is_some() {
            walk_rejected += 1;
            continue;
        }
        roots.push(n);
    }
    let sampling_seconds = sampling.elapsed().as_secs_f64();
    let mut corpus = Digest::default();
    for n in &roots {
        Sink::endpoint(&mut corpus, n);
    }
    println!(
        "{}",
        json!({"type":"config","checkpoint":38,"target":target,"cutoff":cutoff,"roots":nroots,
        "repetitions":reps,"threads":threads,"seed":seed,"sink":sink,"statistics":true,
        "variants":variants.iter().map(|v|v.name()).collect::<Vec<_>>(),"setup_seconds":setup_seconds,
        "tables":setup.tables.iter().map(|t|json!({"width":t.target()-t.start(),"start":t.start(),
            "bytes":t.bytes(),"entries":t.entries(),"states":t.states()})).collect::<Vec<_>>(),
        "sampling_seconds":sampling_seconds,"baseline_draws":draws,"cheap_rejected":cheap_rejected,
        "walk_rejected":walk_rejected,"corpus":corpus.json(),"walkback_cap":65536,
        "root_selection":"Uniform baseline K38 roots rejected by improved13 ancestry and checkpoint walkback; same roots and pruning policy in every variant",
        "packing":"packed sink stores four u32 words per endpoint by F, shared-family flag, black_box flushes every 4096 endpoints; no queue or device",
        "excludes":["table construction","root sampling","archive IO","GPU","endpoint transfer","queue backpressure"]})
    );
    audit(&setup, &variants, seed);
    for &v in &variants {
        black_box(setup.timed_batch(v, &roots[..nroots.min(32)], &pool, sink == "packed"));
    }
    let mut expected = None;
    let mut timings = vec![Vec::<f64>::new(); variants.len()];
    for rep in 0..reps {
        for pos in 0..variants.len() {
            let i = (pos + rep) % variants.len();
            let timer = Instant::now();
            let out = black_box(setup.timed_batch(
                variants[i],
                black_box(&roots),
                &pool,
                sink == "packed",
            ));
            let seconds = timer.elapsed().as_secs_f64();
            if let Some(ref want) = expected {
                assert_eq!(&out, want, "outcome mismatch {}", variants[i].name());
            } else {
                expected = Some(out.clone());
            }
            timings[i].push(seconds);
            println!(
                "{}",
                json!({"type":"trial","variant":variants[i].name(),"repetition":rep,
            "order_position":pos,"seconds":seconds,"endpoints":out.digest.count})
            );
        }
    }
    let out = expected.unwrap();
    for (i, v) in variants.iter().enumerate() {
        let mut times = timings[i].clone();
        times.sort_by(f64::total_cmp);
        println!(
            "{}",
            json!({"type":"summary","variant":v.name(),"median_seconds":times[times.len()/2],
            "minimum_seconds":times[0],"maximum_seconds":times[times.len()-1],
            "digest":out.digest.json(),"tree_stats":out.stats})
        );
    }
}
