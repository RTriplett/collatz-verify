use cv::rules::{walkback_certificate, Rules};
use cv::sample::Sampler;
use cv::tree::{dfs, frontier, Node, Sink, TreeStats};
use rayon::prelude::*;
use serde_json::json;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const K: u32 = 30;
const CAP: u32 = 65_536;
const SPLIT: u32 = 16;
const M: u32 = 62;
const DRAWS: usize = 20_000;
const SEED: u64 = 0x643a_51ce_9b12_a809;

#[derive(Clone, Copy, Debug)]
struct Record {
    n0: u64,
    p: u128,
    s: u32,
    o: u32,
}

struct Collector {
    considered: u64,
    records: Vec<Record>,
}
impl Sink for Collector {
    fn endpoint(&mut self, nd: &Node) {
        self.considered += 1;
        if let Some(c) = walkback_certificate(nd.n0, K, nd.f, nd.m, CAP) {
            self.records.push(Record {
                n0: nd.n0.try_into().unwrap(),
                p: c.p,
                s: c.s,
                o: c.o,
            });
        }
    }
}

// Separate forward arithmetic: never call the reverse search or culling logic.
fn forward(mut n: u128, steps: u32) -> (u128, u32) {
    let mut odds = 0;
    for _ in 0..steps {
        if n & 1 == 0 {
            n /= 2;
        } else {
            n = n.checked_mul(3).unwrap().checked_add(1).unwrap() / 2;
            odds += 1;
        }
    }
    (n, odds)
}

fn sha256(path: &Path) -> String {
    let out = Command::new("sha256sum").arg(path).output().unwrap();
    assert!(out.status.success());
    String::from_utf8(out.stdout)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .into()
}

fn main() {
    let mut args = std::env::args_os().skip(1);
    let out_dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "results/walkback_reuse/cache-k30".into()),
    );
    assert!(
        args.next().is_none(),
        "usage: cache_demo [OUTPUT_DIRECTORY]"
    );
    fs::create_dir_all(&out_dir).unwrap();
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .unwrap();
    let all_start = Instant::now();
    let rules_start = Instant::now();
    let rules = Rules::improved(13);
    let rules_seconds = rules_start.elapsed().as_secs_f64();
    let mut top = TreeStats::new(K);
    let units = frontier(SPLIT, &rules, &mut top);
    eprintln!(
        "building K={K} positive-only certificate cache, {} subtrees, 8 threads",
        units.len()
    );
    let build_start = Instant::now();
    let pieces: Vec<_> = units
        .par_iter()
        .map(|nd| {
            let mut sink = Collector {
                considered: 0,
                records: Vec::new(),
            };
            let mut stats = TreeStats::new(K);
            dfs::<_, false>(nd, K, &rules, &mut sink, &mut stats);
            sink
        })
        .collect();
    let build_seconds = build_start.elapsed().as_secs_f64();
    let considered: u64 = pieces.iter().map(|s| s.considered).sum();
    let mut records: Vec<Record> = pieces.into_iter().flat_map(|s| s.records).collect();
    let sort_start = Instant::now();
    records.sort_unstable_by_key(|r| r.n0);
    assert!(records.windows(2).all(|w| w[0].n0 < w[1].n0));
    let sort_seconds = sort_start.elapsed().as_secs_f64();
    eprintln!("enumerated {considered} improved survivors; {} positive records; build {build_seconds:.3}s", records.len());

    let bin_path = out_dir.join("certificates-k30.bin");
    let write_start = Instant::now();
    let mut file = BufWriter::new(File::create(&bin_path).unwrap());
    for r in &records {
        file.write_all(&r.n0.to_le_bytes()).unwrap();
        file.write_all(&r.p.to_le_bytes()).unwrap();
        file.write_all(&r.s.to_le_bytes()).unwrap();
        file.write_all(&r.o.to_le_bytes()).unwrap();
    }
    file.flush().unwrap();
    drop(file);
    let write_seconds = write_start.elapsed().as_secs_f64();

    // Reload the on-disk encoding before auditing every record.
    let verify_start = Instant::now();
    let bytes = fs::read(&bin_path).unwrap();
    assert_eq!(bytes.len(), 32 * records.len());
    let records: Vec<Record> = bytes
        .chunks_exact(32)
        .map(|b| Record {
            n0: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            p: u128::from_le_bytes(b[8..24].try_into().unwrap()),
            s: u32::from_le_bytes(b[24..28].try_into().unwrap()),
            o: u32::from_le_bytes(b[28..32].try_into().unwrap()),
        })
        .collect();
    for r in &records {
        let (m, f) = forward(r.n0 as u128, K);
        let (merged, odd_steps) = forward(r.p, r.s);
        assert!(r.n0 < 1u64 << K && r.p > 0 && r.p < r.n0 as u128 && r.s >= 1 && r.o <= f);
        assert_eq!((merged, odd_steps), (m, r.o));
        let slope = (1u128.checked_shl(r.s).unwrap())
            .checked_mul(3u128.checked_pow(f - r.o).unwrap())
            .unwrap();
        assert!(slope <= 1u128 << K);
        // Explicitly check three translated family members as well.
        for a in [1u128, 2, (1u128 << (M - K)) - 1] {
            let member = r.n0 as u128 + (1u128 << K) * a;
            let partner = r.p + slope * a;
            assert!(partner < member);
            assert_eq!(forward(partner, r.s).0, forward(member, K).0);
        }
    }
    let verify_seconds = verify_start.elapsed().as_secs_f64();

    // Independent, exactly uniform baseline M=62 draws, then require survival
    // of every current improved rule along the whole path. Use no walkback
    // while sampling, so cache eligibility does not bias the validation set.
    let sample_start = Instant::now();
    let sampler = Sampler::new(&Rules::baseline(), M, None);
    let draws = sampler.draws(DRAWS, SEED, Some(&rules));
    let kept: Vec<_> = draws.iter().filter(|d| d.culled.is_none()).collect();
    let ancestors: Vec<_> = kept
        .iter()
        .map(|d| {
            let n0 = (d.node.n0 & ((1u128 << K) - 1)) as u64;
            let (m, f) = forward(n0 as u128, K);
            (n0, m, f)
        })
        .collect();
    let sample_seconds = sample_start.elapsed().as_secs_f64();
    let lookup_start = Instant::now();
    let cached: Vec<_> = ancestors
        .iter()
        .map(|&(n0, _, _)| records.binary_search_by_key(&n0, |r| r.n0).is_ok())
        .collect();
    let lookup_seconds = lookup_start.elapsed().as_secs_f64();
    let fresh_start = Instant::now();
    let fresh: Vec<_> = ancestors
        .iter()
        .map(|&(n0, m, f)| walkback_certificate(n0 as u128, K, f, m, CAP).is_some())
        .collect();
    let fresh_seconds = fresh_start.elapsed().as_secs_f64();
    let mismatches = cached.iter().zip(&fresh).filter(|(a, b)| a != b).count();
    assert_eq!(mismatches, 0);
    let hits = cached.iter().filter(|&&hit| hit).count();
    let mut unique_ancestors: Vec<_> = ancestors.iter().map(|x| x.0).collect();
    unique_ancestors.sort_unstable();
    unique_ancestors.dedup();

    let metadata = json!({
        "purpose": "Positive walkback certificate reuse experiment, not an end-to-end verification benchmark",
        "checkpoint_depth": K, "walkback_max_states": CAP, "rules": rules.label(),
        "threads": rayon::current_num_threads(), "split_depth": SPLIT,
        "checkpoint_survivors_before_walkback": considered,
        "certificate_count": records.len(), "retained_checkpoint_survivors": considered - records.len() as u64,
        "record_format": "sorted n0:u64 little-endian, p:u128 little-endian, s:u32 little-endian, o:u32 little-endian; no header; 32 bytes/record",
        "binary_bytes": bytes.len(), "binary_sha256": sha256(&bin_path),
        "rules_setup_seconds": rules_seconds, "enumerate_and_search_seconds": build_seconds,
        "sort_seconds": sort_seconds, "write_seconds": write_seconds,
        "verify_all_records_seconds": verify_seconds,
        "verification": {"records_checked": records.len(), "forward_orbit_matches": true,
            "odd_counts_match": true, "positive_smaller_partners": true,
            "exact_slope_inequalities": true, "translated_members_per_record": 3},
        "independent_queries": {"n": 72, "a": 10, "endpoint_depth": M,
            "baseline_uniform_draws": DRAWS, "seed_hex": format!("{SEED:016x}"),
            "improved_survivor_draws": kept.len(), "unique_ancestors": unique_ancestors.len(),
            "cache_hits": hits, "mismatches_vs_fresh_search": mismatches,
            "sampling_and_ancestor_seconds": sample_seconds,
            "single_thread_lookup_seconds": lookup_seconds,
            "single_thread_fresh_search_seconds": fresh_seconds,
            "timing_note": "single pass on the same sampled ancestors; includes neither production integration nor tail work"},
        "elapsed_seconds": all_start.elapsed().as_secs_f64()
    });
    let out = serde_json::to_string_pretty(&metadata).unwrap();
    fs::write(out_dir.join("metadata.json"), format!("{out}\n")).unwrap();
    println!("{out}");
}
