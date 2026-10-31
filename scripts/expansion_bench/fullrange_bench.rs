//! Fixed-draw checkpoint cohorts for extrapolating the live N72 GPU pipeline.
//!
//! Unlike transport_bench, sample size counts baseline draws, including rejected
//! draws and duplicates. Every baseline K38 residue has the same inclusion
//! probability. A rejected draw contributes zero downstream work; an accepted
//! duplicate is dispatched again. Timing repetitions measure noise, while the
//! independent cohorts measure sampling uncertainty.
//!
//! n73-port: N is a parameter (positional argument 8, default 72) of the target,
//! census, validate-wide and validate-device profiles; the complete-small-N
//! `validate` profile takes CV_VALIDATE_N (default 32).  M = N - A must lie in
//! 60..=64 (the parent transport's u64 n0 and device width M - 52 <= 12).
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
use cv::tail::{Screen, TailParams, TailStats};
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

// Optional activity tracing for the warm, concatenated corpus only. These
// host API calls do not instrument or rebuild any CUDA kernel. Keep profiler
// start/stop outside the trial timers, and leave ordinary runs unchanged.
fn profile_aggregate_start(enabled: bool) {
    if enabled {
        unsafe extern "C" { fn cudaProfilerStart() -> i32; }
        assert_eq!(unsafe { cudaProfilerStart() }, 0, "cudaProfilerStart failed");
    }
}

fn profile_aggregate_stop(enabled: bool) {
    if enabled {
        unsafe extern "C" { fn cudaProfilerStop() -> i32; }
        assert_eq!(unsafe { cudaProfilerStop() }, 0, "cudaProfilerStop failed");
    }
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

// Jump tails redefine iterations, checksum and (conservative) flags; they must
// keep the tree statistics and per-class endpoints and cases exactly.
// Warp-queue variants jq/jq2/jq3 (no suffix) run the jx codes with the same
// per-case semantics, so their full signature must equal jx's.  Every other
// tail variant is its own signature family.
/// c3-a: work-preserving per-variant pipeline options appended to a variant name:
///   `<variant>[~s<streams>][~b<log2 endpoints per launch>][~t<producer threads>][~e<parent entry depth>]`
/// e.g. `fxr2_top~s2~b19`.  Streams <= the session's slots (CV_GPU_SLOTS, default 3); launch size a power of
/// two in [2^14, 2^20]; producer threads run the CPU expansion in a dedicated rayon pool; the entry depth moves
/// the CPU/device boundary of the baseline-only levels (the full-check cutoff, CV_FULLRANGE_CUTOFF, is unchanged,
/// so trees, endpoints and cases are identical; the per-cohort work and signature asserts check this).
/// Target profile only.  The trial record keeps the full name.
#[derive(Clone, Copy, Default, Debug)]
struct RunOpts {
    slots: u32,
    batch_log2: u32,
    threads: usize,
    entry: u32,
}

fn split_opts(name: &str) -> (&str, RunOpts) {
    let mut it = name.split('~');
    let base = it.next().unwrap();
    let mut o = RunOpts::default();
    for part in it {
        assert!(!part.is_empty(), "empty option in {name}");
        let (k, v) = part.split_at(1);
        let v: u32 = v.parse().unwrap_or_else(|_| panic!("bad option ~{part} in {name}"));
        match k {
            "s" => { assert!((1..=8).contains(&v), "~s<streams> in 1..=8"); o.slots = v }
            "b" => { assert!((14..=20).contains(&v), "~b<log2 batch> in 14..=20"); o.batch_log2 = v }
            "t" => { assert!((1..=8).contains(&v), "~t<threads> in 1..=8"); o.threads = v as usize }
            "e" => o.entry = v,
            _ => panic!("unknown option ~{part} in {name}"),
        }
    }
    (base, o)
}

fn base_name(name: &str) -> &str {
    name.split('~').next().unwrap()
}

struct Extras {
    engines: std::collections::HashMap<u32, Affine>,
    pools: std::collections::HashMap<usize, rayon::ThreadPool>,
}
static EXTRAS: std::sync::OnceLock<Extras> = std::sync::OnceLock::new();

/// Build the extra entry-depth engines and producer pools the variant options ask for.
fn setup_extras(cfg: &VerifyConfig, cutoff: u32, names: &[&str]) {
    let mut engines = std::collections::HashMap::new();
    let mut pools = std::collections::HashMap::new();
    for name in names {
        let (_, o) = split_opts(name);
        if o.entry > 0 && !engines.contains_key(&o.entry) {
            assert!(o.entry >= cutoff && o.entry < cfg.m() && cfg.m() - o.entry <= 12,
                "~e{}: entry must be in [cutoff {cutoff}, M) with M - entry <= 12", o.entry);
            engines.insert(o.entry, Affine::new(cfg.m(), cutoff, o.entry, &cfg.rules));
        }
        if o.threads > 0 && !pools.contains_key(&o.threads) {
            pools.insert(o.threads, rayon::ThreadPoolBuilder::new().num_threads(o.threads).build().unwrap());
        }
    }
    assert!(EXTRAS.set(Extras { engines, pools }).is_ok());
}

fn engine_for<'a>(affine: &'a Affine, o: &RunOpts) -> &'a Affine {
    if o.entry == 0 {
        return affine;
    }
    &EXTRAS.get().expect("variant options need setup_extras").engines[&o.entry]
}

fn family(name: &str) -> &str {
    let name = base_name(name);
    if matches!(name, "jq" | "jq2" | "jq3") { "jx" } else if tail_spec(name, 10).is_some() { name } else { "base" }
}

/// c-final: tail variants (per-class tail codes, optional per-class warp-queue
/// scheduler bits, explicit completion screen):
///   <fam>[2|3][r[N]][_top|_bot][_dload|_dscan|_dsetup|_pf|_lz]
/// fam = jx (jump tails, CV_JX / CV_JX2 / CV_JX3 codes; c-screen),
///       jd (jd window-dip tails, CV_JD / CV_JD2 / CV_JD3 codes; c-tail),
///       jq (CV_JX codes + warp-queue specs CV_JQ_WQ / CV_JQ2_WQ / CV_JQ3_WQ; c-sched),
///       fx (final: CV_FX / CV_FX2 / CV_FX3 codes, optional warp-queue specs
///           CV_FX_WQ / CV_FX2_WQ / CV_FX3_WQ; unset = per-thread kernels).
/// `r[N]` = (12, 9) row table instead of mod 9 (for classes >= N; N = 0 if
/// omitted); `_top`/`_bot` = completions h >= 2^(A-1) / h < 2^(A-1) (n73-port:
/// also `_u2`/`_l2`/`_m2`, see range_screen; and a trailing `_c<bits>` cutoff); `_d*` =
/// diagnostic kernels (not verifications: load records only / screen only /
/// screen + case setup); `_pf` / `_lz` / `_lb` = work-preserving scan options
/// (mod-9 word prefetch / lazy row words / 48-register launch bound of the jd
/// rows kernels).  Warp-queue specs are
/// v<layout>[.k<K>][.c<log2 fetch>][.o<blocks/SM>] or `-` (per-thread), one for
/// all classes or one per class.
#[derive(Clone, Copy)]
struct TailSpec {
    key: &'static str,
    wq_key: Option<&'static str>,
    screen: Screen,
    diag: u32,
}

/// r3pr-gpu (as pr-measure / v-pr-cost): optional per-variant arithmetic
/// cutoff suffix `_c<bits>` (e.g. `fx3r0_c118~s2`, `fx3r0_top_c123`).  It sets
/// `TailParams::cutoff_bits` for that variant only (host side; the cutoff
/// reaches the kernels as KParams.cutoff_hi).  A distinct name is a distinct
/// signature family (iterations, hash and flags depend on the cutoff), while
/// tree, endpoints and cases must still match (asserted per cohort).
/// n73-port: copied from r3pr-gpu's tree (host only).
fn split_cutoff(name: &str) -> (&str, Option<u32>) {
    if let Some(pos) = name.rfind("_c") {
        let digits = &name[pos + 2..];
        if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
            return (&name[..pos], Some(digits.parse().unwrap()));
        }
    }
    (name, None)
}

/// n73-port: completion ranges of the `_top/_bot/_u2/_l2/_m2` suffixes, relative to
/// 2^N (so a name means the same range at every N and A):
///   `_top` h >= 2^(A-1)               n in [2^(N-1), 2^N)    (N73: [2^72, 2^73))
///   `_bot` h <  2^(A-1)               n <  2^(N-1)
///   `_u2`  h >= 2^(A-2)               n in [2^(N-2), 2^N)    (N73: [2^71, 2^73))
///   `_l2`  h <  2^(A-2)               n <  2^(N-2)
///   `_m2`  2^(A-2) <= h < 2^(A-1)     n in [2^(N-2), 2^(N-1)) (N73: [2^71, 2^72))
/// Partitions: full = _l2 + _u2, _u2 = _m2 + _top, _bot = _l2 + _m2.  The quarter
/// ranges need A >= 8 (64-completion words); Screen::check refuses less at run time.
fn range_screen(suffix: &str, a: u32) -> Option<Screen> {
    let q = |lo: u32, hi: u32| Screen { h_lo: lo, h_hi: hi, ..Screen::default() };
    let a2 = if a >= 2 { 1u32 << (a - 2) } else { 0 };
    Some(match suffix {
        "_top" => Screen::top(a),
        "_bot" => Screen::bottom(a),
        "_u2" => q(a2, 0),
        "_l2" => q(0, a2),
        "_m2" => q(a2, 2 * a2),
        _ => return None,
    })
}

fn tail_spec(name: &str, a: u32) -> Option<TailSpec> {
    let (name, _) = split_cutoff(name);
    let fam = &name[..name.len().min(2)];
    let mut rest = &name[fam.len()..];
    let digit = match rest.as_bytes().first() {
        Some(b'2') => { rest = &rest[1..]; 2 }
        Some(b'3') => { rest = &rest[1..]; 3 }
        _ => 1,
    };
    let (key, wq_key): (&'static str, Option<&'static str>) = match (fam, digit) {
        ("jx", 1) => ("CV_JX", None), ("jx", 2) => ("CV_JX2", None), ("jx", 3) => ("CV_JX3", None),
        ("jd", 1) => ("CV_JD", None), ("jd", 2) => ("CV_JD2", None), ("jd", 3) => ("CV_JD3", None),
        ("jq", 1) => ("CV_JX", Some("CV_JQ_WQ")), ("jq", 2) => ("CV_JX", Some("CV_JQ2_WQ")),
        ("jq", 3) => ("CV_JX", Some("CV_JQ3_WQ")),
        ("fx", 1) => ("CV_FX", Some("CV_FX_WQ")), ("fx", 2) => ("CV_FX2", Some("CV_FX2_WQ")),
        ("fx", 3) => ("CV_FX3", Some("CV_FX3_WQ")),
        _ => return None,
    };
    let mut screen = Screen::default();
    if let Some(r) = rest.strip_prefix('r') {
        // r[N]: rows for bitvector classes >= N (default all)
        let digits: String = r.chars().take_while(|c| c.is_ascii_digit()).collect();
        rest = &r[digits.len()..];
        screen.rows = true;
        screen.rows_from = if digits.is_empty() { 0 } else { digits.parse().ok()? };
    }
    // n73-port: `_top` / `_bot` plus the quarter ranges `_u2` / `_l2` / `_m2` (range_screen).
    for suffix in ["_top", "_bot", "_u2", "_l2", "_m2"] {
        if let Some(r) = rest.strip_prefix(suffix) {
            rest = r;
            let range = range_screen(suffix, a).unwrap();
            screen = Screen { rows: screen.rows, rows_from: screen.rows_from, ..range };
            break;
        }
    }
    let diag = match rest {
        "" => 0,
        "_dload" => cv::gpu::SCREEN_DIAG_LOAD,
        "_dscan" => cv::gpu::SCREEN_DIAG_SCAN,
        "_dsetup" => cv::gpu::SCREEN_DIAG_SETUP,
        // work-preserving scan options (verifications): mod-9 prefetch / lazy rows
        "_pf" => cv::gpu::SCREEN_OPT_M9PF,
        "_lz" => cv::gpu::SCREEN_OPT_LAZY,
        // c-final: jd rows kernels with the 48-register launch bound
        "_lb" => cv::gpu::SCREEN_OPT_LB5,
        _ => return None,
    };
    Some(TailSpec { key, wq_key, screen, diag })
}

fn is_diagnostic(name: &str) -> bool {
    let name = base_name(name);
    tail_spec(name, 10).is_some_and(|s| matches!(s.diag, cv::gpu::SCREEN_DIAG_LOAD | cv::gpu::SCREEN_DIAG_SCAN | cv::gpu::SCREEN_DIAG_SETUP))
}

/// Screen of a variant (default for every non-tail variant).
fn screen_of(name: &str, a: u32) -> Screen {
    let name = base_name(name);
    tail_spec(name, a).map_or(Screen::default(), |s| s.screen)
}

/// Per-class tail codes of a tail variant (from its environment key).
fn tail_codes(spec: &TailSpec) -> Vec<u32> {
    let default = "33,35,35,35,35,35,35,43,43,43,43,43,43,43,43,43";
    let codes = std::env::var(spec.key).unwrap_or_else(|_| {
        assert!(spec.key.starts_with("CV_JX"), "{} must list the per-class tail codes", spec.key);
        default.into()
    });
    codes.split(',').map(|c| c.trim().parse().unwrap()).collect()
}

/// Warp-queue bits of a tail variant (one spec for every class or one per
/// class; classes beyond the list and classes with code 0 stay per-thread).
fn wq_bits(spec: &TailSpec, codes: &[u32]) -> [u32; 32] {
    let mut bits = [0u32; 32];
    if let Some(wk) = spec.wq_key {
        match std::env::var(wk) {
            Ok(specs) => {
                let specs: Vec<&str> = specs.split(',').collect();
                assert!(specs.len() <= 32, "{wk}: at most 32 specs");
                for (i, b) in bits.iter_mut().enumerate() {
                    let sp = if specs.len() == 1 { Some(specs[0]) } else { specs.get(i).copied() };
                    let code = if codes.len() == 1 { codes.first() } else { codes.get(i) }.copied().unwrap_or(0);
                    *b = match sp {
                        Some(sp) if code != 0 => cv::gpu::parse_wq(sp).unwrap(),
                        _ => 0,
                    };
                }
            }
            Err(_) => assert!(spec.key.starts_with("CV_FX"), "{} needs {wk}", spec.key),
        }
    }
    bits
}

/// c3-hyg: the TailParams of a variant on `cfg`: its screen, per-class tail
/// codes and warp-queue schedule.  The CPU reference (validation profiles,
/// CPU-only classes) and the GPU tuning are both built from this value;
/// gpu_run refuses a tuning that differs.  Floor codes are allowed here
/// (benchmark), and make the run measurement-only.
fn variant_tail(name: &str, cfg: &VerifyConfig) -> TailParams {
    let mut tail = cfg.tail;
    if let Some(spec) = tail_spec(name, cfg.a) {
        let codes = tail_codes(&spec);
        tail.screen = spec.screen;
        tail.codes = cv::runspec::TailCodes::measurement(&codes, cfg.num_bv)
            .unwrap_or_else(|e| panic!("{name}: {} = {codes:?}: {e:#}", spec.key));
        let bits = wq_bits(&spec, &codes);
        tail.schedule = cv::runspec::WqSchedule::from_bits(bits).unwrap();
        // r3pr-gpu: per-variant arithmetic cutoff (`_c<bits>`).
        if let Some(bits) = split_cutoff(name).1 {
            tail.cutoff_bits = bits;
        }
    }
    tail
}

/// Does a variant need the measurement-only opt-in (diagnostic kernels or
/// jd floor codes)?
fn needs_measurement(name: &str) -> bool {
    let name = base_name(name);
    is_diagnostic(name)
        || tail_spec(name, 10).is_some_and(|s| tail_codes(&s).iter().any(|&c| cv::runspec::is_floor_code(c)))
}

/// Tree and endpoints must be identical for every variant; cases only within
/// the same screen (rows / h range change the case set by design).
fn work_signature(report: &VerifyReport) -> Value {
    json!({"tree":report.tree,"endpoints":report.tail.endpoints})
}

fn case_signature(report: &VerifyReport) -> Value {
    json!({"cases":report.tail.cases})
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

#[allow(clippy::too_many_arguments)]
fn cohort(
    index: usize,
    count: usize,
    seed: u64,
    sampler: &Sampler,
    descendants: &BackwardDp,
    extra_depths: &[(u32, BackwardDp)],
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
    // n73-port: the same diagnostic at fixed depths (CV_FULLRANGE_CV_DEPTHS), so
    // processes of different A (different M) can share one control variate.
    let at: serde_json::Map<String, Value> = extra_depths
        .iter()
        .map(|(d, dp)| {
            let s: u128 = draws.iter().map(|draw| dp.ways(PState::of(&draw.node))).sum();
            (d.to_string(), json!(s.to_string()))
        })
        .collect();
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
        "baseline_descendants_at":at,
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
        "defer8" => (8, 65_536, 1),
        "defer16" => (16, 65_536, 1),
        "defer24" | "compact_defer24" => (24, 65_536, 1),
        "defer32" | "compact_defer32" => (32, 65_536, 1),
        "defer64" => (64, 65_536, 1),
        "defer128" | "compact_defer128" => (128, 65_536, 1),
        "defer256" | "compact_defer256" => (256, 65_536, 1),
        "defer512" => (512, 65_536, 1),
        "defer_count" => (u32::MAX, 65_536, 1),
        "defer1_cap1" => (1, 1, 1),
        "defer16_w2" => (16, 65_536, 2),
        "defer24_w2" => (24, 65_536, 2),
        "defer24_cap262k_w2" | "compact_defer24_cap262k_w2" => (24, 262_144, 2),
        "defer24_cap262k_w4" => (24, 262_144, 4),
        "defer16_cap262k_w4" => (16, 262_144, 4),
        "defer32_w2" => (32, 65_536, 2),
        "defer64_w2" => (64, 65_536, 2),
        "defer128_w2" => (128, 65_536, 2),
        "defer256_w2" => (256, 65_536, 2),
        _ => return None,
    };
    Some((budget, capacity, workers))
}

// Early-screen experiments select whole launches before verification. Names
// encode the selection interval, CPU method/low-bit width and worker count.
fn early_screen(name: &str) -> Option<(u32, u32, u32, u32)> {
    let body = name.strip_prefix("early")?;
    let mut fields = body.split('_');
    let every: u32 = fields.next()?.parse().ok()?;
    let method = fields.next()?;
    let workers: u32 = fields.next()?.strip_prefix('w')?.parse().ok()?;
    if fields.next().is_some() || every == 0 || !(1..=8).contains(&workers) {
        return None;
    }
    let (mode, width) = if method == "scalar" {
        (0, 10)
    } else {
        let (mode, bits) = if let Some(bits) = method.strip_prefix("bucket") { (1, bits) }
            else if let Some(bits) = method.strip_prefix("table") { (2, bits) }
            else if let Some(bits) = method.strip_prefix("group") { (3, bits) }
            else if let Some(bits) = method.strip_prefix("regroup") { (4, bits) }
            else { return None; };
        (mode, bits.parse().ok()?)
    };
    if ![8, 10, 12, 14].contains(&width) { return None; }
    Some((every, mode, width, workers))
}

fn compact_chain_index(name: &str) -> Option<usize> {
    match name {
        "compact66" | "compact_defer24" | "compact_defer32" | "compact_defer24_cap262k_w2"
        | "compact_defer128" | "compact_defer256" => Some(0),
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
    let (name, opts) = split_opts(name);
    let affine = engine_for(affine, &opts);
    let name = name.strip_suffix("_1k").unwrap_or(name);
    if let Some(spec) = tail_spec(name, 10) {
        // jd dip tables: built once per process (host), uploaded into this
        // session's context before the timed run (c3-hyg: per context).
        let codes: Vec<u32> = tail_codes(&spec).into_iter().filter(|&c| c >= 64).collect();
        gpu.prepare_jd_tables(&codes).expect("jd table upload");
    }
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
    if early_screen(name).is_some() {
        gpu.prepare_early_screen().expect("early screening buffer allocation");
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
    let (name, opts) = split_opts(name);
    let affine = engine_for(affine, &opts);
    let words = affine.table_words();
    let mut tuning = GpuTuning::default();
    tuning.run_slots = opts.slots;
    tuning.run_batch = if opts.batch_log2 > 0 { 1 << opts.batch_log2 } else { 0 };
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
    ) || jump_width(name) > 0 || tail_offload(name).is_some() || early_screen(name).is_some();
    if experiment {
        tuning.overflow_vote = false;
        tuning.readonly_powers = false;
        tuning.oee_mask = true;
        tuning.tail_jump_bits = jump_width(name);
    }
    let mut variant_cfg = None;
    if let Some(spec) = tail_spec(name, cfg.a) {
        // Experimental tails: novote_oee plus per-class variant codes.
        tuning.overflow_vote = false;
        tuning.readonly_powers = false;
        tuning.oee_mask = true;
        // c3-hyg: codes, warp-queue bits and screen from the variant's
        // TailParams (CPU fallback and GPU alike).
        let mut c = cfg.clone();
        c.tail = variant_tail(name, cfg);
        tuning.xtail = c.tail.codes.gpu_xtail();
        tuning.wq = c.tail.schedule.bits();
        tuning.screen_diag = spec.diag;
        variant_cfg = Some(c);
    }
    let cfg = variant_cfg.as_ref().unwrap_or(cfg);
    if let Some((budget, capacity, workers)) = tail_offload(name) {
        tuning.tail_budget = budget;
        tuning.tail_queue_capacity = capacity;
        tuning.tail_cpu_workers = workers;
    }
    if let Some((every, mode, width, workers)) = early_screen(name) {
        tuning.early_every = every;
        tuning.early_mode = mode;
        tuning.early_group_bits = width;
        tuning.early_cpu_workers = workers;
    }
    let pool = (opts.threads > 0).then(|| &EXTRAS.get().expect("variant options need setup_extras").pools[&opts.threads]);
    let mut run = move || match name {
        "early_parents" => gpu.run_transport_tuned(cfg, &compressed.early, GpuTransport::Parents,
                                                   compressed.early.table_words(), tuning),
        "compact66" | "compact48" | "compact_defer24" | "compact_defer32" | "compact_defer24_cap262k_w2"
        | "compact_defer128" | "compact_defer256" => {
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
        name if tail_spec(name, cfg.a).is_some() => {
            gpu.run_transport_tuned(cfg, affine, GpuTransport::Parents, words, tuning)
        }
        name if tail_offload(name).is_some() || early_screen(name).is_some() => {
            gpu.run_transport_tuned(cfg, affine, GpuTransport::Parents, words, tuning)
        }
        _ => panic!("unknown variant {name}"),
    };
    match pool {
        Some(p) => p.install(run),
        None => run(),
    }
    .expect("GPU full-range cohort run")
}

fn validate_tail_queue(name: &str, report: &VerifyReport) {
    if let Some((every, _, _, _)) = early_screen(name) {
        let gpu = report.gpu.as_ref().unwrap();
        assert!(gpu["early_batches"].as_u64().unwrap() > 0,
                "early-screen validation must exercise actual exports");
        assert!(gpu["early_cases"].as_u64().unwrap() > 0,
                "early-screen validation must verify actual CPU survivors");
        if every == 1 {
            assert_eq!(gpu["regular_batches"].as_u64().unwrap(), 0,
                       "all-screen variant unexpectedly used the ordinary verifier");
        }
    }
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

fn validate(wide: bool, n: u32, a: u32, names: &[&str]) {
    // n73-port: validate-wide runs N (argument 8, default 72); the complete
    // small-N profile runs CV_VALIDATE_N (default 32).
    let small_n = env_number("CV_VALIDATE_N", 32) as u32;
    assert!((a + 20..=40).contains(&small_n), "CV_VALIDATE_N in A + 20..=40 (entry 18 < M)");
    let mut cfg = config(if wide { n } else { small_n }, a, 12, 2);
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
    // c-screen / c3-hyg: one CPU reference per variant, with the variant's own
    // TailParams (rows / h range / per-class tail codes), the value its GPU
    // tuning is built from.
    let cpu_for = |name: &str| {
        let mut c = cfg.clone();
        c.tail = variant_tail(name, &cfg);
        let cpu = verify_cpu_with_expander(&c, &compact);
        let affine_cpu = verify_cpu_with_expander(&c, &affine);
        assert_eq!(signature(&cpu), signature(&affine_cpu));
        cpu
    };
    let mut session_cfg = cfg.clone();
    session_cfg.tail.screen.rows = names.iter().any(|n| screen_of(n, cfg.a).rows);
    let mut gpu = GpuSession::new_experimental(&session_cfg).expect("GPU initialization");
    if names.iter().any(|n| needs_measurement(n)) {
        gpu.allow_measurement_only().expect("measurement opt-in");
    }
    gpu.set_terminal_table(affine.table_words())
        .expect("table upload");
    prepare_jumps(&mut gpu, &names);
    for name in names {
        let cpu = cpu_for(name);
        prepare_variant(&mut gpu, &affine, &compressed, name);
        let report = run_variant(&mut gpu, &cfg, &compact, &affine, &chains, &compressed, name);
        assert_eq!(signature(&cpu), signature(&report), "validation {name}");
        require_resolved(&report);
        validate_tail_queue(name, &report);
        assert!(report.tail.endpoints[cfg.num_bv] > 0);
        println!(
            "{}",
            json!({"type":"validation","variant":name,
            "scope":if wide { format!("sixteen K52 packages, N{} with CPU fallback", cfg.n) } else { format!("complete N{} with CPU fallback", cfg.n) },
            "n":cfg.n,"a":cfg.a,"m":cfg.m(),
            "h_range":cv::runspec::h_range(&screen_of(name, cfg.a), cfg.n, cfg.a),
            "screen":screen_of(name, cfg.a),
            "cpu_gpu_equal":true,"roots":cfg.units,"report":report})
        );
    }
}

fn validate_overflow(n: u32, a: u32, names: &[&str]) {
    // c3-a: A is a parameter (was fixed at 10); 24 GPU classes at any A in 8..=12.
    // n73-port: N is a parameter too (argument 8, default 72).
    let mut cfg = config(n, a, 12, 24);
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
    let cpu_for = |name: &str| {
        let mut c = cfg.clone();
        c.tail = variant_tail(name, &cfg);
        let cpu = verify_cpu_with_expander(&c, &compact);
        let affine_cpu = verify_cpu_with_expander(&c, &affine);
        assert_eq!(signature(&cpu), signature(&affine_cpu));
        assert!(
            !cpu.flagged.is_empty(),
            "overflow audit must exercise flag path"
        );
        assert_eq!(cpu.flagged.len(), cpu.flagged_rechecked);
        cpu
    };
    let mut session_cfg = cfg.clone();
    session_cfg.tail.screen.rows = names.iter().any(|n| screen_of(n, cfg.a).rows);
    let mut gpu = GpuSession::new_experimental(&session_cfg).expect("GPU initialization");
    if names.iter().any(|n| needs_measurement(n)) {
        gpu.allow_measurement_only().expect("measurement opt-in");
    }
    gpu.set_terminal_table(affine.table_words())
        .expect("table upload");
    prepare_jumps(&mut gpu, &names);
    for name in names {
        let cpu = cpu_for(name);
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
        if early_screen(name).is_some_and(|(every, _, _, _)| every == 1) {
            assert!(report.gpu.as_ref().unwrap()["early_flags"].as_u64().unwrap() > 0,
                    "all-screen overflow fixture must exercise flagged early CPU verification");
        }
        if *name == "defer1" {
            assert!(report.gpu.as_ref().unwrap()["deferred_flags"].as_u64().unwrap() > 0,
                    "budget-one overflow fixture must exercise flagged CPU continuation");
        }
        println!(
            "{}",
            json!({"type":"validation","variant":name,"screen":screen_of(name, cfg.a),
            "scope":format!("N{} forced GPU overflow, cutoff110, high-F all-one K52 root plus sixteen random K52 roots", cfg.n),
            "n":cfg.n,"a":cfg.a,"m":cfg.m(),
            "h_range":cv::runspec::h_range(&screen_of(name, cfg.a), cfg.n, cfg.a),
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
    // n73-port: N (argument 8; default 72, so every earlier command line is unchanged).
    let n = number(8, 72) as u32;
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
    // c3-a: A in 8..=12 (M = 72 - A in 60..=64: parent transport needs M <= 64,
    // entry 52 + device width M - 52 <= 12; the warp queue screens 2^(A-10) chunks).
    // n73-port: the same limits for any N: A in 8..=12 and M = N - A in 60..=64
    // (N73: A = 9..=12, M = 64..=61; N72: A = 8..=12).
    assert!((8..=12).contains(&a), "A must be in 8..=12");
    assert!(n > a && (60..=64).contains(&(n - a)), "M = N - A must be in 60..=64 (N = {n}, A = {a})");
    assert!(threads > 0 && !names.is_empty());
    assert_eq!(names.iter().collect::<HashSet<_>>().len(), names.len());
    let full_names = names.clone();
    let names: Vec<&str> = names.iter().map(|n| base_name(n)).collect();
    assert!(names.iter().all(|name| tail_offload(name).is_some() || early_screen(name).is_some() || matches!(
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
    ) || tail_spec(name, a).is_some()));
    let names = full_names;
    assert!(profile == "target" || names.iter().all(|n| !n.contains('~')),
        "variant options (~s ~b ~t ~e) are supported by the target profile only");
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
        validate_overflow(n, a, &names);
        return;
    }
    if matches!(profile, "validate" | "validate-wide") {
        validate(profile == "validate-wide", n, a, &names);
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
    let profile_aggregate = env_number("CV_FULLRANGE_PROFILE_AGGREGATE", 0);
    assert!(profile_aggregate <= 1, "CV_FULLRANGE_PROFILE_AGGREGATE must be 0 or 1");
    let profile_aggregate = profile_aggregate == 1;
    assert!(drawn > 0 && repetitions > 0 && count > 0);
    assert!(CHECKPOINT <= cutoff && cutoff <= entry);
    let mut cfg = config(n, a, 27, 16);
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
    setup_extras(&cfg, cutoff, &names);
    let engine_setup_seconds = started.elapsed().as_secs_f64();
    let baseline = Rules::baseline();
    let sampler = Sampler::new(&baseline, CHECKPOINT, None);
    let descendants = BackwardDp::run(ParityRules::of(&baseline), 2, cfg.m(), None);
    let baseline_endpoint_population = descendants.ways(PState::of(&Node::root(&baseline)));
    // n73-port: optional fixed control-variate depths (CV_FULLRANGE_CV_DEPTHS = "62,63"),
    // reported per cohort as baseline_descendants_at and here with their exact populations.
    let extra_depths: Vec<(u32, BackwardDp)> = std::env::var("CV_FULLRANGE_CV_DEPTHS")
        .map(|v| v.split(',').map(|d| {
            let d: u32 = d.trim().parse().expect("CV_FULLRANGE_CV_DEPTHS: depths");
            assert!((CHECKPOINT..=80).contains(&d), "CV_FULLRANGE_CV_DEPTHS: depths in {CHECKPOINT}..=80");
            (d, BackwardDp::run(ParityRules::of(&baseline), 2, d, None))
        }).collect())
        .unwrap_or_default();
    let population_at: serde_json::Map<String, Value> = extra_depths.iter()
        .map(|(d, dp)| (d.to_string(), json!(dp.ways(PState::of(&Node::root(&baseline))).to_string())))
        .collect();
    println!(
        "{}",
        json!({
            "type":"config","experiment":"fullrange","profile":profile,"n":cfg.n,"a":a,"m":cfg.m(),
            "checkpoint":CHECKPOINT,"cutoff":cutoff,"terminal_entry":entry,
            "terminal_width":cfg.m()-entry,"walkback_cap":WALKBACK_CAP,
            "sampling_design":"fixed_baseline_draws_with_replacement",
            "baseline_population":sampler.population().to_string(),
            "baseline_endpoint_population":baseline_endpoint_population.to_string(),
            "baseline_endpoint_population_at":population_at,
            "cohort_count":count,"draws_per_cohort":drawn,"seed":seed,
            "repetitions":repetitions,"aggregate_repetitions":aggregate_repetitions,
            "profile_aggregate":profile_aggregate,
            "threads":threads,"num_bv":cfg.num_bv,"class_b":cfg.class_b,
            "window":cfg.window.label(),"rules":cfg.rules.label(),"variants":names,
            "early_capture_requested":std::env::var_os("CV_EARLY_CAPTURE").is_some(),
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
            &extra_depths,
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
    // c-screen: the session carries the row table when any variant uses it.
    let mut session_cfg = cfg.clone();
    session_cfg.tail.screen.rows = names.iter().any(|n| screen_of(n, a).rows);
    let mut gpu = GpuSession::new_experimental(&session_cfg).expect("GPU initialization");
    // c3-hyg: diagnostic variants and jd floor codes are measurement-only
    // kernels; their reports say measurement_only.
    if names.iter().any(|n| needs_measurement(n)) {
        gpu.allow_measurement_only().expect("measurement opt-in");
    }
    gpu.set_terminal_table(affine.table_words())
        .expect("table upload");
    prepare_jumps(&mut gpu, &names);
    println!(
        "{}",
        json!({"type":"gpu_setup","wall_seconds":started.elapsed().as_secs_f64(),
        "process_cpu_seconds":cpu_seconds()-cpu,
        "row_table":session_cfg.tail.screen.rows,
        "variant_screens":names.iter().map(|n| json!({"variant":n,"screen":screen_of(n, a),"diagnostic":is_diagnostic(n)})).collect::<Vec<_>>()})
    );
    cfg.units = Some(
        cohorts
            .iter()
            .flat_map(|c| c.roots.iter())
            .take(32)
            .copied()
            .collect(),
    );
    let mut warmup_signature: std::collections::HashMap<String, Value> = Default::default();
    let mut warmup_work = None;
    let mut warmup_cases: std::collections::HashMap<String, Value> = Default::default();
    for name in &names {
        prepare_variant(&mut gpu, &affine, &compressed, name);
        let report = run_variant(&mut gpu, &cfg, &compact, &affine, &chains, &compressed, name);
        require_resolved(&report);
        if let Some(want) = warmup_signature.get(family(name)) {
            assert_eq!(*want, signature(&report), "warmup {name}");
        } else {
            warmup_signature.insert(family(name).to_string(), signature(&report));
        }
        if let Some(want) = &warmup_work {
            assert_eq!(*want, work_signature(&report), "warmup work {name}");
        } else {
            warmup_work = Some(work_signature(&report));
        }
        if !is_diagnostic(name) {
            let key = screen_of(name, a).label();
            if let Some(want) = warmup_cases.get(&key) {
                assert_eq!(*want, case_signature(&report), "warmup cases {name}");
            } else {
                warmup_cases.insert(key, case_signature(&report));
            }
        }
        println!(
            "{}",
            json!({"type":"warmup","variant":name,"report":report})
        );
    }
    let mut expected: Vec<Option<Value>> = vec![None; count];
    let mut expected_family: std::collections::HashMap<(usize, String), Value> = Default::default();
    let mut expected_work: Vec<Option<Value>> = vec![None; count];
    let mut expected_cases: std::collections::HashMap<(usize, String), Value> = Default::default();
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
                if family(name) == "base" {
                    if let Some(want) = &expected[index] {
                        assert_eq!(*want, signature(&report), "cohort {index} variant {name}");
                    } else {
                        expected[index] = Some(signature(&report));
                    }
                } else if let Some(want) = expected_family.get(&(index, family(name).to_string())) {
                    assert_eq!(*want, signature(&report), "cohort {index} variant {name}");
                } else {
                    expected_family.insert((index, family(name).to_string()), signature(&report));
                }
                if let Some(want) = &expected_work[index] {
                    assert_eq!(*want, work_signature(&report), "cohort {index} work {name}");
                } else {
                    expected_work[index] = Some(work_signature(&report));
                }
                if !is_diagnostic(name) {
                    let key = (index, screen_of(name, a).label());
                    if let Some(want) = expected_cases.get(&key) {
                        assert_eq!(*want, case_signature(&report), "cohort {index} cases {name}");
                    } else {
                        expected_cases.insert(key, case_signature(&report));
                    }
                }
                println!(
                    "{}",
                    json!({"type":"trial","cohort":index,"variant":name,
                    "screen":screen_of(name, a),"diagnostic":is_diagnostic(name),
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
        // Per signature family (base, or a jump-tail family whose iterations,
        // hash and conservative flags are redefined): the combined cohort
        // signature that the concatenated run must reproduce exactly.
        let mut combined_signatures: std::collections::HashMap<String, Value> = Default::default();
        for name in &names {
            let fam = family(name).to_string();
            if combined_signatures.contains_key(&fam) {
                continue;
            }
            let parts: Vec<Option<Value>> = if fam == "base" {
                expected.clone()
            } else {
                (0..count).map(|i| expected_family.get(&(i, fam.clone())).cloned()).collect()
            };
            combined_signatures.insert(fam, aggregate_signature(&parts, cfg.m(), cfg.num_bv));
        }
        let mut aggregate_expected: std::collections::HashMap<String, Value> = Default::default();
        for repetition in 0..aggregate_repetitions {
            for order_position in 0..names.len() {
                let name = names[(order_position + repetition) % names.len()];
                let preparation = Instant::now();
                prepare_variant(&mut gpu, &affine, &compressed, name);
                let variant_prepare_seconds = preparation.elapsed().as_secs_f64();
                profile_aggregate_start(profile_aggregate);
                let cpu = cpu_seconds();
                let started = Instant::now();
                let report = run_variant(&mut gpu, &cfg, &compact, &affine, &chains, &compressed, name);
                let wall = started.elapsed().as_secs_f64();
                let cpu = cpu_seconds() - cpu;
                profile_aggregate_stop(profile_aggregate);
                require_resolved(&report);
                let fam = family(name).to_string();
                if let Some(want) = aggregate_expected.get(&fam) {
                    assert_eq!(*want, signature(&report), "aggregate {name}");
                } else {
                    aggregate_expected.insert(fam.clone(), signature(&report));
                }
                // Every tree/per-class counter, XOR hash, flagged-origin
                // multiset and recheck total must compose across cohorts.
                assert_eq!(
                    signature(&report),
                    combined_signatures[&fam],
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
    fn quarter_ranges_partition_the_completions_and_parse_with_cutoffs() {
        // n73-port: full = _l2 + _u2, _u2 = _m2 + _top, _bot = _l2 + _m2 (64-completion words).
        for a in 8..=12u32 {
            let w = |s: &str| {
                let sc = range_screen(s, a).unwrap();
                sc.check(a).unwrap();
                sc.word_range(a)
            };
            let all = (1usize << a) / 64;
            assert_eq!(w("_l2"), (0, all / 4));
            assert_eq!(w("_u2"), (all / 4, all));
            assert_eq!(w("_m2"), (all / 4, all / 2));
            assert_eq!(w("_top"), (all / 2, all));
            assert_eq!(w("_bot"), (0, all / 2));
        }
        assert!(range_screen("_m2", 7).unwrap().check(7).is_err(), "quarter ranges need A >= 8");
        let s = tail_spec("fx3r0_u2_c118", 10).unwrap();
        assert!(s.screen.rows && s.screen.rows_from == 0 && s.screen.h_lo == 256 && s.screen.h_hi == 0);
        assert_eq!(split_cutoff("fx3r0_u2_c118"), ("fx3r0_u2", Some(118)));
        assert_eq!(split_cutoff("fx3r0_top"), ("fx3r0_top", None));
        let s = tail_spec("fx3r0_m2", 11).unwrap();
        assert_eq!((s.screen.h_lo, s.screen.h_hi), (512, 1024));
        assert!(tail_spec("fx3r0_q2", 10).is_none());
        assert_eq!(screen_of("fx3r0_l2~s2", 12).h_hi, 1024);
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
        let sampled = cohort(0, 256, 1234, &sampler, &descendants, &[], &rules, 16);
        let same = cohort(0, 256, 1234, &sampler, &descendants, &[], &rules, 16);
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
