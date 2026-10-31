//! End-to-end drivers: full verification on the CPU, and step-1 censuses.

use crate::lookahead::{BvMode, Lookahead, WindowRules};
use crate::expansion::{DfsExpander, Expander};
use crate::rules::Rules;
use crate::tail::{CpuTailSink, TailParams, TailStats};
use crate::tree::{dfs, frontier, CountSink, Node, TreeStats};
use rayon::prelude::*;
use serde::Serialize;
use std::time::Instant;

#[derive(Clone)]
pub struct VerifyConfig {
    /// Verify all n < 2^n.
    pub n: u32,
    /// A: bits added by the bitvector stage (M = N - A).
    pub a: u32,
    /// B: lookahead window of the bitvectors.
    pub b: u32,
    pub num_bv: usize,
    pub bv_mode: BvMode,
    /// Extra merge credits baked into the bitvectors.
    pub window: WindowRules,
    /// Per-class windows for the lowest-f classes (the rest use `b`).
    pub class_b: Option<Vec<u32>>,
    /// Depth of the breadth-first split into independent work units.
    pub split: u32,
    pub rules: Rules,
    pub tail: TailParams,
    /// Run only these subtrees (a work package) instead of the whole tree.
    pub units: Option<Vec<Node>>,
}

/// Deepest tree the 128-bit node arithmetic supports (m < 3^80 < 2^127).
pub const MAX_M: u32 = 80;

impl VerifyConfig {
    /// c3-hyg: every limit the CPU and GPU paths rely on, as an error rather
    /// than an assert deep inside a worker: M <= 80, 6 <= A <= B <= 39, the
    /// cutoff (<= 123; >= 96 + K for jump / jd codes), the screen's completion
    /// range, the per-class tail codes and warp-queue specs (one per class),
    /// and the row table's needs (M >= 12, A <= 16).
    pub fn validate(&self) -> anyhow::Result<()> {
        use anyhow::ensure;
        ensure!(self.n > self.a, "N = {} must exceed A = {}", self.n, self.a);
        ensure!(self.n - self.a <= MAX_M, "M = N - A = {} exceeds {MAX_M}; raise A", self.n - self.a);
        ensure!(self.a >= 6 && self.a <= self.b && self.b <= crate::lookahead::MAX_B,
            "need 6 <= A <= B <= {} (A = {}, B = {})", crate::lookahead::MAX_B, self.a, self.b);
        ensure!(self.num_bv >= 1, "need at least one bitvector class");
        if let Some(w) = &self.class_b {
            ensure!(w.len() <= self.num_bv && w.iter().all(|&x| x >= self.a && x <= crate::lookahead::MAX_B),
                "class windows must number at most num_bv and lie in [A, {}]", crate::lookahead::MAX_B);
        }
        ensure!(self.tail.cutoff_bits <= 123, "cutoff-bits {} exceeds 123 (128-bit tail arithmetic)", self.tail.cutoff_bits);
        self.tail.screen.check(self.a)?;
        if self.tail.screen.rows {
            ensure!(self.n - self.a >= 12 && self.a <= 16, "the (12, 9) row table needs M >= 12 and A <= 16");
        }
        self.tail.codes.check(self.num_bv, self.tail.cutoff_bits)?;
        self.tail.schedule.check(&self.tail.codes, self.num_bv)?;
        Ok(())
    }

    pub fn m(&self) -> u32 {
        assert!(self.n - self.a <= MAX_M, "M = N - A = {} exceeds {MAX_M}; raise A", self.n - self.a);
        self.n - self.a
    }
    pub fn split_depth(&self) -> u32 {
        self.split.clamp(2, self.m())
    }
    /// Host arrays up to B = 30; beyond that windows are evaluated directly on
    /// the CPU and the GPU builds its arrays on the device.
    pub fn lookahead(&self) -> Lookahead {
        let la = if let Some(w) = &self.class_b {
            Lookahead::meta(self.m(), self.a, self.b, self.num_bv, self.bv_mode, &self.window).with_class_windows(w)
        } else if self.b > 30 {
            Lookahead::meta(self.m(), self.a, self.b, self.num_bv, self.bv_mode, &self.window)
        } else {
            Lookahead::build_ext(self.m(), self.a, self.b, self.num_bv, self.bv_mode, &self.window)
        };
        // c-screen: the row table is attached only when this configuration's
        // screen asks for it (explicit; cached per (M, A)).
        if self.tail.screen.rows {
            la.with_rows(Some(crate::rowsieve::RowSieve::standard(self.m(), self.a)))
        } else {
            la
        }
    }
    /// Work units: the given package, or the breadth-first split frontier.
    pub fn work_units(&self, top: &mut TreeStats) -> Vec<Node> {
        match &self.units {
            Some(u) => u.clone(),
            None => frontier(self.split_depth(), &self.rules, top),
        }
    }
}

#[derive(Serialize)]
pub struct VerifyReport {
    pub n: u32,
    pub a: u32,
    pub b: u32,
    pub num_bv: usize,
    pub bv_mode: BvMode,
    pub class_b: Option<Vec<u32>>,
    pub window: String,
    pub rules: String,
    pub backend: String,
    /// What this invocation covers: `full_range`, `selected_work_units`,
    /// `top_half`, `bottom_half`, `h_range[lo,hi)` (and combinations with
    /// `selected_work_units+`), or `measurement_only_not_a_verification`.
    /// c3-hyg: derived from `config` (the h range and the units).
    pub verification_scope: String,
    /// c3-hyg: true for benchmark runs with measurement-only kernels.
    pub measurement_only: bool,
    /// c3-hyg: the configuration record (identical for the CPU and GPU paths):
    /// h range, screen, per-class codes and windows, bitvector rules,
    /// warp-queue specs, cutoff, and the work-definition version.
    pub config: crate::runspec::RunRecord,
    pub cutoff_bits: u32,
    pub tail_oee: bool,
    pub threads: usize,
    pub units: usize,
    pub tree_nodes: u64,
    pub endpoints: u64,
    pub cases: u64,
    pub iters: u64,
    pub hash: String,
    pub flagged: Vec<String>,
    /// Flagged starts re-verified with 256-bit arithmetic.
    pub flagged_rechecked: usize,
    pub seconds: f64,
    pub tree: TreeStats,
    pub tail: TailStats,
    /// Backend-specific timings (GPU kernel time per class etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<serde_json::Value>,
    /// c3-hyg: plan runs: the work tuple of every part (package, slice).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parts: Option<Vec<crate::runspec::PartRecord>>,
}

impl VerifyReport {
    /// c3-hyg: mark a report as measurement-only (not a verification).
    pub fn mark_measurement_only(&mut self) {
        self.measurement_only = true;
        self.config.measurement_only = true;
        self.verification_scope = self.config.scope_label();
    }
}

impl VerifyReport {
    pub fn new(cfg: &VerifyConfig, backend: &str, units: usize, tree: TreeStats, tail: TailStats, seconds: f64) -> Self {
        let config = crate::runspec::RunRecord::of(cfg);
        VerifyReport {
            n: cfg.n,
            a: cfg.a,
            b: cfg.b,
            num_bv: cfg.num_bv,
            bv_mode: cfg.bv_mode,
            class_b: cfg.class_b.clone(),
            window: cfg.window.label(),
            rules: cfg.rules.label(),
            backend: backend.into(),
            verification_scope: config.scope_label(),
            measurement_only: config.measurement_only,
            config,
            cutoff_bits: cfg.tail.cutoff_bits,
            tail_oee: cfg.tail.tail_oee,
            threads: rayon::current_num_threads(),
            units,
            tree_nodes: tree.total_generated(),
            endpoints: tail.total_endpoints(),
            cases: tail.total_cases(),
            iters: tail.total_iters(),
            hash: format!("{:08X}", tail.hash),
            flagged: tail.flagged.iter().map(|x| x.to_string()).collect(),
            flagged_rechecked: tail.flagged.iter().filter(|&&x| crate::tail::recheck_wide(x).is_some()).count(),
            seconds,
            tree,
            tail,
            gpu: None,
            parts: None,
        }
    }
}

/// Verify n < 2^N entirely on the CPU.
pub fn verify_cpu(cfg: &VerifyConfig) -> VerifyReport {
    verify_cpu_with_expander(cfg, &DfsExpander)
}

/// Run the complete CPU verifier using a caller-selected host expander.
/// Work-unit generation still uses `cfg.rules`; the custom expansion policy
/// governs descendants of those already-surviving units.
pub fn verify_cpu_with_expander<E: Expander>(cfg: &VerifyConfig, expander: &E) -> VerifyReport {
    // c3-hyg: refuse invalid configurations and the round-2 environment
    // selectors loudly, before any work (callers that want an error value use
    // `VerifyConfig::validate` first, as `cv verify` does).
    if let Err(e) = cfg.validate().and_then(|_| crate::runspec::reject_legacy_env()) {
        panic!("invalid verification configuration: {e:#}");
    }
    let t0 = Instant::now();
    let m = cfg.m();
    let la = cfg.lookahead();
    let mut top = TreeStats::new(m);
    let units = cfg.work_units(&mut top);
    let (tree, tail) = units
        .par_iter()
        .fold(
            || (TreeStats::new(m), CpuTailSink::new(&la, cfg.tail)),
            |(mut st, mut sink), u| {
                expander.expand::<_, true>(u, m, &cfg.rules, &mut sink, &mut st);
                (st, sink)
            },
        )
        .map(|(st, sink)| (st, sink.st))
        .reduce(
            || (TreeStats::new(m), TailStats::new(cfg.num_bv)),
            |(mut a, mut b), (c, d)| {
                a.merge(&c);
                b.merge(&d);
                (a, b)
            },
        );
    top.merge(&tree);
    VerifyReport::new(cfg, "cpu", units.len(), top, tail, t0.elapsed().as_secs_f64())
}

#[derive(Serialize)]
pub struct Census {
    pub depth: u32,
    pub rules: String,
    pub seconds: f64,
    pub stats: TreeStats,
    pub endpoints_by_f: Vec<u64>,
}

/// Exact step-1 survivor counts per depth (no steps 2/3).
pub fn census(rules: &Rules, depth: u32, split: u32) -> Census {
    let t0 = Instant::now();
    let mut top = TreeStats::new(depth);
    let units: Vec<Node> = frontier(split.clamp(2, depth), rules, &mut top);
    let (tree, sink) = units
        .par_iter()
        .fold(
            || (TreeStats::new(depth), CountSink::default()),
            |(mut st, mut sink), u| {
                dfs::<_, true>(u, depth, rules, &mut sink, &mut st);
                (st, sink)
            },
        )
        .reduce(
            || (TreeStats::new(depth), CountSink::default()),
            |(mut a, mut b), (c, d)| {
                a.merge(&c);
                if b.by_f.len() < d.by_f.len() {
                    b.by_f.resize(d.by_f.len(), 0);
                }
                for (x, y) in b.by_f.iter_mut().zip(&d.by_f) {
                    *x += y;
                }
                (a, b)
            },
        );
    top.merge(&tree);
    Census { depth, rules: rules.label(), seconds: t0.elapsed().as_secs_f64(), stats: top, endpoints_by_f: sink.by_f }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::Sink;

    struct CompactDfs;

    impl Expander for CompactDfs {
        fn expand<S: Sink, const STATS: bool>(
            &self,
            root: &Node,
            target: u32,
            rules: &Rules,
            sink: &mut S,
            stats: &mut TreeStats,
        ) {
            struct Compact<'a, S>(&'a mut S);
            impl<S: Sink> Sink for Compact<'_, S> {
                fn endpoint(&mut self, node: &Node) {
                    self.0.compact_endpoint(node.n0 as u64, node.k, node.f, node.oee, node.ne);
                }
            }
            dfs::<_, STATS>(root, target, rules, &mut Compact(sink), stats);
        }
    }

    fn small_cfg(screen: crate::tail::Screen, codes: crate::runspec::TailCodes) -> VerifyConfig {
        let rules = Rules::improved(3);
        VerifyConfig {
            n: 26,
            a: 8,
            b: 12,
            num_bv: 4,
            bv_mode: BvMode::Shared,
            window: WindowRules { pm_inf: true, mix: rules.mix.clone() },
            class_b: None,
            split: 10,
            rules,
            tail: TailParams { screen, codes, ..TailParams::default() },
            units: None,
        }
    }

    #[test]
    fn reports_name_their_h_range_and_halves_partition_the_full_run() {
        // c3-hyg regression: `cv verify --range top|bottom` reported "full_range".
        use crate::runspec::TailCodes;
        use crate::tail::Screen;
        for codes in [TailCodes::original(), TailCodes::parse("93,91,90,111", 4).unwrap()] {
            let full = verify_cpu(&small_cfg(Screen::default(), codes));
            let top = verify_cpu(&small_cfg(Screen::top(8), codes));
            let bot = verify_cpu(&small_cfg(Screen::bottom(8), codes));
            assert_eq!(full.verification_scope, "full_range");
            assert_eq!(top.verification_scope, "top_half");
            assert_eq!(bot.verification_scope, "bottom_half");
            assert_eq!((top.config.h_range.h_lo, top.config.h_range.h_hi), (128, 256));
            assert_eq!((bot.config.h_range.h_lo, bot.config.h_range.h_hi), (0, 128));
            assert_eq!(top.config.tail_codes, codes.per_class(4));
            assert!(top.cases > 0 && bot.cases > 0);
            assert_eq!(top.cases + bot.cases, full.cases);
            assert_eq!(top.iters + bot.iters, full.iters);
            let h = |r: &VerifyReport| u32::from_str_radix(&r.hash, 16).unwrap();
            assert_eq!(h(&top) ^ h(&bot), h(&full));
            assert_eq!(top.tail.endpoints, full.tail.endpoints);
            let json = serde_json::to_value(&top).unwrap();
            assert_eq!(json["config"]["h_range"]["kind"], "top_half");
            assert_eq!(json["config"]["work_definition"]["version"], crate::runspec::WORK_DEFINITION_VERSION);
            assert_eq!(json["config"]["bitvector"]["class_windows"], serde_json::json!([12, 12, 12, 12]));
        }
        let mut units = small_cfg(Screen::top(8), TailCodes::original());
        units.units = Some(vec![Node::root(&units.rules)]);
        assert_eq!(verify_cpu(&units).verification_scope, "selected_work_units+top_half");
    }

    #[test]
    fn cpu_reference_takes_every_class_code_from_the_run() {
        // c3-hyg regression: the CPU semantics came from CV_XTAIL_CPU and
        // classes beyond its list silently ran the original loop.  Now each
        // class runs exactly the configured code: changing one class's code
        // changes only that class's iterations.
        use crate::runspec::TailCodes;
        use crate::tail::Screen;
        let base = verify_cpu(&small_cfg(Screen::default(), TailCodes::parse("93,93,93,93", 4).unwrap()));
        let alt = verify_cpu(&small_cfg(Screen::default(), TailCodes::parse("90,93,93,93", 4).unwrap()));
        let orig = verify_cpu(&small_cfg(Screen::default(), TailCodes::original()));
        assert_eq!(base.tail.cases, orig.tail.cases, "codes do not change the case set");
        assert!(base.tail.cases[0] > 0 && base.tail.endpoints[4] > 0, "fixture exercises class 0 and the CPU-only classes");
        assert_eq!(base.tail.iters[1..], alt.tail.iters[1..]);
        assert_ne!(base.tail.iters[0], alt.tail.iters[0]);
        assert_ne!(base.tail.iters[0], orig.tail.iters[0], "jumps are not double iterations");
        assert_eq!(base.tail.iters[4], orig.tail.iters[4], "CPU-only classes run the original loop");
        let mut bad = small_cfg(Screen::default(), TailCodes::parse("93,91", 2).unwrap());
        assert!(bad.validate().is_err(), "two codes for four classes");
        bad.tail.codes = TailCodes::parse("93", 4).unwrap();
        bad.tail.cutoff_bits = 102;
        assert!(bad.validate().is_err(), "W11 needs cutoff >= 103");
        bad.tail.cutoff_bits = 124;
        assert!(bad.validate().is_err());
        bad.tail.cutoff_bits = 120;
        bad.tail.screen = Screen::top(8);
        bad.a = 6;
        assert!(bad.validate().is_err(), "halves need A >= 7");
    }

    #[test]
    fn compact_expander_preserves_complete_cpu_verification() {
        let rules = Rules::improved(3);
        let cfg = VerifyConfig {
            n: 24,
            a: 8,
            b: 12,
            num_bv: 4,
            bv_mode: BvMode::Shared,
            window: WindowRules { pm_inf: true, mix: rules.mix.clone() },
            class_b: None,
            split: 10,
            rules,
            tail: TailParams::default(),
            units: None,
        };
        let reference = verify_cpu(&cfg);
        let compact = verify_cpu_with_expander(&cfg, &CompactDfs);
        assert!(reference.cases > 0);
        assert!(reference.tail.endpoints[cfg.num_bv] > 0, "exercise classes without bitvectors");
        assert_eq!(compact.tree.generated, reference.tree.generated);
        assert_eq!(compact.tree.survived, reference.tree.survived);
        assert_eq!(compact.tree.culls, reference.tree.culls);
        assert_eq!(compact.tail.endpoints, reference.tail.endpoints);
        assert_eq!(compact.tail.cases, reference.tail.cases);
        assert_eq!(compact.tail.iters, reference.tail.iters);
        assert_eq!(compact.hash, reference.hash);
        assert_eq!(compact.flagged, reference.flagged);
    }
}
