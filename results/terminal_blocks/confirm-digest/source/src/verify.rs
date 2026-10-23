//! End-to-end drivers: full verification on the CPU, and step-1 censuses.

use crate::lookahead::{BvMode, Lookahead, WindowRules};
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
        if let Some(w) = &self.class_b {
            Lookahead::meta(self.m(), self.a, self.b, self.num_bv, self.bv_mode, &self.window).with_class_windows(w)
        } else if self.b > 30 {
            Lookahead::meta(self.m(), self.a, self.b, self.num_bv, self.bv_mode, &self.window)
        } else {
            Lookahead::build_ext(self.m(), self.a, self.b, self.num_bv, self.bv_mode, &self.window)
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
}

impl VerifyReport {
    pub fn new(cfg: &VerifyConfig, backend: &str, units: usize, tree: TreeStats, tail: TailStats, seconds: f64) -> Self {
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
        }
    }
}

/// Verify n < 2^N entirely on the CPU.
pub fn verify_cpu(cfg: &VerifyConfig) -> VerifyReport {
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
                dfs::<_, true>(u, m, &cfg.rules, &mut sink, &mut st);
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
