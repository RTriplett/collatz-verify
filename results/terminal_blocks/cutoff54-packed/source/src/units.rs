//! Work units for distributed runs.
//!
//! Every subtree of the step-1 search is independent.  The backward DP gives
//! the exact number of baseline endpoints below any node, split by
//! bitvector class, so with measured per-class GPU seconds per endpoint the
//! cost of a subtree is known *before* it is run.  The planner splits the
//! tree until each subtree is below a target cost and then packs subtrees
//! into packages of roughly equal predicted cost.  A package's result is a
//! deterministic tuple (endpoints, cases, iterations, checksum), so any
//! package can be re-run elsewhere as a spot check.

use crate::dp::{BackwardDp, ParityRules, PState};
use crate::lookahead::Lookahead;
use crate::rules::Rules;
use crate::tree::Node;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Package {
    pub id: usize,
    pub nodes: Vec<Node>,
    /// Exact baseline endpoints below these nodes.
    pub endpoints: f64,
    /// Predicted GPU seconds (from per-class seconds per endpoint).
    pub gpu_seconds: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plan {
    pub n: u32,
    pub a: u32,
    pub b: u32,
    pub rules: String,
    pub target_seconds: f64,
    pub packages: Vec<Package>,
    pub total_endpoints: f64,
    pub total_gpu_seconds: f64,
}

/// Per-class backward DPs: `cost(node) = sum_i ways_i(node) * sec_i`.
pub struct CostOracle {
    classes: Vec<(BackwardDp, f64)>,
}

impl CostOracle {
    /// `sec_per_endpoint[i]` for bitvector class i; the last entry covers
    /// all larger f (CPU-finished endpoints).
    pub fn new(rules: &Rules, la: &Lookahead, sec_per_endpoint: &[f64]) -> Self {
        let pr = ParityRules::of(rules);
        let root = Node::root(rules);
        let mut classes = Vec::new();
        for (i, &sec) in sec_per_endpoint.iter().enumerate() {
            let lo = la.offset + i as u32;
            let hi = if i + 1 == sec_per_endpoint.len() { la.m } else { lo };
            classes.push((BackwardDp::run(pr, root.k, la.m, Some((lo, hi))), sec));
        }
        CostOracle { classes }
    }
    pub fn endpoints(&self, nd: &Node) -> f64 {
        let s = PState::of(nd);
        self.classes.iter().map(|(bw, _)| bw.ways(s) as f64).sum()
    }
    pub fn seconds(&self, nd: &Node) -> f64 {
        let s = PState::of(nd);
        self.classes.iter().map(|(bw, sec)| bw.ways(s) as f64 * sec).sum()
    }
}

/// Split until every subtree costs at most `target` seconds, then pack.
pub fn plan(rules: &Rules, la: &Lookahead, n: u32, oracle: &CostOracle, target: f64) -> Plan {
    let mut leaves: Vec<(Node, f64, f64)> = Vec::new();
    let mut stack = vec![Node::root(rules)];
    while let Some(nd) = stack.pop() {
        let cost = oracle.seconds(&nd);
        if cost <= target || nd.k == la.m {
            if cost > 0.0 {
                leaves.push((nd, oracle.endpoints(&nd), cost));
            }
            continue;
        }
        for c in nd.children(rules).into_iter().flatten() {
            if c.cull(rules).is_none() {
                stack.push(c);
            }
        }
    }
    // Deterministic order (by depth, then n0), then greedy packing.
    leaves.sort_by(|a, b| (a.0.k, a.0.n0).cmp(&(b.0.k, b.0.n0)));
    let mut packages = Vec::new();
    let mut cur = Package { id: 0, nodes: Vec::new(), endpoints: 0.0, gpu_seconds: 0.0 };
    for (nd, eps, cost) in leaves {
        if !cur.nodes.is_empty() && cur.gpu_seconds + cost > target {
            let id = packages.len() + 1;
            packages.push(std::mem::replace(&mut cur, Package { id, nodes: Vec::new(), endpoints: 0.0, gpu_seconds: 0.0 }));
        }
        cur.nodes.push(nd);
        cur.endpoints += eps;
        cur.gpu_seconds += cost;
    }
    if !cur.nodes.is_empty() {
        packages.push(cur);
    }
    let total_endpoints = packages.iter().map(|p| p.endpoints).sum();
    let total_gpu_seconds = packages.iter().map(|p| p.gpu_seconds).sum();
    Plan { n, a: la.a, b: la.b, rules: rules.label(), target_seconds: target, packages, total_endpoints, total_gpu_seconds }
}
