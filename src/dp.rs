//! Exact workload counting for the baseline (parity-state) rules.
//!
//! Every baseline rule (descent by `306k >= 485f`, pm1, OEE) depends only on
//! the parity vector of `n0` through the state `(k, f, oee, ne)`.  Since the
//! map from `n0 mod 2^k` to its parity vector is a bijection, the number of
//! surviving classes at any depth is a lattice-path count, computed here by
//! dynamic programming in O(k^2) — for k = 66 as easily as for k = 30.
//! (The value test `m < n0` never fires beyond the slope test; the counts
//! match the DFS exactly, see tests.)
//!
//! The same states give exact subtree sizes for work units and the backward
//! weights used to draw uniformly random endpoints (`crate::sample`).

use crate::arith::slope_descends;
use crate::lookahead::Lookahead;
use crate::rules::Rules;
use crate::tree::{next_even_run, Node};
use serde::Serialize;

pub const OEE_STATES: usize = 3;
pub const NE_STATES: usize = 4;

/// Parity state of a class, independent of its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PState {
    pub k: u32,
    pub f: u32,
    pub oee: u8,
    pub ne: u8,
}

impl PState {
    pub fn of(nd: &Node) -> Self {
        PState { k: nd.k, f: nd.f, oee: nd.oee, ne: nd.ne }
    }
}

/// Parity-state transitions with the baseline culls.
#[derive(Clone, Copy)]
pub struct ParityRules {
    pub pm1: bool,
    pub oee: bool,
}

impl ParityRules {
    pub fn of(r: &Rules) -> Self {
        ParityRules { pm1: r.pm1, oee: r.oee }
    }
    /// Children of a surviving state: `(odd child, even child)`, each `None`
    /// if suppressed (OEE) or culled.  Also returns how many were generated.
    #[inline]
    pub fn children(&self, s: PState) -> ([Option<PState>; 2], u32) {
        let arm = if self.oee && slope_descends(s.k + 1, s.f) { 1 } else { 0 };
        let odd = PState { k: s.k + 1, f: s.f + 1, oee: (s.oee | arm) & !2, ne: 0 };
        let even = (s.oee < 2).then(|| PState { k: s.k + 1, f: s.f, oee: s.oee << 1, ne: next_even_run(s.ne) });
        let generated = 1 + even.is_some() as u32;
        ([self.survives(odd).then_some(odd), even.filter(|&e| self.survives(e))], generated)
    }
    #[inline]
    pub fn survives(&self, s: PState) -> bool {
        if slope_descends(s.k, s.f) {
            return false;
        }
        !(self.pm1 && s.ne == 2 && slope_descends(s.k - 1, s.f - 1))
    }
}

/// Dense table over `(f, oee, ne)` at one depth.
#[derive(Clone)]
pub struct Layer {
    pub k: u32,
    pub cells: Vec<u128>,
}

impl Layer {
    fn new(k: u32, fmax: u32) -> Self {
        Layer { k, cells: vec![0; (fmax as usize + 1) * OEE_STATES * NE_STATES] }
    }
    #[inline]
    fn idx(f: u32, oee: u8, ne: u8) -> usize {
        (f as usize * OEE_STATES + oee as usize) * NE_STATES + ne as usize
    }
    #[inline]
    pub fn get(&self, f: u32, oee: u8, ne: u8) -> u128 {
        self.cells.get(Self::idx(f, oee, ne)).copied().unwrap_or(0)
    }
    #[inline]
    fn add(&mut self, s: PState, c: u128) {
        self.cells[Self::idx(s.f, s.oee, s.ne)] += c;
    }
    pub fn fmax(&self) -> u32 {
        (self.cells.len() / (OEE_STATES * NE_STATES)) as u32 - 1
    }
    pub fn states(&self) -> impl Iterator<Item = (PState, u128)> + '_ {
        let k = self.k;
        self.cells.iter().enumerate().filter(|(_, &c)| c > 0).map(move |(i, &c)| {
            let ne = (i % NE_STATES) as u8;
            let oee = ((i / NE_STATES) % OEE_STATES) as u8;
            let f = (i / (NE_STATES * OEE_STATES)) as u32;
            (PState { k, f, oee, ne }, c)
        })
    }
    pub fn total(&self) -> u128 {
        self.cells.iter().sum()
    }
    pub fn by_f(&self) -> Vec<u128> {
        let mut v = vec![0u128; self.fmax() as usize + 1];
        for (s, c) in self.states() {
            v[s.f as usize] += c;
        }
        v
    }
    pub fn by_f_oee(&self) -> Vec<[u128; OEE_STATES]> {
        let mut v = vec![[0u128; OEE_STATES]; self.fmax() as usize + 1];
        for (s, c) in self.states() {
            v[s.f as usize][s.oee as usize] += c;
        }
        v
    }
}

/// Forward counts from a start state down to `kmax`.
pub struct ForwardDp {
    pub start: PState,
    /// `layers[i]` is depth `start.k + i`.
    pub layers: Vec<Layer>,
    /// Nodes generated (rule evaluations) at each depth.
    pub generated: Vec<u128>,
}

impl ForwardDp {
    pub fn run(pr: ParityRules, start: PState, kmax: u32) -> Self {
        let mut layers = Vec::new();
        let mut generated = vec![1u128];
        let mut cur = Layer::new(start.k, kmax);
        cur.add(start, 1);
        layers.push(cur.clone());
        for k in start.k..kmax {
            let mut next = Layer::new(k + 1, kmax);
            let mut gen = 0u128;
            for (s, c) in cur.states() {
                let (kids, g) = pr.children(s);
                gen += g as u128 * c;
                for kid in kids.into_iter().flatten() {
                    next.add(kid, c);
                }
            }
            generated.push(gen);
            layers.push(next.clone());
            cur = next;
        }
        ForwardDp { start, layers, generated }
    }
    /// From the tree root (class 3 mod 4).
    pub fn from_root(rules: &Rules, kmax: u32) -> Self {
        let root = Node::root(rules);
        Self::run(ParityRules::of(rules), PState::of(&root), kmax)
    }
    pub fn layer(&self, k: u32) -> &Layer {
        &self.layers[(k - self.start.k) as usize]
    }
    pub fn generated_at(&self, k: u32) -> u128 {
        self.generated[(k - self.start.k) as usize]
    }
    /// Nodes generated at all depths up to and including `k`.
    pub fn generated_through(&self, k: u32) -> u128 {
        self.generated[..=(k - self.start.k) as usize].iter().sum()
    }
}

/// `ways[k][state]` = surviving depth-`target` descendants of `state`
/// (optionally restricted to endpoints with `f` in `f_filter`).
pub struct BackwardDp {
    pub target: u32,
    pub kmin: u32,
    layers: Vec<Layer>,
    pr: ParityRules,
}

impl BackwardDp {
    pub fn run(pr: ParityRules, kmin: u32, target: u32, f_filter: Option<(u32, u32)>) -> Self {
        let fmax = target;
        let mut layers = vec![Layer::new(target, fmax)];
        // depth `target`: weight 1 for every surviving state (in the filter)
        for f in 0..=fmax {
            if f_filter.is_some_and(|(lo, hi)| f < lo || f > hi) {
                continue;
            }
            for oee in 0..OEE_STATES as u8 {
                for ne in 0..NE_STATES as u8 {
                    let s = PState { k: target, f, oee, ne };
                    if pr.survives(s) {
                        layers[0].add(s, 1);
                    }
                }
            }
        }
        for k in (kmin..target).rev() {
            let up = layers.last().unwrap().clone();
            let mut l = Layer::new(k, fmax);
            for f in 0..=k.min(fmax) {
                for oee in 0..OEE_STATES as u8 {
                    for ne in 0..NE_STATES as u8 {
                        let s = PState { k, f, oee, ne };
                        if !pr.survives(s) {
                            continue;
                        }
                        let (kids, _) = pr.children(s);
                        let w: u128 = kids.iter().flatten().map(|c| up.get(c.f, c.oee, c.ne)).sum();
                        if w > 0 {
                            l.add(s, w);
                        }
                    }
                }
            }
            layers.push(l);
        }
        layers.reverse();
        BackwardDp { target, kmin, layers, pr }
    }
    #[inline]
    pub fn ways(&self, s: PState) -> u128 {
        if s.k < self.kmin || s.k > self.target {
            return 0;
        }
        self.layers[(s.k - self.kmin) as usize].get(s.f, s.oee, s.ne)
    }
    pub fn rules(&self) -> ParityRules {
        self.pr
    }
}

/// Predicted step-2 output ("cases to check") for verification below 2^N.
#[derive(Clone, Debug, Serialize)]
pub struct CasePrediction {
    pub n: u32,
    pub a: u32,
    pub b: u32,
    pub endpoints: f64,
    pub cases: f64,
    /// Endpoints with `f` beyond the last bitvector (handled without one).
    pub endpoints_no_bv: f64,
    /// Endpoints and cases per bitvector class (last entry = no bitvector).
    pub endpoints_by_class: Vec<f64>,
    pub cases_by_class: Vec<f64>,
    pub tree_nodes: f64,
    pub fraction: f64,
}

/// `cases = (5/9) 2^A sum_{f,oee} S_M(f,oee) rho(f,oee)`: completions are
/// equidistributed over window patterns and over residues mod 9.
pub fn predict_cases(fw: &ForwardDp, la: &Lookahead, n: u32, tail_oee: bool) -> CasePrediction {
    let layer = fw.layer(la.m);
    let per = (1u64 << la.a) as f64 * 5.0 / 9.0;
    let mut cases_by = vec![0f64; la.num_bv + 1];
    let mut eps_by = vec![0f64; la.num_bv + 1];
    for (s, c) in layer.states() {
        let arr = la.array(s.f, s.ne);
        let slot = la.bigindex(s.f).unwrap_or(la.num_bv);
        let rho = match (arr, tail_oee, s.oee) {
            (Some(i), true, 2) => la.density_oee2[i],
            (Some(i), true, 1) => la.density_oee1[i],
            (Some(i), _, _) => la.density[i],
            (None, true, 2) => 0.5,
            (None, true, 1) => 0.75,
            (None, _, _) => 1.0,
        };
        eps_by[slot] += c as f64;
        cases_by[slot] += c as f64 * rho * per;
    }
    let endpoints = eps_by.iter().sum();
    let cases = cases_by.iter().sum::<f64>();
    CasePrediction {
        n,
        a: la.a,
        b: la.b,
        endpoints,
        cases,
        endpoints_no_bv: eps_by[la.num_bv],
        endpoints_by_class: eps_by,
        cases_by_class: cases_by,
        tree_nodes: fw.generated_through(la.m) as f64,
        fraction: cases / 2f64.powi(n as i32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{frontier, TreeStats};

    #[test]
    fn dp_matches_tree() {
        for rules in [Rules::baseline(), Rules::descent_only(), Rules { root_oee: false, ..Rules::baseline() }] {
            let fw = ForwardDp::from_root(&rules, 22);
            let mut st = TreeStats::new(22);
            frontier(22, &rules, &mut st);
            for k in 3..=22u32 {
                assert_eq!(fw.layer(k).total(), st.survived[k as usize] as u128, "k={k} {}", rules.label());
                assert_eq!(fw.generated_at(k), st.generated[k as usize] as u128, "gen k={k}");
            }
        }
    }

    #[test]
    fn known_counts() {
        // Angeltveit set (root-armed OEE) == "descent+pm(3^1)+oee" column of
        // collatz-probe/cull_survivors_k1_k37.csv.
        let fw = ForwardDp::from_root(&Rules::baseline(), 37);
        assert_eq!(fw.layer(30).total(), 7_227_826);
        assert_eq!(fw.layer(37).total(), 575_349_866);
        let d = ForwardDp::from_root(&Rules::descent_only(), 30);
        assert_eq!(d.layer(30).total(), 12_771_274); // OEIS A076227
    }

    #[test]
    fn backward_matches_forward() {
        let rules = Rules::baseline();
        let pr = ParityRules::of(&rules);
        let root = PState::of(&Node::root(&rules));
        let fw = ForwardDp::run(pr, root, 28);
        let bw = BackwardDp::run(pr, 2, 28, None);
        assert_eq!(bw.ways(root), fw.layer(28).total());
    }
}
