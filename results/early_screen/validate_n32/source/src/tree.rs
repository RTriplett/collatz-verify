//! Step 1: the least-significant-bit-first residue tree (Angeltveit, Alg. 2).
//!
//! A node is the class `n0 (mod 2^k)` together with `m = T^k(n0)`, the odd
//! step count `f`, and two small parity-state counters used by the baseline
//! rules.  Its two children add bit `k` of the starting value; exactly one of
//! them takes an even step and the other an odd step (Corollary 2.4).

use crate::arith::{slope_descends, POW3};
use crate::rules::{Cull, Rules, NUM_CULLS};
use serde::{Deserialize, Serialize};

/// Parity state of the OEE machine.
/// 0 = idle, 1 = armed (an odd run whose `(x-1)/2` partner is low enough),
/// 2 = armed and one even step taken: a second even step is a merge.
pub type OeeState = u8;

/// Trailing even-step counter, folded to {0, 1, 2 = even >= 2, 3 = odd >= 3}.
/// `ne == 2` is exactly `m = 2 (mod 3)` after an even step (Lemma 2.8).
pub type EvenRun = u8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Node {
    #[serde(with = "u128_str")]
    pub n0: u128,
    pub k: u32,
    pub f: u32,
    #[serde(with = "u128_str")]
    pub m: u128,
    /// `m mod 3^16`, carried incrementally for the mixed-word tables.
    pub m3: u32,
    pub oee: OeeState,
    pub ne: EvenRun,
}

/// A compiled baseline suffix. The layout is shared with experimental CUDA
/// dispatch; all arithmetic fields describe the canonical residue `r`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerminalWord {
    pub r: u32,
    pub q_coeff: u32,
    pub iterate: u32,
    pub pow3: u32,
    pub f: u32,
    pub oee: u32,
    pub ne: u32,
    pub pad: u32,
}

/// Resident two-stage expansion program for one parent parity state and final F.
/// The layout is shared with the experimental CUDA compact-chain transport.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainProgram {
    pub route_start: u32,
    pub route_len: u32,
    pub endpoints: u32,
    pub first_width: u32,
    pub second_width: u32,
    pub first_start: u32,
    pub first_len: u32,
    pub leaf_start: u32,
}

/// A reusable first-stage word followed by a contiguous second-stage range.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainRoute {
    pub first_index: u32,
    pub second_start: u32,
    pub second_len: u32,
    pub output_start: u32,
}

/// Dense output scheduling map; indices reference the two short word tables.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainLeaf {
    pub first_slot: u32,
    pub second_index: u32,
}

/// Both short transition tables and the references that compose their routes.
/// Words are retained separately; no full-width composed affine words are stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainTables {
    pub programs: Vec<ChainProgram>,
    pub routes: Vec<ChainRoute>,
    pub words: Vec<TerminalWord>,
    pub leaves: Vec<ChainLeaf>,
}

/// Translation from canonical suffix residues to extensions of this parent.
#[inline]
pub fn terminal_delta(parent: &Node, width: u32) -> u32 {
    debug_assert!((1..=12).contains(&width));
    let inverse = 0xaaaa_aaaa_aaaa_aaabu64.wrapping_pow(parent.f);
    ((parent.m as u64).wrapping_mul(inverse) & ((1u64 << width) - 1)) as u32
}

impl TerminalWord {
    /// Exact affine reconstruction, retaining bits above the compact u64 state.
    #[inline]
    pub fn endpoint(&self, parent: &Node, target: u32) -> Node {
        let width = target - parent.k;
        debug_assert!(target <= 64 && (1..=12).contains(&width));
        let q = self.q_coeff.wrapping_sub(terminal_delta(parent, width))
            & ((1u32 << width) - 1);
        let x = parent.m + POW3[parent.f as usize] * q as u128;
        debug_assert_eq!(x & ((1u128 << width) - 1), self.r as u128);
        debug_assert_eq!(self.pow3 as u128, POW3[(self.f - parent.f) as usize]);
        let m = ((x - self.r as u128) >> width) * self.pow3 as u128
            + self.iterate as u128;
        Node {
            n0: parent.n0 | ((q as u128) << parent.k),
            k: target, f: self.f, m, m3: (m % P16 as u128) as u32,
            oee: self.oee as u8, ne: self.ne as u8,
        }
    }
}

pub fn expand_terminal_group<S: Sink + ?Sized>(
    sink: &mut S, parent: &Node, entries: &[TerminalWord], target: u32,
) {
    for entry in entries {
        sink.endpoint(&entry.endpoint(parent, target));
    }
}

/// Exact CPU fallback for a compact two-stage group. The GPU implementation
/// retains the same route structure and reconstructs intermediate nodes locally.
pub fn expand_chained_group<S: Sink + ?Sized>(
    sink: &mut S,
    parent: &Node,
    program: &ChainProgram,
    routes: &[ChainRoute],
    words: &[TerminalWord],
    target: u32,
) {
    debug_assert_eq!(routes.len(), program.route_len as usize);
    debug_assert_eq!(parent.k + program.first_width + program.second_width, target);
    let middle = parent.k + program.first_width;
    let mut emitted = 0;
    for route in routes {
        debug_assert_eq!(route.output_start, emitted);
        let prefix = words[route.first_index as usize].endpoint(parent, middle);
        let start = route.second_start as usize;
        for word in &words[start..start + route.second_len as usize] {
            sink.endpoint(&word.endpoint(&prefix, target));
            emitted += 1;
        }
    }
    debug_assert_eq!(emitted, program.endpoints);
}

/// 3^16, the largest mixed-word table modulus.
pub const P16: u64 = 43_046_721;
/// `3^f mod 3^16` (zero from f = 16 on).
const POW3_MOD_P16: [u32; 17] = {
    let mut t = [0u32; 17];
    let mut i = 0;
    while i < 16 {
        t[i] = POW3[i] as u32;
        i += 1;
    }
    t
};

/// `a + b mod 3^16` for `a, b < 3^16`.
#[inline(always)]
fn add_mod_p16(a: u32, b: u32) -> u32 {
    let s = a + b;
    if s >= P16 as u32 { s - P16 as u32 } else { s }
}

/// `x / 2 mod 3^16` for `x < 3 * 3^16 + 1` (P16 is odd: halve x or x + P16).
#[inline(always)]
fn half_mod_p16(x: u64) -> u32 {
    let mut y = x as u32;
    if y >= 2 * P16 as u32 {
        y -= 2 * P16 as u32;
    }
    if y >= P16 as u32 {
        y -= P16 as u32;
    }
    ((y as u64 + (y as u64 & 1) * P16) >> 1) as u32
}

#[inline(always)]
pub fn next_even_run(ne: EvenRun) -> EvenRun {
    if ne >= 2 {
        5 - ne
    } else {
        ne + 1
    }
}

impl Node {
    /// Restore the full iterate for a compact endpoint emitted by an expander.
    /// The CPU tail needs the exact iterate, including bits above bit 63.
    pub fn from_compact(n0: u64, k: u32, f: u32, oee: OeeState, ne: EvenRun) -> Self {
        assert!((2..=64).contains(&k));
        assert!((n0 as u128) < (1u128 << k));
        assert!(oee < 3 && ne < 4);
        let (m, actual_f) = crate::arith::t_iter(n0 as u128, k);
        assert_eq!(f, actual_f, "compact endpoint odd-step count is inconsistent");
        Node { n0: n0 as u128, k, f, m, m3: (m % P16 as u128) as u32, oee, ne }
    }

    /// The class `3 (mod 4)`.  Classes 0, 1, 2 (mod 4) descend in <= 2 steps.
    pub fn root(rules: &Rules) -> Node {
        Node { n0: 3, k: 2, f: 2, m: 8, m3: 8, oee: (rules.oee && rules.root_oee) as u8, ne: 0 }
    }

    /// The two children `(bit 0, bit 1)` before any culls.  `None` marks the
    /// even child suppressed by the OEE machine (an OEE cull).
    #[inline(always)]
    pub fn children(&self, rules: &Rules) -> [Option<Node>; 2] {
        let k = self.k;
        // Angeltveit's "doubleinplay": the (x-1)/2 partner of an odd step taken
        // here would have slope 3^f <= 2^(k+1).
        let arm = if rules.oee && slope_descends(k + 1, self.f) { 1 } else { 0 };
        let odd_state = (self.oee | arm) & !2;
        let even_ok = self.oee < 2;
        let hi = 1u128 << k;
        let odd_of = |n0: u128, x: u128, x3: u32| Node {
            n0,
            k: k + 1,
            f: self.f + 1,
            m: x + (x >> 1) + 1,
            m3: half_mod_p16(3 * x3 as u64 + 1),
            oee: odd_state,
            ne: 0,
        };
        let even_of = |n0: u128, x: u128, x3: u32| Node {
            n0,
            k: k + 1,
            f: self.f,
            m: x >> 1,
            m3: half_mod_p16(x3 as u64),
            oee: self.oee << 1,
            ne: next_even_run(self.ne),
        };
        let x1 = self.m + POW3[self.f as usize];
        let x13 = add_mod_p16(self.m3, POW3_MOD_P16[self.f.min(16) as usize]);
        if self.m & 1 == 0 {
            [even_ok.then(|| even_of(self.n0, self.m, self.m3)), Some(odd_of(self.n0 | hi, x1, x13))]
        } else {
            [Some(odd_of(self.n0, self.m, self.m3)), even_ok.then(|| even_of(self.n0 | hi, x1, x13))]
        }
    }

    /// Apply every rule to this (freshly generated) node.
    #[inline(always)]
    pub fn cull(&self, rules: &Rules) -> Option<Cull> {
        if self.m < self.n0 || slope_descends(self.k, self.f) {
            return Some(Cull::Descent);
        }
        if rules.pm1 && self.ne == 2 && slope_descends(self.k - 1, self.f - 1) {
            return Some(Cull::Pm1);
        }
        rules.extra_cull(self.n0, self.k, self.f, self.m, self.m3)
    }
}

/// Receives the depth-`M` survivors.
pub trait Sink {
    fn endpoint(&mut self, node: &Node);

    /// Accept an endpoint without a final iterate. GPU sinks can pack this
    /// directly; ordinary sinks reconstruct the exact node for CPU processing.
    fn compact_endpoint(&mut self, n0: u64, k: u32, f: u32, oee: OeeState, ne: EvenRun) {
        self.endpoint(&Node::from_compact(n0, k, f, oee, ne));
    }

    /// A group of suffixes with one final F. Ordinary sinks receive full nodes;
    /// experimental GPU sinks may send the parent and resident-table indices.
    fn terminal_group(&mut self, parent: &Node, _entry_start: u32,
                      entries: &[TerminalWord], target: u32) {
        expand_terminal_group(self, parent, entries, target);
    }

    /// Dispatch a parent and resident two-stage program without materializing
    /// intermediate or final nodes on the host. Ordinary sinks expand exactly.
    fn chained_group(&mut self, parent: &Node, _program_index: u32,
                     program: &ChainProgram, routes: &[ChainRoute],
                     words: &[TerminalWord], target: u32) {
        expand_chained_group(self, parent, program, routes, words, target);
    }
}

/// Per-depth node accounting.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TreeStats {
    /// Nodes generated (= rule evaluations) at each depth.
    pub generated: Vec<u64>,
    /// Nodes surviving all rules at each depth.
    pub survived: Vec<u64>,
    /// Culls by rule at each depth (`Cull as usize` index).
    pub culls: Vec<[u64; NUM_CULLS]>,
}

impl TreeStats {
    pub fn new(max_depth: u32) -> Self {
        let n = max_depth as usize + 1;
        TreeStats { generated: vec![0; n], survived: vec![0; n], culls: vec![[0; NUM_CULLS]; n] }
    }
    pub fn merge(&mut self, o: &TreeStats) {
        for (a, b) in self.generated.iter_mut().zip(&o.generated) {
            *a += b;
        }
        for (a, b) in self.survived.iter_mut().zip(&o.survived) {
            *a += b;
        }
        for (a, b) in self.culls.iter_mut().zip(&o.culls) {
            for i in 0..NUM_CULLS {
                a[i] += b[i];
            }
        }
    }
    pub fn total_generated(&self) -> u64 {
        self.generated.iter().sum()
    }
}

/// Depth-first search below `node` (which must already have survived) down
/// to depth `max_depth`, calling `sink` on every surviving depth-`max_depth`
/// class.  With `STATS = false` the counters compile away.
pub fn dfs<S: Sink, const STATS: bool>(
    node: &Node,
    max_depth: u32,
    rules: &Rules,
    sink: &mut S,
    stats: &mut TreeStats,
) {
    if node.k == max_depth {
        sink.endpoint(node);
        return;
    }
    for child in node.children(rules) {
        let Some(c) = child else {
            if STATS {
                stats.culls[node.k as usize + 1][Cull::Oee as usize] += 1;
            }
            continue;
        };
        if STATS {
            stats.generated[c.k as usize] += 1;
        }
        match c.cull(rules) {
            Some(why) => {
                if STATS {
                    stats.culls[c.k as usize][why as usize] += 1;
                }
            }
            None => {
                if STATS {
                    stats.survived[c.k as usize] += 1;
                }
                dfs::<S, STATS>(&c, max_depth, rules, sink, stats);
            }
        }
    }
}

/// Breadth-first expansion from the root to `depth`, returning the surviving
/// nodes (the natural work units) and the stats of the levels above.
pub fn frontier(depth: u32, rules: &Rules, stats: &mut TreeStats) -> Vec<Node> {
    let root = Node::root(rules);
    assert!(depth >= root.k);
    stats.generated[root.k as usize] += 1;
    stats.survived[root.k as usize] += 1;
    let mut level = vec![root];
    for _ in root.k..depth {
        let mut next = Vec::with_capacity(level.len() * 2);
        for nd in &level {
            for child in nd.children(rules) {
                let Some(c) = child else {
                    stats.culls[nd.k as usize + 1][Cull::Oee as usize] += 1;
                    continue;
                };
                stats.generated[c.k as usize] += 1;
                match c.cull(rules) {
                    Some(why) => stats.culls[c.k as usize][why as usize] += 1,
                    None => {
                        stats.survived[c.k as usize] += 1;
                        next.push(c);
                    }
                }
            }
        }
        level = next;
    }
    level
}

/// Sink that only counts endpoints by odd-step count `f`.
#[derive(Default, Clone)]
pub struct CountSink {
    pub by_f: Vec<u64>,
}

impl Sink for CountSink {
    #[inline]
    fn endpoint(&mut self, node: &Node) {
        let f = node.f as usize;
        if self.by_f.len() <= f {
            self.by_f.resize(f + 1, 0);
        }
        self.by_f[f] += 1;
    }
}

/// Serialize u128 as decimal strings (JSON numbers are f64 in most readers).
pub mod u128_str {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(x: &u128, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&x.to_string())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<u128, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::t_iter;

    #[test]
    fn compact_endpoint_restores_full_nodes_and_wide_iterates() {
        #[derive(Default)]
        struct Collect(Vec<Node>);
        impl Sink for Collect {
            fn endpoint(&mut self, node: &Node) {
                self.0.push(*node);
            }
        }

        let rules = Rules::baseline();
        let nodes = frontier(14, &rules, &mut TreeStats::new(14));
        let mut sink = Collect::default();
        for node in &nodes {
            sink.compact_endpoint(node.n0 as u64, node.k, node.f, node.oee, node.ne);
        }
        assert_eq!(sink.0, nodes);

        // T^64(2^64-1) = 3^64-1: reconstructing from low iterate bits would
        // corrupt the CPU fallback even though the starting residue fits u64.
        sink.compact_endpoint(u64::MAX, 64, 64, 1, 0);
        let wide = sink.0.last().unwrap();
        assert_eq!(wide.m, POW3[64] - 1);
        assert!(wide.m > u64::MAX as u128);
        assert_eq!(wide.m3, (wide.m % P16 as u128) as u32);
        assert_eq!((wide.oee, wide.ne), (1, 0));
    }

    #[test]
    fn children_track_iterates() {
        let rules = Rules::baseline();
        let mut stats = TreeStats::new(20);
        let nodes = frontier(14, &rules, &mut stats);
        for nd in &nodes {
            let (m, f) = t_iter(nd.n0, nd.k);
            assert_eq!((m, f), (nd.m, nd.f), "n0={}", nd.n0);
            assert_eq!(nd.m3 as u64, (nd.m % P16 as u128) as u64);
        }
    }

    #[test]
    fn descent_only_matches_a076227() {
        // OEIS A076227: residues mod 2^k with stopping time > k (k >= 2).
        let want = [(10u32, 64u64), (16, 2114), (20, 27328)];
        let rules = Rules::descent_only();
        for (k, w) in want {
            let mut stats = TreeStats::new(k);
            let level = frontier(k, &rules, &mut stats);
            assert_eq!(level.len() as u64, w, "k={k}");
        }
    }
}
