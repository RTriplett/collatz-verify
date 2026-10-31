//! Chain exact affine tables across baseline-only expansion boundaries.
//!
//! The grouped variant buckets full nodes by their parity-rule state and low
//! iterate translation. A table is dispatched once per bucket, then each word
//! is applied to all parents in that bucket. This shares word translation and
//! its wide `3^f * q` multiplication, rather than only changing endpoint order.
//! Exact high parts and the modulo-3^16 state are retained between every block.
use crate::terminal::Table;
use crate::ternary::{self, Screens};
use cv::arith::POW3;
use cv::expansion::Expander;
use cv::rules::{Cull, Rules};
use cv::tree::{terminal_delta, Node, Sink, TerminalWord, TreeStats, P16};
use serde::Serialize;

const INV3_POW: [u64; 65] = {
    let mut values = [1u64; 65];
    let mut i = 1;
    while i < values.len() {
        values[i] = values[i - 1].wrapping_mul(0xaaaa_aaaa_aaaa_aaab);
        i += 1;
    }
    values
};

#[derive(Default)]
struct Nodes(Vec<Node>);

impl Sink for Nodes {
    fn endpoint(&mut self, node: &Node) {
        self.0.push(*node);
    }
}

/// Optional untimed census; normal expansion compiles out these counters.
#[derive(Clone, Debug, Default, Serialize)]
pub struct LayerStats {
    pub start: u32,
    pub target: u32,
    pub parents: u64,
    pub buckets: u64,
    pub parents_in_shared_buckets: u64,
    pub largest_bucket: u64,
    pub table_dispatches: u64,
    pub word_translations: u64,
    pub output_nodes: u64,
}

pub struct Chained {
    target: u32,
    cutoff: u32,
    width: u32,
    grouped: bool,
    baseline: Rules,
    screens: Screens,
    tables: Vec<Table>,
}

impl Chained {
    pub fn new(target: u32, cutoff: u32, width: u32, rules: &Rules, grouped: bool) -> Self {
        assert!(2 <= cutoff && cutoff < target && target <= 64);
        assert!((1..=12).contains(&width));
        assert!(rules.walkback.is_none_or(|w| w.depth <= cutoff));
        let mut baseline = rules.clone();
        baseline.pm_inf = false;
        baseline.mix = None;
        baseline.walkback = None;
        let mut tables = Vec::new();
        let mut start = cutoff;
        while start < target {
            let end = (start + width).min(target);
            tables.push(Table::new(start, end, &baseline));
            start = end;
        }
        Self {
            target,
            cutoff,
            width,
            grouped,
            baseline,
            screens: Screens::new(target),
            tables,
        }
    }

    pub fn name(&self) -> String {
        format!(
            "chained{}x{}{}",
            self.cutoff,
            self.width,
            if self.grouped { "g" } else { "u" }
        )
    }

    pub fn table_bytes(&self) -> usize {
        self.tables.iter().map(Table::bytes).sum()
    }

    pub fn table_boundaries(&self) -> Vec<(u32, u32)> {
        self.tables
            .iter()
            .map(|t| (t.start(), t.target()))
            .collect()
    }

    /// Repeat expansion without a result sink to measure sharing. This is
    /// deliberately separate from timed runs and contains no clocks or globals.
    pub fn census(&self, root: &Node, rules: &Rules) -> Vec<LayerStats> {
        struct Discard;
        impl Sink for Discard {
            fn endpoint(&mut self, _: &Node) {}
        }
        let mut layers = Vec::new();
        self.expand_inner::<_, false, true>(
            root,
            rules,
            &mut Discard,
            &mut TreeStats::default(),
            &mut layers,
        );
        layers
    }

    fn prefix<S: Sink, const STATS: bool>(
        &self,
        node: &Node,
        stop: u32,
        rules: &Rules,
        sink: &mut S,
        stats: &mut TreeStats,
    ) {
        if node.k == stop {
            sink.endpoint(node);
            return;
        }
        let full_checks = node.k < self.cutoff;
        let active = if full_checks { rules } else { &self.baseline };
        for child in node.children(active) {
            let Some(child) = child else {
                if STATS {
                    stats.culls[node.k as usize + 1][Cull::Oee as usize] += 1;
                }
                continue;
            };
            if STATS {
                stats.generated[child.k as usize] += 1;
            }
            let why = if full_checks {
                ternary::cull::<2>(&child, rules, &self.screens)
            } else {
                child.cull(&self.baseline)
            };
            if let Some(why) = why {
                if STATS {
                    stats.culls[child.k as usize][why as usize] += 1;
                }
            } else {
                if STATS {
                    stats.survived[child.k as usize] += 1;
                }
                self.prefix::<S, STATS>(&child, stop, rules, sink, stats);
            }
        }
    }

    fn expand_inner<S: Sink, const STATS: bool, const CENSUS: bool>(
        &self,
        root: &Node,
        rules: &Rules,
        sink: &mut S,
        stats: &mut TreeStats,
        layers: &mut Vec<LayerStats>,
    ) {
        assert!(root.k >= 2 && root.k <= self.target);
        assert_eq!(
            (rules.pm1, rules.oee, rules.root_oee),
            (self.baseline.pm1, self.baseline.oee, self.baseline.root_oee)
        );
        assert!(rules.walkback.is_none_or(|w| w.depth <= self.cutoff));

        // A supplied root may already lie inside one of the configured blocks.
        // Finish that partial block with ordinary exact baseline traversal.
        let Some(first) = self.tables.iter().position(|t| t.start() >= root.k) else {
            self.prefix::<_, STATS>(root, self.target, rules, sink, stats);
            return;
        };
        let mut current = Nodes::default();
        let mut next = Nodes::default();
        self.prefix::<_, STATS>(root, self.tables[first].start(), rules, &mut current, stats);
        for table in &self.tables[first..] {
            let mut census = LayerStats {
                start: table.start(),
                target: table.target(),
                ..LayerStats::default()
            };
            if table.target() == self.target {
                self.layer::<_, STATS, CENSUS>(table, &mut current.0, sink, stats, &mut census);
            } else {
                next.0.clear();
                self.layer::<_, STATS, CENSUS>(
                    table,
                    &mut current.0,
                    &mut next,
                    stats,
                    &mut census,
                );
                std::mem::swap(&mut current, &mut next);
            }
            if CENSUS {
                layers.push(census);
            }
        }
    }

    fn layer<S: Sink, const STATS: bool, const CENSUS: bool>(
        &self,
        table: &Table,
        parents: &mut [Node],
        sink: &mut S,
        stats: &mut TreeStats,
        census: &mut LayerStats,
    ) {
        if CENSUS {
            census.parents = parents.len() as u64;
        }
        if !self.grouped {
            for parent in parents.iter() {
                if CENSUS {
                    let count = table.descendants(parent.f, parent.oee, parent.ne);
                    census.buckets += 1;
                    census.largest_bucket = 1;
                    census.table_dispatches += 1;
                    census.word_translations += count;
                    census.output_nodes += count;
                }
                table.emit_affine::<_, STATS>(parent, sink, stats);
            }
            return;
        }

        let width = table.target() - table.start();
        let mask = (1u64 << width) - 1;
        let key = |parent: &Node| {
            let delta = (parent.m as u64).wrapping_mul(INV3_POW[parent.f as usize]) & mask;
            let state = (parent.f * 3 + parent.oee as u32) * 4 + parent.ne as u32;
            (state << width) | delta as u32
        };
        parents.sort_unstable_by_key(key);
        let mut offset = 0;
        while offset < parents.len() {
            let first = &parents[offset];
            let first_key = key(first);
            let len = parents[offset..]
                .iter()
                .take_while(|p| key(p) == first_key)
                .count();
            if CENSUS {
                let count = table.descendants(first.f, first.oee, first.ne);
                census.buckets += 1;
                census.largest_bucket = census.largest_bucket.max(len as u64);
                if len > 1 {
                    census.parents_in_shared_buckets += len as u64;
                }
                census.table_dispatches += 1;
                census.word_translations += count;
                census.output_nodes += count * len as u64;
            }
            let mut bucket = Bucket {
                parents: &parents[offset..offset + len],
                sink,
            };
            if STATS {
                // A program's per-depth counters depend only on the bucket
                // state. Evaluate them once and multiply by its parent count.
                let mut per_parent = TreeStats::new(table.target());
                table.emit_affine::<_, true>(first, &mut bucket, &mut per_parent);
                for k in table.start() as usize + 1..=table.target() as usize {
                    stats.generated[k] += per_parent.generated[k] * len as u64;
                    stats.survived[k] += per_parent.survived[k] * len as u64;
                    for (total, count) in stats.culls[k].iter_mut().zip(per_parent.culls[k]) {
                        *total += count * len as u64;
                    }
                }
            } else {
                table.emit_affine::<_, false>(first, &mut bucket, stats);
            }
            offset += len;
        }
    }
}

/// All parents have the same parity state and modular translation. The word
/// loop shares both table loads and affine setup across their distinct high bits.
struct Bucket<'a, S> {
    parents: &'a [Node],
    sink: &'a mut S,
}

impl<S: Sink> Sink for Bucket<'_, S> {
    fn endpoint(&mut self, _: &Node) {
        unreachable!("table dispatch must retain its affine word groups")
    }

    fn terminal_group(&mut self, parent: &Node, _: u32, words: &[TerminalWord], target: u32) {
        let width = target - parent.k;
        let mask = (1u32 << width) - 1;
        let delta = terminal_delta(parent, width);
        let parent_power = POW3[parent.f as usize];
        for word in words {
            let q = word.q_coeff.wrapping_sub(delta) & mask;
            let addition = parent_power * q as u128;
            let residue_bits = (q as u128) << parent.k;
            for input in self.parents {
                let x = input.m + addition;
                debug_assert_eq!(x & mask as u128, word.r as u128);
                let m = ((x - word.r as u128) >> width) * word.pow3 as u128 + word.iterate as u128;
                self.sink.endpoint(&Node {
                    n0: input.n0 | residue_bits,
                    k: target,
                    f: word.f,
                    m,
                    m3: (m % P16 as u128) as u32,
                    oee: word.oee as u8,
                    ne: word.ne as u8,
                });
            }
        }
    }
}

impl Expander for Chained {
    fn expand<S: Sink, const STATS: bool>(
        &self,
        root: &Node,
        target: u32,
        rules: &Rules,
        sink: &mut S,
        stats: &mut TreeStats,
    ) {
        assert_eq!(target, self.target);
        self.expand_inner::<S, STATS, false>(root, rules, sink, stats, &mut Vec::new());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affine::Affine;
    use cv::rules::Walkback;
    use cv::tree::dfs;

    fn compare(root: &Node, target: u32, cutoff: u32, width: u32, rules: &Rules) -> Vec<Node> {
        let grouped = Chained::new(target, cutoff, width, rules, true);
        let plain = Chained::new(target, cutoff, width, rules, false);
        let mut boundary = Nodes::default();
        let mut want_stats = TreeStats::new(target);
        dfs::<_, true>(
            root,
            cutoff.max(root.k),
            rules,
            &mut boundary,
            &mut want_stats,
        );
        let mut want = Nodes::default();
        for parent in boundary.0 {
            dfs::<_, true>(
                &parent,
                target,
                &grouped.baseline,
                &mut want,
                &mut want_stats,
            );
        }
        want.0.sort_unstable_by_key(|n| n.n0);

        for engine in [&grouped, &plain] {
            let mut actual = Nodes::default();
            let mut stats = TreeStats::new(target);
            engine.expand::<_, true>(root, target, rules, &mut actual, &mut stats);
            actual.0.sort_unstable_by_key(|n| n.n0);
            assert_eq!(actual.0, want.0, "{}", engine.name());
            assert_eq!(stats.generated, want_stats.generated);
            assert_eq!(stats.survived, want_stats.survived);
            assert_eq!(stats.culls, want_stats.culls);

            let mut unrecorded = Nodes::default();
            let mut untouched = TreeStats::default();
            engine.expand::<_, false>(root, target, rules, &mut unrecorded, &mut untouched);
            unrecorded.0.sort_unstable_by_key(|n| n.n0);
            assert_eq!(unrecorded.0, want.0);
            assert!(
                untouched.generated.is_empty()
                    && untouched.survived.is_empty()
                    && untouched.culls.is_empty()
            );
        }

        let entry = (target - 12).max(cutoff).max(root.k);
        if entry < target {
            let affine = Affine::new(target, cutoff, entry, rules);
            let mut actual = Nodes::default();
            let mut stats = TreeStats::new(target);
            affine.expand::<_, true>(root, target, rules, &mut actual, &mut stats);
            actual.0.sort_unstable_by_key(|n| n.n0);
            assert_eq!(actual.0, want.0);
            assert_eq!(stats.generated, want_stats.generated);
            assert_eq!(stats.survived, want_stats.survived);
            assert_eq!(stats.culls, want_stats.culls);
        }
        want.0
    }

    #[test]
    fn chained_small_tree_matches_affine_and_independent_dfs() {
        let mut rules = Rules::improved(7);
        rules.walkback = Some(Walkback {
            depth: 10,
            max_states: 256,
        });
        for width in [1, 4, 5, 6, 12] {
            compare(&Node::root(&rules), 22, 10, width, &rules);
        }
    }

    #[test]
    fn chained_preserves_wide_iterates_bit63_and_partial_entry_blocks() {
        let rules = Rules::improved(7);
        let mut root = Node::root(&rules);
        while root.k < 52 {
            root = root
                .children(&rules)
                .into_iter()
                .flatten()
                .find(|n| n.f == root.f + 1)
                .unwrap();
        }
        assert!(root.m > u64::MAX as u128);
        for (cutoff, width) in [(50, 4), (52, 4), (52, 5), (52, 6), (52, 12)] {
            let endpoints = compare(&root, 64, cutoff, width, &rules);
            assert!(endpoints.iter().any(|n| n.n0 & (1u128 << 63) != 0));
            assert!(endpoints.iter().any(|n| n.m > u64::MAX as u128));
        }
        compare(&root, 52 + 1, 50, 4, &rules);
    }

    #[test]
    fn chained_census_shows_shared_dispatch_and_equal_endpoint_counts() {
        let rules = Rules::baseline();
        let root = Node::root(&rules);
        let grouped = Chained::new(22, 10, 4, &rules, true).census(&root, &rules);
        let plain = Chained::new(22, 10, 4, &rules, false).census(&root, &rules);
        assert_eq!(grouped.len(), 3);
        let mut shared = false;
        for (g, u) in grouped.iter().zip(plain) {
            assert_eq!(
                (g.start, g.target, g.parents, g.output_nodes),
                (u.start, u.target, u.parents, u.output_nodes)
            );
            assert_eq!(u.table_dispatches, u.parents);
            assert_eq!(u.word_translations, u.output_nodes);
            assert!(g.table_dispatches <= g.parents);
            assert!(g.word_translations <= g.output_nodes);
            shared |= g.word_translations < g.output_nodes;
        }
        assert!(
            shared,
            "grouping must share arithmetic, not merely reorder nodes"
        );
    }
}
