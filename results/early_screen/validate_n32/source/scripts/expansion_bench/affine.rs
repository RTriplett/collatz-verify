//! Preserve a terminal parent's exact iterate across table expansion.
//!
//! This uses the same full-check cutoff and terminal parity words as Combined.
//! The short baseline prefix retains genuine u128 Nodes. At the table boundary
//! it dispatches one parent plus a resident word range per final odd-count
//! class, allowing sinks to choose compact, affine, or parent/table transport.
use crate::terminal::Table;
use crate::ternary::{self, Screens};
use cv::expansion::Expander;
use cv::rules::{Cull, Rules};
use cv::tree::{Node, Sink, TerminalWord, TreeStats};
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub struct Affine {
    target: u32,
    cutoff: u32,
    baseline: Rules,
    screens: Screens,
    table: Table,
}

impl Affine {
    pub fn new(target: u32, cutoff: u32, entry: u32, rules: &Rules) -> Self {
        assert!(2 <= cutoff && cutoff <= entry && entry < target && target <= 64);
        assert!(target - entry <= 12);
        assert!(
            rules.walkback.is_none_or(|w| w.depth <= cutoff),
            "a configured checkpoint must precede the full-check cutoff"
        );
        let mut baseline = rules.clone();
        baseline.pm_inf = false;
        baseline.mix = None;
        baseline.walkback = None;
        let table = Table::new(entry, target, &baseline);
        Self {
            target,
            cutoff,
            baseline,
            screens: Screens::new(target),
            table,
        }
    }

    pub fn name(&self) -> String {
        format!("affine{}x{}", self.cutoff, self.target - self.table.start())
    }

    pub fn table_bytes(&self) -> usize {
        self.table.bytes()
    }

    pub fn table_words(&self) -> &[TerminalWord] {
        self.table.words()
    }

    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Exact baseline candidate reductions from one or two additional bits,
    /// attributed to the original final-F class. Count group visits while
    /// expanding parents, then inspect each distinct table word only once.
    /// This avoids materializing the potentially billions of endpoints.
    pub fn class_probe(&self, roots: &[Node], rules: &Rules) -> Value {
        assert!(self.target <= 62, "two-bit probe requires M <= 62");
        struct Visits {
            visits: Vec<u64>,
            lengths: Vec<usize>,
            previous_parent: Option<(u128, u32)>,
            emitting_parents: u64,
        }
        impl Sink for Visits {
            fn endpoint(&mut self, _: &Node) {
                panic!("class probe requires grouped terminal dispatch");
            }
            fn terminal_group(
                &mut self,
                parent: &Node,
                entry_start: u32,
                entries: &[TerminalWord],
                _: u32,
            ) {
                let offset = entry_start as usize;
                self.visits[offset] += 1;
                self.lengths[offset] = entries.len();
                let key = (parent.n0, parent.k);
                if self.previous_parent != Some(key) {
                    self.previous_parent = Some(key);
                    self.emitting_parents += 1;
                }
            }
        }
        let mut sink = Visits {
            visits: vec![0; self.table.words().len()],
            lengths: vec![0; self.table.words().len()],
            previous_parent: None,
            emitting_parents: 0,
        };
        for root in roots {
            sink.previous_parent = None;
            self.expand::<_, false>(
                root,
                self.target,
                rules,
                &mut sink,
                &mut TreeStats::default(),
            );
        }
        let one = Table::new(self.target, self.target + 1, &self.baseline);
        let two = Table::new(self.target, self.target + 2, &self.baseline);
        let mut classes: BTreeMap<u32, [u64; 3]> = BTreeMap::new();
        for (offset, &visits) in sink.visits.iter().enumerate() {
            if visits == 0 {
                continue;
            }
            for word in &self.table.words()[offset..offset + sink.lengths[offset]] {
                let counts = classes.entry(word.f).or_default();
                counts[0] += visits;
                counts[1] += visits * one.descendants(word.f, word.oee as u8, word.ne as u8);
                counts[2] += visits * two.descendants(word.f, word.oee as u8, word.ne as u8);
            }
        }
        let mut totals = [0u64; 3];
        let rows: Vec<_> = classes
            .into_iter()
            .map(|(f, counts)| {
                for (sum, count) in totals.iter_mut().zip(counts) {
                    *sum += count;
                }
                json!({
                    "f": f,
                    "endpoints": counts[0],
                    "one_bit_endpoints": counts[1],
                    "two_bit_endpoints": counts[2],
                    "one_bit_candidate_fraction": counts[1] as f64 / (2.0 * counts[0] as f64),
                    "two_bit_candidate_fraction": counts[2] as f64 / (4.0 * counts[0] as f64),
                })
            })
            .collect();
        json!({
            "type": "selective_class_probe",
            "target": self.target,
            "cutoff": self.cutoff,
            "terminal_start": self.table.start(),
            "roots": roots.len(),
            "emitting_parents": sink.emitting_parents,
            "group_visits": sink.visits.iter().sum::<u64>(),
            "distinct_groups": sink.visits.iter().filter(|&&n| n > 0).count(),
            "classes": rows,
            "endpoints": totals[0],
            "one_bit_endpoints": totals[1],
            "two_bit_endpoints": totals[2],
            "one_bit_candidate_fraction": totals[1] as f64 / (2.0 * totals[0] as f64),
            "two_bit_candidate_fraction": totals[2] as f64 / (4.0 * totals[0] as f64),
            "scope": "Exact baseline candidate counts by original final F; continuation halves or quarters the candidates per endpoint. No lookahead-window, tail-time, or mixed-M GPU timing prediction."
        })
    }

    fn prefix<S: Sink, const STATS: bool>(
        &self,
        node: &Node,
        rules: &Rules,
        sink: &mut S,
        stats: &mut TreeStats,
    ) {
        if node.k >= self.cutoff {
            self.baseline_prefix::<S, STATS>(node, sink, stats);
            return;
        }
        for child in node.children(rules) {
            let Some(child) = child else {
                if STATS {
                    stats.culls[node.k as usize + 1][Cull::Oee as usize] += 1;
                }
                continue;
            };
            if STATS {
                stats.generated[child.k as usize] += 1;
            }
            match ternary::cull::<2>(&child, rules, &self.screens) {
                Some(why) => {
                    if STATS {
                        stats.culls[child.k as usize][why as usize] += 1;
                    }
                }
                None => {
                    if STATS {
                        stats.survived[child.k as usize] += 1;
                    }
                    self.prefix::<S, STATS>(&child, rules, sink, stats);
                }
            }
        }
    }

    fn baseline_prefix<S: Sink, const STATS: bool>(
        &self,
        node: &Node,
        sink: &mut S,
        stats: &mut TreeStats,
    ) {
        if node.k == self.table.start() {
            self.table.emit_affine::<S, STATS>(node, sink, stats);
            return;
        }
        for child in node.children(&self.baseline) {
            let Some(child) = child else {
                if STATS {
                    stats.culls[node.k as usize + 1][Cull::Oee as usize] += 1;
                }
                continue;
            };
            if STATS {
                stats.generated[child.k as usize] += 1;
            }
            match child.cull(&self.baseline) {
                Some(why) => {
                    if STATS {
                        stats.culls[child.k as usize][why as usize] += 1;
                    }
                }
                None => {
                    if STATS {
                        stats.survived[child.k as usize] += 1;
                    }
                    self.baseline_prefix::<S, STATS>(&child, sink, stats);
                }
            }
        }
    }
}

impl Expander for Affine {
    fn expand<S: Sink, const STATS: bool>(
        &self,
        root: &Node,
        target: u32,
        rules: &Rules,
        sink: &mut S,
        stats: &mut TreeStats,
    ) {
        assert_eq!(target, self.target);
        assert!(root.k >= 2 && root.k <= self.table.start());
        assert_eq!(
            (rules.pm1, rules.oee, rules.root_oee),
            (self.baseline.pm1, self.baseline.oee, self.baseline.root_oee),
        );
        assert!(rules.walkback.is_none_or(|w| w.depth <= self.cutoff));
        self.prefix::<S, STATS>(root, rules, sink, stats);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combined::{Combined, Mode};
    use cv::rules::Walkback;
    use cv::sample::{Rng, Sampler};
    use cv::tree::dfs;

    #[derive(Default)]
    struct Nodes(Vec<Node>);
    impl Sink for Nodes {
        fn endpoint(&mut self, node: &Node) {
            self.0.push(*node);
        }
    }

    fn compare(root: &Node, target: u32, cutoff: u32, entry: u32, rules: &Rules) {
        let affine = Affine::new(target, cutoff, entry, rules);
        let compact = Combined::new(
            target,
            rules,
            Mode::Terminal {
                cutoff,
                width: target - entry,
            },
        );
        let mut want = Nodes::default();
        let mut expected = TreeStats::new(target);
        compact.expand::<_, true>(root, target, rules, &mut want, &mut expected);
        want.0.sort_unstable_by_key(|node| node.n0);

        let mut got = Nodes::default();
        let mut actual = TreeStats::new(target);
        affine.expand::<_, true>(root, target, rules, &mut got, &mut actual);
        got.0.sort_unstable_by_key(|node| node.n0);
        assert_eq!(got.0, want.0);
        assert_eq!(actual.generated, expected.generated);
        assert_eq!(actual.survived, expected.survived);
        assert_eq!(actual.culls, expected.culls);

        // Also compare with genuine DFS on both sides of the policy boundary.
        let mut boundary = Nodes::default();
        let mut independent = TreeStats::new(target);
        dfs::<_, true>(
            root,
            cutoff.max(root.k),
            rules,
            &mut boundary,
            &mut independent,
        );
        let mut oracle = Nodes::default();
        for node in boundary.0 {
            dfs::<_, true>(
                &node,
                target,
                &affine.baseline,
                &mut oracle,
                &mut independent,
            );
        }
        oracle.0.sort_unstable_by_key(|node| node.n0);
        assert_eq!(got.0, oracle.0);
        assert_eq!(actual.generated, independent.generated);
        assert_eq!(actual.survived, independent.survived);
        assert_eq!(actual.culls, independent.culls);

        let mut unrecorded = Nodes::default();
        let mut untouched = TreeStats::default();
        affine.expand::<_, false>(root, target, rules, &mut unrecorded, &mut untouched);
        unrecorded.0.sort_unstable_by_key(|node| node.n0);
        assert_eq!(unrecorded.0, want.0);
        assert!(untouched.generated.is_empty());
        assert!(untouched.survived.is_empty());
        assert!(untouched.culls.is_empty());
    }

    #[test]
    fn affine_complete_small_tree_matches_compact_and_two_stage_dfs() {
        let mut rules = Rules::improved(7);
        rules.walkback = Some(Walkback {
            depth: 10,
            max_states: 256,
        });
        let root = Node::root(&rules);
        for (cutoff, entry) in [(10, 10), (10, 12), (14, 18), (18, 18)] {
            compare(&root, 22, cutoff, entry, &rules);
        }
    }

    #[test]
    fn affine_exact_nodes_include_wide_iterates_and_target64() {
        let rules = Rules::improved(7);
        let sampler = Sampler::new(&Rules::baseline(), 52, Some((40, 52)));
        let mut rng = Rng(31_421);
        let mut roots = Vec::new();
        while roots.len() < 8 {
            let draw = sampler.draw(&mut rng, Some(&rules));
            if draw.culled.is_none() && draw.node.m > u64::MAX as u128 {
                roots.push(draw.node);
            }
        }
        let mut all_odd = Node::root(&rules);
        while all_odd.k < 52 {
            all_odd = all_odd
                .children(&rules)
                .into_iter()
                .flatten()
                .find(|node| node.f == all_odd.f + 1)
                .unwrap();
        }
        roots.push(all_odd);
        for root in roots {
            for (target, cutoff, entry) in [(62, 52, 52), (64, 52, 52), (64, 54, 54)] {
                compare(&root, target, cutoff, entry, &rules);
            }
        }
    }

    #[test]
    fn affine_dispatch_ranges_are_resident_and_single_class() {
        struct Groups<'a> {
            words: &'a [TerminalWord],
            nodes: Vec<Node>,
            groups: usize,
        }
        impl Sink for Groups<'_> {
            fn endpoint(&mut self, _: &Node) {
                panic!("affine expander must dispatch terminal groups");
            }
            fn terminal_group(
                &mut self,
                parent: &Node,
                entry_start: u32,
                entries: &[TerminalWord],
                target: u32,
            ) {
                assert!(!entries.is_empty());
                assert_eq!(
                    entries,
                    &self.words[entry_start as usize..][..entries.len()]
                );
                assert!(entries.iter().all(|entry| entry.f == entries[0].f));
                self.nodes
                    .extend(entries.iter().map(|entry| entry.endpoint(parent, target)));
                self.groups += 1;
            }
        }
        let rules = Rules::improved(7);
        let affine = Affine::new(22, 10, 12, &rules);
        let root = Node::root(&rules);
        let mut groups = Groups {
            words: affine.table_words(),
            nodes: Vec::new(),
            groups: 0,
        };
        affine.expand::<_, false>(&root, 22, &rules, &mut groups, &mut TreeStats::default());
        let mut expected = Nodes::default();
        affine.expand::<_, false>(&root, 22, &rules, &mut expected, &mut TreeStats::default());
        groups.nodes.sort_unstable_by_key(|node| node.n0);
        expected.0.sort_unstable_by_key(|node| node.n0);
        assert_eq!(groups.nodes, expected.0);
        assert!(groups.groups > 0 && groups.groups < groups.nodes.len());
    }

    #[test]
    fn selective_class_counts_match_explicit_baseline_continuations() {
        let rules = Rules::improved(7);
        let target = 20;
        let cutoff = 10;
        let root = Node::root(&rules);
        let affine = Affine::new(target, cutoff, 12, &rules);
        let probe = affine.class_probe(&[root], &rules);
        let reference = Combined::new(target, &rules, Mode::ReferenceCutoff(cutoff));
        let mut endpoints = Nodes::default();
        reference.expand::<_, false>(
            &root,
            target,
            &rules,
            &mut endpoints,
            &mut TreeStats::default(),
        );
        let mut expected: BTreeMap<u32, [u64; 3]> = BTreeMap::new();
        for endpoint in endpoints.0 {
            let counts = expected.entry(endpoint.f).or_default();
            counts[0] += 1;
            for extra in [1, 2] {
                let mut descendants = Nodes::default();
                dfs::<_, false>(
                    &endpoint,
                    target + extra,
                    &affine.baseline,
                    &mut descendants,
                    &mut TreeStats::default(),
                );
                counts[extra as usize] += descendants.0.len() as u64;
            }
        }
        let rows = probe["classes"].as_array().unwrap();
        assert_eq!(rows.len(), expected.len());
        for (row, (f, counts)) in rows.iter().zip(expected) {
            assert_eq!(row["f"], f);
            assert_eq!(row["endpoints"], counts[0]);
            assert_eq!(row["one_bit_endpoints"], counts[1]);
            assert_eq!(row["two_bit_endpoints"], counts[2]);
        }
    }
}
