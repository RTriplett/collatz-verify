//! Experimental expansion engines connected to the real verifier sinks.
//!
//! Full-check prefixes use the previously measured minimum-divisibility
//! screen. After one policy boundary, expansion calls the existing original,
//! compact, or terminal-table traversal. No rule-policy branch is added to the
//! inner compact/table loop. Compact payloads go through Sink::compact_endpoint,
//! so production GPU packing can consume them directly while ordinary CPU sinks
//! reconstruct genuine full-width nodes.
use crate::fast_tail::{self, Endpoints};
use crate::terminal::Table;
use crate::terminal_walk;
use crate::ternary::{self, Screens};
use cv::expansion::Expander;
use cv::rules::{Cull, Rules};
use cv::tree::{dfs, Node, Sink, TreeStats};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Full,
    Screened,
    ReferenceCutoff(u32),
    CompactCutoff(u32),
    Terminal { cutoff: u32, width: u32 },
}

impl Mode {
    pub fn name(self) -> String {
        match self {
            Self::Full => "full".into(),
            Self::Screened => "screened".into(),
            Self::ReferenceCutoff(k) => format!("reference{k}"),
            Self::CompactCutoff(k) => format!("compact{k}"),
            Self::Terminal { cutoff, width } => format!("terminal{cutoff}x{width}"),
        }
    }

    fn cutoff(self) -> Option<u32> {
        match self {
            Self::Full | Self::Screened => None,
            Self::ReferenceCutoff(k) | Self::CompactCutoff(k) => Some(k),
            Self::Terminal { cutoff, .. } => Some(cutoff),
        }
    }
}

pub struct Combined {
    target: u32,
    mode: Mode,
    baseline: Rules,
    screens: Screens,
    table: Option<Table>,
}

impl Combined {
    pub fn new(target: u32, rules: &Rules, mode: Mode) -> Self {
        assert!((2..=80).contains(&target));
        if let Some(cutoff) = mode.cutoff() {
            assert!((2..=target).contains(&cutoff));
            assert!(
                rules.walkback.is_none_or(|w| w.depth <= cutoff),
                "a configured checkpoint must precede the full-check cutoff"
            );
        }
        if matches!(mode, Mode::CompactCutoff(_) | Mode::Terminal { .. }) {
            assert!(
                target <= 64,
                "compact endpoint representation requires M <= 64"
            );
        }
        let mut baseline = rules.clone();
        baseline.pm_inf = false;
        baseline.mix = None;
        baseline.walkback = None;
        let table = match mode {
            Mode::Terminal { cutoff, width } => {
                assert!((1..=12).contains(&width));
                assert!(
                    width <= target - cutoff,
                    "the terminal table cannot bypass full prefix checks"
                );
                Some(Table::new(target - width, target, &baseline))
            }
            _ => None,
        };
        Self {
            target,
            mode,
            baseline,
            screens: Screens::new(target),
            table,
        }
    }

    pub fn name(&self) -> String {
        self.mode.name()
    }

    pub fn table_bytes(&self) -> usize {
        self.table.as_ref().map_or(0, Table::bytes)
    }

    /// SUFFIX 0 is original baseline DFS; 1 is compact DFS; 2 is the flat
    /// terminal table with the existing compact traversal to its boundary.
    fn prefix<S: Sink, const STATS: bool, const SCREENED: bool, const SUFFIX: u8>(
        &self,
        node: &Node,
        cutoff: u32,
        rules: &Rules,
        sink: &mut S,
        stats: &mut TreeStats,
    ) {
        if node.k >= cutoff {
            if SUFFIX == 0 {
                dfs::<S, STATS>(node, self.target, &self.baseline, sink, stats);
            } else {
                let mut adapter = Dispatch {
                    sink,
                    target: self.target,
                };
                if SUFFIX == 1 {
                    fast_tail::dfs::<_, STATS>(
                        node,
                        self.target,
                        &self.baseline,
                        &mut adapter,
                        stats,
                    );
                } else {
                    let table = self.table.as_ref().unwrap();
                    // This call starts at the current node: terminal_walk's
                    // original-rule prefix is bypassed, leaving only its
                    // already-tested compact traversal and table emission.
                    terminal_walk::run::<_, STATS, false>(
                        node,
                        node.k,
                        rules,
                        &self.baseline,
                        table,
                        &mut adapter,
                        stats,
                    );
                }
            }
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
            let why = if SCREENED {
                ternary::cull::<2>(&child, rules, &self.screens)
            } else {
                child.cull(rules)
            };
            if let Some(why) = why {
                if STATS {
                    stats.culls[child.k as usize][why as usize] += 1;
                }
            } else {
                if STATS {
                    stats.survived[child.k as usize] += 1;
                }
                self.prefix::<S, STATS, SCREENED, SUFFIX>(&child, cutoff, rules, sink, stats);
            }
        }
    }
}

struct Dispatch<'a, S> {
    sink: &'a mut S,
    target: u32,
}

impl<S: Sink> Endpoints for Dispatch<'_, S> {
    #[inline(always)]
    fn endpoint(&mut self, n0: u64, f: u32, oee: u8, ne: u8) {
        self.sink.compact_endpoint(n0, self.target, f, oee, ne);
    }
}

impl Expander for Combined {
    fn expand<S: Sink, const STATS: bool>(
        &self,
        root: &Node,
        target: u32,
        rules: &Rules,
        sink: &mut S,
        stats: &mut TreeStats,
    ) {
        assert_eq!(target, self.target);
        assert!(root.k >= 2 && root.k <= target);
        assert_eq!(
            (rules.pm1, rules.oee, rules.root_oee),
            (self.baseline.pm1, self.baseline.oee, self.baseline.root_oee),
        );
        if let Some(cutoff) = self.mode.cutoff() {
            assert!(rules.walkback.is_none_or(|w| w.depth <= cutoff));
        }
        match self.mode {
            Mode::Full => dfs::<_, STATS>(root, target, rules, sink, stats),
            Mode::Screened => {
                ternary::dfs::<_, 2, STATS>(root, target, rules, &self.screens, sink, stats)
            }
            Mode::ReferenceCutoff(k) => {
                self.prefix::<_, STATS, false, 0>(root, k, rules, sink, stats)
            }
            Mode::CompactCutoff(k) => self.prefix::<_, STATS, true, 1>(root, k, rules, sink, stats),
            Mode::Terminal { cutoff, .. } => {
                assert!(root.k <= self.table.as_ref().unwrap().start());
                self.prefix::<_, STATS, true, 2>(root, cutoff, rules, sink, stats);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv::rules::Walkback;
    use cv::sample::{Rng, Sampler};

    #[derive(Default)]
    struct Nodes(Vec<Node>);

    impl Sink for Nodes {
        fn endpoint(&mut self, node: &Node) {
            self.0.push(*node);
        }
    }

    fn same_stats(a: &TreeStats, b: &TreeStats) {
        assert_eq!(a.generated, b.generated);
        assert_eq!(a.survived, b.survived);
        assert_eq!(a.culls, b.culls);
    }

    fn seeded_stats(target: u32) -> TreeStats {
        let mut st = TreeStats::new(target);
        for k in 0..=target as usize {
            st.generated[k] = 17 + k as u64;
            st.survived[k] = 31 + k as u64;
            for (why, count) in st.culls[k].iter_mut().enumerate() {
                *count = 5 + (k + why) as u64;
            }
        }
        st
    }

    // Independent two-stage oracle: ordinary DFS materializes the complete
    // cutoff frontier, then ordinary baseline DFS expands every survivor.
    fn oracle(root: &Node, target: u32, cutoff: u32, rules: &Rules) -> (Nodes, TreeStats) {
        let mut st = seeded_stats(target);
        let mut frontier = Nodes::default();
        dfs::<_, true>(root, cutoff.max(root.k), rules, &mut frontier, &mut st);
        let mut base = rules.clone();
        base.pm_inf = false;
        base.mix = None;
        base.walkback = None;
        let mut endpoints = Nodes::default();
        for node in frontier.0 {
            dfs::<_, true>(&node, target, &base, &mut endpoints, &mut st);
        }
        endpoints.0.sort_unstable_by_key(|n| n.n0);
        (endpoints, st)
    }

    fn audit(root: &Node, target: u32, rules: &Rules, modes: &[Mode]) {
        for &mode in modes {
            let cutoff = mode.cutoff().unwrap_or(target);
            let (expected, expected_stats) = oracle(root, target, cutoff, rules);
            let engine = Combined::new(target, rules, mode);
            let mut got = Nodes::default();
            let mut st = seeded_stats(target);
            engine.expand::<_, true>(root, target, rules, &mut got, &mut st);
            got.0.sort_unstable_by_key(|n| n.n0);
            assert_eq!(got.0, expected.0, "complete Nodes differ for {mode:?}");
            assert!(got.0.windows(2).all(|p| p[0].n0 != p[1].n0));
            same_stats(&st, &expected_stats);

            let mut unrecorded = Nodes::default();
            let mut st = seeded_stats(target);
            engine.expand::<_, false>(root, target, rules, &mut unrecorded, &mut st);
            unrecorded.0.sort_unstable_by_key(|n| n.n0);
            assert_eq!(unrecorded.0, expected.0);
            same_stats(&st, &seeded_stats(target));
        }
    }

    #[test]
    fn combined_complete_small_tree_matches_independent_two_stage_oracle() {
        let mut rules = Rules::improved(7);
        rules.walkback = Some(Walkback {
            depth: 10,
            max_states: 256,
        });
        let root = Node::root(&rules);
        audit(&root, 22, &rules, &[Mode::Full, Mode::Screened]);
        for cutoff in [10, 14, 18] {
            audit(
                &root,
                22,
                &rules,
                &[
                    Mode::ReferenceCutoff(cutoff),
                    Mode::CompactCutoff(cutoff),
                    Mode::Terminal { cutoff, width: 4 },
                    Mode::Terminal {
                        cutoff,
                        width: 22 - cutoff,
                    },
                ],
            );
        }
    }

    #[test]
    fn combined_reconstructed_nodes_match_at_target64_and_wrapped_iterates() {
        let rules = Rules::improved(7);
        let sampler = Sampler::new(&Rules::baseline(), 54, Some((42, 54)));
        let mut rng = Rng(31_417);
        let mut roots = Vec::new();
        while roots.len() < 8 {
            let draw = sampler.draw(&mut rng, Some(&rules));
            if draw.culled.is_none() && draw.node.m > u64::MAX as u128 {
                roots.push(draw.node);
            }
        }
        let mut all_odd = Node::root(&rules);
        while all_odd.k < 54 {
            all_odd = all_odd
                .children(&rules)
                .into_iter()
                .flatten()
                .find(|n| n.f == all_odd.f + 1)
                .unwrap();
        }
        roots.push(all_odd);
        for root in roots {
            audit(
                &root,
                64,
                &rules,
                &[
                    Mode::Full,
                    Mode::Screened,
                    Mode::ReferenceCutoff(54),
                    Mode::CompactCutoff(54),
                    Mode::Terminal {
                        cutoff: 54,
                        width: 8,
                    },
                    Mode::Terminal {
                        cutoff: 58,
                        width: 6,
                    },
                ],
            );
        }
        let (nodes, _) = oracle(&all_odd, 64, 54, &rules);
        assert!(nodes.0.iter().any(|n| n.n0 == u64::MAX as u128));
    }

    #[test]
    fn combined_terminal_dispatches_compact_payloads_to_overridden_sink() {
        #[derive(Default)]
        struct Direct(Vec<(u64, u32, u32, u8, u8)>);
        impl Sink for Direct {
            fn endpoint(&mut self, _: &Node) {
                panic!("terminal endpoints must use the compact sink hook");
            }
            fn compact_endpoint(&mut self, n0: u64, k: u32, f: u32, oee: u8, ne: u8) {
                self.0.push((n0, k, f, oee, ne));
            }
        }
        let rules = Rules::improved(7);
        let root = Node::root(&rules);
        let (expected, _) = oracle(&root, 20, 10, &rules);
        let engine = Combined::new(
            20,
            &rules,
            Mode::Terminal {
                cutoff: 10,
                width: 8,
            },
        );
        let mut sink = Direct::default();
        engine.expand::<_, true>(&root, 20, &rules, &mut sink, &mut TreeStats::new(20));
        sink.0.sort_unstable();
        let expected: Vec<_> = expected
            .0
            .iter()
            .map(|n| (n.n0 as u64, n.k, n.f, n.oee, n.ne))
            .collect();
        assert_eq!(sink.0, expected);
    }

    #[test]
    #[should_panic(expected = "cannot bypass full prefix")]
    fn combined_rejects_table_that_bypasses_full_checks() {
        Combined::new(
            62,
            &Rules::improved(0),
            Mode::Terminal {
                cutoff: 54,
                width: 10,
            },
        );
    }

    #[test]
    #[should_panic(expected = "checkpoint must precede")]
    fn combined_rejects_checkpoint_after_cutoff() {
        let mut rules = Rules::improved(0);
        rules.walkback = Some(Walkback {
            depth: 54,
            max_states: 256,
        });
        Combined::new(62, &rules, Mode::CompactCutoff(50));
    }
}
