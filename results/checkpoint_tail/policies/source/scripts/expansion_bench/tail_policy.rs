//! Rule-policy experiments below a frontier that already survived the full rules.
//! Disabling extra rules admits more work; it does not reject additional classes.

use crate::ternary::{self, Screens};
use cv::rules::{Cull, Rules};
use cv::tree::{dfs, Node, Sink, TreeStats};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    Full,
    Screened,
    Baseline,
    PmOnly,
    PmScreened,
    /// Apply all rules through this depth, then only the baseline rules.
    Cutoff(u32),
}

impl Policy {
    pub fn name(self) -> String {
        match self {
            Self::Full => "full".into(),
            Self::Screened => "screened".into(),
            Self::Baseline => "baseline".into(),
            Self::PmOnly => "pm_only".into(),
            Self::PmScreened => "pm_screened".into(),
            Self::Cutoff(k) => format!("cutoff{k}"),
        }
    }
}

pub struct Engines {
    full: Rules,
    baseline: Rules,
    pm_only: Rules,
    target: u32,
    screens: Screens,
}

impl Engines {
    pub fn new(full: Rules, target: u32) -> Self {
        // The benchmark compares the Angeltveit baseline against its extra
        // pruning rules, so all policies must agree on the parity-state rules.
        assert!(full.pm1 && full.oee && full.root_oee);
        Self {
            full,
            baseline: Rules::baseline(),
            pm_only: Rules::improved(0),
            target,
            screens: Screens::new(target),
        }
    }

    /// `root` must already have survived the complete checkpoint policy.
    pub fn run<S: Sink, const STATS: bool>(
        &self,
        policy: Policy,
        root: &Node,
        sink: &mut S,
        stats: &mut TreeStats,
    ) {
        assert!(root.k <= self.target);
        match policy {
            Policy::Full => dfs::<S, STATS>(root, self.target, &self.full, sink, stats),
            Policy::Screened => ternary::dfs::<S, 2, STATS>(
                root,
                self.target,
                &self.full,
                &self.screens,
                sink,
                stats,
            ),
            Policy::Baseline => dfs::<S, STATS>(root, self.target, &self.baseline, sink, stats),
            Policy::PmOnly => dfs::<S, STATS>(root, self.target, &self.pm_only, sink, stats),
            Policy::PmScreened => ternary::dfs::<S, 2, STATS>(
                root,
                self.target,
                &self.pm_only,
                &self.screens,
                sink,
                stats,
            ),
            Policy::Cutoff(k) if k <= root.k => {
                dfs::<S, STATS>(root, self.target, &self.baseline, sink, stats)
            }
            Policy::Cutoff(k) if k >= self.target => {
                dfs::<S, STATS>(root, self.target, &self.full, sink, stats)
            }
            Policy::Cutoff(k) => self.until_cutoff::<S, STATS>(root, k, sink, stats),
        }
    }

    fn until_cutoff<S: Sink, const STATS: bool>(
        &self,
        node: &Node,
        cutoff: u32,
        sink: &mut S,
        stats: &mut TreeStats,
    ) {
        if node.k == cutoff {
            // The original DFS below this point has no per-node cutoff branch.
            dfs::<S, STATS>(node, self.target, &self.baseline, sink, stats);
            return;
        }
        for child in node.children(&self.full) {
            let Some(child) = child else {
                if STATS {
                    stats.culls[node.k as usize + 1][Cull::Oee as usize] += 1;
                }
                continue;
            };
            if STATS {
                stats.generated[child.k as usize] += 1;
            }
            match child.cull(&self.full) {
                Some(why) => {
                    if STATS {
                        stats.culls[child.k as usize][why as usize] += 1;
                    }
                }
                None => {
                    if STATS {
                        stats.survived[child.k as usize] += 1;
                    }
                    self.until_cutoff::<S, STATS>(&child, cutoff, sink, stats);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[derive(Default)]
    struct Collect(Vec<Node>);

    impl Sink for Collect {
        fn endpoint(&mut self, node: &Node) {
            self.0.push(*node);
        }
    }

    fn assert_stats_eq(a: &TreeStats, b: &TreeStats) {
        assert_eq!(a.generated, b.generated);
        assert_eq!(a.survived, b.survived);
        assert_eq!(a.culls, b.culls);
    }

    // Independent breadth-first expansion, switching policies between the two
    // phases. It does not call Engines::run or either recursive DFS.
    fn expand_frontier(
        mut level: Vec<Node>,
        target: u32,
        rules: &Rules,
        stats: &mut TreeStats,
    ) -> Vec<Node> {
        while level.first().is_some_and(|node| node.k < target) {
            let mut next = Vec::new();
            for parent in level {
                for child in parent.children(rules) {
                    let Some(child) = child else {
                        stats.culls[parent.k as usize + 1][Cull::Oee as usize] += 1;
                        continue;
                    };
                    stats.generated[child.k as usize] += 1;
                    if let Some(why) = child.cull(rules) {
                        stats.culls[child.k as usize][why as usize] += 1;
                    } else {
                        stats.survived[child.k as usize] += 1;
                        next.push(child);
                    }
                }
            }
            level = next;
        }
        level
    }

    fn endpoints(engines: &Engines, policy: Policy, root: &Node) -> (Vec<Node>, TreeStats) {
        let mut output = Collect::default();
        let mut stats = TreeStats::new(engines.target);
        engines.run::<_, true>(policy, root, &mut output, &mut stats);
        output.0.sort_unstable_by_key(|node| node.n0);
        (output.0, stats)
    }

    #[test]
    fn cutoff_matches_two_stage_frontier_and_counters() {
        let engines = Engines::new(Rules::improved(5), 20);
        let root = Node::root(&engines.full);
        let (full, _) = endpoints(&engines, Policy::Full, &root);
        let (baseline, _) = endpoints(&engines, Policy::Baseline, &root);
        let baseline_keys: HashSet<_> = baseline.iter().map(|node| node.n0).collect();
        for cutoff in [2, 14, 16, 20, 22] {
            let (actual, actual_stats) = endpoints(&engines, Policy::Cutoff(cutoff), &root);
            let mut expected_stats = TreeStats::new(20);
            let checkpoint = expand_frontier(
                vec![root],
                cutoff.min(20),
                &engines.full,
                &mut expected_stats,
            );
            let mut expected =
                expand_frontier(checkpoint, 20, &engines.baseline, &mut expected_stats);
            expected.sort_unstable_by_key(|node| node.n0);
            assert_eq!(actual, expected, "cutoff={cutoff}");
            assert_stats_eq(&actual_stats, &expected_stats);
            let actual_keys: HashSet<_> = actual.iter().map(|node| node.n0).collect();
            assert!(full.iter().all(|node| actual_keys.contains(&node.n0)));
            assert!(actual.iter().all(|node| baseline_keys.contains(&node.n0)));

            let mut no_stats_output = Collect::default();
            engines.run::<_, false>(
                Policy::Cutoff(cutoff),
                &root,
                &mut no_stats_output,
                &mut TreeStats::default(),
            );
            no_stats_output.0.sort_unstable_by_key(|node| node.n0);
            assert_eq!(actual, no_stats_output.0);
        }
    }

    #[test]
    fn screened_policies_preserve_endpoints_and_counters() {
        let engines = Engines::new(Rules::improved(5), 20);
        let root = Node::root(&engines.full);
        for (plain, screened) in [
            (Policy::Full, Policy::Screened),
            (Policy::PmOnly, Policy::PmScreened),
        ] {
            let (plain_nodes, plain_stats) = endpoints(&engines, plain, &root);
            let (screened_nodes, screened_stats) = endpoints(&engines, screened, &root);
            assert_eq!(plain_nodes, screened_nodes);
            assert_stats_eq(&plain_stats, &screened_stats);
        }
        let (full, _) = endpoints(&engines, Policy::Full, &root);
        let (pm, _) = endpoints(&engines, Policy::PmOnly, &root);
        let (baseline, _) = endpoints(&engines, Policy::Baseline, &root);
        let pm_keys: HashSet<_> = pm.iter().map(|node| node.n0).collect();
        let baseline_keys: HashSet<_> = baseline.iter().map(|node| node.n0).collect();
        assert!(full.iter().all(|node| pm_keys.contains(&node.n0)));
        assert!(pm.iter().all(|node| baseline_keys.contains(&node.n0)));
    }
}
