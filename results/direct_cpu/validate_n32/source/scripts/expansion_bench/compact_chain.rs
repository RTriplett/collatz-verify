//! Keep a two-stage affine expansion compressed from its entry to GPU dispatch.
//!
//! Runtime host work visits entry nodes and emits one program reference per
//! final odd-count class. Programs refer to short first/second transition tables;
//! they never store composed full-width words or materialize intermediate nodes.
//! On the GPU each first-stage result can be shared by its second-stage range.
use crate::terminal::Table;
use crate::ternary::{self, Screens};
use cv::dp::ForwardDp;
use cv::expansion::Expander;
use cv::rules::{Cull, Rules};
use cv::tree::{ChainLeaf, ChainProgram, ChainRoute, ChainTables, Node, Sink, TreeStats};
use std::collections::BTreeMap;
use std::mem::size_of;

fn index(f: u32, oee: u8, ne: u8) -> usize {
    (f as usize * 3 + oee as usize) * 4 + ne as usize
}

struct StateProgram {
    programs: Vec<u32>,
    stats: TreeStats,
}

pub struct CompactChain {
    target: u32,
    cutoff: u32,
    entry: u32,
    first_width: u32,
    baseline: Rules,
    screens: Screens,
    tables: ChainTables,
    states: Vec<Option<StateProgram>>,
}

impl CompactChain {
    pub fn new(target: u32, cutoff: u32, entry: u32, first_width: u32, rules: &Rules) -> Self {
        assert!(2 <= cutoff && cutoff <= entry && entry < target && target <= 64);
        assert!(
            (1..=6).contains(&first_width),
            "GPU shared prefix storage supports at most 64 words"
        );
        let middle = entry + first_width;
        assert!(middle < target && target - middle <= 12);
        assert!(rules.walkback.is_none_or(|w| w.depth <= cutoff));
        let mut baseline = rules.clone();
        baseline.pm_inf = false;
        baseline.mix = None;
        baseline.walkback = None;
        let first = Table::new(entry, middle, &baseline);
        let second = Table::new(middle, target, &baseline);
        let second_offset = u32::try_from(first.words().len()).unwrap();
        let mut tables = ChainTables {
            programs: Vec::new(),
            routes: Vec::new(),
            words: first.words().to_vec(),
            leaves: Vec::new(),
        };
        tables.words.extend_from_slice(second.words());
        let mut states: Vec<Option<StateProgram>> =
            (0..(entry as usize + 1) * 12).map(|_| None).collect();
        let reachable = ForwardDp::from_root(&baseline, entry);
        for (state, _) in reachable.layer(entry).states() {
            // The table only inspects this node's parity state, not its residue.
            let parent = Node {
                n0: 0,
                k: entry,
                f: state.f,
                m: 0,
                m3: 0,
                oee: state.oee,
                ne: state.ne,
            };
            let first_groups = first.groups(&parent);
            let first_start = first_groups.first().map_or(0, |g| g.start);
            let first_len: u32 = first_groups.iter().map(|g| g.len).sum();
            let mut stats = TreeStats::new(target);
            first.accumulate_stats(state.f, state.oee, state.ne, &mut stats);
            let mut by_f: BTreeMap<u32, Vec<ChainRoute>> = BTreeMap::new();
            for first_index in first_start..first_start + first_len {
                let word = &first.words()[first_index as usize];
                let prefix = Node {
                    n0: 0,
                    k: middle,
                    f: word.f,
                    m: 0,
                    m3: 0,
                    oee: word.oee as u8,
                    ne: word.ne as u8,
                };
                second.accumulate_stats(prefix.f, prefix.oee, prefix.ne, &mut stats);
                for group in second.groups(&prefix) {
                    by_f.entry(group.f).or_default().push(ChainRoute {
                        first_index,
                        second_start: second_offset + group.start,
                        second_len: group.len,
                        output_start: 0,
                    });
                }
            }
            let mut programs = Vec::new();
            for (_, mut routes) in by_f {
                let mut endpoints = 0;
                let leaf_start = u32::try_from(tables.leaves.len()).unwrap();
                for route in &mut routes {
                    route.output_start = endpoints;
                    endpoints += route.second_len;
                    for second_index in route.second_start..route.second_start + route.second_len {
                        tables.leaves.push(ChainLeaf {
                            first_slot: route.first_index - first_start,
                            second_index,
                        });
                    }
                }
                debug_assert!(endpoints > 0 && routes.len() <= first_len as usize);
                programs.push(u32::try_from(tables.programs.len()).unwrap());
                tables.programs.push(ChainProgram {
                    route_start: u32::try_from(tables.routes.len()).unwrap(),
                    route_len: u32::try_from(routes.len()).unwrap(),
                    endpoints,
                    first_width,
                    second_width: target - middle,
                    first_start,
                    first_len,
                    leaf_start,
                });
                tables.routes.extend(routes);
            }
            states[index(state.f, state.oee, state.ne)] = Some(StateProgram { programs, stats });
        }
        Self {
            target,
            cutoff,
            entry,
            first_width,
            baseline,
            screens: Screens::new(target),
            tables,
            states,
        }
    }

    pub fn name(&self) -> String {
        format!(
            "compact{}x{}+{}",
            self.entry,
            self.first_width,
            self.target - self.entry - self.first_width
        )
    }

    pub fn tables(&self) -> &ChainTables {
        &self.tables
    }

    pub fn table_bytes(&self) -> usize {
        size_of::<Self>()
            + self.tables.programs.capacity() * size_of::<ChainProgram>()
            + self.tables.routes.capacity() * size_of::<ChainRoute>()
            + self.tables.words.capacity() * size_of::<cv::tree::TerminalWord>()
            + self.tables.leaves.capacity() * size_of::<ChainLeaf>()
            + self.states.capacity() * size_of::<Option<StateProgram>>()
            + self
                .states
                .iter()
                .flatten()
                .map(|state| {
                    state.programs.capacity() * size_of::<u32>()
                        + state.stats.generated.capacity() * size_of::<u64>()
                        + state.stats.survived.capacity() * size_of::<u64>()
                        + state.stats.culls.capacity() * size_of::<[u64; cv::rules::NUM_CULLS]>()
                })
                .sum::<usize>()
    }

    fn prefix<S: Sink, const STATS: bool>(
        &self,
        node: &Node,
        rules: &Rules,
        sink: &mut S,
        stats: &mut TreeStats,
    ) {
        if node.k == self.entry {
            let state = self.states[index(node.f, node.oee, node.ne)]
                .as_ref()
                .expect("unreachable compact-chain entry state");
            if STATS {
                // All cached counters through entry are zero; prefix traversal
                // already accounted for those depths. Merge only chain work.
                for k in self.entry as usize + 1..=self.target as usize {
                    stats.generated[k] += state.stats.generated[k];
                    stats.survived[k] += state.stats.survived[k];
                    for (total, count) in stats.culls[k].iter_mut().zip(state.stats.culls[k]) {
                        *total += count;
                    }
                }
            }
            for &program_index in &state.programs {
                let program = &self.tables.programs[program_index as usize];
                let start = program.route_start as usize;
                sink.chained_group(
                    node,
                    program_index,
                    program,
                    &self.tables.routes[start..start + program.route_len as usize],
                    &self.tables.words,
                    self.target,
                );
            }
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
                self.prefix::<S, STATS>(&child, rules, sink, stats);
            }
        }
    }
}

impl Expander for CompactChain {
    fn expand<S: Sink, const STATS: bool>(
        &self,
        root: &Node,
        target: u32,
        rules: &Rules,
        sink: &mut S,
        stats: &mut TreeStats,
    ) {
        assert_eq!(target, self.target);
        assert!(root.k >= 2 && root.k <= self.entry);
        assert_eq!(
            (rules.pm1, rules.oee, rules.root_oee),
            (self.baseline.pm1, self.baseline.oee, self.baseline.root_oee)
        );
        assert!(rules.walkback.is_none_or(|w| w.depth <= self.cutoff));
        self.prefix::<S, STATS>(root, rules, sink, stats);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affine::Affine;
    use cv::rules::Walkback;
    use cv::tree::dfs;

    #[derive(Default)]
    struct Nodes(Vec<Node>);
    impl Sink for Nodes {
        fn endpoint(&mut self, node: &Node) {
            self.0.push(*node);
        }
    }

    fn compare(
        root: &Node,
        target: u32,
        cutoff: u32,
        entry: u32,
        width: u32,
        rules: &Rules,
    ) -> Vec<Node> {
        let chain = CompactChain::new(target, cutoff, entry, width, rules);
        let mut boundary = Nodes::default();
        let mut expected_stats = TreeStats::new(target);
        dfs::<_, true>(
            root,
            cutoff.max(root.k),
            rules,
            &mut boundary,
            &mut expected_stats,
        );
        let mut expected = Nodes::default();
        for parent in boundary.0 {
            dfs::<_, true>(
                &parent,
                target,
                &chain.baseline,
                &mut expected,
                &mut expected_stats,
            );
        }
        expected.0.sort_unstable_by_key(|n| n.n0);
        let affine = Affine::new(target, cutoff, entry, rules);
        let mut affine_nodes = Nodes::default();
        let mut affine_stats = TreeStats::new(target);
        affine.expand::<_, true>(root, target, rules, &mut affine_nodes, &mut affine_stats);
        affine_nodes.0.sort_unstable_by_key(|n| n.n0);
        assert_eq!(affine_nodes.0, expected.0);
        assert_eq!(affine_stats.generated, expected_stats.generated);
        assert_eq!(affine_stats.survived, expected_stats.survived);
        assert_eq!(affine_stats.culls, expected_stats.culls);

        let mut actual = Nodes::default();
        let mut actual_stats = TreeStats::new(target);
        chain.expand::<_, true>(root, target, rules, &mut actual, &mut actual_stats);
        actual.0.sort_unstable_by_key(|n| n.n0);
        assert_eq!(actual.0, expected.0);
        assert_eq!(actual_stats.generated, expected_stats.generated);
        assert_eq!(actual_stats.survived, expected_stats.survived);
        assert_eq!(actual_stats.culls, expected_stats.culls);
        let mut unrecorded = Nodes::default();
        let mut untouched = TreeStats::default();
        chain.expand::<_, false>(root, target, rules, &mut unrecorded, &mut untouched);
        unrecorded.0.sort_unstable_by_key(|n| n.n0);
        assert_eq!(unrecorded.0, expected.0);
        assert!(
            untouched.generated.is_empty()
                && untouched.survived.is_empty()
                && untouched.culls.is_empty()
        );
        expected.0
    }

    #[test]
    fn compact_chain_matches_exact_nodes_and_all_prefix_counters() {
        let mut rules = Rules::improved(7);
        rules.walkback = Some(Walkback {
            depth: 10,
            max_states: 256,
        });
        for (entry, width) in [(10, 4), (10, 6), (12, 4), (14, 3)] {
            compare(&Node::root(&rules), 22, 10, entry, width, &rules);
        }
    }

    #[test]
    fn compact_chain_keeps_bit63_and_wide_iterates() {
        let rules = Rules::improved(7);
        let mut root = Node::root(&rules);
        while root.k < 52 {
            root = root
                .children(&rules)
                .into_iter()
                .flatten()
                .find(|child| child.f == root.f + 1)
                .unwrap();
        }
        assert!(root.m > u64::MAX as u128);
        for (cutoff, entry, width) in [(50, 52, 6), (52, 52, 4), (50, 54, 4)] {
            let nodes = compare(&root, 64, cutoff, entry, width, &rules);
            assert!(nodes.iter().any(|n| n.n0 & (1u128 << 63) != 0));
            assert!(nodes.iter().any(|n| n.m > u64::MAX as u128));
        }
    }

    #[test]
    fn compact_chain_dispatch_does_not_emit_host_nodes() {
        struct Dispatch {
            groups: usize,
            endpoints: usize,
            prefixes: usize,
        }
        impl Sink for Dispatch {
            fn endpoint(&mut self, _: &Node) {
                panic!("compact dispatch materialized an endpoint");
            }
            fn chained_group(
                &mut self,
                _: &Node,
                _: u32,
                program: &ChainProgram,
                routes: &[ChainRoute],
                words: &[cv::tree::TerminalWord],
                _: u32,
            ) {
                assert_eq!(routes.len(), program.route_len as usize);
                let mut count = 0;
                let final_f = words[routes[0].second_start as usize].f;
                for route in routes {
                    assert_eq!(route.output_start, count);
                    assert!(
                        (program.first_start..program.first_start + program.first_len)
                            .contains(&route.first_index)
                    );
                    for word in &words[route.second_start as usize
                        ..(route.second_start + route.second_len) as usize]
                    {
                        assert_eq!(word.f, final_f);
                    }
                    count += route.second_len;
                }
                assert_eq!(count, program.endpoints);
                self.groups += 1;
                self.endpoints += count as usize;
                self.prefixes += routes.len();
            }
        }
        let rules = Rules::baseline();
        let root = Node::root(&rules);
        let chain = CompactChain::new(22, 10, 10, 6, &rules);
        for program in &chain.tables.programs {
            assert!(program.first_len <= 64);
            let expected: Vec<_> = chain.tables.routes
                [program.route_start as usize..(program.route_start + program.route_len) as usize]
                .iter()
                .flat_map(|route| {
                    (route.second_start..route.second_start + route.second_len).map(
                        move |second_index| ChainLeaf {
                            first_slot: route.first_index - program.first_start,
                            second_index,
                        },
                    )
                })
                .collect();
            assert_eq!(expected.len(), program.endpoints as usize);
            assert_eq!(
                &chain.tables.leaves[program.leaf_start as usize
                    ..(program.leaf_start + program.endpoints) as usize],
                expected.as_slice()
            );
        }
        let mut dispatch = Dispatch {
            groups: 0,
            endpoints: 0,
            prefixes: 0,
        };
        let mut stats = TreeStats::new(22);
        chain.expand::<_, true>(&root, 22, &rules, &mut dispatch, &mut stats);
        assert_eq!(dispatch.endpoints as u64, stats.survived[22]);
        assert!(dispatch.groups > 0 && dispatch.prefixes < dispatch.endpoints);
        assert_eq!(size_of::<ChainProgram>(), 32);
        assert_eq!(size_of::<ChainRoute>(), 16);
        assert_eq!(size_of::<ChainLeaf>(), 8);
    }
}
