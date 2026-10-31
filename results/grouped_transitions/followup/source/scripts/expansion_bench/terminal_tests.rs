//! Integration audits against the original full-width production traversal.
//! These compare complete endpoint payloads, not the benchmark's checksum.
use crate::fast_tail::Endpoints;
use crate::tail_policy::{Engines, Policy};
use crate::terminal::Table;
use crate::terminal_walk;
use cv::rules::{walkback_certificate, Rules};
use cv::sample::{Rng, Sampler};
use cv::tree::{Node, Sink, TreeStats};
use std::sync::OnceLock;

type Payload = (u64, u32, u8, u8);

#[derive(Default)]
struct Payloads(Vec<Payload>);

impl Endpoints for Payloads {
    fn endpoint(&mut self, n0: u64, f: u32, oee: u8, ne: u8) {
        self.0.push((n0, f, oee, ne));
    }
}

impl Sink for Payloads {
    fn endpoint(&mut self, node: &Node) {
        assert!(node.n0 <= u64::MAX as u128);
        Endpoints::endpoint(self, node.n0 as u64, node.f, node.oee, node.ne);
    }
}

fn full_rules() -> &'static Rules {
    static RULES: OnceLock<Rules> = OnceLock::new();
    RULES.get_or_init(|| Rules::improved(13))
}

fn initial_stats(target: u32) -> TreeStats {
    let mut stats = TreeStats::new(target);
    // Nonzero initial values detect accidental replacement of the caller's
    // counters and accounting of the already-surviving root.
    for k in 0..=target as usize {
        stats.generated[k] = 31 + k as u64;
        stats.survived[k] = 47 + k as u64;
        for (why, count) in stats.culls[k].iter_mut().enumerate() {
            *count = 7 + (k + why) as u64;
        }
    }
    stats
}

fn assert_stats(a: &TreeStats, b: &TreeStats) {
    assert_eq!(a.generated, b.generated, "generated counts");
    assert_eq!(a.survived, b.survived, "survived counts");
    assert_eq!(a.culls, b.culls, "culls by rule and depth");
}

fn sort_unique(payloads: &mut Payloads) {
    payloads.0.sort_unstable();
    assert!(
        payloads.0.windows(2).all(|pair| pair[0].0 != pair[1].0),
        "an extension was emitted more than once"
    );
}

fn compare<const STATS: bool, const GROUPED: bool>(
    root: &Node,
    cutoff: u32,
    table: &Table,
    expected: &[Payload],
    expected_stats: &TreeStats,
) {
    let mut got = Payloads::default();
    let mut stats = initial_stats(table.target());
    terminal_walk::run::<_, STATS, GROUPED>(
        root,
        cutoff,
        full_rules(),
        &Rules::baseline(),
        table,
        &mut got,
        &mut stats,
    );
    sort_unique(&mut got);
    assert_eq!(
        got.0,
        expected,
        "root K{} F{}, cutoff {}, terminal start {}, stats {}, grouped {}",
        root.k,
        root.f,
        cutoff,
        table.start(),
        STATS,
        GROUPED
    );
    if STATS {
        assert_stats(&stats, expected_stats);
    } else {
        assert_stats(&stats, &initial_stats(table.target()));
    }
}

fn audit(roots: &[Node], target: u32, cutoff: u32, widths: &[u32]) {
    let base = Rules::baseline();
    let engines = Engines::new(full_rules().clone(), target);
    let tables: Vec<_> = widths
        .iter()
        .map(|&width| Table::new(target - width, target, &base))
        .collect();
    for root in roots {
        let mut expected = Payloads::default();
        let mut stats = initial_stats(target);
        // This path uses Node.children/cull and the original production DFS;
        // it shares neither the compact iterate nor terminal-word mapping.
        engines.run::<_, true>(Policy::Cutoff(cutoff), root, &mut expected, &mut stats);
        sort_unique(&mut expected);
        for table in &tables {
            compare::<true, false>(root, cutoff, table, &expected.0, &stats);
            compare::<true, true>(root, cutoff, table, &expected.0, &stats);
            compare::<false, false>(root, cutoff, table, &expected.0, &stats);
            compare::<false, true>(root, cutoff, table, &expected.0, &stats);
        }
    }
}

fn sampled_roots(depth: u32, seed: u64, count: usize, wide: bool) -> Vec<Node> {
    let base = Rules::baseline();
    let sampler = Sampler::new(&base, depth, wide.then_some((42, depth)));
    let mut rng = Rng(seed);
    let mut roots = Vec::new();
    for _ in 0..100_000 {
        let draw = sampler.draw(&mut rng, Some(full_rules()));
        if draw.culled.is_some() {
            continue;
        }
        let node = draw.node;
        if wide && node.m <= u64::MAX as u128 {
            continue;
        }
        if depth == 38 && walkback_certificate(node.n0, node.k, node.f, node.m, 65_536).is_some() {
            continue;
        }
        roots.push(node);
        if roots.len() == count {
            return roots;
        }
    }
    panic!("failed to draw the requested test corpus");
}

fn all_odd_root(depth: u32) -> Node {
    let full = full_rules();
    let mut node = Node::root(full);
    while node.k < depth {
        node = node
            .children(full)
            .into_iter()
            .flatten()
            .find(|child| child.f == node.f + 1)
            .unwrap();
        assert!(node.cull(full).is_none());
    }
    assert_eq!(node.n0, (1u128 << depth) - 1);
    node
}

#[test]
fn terminal_integration_complete_small_tree_and_all_widths() {
    let root = Node::root(full_rules());
    for cutoff in [8, 12, 16] {
        let widths: Vec<_> = (1..=(20 - cutoff).min(12)).collect();
        audit(&[root], 20, cutoff, &widths);
    }
}

#[test]
fn terminal_integration_sampled_checkpoint_and_compact_prefixes() {
    let roots = sampled_roots(38, 28_041, 12, false);
    for cutoff in [38, 40, 44] {
        let widths: Vec<_> = [1, 2, 4, 6, 8, 10, 12]
            .into_iter()
            .filter(|&width| width <= 50 - cutoff)
            .collect();
        audit(&roots, 50, cutoff, &widths);
    }
}

#[test]
fn terminal_integration_wrapped_iterates_and_bit_63() {
    let mut roots50 = sampled_roots(50, 28_042, 8, true);
    roots50.push(all_odd_root(50));
    assert!(roots50.iter().all(|node| node.m > u64::MAX as u128));
    // K50 -> K52 -> table exercises compact wrapping before a width-12
    // table; cutoff54 instead exercises full checks before smaller tables.
    audit(&roots50, 64, 50, &[4, 8, 10, 12]);
    audit(&roots50, 64, 54, &[4, 8, 10]);

    let mut roots54 = sampled_roots(54, 28_043, 4, true);
    roots54.push(all_odd_root(54));
    audit(&roots54, 64, 54, &[4, 8, 10]);

    let root = all_odd_root(50);
    let table = Table::new(52, 64, &Rules::baseline());
    let mut payloads = Payloads::default();
    terminal_walk::run::<_, true, false>(
        &root,
        50,
        full_rules(),
        &Rules::baseline(),
        &table,
        &mut payloads,
        &mut TreeStats::new(64),
    );
    assert!(payloads.0.iter().any(|p| p.0 == u64::MAX && p.1 == 64));
}

#[test]
#[should_panic]
fn terminal_integration_rejects_table_before_full_check_cutoff() {
    let root = all_odd_root(50);
    let base = Rules::baseline();
    let table = Table::new(52, 64, &base);
    terminal_walk::run::<_, true, false>(
        &root,
        54,
        full_rules(),
        &base,
        &table,
        &mut Payloads::default(),
        &mut TreeStats::new(64),
    );
}
