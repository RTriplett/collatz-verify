//! Baseline-only expansion with the endpoint payload used by the GPU producer.
//!
//! Once value-dependent checks have stopped, the tree needs only the parity of
//! each successive iterate. Starting with `m mod 2^64`, each even or odd step
//! preserves one fewer low bit. A real root has k >= 2, and target <= 64, so at
//! most 62 steps are taken: every parity decision is exact despite wrapping.
//!
//! The omitted `m < n0` test cannot add a baseline cull in this range. For all
//! integer 0 <= f <= k <= 64, the rational slope predicate agrees with
//! `3^f <= 2^k` (exhaustively checked below using exact integer arithmetic).
//! If that predicate is false, the affine iterate
//! `m = (3^f*n0 + c)/2^k`, with c >= 0, is at least n0. If it is true, both
//! implementations attribute the removal to `Cull::Descent` anyway.
//!
//! This emits only n0/f/oee/ne; it does not pretend to reconstruct full m/m3.
use cv::arith::slope_descends;
use cv::rules::{Cull, Rules};
use cv::tree::{next_even_run, Node, TreeStats};

const POW3_U64: [u64; 65] = {
    let mut values = [1u64; 65];
    let mut i = 1;
    while i < values.len() {
        values[i] = values[i - 1].wrapping_mul(3);
        i += 1;
    }
    values
};

/// The exact payload consumed by the production GPU endpoint producer.
pub trait Endpoints {
    fn endpoint(&mut self, n0: u64, f: u32, oee: u8, ne: u8);
}

#[derive(Clone, Copy)]
struct SmallNode {
    n0: u64,
    m: u64,
    k: u32,
    f: u32,
    oee: u8,
    ne: u8,
}

impl SmallNode {
    #[inline(always)]
    fn children(self, rules: &Rules) -> [Option<Self>; 2] {
        let arm = u8::from(rules.oee && slope_descends(self.k + 1, self.f));
        let odd_state = (self.oee | arm) & !2;
        let even_ok = self.oee < 2;
        let hi = 1u64 << self.k;
        let odd = |n0: u64, x: u64| Self {
            n0,
            m: x.wrapping_add(x >> 1).wrapping_add(1),
            k: self.k + 1,
            f: self.f + 1,
            oee: odd_state,
            ne: 0,
        };
        let even = |n0: u64, x: u64| Self {
            n0,
            m: x >> 1,
            k: self.k + 1,
            f: self.f,
            oee: self.oee << 1,
            ne: next_even_run(self.ne),
        };
        let x1 = self.m.wrapping_add(POW3_U64[self.f as usize]);
        if self.m & 1 == 0 {
            [
                even_ok.then(|| even(self.n0, self.m)),
                Some(odd(self.n0 | hi, x1)),
            ]
        } else {
            [
                Some(odd(self.n0, self.m)),
                even_ok.then(|| even(self.n0 | hi, x1)),
            ]
        }
    }

    #[inline(always)]
    fn cull(self, rules: &Rules) -> Option<Cull> {
        if slope_descends(self.k, self.f) {
            return Some(Cull::Descent);
        }
        if rules.pm1 && self.ne == 2 && slope_descends(self.k - 1, self.f - 1) {
            return Some(Cull::Pm1);
        }
        None
    }
}

/// Expand a genuine, canonical, already-surviving tree node using only the
/// baseline parity-state rules. The supplied root's OEE/even state is retained.
/// As with the reference DFS, the root itself is neither culled nor counted.
pub fn dfs<S: Endpoints, const STATS: bool>(
    root: &Node,
    target: u32,
    rules: &Rules,
    sink: &mut S,
    stats: &mut TreeStats,
) {
    assert!(
        rules.is_parity_only(),
        "fast tail requires parity-only rules"
    );
    assert!((2..=64).contains(&root.k) && root.k <= target && target <= 64);
    assert!(root.f <= root.k && root.n0 < (1u128 << root.k));
    expand::<S, STATS>(
        SmallNode {
            n0: root.n0 as u64,
            m: root.m as u64,
            k: root.k,
            f: root.f,
            oee: root.oee,
            ne: root.ne,
        },
        target,
        rules,
        sink,
        stats,
    );
}

fn expand<S: Endpoints, const STATS: bool>(
    node: SmallNode,
    target: u32,
    rules: &Rules,
    sink: &mut S,
    stats: &mut TreeStats,
) {
    if node.k == target {
        sink.endpoint(node.n0, node.f, node.oee, node.ne);
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
        if let Some(why) = child.cull(rules) {
            if STATS {
                stats.culls[child.k as usize][why as usize] += 1;
            }
        } else {
            if STATS {
                stats.survived[child.k as usize] += 1;
            }
            expand::<S, STATS>(child, target, rules, sink, stats);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv::arith::POW3;
    use cv::sample::{Rng, Sampler};
    use cv::tree::{dfs as reference_dfs, Sink};

    #[derive(Default)]
    struct Payloads(Vec<(u64, u32, u8, u8)>);

    impl Endpoints for Payloads {
        fn endpoint(&mut self, n0: u64, f: u32, oee: u8, ne: u8) {
            self.0.push((n0, f, oee, ne));
        }
    }

    impl Sink for Payloads {
        fn endpoint(&mut self, node: &Node) {
            assert!(node.n0 <= u64::MAX as u128);
            self.0.push((node.n0 as u64, node.f, node.oee, node.ne));
        }
    }

    fn compare(roots: &[Node], target: u32, rules: &Rules) {
        let mut expected = Payloads::default();
        let mut actual = Payloads::default();
        let mut expected_stats = TreeStats::new(target);
        let mut actual_stats = TreeStats::new(target);
        for root in roots {
            reference_dfs::<_, true>(root, target, rules, &mut expected, &mut expected_stats);
            dfs::<_, true>(root, target, rules, &mut actual, &mut actual_stats);
        }
        assert!(!expected.0.is_empty(), "comparison must exercise endpoints");
        assert_eq!(actual.0, expected.0, "ordered endpoint payloads differ");
        assert_eq!(actual_stats.generated, expected_stats.generated);
        assert_eq!(actual_stats.survived, expected_stats.survived);
        assert_eq!(actual_stats.culls, expected_stats.culls);

        let mut unrecorded = Payloads::default();
        let mut empty_stats = TreeStats::default();
        for root in roots {
            dfs::<_, false>(root, target, rules, &mut unrecorded, &mut empty_stats);
        }
        assert_eq!(unrecorded.0, expected.0);
        assert!(empty_stats.generated.is_empty());
        assert!(empty_stats.survived.is_empty());
        assert!(empty_stats.culls.is_empty());
    }

    #[test]
    fn rational_slope_matches_exact_powers_through_64() {
        for k in 1..=64 {
            for f in 0..=k {
                assert_eq!(
                    slope_descends(k, f),
                    POW3[f as usize] <= (1u128 << k),
                    "slope discrepancy at k={k}, f={f}"
                );
            }
        }
        for (f, &value) in POW3_U64.iter().enumerate() {
            assert_eq!(value, POW3[f] as u64);
        }
    }

    #[test]
    fn small_complete_expansion_matches_payloads_and_counters() {
        let mut no_pm1 = Rules::baseline();
        no_pm1.pm1 = false;
        let mut unarmed = Rules::baseline();
        unarmed.root_oee = false;
        for rules in [Rules::baseline(), Rules::descent_only(), no_pm1, unarmed] {
            compare(&[Node::root(&rules)], 22, &rules);
        }
    }

    #[test]
    fn wide_sampled_expansion_matches_with_wrapped_iterates() {
        let mut saw_wide_m = false;
        let mut saw_top_n0_bit = false;
        for rules in [Rules::baseline(), Rules::descent_only()] {
            for (start, target) in [(38, 46), (50, 58), (58, 64), (64, 64)] {
                let sampler = Sampler::new(&rules, start, None);
                let mut rng = Rng(1_738_620 ^ start as u64);
                let roots: Vec<_> = (0..64).map(|_| sampler.draw(&mut rng, None).node).collect();
                saw_wide_m |= roots.iter().any(|node| node.m > u64::MAX as u128);
                saw_top_n0_bit |= roots.iter().any(|node| node.n0 >= (1u128 << 63));
                compare(&roots, target, &rules);
            }
        }
        assert!(saw_wide_m, "test must exercise truncated initial iterates");
        assert!(
            saw_top_n0_bit,
            "test must exercise the full u64 payload range"
        );
    }
}
