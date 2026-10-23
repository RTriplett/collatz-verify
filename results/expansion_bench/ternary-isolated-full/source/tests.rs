//! Reproducible correctness checks for the experimental expansion engines.
//! These compare complete endpoint values, independently of timed digests.

use super::{blocks, gating, Collect, Engines, Variant, VARIANTS};
use cv::arith::POW3;
use cv::rules::{Rules, Walkback};
use cv::sample::{Rng, Sampler};
use cv::tree::{Node, P16};

fn sorted_endpoints(engines: &Engines, variant: Variant, roots: &[Node]) -> Vec<Node> {
    let mut sink = Collect::default();
    for root in roots {
        engines.run(variant, root, &mut sink);
    }
    sink.0.sort_unstable_by_key(|node| node.n0);
    sink.0
}

fn compare_engines(start: u32, target: u32, rules: &Rules, roots: &[Node]) {
    assert!(!roots.is_empty());
    let engines = Engines::new(start, target, rules);
    let expected = sorted_endpoints(&engines, Variant::Reference, roots);
    assert!(
        !expected.is_empty(),
        "audit must exercise surviving endpoints"
    );
    for variant in VARIANTS.into_iter().skip(1) {
        let actual = sorted_endpoints(&engines, variant, roots);
        assert_eq!(
            actual,
            expected,
            "{} differs for {} at {start}->{target}",
            variant.name(),
            rules.label()
        );
    }
    // Width one exercises the same compiled transition machinery without
    // relying on any multi-step block or terminal partial-block behavior.
    let one_step = blocks::Blocks::new(start, target, 1, rules);
    let mut actual = Collect::default();
    for root in roots {
        blocks::dfs_blocks(root, target, rules, &one_step, &mut actual);
    }
    actual.0.sort_unstable_by_key(|node| node.n0);
    assert_eq!(
        actual.0,
        expected,
        "width-one block differs for {}",
        rules.label()
    );
}

#[test]
fn complete_small_frontiers_match_all_node_fields() {
    let mut checkpoint = Rules::improved(5);
    checkpoint.mix_from = 15;
    checkpoint.walkback = Some(Walkback {
        depth: 18,
        max_states: 64,
    });
    for rules in [
        Rules::baseline(),
        Rules::descent_only(),
        Rules::improved(5),
        checkpoint,
    ] {
        compare_engines(2, 22, &rules, &[Node::root(&rules)]);
    }
}

#[test]
fn sampled_wide_frontiers_match_all_node_fields() {
    let baseline = Rules::baseline();
    for rules in [Rules::baseline(), Rules::improved(13)] {
        for (start, target) in [(38, 44), (50, 56)] {
            let sampler = Sampler::new(&baseline, start, None);
            let mut rng = Rng(17_173 ^ start as u64);
            let roots: Vec<_> = (0..64)
                .map(|_| sampler.draw(&mut rng, Some(&rules)))
                .filter(|draw| draw.culled.is_none())
                .map(|draw| draw.node)
                .collect();
            compare_engines(start, target, &rules, &roots);
        }
    }
}

#[test]
fn compact_mix_threshold_matches_full_gate_on_tree_states() {
    // f=0 cannot select a mixed word, and J=0 disables the mixed table.
    // All actual mixed-rule applications have J>=1 and 1<=f<=k.
    for j in 1..=16 {
        let gates = gating::Gates::new(80, j);
        for k in 1..=80 {
            for f in 1..=k {
                assert_eq!(
                    gates.mix_possible_compact(k, f),
                    gates.mix_possible(k, f),
                    "mixed-word gate differs at J={j}, k={k}, f={f}"
                );
            }
        }
    }
}

#[test]
fn maximal_ternary_valuation_uses_full_width_fallback() {
    let rules = Rules::improved(0);
    let gates = gating::Gates::new(80, 0);
    let mut kept = 0usize;
    let mut culled = 0usize;
    for k in [30, 38, 50, 62, 72] {
        for f in 1..=k {
            for exponent in 16..=40 {
                for multiplier in [1u128, 2, 7, 123] {
                    let m = POW3[exponent] * multiplier - 1;
                    for n0 in [m / 2, m, m + 1, 2 * m] {
                        let node = Node {
                            n0,
                            k,
                            f,
                            m,
                            m3: (P16 - 1) as u32,
                            oee: 0,
                            ne: 0,
                        };
                        let expected = rules.extra_cull(n0, k, f, m, node.m3).is_some();
                        assert_eq!(
                            gating::extra_culls_ternary(&node, &rules, &gates),
                            expected,
                            "ternary fallback differs at k={k}, f={f}, m={m}, n0={n0}"
                        );
                        if expected {
                            culled += 1;
                        } else {
                            kept += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(kept + culled, 100_800);
    assert!(kept > 0 && culled > 0, "audit must exercise both outcomes");
}
