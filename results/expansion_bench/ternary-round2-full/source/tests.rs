//! Reproducible correctness checks for the experimental expansion engines.
//! These compare complete endpoint values, independently of timed digests.

use super::{blocks, gating, ternary, Collect, Engines, Variant, VARIANTS};
use cv::arith::POW3;
use cv::rules::{pm_inf_culls, Rules, Walkback};
use cv::sample::{Rng, Sampler};
use cv::tree::{Node, TreeStats, P16};

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

fn recorded_endpoints(
    engines: &Engines,
    variant: Variant,
    roots: &[Node],
) -> (Vec<Node>, TreeStats) {
    let mut sink = Collect::default();
    let mut stats = TreeStats::new(engines.target);
    for root in roots {
        engines.run_recorded::<_, true>(variant, root, &mut sink, &mut stats);
    }
    sink.0.sort_unstable_by_key(|node| node.n0);
    (sink.0, stats)
}

fn compare_recorded_engines(start: u32, target: u32, rules: &Rules, roots: &[Node]) {
    assert!(!roots.is_empty());
    let engines = Engines::new(start, target, rules);
    let (expected_nodes, expected_stats) = recorded_endpoints(&engines, Variant::Reference, roots);
    assert!(!expected_nodes.is_empty());
    assert!(expected_stats.total_generated() > 0);
    for variant in VARIANTS
        .into_iter()
        .skip(1)
        .filter(|variant| variant.supports_stats())
    {
        let (actual_nodes, actual_stats) = recorded_endpoints(&engines, variant, roots);
        assert_eq!(
            actual_nodes,
            expected_nodes,
            "{} recorded endpoints differ for {} at {start}->{target}",
            variant.name(),
            rules.label(),
        );
        assert_eq!(
            actual_stats.generated,
            expected_stats.generated,
            "{} generated counts differ for {} at {start}->{target}",
            variant.name(),
            rules.label(),
        );
        assert_eq!(
            actual_stats.survived,
            expected_stats.survived,
            "{} survival counts differ for {} at {start}->{target}",
            variant.name(),
            rules.label(),
        );
        assert_eq!(
            actual_stats.culls,
            expected_stats.culls,
            "{} cull categories differ for {} at {start}->{target}",
            variant.name(),
            rules.label(),
        );
    }
}

#[test]
fn recorded_traversals_match_endpoints_and_every_counter() {
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
        compare_recorded_engines(2, 20, &rules, &[Node::root(&rules)]);
    }
    let baseline = Rules::baseline();
    for rules in [Rules::baseline(), Rules::improved(13)] {
        for (start, target) in [(38, 44), (50, 56)] {
            let sampler = Sampler::new(&baseline, start, None);
            let mut rng = Rng(91_133 ^ start as u64);
            let roots: Vec<_> = (0..64)
                .map(|_| sampler.draw(&mut rng, Some(&rules)))
                .filter(|draw| draw.culled.is_none())
                .map(|draw| draw.node)
                .collect();
            compare_recorded_engines(start, target, &rules, &roots);
        }
    }
}

/// Independent of the production modular inverses, slope helper, and screening
/// tables. Synthetic inputs satisfy m+1 fits, f<=80, and k<=127.
fn ordinary_division_pm(node: &Node) -> bool {
    let mut q = node.m + 1;
    let mut t = 0;
    while t < node.f && q % 3 == 0 {
        q /= 3;
        t += 1;
    }
    if t == 0 || t > node.k {
        return false;
    }
    let partner = (q << t) - 1;
    partner < node.n0 && POW3[(node.f - t) as usize] <= (1u128 << (node.k - t))
}

#[test]
fn ternary_predicates_match_ordinary_division_at_arithmetic_boundaries() {
    let screens = ternary::Screens::new(127);
    let mut checked = 0usize;
    let mut culled = 0usize;
    let mut fallback_cases = 0usize;
    let mut clipped_cases = 0usize;
    for valuation in (0..=40).chain([79, 80]) {
        let divisor = POW3[valuation];
        // Include ordinary magnitudes and values close to the u128 ceiling.
        // Remove factors of three from q so the intended valuation is exact.
        let mut top_q = u128::MAX / divisor;
        while top_q % 3 == 0 {
            top_q -= 1;
        }
        let mut values: Vec<_> = [1u128, 2, 7, top_q]
            .into_iter()
            .filter_map(|q| divisor.checked_mul(q).map(|x| x - 1))
            .collect();
        values.sort_unstable();
        values.dedup();
        for m in values {
            let m3 = (m % P16 as u128) as u32;
            assert_eq!(ternary::valuation(m3), (valuation as u32).min(16));
            for f in 0..=80u32 {
                let t = (valuation as u32).min(f);
                let q = (m + 1) / POW3[t as usize];
                let partner = (q << t) - 1;
                // ceil(log2(3^(f-t))) calculated solely with integer bits.
                let exponent = POW3[(f - t) as usize];
                let slope_boundary = t + (128 - (exponent - 1).leading_zeros());
                assert!(slope_boundary <= 127);
                let mut depths = vec![
                    0,
                    38,
                    62,
                    127,
                    slope_boundary.saturating_sub(1),
                    slope_boundary,
                    (slope_boundary + 1).min(127),
                ];
                depths.sort_unstable();
                depths.dedup();
                let mut starts = vec![partner.saturating_sub(1), partner, partner + 1];
                starts.sort_unstable();
                starts.dedup();
                for k in depths {
                    for &n0 in &starts {
                        let node = Node {
                            n0,
                            k,
                            f,
                            m,
                            m3,
                            oee: 0,
                            ne: 0,
                        };
                        let expected = ordinary_division_pm(&node);
                        for (name, actual) in [
                            ("reference", pm_inf_culls(n0, k, f, m)),
                            ("screen", screens.pm_screen(&node)),
                            ("fused", screens.pm_fused(&node)),
                            ("threshold", screens.pm_threshold(&node)),
                            ("lookup", screens.pm_lookup(&node)),
                            (
                                "known",
                                screens.pm_known_valuation(&node, (valuation as u32).min(16)),
                            ),
                        ] {
                            assert_eq!(actual, expected,
                                "{name} differs at valuation={valuation}, f={f}, k={k}, m={m}, n0={n0}");
                        }
                        checked += 1;
                        culled += expected as usize;
                        fallback_cases += (valuation >= 16) as usize;
                        clipped_cases += (f < valuation as u32) as usize;
                    }
                }
            }
        }
    }
    assert!((100_000..1_000_000).contains(&checked));
    assert!(culled > 0 && culled < checked);
    assert!(fallback_cases > 0 && clipped_cases > 0);
}
