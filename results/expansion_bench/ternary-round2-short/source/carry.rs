//! Carry the capped ternary valuation through the DFS without changing Node.
//!
//! Once f >= 16, either high-bit lift adds a multiple of 3^16. For an
//! odd child, m' + 1 = 3(x + 1)/2, so the capped valuation rises by one.
//! For an even child, m' + 1 = (x + 2)/2. If the parent's valuation is
//! positive, x + 2 is nonzero modulo 3 and the child's valuation is zero.
//! Other even children, and all children below f = 16, use the residue.
use crate::ternary::{valuation, Screens};
use cv::arith::slope_descends;
use cv::rules::{walkback_certificate, Cull, Rules};
use cv::tree::{Node, Sink, TreeStats};

#[inline]
fn child_valuation(parent: &Node, child: &Node, capped_t: u32) -> u32 {
    if parent.f < 16 {
        valuation(child.m3)
    } else if child.f > parent.f {
        (capped_t + 1).min(16)
    } else if capped_t > 0 {
        0
    } else {
        valuation(child.m3)
    }
}

#[inline]
fn extra_cull(node: &Node, capped_t: u32, rules: &Rules, screens: &Screens) -> Option<Cull> {
    if rules.pm_inf && screens.pm_known_valuation(node, capped_t) {
        return Some(Cull::PmInf);
    }
    if let Some(table) = &rules.mix {
        if node.k >= rules.mix_from && table.culls_r(node.n0, node.k, node.f, node.m, node.m3) {
            return Some(Cull::Mix);
        }
    }
    if let Some(walkback) = rules.walkback {
        if node.k == walkback.depth
            && walkback_certificate(node.n0, node.k, node.f, node.m, walkback.max_states).is_some()
        {
            return Some(Cull::Walkback);
        }
    }
    None
}

/// Original child construction, rule order, and statistics, with a capped
/// valuation carried as one extra recursion argument. No production changes.
pub fn dfs<S: Sink, const STATS: bool>(
    node: &Node,
    target: u32,
    rules: &Rules,
    screens: &Screens,
    sink: &mut S,
    stats: &mut TreeStats,
) {
    inner::<S, STATS>(
        node,
        valuation(node.m3),
        target,
        rules,
        screens,
        sink,
        stats,
    );
}

fn inner<S: Sink, const STATS: bool>(
    node: &Node,
    capped_t: u32,
    target: u32,
    rules: &Rules,
    screens: &Screens,
    sink: &mut S,
    stats: &mut TreeStats,
) {
    if node.k == target {
        sink.endpoint(node);
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
        // Delay valuation work until the two baseline checks have passed.
        let mut child_t = 0;
        let reason = if child.m < child.n0 || slope_descends(child.k, child.f) {
            Some(Cull::Descent)
        } else if rules.pm1 && child.ne == 2 && slope_descends(child.k - 1, child.f - 1) {
            Some(Cull::Pm1)
        } else {
            child_t = child_valuation(node, &child, capped_t);
            extra_cull(&child, child_t, rules, screens)
        };
        match reason {
            Some(why) => {
                if STATS {
                    stats.culls[child.k as usize][why as usize] += 1;
                }
            }
            None => {
                if STATS {
                    stats.survived[child.k as usize] += 1;
                }
                inner::<S, STATS>(&child, child_t, target, rules, screens, sink, stats);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::child_valuation;
    use crate::ternary::valuation;
    use cv::arith::POW3;
    use cv::rules::Rules;
    use cv::tree::{Node, P16};

    fn full_valuation(m: u128) -> u32 {
        let mut q = m + 1;
        let mut t = 0;
        while q % 3 == 0 && t < 16 {
            q /= 3;
            t += 1;
        }
        t
    }

    #[test]
    fn carried_child_valuations_match_full_values() {
        let rules = Rules::descent_only();
        let mut checked = 0;
        for f in [0, 1, 2, 14, 15, 16, 17, 26, 40] {
            for exponent in 0..=20 {
                for multiplier in [1, 2, 4, 7, 11, 16] {
                    let m = POW3[exponent] * multiplier - 1;
                    let parent = Node {
                        n0: 3,
                        k: 50,
                        f,
                        m,
                        m3: (m % P16 as u128) as u32,
                        oee: 0,
                        ne: 0,
                    };
                    let parent_t = full_valuation(m);
                    assert_eq!(valuation(parent.m3), parent_t);
                    for child in parent.children(&rules).into_iter().flatten() {
                        assert_eq!(
                            child_valuation(&parent, &child, parent_t),
                            full_valuation(child.m),
                            "valuation mismatch at f={f}, m={m}, child={child:?}",
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 2_268);
    }
}
