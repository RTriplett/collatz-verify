//! Reuse a proven impossible pm-infinity slope along odd-step edges.
//!
//! Let t=min(v3(m+1),f). An odd child has t'=t+1, including a high-bit
//! lift m -> m+b*3^f: if t<f its valuation is unchanged by the lift,
//! and otherwise clipping at f makes any cancellation immaterial. Thus
//! 2^t'*3^(f+1-t') and 2^(k+1) both double, preserving slope eligibility.
use super::ternary::Screens;
use cv::arith::{slope_descends, weight_le_pow2k, POW3};
use cv::rules::{pm_inf_culls, walkback_certificate, Cull, Rules};
use cv::tree::{Node, Sink, TreeStats, P16};

const INV3_U32: u32 = 0xAAAA_AAAB;

pub struct Checks {
    min_t: Vec<[u8; 81]>,
    divisor_inverse: [u32; 17],
    divisor_limit: [u32; 17],
}

impl Checks {
    pub fn new(max_depth: u32) -> Self {
        let min_t = (0..=max_depth)
            .map(|k| {
                let mut row = [81; 81];
                for f in 0..=80u32 {
                    // Include zero: a parent with t=0 can have a valid slope
                    // even though pm-infinity itself requires positive t.
                    if let Some(t) = (0..=f).find(|&t| weight_le_pow2k(t, f - t, k)) {
                        row[f as usize] = t as u8;
                    }
                }
                row
            })
            .collect();
        let mut divisor_inverse = [1u32; 17];
        let mut divisor_limit = [u32::MAX; 17];
        for t in 1..=16 {
            divisor_inverse[t] = divisor_inverse[t - 1].wrapping_mul(INV3_U32);
            divisor_limit[t] = u32::MAX / POW3[t] as u32;
        }
        Self {
            min_t,
            divisor_inverse,
            divisor_limit,
        }
    }

    pub fn bytes(&self) -> usize {
        self.min_t.len() * std::mem::size_of::<[u8; 81]>()
            + std::mem::size_of_val(&self.divisor_inverse)
            + std::mem::size_of_val(&self.divisor_limit)
    }

    /// True proves the maximal clipped odd-path slope exceeds 2^k. False
    /// may also mean its valuation is unknown above the carried 3^16 limit.
    /// This predicate deliberately does not test the partner's value.
    #[inline(always)]
    pub fn slope_impossible(&self, node: &Node) -> bool {
        let needed = self.min_t[node.k as usize][node.f as usize] as u32;
        if needed == 0 {
            return false;
        }
        if needed > node.f {
            return true;
        }
        if needed <= 16 {
            let i = needed as usize;
            (node.m3 + 1).wrapping_mul(self.divisor_inverse[i]) > self.divisor_limit[i]
        } else {
            node.m3 != P16 as u32 - 1
        }
    }
}

#[inline]
fn extra_cull(
    rules: &Rules,
    n0: u128,
    k: u32,
    f: u32,
    m: u128,
    m3: u32,
    impossible: bool,
) -> Option<Cull> {
    // Match the production scalar interface so child addresses do not
    // escape the DFS into this potentially outlined helper.
    if rules.pm_inf && !impossible && pm_inf_culls(n0, k, f, m) {
        return Some(Cull::PmInf);
    }
    if let Some(table) = &rules.mix {
        if k >= rules.mix_from && table.culls_r(n0, k, f, m, m3) {
            return Some(Cull::Mix);
        }
    }
    if let Some(w) = rules.walkback {
        if k == w.depth && walkback_certificate(n0, k, f, m, w.max_states).is_some() {
            return Some(Cull::Walkback);
        }
    }
    None
}

/// The shared Screens argument keeps the experimental call interface uniform;
/// this variant needs its own zero-inclusive slope thresholds instead.
pub fn dfs<S: Sink, const STATS: bool>(
    node: &Node,
    target: u32,
    rules: &Rules,
    _screens: &Screens,
    checks: &Checks,
    sink: &mut S,
    stats: &mut TreeStats,
) {
    debug_assert!(target < checks.min_t.len() as u32);
    let impossible = rules.pm_inf && checks.slope_impossible(node);
    inner::<S, STATS>(node, target, rules, checks, impossible, sink, stats);
}

fn inner<S: Sink, const STATS: bool>(
    node: &Node,
    target: u32,
    rules: &Rules,
    checks: &Checks,
    impossible: bool,
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
        // Defer the extra slope check until the ordinary cheap culls pass.
        let mut child_impossible = false;
        let why = if child.m < child.n0 || slope_descends(child.k, child.f) {
            Some(Cull::Descent)
        } else if rules.pm1 && child.ne == 2 && slope_descends(child.k - 1, child.f - 1) {
            Some(Cull::Pm1)
        } else {
            child_impossible = rules.pm_inf
                && ((child.f == node.f + 1 && impossible) || checks.slope_impossible(&child));
            extra_cull(
                rules,
                child.n0,
                child.k,
                child.f,
                child.m,
                child.m3,
                child_impossible,
            )
        };
        match why {
            Some(why) => {
                if STATS {
                    stats.culls[child.k as usize][why as usize] += 1;
                }
            }
            None => {
                if STATS {
                    stats.survived[child.k as usize] += 1;
                }
                inner::<S, STATS>(&child, target, rules, checks, child_impossible, sink, stats);
            }
        }
    }
}

/// An odd child without a high-bit lift has the same maximal-path partner
/// and the same slope eligibility as its parent. If the parent's t was
/// zero, its new positive-t partner is the parent's iterate m >= n0.
/// Thus neither case can newly cull an already surviving parent.
///
/// The starting node must have passed descent and pm-infinity, as all stored
/// frontier roots and recursively reached survivors do. Unlike the other
/// experiments, this variant relies on that root precondition.
pub fn dfs_bit0<S: Sink, const STATS: bool>(
    node: &Node,
    target: u32,
    rules: &Rules,
    screens: &Screens,
    sink: &mut S,
    stats: &mut TreeStats,
) {
    debug_assert!(node.m >= node.n0);
    debug_assert!(
        !rules.pm_inf || !pm_inf_culls(node.n0, node.k, node.f, node.m),
        "bit-zero reuse requires a surviving root"
    );
    bit0_inner::<S, STATS>(node, target, rules, screens, sink, stats);
}

#[inline]
fn bit0_extra_cull(
    rules: &Rules,
    n0: u128,
    k: u32,
    f: u32,
    m: u128,
    m3: u32,
    screens: &Screens,
    skip_pm: bool,
) -> Option<Cull> {
    let node = Node {
        n0,
        k,
        f,
        m,
        m3,
        oee: 0,
        ne: 0,
    };
    if rules.pm_inf && !skip_pm && screens.pm_threshold(&node) {
        return Some(Cull::PmInf);
    }
    if let Some(table) = &rules.mix {
        if k >= rules.mix_from && table.culls_r(n0, k, f, m, m3) {
            return Some(Cull::Mix);
        }
    }
    if let Some(w) = rules.walkback {
        if k == w.depth && walkback_certificate(n0, k, f, m, w.max_states).is_some() {
            return Some(Cull::Walkback);
        }
    }
    None
}

fn bit0_inner<S: Sink, const STATS: bool>(
    node: &Node,
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
        let why = if child.m < child.n0 || slope_descends(child.k, child.f) {
            Some(Cull::Descent)
        } else if rules.pm1 && child.ne == 2 && slope_descends(child.k - 1, child.f - 1) {
            Some(Cull::Pm1)
        } else {
            bit0_extra_cull(
                rules,
                child.n0,
                child.k,
                child.f,
                child.m,
                child.m3,
                screens,
                child.f == node.f + 1 && child.n0 == node.n0,
            )
        };
        match why {
            Some(why) => {
                if STATS {
                    stats.culls[child.k as usize][why as usize] += 1;
                }
            }
            None => {
                if STATS {
                    stats.survived[child.k as usize] += 1;
                }
                bit0_inner::<S, STATS>(&child, target, rules, screens, sink, stats);
            }
        }
    }
}
