//! Exact experimental DFS variants. Production traversal is unchanged.
use cv::arith::{slope_descends, weight_le_pow2k, POW3};
use cv::rules::{pm_inf_culls, walkback_certificate, Rules};
use cv::tree::{next_even_run, Node, Sink, P16};

/// Necessary slope conditions for any mixed word with at most `mix_j` odds.
/// A word has s >= o, so its smallest possible slope is obtained at
/// s = o = min(f, mix_j). Failure of that optimistic bound rules out every
/// entry of the existing mixed-word table, independent of its residue.
pub struct Gates {
    possible: Vec<[bool; 81]>,
    min_pm_t: Vec<[u8; 81]>,
    mix_j: u32,
}

impl Gates {
    pub fn new(max_depth: u32, mix_j: u32) -> Self {
        assert!(mix_j <= 16);
        let possible = (0..=max_depth)
            .map(|k| {
                let mut row = [false; 81];
                for f in 0..=80u32 {
                    let j = f.min(mix_j);
                    row[f as usize] = j > 0 && weight_le_pow2k(j, f - j, k);
                }
                row
            })
            .collect();
        let min_pm_t = (0..=max_depth)
            .map(|k| {
                let mut row = [81; 81];
                for f in 0..=80u32 {
                    if let Some(t) = (0..=f).find(|&t| weight_le_pow2k(t, f - t, k)) {
                        row[f as usize] = t as u8;
                    }
                }
                row
            })
            .collect();
        Self {
            possible,
            min_pm_t,
            mix_j,
        }
    }

    #[inline(always)]
    pub fn mix_possible(&self, k: u32, f: u32) -> bool {
        self.possible[k as usize][f as usize]
    }

    /// m3 determines v3(m+1) exactly unless m3+1 is 3^16. In that
    /// exceptional case preserve the full-width reference calculation.
    #[inline(always)]
    fn pm_possible(&self, node: &Node) -> bool {
        if node.m3 == P16 as u32 - 1 {
            return true;
        }
        let mut q = node.m3 + 1;
        let mut t = 0;
        while q % 3 == 0 {
            q /= 3;
            t += 1;
        }
        let t = t.min(node.f);
        t != 0 && t >= self.min_pm_t[node.k as usize][node.f as usize] as u32
    }
}

/// The reference value-dependent rules, with a necessary-condition guard
/// before the table lookup. Also usable by the block-expansion experiment.
#[inline(always)]
pub fn extra_culls(node: &Node, rules: &Rules, gates: &Gates) -> bool {
    extra_inner::<false>(node, rules, gates)
}

/// Additionally screen maximal odd-path merges using the carried ternary
/// residue, retaining the reference value and slope tests on eligible nodes.
#[inline(always)]
pub fn extra_culls_ternary(node: &Node, rules: &Rules, gates: &Gates) -> bool {
    extra_inner::<true>(node, rules, gates)
}

#[inline(always)]
fn extra_inner<const TERNARY: bool>(node: &Node, rules: &Rules, gates: &Gates) -> bool {
    if rules.pm_inf
        && (!TERNARY || gates.pm_possible(node))
        && pm_inf_culls(node.n0, node.k, node.f, node.m)
    {
        return true;
    }
    if let Some(table) = &rules.mix {
        if node.k >= rules.mix_from
            && gates.mix_possible(node.k, node.f)
            && table.culls_r(node.n0, node.k, node.f, node.m, node.m3)
        {
            return true;
        }
    }
    if let Some(w) = rules.walkback {
        if node.k == w.depth
            && walkback_certificate(node.n0, node.k, node.f, node.m, w.max_states).is_some()
        {
            return true;
        }
    }
    false
}

#[inline(always)]
fn culls(node: &Node, rules: &Rules, gates: &Gates) -> bool {
    if node.m < node.n0 || slope_descends(node.k, node.f) {
        return true;
    }
    if rules.pm1 && node.ne == 2 && slope_descends(node.k - 1, node.f - 1) {
        return true;
    }
    extra_culls(node, rules, gates)
}

/// Retains reference children construction and rule order; only guards the
/// mixed-word lookup using the precomputed necessary slope condition.
pub fn dfs_gated<S: Sink>(node: &Node, target: u32, rules: &Rules, gates: &Gates, sink: &mut S) {
    debug_assert_eq!(gates.mix_j, rules.mix_j());
    debug_assert!(target < gates.possible.len() as u32);
    gated_inner(node, target, rules, gates, sink);
}

fn gated_inner<S: Sink>(node: &Node, target: u32, rules: &Rules, gates: &Gates, sink: &mut S) {
    if node.k == target {
        sink.endpoint(node);
        return;
    }
    for child in node.children(rules).into_iter().flatten() {
        if !culls(&child, rules, gates) {
            gated_inner(&child, target, rules, gates, sink);
        }
    }
}

#[inline(always)]
fn half_mod_p16(mut x: u32) -> u32 {
    if x >= 2 * P16 as u32 {
        x -= 2 * P16 as u32;
    }
    if x >= P16 as u32 {
        x -= P16 as u32;
    }
    (x + (x & 1) * P16 as u32) >> 1
}

/// Checks baseline state rules before constructing wide child values.
/// Bit order and every surviving Node field match Node::children exactly.
#[inline(always)]
fn early_child<const TERNARY: bool>(
    node: &Node,
    bit: u32,
    rules: &Rules,
    gates: &Gates,
) -> Option<Node> {
    let odd = ((node.m as u32) ^ bit) & 1 != 0;
    let k = node.k + 1;
    let f = node.f + odd as u32;
    if (!odd && node.oee >= 2) || slope_descends(k, f) {
        return None;
    }
    let ne = if odd { 0 } else { next_even_run(node.ne) };
    if rules.pm1 && ne == 2 && slope_descends(k - 1, f - 1) {
        return None;
    }

    let n0 = node.n0 | ((bit as u128) << node.k);
    let x = if bit == 0 {
        node.m
    } else {
        node.m + POW3[node.f as usize]
    };
    let m = if odd { x + (x >> 1) + 1 } else { x >> 1 };
    if m < n0 {
        return None;
    }

    // For the sampled K=38 roots f>=16, the high-bit lift does not alter
    // m modulo 3^16. Retain the general case for audits near the root.
    let mut x3 = node.m3;
    if bit != 0 && node.f < 16 {
        x3 += POW3[node.f as usize] as u32;
        if x3 >= P16 as u32 {
            x3 -= P16 as u32;
        }
    }
    let m3 = half_mod_p16(if odd { 3 * x3 + 1 } else { x3 });
    let oee = if odd {
        let arm = (rules.oee && slope_descends(k, node.f)) as u8;
        (node.oee | arm) & !2
    } else {
        node.oee << 1
    };
    let child = Node {
        n0,
        k,
        f,
        m,
        m3,
        oee,
        ne,
    };
    if extra_inner::<TERNARY>(&child, rules, gates) {
        None
    } else {
        Some(child)
    }
}

/// Adds early baseline filters to the lookup-gated DFS.
pub fn dfs_early<S: Sink>(node: &Node, target: u32, rules: &Rules, gates: &Gates, sink: &mut S) {
    debug_assert_eq!(gates.mix_j, rules.mix_j());
    debug_assert!(target < gates.possible.len() as u32);
    early_inner::<S, false>(node, target, rules, gates, sink);
}

/// Early child filtering plus carried-ternary screening of maximal merges.
pub fn dfs_ternary<S: Sink>(node: &Node, target: u32, rules: &Rules, gates: &Gates, sink: &mut S) {
    debug_assert_eq!(gates.mix_j, rules.mix_j());
    debug_assert!(target < gates.possible.len() as u32);
    early_inner::<S, true>(node, target, rules, gates, sink);
}

fn early_inner<S: Sink, const TERNARY: bool>(
    node: &Node,
    target: u32,
    rules: &Rules,
    gates: &Gates,
    sink: &mut S,
) {
    if node.k == target {
        sink.endpoint(node);
        return;
    }
    for bit in 0..=1 {
        if let Some(child) = early_child::<TERNARY>(node, bit, rules, gates) {
            early_inner::<S, TERNARY>(&child, target, rules, gates, sink);
        }
    }
}
