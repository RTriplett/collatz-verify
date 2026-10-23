//! Experiments that use the carried residue to reduce maximal-path-merge work.
//! These preserve every original rule and differ only in pm-infinity evaluation.
use cv::arith::{slope_descends, weight_le_pow2k, POW3};
use cv::rules::{pm_inf_culls, walkback_certificate, Cull, Rules};
use cv::tree::{Node, Sink, TreeStats, P16};

const INV3: u128 = 0xAAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAB;
const INV3_U32: u32 = 0xAAAA_AAAB;
const LOOKUP_MODULUS: u32 = 6_561; // 3^8, one byte per residue.

/// The exact valuation when below 16, otherwise the carried-residue limit.
#[inline]
pub fn valuation(m3: u32) -> u32 {
    let mut q = m3 + 1;
    let mut t = 0;
    while q % 3 == 0 {
        q /= 3;
        t += 1;
    }
    t
}

pub struct Screens {
    min_t: Vec<[u8; 81]>,
    inverse: [u128; 16],
    divisor_inverse: [u32; 17],
    divisor_limit: [u32; 17],
    valuation: Box<[u8; LOOKUP_MODULUS as usize]>,
}

impl Screens {
    pub fn new(max_depth: u32) -> Self {
        let min_t = (0..=max_depth)
            .map(|k| {
                let mut row = [81; 81];
                for f in 1..=80u32 {
                    if let Some(t) = (1..=f).find(|&t| weight_le_pow2k(t, f - t, k)) {
                        row[f as usize] = t as u8;
                    }
                }
                row
            })
            .collect();
        let mut inverse = [1u128; 16];
        for t in 1..16 {
            inverse[t] = inverse[t - 1].wrapping_mul(INV3);
        }
        let mut divisor_inverse = [1u32; 17];
        let mut divisor_limit = [u32::MAX; 17];
        for t in 1..=16 {
            divisor_inverse[t] = divisor_inverse[t - 1].wrapping_mul(INV3_U32);
            divisor_limit[t] = u32::MAX / POW3[t] as u32;
        }
        let mut valuation = Box::new([0u8; LOOKUP_MODULUS as usize]);
        valuation[0] = 8;
        for r in 1..LOOKUP_MODULUS {
            let mut q = r;
            while q % 3 == 0 {
                valuation[r as usize] += 1;
                q /= 3;
            }
        }
        Self {
            min_t,
            inverse,
            divisor_inverse,
            divisor_limit,
            valuation,
        }
    }

    pub fn bytes(&self) -> usize {
        self.min_t.len() * std::mem::size_of::<[u8; 81]>()
            + std::mem::size_of_val(&self.inverse)
            + std::mem::size_of_val(&self.divisor_inverse)
            + std::mem::size_of_val(&self.divisor_limit)
            + std::mem::size_of_val(self.valuation.as_ref())
    }

    /// Smallest positive odd-path length whose slope can pass. The slope
    /// decreases strictly with t, so testing t >= this value is exact.
    #[inline(always)]
    pub fn minimum_t(&self, k: u32, f: u32) -> u32 {
        self.min_t[k as usize][f as usize] as u32
    }

    /// If 3^16 does not divide m+1, its valuation is exactly that of m3+1.
    /// Once t is known, division by 3^t is one multiplication modulo 2^128.
    /// Since 3^t divides m+1, the modular result is the ordinary quotient.
    #[inline]
    pub fn pm_fused(&self, node: &Node) -> bool {
        if node.m3 == P16 as u32 - 1 {
            return pm_inf_culls(node.n0, node.k, node.f, node.m);
        }
        self.finish(node, valuation(node.m3).min(node.f))
    }

    /// For traversal variants that maintain the capped valuation alongside
    /// the node. A cap of 16 needs the original full-width calculation.
    #[inline]
    pub fn pm_known_valuation(&self, node: &Node, capped_t: u32) -> bool {
        if capped_t == 16 {
            return pm_inf_culls(node.n0, node.k, node.f, node.m);
        }
        self.finish(node, capped_t.min(node.f))
    }

    /// Original two-stage screening, retained as a matched traversal control.
    #[inline]
    pub fn pm_screen(&self, node: &Node) -> bool {
        if node.m3 != P16 as u32 - 1 {
            let mut q = node.m3 + 1;
            let mut t = 0;
            while q % 3 == 0 {
                q /= 3;
                t += 1;
            }
            if t.min(node.f) < self.minimum_t(node.k, node.f) {
                return false;
            }
        }
        pm_inf_culls(node.n0, node.k, node.f, node.m)
    }

    /// Reject impossible slopes by testing only the minimum needed ternary
    /// divisibility. For an odd divisor d, d|x iff x*inverse(d) modulo 2^32
    /// is at most floor((2^32-1)/d); no variable division is needed here.
    #[inline]
    pub fn pm_threshold(&self, node: &Node) -> bool {
        let needed = self.minimum_t(node.k, node.f);
        if needed > node.f {
            return false;
        }
        if needed <= 16 {
            let i = needed as usize;
            if (node.m3 + 1).wrapping_mul(self.divisor_inverse[i]) > self.divisor_limit[i] {
                return false;
            }
        } else if node.m3 != P16 as u32 - 1 {
            return false;
        }
        pm_inf_culls(node.n0, node.k, node.f, node.m)
    }

    /// Same fused arithmetic, with the common valuation range answered by
    /// a 6.4 KiB table. Multiples of 3^8 continue the small-integer loop.
    #[inline]
    pub fn pm_lookup(&self, node: &Node) -> bool {
        if node.m3 == P16 as u32 - 1 {
            return pm_inf_culls(node.n0, node.k, node.f, node.m);
        }
        let q = node.m3 + 1;
        let r = q % LOOKUP_MODULUS;
        let mut t = self.valuation[r as usize] as u32;
        if r == 0 {
            let mut tail = q / LOOKUP_MODULUS;
            while tail % 3 == 0 {
                tail /= 3;
                t += 1;
            }
        }
        self.finish(node, t.min(node.f))
    }

    #[inline(always)]
    fn finish(&self, node: &Node, t: u32) -> bool {
        if t < self.minimum_t(node.k, node.f) {
            return false;
        }
        // Here 1 <= t < 16. q*2^t-1 < n0 iff q*2^t <= n0, hence this
        // comparison avoids a wide left shift and subtraction. m+1 must fit,
        // as in the reference rule.
        let q = (node.m + 1).wrapping_mul(self.inverse[t as usize]);
        q <= (node.n0 >> t)
    }
}

#[inline]
fn extra_cull<const MODE: u8>(node: &Node, rules: &Rules, screens: &Screens) -> Option<Cull> {
    if rules.pm_inf
        && match MODE {
            0 => pm_inf_culls(node.n0, node.k, node.f, node.m),
            1 => screens.pm_fused(node),
            2 => screens.pm_threshold(node),
            3 => screens.pm_lookup(node),
            4 => screens.pm_screen(node),
            _ => unreachable!(),
        }
    {
        return Some(Cull::PmInf);
    }
    if let Some(table) = &rules.mix {
        if node.k >= rules.mix_from && table.culls_r(node.n0, node.k, node.f, node.m, node.m3) {
            return Some(Cull::Mix);
        }
    }
    if let Some(w) = rules.walkback {
        if node.k == w.depth
            && walkback_certificate(node.n0, node.k, node.f, node.m, w.max_states).is_some()
        {
            return Some(Cull::Walkback);
        }
    }
    None
}

#[inline(always)]
fn cull<const MODE: u8>(node: &Node, rules: &Rules, screens: &Screens) -> Option<Cull> {
    if node.m < node.n0 || slope_descends(node.k, node.f) {
        return Some(Cull::Descent);
    }
    if rules.pm1 && node.ne == 2 && slope_descends(node.k - 1, node.f - 1) {
        return Some(Cull::Pm1);
    }
    extra_cull::<MODE>(node, rules, screens)
}

/// Source-matched reference traversal. MODE 0 uses the original pm-infinity
/// rule, 1 fuses quotient evaluation, 2 screens minimum divisibility,
/// 3 uses a valuation lookup, and 4 is the earlier two-stage screen.
/// STATS exactly mirrors production generated/survived/reason counters.
pub fn dfs<S: Sink, const MODE: u8, const STATS: bool>(
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
        match cull::<MODE>(&child, rules, screens) {
            Some(why) => {
                if STATS {
                    stats.culls[child.k as usize][why as usize] += 1;
                }
            }
            None => {
                if STATS {
                    stats.survived[child.k as usize] += 1;
                }
                dfs::<S, MODE, STATS>(&child, target, rules, screens, sink, stats);
            }
        }
    }
}
