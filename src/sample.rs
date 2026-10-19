//! Exactly uniform sampling of baseline endpoints at any depth.
//!
//! The backward DP gives, for every parity state, the number of surviving
//! depth-`M` descendants.  Descending from the root and choosing each child
//! with probability proportional to that count draws every depth-`M`
//! baseline survivor with probability exactly `1 / S_M` — at M = 62 as
//! easily as at M = 30.  Each draw is a real class `(n0, m, f)`, so the
//! value-dependent rules (pm_inf, mix) can be evaluated along its path and
//! its steps 2+3 can be run for real.  Means over draws times the exact
//! counts give unbiased totals at depths no enumeration can reach.

use crate::dp::{BackwardDp, ParityRules, PState};
use crate::rules::{Cull, Rules};
use crate::tree::Node;
use rayon::prelude::*;

/// SplitMix64.
#[derive(Clone)]
pub struct Rng(pub u64);

impl Rng {
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, n) (bias below 2^-64 for n < 2^64·something; n < 2^100 here).
    #[inline]
    pub fn below(&mut self, n: u128) -> u128 {
        let r = ((self.next_u64() as u128) << 64) | self.next_u64() as u128;
        r % n
    }
}

/// One uniformly drawn endpoint, with the first improved-rule cull on its path.
#[derive(Clone, Copy, Debug)]
pub struct Draw {
    pub node: Node,
    pub culled: Option<(u32, Cull)>,
}

pub struct Sampler {
    baseline: Rules,
    bw: BackwardDp,
    root: Node,
}

impl Sampler {
    /// Endpoints at depth `m`, optionally only those with `f` in `[lo, hi]`.
    pub fn new(baseline: &Rules, m: u32, f_filter: Option<(u32, u32)>) -> Self {
        assert!(baseline.is_parity_only(), "sample from a parity-state rule set");
        let root = Node::root(baseline);
        let bw = BackwardDp::run(ParityRules::of(baseline), root.k, m, f_filter);
        Sampler { baseline: baseline.clone(), bw, root }
    }

    /// Exact number of endpoints in the sampled population.
    pub fn population(&self) -> u128 {
        self.bw.ways(PState::of(&self.root))
    }

    /// Draw one endpoint; `extra` rules are evaluated at every depth.
    pub fn draw(&self, rng: &mut Rng, extra: Option<&Rules>) -> Draw {
        let mut nd = self.root;
        let mut culled = None;
        while nd.k < self.bw.target {
            let kids = nd.children(&self.baseline);
            let w: [u128; 2] = kids.map(|c| c.map_or(0, |c| self.bw.ways(PState::of(&c))));
            let pick = if rng.below(w[0] + w[1]) < w[0] { 0 } else { 1 };
            nd = kids[pick].unwrap();
            debug_assert!(nd.cull(&self.baseline).is_none());
            if culled.is_none() {
                if let Some(r) = extra {
                    if let Some(why) = r.extra_cull(nd.n0, nd.k, nd.f, nd.m, nd.m3) {
                        culled = Some((nd.k, why));
                    }
                }
            }
        }
        Draw { node: nd, culled }
    }

    /// `count` draws in parallel, deterministic in `seed`.
    pub fn draws(&self, count: usize, seed: u64, extra: Option<&Rules>) -> Vec<Draw> {
        const BLOCK: usize = 4096;
        (0..count.div_ceil(BLOCK))
            .into_par_iter()
            .flat_map_iter(|b| {
                let mut rng = Rng(seed ^ (b as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
                let n = BLOCK.min(count - b * BLOCK);
                (0..n).map(move |_| self.draw(&mut rng, extra)).collect::<Vec<_>>()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::census;

    #[test]
    fn draws_are_uniform_and_real() {
        let rules = Rules::baseline();
        let s = Sampler::new(&rules, 20, None);
        let draws = s.draws(20000, 7, None);
        for d in &draws {
            let (m, f) = crate::arith::t_iter(d.node.n0, 20);
            assert_eq!((m, f), (d.node.m, d.node.f));
            assert!(d.node.cull(&rules).is_none());
        }
        // f distribution matches the exact census within sampling error
        let c = census(&rules, 20, 12);
        let total: u64 = c.endpoints_by_f.iter().sum();
        assert_eq!(total as u128, s.population());
        for (f, &cnt) in c.endpoints_by_f.iter().enumerate() {
            let p = cnt as f64 / total as f64;
            let got = draws.iter().filter(|d| d.node.f as usize == f).count() as f64 / draws.len() as f64;
            let sd = (p * (1.0 - p) / draws.len() as f64).sqrt();
            assert!((got - p).abs() < 5.0 * sd + 1e-4, "f={f} p={p} got={got}");
        }
    }

    #[test]
    fn improved_survival_matches_census() {
        let base = Rules::baseline();
        let imp = Rules::improved(8);
        let k = 26;
        let exact = census(&imp, k, 14).stats.survived[k as usize] as f64
            / census(&base, k, 14).stats.survived[k as usize] as f64;
        let s = Sampler::new(&base, k, None);
        let draws = s.draws(200_000, 11, Some(&imp));
        let surv = draws.iter().filter(|d| d.culled.is_none()).count() as f64 / draws.len() as f64;
        let sd = (exact * (1.0 - exact) / draws.len() as f64).sqrt();
        assert!((surv - exact).abs() < 5.0 * sd, "exact={exact} sampled={surv}");
    }
}
