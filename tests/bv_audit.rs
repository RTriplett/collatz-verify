//! Exhaustive soundness audit of the step-2 bitvector at small N.
//!
//! For every depth-M endpoint and every completion `n = n0 + h 2^M` that the
//! bitvector excludes, search for an explicit witness inside the B-step
//! window by direct iteration of `n`:
//!   * descent:  `T^j(n) < n`,
//!   * pm (mod 3 merge): `y = T^j(n) = 2 (mod 3)` and `(2y - 1)/3 < n`,
//!   * OEE: `x = T^j(n)` odd, then >= 1 odd steps, then two even steps, and
//!     `(x - 1)/2 < n`.
//!   * pm_inf chain: `t = v3(y+1)`, `u = 2^t (y+1)/3^t - 1 < n`,
//!   * mixed word (from the tables): `p = (2^s y - b)/3^o` with `T^s(p) = y`
//!     checked by direct iteration, and `p < n`.
//! An exclusion without any witness is an unjustified cull.

use cv::arith::{t_iter, t_step, POW3};
use cv::lookahead::{BvMode, Lookahead, WindowRules};
use cv::rules::{MixTables, Rules};
use std::sync::Arc;
use cv::tree::{dfs, frontier, Node, Sink, TreeStats};

struct Audit<'a> {
    la: &'a Lookahead,
    mix: Option<&'a MixTables>,
    excluded: u64,
    unjustified: u64,
    by_ne: [u64; 4],
    example: Option<(u128, u32, u8)>,
}

fn witness(n: u128, m_start: u32, b: u32, mix: Option<&MixTables>) -> bool {
    // trajectory from T^M(n) through the window
    let mut ys = Vec::with_capacity(b as usize + 2);
    let mut y = n;
    for _ in 0..m_start {
        y = t_step(y);
    }
    ys.push(y);
    for _ in 0..b {
        y = t_step(y);
        ys.push(y);
    }
    for (j, &y) in ys.iter().enumerate() {
        if y < n {
            return true;
        }
        if y % 3 == 2 && (2 * y - 1) / 3 < n {
            return true;
        }
        // longest inverse-odd chain
        let (mut q, mut t) = (y + 1, 0u32);
        while q % 3 == 0 {
            q /= 3;
            t += 1;
        }
        if t > 0 && (q << t) - 1 < n {
            return true;
        }
        if let Some(tab) = mix {
            for jp in 1..=tab.j {
                if let Some((s, o, bb)) = tab.lookup(jp, (y % POW3[jp as usize]) as u64) {
                    let num = (y << s).checked_sub(bb);
                    if let Some(num) = num {
                        if num % POW3[o as usize] == 0 {
                            let p = num / POW3[o as usize];
                            if p >= 1 && p < n && t_iter(p, s).0 == y {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        if y & 1 == 1 && (y - 1) / 2 < n {
            // odd run of length >= 1 starting at y, then two evens, inside the window
            let mut i = j;
            while i < ys.len() && ys[i] & 1 == 1 {
                i += 1;
            }
            if i > j && i + 1 < ys.len() && ys[i] & 1 == 0 && ys[i + 1] & 1 == 0 {
                return true;
            }
        }
    }
    false
}

impl Sink for Audit<'_> {
    fn endpoint(&mut self, nd: &Node) {
        let la = self.la;
        let Some(arr) = la.array(nd.f, nd.ne) else { return };
        let start = la.start(arr, nd.m);
        for h in 0..(1u64 << la.a) {
            let pos = start + h;
            let word = la.bv[arr * la.words + (pos / 64) as usize];
            if word >> (pos % 64) & 1 == 1 {
                continue;
            }
            self.excluded += 1;
            let n = nd.n0 + ((h as u128) << la.m);
            if !witness(n, la.m, la.b, self.mix) {
                self.unjustified += 1;
                self.by_ne[nd.ne as usize] += 1;
                self.example.get_or_insert((n, nd.f, nd.ne));
            }
        }
    }
}

fn audit(n_bits: u32, a: u32, legacy: bool) -> u64 {
    audit_with(n_bits, a, if legacy { BvMode::Legacy } else { BvMode::Split }, &WindowRules::default())
}

fn audit_with(n_bits: u32, a: u32, mode: BvMode, wr: &WindowRules) -> u64 {
    let rules = Rules::baseline();
    let legacy = mode == BvMode::Legacy;
    {
        let m = n_bits - a;
        let la = Lookahead::build_ext(m, a, 24, 16, mode, wr);
        let mut st = TreeStats::new(m);
        let units = frontier(m.min(14), &rules, &mut st);
        let mut audit = Audit { la: &la, mix: wr.mix.as_deref(), excluded: 0, unjustified: 0, by_ne: [0; 4], example: None };
        for u in &units {
            dfs::<_, false>(u, m, &rules, &mut audit, &mut st);
        }
        eprintln!(
            "N={n_bits} A={a} legacy={legacy} window={}: excluded={} unjustified={} by_ne={:?} example={:?}",
            wr.label(),
            audit.excluded, audit.unjustified, audit.by_ne, audit.example
        );
        audit.unjustified
    }
}

#[test]
fn bitvector_exclusions_have_witnesses() {
    for (n_bits, a) in [(26u32, 6u32), (30, 10), (32, 8)] {
        assert_eq!(audit(n_bits, a, false), 0, "N={n_bits} A={a}");
    }
}

#[test]
fn reference_bitvector_has_unjustified_exclusions() {
    // Documents the defect in the single-family construction.
    assert!(audit(26, 6, true) > 0);
}

#[test]
fn improved_bitvector_exclusions_have_witnesses() {
    let wr = WindowRules { pm_inf: true, mix: Some(Arc::new(MixTables::build(12))) };
    for (n_bits, a) in [(26u32, 6u32), (30, 10)] {
        assert_eq!(audit_with(n_bits, a, BvMode::Split, &wr), 0, "N={n_bits} A={a}");
    }
}

#[test]
fn shared_bitvector_exclusions_have_witnesses() {
    let wr = WindowRules { pm_inf: true, mix: Some(Arc::new(MixTables::build(12))) };
    assert_eq!(audit_with(26, 6, BvMode::Shared, &WindowRules::default()), 0);
    assert_eq!(audit_with(30, 10, BvMode::Shared, &wr), 0);
}

/// Shared mode reduces to Split: for every window pattern its dip is the
/// smaller of the two split-family dips (it withholds exactly the credits
/// whose validity depends on the unknown trailing-even parity).  With the
/// split families audited above, Shared is sound for both parities.
#[test]
fn shared_dip_is_min_of_split_dips() {
    use cv::lookahead::{window_dip_ext, START_UNKNOWN};
    for wr in [WindowRules::default(), WindowRules { pm_inf: true, mix: Some(Arc::new(MixTables::build(13))) }] {
        for b in [12u32, 18, 22] {
            for x in 0..(1u64 << b) {
                let s = window_dip_ext(x, b, START_UNKNOWN, &wr);
                let m = window_dip_ext(x, b, 0, &wr).min(window_dip_ext(x, b, 1, &wr));
                assert_eq!(s, m, "B={b} x={x} window={}", wr.label());
            }
        }
    }
}
