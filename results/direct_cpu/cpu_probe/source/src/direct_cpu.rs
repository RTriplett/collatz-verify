//! Experimental CPU ownership of complete endpoint families before GPU dispatch.
//!
//! A source family is expanded with reusable baseline suffixes to the CPU's
//! independently chosen M. The CPU then performs mod-9/OEE screening, optional
//! resident B-bitvector screening, and the ordinary scalar verification tail.
//! No GPU transfer or GPU-produced survivor mask is required.

use crate::arith::{slope_descends, POW3};
use crate::dp::ForwardDp;
use crate::lookahead::{BvMode, Lookahead, WindowRules};
use crate::rules::Rules;
use crate::tail::{step_until_below, TailContinuation};
use crate::tree::{next_even_run, Node, P16};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DirectWork {
    pub source_endpoints: u64,
    pub endpoints: u64,
    pub cases: u64,
    pub iters: u64,
    pub hash: u32,
    pub flagged: Vec<u128>,
}

impl DirectWork {
    pub fn merge(&mut self, other: &Self) {
        self.source_endpoints += other.source_endpoints;
        self.endpoints += other.endpoints;
        self.cases += other.cases;
        self.iters += other.iters;
        self.hash ^= other.hash;
        self.flagged.extend_from_slice(&other.flagged);
    }
}

#[derive(Clone, Copy)]
struct Word {
    residue: u32,
    q_coeff: u32,
    iterate: u32,
    pow3: u32,
    f: u32,
    oee: u8,
    ne: u8,
}

struct Program {
    inverse: u64,
    words: Vec<Word>,
}

#[derive(Clone, Copy)]
struct State {
    k: u32,
    f: u32,
    oee: u8,
    ne: u8,
}

fn index(f: u32, oee: u8, ne: u8) -> usize {
    (f as usize * 3 + oee as usize) * 4 + ne as usize
}

/// Immutable setup shared by CPU worker threads. Array/table construction is
/// intentionally separate from per-source processing and should be timed as setup.
pub struct Plan {
    n: u32,
    source_m: u32,
    a: u32,
    b: Option<u32>,
    cutoff_bits: u32,
    lookahead: Lookahead,
    programs: Vec<Option<Program>>,
}

impl Plan {
    pub fn new(
        n: u32,
        source_m: u32,
        cpu_a: u32,
        b: Option<u32>,
        num_bv: usize,
        window: &WindowRules,
    ) -> Self {
        assert!((2..=72).contains(&n));
        assert!(cpu_a <= 10 && cpu_a < n);
        let target = n - cpu_a;
        assert!(source_m >= 2 && source_m <= target && target - source_m <= 12);
        if let Some(b) = b {
            assert!((cpu_a.max(6)..=24).contains(&b), "CPU resident B must be in [max(A,6),24]");
            assert!(num_bv > 0);
        }
        // The baseline rational predicate and exact slope sign agree in this
        // supported range. Thus m<n0 cannot remove additional suffixes after
        // the state-only compiler has retained a strictly increasing slope.
        for k in source_m..=target {
            for f in 0..=k {
                assert_eq!(slope_descends(k, f), POW3[f as usize] <= 1u128 << k);
            }
        }
        let rules = Rules::baseline();
        let reachable = ForwardDp::from_root(&rules, source_m);
        let width = target - source_m;
        let mask = (1u64 << width) - 1;
        let mut programs: Vec<Option<Program>> =
            (0..(source_m as usize + 1) * 12).map(|_| None).collect();
        for (state, _) in reachable.layer(source_m).states() {
            let inverse = 0xaaaa_aaaa_aaaa_aaabu64.wrapping_pow(state.f);
            let mut words = Vec::new();
            compile(
                State { k: source_m, f: state.f, oee: state.oee, ne: state.ne },
                source_m, target, 0, 0, 1, inverse, mask, &mut words,
            );
            words.sort_by_key(|word| word.f);
            programs[index(state.f, state.oee, state.ne)] = Some(Program { inverse, words });
        }
        // Existing Lookahead stores at least one 64-bit completion word. For
        // A<6 its M-correct A6 masks/arrays remain valid; iteration truncates
        // the word to exactly 2^A completions below.
        let stored_a = cpu_a.max(6);
        let lookahead = match b {
            Some(b) => Lookahead::build_ext(target, stored_a, b, num_bv, BvMode::Shared, window),
            None => Lookahead::meta(target, stored_a, stored_a, num_bv, BvMode::Shared, window),
        };
        Self { n, source_m, a: cpu_a, b, cutoff_bits: 120, lookahead, programs }
    }

    pub fn with_cutoff_bits(mut self, bits: u32) -> Self {
        assert!((20..=123).contains(&bits));
        self.cutoff_bits = bits;
        self
    }

    pub fn n(&self) -> u32 { self.n }
    pub fn source_m(&self) -> u32 { self.source_m }
    pub fn m(&self) -> u32 { self.n - self.a }
    pub fn a(&self) -> u32 { self.a }
    pub fn b(&self) -> Option<u32> { self.b }
    pub fn cutoff_bits(&self) -> u32 { self.cutoff_bits }
    pub fn lookahead(&self) -> &Lookahead { &self.lookahead }

    pub fn table_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.programs.capacity() * std::mem::size_of::<Option<Program>>()
            + self.programs.iter().flatten()
                .map(|program| program.words.capacity() * std::mem::size_of::<Word>())
                .sum::<usize>()
    }

    pub fn lookahead_bytes(&self) -> usize {
        self.lookahead.bv.len() * 8 + self.lookahead.mod9.len() * 8
    }

    /// Enumerate complete, exact descendant nodes. No narrow u64 endpoint
    /// conversion is used, including when the CPU chooses M>64.
    #[inline]
    pub fn for_each_endpoint(&self, source: &Node, mut visit: impl FnMut(&Node)) {
        assert_eq!(source.k, self.source_m);
        assert!(source.f <= source.k && source.oee < 3 && source.ne < 4);
        if self.m() == source.k {
            visit(source);
            return;
        }
        let program = self.programs[index(source.f, source.oee, source.ne)]
            .as_ref().expect("source parity state is not reachable under the baseline rules");
        let width = self.m() - source.k;
        let mask = (1u64 << width) - 1;
        let delta = (source.m as u64).wrapping_mul(program.inverse) & mask;
        let source_power = POW3[source.f as usize];
        for word in &program.words {
            let q = (word.q_coeff as u64).wrapping_sub(delta) & mask;
            let x = source.m + source_power * q as u128;
            debug_assert_eq!(x & mask as u128, word.residue as u128);
            let m = ((x - word.residue as u128) >> width) * word.pow3 as u128
                + word.iterate as u128;
            visit(&Node {
                n0: source.n0 | ((q as u128) << source.k),
                k: self.m(), f: word.f, m, m3: (m % P16 as u128) as u32,
                oee: word.oee, ne: word.ne,
            });
        }
    }

    pub fn process(&self, source: &Node) -> DirectWork {
        let mut result = DirectWork::default();
        self.process_into(source, &mut result);
        result
    }

    /// Fused expansion, screening, and scalar tails. Flagged starts are returned
    /// unchanged; the caller must perform the ordinary wide recheck.
    pub fn process_into(&self, source: &Node, work: &mut DirectWork) {
        work.source_endpoints += 1;
        self.for_each_endpoint(source, |node| {
            self.candidates(node, work, |base, number, work| {
                let (hash, iters, flagged) = step_until_below(base, number, 1u128 << self.cutoff_bits);
                work.hash ^= hash;
                work.iters += iters;
                if flagged { work.flagged.push(base); }
            });
        });
    }

    /// Diagnostic/replay API: collect exactly the screened states before their
    /// tails. Initial arithmetic-bound flags remain in `work.flagged` instead.
    pub fn screen_source(&self, source: &Node, states: &mut Vec<TailContinuation>) -> DirectWork {
        let mut work = DirectWork { source_endpoints: 1, ..DirectWork::default() };
        self.for_each_endpoint(source, |node| {
            self.candidates(node, &mut work, |base, number, _| {
                states.push(TailContinuation::from_values(base, number));
            });
        });
        work
    }

    #[inline]
    fn candidates(&self, node: &Node, work: &mut DirectWork,
                  mut visit: impl FnMut(u128, u128, &mut DirectWork)) {
        work.endpoints += 1;
        let la = &self.lookahead;
        let arr = if self.b.is_some() { la.array(node.f, node.ne) } else { None };
        let start = arr.map_or(0, |i| la.start(i, node.m));
        let r9 = (node.n0 % 9) as usize;
        let power = POW3[node.f as usize];
        let words = ((1usize << self.a) + 63) / 64;
        let truncate = if self.a < 6 { (1u64 << (1u32 << self.a)) - 1 } else { u64::MAX };
        for word in 0..words {
            let mut bits = la.completion_word(arr, start, r9, word) & truncate;
            while bits != 0 {
                let h = (word * 64 + bits.trailing_zeros() as usize) as u128;
                bits &= bits - 1;
                let base = node.n0 + (h << self.m());
                let Some(x) = power.checked_mul(h).and_then(|x| x.checked_add(node.m))
                    .filter(|&x| x < 1u128 << 123) else {
                    work.cases += 1;
                    work.flagged.push(base);
                    continue;
                };
                if node.oee == 2 && x & 1 == 0 { continue; }
                if node.oee == 1 && x & 3 == 0 { continue; }
                work.cases += 1;
                visit(base, x + 1, work);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compile(state: State, source_m: u32, target: u32, residue: u64, iterate: u64,
           power: u64, inverse: u64, mask: u64, words: &mut Vec<Word>) {
    if state.k == target {
        words.push(Word {
            residue: residue as u32,
            q_coeff: (residue.wrapping_mul(inverse) & mask) as u32,
            iterate: iterate as u32, pow3: power as u32, f: state.f,
            oee: state.oee, ne: state.ne,
        });
        return;
    }
    let arm = u8::from(slope_descends(state.k + 1, state.f));
    for odd in [false, true] {
        if !odd && state.oee >= 2 { continue; }
        let next = State {
            k: state.k + 1,
            f: state.f + u32::from(odd),
            oee: if odd { (state.oee | arm) & !2 } else { state.oee << 1 },
            ne: if odd { 0 } else { next_even_run(state.ne) },
        };
        if slope_descends(next.k, next.f)
            || (next.ne == 2 && slope_descends(next.k - 1, next.f - 1)) { continue; }
        let bit = (iterate & 1) ^ u64::from(odd);
        let x = iterate + power * bit;
        let next_iterate = if odd { x + (x >> 1) + 1 } else { x >> 1 };
        compile(next, source_m, target, residue | (bit << (state.k - source_m)),
                next_iterate, if odd { power * 3 } else { power }, inverse, mask, words);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::{Rng, Sampler};
    use crate::tail::recheck_wide;
    use crate::tree::{dfs, Sink, TreeStats};

    #[derive(Default)]
    struct Nodes(Vec<Node>);
    impl Sink for Nodes {
        fn endpoint(&mut self, node: &Node) { self.0.push(*node); }
    }

    fn sources(m: u32, count: usize) -> Vec<Node> {
        let sampler = Sampler::new(&Rules::baseline(), m, None);
        let mut rng = Rng(0xe26f_889d_1482_7511);
        (0..count).map(|_| sampler.draw(&mut rng, None).node).collect()
    }

    fn exact_suffix(source: &Node, target: u32, q: u128) -> Option<Node> {
        let n0 = source.n0 | (q << source.k);
        let mut m = source.m + POW3[source.f as usize] * q;
        let (mut f, mut oee, mut ne) = (source.f, source.oee, source.ne);
        for k in source.k + 1..=target {
            if m % 2 == 0 {
                if oee == 2 { return None; }
                m /= 2;
                oee *= 2;
                ne = if ne < 2 { ne + 1 } else { 5 - ne };
            } else {
                let arm = u8::from(POW3[f as usize] <= 1u128 << k);
                m = (3 * m + 1) / 2;
                f += 1;
                oee = (oee | arm) & 1;
                ne = 0;
            }
            if POW3[f as usize] <= 1u128 << k
                || (ne == 2 && POW3[(f - 1) as usize] <= 1u128 << (k - 1)) {
                return None;
            }
        }
        Some(Node { n0, k: target, f, m, m3: (m % P16 as u128) as u32, oee, ne })
    }

    fn reference(plan: &Plan, source: &Node) -> DirectWork {
        let mut out = DirectWork { source_endpoints: 1, ..DirectWork::default() };
        for q in 0..1u128 << (plan.m() - source.k) {
            let Some(node) = exact_suffix(source, plan.m(), q) else { continue };
            out.endpoints += 1;
            for h in 0..1u128 << plan.a() {
                let base = node.n0 + (h << plan.m());
                if matches!(base % 9, 2 | 4 | 5 | 8) { continue; }
                let x = node.m + POW3[node.f as usize] * h;
                if plan.b().is_some() && !plan.lookahead().keeps(node.f, node.ne, (x + 1) as u64) {
                    continue;
                }
                if x >= 1u128 << 123 {
                    out.cases += 1;
                    out.flagged.push(base);
                    continue;
                }
                if (node.oee == 2 && x % 2 == 0) || (node.oee == 1 && x % 4 == 0) {
                    continue;
                }
                let (hash, iters, flagged) = step_until_below(base, x + 1, 1u128 << plan.cutoff_bits());
                out.cases += 1;
                out.iters += iters;
                out.hash ^= hash;
                if flagged { out.flagged.push(base); }
            }
        }
        out.flagged.sort_unstable();
        out
    }

    #[test]
    fn compiled_suffixes_match_dfs_and_exact_start_ledger() {
        for (n, m) in [(32, 22), (72, 62)] {
            let sources = sources(m, 24);
            for a in [10, 8, 6, 4, 2, 0] {
                let plan = Plan::new(n, m, a, None, 16, &WindowRules::default());
                for source in &sources {
                    let mut actual = Vec::new();
                    plan.for_each_endpoint(source, |node| actual.push(*node));
                    actual.sort_unstable_by_key(|node| node.n0);
                    let mut dfs_nodes = Nodes::default();
                    dfs::<_, false>(source, n - a, &Rules::baseline(), &mut dfs_nodes, &mut TreeStats::default());
                    dfs_nodes.0.sort_unstable_by_key(|node| node.n0);
                    assert_eq!(actual, dfs_nodes.0);
                    let exact: Vec<_> = (0..1u128 << (n - a - m))
                        .filter_map(|q| exact_suffix(source, n - a, q)).collect();
                    assert_eq!(actual, exact);
                    let mut ledger = vec![false; 1 << (n - m)];
                    for node in &actual {
                        assert_eq!(crate::arith::t_iter(node.n0, node.k), (node.m, node.f));
                        for h in 0..1u128 << a {
                            let base = node.n0 + (h << node.k);
                            let original_h = ((base - source.n0) >> m) as usize;
                            assert!(!ledger[original_h], "duplicated original start");
                            ledger[original_h] = true;
                        }
                    }
                    for (h, &present) in ledger.iter().enumerate() {
                        let q = h as u128 & ((1u128 << (n - a - m)) - 1);
                        assert_eq!(present, exact_suffix(source, n - a, q).is_some());
                    }
                }
            }
        }
    }

    #[test]
    fn screening_tail_and_forced_flags_match_scalar_reference_at_every_a() {
        let sources = sources(62, 16);
        let rules = Rules::improved(5);
        let window = WindowRules { pm_inf: true, mix: rules.mix.clone() };
        let mut forced_flags = 0;
        for a in [10, 8, 6, 4, 2, 0] {
            for b in [None, Some(12)] {
                let plan = Plan::new(72, 62, a, b, 16, &window).with_cutoff_bits(70);
                for source in &sources {
                    let mut actual = plan.process(source);
                    actual.flagged.sort_unstable();
                    let expected = reference(&plan, source);
                    assert_eq!(actual, expected, "a={a} b={b:?} source={source:?}");
                    let mut states = Vec::new();
                    let mut screened = plan.screen_source(source, &mut states);
                    for state in states {
                        let (hash, iters, flagged) = step_until_below(state.base(), state.number(), 1u128 << 70);
                        screened.hash ^= hash;
                        screened.iters += iters;
                        if flagged { screened.flagged.push(state.base()); }
                    }
                    screened.flagged.sort_unstable();
                    assert_eq!(screened, expected);
                    forced_flags += actual.flagged.len();
                    for flag in actual.flagged { assert!(recheck_wide(flag).is_some()); }
                }
            }
        }
        assert!(forced_flags > 1000);
    }
}
