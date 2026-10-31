//! Compile the last few baseline-only levels into direct endpoint emission.
//!
//! A parity word of length b has a unique input residue r modulo 2^b. For a
//! depth-k class with iterate m and odd count f, its extensions have iterates
//! m + 3^f q, 0 <= q < 2^b. Thus the extension realizing that word is exactly
//!
//!     q = (r - m) * 3^(-f) mod 2^b.
//!
//! Compilation stores R = r * 3^(-f) mod 2^b, so emission needs one modular
//! multiplication per root, then one subtraction/mask/shift per endpoint.
//! The compact emission path needs neither intermediate nor final iterates. An
//! additional affine representation retains r, T^b(r), and the local 3-power,
//! allowing exact endpoint reconstruction or resident-table GPU expansion. All
//! baseline prefix checks depend only on (k, f, oee, ne), and are performed
//! during compilation. The baseline's m < n0 condition adds no removal here:
//! for k <= 64 the slope predicate agrees with 3^f <= 2^k, and otherwise the
//! affine iterate (3^f*n0+c)/2^k, c >= 0, cannot be below n0.
//!
//! The table also stores complete per-depth counters, including paths that
//! die before the terminal level. Counters are merged once per root. Leaf
//! order differs from residue-bit DFS; payload sets and every counter agree.
use crate::fast_tail::Endpoints;
use cv::arith::slope_descends;
use cv::dp::ForwardDp;
use cv::rules::{Cull, Rules, NUM_CULLS};
use cv::tree::{next_even_run, Node, Sink, TerminalWord, TreeStats};
use std::mem::size_of;

const INV3: u64 = 0xaaaa_aaaa_aaaa_aaab;
const INV3_POW: [u64; 65] = {
    let mut result = [1u64; 65];
    let mut i = 1;
    while i < result.len() {
        result[i] = result[i - 1].wrapping_mul(INV3);
        i += 1;
    }
    result
};

#[derive(Clone, Copy)]
struct State {
    k: u32,
    f: u32,
    oee: u8,
    ne: u8,
}

#[derive(Clone, Copy)]
struct Entry {
    r: u16,
    f: u8,
    oee: u8,
    ne: u8,
}

struct Group {
    f: u32,
    oee: u8,
    ne: u8,
    residues: Box<[u16]>,
}

/// One contiguous, single-final-F range in the resident affine word table.
/// Within a range, words retain the original parity-word DFS order.
#[derive(Clone, Copy, Debug)]
pub struct WordGroup {
    pub f: u32,
    pub start: u32,
    pub len: u32,
}

#[derive(Clone, Copy, Default)]
struct Counts {
    generated: u64,
    survived: u64,
    culls: [u64; NUM_CULLS],
}

struct Program {
    flat: Box<[Entry]>,
    grouped: Box<[Group]>,
    word_groups: Box<[WordGroup]>,
    counts: Box<[Counts]>,
}

/// Fixed terminal depths and baseline flags; reusable across all roots with
/// parity states reachable from the corresponding baseline tree root.
pub struct Table {
    start: u32,
    target: u32,
    mask: u64,
    programs: Vec<Option<Program>>,
    words: Vec<TerminalWord>,
}

#[inline(always)]
fn index(f: u32, oee: u8, ne: u8) -> usize {
    (f as usize * 3 + oee as usize) * 4 + ne as usize
}

impl Table {
    pub fn new(start: u32, target: u32, rules: &Rules) -> Self {
        assert!(
            rules.is_parity_only(),
            "terminal tables require parity-only rules"
        );
        assert!(start >= 2 && target <= 64 && target > start);
        assert!(target - start <= 12, "terminal width must be in 1..=12");
        let width = target - start;
        let mask = (1u64 << width) - 1;
        let reachable = ForwardDp::from_root(rules, start);
        let mut programs: Vec<Option<Program>> =
            (0..(start as usize + 1) * 12).map(|_| None).collect();
        let mut words = Vec::new();
        for (state, _) in reachable.layer(start).states() {
            let mut affine = Vec::new();
            let mut counts = vec![Counts::default(); width as usize];
            compile(
                State {
                    k: start,
                    f: state.f,
                    oee: state.oee,
                    ne: state.ne,
                },
                start,
                target,
                0,
                0,
                0,
                1,
                INV3_POW[state.f as usize],
                mask,
                rules,
                &mut affine,
                &mut counts,
            );
            // Preserve the original compact payload representation and its
            // original iteration order. The affine copy retains exact local
            // iterates and is grouped only by final F for GPU dispatch.
            let entries: Vec<_> = affine
                .iter()
                .map(|word| Entry {
                    r: word.q_coeff as u16,
                    f: word.f as u8,
                    oee: word.oee as u8,
                    ne: word.ne as u8,
                })
                .collect();
            affine.sort_by_key(|word| word.f);
            let mut word_groups = Vec::new();
            let mut offset = 0;
            while offset < affine.len() {
                let f = affine[offset].f;
                let len = affine[offset..]
                    .iter()
                    .take_while(|word| word.f == f)
                    .count();
                word_groups.push(WordGroup {
                    f,
                    start: u32::try_from(words.len() + offset)
                        .expect("terminal table is too large"),
                    len: len as u32,
                });
                offset += len;
            }
            words.extend(affine);
            let mut sorted = entries.clone();
            sorted.sort_unstable_by_key(|e| (e.f, e.oee, e.ne));
            let mut groups: Vec<Group> = Vec::new();
            let mut offset = 0;
            while offset < sorted.len() {
                let first = sorted[offset];
                let len = sorted[offset..]
                    .iter()
                    .take_while(|e| (e.f, e.oee, e.ne) == (first.f, first.oee, first.ne))
                    .count();
                groups.push(Group {
                    f: first.f as u32,
                    oee: first.oee,
                    ne: first.ne,
                    residues: sorted[offset..offset + len]
                        .iter()
                        .map(|e| e.r)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                });
                offset += len;
            }
            programs[index(state.f, state.oee, state.ne)] = Some(Program {
                flat: entries.into_boxed_slice(),
                grouped: groups.into_boxed_slice(),
                word_groups: word_groups.into_boxed_slice(),
                counts: counts.into_boxed_slice(),
            });
        }
        Self {
            start,
            target,
            mask,
            programs,
            words,
        }
    }

    pub fn start(&self) -> u32 {
        self.start
    }
    pub fn target(&self) -> u32 {
        self.target
    }

    /// Resident affine words shared by every parent and worker. Group indices
    /// are absolute indices into this slice, including across input states.
    pub fn words(&self) -> &[TerminalWord] {
        &self.words
    }

    pub fn groups(&self, parent: &Node) -> &[WordGroup] {
        assert_eq!(parent.k, self.start);
        &self.programs[index(parent.f, parent.oee, parent.ne)]
            .as_ref()
            .expect("terminal root state is unreachable under the supplied baseline rules")
            .word_groups
    }

    /// Number of descendants of a reachable parity state. This depends only
    /// on the state's flags and odd count, not on the parent's residue.
    pub fn descendants(&self, f: u32, oee: u8, ne: u8) -> u64 {
        self.programs[index(f, oee, ne)]
            .as_ref()
            .expect("terminal root state is unreachable under the supplied baseline rules")
            .flat
            .len() as u64
    }

    /// Total valid parity-word entries across reachable root states. Flat and
    /// grouped copies represent the same entries, so each is counted once.
    pub fn entries(&self) -> usize {
        self.programs.iter().flatten().map(|p| p.flat.len()).sum()
    }

    pub fn states(&self) -> usize {
        self.programs.iter().flatten().count()
    }

    /// Allocated payload bytes, including all experimental representations,
    /// counters, dense program slots and vector/box headers; allocator overhead
    /// is excluded. Table construction and this storage are shared by workers.
    pub fn bytes(&self) -> usize {
        size_of::<Self>()
            + self.programs.capacity() * size_of::<Option<Program>>()
            + self.words.capacity() * size_of::<TerminalWord>()
            + self
                .programs
                .iter()
                .flatten()
                .map(|p| {
                    p.flat.len() * size_of::<Entry>()
                        + p.grouped.len() * size_of::<Group>()
                        + p.word_groups.len() * size_of::<WordGroup>()
                        + p.grouped
                            .iter()
                            .map(|g| g.residues.len() * size_of::<u16>())
                            .sum::<usize>()
                        + p.counts.len() * size_of::<Counts>()
                })
                .sum::<usize>()
    }

    /// Emit exact affine descendants grouped by final F. The sink may retain
    /// the parent and table range for device expansion, or use its default
    /// implementation to reconstruct genuine full-width endpoint Nodes.
    #[inline]
    pub fn emit_affine<S: Sink, const STATS: bool>(
        &self,
        parent: &Node,
        sink: &mut S,
        stats: &mut TreeStats,
    ) {
        debug_assert_eq!(parent.k, self.start);
        let program = self.programs[index(parent.f, parent.oee, parent.ne)]
            .as_ref()
            .expect("terminal root state is unreachable under the supplied baseline rules");
        if STATS {
            for (offset, counts) in program.counts.iter().enumerate() {
                let k = self.start as usize + offset + 1;
                stats.generated[k] += counts.generated;
                stats.survived[k] += counts.survived;
                for c in 0..NUM_CULLS {
                    stats.culls[k][c] += counts.culls[c];
                }
            }
        }
        for group in &program.word_groups {
            let begin = group.start as usize;
            let entries = &self.words[begin..begin + group.len as usize];
            sink.terminal_group(parent, group.start, entries, self.target);
        }
    }

    /// Emit the terminal descendants of a genuine, already-surviving class.
    /// `m` need only be correct modulo 2^(target-start). The root itself is
    /// neither checked nor counted. GROUPED changes endpoint order only.
    #[inline]
    pub fn emit<S: Endpoints, const STATS: bool, const GROUPED: bool>(
        &self,
        n0: u64,
        f: u32,
        m: u64,
        oee: u8,
        ne: u8,
        sink: &mut S,
        stats: &mut TreeStats,
    ) {
        debug_assert!(n0 < (1u64 << self.start));
        debug_assert!(f <= self.start && oee < 3 && ne < 4);
        let program = self.programs[index(f, oee, ne)]
            .as_ref()
            .expect("terminal root state is unreachable under the supplied baseline rules");
        if STATS {
            for (offset, counts) in program.counts.iter().enumerate() {
                let k = self.start as usize + offset + 1;
                stats.generated[k] += counts.generated;
                stats.survived[k] += counts.survived;
                for c in 0..NUM_CULLS {
                    stats.culls[k][c] += counts.culls[c];
                }
            }
        }
        let d = m.wrapping_mul(INV3_POW[f as usize]) & self.mask;
        if GROUPED {
            for group in &program.grouped {
                for &r in &group.residues {
                    let q = (r as u64).wrapping_sub(d) & self.mask;
                    sink.endpoint(n0 | (q << self.start), group.f, group.oee, group.ne);
                }
            }
        } else {
            for entry in &program.flat {
                let q = (entry.r as u64).wrapping_sub(d) & self.mask;
                sink.endpoint(n0 | (q << self.start), entry.f as u32, entry.oee, entry.ne);
            }
        }
    }
}

// Simultaneously traverse abstract parity states and the canonical input
// residue realizing each parity prefix. `m = T^depth(r)` and `pow3 = 3^g`
// for the prefix's local odd count g; these values stay tiny for width <= 12.
#[allow(clippy::too_many_arguments)]
fn compile(
    state: State,
    start: u32,
    target: u32,
    depth: u32,
    r: u64,
    m: u64,
    pow3: u64,
    inv3_f: u64,
    mask: u64,
    rules: &Rules,
    entries: &mut Vec<TerminalWord>,
    counts: &mut [Counts],
) {
    if state.k == target {
        entries.push(TerminalWord {
            r: r as u32,
            q_coeff: (r.wrapping_mul(inv3_f) & mask) as u32,
            iterate: m as u32,
            pow3: pow3 as u32,
            f: state.f,
            oee: state.oee as u32,
            ne: state.ne as u32,
            pad: 0,
        });
        return;
    }
    let arm = u8::from(rules.oee && slope_descends(state.k + 1, state.f));
    for odd in [false, true] {
        let count = &mut counts[(state.k - start) as usize];
        if !odd && state.oee >= 2 {
            count.culls[Cull::Oee as usize] += 1;
            continue;
        }
        let next = State {
            k: state.k + 1,
            f: state.f + u32::from(odd),
            oee: if odd {
                (state.oee | arm) & !2
            } else {
                state.oee << 1
            },
            ne: if odd { 0 } else { next_even_run(state.ne) },
        };
        count.generated += 1;
        if slope_descends(next.k, next.f) {
            count.culls[Cull::Descent as usize] += 1;
            continue;
        }
        if rules.pm1 && next.ne == 2 && slope_descends(next.k - 1, next.f - 1) {
            count.culls[Cull::Pm1 as usize] += 1;
            continue;
        }
        count.survived += 1;
        let bit = (m & 1) ^ u64::from(odd);
        let x = m + pow3 * bit;
        let next_m = if odd { x + (x >> 1) + 1 } else { x >> 1 };
        compile(
            next,
            start,
            target,
            depth + 1,
            r | (bit << depth),
            next_m,
            if odd { pow3 * 3 } else { pow3 },
            inv3_f,
            mask,
            rules,
            entries,
            counts,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv::arith::POW3;
    use cv::sample::{Rng, Sampler};
    use cv::tree::{dfs, frontier, Node, Sink};

    #[derive(Default)]
    struct Payloads(Vec<(u64, u32, u8, u8)>);
    impl Endpoints for Payloads {
        fn endpoint(&mut self, n0: u64, f: u32, oee: u8, ne: u8) {
            self.0.push((n0, f, oee, ne));
        }
    }
    impl Sink for Payloads {
        fn endpoint(&mut self, node: &Node) {
            self.0.push((node.n0 as u64, node.f, node.oee, node.ne));
        }
    }

    #[derive(Default)]
    struct ExactNodes(Vec<Node>);
    impl Sink for ExactNodes {
        fn endpoint(&mut self, node: &Node) {
            self.0.push(*node);
        }
    }

    // An independent oracle loops over starting-number suffixes and simulates
    // their actual trajectories. Its baseline tests use exact powers rather
    // than the production rational slope predicate or parity transitions.
    fn brute(root: &Node, target: u32, rules: &Rules) -> Payloads {
        let mut out = Payloads::default();
        for q in 0..1u128 << (target - root.k) {
            let n0 = root.n0 | (q << root.k);
            let mut m = root.m + POW3[root.f as usize] * q;
            let mut f = root.f;
            let mut oee = root.oee;
            let mut ne = root.ne;
            let mut survives = true;
            for k in root.k + 1..=target {
                if m & 1 == 0 {
                    if oee >= 2 {
                        survives = false;
                        break;
                    }
                    m /= 2;
                    oee <<= 1;
                    ne = if ne >= 2 { 5 - ne } else { ne + 1 };
                } else {
                    let arm = u8::from(rules.oee && POW3[f as usize] <= 1u128 << k);
                    m = (3 * m + 1) / 2;
                    f += 1;
                    oee = (oee | arm) & !2;
                    ne = 0;
                }
                if POW3[f as usize] <= 1u128 << k
                    || (rules.pm1 && ne == 2 && POW3[(f - 1) as usize] <= 1u128 << (k - 1))
                {
                    survives = false;
                    break;
                }
            }
            if survives {
                out.0.push((n0 as u64, f, oee, ne));
            }
        }
        out
    }

    fn compare(roots: &[Node], target: u32, rules: &Rules, brute_check: bool) {
        let table = Table::new(roots[0].k, target, rules);
        for root in roots {
            let mut expected = Payloads::default();
            let mut expected_stats = TreeStats::new(target);
            dfs::<_, true>(root, target, rules, &mut expected, &mut expected_stats);
            expected.0.sort_unstable();
            if brute_check {
                let mut independent = brute(root, target, rules);
                independent.0.sort_unstable();
                assert_eq!(expected.0, independent.0, "independent suffix oracle");
            }
            let mut affine = ExactNodes::default();
            let mut affine_stats = TreeStats::new(target);
            table.emit_affine::<_, true>(root, &mut affine, &mut affine_stats);
            affine.0.sort_unstable_by_key(|node| node.n0);
            let exact: Vec<_> = expected
                .0
                .iter()
                .map(|&(n0, f, oee, ne)| Node::from_compact(n0, target, f, oee, ne))
                .collect();
            assert_eq!(
                affine.0, exact,
                "affine Nodes start={} target={target}",
                root.k
            );
            assert_eq!(affine_stats.generated, expected_stats.generated);
            assert_eq!(affine_stats.survived, expected_stats.survived);
            assert_eq!(affine_stats.culls, expected_stats.culls);
            for group in table.groups(root) {
                let words = &table.words()[group.start as usize..][..group.len as usize];
                assert!(words.iter().all(|word| word.f == group.f));
                let flat: Vec<_> = table.programs[index(root.f, root.oee, root.ne)]
                    .as_ref()
                    .unwrap()
                    .flat
                    .iter()
                    .filter(|entry| entry.f as u32 == group.f)
                    .map(|entry| (entry.r as u32, entry.oee as u32, entry.ne as u32))
                    .collect();
                let grouped: Vec<_> = words
                    .iter()
                    .map(|word| (word.q_coeff, word.oee, word.ne))
                    .collect();
                assert_eq!(
                    grouped, flat,
                    "final-F grouping must preserve within-class order"
                );
            }
            for grouped in [false, true] {
                let mut actual = Payloads::default();
                let mut actual_stats = TreeStats::new(target);
                if grouped {
                    table.emit::<_, true, true>(
                        root.n0 as u64,
                        root.f,
                        root.m as u64,
                        root.oee,
                        root.ne,
                        &mut actual,
                        &mut actual_stats,
                    );
                } else {
                    table.emit::<_, true, false>(
                        root.n0 as u64,
                        root.f,
                        root.m as u64,
                        root.oee,
                        root.ne,
                        &mut actual,
                        &mut actual_stats,
                    );
                }
                actual.0.sort_unstable();
                assert_eq!(
                    actual.0, expected.0,
                    "payloads start={} target={target} grouped={grouped}",
                    root.k
                );
                assert_eq!(actual_stats.generated, expected_stats.generated);
                assert_eq!(actual_stats.survived, expected_stats.survived);
                assert_eq!(actual_stats.culls, expected_stats.culls);

                let mut unrecorded = Payloads::default();
                // Empty counters ensure STATS=false cannot accidentally read
                // or write them, including counters for rejected prefixes.
                let mut untouched = TreeStats::default();
                if grouped {
                    table.emit::<_, false, true>(
                        root.n0 as u64,
                        root.f,
                        root.m as u64,
                        root.oee,
                        root.ne,
                        &mut unrecorded,
                        &mut untouched,
                    );
                } else {
                    table.emit::<_, false, false>(
                        root.n0 as u64,
                        root.f,
                        root.m as u64,
                        root.oee,
                        root.ne,
                        &mut unrecorded,
                        &mut untouched,
                    );
                }
                unrecorded.0.sort_unstable();
                assert_eq!(unrecorded.0, expected.0);
                assert!(
                    untouched.generated.is_empty()
                        && untouched.survived.is_empty()
                        && untouched.culls.is_empty()
                );
            }
        }
    }

    #[test]
    fn modular_extension_mapping_is_bijective_for_every_supported_width() {
        for width in 1..=12 {
            let mask = (1u64 << width) - 1;
            for f in [0usize, 1, 17, 38, 52, 64] {
                for m in [0u64, 1, 0x5432_1012_3456_789a, u64::MAX] {
                    let mut seen = vec![false; mask as usize + 1];
                    let d = m.wrapping_mul(INV3_POW[f]) & mask;
                    for r in 0..=mask {
                        let stored = r.wrapping_mul(INV3_POW[f]) & mask;
                        let q = stored.wrapping_sub(d) & mask;
                        assert!(!seen[q as usize]);
                        seen[q as usize] = true;
                        assert_eq!(m.wrapping_add((POW3[f] as u64).wrapping_mul(q)) & mask, r);
                    }
                    assert!(seen.into_iter().all(|b| b));
                }
            }
        }
    }

    #[test]
    fn complete_small_frontiers_match_reference_and_suffix_oracle() {
        let mut no_pm1 = Rules::baseline();
        no_pm1.pm1 = false;
        let mut unarmed = Rules::baseline();
        unarmed.root_oee = false;
        for rules in [Rules::baseline(), Rules::descent_only(), no_pm1, unarmed] {
            let roots = frontier(8, &rules, &mut TreeStats::new(8));
            for width in 1..=12 {
                compare(&roots, 8 + width, &rules, true);
            }
        }
    }

    #[test]
    fn sampled_wide_frontiers_match_wrapping_and_target_64() {
        let mut saw_wide = false;
        let mut states_seen = [[false; 4]; 3];
        for rules in [Rules::baseline(), Rules::descent_only()] {
            for (start, target) in [(38, 50), (50, 62), (54, 62), (54, 64), (60, 64)] {
                let sampler = Sampler::new(&rules, start, None);
                let mut rng = Rng(1_842_670 ^ start as u64);
                let roots: Vec<_> = (0..32).map(|_| sampler.draw(&mut rng, None).node).collect();
                for root in &roots {
                    saw_wide |= root.m > u64::MAX as u128;
                    states_seen[root.oee as usize][root.ne as usize] = true;
                }
                compare(&roots, target, &rules, true);
            }
        }
        assert!(saw_wide, "must exercise truncated full-width iterates");
        assert!(
            states_seen[1][0] && states_seen[2][1],
            "must exercise inherited OEE state"
        );
        assert!(
            states_seen.iter().any(|s| s[2]) && states_seen.iter().any(|s| s[3]),
            "must exercise folded even states"
        );
    }

    #[test]
    fn dead_prefixes_are_counted_even_though_they_emit_no_endpoints() {
        let rules = Rules::baseline();
        let roots = frontier(14, &rules, &mut TreeStats::new(14));
        let mut found = false;
        for target in 15..=26 {
            let table = Table::new(14, target, &rules);
            for root in &roots {
                let program = table.programs[index(root.f, root.oee, root.ne)]
                    .as_ref()
                    .unwrap();
                if program.counts[..program.counts.len() - 1]
                    .iter()
                    .any(|c| c.culls.iter().any(|&n| n > 0))
                {
                    compare(&[*root], target, &rules, true);
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
        }
        assert!(
            found,
            "must exercise a branch rejected before the terminal level"
        );
    }
}
