//! Full checks through a fixed cutoff, compact baseline traversal to the
//! terminal table boundary, then direct enumeration of its surviving leaves.
use crate::fast_tail::Endpoints;
use crate::terminal::Table;
use cv::arith::slope_descends;
use cv::rules::{Cull, Rules};
use cv::tree::{next_even_run, Node, TreeStats};

const POW3: [u64; 65] = {
    let mut p = [1u64; 65];
    let mut i = 1;
    while i < 65 {
        p[i] = p[i - 1].wrapping_mul(3);
        i += 1;
    }
    p
};

pub fn run<S: Endpoints, const STATS: bool, const GROUPED: bool>(
    root: &Node,
    cutoff: u32,
    full: &Rules,
    base: &Rules,
    table: &Table,
    sink: &mut S,
    stats: &mut TreeStats,
) {
    assert!(base.is_parity_only());
    assert_eq!(
        (full.pm1, full.oee, full.root_oee),
        (base.pm1, base.oee, base.root_oee)
    );
    assert!(root.k >= 2 && root.k <= table.start() && cutoff <= table.start());
    assert!(table.target() <= 64);
    assert!(root.n0 < (1u128 << root.k));
    full_prefix::<S, STATS, GROUPED>(root, cutoff, full, base, table, sink, stats);
}

fn full_prefix<S: Endpoints, const STATS: bool, const GROUPED: bool>(
    node: &Node,
    cutoff: u32,
    full: &Rules,
    base: &Rules,
    table: &Table,
    sink: &mut S,
    stats: &mut TreeStats,
) {
    if node.k >= cutoff {
        compact::<S, STATS, GROUPED>(Small::of(node), base, table, sink, stats);
        return;
    }
    for child in node.children(full) {
        let Some(child) = child else {
            if STATS {
                stats.culls[node.k as usize + 1][Cull::Oee as usize] += 1;
            }
            continue;
        };
        if STATS {
            stats.generated[child.k as usize] += 1;
        }
        if let Some(why) = child.cull(full) {
            if STATS {
                stats.culls[child.k as usize][why as usize] += 1;
            }
        } else {
            if STATS {
                stats.survived[child.k as usize] += 1;
            }
            full_prefix::<S, STATS, GROUPED>(&child, cutoff, full, base, table, sink, stats);
        }
    }
}

#[derive(Clone, Copy)]
struct Small {
    n0: u64,
    m: u64,
    k: u32,
    f: u32,
    oee: u8,
    ne: u8,
}
impl Small {
    fn of(n: &Node) -> Self {
        Self {
            n0: n.n0 as u64,
            m: n.m as u64,
            k: n.k,
            f: n.f,
            oee: n.oee,
            ne: n.ne,
        }
    }
    #[inline(always)]
    fn children(self, rules: &Rules) -> [Option<Self>; 2] {
        let arm = u8::from(rules.oee && slope_descends(self.k + 1, self.f));
        let odd_state = (self.oee | arm) & !2;
        let even_ok = self.oee < 2;
        let hi = 1u64 << self.k;
        let odd = |n0, x: u64| Self {
            n0,
            m: x.wrapping_add(x >> 1).wrapping_add(1),
            k: self.k + 1,
            f: self.f + 1,
            oee: odd_state,
            ne: 0,
        };
        let even = |n0, x: u64| Self {
            n0,
            m: x >> 1,
            k: self.k + 1,
            f: self.f,
            oee: self.oee << 1,
            ne: next_even_run(self.ne),
        };
        let x1 = self.m.wrapping_add(POW3[self.f as usize]);
        if self.m & 1 == 0 {
            [
                even_ok.then(|| even(self.n0, self.m)),
                Some(odd(self.n0 | hi, x1)),
            ]
        } else {
            [
                Some(odd(self.n0, self.m)),
                even_ok.then(|| even(self.n0 | hi, x1)),
            ]
        }
    }
    #[inline(always)]
    fn cull(self, rules: &Rules) -> Option<Cull> {
        if slope_descends(self.k, self.f) {
            return Some(Cull::Descent);
        }
        if rules.pm1 && self.ne == 2 && slope_descends(self.k - 1, self.f - 1) {
            return Some(Cull::Pm1);
        }
        None
    }
}

fn compact<S: Endpoints, const STATS: bool, const GROUPED: bool>(
    node: Small,
    base: &Rules,
    table: &Table,
    sink: &mut S,
    stats: &mut TreeStats,
) {
    if node.k == table.start() {
        table.emit::<S, STATS, GROUPED>(node.n0, node.f, node.m, node.oee, node.ne, sink, stats);
        return;
    }
    for child in node.children(base) {
        let Some(child) = child else {
            if STATS {
                stats.culls[node.k as usize + 1][Cull::Oee as usize] += 1;
            }
            continue;
        };
        if STATS {
            stats.generated[child.k as usize] += 1;
        }
        if let Some(why) = child.cull(base) {
            if STATS {
                stats.culls[child.k as usize][why as usize] += 1;
            }
        } else {
            if STATS {
                stats.survived[child.k as usize] += 1;
            }
            compact::<S, STATS, GROUPED>(child, base, table, sink, stats);
        }
    }
}
