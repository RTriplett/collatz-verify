//! Experimental finite-control-state block expansion.
//!
//! The compiled programs contain only prefixes that survive all baseline
//! parity tests.  Each remaining prefix is evaluated once, and all value
//! dependent rules are still checked at its original depth.  A failed prefix
//! jumps over its whole subtree.  This changes visitation order (odd parity
//! first), but not the complete set of endpoint Nodes.

use cv::arith::POW3;
use cv::dp::{ForwardDp, PState, ParityRules};
use cv::rules::Rules;
use cv::tree::{Node, Sink, P16};
use std::mem::size_of;

const MAX_WIDTH: usize = 8;

#[derive(Clone, Copy)]
struct Instruction {
    // The child is at depth root.k + depth; its parent occupies depth - 1.
    depth: u8,
    odd: bool,
    state: PState,
    // First instruction after this entire prefix subtree.
    skip: u32,
    hi: u128,
    pow3: u128,
    pow3_mod: u32,
}

pub struct Blocks {
    start: u32,
    target: u32,
    width: u32,
    // A dense state index is cheap enough to use at every block boundary.
    programs: Vec<Vec<Instruction>>,
}

impl Blocks {
    pub fn new(start: u32, target: u32, width: u32, rules: &Rules) -> Self {
        assert!(start >= 2 && start <= target);
        assert!((1..=MAX_WIDTH as u32).contains(&width));
        assert!(target < 80, "prototype uses the existing u128 powers of 3");
        let levels = (target - start).div_ceil(width) as usize;
        let count = levels * (target as usize + 1) * 12;
        let mut blocks = Self {
            start,
            target,
            width,
            programs: vec![Vec::new(); count],
        };
        let dp = ForwardDp::from_root(rules, target);
        let pr = ParityRules::of(rules);
        for k in (start..target).step_by(width as usize) {
            for (state, _) in dp.layer(k).states() {
                let mut code = Vec::new();
                compile(&mut code, state, k, (k + width).min(target), pr);
                let id = blocks.index(state);
                blocks.programs[id] = code;
            }
        }
        blocks
    }

    #[inline(always)]
    fn index(&self, state: PState) -> usize {
        (((state.k - self.start) / self.width) as usize * (self.target as usize + 1)
            + state.f as usize)
            * 12
            + state.oee as usize * 4
            + state.ne as usize
    }

    /// Allocated program storage, including the dense Vec directory.
    pub fn bytes(&self) -> usize {
        size_of::<Self>()
            + self.programs.capacity() * size_of::<Vec<Instruction>>()
            + self
                .programs
                .iter()
                .map(|p| p.capacity() * size_of::<Instruction>())
                .sum::<usize>()
    }
}

fn compile(code: &mut Vec<Instruction>, parent: PState, root: u32, end: u32, pr: ParityRules) {
    let (children, _) = pr.children(parent);
    for (parity, state) in children.into_iter().enumerate() {
        let Some(state) = state else { continue };
        let here = code.len();
        code.push(Instruction {
            depth: (state.k - root) as u8,
            odd: parity == 0,
            state,
            skip: 0,
            hi: 1u128 << parent.k,
            pow3: POW3[parent.f as usize],
            pow3_mod: if parent.f < 16 {
                POW3[parent.f as usize] as u32
            } else {
                0
            },
        });
        if state.k < end {
            compile(code, state, root, end, pr);
        }
        code[here].skip = code.len().try_into().unwrap();
    }
}

#[inline(always)]
fn half_mod(mut value: u32) -> u32 {
    if value >= 2 * P16 as u32 {
        value -= 2 * P16 as u32;
    }
    if value >= P16 as u32 {
        value -= P16 as u32;
    }
    (value + (value & 1) * P16 as u32) >> 1
}

/// Expand an already-surviving root.  Descendants are visited in parity order.
pub fn dfs_blocks<S: Sink>(node: &Node, target: u32, rules: &Rules, blocks: &Blocks, sink: &mut S) {
    assert_eq!(node.k, blocks.start);
    assert_eq!(target, blocks.target);
    expand(node, rules, blocks, sink);
}

fn expand<S: Sink>(node: &Node, rules: &Rules, blocks: &Blocks, sink: &mut S) {
    if node.k == blocks.target {
        sink.endpoint(node);
        return;
    }
    let end = (node.k + blocks.width).min(blocks.target);
    let program = &blocks.programs[blocks.index(PState::of(node))];
    // A depth slot remains live until all of that prefix's descendants have
    // been consumed.  The skip links ensure no failed prefix is ever read.
    let mut values = [*node; MAX_WIDTH + 1];
    let mut ip = 0usize;
    while ip < program.len() {
        let op = &program[ip];
        let parent = &values[op.depth as usize - 1];
        // Since 3^f is odd, setting the new starting bit toggles this step's
        // parity.  This identifies the unique desired parity child.
        let bit = ((parent.m & 1) as u8 ^ op.odd as u8) as u128;
        let x = parent.m + bit * op.pow3;
        let n0 = parent.n0 | (bit * op.hi);
        let m = if op.odd { x + (x >> 1) + 1 } else { x >> 1 };
        let mut x3 = parent.m3 + bit as u32 * op.pow3_mod;
        if x3 >= P16 as u32 {
            x3 -= P16 as u32;
        }
        let m3 = half_mod(if op.odd { 3 * x3 + 1 } else { x3 });
        if m < n0
            || rules
                .extra_cull(n0, op.state.k, op.state.f, m, m3)
                .is_some()
        {
            ip = op.skip as usize;
            continue;
        }
        let child = Node {
            n0,
            k: op.state.k,
            f: op.state.f,
            m,
            m3,
            oee: op.state.oee,
            ne: op.state.ne,
        };
        if child.k == end {
            expand(&child, rules, blocks, sink);
        } else {
            values[op.depth as usize] = child;
        }
        ip += 1;
    }
}
