//! Pluggable host expansion with the verifier's normal endpoint sinks.
//!
//! The default expander is the reference DFS. Alternative implementations can
//! reuse the full CPU/GPU pipeline, including accounting and CPU fallback.

use crate::rules::Rules;
use crate::tree::{dfs, Node, Sink, TreeStats};

pub trait Expander: Sync {
    /// Expand an already-surviving root to `target`. As with [`dfs`], the root
    /// itself is neither checked nor counted. Implementations must preserve
    /// complete endpoint state and their chosen policy's per-depth counters.
    fn expand<S: Sink, const STATS: bool>(
        &self,
        root: &Node,
        target: u32,
        rules: &Rules,
        sink: &mut S,
        stats: &mut TreeStats,
    );
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DfsExpander;

impl Expander for DfsExpander {
    fn expand<S: Sink, const STATS: bool>(
        &self,
        root: &Node,
        target: u32,
        rules: &Rules,
        sink: &mut S,
        stats: &mut TreeStats,
    ) {
        dfs::<S, STATS>(root, target, rules, sink, stats);
    }
}
