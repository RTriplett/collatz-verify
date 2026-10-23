//! Collatz convergence verification for all `n < 2^N` by LSB-first residue
//! tree search (Angeltveit 2026) with additional path-merge certificates,
//! plus exact workload counting and a measured cost model.
//!
//! Pipeline (`A` = bitvector bits, `M = N - A`, `B` = lookahead window):
//! 1. [`tree`]: DFS over classes `n0 mod 2^k`, `k <= M`, pruned by [`rules`].
//! 2. [`lookahead`]: each depth-`M` endpoint's `2^A` completions are filtered
//!    by one `B`-step bitvector and the mod-9 preimage sieve.
//! 3. [`tail`] (CPU) / `gpu` (CUDA): survivors are iterated until they fall
//!    below their starting value.
//!
//! [`dp`] counts the baseline workload exactly at any depth, [`sample`]
//! estimates the effect of the value-dependent rules at depths that cannot
//! be enumerated, and [`model`] turns counts and measured unit costs into
//! resource estimates.

pub mod arith;
pub mod dp;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod lookahead;
pub mod measure;
pub mod rules;
pub mod sample;
pub mod tail;
pub mod tree;
pub mod units;
pub mod verify;
