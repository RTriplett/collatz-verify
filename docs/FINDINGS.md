# Measured findings and optimization tradeoffs

This document summarizes the verification workload, the benefits of individual
optimizations, and the evidence behind the recommended configurations. Baseline
endpoint counts come from exact dynamic programming. Additional pruning, cases,
and tail iterations are estimated from uniformly sampled real endpoints at the
target N. GPU-year estimates use the RTX 4090 kernel timings and fitted
per-operation model described in [COST_MODEL.md](COST_MODEL.md).

## Verification cost

| Configuration at N = 72 | Estimated RTX 4090-years | Evidence |
|---|---|---|
| Sound baseline, A = 10, B = 24 | 1.03 | Exact baseline endpoint counts, sampled work, and kernel timings |
| Step-1 pm∞ + mix(3^13), A = 10, B = 24 | 0.983 | About 4.3% less GPU time than the sound baseline |
| B = 28 with window credits, baseline step-1 rules | 0.889 | About 13.5% less GPU time than the sound baseline |
| Step-1 pm∞ + mix(3^13) and window credits, A = 10, B = 28 | 0.851 | About 17% less GPU time than the sound baseline |
| Per-class windows on a 24 GB card, shared bitvectors, A = 10 | 0.675 | About 35% less GPU time than the sound baseline; see [HARDWARE.md](HARDWARE.md) |

These are estimates for a complete run below 2^72, not completed record runs.
Only the RTX 4090 is timed here; figures for other GPUs are hardware-scaling
estimates. Per-bit growth is not constant: the measured workload grows by about
1.83–1.94× between successive N, oscillating with the log2 3 lattice. Use the
[cost table](../results/cost_table.md) rather than a fixed growth factor.

Complete runs at N = 50, 54, and 56 validate the estimator. In the recommended
24 GB configuration at N = 56 (A = 12), the pipeline took 765 s against 756
GPU-seconds predicted, plus an 83 s array build. Cases and iterations agreed
within 0.04%. The saved runs are in [results/validation](../results/validation/).

## Step-1 pruning and CPU cost

The exact k = 37 census reproduces 575,349,866 baseline survivors and
543,518,371 with pm∞ + mix(3^16), a 5.53% reduction. At the N = 72 endpoint depth
k = 62, sampling gives a 7.39% reduction for the same rules. Endpoint reduction
overstates the reduction in GPU work because the rules mostly remove cheaper
low-f endpoints.

The following comparisons use A = 10, B = 24, sound split bitvectors, and no
extra window credits. Percentages are reductions relative to the common baseline
with descent, pm1, and root-armed OEE. They come from the individual sampled runs
in [results/n72](../results/n72/README.md), rather than the fitted timing sweep.

| Additional step-1 rules | Endpoints removed | Cases removed | Tail iterations removed |
|---|---|---|---|
| pm∞ + mix(3^13) | 6.93% | 4.25% | 4.04% |
| pm∞ + mix(3^16) | 7.39% | 4.63% | 4.40% |
| pm∞ + mix(3^13) + walkback at k = 50, capped at 65,536 states | 7.66% | 4.91% | 4.67% |

The CPU tradeoff matters. In the 16-thread Ryzen 7700X census benchmark, the
cost per node per thread is 16.44 ns for the baseline, 32.93 ns with pm∞ +
mix(3^13), and 79.97 ns with pm∞ + mix(3^16). The J = 13 tables occupy about
19 MB and fit in the 32 MB L3 cache; J = 16 uses about 0.5 GB. The extra pruning
from J = 16 is small relative to its CPU cost. See
[cpu_bench.csv](../results/cpu_bench.csv) and [ALGORITHM.md](ALGORITHM.md#improvements).

### Optional checkpoint walkback

Walkback is implemented as an optional bounded inverse-tree search at one
checkpoint. `--rules improved` enables pm∞ and mix only; walkback requires
`--walkback-depth K`. The state budget defaults to 65,536 and is set with
`--walkback-states S`. It keeps a class if it exhausts the budget without finding
a certificate. This is not an exhaustive walkback at every depth.

The saved N = 72 experiment compares
[pm∞ + mix(3^13)](../results/n72/measure_n72_b24_mix13.json) with
[the same rules plus walkback at k = 50](../results/n72/measure_n72_mix13_wb50.json).
Adding walkback removes about **0.79% of the remaining endpoints, 0.68% of the
remaining cases, and 0.66% of the remaining tail iterations**. Equivalently, the
iteration reduction rises from 4.04% to 4.67% of the original baseline, a gain
of about 0.64 percentage points. These are sampled estimates from runs with
different sample sizes, so the rounded differences should not be read as exact
counts.

The small marginal gain reflects overlapping certificates: mix already
precomputes useful inverse words that an unrestricted walkback could discover.
Gains measured separately against a baseline cannot simply be multiplied.
The checkpoint and state cap also limit which additional certificates this
implementation can find.

That original experiment predates the window-credit and large-window comparisons.
The follow-up below measures CPU search costs and samples the work removed with
the recommended 24 GB windows. There is still **no direct GPU timing or complete
verification benchmark with walkback toggled**, so walkback remains outside the
headline configurations.

### Reusable checkpoint certificates

The deeper tree dominates verification cost. For N = 72, A = 10 (endpoint depth
62), exact baseline DP puts only **0.0467%** of generated nodes at depths through
50, and **0.0000242%** through 38. Merely loading a precomputed shallow frontier
saves little traversal. Finding a certificate there is more valuable: it removes
the class's entire deeper subtree and its GPU work.

The current online walkback already searches once per checkpoint node, rather
than once per final endpoint or high-bit completion. Persistence provides reuse
across runs, bounds, window choices, and machines; it does not add another
within-run factor of `2^(N-K)`.

Fresh probes use pm∞ + mix(3^13), a 65,536-state search cap, N = 72, A = 10,
shared bitvectors with window credits, and class windows
`32,35,35,35,35,35,32,32,27,27,27,27,27,27,27,27`.
Each row is a **separate single-checkpoint alternative**; the savings cannot be
added together.

| Checkpoint K | Estimated search CPU-hours, one thread | Modeled deeper CPU core-hours removed | Modeled RTX 4090-hours removed | Compact exclusion keys |
|---|---:|---:|---:|---:|
| 30 | 0.0062 (22 s) | 89.5 ± 11.6 | 12.8 ± 0.6 | 0.18 MB |
| 34 | 0.135 (8.1 min) | 162.1 ± 15.4 | 20.7 ± 0.8 | 3.3 MB |
| 38 | 2.83 | 227.4 ± 18.4 | 26.7 ± 0.9 | 51 MB |
| 42 | 63.6 | 253.5 ± 19.1 | 32.8 ± 1.0 | 881 MB |
| 50 | 30,422 | 369.3 ± 22.9 | 43.9 ± 1.2 | 174 GB |

Search estimates multiply sampled single-thread timings on a Ryzen 7700X by
the checkpoint population, including nodes whose descendants eventually die
before reaching the GPU. They exclude generating those nodes, building and
validating the cache, and I/O. CPU traversal savings use the existing 32.93 ns
per thread-node census measurement, divided by two threads per core, without
the hardware report's additional 10% production allowance. These columns are
resource estimates, not elapsed time at a specified parallelism; SMT scaling
of reverse searches was not measured. Key sizes are decimal MB/GB for eight
bytes per excluded residue; the demonstrated certificate format uses four times
that space, and lookup structures may add overhead.

The GPU work removed at K = 34 and 38 is approximately **0.35% and 0.45%** of
the recommended 24 GB model. These gains are modest, but the shallow search
costs are small enough to make them promising even for a single full run.
Depth 50 spends vastly more CPU time finding somewhat more certificates.
Reusing a cache improves this tradeoff further, and a GPU builder could change
the preprocessing cost, but neither implies that deeper is always better.

The ± figures are **one sampling standard error**, conditional on the cost
models. GPU savings use sampled operation counts and the saved RTX 4090 timing
fit, not new kernel timings. They exclude changes in batching, warp behavior,
transfers, and cache integration. The percentage denominator comes from the
older window sweep, so it is an approximate relative comparison rather than a
paired measurement. A sub-percent wall-time claim would need direct validation:
the timing model's absolute uncertainty is larger than this marginal gain.

#### Complete depth-30 cache demonstration

The [cache prototype](../scripts/walkback_reuse/cache_demo.rs) enumerated all
**6,936,282** depth-30 survivors under the current improved rules and found
**21,101** additional certificates. Enumeration and reverse searches took
**3.00 s on eight CPU threads**; the serialized cache occupies **675,232 bytes**.
Every serialized record passed a separate forward-orbit and exact-slope check
in Rust and an independent Python arbitrary-precision audit.

On 20,000 independent baseline endpoint draws at depth 62, 18,623 survived the
improved rules. Their checkpoint ancestors produced 39 cache hits and **zero
mismatches** against fresh capped searches. Lookup took 0.681 ms versus 48.547 ms
for fresh searches on that same list. These are single-pass query timings, not
production throughput or an end-to-end verification speedup. The saved
[metadata](../results/walkback_reuse/cache-k30/metadata.json),
[certificates](../results/walkback_reuse/cache-k30/certificates-k30.bin), and
[independent audit](../results/walkback_reuse/cache-k30/python-audit.json)
make the small demonstration fully reproducible.

#### Integration and further reuse

The useful artifact is a sparse set of **positive certificates**, indexed by
`(K, n0)`, rather than a complete list of surviving nodes. A record proves
`T^s(p) = T^K(n0)` with the required odd-step counts, `0 < p < n0`, and
`2^s * 3^(f-o) <= 2^K`. It therefore certifies every member `n0 + 2^K*a` for
`a >= 0`. The same record remains valid at other N, A, and B whenever the
checkpoint is reached, and independently of the cheap rules that discovered
it. Missing entries merely lose pruning; they never authorize an exclusion.

Validate the certificates once on loading, then check a class at the checkpoint
and enter the existing deep DFS. A work-package root already deeper than K
needs its low-K ancestor checked before DFS starts. Roots shallower than K
can traverse to K before checking. Preserve the original package coverage;
importing an entire survivor frontier would also require a separate coverage
proof. Avoid adding a cache lookup to every deep node: the run generates about
9 quadrillion improved-tree nodes, so even small hot-loop overhead matters.
The prototype is separate from `cv`; production cache loading is not implemented.

There is another form of reuse: retain successful **inverse words** rather than
only their decisions for fixed classes. A word with s steps and o odd steps
defines `p = (2^s*m - b)/3^o` on one residue of `m mod 3^o`, so it can apply at
other depths. The current mix tables already do this for a bounded word family.
A sparse dictionary of longer words discovered by walkback could extend that
family without a dense `3^J` table. Each use still needs a valid word, the residue
condition, `o <= f`, a positive smaller partner, and the exact slope inequality;
a favorable step-count ratio alone is insufficient. Longer words match finer
residues and add lookup costs. Their usefulness should be measured on held-out
paths and depths after the existing rules. This extension has not been tested.

#### Reproducing the probes

The CPU-cost probe uses 100,000 uniform baseline checkpoint draws per depth,
then filters improved-rule ancestry. The GPU-savings probe uses 30,000 uniform
baseline endpoint draws in each of 16 exact f-classes, retaining zero savings
for paths already culled; it reconstructs their checkpoint ancestors and
evaluates actual window and tail work for removed endpoints. The tree probe
uses 10,000 uniform parent draws at each depth 30–61 to count generated children
removed below each checkpoint. Search repetitions, seeds, populations, class
results, and sampling errors are saved in
[results/walkback_reuse](../results/walkback_reuse/).

From the repository root, with Rust, Python 3, and `sha256sum` installed:

```sh
cargo build --release --offline --manifest-path scripts/walkback_reuse/Cargo.toml \
    --target-dir /tmp/collatz-reuse-target
mkdir -p /tmp/collatz-reuse-results
python3 scripts/walkback_reuse/prepare_model.py > /tmp/collatz-reuse-results/model.json
/tmp/collatz-reuse-target/release/checkpoint_cost 100000 > /tmp/collatz-reuse-results/checkpoint-cost.jsonl
/tmp/collatz-reuse-target/release/target_savings 30000 17391 /tmp/collatz-reuse-results/model.json \
    > /tmp/collatz-reuse-results/target-savings.jsonl
/tmp/collatz-reuse-target/release/tree_savings 10000 > /tmp/collatz-reuse-results/tree-savings.jsonl
/tmp/collatz-reuse-target/release/cache_demo /tmp/collatz-reuse-results/cache-k30
python3 scripts/walkback_reuse/verify_cache.py /tmp/collatz-reuse-results/cache-k30
```

All probes are CPU-only. `--offline` requires dependencies already cached;
omit it for the first build if needed. Timing results vary by host.

## Bitvectors and the host pipeline

1. **The reference bitvectors have a soundness gap.** The window's mod-3 merge
   credit miscounts trailing even steps for endpoints whose trailing-even count
   is odd. At N = 26, 386 of 857,601 exclusions lack a valid witness for the
   credited window rules. Two bitvector families fix this; shared bitvectors
   provide a conservative alternative with half the memory. The worked example
   and exhaustive witness audits are described in
   [ALGORITHM.md](ALGORITHM.md#soundness-fix-two-bitvector-families).
2. **The host pipeline can limit GPU throughput.** With multithreaded step 1,
   16-byte endpoints, and pinned staging, the saved N = 50 run takes 23.4 s on
   an RTX 4090, versus Angeltveit's reported 166 s on an RTX 3060. Earlier
   single-process reference runs were limited by one CPU thread. The roughly
   7× ratio compares both hardware and implementations.
3. **Lookahead depth saves more GPU work than the additional step-1 rules.**
   B = 28 with window credits cuts GPU time by 13.5% at N = 72. Larger windows
   continue to help; device construction and per-class allocation allow windows
   of 35 for the heavy classes on a 24 GB card. Timing arrays much larger than
   L2 agrees with the fitted model within a few percent. The extra memory on
   larger cards buys smaller additional savings; see [HARDWARE.md](HARDWARE.md).
4. **Window credits have no per-endpoint search cost.**
   `--bv-pm-inf --bv-mix-j 13` incorporates extra merge certificates during
   bitvector construction. At N = 72, A = 10, B = 24, these credits remove
   about 3.42% of cases before the step-1 improvements are applied.
5. **Larger A reduces CPU demand.** A = 12 needs about 3.6× less step-1 CPU
   work than A = 10, at about 4% more GPU time with the measured large windows.
   It is the recommended setting when the host cannot keep the GPU fed at
   A = 10.
6. **The reference CPU and CUDA programs use different OEE initial states.**
   The CPU program starts unarmed; the CUDA input starts armed. At k = 37,
   those choices give 678,988,765 and 575,349,866 survivors respectively.
   This project uses the root-armed state by default and exposes
   `--no-root-oee` for comparison.
