# Measured findings and optimization tradeoffs

This document summarizes the verification workload, the benefits of individual
optimizations, and the evidence behind the recommended configurations. Baseline
endpoint counts come from exact dynamic programming. Additional pruning, cases,
and tail iterations are estimated from uniformly sampled real endpoints at the
target N. Historical GPU-year estimates use the RTX 4090 kernel timings and
fitted per-operation model described in [COST_MODEL.md](COST_MODEL.md). The newer
checkpoint-pipeline estimate uses fixed baseline K38 cohorts and live elapsed
pipeline timings, with its different scope stated separately.

## Verification cost

| Configuration at N = 72 | Estimated RTX 4090-years | Evidence |
|---|---|---|
| Sound baseline, A = 10, B = 24 | 1.03 | Exact baseline endpoint counts, sampled work, and kernel timings |
| Step-1 pm∞ + mix(3^13), A = 10, B = 24 | 0.983 | About 4.3% less GPU time than the sound baseline |
| B = 28 with window credits, baseline step-1 rules | 0.889 | About 13.5% less GPU time than the sound baseline |
| Step-1 pm∞ + mix(3^13) and window credits, A = 10, B = 28 | 0.851 | About 17% less GPU time than the sound baseline |
| Per-class windows on a 24 GB card, shared bitvectors, A = 10 | 0.675 | About 35% less GPU time than the sound baseline; see [HARDWARE.md](HARDWARE.md) |
| Experimental prepared-K38 parent pipeline, A = 10, tuned kernel | **0.586 [0.565, 0.607]** | 8,192 baseline draws, repeated paired timings and aggregate checks; excludes frontier preparation and setup; [full report](../results/fullrange_pipeline/README.md) |

These are estimates for a complete run below 2^72, not completed record runs.
Only the RTX 4090 is timed here; figures for other GPUs are hardware-scaling
estimates. Per-bit growth is not constant: the measured workload grows by about
1.83–1.94× between successive N, oscillating with the log2 3 lattice. Use the
[cost table](../results/cost_table.md) rather than a fixed growth factor.

Complete runs at N = 50, 54, and 56 validate the earlier per-operation estimator. In the recommended
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
paths and depths after the existing rules. A dictionary has not been tested in
CPU tree traversal; the following experiment tests it during window construction.

#### Harvested words in lookahead

The 21,101 depth-30 certificates yield **14,169 distinct inverse words**, with
14–21 odd steps and 17–28 total steps. All lie beyond mix13's word family.
A CPU-only [window probe](../scripts/walkback_reuse/word_window.rs) recovers and
validates their parity words, indexes them by odd count and ternary residue,
and adds their dip credits after the window has accumulated enough odd steps.
It also requires a positive partner for the least window representative.
The construction and cutoff condition are explained in
[ALGORITHM.md](ALGORITHM.md#harvesting-longer-inverse-words).

On **10 million uniform B = 35 patterns**, adding the dictionary increased
the maximum dip for 22,655 patterns over pm∞ + mix13, or 3,180 over pm∞ + mix16.
At the N = 72, A = 10 class cutoffs, some of those increases clear additional bits:

| Endpoint odd count F | Window B | Extra bits cleared / previously kept, over mix13 | Extra bits cleared / previously kept, over mix16 |
|---|---:|---:|---:|
| 40 | 32 | 1,022 / 267,948 (0.381%) | 230 / 267,059 (0.0861%) |
| 41 | 35 | 2,254 / 702,470 (0.321%) | 496 / 700,358 (0.0708%) |
| 42 | 35 | 2,043 / 1,546,020 (0.132%) | 304 / 1,543,337 (0.0197%) |

These percentages describe **uniform bitvector patterns within each class**,
not target endpoint distributions or GPU-time savings. The table uses the
recommended 24 GB windows, with a separate 10-million-pattern sample at B = 32.
The dictionary was mined from checkpoint classes, not
optimized for window filtering. It is evidence of additional pruning, not an
end-to-end performance result. Checkpoint and window savings can overlap.

The probe verifies 42,507 translated word mergers and records a cutoff-changing
example for each run, with four translated window starts. The independent
[Python auditor](../scripts/walkback_reuse/verify_word_windows.py) checks the
example words and translated orbits using arbitrary-precision arithmetic. Raw counts,
seeds, examples, and audit results are in
[results/walkback_reuse/word-window](../results/walkback_reuse/word-window/).
This does not replace a complete exclusion-witness audit and host/device
comparison before production integration.

The implementation opportunity is to compile the sparse dictionary into the
arrays once. Final bitvector dimensions and verification lookups would be
unchanged; construction takes additional work and memory. An existing option
provides a useful comparison without implementing a new dictionary loader:
**`--mix-j 13 --bv-mix-j 16`** keeps the small table in the CPU tree while using
the larger dense family only for lookahead. The latter adds about 0.5 GB of
table storage in the builder; current CUDA code retains that allocation for
the context's lifetime. The dictionary still adds pruning over mix16, but its
increment is smaller.

With the probe package built using the command below, reproduce the B = 35
comparison from the repository root:

```sh
/tmp/collatz-reuse-target/release/word_window 10000000 35 13 > /tmp/words-b35-mix13.json
/tmp/collatz-reuse-target/release/word_window 10000000 35 16 > /tmp/words-b35-mix16.json
/tmp/collatz-reuse-target/release/word_window 10000000 32 13 > /tmp/words-b32-mix13.json
/tmp/collatz-reuse-target/release/word_window 10000000 32 16 > /tmp/words-b32-mix16.json
python3 scripts/walkback_reuse/verify_word_windows.py
```

The Python command audits the saved repository results by default. The prototype
does not alter `cv` or the CUDA builder.

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

### Expanding a depth-38 frontier

The [expansion benchmark](../scripts/expansion_bench/main.rs) compares exact
CPU traversals below sampled checkpoint survivors. The initial experiments change
how the existing rules are evaluated, so every variant must produce the same
endpoint nodes. They do not add exclusions or change A, B, or M = N − A.
The later cutoff experiment deliberately admits additional descendants and
measures their downstream cost. Production traversal remains unchanged.

The first comparison used 8,192 roots, seven repetitions, and pm∞ + mix13.
These are median wall times on a Ryzen 7700X, with eight workers restricted to
eight physical cores (CPUs 0–7) and variant order rotated between repetitions:

| Expansion implementation | Depth 38 → 50 | Depth 50 → 62 |
|---|---:|---:|
| Existing DFS, without per-depth statistics | 74.1 ms | 80.6 ms |
| Necessary slope gate before the mix lookup | 107.4 ms | 113.7 ms |
| Gate plus baseline checks before wide child construction | 79.2 ms | 85.1 ms |
| Previous combination plus ternary screening of pm∞ | 75.0 ms | 78.1 ms |
| Precompiled baseline prefixes, blocks of four levels | 102.4 ms | 107.9 ms |
| Precompiled baseline prefixes, blocks of six levels | 100.5 ms | 110.1 ms |

The first gate implementation regressed badly, but it also changed helper
structure and inlining directives. A controlled follow-up uses identical
traversal source with the gate enabled or disabled at compile time. Its gate
stores just one maximum eligible odd count per depth, instead of a two-dimensional
feasibility table. That version removes most of the regression but gives no
consistent win over the reference or its matched control in the short trials.

Ternary screening uses the carried `m mod 3^16` to determine `v3(m+1)` and reject
impossible pm∞ certificates before the full-width calculation. When the residue
is `3^16 − 1`, it falls back to the original calculation. The initial late-stage
median suggests a small gain, but this is **not an established speedup**: the
eight-thread ratio of separate medians is 1.032, while the median of ratios
within the same repetition is 1.002. The nine-repetition controlled follow-up
also places this combination near parity. One-thread trials are noisier.

Expanding 64 depth-38 roots all the way to M = 62 produced **240,406,574
endpoints per trial**. Five repetitions gave the following medians:

| Implementation | One worker | Eight workers |
|---|---:|---:|
| Existing DFS | 10.485 s | 1.251 s |
| Matched control, neither new screening rule | 11.044 s | 1.382 s |
| Compact mix gate | 10.742 s | 1.493 s |
| Early checks + mix gate + ternary screening | 10.296 s | 1.294 s |

The combination's apparent gain reverses between one and eight workers.
Reference times ranged from 10.211–10.763 s and 1.188–1.484 s respectively;
these five-trial medians do not establish a small improvement. Both full-span
and short-subtree results argue against adopting the combined variant yet.

A final ablation isolates **ternary screening alone** using the same source
as the matched control, with no new mix gate or early child construction.
On eight workers, 32,768 roots and nine repetitions give:

| Implementation | Depth 38 → 50 | Depth 50 → 62 |
|---|---:|---:|
| Existing DFS | 307.8 ms | 330.8 ms |
| Matched control | 311.4 ms | 334.7 ms |
| Ternary screening alone | 296.1 ms | 315.1 ms |
| Early checks + mix gate + ternary screening | 303.3 ms | 330.4 ms |

The isolated change reduces elapsed time by **3.8% and 4.7%** on these corpora.
It beats the reference in nine of nine early repetitions and seven of nine
late repetitions, and also beats its matched control. However, a full 38 → 62
comparison with a different seed, 64 roots, seven repetitions, and
232,009,458 endpoints per trial gives **1.443 s reference versus 1.431 s
ternary-only**, just **0.85% less time**. Ternary-only wins four of seven paired
repetitions; its 1.256–1.651 s range overlaps the reference's 1.294–1.575 s.
The combined variant's ratio of medians also fluctuates substantially between
the two full-span corpora. The isolated ternary test is the most promising
short-subtree result, but a stable full-expansion or production gain remains
unproven. No experimental variant is enabled in production.

The block implementations are **prefix interpreters**, not affine jumps that
eliminate intermediate wide arithmetic. They precompute baseline transitions,
then still construct values and apply value-dependent rules at every surviving
prefix. Instruction loads, temporary nodes, and dispatch appear to outweigh
the saved baseline work here; attribution would require profiling. Their
consistent 34–49% elapsed-time regression across the one/eight-thread trials
argues against these implementations on this CPU with this sink. A
genuine affine-jump experiment would need to avoid intermediate wide arithmetic
while preserving every intermediate exclusion.

Roots are sampled uniformly with replacement from the baseline frontier, then
filtered by improved-rule ancestry and capped walkback at their depth-38
ancestor. The late-stage corpus is sampled at depth 50; it is not obtained by
giving every depth-38 subtree equal weight. This represents expansion below
conditioned survivors, not loading a canonical stored archive. Setup, archive
I/O, per-depth accounting, endpoint transfers, and GPU execution are excluded.
The endpoint sink only accumulates counts and a digest. These CPU timings do
not revise the GPU-year estimates or establish a production pipeline speedup.
The first round excludes per-depth statistics; the follow-up below also tests
with the production counters enabled.

All variants pass exact sorted comparisons of every `Node` field on complete
small trees and sampled deep subtrees. Separate tests cover the mix gate's
necessary condition and 100,800 maximal-ternary-valuation fallback cases.
Each timed run also checks an endpoint count and order-independent digest.
Per-run exact vector comparisons cover the first 64 roots for up to twelve
levels; a full 38 → 62 run's remaining levels are checked by the digest, not
by storing all endpoints for an exact comparison.

Raw trials, corpus digests, validation counts, and source snapshots are in
[results/expansion_bench](../results/expansion_bench/). The runner records compiler,
binary and source hashes, affinity, and the CPU governor (`powersave` here).
Its CPU identity and affinity mapping are specific to this host; adapt them
before using it on another machine. It runs benchmark processes serially,
without changing system power settings or isolating unrelated background work.
The [summary](../results/expansion_bench/summary.json) includes both ratios of
medians and medians of within-repetition ratios, plus observed ranges. These
ranges are not confidence intervals. [Test output](../results/expansion_bench/tests.txt)
records the four passing correctness tests with the nine first-round variants.

```sh
cargo test --release --offline --manifest-path scripts/expansion_bench/Cargo.toml \
    --target-dir /tmp/collatz-expansion-target
cargo build --release --offline --manifest-path scripts/expansion_bench/Cargo.toml \
    --target-dir /tmp/collatz-expansion-target
python3 scripts/expansion_bench/run.py \
    /tmp/collatz-expansion-target/release/expansion_bench /tmp/expansion-comparison
python3 scripts/expansion_bench/run.py \
    /tmp/collatz-expansion-target/release/expansion_bench /tmp/expansion-full \
    --roots 64 --reps 5 --cases full --variants reference,control,compact,ternary
python3 scripts/expansion_bench/run.py \
    /tmp/collatz-expansion-target/release/expansion_bench /tmp/expansion-ternary \
    --roots 32768 --reps 9 --threads 8 --variants reference,control,ternary_only,ternary
python3 scripts/expansion_bench/run.py \
    /tmp/collatz-expansion-target/release/expansion_bench /tmp/expansion-ternary-full \
    --roots 64 --reps 7 --threads 8 --cases full --seed 17392 \
    --variants reference,control,ternary_only,ternary
python3 scripts/expansion_bench/summarize.py > /tmp/expansion-summary.json
```

The last command summarizes the saved repository runs by default. Each result
directory's metadata records its exact arguments; its source snapshot preserves
the experimental revision used for that comparison.

#### Further ternary screening experiments

The [untimed profiler](../scripts/expansion_bench/profile.rs) explains where the
opportunity lies. On 1,024 roots from each short-subtree corpus, the residue
screen proves slope failure for **99.724% and 99.853%** of pm∞ calls. The median minimum required
valuation is seven, but about **81%** of actual valuations are at most two.
The original residue screen replaces roughly 2.4 wide divisibility tests per
call with roughly 2.4 narrow ones. Very few calls reach the second, wide
calculation, so optimizing that remaining calculation has limited scope.

The better screening question is whether the required power of three divides
`m+1`, without first finding its exact valuation. Precompute

```text
t_min(k,f) = min { t in 1..=f : 2^t * 3^(f-t) <= 2^k }.
```

An empty set means the pm∞ slope cannot pass and its check can be skipped.
For `t_min <= 16`, test divisibility of `m3+1` by `3^t_min`, where
`m3 = m mod 3^16`. The implementation uses a precomputed 32-bit modular
inverse and one multiply/compare instead of variable division. For larger
thresholds, only the saturated residue can remain possible. Eligible or
unresolved nodes still run the original full-width predicate; mixed-word
and walkback rules run in their original order. The
[implementation](../scripts/expansion_bench/ternary.rs) also compares fused
quotient calculation and a 6,561-byte valuation lookup.

There are two exact forms of [reuse](../scripts/expansion_bench/reuse.rs).
For the clipped valuation `t = min(v3(m+1),f)`, an odd child has `t' = t+1`,
including a high-bit lift. Both `2^t * 3^(f-t)` and `2^k` double. A proven
failed slope test can therefore be inherited along that edge.
The implementation preserves unresolved saturated cases and handles `t=0`
using a separate zero-inclusive threshold. An odd child that adds no starting
bit also inherits its surviving parent's failed pm∞ test, since its inverse
partner is unchanged. If the parent has `t=0`, the child's partner is the
parent's iterate `m >= n0`, so it still cannot cull. That second shortcut
requires an already-surviving
canonical root, as supplied by this harness and the production DFS contract.

The counter-enabled comparison exposed an implementation issue: passing a
whole `Node` to an outlined extra-rule helper forced additional child state
onto the stack. Even the unchanged-rule control became 26% slower than the
reference. The repaired variants pass scalar fields, matching the original
helper interface. All earlier timings and source versions remain saved;
the `ternary-scalar-*` directories contain the repaired comparisons.

With eight workers, counters enabled, 32,768 roots and nine repetitions,
the repaired short-subtree comparisons give these median times:

| pm∞ implementation | Depth 38 → 50 | Depth 50 → 62 |
|---|---:|---:|
| Original reference | 319.9 ms | 342.2 ms |
| Unchanged-rule control | 321.9 ms | 342.7 ms |
| Original residue screen | 324.1 ms | 339.5 ms |
| Fused valuation and quotient | 308.0 ms | 330.3 ms |
| Minimum-required-power divisibility gate | **299.3 ms** | **321.1 ms** |
| Small valuation lookup with fused quotient | 305.2 ms | 325.0 ms |
| Carry the capped valuation between nodes | 314.8 ms | 337.4 ms |
| Inherit a failed slope test along odd edges | 306.4 ms | 331.3 ms |
| Skip the unchanged-start odd child's pm∞ test | 301.4 ms | 322.4 ms |

The simple divisibility gate saves **6.45% and 6.18%** elapsed time here.
Explicit reuse is mathematically sound, but its extra state and branching
do not improve on that simpler variant in these measurements. The matched
control is within 0.7% of the reference, and their generated instruction
sequences agree apart from relocations and cold panic metadata. The saved
[assembly comparison](../results/expansion_bench/ternary-scalar-assembly.json)
also records the restored stack sizes; it is diagnostic evidence, not a
hardware-counter attribution of the entire timing change.

The final full-span comparisons use **256 depth-38 roots, nine repetitions,
eight workers, and production per-depth counters**, with a different root
sample for each seed. Each variant must reproduce the same counters and
endpoint digest. This measures CPU expansion with the digest sink; GPU work,
endpoint packing, transfers, and queue backpressure remain unmeasured.

| Sample seed | Endpoints per trial | Reference median | Control median | Divisibility gate median | Elapsed reduction vs reference |
|---|---:|---:|---:|---:|---:|
| 17393 | 902,846,623 | 5.021 s | 4.985 s | **4.479 s** | **10.79%** |
| 17394 | 1,113,119,517 | 5.973 s | 5.967 s | **5.500 s** | **7.91%** |

The gate wins nine of nine paired repetitions in the first sample and eight
of nine in the second. Median within-repetition speedup ratios are 1.121 and
1.078, consistent with the ratios of medians. Unlike the original small
screening gain, this improvement persists over two much larger full-span
samples with counters enabled. The **minimum-required-power divisibility
gate is the preferred candidate when preserving the full pruning policy**. The measured
claim is roughly **8–11% less CPU expansion time in these configurations**;
it is not a GPU or end-to-end verification speedup. Explicitly carrying a
valuation or a failed-slope flag did not improve on it in the short-subtree
comparisons and was not selected for these final full-span trials.

To reproduce these follow-ups after building the experimental package:

```sh
python3 scripts/expansion_bench/run.py \
    /tmp/collatz-expansion-target/release/expansion_bench /tmp/ternary-profile \
    --roots 32768 --reps 9 --threads 8 --profile-roots 1024 \
    --variants reference,pm_control,pm_screen,pm_fused,pm_threshold,pm_lookup,pm_carry,pm_reuse,pm_bit0 \
    --stats
python3 scripts/expansion_bench/run.py \
    /tmp/collatz-expansion-target/release/expansion_bench /tmp/ternary-full-a \
    --roots 256 --reps 9 --threads 8 --cases full --seed 17393 \
    --variants reference,pm_control,pm_threshold --stats
python3 scripts/expansion_bench/run.py \
    /tmp/collatz-expansion-target/release/expansion_bench /tmp/ternary-full-b \
    --roots 256 --reps 9 --threads 8 --cases full --seed 17394 \
    --variants reference,pm_control,pm_threshold --stats
```

The [eight passing tests](../results/expansion_bench/ternary-scalar-tests.txt)
cover all 17 experimental/reference variants, independent ordinary-division
checks at arithmetic boundaries, odd-edge identities, and exact per-depth
generated/survived/cull-category counts for the counter-capable variants.
Timed counter-enabled runs require those vectors and endpoint digests to
match as well. The full-span digest and twelve-level exact-vector limitations
from the first round still apply. Production `src/` remains unchanged.

#### Discontinuing extra checks below the checkpoint

Dropping pm∞ and mixed-word checks after a stored checkpoint is sound: it
admits additional descendants for verification while retaining every proof
that removed a checkpoint class. The inherited OEE and trailing-even states
must be preserved. The cheap descent, pm1 and OEE rules continue throughout
expansion. The expensive bounded walkback remains a checkpoint-only operation.

This also permits a simpler representation below the cutoff. The
[baseline tail prototype](../scripts/expansion_bench/fast_tail.rs) carries
`n0`, a wrapping 64-bit iterate, and the parity counters, dropping the
full-width iterate and ternary residue. Each step loses at most one known
low bit of the iterate; from K38 to M62 there are only 24 steps, so all parity
decisions remain exact. Through depth 64 the rational slope predicate agrees
with exact integer powers. If it fails to prove descent, the nonnegative
affine intercept ensures `m >= n0`, making the separate value comparison
redundant. The prototype emits the fields needed for GPU dispatch; it does
not claim to reconstruct a complete `Node`.

The [policy benchmark](../scripts/expansion_bench/tail_bench.rs) uses the
**same improved13 plus capped-walkback K38 roots for every variant**.
It compares full pruning, the divisibility-screened version, baseline-only
expansion, pm∞-only expansion, and later cutoffs. Root selection samples the
checkpoint survivor distribution; it does not read a stored frontier archive.
Counters remain enabled, trials rotate their order, and the sink digests the
same dispatch fields and counts endpoints by F. Policies with different
pruning intentionally produce different endpoint sets. Repeated outcomes
must agree within each policy; equivalent implementations must also match
every per-depth counter.

On the Ryzen 7 7700X with eight physical cores, the first 128-root sample
(seed 17395, five repetitions) gave these medians:

| Expansion after K38 | CPU time | Reduction vs screened full checks | Additional endpoints in this sample |
|---|---:|---:|---:|
| Original full checks | 3.001 s | — | 0% |
| Screened full checks | 2.651 s | — | 0% |
| Original DFS, baseline rules only | 1.463 s | 44.80% | 4.09% |
| Compact baseline tail | 0.861 s | 67.51% | 4.09% |
| pm∞ only, screened | 1.885 s | 28.89% | 3.16% |

Keeping pm∞ alone is a less attractive CPU tradeoff in this comparison.
The [follow-up cutoff sweep](../results/checkpoint_tail/fast-cutoffs/cpu.jsonl)
keeps full checks through a later depth, then switches once into the compact
baseline loop. On the same roots, screened full checks took 2.774 s; cutoffs
38/46/50/54/58 took 0.857/0.849/0.867/0.878/1.100 s. Thus keeping full checks
through depth 54 costs little CPU time here while retaining more pruning.
These are within-run comparisons; the two runs' control times differ.

An independent **256-root sample, seven repetitions, seed 17396** confirms
the result. Its full-check policy produced 1,135,372,311 endpoints per trial:

| Expansion policy | Median CPU time | Reduction vs screened full checks |
|---|---:|---:|
| Original full checks | 6.129 s | — |
| Screened full checks | 5.523 s | — |
| Original DFS, baseline after K38 | 3.030 s | 45.14% |
| Compact baseline after K38 | 1.730 s | 68.67% |
| Compact baseline after depth 50 | 1.722 s | 68.82% |
| Compact baseline after depth 54 | 1.792 s | 67.56% |
| Compact baseline after depth 58 | 2.140 s | 61.26% |

All five weakened-policy variants beat screened full checks in all seven
paired repetitions. This supports roughly **45% less CPU expansion time
from changing the rules alone**, or **68–69% less with the specialized
baseline suffix**, for these configurations. It does not establish a
universal optimal cutoff; the close timings around depths 38–54 should not
be treated as a precise ranking.

The additional endpoints do not have average GPU cost. A separate
[paired work probe](../scripts/expansion_bench/checkpoint_work.rs) sampled
30,000 baseline M62 endpoints within each of the 16 GPU F classes: **480,000
draws total**, seed 17397. It reconstructs each path, applies the common
K38 checkpoint policy, and evaluates both retained and newly admitted
endpoints with the unchanged shared pm∞+mix13 windows. Exact baseline class
populations supply the weights. The N72/A10 window sizes are
`32,35,35,35,35,35,32,32,27,27,27,27,27,27,27,27`.

| Last depth with full extra checks | Additional endpoints | Additional tail iterations | Additional modeled GPU time |
|---|---:|---:|---:|
| 38 | 4.55% | 1.65% | **2.05%** |
| 46 | 3.60% | 1.09% | **1.43%** |
| 50 | 3.00% | 0.81% | **1.11%** |
| 54 | 2.35% | 0.56% | **0.80%** |
| 58 | 1.35% | 0.29% | **0.43%** |
| 62 | 0% | 0% | 0% |

Keeping only pm∞ after K38 adds 3.50% endpoints and 1.65% modeled GPU time.
All percentages use the full-check survivors with the **same checkpoint**
as denominator, estimated on the same draws. They are not divided by the
older model's unconditioned endpoint population. The GPU model includes
per-endpoint overhead, surviving cases and tail iterations, using the existing
RTX 4090 fit. Sampling standard errors for the modeled percentages above
are respectively 0.031/0.025/0.021/0.017/0.012 percentage points; they exclude
model error. **No GPU was timed.** The probe covers F40–55; the omitted
F56–62 baseline population is 49,063,601, about 1.12e-8 of all baseline
endpoints. Those endpoints require the production CPU fallback, whose cost
is not included here.

The enhanced bitvectors remain valid independently of CPU extra-pruning
ancestry. Their credits use the within-window parity pattern and endpoint
slope; the existing [witness audits](../tests/bv_audit.rs) already exercise
enhanced windows on baseline-only endpoints. Dropping tree checks does not
require discarding those reusable window credits.

This is a CPU/GPU resource tradeoff. If CPU expansion takes C and GPU work
takes G with full checks, an ideally overlapped pipeline takes approximately
`max(C,G)`. A cutoff with CPU ratio r and GPU ratio s instead takes
`max(r*C,s*G)`. It helps elapsed time when CPU expansion is the bottleneck
and `C/G > s`; a GPU-bound run instead pays the additional GPU work. Depth
50–54 is a promising compromise here, retaining roughly two-thirds less
CPU expansion time for about 0.8–1.1% modeled additional GPU time. The
optimum depends on available CPU capacity and requires pipeline timing.

The [saved results and reproduction commands](../results/checkpoint_tail/README.md)
include source snapshots, binary hashes, all trials and a
[machine-readable summary](../results/checkpoint_tail/summary.json).
There are 18 passing experimental tests across the three binaries, including
exact small-tree policy enumeration, wrapped-iterate boundary cases, matching
payloads/counters, and paired sampling arithmetic. Timings exclude archive
I/O, endpoint packing, queue backpressure and transfers. Integrating the
compact representation also requires reconstructing complete iterates for
the rare endpoints sent to CPU fallback. Production `src/` remains unchanged.

#### Direct enumeration of terminal bit blocks

The [terminal-block prototype](../scripts/expansion_bench/terminal.rs) turns
the final baseline-only levels into a reusable table indexed by incoming
`(k, f, oee, ne)`. It compiles parity words and tests every intermediate
descent, pm1, and OEE condition once per state. At runtime it emits surviving
endpoints directly. This shares work across different numeric residues,
beyond the prefix sharing already provided by DFS.

Starting-number extension bits and trajectory parity bits are different.
For a W-step parity word, let `r` be its input residue modulo `2^W`. At a
depth-k class with iterate m and odd count f, extension q starts that word
exactly when `m + 3^f*q = r (mod 2^W)`. Thus:

```text
R = r * 3^(-f) mod 2^W       # stored in the table
D = m * 3^(-f) mod 2^W       # computed once for the incoming class
q = (R - D) mod 2^W
n0_extended = n0 + q * 2^k
```

Only the low W iterate bits are needed at the boundary. Emission performs
one modular multiplication per incoming class and a subtraction, mask, and
shift per endpoint; it does not reconstruct intermediate or final iterates.
Tables also store exact generated, survived, and cull-category counts at
every intermediate depth, including prefixes that emit no endpoints. Those
counts are merged once per incoming class, preserving production statistics.

The [benchmark](../scripts/expansion_bench/terminal_bench.rs) keeps **one
fixed pruning policy per run**: identical improved13 plus capped-walkback
K38 roots, full extra checks through a chosen cutoff, and baseline rules
thereafter to M62. `compact` is the previous specialized baseline DFS;
`production` uses original full-width `Node` traversal under the same cutoff
policy. Neither control continues extra pruning past the cutoff. `flatW`
stores terminal words in one array; `groupW` groups them by final
`(f, oee, ne)`. All variants preserve the same endpoint set and every
per-depth counter. The full-check cutoff stays fixed while table width
changes, so width improvements cannot be explained by dropping more checks.

The initial **128-root sample, five repetitions, seed 17411**, used cutoff
50 on eight physical Ryzen 7 7700X cores. It emitted 455,980,448 endpoints
per trial. Both sinks include counters and the validation digest; `packed`
also writes four `u32` words per endpoint into F-specific buffers, exposing
each 4,096-endpoint chunk to `black_box`. Median CPU times were:

| Implementation | Digest sink | Packed sink | Shared table bytes |
|---|---:|---:|---:|
| Compact baseline DFS | 0.71851 s | 0.75145 s | — |
| Flat, 4 bits | 0.15157 s | 0.26414 s | 102,600 |
| Flat, 6 bits | 0.08708 s | 0.20725 s | 166,480 |
| Flat, 8 bits | 0.06663 s | 0.18219 s | 317,768 |
| Flat, 10 bits | 0.06144 s | 0.17859 s | 811,656 |
| Flat, 12 bits | 0.05924 s | 0.16909 s | 2,582,072 |

The original full-width control took 1.26472 s with the digest and 1.32921 s
with packing. The table sizes include **both experimental layouts**, counters,
program slots, and container headers, excluding allocator overhead. All
workers share them. Setup with all five tables took 0.0371 s in the digest
sweep, including rule setup and excluded from expansion timing; root sampling
and archive I/O are separate. A production implementation retaining one
layout would not require both copies.

An independent **256-root sample, seven repetitions, seed 17412**, produced
**1,085,114,524 endpoints per trial** at cutoff 50:

| Sink | Compact DFS | Flat 10-bit table | Flat 12-bit table | CPU time reduction vs compact |
|---|---:|---:|---:|---:|
| Digest | 1.61134 s | 0.13715 s | 0.13062 s | 91.49–91.89% |
| Packed | 1.65550 s | 0.37997 s | 0.38828 s | 76.55–77.05% |

Both widths beat compact DFS in all seven paired repetitions with each sink.
The small 10/12-bit ranking reverses in the packed confirmation; the evidence
supports a large direct-enumeration gain, not a universally best 12-bit
width. Grouping by final state did not consistently improve on the flat
layout. All grouped results and within-repetition ranges are retained in the
[machine-readable summary](../results/terminal_blocks/summary.json).

Later cutoffs also benefit, with shorter terminal blocks. On the same
256-root corpus and packed sink:

| Full checks through | Table | Matched compact control | Table implementation | CPU time reduction |
|---|---|---:|---:|---:|
| 50 | Flat 10 bits | 1.65550 s | 0.37997 s | 77.05% |
| 54 | Flat 8 bits | 1.68152 s | 0.43784 s | 73.96% |
| 58 | Flat 4 bits | 2.14556 s | 1.04117 s | 51.47% |

Each row compares implementations of its own pruning policy; different
cutoffs have different endpoint sets. Tables add no new pruning and no new
endpoints. The earlier paired work probe's estimates of **1.11%, 0.80%, and
0.43% additional GPU time** for cutoffs 50, 54, and 58 relative to full checks
therefore remain the relevant modeled work tradeoffs. They were not
remeasured here, and the CPU savings above are relative to compact DFS,
not to full checks or to total verification time.

[Assembly inspection](../results/terminal_blocks/assembly-notes.json) verifies
that optimized code retains a per-residue reconstruction loop and all four
payload stores in the packed sink. The compiler aggregates some grouped
digest updates, while precomputed tree counters intentionally replace
per-node accounting. These timings include gains from both endpoint
enumeration and accounting; they do not isolate their separate contributions.

There are [eight new tests](../results/terminal_blocks/tests.txt), including
independent suffix enumeration, modular-mapping bijection, rejected-prefix
counts, wide iterates and bit 63, both layouts, and enabled/disabled counters.
The terminal binary passes 15 tests including seven reused tail tests; the
whole experimental package records **33 successful executions covering 26
distinct tests**. Each benchmark additionally audits exact endpoint payloads
against original traversal on 64 sampled roots and checks every timed
variant's digest and per-depth counters.

These are CPU experiments on sampled frontiers, not archive replay or GPU
timings. The packed sink includes its validation digest and buffers every
F class; production instead processes classes without a suitable GPU array
through its CPU fallback. Queueing, locks, transfers, GPU execution, and
backpressure are excluded. Endpoint ordering changes, so actual device timing
is still needed to determine its effect on GPU execution. The next integration
step needs production dispatch and the rare CPU-fallback path. Production
`src/` remains unchanged. The
[saved results and reproduction commands](../results/terminal_blocks/README.md)
include all six runs, table sizes, source snapshots, and binary hashes.

#### Integrated GPU pipeline

The [overall benchmark](../results/overall_pipeline/README.md) now connects the
screened prefix, compact suffix and terminal tables to the real GPU verifier.
It includes production packing, queues, transfers, kernels and CPU fallback,
with shared pm∞+mix13 lookahead arrays retained across trials.

On a Ryzen 7 7700X and RTX 4090, three rotating repetitions expanded the same
1,024 saved K38 packages at **N72/A10/M62**, emitting 4.20–4.33 billion endpoints
per trial. With eight physical CPU cores, median results were:

| Expansion | Pipeline wall | Process CPU time |
|---|---:|---:|
| Original full checks | 28.955 s | 213.846 core-s |
| Screened full checks | 29.973 s | 212.361 core-s |
| Screened through K54, compact suffix | 26.379 s | 74.877 core-s |
| Screened through K54, terminal8 | 26.441 s | 31.513 core-s |
| Screened through K50, compact suffix | 26.463 s | 73.271 core-s |
| Screened through K50, terminal10 | 26.549 s | 27.803 core-s |

The combined K50 path uses **87% less process CPU time** and **8.3% less median
elapsed time** than full checks. Terminal tables reduce CPU use substantially
even against matched compact controls, but provide little further wall-time
benefit once the GPU is fed. Ternary screening alone does not establish an
integrated speedup here. Full-check timings varied from 27.766 to 31.491 seconds;
combined K50 timings varied from 26.240 to 26.562 seconds on the active desktop.

A second run pins the entire process to **two physical cores**, with two
producer workers, on the identical corpus. Median times remain **26.517 seconds
for K54/terminal8** and **26.535 seconds for K50/terminal10**. Process CPU use is
30.491 and 29.605 core-seconds respectively. Counts, tree statistics and hashes
match the eight-core runs. The main practical gain is feeding this GPU with a
much smaller CPU allocation, rather than a comparable reduction in GPU-years.

Cutoff54 adds 0.537% actual tail iterations and cutoff50 adds 0.785% on this
corpus. Counts, every tree counter and checksums agree exactly within each
pruning policy. Twelve complete N32 CPU/GPU comparisons also pass, including
checkpoint walkback, host-built and device-built arrays, and forced CPU fallback
from reconstructed compact endpoints. No trajectories were flagged.

This comparison starts from the same walkback-filtered K38 checkpoint for every
variant. Checkpoint preparation, archive I/O and one-time setup are excluded
from trial times. GPU initialization took 89.745 seconds; the arrays occupy
19.625 GiB after reducing one B35 class to B34 to fit alongside the desktop.
These are sampled N72 packages, not a complete 2^72 run or a new RTX PRO 6000
measurement. CUDA event durations sum overlapping streams and must not be read
as GPU busy wall time. The [raw results](../results/overall_pipeline/summary.json)
retain timing ranges, CPU accounting, telemetry and source provenance.

The new `Expander`/compact-sink hooks and reusable `GpuSession` expose the real
pipeline to the experimental engine. Default `cv verify` still uses original
DFS. Harvested reverse-word lookahead credits remain a separate prototype.

#### Moving more expansion onto the CPU

An [A8/A9/A10 pipeline comparison](../results/a_pipeline/README.md) tests whether
the saved CPU time can buy less GPU time. It reuses the same 1,024 K38 packages,
full-check cutoff K50 and terminal entry K52. Changing A from 10 to 9 to 8
increases endpoint depth M from 62 to 63 to 64 and terminal width from 10 to 11
to 12. Eight physical CPU cores and the same 19.625 GiB relative-class window
allocation are used throughout.

| A / M | Pipeline wall, median [min, max] | Process CPU, median | Average busy cores | Tail iterations vs A10 |
|---|---:|---:|---:|---:|
| 10 / 62 | 25.367 [25.093, 25.507] s | 27.970 core-s | 1.102 | reference |
| 9 / 63 | 25.448 [25.085, 25.891] s | 51.105 core-s | 2.008 | −4.23% |
| 8 / 64 | 27.539 [27.516, 27.609] s | 97.871 core-s | 3.550 | −8.11% |

A10 pools three opening and three closing trials; the other configurations
have three trials each. Fresh A10 control medians are 25.108 and 25.443 seconds.
A9's +0.32% difference falls within the observed timing variation, while A8
takes 8.56% longer. The CPU has enough capacity, but **neither smaller A
demonstrates a pipeline improvement with these windows**. A10 remains the
practical choice for this experimental path.

Fewer tail iterations do not capture the whole GPU workload. A9 emits 1.906×
and A8 3.663× as many GPU endpoints, increasing transfer volume from 69.211 GB
to 131.893 and 253.508 GB. The kernel also replays all M steps to reconstruct
each endpoint's iterate before lookahead filtering. Those replay steps, absent
from the tail counter, increase 1.936× and 3.781×. These measurements do not
isolate how much of the wall time comes from replay, transfers, dispatch or
window behavior. Reducing those endpoint costs is a better next experiment
than simply deepening CPU expansion again.

The common window vector is indexed relative to minimum F, which shifts from
40 to 41 at A8. It preserves the memory budget, not the allocation by absolute
F; this comparison does not establish A8's optimum after window retuning.
Setup is excluded (85.2–86.7 seconds per block). All runs share one binary and
ordered corpus, with identical repeated work signatures within each A and no
flagged trajectories. Eighteen complete N32 and nine sampled N72 CPU/GPU
comparisons pass, including M64 bit 63, iterates above `u64`, and CPU fallback.
The [saved results](../results/a_pipeline/summary.json) retain all trials,
accounting and provenance; they do not replace the full-range GPU-year model.

#### Preserving terminal reuse across the GPU boundary

The [transport experiments](../results/transport_pipeline/README.md) compare
five ways to process the same N72/A10 corpus, with the same windows, resident
arrays, three CUDA streams and eight physical CPU cores. All variants perform
identical tree and tail work, including per-class counts and checksums.

| Variant | Pipeline wall, median [min, max] | Process CPU, median | Transfer volume |
|---|---:|---:|---:|
| Original 16-byte replay | 24.819 [24.499, 24.890] s | 28.943 core-s | 69.211 GB |
| Packed 8-byte replay | 24.720 [24.646, 24.830] s | 25.640 core-s | 34.606 GB |
| CPU affine reconstruction, 16-byte replay | 24.939 [24.797, 25.166] s | 49.464 core-s | 69.211 GB |
| CPU affine reconstruction, 32-byte exact iterate | 24.737 [24.415, 25.007] s | 59.259 core-s | 138.423 GB |
| Parent/class descriptors with GPU expansion | 24.414 [24.091, 24.582] s | 6.543 core-s | 2.955 GB |

The strongest measured benefit is **less CPU work and traffic**. Parent
dispatch cuts process CPU by 77.4% and transfers by 95.7%; its median wall
time is only 1.6% lower, with overlapping observed ranges. It sends exact K52
parents and ranges of resident ten-bit terminal words, grouped by final F.
A GPU expansion kernel reconstructs exact M62 endpoints before verification.
The transfer accounting includes repeated 64-byte class headers and batch
splits. The terminal words are uploaded once before timing.

Packing eight-byte endpoints halves traffic and reduces CPU time by 11.4%,
without a clear wall-time gain. Computing exact iterates on the CPU and sending
them doubles traffic and roughly doubles CPU time, also without a clear wall
improvement. The `affine16` control shows the extra host reconstruction cost;
avoiding GPU prefix replay alone does not justify that cost here.

A second run pins the entire process to one physical core. Parent dispatch
takes **24.315 [21.952, 24.465] seconds**, versus
**26.931 [26.058, 27.035] seconds** for 16-byte replay. Process CPU is 2.130
versus 24.721 core-seconds. Parent dispatch retains the throughput measured
with eight cores; it also uses less CPU with one worker, consistent with
reduced producer synchronization overhead. Both variants are faster in the
last repetition, so the observed ranges should accompany the medians. The
same ordered corpus and full work signatures match across both runs. This
establishes a useful host-allocation saving for one GPU, without extrapolating
the 0.088 average busy-core figure into unmeasured multi-GPU capacity.

These are three rotating repetitions per variant on the same saved 1,024
packages, not a full N72 verification. All variants share the experimental
larger device buffers for a controlled comparison. The transport
implementations pass 22 CPU/GPU comparisons, including split arrays, M64,
wide iterates and CPU fallback. Production `cv verify` remains unchanged.

#### Selective extra bits for the lowest F class

The [class-count probe](../results/transport_pipeline/CLASS_SPLIT.md) finds
that original F40 contains nearly all the useful baseline pruning from one
or two more bits. Expanding F40 alone captures 98.48% or 95.46% of the
all-class candidate reduction, with total endpoint factors of 1.136× or
1.365× rather than 1.906× or 3.663×. These counts precede lookahead screening.

A [GPU follow-up](../results/transport_pipeline/class_gpu_large_summary.md)
samples 16,777,216 conditional M62/F40 endpoints and compares serialized
replay-kernel cost with seven repetitions and shared B32/B35 relative-class
windows. A10 takes 20.890 ms, A9 19.781 ms (**5.31% less**), and A8 22.382 ms
(**7.14% more**). A closing A10 control takes 20.894 ms, just 0.02% above the
opening control. A smaller initial sample found the same direction. One extra
bit is more promising than two: both reduce counted tail iterations by about
21%, but two bits generate substantially more endpoints to reconstruct and
screen.

This is a different conditional corpus, with improved13 checks through K50
and no K38 walkback; it measures the original replay kernel, not parent
dispatch. It excludes CPU expansion, transfers and setup. A mixed-depth
pipeline remains unmeasured, and the extra depth-dependent arrays would add
roughly 4 GiB net to the current 19.625 GiB resident layout. Window retuning
or staging is needed on this desktop's 24 GiB GPU. The 5.31% class-kernel
gain is therefore not an overall throughput claim.

### Full-range checkpoint estimate and kernel tuning

The [full-range follow-up](../results/fullrange_pipeline/README.md) replaces
fixed accepted-root packages with 16 independent cohorts of 512 baseline K38
draws. Improved13 ancestry and bounded K38 walkback reject 468 draws; all 8,192
draws remain in the denominator. Each accepted subtree is expanded with extra
checks through K50 and terminal parent dispatch from K52 to M62. Two rotating
timing repetitions compare identical work on an RTX 4090 with two physical
7700X cores available and 19.625 GiB of shared windows.

The original parent kernel estimates **0.6535 GPU-years [0.6299, 0.6771]**;
removing its redundant warp overflow vote and moving the carried OEE filter
before candidate extraction gives **0.5859 [0.5649, 0.6069]**. The paired
saving is **10.34% [10.27%, 10.42%]**, about **593 GPU-hours**. These are
approximate 95% cohort intervals. The optimization preserves cases, iterations,
checksums and overflow handling; it makes the same work cheaper. All 32 paired
timings improve. Concatenating the accepted corpus gives aggregate times
within 0.08% of summed cohort means, with exact matching aggregate counts and
hashes. This is an extrapolation, not a completed N72 verification.

Pilot alternatives are less promising: a read-only power table costs 6.5%
more than the original kernel, 128- and 256-thread blocks are effectively tied,
A9 is within 1% of A10, and A8 takes 5.4% longer. A tested A8 window remapping
takes 6.7% longer than A10. The A comparisons use the same saved pilot roots
but cross separate runs. They do not establish optimal windows for every A.

The tuned CPU demand extrapolates to **0.0681 core-years**, measured as process
CPU seconds without an SMT conversion. Eight naturally overflowing starts in
six cohorts all complete wide rechecks inside the timed pipeline. A separate
65,536-endpoint conditional probe estimates **1.37 CPU-hours** for every
baseline F56–62 endpoint, deliberately omitting extra screening. Even a fully
serialized idle-GPU allowance for those tails is only 0.000156 GPU-years.
The sample does not construct the full K38 frontier; preprocessing, archive
I/O, resident-array setup and distributed coordination are excluded. The
cohort intervals also exclude long-run device and deployment effects.

Validation includes 34 CPU/GPU comparisons, including forced overflow and
M64 inputs, plus exact work agreement throughout the confirmation. The first
confirmation attempt exposed a benchmark assertion that incorrectly rejected
successfully rechecked flags; it was fixed and the same predeclared sample was
rerun. The failed artifact is retained. Default production settings remain
unchanged; tuning and parent dispatch are opt-in experimental paths.

## Chained low-bit transitions

The [grouped-transition experiments](../results/grouped_transitions/README.md)
chain exact affine tables across K50→54→58→62 or K50→56→62, regrouping at each
boundary by parity-rule state and the current low-bit translation. This shares
real work: final-block extension translations fall by **99.83%** for four-bit
blocks and **98.14%** for six-bit blocks in a 32-root structural census. All
distinct starting values and high iterate bits are retained.

The paired pilot reduces process CPU time by about **20%** relative to the
same block chains without grouping. It does not beat the existing K52
parent-descriptor pipeline. These prototypes emit exact 32-byte endpoints,
transferring about **47 times** as much data as the compressed parent path.
Sorting and intermediate frontier construction also remain in the timed scope.
Thus the arithmetic-sharing result does not establish an overall GPU saving.

GPU tail experiments reindex a low-bit affine table after each exact jump,
retaining every original checksum point and conservatively guarding intermediate
cutoff peaks and descent. Widths 8/10/12/14, a combined CPU/GPU path, and a
follow-up with aligned vector loads and an unrolled checksum loop all preserve
the reference work. They are slower than ordinary stepping in the measured
pipeline. The saved report contains paired timings, a continuous-run check,
source snapshots, and validation including forced-overflow origins.

The [compact-dispatch follow-up](../results/compact_chain/README.md) below
keeps parent/program descriptions through device expansion. Merely dispatching
the final short block as parents would still produce more host parents than
the current K52 design. These results leave the confirmed **0.586 RTX
4090-year** estimate unchanged.

## Compact chains through GPU dispatch

The new experiment dispatches exact K50 parents with references to two resident
short transition tables. A GPU block reconstructs first-stage prefixes in
shared memory and reuses them for second-stage descendants. The CPU constructs
neither intermediate K54/K56 nodes nor final M62 endpoints for GPU-bound
classes. Final endpoints are still materialized on the device for the existing
verifier; prefix sharing is within a parent/final-class descriptor, not across
multiple unrelated parents.

Four fresh cohorts of 256 baseline K38 draws give **68.6% less host-to-device
traffic** and about **49% less process CPU time** for compact 6+6 expansion,
relative to current K52 parent dispatch. The elapsed-time ratio is **0.999
[0.975, 1.022]**. A flat twelve-bit K50 control has the same traffic reduction
and slightly lower CPU cost: earlier dispatch accounts for those savings.
Chaining reduces the resident device transition table from **9.54 MB to
2.76 MB**; the common 19.625 GiB lookahead windows are unchanged.

An initial continuous-run comparison appeared slower for K50 dispatch, but
the fast K52 baseline did not reproduce. Four balanced continuous repetitions
on the identical corpus average **20.219 s** for K52, **20.405 s** for flat K50,
**20.292 s** for compact 6+6, and **20.310 s** for compact 4+8. Compact chains
use about **45% less process CPU time** in these runs. Recorded GPU clocks do
not explain the timing-regime change, so small differences should not be
interpreted as established GPU savings.

A matched dispatch-cadence experiment flushes 1,024 instead of 4,096 parent
descriptors for both the existing and compact pipelines. The smaller compact
chunks improve continuous time by only 0.7% against ordinary compact dispatch;
comparisons with K52 change sign between cohort and continuous timings. This
does not establish a GPU-time saving, and default flushing remains unchanged.

Exact endpoints and per-depth counters match independent CPU expansion.
Direct arithmetic checks cover wide values, carries and near-128-bit limits;
complete N32, sampled N72, M64 and forced-overflow GPU comparisons all pass.
The [full report](../results/compact_chain/README.md) preserves the controls,
source snapshots, telemetry and raw work signatures. These comparative
samples do not replace the earlier full-range cost confirmation.

## CPU continuation of long GPU tails

The [CPU-offload experiment](../results/tail_offload/README.md) tests whether
the host capacity freed by compact dispatch can reduce GPU runtime. After a
candidate exceeds a double-iteration budget, the GPU can send its exact start
and current state to bounded CPU workers. A full device queue leaves that
candidate on the GPU. Combined counts, checksums, flags and wide rechecks
remain identical to ordinary verification.

This design does **not** save GPU time. In a fresh four-cohort comparison
covering 512 baseline K38 draws, enabling the machinery without exporting any
work costs **9.16% [8.07%, 10.25%]**. Budget 16 with 262,144 continuation slots
per launch and four CPU workers costs **10.16% more [9.13%, 11.19%]** than
ordinary K52 dispatch. Compact dispatch with budget 24 and two workers is
**9.49% slower [8.19%, 10.80%]** than its own compact control. These are
approximate paired 95% cohort intervals.

Two continuous repeats confirm the direction: ordinary K52 takes **10.199 s**
and **1.376 CPU core-s**; the four-worker offload takes **11.014 s** and
**18.749 CPU core-s**. It uses **1.70 average busy CPU cores** while removing
only **0.0965%** of GPU-origin tail iterations. Compact dispatch takes
**10.154 s** and **0.656 CPU core-s**, versus **10.986 s** and **7.740 CPU
core-s** with the larger-queue budget-24 handoff. CPU queue waits and final
drains are only milliseconds. Extra host work does not overcome the cost of
the enabled GPU path.

The initial threshold sweep explains why: only **0.0108%** of verified cases
survive beyond 32 double iterations, and accepted continuations typically
finish after another 3–4. A budget of eight reaches many more candidates but
causes heavy queue saturation and about **80% more elapsed time**. These
measurements argue against individual-tail offload, while leaving open CPU
pruning that eliminates whole candidate families. All 54 CPU/GPU validation
comparisons pass, including forced queue saturation and overflow after handoff;
813 of 985 forced-overflow flags arise on CPU continuations in the budget-one
fixture. Default settings and the confirmed **0.586 RTX 4090-year** estimate
remain unchanged.

## Early screening and grouped CPU verification

The [early-handoff experiment](../results/early_screen/README.md) lets the GPU
finish A expansion and B screening, then returns compact survivor masks to
four CPU workers before any tail stepping. Unselected launches use the
unchanged fused GPU verifier. Five CPU methods cover scalar stepping, sorting,
independent table jumps, initial shared transitions and repeated regrouping.

Two continuous timings on 512 baseline K38 draws average **9.875 s** for ordinary
GPU verification and **10.046–10.169 s** for the early-handoff variants:
**1.7–3.0% slower**. The selected batches move only **0.025–0.037%** of tail
iterations to the CPU. Returning 320 MiB takes about **12.8 ms** of overlapping
CUDA copy-event time, while host mask scans and owned-buffer copies consume
**245–260 ms of dispatch-thread CPU time**. CPU worker execution mostly
overlaps the GPU; final drains are about 0.1 ms. These different timing measures
must not be added together.

An isolated pinned-memory probe attributes much of the host-copy cost to
first touching destination pages: fresh-vector copies take **89.3 ms** per
160 MiB launch, versus **23.3 ms** with resident reused vectors. The software
mask scan adds **13.3 ms**. These are standalone medians, not measured pipeline
savings; buffer reuse has not been applied to the timed variants.

An identical-input replay tests 51 configurations with five repetitions each.
Repeated 10-bit grouping eliminates **97.96% of table lookups**, but is **19.2%
slower** than scalar stepping with the same chunk size. Regrouping consumes
23.4% of its runtime. No grouped configuration beats the fastest scalar
configuration on this conditional, first-launch fixture. Sharing a transition
description leaves per-state arithmetic, descent checks and checksums to do.

All methods preserve work, checksums, flagged origins and wide rechecks.
Validation includes complete N32, sampled N72/M64, an independent GPU mask
oracle and 985 forced-cutoff flags completed by the early CPU workers.
The saved report distinguishes varying selected class mixes from fixed-state
CPU replay. These comparative timings leave defaults and the full-range
**0.586 RTX 4090-year** estimate unchanged.

## Direct CPU ownership before GPU dispatch

The [direct-routing experiment](../results/direct_cpu/README.md) sends selected
compact tree groups to CPU workers before the GPU sees them. A worker reconstructs
the original M62 endpoints, expands them to its own depth, and applies its own
A/B policy. The GPU retains M62/A10. Admission is nonblocking: if the bounded
CPU pool is full, the unchanged group goes to the ordinary GPU sink. Only
original F40 groups are eligible in the timed runs; existing rare-F fallback
continues unchanged.

The smaller CPU A is useful. On identical conditional M62/F40–42 source
families, **A6/B20 uses 18–20% less CPU time than A10/B20**, including expansion
to M66. Expanding further to A0 loses that advantage. With a stronger B24
screen the same A change saves **8–12% by mean CPU time**, although A6 and A8
are close for F40. B24 uses 32 MiB of resident arrays per plan, shared by the
workers; construction is excluded from steady processing time.

The initial live comparison uses two tree producers and four CPU workers on
the Ryzen 7 7700X alongside the RTX 4090. Three continuous repetitions of 512
baseline K38 draws average **9.414 s** for the GPU control and **9.368 s** for
CPU A6/B24 routing: **0.49% lower elapsed time**, or 0.40% against the routing
control with no diversion. The difference is small relative to timing
variation. The 64-group queue keeps only **1.83 CPU worker cores** busy on
average, despite rejecting 97.2% of selected groups when full. This motivates
the larger-queue follow-up.

With capacity increased to 1,024 groups, six repetitions on a fresh 512-draw
corpus keep **3.94 CPU worker cores** busy. Mean elapsed times are **8.038 s**
for ordinary GPU processing, **8.018 s** for the routing control, and
**8.024 s** for CPU A6/B24: **no reliable overall speedup**. The routed median
is essentially equal to the ordinary GPU median, and comparisons cross equality
against both controls. Workers divert 21.57 million original endpoints
(**1.22%**) while removing **0.164% of GPU tail iterations**. Total process CPU
time rises from 1.318 to 32.958 core-seconds; final CPU drain remains only
0.090 ms. The larger queue makes spare cores useful to the computation but
does not establish GPU elapsed-time savings. The changed seed means these
absolute timings should not be compared with the pilot's timings.

Complete N32, sampled wide N72, and forced-cutoff checks pass. The tests compare
each CPU policy with an independent DFS/window oracle, including wide rechecks,
and audit the partition of original endpoint families between GPU and CPU.
Different A/B policies can prune different starts and change native cases,
iterations and checksums; those counters are not compared across policies as
if the work were identical. Timed runs include routing, CPU work, worker drain
and wide rechecks, with resident setup excluded. The initial GPU-forwarding
guard is not a live queue watermark. These are sampled package comparisons;
production defaults and the **0.586 RTX 4090-year** estimate remain unchanged.

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
