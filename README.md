# collatz-verify (`cv`)

A clean implementation of Collatz convergence verification for all `n < 2^N`,
using Angeltveit's least-significant-bit-first tree search with GPU bitvector
lookahead (2026). It adds per-class path-merge certificates and a fix for a
soundness gap in the reference bitvectors. It also measures what a full run costs
at any N.

* **Reproduces the reference exactly.** Angeltveit's Fig. 5.1 counts
  (756,583,624 cases at N = 40), OEIS A076227, and every k = 37 survivor count in
  `collatz-probe/cull_survivors_k1_k37.csv`.
* **Sound.** Every step-1 cull carries a merge certificate. Every bitvector
  exclusion is audited against an explicit witness. CPU and GPU agree bit for bit.
* **Exactly countable.** The baseline workload at any N is a lattice-path count
  (DP, milliseconds at N = 72). Unit costs are measured on endpoints drawn
  exactly uniformly at the target N, not extrapolated from small N.
* **Distributable.** Work packages with exactly predicted GPU cost and
  deterministic checksums, so any package can be recomputed elsewhere as a spot check.

## Headline numbers

Resources to verify every n < 2^N. GPU time is RTX 4090-years (per-operation
model fitted to the measured timings, ±5%). CPU time is Ryzen 7 7700X core-years
for step 1.

| N | Angeltveit's algorithm, sound bitvectors (A=10, B=24) | all improvements (A=10, B=28) | CPU-light (A=12, B=28) |
|---|---|---|---|
| 72 | 1.03 GPU-yr + 2.6 core-yr | **0.85** (−17%) + 4.8 | 0.90 + 1.3 |
| 73 | 1.96 + 4.9 | **1.63** + 9.1 | 1.70 + 2.5 |
| 74 | 3.78 + 9.1 | **3.05** + 17 | 3.24 + 4.8 |
| 75 | 7.05 + 17 | **5.80** + 32 | 6.21 + 9.1 |
| 76 | 13.4 + 33 | **11.1** + 61 | 11.6 + 17 |
| 80 | 176 + 423 | **147** + 786 | 156 + 214 |

The step-1 tree needs about 2.5 CPU cores per GPU for the baseline, 5.6 for all
improvements, and 1.5 for the CPU-light variant. A 1-GPU-hour work package
plan for 2^72 has 11,616 packages.

**With larger, per-class lookahead windows** (device-built arrays, shared
bitvectors), a 24 GB card does 2^72 in **0.675 RTX 4090-years** (A=10; A=12 costs
about 4% more GPU time and 3.6× less CPU), against 1.03 for Angeltveit's
configuration. That is about $3k on vast.ai 4090s on demand, or $1.7k on interruptible
instances. Renting a 288 GB B300 for even larger windows saves only another ~10% of
the work and costs ~20× more; see [docs/HARDWARE.md](docs/HARDWARE.md).

See [docs/COST_MODEL.md](docs/COST_MODEL.md) for all N from 64 to 80, the
method, and its validation. [docs/HARDWARE.md](docs/HARDWARE.md) covers lookahead
windows, VRAM, and which GPU to rent. [docs/FINDINGS.md](docs/FINDINGS.md) summarizes
the measured benefits and tradeoffs of the optimizations.

## What is new relative to the reference

| | effect at 2^72 (RTX 4090) | cost |
|---|---|---|
| **Soundness fix**: bitvectors split by trailing-even parity | 386 unjustified exclusions at N = 26 → 0; −0.18% cases | none |
| **Window credits**: pm∞ and mixed-word merges baked into the bitvectors | −3.4% cases, −4.8% GPU time | precompute only |
| **Deeper lookahead** B = 28 (32 MB arrays) | −18% cases, about −9% GPU time | 1 GB of arrays, ~40 s precompute |
| **Step-1 pm∞ + mix(3^13)** per-class certificates | −6.9% endpoints, about −4% GPU time | 2× CPU per tree node |
| **Host pipeline**: multi-threaded step 1, 16-byte endpoints, pinned streams | 4090 ≈ 7× a 3060 (was ≈ 2× when CPU-bound) | |
| **Large windows**: arrays built on the GPU, 64-bit indexing, per-class windows (`--class-b`) | windows of 35 for the heavy classes on a 24 GB card: 0.851 → 0.675 4090-years | 21.5 GiB of VRAM, 1.5 min build per process |
| **Shared bitvectors** (`--bv-shared`): one sound array per class | half the bitvector memory | +0.2–0.3% work |

**Optional checkpoint walkback** (`--walkback-depth`, `--walkback-states`) is also
implemented, but is not enabled by `--rules improved` or included in the headline
configurations. At N = 72, A = 10, B = 24, adding a depth-50 search capped at 65,536
states to pm∞ + mix(3^13) removes about **0.66% of the remaining tail iterations**.
That is a sampled work reduction; its net running-time benefit, including the
extra CPU search, has not been established. See the
[walkback comparison](docs/FINDINGS.md#optional-checkpoint-walkback).
Follow-up [reuse probes](docs/FINDINGS.md#reusable-checkpoint-certificates) suggest
shallow checkpoints are more economical: depth 34 removes a modeled 20.7 GPU-hours
at N = 72 for about eight CPU-minutes of reverse searches, with the recommended
24 GB windows. A validated depth-30 certificate cache is included as an experiment;
production cache loading is not implemented, and an end-to-end speedup remains
unmeasured.

## Quick start

```sh
cargo build --release                       # needs nvcc; CPU-only: --no-default-features
export PATH=$PWD/target/release:$PATH
cargo test --release                        # unit tests + bitvector soundness audit
scripts/reproduce.sh                        # regression anchors, PASS/FAIL (~6 min)

cv verify --n 40                            # full verification below 2^40 on the CPU
cv verify --n 50 --gpu                      # ... steps 2+3 on the GPU (~23 s on a 4090)
cv verify --n 50 --gpu --b 28 --bv-pm-inf --bv-mix-j 13 --rules improved --mix-j 13

# 24 GB card, recommended: per-class windows, shared arrays, A=12
cv verify --n 56 --gpu --a 12 --b 28 --class-b 32,35,35,35,35,35,32,32 --bv-shared \
    --bv-pm-inf --bv-mix-j 13 --rules improved --mix-j 13
cv gpu-check --n 60 --b 28 --bv-shared --bv-pm-inf --bv-mix-j 13   # GPU build == host build

cv census --depth 37 --rules improved --mix-j 16   # step-1 survivors per depth
cv predict --from 60 --to 80                       # exact baseline workload (DP)
cv measure --n 72 --gpu --rules improved --mix-j 13 --json m72.json
cv plan --n 72 --costs m72.json --target-hours 1 --out plan72.json
cv verify --n 72 --gpu --plan plan72.json --package 0..40   # many packages, one array build
python3 scripts/hardware_cost.py   # per-card windows and cost per 2^72 from the measurements
```

`cv measure --gpu` on a new GPU takes 1–2 minutes and prints that device's GPU
time for the full verification at that N.

## Layout

```
src/arith.rs      exact helpers (3^k table, 306/485 slope test, 256-bit compares)
src/rules.rs      step-1 rules and certificates: pm1, OEE, pm_inf, mix(3^J), walkback
src/tree.rs       step-1 DFS over residue classes (n0, k, f, m, oee, ne)
src/lookahead.rs  step-2 bitvectors (two families; window credits), mod-9 sieve
src/tail.rs       steps 2+3 on the CPU, bit-identical to the GPU; 256-bit re-check
cuda/kernel.cu    steps 2+3 on the GPU (Angeltveit's arithmetic core)
src/gpu.rs        GPU driver: batching, pinned streams, per-class timing
src/dp.rs         exact parity-state DP: survivors, tree nodes, cases
src/sample.rs     exactly uniform endpoint draws at any depth
src/measure.rs    exact counts x sampled per-endpoint work and GPU time
src/units.rs      work packages with exactly predicted cost
tests/bv_audit.rs exhaustive witness audit of every bitvector exclusion
scripts/          reproduce.sh (regression anchors), gpu_fit.py + cost_table.py (tables),
                  hardware_cost.py (per-card windows and cost)
results/          sweeps, CPU benchmarks, cost tables
docs/             ALGORITHM.md, COST_MODEL.md, HARDWARE.md, FINDINGS.md
```

## Provenance

The baseline algorithm, the CUDA arithmetic core, and the 306/485 dip framework
are Angeltveit's (`../collatz_bitvec`, github.com/vigleik0/collatz). The pm∞,
mix(3^J), and walkback rules follow `../collatz-residue-sieve` and
`../collatz-probe`, and are re-implemented here, cross-checked against those
projects' counts.
