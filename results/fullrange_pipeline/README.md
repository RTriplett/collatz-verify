# Full-range N72 estimates and GPU tuning

These experiments estimate the GPU allocation needed to verify every starting
value below 2^72 using the experimental checkpoint/terminal-table pipeline.
They measure a Ryzen 7 7700X with an RTX 4090, shared lookahead arrays, and a
prepared K38 frontier. They are sampled estimates, not a completed verification.

The [method](METHOD.md) describes the exact population weights, independent
cohorts, uncertainty intervals, and exclusions. The [validation record](VALIDATION.md)
describes CPU/GPU comparisons, including deliberately flagged trajectories.

## Confirmed full-range estimate

The [fresh confirmation](confirmation_summary.md) uses **16 independent cohorts
of 512 baseline K38 draws**, with two timing repetitions per variant. The
7,724 accepted draws retain their original 8,192-draw denominator. Each variant
also processes the concatenated corpus once in a continuous run.

| N72/A10 parent-dispatch pipeline | RTX 4090-years | Approximate 95% sampling interval | CPU core-years |
|---|---:|---:|---:|
| Original kernel | 0.6535 | 0.6299–0.6771 | 0.0690 |
| No overflow vote, early OEE mask | **0.5859** | **0.5649–0.6069** | **0.0681** |

The paired reduction is **10.34%**, with a 95% interval of **10.27–10.42%**.
That saves an estimated **593 GPU-hours**, or **24.7 GPU-days**, over the full
range. Every one of the 32 paired timings improves. Counts, iterations,
checksums, and flagged-origin multisets match exactly between kernels. Eight
distinct trajectories in six cohorts require overflow rechecks; all succeed,
and their recheck time is included.

The aggregate runs take 155.396 and 139.432 seconds, compared with sums of
cohort means of 155.511 and 139.425 seconds. Their ratios are **0.99926** and
**1.00005**, and all aggregate work counters and hashes match the combined
cohorts. Thus this batching check agrees within 0.08%; no correction factor is
applied. The exact-baseline endpoint diagnostic differs by −0.335%, also
without adjusting the primary estimate.

These intervals describe cohort sampling and observed timing variation, not
all uncertainty in a months-long deployment. The GPU clock holds at 2685 MHz
through the paired trials and drops to 2670 MHz late in the tuned aggregate;
see the [telemetry and order audit](TELEMETRY.md). This is a measured RTX 4090
estimate for the prepared-frontier pipeline. It does not revise the historical,
unmeasured RTX PRO 6000 projection. Frontier construction, archive I/O, setup,
and distributed coordination remain outside timing. Resident-array setup
takes 75.2 seconds per process in this run.

### Rare CPU fallback allowance

No F56–62 fallback endpoints occur in the checkpoint confirmation. Their exact
baseline population is **49,063,601**. A [separate CPU probe](rare_tail_summary.md)
draws 65,536 endpoints from that conditional population, in 16 cohorts with
two repetitions, and deliberately retains every baseline draw without improved
screening. It estimates **4,914 seconds (1.37 CPU-hours)** for scalar tails and
wide rechecks, with a 4,894–4,933 second sampling interval. All seven sampled
overflow trajectories finish wide rechecking.

This is an allowance for a conservative superset of the pipeline's rare
endpoints, not a rigorous runtime upper bound. Even if one GPU remained idle
throughout the entire allowance, the additional allocation would be only
**0.000156 GPU-years**, about **0.027%** of the tuned estimate. No allowance is
silently added to the cohort result; CPU/GPU overlap for this rare work is not
measured. The probe excludes endpoint generation, which is part of the main
pipeline's expansion rather than the isolated tail timing.

## Configurations

The default experiment retains improved13 ancestry and bounded walkback at
K38, uses improved checks through K50, and enters a terminal table at K52.
Parent/class descriptors cross the host/device boundary; the GPU reconstructs
the terminal descendants before running lookahead and the tail verifier.

The A10 resident windows, indexed by final F from 40, are:

```text
32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27
```

They consume about 19.625 GiB. Two producer threads and the whole process are
pinned to physical CPUs 0–1. Three CUDA streams share a persistent session.
The desktop remains active; GPU clocks are not locked. Timing includes host
expansion, descriptor packing, transfers, GPU expansion, verification, and any
CPU fallback. Array construction and frontier preparation precede timing.

| Kernel variant | Overflow check | Power-of-three lookup | Carried OEE filter | Threads/block |
|---|---|---|---|---:|
| `parents` | Warp vote followed by per-lane guard | Constant memory | After candidate reconstruction | 256 |
| `novote` | Per-lane guard | Constant memory | After reconstruction | 256 |
| `readonly` | Warp vote and guard | Read-only cached device table | After reconstruction | 256 |
| `both` | Per-lane guard | Read-only cached table | After reconstruction | 256 |
| `both_oee` | Per-lane guard | Read-only cached table | Before set-bit extraction | 256 |
| `block128` | Per-lane guard | Read-only cached table | Before extraction | 128 |
| `block512` | Per-lane guard | Read-only cached table | Before extraction | 512 |
| `novote_oee` | Per-lane guard | Constant memory | Before extraction | 256 |
| `novote_oee128` | Per-lane guard | Constant memory | Before extraction | 128 |
| `novote_oee512` | Per-lane guard | Constant memory | Before extraction | 512 |

These variants preserve the same mathematical work. Their tree counters,
per-class endpoint/case/iteration counts, checksums, and overflow handling are
checked against one another. Changing A or the lookahead windows changes the
workload legitimately and is recorded as a separate configuration.

## Kernel selection pilot

The first pilot uses two independent cohorts of 256 baseline draws, two timing
repetitions, seed 20261027. It is used for choosing experiments, not as the
final precision estimate. Ratios compare sums of cohort-mean wall times for
identical work, against `parents` in the same persistent session:

| Variant | Wall-time ratio | Change |
|---|---:|---:|
| `parents` | 1.0000 | reference |
| `novote` | 0.9526 | −4.74% |
| `readonly` | 1.0651 | +6.51% |
| `both` | 1.0152 | +1.52% |
| `both_oee` | 0.9752 | −2.48% |
| `block128` | 0.9745 | −2.55% |
| `block512` | 1.0196 | +1.96% |

Removing the warp-wide overflow vote helps; the read-only table is slower on
this workload. Early OEE filtering helps relative to the otherwise identical
`both` variant. A follow-up tests early OEE with the original constant-memory
table, avoiding the slower lookup. See [the complete pilot accounting](kernel_pilot_summary.md).

The old per-operation window model predicts less than 0.03% gain from
reallocating the same 19.625 GiB among classes. That is below its uncertainty
and the observed timing variation, so nearby memory reallocations were not
chosen for further benchmarking. This historical model does not establish
optimal windows for every new A/kernel combination.

The [constant-memory OEE follow-up](oee_pilot_summary.md) uses the same two
cohorts and two repetitions. Relative to `novote`, adding the early OEE mask
reduces wall time by **5.91%** with 256-thread blocks or **5.95%** with
128-thread blocks. Their difference is only 0.04%, so the 256-thread option
is retained for confirmation. The 512-thread option saves 5.28%. The final
comparison uses fresh random cohorts, rather than treating the fastest pilot
observation as the final estimate.

The [complete pilot comparison](PILOTS.md) also tests A9 and A8 with parent
dispatch and the selected kernel. A9 takes 0.53% longer than A10, while A8
takes 5.37% longer despite 8.14% fewer tail iterations. A second A8 window
allocation, preserving the A10 allocation by absolute F where possible,
takes 6.67% longer than A10. All use the same 19.625 GiB array budget and
the same saved roots. These comparisons cross separate runs, so the small
A9 difference does not establish a loss; neither deeper split shows a useful
gain. A10 is retained for confirmation.

The kernel improvements preserve the exact cases, tail iterations, and
checksums. Applying the carried OEE rule before extracting candidate bits
avoids reconstructing candidates that the original kernel rejects afterward.
Removing the redundant warp vote retains each lane's overflow check and wide
reverification. These changes make the same verification work cheaper.

## Further opportunities

A focused next experiment would assign two or four GPU threads to disjoint
completion-word ranges of each M62 endpoint. This preserves the frontier,
lookahead, and transport volume while testing whether different tail scheduling
outweighs duplicated endpoint initialization. It is not implemented or measured
here. Another possibility is staging F classes to afford deeper windows within
the same VRAM; that requires measuring repeated expansion and scheduling costs,
which the old independent-class model does not include.

## Reproduction

Build outside timed runs. Adapt the architecture and CPU affinity for the
machine; use fresh output directories.

```sh
CUDA_ARCH=sm_89 cargo build --release --offline --features cuda \
  --manifest-path scripts/expansion_bench/Cargo.toml \
  --bin fullrange_bench --target-dir /tmp/collatz-fullrange-target

python3 scripts/expansion_bench/run_overall.py \
  --binary /tmp/collatz-fullrange-target/release/fullrange_bench \
  --out /tmp/collatz-fullrange/confirmation_a10 \
  --roots 512 --cohorts 16 --repetitions 2 --aggregate-runs 1 \
  --threads 2 --affinity 0-1 --seed 20261028 --a 10 \
  --variants parents,novote_oee \
  --class-b 32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27

python3 -B scripts/expansion_bench/summarize_fullrange.py \
  --out /tmp/collatz-fullrange/confirmation_summary.json \
  /tmp/collatz-fullrange/confirmation_a10

# Run the CPU allowance separately, after GPU timing has finished.
cargo build --release --offline \
  --manifest-path scripts/expansion_bench/Cargo.toml \
  --bin rare_tail_bench --target-dir /tmp/collatz-fullrange-target
python3 -B scripts/expansion_bench/run_rare_tail.py \
  --binary /tmp/collatz-fullrange-target/release/rare_tail_bench \
  --out /tmp/collatz-fullrange/rare_cpu_tail \
  --cpu 0 --draws 4096 --cohorts 16 --repetitions 2 --seed 20261030
python3 -B scripts/expansion_bench/summarize_rare_tail.py \
  --out /tmp/collatz-fullrange/rare_tail_summary.json \
  /tmp/collatz-fullrange/rare_cpu_tail
```

For `fullrange_bench`, `--roots` counts baseline draws per cohort, including
rejected draws; it does not request a fixed number of accepted roots.
`--cohorts`, `--cutoff`, `--entry`, and `--aggregate-runs` are passed through
the recorded `CV_FULLRANGE_*` environment. Aggregate runs concatenate the
same accepted draws, retaining multiplicities, to check batching effects.

Each raw run directory retains the exact command, source snapshot, binary and
source hashes, accepted root records, rejection accounting, JSON results and
GPU telemetry. Hashes identify artifacts, rather than independently proving
compiler provenance. The tuning paths are opt-in; default `cv verify` retains
its previous settings.
