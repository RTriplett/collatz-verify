# CPU ownership before GPU dispatch

The CPU benefits from its own completion width: expanding assigned M62 families to M66/A6 reduced CPU cost roughly **18–20%** with B20. Direct routing works, but neither integrated experiment establishes a reliable overall speedup on this Ryzen 7 7700X / RTX 4090. Even with nearly four CPU cores busy, mean pipeline time was **8.024 s**, versus **8.038 s** for ordinary GPU processing and **8.018 s** for a routing control with no diversion. Production defaults remain unchanged.

## Ownership and scheduling

The experimental path starts with the existing tree expansion and compact terminal groups:

```text
K38 roots → improved checks through K50 → terminal groups describing M62 families
                                      ├─ ordinary GPU sink → A10 + GPU B arrays → GPU tail
                                      └─ admitted CPU group → expand to CPU M → CPU A/B → CPU tail
```

Each accepted CPU job owns complete original M62 source families; those families never enter GPU dispatch. The CPU reconstructs their exact endpoints and uses reusable suffix tables when its A is smaller. For example, A6 expands four further bits to M66, then screens and verifies the surviving completions. The GPU retains its existing A10 policy and class-specific windows. The existing out-of-range-F CPU fallback is unchanged.

Routing selects eligible compact groups by descriptor hash. Admission reserves bounded in-flight capacity before copying; a full CPU pool immediately forwards the group to the ordinary GPU sink. Thus CPU backpressure does not block the producer waiting for a free CPU worker. The pilot targets original F40, uses four workers and a capacity of 64 groups, and initially forwards 4,194,304 endpoints to the ordinary sink. This startup guard is **not** a live GPU queue-occupancy measurement.

CPU and GPU ownership counts are conserved at the original source depth. Their native endpoint counts, cases, iterations and checksums are reported separately: changing M/A/B changes the proof and checksum boundary. In particular, GPU iterations removed are measured against the unchanged GPU policy on the same root corpus; they are not inferred from the CPU's different iteration count.

## CPU completion-width experiments

Both CPU runs used one Ryzen 7 7700X core pinned to CPU 6, five rotated/reversed repetitions, and identical source families across policies within each run. Sources are conditional M62/F draws passing improved13 ancestry checks through K50, with duplicates retained and no K38 walkback filtering.

| Run | Sources per F class | Policies | Observed result |
|---|---:|---|---|
| [Initial sweep](cpu_probe/README.md) | 2,048 | A10/8/6/4/2/0; B none/16/20 | A6/B20 lowest mean for F40/41/42; 18–20% below same-B A10 |
| [B24 follow-up](cpu_b24/README.md) | 8,192 | A10/8/6/4; B20/24 | A6/B24 lowest mean; F40 ranking changes when using medians |

For F41/F42, A6/B24 costs 4.787/9.608 µs per original source by mean CPU time, about 7–8% below A6/B20. F40 remains less decisive: A6/B24 has mean/median 1.603/1.651 µs, A8/B24 has 1.632/1.641 µs, and A6/B20 has 1.645/1.656 µs. The median advantage of A8/B24 over A6/B20 is less than 1%.

B24 needs about 32 MiB per shared resident plan versus 2 MiB for B20. Single-core setup took about 6.5 seconds versus 0.22 seconds, excluded from the steady processing timings. These CPU windows are smaller than the GPU's tuned windows; a CPU improvement does not imply equal work per endpoint across devices.

## Integrated pilot

[Audited results](pipeline_pilot/summary.md) use seed 20261105, four cohorts of 128 baseline K38 draws (488 accepted of 512), two tree threads, a 4090, and three continuous-corpus repetitions per variant. GPU windows are `32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27`. Setup and warmup are excluded; timing includes worker creation, dispatch, final CPU drain, and all wide rechecks.

| Variant | Mean elapsed s | Observed range s | CPU worker core-s | GPU iterations removed |
|---|---:|---:|---:|---:|
| Ordinary GPU | 9.4145 | 9.3892–9.4437 | 0 | 0 |
| Routing wrapper, no CPU admissions | 9.4053 | 9.3698–9.4622 | 0 | 0 |
| CPU A10/B24, all eligible groups offered | 9.3852 | 9.3504–9.4379 | 19.353 | 0.06832% |
| CPU A6/B24, all eligible groups offered | 9.3680 | 9.3615–9.3733 | 17.110 | 0.07366% |
| CPU A6/B20, all eligible groups offered | 9.3998 | 9.3613–9.4526 | 18.707 | 0.06973% |
| CPU A6/B24, one in 64 groups offered | 9.4132 | 9.3791–9.4468 | 11.237 | 0.04327% |

A6/B24 averaged 0.49% below ordinary GPU time and 0.40% below the routing control, but the ranges overlap and only three repetitions were measured. It diverted about 11.36 million original endpoints per continuous run while removing only 0.07366% of GPU tail iterations. F40 is cheap enough for CPU admission but also cheap under the GPU policy.

Four configured CPU workers averaged only **1.83 busy cores** (17.110 core-seconds over 9.368 wall-seconds). The final drain was negligible, about 0.079 ms. This motivates testing whether a larger queue can carry CPU work between producer bursts; these counters alone do not prove the reason for idle time. Hash selection is deterministic, but queue admission depends on worker availability, so admitted groups can vary across repetitions.

## Larger-queue follow-up

The [audited follow-up](pipeline_queue1024/summary.md) increases bounded in-flight capacity from 64 to 1,024 groups. It uses the same frozen executable, seed 20261106, four cohorts of 128 baseline draws (485 accepted of 512), six continuous-corpus repetitions, and three variants. All threads inherit process affinity 0–7; the two tree producers and four verification workers are not individually pinned to different cores.

| Variant | Mean elapsed s | Median elapsed s | Observed range s | Total process CPU core-s |
|---|---:|---:|---:|---:|
| Ordinary GPU | 8.0384 | 8.0324 | 8.0144–8.0706 | 1.318 |
| Routing wrapper, no CPU admissions | 8.0175 | 8.0188 | 8.0008–8.0382 | 1.398 |
| CPU A6/B24, four workers | 8.0240 | 8.0327 | 7.9923–8.0491 | 32.958 |

The workers now average **3.94 busy cores**, consuming 31.589 worker core-seconds per run, with only **0.090 ms** of final drain. They divert **21.57 million original endpoints**, or **1.22%** of the original source families. Under the unchanged GPU policy this removes **0.193% of cases and 0.164% of tail iterations**. The CPU's weaker B24 screen and different A produce their own native work totals. Diverting these cheap F40 families is not equivalent to removing the same fraction of all GPU work.

Mean time is 0.18% below the ordinary GPU control but 0.08% above the routing control. Median time is essentially equal to the ordinary GPU control and 0.17% above the routing control. Per-repetition comparisons cross equality against both controls. **There is no stable elapsed-time win.** Higher CPU utilization alone does not establish useful GPU savings; these measurements do not justify changing defaults or the full-range GPU-year estimate.

The changed seed makes this a separate corpus; compare variants within this follow-up, not absolute times against the pilot. The larger queue's high utilization supports the producer-burst explanation, but the two runs are not a paired queue-only experiment. Per-class CUDA intervals also change with scheduling and stream overlap: their sums cannot be converted into additive GPU savings.

## Correctness checks

- CPU-only source ledgers enumerate all 1,024 original starts per audited source, check suffixes against ordinary DFS and independent exact-power arithmetic, and independently wide-check all starts, including screened starts. M>64 arithmetic remains u128. For A<6, M-correct A6 storage is truncated to exactly 2^A completion bits.
- [Complete N32 validation](validate_n32/pipeline.jsonl) and [N72 wide-state validation](validate_wide/pipeline.jsonl) pass six variants each. Same-policy routed and unrouted totals match exactly. Different-A CPU results are independently replayed under their own policy rather than incorrectly compared to the GPU checksum.
- [Forced-cutoff validation](validate_cutoff/pipeline.jsonl), cutoff 110, preserves all 985 baseline flags in the same-policy A10 run: 909 remain in ordinary processing and 76 move to the CPU. Mixed-A CPU variants produce 68/64/46 flags under their own policies; every flag passes wide recheck, and each policy matches its CPU oracle.
- Runtime checks conserve original per-F endpoint counts and descriptor count/XOR/sum identifiers across CPU and GPU ownership. These identifiers are consistency checks, not cryptographic proofs of identical descriptor sets. Timed runs disable source/diagnostic capture.

## Interpretation and provenance

The CPU source cohorts are conditional strata and cannot be equally averaged into a full-range population estimate. Continuous repetitions reuse the same roots: their ranges describe timing variation, not independent sampling confidence intervals. CPU core-seconds, worker phase intervals, CUDA event sums, and pipeline elapsed time overlap and cannot be added into a wall-time breakdown. No GPU-year or rental-price extrapolation is made here.

Each run saves `pipeline.jsonl`, `metadata.json`, a source snapshot, executable/source hashes, and applicable GPU telemetry. Hashes identify saved artifacts; they do not independently prove how an executable was built. CPU audit reports are in the two CPU directories; both integrated reports check ownership, work, repeated order, setup separation, and provenance. The 21 Python auditor tests and 22 release tests for the integrated benchmark pass. Implementation is experimental in [direct_cpu.rs](../../src/direct_cpu.rs) and [direct_pipeline_bench.rs](../../scripts/expansion_bench/direct_pipeline_bench.rs); [implementation notes](IMPLEMENTATION.md) describe the queue and validation semantics.

Reproduce the existing summaries without rerunning computation:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/expansion_bench/summarize_direct_cpu.py \
  --out results/direct_cpu/cpu_probe/summary.json results/direct_cpu/cpu_probe/pipeline.jsonl
PYTHONDONTWRITEBYTECODE=1 python3 scripts/expansion_bench/summarize_direct_cpu.py \
  --out results/direct_cpu/cpu_b24/summary.json results/direct_cpu/cpu_b24/pipeline.jsonl
PYTHONDONTWRITEBYTECODE=1 python3 scripts/expansion_bench/summarize_direct_pipeline.py \
  --out results/direct_cpu/pipeline_pilot/summary.json results/direct_cpu/pipeline_pilot
PYTHONDONTWRITEBYTECODE=1 python3 scripts/expansion_bench/summarize_direct_pipeline.py \
  --out results/direct_cpu/pipeline_queue1024/summary.json results/direct_cpu/pipeline_queue1024
```

To rerun an experiment, use a fresh output directory with the command and environment saved in its `metadata.json`; CPU probes use `run_direct_cpu.py`, and integrated runs use `run_overall.py` with the `direct_pipeline_bench` executable. Build under `/tmp`, and run CPU-only sweeps separately from GPU timing runs.
