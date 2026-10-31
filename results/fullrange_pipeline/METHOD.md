# Full-range estimates from checkpoint cohorts

This experiment estimates the resources needed to process the complete
checkpoint frontier for verification below `2^72`. It measures the live CPU/GPU
pipeline on an RTX 4090, starting with screened K38 roots and resident lookahead
arrays. It does not perform a complete verification below `2^72` or construct
the complete stored frontier.

The numerical results belong to the exact configuration recorded by each run:
`N`, `A`, pruning cutoff, terminal-table entry depth, per-class windows, kernel
variant, CPU allocation, and GPU. The default experiment targets N72/A10,
improved checks through K50, and terminal expansion from K52 to M62. A timing
from one configuration must not be applied to a different pruning policy or
window allocation without another measurement.

## Relationship to the earlier cost model

[`docs/COST_MODEL.md`](../../docs/COST_MODEL.md) estimates GPU kernel work from
exact baseline endpoint populations, samples of real target-depth endpoints,
and measured or fitted costs per endpoint, case, and tail iteration. Its
full-range tables describe that algorithm, kernel, window allocation, and
hardware model. The larger-window and other-device estimates in
[`docs/HARDWARE.md`](../../docs/HARDWARE.md) also depend on the documented memory
allocation and device-throughput scaling.

This experiment estimates a different quantity: **elapsed time on one GPU
while the complete experimental checkpoint pipeline supplies and verifies
work**. Timed regions include CPU expansion after K38, group construction,
queueing, transfers, device expansion, verification, and any observed CPU
fallback. The estimate therefore includes the effects of the actual transport,
kernel, and host allocation. It uses measured cases and iterations and does
not infer them from bitvector density or an equidistribution assumption.

The two approaches provide useful cross-checks, but their numbers are not
interchangeable. The newer measurement does not refit the earlier per-operation
model, validate another GPU, or automatically revise its entire N64–N80 table.
In particular, the historical RTX PRO 6000 estimate is a hardware projection;
this experiment measures only the available RTX 4090.

## Population and fixed-draw sampling

The baseline rules depend only on the parity state. Dynamic programming gives
the exact number of baseline residues surviving to K38:

```text
P38 = 1,086,320,709
```

`Sampler` uses backward descendant counts to draw from that population
uniformly. Each cohort contains a **fixed, predeclared number D of baseline
draws with replacement**. Separate deterministic SplitMix64 seeds identify the
cohorts. The sampler's finite-width modulo bias is negligible at these
population sizes; the statistical model treats its pseudorandom draws as
independent uniform draws.

For every draw the harness applies improved13 ancestry screening and a bounded
K38 walkback certificate search with a 65,536-state cap. A rejected draw
contributes zero downstream work. An accepted draw contributes its entire
subtree under the configured continuation policy. Duplicate accepted draws
remain in the workload, including their multiplicity in counters and the XOR
checksum.

Consequently, every baseline root has the same sampling weight. No root class
is selected because it is cheap, common among accepted roots, or profitable
for a particular kernel. Rejections remain in D, and sampling never continues
until a target number of accepted roots is reached. There is no separately
estimated survival multiplier in the main estimator.

The confirmation design uses 16 independent cohorts of 512 baseline
draws, with two timing repetitions per cohort and variant. Pilot runs use
smaller designs to compare implementation choices. The saved `config` record
is authoritative for a particular run. Selecting a variant from a pilot
requires a fresh confirmation corpus; the fastest pilot observation alone
would give an optimistically selected estimate.

## Full-range estimator and uncertainty

For cohort c, let `W_c` be an observed work count and let `T_c` be the mean
pipeline wall time over its timing repetitions. Then:

```text
estimated full-range work = P38 × mean_c(W_c / D)
estimated GPU-years      = P38 × mean_c(T_c / D) / 31,557,600
```

The work-count estimator is unbiased under the sampling model, because
screened-out roots have zero work and all baseline draws remain represented.
The time estimate additionally assumes that sustained processing cost scales
with the sampled workload. Finite batches, pipeline draining, class scheduling,
and clock changes can violate exact additivity; the aggregate calibration
below checks their combined effect on the measured corpus.

Timing repetitions use the same roots. They measure timing variation and are
averaged **within each cohort**, rather than counted as additional independent
sampling observations. The primary estimate uses arithmetic means. Medians
would not preserve the population weighting of total work and time.

For C cohorts the reported interval is:

```text
mean ± t(0.975, C−1) × sample_standard_deviation / sqrt(C)
```

The scale factor `P38 / D` and the conversion to years are then applied. This
is an approximate Student-t interval over independent cohort means, not a
distribution-free bound. Sixteen cohorts give 15 degrees of freedom; doubling
the repetitions does not double that number. Small pilot designs can have very
wide intervals and should not be used as the final precision claim.

Paired performance comparisons use the ratio of the two variants' cohort
means, which is also the ratio of their estimated total costs. They do not
average per-cohort speedup ratios. The uncertainty uses paired cohort
residuals and a delta-method t interval, preserving the correlation caused by
running identical roots through both variants.

These intervals cover root sampling variation and timing variation present
within the experiment. They do not establish an interval for sustained clock
behavior across months, desktop interference outside the measurement, device
availability, interruptions, production scheduling, or another GPU. Drift
shared by many trials can also be missed by a cohort interval. Cohort and
variant order rotate between repetitions, and the runner records GPU telemetry
so those limitations can be assessed separately.

## Exact-count diagnostic

For every baseline draw, including rejected draws, the harness also computes
its exact number of baseline M-depth descendants using backward DP. Scaling
the sampled sum by `P38 / D` estimates a total independently known exactly:
the entire baseline M-depth endpoint population.

The summary reports this sampled-versus-exact comparison as a check on cohort
composition and the weighting scheme. It is not a fitted adjustment to the
primary estimate. Using it as a control variate would require a separately
specified estimator and uncertainty calculation.

## Aggregate calibration and correctness

After the cohort trials, the harness concatenates all accepted draws in cohort
order and runs the combined workload as one continuous invocation per variant.
It preserves duplicate roots. For each aggregate invocation:

* Every tree counter and per-class endpoint, case, and iteration count must
  equal the sum of the separate cohort reports.
* Its trajectory checksum must equal the XOR of the cohort checksums.
* Repetitions and same-work transport/kernel variants must agree on the full
  work signature.

The aggregate wall time divided by the sum of cohort mean wall times measures
the observed effect of joining the workloads, together with any elapsed-time
drift between the two measurements. A ratio different from one cannot by itself
be attributed entirely to partial batches or entirely to clocks.

This check retains the sampled root order. A complete frontier archive can use
a different order and package size, changing locality or scheduling behavior;
the aggregate check does not measure every such deployment choice.

The summary reports both that ratio and the direct full-range scaling of the
aggregate time. It does not silently multiply the primary estimate by a
calibration factor. The aggregate uses the same roots, so it is a correlated
check, not another independent population sample or an extra degree of freedom.

## What is outside the timed estimate

The frontier is treated as already prepared and available in memory. Excluded
work includes building the K38 frontier, its ancestry/walkback preprocessing,
writing or reading a checkpoint archive, sampled-root construction, expansion
table preparation, lookahead construction/upload, and distributed coordination.
Setup and sampling durations are recorded separately.

Sampling a root by walking a backward DP is not the same computation as
enumerating a complete production frontier. Scaling the harness's sampling
seconds by P38 would therefore not provide a justified frontier-construction
estimate. Reusing a stored frontier can amortize its preparation cost, but its
construction and archive throughput require separate measurements.

Lookahead arrays and suffix tables remain resident across trials. A production
worker must pay setup at every restart; the full deployment budget should add
those costs at its expected restart frequency. Initial/final draining inside a
timed invocation is included. CUDA event totals from concurrent streams are
not added as if they were elapsed GPU time, because those intervals overlap.

CPU core-years are measured process CPU seconds divided by 31,557,600, without
an SMT conversion. Average busy CPU cores are CPU seconds divided by pipeline
wall seconds. This differs from historical tables that express one physical
core as two SMT threads. It also does not establish how many GPUs one CPU core
could serve concurrently.

Any fallback observed in a timed sample is included in its wall time and
counters. Seeing no flags or high-class fallback in a sample does not prove
those events are absent from the full frontier. Unsampled rare work, exceptional
trajectories, and retry policy remain outside what the cohort interval alone
can establish.

The completed experiment also reports a [separate rare-tail allowance](rare_tail_summary.md):
conditional baseline endpoint sampling measures the scalar CPU cost of F56–62,
using its exact population and including wide rechecks. All baseline draws
are retained without extra screening. This supplement addresses a class absent
from the checkpoint sample, but is not silently added to that sample's GPU
estimate or confidence interval. Its hypothetical idle-GPU scenario assumes
no CPU/GPU overlap; it is not a measured GPU workload.

## Reproducibility and audit

`scripts/expansion_bench/run_overall.py` saves each run in a fresh directory:

| Artifact | Contents |
|---|---|
| `metadata.json` | Command, arguments, full-range environment, affinity, toolchain, device/host information, source and binary hashes, run status |
| `pipeline.jsonl` | Configuration, exact populations, cohort seeds, every accepted root, rejection accounting, warmups, full trial reports, aggregate calibration |
| `pipeline.stderr.txt` | Harness diagnostics |
| `gpu.csv`, `gpu.stderr.txt` | GPU telemetry and monitor diagnostics |
| `source/` | Captured Rust/CUDA implementation, Cargo/build inputs, and runner sources |

Captured source and binary hashes identify the inputs; they do not
independently prove that a particular binary was built from a particular source
snapshot. Runs with different binaries, policies, devices, or corpora retain
their own provenance and are not silently pooled.

The repository's analysis script audits successful, complete runs before
writing JSON and Markdown summaries:

```sh
python3 -B scripts/expansion_bench/summarize_fullrange.py \
  --out results/fullrange_pipeline/summary.json RUN_DIRECTORY...
```

It rejects incomplete trials, missing or duplicate cohorts/repetitions,
inconsistent draw accounting, incompatible work signatures, malformed report
totals, and aggregate sums/checksums that do not match. The exact saved roots
and ordered corpus SHA-256 allow later implementations to rerun the same work.
The generated full-range totals remain extrapolations; correctness validation
of the implementation and completion of an actual range are separate claims.
