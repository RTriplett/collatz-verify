# Confirmation timing and telemetry audit

This read-only audit covers the completed `confirmation_a10_v2` run: 16
independent cohorts, two repetitions, and one concatenated-corpus run per
variant. It completed successfully on 2026-10-24 at 05:15:21 UTC. Sources are
[the raw trials](confirmation_a10_v2/pipeline.jsonl),
[GPU telemetry](confirmation_a10_v2/gpu.csv), and
[run metadata](confirmation_a10_v2/metadata.json).

## Repetition and order checks

The following ratios compare `novote_oee` with `parents`. Each entry is the
sum of the selected tuned times divided by the sum of the corresponding
baseline times. These are diagnostic subdivisions of the same experiment,
not additional independent samples or confidence intervals.

| Selection | Paired timings | Tuned / baseline |
|---|---:|---:|
| All | 32 | 0.896559 |
| Repetition 0 | 16 | 0.896302 |
| Repetition 1 | 16 | 0.896816 |
| Tuned variant runs first | 16 | 0.896392 |
| Tuned variant runs second | 16 | 0.896726 |

Every paired timing was faster with the tuned kernel; individual ratios
range from 0.893252 to 0.901990. Total repetition-1 / repetition-0 time is
0.999945 for `parents` and 1.000520 for `novote_oee`. Each cohort reverses
variant order between repetitions. Summing comparable cohorts, running
second rather than first changes timing by +0.0252% for `parents` and
+0.0121% for `novote_oee`. These checks show no material repetition or order
effect relative to the approximately 10.34% improvement.

## Device telemetry

The monitor samples approximately once per second. Its timestamps use
America/Chicago local time; metadata uses UTC. GPU setup takes 75.195 seconds.
The table below excludes the first 80 telemetry samples, retaining 882
samples from 05:00:39 through 05:15:20 UTC. This conservative post-setup
selection omits a few seconds near the start of measurement; trial records
do not carry wall-clock timestamps for exact sample-to-trial alignment.

| Metric | Minimum | Median | Maximum |
|---|---:|---:|---:|
| GPU utilization | 98% | 100% | 100% |
| Temperature | 68°C | 73°C | 73°C |
| Board power | 419.12 W | 431.27 W | 438.40 W |
| SM clock | 2670 MHz | 2685 MHz | 2685 MHz |
| Memory clock | 10251 MHz | 10251 MHz | 10251 MHz |

GPU utilization is 100% in 877 of these 882 samples. The SM clock remains
2685 MHz throughout the cohort trials and baseline aggregate. At 05:14:33
UTC it falls to 2670 MHz, a 0.56% reduction, for the final 48 telemetry
samples during the tuned aggregate. This late change does not affect the
primary paired cohort estimate. It is a small confound in the single
aggregate comparison; no clock-based correction is applied.

## Aggregate check

The aggregate runs concatenate the same sampled roots. They check batch
draining and partial-launch overhead; they do not add independent samples.

| Variant | Sum of cohort mean times | Aggregate time | Aggregate / cohort sum |
|---|---:|---:|---:|
| `parents` | 155.511472 s | 155.395675 s | 0.999255 |
| `novote_oee` | 139.425209 s | 139.432466 s | 1.000052 |

The aggregate tuned / baseline ratio is 0.897274. Both aggregate runs
contain the same eight flagged trajectories, with successful wide rechecks,
and their complete work signatures equal the combined cohort signatures.
The aggregate timings differ from their cohort sums by -0.0745% and +0.0052%.
Together with the balanced order and stable cohort clocks, this provides no
indication that another timing control is needed for the reported paired
comparison. It does not address long-run availability, other GPU models,
or costs excluded from the experiment.
