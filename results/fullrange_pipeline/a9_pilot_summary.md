# Full-range checkpoint-pipeline extrapolation

Full-range N-target extrapolation of measured resident-array CPU/GPU pipeline wall time. This is not a completed full-range verification.

Fixed baseline checkpoint draws with replacement; rejected draws contribute zero work. Full-range total = exact baseline population × mean cost per baseline draw. Timing repetitions are averaged within each independent cohort. Student-t intervals use cohort degrees of freedom. Paired performance ratios use ratios of means and a paired delta-method interval. One year is 365.25 days.

Intervals describe cohort sampling and observed timing variation only; they do not include systematic desktop clock drift, long-run availability, checkpoint construction or archive I/O, setup, scheduler overhead, rare unsampled fallback/overflow work, or extrapolation to another GPU. Pipeline wall time is used because overlapping CUDA event sums are not GPU busy wall time. Cohort drain/partial-batch effects are checked by a separately reported aggregate run; no correction factor is silently applied. CPU core-years mean process CPU seconds divided by seconds per year, without an SMT conversion.

## results/fullrange_pipeline/a9_pilot

N=72, A=9, K=38; 2 independent cohorts × 256 baseline draws, 2 timing repetitions per cohort. Exact baseline population: 1086320709. 477/512 draws survive checkpoint screening.

| Variant | GPU-years [95% interval] | CPU core-years [95% interval] | Busy cores | Ratio to comparison baseline |
|---|---:|---:|---:|---:|
| novote_oee | 0.5988 [0.5336, 0.6639] | 0.0781 [0.0594, 0.0969] | 0.131 | 1.0000 [1.0000, 1.0000] |

Full-range work (sampling estimates, without window-density assumptions):

| Quantity | Estimate | 95% interval |
|---|---:|---:|
| endpoints | 8.100574e+15 | [7.311537e+15, 8.889611e+15] |
| cases | 3.978173e+17 | [3.555764e+17, 4.400582e+17] |
| iters | 3.168732e+18 | [2.818992e+18, 3.518471e+18] |
| tree_nodes | 1.80351e+16 | [1.626324e+16, 1.980696e+16] |

Baseline descendant diagnostic: the same sampled roots predict 8.524215e+15 baseline endpoints versus the independently known exact total 8374815028830481 (+1.784%). This does not adjust the primary estimate.

One-time GPU setup: 74.715 seconds, outside trials. Ordered aggregate corpus SHA-256: `0b57c40791276fccb3f5d12371a793dec8e6b40fc7eb493a274c4ceaf1adc744`.
