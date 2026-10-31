# Full-range checkpoint-pipeline extrapolation

Full-range N-target extrapolation of measured resident-array CPU/GPU pipeline wall time. This is not a completed full-range verification.

Fixed baseline checkpoint draws with replacement; rejected draws contribute zero work. Full-range total = exact baseline population × mean cost per baseline draw. Timing repetitions are averaged within each independent cohort. Student-t intervals use cohort degrees of freedom. Paired performance ratios use ratios of means and a paired delta-method interval. One year is 365.25 days.

Intervals describe cohort sampling and observed timing variation only; they do not include systematic desktop clock drift, long-run availability, checkpoint construction or archive I/O, setup, scheduler overhead, rare unsampled fallback/overflow work, or extrapolation to another GPU. Pipeline wall time is used because overlapping CUDA event sums are not GPU busy wall time. Cohort drain/partial-batch effects are checked by a separately reported aggregate run; no correction factor is silently applied. CPU core-years mean process CPU seconds divided by seconds per year, without an SMT conversion.

## results/fullrange_pipeline/a8_relative_pilot

N=72, A=8, K=38; 2 independent cohorts × 256 baseline draws, 2 timing repetitions per cohort. Exact baseline population: 1086320709. 477/512 draws survive checkpoint screening.

| Variant | GPU-years [95% interval] | CPU core-years [95% interval] | Busy cores | Ratio to comparison baseline |
|---|---:|---:|---:|---:|
| novote_oee | 0.6276 [0.5574, 0.6978] | 0.0842 [0.0722, 0.0962] | 0.134 | 1.0000 [1.0000, 1.0000] |

Full-range work (sampling estimates, without window-density assumptions):

| Quantity | Estimate | 95% interval |
|---|---:|---:|
| endpoints | 1.556241e+16 | [1.405137e+16, 1.707346e+16] |
| cases | 3.822512e+17 | [3.414962e+17, 4.230062e+17] |
| iters | 3.040581e+18 | [2.703504e+18, 3.377658e+18] |
| tree_nodes | 3.380635e+16 | [3.050236e+16, 3.711034e+16] |

Baseline descendant diagnostic: the same sampled roots predict 1.636645e+16 baseline endpoints versus the independently known exact total 16079624585697968 (+1.784%). This does not adjust the primary estimate.

One-time GPU setup: 74.947 seconds, outside trials. Ordered aggregate corpus SHA-256: `0b57c40791276fccb3f5d12371a793dec8e6b40fc7eb493a274c4ceaf1adc744`.
