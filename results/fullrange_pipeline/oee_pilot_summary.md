# Full-range checkpoint-pipeline extrapolation

Full-range N-target extrapolation of measured resident-array CPU/GPU pipeline wall time. This is not a completed full-range verification.

Fixed baseline checkpoint draws with replacement; rejected draws contribute zero work. Full-range total = exact baseline population × mean cost per baseline draw. Timing repetitions are averaged within each independent cohort. Student-t intervals use cohort degrees of freedom. Paired performance ratios use ratios of means and a paired delta-method interval. One year is 365.25 days.

Intervals describe cohort sampling and observed timing variation only; they do not include systematic desktop clock drift, long-run availability, checkpoint construction or archive I/O, setup, scheduler overhead, rare unsampled fallback/overflow work, or extrapolation to another GPU. Pipeline wall time is used because overlapping CUDA event sums are not GPU busy wall time. Cohort drain/partial-batch effects are checked by a separately reported aggregate run; no correction factor is silently applied. CPU core-years mean process CPU seconds divided by seconds per year, without an SMT conversion.

## results/fullrange_pipeline/oee_pilot

N=72, A=10, K=38; 2 independent cohorts × 256 baseline draws, 2 timing repetitions per cohort. Exact baseline population: 1086320709. 477/512 draws survive checkpoint screening.

| Variant | GPU-years [95% interval] | CPU core-years [95% interval] | Busy cores | Ratio to comparison baseline |
|---|---:|---:|---:|---:|
| novote | 0.6330 [0.5806, 0.6855] | 0.0707 [0.0686, 0.0729] | 0.112 | 1.0000 [1.0000, 1.0000] |
| novote_oee | 0.5956 [0.5330, 0.6583] | 0.0687 [0.0625, 0.0749] | 0.115 | 0.9409 [0.9199, 0.9619] |
| novote_oee128 | 0.5954 [0.5450, 0.6457] | 0.0682 [0.0557, 0.0807] | 0.115 | 0.9405 [0.9390, 0.9421] |
| novote_oee512 | 0.5996 [0.5381, 0.6610] | 0.0690 [0.0567, 0.0813] | 0.115 | 0.9472 [0.9285, 0.9658] |

Full-range work (sampling estimates, without window-density assumptions):

| Quantity | Estimate | 95% interval |
|---|---:|---:|
| endpoints | 4.253022e+15 | [3.837461e+15, 4.668583e+15] |
| cases | 4.144454e+17 | [3.70662e+17, 4.582287e+17] |
| iters | 3.309931e+18 | [2.946411e+18, 3.673451e+18] |
| tree_nodes | 9.582755e+15 | [8.635968e+15, 1.052954e+16] |

Baseline descendant diagnostic: the same sampled roots predict 4.478258e+15 baseline endpoints versus the independently known exact total 4399723745080828 (+1.785%). This does not adjust the primary estimate.

One-time GPU setup: 75.282 seconds, outside trials. Ordered aggregate corpus SHA-256: `0b57c40791276fccb3f5d12371a793dec8e6b40fc7eb493a274c4ceaf1adc744`.
