# Full-range checkpoint-pipeline extrapolation

Full-range N-target extrapolation of measured resident-array CPU/GPU pipeline wall time. This is not a completed full-range verification.

Fixed baseline checkpoint draws with replacement; rejected draws contribute zero work. Full-range total = exact baseline population × mean cost per baseline draw. Timing repetitions are averaged within each independent cohort. Student-t intervals use cohort degrees of freedom. Paired performance ratios use ratios of means and a paired delta-method interval. One year is 365.25 days.

Intervals describe cohort sampling and observed timing variation only; they do not include systematic desktop clock drift, long-run availability, checkpoint construction or archive I/O, setup, scheduler overhead, rare unsampled fallback/overflow work, or extrapolation to another GPU. Pipeline wall time is used because overlapping CUDA event sums are not GPU busy wall time. Cohort drain/partial-batch effects are checked by a separately reported aggregate run; no correction factor is silently applied. CPU core-years mean process CPU seconds divided by seconds per year, without an SMT conversion.

## results/fullrange_pipeline/kernel_pilot

N=72, A=10, K=38; 2 independent cohorts × 256 baseline draws, 2 timing repetitions per cohort. Exact baseline population: 1086320709. 477/512 draws survive checkpoint screening.

| Variant | GPU-years [95% interval] | CPU core-years [95% interval] | Busy cores | Ratio to comparison baseline |
|---|---:|---:|---:|---:|
| parents | 0.6642 [0.5938, 0.7345] | 0.0703 [0.0697, 0.0710] | 0.106 | 1.0000 [1.0000, 1.0000] |
| novote | 0.6327 [0.5730, 0.6923] | 0.0693 [0.0650, 0.0735] | 0.109 | 0.9526 [0.9415, 0.9636] |
| readonly | 0.7074 [0.6308, 0.7840] | 0.0700 [0.0695, 0.0706] | 0.099 | 1.0651 [1.0626, 1.0676] |
| both | 0.6742 [0.6011, 0.7474] | 0.0697 [0.0605, 0.0789] | 0.103 | 1.0152 [1.0127, 1.0177] |
| both_oee | 0.6477 [0.5787, 0.7168] | 0.0687 [0.0674, 0.0700] | 0.106 | 0.9752 [0.9746, 0.9759] |
| block128 | 0.6472 [0.5772, 0.7172] | 0.0693 [0.0658, 0.0728] | 0.107 | 0.9745 [0.9723, 0.9767] |
| block512 | 0.6772 [0.6140, 0.7404] | 0.0696 [0.0685, 0.0707] | 0.103 | 1.0196 [1.0067, 1.0325] |

Full-range work (sampling estimates, without window-density assumptions):

| Quantity | Estimate | 95% interval |
|---|---:|---:|
| endpoints | 4.253022e+15 | [3.837461e+15, 4.668583e+15] |
| cases | 4.144454e+17 | [3.70662e+17, 4.582287e+17] |
| iters | 3.309931e+18 | [2.946411e+18, 3.673451e+18] |
| tree_nodes | 9.582755e+15 | [8.635968e+15, 1.052954e+16] |

Baseline descendant diagnostic: the same sampled roots predict 4.478258e+15 baseline endpoints versus the independently known exact total 4399723745080828 (+1.785%). This does not adjust the primary estimate.

One-time GPU setup: 75.319 seconds, outside trials. Ordered aggregate corpus SHA-256: `0b57c40791276fccb3f5d12371a793dec8e6b40fc7eb493a274c4ceaf1adc744`.
