# Full-range checkpoint-pipeline extrapolation

Full-range N-target extrapolation of measured resident-array CPU/GPU pipeline wall time. This is not a completed full-range verification.

Fixed baseline checkpoint draws with replacement; rejected draws contribute zero work. Full-range total = exact baseline population × mean cost per baseline draw. Timing repetitions are averaged within each independent cohort. Student-t intervals use cohort degrees of freedom. Paired performance ratios use ratios of means and a paired delta-method interval. One year is 365.25 days.

Intervals describe cohort sampling and observed timing variation only; they do not include systematic desktop clock drift, long-run availability, checkpoint construction or archive I/O, setup, scheduler overhead, rare unsampled fallback/overflow work, or extrapolation to another GPU. Pipeline wall time is used because overlapping CUDA event sums are not GPU busy wall time. Cohort drain/partial-batch effects are checked by a separately reported aggregate run; no correction factor is silently applied. CPU core-years mean process CPU seconds divided by seconds per year, without an SMT conversion.

## results/grouped_transitions/pilot

N=72, A=10, K=38; 2 independent cohorts × 128 baseline draws, 2 timing repetitions per cohort. Exact baseline population: 1086320709. 239/256 draws survive checkpoint screening.

| Variant | GPU-years [95% interval] | CPU core-years [95% interval] | Busy cores | Ratio to comparison baseline |
|---|---:|---:|---:|---:|
| novote_oee | 0.6303 [-0.2130, 1.4735] | 0.0742 [0.0255, 0.1228] | 0.118 | 1.0000 [1.0000, 1.0000] |
| affine32 | 0.8487 [0.1664, 1.5311] | 1.6012 [0.5484, 2.6539] | 1.887 | 1.3466 [0.6276, 2.0655] |
| chain4 | 1.0963 [0.5519, 1.6407] | 2.0971 [1.0023, 3.1919] | 1.913 | 1.7393 [0.2760, 3.2027] |
| group4 | 0.9027 [0.2221, 1.5832] | 1.6868 [0.7731, 2.6004] | 1.869 | 1.4322 [0.5958, 2.2685] |
| chain6 | 0.8954 [0.3804, 1.4105] | 1.7087 [0.8480, 2.5693] | 1.908 | 1.4207 [0.3371, 2.5042] |
| group6 | 0.8016 [-0.1203, 1.7234] | 1.3751 [0.8348, 1.9153] | 1.715 | 1.2718 [1.0329, 1.5106] |
| jump8 | 1.4880 [-0.5355, 3.5115] | 0.0803 [0.0599, 0.1007] | 0.054 | 2.3608 [2.3089, 2.4128] |
| jump10 | 1.6135 [-0.5788, 3.8058] | 0.0789 [0.0179, 0.1400] | 0.049 | 2.5600 [2.5067, 2.6133] |
| jump12 | 1.8938 [-0.9157, 4.7033] | 0.0802 [0.0391, 0.1213] | 0.042 | 3.0048 [2.5672, 3.4423] |
| jump14 | 3.2823 [-0.9794, 7.5440] | 0.0844 [0.0517, 0.1170] | 0.026 | 5.2078 [5.0020, 5.4136] |
| group4_jump12 | 1.7851 [-0.7533, 4.3234] | 1.7398 [0.8373, 2.6423] | 0.975 | 2.8322 [2.5940, 3.0704] |

Full-range work (sampling estimates, without window-density assumptions):

| Quantity | Estimate | 95% interval |
|---|---:|---:|
| endpoints | 4.298348e+15 | [1.909386e+15, 6.68731e+15] |
| cases | 4.195536e+17 | [-1.542133e+17, 9.933206e+17] |
| iters | 3.350647e+18 | [-1.500796e+18, 8.202091e+18] |
| tree_nodes | 9.683548e+15 | [5.019786e+15, 1.434731e+16] |
| flagged_trajectories | 0 | [0, 0] |
| wide_rechecks | 0 | [0, 0] |

Baseline descendant diagnostic: the same sampled roots predict 4.516912e+15 baseline endpoints versus the independently known exact total 4399723745080828 (+2.664%). This does not adjust the primary estimate.

One pass through the sampled cohorts observed 0 flagged trajectories (0 distinct origins) in 0 cohorts. Every observed flag completed a wide recheck. These rechecks are included in the outer pipeline wall time. Sparse/clustered flags do not support a precise full-range frequency estimate; their displayed t interval is only a diagnostic.

One-time GPU setup: 87.546 seconds, outside trials. Ordered aggregate corpus SHA-256: `84e9dc9c374bb27aba73d8b4efb0df8b3ad704c8040c38781349866d8dffb734`.
