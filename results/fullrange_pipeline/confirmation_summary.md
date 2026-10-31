# Full-range checkpoint-pipeline extrapolation

Full-range N-target extrapolation of measured resident-array CPU/GPU pipeline wall time. This is not a completed full-range verification.

Fixed baseline checkpoint draws with replacement; rejected draws contribute zero work. Full-range total = exact baseline population × mean cost per baseline draw. Timing repetitions are averaged within each independent cohort. Student-t intervals use cohort degrees of freedom. Paired performance ratios use ratios of means and a paired delta-method interval. One year is 365.25 days.

Intervals describe cohort sampling and observed timing variation only; they do not include systematic desktop clock drift, long-run availability, checkpoint construction or archive I/O, setup, scheduler overhead, rare unsampled fallback/overflow work, or extrapolation to another GPU. Pipeline wall time is used because overlapping CUDA event sums are not GPU busy wall time. Cohort drain/partial-batch effects are checked by a separately reported aggregate run; no correction factor is silently applied. CPU core-years mean process CPU seconds divided by seconds per year, without an SMT conversion.

## results/fullrange_pipeline/confirmation_a10_v2

N=72, A=10, K=38; 16 independent cohorts × 512 baseline draws, 2 timing repetitions per cohort. Exact baseline population: 1086320709. 7,724/8,192 draws survive checkpoint screening.

| Variant | GPU-years [95% interval] | CPU core-years [95% interval] | Busy cores | Ratio to comparison baseline |
|---|---:|---:|---:|---:|
| parents | 0.6535 [0.6299, 0.6771] | 0.0690 [0.0676, 0.0703] | 0.106 | 1.0000 [1.0000, 1.0000] |
| novote_oee | 0.5859 [0.5649, 0.6069] | 0.0681 [0.0666, 0.0696] | 0.116 | 0.8966 [0.8958, 0.8973] |

Full-range work (sampling estimates, without window-density assumptions):

| Quantity | Estimate | 95% interval |
|---|---:|---:|
| endpoints | 4.17758e+15 | [4.084162e+15, 4.270998e+15] |
| cases | 4.074301e+17 | [3.924455e+17, 4.224146e+17] |
| iters | 3.256209e+18 | [3.130185e+18, 3.382232e+18] |
| tree_nodes | 9.416181e+15 | [9.216014e+15, 9.616349e+15] |
| flagged_trajectories | 1060860 | [235197.6, 1886523] |
| wide_rechecks | 1060860 | [235197.6, 1886523] |

Baseline descendant diagnostic: the same sampled roots predict 4.384986e+15 baseline endpoints versus the independently known exact total 4399723745080828 (-0.335%). This does not adjust the primary estimate.

One pass through the sampled cohorts observed 8 flagged trajectories (8 distinct origins) in 6 cohorts. Every observed flag completed a wide recheck. These rechecks are included in the outer pipeline wall time. Sparse/clustered flags do not support a precise full-range frequency estimate; their displayed t interval is only a diagnostic.

Aggregate calibration joins the same roots into one continuous run. Its direct estimate is a correlated check, not an additional independent sample.

| Variant | Aggregate wall s | Aggregate / sum of cohorts | Direct scaled GPU-years |
|---|---:|---:|---:|
| parents | 155.396 | 0.9993 | 0.6530 |
| novote_oee | 139.432 | 1.0001 | 0.5859 |

One-time GPU setup: 75.195 seconds, outside trials. Ordered aggregate corpus SHA-256: `b25401329482d36d853a1a01f883d733785ce936d8c9e7aebc592a3412db2e09`.
