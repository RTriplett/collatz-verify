# Full-range checkpoint-pipeline extrapolation

Full-range N-target extrapolation of measured resident-array CPU/GPU pipeline wall time. This is not a completed full-range verification.

Fixed baseline checkpoint draws with replacement; rejected draws contribute zero work. Full-range total = exact baseline population × mean cost per baseline draw. Timing repetitions are averaged within each independent cohort. Student-t intervals use cohort degrees of freedom. Paired performance ratios use ratios of means and a paired delta-method interval. One year is 365.25 days.

Intervals describe cohort sampling and observed timing variation only; they do not include systematic desktop clock drift, long-run availability, checkpoint construction or archive I/O, setup, scheduler overhead, rare unsampled fallback/overflow work, or extrapolation to another GPU. Pipeline wall time is used because overlapping CUDA event sums are not GPU busy wall time. Cohort drain/partial-batch effects are checked by a separately reported aggregate run; no correction factor is silently applied. CPU core-years mean process CPU seconds divided by seconds per year, without an SMT conversion.

## results/compact_chain/confirmation

N=72, A=10, K=38; 4 independent cohorts × 256 baseline draws, 2 timing repetitions per cohort. Exact baseline population: 1086320709. 955/1,024 draws survive checkpoint screening.

| Variant | GPU-years [95% interval] | CPU core-years [95% interval] | Busy cores | Ratio to comparison baseline |
|---|---:|---:|---:|---:|
| novote_oee | 0.5920 [0.4822, 0.7019] | 0.0766 [0.0672, 0.0859] | 0.129 | 1.0000 [1.0000, 1.0000] |
| early_parents | 0.5873 [0.4813, 0.6933] | 0.0353 [0.0298, 0.0408] | 0.060 | 0.9920 [0.9746, 1.0095] |
| compact66 | 0.5912 [0.4821, 0.7004] | 0.0390 [0.0331, 0.0450] | 0.066 | 0.9986 [0.9749, 1.0224] |
| compact48 | 0.5904 [0.4826, 0.6982] | 0.0392 [0.0339, 0.0446] | 0.066 | 0.9972 [0.9793, 1.0152] |

Full-range work (sampling estimates, without window-density assumptions):

| Quantity | Estimate | 95% interval |
|---|---:|---:|
| endpoints | 4.129541e+15 | [3.679914e+15, 4.579169e+15] |
| cases | 4.076911e+17 | [3.326633e+17, 4.827189e+17] |
| iters | 3.26581e+18 | [2.631267e+18, 3.900353e+18] |
| tree_nodes | 9.302516e+15 | [8.343946e+15, 1.026109e+16] |
| flagged_trajectories | 2121720 | [-4630540, 8873981] |
| wide_rechecks | 2121720 | [-4630540, 8873981] |

Baseline descendant diagnostic: the same sampled roots predict 4.340829e+15 baseline endpoints versus the independently known exact total 4399723745080828 (-1.339%). This does not adjust the primary estimate.

One pass through the sampled cohorts observed 2 flagged trajectories (2 distinct origins) in 1 cohorts. Every observed flag completed a wide recheck. These rechecks are included in the outer pipeline wall time. Sparse/clustered flags do not support a precise full-range frequency estimate; their displayed t interval is only a diagnostic.

Aggregate calibration joins the same roots into one continuous run. Its direct estimate is a correlated check, not an additional independent sample.

| Variant | Aggregate wall s | Aggregate / sum of cohorts | Direct scaled GPU-years |
|---|---:|---:|---:|
| novote_oee | 18.011 | 1.0226 | 0.6055 |
| early_parents | 20.396 | 1.1674 | 0.6856 |
| compact66 | 20.520 | 1.1667 | 0.6898 |
| compact48 | 20.834 | 1.1862 | 0.7004 |

One-time GPU setup: 74.871 seconds, outside trials. Ordered aggregate corpus SHA-256: `83477b76d8fc1ab57d3d7e25265bfb7853f67d2a92a849d9afc431ba91fa3a9a`.
