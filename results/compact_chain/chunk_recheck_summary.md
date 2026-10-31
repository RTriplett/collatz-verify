# Full-range checkpoint-pipeline extrapolation

Full-range N-target extrapolation of measured resident-array CPU/GPU pipeline wall time. This is not a completed full-range verification.

Fixed baseline checkpoint draws with replacement; rejected draws contribute zero work. Full-range total = exact baseline population × mean cost per baseline draw. Timing repetitions are averaged within each independent cohort. Student-t intervals use cohort degrees of freedom. Paired performance ratios use ratios of means and a paired delta-method interval. One year is 365.25 days.

Intervals describe cohort sampling and observed timing variation only; they do not include systematic desktop clock drift, long-run availability, checkpoint construction or archive I/O, setup, scheduler overhead, rare unsampled fallback/overflow work, or extrapolation to another GPU. Pipeline wall time is used because overlapping CUDA event sums are not GPU busy wall time. Cohort drain/partial-batch effects are checked by a separately reported aggregate run; no correction factor is silently applied. CPU core-years mean process CPU seconds divided by seconds per year, without an SMT conversion.

## results/compact_chain/chunk_recheck

N=72, A=10, K=38; 4 independent cohorts × 256 baseline draws, 1 timing repetitions per cohort. Exact baseline population: 1086320709. 955/1,024 draws survive checkpoint screening.

| Variant | GPU-years [95% interval] | CPU core-years [95% interval] | Busy cores | Ratio to comparison baseline |
|---|---:|---:|---:|---:|
| novote_oee | 0.6780 [0.5513, 0.8046] | 0.0771 [0.0684, 0.0859] | 0.114 | 1.0000 [1.0000, 1.0000] |
| novote_oee_1k | 0.6845 [0.5520, 0.8170] | 0.0740 [0.0666, 0.0814] | 0.108 | 1.0096 [0.9999, 1.0193] |
| compact66 | 0.6970 [0.5577, 0.8362] | 0.0399 [0.0360, 0.0437] | 0.057 | 1.0280 [0.9790, 1.0770] |
| compact66_1k | 0.6874 [0.5557, 0.8191] | 0.0416 [0.0352, 0.0479] | 0.060 | 1.0139 [0.9902, 1.0377] |

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
| novote_oee | 20.671 | 1.0250 | 0.6949 |
| novote_oee_1k | 20.761 | 1.0196 | 0.6979 |
| compact66 | 20.661 | 0.9965 | 0.6946 |
| compact66_1k | 20.388 | 0.9970 | 0.6854 |
| novote_oee_1k | 20.453 | 1.0045 | 0.6876 |
| compact66 | 20.408 | 0.9844 | 0.6861 |
| compact66_1k | 20.392 | 0.9972 | 0.6855 |
| novote_oee | 20.427 | 1.0128 | 0.6867 |

One-time GPU setup: 87.937 seconds, outside trials. Ordered aggregate corpus SHA-256: `83477b76d8fc1ab57d3d7e25265bfb7853f67d2a92a849d9afc431ba91fa3a9a`.
