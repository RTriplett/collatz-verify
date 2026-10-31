# Full-range checkpoint-pipeline extrapolation

Full-range N-target extrapolation of measured resident-array CPU/GPU pipeline wall time. This is not a completed full-range verification.

Fixed baseline checkpoint draws with replacement; rejected draws contribute zero work. Full-range total = exact baseline population × mean cost per baseline draw. Timing repetitions are averaged within each independent cohort. Student-t intervals use cohort degrees of freedom. Paired performance ratios use ratios of means and a paired delta-method interval. One year is 365.25 days.

Intervals describe cohort sampling and observed timing variation only; they do not include systematic desktop clock drift, long-run availability, checkpoint construction or archive I/O, setup, scheduler overhead, rare unsampled fallback/overflow work, or extrapolation to another GPU. Pipeline wall time is used because overlapping CUDA event sums are not GPU busy wall time. Cohort drain/partial-batch effects are checked by a separately reported aggregate run; no correction factor is silently applied. CPU core-years mean process CPU seconds divided by seconds per year, without an SMT conversion.

## results/compact_chain/continuous_recheck

N=72, A=10, K=38; 4 independent cohorts × 256 baseline draws, 1 timing repetitions per cohort. Exact baseline population: 1086320709. 955/1,024 draws survive checkpoint screening.

| Variant | GPU-years [95% interval] | CPU core-years [95% interval] | Busy cores | Ratio to comparison baseline |
|---|---:|---:|---:|---:|
| compact48 | 0.6816 [0.5528, 0.8105] | 0.0404 [0.0345, 0.0463] | 0.059 | 1.0000 [1.0000, 1.0000] |
| compact66 | 0.6863 [0.5659, 0.8067] | 0.0404 [0.0364, 0.0444] | 0.059 | 1.0068 [0.9794, 1.0342] |
| early_parents | 0.6896 [0.5649, 0.8142] | 0.0373 [0.0329, 0.0417] | 0.054 | 1.0116 [0.9742, 1.0491] |
| novote_oee | 0.6975 [0.5498, 0.8452] | 0.0704 [0.0650, 0.0757] | 0.101 | 1.0232 [0.9756, 1.0709] |

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
| compact48 | 20.677 | 1.0197 | 0.6951 |
| compact66 | 20.483 | 1.0034 | 0.6886 |
| early_parents | 20.215 | 0.9855 | 0.6796 |
| novote_oee | 20.317 | 0.9792 | 0.6830 |
| compact66 | 20.697 | 1.0138 | 0.6957 |
| early_parents | 20.349 | 0.9920 | 0.6841 |
| novote_oee | 20.282 | 0.9775 | 0.6818 |
| compact48 | 20.338 | 1.0030 | 0.6837 |
| early_parents | 20.630 | 1.0057 | 0.6935 |
| novote_oee | 20.364 | 0.9815 | 0.6846 |
| compact48 | 20.071 | 0.9898 | 0.6747 |
| compact66 | 19.980 | 0.9787 | 0.6717 |
| novote_oee | 19.911 | 0.9597 | 0.6694 |
| compact48 | 20.156 | 0.9940 | 0.6776 |
| compact66 | 20.007 | 0.9800 | 0.6726 |
| early_parents | 20.427 | 0.9958 | 0.6867 |

One-time GPU setup: 87.262 seconds, outside trials. Ordered aggregate corpus SHA-256: `83477b76d8fc1ab57d3d7e25265bfb7853f67d2a92a849d9afc431ba91fa3a9a`.
