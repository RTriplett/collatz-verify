# Full-range checkpoint-pipeline extrapolation

Full-range N-target extrapolation of measured resident-array CPU/GPU pipeline wall time. This is not a completed full-range verification.

Fixed baseline checkpoint draws with replacement; rejected draws contribute zero work. Full-range total = exact baseline population × mean cost per baseline draw. Timing repetitions are averaged within each independent cohort. Student-t intervals use cohort degrees of freedom. Paired performance ratios use ratios of means and a paired delta-method interval. One year is 365.25 days.

Intervals describe cohort sampling and observed timing variation only; they do not include systematic desktop clock drift, long-run availability, checkpoint construction or archive I/O, setup, scheduler overhead, rare unsampled fallback/overflow work, or extrapolation to another GPU. Pipeline wall time is used because overlapping CUDA event sums are not GPU busy wall time. Cohort drain/partial-batch effects are checked by a separately reported aggregate run; no correction factor is silently applied. CPU core-years mean process CPU seconds divided by seconds per year, without an SMT conversion.

## results/compact_chain/pilot

N=72, A=10, K=38; 2 independent cohorts × 128 baseline draws, 2 timing repetitions per cohort. Exact baseline population: 1086320709. 242/256 draws survive checkpoint screening.

| Variant | GPU-years [95% interval] | CPU core-years [95% interval] | Busy cores | Ratio to comparison baseline |
|---|---:|---:|---:|---:|
| novote_oee | 0.7135 [0.5138, 0.9132] | 0.0802 [0.0734, 0.0869] | 0.112 | 1.0000 [1.0000, 1.0000] |
| early_parents | 0.7145 [0.5163, 0.9127] | 0.0385 [0.0384, 0.0387] | 0.054 | 1.0014 [0.9989, 1.0038] |
| compact66 | 0.7160 [0.4832, 0.9487] | 0.0490 [0.0484, 0.0496] | 0.068 | 1.0034 [0.9581, 1.0488] |
| compact48 | 0.7160 [0.4358, 0.9962] | 0.0492 [0.0479, 0.0506] | 0.069 | 1.0034 [0.8916, 1.1153] |

Full-range work (sampling estimates, without window-density assumptions):

| Quantity | Estimate | 95% interval |
|---|---:|---:|
| endpoints | 4.400392e+15 | [4.011471e+15, 4.789313e+15] |
| cases | 4.258687e+17 | [2.691042e+17, 5.826332e+17] |
| iters | 3.397174e+18 | [2.041299e+18, 4.75305e+18] |
| tree_nodes | 9.918426e+15 | [9.263026e+15, 1.057383e+16] |
| flagged_trajectories | 4243440 | [-4.967458e+07, 5.816146e+07] |
| wide_rechecks | 4243440 | [-4.967458e+07, 5.816146e+07] |

Baseline descendant diagnostic: the same sampled roots predict 4.561233e+15 baseline endpoints versus the independently known exact total 4399723745080828 (+3.671%). This does not adjust the primary estimate.

One pass through the sampled cohorts observed 1 flagged trajectories (1 distinct origins) in 1 cohorts. Every observed flag completed a wide recheck. These rechecks are included in the outer pipeline wall time. Sparse/clustered flags do not support a precise full-range frequency estimate; their displayed t interval is only a diagnostic.

One-time GPU setup: 85.807 seconds, outside trials. Ordered aggregate corpus SHA-256: `f5a0a45b0e6acf37f23e284ca874a632af778ce5facc8ba5c454b75ebb6f2303`.
