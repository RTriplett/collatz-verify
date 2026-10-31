# Full-range checkpoint-pipeline extrapolation

Full-range N-target extrapolation of measured resident-array CPU/GPU pipeline wall time. This is not a completed full-range verification.

Fixed baseline checkpoint draws with replacement; rejected draws contribute zero work. Full-range total = exact baseline population × mean cost per baseline draw. Timing repetitions are averaged within each independent cohort. Student-t intervals use cohort degrees of freedom. Paired performance ratios use ratios of means and a paired delta-method interval. One year is 365.25 days.

Intervals describe cohort sampling and observed timing variation only; they do not include systematic desktop clock drift, long-run availability, checkpoint construction or archive I/O, setup, scheduler overhead, rare unsampled fallback/overflow work, or extrapolation to another GPU. Pipeline wall time is used because overlapping CUDA event sums are not GPU busy wall time. Cohort drain/partial-batch effects are checked by a separately reported aggregate run; no correction factor is silently applied. CPU core-years mean process CPU seconds divided by seconds per year, without an SMT conversion.

## results/grouped_transitions/followup

N=72, A=10, K=38; 4 independent cohorts × 128 baseline draws, 2 timing repetitions per cohort. Exact baseline population: 1086320709. 479/512 draws survive checkpoint screening.

| Variant | GPU-years [95% interval] | CPU core-years [95% interval] | Busy cores | Ratio to comparison baseline |
|---|---:|---:|---:|---:|
| novote_oee | 0.6676 [0.5617, 0.7736] | 0.0712 [0.0679, 0.0746] | 0.107 | 1.0000 [1.0000, 1.0000] |
| affine32 | 0.8531 [0.7397, 0.9665] | 1.5531 [1.4499, 1.6563] | 1.821 | 1.2778 [1.2304, 1.3251] |
| chain6 | 0.8873 [0.7547, 1.0198] | 1.6459 [1.4984, 1.7934] | 1.855 | 1.3290 [1.3110, 1.3469] |
| group6 | 0.7745 [0.6434, 0.9056] | 1.3231 [1.2242, 1.4221] | 1.708 | 1.1600 [1.1363, 1.1838] |
| jump8 | 1.3099 [1.1125, 1.5074] | 0.0738 [0.0656, 0.0821] | 0.056 | 1.9620 [1.9098, 2.0142] |
| jump12 | 1.6568 [1.3092, 2.0044] | 0.0780 [0.0712, 0.0849] | 0.047 | 2.4815 [2.3324, 2.6306] |

Full-range work (sampling estimates, without window-density assumptions):

| Quantity | Estimate | 95% interval |
|---|---:|---:|
| endpoints | 4.04511e+15 | [3.784027e+15, 4.306193e+15] |
| cases | 3.921278e+17 | [3.334059e+17, 4.508497e+17] |
| iters | 3.133197e+18 | [2.628125e+18, 3.638269e+18] |
| tree_nodes | 9.126498e+15 | [8.584241e+15, 9.668755e+15] |
| flagged_trajectories | 0 | [0, 0] |
| wide_rechecks | 0 | [0, 0] |

Baseline descendant diagnostic: the same sampled roots predict 4.261288e+15 baseline endpoints versus the independently known exact total 4399723745080828 (-3.146%). This does not adjust the primary estimate.

One pass through the sampled cohorts observed 0 flagged trajectories (0 distinct origins) in 0 cohorts. Every observed flag completed a wide recheck. These rechecks are included in the outer pipeline wall time. Sparse/clustered flags do not support a precise full-range frequency estimate; their displayed t interval is only a diagnostic.

Aggregate calibration joins the same roots into one continuous run. Its direct estimate is a correlated check, not an additional independent sample.

| Variant | Aggregate wall s | Aggregate / sum of cohorts | Direct scaled GPU-years |
|---|---:|---:|---:|
| novote_oee | 10.231 | 1.0303 | 0.6879 |
| affine32 | 12.585 | 0.9918 | 0.8461 |
| chain6 | 13.123 | 0.9944 | 0.8823 |
| group6 | 11.237 | 0.9755 | 0.7555 |
| jump8 | 19.246 | 0.9878 | 1.2940 |
| jump12 | 24.672 | 1.0012 | 1.6588 |

One-time GPU setup: 87.931 seconds, outside trials. Ordered aggregate corpus SHA-256: `fd6800f03ed19f6876725bbba497baf94d0d01b45968570427ad1e90017d6392`.
