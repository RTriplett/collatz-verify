# Separate allowance for rare CPU tails

A separately measured scalar CPU workload allowance for all baseline N72/A10 endpoints in F56–62, without improved ancestry or K38 walkback screening. It is not a GPU performance estimate or a correction factor for the main checkpoint sample.

Fixed conditional baseline endpoint draws with replacement. Average repetitions within each independent cohort, then multiply the mean per-draw cost by the exact population 49,063,601. Approximate Student-t95 intervals use cohort degrees of freedom; one year is 365.25 days. The total includes wide rechecking, with tail and recheck timings also reported separately.

Omitting improved screening makes the selected workload a conservative superset of the experimental pipeline's rare endpoints; the sampled runtime estimate is not a rigorous upper bound. It excludes endpoint generation, checkpoint preprocessing, storage, scheduling and GPU work. Rare/clustered flags and high-F strata may be sampled imprecisely; cohort intervals do not include systematic host performance differences or long-term load. CPU/GPU overlap is not measured here.

Exact conditional baseline population: 49,063,601 M62 endpoints with F56–62. 16 cohorts × 4096 draws; 2 repetitions per cohort. Every draw is retained, including duplicates.

| CPU stage | Full-population wall seconds [95% interval] | CPU core-years [95% interval] |
|---|---:|---:|
| Total tail and wide rechecks | 4913.842 [4894.372, 4933.311] | 0.0001557 [0.0001551, 0.0001563] |
| Scalar tail | 4913.801 [4894.333, 4933.270] | 0.0001557 [0.0001551, 0.0001563] |
| Wide rechecks | 0.014 [0.009, 0.020] | 0.0000000 [0.0000000, 0.0000000] |

One pass through the sampled endpoints contains 37,282,698 cases and 639,427,584 tail iterations. All 7 flagged trajectories (7 distinct origins) finish wide rechecking, totaling 2,814 wide steps. Repetitions agree on the full work signature; the one-pass XOR checksum is `72191D6C`.

If a GPU remained allocated and completely idle throughout this CPU work, the **hypothetical no-overlap allocation addition** would be 0.0001557 GPU-years [0.0001551, 0.0001563]. This is a pessimistic accounting scenario, not GPU computation measured by this probe. No allowance is silently added to the main pipeline estimate.

Ordered corpus SHA-256: `013c22c6bd229bc067a25099f167f4026f7a4e2bd7d61001f68d064c2cb87bc2`. Setup takes 0.000618 seconds and is outside the timed samples.
