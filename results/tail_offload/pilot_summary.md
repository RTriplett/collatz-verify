# CPU tail-offload experiment

Directly measured CPU/GPU continuation experiments; no new absolute full-range GPU-year estimate.

GPU-origin cases, iterations, flags and checksums include exact CPU continuations. Deferred work is a subset of these totals. Budgets and iteration counts use the existing double-iteration loop, not individual Collatz steps. A full queue leaves the candidate on the GPU. GPU time comparisons use outer pipeline wall time, including queue backpressure, CPU continuation drain and wide rechecks. Timing repetitions are averaged within each independent cohort before paired inference.

## results/tail_offload/pilot

2 independent cohorts, 256 baseline draws (238 accepted), 2 timing repetitions. All work signatures and accounting checks passed.

| Variant | Wall ratio [95% paired interval] | Wall s | CPU core-s | Offloaded cases | CPU remainder iterations | Queue rejected |
|---|---:|---:|---:|---:|---:|---:|
| novote_oee | 1.0000 [1.0000, 1.0000] | 5.093 | 0.721 | 0 | 0 | 0 |
| defer_count | 1.0901 [1.0469, 1.1333] | 5.552 | 0.717 | 0 | 0 | 0 |
| defer8 | 1.8030 [1.6032, 2.0028] | 9.183 | 3.295 | 62,720,241 | 202,024,628 | 30,716,348,950 |
| defer16 | 1.1085 [1.0725, 1.1445] | 5.646 | 3.180 | 62,627,552 | 213,169,156 | 1,921,559,606 |
| defer24 | 1.0860 [0.9726, 1.1994] | 5.531 | 2.364 | 40,757,566 | 144,781,202 | 97,818,358 |
| defer32 | 1.0911 [1.0769, 1.1053] | 5.557 | 1.180 | 8,976,924 | 32,901,056 | 1,373,231 |
| defer64 | 1.0923 [1.0897, 1.0950] | 5.563 | 0.736 | 474 | 1,872 | 0 |
| defer16_w2 | 1.0973 [0.9718, 1.2228] | 5.589 | 3.357 | 62,627,630 | 213,162,306 | 1,921,559,528 |
| compact66 | 1.0209 [0.8639, 1.1780] | 5.200 | 0.335 | 0 | 0 | 0 |
| compact_defer24 | 1.0906 [1.0280, 1.1532] | 5.554 | 2.201 | 40,757,188 | 144,763,214 | 97,818,736 |

Rows sum cohort means, retaining each original draw once. Detailed JSON includes continuation CPU time, queue wait, final drain, threshold exceedance, and continuous aggregate measurements.

Continuous aggregate: the same accepted roots concatenated; this measures pipeline drain overhead and is not an independent sample.

| Variant | Wall s | CPU core-s | CPU continuation s | Queue wait s | Final drain s |
|---|---:|---:|---:|---:|---:|
| novote_oee | 5.067 | 0.754 | 0.000 | 0.000 | 0.000 |
| defer_count | 5.486 | 0.754 | 0.000 | 0.000 | 0.000 |
| defer8 | 8.953 | 3.283 | 2.486 | 0.013 | 0.007 |
| defer16 | 5.540 | 3.164 | 2.428 | 0.006 | 0.004 |
| defer24 | 5.517 | 2.427 | 1.690 | 0.007 | 0.001 |
| defer32 | 5.526 | 1.227 | 0.452 | 0.003 | 0.000 |
| defer64 | 5.522 | 0.772 | 0.000 | 0.001 | 0.000 |
| defer16_w2 | 5.530 | 3.280 | 2.523 | 0.002 | 0.003 |
| compact66 | 5.125 | 0.340 | 0.000 | 0.000 | 0.000 |
| compact_defer24 | 5.529 | 2.105 | 1.708 | 0.008 | 0.001 |
