# CPU tail-offload experiment

Directly measured CPU/GPU continuation experiments; no new absolute full-range GPU-year estimate.

GPU-origin cases, iterations, flags and checksums include exact CPU continuations. Deferred work is a subset of these totals. Budgets and iteration counts use the existing double-iteration loop, not individual Collatz steps. A full queue leaves the candidate on the GPU. GPU time comparisons use outer pipeline wall time, including queue backpressure, CPU continuation drain and wide rechecks. Timing repetitions are averaged within each independent cohort before paired inference.

## results/tail_offload/confirmation

4 independent cohorts, 512 baseline draws (484 accepted), 2 timing repetitions. All work signatures and accounting checks passed.

| Variant | Wall ratio [95% paired interval] | Wall s | CPU core-s | Offloaded cases | CPU remainder iterations | Queue rejected |
|---|---:|---:|---:|---:|---:|---:|
| novote_oee | 1.0000 [1.0000, 1.0000] | 10.176 | 1.343 | 0 | 0 | 0 |
| defer_count | 1.0916 [1.0807, 1.1025] | 11.108 | 1.389 | 0 | 0 | 0 |
| defer24 | 1.0978 [1.0767, 1.1189] | 11.171 | 4.878 | 80,310,372 | 285,372,562 | 202,147,649 |
| defer24_cap262k_w2 | 1.0964 [1.0752, 1.1176] | 11.157 | 8.499 | 168,945,304 | 603,785,992 | 113,512,717 |
| defer16_cap262k_w4 | 1.1016 [1.0913, 1.1119] | 11.210 | 18.988 | 437,460,809 | 1,492,070,311 | 3,576,547,485 |
| compact66 | 1.0018 [0.9985, 1.0051] | 10.194 | 0.630 | 0 | 0 | 0 |
| compact_defer24_cap262k_w2 | 1.0969 [1.0811, 1.1127] | 11.162 | 7.982 | 168,945,176 | 603,793,396 | 113,512,845 |

Rows sum cohort means, retaining each original draw once. Detailed JSON includes continuation CPU time, queue wait, final drain, threshold exceedance, and continuous aggregate measurements.

Continuous aggregate: the same accepted roots concatenated; this measures pipeline drain overhead and is not an independent sample.

| Variant | Wall s | CPU core-s | CPU continuation s | Queue wait s | Final drain s |
|---|---:|---:|---:|---:|---:|
| novote_oee | 10.199 | 1.376 | 0.000 | 0.000 | 0.000 |
| defer_count | 11.000 | 1.433 | 0.000 | 0.000 | 0.000 |
| defer24 | 10.979 | 4.752 | 3.324 | 0.005 | 0.004 |
| defer24_cap262k_w2 | 10.966 | 8.305 | 6.783 | 0.003 | 0.017 |
| defer16_cap262k_w4 | 11.014 | 18.749 | 16.797 | 0.004 | 0.013 |
| compact66 | 10.154 | 0.656 | 0.000 | 0.000 | 0.000 |
| compact_defer24_cap262k_w2 | 10.986 | 7.740 | 6.815 | 0.006 | 0.014 |
