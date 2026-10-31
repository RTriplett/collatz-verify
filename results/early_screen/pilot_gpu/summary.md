# Early GPU screening and CPU verification

Directly measured costs only; no full-range GPU-year or rental-price extrapolation.

Logical GPU-origin totals include the exact CPU-finished survivors; early work is a subset, never additional work.

CPU workers, producer threads, CUDA compute and copies overlap. Do not add their times into an elapsed-time breakdown.

early_copy_seconds sums CUDA stream event intervals around D2H copies; it is not exclusive device busy time.

early_host_copy_seconds is host-thread CPU time scanning mask popcounts and copying compact jobs; early_cpu_seconds is summed worker CPU time. Decode/group/verification are summed worker elapsed intervals, not exclusive process CPU time.

Every-Nth launch selection starts at launch zero and resets each run. Producer interleaving can change selected classes and actual CPU work between variants or repetitions. Compare recorded class mix and CPU counts; identical selected states are established only by separate fixture replay.

Continuous aggregates concatenate the same cohort roots. Repeats share work; their min/mean/max describe measured variation, not independent sampling uncertainty.

Warmup and setup are excluded. Every summarized trial must report zero captured states.

## /home/ryan/Projects/Collatz/distributed/results/early_screen/pilot_gpu

512 baseline draws, 485 accepted; all exact-work and accounting audits passed.

Diagnostic capture was requested or occurred during warmup. Warmup is excluded; all summarized trials captured zero states. Keep the capture context when interpreting this run.

Continuous aggregate means; wall range is min–max across repeated timings.

| Variant | Wall s [range] | Wall / baseline | Process CPU core-s | CPU tail cases | CPU iteration share | D2H GiB |
|---|---:|---:|---:|---:|---:|---:|
| novote_oee | 9.8752 [9.7821–9.9684] | 1.0000 | 1.455 | 0 | 0.00000% | 0.0000 |
| early1024_scalar_w4 | 10.0463 [10.0369–10.0556] | 1.0173 | 9.095 | 82,867,657 | 0.03685% | 0.3125 |
| early1024_bucket10_w4 | 10.1068 [10.0641–10.1495] | 1.0235 | 7.170 | 57,706,130 | 0.02509% | 0.3125 |
| early1024_table10_w4 | 10.0493 [10.0464–10.0522] | 1.0176 | 7.198 | 57,699,894 | 0.02509% | 0.3125 |
| early1024_group10_w4 | 10.1691 [10.0270–10.3111] | 1.0298 | 8.539 | 69,893,888 | 0.03078% | 0.3125 |
| early1024_regroup10_w4 | 10.1115 [10.0053–10.2176] | 1.0239 | 10.513 | 82,859,706 | 0.03684% | 0.3125 |

Separate timing measurements below overlap and must not be added together.

| Variant | D2H event s | Host scan/copy CPU s | Worker CPU s | Decode elapsed s | Group elapsed s | Verify elapsed s | Queue send s | Drain s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| novote_oee | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| early1024_scalar_w4 | 0.0128 | 0.2465 | 7.4306 | 0.6912 | 0.0000 | 6.7397 | 0.0000 | 0.0001 |
| early1024_bucket10_w4 | 0.0128 | 0.2455 | 5.4987 | 0.5915 | 0.2603 | 4.6456 | 0.0000 | 0.0001 |
| early1024_table10_w4 | 0.0128 | 0.2468 | 5.5272 | 0.5892 | 0.0000 | 4.9379 | 0.0000 | 0.0001 |
| early1024_group10_w4 | 0.0128 | 0.2604 | 6.8400 | 0.6433 | 0.3676 | 5.8370 | 0.0000 | 0.0001 |
| early1024_regroup10_w4 | 0.0128 | 0.2461 | 8.8555 | 0.6863 | 1.9145 | 6.2484 | 0.0000 | 0.0001 |

JSON retains all min/mean/max metrics, individual repetitions, selected per-class work, paired cohort ratios, corpus/work fingerprints, configuration and raw runner metadata. Different CPU methods can receive different selected trajectories in these pipeline runs; use the separate fixed-state CPU replay to isolate their algorithmic costs.
