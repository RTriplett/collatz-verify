# Selective deeper expansion by original F

Expanding only the lowest A10 class, **F = 40**, captures **98.48%** of the baseline candidate reduction available from one extra bit, or **95.46%** from two extra bits. This class contains **22.90%** of the M62 endpoints. It is the clear first class to investigate for selective deeper expansion.

These are exact continuation counts for the saved sample, not measured savings from a mixed-A GPU pipeline. Such a pipeline is not implemented by this experiment.

## Method

The probe uses the same 1,024 K38 packages as the transport and earlier A-pipeline experiments: seed 20261023, full improved checks through K50, terminal entry K52, and target M62. The ordered root records were checked against `results/a_pipeline/a10_start/pipeline.jsonl`. Their corpus SHA-256 is `7b0b5acf7274d1697f2ccdcf5a4d1ea8d33d0e617f5478b417416dcc97f30363`.

It records visits to the resident terminal-word groups, then aggregates each word's final `(F, oee, ne)` state. Baseline continuation tables count descendants at M63 and M64. The probe visits 7,213,167 emitting K52 parents and 46,175,333 groups, of which only 486 group descriptors are distinct; it does not reconstruct all 4,325,704,552 endpoints. The recorded probe took 0.268 seconds, outside the GPU trial timings.

For an original class containing E endpoints and D descendants after b additional bits, the retained candidate fraction is **D / (2^b E)**. Each descendant represents half or one quarter as many starting values in the fixed N72 range. This counts candidates before lookahead screening and tail execution.

## Counts by original M62 class

Fractions are rounded to six decimals; the raw record retains full precision.

| Original F | M62 endpoints | M63 descendants | Retained after 1 bit | M64 descendants | Retained after 2 bits |
|---:|---:|---:|---:|---:|---:|
| 40 | 990,486,807 | 1,579,074,543 | 0.797120 | 2,569,561,350 | 0.648560 |
| 41 | 1,163,566,073 | 2,323,547,649 | 0.998460 | 4,599,492,314 | 0.988232 |
| 42 | 931,873,032 | 1,861,451,718 | 0.998769 | 3,719,318,939 | 0.997807 |
| 43 | 607,693,092 | 1,215,112,450 | 0.999775 | 2,427,930,554 | 0.998831 |
| 44 | 342,001,024 | 683,979,349 | 0.999967 | 1,367,684,964 | 0.999767 |
| 45 | 170,195,992 | 340,376,891 | 0.999956 | 680,731,083 | 0.999922 |
| 46 | 75,485,591 | 150,968,995 | 0.999986 | 301,922,897 | 0.999936 |
| 47 | 29,820,023 | 59,639,769 | 0.999995 | 119,277,351 | 0.999977 |
| 48 | 10,405,276 | 20,810,524 | 0.999999 | 41,620,771 | 0.999992 |
| 49 | 3,157,622 | 6,315,230 | 0.999998 | 12,630,432 | 0.999996 |
| 50 | 814,402 | 1,628,804 | 1.000000 | 3,257,594 | 0.999996 |
| 51 | 172,899 | 345,798 | 1.000000 | 691,596 | 1.000000 |
| 52 | 28,881 | 57,762 | 1.000000 | 115,524 | 1.000000 |
| 53 | 3,544 | 7,088 | 1.000000 | 14,176 | 1.000000 |
| 54 | 283 | 566 | 1.000000 | 1,132 | 1.000000 |
| 55 | 11 | 22 | 1.000000 | 44 | 1.000000 |
| **All** | **4,325,704,552** | **8,243,317,158** | **0.952829** | **15,844,250,721** | **0.915703** |

## Selective continuation versus expanding every class

The following totals retain M62 endpoints in unselected classes and replace selected endpoints with their deeper descendants. Candidate reductions are relative to the original A10 candidate population. Endpoint growth counts records at their respective depths and therefore is not itself a candidate count.

| Extra bits | Classes expanded | Overall candidate reduction | Share of all-class reduction captured | Endpoint count factor |
|---:|:---|---:|---:|---:|
| 1 | F40 | 4.645% | 98.48% | 1.136× |
| 1 | F40–41 | 4.687% | 99.36% | 1.404× |
| 1 | All | 4.717% | 100.00% | 1.906× |
| 2 | F40 | 8.047% | 95.46% | 1.365× |
| 2 | F40–41 | 8.364% | 99.22% | 2.159× |
| 2 | All | 8.430% | 100.00% | 3.663× |

Within F40 itself, one extra bit removes **20.29%** of candidates and two remove **35.14%**. Selectively applying those continuations increases the total endpoint count by **13.61%** or **36.50%**, compared with **90.57%** or **266.28%** when every class is expanded.

F41 has a much smaller payoff: **0.154%** or **1.177%** of its own candidates. Expanding it together with F40 raises the total endpoint factors to 1.404× or 2.159×, while adding only 0.041 or 0.317 percentage points of overall candidate reduction beyond F40 alone. Higher classes offer still less pruning.

## Scope and GPU follow-up

The total M63 and M64 descendant counts exactly match the earlier global A9 and A8 endpoint counts. The retained candidate fractions do **not** equal the earlier tail-case or tail-iteration ratios: for example, global A8 removes 8.430% of these baseline candidates, whereas that benchmark measured 7.729% fewer lookahead-surviving tail cases and 8.106% fewer tail iterations. Different work measures should not be substituted for one another.

The subsequent [serialized GPU probe](class_gpu_large_summary.md) tests 16,777,216 conditional F40 sources with seven repetitions. One extra bit reduces kernel time by 5.31%; two increase it by 7.14%, relative to the opening A10 control. Opening and closing controls differ by only 0.02%. This probe uses replay16, B32/B35 relative-class windows and a different source corpus without K38 walkback. Host expansion, transfers and setup are excluded. See [the experiment report](README.md#selective-deeper-expansion) for details and the approximately 4 GiB additional array requirement for integration.

Neither the counts nor the kernel microprobe establish an overall wall-time improvement. A mixed-depth implementation must route descendants using their new depth and final F, apply the corresponding lookahead windows and cutoffs, and account for expansion, reconstruction, traffic, scheduling, and any CPU fallback. The candidate-count probe itself applies no lookahead windows and measures none of those GPU costs. Its fixed sample is also not a complete verification of the N72 range.

The implementation is `Affine::class_probe` in [affine.rs](../../scripts/expansion_bench/affine.rs); its test compares per-class counts with explicit baseline DFS continuations. The raw source is the `class_probe` record in [main8/pipeline.jsonl](main8/pipeline.jsonl). The earlier global measurements are in [the A-pipeline results](../a_pipeline/README.md).
