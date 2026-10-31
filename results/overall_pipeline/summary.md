# Overall pipeline benchmark

Resident-array CPU→GPU pipeline wall times include expansion, packing, queueing, transfers, GPU execution and CPU fallback. Checkpoint preparation, archive I/O and table/array setup are excluded from trials and recorded separately. N72 target runs are sampled work packages.

GPU kernel seconds sum CUDA stream event intervals. Concurrent streams overlap, so these totals are not GPU busy wall time.

Speedup and CPU ratios are paired by repetition against full within each run. Different policies intentionally have different work and checksums. Runs with different thread counts or corpora are kept separate. Reported ranges are observed extrema, not confidence intervals.

## results/overall_pipeline/validation

6 complete-range variants passed CPU/GPU and same-policy checks.

| Variant | Range | Endpoints | Cases | CPU fallback endpoints | Hash |
|---|---:|---:|---:|---:|---|
| full | 2^32 | 52,176 | 18,615,811 | 29,310 | 1A7D4E28 |
| screened | 2^32 | 52,176 | 18,615,811 | 29,310 | 1A7D4E28 |
| reference16 | 2^32 | 53,406 | 18,803,234 | 29,492 | 3CBADFDE |
| compact16 | 2^32 | 53,406 | 18,803,234 | 29,492 | 3CBADFDE |
| terminal16x4 | 2^32 | 53,406 | 18,803,234 | 29,492 | 3CBADFDE |
| terminal16x6 | 2^32 | 53,406 | 18,803,234 | 29,492 | 3CBADFDE |

## results/overall_pipeline/validation_device

6 complete-range variants passed CPU/GPU and same-policy checks.

| Variant | Range | Endpoints | Cases | CPU fallback endpoints | Hash |
|---|---:|---:|---:|---:|---|
| full | 2^32 | 52,176 | 18,317,914 | 29,310 | 4510A8AC |
| screened | 2^32 | 52,176 | 18,317,914 | 29,310 | 4510A8AC |
| reference16 | 2^32 | 53,406 | 18,494,045 | 29,492 | CEA9098C |
| compact16 | 2^32 | 53,406 | 18,494,045 | 29,492 | CEA9098C |
| terminal16x4 | 2^32 | 53,406 | 18,494,045 | 29,492 | CEA9098C |
| terminal16x6 | 2^32 | 53,406 | 18,494,045 | 29,492 | CEA9098C |

## results/overall_pipeline/main8_desktop

N=72, A=10, M=62; 1024 common K=38 roots; 8 workers; 3 repetitions. Corpus SHA-256: `7b0b5acf7274d1697f2ccdcf5a4d1ea8d33d0e617f5478b417416dcc97f30363`.

GPU setup: 89.745 s; device array build: 89.366 s; expansion-engine setup: 0.004 s.

| Variant | Wall median [min, max], s | CPU median, core-s | Busy CPU cores | Paired speedup vs full | Paired CPU reduction |
|---|---:|---:|---:|---:|---:|
| full | 28.9553 [27.7656, 31.4911] | 213.8457 | 7.281 | 1.000× | 0.00% |
| screened | 29.9726 [29.8090, 32.2372] | 212.3607 | 7.085 | 0.931× | 3.14% |
| compact54 | 26.3792 [26.2260, 26.7274] | 74.8765 | 2.838 | 1.083× | 64.61% |
| combined54 | 26.4409 [26.2607, 26.5264] | 31.5128 | 1.191 | 1.103× | 85.39% |
| compact50 | 26.4634 [26.2972, 26.4944] | 73.2714 | 2.769 | 1.101× | 65.74% |
| combined50 | 26.5485 [26.2399, 26.5617] | 27.8029 | 1.060 | 1.103× | 87.00% |

| Variant | Endpoints | Cases | Iterations | Transfer bytes | CPU fallback endpoints | Hash |
|---|---:|---:|---:|---:|---:|---|
| full | 4,201,887,155 | 423,675,010,079 | 3,389,613,246,168 | 67,230,194,480 | 0 | CF7B0C23 |
| screened | 4,201,887,155 | 423,675,010,079 | 3,389,613,246,168 | 67,230,194,480 | 0 | CF7B0C23 |
| compact54 | 4,299,192,388 | 426,191,875,427 | 3,407,832,363,302 | 68,787,078,208 | 0 | 7BCE1D4D |
| combined54 | 4,299,192,388 | 426,191,875,427 | 3,407,832,363,302 | 68,787,078,208 | 0 | 7BCE1D4D |
| compact50 | 4,325,704,552 | 427,313,317,731 | 3,416,237,911,342 | 69,211,272,832 | 0 | F782B001 |
| combined50 | 4,325,704,552 | 427,313,317,731 | 3,416,237,911,342 | 69,211,272,832 | 0 | F782B001 |

Counts, every tree counter, and hashes match across repetitions and within each pruning policy. Policies with only one selected implementation have repetition consistency checks only.

## results/overall_pipeline/two_cores

N=72, A=10, M=62; 1024 common K=38 roots; 2 workers; 3 repetitions. Corpus SHA-256: `7b0b5acf7274d1697f2ccdcf5a4d1ea8d33d0e617f5478b417416dcc97f30363`.

GPU setup: 89.486 s; device array build: 89.099 s; expansion-engine setup: 0.003 s.

| Variant | Wall median [min, max], s | CPU median, core-s | Busy CPU cores | Paired speedup vs full | Paired CPU reduction |
|---|---:|---:|---:|---:|---:|
| combined54 | 26.5171 [26.4228, 26.5206] | 30.4907 | 1.153 | — | — |
| combined50 | 26.5348 [26.5099, 26.5775] | 29.6048 | 1.114 | — | — |

| Variant | Endpoints | Cases | Iterations | Transfer bytes | CPU fallback endpoints | Hash |
|---|---:|---:|---:|---:|---:|---|
| combined54 | 4,299,192,388 | 426,191,875,427 | 3,407,832,363,302 | 68,787,078,208 | 0 | 7BCE1D4D |
| combined50 | 4,325,704,552 | 427,313,317,731 | 3,416,237,911,342 | 69,211,272,832 | 0 | F782B001 |

Counts, every tree counter, and hashes match across repetitions and within each pruning policy. Policies with only one selected implementation have repetition consistency checks only.
