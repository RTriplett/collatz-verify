# Transport and terminal reuse experiments

Resident-array complete CPU/GPU pipeline; sampled N72 packages, not full 2^72 verification.

Observed ranges are not confidence intervals. Paired ratios use repetition within one run. CUDA event intervals overlap. Parent traffic includes repeated group headers and batch splits.

## results/transport_pipeline/main8

1024 K38 roots; 8 workers; A=10; cutoff50; table entry52; 3 rotating repetitions.

| Variant | Wall s, median [min, max] | CPU core-s, median [min, max] | Busy cores | Transfer GB | Paired wall ratio |
|---|---:|---:|---:|---:|---:|
| replay16 | 24.819 [24.499, 24.890] | 28.943 [28.907, 28.978] | 1.165 | 69.211 | 1.0000 |
| packed8 | 24.720 [24.646, 24.830] | 25.640 [25.489, 25.708] | 1.035 | 34.606 | 0.9976 |
| affine16 | 24.939 [24.797, 25.166] | 49.464 [49.368, 49.564] | 1.983 | 69.211 | 1.0122 |
| affine32 | 24.737 [24.415, 25.007] | 59.259 [59.170, 59.712] | 2.392 | 138.423 | 0.9966 |
| parents | 24.414 [24.091, 24.582] | 6.543 [6.474, 6.602] | 0.268 | 2.955 | 0.9834 |

GPU setup 84.145s, outside trials. Corpus `7b0b5acf7274d1697f2ccdcf5a4d1ea8d33d0e617f5478b417416dcc97f30363`; all work signatures match.

## results/transport_pipeline/one_core

1024 K38 roots; 1 workers; A=10; cutoff50; table entry52; 3 rotating repetitions.

| Variant | Wall s, median [min, max] | CPU core-s, median [min, max] | Busy cores | Transfer GB | Paired wall ratio |
|---|---:|---:|---:|---:|---:|
| replay16 | 26.931 [26.058, 27.035] | 24.721 [24.540, 24.761] | 0.918 | 69.211 | 1.0000 |
| parents | 24.315 [21.952, 24.465] | 2.130 [2.033, 2.148] | 0.088 | 2.955 | 0.8994 |

GPU setup 83.362s, outside trials. Corpus `7b0b5acf7274d1697f2ccdcf5a4d1ea8d33d0e617f5478b417416dcc97f30363`; all work signatures match.
