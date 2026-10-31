# A=8/9/10 pipeline comparison

Resident-array expansion, packing, queues, transfers, GPU execution and CPU fallback. Checkpoint preparation, archive I/O, engine setup and array construction are excluded from trials.

Different A values ran in separate process blocks with resident arrays. Cross-A ratios are ratios of medians, not repetition-paired comparisons. Replicated blocks are preserved individually and pooled only within the same A/configuration. Ranges are observed minima and maxima, not confidence intervals.

Changing A changes M=72-A, endpoint grouping, bitvector thresholds and tail work. Counts and hashes may differ across A. Within each A, every repetition and rerun must have the identical complete work signature. All A values use the same ordered K38 roots, full-check cutoff and terminal-entry depth.

N=72; 1024 K38 roots; 8 workers; affinity [0, 1, 2, 3, 4, 5, 6, 7]; NVIDIA GeForce RTX 4090 (sm_89, 128 SMs); 3 streams. Full checks end at 50; terminal tables start at 52. Corpus SHA-256: `7b0b5acf7274d1697f2ccdcf5a4d1ea8d33d0e617f5478b417416dcc97f30363`.

Class windows: `32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27`. Window rules: `descent+pm1+oee+pm_inf+mix(3^13)`.

| A | M | Table width | Blocks / trials | Wall seconds, median [min, max] | CPU core-seconds, median [min, max] | Busy cores, median [min, max] | Wall ratio vs A10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 64 | 12 | 1 / 3 | 27.539 [27.516, 27.609] | 97.871 [97.368, 98.014] | 3.550 [3.539, 3.554] | 1.0856 |
| 9 | 63 | 11 | 1 / 3 | 25.448 [25.085, 25.891] | 51.105 [50.255, 52.859] | 2.008 [1.941, 2.107] | 1.0032 |
| 10 | 62 | 10 | 2 / 6 | 25.367 [25.093, 25.507] | 27.970 [27.385, 28.428] | 1.102 [1.074, 1.132] | 1.0000 |

GB is decimal (10^9 bytes). Transfer GB/s is payload bytes divided by full pipeline wall time, not an isolated PCIe bandwidth measurement.

| A | Endpoints | Cases | Tail iterations | Transfer GB/trial | Transfer GB/s, median [min, max] | Endpoints/s, median [min, max] | Hash |
|---|---:|---:|---:|---:|---:|---:|---|
| 8 | 15,844,250,721 | 394,286,582,991 | 3,139,327,674,219 | 253.508 | 9.205 [9.182, 9.213] | 575340717 [573873371, 575821660] | BA748CE9 |
| 9 | 8,243,317,158 | 410,373,026,230 | 3,271,717,460,118 | 131.893 | 5.183 [5.094, 5.258] | 323925715 [318384288, 328621537] | 635A2D7D |
| 10 | 4,325,704,552 | 427,313,317,731 | 3,416,237,911,342 | 69.211 | 2.728 [2.713, 2.758] | 170528776 [169590876, 172385263] | F782B001 |

A changes M, terminal width/table size, mod-9 masks and lookahead cutoff parameters. Each A receives its own session/array build; setup is recorded separately even though class window widths match.

| Run | A | Repetitions | Wall seconds, median [min, max] | CPU core-seconds, median [min, max] | GPU setup s | Device build s | Engine setup s | Table bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| results/a_pipeline/a10_start | 10 | 3 | 25.108 [25.093, 25.315] | 28.339 [27.813, 28.428] | 86.098 | 85.680 | 0.002 | 811,656 |
| results/a_pipeline/a9 | 9 | 3 | 25.448 [25.085, 25.891] | 51.105 [50.255, 52.859] | 85.209 | 84.849 | 0.004 | 1,454,096 |
| results/a_pipeline/a8 | 8 | 3 | 27.539 [27.516, 27.609] | 97.871 [97.368, 98.014] | 86.655 | 86.279 | 0.008 | 2,717,144 |
| results/a_pipeline/a10_end | 10 | 3 | 25.443 [25.418, 25.507] | 27.532 [27.385, 28.128] | 85.532 | 85.079 | 0.002 | 811,656 |

Complete work signatures match between repetitions and reruns within each A. The JSON preserves original validated run summaries, individual trial metrics and unpaired A10 ratios. These samples do not constitute a complete 2^72 verification.
