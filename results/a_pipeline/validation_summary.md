# Overall pipeline benchmark

Resident-array CPU→GPU pipeline wall times include expansion, packing, queueing, transfers, GPU execution and CPU fallback. Checkpoint preparation, archive I/O and table/array setup are excluded from trials and recorded separately. N72 target runs are sampled work packages.

GPU kernel seconds sum CUDA stream event intervals. Concurrent streams overlap, so these totals are not GPU busy wall time.

Speedup and CPU ratios are paired by repetition against full within each run. Different policies intentionally have different work and checksums. Runs with different thread counts or corpora are kept separate. Reported ranges are observed extrema, not confidence intervals.

## results/a_pipeline/validation_complete_a8

6 complete-range variants passed CPU/GPU and same-policy checks.

| Variant | Range | Endpoints | Cases | CPU fallback endpoints | Hash |
|---|---:|---:|---:|---:|---|
| full | 2^32 | 172,927 | 12,471,991 | 60,191 | D6963196 |
| screened | 2^32 | 172,927 | 12,471,991 | 60,191 | D6963196 |
| reference16 | 2^32 | 177,870 | 12,639,847 | 60,448 | 84E5FC5E |
| compact16 | 2^32 | 177,870 | 12,639,847 | 60,448 | 84E5FC5E |
| terminal16x6 | 2^32 | 177,870 | 12,639,847 | 60,448 | 84E5FC5E |
| terminal16x8 | 2^32 | 177,870 | 12,639,847 | 60,448 | 84E5FC5E |

## results/a_pipeline/validation_complete_a9

6 complete-range variants passed CPU/GPU and same-policy checks.

| Variant | Range | Endpoints | Cases | CPU fallback endpoints | Hash |
|---|---:|---:|---:|---:|---|
| full | 2^32 | 97,991 | 14,874,875 | 42,524 | 6845E1B2 |
| screened | 2^32 | 97,991 | 14,874,875 | 42,524 | 6845E1B2 |
| reference16 | 2^32 | 101,030 | 15,042,215 | 42,742 | E7B7D706 |
| compact16 | 2^32 | 101,030 | 15,042,215 | 42,742 | E7B7D706 |
| terminal16x5 | 2^32 | 101,030 | 15,042,215 | 42,742 | E7B7D706 |
| terminal16x7 | 2^32 | 101,030 | 15,042,215 | 42,742 | E7B7D706 |

## results/a_pipeline/validation_complete_a10

6 complete-range variants passed CPU/GPU and same-policy checks.

| Variant | Range | Endpoints | Cases | CPU fallback endpoints | Hash |
|---|---:|---:|---:|---:|---|
| full | 2^32 | 52,176 | 18,317,914 | 29,310 | 4510A8AC |
| screened | 2^32 | 52,176 | 18,317,914 | 29,310 | 4510A8AC |
| reference16 | 2^32 | 53,406 | 18,494,045 | 29,492 | CEA9098C |
| compact16 | 2^32 | 53,406 | 18,494,045 | 29,492 | CEA9098C |
| terminal16x4 | 2^32 | 53,406 | 18,494,045 | 29,492 | CEA9098C |
| terminal16x6 | 2^32 | 53,406 | 18,494,045 | 29,492 | CEA9098C |

## results/a_pipeline/validation_wide_a8

3 sampled-package variants passed CPU/GPU and same-policy checks.

Sixteen improved-surviving K52 roots; not complete 2^72 verification. A=8, M=64; 16 saved roots at depths [52]. Corpus SHA-256: `84e01c1aa37831ce3bdb22a9dc93f1e953152b668eb184a0dd1ee86406089df7`.

Coverage includes 8,509 GPU endpoints with payload bit 63 set, 11,869 GPU endpoints with iterates above u64, and 20,381 CPU-fallback endpoints. This validates the sampled packages, not the complete range.

| Variant | Range | Endpoints | Cases | CPU fallback endpoints | Hash |
|---|---:|---:|---:|---:|---|
| reference52 | N=72 sampled packages | 37,337 | 3,413,922 | 20,381 | 4063213E |
| compact52 | N=72 sampled packages | 37,337 | 3,413,922 | 20,381 | 4063213E |
| terminal52x12 | N=72 sampled packages | 37,337 | 3,413,922 | 20,381 | 4063213E |

## results/a_pipeline/validation_wide_a9

3 sampled-package variants passed CPU/GPU and same-policy checks.

Sixteen improved-surviving K52 roots; not complete 2^72 verification. A=9, M=63; 16 saved roots at depths [52]. Corpus SHA-256: `84e01c1aa37831ce3bdb22a9dc93f1e953152b668eb184a0dd1ee86406089df7`.

Coverage includes 3,364 GPU endpoints with payload bit 62 set, 2,574 GPU endpoints with iterates above u64, and 12,691 CPU-fallback endpoints. This validates the sampled packages, not the complete range.

| Variant | Range | Endpoints | Cases | CPU fallback endpoints | Hash |
|---|---:|---:|---:|---:|---|
| reference52 | N=72 sampled packages | 19,466 | 3,909,224 | 12,691 | C8DCF9A4 |
| compact52 | N=72 sampled packages | 19,466 | 3,909,224 | 12,691 | C8DCF9A4 |
| terminal52x11 | N=72 sampled packages | 19,466 | 3,909,224 | 12,691 | C8DCF9A4 |

## results/a_pipeline/validation_wide_a10

3 sampled-package variants passed CPU/GPU and same-policy checks.

Sixteen improved-surviving K52 roots; not complete 2^72 verification. A=10, M=62; 16 saved roots at depths [52]. Corpus SHA-256: `84e01c1aa37831ce3bdb22a9dc93f1e953152b668eb184a0dd1ee86406089df7`.

Coverage includes 2,652 GPU endpoints with payload bit 61 set, 1,398 GPU endpoints with iterates above u64, and 4,926 CPU-fallback endpoints. This validates the sampled packages, not the complete range.

| Variant | Range | Endpoints | Cases | CPU fallback endpoints | Hash |
|---|---:|---:|---:|---:|---|
| reference52 | N=72 sampled packages | 10,238 | 3,558,096 | 4,926 | 65C62E5C |
| compact52 | N=72 sampled packages | 10,238 | 3,558,096 | 4,926 | 65C62E5C |
| terminal52x10 | N=72 sampled packages | 10,238 | 3,558,096 | 4,926 | 65C62E5C |
