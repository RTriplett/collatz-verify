# Chained low-bit transitions

This experiment tests whether low-bit indexing can eliminate repeated arithmetic
across expansion blocks and inside the GPU tail. All paths are opt-in. The
production defaults and the previously confirmed full-range estimate are unchanged.

**Result:** grouping saves about 20% of CPU time against ungrouped chains, but
the existing parent-descriptor pipeline remains faster overall. The aligned GPU
tail tables also lose to ordinary stepping. Further work should preserve compact
group descriptions through device expansion instead of emitting every endpoint
on the CPU.

## Implementations

`chain4` and `chain6` replace the final expansion with successive exact affine
tables: K50→54→58→62 or K50→56→62. `group4` and `group6` additionally regroup
parents at each boundary by `(f, oee, ne, delta)`, where
`delta = m * 3^(-f) mod 2^width`. A bucket shares the extension translation and
the wide multiplication `3^f*q`; every parent's exact high iterate and starting
residue are retained. No trajectories are merged or dropped. These paths send
32-byte exact endpoints; `affine32` is the same-transport single-table control.
`novote_oee` is the existing K52 parent-descriptor pipeline and practical baseline.

`jump8`, `jump10`, `jump12`, and `jump14` retain parent transport and index a
GPU tail table by the current iterate's low bits. Each entry combines complete
iterations of the existing two-half tail loop. It reconstructs every original
low-32-bit checksum point but performs full 128-bit arithmetic only at the last
point. After each jump it indexes again from the exact updated state. Conservative
directed-rounding guards prove that no skipped point descends and no intermediate
odd-run peak reaches the arithmetic cutoff; otherwise ordinary stepping resumes.
Floating point is used only for these guards, never for trajectory state.
`group4_jump12` combines grouped CPU expansion with the 12-bit GPU table.

Tail tables occupy 16 KiB, 64 KiB, 256 KiB, and 1 MiB. They encode at most two
original iterations for widths 8/10 and three for widths 12/14. The initial sm_89
jump kernel used 48 registers versus 40 for the tuned baseline. A follow-up
aligns the header and each prefix to 16 bytes, uses vector loads, and explicitly
unrolls the three possible checksum points. Its exact-iterate kernel uses
46 registers, with no spills; the row remains 64 bytes.

## Correctness

- The expansion benchmark's 26 Rust tests pass, including exact chained
  endpoints and per-depth statistics against independent DFS and the existing
  affine expander, partial blocks, bit 63, and iterates wider than 64 bits.
- The independent [host validator](../../scripts/grouped_tail_bench/table_check.cpp)
  checks all 21,760 table keys and 499,008 lifted states, including 677,496 exact
  checksum endpoints and 1,354,992 odd-run peaks. Stored floating-point bounds
  are checked as exact rational numbers.
- [Direct GPU validation](validate_jump/pipeline.stderr.txt) compares all four
  table widths on 16,384 adversarial/random inputs at cutoffs 97/104/112/120.
  Every width matches 49,920 original iterations and 9,123 cutoff flags; accepted
  jumps are counted to confirm that the optimized path was exercised.
- All eleven variants match the CPU reference on a [complete N32 run](validate_small/pipeline.jsonl),
  [sixteen K52 packages at N72](validate_wide/pipeline.jsonl), and a
  [forced-overflow N72 run](validate_overflow/pipeline.jsonl). The last checks
  all 985 GPU-flagged origins and successful wide rechecks. Comparisons include
  full tree statistics, per-class counts, iterations, checksums, and flagged-origin
  multisets, including multiplicities.
- The aligned-table revision passes the same exhaustive host validator, all four
  [direct GPU comparisons](validate_jump_aligned/pipeline.stderr.txt), and six
  [forced-overflow pipeline comparisons](validate_overflow_aligned/pipeline.jsonl).
  Five Rust tuning tests and twelve Python report-auditor tests also pass.

## Arithmetic sharing and initial timing

The [untimed census](census/pipeline.jsonl) follows 32 accepted K38 roots,
sixteen from each pilot cohort. Sharing is confined to descendants of each root;
the diagnostic does not pool unrelated checkpoints into larger buckets.

| Grouped block | Mean parents/bucket | Reduction in extension translations |
|---|---:|---:|
| K54→58, width 4 | 61.56 | 98.43% |
| K58→62, width 4 | 573.83 | 99.83% |
| K50→56, width 6 | 2.60 | 64.95% |
| K56→62, width 6 | 53.26 | 98.14% |

These are reductions in one shared setup operation, not reductions in emitted
endpoints or in all arithmetic. The 148,412,492 final endpoints remain distinct.

The [initial pilot](pilot_summary.md) uses two independent cohorts of 128
baseline draws, two timing repetitions, all eleven variants, seed 20261027.
The whole process is pinned to physical cores 0–1 of the Ryzen 7700X; the RTX
4090 runs the same N72/A10 windows as the confirmed parent pipeline. Setup and
warmup precede the paired timings. Each pair executes exactly the same work.

Grouping cuts process CPU time by **19.57%** for four-bit chains and **19.52%**
for six-bit chains, relative to their ungrouped controls. Corresponding elapsed
reductions are **17.66%** and **10.48%**. The fastest grouped path, `group6`,
still takes **27.18% longer** than `novote_oee`. Exact endpoint transport averages
**16.207 GB per cohort**, versus **0.347 GB** for parent descriptors: **46.71×**
as much traffic. The grouped six-bit path also consumes 18.54× the process CPU
time of that parent baseline.

The initial GPU tail variants take 2.36×, 2.56×, 3.00×, and 5.21× the baseline
elapsed time for widths 8/10/12/14. Combining grouped four-bit expansion and
12-bit jumps takes 2.83×. These results motivated the aligned-table follow-up;
the original sources and binary hashes remain in the pilot directory.

The pilot is a comparative experiment, not a replacement full-range cost
estimate. Its two-cohort extrapolation has very wide sampling intervals.

## Fresh follow-up

The [follow-up report](followup_summary.md) uses **four new cohorts × 128 baseline
draws**, seed 20261028, two repetitions, and the aligned GPU tables. Its 479
accepted draws retain their original 512-draw denominator. Each variant also
processes all accepted draws together once. Every per-cohort and aggregate work
signature matches, including the aggregate composition from the separate cohorts.

| Variant | Mean cohort wall s | Mean cohort CPU s | Wall / current baseline | Continuous-run wall s |
|---|---:|---:|---:|---:|
| Current parent pipeline | 2.483 | 0.265 | 1.000 | 10.231 |
| Single table, exact endpoints | 3.172 | 5.775 | 1.278 | 12.585 |
| Ungrouped six-bit chains | 3.299 | 6.120 | 1.329 | 13.123 |
| Grouped six-bit chains | 2.880 | 4.920 | **1.160** | **11.237** |
| Aligned eight-bit GPU jumps | 4.871 | 0.275 | **1.962** | **19.246** |
| Aligned twelve-bit GPU jumps | 6.161 | 0.290 | **2.482** | **24.672** |

Against the matched ungrouped six-bit chain, grouping saves **19.61% CPU time**
and **12.71% elapsed time**. Against the single-table exact-endpoint control it
saves **14.81% CPU time** and **9.21% elapsed time**. This confirms a benefit from
changing how indexed work is executed, rather than merely adding an index.

Against the current parent pipeline it still costs **16.00% more elapsed time**
in the paired cohorts and **9.83% more** in the continuous run. The aligned
eight-bit GPU jumps cost 1.962× and 1.881× respectively; twelve-bit jumps cost
2.482× and 2.411×. Smaller tables, vector loads, and bounded loop unrolling do
not recover the added lookup/guard costs in this design. Attribution among
instruction count, cache behavior, register occupancy, and warp divergence
would require profiling; the experiment measures their combined effect.

The continuous/cohort calibration ratios range from **0.9755 to 1.0303**. This
active-desktop variation limits precision, but every comparison retains the
same winner. The four-cohort paired interval for grouped/baseline wall is
1.136–1.184; it excludes systematic effects such as the difference seen in the
continuous run. Do not substitute these small comparative samples for the
earlier 8,192-draw cost confirmation. The **0.586 RTX 4090-year** estimate remains
the existing estimate, with no new saving claimed here.

The representation to investigate next is a compact parent array plus bucket
metadata and resident transition ranges. Merely using parent dispatch at K56
or K58 would reduce this prototype's traffic but still generate more CPU
parents than the current K52 approach. To improve on that baseline, the chain
should remain compressed across boundaries and into device expansion. The
current experiment instead regroups full intermediate nodes, emits individual
final endpoints, and then reindexes exact GPU states independently.

The [compact-dispatch follow-up](../compact_chain/README.md) implements a
two-stage representation that crosses the intermediate boundary without
constructing host nodes. It cuts CPU cost and transfers relative to K52 parent
dispatch, and compares shared-prefix GPU expansion against a flat K50 control.
See that report for paired and repeated continuous timings.

## Reproduction

```bash
CUDA_ARCH=sm_89 cargo build --release --offline --features cuda \
  --manifest-path scripts/expansion_bench/Cargo.toml --bin fullrange_bench \
  --target-dir /tmp/collatz-grouped-target
g++ -O2 -std=c++17 scripts/grouped_tail_bench/table_check.cpp \
  -o /tmp/collatz-tail-table-check
/tmp/collatz-tail-table-check
```

Each measurement directory preserves its exact command in `metadata.json`,
source snapshots, raw JSON records, and GPU telemetry (except the CPU-only
census). `run_overall.py --profile census` records arithmetic sharing without
initializing CUDA. Validation profiles are `validate-jump`, `validate`,
`validate-wide`, and `validate-device`; timing uses `target`.
