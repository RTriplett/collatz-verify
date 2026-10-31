# Early GPU screening followed by grouped CPU verification

This experiment implements CPU tree expansion → GPU terminal/A expansion and
B-bitvector screening → CPU verification of selected surviving batches. It does
not establish a runtime saving. On the measured RTX 4090 pipeline, the five
handoff variants take **1.7–3.0% longer** than ordinary GPU verification.
Repeated low-bit grouping greatly reduces table lookups, but its sorting costs
outweigh the savings on the captured CPU fixture. Production defaults and the
existing full-range GPU-year estimate are unchanged.

## Implementation and controls

The GPU exports a survivor mask and exact affine endpoint for selected launches:
128 bytes of masks plus 32 bytes of context per M62 endpoint at A10. It does not
enqueue a separate record for each survivor. B screening filters candidates;
it does not advance their trajectories. CPU workers reconstruct the original
start and current shifted iterate `y = T^M(n) + 1`, then verify to descent or the
arithmetic cutoff. Grouping uses low bits of **y**, not of the original start.

The ordinary fused GPU kernel remains unchanged. The experiment sends every
1,024th launch, starting at launch zero, to the screen-only kernel. Four bounded
CPU workers consume compact shards and reconstruct at most 1,024 endpoints at
a time. Each verifier sorts at most 65,536 states at once. These limits avoid
materializing an entire batch's tens of millions of 32-byte states.

Five CPU methods are compared:

1. Original scalar stepping.
2. Counting-sort by low bits, followed by scalar stepping.
3. Independent affine table lookups.
4. Initial sorting with a shared first table lookup, followed by independent
   lookups as trajectories diverge.
5. Repeated sorting and shared transitions of the unfinished states.

Tables preserve the reference double-iteration count and every checksum
contribution. Exact integer bounds check skipped descent points and arithmetic
peaks; a failed bound falls back to ordinary stepping. Individual starts,
cutoff flags, duplicate flagged origins and wide rechecks remain distinct.
The prototype shares transition descriptions; it does not implement a new
proof that permits verifying an entire group with one arithmetic operation.

## Continuous pipeline measurements

The [pipeline report](pilot_gpu/summary.md) uses four cohorts of 128 fixed
baseline K38 draws, seed 20261102, with 485 accepted draws. A concatenated run
processes 2,026,356,266 GPU-bound endpoints, 207,885,256,605 surviving cases and
1,670,181,182,546 double iterations. All variants match the complete work
signature, checksum `191F29E9`, flags and wide rechecks. The common windows use
19.625 GiB; N72/M62/A10 and improved checks through K50 are unchanged.

Each method has one timing per cohort and two continuous timings of the same
concatenated corpus. Setup and warmup are excluded. These are comparative
timings, not a new full-range estimate or independent sampling interval.

| Method | Mean continuous seconds | Relative to ordinary GPU |
|---|---:|---:|
| Ordinary GPU | 9.875 | 1.000 |
| Early scalar CPU | 10.046 | 1.017 |
| Sorted scalar, 10 bits | 10.107 | 1.023 |
| Independent table, 10 bits | 10.049 | 1.018 |
| Initial shared table, 10 bits | 10.169 | 1.030 |
| Repeated shared table, 10 bits | 10.111 | 1.024 |

Each continuous run selects two of 1,943 launches. Depending on their class,
these contain **0.025–0.037% of the total tail iterations**. Launch order can
vary with producer scheduling, so the selected CPU states are not identical
between pipeline variants. Recorded class vectors and actual work counts are
retained; the fixed-state replay below isolates the CPU algorithms.

The returned payload is 320 MiB per continuous run. D2H event intervals sum
to approximately **12.8 ms**, while mask scans and owned-buffer copies consume
**245–260 ms of dispatch-thread CPU time**. CPU workers consume 5.5–8.9 core-s,
but final drains are about 0.1 ms: most worker execution overlaps GPU work.
The dispatch-thread work is a more substantial observed cost than PCIe copying.
These measurements overlap and must not be added into a wall-time breakdown.
CUDA event intervals can also include scheduling delays.

### What the host-buffer cost contains

A separate [host-export probe](host_export/README.md) uses pinned input arrays
and the same four-shard, 160 MiB layout as one exported launch. Five rotated
repetitions on CPU 6 separate scans, allocation, first touching output pages
and copying. Median thread CPU times are:

| Operation per exported launch | CPU milliseconds |
|---|---:|
| Software mask popcount, as compiled in the pipeline | 13.30 |
| Hardware POPCNT64 scan | 2.67 |
| Allocating and copying output vectors | 89.28 |
| Copy into newly allocated, untouched pages | 90.13 |
| Copy into resident reusable vectors | 23.34 |
| Current-style scan + allocation + copy | 103.10 |
| Scan + resident reused copy | 36.78 |

Fresh copies incur approximately 40,932 minor page faults per launch; resident
copies incur none. Allocation without touching the pages costs only 0.035 ms.
This identifies destination first-touch costs as a substantial part of the
host handoff. Workers also count the masks while decoding, so the dispatch
scan duplicates work. Buffer reuse and removing that duplicate scan are
concrete implementation opportunities, but have **not** been applied to the
timed full pipeline. This standalone probe uses synthetic masks and no live
DMA or concurrent verifier, so its differences are not promised wall-time
savings. It does not change the fixed-input grouping result below.

## Identical-input CPU replay

During the first early warmup, the runner captures 262,144 real screened states.
No timed pipeline trial captures data. This first-launch fixture is conditional
on its selected class (F41, B35) and is not a population sample across all F
classes. [Fixture details](cpu_replay/README.md) distinguish decoding raw
32-byte states from reconstructing them from GPU masks.

The [CPU replay](cpu_replay/summary.md) tests 8/10/12/14-bit keys and
16,384/65,536/262,144-state chunks, for 51 configurations and five rotated
timing repetitions each. All 255 trials reproduce 1,949,898 double iterations
and checksum 3367139950 on the same states.

The fastest scalar configuration takes **19.737 ms**. The best initial shared
table configuration takes **20.383 ms**. The best repeated grouping takes
**23.911 ms**, 19.2% longer than scalar with the same chunk size. It eliminates
**97.96% of table lookups** (1,921,077 → 39,167), but grouping takes **5.60 ms**,
or 23.4% of its runtime. Larger groups often increase memory costs.

A 12-bit independent table takes 20.291 ms: 3.2% faster than scalar at its
256K chunk size, but 2.8% slower than the best scalar configuration. This is
not evidence of a general table advantage. A broader fixture would be needed
to establish a small improvement across classes.

## Correctness and retained artifacts

- [Independent GPU mask oracle](mask_oracle/README.md): 128 launches,
  111,680 mask words and 3,573,760 candidate bits; A8/A10, M62/M64, both
  bitvector families, OEE states, wrap boundaries, partial warps and all three
  supported endpoint transports.
- [Complete N32](validate_n32/pipeline.jsonl) and
  [sampled N72/M64](validate_wide_a8/pipeline.jsonl): seven pipeline variants
  each agree with independent CPU expansion and verification.
- [Forced arithmetic cutoff](validate_cutoff/pipeline.jsonl): all 985 flags
  originate on early CPU workers in each all-screen variant; the mixed variant
  returns 488 from early CPU workers. Every origin and wide recheck matches.
- CPU table tests cover every supported low-bit residue, adversarial wide
  states, carries, long zero runs, cutoffs 97–123 and chunk boundaries. Worker
  tests cover bounded queues, accounting and multi-chunk shards.
- The pipeline and CPU auditors reject missing work, mismatched counts,
  malformed class vectors, incomplete repetitions and captures during timed
  trials. Raw telemetry, source snapshots, executable hashes and commands are
  retained with each run.

`pilot/metadata.json` records an unsuccessful sandbox GPU-access attempt;
`pilot_gpu/` contains the completed measurement. It is not a second sample.

## Reproduction

Build outside the repository's tracked target directory:

```sh
CUDA_ARCH=sm_89 cargo build --release --offline --features cuda \
  --manifest-path scripts/expansion_bench/Cargo.toml \
  --bin fullrange_bench --bin early_cpu_bench \
  --target-dir /tmp/collatz-early-screen-target
```

The exact pipeline command and environment are in
[pilot_gpu/metadata.json](pilot_gpu/metadata.json). Use a fresh output directory
and capture filename when reproducing it. Labels have the form
`early1024_group10_w4`: launch stride, CPU method/key width and worker count.
`early1_*` validates all-screen execution; `early2_*` exercises mixed execution.

The standalone CPU runner and both auditors are:

```sh
python3 scripts/expansion_bench/run_early_cpu.py \
  --fixture results/early_screen/pilot_gpu/screened_states.bin \
  --binary /tmp/collatz-early-screen-target/release/early_cpu_bench \
  --out /tmp/early-cpu-replay --core 6 --repetitions 5
python3 scripts/expansion_bench/summarize_early_pipeline.py --help
python3 scripts/expansion_bench/summarize_early_cpu.py --help
```

The useful next distinction is between eliminating per-state arithmetic and
merely sharing its table lookup. This experiment demonstrates lookup reuse,
but does not make CPU verification fast enough to remove substantial GPU work.
