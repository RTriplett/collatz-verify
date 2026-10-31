# Integrated CPU/GPU benchmark

Measured on 2026-10-23 with a Ryzen 7 7700X and RTX 4090. These runs use the
actual verifier's packing, bounded queues, pinned transfers, three CUDA streams,
tail kernels, statistics and CPU fallback. They do not multiply results from
separate microbenchmarks. [All trials and accounting](summary.json) and a
[generated table](summary.md) are available.

## Eight physical cores

Each variant expands the same 1,024 saved K38 work packages to M62 and completes
the N72/A10 verification for those packages. There are three repetitions in
rotating order, after warming each path. Medians:

| Expansion | Pipeline wall, s | Process CPU, core-s | Average busy CPU cores |
|---|---:|---:|---:|
| Original full checks through M62 | 28.955 | 213.846 | 7.28 |
| Ternary-screened full checks | 29.973 | 212.361 | 7.08 |
| Screened through K54, compact suffix | 26.379 | 74.877 | 2.84 |
| Screened through K54, terminal table of 8 bits | 26.441 | 31.513 | 1.19 |
| Screened through K50, compact suffix | 26.463 | 73.271 | 2.77 |
| Screened through K50, terminal table of 10 bits | 26.549 | 27.803 | 1.06 |

The combined K50 path reduces median process CPU consumption by **87.0%**,
with **8.3% less median elapsed time** than full checks. Terminal tables alone
reduce CPU consumption by 57.9% at K54 and 62.1% at K50 versus their matched
compact controls, while elapsed times remain close. This workload is limited by
GPU throughput once the host expansion is inexpensive enough.

Full-check elapsed times range from 27.766 to 31.491 seconds; the combined K50
range is 26.240–26.562 seconds. These are observed ranges over three runs, not
confidence intervals. Ternary screening alone does not establish an integrated
speedup here. The combined paths retain screening in their shorter prefix.

## Two physical cores

The same corpus and arrays were then run with two producer workers and the
whole process pinned to physical CPUs 0–1. Three rotating repetitions give:

| Expansion | Pipeline wall, median [min, max], s | Process CPU, median core-s | Average busy CPU cores |
|---|---:|---:|---:|
| K54, terminal8 | 26.517 [26.423, 26.521] | 30.491 | 1.153 |
| K50, terminal10 | 26.535 [26.510, 26.577] | 29.605 | 1.114 |

Both configurations preserve eight-core GPU throughput on this corpus.
**Two physical cores keep the RTX 4090 fed** with the combined expansion.
Counts, per-class work, every tree counter and hashes also match the corresponding
eight-core runs exactly. This is a measured allocation result on this CPU/GPU,
not a claim that two arbitrary cloud vCPUs will always suffice.

## Work and correctness

The full policy emits 4,201,887,155 endpoints per trial; K54 emits 4,299,192,388
and K50 emits 4,325,704,552. Relative to full checks, K54 increases actual tail
iterations by 0.537% and K50 by 0.785%. The tables preserve the exact endpoint
set for their cutoff policy. Every tree counter, per-class endpoint/case/iteration
count and checksum agrees between implementations of that policy and across
repetitions. Different cutoff policies intentionally have different checksums.
No trajectories were flagged.

The final regression run passes [19 library tests plus five bitvector witness
audits](cpu-tests.txt) and [16 expansion tests](expansion-tests.txt). Test commands:

```sh
cargo test --offline --no-default-features --target-dir /tmp/collatz-overall-host-tests
CUDA_ARCH=sm_89 cargo test --release --offline --features cuda \
  --manifest-path scripts/expansion_bench/Cargo.toml --bin overall_bench \
  --target-dir /tmp/collatz-overall-target
```

Two complete N32 CPU/GPU validations cover six expansion variants each, using
host-built arrays and then device-built per-class arrays. A depth-12 walkback
checkpoint is active. Only two bitvector classes are supplied, forcing
29,310–29,492 endpoints through CPU fallback on every run. Compact payloads
reconstruct the true full-width iterate for that fallback. The large sampled
corpus itself has no fallback endpoints. Unit tests additionally compare full
nodes and counters against an independent two-stage DFS, including wrapped
64-bit iterates, M64 and disabled statistics.

## Scope and setup

* K38 roots are sampled without duplicates from the baseline population,
  rejecting improved13 ancestry and bounded checkpoint walkback (65,536 states).
  The saved roots are included in each run's configuration record. All controls
  use this same filtered checkpoint: this comparison does **not** measure a new
  walkback gain against an unfiltered verifier.
* Windows are shared and include pm∞ plus mix13 credits. The class windows are
  `32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27`, requiring **19.625 GiB**
  for the arrays. The recommended 21.625 GiB configuration failed the VRAM guard
  alongside the active desktop; reducing the F45 window from 35 to 34 saved
  2 GiB. The rejected attempt is retained in `main8/` and excluded from summaries.
* GPU arrays stay resident throughout a run. Eight-core setup took 89.745 seconds,
  including 89.366 seconds building arrays. Expansion-engine setup took 0.0041
  seconds, and generating/filtering the sampled checkpoint took 0.0131 seconds.
  Setup, archive I/O and checkpoint preparation are outside the timed trials.
* Terminal tables retain both experimental layouts: K54/table8 uses 317,768
  bytes and K50/table10 uses 811,656 bytes, shared across workers.
* CPU time uses the Linux process CPU clock, including all producer and GPU
  worker threads. Average busy cores are CPU seconds divided by elapsed seconds;
  blocked workers are not charged as busy. CUDA event totals sum overlapping
  stream durations and are **not GPU busy wall time**.
* The desktop remained active, with CPU governor `powersave`. GPU telemetry,
  CPU topology, affinity, commands, source snapshots and hashes are saved.
  No concurrent benchmark/build was run during measurement.

This is a target-shaped **sample of N72 work**, not complete verification below
2^72. It measures the downstream pipeline from stored K38 packages, not full
frontier generation/storage. It does not replace the global GPU-year model or
measure an RTX PRO 6000. Harvested reverse-word window credits remain a separate
research prototype and are not included.

## Reproduce

Build separately from measurement; adjust `sm_89` and CPU affinity for the host:

```sh
CUDA_ARCH=sm_89 cargo build --release --offline --features cuda \
  --manifest-path scripts/expansion_bench/Cargo.toml --bin overall_bench \
  --target-dir /tmp/collatz-overall-target

python3 scripts/expansion_bench/run_overall.py --profile validate --out /tmp/overall-validation
python3 scripts/expansion_bench/run_overall.py --profile validate-device --out /tmp/overall-validation-device

python3 scripts/expansion_bench/run_overall.py --out /tmp/overall-eight \
  --roots 1024 --repetitions 3 --threads 8 --affinity 0-7 --seed 20261023 \
  --class-b 32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27

python3 scripts/expansion_bench/run_overall.py --out /tmp/overall-two \
  --roots 1024 --repetitions 3 --threads 2 --affinity 0-1 --seed 20261023 \
  --variants combined54,combined50 \
  --class-b 32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27

python3 scripts/expansion_bench/summarize_overall.py --out /tmp/overall-summary.json \
  /tmp/overall-validation /tmp/overall-validation-device /tmp/overall-eight /tmp/overall-two
```

The experimental engines live in `scripts/expansion_bench/combined.rs`.
`Expander`, the compact endpoint sink hook, and `GpuSession` expose the existing
production pipeline to them. Default `cv verify` still uses the original DFS;
its one-shot report includes setup and teardown. A session's `run` reports warm
pipeline time separately. Work-unit generation still uses `VerifyConfig.rules`;
the experiment's cutoff is deeper than the initial split.
