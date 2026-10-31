# Moving two more bits from GPU completion to CPU expansion

This experiment compares A = 10, 9 and 8 in the actual CPU/GPU pipeline at
N = 72. It uses the combined expansion from the
[previous integrated benchmark](../overall_pipeline/README.md): screened full
checks through K50, compact baseline traversal to K52, and direct terminal
enumeration. The terminal entry stays fixed while its width changes:

| Completion bits A | CPU endpoint depth M = 72 − A | Terminal entry | Terminal width |
|---|---:|---:|---:|
| 10 | 62 | 52 | 10 |
| 9 | 63 | 52 | 11 |
| 8 | 64 | 52 | 12 |

Every configuration processes the same 1,024 saved K38 roots, generated with
seed 20261023. Roots are sampled without duplicates from the baseline frontier,
then rejected if improved13 ancestry or bounded K38 walkback supplies a
certificate; walkback uses a 65,536-state cap. Thus each A covers the same
starting residue classes below 2^72. This is a sample of target-sized work
packages, not complete verification below 2^72 or a measurement of generating
and storing the entire K38 frontier.

## Results

**The CPU can afford deeper expansion, but it does not improve throughput
with this pipeline and window allocation.** Keep A10 for this configuration.
Measured on 2026-10-23; medians and observed ranges:

| A / M | Pipeline wall, s [min, max] | Process CPU, core-s | Average busy cores | Transfer GB/trial | Tail iterations vs A10 |
|---|---:|---:|---:|---:|---:|
| 10 / 62 | 25.367 [25.093, 25.507] | 27.970 | 1.102 | 69.211 | reference |
| 9 / 63 | 25.448 [25.085, 25.891] | 51.105 | 2.008 | 131.893 | −4.23% |
| 8 / 64 | 27.539 [27.516, 27.609] | 97.871 | 3.550 | 253.508 | −8.11% |

A10 pools six trials from the opening and closing control blocks; A9 and A8
have three trials each. The A10 block medians are 25.108 and 25.443 seconds
(1.34% apart). A9's pooled-reference difference is only +0.32%, with overlapping
ranges: there is **no demonstrated wall-time benefit**, despite 1.83× the CPU
time. A8 takes **8.56% longer**, with 3.50× the CPU time. Even its fastest trial
is slower than every A10 trial. These are comparisons between process blocks,
not paired trials or confidence intervals. The previous session's 26.549-second
A10 measurement is not used as the reference.

The extra CPU work uses only about 3.55 of the eight available physical cores
at A8. The tradeoff includes substantially more work outside the counted tail:

| A | GPU endpoints relative to A10 | GPU prefix replay steps, M × GPU endpoints | Replay steps relative to A10 |
|---|---:|---:|---:|
| 10 | 1.000× | 268,193,682,224 | 1.000× |
| 9 | 1.906× | 519,328,980,261 | 1.936× |
| 8 | 3.663× | 1,014,032,045,440 | 3.781× |

The GPU reconstructs each endpoint's iterate by replaying M steps before
lookahead filtering, even when that endpoint has no surviving completions.
More endpoints also mean more packing, transfers and kernel launches.
Payload traffic rises from 2.728 GB/s at A10 to 5.183 at A9 and 9.205 at A8
(decimal GB, divided by pipeline wall time). This is not an isolated bandwidth
measurement. The benchmark does not isolate the time attributable to replay,
transfers, dispatch or changed window behavior; replay steps and tail iterations
are different units of arithmetic work. Reducing endpoint/reconstruction costs
is a more promising next experiment than simply lowering A further.

GPU setup takes 85.2–86.7 seconds per block and is excluded above. Shared CPU
terminal tables occupy 811,656 / 1,454,096 / 2,717,144 bytes for A10/A9/A8.
All four blocks use the same binary and ordered corpus; every repeated work
signature matches within its A. No trajectories were flagged. A10 has no CPU
fallback endpoints; A9 and A8 each have eleven per trial, included in the totals.

[All trials and accounting](summary.json), a [generated table](summary.md), and
each block's complete `pipeline.jsonl` retain the measurements.

## Measurement

The machine is a Ryzen 7 7700X with an RTX 4090. Eight producer workers and
the complete process are pinned to physical CPUs 0–7. Each block warms the
selected expansion path, then records three repetitions with arrays resident.
The blocks run in the order `a10_start`, `a9`, `a8`, `a10_end`; the closing
A10 block checks how much the reference changed during the comparison.
The desktop stays active and the CPU governor is `powersave`.

All runs use shared bitvectors with pm∞ and mix13 window credits. The common
window vector, indexed relative to each configuration's minimum F, is:

```text
32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27
```

The arrays occupy approximately 19.625 GiB. The small wraparound allocation
varies with A. Holding this vector fixed gives essentially the same memory
budget; it does not independently optimize the window allocation for each A.
Changing M changes the minimum F and the window cutoffs, so any ranking applies
to these windows on this machine. In particular, the minimum F shifts from
40 to 41 at A8: F41 receives B32 there, versus B35 at A10/A9. Separately tuning
A8's allocation is not measured here.

Timed wall time includes expansion, endpoint packing, queueing, pinned
host-to-device transfers, CUDA kernels and any CPU fallback. Endpoint payloads
remain 16 bytes each, so deeper expansion can increase transfer and launch
cost even when fewer completed trajectories survive. The GPU also replays all
M steps from each endpoint's starting residue to reconstruct its iterate before
lookahead filtering (`cuda/kernel.cu`). This replay is outside the reported
tail-iteration count. Setup, checkpoint
sampling/filtering, archive I/O and terminal-table construction are outside
the timed trials and recorded separately. Each block builds its GPU arrays
once in a reusable session.

Process CPU time includes all producer and GPU worker threads. Dividing it
by wall time gives average busy cores; blocked workers do not count as busy.
CUDA kernel seconds sum event intervals from three concurrent streams and
are **not GPU busy wall time**. Reported minima and maxima are observed
repetition ranges, not confidence intervals.

The runner preserves inputs, commands, CPU topology and affinity, source
snapshots, source and binary SHA-256 hashes, and one-second GPU telemetry in
each block. Hashes identify the saved files and binary; they do not by
themselves establish that the binary was built from those sources. Builds
and other benchmarks are kept separate from measurements.

## Correctness

[Validation results](validation_summary.md) and their
[machine-readable counterpart](validation_summary.json) contain 27 exact
CPU/GPU comparisons:

* Eighteen complete N32 runs: six expansion variants for each A, with a
  depth-12 walkback checkpoint, device-built B12/B13 windows and only two
  bitvector classes to force CPU fallback.
* Nine N72 package runs: reference, compact and terminal expansion for each
  A, using the same sixteen improved-surviving K52 roots. Full checks stop at
  K52 in this validation, with terminal widths 10, 11 and 12.

Within each policy and A, every tree counter, per-class endpoint/case/iteration
count and checksum agrees. Wide validation explicitly requires GPU endpoints
with the top residue bit set, true iterates above `u64`, and CPU fallback.
For A8/M64 it covers 8,509 GPU endpoints with bit63 set, 11,869 GPU endpoints
whose iterates exceed `u64`, and 20,381 CPU fallback endpoints. This exercises
the compact payload boundary, GPU reconstruction and exact fallback state.

The timed corpus and all repetition accounting are checked for consistency.
**Counts and verification hashes are not expected to match across A:** the
CPU/GPU boundary and subsequent pruning differ. The shared K38 corpus hash
establishes that the configurations cover the same starting packages.

The experimental expander uses the production packing, queues, transfers and
tail kernel through `Expander` and `GpuSession`. Default `cv verify` continues
to use the original DFS. Harvested reverse-word window credits are outside
this experiment.

## Reproduce

Build before measurement; adapt the CUDA architecture and CPU affinity for
another machine:

```sh
CUDA_ARCH=sm_89 cargo build --release --offline --features cuda \
  --manifest-path scripts/expansion_bench/Cargo.toml --bin overall_bench \
  --target-dir /tmp/collatz-overall-target

for a in 8 9 10; do
  python3 scripts/expansion_bench/run_overall.py --profile validate-device \
    --a "$a" --out "/tmp/a-pipeline/validation_complete_a$a"
  python3 scripts/expansion_bench/run_overall.py --profile validate-wide \
    --a "$a" --out "/tmp/a-pipeline/validation_wide_a$a"
done

python3 scripts/expansion_bench/run_overall.py --a 10 \
  --out /tmp/a-pipeline/a10_start --variants combined50 \
  --roots 1024 --repetitions 3 --threads 8 --affinity 0-7 --seed 20261023 \
  --class-b 32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27
python3 scripts/expansion_bench/run_overall.py --a 9 \
  --out /tmp/a-pipeline/a9 --variants combined50 \
  --roots 1024 --repetitions 3 --threads 8 --affinity 0-7 --seed 20261023 \
  --class-b 32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27
python3 scripts/expansion_bench/run_overall.py --a 8 \
  --out /tmp/a-pipeline/a8 --variants combined50 \
  --roots 1024 --repetitions 3 --threads 8 --affinity 0-7 --seed 20261023 \
  --class-b 32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27
python3 scripts/expansion_bench/run_overall.py --a 10 \
  --out /tmp/a-pipeline/a10_end --variants combined50 \
  --roots 1024 --repetitions 3 --threads 8 --affinity 0-7 --seed 20261023 \
  --class-b 32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27

python3 scripts/expansion_bench/summarize_overall.py \
  --out /tmp/a-pipeline/validation_summary.json \
  /tmp/a-pipeline/validation_complete_a8 /tmp/a-pipeline/validation_complete_a9 \
  /tmp/a-pipeline/validation_complete_a10 /tmp/a-pipeline/validation_wide_a8 \
  /tmp/a-pipeline/validation_wide_a9 /tmp/a-pipeline/validation_wide_a10
python3 scripts/expansion_bench/summarize_a_pipeline.py \
  --out /tmp/a-pipeline/summary.json \
  /tmp/a-pipeline/a10_start /tmp/a-pipeline/a9 \
  /tmp/a-pipeline/a8 /tmp/a-pipeline/a10_end
```

Use new output directories for another run; the runner refuses to overwrite
existing measurement files. Select windows that fit the available VRAM if
the desktop or another workload occupies more memory.
