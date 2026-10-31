# GPU transport and terminal-table reuse experiments

This experiment tests whether the faster CPU expansion can reduce total pipeline
time by packing smaller endpoint records or preserving terminal-table reuse on
the GPU. It uses sampled N72 packages on a Ryzen 7 7700X and RTX 4090; it is not
a complete verification through 2^72.

The main eight-core comparison shows substantial CPU and transfer savings from
parent/table dispatch, with a small wall-time improvement. Halving the ordinary
endpoint payload or supplying CPU-computed iterates did not materially improve
wall time in this run.

| Variant | Wall seconds, median [min, max] | CPU core-seconds | Average busy cores | Transfer GB |
|---|---:|---:|---:|---:|
| `replay16` | 24.819 [24.499, 24.890] | 28.943 | 1.165 | 69.211 |
| `packed8` | 24.720 [24.646, 24.830] | 25.640 | 1.035 | 34.606 |
| `affine16` | 24.939 [24.797, 25.166] | 49.464 | 1.983 | 69.211 |
| `affine32` | 24.737 [24.415, 25.007] | 59.259 | 2.392 | 138.423 |
| `parents` | 24.414 [24.091, 24.582] | 6.543 | 0.268 | 2.955 |

GB means decimal gigabytes. Each variant has three trials. Parent dispatch uses
about **77% less CPU time and 95.7% less transfer traffic** than replay16. Its
wall-time ratio, pairing each trial with replay16 in the same repetition, has
a median of 0.9834 and a range of 0.9809–0.9904. This roughly 1.7% improvement
is modest and comes from one short comparison on an active desktop; it should
not be projected directly into a general GPU-year saving.

Packed8 halves transfers and reduces CPU time by about 11%, but its paired
wall ratios span both sides of 1.0. Affine32 doubles traffic and uses about
twice replay16's CPU time, with similarly inconclusive wall-time differences.
The affine16 control shows that materializing full iterates on the CPU itself
has a substantial cost. These results favor retaining the shared parent on the
GPU over transmitting an exact iterate for every endpoint. They also show
that large transfer reductions need not yield comparable elapsed-time savings
when the current host already keeps the GPU supplied.

The corpus has 7,213,167 emitting K52 parents, 46,175,333 parent/class group
visits, and 4,325,704,552 M62 endpoints. Batch boundaries increase the parent
record count to about 46.179 million per trial. This explains why measured
parent traffic is 2.955 GB: each 64-byte header belongs to a class group, with
some repeated headers at batch boundaries. It is not one header per parent.

Every timed trial has 427,313,317,731 tested cases, 3,416,237,911,342 tail
iterations and checksum `F782B001`, with no CPU fallback or flags. GPU setup
takes 84.145 seconds and is excluded above. The ordered corpus SHA-256 is
`7b0b5acf7274d1697f2ccdcf5a4d1ea8d33d0e617f5478b417416dcc97f30363`.
See the [generated summary](summary.md), [full accounting](summary.json), and
[raw main-run records](main8/pipeline.jsonl).

The one-core comparison pins the entire process, including the GPU worker,
to physical CPU 0. Parent dispatch retains the eight-core throughput:

| Variant, one core | Wall seconds, median [min, max] | CPU core-seconds | Average busy cores |
|---|---:|---:|---:|
| `replay16` | 26.931 [26.058, 27.035] | 24.721 | 0.918 |
| `parents` | 24.315 [21.952, 24.465] | 2.130 | 0.088 |

The median paired wall ratio is 0.8994, about a 10% improvement on the
constrained host. Both variants are faster in the final repetition; the
active-desktop timing range is wider than in the eight-core run. The ordered
corpus and full work signatures match both runs exactly. One worker also
reduces parent dispatch's CPU time, consistent with less synchronization
overhead. This is evidence that one physical host core suffices for this GPU
and workload, not a measurement of how many GPUs one core could feed. GPU
setup takes 83.362 seconds, outside these trials.

## Selective deeper expansion

The [exact candidate-count probe](CLASS_SPLIT.md) finds that expanding only
original F40 endpoints captures 98.48% of the one-bit candidate reduction
available from expanding every class, or 95.46% of the two-bit reduction.
The corresponding total endpoint growth is only 1.136× or 1.365×, compared
with 1.906× or 3.663× for expanding every class.

A separate GPU microprobe tests descendants of **16,777,216 conditional
M62/F40 sources**, using seven repetitions per class and shape. It measures
the original replay kernel, with shared relative-class windows B32/B35.
Opening and closing A10 controls bracket the A9 and A8 blocks:

| Expansion of original F40 sources | Serialized kernel milliseconds, median [min, max] | Versus opening A10 |
|---|---:|---:|
| None, A10 opening | 20.890 [20.858, 20.900] | reference |
| One bit, A9 | 19.781 [19.738, 19.866] | −5.31% |
| Two bits, A8 | 22.382 [22.311, 22.427] | +7.14% |
| None, A10 closing | 20.894 [20.887, 20.935] | +0.02% |

One extra bit offers a modest kernel benefit in this probe. Two extra bits
lose despite removing more baseline candidates. Both deeper configurations
reduce tail iterations by about 21%, but A8 creates 2.594× the endpoints
compared with 1.594× for A9. The endpoint reconstruction and screening costs
are included in the kernel timings. The earlier
[2,097,152-source probe](class_gpu_summary.md), with three repetitions, found
the same direction: A9 −4.51% and A8 +8.00% versus its opening control. The
larger run extends the same deterministic source sequence. See the
[larger-run summary](class_gpu_large_summary.md) and
[audited JSON](class_gpu_large_summary.json).

This is **kernel timing, not a mixed-depth pipeline measurement**. Sources
are unique uniform baseline M62/F40 draws, screened with improved13 ancestry
through K50 but without K38 walkback. They differ from the saved K38 packages.
Each class is timed separately with its array resident, and the class times
are summed. Host expansion, packing, transfers and setup are excluded. The
probe uses replay16, not the new parent dispatch, and compares fixed relative
windows rather than independently optimized windows. Both runs pass six
class-level CPU/GPU audits on descendants of their common first 16 sources,
with matching repeated work signatures and zero flags.

Combining either split with the main A10 layout would require new
M-dependent B32/B35 arrays. Replacing the old F40 B32 array adds roughly
**4 GiB net**, taking the 19.625 GiB array layout to about 23.625 GiB before
buffers and desktop use. Integration on this 24 GiB GPU therefore needs
window retuning or staged arrays. The measured F40 kernel reduction cannot
be substituted for an overall wall-time reduction.

## Controlled variants

All variants cover the same endpoint set and verification work. The transport
and the way terminal descendants are materialized change:

| Variant | CPU expansion | Host-to-device payload | GPU endpoint preparation |
|---|---|---|---|
| `replay16` | Existing `Combined` compact terminal expansion | 16 bytes per endpoint | Replay M steps |
| `packed8` | Same `Combined` expansion | 8 bytes per endpoint: 62 residue bits and two OEE bits | Replay M steps |
| `affine16` | `Affine` expansion reconstructs exact iterates from the terminal table | Original 16-byte endpoint | Replay M steps; the CPU iterate is not transmitted |
| `affine32` | Same `Affine` expansion | 32 bytes per endpoint, including the exact 128-bit iterate | Read the iterate, without M-step replay |
| `parents` | `Affine` expansion dispatches terminal parents grouped by final F | 64 bytes per parent/class descriptor | Expand resident suffix words into dense 32-byte endpoint records, then verify |

`affine16` controls for the different CPU expansion and grouping before the
`affine32` comparison. The latter trades extra transfers for avoided replay;
it does not isolate either component's time by itself. Parent expansion uses
the same verification kernel as `affine32`, with one preceding expansion kernel
on the same stream. Its output contains exactly the surviving endpoints, with
no padded endpoint verification.

Each parent descriptor contains the starting residue, exact iterate, 3^F,
residue translation, terminal-table range, output offset and depths. A parent
may need several descriptors, one per final-F group; a group can also split
across batches. Reported transfer bytes include all such headers and splits.
The resident terminal words contain the affine suffix coefficients and final
parity state. Their upload is a setup cost.

## Measurement

The main comparison uses the same 1,024 K38 roots and seed `20261023` as the
earlier overall and A-pipeline experiments. Uniform baseline draws reject
improved13 ancestry and bounded K38 walkback with a 65,536-state search cap.
Every variant keeps full checks through K50 and enters the terminal table at
K52, with target endpoint depth M62 and A10. The terminal width is ten bits.
Flat transports materialize M62 endpoints on the CPU; `parents` sends K52
parent/class groups and materializes their M62 descendants on the GPU.

Eight producer workers and the process are pinned to physical CPUs 0–7. Each
variant first warms 32 packages, then runs three repetitions over the complete
saved corpus. Variant order rotates by one position each repetition. All five
variants share one persistent GPU session with three CUDA streams, resident
bitvectors and a resident suffix table. They also use the same experimental
buffer allocation: 64 bytes of input capacity and 32 bytes of expansion scratch
per logical endpoint in each device slot. The production constructor retains
its original allocation.

Shared bitvectors use pm∞ and mix13 window credits. The class-indexed window
vector is:

```text
32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27
```

These arrays occupy approximately 19.625 GiB. The desktop remains active and
the CPU governor is `powersave`. This is a comparison under the stated memory
allocation, without independently retuning the windows for each variant.

Timed wall time includes CPU expansion, packing, queues, pinned transfers,
device expansion where applicable, verification and CPU fallback. Sampling,
terminal compilation, bitvector construction and resident-table upload occur
before the timed trials. The runner records their setup costs separately.
Process CPU time includes producer and GPU worker threads; CPU seconds divided
by wall seconds gives average busy cores. CUDA event intervals include device
expansion for `parents` but exclude transfers. They overlap across streams and
must not be interpreted as GPU busy wall time. Repetition ranges are observed
ranges, not confidence intervals.

A separate, untimed class probe counts the effect of one or two extra baseline
expansion bits by current F class. It measures candidate reductions and parent
reuse on this corpus; it does not measure a mixed-A GPU pipeline or predict its
wall-time saving. The raw candidate reductions concentrate at F40:

| Original M62 class | Candidate reduction with one extra bit | With two extra bits |
|---|---:|---:|
| F40 | 20.288% | 35.144% |
| F41 | 0.154% | 1.177% |
| F42 | 0.123% | 0.219% |
| All classes | 4.717% | 8.430% |

These fractions account for halving or quartering completions per endpoint;
the number of endpoints still increases. Lookahead windows may already remove
many of the same candidates, which is why GPU timing by class is needed.

Each transport run directory preserves the command, K38 corpus, source snapshot, file and
binary SHA-256 hashes, affinity/topology, output and one-second GPU telemetry.
The F40 probes preserve the deterministic sampling method and seed, ordered
FNV identifier, sum/XOR and first 16 source nodes rather than all source nodes.
FNV is an identifier, not a cryptographic hash.
Hashes identify those artifacts; they do not independently prove build
provenance. Compilation and unrelated CPU/GPU work are kept outside timing.

## Correctness

The implementations passed 22 CPU/GPU comparisons:

* Nine complete N32 comparisons: all five variants with shared bitvectors,
  plus four with split families. Packed8 requires shared arrays. Two bitvector
  classes and B12/B13 windows force CPU fallback; a depth-12 walkback checkpoint
  exercises the policy boundary.
* Thirteen N72 package comparisons: five shared-array variants at A10/M62,
  four shared-array variants at A8/M64, and four split-family variants at
  A8/M64. These use sixteen sampled K52 parents and explicitly require GPU
  endpoints with the top residue bit set, iterates above 64 bits and fallback.

Every comparison checks all tree counters, per-class endpoint/case/iteration
counts and checksums against the CPU reference. A8 validation exercises bit 63
and exact affine arithmetic above 64 bits. CPU fallback receives genuine full
`Node` values. The target harness also requires identical work signatures
across warmups and timed variants, with no flagged trajectories.

The validation and measurement binaries are not all byte-identical. A
cross-variant warmup assertion was added after the validation runs, before
`main8`. Later `GpuBench`-only accounting changes, used for serialized class
probes, were added after `main8`; subsequent benchmark binaries therefore have
different hashes. Each run preserves its actual binary hash and source snapshot.
The completed target run independently checks every timed work signature, so
the 22 comparisons are attributed to the implementations rather than to a
single measured executable.

[Expansion tests](expansion-tests.txt) contain 20 passing tests, including exact
affine nodes against independent DFS and resident group ranges. The
[CPU test log](cpu-tests.txt) contains 19 library tests and five bitvector
audits, all passing.

The first [wide validation attempt](validation_wide_a10/metadata.json) failed a
coverage assertion because the coverage classifier used an F offset one below
the actual first GPU class. That classifier was corrected before measurement.
The failed record is retained and excluded from completed validation results.
The final complete-range validations are `validation_complete_v2` and
`validation_complete_split_v2`; the earlier complete-range runs are also
retained but are not counted twice. The final wide validations are
`validation_wide_a10_v2`, `validation_wide_a8` and `validation_wide_a8_split`.

## Reproduce

Build before measurement, adapting CUDA architecture and CPU affinity for the
machine. Use new output directories; the runner refuses to overwrite results.

```sh
CUDA_ARCH=sm_89 cargo build --release --offline --features cuda \
  --manifest-path scripts/expansion_bench/Cargo.toml --bin transport_bench \
  --bin class_split_bench \
  --target-dir /tmp/collatz-transport-target

python3 scripts/expansion_bench/run_overall.py \
  --binary /tmp/collatz-transport-target/release/transport_bench \
  --profile validate --a 10 --out /tmp/transport-pipeline/validation_complete
python3 scripts/expansion_bench/run_overall.py \
  --binary /tmp/collatz-transport-target/release/transport_bench \
  --profile validate --a 10 --variants split \
  --out /tmp/transport-pipeline/validation_complete_split
python3 scripts/expansion_bench/run_overall.py \
  --binary /tmp/collatz-transport-target/release/transport_bench \
  --profile validate-wide --a 10 --out /tmp/transport-pipeline/validation_wide_a10
python3 scripts/expansion_bench/run_overall.py \
  --binary /tmp/collatz-transport-target/release/transport_bench \
  --profile validate-wide --a 8 --out /tmp/transport-pipeline/validation_wide_a8
python3 scripts/expansion_bench/run_overall.py \
  --binary /tmp/collatz-transport-target/release/transport_bench \
  --profile validate-wide --a 8 --variants split \
  --out /tmp/transport-pipeline/validation_wide_a8_split

python3 scripts/expansion_bench/run_overall.py \
  --binary /tmp/collatz-transport-target/release/transport_bench \
  --a 10 --out /tmp/transport-pipeline/main8 \
  --variants replay16,packed8,affine16,affine32,parents \
  --roots 1024 --repetitions 3 --threads 8 --affinity 0-7 --seed 20261023 \
  --class-b 32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27

python3 scripts/expansion_bench/run_overall.py \
  --binary /tmp/collatz-transport-target/release/transport_bench \
  --a 10 --out /tmp/transport-pipeline/one_core \
  --variants replay16,parents \
  --roots 1024 --repetitions 3 --threads 1 --affinity 0 --seed 20261023 \
  --class-b 32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27

python3 scripts/expansion_bench/run_overall.py \
  --binary /tmp/collatz-transport-target/release/class_split_bench \
  --out /tmp/transport-pipeline/class_gpu \
  --roots 2097152 --repetitions 3 --threads 8 --affinity 0-7 \
  --seed 20261025 --a 10

python3 scripts/expansion_bench/run_overall.py \
  --binary /tmp/collatz-transport-target/release/class_split_bench \
  --out /tmp/transport-pipeline/class_gpu_large \
  --roots 16777216 --repetitions 7 --threads 8 --affinity 0-7 \
  --seed 20261025 --a 10

python3 scripts/expansion_bench/summarize_class_split.py \
  --out /tmp/transport-pipeline/class_gpu_summary.json \
  /tmp/transport-pipeline/class_gpu
python3 scripts/expansion_bench/summarize_class_split.py \
  --out /tmp/transport-pipeline/class_gpu_large_summary.json \
  /tmp/transport-pipeline/class_gpu_large

python3 scripts/expansion_bench/summarize_transport.py \
  --out /tmp/transport-pipeline/summary.json \
  /tmp/transport-pipeline/main8 \
  /tmp/transport-pipeline/one_core \
  /tmp/transport-pipeline/validation_complete \
  /tmp/transport-pipeline/validation_complete_split \
  /tmp/transport-pipeline/validation_wide_a10 \
  /tmp/transport-pipeline/validation_wide_a8 \
  /tmp/transport-pipeline/validation_wide_a8_split
```

The default `cv verify` command still uses the original DFS and transport.
These changes are experimental paths exercised through the live production
queues, streams and tail kernel.
