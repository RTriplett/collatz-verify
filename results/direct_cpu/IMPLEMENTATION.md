# CPU routing before GPU dispatch

The experiment leaves the GPU verifier unchanged. A benchmark-only adapter
around the existing affine tree expander intercepts compact terminal groups
before the GPU sink packs them. Selected groups go to CPU workers, which
reconstruct their exact original endpoints, expand to their own depth, apply
their own lookahead, and finish the surviving trajectories.

The implementation is
[`direct_pipeline_bench.rs`](../../scripts/expansion_bench/direct_pipeline_bench.rs).
CPU suffix expansion and verification use
[`src/direct_cpu.rs`](../../src/direct_cpu.rs). Production defaults do not
select this adapter.

## Routing and bounded work

Each group contains one original odd-count class F. Candidate selection hashes
the canonical parent residue/depth and its range in the immutable terminal
table, then selects one residue modulo the configured stride. By default only
the lowest GPU class is eligible: F40 at N72/M62/A10. The same descriptor always
passes or fails this hash test; actual CPU admission also depends on available
worker capacity.

The capacity limit covers **queued, processing, and reserved groups together**.
A producer first reserves a credit with a nonblocking atomic operation. Only
after that succeeds does it copy the parent and terminal-word range. Workers
release credits after finishing a group. If no credit is available, the
unchanged group immediately continues to the ordinary GPU sink. The bounded
channel has the same capacity, with a defensive fallback if its nonblocking
send ever reports full. A full CPU pool therefore does not allocate or copy
the rejected group.

The default capacity is 64 groups. This is a limit on compact source groups,
not the potentially much larger number of their CPU descendants. Workers
materialize one original group at a time; the CPU suffix engine streams its
descendants. Repeated variants with the same CPU A/B share an immutable Plan
and its resident tables.

An initial guard forwards at least 2^22 original endpoints belonging to GPU
classes before allowing CPU admission. It counts endpoints forwarded to the
existing GPU sink, before that sink's packing and queueing. It is **not a live
measurement of GPU queue occupancy**. After the guard opens, admission uses
only the hash test and CPU capacity. Existing rare out-of-range-F fallback
continues to run on the producer as before.

## Accounting and validation

Original endpoint counts are indexed by actual F in vectors of length 81.
For every F, original endpoints equal forwarded plus diverted endpoints.
The forwarded vector includes the existing CPU fallback; its in-range entries
match actual GPU-class counts, while its out-of-range sum matches the original
fallback count. The diverted vector must exactly match what workers completed.

Descriptor count/XOR/wrapping-sum identifiers provide an additional partition
check. They are deterministic diagnostics, not cryptographic proofs. Accepted
groups may differ between runs when worker capacity is exhausted. Changes in
CPU policy can also change descendant counts, cases, iterations, hashes, and
flagged origins. The benchmark therefore keeps the original-depth partition
separate from CPU-native work. The nested GPU report contains the ordinary
pipeline's remaining work; the adjacent `cpu_native` record contains diverted
work at its CPU depth and window policy.

Preflights capture every admitted original CPU endpoint and replay those exact
sources using ordinary Node DFS, direct window evaluation, and scalar tail
verification. This bypasses the compiled CPU suffix table. A same-A/B12
preflight additionally requires the combined CPU/GPU result to match the
complete CPU oracle. N72 packages exercise wide iterates. A separate forced
cutoff case diverts the all-odd F62 source and requires actual CPU cutoff flags
and successful wide rechecks. Target timing runs do not capture source nodes.

In validation JSON, `same_policy_complete_signature_equal: false` means that
the cross-policy complete-signature comparison is inapplicable, not that a
comparison failed. That field is asserted true for matching CPU/GPU policies.
Different-A policies instead require `cpu_policy_oracle_equal: true` and a
valid original-source partition. The frozen benchmark retains this field name
for consistency with its saved executable and logs.

The release test suite also checks that a full CPU pool forwards a group
without copying it, and compares A10/A8/A6/A0 policies against an independent
DFS/window oracle. These checks accompany the CPU module's own arithmetic and
coverage tests.

## Timings

Outer pipeline time includes thread creation, tree expansion, routing,
ordinary GPU work, CPU verification, final worker drain, and all wide rechecks.
Sampling and resident table construction are excluded and recorded separately.
CPU-worker core-seconds, reconstruction/verification intervals, drain time,
copy bytes, admission counts, and peak in-flight groups are reported separately.
CUDA class intervals overlap across streams and are not additive GPU wall time.

The `gpu` control bypasses the routing adapter entirely. `route_control`
executes its descriptor accounting and forwarding with no CPU admission,
exposing adapter overhead. Neither control starts CPU verification workers.

## Running the experiment

Build outside the repository's tracked `target` directory:

```sh
CUDA_ARCH=sm_89 cargo build --release --offline --features cuda \
  --manifest-path scripts/expansion_bench/Cargo.toml \
  --bin direct_pipeline_bench --target-dir /tmp/collatz-direct-pipeline
```

The binary accepts the existing runner's positional protocol. Validation
profiles choose their own fixtures and policies:

```sh
python3 scripts/expansion_bench/run_overall.py \
  --binary /tmp/collatz-direct-pipeline/release/direct_pipeline_bench \
  --profile validate --threads 2 --affinity 0-7 \
  --out /tmp/direct-pipeline/validate
```

Repeat with new output directories for `--profile validate-wide` and
`--profile validate-device`. The latter uses a cutoff of 110 bits; ordinary
target timings use the default 120-bit cutoff.

Variant labels encode CPU A, CPU window B (or `none`), selection stride, and
worker count. Any positive stride is supported. `s1` offers every eligible
group and lets bounded worker capacity determine admission; `s64` offers one
in 64 hash buckets. For example:

```sh
CV_DIRECT_CLASSES=40 CV_DIRECT_QUEUE=64 CV_DIRECT_STARTUP_ENDPOINTS=4194304 \
python3 scripts/expansion_bench/run_overall.py \
  --binary /tmp/collatz-direct-pipeline/release/direct_pipeline_bench \
  --profile target --roots 128 --cohorts 4 --repetitions 1 \
  --aggregate-runs 3 --threads 2 --affinity 0-7 --a 10 --seed 20261103 \
  --variants gpu,route_control,direct_a10_b24_s1_w4,direct_a6_b24_s1_w4,direct_a6_b20_s1_w4,direct_a6_b24_s64_w4 \
  --class-b 32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27 \
  --out /tmp/direct-pipeline/target
```

`--threads 2` sets tree producers; `w4` starts four additional CPU verification
workers. All threads inherit the process affinity. The example does not pin
individual thread roles to distinct cores. The GPU keeps A10/M62 while CPU A6
uses M66, so CPU descendants above the compact 64-bit boundary are represented
with full-width integers.

The target corpus uses the same fixed baseline K38 draw protocol as the
full-range benchmark: duplicates are retained, improved ancestry and bounded
checkpoint walkback are applied, and rejected draws contribute no downstream
work. The concatenated run combines those same accepted cohorts to reduce
small-run flushing effects. These timings are comparisons of sampled packages,
not a completed N72 verification or a new full-range GPU-year estimate.
