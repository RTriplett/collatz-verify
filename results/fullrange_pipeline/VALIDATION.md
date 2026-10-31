# Full-range pipeline validation

All eight selected variants passed each of three CPU/GPU comparisons: **24
successful comparisons**. The variants were `parents`, `novote`, `readonly`,
`both`, `both_oee`, `block128`, `block512`, and `replay16`. Every comparison
checked the complete tree counters, per-class endpoint/case/iteration counts,
XOR hash, flagged starting values with multiplicity, and successful wide
rechecks against the CPU implementation. Both CPU expansion engines also
matched before the GPU runs.

The constant-memory OEE follow-up adds ten successful comparisons: four
complete N32 comparisons (`novote` plus three new variants), three M64
comparisons, and three forced-overflow comparisons. The new variants are
`novote_oee`, `novote_oee128`, and `novote_oee512`; they use the same OEE mask
with constant-memory powers and 256/128/512 threads per block. Thus the saved
logs contain **34 successful CPU/GPU comparisons**, including the repeated
`novote` control. Their results match the corresponding rows below, including
985 flags and successful rechecks in each overflow comparison.

Follow-up records: [N32](validate_oee_variants/pipeline.jsonl),
[M64](validate_oee_wide/pipeline.jsonl), and
[overflow](validate_oee_overflow/pipeline.jsonl). These use the later harness
build with the additional aliases; the CUDA and GPU backend implementations
are the same as in the initial comparisons.

The figures below are the common result **per variant**, rather than the sum
of repeated comparisons. An iteration is the verifier's double iteration.

| Profile | N / A / M | Endpoints | Cases | Iterations | Hash | CPU fallback endpoints |
|---|---|---:|---:|---:|---|---:|
| Complete small range | 32 / 10 / 22 | 53,406 | 18,494,045 | 66,053,608 | `CEA9098C` | 29,492 |
| Sixteen sampled K52 packages | 72 / 8 / 64 | 37,337 | 3,413,922 | 15,340,315 | `4063213E` | 20,381 |
| Forced overflow, seventeen K52 packages | 72 / 10 / 62 | 10,751 | 3,257,107 | 20,744,260 | `2C39A418` | 0 |

The first two profiles had zero GPU flags and zero flagged starts. The forced
overflow profile had **985 GPU flags, 985 flagged starts, and 985 successful
256-bit rechecks** for every variant. There were zero unresolved flagged
starts. `VerifyReport.flagged` retains flagged starts even after successful
rechecking; it is not a list of unresolved failures. Flagged origins were
sorted for equality checks because atomic writes and parallel processing can
change their order; duplicates were retained.

The overflow input includes the valid all-one residue `2^52 - 1` at K52, with
F52, plus sixteen deterministic sampled K52 packages. It uses twenty-four
shared bitvector classes with alternating B12/B13 windows and lowers the
overflow threshold to `2^110`. This exercises the real GPU overflow buffer and
the wide recheck path, including the variant that removes the warp vote.

Saved records:

- [Complete N32 comparisons](validate_n32/pipeline.jsonl),
  [metadata](validate_n32/metadata.json).
- [N72 / M64 comparisons](validate_wide_a8/pipeline.jsonl),
  [metadata](validate_wide_a8/metadata.json).
- [Forced-overflow comparisons](validate_overflow_v2/pipeline.jsonl),
  [metadata](validate_overflow_v2/metadata.json).

The initial forced-overflow attempt used the lower threshold `2^97` and
stopped when one launch produced **67,484 flags**, exceeding its **65,536-entry
flag buffer**. This was the buffer-capacity guard, not an observed arithmetic
mismatch. The failed run is retained in
[validate_overflow](validate_overflow/metadata.json), with its
[error log](validate_overflow/pipeline.stderr.txt). Raising the deliberately
low validation threshold to `2^110` preserved actual overflow coverage while
keeping every launch within the buffer capacity.

The complete N32 and M64 runs used binary SHA-256
`150895c9adfa672c5a218c9f83e5ff6bd7889776710885dba21911b5d5858b87`.
The successful overflow run used
`4636317d1877edb9a4657bba6b8571664086451a39fd31d3dc460ba8bdca1f80`.
Comparing the saved source manifests shows that only
`scripts/expansion_bench/fullrange_bench.rs` changed between those builds: the
forced validation threshold and its description. CUDA and Rust GPU backend
sources were unchanged. Each run preserves its own source snapshot and hashes;
these records identify artifacts but do not independently prove build provenance.

The initial full-range confirmation attempt,
[confirmation_a10](confirmation_a10/metadata.json), stopped at a harness
assertion that required the list of flagged starts to be empty. A natural
overflow reached the existing wide-recheck path. Since the verifier retains
successfully rechecked origins in that list, emptiness was an incorrect
benchmark requirement. This was a harness accounting failure; it did not
report a CPU/GPU arithmetic mismatch. The
[failure log](confirmation_a10/pipeline.stderr.txt) and partial records are
preserved, and the incomplete run is excluded from full-range estimates.

The corrected harness requires every flagged start to have a successful wide
recheck. Internal work signatures now explicitly include sorted decimal-string
origins from both the report and its `TailStats`, retaining duplicates and
checking that the two lists agree. (`TailStats` deliberately omits its raw u128
flag list from public JSON.) Aggregate verification compares the summed origin
multiset and successful-recheck count, in addition to all tree/class counters
and XOR hashes. It still rejects unresolved flagged starts.

The corrected binary used for
[confirmation_a10_v2](confirmation_a10_v2/metadata.json) has SHA-256
`2a9e005e9a34a2c84ccb1c62270415b1c41f3998d96cb60bff043cfabb981c47`.
This repair changes the benchmark harness only; CUDA arithmetic and Rust GPU
backend sources are unchanged. The restarted confirmation retains the same
predeclared seed, cohort count, draw count, repetition count, variants, and
aggregate calibration plan.

## CPU checks

[Expansion test output](expansion-tests.txt) records **21 passed tests**,
including the fixed-draw sampler's rejection accounting and duplicate retention,
exact affine reconstruction through M64, terminal-table mappings, and
independent expansion/counter comparisons. The sample test compares the
parallel classification result with a direct sequential classification of the
same saved random draws.

[GPU tuning test output](tuning-tests.txt) records **2 passed tests**: default
tuning encoding and supported block sizes, and exhaustive low-residue checks
of the periodic OEE candidate mask against the original candidate filter.
These two tests were selected specifically; nineteen other library tests were
filtered out in that invocation.

After correcting natural-overflow accounting,
[the repeated expansion test run](expansion-tests-v2.txt) records **23 passed
tests**. The two added regressions verify that fully rechecked reports are
accepted while unresolved starts are rejected, and that aggregate signatures
preserve flag multiplicity, order independence, decimal values above 64 bits,
recheck totals, additive counters, and XOR hashes. The other twenty-one
expansion and sampling checks passed again. This rerun did not change the GPU
kernel or require new GPU arithmetic variants.

## Source audit and measurement scope

`GpuTuning::default()` remains 256 threads per verification block, overflow
votes enabled, constant-memory powers, and the original per-candidate OEE
check. Its parameter encoding is zero. `GpuSession::run_transport()` delegates
to these defaults, and CUDA selects the corresponding compile-time template.
The ordinary verifier still selects its existing DFS and replay transport;
the new tuning variants are opt-in benchmark paths.

A source comparison against the prior transport benchmark's saved CUDA kernel
found the same default overflow condition, atomic flag recording, cutoff reset,
iteration accounting, and XOR accumulation. With default options the added OEE
word mask is all ones and the original per-candidate filter remains active.
The vote-free variant changes the branch condition to each lane's own overflow
predicate while retaining the flag-recording and arithmetic body. The successful
overflow comparisons provide runtime evidence for that path; this audit does
not claim that two separately compiled binaries have identical machine code or
identical timing.

The full-range experiment times the live pipeline after K38: expansion,
packing, queueing, host/device transfers, GPU parent expansion, verification,
and any CPU fallback. Arrays and suffix tables are already resident. Frontier
construction through K38, checkpoint walkback preprocessing, sampling,
one-time resident-array construction, storage, and distributed coordination
are excluded and separately identified in the configuration.

The [completed confirmation](confirmation_a10_v2/pipeline.jsonl) has 64 cohort
trials and two aggregate trials. All same-corpus work signatures agree. Eight
distinct natural overflow starts in six cohorts all complete wide rechecks;
each aggregate reproduces their exact multiset and the summed cohort counters
and XOR checksum. The successful run reuses the original predeclared seeds
after the harness assertion fix; no roots are filtered or replaced because
they overflow.

The [isolated rare CPU probe](rare_cpu_tail/pipeline.jsonl) runs only after the
GPU confirmation finishes. Its 16 cohorts contain 65,536 conditional baseline
F56–62 endpoints, timed twice. Both repetitions agree on all counters, hashes,
and flagged origins. The one-pass totals are 37,282,698 cases, 639,427,584
double iterations, and seven successful wide rechecks, with XOR `72191D6C`.
Forward and backward DP independently give the same 49,063,601-endpoint
population. The [summary](rare_tail_summary.md) is a separate CPU allowance;
it does not alter the main GPU estimate or add to the CPU/GPU comparison count.

Each cohort takes a fixed number of uniform baseline K38 draws with replacement.
Improved13 ancestry and bounded K38 walkback rejections remain in the sample
denominator and contribute zero downstream work; accepted duplicates are
processed again. Independent cohorts estimate sampling variation, while timing
repetitions measure variation on the same work. Extrapolation multiplies mean
time per baseline draw by the exact baseline population. It still assumes that
sampled pipeline throughput represents sustained full-range operation. The
additional concatenated run measures cohort draining and partial-batch effects
and is not an independent sampling replicate. Overlapping CUDA event sums are
diagnostics, not elapsed GPU busy time and not the GPU-year scaling quantity.
