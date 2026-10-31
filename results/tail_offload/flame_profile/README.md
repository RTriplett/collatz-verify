# Before/after device activity profile

These two Nsight Systems captures show where device operations ran in the
ordinary K52 parent pipeline and the four-worker CPU continuation experiment.
They use the exact confirmation corpus: 512 baseline K38 draws, 484 accepted
roots, N72/M62/A10, two producer threads, affinity to physical cores 0–7, and
the same 19.625 GiB lookahead windows.

The capture begins with `cudaProfilerStart()` immediately before each warm
aggregate trial and ends with `cudaProfilerStop()` after the trial returns.
Those calls are enabled only by `CV_FULLRANGE_PROFILE_AGGREGATE=1`; setup,
table construction, warmups and individual cohort trials are outside the
captured ranges. CUDA kernels are unchanged. Only the benchmark executable
was rebuilt; build output and the previous binary hash are retained here.

## What the widths mean

[trace_trees.json](trace_trees.json) contains additive stage trees for the
interactive flame graph. GPU widths are **summed device-operation seconds**,
with compute, copies and buffer-clearing branches and per-stream children.
They are measured activity durations, not sampled call stacks. Overlapping
streams contribute separately, so the GPU total exceeds elapsed runtime.
CPU widths are busy CPU core-seconds and are shown on a separate scale.

The verifier is a fused CUDA kernel: its screening, integer stepping and
handoff bookkeeping are not separately timed. The trace cannot attribute
cost among those internals. CPU sampling was disabled, so CPU attribution
stops at continuation workers versus other host work.

| Measured duration | Before: `novote_oee` | After: `defer16_cap262k_w4` |
|---|---:|---:|
| Fused candidate screening and tail verifier, summed kernel s | 11.83423 | 13.11756 |
| Parent expansion, summed kernel s | 0.23541 | 0.26593 |
| Host-to-device copies, summed operation s | 0.05312 | 0.05370 |
| Device-to-host copies, summed operation s | 0.00199 | 0.54427 |
| Buffer clears, summed operation s | 0.00117 | 0.00351 |
| All recorded device operations, summed s | 12.12592 | 13.98497 |
| Pipeline elapsed s | 10.02949 | 11.13059 |
| CPU continuation core-s | 0 | 16.63872 |
| Other process CPU core-s | 1.45012 | 1.95729 |

Each capture includes 1,858 parent-expansion kernels and 1,858 verifier
kernels. Exact input-copy bytes match each benchmark report. Returned bytes
match the 24-byte result per launch, the additional four-byte offload counts,
and all admitted 32-byte continuation records. The after capture returns
13,823,839,616 bytes of continuation records.

These are single captures with profiling enabled. Host activity tracing can
add CPU overhead. They explain attribution and do not replace the repeated,
unprofiled performance measurements in the [experiment report](../README.md).
Device operations overlap CPU execution and one another; none of the summed
durations should be added to elapsed runtime.

## Validation and artifacts

The existing strict report auditor passes all warmups, cohort trials and
aggregate trials, including counts, checksums, flags, queue accounting and
variant settings. The accepted corpus and both aggregate work signatures
match the saved confirmation exactly. Trace launch counts and transfer bytes
are checked against the corresponding benchmark reports.

- [Raw benchmark JSONL](pipeline.jsonl)
- [Before Nsight report](direct_trace.1.nsys-rep) and [SQLite export](direct_trace.1.sqlite)
- [After Nsight report](direct_trace.2.nsys-rep) and [SQLite export](direct_trace.2.sqlite)
- [Metadata, sources and binary hash](metadata.json), with snapshots under `source/`
- [Capture command](command.sh)

Nsight stores application stdout in each SQLite `ProcessStreams` table.
`pipeline.jsonl` uses the second report's complete stdout prefix, followed by
the unchanged final aggregate row saved in [final_aggregate.json](final_aggregate.json).
The final row is printed after the last `cudaProfilerStop()`, so it was retained
from the profiling command's stdout instead of the report.

Two sandbox startup attempts failed before launching the workload with
`open: Operation not permitted`. Their outputs and commands are retained.
A single approved process-scoped capture then succeeded with CPU sampling
and context-switch tracing disabled. No system or driver settings changed.

Regenerate the audited stage trees from the saved data:

```bash
python3 scripts/expansion_bench/summarize_flame_profile.py
```

For a fresh capture, build with the existing CUDA architecture setting and
use the saved command with a new `--output` destination. The capture command
sets `CV_FULLRANGE_PROFILE_AGGREGATE=1` and takes two repeated capture ranges,
one per variant; normal benchmark execution leaves profiling disabled.
