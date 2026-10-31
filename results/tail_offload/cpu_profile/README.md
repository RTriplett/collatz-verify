# Sampled CPU algorithms, before and after tail offload

These captures add real CPU instruction-pointer samples and partial call
stacks to the before/after comparison. GPU operations come from the same
captures. [algorithm_trees.json](algorithm_trees.json) is the machine-readable
input for the algorithm flame graph.

**CPU widths are samples, not seconds.** Algorithm categories assign every
running-state sample once; expandable frames retain the measured backtrace
and separately tagged DWARF inline functions. Measured process CPU time is
provided separately. GPU widths are summed device-operation durations, which
overlap across streams and with CPU execution.

## Stack coverage and limits

Nsight marks 3,959 of 4,068 before samples and 59,475 of 59,577 after samples
with a broken-backtrace sentinel. Most still reach the outer thread entry:
3,675 before and 57,238 after contain `Thread::new::thread_start` or libc
`start_thread` before the sentinel. Inspected normal stacks reach `__clone3`
before the unwinder stops. The sentinel therefore does not erase valid
algorithm frames below it, but these should be described as **partial stacks**,
not complete backtraces.

Some stacks stop much earlier. In particular, 1,848 after samples reach
`memcpy`/`memmove` without a usable caller. They appear under **Unattributed
host memory copies**; they are not assigned to continuation transfers by
inference. Unresolved sampled leaves number 268 before and 380 after. Nsight
marks 9 before and 34 after leaves as kernel-mode locations, without resolved
kernel symbols; those records are retained. No non-running samples were
included, and every sampled thread belongs to the launched benchmark process.

The profiled executable uses release optimization, debug level 2 and frame
pointers. Inline resolution succeeds for 3,636 before and 57,186 after samples.
For example, an outer Rust thread-wrapper symbol resolves to
`finish_continuation` / `step_until_below`, and tree code resolves to
`Node::children`, mixed-path screening and pm∞ screening where DWARF supports
that attribution. Otherwise the enclosing operation retains the sample.

## Sampled allocation of CPU work

| Algorithm owner | Before samples | After samples |
|---|---:|---:|
| Host descriptor packing, batch buffers and queues | 2,577 | 2,058 |
| CPU tree expansion and pruning | 1,053 | 1,041 |
| GPU dispatch and result collection on the host | 331 | 405 |
| CPU continuation workers | 0 | 54,077 |
| Host memory copies with unavailable caller | 0 | 1,848 |
| Profiling interception and injected workers | 103 | 136 |
| Other host runtime | 2 | 11 |
| CPU fallback / wide rechecks | 1 | 0 |
| Unresolved algorithm owner | 1 | 1 |
| **All running samples** | **4,068** | **59,577** |

The tree branch expands into reusable suffix tables, K50–K52 baseline
extension, child construction/OEE, mixed-path screening, ternary/path-merge
screening, K38–K50 traversal, and pm∞ screening. The raw resolved functions
and source locations remain available within each category. The categories
are an exclusive accounting of samples, not independently timed functions.
The before/after difference among small stages should not be interpreted as
a performance improvement from this single capture.

Measured pipeline time is 9.917 s before and 10.884 s after. Measured process
CPU time is 1.453 and 18.992 core-seconds, respectively; the after continuation
workers account for 17.005 core-seconds. These instrumented, frame-pointer
build measurements provide context and do not replace the repeated,
unprofiled [experiment results](../README.md). CPU sampling used the same
configured period of 1,393,964 cycle events and one backtrace per sample in
both captures; sample counts are never converted into estimated seconds.

## Scope and validation

The workload is the same 512 baseline K38 draws, 484 accepted roots,
N72/M62/A10, seed 20261101, two producer threads and cores 0–7 used by the
confirmation experiment. The variants are `novote_oee` and
`defer16_cap262k_w4`. Only their warm aggregate calls are captured. Checkpoint
construction, sampling, resident lookahead arrays, suffix-table preparation,
warmups and individual cohort trials are outside the capture ranges. Array
construction uses precomputed path-merge credits; it is not an additional
live CPU cost in these flame graphs.

The strict benchmark auditor validates all 19 JSONL rows, including counts,
checksums, flags, queue accounting, variant settings and exact aggregation.
The corpus and work signatures match the saved confirmation exactly. Each
GPU capture contains 1,858 parent-expansion and 1,858 verification kernels.
Input bytes and output result/count/continuation bytes match the corresponding
benchmark reports. Every CPU sample is assigned once, and every tree node's
width is checked against its children and exclusive samples.

The user enabled process CPU profiling by setting `perf_event_paranoid=2`.
This capture made no global setting changes. Nsight used process-tree CPU
sampling, disabled context-switch tracing, and traced CUDA only in the
launched benchmark. The CUDA source hash matches the established offload
kernel; the subsequent GPU internal-stage diagnostic is a separate build.

## Artifacts and regeneration

- [Raw benchmark JSONL](pipeline.jsonl), [command](command.sh), and [metadata](metadata.json)
- [Before report](cpu_trace.1.nsys-rep) / [SQLite](cpu_trace.1.sqlite)
- [After report](cpu_trace.2.nsys-rep) / [SQLite](cpu_trace.2.sqlite)
- [Before folded stacks](before.folded) / [after folded stacks](after.folded)
- [Cached DWARF inline resolution](inline_frames.json) and [original addr2line output](addr2line.stdout.txt)
- [Executable mapping ranges](process_maps.txt), build logs, and capture source snapshots under `source/`

Regenerate the audited trees using the saved inline cache and traces:

```bash
python3 scripts/expansion_bench/summarize_cpu_profile.py
```

For a fresh capture, the symbolized executable was built with:

```bash
CUDA_ARCH=sm_89 CARGO_PROFILE_RELEASE_DEBUG=2 \
RUSTFLAGS='-C force-frame-pointers=yes' \
cargo build --release --offline --features cuda \
  --manifest-path scripts/expansion_bench/Cargo.toml --bin fullrange_bench \
  --target-dir /tmp/collatz-cpu-profile-target
```

Use the saved profiling command with a fresh output destination. Save the
launched executable's mapping ranges before it exits if generating new inline
resolution: sampled absolute addresses need the process's load base. The
saved cache allows regeneration of this capture without the temporary binary.
