# Diagnostic phases inside the fused GPU verifier

This experiment separates the verifier into endpoint setup, Step 2 completion
screening and assembly, Step 3 exact trajectory stepping, and warp completion
with checksum reduction. The normal verifier keeps these stages in one CUDA
kernel. Actual transfer and endpoint-expansion durations remain available in
the [combined CPU sampling and CUDA activity capture](../cpu_profile/README.md).

The measurement is **sampled lane elapsed cycles**, not GPU execution seconds.
All live lanes in one block per 128 block indices record four disjoint
`clock64` intervals. The selected block offset advances with every launch and
changes by 53 between aggregate captures. The intervals partition each sampled
lane's measured residence exactly, including stalls, scheduling and profiling
instructions. Diagnostic counter writes happen after the final timestamp.
Those writes can still affect other warps' scheduling.

Warp-completion time includes waiting for longer-running lanes at
reconvergence and full-warp reductions. It must not be interpreted as pure
checksum arithmetic. The other intervals can also include SIMT waiting.
Across concurrent lanes these cycle counts overlap extensively; converting
their sum into wall time or stacking them with CUDA memcpy durations is invalid.

## Scope and validation

The two variants are the K52-parent baseline `novote_oee` and experimental
CPU tail continuation `defer16_cap262k_w4`. They use the same 512 baseline
draws, 484 accepted roots, seed 20261101, N72/M62/A10, K50 improved-check
cutoff, per-class 19.625 GiB windows, two CPU producers and cores 0–7 as the
[saved confirmation](../confirmation/). The GPU is an RTX 4090.

There are two warm aggregate captures per variant with rotated execution
order. The captures reuse the same roots; different sampling offsets provide
a sensitivity check, not independent root samples or a precision estimate.
Setup, table construction, warmup and individual-cohort trials are excluded
from the phase counters.

The benchmark checks exact tree counters, per-class cases and double
iterations, XOR checksum, flagged-origin multiset and wide rechecks, both
between variants and against concatenated cohort results. The summarizer
additionally runs the existing strict full-range auditor, compares the saved
confirmation's ordered corpus and complete work signatures, checks launch
counts and sampled-work bounds, and verifies exact integer cycle partitions.

## Perturbation and interpretation

The overlay changes generated code. In particular, the normal baseline and
offload kernels use 40 registers, while the diagnostic versions use 52 and 53
respectively. Neither diagnostic kernel spills. Sampling sparsely does not
remove this change in register allocation from unsampled blocks.

The runtime control in [control/](control/) runs the original uninstrumented
binary on the same corpus with the same two aggregate repetitions, after the
diagnostic run. [phase_trees.json](phase_trees.json) reports the individual
captures, phase shares, fresh-control runtime ratios, and older confirmation
ratios. These sequential runtime comparisons include ordinary run-to-run
variation and do not isolate a universal profiler overhead.

| Diagnostic phase | Baseline lane cycles | CPU-offload lane cycles |
|---|---:|---:|
| Endpoint and filter setup | 0.084% | 0.074% |
| Step 2: screening and candidate assembly | 5.379% | 5.756% |
| Step 3: exact stepping and handoff | 85.637% | 85.252% |
| Warp completion and checksum reduction, including waiting | 8.901% | 8.919% |

These shares combine two captures, each sampling about 15.14 million endpoint
lanes and 1.489 billion cases. The Step 3 shares differ by less than 0.06
percentage points between the two baseline captures and less than 0.01 points
between the offload captures. This stability does not remove instrumentation
bias.

| Variant | Instrumented wall time, mean | Fresh control, mean | Observed ratio |
|---|---:|---:|---:|
| Baseline | 9.527 s | 8.673 s | 1.0984× |
| CPU offload | 10.271 s | 9.420 s | 1.0902× |

The instrumented runs are roughly 9–10% slower than the fresh control. Both
fresh control times also differ from the older confirmation; these diagnostic
runs do not replace the established full-range performance estimate.

The phase shares explain the instrumented program. They should remain a
separate diagnostic panel and must not be used to subdivide measured durations
from the uninstrumented CUDA trace into supposedly exact seconds.

## Files and reproduction

[`profile_gpu_phases.py`](../../../scripts/expansion_bench/profile_gpu_phases.py)
copies the project into `/tmp`, inserts instrumentation using checked source
anchors, and builds there. Production `cuda/kernel.cu`, `build.rs`, and the
benchmark are unchanged. The only replaced benchmark hooks are the existing
optional aggregate `cudaProfilerStart`/`cudaProfilerStop` calls in the copied
source. They reset/read phase counters instead.

```bash
python3 scripts/expansion_bench/profile_gpu_phases.py --build
bash results/tail_offload/gpu_phase_profile/command.sh
bash results/tail_offload/gpu_phase_profile/control/command.sh
python3 scripts/expansion_bench/profile_gpu_phases.py --summarize
```

Use `--refresh` when regenerating a previously created overlay. The generator
accepts only marked overlay directories under `/tmp` for replacement. A fresh
machine also needs to build the original benchmark binary at the path recorded
in the control command, using the normal CUDA build; its checksum must be
recorded in the control metadata. Commands overwrite result logs, so use a new
output directory for an independent capture and record its command, binary
hash, exit status and metadata before summarizing.

The raw captures are `pipeline.jsonl` and `pipeline.stderr.txt`; the latter
contains four `gpu_phase_profile` JSON records. `build_metadata.json` preserves
input hashes, generated-source hashes and build settings. `source/` preserves
the four changed overlay files. `ptxas.log` and
`control/cuda_resource_usage.txt` record register usage. `command.sh` and
`control/command.sh` preserve the exact executions. The initial sandbox could
not query GPU memory; its failed logs are preserved as `sandbox_failed.*`.
The successful runs used process-scoped access outside the sandbox and changed
no driver permissions or settings.
