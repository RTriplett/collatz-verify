# Algorithm-level before/after flame graph

[Open the interactive comparison](index.html), or use the [PNG](comparison.png)
and [SVG](comparison.svg). This compares the ordinary K52 parent pipeline with
the experimental four-worker CPU tail offload, on the same 512 sampled K38 draws
(484 accepted roots) used in the confirmation benchmark. It is sampled N72
work, not a completed verification to 2^72.

The initial view groups CPU samples by algorithm stage. Click a stage to
expand its actual sampled function stacks, or enable the full-stack view.
Hover shows the underlying counts, source locations and attribution method.
The default uses a common absolute scale for matching resources; normalize
each tree to inspect the smaller before CPU profile.

## What the three rows measure

1. **CPU algorithm stages:** running CPU samples, with inline function names
   recovered from the captured executable's debug information. Samples are
   assigned once to tree expansion and pruning, packing and queues, dispatch,
   continuation arithmetic, or other explicit categories. CPU core-seconds
   and elapsed seconds are shown separately; samples are not converted to time.
2. **GPU operations:** measured CUDA activity durations from those same
   captures. These separate terminal tree expansion, the fused verifier,
   host-to-GPU copies, GPU-to-host copies and buffer clears. Concurrent
   operations overlap, so their sum is not elapsed runtime.
3. **Inside the fused verifier:** lane-clock measurements from a separate
   instrumented build. These distinguish endpoint setup, Step 2 screening and
   candidate assembly, Step 3 trajectory work, and warp completion/checksum
   reduction. They include stalls, scheduling and instrumentation. They are
   not subdivisions of the ordinary kernel's measured seconds.

The CPU profiles show roughly 63% of before samples in packing/queues and 26%
in tree expansion/pruning. After offload, roughly 91% of samples are in CPU
continuation work. The GPU diagnostic puts about 85% of sampled lane clocks in
trajectory verification. Time awaiting other lanes can accrue at warp
completion, so that branch must not be read as pure checksum arithmetic.

Most CPU stacks include a truncation marker after reaching thread entry; the
captured algorithm frames remain available. Shorter, unattributed stacks are
retained explicitly. In particular, host memory-copy samples with no caller
are not assigned to a presumed transfer or continuation stage. See the
[CPU coverage report](../cpu_profile/README.md) for exact counts.

## Algorithm and scope

Step 1 expands low-bit residue classes from the stored K38 frontier. The CPU
checks pruning certificates and emits grouped terminal work; a GPU kernel
expands those groups into endpoints. Step 2 screens each endpoint's `2^A`
completions using B-step lookahead bitvectors, mod-9 masks and carried OEE.
Step 3 iterates the surviving integer trajectories until descent.

Here A=10 and M=62; B varies by odd-step class. A and B are parameters, not
separate algorithms. The pm∞ and mixed-word lookahead credits are already
encoded in the bitvectors and have no separate live GPU-check cost. Their CPU
tree counterparts can still appear as sampled pruning functions.

The profiles cover the warm pipeline after K38. Frontier construction,
checkpoint walkback and shared lookup-table construction are outside the
captured interval. These attribution runs do not replace the repeated,
uninstrumented performance comparison or its full-range cost estimates.

## Data and reproduction

- [Combined data](data.json), including source hashes and measurement notes.
- [CPU and CUDA captures](../cpu_profile/README.md), with raw traces, inline
  symbolization and exact-work/transfer accounting.
- [GPU phase diagnostic](../gpu_phase_profile/README.md), including independent
  capture offsets, an uninstrumented control and its measurement limits.

From the repository root, after regenerating the two audited input datasets:

```bash
python3 scripts/expansion_bench/combine_algorithm_profile.py
python3 scripts/expansion_bench/flame_offload.py \
  --input results/tail_offload/algorithm_flamegraph/data.json \
  --out results/tail_offload/algorithm_flamegraph/index.html
python3 scripts/expansion_bench/plot_flame_offload.py \
  --input results/tail_offload/algorithm_flamegraph/data.json \
  --out results/tail_offload/algorithm_flamegraph/comparison
```

Only the static plot generator requires Matplotlib. The interactive page is
self-contained and needs no server or network connection.
