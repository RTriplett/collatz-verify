# Early-screen host export diagnostic

The main cost of preparing the compact CPU jobs was writing fresh destination
pages, followed by the software population count. This standalone diagnostic
isolates the operations in `EarlyWorkers::submit` without changing production
code. It does not establish an end-to-end improvement for early CPU offload.

One million endpoints require 32 MiB of affine endpoint data and 128 MiB of
completion masks at A=10. The benchmark divides those arrays into four jobs,
matching a four-worker dispatch. Sources are allocated with
`cudaHostAllocDefault` and filled deterministically on the CPU. No GPU kernel
or transfer runs. The Ryzen 7 7700X executes one benchmark thread pinned to
logical CPU 6, after the other experimental workloads finished.

The table reports medians of five measured repetitions after one complete
warmup. Each repetition rotates the eight variants by three positions and
reverses odd repetitions. This changes order without claiming a complete
balanced Latin-square design. Ranges are observed thread CPU minima and maxima.

| Operation, for all 160 MiB | CPU ms | CPU range ms | Wall ms | Minor faults |
| --- | ---: | ---: | ---: | ---: |
| Software count of the 128 MiB masks | 13.30 | 13.27–13.53 | 13.33 | 0 |
| Runtime-selected hardware POPCNT64 | 2.67 | 2.63–2.85 | 2.68 | 0 |
| Allocate destination capacity without touching | 0.035 | 0.026–0.038 | 0.035 | 8 |
| Allocate and copy | 89.28 | 85.56–90.48 | 89.56 | 40,932 |
| Copy after destination allocation, first touch timed | 90.13 | 86.32–90.71 | 90.44 | 40,924 |
| Copy into reused, resident destinations | 23.34 | 23.25–23.54 | 23.39 | 0 |
| Software count, allocate, copy | 103.10 | 99.01–103.94 | 103.45 | 40,932 |
| Software count, copy into reused destinations | 36.78 | 36.50–36.85 | 36.88 | 0 |

The scan-and-copy case follows the production order within each shard: count
its masks, copy its affine words, then copy its masks. Its 103 ms local cost
accounts for most of the approximately 119 ms of host preparation observed
in the integrated pilot. The different contexts prevent an exact attribution
of the remaining time.

The allocation-only case is cheap, and moving allocation outside the timer
does not materially reduce copying into untouched storage. Fresh copying
generates roughly one minor page fault per destination page and about 42 ms
of median reported kernel CPU time. Reusing resident buffers removes those
faults and saves about 66 ms for the measured job volume. That is evidence
for investigating a bounded reusable buffer pool if this pipeline is pursued;
the full pipeline would still need to pay its initial allocation and first
touch and preserve buffer ownership while workers run.

The production mask expression is
`masks.iter().map(|w| w.count_ones() as u64).sum::<u64>()`.
Default x86-64 compilation emits an SSE2 software population-count loop.
The isolated hardware variant uses runtime CPU-feature detection and a
`#[target_feature(enable = "popcnt")]` helper with four independent u64
accumulators. Its saved disassembly confirms actual POPCNT instructions.
This is a comparison of those implementations, including their loop widths,
not a claim that the instruction alone determines the difference. Globally
enabling native code generation would also change unrelated algorithms and
is unnecessary for this experiment.

The current worker decoder also counts mask bits before allocating each
state vector. A future change could avoid the dispatcher's duplicate count,
but it must preserve the independent accounting check: record submitted jobs
and endpoints, independently count expected candidates in workers, and verify
completed jobs, endpoints, and candidate counts. Simply replacing the expected
count with the number verified would weaken the existing completeness check.
No such change or buffer-pool change was made here.

The source is CPU-filled rather than freshly written by GPU DMA, so cache and
coherency state differ from the real export. The arrays exceed this CPU's
32 MiB last-level cache, but they are deliberately not flushed between trials.
There are no concurrent worker reads, tree expansion, queue waits, or transfers.
Allocator residency and page-fault behavior can differ when worker threads
free jobs. Destruction and validation occur outside timing, as destruction is
outside the production host-copy timer. Thread CPU and wall clocks are the
primary timing measurements; the separately reported `getrusage` user/system
values have coarser accounting and should not be treated as an exact partition
of those clocks. Five repetitions show local consistency, not statistical
confidence or a predicted pipeline speedup. In particular, reducing this
overhead does not establish that moving these short trajectories to the CPU
is worthwhile.

All four counting cases match the independently accumulated population of
536,880,681 bits. Copy checks validate lengths and representative words in
each array, while escaping buffers through `black_box` prevents removal of
the copies. All measured and warmup checks passed, with no major page faults.

From the repository root:

```sh
rustc --edition=2021 -O -C codegen-units=1 \
  results/early_screen/host_export/bench.rs \
  -L native=/usr/local/cuda/lib64 \
  -C link-arg=-Wl,-rpath,/usr/local/cuda/lib64 \
  -o /tmp/collatz-host-export-bench
taskset -c 6 /tmp/collatz-host-export-bench \
  > results/early_screen/host_export/trials.jsonl \
  2> results/early_screen/host_export/stderr.txt
python3 results/early_screen/host_export/summarize.py
```

CUDA device access is needed for the pinned allocation even though no GPU
computation runs. The benchmark assumes Linux x86-64 for its libc timing
structures. See [trials.jsonl](trials.jsonl) for every sample,
[summary.json](summary.json) for machine-readable aggregates, and
[metadata.json](metadata.json) for build, source, hardware, and capture hashes.
The executable remains under `/tmp`; it is not part of the saved results.
