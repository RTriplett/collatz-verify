# CPU continuation of long GPU tails

This experiment tests whether spare CPU capacity from the
[compact-dispatch improvements](../compact_chain/README.md) can reduce overall
GPU runtime. The GPU screens candidates normally, then offers unusually long
trajectories to bounded CPU queues while continuing other verification work.

**Result:** this form of CPU offload does not save GPU time. A fresh four-cohort
comparison is about **9–10% slower**, and continuous runs are about **8% slower**.
The largest tested workload shift uses **1.70 busy CPU cores** but removes only
**0.0965%** of GPU-origin tail iterations. Enabling the handoff machinery without
actually handing off work already incurs most of the regression. All correctness
checks pass; production defaults and the confirmed **0.586 RTX 4090-year**
estimate remain unchanged.

The [algorithm flame graph](algorithm_flamegraph/index.html) shows before/after
CPU tree expansion and pruning, GPU endpoint expansion, transfers, screening,
and exact trajectory verification. The [CPU profile](cpu_profile/README.md)
contains sampled call stacks with inline source attribution. A separate
[GPU phase diagnostic](gpu_phase_profile/README.md) splits the fused verifier
using instrumented lane clocks. CPU samples, CUDA activity seconds and
diagnostic clocks retain separate scales; the clocks are not converted into
ordinary kernel seconds. Click frames to zoom into functions, or export the
overview as [PNG](algorithm_flamegraph/comparison.png) or
[SVG](algorithm_flamegraph/comparison.svg).

The earlier [stage-only flame view](flamegraph/index.html) and its
[CUDA trace](flame_profile/README.md) remain available. That capture predates
CPU sampling and leaves GPU screening and stepping in one fused frame.

The earlier [bar-chart comparison](visualization/index.html) uses the original
unprofiled timings and includes both parent and compact dispatch. Its
[PNG](visualization/time_parents.png) and [SVG](visualization/time_parents.svg)
show elapsed runtime separately from CPU continuation and other host work.

## Method and controls

A budget counts completed **double iterations** of the existing tail loop:
an odd run followed by an even run, repeated twice. It does not count individual
Collatz steps. After the budget, a candidate is eligible for handoff only after
its checksum update and descent test. Each 32-byte continuation contains its
128-bit starting value and current shifted state, `T^j(start) + 1`.

The CPU resumes at the next double iteration with identical arithmetic and
cutoff handling. The GPU has already counted the case and its prefix; the CPU
adds only the remaining iterations, checksum contribution, and any eventual
overflow flag. Flagged starts receive the same wide recheck as ordinary GPU
flags. CPU fallback for high odd-count classes remains in the host pipeline.

The initial GPU queue retains up to 65,536 continuations per launch; follow-up
variants increase this to 262,144. A candidate encountering
a full queue finishes on the GPU, without retrying its enqueue. Completed
launches copy their records to owned host storage and submit them to one, two, or four
CPU workers. The host queue holds at most two batches per worker in addition
to work being processed. Workers can overlap later GPU launches and other
streams; a full host queue applies backpressure to dispatch. All CPU work is
joined before the timed pipeline returns.

| Variant family | Purpose |
|---|---|
| `novote_oee` | Existing K52 parent dispatch and tuned GPU tail, with no handoff |
| `defer_count` | Same parent dispatch, with the handoff counter enabled but a budget of `u32::MAX`; validation confirms no handoffs |
| `deferB` | K52 dispatch with budget B and one CPU continuation worker |
| `deferB_w2` | Same budget with two continuation workers |
| `deferB_cap262k_wW` | Budget B, four times the device queue capacity, and W continuation workers |
| `compact66`, `compact_deferB` | Matched K50 compact 6+6 dispatch, without and with CPU continuation |
| `defer1_cap1` | Validation control with budget one and one retained record per launch, forcing full-queue GPU completion |

The offload specialization uses the existing no-vote, constant-powers,
early-OEE verifier with ordinary integer stepping. The
[CUDA resource report](cuda_resources.txt) records 40 registers and no spills
for both this specialization and its ordinary affine verifier control.
Production defaults do not enable handoff.

## Measurement scope

Target runs retain N72/M62/A10, improved checks through K50, K38 checkpoint
screening, and the same 19.625 GiB of lookahead windows. Each cohort starts with
independent uniform baseline K38 draws; rejected roots contribute zero
downstream work. Variants consume identical accepted roots and rotate timing
positions. Timing repetitions are averaged within each cohort. Concatenated
continuous runs reuse those roots and do not increase the independent sample.

The initial pilot uses two cohorts of 128 draws, two repetitions, and seed
20261031. It runs two tree-producer threads with the entire process restricted
to physical CPU cores 0–7; this CPU allocation also contains the GPU dispatch
thread and continuation workers. CPU worker count is separate from the number
of available cores. The runner records actual affinity, GPU telemetry, sources,
binary hash, command, and configuration in each run directory.

Use outer pipeline elapsed time to judge savings: it includes continuation
copies, bounded-queue waits, final CPU draining, and wide rechecks. Summed CUDA
stream event times overlap and exclude the CPU remainder. Total process CPU
time includes host overhead; `deferred_cpu_seconds` measures just continuation
workers while processing their batches. Array construction and resident-buffer
preparation precede trial timers, with variant preparation recorded separately.

Reported GPU-origin cases, iterations, flags, and checksum include the CPU
suffixes so they remain comparable with the ordinary verifier. Device-executed
iterations equal the reported total minus `deferred_iterations`. Continuation
traffic is separate from input traffic and equals 32 bytes per accepted
handoff. Attempts equal accepted handoffs plus full-queue rejections. When a
queue saturates, accepted records reflect arrival order; their mean CPU cost
is not an unbiased estimate for every trajectory beyond the threshold.

These measurements compare implementations on sampled work. They do not
complete an N72 verification or independently replace the earlier full-range
GPU-year confirmation.

## Initial threshold sweep

The [pilot](pilot_summary.md) contains 256 baseline K38 draws, 238 accepted
roots, and two timing repetitions. The table sums cohort means, counting each
input once. Budget values are double iterations.

| Budget / workers | Wall / ordinary K52 | Process CPU core-s | Cases exceeding budget | GPU-origin iterations moved to CPU |
|---|---:|---:|---:|---:|
| Ordinary K52 | 1.000 | 0.721 | — | 0 |
| Count only | 1.090 | 0.717 | 0 | 0 |
| 8 / 1 | 1.803 | 3.295 | 32.065% | 0.02637% |
| 16 / 1 | 1.109 | 3.180 | 2.067% | 0.02782% |
| 24 / 1 | 1.086 | 2.364 | 0.1444% | 0.01889% |
| 32 / 1 | 1.091 | 1.180 | 0.01078% | 0.00429% |
| 64 / 1 | 1.092 | 0.736 | 0.000000494% | 0.000000244% |
| 16 / 2 | 1.097 | 3.357 | 2.067% | 0.02782% |

Only 474 cases across this corpus survive beyond 64 double iterations, out of
roughly 96 billion verified cases. A high threshold leaves essentially no
useful work to move on this corpus. Lower thresholds reach more cases, but fill the bounded
device queues: 99.80%, 96.84%, 70.59%, and 13.27% of handoff attempts are
rejected at budgets 8, 16, 24, and 32 respectively. Rejected work completes
on the GPU and remains in every signature.

Even the largest accepted offload removes just **0.02782%** of GPU-origin tail
iterations. The count-only control is already **9.0% slower**, without any
continuation records or CPU arithmetic. The budget-eight regression is
consistent with additional reservation overhead under heavy saturation;
this experiment does not isolate its exact instruction-level cause. CPU
queue waiting and final draining are small in these runs. Two workers at
budget 16 do not increase the per-launch device admission limit and did not
materially enlarge the admitted workload.

The compact budget-24 variant is also slower than ordinary compact dispatch:
5.529 versus 5.125 seconds in the continuous run, with 2.105 versus 0.340
process CPU core-seconds. All variants reproduce the same cases, iterations,
checksum and flags after GPU and CPU contributions are combined.

## Larger queues and fresh confirmation

The [follow-up](confirmation_summary.md) uses four independent cohorts of 128
baseline draws, seed 20261101, two cohort timing repetitions, and two continuous
repetitions. It accepts 484 of 512 draws. Hardware, windows, two producer
threads, and affinity to physical cores 0–7 match the pilot. The binary adds
larger-capacity variant settings; CUDA and continuation arithmetic are unchanged.
Device queues now retain up to 262,144 records per launch, with two or four
CPU workers. Buffer replacement happens before the timer.

In the cohort comparisons, count-only takes **9.16% more time [8.07%, 10.25%]**
than ordinary K52 dispatch. Budget 24 with the larger queue and two workers
takes **9.64% more [7.52%, 11.76%]**; budget 16 with four workers takes
**10.16% more [9.13%, 11.19%]**. Compact budget 24 takes
**9.49% more [8.19%, 10.80%]** than its matched ordinary compact control.
These are approximate 95% paired cohort intervals, not new absolute cost
estimates. None of the offload variants shows an established improvement
even against count-only instrumentation.

Continuous means over the identical concatenated corpus are:

| Dispatch / budget / queue / workers | Wall s | Process CPU core-s | Average busy CPU cores | GPU-origin iterations moved to CPU |
|---|---:|---:|---:|---:|
| K52, ordinary | 10.199 | 1.376 | 0.135 | 0 |
| K52, count only | 11.000 | 1.433 | 0.130 | 0 |
| K52 / 24 / 65,536 / 1 | 10.979 | 4.752 | 0.433 | 0.01847% |
| K52 / 24 / 262,144 / 2 | 10.966 | 8.305 | 0.757 | 0.03882% |
| K52 / 16 / 262,144 / 4 | 11.014 | 18.749 | 1.702 | 0.09646% |
| Compact 6+6, ordinary | 10.154 | 0.656 | 0.065 | 0 |
| Compact / 24 / 262,144 / 2 | 10.986 | 7.740 | 0.705 | 0.03882% |

Thus larger queues really do use more CPU and transfer more arithmetic, but
neither representation becomes faster. The four-worker run returns **13.82 GB**
of continuation records; host queue waiting averages only **4 ms** and final
draining **13 ms**. The two-worker budget-24 run waits about **3 ms** and drains
in **17 ms**. The measured loss is not explained by CPU workers holding up
the pipeline at the end. Queue saturation remains substantial, but spare CPU
throughput buys little arithmetic relative to the GPU's bulk workload.

The pilot's ordinary pipeline processes about 150 billion double iterations
per second overall, versus roughly 81–88 million per CPU core-second for
the accepted continuations at budgets 8–24. These measurements cover different
work distributions and include different overheads; they are throughput context,
not a conversion from removed iterations into GPU time saved. Likewise, the
count-only control measures the whole enabled configuration, not the cost of
one instruction. The [static SASS comparison](cuda_resources.txt) shows extra
loop instructions with unchanged register count and no spills.

Two continuous rotations do not fully balance seven variant positions. Their
roughly 8% regression agrees in direction with the independently paired cohort
comparison; sub-percent differences among offload settings should not be ranked
as established improvements. This experiment argues against moving individual
tails to the CPU in this implementation. It does not rule out CPU work that
eliminates whole families of GPU candidates through reusable certificates.

## Correctness

All 12 variants agree with the CPU reference in each of four successful runs:

| Run | Coverage |
|---|---|
| [Complete N32](validate_small/pipeline.jsonl) | Full small-range verification, including CPU fallback classes |
| [Sampled N72](validate_wide_direct/pipeline.jsonl) | Sixteen K52 packages with wide iterates and CPU fallback |
| [Forced overflow](validate_overflow/pipeline.jsonl) | N72, cutoff 110, an all-one K52 root plus sixteen sampled roots |
| [M64/A8](validate_m64/pipeline.jsonl) | Wide starts and the bit-64 CPU/GPU boundary |

Comparisons preserve complete tree statistics, per-class endpoints, cases and
iterations, checksum, and flagged-origin multisets including multiplicity.
The forced-overflow fixture finds the same **985 flagged origins** in every
variant, and all finish wide rechecks. With budget one, **813 flags arise while
finishing CPU continuations**, exercising overflow after the handoff rather
than only before it. Budget-one fixtures require a nonempty continuation
queue; the one-record control requires rejected enqueue attempts; the
instrumentation control requires an empty queue.

The [larger-queue overflow check](validate_large_queue/pipeline.jsonl) also
passes for all four capacity-262,144 variants, including four CPU workers,
against the ordinary and count-only controls. All six reports preserve the
same 985 flagged origins and successful wide rechecks.

CPU tests compare independently scalar-stepped prefixes with uninterrupted
verification across 323 surviving randomized N72/M62 trajectories and budgets
1, 2, 4, 8, 16, 32, 64, and 128. Other fixtures cover forced overflow after
handoff, states wider than 120 bits, the 32-byte limb layout, and malformed or
already-descended records. Worker tests preserve duplicate cases, class
accounting and flags with one and two workers, and verify that invalid records
are reported after the bounded queue drains. Report audits check variant
configuration, exact work, queue accounting, and continuation transfer bytes.
All 39 library tests, 29 expansion-benchmark tests, and 19 report-auditor tests
pass. All 54 validation reports and both completed timing runs pass the saved
work and accounting audits.

Two separate startup attempts, [validate_wide](validate_wide/metadata.json)
and [validate_wide_retry](validate_wide_retry/metadata.json), failed because
`nvidia-smi` returned exit 9 before the benchmark launched. They are retained
as failed artifacts and are separate from the successful `validate_wide_direct`
verification.

## Reproduction

Regenerate the interactive flame view from the saved stage trees with Python;
PNG/SVG rendering also requires Matplotlib:

```bash
python3 scripts/expansion_bench/flame_offload.py \
  --input results/tail_offload/flame_profile/trace_trees.json \
  --out results/tail_offload/flamegraph/index.html
python3 scripts/expansion_bench/plot_flame_offload.py
```

Regenerate the bar-chart comparison from audited raw timings with Python and
Matplotlib installed:

```bash
python3 scripts/expansion_bench/plot_offload.py
```

Build into a separate target directory, then reproduce the pilot:

```bash
CUDA_ARCH=sm_89 cargo build --release --offline --features cuda \
  --manifest-path scripts/expansion_bench/Cargo.toml --bin fullrange_bench \
  --target-dir /tmp/collatz-tail-offload-target

python3 scripts/expansion_bench/run_overall.py \
  --binary /tmp/collatz-tail-offload-target/release/fullrange_bench \
  --out /tmp/tail-offload-pilot --profile target \
  --roots 128 --cohorts 2 --repetitions 2 --aggregate-runs 1 \
  --threads 2 --affinity 0-7 --seed 20261031 \
  --class-b 32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27 \
  --variants novote_oee,defer_count,defer8,defer16,defer24,defer32,defer64,defer16_w2,compact66,compact_defer24

python3 scripts/expansion_bench/summarize_offload.py \
  --out /tmp/tail-offload-summary.json /tmp/tail-offload-pilot

python3 scripts/expansion_bench/run_overall.py \
  --binary /tmp/collatz-tail-offload-target/release/fullrange_bench \
  --out /tmp/tail-offload-confirmation --profile target \
  --roots 128 --cohorts 4 --repetitions 2 --aggregate-runs 2 \
  --threads 2 --affinity 0-7 --seed 20261101 \
  --class-b 32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27 \
  --variants novote_oee,defer_count,defer24,defer24_cap262k_w2,defer16_cap262k_w4,compact66,compact_defer24_cap262k_w2

python3 scripts/expansion_bench/summarize_offload.py \
  --out /tmp/tail-offload-confirmation-summary.json /tmp/tail-offload-confirmation
```

Validation uses the same runner with profiles `validate`, `validate-wide`,
and `validate-device`; the M64 fixture uses `validate-wide --a 8`. Exact
validation commands and the 12-variant lists are preserved in each successful
directory's `metadata.json`. The offload summarizer audits completed runs and
reports directly measured comparative costs without promoting small pilots to
new absolute GPU-year estimates.
