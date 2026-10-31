# Compact chains through GPU dispatch

This experiment retains a K50 parent and references to two resident transition
tables through GPU dispatch. It addresses the expanded endpoint traffic in the
[previous grouped-transition experiment](../grouped_transitions/README.md).
The CPU constructs neither the intermediate K56/K54 frontier nor final M62
endpoints for GPU-bound classes. The ordinary GPU verifier and production
defaults remain unchanged.

**Result:** compact dispatch reduces transfers by about **69%** and process
CPU time by about **45%** in the repeated continuous runs. GPU elapsed time
remains essentially tied with the existing K52 parent pipeline. A flat K50
control obtains the transfer saving too; two-stage sharing mainly reduces
transition-table storage. This experiment establishes no reduction in the
previously confirmed **0.586 RTX 4090-year** estimate.

## Representation and controls

Each 64-byte descriptor contains the exact parent, its power of three, an
inverse modulo 2^32, and a reference to a program for one final odd-count class.
A program references two short affine tables and a dense array of eight-byte
leaf references. These references identify reusable first- and second-stage
words; they do not contain composed twelve-bit affine transformations.

The GPU uses one block per descriptor. It reconstructs up to 64 first-stage
prefixes once in 3 KiB of shared memory, then reuses those exact states for
their second-stage descendants. No intermediate frontier is written to device
global memory. Final 32-byte endpoint records are produced on the device for
the existing tuned lookahead/tail verifier. Batches split by logical leaf
offset without duplicating or dropping work.

| Variant | CPU/GPU boundary | Device expansion |
|---|---|---|
| `novote_oee` | Existing K52 parents | One ten-bit table |
| `early_parents` | K50 parents | One twelve-bit table; control for earlier dispatch |
| `compact66` | K50 parents | Shared six-bit prefixes followed by six-bit suffixes |
| `compact48` | K50 parents | Shared four-bit prefixes followed by eight-bit suffixes |

Optional `_1k` variants flush each producer's class buffer after 1,024
descriptors instead of 4,096. This is explicit host tuning recorded in the
report, with no change to CUDA flags or the 1,048,576-endpoint GPU batch size.
`novote_oee_1k` provides the same scheduling control for the existing pipeline.

All variants use improved checks through K50, the same K38 checkpoint screening,
M62/A10, 19.625 GiB of lookahead windows, and the existing no-vote/early-OEE tail
kernel. Compact programs emit CPU fallback classes through an exact two-stage
CPU traversal. Complete per-depth tree statistics are precomputed and composed,
including prefixes that die before producing a final endpoint.

This prototype shares prefixes within each parent/class descriptor. It does not
combine multiple K50 parents into one physical GPU block. Endpoint order also
changes relative to flat twelve-bit expansion, so timing differences can include
warp coherence as well as arithmetic and transport effects.

## Measurements

The [initial pilot](pilot_summary.md) uses two independent cohorts of 128 K38
baseline draws, two timing repetitions, and seed 20261029. All three K50 paths
take essentially the same elapsed time as K52 parent dispatch (ratios
1.001–1.003), while transferring about 69% fewer bytes. Before the larger run,
two host costs were removed: additions of known-zero prefix counters and
repeated inverse-power calculations. CUDA code is identical between these
runs; their CPU timings should not be pooled.

The [fresh confirmation](confirmation_summary.md) uses four independent
cohorts of 256 baseline draws, two repetitions, and seed 20261030. There are
955 accepted roots from 1,024 draws; rejected draws retain zero downstream
cost in the sampling denominator. Four variant positions rotate across
cohorts and repetitions. The process is pinned to Ryzen 7700X physical cores
0–1 and uses the same RTX 4090 for every variant.

| Variant | Mean cohort wall s | Mean cohort CPU s | Transfer GB/cohort | Wall / baseline [paired 95% interval] |
|---|---:|---:|---:|---:|
| Existing K52 parents | 4.403 | 0.569 | 0.668 | 1.000 |
| K50 single table | 4.368 | 0.262 | 0.210 | 0.992 [0.975, 1.010] |
| Compact 6+6 | 4.397 | 0.290 | 0.210 | 0.999 [0.975, 1.022] |
| Compact 4+8 | 4.391 | 0.292 | 0.210 | 0.997 [0.979, 1.015] |

Thus compact dispatch reduces process CPU time by about **49%** and transfer
bytes by **68.6%** against K52 parents, but the paired samples establish no
elapsed-time saving. The single-table K50 control accounts for the traffic
reduction and most of the CPU benefit. Chaining itself adds some host work
relative to that control.

Chaining does reduce the resident transition representation:

| K50 representation | Device table bytes | Host table bytes |
|---|---:|---:|
| Single twelve-bit table | 9,542,944 | 19,906,556 |
| Compact 6+6 | 2,761,832 | 5,798,144 |
| Compact 4+8 | 3,141,512 | 5,719,072 |

These totals include the dense leaf references. They exclude the common
19.625 GiB of lookahead windows and runtime buffers. The compact expansion
kernel uses 38 registers and 3 KiB shared memory, compared with 40 registers
and no shared memory for the parent expander; neither spills. See the
[compiler resource report](cuda_resources.txt). The verifier is unchanged.

The initial continuous-run check concatenates the same 955 accepted roots.
It takes 18.011 s for K52 parents, 20.396 s for flat K50, 20.520 s for compact
6+6, and 20.834 s for compact 4+8. This disagrees with the small-cohort tie and
motivated a balanced repeated check. All work signatures and per-class launch counts agree,
including both naturally flagged trajectories and their wide rechecks.

The [continuous recheck](continuous_recheck_summary.md) uses the same binary
and identical ordered corpus, reverses the initial variant order, and runs
four continuous repetitions with rotating positions. Each variant occupies
each position once. It also reruns the individual cohorts once. These are
timing repeats of the same sample, not additional independent cohorts.

| Variant | Mean continuous wall s | Wall / K52 baseline | Mean continuous CPU s |
|---|---:|---:|---:|
| Existing K52 parents | 20.219 | 1.0000 | 2.195 |
| K50 single table | 20.405 | 1.0092 | 1.076 |
| Compact 6+6 | 20.292 | 1.0036 | 1.200 |
| Compact 4+8 | 20.310 | 1.0045 | 1.207 |

The earlier 18.011-second baseline did not reproduce. All four designs now
take about 20 seconds, and their ordering changes between repetitions.
Clocks alone do not explain the change: earlier fast periods drew about
429 W at 2670 MHz, whereas the recheck generally draws about 390–400 W at
2685 MHz. The run records do not establish the cause. The within-run
comparison supports an elapsed-time tie, with about **45% less process CPU
time** for compact chains. The automated recheck summary uses its first
variant, `compact48`, as its comparison denominator; the table above uses
the practical K52 baseline explicitly.

## Host dispatch cadence

Compression changes the amount of work represented by each producer flush.
On this corpus, K52 descriptors average about 93 endpoints each, whereas K50
descriptors average about 297. The default 4,096-descriptor chunk therefore
represents roughly 382,000 versus 1.215 million endpoints, respectively. The
latter exceeds one full GPU batch, changing class arrival order and possible
overlap across the three streams. This is a scheduling hypothesis, not an
established explanation for the earlier timing-regime change.

The [chunk recheck](chunk_recheck_summary.md) compares 4,096- and
1,024-descriptor flushing for both K52 and compact 6+6. It uses the same
1,024 draws, one timing repetition per cohort with rotating variant positions,
and two continuous repetitions. The new binary differs only in the recorded
host chunk tuning and corresponding benchmark controls; CUDA code is unchanged.

| Variant | Mean cohort wall s | Mean continuous wall s | Mean continuous CPU s |
|---|---:|---:|---:|
| K52, 4,096 descriptors | 5.042 | 20.549 | 2.138 |
| K52, 1,024 descriptors | 5.090 | 20.607 | 2.105 |
| Compact 6+6, 4,096 descriptors | 5.183 | 20.535 | 1.192 |
| Compact 6+6, 1,024 descriptors | 5.112 | 20.390 | 1.245 |

Smaller chunks improve compact-chain wall time by 1.4% in the cohort means
and 0.7% in the continuous means, at about 4.4% more process CPU time in the
continuous runs. Relative to ordinary K52 dispatch, that compact variant is
1.4% slower in the cohorts and 0.8% faster in the continuous runs. Its paired
cohort ratio is **1.014 [0.990, 1.038]**. Two aggregate rotations do not fully
balance four variant positions, and these small changes do not establish an
overall speedup. Default flushing remains 4,096 descriptors.

All measured variants preserve identical work, including the two natural
flags and wide rechecks. The fresh confirmation and both rechecks reuse the
same four independent cohorts; repeating their timing does not increase the
effective sampling population. These comparisons are not a replacement for
the earlier 8,192-draw full-range cost confirmation. The dominant GPU tail
still executes the same cases and iterations; this design chiefly saves host
work, transfer traffic, and transition-table storage.

## Correctness

- All 29 expansion-benchmark Rust tests pass, including exact endpoints and
  per-depth counters against independent DFS, wide iterates, bit 63, and a sink
  that fails if compact dispatch emits any host endpoint.
- The independent [arithmetic validator](../../scripts/compact_chain_bench/arithmetic_check.cpp)
  executes the actual CUDA-header arithmetic against direct Collatz calculations:
  28,486 cases, 10,993 wide states, 1,042 carrying products, and 304 cases near
  the 128-bit limit. It also checks 8,574 sliced kernel outputs and metadata.
  [Recorded output](arithmetic_check.txt). AddressSanitizer and UBSan pass;
  LeakSanitizer is disabled because this environment uses ptrace.
- All four variants match the CPU reference on a [complete N32 run](validate_small/pipeline.jsonl),
  [sampled N72 work](validate_wide_retry/pipeline.jsonl),
  [forced GPU overflow](validate_overflow/pipeline.jsonl), and
  [M64/A8 work](validate_m64/pipeline.jsonl). Comparisons include full tree
  statistics, per-class counts, iterations, checksums, and flagged-origin
  multisets with multiplicity. All 985 forced-overflow origins complete wide
  CPU rechecking.
- All four reduced-chunk variants also pass the [forced-overflow comparison](validate_chunks/pipeline.jsonl)
  against the existing pipeline, including the same flagged-origin multiset.
- Fifteen report-auditor tests pass, including compact tuning, transport labels,
  descriptor counts, exactly 64 transferred bytes per descriptor, and explicit
  host chunk limits. Six Rust GPU tuning tests also pass.

`validate_wide` records a failed runner startup: `nvidia-smi` returned exit 9
before the benchmark launched. The immediate retry succeeded. This artifact is
retained separately from the successful verification results.

## Reproduction

```bash
CUDA_ARCH=sm_89 cargo build --release --offline --features cuda \
  --manifest-path scripts/expansion_bench/Cargo.toml --bin fullrange_bench \
  --target-dir /tmp/collatz-grouped-target
c++ -O2 -std=c++20 scripts/compact_chain_bench/arithmetic_check.cpp \
  -o /tmp/compact-chain-check
/tmp/compact-chain-check

python3 scripts/expansion_bench/run_overall.py \
  --binary /tmp/collatz-grouped-target/release/fullrange_bench \
  --out /tmp/compact-chain-comparison --profile target \
  --roots 256 --cohorts 4 --repetitions 2 --aggregate-runs 4 \
  --threads 2 --affinity 0,1 --seed 20261030 \
  --class-b 32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27 \
  --variants novote_oee,early_parents,compact66,compact48
```

Each run directory preserves the exact command, source snapshot, binary hash,
raw JSON, and GPU telemetry. Table uploads and variant switches precede trial
timers and are recorded separately as `variant_prepare_seconds`. Resident array
construction, K38 frontier preparation, and distributed coordination remain
outside timed pipeline work.
