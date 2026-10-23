# Direct enumeration of terminal blocks

These experiments replace the final baseline-only DFS levels with reusable
tables of surviving parity words. Every variant in a run starts from the same
sampled K=38 frontier and applies the same pruning policy: pm∞ + mix13 through
the chosen cutoff, then Angeltveit's descent, pm1, and root-armed OEE rules to
M=62. With N=72 this corresponds to A=10. Changing the terminal block width
changes how those decisions are computed, not which endpoints survive.
Production `src/` is unchanged.

Roots are sampled uniformly from the baseline K=38 frontier and rejected when
improved-rule ancestry or capped checkpoint walkback provides a certificate.
The walkback cap is 65,536. This samples a frontier rather than replaying a
stored archive. The same accepted roots are reused for every timed variant.

## Implementations

| Variant | Expansion below the common full-check cutoff |
|---|---|
| `production` | Original full-width `Node` traversal with baseline rules |
| `compact` | Specialized baseline DFS carrying only the iterate bits needed for subsequent parity decisions |
| `flatW` | Compact DFS to M−W, followed by direct emission from a flat table of valid terminal words |
| `groupW` | The same table computation, with words grouped by final `(f, oee, ne)` state |

`production` is a control using the original traversal under this experiment's
cutoff policy; it does not mean that the repository's production verifier has
adopted that policy. `compact` is the main performance control, since it already
includes the earlier baseline-suffix optimization.

The width sweep supports W=4, 6, 8, 10, and 12. A width is available only when
M−W is at or beyond the full-check cutoff: all five widths fit cutoff 50 at
M=62, whereas cutoff 54 admits widths 4, 6, and 8.

For a table-boundary class `(n0, k, f, m)`, a terminal parity word has an input
residue `r mod 2^W`. Its starting-number extension is

```text
q = (r - m) * 3^(-f) mod 2^W
n0_extended = n0 + q * 2^k
```

The inverse exists because 3 is odd. Writing
`R = r * 3^(-f) mod 2^W` and `D = m * 3^(-f) mod 2^W` gives
`q = (R - D) mod 2^W`. The table precomputes R, checks
every intermediate baseline rule, and stores the final endpoint state. Runtime
emission needs one modular multiplication per table-boundary class, followed
by a subtraction, mask, and shift for each surviving word. It does not compute
intermediate or final iterates. Tables also store complete per-depth counters,
including branches that die before the terminal depth, so counters match the
ordinary traversal even though those branches are never visited at runtime.

All workers share each immutable table. The recorded `tables[].bytes` includes
both flat and grouped representations, counter arrays, program slots, and
container headers; allocator overhead is excluded. It is not a per-worker
allocation or the memory requirement of a production implementation retaining
only its chosen layout. Table sizes, entry/state counts, and total setup time
are retained in each run's config. Construction is outside the timed region.

## Timing scope and validation

Per-depth counters are enabled in every timed variant. Each run warms the
selected variants and rotates their execution order across repetitions. The
reported ratios compare variants within one run, with its fixed cutoff,
corpus, and sink; different cutoff policies are not pooled.

The `digest` sink counts endpoints, accumulates a wrapping sum of `n0`, XORs
the final state, and counts endpoints by F. The `packed` sink also stores four
`u32` words per endpoint in separate F buffers, exposing each 4,096-endpoint
chunk to `black_box` before reuse. This exercises endpoint stores and bounded
buffering. It retains the digest work and does not include the production CPU
fallback, queueing, locks, GPU transfer, GPU execution, or queue backpressure.
Both sinks exclude table construction, root sampling, correctness audits, and
archive I/O. Neither is a complete verifier-pipeline measurement.
The packed sink buffers every F class; production instead sends classes
without an applicable GPU array to CPU endpoint processing. Table emission
also changes endpoint order, whose effect on GPU execution needs device timing.

All variants must match endpoint digests and every per-depth counter on every
timed repetition. Each run additionally compares complete endpoint payloads
against the original traversal on 64 sampled roots near the terminal region;
the audit is outside timing. Endpoint order may differ between layouts.

[tests.txt](tests.txt) records the package test run: **33 successful test
executions covering 26 distinct tests**. The terminal binary runs 15 tests:
eight new table/integration tests plus seven reused tail tests. The new tests
cover the modular mapping's bijection for every supported width, complete
small frontiers, an independent suffix oracle, branches producing no terminal
endpoints, sampled wide iterates, target depth 64 and bit 63, both layouts,
enabled/disabled counters, and rejection of a table that would bypass full
checks before the cutoff. The seven tail tests also run in their original
binary, accounting for the repeated executions.

The [assembly inspection](assembly-notes.json) confirms that the optimized
digest implementations still reconstruct every endpoint in a residue loop and
that both packed implementations retain four `u32` stores per endpoint. Some
grouped digest updates are aggregated by the compiler. Precomputed tree counts
are deliberately merged once per table-boundary class, so the measured gain
includes cheaper accounting as well as cheaper endpoint enumeration.

## Measured results

All six runs completed successfully on eight physical Ryzen 7 7700X cores.
Every variant matched its run's endpoint digest and every per-depth counter.
The initial 128-root corpus (seed 17411, five repetitions) produced 455,980,448
endpoints per trial under cutoff 50. Median wall times were:

| Variant | Digest sink | Packed sink | Table bytes, both layouts |
|---|---:|---:|---:|
| `compact` | 0.71851 s | 0.75145 s | — |
| `flat4` | 0.15157 s | 0.26414 s | 102,600 |
| `flat6` | 0.08708 s | 0.20725 s | 166,480 |
| `flat8` | 0.06663 s | 0.18219 s | 317,768 |
| `flat10` | 0.06144 s | 0.17859 s | 811,656 |
| `flat12` | 0.05924 s | 0.16909 s | 2,582,072 |

The original full-width `production` control took 1.26472 s with the digest
sink and 1.32921 s with packing. Grouped layouts were also timed; all of their
results are in [summary.json](summary.json). Building the setup with all five
tables took 0.0371 s in the digest sweep, outside the timed expansion. This
includes rule setup; root sampling and archive I/O are separate.

An independent 256-root corpus (seed 17412, seven repetitions) produced
1,085,114,524 endpoints per trial at cutoff 50:

| Sink | `compact` | `flat10` | `flat12` | Reduction vs compact |
|---|---:|---:|---:|---:|
| Digest | 1.61134 s | 0.13715 s | 0.13062 s | 91.49–91.89% |
| Packed | 1.65550 s | 0.37997 s | 0.38828 s | 76.55–77.05% |

Both widths beat compact DFS in all seven paired repetitions for both sinks.
The small 10/12-bit ranking reverses in the packed confirmation; these results
support a strong table-enumeration gain, not a universal best width of 12.

Keeping full checks later still helps when the terminal table fits entirely
below that cutoff. On the same 256 roots with packing, cutoff 54's `flat8`
took **0.43784 s versus 1.68152 s** for its compact control (**73.96% less**).
Cutoff 58's `flat4` took **1.04117 s versus 2.14556 s** (**51.47% less**).
Each comparison uses its own matching policy; those cutoffs intentionally
have different endpoint sets.

Tables introduce no extra pruning and admit no extra endpoints relative to
their matched cutoff policy. The earlier checkpoint-tail probe estimated
additional GPU work of 1.11%, 0.80%, and 0.43% for cutoffs 50, 54, and 58,
respectively, relative to full checks with the same checkpoint. Those are
[previous modeled estimates](../checkpoint_tail/README.md), not GPU timings
or new measurements from this experiment.

## Artifacts and provenance

Each run directory contains `cpu.jsonl`, `cpu.stderr.txt`, and `metadata.json`.
The metadata records the command, timestamps, completion status, binary/source
hashes, compiler version, CPU information, scaling governors, and affinity.
`source/` snapshots inputs relative to the repository root, including the Rust
experiment files, Cargo manifests/locks, production Rust sources, and runners.
The runner records the binary hash but does not itself establish that the
binary was built from those sources; build immediately before running.

| Directory | Roots / repetitions / seed | Cutoff | Sink and variants |
|---|---|---:|---|
| [width-sweep/](width-sweep/) | 128 / 5 / 17411 | 50 | Digest; production, compact, flat/group widths 4, 6, 8, 10, 12 |
| [packed-sweep/](packed-sweep/) | 128 / 5 / 17411 | 50 | Packed; the same complete width sweep |
| [confirm-digest/](confirm-digest/) | 256 / 7 / 17412 | 50 | Digest; compact, flat8, flat10, flat12, group12 |
| [confirm-packed/](confirm-packed/) | 256 / 7 / 17412 | 50 | Packed; the same confirmation variants |
| [cutoff54-packed/](cutoff54-packed/) | 256 / 7 / 17412 | 54 | Packed; compact, flat4, flat6, flat8, group8 |
| [cutoff58-packed/](cutoff58-packed/) | 256 / 7 / 17412 | 58 | Packed; compact, flat4, group4 |

[summary.json](summary.json) contains all six comparisons. The summarizer
preserves complete config records,
including table memory, and skips incomplete, failed, or output-mismatched
runs. Timing summaries contain medians, observed extrema, and ratios paired
by repetition against `compact` and, when present, `production`. Paired ranges
are observed extrema, not confidence intervals.

For exact reruns, use a run's saved command and source snapshot; rebuilding
from later source can change code layout and timings. Independent-corpus
confirmation and the packed sink help distinguish expansion gains from
effects specific to one root sample or endpoint sink.

## Reproduction

Run these commands sequentially from the repository root. They reproduce the
saved run parameters and selected variants into fresh output directories.
The default affinity, CPUs 0–7, uses the eight physical cores on the Ryzen
7700X benchmark host; use `--affinity` to adapt it on another machine. Use
fresh output directories because the runner refuses to overwrite measurements.
`--offline` requires cached dependencies.

```sh
cargo test --offline --manifest-path scripts/expansion_bench/Cargo.toml \
    --target-dir /tmp/collatz-expansion-target
cargo build --release --offline --manifest-path scripts/expansion_bench/Cargo.toml \
    --target-dir /tmp/collatz-expansion-target --bin terminal_bench

python3 scripts/expansion_bench/run_blocks.py \
    --out /tmp/terminal-block-repro/width-sweep \
    --roots 128 --repetitions 5 --threads 8 --seed 17411 \
    --target 62 --cutoff 50 --variants all --sink digest

python3 scripts/expansion_bench/run_blocks.py \
    --out /tmp/terminal-block-repro/packed-sweep \
    --roots 128 --repetitions 5 --threads 8 --seed 17411 \
    --target 62 --cutoff 50 --variants all --sink packed

python3 scripts/expansion_bench/run_blocks.py \
    --out /tmp/terminal-block-repro/confirm-digest \
    --roots 256 --repetitions 7 --threads 8 --seed 17412 \
    --target 62 --cutoff 50 --variants compact,flat8,flat10,flat12,group12 --sink digest

python3 scripts/expansion_bench/run_blocks.py \
    --out /tmp/terminal-block-repro/confirm-packed \
    --roots 256 --repetitions 7 --threads 8 --seed 17412 \
    --target 62 --cutoff 50 --variants compact,flat8,flat10,flat12,group12 --sink packed

python3 scripts/expansion_bench/run_blocks.py \
    --out /tmp/terminal-block-repro/cutoff54-packed \
    --roots 256 --repetitions 7 --threads 8 --seed 17412 \
    --target 62 --cutoff 54 --variants compact,flat4,flat6,flat8,group8 --sink packed

python3 scripts/expansion_bench/run_blocks.py \
    --out /tmp/terminal-block-repro/cutoff58-packed \
    --roots 256 --repetitions 7 --threads 8 --seed 17412 \
    --target 62 --cutoff 58 --variants compact,flat4,group4 --sink packed

python3 scripts/expansion_bench/summarize_blocks.py /tmp/terminal-block-repro \
    --out /tmp/terminal-block-repro/summary.json
```

The runner defaults to
`/tmp/collatz-expansion-target/release/terminal_bench`; `--binary` overrides
that location. Narrow a sweep with a comma-separated `--variants` list,
always including `compact` as its control.
