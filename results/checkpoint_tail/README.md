# Checks after a depth-38 checkpoint

These experiments compare expansion from the same K=38 roots to M=62
(N=72, A=10). Roots are sampled uniformly from the baseline frontier, then
rejected if pm∞ + mix13 anywhere through K=38, or capped checkpoint walkback,
provides a certificate. This is a sampled frontier, not a stored archive replay.
Production `src/` is unchanged.

`full` continues pm∞ + mix13 at every depth; `screened` preserves its exact
decisions using the ternary divisibility screen. `baseline` retains Angeltveit's
descent, pm1, and root-armed OEE rules. `pm_only` drops mix, and `pm_screened`
screens that policy's pm∞ checks. `cutoffK` applies full rules through K, then
baseline rules. The corresponding `fast_baseline` and `fast_cutoffK` variants
specialize the baseline suffix, using only the iterate bits needed for future
parity and emitting the GPU producer's endpoint payload. They do not reconstruct
the full iterate or its ternary residue.

CPU timings include per-depth production counters and an endpoint digest sink.
Sampling, audits, archive I/O, packing/transfers, and GPU execution are excluded.
Policies intentionally admit different numbers of endpoints; comparisons use
elapsed time for identical roots rather than equal endpoint counts.

The separate work probe samples 30,000 baseline endpoints in each of the 16
GPU classes, F=40 through 55. It evaluates every policy on paired draws and
uses the saved RTX 4090 fit to estimate GPU work, including endpoint overhead.
Shared lookahead keeps pm∞ + mix13 credits and the saved 24 GB windows for every
CPU policy. Its `all` mode estimates the full-policy denominator from the same
sample; this paired ratio is the primary comparison. Errors are sampling
standard errors and exclude uncertainty in the GPU timing fit. These are not
GPU timings or complete verification runs; F≥56 work is outside this probe.

## Saved runs

| Directory or file | Contents |
|---|---|
| [policies/](policies/) | Initial ten-policy CPU comparison: 128 roots, five repetitions, seed 17395 |
| [fast-cutoffs/](fast-cutoffs/) | Specialized cutoff comparison: same 128 roots, five repetitions, seven selected variants |
| [confirmation/](confirmation/) | Independent 256-root corpus, seven repetitions, seed 17396, seven selected variants |
| [work/](work/) | Paired `all` work probe: 30,000 draws per class, seed 17397 |
| [summary.json](summary.json) | Machine-readable CPU comparisons, equivalence checks, and paired work estimates |
| [tests.txt](tests.txt) | Initial tests: three work-estimator, eight expansion, and five tail-policy tests |
| [fast-cutoff-tests.txt](fast-cutoff-tests.txt) | Updated seven-test tail suite, including two added specialized-cutoff checks |

The two logs cover **18 distinct passing tests**: three work-estimator tests,
eight expansion tests, and the updated seven-test tail suite. Checks include
complete small-tree comparisons, wide sampled roots, endpoint payloads and
per-depth counters, policy subsets, independent first-failure classification,
and paired-ratio variance arithmetic. Timed trials also validate digests and
counters, and each CPU run performs a separate exact payload audit.

Each run records raw JSONL, stderr, and `metadata.json` with arguments, timestamps,
exit status, binary/source hashes, compiler, CPU, governor, and affinity.
`source/` preserves inputs relative to the repository root; the work probe reads
its saved model snapshot. The initial `policies` run used `--variants all` when
there were ten variants. There are now fourteen, so its reproduction command
below lists the original ten explicitly. Use the saved source snapshot for
the original binary revision; current-source reruns can have different code
layout and timings. Compare timing ratios within a run.

## Reproduction

Run commands sequentially from the repository root. The runners pin CPUs 0–7,
the eight physical cores on this Ryzen 7700X host; adapt that mapping on other
machines. Choose fresh output directories because runners refuse to overwrite
measurements. `--offline` requires the dependencies to be cached.

```sh
cargo test --offline --manifest-path scripts/expansion_bench/Cargo.toml \
    --target-dir /tmp/collatz-expansion-target
cargo build --release --offline --manifest-path scripts/expansion_bench/Cargo.toml \
    --target-dir /tmp/collatz-expansion-target

python3 scripts/expansion_bench/run_tail.py \
    --out /tmp/checkpoint-tail-repro/policies \
    --roots 128 --repetitions 5 --threads 8 --seed 17395 --target 62 \
    --variants full,screened,baseline,fast_baseline,pm_only,pm_screened,cutoff46,cutoff50,cutoff54,cutoff58

python3 scripts/expansion_bench/run_tail.py \
    --out /tmp/checkpoint-tail-repro/fast-cutoffs \
    --roots 128 --repetitions 5 --threads 8 --seed 17395 --target 62 \
    --variants screened,baseline,fast_baseline,fast_cutoff46,fast_cutoff50,fast_cutoff54,fast_cutoff58

python3 scripts/expansion_bench/run_tail.py \
    --out /tmp/checkpoint-tail-repro/confirmation \
    --roots 256 --repetitions 7 --threads 8 --seed 17396 --target 62 \
    --variants full,screened,baseline,fast_baseline,fast_cutoff50,fast_cutoff54,fast_cutoff58

python3 scripts/expansion_bench/run_work.py \
    --out /tmp/checkpoint-tail-repro/work \
    --draws 30000 --seed 17397 --threads 8 --mode all \
    --model results/walkback_reuse/model.json

python3 scripts/expansion_bench/summarize_tail.py /tmp/checkpoint-tail-repro \
    --out /tmp/checkpoint-tail-repro/summary.json
```

The runners' default binaries are `tail_bench` and `checkpoint_work` under
`/tmp/collatz-expansion-target/release/`; `--binary` overrides either path.
