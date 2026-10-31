# Direct CPU ownership: B20 versus B24

B24 reduced the mean cost of the F41/F42 CPU work by about **7–8% at A6**, compared with B20. F40 is less decisive: changing between means and medians changes the apparent best A, and its small advantage over A6/B20 does not establish a clear winner.

| Source F | Policy | Mean CPU µs/source | Median CPU µs/source | Mean / same-B A10 |
|---:|---|---:|---:|---:|
| 40 | A6/B20 | 1.645 | 1.656 | 0.7953 |
| 40 | A8/B24 | 1.632 | 1.641 | 0.9400 |
| 40 | A6/B24 | 1.603 | 1.651 | 0.9237 |
| 41 | A6/B20 | 5.136 | 5.140 | 0.8228 |
| 41 | A6/B24 | 4.787 | 4.815 | 0.9199 |
| 42 | A6/B20 | 10.481 | 10.479 | 0.8223 |
| 42 | A6/B24 | 9.608 | 9.656 | 0.8784 |

The lowest observed **mean** for each class is A6/B24. For F40, however, A8/B24 has the lower **median**, only 0.9% below A6/B20's median. A6/B24's F40 mean is pulled down by faster repetitions: its range is 1.515–1.664 µs/source. This is a timing sensitivity on one fixed cohort, not evidence of a stable population winner. For F42 with B20, A4 and A6 are also close (means 10.436 and 10.481 µs/source).

Reducing A from 10 to 6 at B24 saves 7.6%, 8.0%, and 12.2% by mean CPU time for F40, F41, and F42 respectively. The same A change at B20 saves about 18–20%. The stronger B screen therefore reduces some of the advantage of further suffix expansion. All policies use CPU windows smaller than the tuned GPU windows; these numbers are not comparisons against GPU execution time.

## Memory, setup, and method

B24 uses about **32 MiB per resident plan**, versus **2 MiB for B20**. On the single pinned setup thread, B24 array/table setup took 6.41–6.57 seconds per plan; B20 took 0.22–0.25 seconds. Those costs are recorded separately and excluded from steady processing time. In a long-lived CPU worker the immutable plan can be shared, but these experiments do not measure contention among simultaneous workers.

This follow-up used 8,192 fixed M62 sources per F=40/41/42 class, seed 20261104, on one AMD Ryzen 7 7700X core pinned to logical CPU 6. It measured A=10/8/6/4 with B=20/24, with one warmup and five rotated/reversed timing repetitions for all 24 configurations. Sampling remains conditional on baseline M62/F survival and improved13 ancestry checks through K50; duplicates are retained and K38 walkback is absent. The larger follow-up uses a different cohort from [the first sweep](../cpu_probe/README.md), so cross-run timing differences are not paired comparisons.

Timed work includes suffix expansion, mod-9/OEE and shared B screening, scalar tails, and flagged wide rechecks. Sampling, setup, independent validation, and output are excluded. CPU-thread time and wall time are reported separately. The table above distinguishes means from medians rather than treating them as interchangeable.

## Correctness, provenance, and limits

Each A/B policy owns the same original source families. Four sources per F class and policy were exhaustively audited over all 1,024 starts each: compiled suffixes match ordinary DFS and an independent exact-power suffix oracle, and every start—including screened starts—also receives an independent wide trajectory check. Native cases, iteration counts, and hashes need not agree across policies; exact equality is enforced within each policy across warmup and all repetitions.

M>64 suffix states use exact u128 arithmetic. For A4, the underlying M-correct A6 completion mask is truncated to the first 16 bits; it does not introduce additional starts. Shared windows remain sound for all trailing-even states by omitting leading-even merge credit. Forced-cutoff flag handling is covered by library tests, and ordinary wide rechecks are included for all flags produced by timed processing.

The [strict audit](summary.json) passes all 120 trials, source fingerprints, independent source F arithmetic, complete policy/repetition coverage, rotated ordering, work signatures, phase accounting, summary arithmetic, and saved source hashes. [pipeline.jsonl](pipeline.jsonl) contains the full ordered source lists and raw measurements; [metadata.json](metadata.json) records command, executable hash, machine, affinity, and source hashes; [source/](source/) preserves the matching source snapshot. Hashes establish saved-file identity, not independent build provenance. Full mean/median/min/max results are in [summary.json](summary.json); [summary.md](summary.md) lists mean results for every policy.

These are conditional single-core CPU measurements. They do not establish GPU savings, simultaneous CPU/GPU scheduling behavior, or full-range costs. Repetitions measure timing variation on fixed sources; source classes cannot be equally averaged into a population estimate.

Reproduce the report without rerunning computation:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/expansion_bench/summarize_direct_cpu.py \
  --out results/direct_cpu/cpu_b24/summary.json \
  results/direct_cpu/cpu_b24/pipeline.jsonl
```
