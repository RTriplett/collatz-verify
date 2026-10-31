# Direct CPU ownership: completion-width sweep

Moving the CPU endpoint depth from M62/A10 to M66/A6 reduced steady CPU time about **18–20% with the same B20 screen**. Among the tested policies, A6/B20 had the lowest mean time for all three source classes. Further expansion to A2/A0 increased enumeration costs enough to lose the benefit.

| Source F | A10/B20 mean µs/source | A6/B20 mean | A6/B20 median | Mean reduction | A6/B16 mean |
|---:|---:|---:|---:|---:|---:|
| 40 | 2.134 | 1.709 | 1.737 | 19.9% | 1.999 |
| 41 | 6.338 | 5.200 | 5.197 | 17.9% | 6.111 |
| 42 | 12.932 | 10.550 | 10.512 | 18.4% | 12.123 |

These are **CPU-thread times**, not GPU throughput measurements. The CPU uses a smaller resident B window than the tuned GPU pipeline. B20 beat B16 at A6 by about 13–15%; omitting B was substantially slower. B16 favored A4, while B20 favored A6. See the [B24 follow-up](../cpu_b24/README.md) for the memory/setup tradeoff and sensitivity of the low-F result.

## Method and scope

One AMD Ryzen 7 7700X core, affinity 6, processed 2,048 fixed M62 sources in each F=40/41/42 class. Seed: 20261103. Each cohort consists of uniform conditional baseline draws after rejecting improved13 ancestry culls through K50; duplicate draws are retained. There is no K38 walkback filtering. The source classes are conditional strata, not equal shares of full-range work.

All combinations of A=10/8/6/4/2/0 and B=none/16/20 were warmed once and timed five times in rotating, reversed order. Every policy owns the same 1,024 original starts per source family. The timed operation includes reusable suffix expansion to M=72−A, mod-9/OEE screening, optional shared B screening, scalar tails, and flagged wide rechecks. Sampling, setup, validation, and JSON output are excluded. Thread CPU time and elapsed wall time are recorded separately; the table above uses means, with the selected policy's median shown explicitly.

B20 arrays occupy about 2 MiB per plan and took roughly 0.225–0.230 seconds to build on the pinned setup thread. B16 arrays occupy about 128 KiB and took 0.011–0.014 seconds. Suffix table sizes and all setup measurements are in [summary.json](summary.json); plans were resident before timing.

## Correctness and provenance

The reusable suffix compiler was checked against ordinary DFS and independent exact-power suffix evaluation. Four sources per F class and policy were audited over all 1,024 original starts, including an independent wide trajectory for starts removed by screening. The source ledger checks complete coverage without duplication among retained descendants. Smaller A policies can prove away more starts, so endpoint counts, cases, tail iterations, and checksums are compared exactly **within each policy**, not across A/B policies.

For A<6, storage retains a 64-completion mask built for the correct M; enumeration truncates it to exactly the first 2^A completions. Exact u128 endpoint arithmetic is retained beyond M64. Shared B windows omit leading-even merge credit, preserving soundness for every trailing-even state. Unit tests also cover forced cutoff flags; all recorded flags receive wide rechecks.

The [strict audit](summary.json) passes for all 54 configurations and 270 trials. It independently recomputes ordered source fingerprints and source F values, checks trial ordering, original-start audit counts, per-policy signatures, phase accounting, summary arithmetic, and saved source hashes. [pipeline.jsonl](pipeline.jsonl) contains the full ordered source lists and measurements. [metadata.json](metadata.json) records hardware, affinity, command, executable hash and source hashes; [source/](source/) is the matching snapshot. Hashes establish saved-file identity, not independent build provenance.

These repeated timings are not independent root samples and do not establish GPU savings, optimal production scheduling, or a full-range cost estimate.

Reproduce the report without rerunning computation:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/expansion_bench/summarize_direct_cpu.py \
  --out results/direct_cpu/cpu_probe/summary.json \
  results/direct_cpu/cpu_probe/pipeline.jsonl
```
