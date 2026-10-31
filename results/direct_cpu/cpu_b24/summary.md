# Direct CPU endpoint ownership

CPU-only comparisons on fixed conditional M62/F source cohorts. Every policy owns the same original source families; native endpoint/case/iteration counts and hashes can differ across A/B. Repetitions measure timing variation, not independent root samples. No full-range cost or GPU saving is inferred.

One pinned timing thread includes reusable suffix expansion, mod9/OEE and optional shared B screening, scalar verification and all flagged wide rechecks. Sampling, table/array setup, independent validation and JSON output are excluded. Different F classes are conditional strata, not equal population shares.

## results/direct_cpu/cpu_b24/pipeline.jsonl

8,192 fixed sources per F class, 5 timing repetitions. All run, source-identity, work, rotation and accounting audits passed. Independent original-start validation covers 4 sources per class and policy.

| Source F | CPU A | CPU M | B | Mean CPU µs/source | Mean wall µs/source | CPU / same-B A10 | Cases/source |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 10 | 62 | 20 | 2.068 | 2.068 | 1.0000 | 37.40 |
| 40 | 8 | 64 | 20 | 1.809 | 1.809 | 0.8749 | 31.33 |
| 40 | 6 | 66 | 20 | 1.645 | 1.645 | 0.7953 | 27.24 |
| 40 | 4 | 68 | 20 | 1.679 | 1.679 | 0.8119 | 23.34 |
| 40 | 10 | 62 | 24 | 1.736 | 1.736 | 1.0000 | 27.63 |
| 40 | 8 | 64 | 24 | 1.632 | 1.632 | 0.9400 | 23.60 |
| 40 | 6 | 66 | 24 | 1.603 | 1.603 | 0.9237 | 20.14 |
| 40 | 4 | 68 | 24 | 1.802 | 1.802 | 1.0383 | 16.40 |
| 41 | 10 | 62 | 20 | 6.242 | 6.243 | 1.0000 | 111.68 |
| 41 | 8 | 64 | 20 | 5.747 | 5.748 | 0.9207 | 100.33 |
| 41 | 6 | 66 | 20 | 5.136 | 5.136 | 0.8228 | 87.07 |
| 41 | 4 | 68 | 20 | 5.287 | 5.287 | 0.8469 | 77.49 |
| 41 | 10 | 62 | 24 | 5.204 | 5.204 | 1.0000 | 84.72 |
| 41 | 8 | 64 | 24 | 4.990 | 4.992 | 0.9589 | 77.10 |
| 41 | 6 | 66 | 24 | 4.787 | 4.788 | 0.9199 | 65.65 |
| 41 | 4 | 68 | 24 | 5.158 | 5.158 | 0.9912 | 55.52 |
| 42 | 10 | 62 | 20 | 12.746 | 12.747 | 1.0000 | 220.53 |
| 42 | 8 | 64 | 20 | 11.548 | 11.549 | 0.9060 | 198.28 |
| 42 | 6 | 66 | 20 | 10.481 | 10.483 | 0.8223 | 174.62 |
| 42 | 4 | 68 | 20 | 10.436 | 10.437 | 0.8188 | 158.18 |
| 42 | 10 | 62 | 24 | 10.938 | 10.939 | 1.0000 | 173.91 |
| 42 | 8 | 64 | 24 | 10.103 | 10.104 | 0.9237 | 157.56 |
| 42 | 6 | 66 | 24 | 9.608 | 9.608 | 0.8784 | 135.97 |
| 42 | 4 | 68 | 24 | 10.319 | 10.319 | 0.9433 | 117.55 |

Best observed A/B per source class (by mean CPU time; descriptive selection):

- F=40: A=6, B=24, 1.603 µs/source.
- F=41: A=6, B=24, 4.787 µs/source.
- F=42: A=6, B=24, 9.608 µs/source.

Suffix-table and resident-array setup is recorded separately in JSON. The source classes are not population-weighted; their timings must not be averaged into a full-range estimate.
