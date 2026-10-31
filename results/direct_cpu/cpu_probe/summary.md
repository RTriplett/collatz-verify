# Direct CPU endpoint ownership

CPU-only comparisons on fixed conditional M62/F source cohorts. Every policy owns the same original source families; native endpoint/case/iteration counts and hashes can differ across A/B. Repetitions measure timing variation, not independent root samples. No full-range cost or GPU saving is inferred.

One pinned timing thread includes reusable suffix expansion, mod9/OEE and optional shared B screening, scalar verification and all flagged wide rechecks. Sampling, table/array setup, independent validation and JSON output are excluded. Different F classes are conditional strata, not equal population shares.

## results/direct_cpu/cpu_probe/pipeline.jsonl

2,048 fixed sources per F class, 5 timing repetitions. All run, source-identity, work, rotation and accounting audits passed. Independent original-start validation covers 4 sources per class and policy.

| Source F | CPU A | CPU M | B | Mean CPU µs/source | Mean wall µs/source | CPU / same-B A10 | Cases/source |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 10 | 62 | none | 10.524 | 10.524 | 1.0000 | 499.46 |
| 40 | 8 | 64 | none | 7.557 | 7.558 | 0.7180 | 306.69 |
| 40 | 6 | 66 | none | 4.783 | 4.783 | 0.4545 | 167.56 |
| 40 | 4 | 68 | none | 4.077 | 4.077 | 0.3874 | 134.51 |
| 40 | 2 | 70 | none | 3.896 | 3.897 | 0.3702 | 102.65 |
| 40 | 0 | 72 | none | 5.314 | 5.313 | 0.5049 | 84.87 |
| 40 | 10 | 62 | 16 | 2.697 | 2.696 | 1.0000 | 53.85 |
| 40 | 8 | 64 | 16 | 2.264 | 2.264 | 0.8397 | 44.85 |
| 40 | 6 | 66 | 16 | 1.999 | 1.999 | 0.7413 | 37.03 |
| 40 | 4 | 68 | 16 | 1.931 | 1.931 | 0.7161 | 31.11 |
| 40 | 2 | 70 | 16 | 2.257 | 2.257 | 0.8369 | 26.38 |
| 40 | 0 | 72 | 16 | 3.433 | 3.432 | 1.2729 | 23.02 |
| 40 | 10 | 62 | 20 | 2.134 | 2.134 | 1.0000 | 37.51 |
| 40 | 8 | 64 | 20 | 1.877 | 1.877 | 0.8798 | 31.54 |
| 40 | 6 | 66 | 20 | 1.709 | 1.708 | 0.8010 | 27.40 |
| 40 | 4 | 68 | 20 | 1.739 | 1.739 | 0.8150 | 23.46 |
| 40 | 2 | 70 | 20 | 2.048 | 2.048 | 0.9597 | 19.50 |
| 40 | 0 | 72 | 20 | 3.242 | 3.241 | 1.5193 | 16.17 |
| 41 | 10 | 62 | none | 17.746 | 17.746 | 1.0000 | 561.04 |
| 41 | 8 | 64 | none | 16.055 | 16.056 | 0.9047 | 556.51 |
| 41 | 6 | 66 | none | 12.827 | 12.827 | 0.7228 | 427.08 |
| 41 | 4 | 68 | none | 11.599 | 11.601 | 0.6536 | 364.94 |
| 41 | 2 | 70 | none | 11.737 | 11.737 | 0.6614 | 298.42 |
| 41 | 0 | 72 | none | 16.260 | 16.263 | 0.9163 | 257.05 |
| 41 | 10 | 62 | 16 | 7.848 | 7.848 | 1.0000 | 153.68 |
| 41 | 8 | 64 | 16 | 7.017 | 7.018 | 0.8941 | 138.05 |
| 41 | 6 | 66 | 16 | 6.111 | 6.111 | 0.7787 | 114.79 |
| 41 | 4 | 68 | 16 | 6.021 | 6.021 | 0.7672 | 100.63 |
| 41 | 2 | 70 | 16 | 7.131 | 7.131 | 0.9087 | 88.15 |
| 41 | 0 | 72 | 16 | 11.103 | 11.103 | 1.4147 | 78.77 |
| 41 | 10 | 62 | 20 | 6.338 | 6.337 | 1.0000 | 111.35 |
| 41 | 8 | 64 | 20 | 5.764 | 5.764 | 0.9095 | 100.12 |
| 41 | 6 | 66 | 20 | 5.200 | 5.200 | 0.8205 | 86.97 |
| 41 | 4 | 68 | 20 | 5.316 | 5.316 | 0.8388 | 77.33 |
| 41 | 2 | 70 | 20 | 6.365 | 6.365 | 1.0043 | 66.38 |
| 41 | 0 | 72 | 20 | 10.026 | 10.026 | 1.5820 | 56.31 |
| 42 | 10 | 62 | none | 22.864 | 22.869 | 1.0000 | 567.44 |
| 42 | 8 | 64 | none | 21.170 | 21.174 | 0.9259 | 566.97 |
| 42 | 6 | 66 | none | 19.631 | 19.634 | 0.8586 | 541.61 |
| 42 | 4 | 68 | none | 18.799 | 18.799 | 0.8222 | 516.98 |
| 42 | 2 | 70 | none | 19.569 | 19.570 | 0.8559 | 462.46 |
| 42 | 0 | 72 | none | 27.759 | 27.763 | 1.2141 | 420.94 |
| 42 | 10 | 62 | 16 | 15.413 | 15.413 | 1.0000 | 287.47 |
| 42 | 8 | 64 | 16 | 13.764 | 13.764 | 0.8930 | 259.21 |
| 42 | 6 | 66 | 16 | 12.123 | 12.124 | 0.7865 | 221.47 |
| 42 | 4 | 68 | 16 | 11.762 | 11.764 | 0.7632 | 199.21 |
| 42 | 2 | 70 | 16 | 13.484 | 13.485 | 0.8749 | 177.37 |
| 42 | 0 | 72 | 16 | 19.960 | 19.961 | 1.2950 | 161.12 |
| 42 | 10 | 62 | 20 | 12.932 | 12.932 | 1.0000 | 220.47 |
| 42 | 8 | 64 | 20 | 11.686 | 11.687 | 0.9036 | 198.35 |
| 42 | 6 | 66 | 20 | 10.550 | 10.550 | 0.8158 | 174.73 |
| 42 | 4 | 68 | 20 | 10.613 | 10.614 | 0.8207 | 158.29 |
| 42 | 2 | 70 | 20 | 12.410 | 12.410 | 0.9596 | 137.93 |
| 42 | 0 | 72 | 20 | 19.085 | 19.085 | 1.4758 | 119.42 |

Best observed A/B per source class (by mean CPU time; descriptive selection):

- F=40: A=6, B=20, 1.709 µs/source.
- F=41: A=6, B=20, 5.200 µs/source.
- F=42: A=6, B=20, 10.550 µs/source.

Suffix-table and resident-array setup is recorded separately in JSON. The source classes are not population-weighted; their timings must not be averaged into a full-range estimate.
