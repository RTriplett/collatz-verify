Single `cv measure` runs at N = 72 (sound bitvectors) behind the rule
comparisons in docs/FINDINGS.md:

| file | step-1 extra rules | shape |
|---|---|---|
| measure_n72_b24_mix13.json | pm_inf + mix(3^13) | A10 B24 |
| measure_n72_mix16_cpu.json | pm_inf + mix(3^16) (CPU only, no GPU timing) | A10 B24 |
| measure_n72_mix13_wb50.json | pm_inf + mix(3^13) + walkback at k=50, 65,536 states (CPU only) | A10 B24 |
| measure_n72_a10b24bvpminfbvmixj13.json | pm_inf + mix(3^13), window credits | A10 B24 |
| measure_n72_a10b28bvpminfbvmixj13.json | pm_inf + mix(3^13), window credits | A10 B28 |
| measure_n72_a12b28bvpminfbvmixj13.json | pm_inf + mix(3^13), window credits | A12 B28 |

These are single cold-start timings; the cost tables use the sweep and the
fitted per-operation model instead.
