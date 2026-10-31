# CPU routing before GPU dispatch

Measured continuous-corpus timings only; no GPU-year or rental-price extrapolation.

GPU reports exclude diverted CPU work. Original-depth source endpoints are conserved separately from CPU-native descendants, cases and checksums.

Different CPU A/B policies can remove different candidates and change checksums. Exact native signatures are compared only for the same policy and the same recorded admitted descriptor identity.

Admission is nonblocking and depends on worker availability, so admitted groups can differ between repetitions. Descriptor count/XOR/sum identifiers are consistency checks, not cryptographic proofs of identical sets.

The startup guard counts endpoints initially forwarded to the ordinary sink; it is not a live measurement of GPU queue occupancy.

Process CPU, summed worker elapsed intervals, CUDA event sums and elapsed pipeline time overlap; they cannot be added into a wall-time breakdown.

GPU case/iteration reduction is measured against the same corpus using the unchanged GPU policy. CPU-native iterations are not a substitute for that reduction.

Warmup, source sampling and resident-table construction are excluded. Continuous repetitions reuse the same roots; observed ranges are not sampling confidence intervals.

Target trials must disable CPU source capture and GPU diagnostic capture. Validation replay may capture sources outside the performance comparison.

## results/direct_cpu/pipeline_pilot

512 baseline draws, 488 accepted; all source-partition and work-accounting audits passed.

| Variant | Continuous wall s, mean [min,max] | vs GPU | vs route control | CPU core-s | CPU worker core-s | Drain ms | Original endpoints diverted | GPU iterations removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpu | 9.4145 [9.3892, 9.4437] | 1.0000 | 1.0010 | 1.489 | 0.000 | 0.001 | 0 | 0.00000% |
| route_control | 9.4053 [9.3698, 9.4622] | 0.9990 | 1.0000 | 1.547 | 0.000 | 0.001 | 0 | 0.00000% |
| direct_a10_b24_s1_w4 | 9.3852 [9.3504, 9.4379] | 0.9969 | 0.9979 | 20.811 | 19.353 | 0.091 | 10,537,378 | 0.06832% |
| direct_a6_b24_s1_w4 | 9.3680 [9.3615, 9.3733] | 0.9951 | 0.9960 | 18.565 | 17.110 | 0.079 | 11,360,930 | 0.07366% |
| direct_a6_b20_s1_w4 | 9.3998 [9.3613, 9.4526] | 0.9984 | 0.9994 | 20.152 | 18.707 | 0.076 | 10,755,288 | 0.06973% |
| direct_a6_b24_s64_w4 | 9.4132 [9.3791, 9.4468] | 0.9999 | 1.0008 | 12.750 | 11.237 | 0.097 | 6,676,463 | 0.04327% |
