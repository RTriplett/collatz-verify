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

## results/direct_cpu/pipeline_queue1024

512 baseline draws, 485 accepted; all source-partition and work-accounting audits passed.

| Variant | Continuous wall s, mean [min,max] | vs GPU | vs route control | CPU core-s | CPU worker core-s | Drain ms | Original endpoints diverted | GPU iterations removed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpu | 8.0384 [8.0144, 8.0706] | 1.0000 | 1.0026 | 1.318 | 0.000 | 0.001 | 0 | 0.00000% |
| route_control | 8.0175 [8.0008, 8.0382] | 0.9974 | 1.0000 | 1.398 | 0.000 | 0.001 | 0 | 0.00000% |
| direct_a6_b24_s1_w4 | 8.0240 [7.9923, 8.0491] | 0.9982 | 1.0008 | 32.958 | 31.589 | 0.090 | 21,568,193 | 0.16428% |
