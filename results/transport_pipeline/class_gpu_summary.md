# F40 selective-expansion GPU microprobe

Serialized GPU kernel event seconds, including endpoint reconstruction and tail work. Host expansion, packing, transfer and array construction are excluded. This is a conditional M62/F40 sample with improved checks through K50 and no K38 walkback, not the earlier saved K38 corpus or an end-to-end mixed-A pipeline.

A10, A9, A8 and the closing A10 control ran in separate blocks. Ratios compare medians, not paired repetitions. Each trial total sums the corresponding separately timed class runs. Observed timing ranges are not confidence intervals.

Sources: 2,097,152 unique M62/F40 endpoints, seed 20261025; ordered FNV-1a64 `E209A5F847EBA0B0`. This identifier is not a cryptographic hash. Shared class windows: 32 and 35 bits.

| Block | M | Kernel median (ms) | Observed range (ms) | Ratio to opening A10 | Ratio to closing A10 |
|:---|---:|---:|:---|---:|---:|
| a10 | 62 | 2.625 | 2.623–2.640 | 1.0000 | 1.0031 |
| a9 | 63 | 2.506 | 2.503–2.507 | 0.9549 | 0.9579 |
| a8 | 64 | 2.835 | 2.831–2.838 | 1.0800 | 1.0834 |
| a10_end | 62 | 2.616 | 2.608–2.625 | 0.9969 | 1.0000 |

Pooled A10 controls: median 2.624 ms, observed range 2.608–2.640 ms. Closing/opening median ratio: 0.9969.

| Block | Endpoints | Endpoint factor | Baseline candidate fraction retained | Tail cases | Tail iterations | Hash |
|:---|---:|---:|---:|---:|---:|:---|
| a10 | 2,097,152 | 1.0000 | 1.000000 | 31,763,392 | 216,364,002 | `A8643EBC` |
| a9 | 3,344,318 | 1.5947 | 0.797348 | 23,649,771 | 170,343,774 | `619A6040` |
| a8 | 5,441,470 | 2.5947 | 0.648674 | 23,824,528 | 170,156,581 | `7ABF9BBE` |
| a10_end | 2,097,152 | 1.0000 | 1.000000 | 31,763,392 | 216,364,002 | `A8643EBC` |

Candidate fractions divide the endpoint factor by 2 or 4 after one or two extra bits. They describe baseline candidates before lookahead, so they differ from tail-case and iteration ratios.

All 6 class-level CPU/GPU audits passed on descendants of the first 16 sources. All warmups, repetitions and same-A control work signatures agree, with zero flagged trajectories. Class counts, XOR hashes and kernel seconds were independently summed and checked against each shape summary.

These measurements test GPU work for a conditional F40 corpus. They do not establish the throughput of a mixed-depth queue or predict the benefit for the saved K38 packages. The JSON retains each class trial, setup timing, source identifiers and validation record.
