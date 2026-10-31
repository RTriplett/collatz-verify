# F40 selective-expansion GPU microprobe

Serialized GPU kernel event seconds, including endpoint reconstruction and tail work. Host expansion, packing, transfer and array construction are excluded. This is a conditional M62/F40 sample with improved checks through K50 and no K38 walkback, not the earlier saved K38 corpus or an end-to-end mixed-A pipeline.

A10, A9, A8 and the closing A10 control ran in separate blocks. Ratios compare medians, not paired repetitions. Each trial total sums the corresponding separately timed class runs. Observed timing ranges are not confidence intervals.

Sources: 16,777,216 unique M62/F40 endpoints, seed 20261025; ordered FNV-1a64 `422B75171FE26ECE`. This identifier is not a cryptographic hash. Shared class windows: 32 and 35 bits.

| Block | M | Kernel median (ms) | Observed range (ms) | Ratio to opening A10 | Ratio to closing A10 |
|:---|---:|---:|:---|---:|---:|
| a10 | 62 | 20.890 | 20.858–20.900 | 1.0000 | 0.9998 |
| a9 | 63 | 19.781 | 19.738–19.866 | 0.9469 | 0.9467 |
| a8 | 64 | 22.382 | 22.311–22.427 | 1.0714 | 1.0712 |
| a10_end | 62 | 20.894 | 20.887–20.935 | 1.0002 | 1.0000 |

Pooled A10 controls: median 20.893 ms, observed range 20.858–20.935 ms. Closing/opening median ratio: 1.0002.

| Block | Endpoints | Endpoint factor | Baseline candidate fraction retained | Tail cases | Tail iterations | Hash |
|:---|---:|---:|---:|---:|---:|:---|
| a10 | 16,777,216 | 1.0000 | 1.000000 | 254,070,186 | 1,730,745,550 | `451AFB96` |
| a9 | 26,748,211 | 1.5943 | 0.797159 | 189,185,082 | 1,362,737,958 | `7C168BD8` |
| a8 | 43,525,427 | 2.5943 | 0.648579 | 190,568,465 | 1,361,165,563 | `D20504B0` |
| a10_end | 16,777,216 | 1.0000 | 1.000000 | 254,070,186 | 1,730,745,550 | `451AFB96` |

Candidate fractions divide the endpoint factor by 2 or 4 after one or two extra bits. They describe baseline candidates before lookahead, so they differ from tail-case and iteration ratios.

All 6 class-level CPU/GPU audits passed on descendants of the first 16 sources. All warmups, repetitions and same-A control work signatures agree, with zero flagged trajectories. Class counts, XOR hashes and kernel seconds were independently summed and checked against each shape summary.

These measurements test GPU work for a conditional F40 corpus. They do not establish the throughput of a mixed-depth queue or predict the benefit for the saved K38 packages. The JSON retains each class trial, setup timing, source identifiers and validation record.
