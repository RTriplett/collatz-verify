# Exploratory GPU-pipeline pilots

The pilots identify `novote_oee` at A10 with 256-thread blocks for confirmation: remove the per-iteration warp vote and apply the existing OEE rejection mask before candidate extraction. The overflow check and flagged-trajectory handling remain enabled. The separate-run comparison with the original parent kernel suggests a 10.323% reduction in pipeline time, which requires confirmation on fresh roots.

All five pilots use the same **512 baseline K38 draws**, arranged as two independent 256-draw cohorts. Of these, 32 fail improved ancestry screening, 3 fail bounded walkback, and 477 reach continuation. Every variant has two timing repetitions per cohort. The entire process is pinned to CPU cores 0–1 with two workers; the GPU is an RTX 4090. The arithmetic means below describe full pipeline wall time after the checkpoint, with arrays and suffix tables resident.

Ordered accepted-root corpus SHA-256: `0b57c40791276fccb3f5d12371a793dec8e6b40fc7eb493a274c4ceaf1adc744`. Both cohort hashes, seeds, complete saved root lists, and rejection accounting agree across all five runs. The A10 kernel and OEE runs also have identical full work signatures. The sample has 2,004,515,942 A10 endpoints, 195,334,608,722 cases, and 1,560,022,405,138 iterations. All pilot reports have zero flags and zero CPU-fallback endpoints.

## Kernel settings

Each ratio in this table uses the `parents` control in the same run. All settings process exactly the same work. `readonly` changes the power-table load path; the `block128` and `block512` labels include both the read-only load path and OEE masking.

| Variant | Mean cohort wall s | Ratio to parents | Time change |
|---|---:|---:|---:|
| `parents` | 4.939337 | 1.000000 | +0.000% |
| `novote` | 4.704996 | 0.952556 | -4.744% |
| `readonly` | 5.260826 | 1.065087 | +6.509% |
| `both` | 5.014245 | 1.015166 | +1.517% |
| `both_oee` | 4.817004 | 0.975233 | -2.477% |
| `block128` | 4.813293 | 0.974481 | -2.552% |
| `block512` | 5.036210 | 1.019613 | +1.961% |

Removing the warp vote saves 4.744% here. The read-only power-table path loses 6.509%; combining it with the other changes leaves performance worse than retaining the original power-table path.

## OEE mask and block size

| Variant | Mean cohort wall s | Ratio to same-run novote | Time change |
|---|---:|---:|---:|
| `novote` | 4.707671 | 1.000000 | +0.000% |
| `novote_oee` | 4.429449 | 0.940900 | -5.910% |
| `novote_oee128` | 4.427597 | 0.940507 | -5.949% |
| `novote_oee512` | 4.458877 | 0.947152 | -5.285% |

The OEE mask saves another 5.910% relative to the same-run `novote` control. Changing 256 threads to 128 changes time by -0.042%; 512 threads changes it by +0.664%. These small block-size differences do not justify selecting 128 threads as a proven improvement. The fresh confirmation retains 256 threads.

The repeated `novote` control changes from 4.704996 to 4.707671 seconds between the two runs (+0.057%). That is a useful consistency check, but the combined `parents`→`novote_oee` comparison still crosses runs and is exploratory.

## More expansion on the CPU

These runs retain the same roots, improved-check cutoff K50, terminal entry K52, and optimized kernel. Deeper endpoints intentionally change work counts. Comparisons with A10 cross separate runs and therefore also include possible timing drift.

| Shape | Mean cohort wall s | CPU core-s | Wall / A10 | Endpoints / A10 | Tail iterations / A10 |
|---|---:|---:|---:|---:|---:|
| A10 | 4.429449 | 0.510966 | 1.000000 | 1.000000 | 1.000000 |
| A9 | 4.452942 | 0.581152 | 1.005304 | 1.904663 | 0.957341 |
| A8, relative windows | 4.667195 | 0.626188 | 1.053674 | 3.659142 | 0.918624 |
| A8, remapped windows | 4.724958 | 0.632056 | 1.066715 | 3.659142 | 0.929109 |

A9 changes time by +0.530% and shows no persuasive gain. A8 with relative windows takes 5.367% longer despite 8.138% fewer tail iterations. Its endpoint count grows 3.659×. The tested remapping is 1.238% slower than the relative allocation; it does not recover a benefit.

The relative allocation uses B = `32,35,35,35,35,34,32,32,27,27,27,27,27,27,27,27`. The A8 remapping uses `35,35,35,35,34,32,32,32,27,27,27,27,27,27,27,27`. Both have 19.625 GiB of shared array bits before small padding allocations. They are two tested allocations, not an exhaustive optimization of per-A windows.

## Limits and records

These are implementation-selection pilots, not the final full-range estimate. Two independent cohorts give only one degree of freedom for a population interval. Repeated timing on the same roots does not increase the independent sample count. Within-run ratios benefit from paired workloads and rotating order; they still do not capture every systematic effect. Cross-run ratios remain descriptive, and timing plus selection of the fastest variant can exaggerate a pilot benefit. The main confirmation uses a new seed and a larger predeclared corpus.

No pilot includes the concatenated-corpus calibration. Array construction takes about 75 seconds per run and is outside the trial timings, as are frontier construction, checkpoint I/O, and distributed coordination. The full estimator and exclusions are described in [METHOD.md](METHOD.md). The pilots contain sampled N72 work, not a completed verification below 2^72.

The [machine-readable comparison](pilot_comparison.json) records unrounded ratios, actual kernel tuning, exact one-pass sample counts, run hashes, and comparison scope. The underlying records are [kernel](kernel_pilot/pipeline.jsonl), [OEE](oee_pilot/pipeline.jsonl), [A9](a9_pilot/pipeline.jsonl), [A8 relative](a8_relative_pilot/pipeline.jsonl), and [A8 remapped](a8_remapped_pilot/pipeline.jsonl). Each was checked by the strict full-range summarizer for completed trials, matching metadata/tuning, exact draw accounting, and repeated work signatures.
