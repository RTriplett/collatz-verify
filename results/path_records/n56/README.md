# Full verification below 2^56 with path-candidate export

The RTX 4090 run completed successfully on 2026-10-26 in **922.73 seconds
(15 minutes 23 seconds)** including setup and export. All **2,236 unique
overflow candidates** passed the verifier's 256-bit recheck and an independent
Python replay of their complete trajectories.

The five successive maximum setters among the captured candidates exactly
match records #73–77 in [David Barina's table](https://pcbarina.fit.vut.cz/path-records.htm).
The additional published record in `[2^55, 2^56)` is:

| Metric | Value |
|---|---:|
| Start | `49163256101584231` |
| Shortcut maximum | `301753104069007668258074264675786` |
| Standard maximum | `603506208138015336516148529351572` |
| Shortcut / standard step attaining the peak | 301 / 523 |
| Total shortcut / standard steps to 1 | 772 / 1224 |
| First descent, in shortcut steps | 623 |

The other recovered published record starts are `1254251874774375`,
`5323048232813247`, `8562235014026655`, and `10709980568908647`, all below
2^55. The run used a **97-bit arithmetic cutoff** to capture these five;
the default 120-bit cutoff would miss them. This establishes agreement on
these records. The artifact remains an export of captured overflow candidates,
not a complete census of all earlier path records.

## Configuration and counts

The run uses the production `cv verify` DFS pipeline with improved(13) tree
rules, shared bitvectors with pm∞ and mixed-word credits, A=12, and sixteen
CPU threads. The per-class windows start with
`32,35,35,35,35,34,32,32`; the remaining eight use B=28. The experimental
checkpoint/parent-dispatch pipeline is not used by this command.

| Measurement | Result |
|---|---:|
| Work units | 172,927 |
| Tree nodes | 101,513,615,194 |
| Endpoints | 44,845,477,535 |
| Cases | 15,105,881,916,633 |
| Tail iterations | 118,945,789,566,957 |
| Checksum | `C3E654D3` |
| Flagged / successfully rechecked starts | 2,236 / 2,236 |
| Pipeline time | 837.33 s |
| Setup time, including array construction | 85.05 s |
| Total process CPU time | 3,871.93 s |

Tree node counts, total endpoints, tree generation/survival/cull vectors, and
per-class endpoint counts exactly match the
[earlier N56 verification](../../validation/verify_n56_a12_cls.json).
Case counts are not compared for equality because the windows differ;
iteration counts and checksums also depend on the arithmetic cutoff.
The GPU report's summed CUDA stream event times overlap and are not elapsed
GPU wall time.

The initial allocation attempt stopped before verification: it needed 22.06 GiB
with class 5 at B=35, while 21.42 GiB was free. Reducing that class to B=34
allowed the complete run. The `attempt1_*` files retain that preflight result.
Telemetry in `gpu.csv` starts partway through verification and continues into
the independent checking phase; it does not cover the whole run.

## Artifacts and reproduction

* [Verification report](verification.json), [stdout](stdout.txt), and
  [process timing](time.txt).
* [All exported candidates](candidates.json), with exact decimal peaks and
  step counts under both conventions.
* [Independent comparison](comparison.json), including the five matched
  published records and six matching historical count checks.
* [Command, build settings and source hashes](metadata.json).

From the repository root:

```sh
CUDA_ARCH=sm_89 cargo build --release --offline --target-dir /tmp/collatz-n56-target
/tmp/collatz-n56-target/release/cv --threads 16 verify --n 56 --gpu \
    --a 12 --b 28 --class-b 32,35,35,35,35,34,32,32 --bv-shared \
    --bv-pm-inf --bv-mix-j 13 --rules improved --mix-j 13 --cutoff-bits 97 \
    --json results/path_records/n56/verification.json \
    --path-records results/path_records/n56/candidates.json
```

Recheck the saved artifacts without running the verifier or using a GPU:

```sh
python3 results/path_records/n56/check_records.py
```

The checker iterates the ordinary map with Python arbitrary-precision integers
and extracts shortcut states after even-input transitions. It checks every
exported metric, occurrence count, sorting, published maximum, and compatible
historical counter, failing on any discrepancy.
