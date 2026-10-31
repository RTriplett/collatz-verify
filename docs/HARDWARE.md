# Lookahead window, VRAM, and which GPU to rent

**Short answer:** a larger lookahead window is the biggest remaining lever, but it
does not justify renting large-memory datacenter GPUs. Per-SM integer throughput is
identical from Turing through Blackwell, so a B300 does about 0.87× a 4090's work
per hour at roughly 20× the hourly price. A 24–32 GB consumer card already holds windows
of 35–36 bits for the classes that matter, and the B300's extra memory saves only
another ~10% of the work. On rented hardware, use many consumer cards with the
per-class windows below. Large memory is worth using only when it is free, as with
donated or idle datacenter capacity; there, use the largest windows that fit.

All numbers are for verifying every n < 2^72, A = 10 unless stated, with step-1
pm∞ + mix(3^13), window credits, and shared bitvectors.

## 1. What a deeper window buys (measured)

Each extra window bit removes the cases that would have descended or merged one
step later. Cases fall by about 5% per bit. GPU time falls by about 3%, because the
removed cases are the short ones.

| window B (all classes) | 24 | 28 | 30 | 32 | 34 | 36 | 38 |
|---|---|---|---|---|---|---|---|
| model, RTX 4090-years (exact op counts × fitted costs) | 0.958 | 0.851 | 0.790 | 0.731 | 0.698 | 0.653 | 0.608 |
| real kernel on the 4090, all 16 classes timed | 0.940 | 0.868 | 0.826 | 0.731 | 0.710 | 0.662 | — |
| cases (×10^17) | 6.91 | 5.66 | 5.03 | 4.60 | 4.25 | 3.82 | 3.42 |

The real-kernel row for B ≥ 30 uses device-built arrays. Per class that means
2 × 8 GiB at B = 36, more than 100× the L2. Timing every class at B = 36 still
agrees with the model within 1.4%. Single runs scatter by about ±5%: the B = 24 value
is a cold-GPU run, B = 30 a slow one. The fitted per-endpoint overhead stays flat from B = 24 to 36
(0.61–0.75 ns), so arrays far larger than the cache cost nothing measurable:
each endpoint reads one contiguous 132-byte run.

The benefit is concentrated in a few classes. Classes 0–8 carry 99% of GPU time.
Classes 1–4 gain most (time at B = 36 is 0.72–0.78 of B = 28); classes 9 and up
gain nothing.

## 2. What it costs in memory

With `--bv-shared` (one array per class; sound, +0.2–0.3% work) each class `i`
needs `2^B_i + 2^A` bits on the device. `scripts/hardware_cost.py` chooses per-class
windows by an exact DP that minimises measured GPU time within `VRAM − 2 GiB`. Pass
the result with `--class-b`.

| card | VRAM | windows for classes 0–7 (others smaller) | RTX 4090-years at 2^72 |
|---|---|---|---|
| RTX 3090 / 4090 | 24 GB | 32,35,35,35,35,35,32,32 | 0.675 |
| RTX 5090 | 32 GB | 32,35,36,36,35,35,32,32 | 0.667 |
| L40S | 48 GB | 32,35,36,37,36,35,33,32 | 0.659 |
| A100 / H100 | 80 GB | 34,35,38,37,36,35,35,33 | 0.646 |
| RTX PRO 6000 | 96 GB | 32,36,38,38,36,36,35,33 | 0.636 |
| H200 | 141 GB | 32,37,38,38,38,36,35,35 | 0.625 |
| B200 | 192 GB | 35,38,38,38,38,38,36,35 | 0.616 |
| B300 | 288 GB | 38 for all of classes 0–7 | 0.610 |
| any card, uniform B = 24 (Angeltveit's window), for comparison | | | 0.958 |

Every card's line, with its build time and CPU need, is in
`results/hardware/cost_2p72.md`.

Two limits apply:

* **Window size.** The window iterate must fit a u64, so B is at most 39.
* **Build time.** Arrays are built on the device in about 0.5 s per 2^30 patterns on
  a 4090. That is 1.5 minutes at the 24 GB windows and about 20 minutes for B = 38 on
  a B300. Run many packages per process (`--package 0..40`) so the build is paid once.

## 3. Integer throughput is the same per SM on every architecture

The kernel is bound by 32-bit integer multiply-add, shifts and bit-finds.
NVIDIA's throughput table gives the same per-SM rates for compute capability 7.5
through 12.x: IMAD 64 per clock (32 for the wide forms), shift 64, clz/popc 16. That
table is in the CUDA C++ Best Practices Guide 13.4, Table 5. Datacenter Blackwell (B200,
cc 10.0; B300, cc 10.3) changed tensor and FP64 throughput, not this. So speed ≈
SM count × sustained clock:

| card | SMs | clock (GHz) | speed vs 4090 (estimate) |
|---|---|---|---|
| RTX 4090 (measured) | 128 | 2.52 rated, ~2.7 sustained | 1.00 |
| RTX 5090 | 170 | 2.41 | 1.27 |
| RTX PRO 6000 Blackwell | 188 | 2.62 | 1.5 |
| L40S | 142 | 2.52 | 1.03 |
| H100 SXM | 132 | 1.98 | 0.75 |
| B200 | 148 | 1.97 | 0.84 |
| B300 | 148 | 2.03 | 0.87 |
| RTX 3090 | 82 | 1.70 | 0.43 |

Only the 4090 is measured. Treat the others as ±10–15%, and check any card in
1–2 minutes with `cv measure --n 72 --gpu`. Angeltveit's own RTX 3060 run fits the
scaling to within 9%. Sources are in `results/hardware_research.json`, where two
independent verifiers confirmed or corrected every number.

## 4. Cost of a 2^72 verification

Prices are a vast.ai snapshot from 16 Oct 2026: public pricing feed plus live
offers. Only 9 B300s were listed, on 2 hosts.

| card | GPU-years | on-demand ($/h) | cost, on-demand | cost, lowest interruptible bid |
|---|---|---|---|---|
| RTX 3090 | 1.57 | 0.20 | $2,764 | $1,464 |
| RTX 4090 | 0.67 | 0.51 | $3,010 | $1,735 |
| RTX 5090 | 0.53 | 0.75 | $3,472 | $1,720 |
| L40S | 0.64 | 0.64 | $3,593 | $1,869 |
| RTX PRO 6000 (server) | 0.44 | 1.40 | $5,385 | $2,216 |
| A100 80GB PCIe | 1.47 | 0.93 | $12,078 | $2,868 |
| H100 SXM | 0.86 | 2.40 | $18,072 | $7,529 |
| H200 | 0.83 | 3.95 | $28,780 | $19,182 |
| B200 | 0.74 | 8.61 | $55,491 | $40,297 |
| B300 | 0.71 | 9.93 | $61,407 | $30,145 |

Per unit of integer throughput the B300 costs 19–23× as much as a 4090:
$11.48 vs $0.51 per 4090-equivalent hour on demand, or $5.64 vs $0.29
interruptible. Its extra memory cuts the work by about 10% (0.610 vs 0.675
4090-years), so a 2^72 run still costs 17–20× more. It would need to rent for
about $0.48/h to break even with a 4090 fleet.

## 5. Keep the GPU fed: CPU cores per GPU

The figures below describe the original DFS used by default. A newer
[integrated benchmark](../results/overall_pipeline/README.md) of experimental
checkpoint/table expansion at N72/A10 reduces process CPU consumption by 87%
on the measured packages, while median pipeline time falls from 29.0 to 26.5
seconds. A follow-up retains 26.5-second throughput with the entire process
pinned to two physical 7700X cores. Its arrays and checkpoint are prepared before
timing. The earlier
CPU-core estimates below should not be applied to that experimental path;
the global GPU-year estimates have not been replaced by this package benchmark.

A subsequent [A comparison](../results/a_pipeline/README.md) uses this spare
CPU capacity to expand to M63/A9 and M64/A8. With fresh A10 controls and the
same window-memory allocation, median pipeline times are 25.367, 25.448 and
27.539 seconds for A10, A9 and A8. Average busy CPU cores rise from 1.10 to 2.01
and 3.55. Smaller A reduces counted tail iterations but multiplies endpoint
traffic and GPU prefix reconstruction; A9 shows no clear benefit and A8 is
8.6% slower. Retain A10 for this experimental configuration. Per-A window
retuning is not measured, and the original-DFS guidance below remains separate.

The [transport follow-up](../results/transport_pipeline/README.md) realizes
larger host savings by sending K52 parent/class groups and expanding their
terminal words on the GPU. With eight cores available, process CPU falls from
28.943 to 6.543 core-seconds and transfers from 69.211 to 2.955 GB per measured
batch; wall time falls only from 24.819 to 24.414 seconds. With the **entire
process pinned to one physical core**, parent dispatch takes a median 24.315
seconds and 2.130 CPU-seconds, retaining the eight-core throughput. The original
16-byte experimental path takes 26.931 seconds on that same single core.
These are three repetitions on an active desktop, with a wider 21.952–24.465
second range for the single-core parent run. This supports allocating fewer
host cores to this path; it does not establish multi-GPU scaling or a revised
full-range GPU-year estimate. Packed eight-byte endpoints and CPU-supplied
exact iterates show no clear elapsed-time benefit in the same comparison.

The subsequent [full-range benchmark](../results/fullrange_pipeline/README.md)
uses 16 independent cohorts of 512 baseline K38 draws, retains rejected draws
in the denominator, and measures two repetitions plus a concatenated-corpus
check. On this RTX 4090, the A10 parent pipeline estimates **0.6535 GPU-years**
with its original kernel and **0.5859 GPU-years** with a per-lane overflow
guard and an OEE mask applied before candidate extraction. The latter has a
**0.5649–0.6069** 95% sampling interval; the paired saving is **10.34%**.
Both aggregate checks agree within 0.08%. The arrays use 19.625 GiB, and two
physical CPU cores are available; measured CPU demand extrapolates to 0.0681
core-years after the checkpoint. A separate rare-class probe gives a 1.37
CPU-hour allowance. Frontier construction, archive I/O, resident-array setup,
and distributed coordination are excluded. This new RTX 4090 result does not
remeasure or update the RTX PRO 6000 and other-device projections above.

A deeper window shrinks GPU work but leaves the step-1 tree unchanged, so a GPU
needs more CPU behind it. A complete N = 56 run on this machine (RTX 4090 plus a
16-thread Ryzen 7700X, A = 10, classes 0–7 at B = 34, 16 GiB of shared arrays
resident) matched the predicted counts to within 0.03%. It still took 941 s, against
760 GPU-seconds of work. The 16 threads ran 13.9 busy on average: the tree needs
about 800 CPU-seconds, so the run was CPU-bound. The fix is to give the GPU more
bits:

| A | step-1 tree nodes at 2^72 | 7700X threads per 4090 | GPU time vs A = 10 (real kernel) |
|---|---|---|---|
| 10 | 9.25×10^15 | ~16 | 1.00 |
| 12 | 2.57×10^15 | ~4 | 1.04 (B = 36) |
| 14 | 7.04×10^14 | ~1.1 | 1.12 (B = 32 and B = 36) |

The recommended 24 GB configuration was validated by a complete N = 56 run:
A = 12, windows 32,35,35,35,35,35,32,32, with 21.5 GiB of shared arrays built on
the GPU. Endpoints, cases and iterations came within 0.07% of the prediction. The
pipeline took 765 s against 756 GPU-seconds predicted (+1.1%), plus a one-time 83 s
array build. Only 4.5 threads were busy on average, so the run was GPU-bound; the
A = 10 run above was CPU-bound at 941 s.

So on a typical rented host with 8–16 vCPUs per GPU, use A = 12. Use A = 14
behind a slow or shared CPU. Host-to-device traffic is 16 bytes per endpoint:
about 3 GB/s per 4090 at A = 10 and under 1 GB/s at A = 12. Only x1 mining risers are
a problem.

## 6. Recommendation

* **Renting:** RTX 4090s or 5090s (3090s are slightly cheaper per unit of work, but
  their 6 MB L2 is untested here). Use `--bv-shared`, the per-class windows from §2,
  and A = 12. That is about $3k on demand or $1.7k interruptible for 2^72. Each
  further bit costs 1.8–1.95× more, so 2^75 is about 6.8× that (see COST_MODEL.md). Run `cv measure --gpu` on the actual instance type
  first. It takes 2 minutes, and the 24 GB windows need about 21.5 GiB free.
* **Donated datacenter GPUs (no hourly price):** use every gigabyte. Windows of
  36–38 on 80–288 GB cards save another 5–10% compared with consumer windows.
* **Not worth it:** renting B200/B300/H200 for their memory alone.
