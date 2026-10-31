# Round 3 final configuration (c3-final)

This note describes what round 3 adds to the round-2 tree (`docs/C_FINAL.md`).
Measurements, proofs and limitations are in the c3-final report (`FINAL3.md`
in the c3-final scratch directory). `docs/ADOPTION.md` is the guide to a
production run.

## What is in the tree

* **Base: c3-hyg's production hygiene.** Per-context device tables
  (`cuda/devtables.cuh`), one configuration source for CPU and GPU
  (`src/runspec.rs`), measurement-only codes refused on verification paths, and
  scope and semantics records in every report.
* **c3-2ph's two-phase kernel** (`cuda/twophase.cuh`, `cv_tp_kernel<TV, SCR, V, MLIMB>`).
  * Phase 1 is warp-cooperative screening into a per-warp shared-memory case
    list. Lane i forms completion word i with c-screen's `scr_word`; the
    per-endpoint setup is batched 32-wide; a warp prefix sum compacts the
    survivors.
  * Phase 2 runs per-lane jd tails as a dynamic trip-pop of masked 4-limb FD3
    steps. Block-owned ranges keep the kernel non-persistent.
  * The device code is c3-2ph's, byte for byte (SASS-identical). c3-final
    changed only the host launcher: the FD3 image and the occupancy query now
    come from the launching context's `DevTables`.
* **c3-final: chunk layouts 29, 30, 31** for A >= 11.
  * In these layouts a screening pass forms one 32-word chunk of an endpoint's
    completion range: lane i forms word 32 c + i.
  * Every chunk the range touches is screened in order into the same endpoint
    slot. At A = 11 the top half is chunk 1, the bottom half chunk 0, and the
    full range both.
  * 29 = layout 3 (U = 4) with chunks, 30 = layout 5 (U = 5) with chunks,
    31 = 29 with the per-case digest (audit copy).
  * Layouts without the chunk flag are unchanged and need A <= 10.
* **c3-a's work-preserving pipeline options.**
  * fullrange_bench takes A in 8..12; validate-device takes A.
  * Per-variant options `~s<streams> ~b<log2 launch> ~t<producer threads> ~e<parent entry>`
    (target profile only).
  * `CV_GPU_BATCH`, plus `GpuTuning::run_slots` / `run_batch`.
  * screenbench gains `CV_SB_PARENT_W`, `CV_SB_CACHE` and `screenbench sample`.

## The final configuration

Full range [0, 2^72), A = 10, windows F24 = 28,35,35,35,35,34,33,33,27x8:

    codes   91 on every class (W12 FD3 jd tail, KA1)
    sched   t3,t3,t3,t3,t5,t5,t5,t5,t5,t5,t5,t5,t5,t5,t5,t5   (two-phase; U = 4 on classes 0-3, U = 5 on 4-15)
    screen  (12, 9) row table from class 0 (--rows-from 0)
    streams 2 (CV_GPU_SLOTS=2)

Top half [2^71, 2^72) fallback, A = 11, windows W11 = 28,34,35,36,34,34,33,33,29,26,25,26,27x4:

    codes   91 on every class
    sched   t29 x 7, t30 x 9   (chunk layouts; one 32-word chunk per endpoint)
    screen  rows from class 0, h >= 2^10 (--range top)
    streams 2

## Scheduler word (`GpuTuning::wq`, `--gpu-wq`, `CV_FX*_WQ`)

| spec | bits | kernel |
|---|---|---|
| `-` | 0 | per-thread kernel (`cv_kernel_jd` / `cv_kernel_jds`) |
| `v<L>[.k<K>][.c<log2 fetch>][.o<blocks/SM>]` | bit 22, layout bits 16-20 (+ bit 23) | c-sched's warp queue |
| `t<L>[.k<K>][.c<log2 fetch>][.o<blocks/SM>]` | bit 21 (bit 22 clear), layout bits 16-20 | two-phase kernel, layout L in 1..31 (`TpCfg<L>`) |

* A two-phase class must have a jd FD3 code in `TP_CODES` = 90, 91, 92, 93, 95 or 111.
* Bit 21 together with bit 22 is still the warp queue's capped test tail. It is
  measurement-only and refused on verification paths.
* The scheduler word has no semantic effect: the work tuple (cases,
  iterations, hash, flags) of a code and screen is the same under every
  scheduler. The report records it in `config.warp_queue`.

## Environment

* `CV_GPU_SLOTS=2`: CUDA streams of a session; the final configuration uses 2.
* `CV_TP_VERBOSE=1`: print the registers, shared memory and occupancy of each
  two-phase kernel at first launch.
* Unchanged from rounds 1-2 and c3-hyg: `CV_FX`, `CV_FX2`, `CV_FX3`, and their
  `_WQ` forms; `CV_SB_*`; `CV_BENCH_CUTOFF`.
* `CV_XTAIL_CPU` and `CV_TAIL` are refused if set.
