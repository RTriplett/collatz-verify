# Integrated N72 configuration (round 2, c-final)

Measured on one RTX 4090. The full report, with every audit and artifact, is
the workflow's `c-final/FINAL.md`. This page is the short version that ships
with the code.

## Configuration

Windows (24 GB desktop card, 20.16 GiB of arrays):

    28,35,35,35,35,34,33,33,27,27,27,27,27,27,27,27

| class | kernel | code | mask |
|---|---|---|---|
| 0 | per-thread jd tail (`cuda/jdip.cuh`) | 93: W11 FD3, dip tested at the boundary | mod 9 |
| 1 | warp queue (`cuda/wq_jump.cuh`, trip-pop layout 43, K 8) with the jd tail (`cuda/wq_jd.cuh`) | 91: W12 FD3, one window ahead | mod 9 |
| 2 | per-thread jd tail behind the row table (`cuda/jdscreen.cuh`) | 93 | (12,9) rows |
| 3-6 | same | 90: W12 FD3, KA0 | rows |
| 7-15 | same | 111: W12 4-limb, KA1 | rows |

fullrange_bench variant `fxr2` (full range) or `fxr2_top` (h >= 2^(A-1),
i.e. [2^(N-1), 2^N)), with:

    CV_FX=93,91,93,90,90,90,90,111,111,111,111,111,111,111,111,111
    CV_FX_WQ=-,v43.k8,-,-,-,-,-,-,-,-,-,-,-,-,-,-

The `cv verify` equivalent (c3-hyg: `--tail-codes`, formerly `--gpu-xtail`,
sets the per-class semantics of the CPU reference and the GPU alike;
`CV_XTAIL_CPU` is refused):

    cv verify ... --rows-from 2 --tail-codes=<codes> --gpu-wq=<specs> [--gpu]

See `docs/ADOPTION.md` for the production path, the report record and spot
checks.

Smaller cards (a 4090 with less memory):

* 16 GB: 28,33,35,35,35,33,31,29,27x8
* 12 GB: 28,33,34,35,34,33,31,29,27x8
* 8 GB: 28,33,33,34,33,33,31,29,27x8

## Measured (N72/A10 parent-dispatch pipeline, 3 streams, 2 producer threads)

Design: 16 cohorts x 512 baseline K38 draws, 4 repetitions, cross-process
palindrome; the controls ran at production windows.

| variant | wall vs tuned original (novote_oee) | vs jump-only (jx) |
|---|---|---|
| full range | 0.3050 [0.3043, 0.3056] | 0.8610 [0.8605, 0.8614] |
| top half | 0.1718 [0.1713, 0.1722] | 0.4849 [0.4846, 0.4852] |

That is 0.179 RTX 4090-years for [0, 2^72) (ratio x 0.586; 0.184 measured
directly on the sample) and 0.101 for [2^71, 2^72) (0.104 directly).

## Definitions that change

* **Iterations** count W-step jumps.
* **Checksum** XORs the low 32 bits of x after every jump.
* **Case set** depends on the windows and the row table; work tuples depend
  on the per-class codes. Package metadata must record codes, windows,
  `rows_from` and the h range. Warp-queue specs do not change semantics.
* **Flags**: a case is flagged iff x >= 2^(cutoff-K) at an executed-jump
  boundary before its stop (K = 7 for W11, 8 for W12). A merge stop
  establishes convergence, not a path record. Path-record runs (cutoff 97)
  must keep the original kernel, and the host refuses cutoff < 96 + K.
* **Tables** (c3-hyg): the jump and jd dip tables are per context
  (`cuda/devtables.cuh`); a context used from another device or after a
  device reset is refused.
