# Cost model: resources to verify all n < 2^N

The question is how much GPU time and how much CPU time a complete
verification below 2^N costs, for a given algorithm configuration and hardware.
The answer here is not an extrapolation from small N. It is

    exact workload counts at the target N  x  unit costs measured on real
    work drawn uniformly from that same target N.

## 1. Exact counts (no sampling)

The baseline step-1 rules depend only on the parity state `(k, f, oee, ne)`
(ALGORITHM.md §3). So the number of classes surviving to any depth, the
number of tree nodes generated, and the endpoints at depth `M = N − A` split by
`f` and `ne` are computed exactly by dynamic programming (`cv predict`, `dp.rs`).
At N = 72 the DP takes milliseconds:

| N=72, A=10 | exact value |
|---|---|
| step-1 tree nodes | 9.9332×10^15 |
| endpoints at depth 62 | 4.3997×10^15 |
| cases (B = 24) | 7.4664×10^17 (0.0158% of 2^72) |

Cases follow from the endpoint counts and the exact set-bit densities of each
bitvector. They rest on one equidistribution assumption, that completions spread
uniformly over window patterns and residues mod 9. That assumption is checked
against full runs:

| run | measured cases | DP prediction | difference |
|---|---|---|---|
| N=40, A=6, 8 bitvectors (reference) | 756,583,624 | 756,604,200 | +0.003% |
| N=40, A=6, sound bitvectors | 758,607,096 | 758,629,100 | +0.003% |
| N=44, A=10 | 12,137,740,670 | 12,137,760,000 | +0.0002% |

## 2. Unit costs from uniform real endpoints

`cv measure` builds, for each bitvector class `i` (endpoints with `f = offset + i`),
the backward DP of subtree counts restricted to that class. It then descends from
the root, choosing each child with probability proportional to its count. Every
depth-M endpoint of the class is drawn with probability exactly `1/S_i`. That
holds at depth 62 or 70 just as at depth 30. Each draw is a real class
`(n0, m, f)`. On each draw the tool does three things:

* It runs steps 2+3 on the CPU and counts cases and double iterations.
* It evaluates the value-dependent step-1 rules (pm∞, mix, walkback) at every
  depth along the draw's path, which gives the fraction of the class the
  improved rules would have culled.
* It times 2^21 draws per class through the real CUDA kernel, in full 2^20
  batches, serialized, after one warm-up run, taking the median of three.

Totals are `Σ_i S_i × mean_i`. Standard errors come from the draws; they are
below 0.01% for cases and iterations with 50k–100k draws per class.

**Per-operation GPU model.** Identical work re-timed minutes apart differs by up
to ±6%. The card is a desktop GPU that also drives the display, and boost clocks
follow thermal state. At N = 80 the same draws measured 188.6 and then 176.3
GPU-years. Operation counts, by contrast, are known to 0.01%. `scripts/gpu_fit.py`
therefore fits one model to all 40 sweep runs × 16 classes:

    seconds/endpoint = a(A,B) + 14.0 ps · cases/endpoint + 3.95 ps · double iterations/endpoint
    a(10,24) = 0.61 ns,  a(10,28) = 0.73 ns,  a(12,28) = 1.64 ns

The overhead covers recomputing T^M and scanning 2^A bits. Its rise from B = 24 to
B = 28 is the cache cost of 32 MB arrays. The tables use this model on each
configuration's measured operation counts. Raw per-run timings are listed
alongside. Fitted and raw totals agree within ±5% for every A = 10 run. At N = 80 the
model predicted the re-measured value to 0.1%. A = 12 runs scatter up to 9%: their
4× longer threads diverge differently, and the shared per-case cost absorbs only
part of that.

**Validation against a complete run (N = 50, A = 10, B = 24):**

| | sampled × exact | full run |
|---|---|---|
| cases | 5.5116×10^11 ± 5.6×10^7 | 5.51223×10^11 (−0.011%) |
| double iterations | 3.5000×10^12 ± 3.8×10^8 | 3.500259×10^12 (−0.007%) |
| GPU kernel seconds | 23.99 | 23.54 serialized; 23.4 s wall with 3 streams |

The sampled GPU time is 1.9% conservative. Batches in a real run come from depth-first
order and are slightly more uniform than random ones.

**End-to-end check at N = 54.** Two complete production runs were made, each
with 3 CUDA streams and step 1 on 16 threads (`results/validation/`):

| | predicted | complete run |
|---|---|---|
| sound baseline: cases | 6.9416×10^12 ± 6×10^8 | 6,941,716,164,126 (+0.004%) |
| sound baseline: time | 296.8 GPU-s | 287.0 s wall (3% faster) |
| all improvements: endpoints | 4.4820×10^10 | 44,845,477,535 (+0.06%) |
| all improvements: cases | 5.3435×10^12 ± 1.4×10^9 | 5,344,896,148,408 (+0.03%) |
| all improvements: tree nodes | 1.0154×10^11 | 101,513,615,194 (+0.03%) |
| all improvements: time | 252.0 GPU-s | 280.4 s wall, of which 26.5 s is the one-time B = 28 bitvector build (+0.8%) |

The all-improvements run kept 14.5 of 16 threads busy. Production step 1 costs
about 10% more CPU per node than the census rate, from endpoint packing, so
budget about 6.3 rather than 5.6 cores per GPU for that configuration.

**Step-1 CPU time** is node count times measured ns per node
(`results/cpu_bench.csv`, Ryzen 7 7700X). For the improved rules, the node count is
`Σ_k G(k) ψ(k−1)`, with the survival fraction ψ sampled every 4 depths. That
estimate is checked against an exact census at M = 37: 1.2194×10^9 sampled versus
1.22015×10^9 exact (−0.06%).

| rules | ns/node, 1 thread | ns/node/thread, 16 threads | nodes/s, whole 7700X |
|---|---|---|---|
| baseline | 7.6 | 16.4 | 0.97×10^9 |
| + pm∞ | 10.1 | 21.6 | 0.74×10^9 |
| + pm∞ + mix(3^12) | 15.1 | 30.4 | 0.53×10^9 |
| + pm∞ + mix(3^13) | 16.8 | 32.9 | 0.49×10^9 |
| + pm∞ + mix(3^14) | 28.4 | 40.1 | 0.40×10^9 |
| + pm∞ + mix(3^16) | 47.9 | 80.0 | 0.20×10^9 |

The tables below give "CPU core-years" for this CPU, counting one core as two
SMT threads. "Cores per GPU" is the number of such cores needed to keep one 4090
busy.

## 3. Results (RTX 4090 + Ryzen 7 7700X)

| N | configuration | RTX 4090-years | (raw timing) | vs sound baseline | CPU core-years (7700X) | cores per GPU | cases |
|---|---|---|---|---|---|---|---|
| 64 | Angeltveit reference (A10 B24, original bitvectors) | 0.00593 | 0.0059 | +0.3% | 0.0156 | 2.6 | 4.302e+15 |
| 64 | Sound baseline (A10 B24) | 0.00591 | 0.0058 | +0.0% | 0.0156 | 2.6 | 4.287e+15 |
| 64 | + step-1 pm_inf + mix(3^13) (A10 B24) | 0.00568 | 0.00557 | -3.9% | 0.0294 | 5.2 | 4.123e+15 |
| 64 | + B28 + window pm_inf/mix (A10 B28) | 0.0051 | 0.00513 | -13.8% | 0.0156 | 3.1 | 3.366e+15 |
| 64 | All improvements (A10 B28) | 0.0049 | 0.00493 | -17.2% | 0.0294 | 6.0 | 3.243e+15 |
| 64 | All improvements, CPU-light (A12 B28) | 0.00522 | 0.00507 | -11.8% | 0.00822 | 1.6 | 3.658e+15 |
| 68 | Sound baseline (A10 B24) | 0.0771 | 0.0739 | +0.0% | 0.197 | 2.6 | 5.552e+16 |
| 68 | + step-1 pm_inf + mix(3^13) (A10 B24) | 0.0738 | 0.0708 | -4.2% | 0.368 | 5.0 | 5.327e+16 |
| 68 | + B28 + window pm_inf/mix (A10 B28) | 0.0678 | 0.0657 | -12.0% | 0.197 | 2.9 | 4.477e+16 |
| 68 | All improvements (A10 B28) | 0.065 | 0.063 | -15.7% | 0.368 | 5.7 | 4.302e+16 |
| 68 | All improvements, CPU-light (A12 B28) | 0.0679 | 0.0649 | -11.9% | 0.102 | 1.5 | 4.755e+16 |
| 70 | Sound baseline (A10 B24) | 0.283 | 0.274 | +0.0% | 0.718 | 2.5 | 2.051e+17 |
| 70 | + step-1 pm_inf + mix(3^13) (A10 B24) | 0.271 | 0.262 | -4.3% | 1.34 | 5.0 | 1.966e+17 |
| 70 | + B28 + window pm_inf/mix (A10 B28) | 0.249 | 0.252 | -12.1% | 0.718 | 2.9 | 1.656e+17 |
| 70 | All improvements (A10 B28) | 0.238 | 0.242 | -15.8% | 1.34 | 5.6 | 1.590e+17 |
| 70 | All improvements, CPU-light (A12 B28) | 0.247 | 0.238 | -12.8% | 0.368 | 1.5 | 1.721e+17 |
| 71 | Sound baseline (A10 B24) | 0.549 | 0.546 | +0.0% | 1.35 | 2.5 | 4.006e+17 |
| 71 | + step-1 pm_inf + mix(3^13) (A10 B24) | 0.524 | 0.523 | -4.4% | 2.52 | 4.8 | 3.836e+17 |
| 71 | + B28 + window pm_inf/mix (A10 B28) | 0.467 | 0.467 | -14.8% | 1.35 | 2.9 | 3.079e+17 |
| 71 | All improvements (A10 B28) | 0.447 | 0.447 | -18.6% | 2.52 | 5.6 | 2.954e+17 |
| 71 | All improvements, CPU-light (A12 B28) | 0.47 | 0.47 | -14.2% | 0.708 | 1.5 | 3.296e+17 |
| 72 | Angeltveit reference (A10 B24, original bitvectors) | 1.03 | 1.03 | +0.2% | 2.59 | 2.5 | 7.480e+17 |
| 72 | Sound baseline (A10 B24) | 1.03 | 1.01 | +0.0% | 2.59 | 2.5 | 7.466e+17 |
| 72 | + step-1 pm_inf + mix(3^13) (A10 B24) | 0.983 | 0.963 | -4.3% | 4.82 | 4.9 | 7.151e+17 |
| 72 | + B28 + window pm_inf/mix (A10 B28) | 0.889 | 0.906 | -13.5% | 2.59 | 2.9 | 5.901e+17 |
| 72 | All improvements (A10 B28) | 0.851 | 0.868 | -17.2% | 4.82 | 5.7 | 5.663e+17 |
| 72 | All improvements, CPU-light (A12 B28) | 0.905 | 0.867 | -11.9% | 1.34 | 1.5 | 6.360e+17 |
| 73 | Sound baseline (A10 B24) | 1.96 | 1.9 | +0.0% | 4.86 | 2.5 | 1.426e+18 |
| 73 | + step-1 pm_inf + mix(3^13) (A10 B24) | 1.87 | 1.82 | -4.4% | 9.07 | 4.8 | 1.364e+18 |
| 73 | + B28 + window pm_inf/mix (A10 B28) | 1.71 | 1.81 | -12.9% | 4.86 | 2.9 | 1.135e+18 |
| 73 | All improvements (A10 B28) | 1.63 | 1.73 | -16.7% | 9.07 | 5.6 | 1.088e+18 |
| 73 | All improvements, CPU-light (A12 B28) | 1.7 | 1.76 | -13.4% | 2.52 | 1.5 | 1.182e+18 |
| 74 | Sound baseline (A10 B24) | 3.78 | 3.77 | +0.0% | 9.11 | 2.4 | 2.762e+18 |
| 74 | + step-1 pm_inf + mix(3^13) (A10 B24) | 3.61 | 3.6 | -4.5% | 16.9 | 4.7 | 2.640e+18 |
| 74 | + B28 + window pm_inf/mix (A10 B28) | 3.19 | 3.27 | -15.6% | 9.11 | 2.9 | 2.097e+18 |
| 74 | All improvements (A10 B28) | 3.05 | 3.13 | -19.3% | 16.9 | 5.6 | 2.009e+18 |
| 74 | All improvements, CPU-light (A12 B28) | 3.24 | 3.18 | -14.3% | 4.82 | 1.5 | 2.265e+18 |
| 75 | Sound baseline (A10 B24) | 7.05 | 6.98 | +0.0% | 17.4 | 2.5 | 5.116e+18 |
| 75 | + step-1 pm_inf + mix(3^13) (A10 B24) | 6.74 | 6.66 | -4.5% | 32.3 | 4.8 | 4.888e+18 |
| 75 | + B28 + window pm_inf/mix (A10 B28) | 6.07 | 6.33 | -13.9% | 17.4 | 2.9 | 4.017e+18 |
| 75 | All improvements (A10 B28) | 5.8 | 6.04 | -17.7% | 32.3 | 5.6 | 3.846e+18 |
| 75 | All improvements, CPU-light (A12 B28) | 6.21 | 6.05 | -12.0% | 9.07 | 1.5 | 4.351e+18 |
| 76 | Angeltveit reference (A10 B24, original bitvectors) | 13.4 | 13.4 | -0.1% | 32.5 | 2.4 | 9.762e+18 |
| 76 | Sound baseline (A10 B24) | 13.4 | 13.5 | +0.0% | 32.5 | 2.4 | 9.771e+18 |
| 76 | + step-1 pm_inf + mix(3^13) (A10 B24) | 12.9 | 12.9 | -4.4% | 60.5 | 4.7 | 9.344e+18 |
| 76 | + B28 + window pm_inf/mix (A10 B28) | 11.6 | 11.8 | -13.6% | 32.5 | 2.8 | 7.702e+18 |
| 76 | All improvements (A10 B28) | 11.1 | 11.2 | -17.3% | 60.5 | 5.4 | 7.379e+18 |
| 76 | All improvements, CPU-light (A12 B28) | 11.6 | 11.4 | -13.8% | 16.9 | 1.5 | 8.037e+18 |
| 77 | Sound baseline (A10 B24) | 25.9 | 26 | +0.0% | 60.7 | 2.3 | 1.888e+19 |
| 77 | + step-1 pm_inf + mix(3^13) (A10 B24) | 24.7 | 24.9 | -4.5% | 113 | 4.6 | 1.804e+19 |
| 77 | + B28 + window pm_inf/mix (A10 B28) | 22.2 | 22.6 | -14.2% | 60.7 | 2.7 | 1.474e+19 |
| 77 | All improvements (A10 B28) | 21.2 | 21.6 | -18.0% | 113 | 5.3 | 1.411e+19 |
| 77 | All improvements, CPU-light (A12 B28) | 22.1 | 24.3 | -14.7% | 32.3 | 1.5 | 1.538e+19 |
| 78 | Sound baseline (A10 B24) | 48.2 | 48 | +0.0% | 115 | 2.4 | 3.464e+19 |
| 78 | + step-1 pm_inf + mix(3^13) (A10 B24) | 46 | 45.8 | -4.6% | 214 | 4.7 | 3.309e+19 |
| 78 | + B28 + window pm_inf/mix (A10 B28) | 42.8 | 43.6 | -11.2% | 115 | 2.7 | 2.849e+19 |
| 78 | All improvements (A10 B28) | 40.8 | 41.7 | -15.2% | 214 | 5.3 | 2.726e+19 |
| 78 | All improvements, CPU-light (A12 B28) | 42.4 | 42.4 | -12.1% | 60.5 | 1.4 | 2.952e+19 |
| 80 | Angeltveit reference (A10 B24, original bitvectors) | 177 | 196 | +0.1% | 423 | 2.4 | 1.277e+20 |
| 80 | Sound baseline (A10 B24) | 176 | 189 | +0.0% | 423 | 2.4 | 1.275e+20 |
| 80 | + step-1 pm_inf + mix(3^13) (A10 B24) | 168 | 180 | -4.7% | 786 | 4.7 | 1.217e+20 |
| 80 | + B28 + window pm_inf/mix (A10 B28) | 154 | 158 | -12.6% | 423 | 2.7 | 1.023e+20 |
| 80 | All improvements (A10 B28) | 147 | 151 | -16.7% | 786 | 5.3 | 9.782e+19 |
| 80 | All improvements, CPU-light (A12 B28) | 156 | 154 | -11.8% | 214 | 1.4 | 1.090e+20 |

GPU-years assume sustained clocks. CPU core-years are Ryzen 7 7700X cores
(two SMT threads each). Per-bit growth is not constant: the case ratio between
consecutive N swings between 1.83 and 1.94 with the log2 3 lattice, so
interpolate from this table rather than using one growth factor.

Configurations:

* **Angeltveit reference**: his algorithm and parameters (A=10, B=24), with his
  original bitvectors. These have the soundness gap described in ALGORITHM.md §4,
  so they are listed for comparison only.
* **Sound baseline**: the same algorithm with the two-family bitvector fix.
  This is the reference point for the percentages.
* **+ step-1 pm∞ + mix(3^13)**: the improved per-class rules in the tree.
* **+ B28 + window**: lookahead window 28 (32 MB arrays) with pm∞ and mix
  credits baked into the bitvectors. Zero runtime cost; tree rules are baseline.
* **All improvements**: both.
* **CPU-light**: A = 12, which cuts step-1 work 3.6× for about 4% more GPU time.

### Larger windows

The table above stops at B = 28, the largest window built on the host. Device-built
arrays and per-class windows go further. On a 24 GB card the best windows are
32,35,35,35,35,35,32,32 for classes 0–7, giving **0.675 RTX 4090-years at 2^72**. A 32 GB
card gets 0.667 and a 288 GB B300 0.610. These are measured at N = 72 by direct
window evaluation, timed on the real kernel up to B = 36, and validated by a complete
N = 56 run; details in docs/HARDWARE.md. At these windows step 1 is the bottleneck on a
16-thread host at A = 10, so use A = 12, which costs about 4% more GPU time and 3.6× less
CPU.

## 4. Other hardware

Only the RTX 4090 is measured here. Run `cv measure --n 72 --gpu` on the target
GPU. It takes 1–2 minutes and produces the same tables for that device.
docs/HARDWARE.md has verified specs and prices and a cost per 2^72 for each
card. As a guide:

* **RTX 3060**: Angeltveit's own run of N = 50 took 166 s. This implementation on
  the 4090 takes 23.4 s, a ratio of 7.1. Scaling by SM count × boost clock predicts
  6.5, so the kernel scales about as expected within Ampere/Ada.
* **By SM count × clock, unmeasured**: RTX 5090 ≈ 1.27× a 4090, H100 SXM ≈ 0.75×,
  B300 ≈ 0.87×. NVIDIA's table gives the same per-SM integer multiply-add, shift
  and bit-find rates for compute capability 7.5 through 12.x, and this kernel is
  bound by those.
* **L2 cache.** One launch touches the two arrays of one class (2 × 32 MB at
  B = 28, which fits the 4090's 72 MB L2). Each endpoint reads only about 132
  contiguous bytes, so DRAM-resident arrays on smaller-L2 parts (A100: 40 MB,
  H100: 50 MB) should cost little. This is reasoning, not a measurement; B = 26
  (16 MB) is the safe choice there.

## 5. Work units

`cv plan --n N --costs measure.json --target-hours H` splits the tree until
every subtree's predicted GPU time is at most H. The prediction is exact baseline
endpoint counts below the node by class, times measured seconds per endpoint.
Subtrees are then packed into packages of about H hours each. A package is a list of
`(n0, k, f, m, oee, ne)` roots. `cv verify --plan … --package i [--gpu]` runs one
package and prints a deterministic (endpoints, cases, iterations, hash) tuple, so
any package can be recomputed elsewhere as a spot check. The XOR of all
package hashes equals the hash of a single full run (checked at N = 40,
69 packages). Package sizes stay within the target because subtree cost is known
exactly before the run, not conjecturally.

Example, N = 72, A = 10, B = 28 with window credits, baseline step-1 costs
(predicted total 7,940 GPU-hours from that run's raw timings):

| target | packages | median | 1st percentile | max | roots per package | plan file |
|---|---|---|---|---|---|---|
| 1 GPU-hour | 11,616 | 0.77 h | 0.24 h | 1.00 h | median 1, max 4 | 1.9 MB |
| 4 GPU-hours | 2,637 | 3.38 h | 1.26 h | 3.94 h | median 1, max 3 | 0.4 MB |

Planning N = 72 takes about 90 s on 4 threads.

## 6. Caveats

* The kernel is Angeltveit's arithmetic core, not re-tuned. Per-class time grows
  from 1.6 ns/endpoint (class 0, ~29 cases) to 48 ns (class 15, ~570 cases).
  Classes 1–5 carry most of the work.
* Flagged trajectories, those reaching 2^120, are abandoned on the GPU and must
  be re-checked with big integers. `cv measure` reports their expected number
  (`flagged=` in the totals). The estimates are about 10^4 at 2^72, 2×10^9 at 2^77 and
2×10^11 at 2^80, from very few flagged draws, so treat them as orders of magnitude.
`cv verify` re-checks every flagged start with 256-bit arithmetic, microseconds
each, and fails if any cannot be finished.
* GPU timings were taken on a desktop 4090 that also drives the display. They
  vary by roughly ±2% between runs.
* CPU figures are for one desktop Zen 4 part with SMT. Server cores at lower
  clocks will be slower per core, so measure on the target CPU too
  (`cv census --depth 40` and the node counts).
