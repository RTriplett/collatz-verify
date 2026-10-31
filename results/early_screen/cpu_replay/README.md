# CPU replay of screened survivors

Plain scalar verification was the fastest observed configuration on this
fixture: **19.737 ms for 262,144 cases**, or **13.28 million cases per CPU
core-second**. Repeated low-bit grouping shares table work successfully, but
its sorting cost makes the best measured regrouping configuration **19.2%
slower** than the scalar control with the same batch size. Full timings and
counters are in [summary.md](summary.md) and [summary.json](summary.json).

## Fixture and measurement scope

The replay uses one worker pinned to logical CPU **6** of the **AMD Ryzen 7
7700X**, with GPU performance measurements stopped. It tests all five methods
at widths 8/10/12/14 and grouping batches of 16,384/65,536/262,144 states:
51 configurations, five rotated timing repetitions each, after warmup.

The input contains the first 262,144 states captured from the first selected
GPU export during the pilot's 32-root warmup. In the
[pilot output](../pilot_gpu/pipeline.jsonl), `early1024_scalar_w4` has exactly
one early warmup batch, and its nonzero `early_class_cases` entry is index 1.
At M=62 the class offset is `floor(62*306/485)+1 = 40`, so this fixture is
**F=41**, with **B=35** and **A=10**. It is a prefix of one launch's survivors,
not a random sample across classes or independent checkpoint roots. Timing
repetitions do not provide a new full-range sampling confidence interval.

All 255 trials match the scalar oracle: **1,949,898 double iterations**,
checksum **3367139950**, and no flags. Each record stores its original start
and exact current `T^M(n)+1` as eight little-endian 32-bit limbs. The fixture
SHA-256 is `e1114e0f312d4868fdde1077c6e0677e269751f3814ca7ff9826ba43ed57e4d1`.
[metadata.json](metadata.json) records the command, executable and source
hashes, CPU affinity and capture provenance.

Reported times include limb decoding, validation, allocations, grouping and
verification. They exclude GPU filtering/transfers, expansion of returned
endpoint/mask descriptions into these individual states, table construction,
fixture loading and oracle comparisons. Consequently, this replay isolates
the CPU verification mechanism; it is not the complete handoff cost.

## What grouping shares

At the screening boundary,

```text
n = n0 + h*2^M
y = T^M(n)+1 = m + 3^F*h + 1
```

The next parity decisions depend on the low bits of **y**, rather than n.
A table row for `y mod 2^w` describes complete double-iteration boundaries
as `p_j*(y >> d_j) + c_j`. Exact integer bounds preserve every original
descent check and intermediate arithmetic-cutoff peak; each original
low-32-bit checksum contribution is retained.

Initial grouping shares the first row lookup. Repeated grouping sorts the
unfinished states again after each transition and shares the next row.
At width 10 and batch size 65,536, repeated grouping reduces lookups from
**1,921,077 to 39,167 (97.96%)**. Its verification phase takes **18.30 ms**,
but grouping adds **5.60 ms**, yielding **23.91 ms** total versus the matched
scalar's **20.07 ms**. The shared row does not remove each state's wide
affine multiplication, descent/cutoff guards or checksum updates.

The independent width-12 table is 3.2% faster than its scalar control at
batch size 262,144, but remains 2.8% slower than the fastest scalar batch
configuration. This small conditional difference is not evidence of a
general table advantage across classes.

No group-level bounding proof, family pruning, or compact evolution of a
whole group is implemented here. These results measure explicit-state
sorting plus shared transition rows; they do **not** rule out grouped
algorithms that eliminate more per-state work.
