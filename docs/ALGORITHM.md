# Algorithm

This is a precise description of what `cv` computes, the rules it uses, and
why each one is sound. Notation follows Angeltveit, *An improved algorithm for
checking the Collatz conjecture for all n < 2^N* (v0.3, 2026). That paper is the
baseline reproduced here.

## 1. Setup

`T(n) = n/2` for even `n` and `T(n) = (3n+1)/2` for odd `n`. The parity sequence of
the first `k` steps depends only on `n mod 2^k`. So the whole residue class
`n = n0 + 2^k a` (`a >= 0`, `0 <= n0 < 2^k`) satisfies

    T^k(n0 + 2^k a) = m + 3^f a,      m = T^k(n0),  f = number of odd steps.     (1)

A class is the tuple `(n0, k, f, m)`. Its two children add bit `k` of `n`, one
taking an odd step and one an even step. The search verifies all `n < 2^N` by
strong induction on `n`. A number may be skipped once some certificate shows its
trajectory reaches a strictly smaller positive integer.

## 2. Pipeline

With `A` = bitvector bits, `M = N - A`, and `B` = lookahead window:

1. **Step 1, tree search (CPU).** Depth-first over classes `k <= M`, pruned by the
   rules of §3. Survivors at depth `M` are *endpoints* `(n0, f, m)`.
2. **Step 2, bitvectors (GPU).** An endpoint has `2^A` completions
   `n = n0 + h 2^M`, with `T^M(n) = m + 3^f h`. One precomputed `2^B`-bit array per `f`
   marks the B-step parity patterns that survive descent and merges inside the
   window (§4). Bits sit at position `pattern * 3^-f mod 2^B`, so the `2^A`
   completions of an endpoint are consecutive. One funnel shift plus an AND
   against the mod-9 preimage mask filters 32 completions.
3. **Step 3, tail (GPU).** Each surviving completion is iterated, using Barina's
   alternating odd-run/even-run loop on `T(n)+1` with four 32-bit limbs, until it
   falls below `n`. A trajectory reaching `2^cutoff` (default 120 bits) is flagged
   for big-integer re-checking.

Every number below `2^N` either belongs to a culled class, is excluded by a
bitvector or mod-9 bit, or is iterated explicitly.

## 3. Step-1 rules (per class, no global state)

A rule culls class `(n0, k, f, m)` only with a **merge certificate**: a partner
family `p + S a` satisfying

    T^s(p + S a) = m + 3^f a,     p < n0,     S <= 2^k.                         (2)

Then `p + S a < n0 + 2^k a` for all `a >= 0`, so every member merges with a smaller
integer. Descent is the special case where the partner is the member's own
iterate. Every rule is a function of the class alone, so subtrees are
independent and the search stays an embarrassingly parallel DFS with
kilobytes of state.

Certificates are **hereditary**. If (2) holds for a class, it also holds for
both children, with slope `2S <= 2^(k+1)`. So culling a node removes exactly the
endpoints whose ancestors carry a certificate.

### Baseline rules (Angeltveit)

| rule | test | certificate |
|---|---|---|
| descent | `306 k >= 485 f` (Thm 4.1; `306/485 < log 2 / log 3`) | own iterate |
| pm1 | `m = 2 mod 3` after an even number (`>= 2`) of even steps (Lemma 2.8), slope `306(k-1) >= 485(f-1)` | `(2m-1)/3` |
| OEE | an odd step taken where `3^f <= 2^(k+1)` arms a 3-state machine; an odd run followed by two even steps is a merge with `(x-1)/2` (Hercher, Lemma 2.9) | `(x-1)/2` |

These three rules depend on the parity vector only, through the state
`(k, f, oee, ne)`. Here `ne` counts trailing even steps, folded to
{0, 1, even ≥ 2, odd ≥ 3}. This is what makes the workload exactly countable (§6).

**Root-armed OEE.** The reference CUDA program is started with
`3 2 2 9 1 0`, which arms the OEE machine at the root. The reference CPU program
hard-codes 0, the unarmed state. Arming at the root culls the class with
parity `O^l E E` at every depth: `n0 = 2^(k-2) - 1`, or its `+2^(k-1)` twin for odd
`k`. That cull is exactly the `all_ones_filter` added in `collatz_bitvec/main.rs`.
It reduces survivors at k=37 from 678,988,765 to 575,349,866 (−15.3%) and is
part of the baseline here.

### Improvements

* **pm∞ (maximal 3-adic path merge).** Write `m + 1 = 3^t q` with
  `t = min(v3(m+1), f)`. Then `u = 2^t q - 1` satisfies `T^t(u) = m`, and the family
  `u + 2^t 3^(f-t) a` is valid because each inverse odd step needs one factor
  of 3 in the slope. Cull if `u < n0` and `2^t 3^(f-t) <= 2^k`. Both conditions
  improve as `t` grows, so only `t_max` is tested. `v3(m+1)` is computed with
  multiplications by `3^-1 mod 2^128`.
* **mix(3^J) (mixed inverse words).** A forward word of `s` steps with `o` odd
  steps maps `p` to `(3^o p + b)/2^s`. It inverts to `p = (2^s m - b)/3^o`, which
  is integral on exactly one residue of `m mod 3^o`. For every residue
  mod `3^j` (`j <= J`), the tables store the valid word with the largest ratio
  `3^o/2^s`, over `o <= j`, `o <= s <= 2o`, `2^s < 3^o`, and first forward step odd.
  The node looks up `m mod 3^min(f,J)`. That residue is carried incrementally
  in the node as `m mod 3^16`, so no 128-bit division is needed. The node is culled if
  `0 < p < n0` and `2^s 3^(f-o) <= 2^k`. The largest table is 4.3 MB (J=12), 12.8 MB (J=13) or 344 MB (J=16), and the
  smaller moduli add half again. J=13, about 19 MB in total, is the last size that
  stays in a 32 MB L3 cache.

Applying the table only at the final depths loses culls. With mix from
`k >= 30` only, k=37 has 545.4M survivors instead of 543.5M, because a
best-ratio word does not always extend to the child. So the rule is applied at
every depth.

### Optional checkpoint walkback

`--walkback-depth K --walkback-states S` adds a bounded inverse-tree search after
the other step-1 rules, only at depth `K`. Starting at `m`, it explores inverse
even steps `v -> 2v` and integral inverse odd steps `v -> (2v-1)/3`, with at most
`f` inverse odd steps. A path of `s >= 1` inverse steps, `o` of them odd, gives the
partner family `p + 2^s 3^(f-o) a`. It culls only when `0 < p < n0` and
`2^s 3^(f-o) <= 2^k`, the same certificate conditions as the other merge rules.
Search bounds prune branches that cannot meet those conditions. If the state
budget is exhausted, the class is kept.

This option is not enabled by `--rules improved`. It searches a broader family
of inverse words than the mix tables, but at one checkpoint and with a finite
search budget. The measured additional pruning and the limits of the timing
evidence are in [FINDINGS.md](FINDINGS.md#optional-checkpoint-walkback).

## 4. Step-2 bitvectors

For a window pattern `x` (the low `B` bits of `T^M(n)+1`), Angeltveit computes the
maximal *dip* over the window, in units of 1/306 bit:

* `+306` per even step, `-179` per odd step (using `485/306 > log2 3`);
* `+179` (one inverse odd step) after an even number (≥ 2) of even steps (pm1);
* `+306` before an odd run that ends with two even steps (OEE).

`BV_i`, for `f = offset + i` with `offset = floor(306M/485) + 1`, has the bit set iff
`dip <= 485 f - 306 M`. A cleared bit means `n` provably reaches a smaller number
within `M + B` steps: Theorem 4.4, with all values `>= 99781` and a one-unit margin
absorbing the `+1` terms.

### Soundness fix: two bitvector families

The pm1 credit uses Lemma 2.8. That lemma counts even steps **since the last odd
step**. While the window is still in its leading run of even steps, that count
includes the endpoint's trailing even steps from *before* the window. A
bitvector indexed only by `f` cannot know them. The reference construction
starts the count at 0, so for endpoints with an odd trailing-even count it
credits merges that do not exist.

Example (N = 26, A = 6): `n = 55,795,739` reaches depth 20 with `f = 14` and one
trailing even step. The window opens `E E O ...`. The bitvector credits a pm1
merge after the second window step (dip 791 > cutoff 670), but the iterate
there is `63,626,599 ≡ 1 (mod 3)`. There is no merge partner, and no descent or
other merge inside the window. `tests/bv_audit.rs` exhaustively checks
every bitvector exclusion for an explicit witness: descent, a valid inverse
chain, an OEE pattern, or a mixed word verified by direct iteration. The
reference construction fails with 386 unjustified exclusions out of 857,601 at
N = 26.

The fix builds two families of arrays, with the even counter started at 0 or 1, and
selects one per endpoint by `ne & 1`. This is exact. The kernel picks the family
by pointer offset, so batches stay per-`f`. The audit passes with 0 unjustified
exclusions at (N, A) = (26, 6), (30, 10), (32, 8). The cost is +0.27% cases at
A = 6 and −0.18% at A = 10, B = 24. `--bv-legacy` reproduces the reference
exactly: 756,583,624 cases at N = 40, A = 6 (Fig. 5.1).

### Shared bitvectors: one family, half the memory

`--bv-shared` keeps one array per class. It gives no mod-3 credit until the
window has taken its first odd step, the only place where the endpoint's
trailing-even parity matters. Every other credit is exact for both parities,
because the even counter resets at each odd step. For every window pattern the
shared dip is therefore the smaller of the two split-family dips (exhaustively
tested for B = 12, 18, 22), so Shared is sound whenever Split is. It costs +0.20
to 0.32% cases at 2^72 for B from 24 to 38 and halves bitvector memory.

### Window credits (zero runtime cost)

Inside the window, `T^j(z) = (3^(f_w) z + c)/2^j`, where `c` depends only on the
parity pattern. So for `t <= f_w` (odd steps taken inside the window),
`y = T^(M+j)(n)` and `T^j(x - 1)` agree mod `3^t`. **The pattern alone fixes
`y mod 3^min(f_w, J)`**. That allows extra merges to be credited during
precomputation:

* `--bv-pm-inf`: chain of length `t = min(v3(y+1), f_w)`, credit `+179 t`.
* `--bv-mix-j J`: the tabulated mixed word `(s, o)`, credit `+485 o - 306 s`.

The same Theorem 4.4 inequality covers them. A credit beats the cutoff iff
`485(f-o) + 306 s < 306 k` (strict), so `p < 2^s y / 3^o < n` once the
`(1 + 1/(3·99781))^f` factor is absorbed by the unit margin. That holds for `f` below
about 600. The audit test re-verifies every resulting exclusion by direct
iteration (0 unjustified at N = 26 and 30). At N = 72, A = 10, B = 24 this removes
3.42% of cases.

Pattern 0 is iterated as its representative `2^B`, so that `x − 1 >= 0`
carries the true residues mod 3^t (as with every other pattern).

### Large windows: device-built arrays and per-class windows

The lookahead filter's value grows with B: every extra bit removes the cases
that would have descended or merged one step later. Arrays of 2^B bits per
class are built on the GPU by a kernel that mirrors the host construction step
for step. `cv gpu-check` and `scripts/reproduce.sh` compare the two bit for bit.
Indices are 64-bit (`3^-f mod 2^64`), so any B up to 39 works; at B = 39 the window
iterate `x < 2^39 (3/2)^39 ≈ 2^62` still fits a u64. With `--class-b` each class
gets its own window. Classes 0–7 carry 99% of GPU time and gain most from a deeper
window; the high-f classes gain almost nothing. On the CPU, any window can be
evaluated directly per completion (`--direct`), giving the same decision as the
stored bit. That is how windows larger than the host can store are measured.

### Mod-9 sieve and tail OEE

`n ≡ 2, 4, 5, 8 (mod 9)` lies on the path of a smaller number (Lemma 2.6). These
residues are masked per `n0 mod 9`. The endpoint's OEE state also removes
completions whose next one or two steps would complete the pattern. That is two
parity tests per completion, applied on both the CPU and the GPU.

## 5. Step 3 and the checksum

The kernel's arithmetic core is Angeltveit's, unchanged. The kernel
recomputes `T^M(n0)` from `n0` (16 bytes per endpoint on the bus instead of 32) and
accumulates per warp:

* the number of cases;
* double iterations;
* the XOR of `(T(n)+1) mod 2^32` after every double iteration.

The CPU tail reproduces the same loop and hash bit for bit, so a CPU run and a GPU
run over the same work agree exactly. A work package's (endpoints, cases,
iterations, hash) tuple is deterministic.

Trajectories reaching `2^cutoff` (default 120) are abandoned on the GPU and
listed. `cv verify` re-checks each with 256-bit arithmetic and fails if any
cannot be finished, or if one launch flags more starts than its 65,536-entry
buffer holds. Starts that would not fit 128 bits at all, possible only for rare
high-f endpoints finished on the CPU at large N, are flagged the same way. Two
hard limits apply: the tree's 128-bit node arithmetic needs `M = N − A <= 80`,
and the GPU refuses configurations whose largest start could exceed 2^120
before its first overflow check.

## 6. Exact counting

Because the baseline rules depend only on `(k, f, oee, ne)` and the map from
`n0 mod 2^k` to its parity vector is a bijection, survivors at any depth are
lattice-path counts. `dp.rs` computes them in O(k²). The DP matches the DFS exactly
(tests). It reproduces OEIS A076227 (descent only) and the k ≤ 37 table in
`collatz-probe/cull_survivors_k1_k37.csv` (575,349,866 at k = 37). Cases to check
follow from equidistribution of completions over window patterns and residues
mod 9:

    cases(N) = (5/9) 2^A  Σ_states S_M(state) · ρ(array(state), oee(state))

Here ρ is the set-bit density of the endpoint's array, restricted by its OEE
state. The prediction is within 0.003% of measured counts: N = 40 and 44 with
A = 6 and 10.

## 7. What is tested

`cargo test --release` covers:

* the rational approximation and mod helpers;
* the DP against the DFS, the DP against the backward DP, and A076227;
* pm∞ and mix certificates against direct iteration, every certificate fired for k ≤ 18;
* the sampler (uniformity in `f`, and improved survival against the exact census);
* the bitvector audit (§4).

Command-line checks (commands in the README):

* Fig. 5.1 exactly (legacy bitvectors).
* The k = 37 CSV columns exactly: 575,349,866 / 548,786,150 / 543,518,371.
* The residue-sieve `P∞+M8` and `P∞+M16` columns exactly at k = 24 and 30.
* Identical CPU and GPU counts and hashes (see the README for the configurations checked).
* A 69-package decomposition reproducing the full N = 40 run's counts and hash.
