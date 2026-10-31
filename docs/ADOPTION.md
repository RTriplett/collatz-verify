# Adopting the round-3 final configuration for a production run

This is for Ryan. It covers four things: what in the tree is the production
path, how to run and record a production verification, how to spot-check a
finished package on the CPU, and the limits that still apply.

The tree is `c3-final/repo`, which is c3-hyg's production-safe tree plus three
additions:

* c3-2ph's two-phase verification kernel;
* c3-final's chunk layouts for A >= 11;
* c3-a's host-side pipeline options.

This file is c3-hyg's ADOPTION.md, updated for round 3. Everything c3-hyg
established about scope records, the single configuration source,
measurement-only refusals, per-context tables, plan parts and spot checks
still holds. In round 3 the two-phase kernels joined the same paths: they read
their FD3 image from the launching context's `DevTables`, take their per-class
code from `TailParams`, and are recorded in `config.warp_queue`. The
measurements behind the configuration below are in `FINAL3.md` (c3-final
scratch directory).

    S=/tmp/claude-1000/-home-ryan-Projects-Collatz-distributed/f12bd555-dbb4-485f-80c9-1a72198d26b8/scratchpad
    F=$S/c3-final

## 1. What runs in production, and what does not

### The production path

| layer | files | functions / items |
|---|---|---|
| CLI entry points | `src/main.rs` | `cv verify` (`Cmd::Verify`, `run_gpu`, `run_parts`, `PartRunner`), `cv plan`, `cv export-paths` |
| run configuration (single source) | `src/runspec.rs` | `TailCodes`, `WqSchedule`, `verification_kind`, `RunRecord`, `PartRecord`, `PlanRecord`, `reject_legacy_env`, `WORK_DEFINITION_VERSION` |
| configuration and report | `src/verify.rs` | `VerifyConfig` (+ `validate`), `VerifyReport` (+ `config`, `parts`), `verify_cpu_with_expander` |
| step 1 (tree) | `src/tree.rs`, `src/rules.rs`, `src/expansion.rs` | `dfs`, `frontier`, `Rules::improved(13)`, certificates, `DfsExpander` |
| step 2 (screens) | `src/lookahead.rs`, `src/rowsieve.rs` | `Lookahead` (shared arrays, pm_inf + mix(3^13) credits, class windows), `RowSieve::standard` (the (12, 9) rows) |
| step 3, CPU reference | `src/tail.rs`, `src/jump.rs`, `src/jdip.rs` | `TailParams` (+ `codes`, `schedule`), `endpoint_work`, `step_until_below`, `recheck_wide`, `tail_config_code`, `TailConfig::run_h`, `jump_dip_until_k`, `DipTable` (+ `quanta`, `quanta_cap`) |
| GPU host side | `src/gpu.rs` | `GpuSession` (`new`, `new_experimental`, `run_transport_tuned`, `prepare_jd_tables`), `gpu_run`, `check_gpu_config`, `upload_jd_ctx`, `verify_gpu`, `verify_gpu_tuned`, `ScreenSpec::of`, `build_all` |
| work packages | `src/units.rs` | `Plan`, `Package`, `plan` |
| CUDA host API | `cuda/kernel.cu`, `cuda/devtables.cuh` | `cv_gpu_init` / `cv_gpu_free`, `cv_gpu_launch_impl` (`_ex`, `_scr`), `cv_launch_tuned`, `cv_gpu_wait`, `cv_ctx_current`, `cv_jd_upload`, `cv_gpu_build_array`, `DevTables` |
| production kernels (round 3) | `cuda/twophase.cuh`, `cuda/screen.cuh`, `cuda/jdip.cuh`, `cuda/kernel.cu` | full range (A = 10): `cv_tp_kernel<91, 1, 3, 1>` (classes 0-3) and `cv_tp_kernel<91, 1, 5, 1>` (classes 4-15), rows on every class; top half (A = 11): `cv_tp_kernel<91, 3, 29, 1>` (classes 0-6) and `cv_tp_kernel<91, 3, 30, 1>` (classes 7-15); `cv_expand_parents`; `cv_build_kernel` (array build) |
| round-2 kernels (still built, SASS-identical to c-final) | `cuda/jdip.cuh`, `cuda/jdscreen.cuh`, `cuda/wq_jump.cuh`, `cuda/wq_jd.cuh` | `cv_kernel_jd<2,93>`, `cv_wqj2_kernel<JdWqTail3<91>,1,43>`, `cv_kernel_jds<93/90/111, 1..3>`, `cv_wqs_kernel<.., 43, ..>` (the round-2 configuration fxr2, still selectable); `cv_kernel<MODE,false,false,true,false,false>` (novote_oee, the original-loop control) |

### Experimental, benchmark-only, or not for production

* `scripts/expansion_bench/*`. `fullrange_bench` is a sampling benchmark that
  verifies sampled K38 cohorts. It is not a verification driver. Its transport
  is the measured one: the Affine terminal-table expander, K52 parent
  descriptors, and device expansion with `cv_expand_parents`. That code lives
  only in this bench crate (see section 4).
* `src/bin/*`: screenbench, jumpbench, wqaudit, jdaudit, jdsim, jumpaudit,
  jumpstats, wqxbits, flagcost and rowsieve_audit are benchmarks and audits.
* `GpuBench` (used by `cv measure` and the benches), plus
  `GpuSession::allow_measurement_only` / `GpuBench::allow_measurement_only`.
  These are explicit opt-ins for the measurement-only kernels: jd floor codes
  125-127, the screen diagnostics `_dload/_dscan/_dsetup` (SCR_LOAD / SCAN /
  SETUP), and the warp queue's capped test tail (pad bit 21). Production code
  never calls them. A report from such a run says
  `"verification_scope": "measurement_only_not_a_verification"`, and
  `cv verify` refuses one.
* Other experiments: the deferred and early CPU tails (`src/gpu_deferred.rs`,
  `src/gpu_early.rs`); compact chains and grouped tail tables
  (`cuda/compact_chain.cuh`, `cuda/grouped_tail.cuh`); warp-queue layouts
  other than 43 (52 is its per-case digest audit copy); and the xtail jump
  codes (`cuda/xtail.cuh`), which are round 1's jx family.
* Two-phase layouts other than 3, 5, 29 and 30. They compute the same work
  tuples, but they are c3-2ph's exploration layouts, and the pipeline did not
  measure them as production layouts.
  * Audit copies with the per-case digest: 12 (= 3), 24 (= 5), 14 (tiny list)
    and 31 (= 29).
  * Instrumented copies: 13, 15, 25-28.
  * Static rounds: 4, 17, 19, 27.
  * `src/bin/tpsim.rs` is c3-2ph's CPU schedule model.
* `cv measure` and `cv predict`. The cost model and `cv measure`'s CPU
  reference assume the original loop. `cv measure --gpu` times the original
  kernel, or a code given in `CV_XV`, which then fails its CPU comparison.
  Costs from `cv measure` therefore do not describe the final kernels.

## 2. Running a production verification

### Configuration (the measured final configuration)

Full range [0, 2^72), A = 10:

    CODES=91,91,91,91,91,91,91,91,91,91,91,91,91,91,91,91              # code 91 on every class (one per class)
    WQ=t3,t3,t3,t3,t5,t5,t5,t5,t5,t5,t5,t5,t5,t5,t5,t5                 # two-phase kernel (scheduling only)
    F24=28,35,35,35,35,34,33,33,27,27,27,27,27,27,27,27                # windows, 24 GB card (20.16 GiB of arrays)
    SHAPE="--n 72 --a 10 --b 27 --class-b $F24 --bv-shared --bv-pm-inf --bv-mix-j 13 \
           --rules improved --mix-j 13 --walkback-depth 38 --walkback-states 65536"
    SEM="--rows-from 0 --tail-codes $CODES --gpu-wq=$WQ --cutoff-bits 120"
    export CV_GPU_SLOTS=2          # 2 CUDA streams (measured: neutral to -0.3% with the two-phase kernel; free)
    unset CV_XTAIL_CPU CV_TAIL     # refused if set: the CPU semantics come from --tail-codes only

Top half [2^71, 2^72), A = 11. This is sound only together with a
verification of every n < 2^71 (Barina).

    W11=28,34,35,36,34,34,33,33,29,26,25,26,27,27,27,27                # windows at A = 11 (20.18 GiB)
    WQ11=t29,t29,t29,t29,t29,t29,t29,t30,t30,t30,t30,t30,t30,t30,t30,t30
    SHAPE11="--n 72 --a 11 --b 27 --class-b $W11 --bv-shared --bv-pm-inf --bv-mix-j 13 \
             --rules improved --mix-j 13 --walkback-depth 38 --walkback-states 65536"
    SEM11="--rows-from 0 --tail-codes $CODES --gpu-wq=$WQ11 --cutoff-bits 120 --range top"

How the pieces fit together:

* **Kernels.**
  * At A = 10, the two-phase layouts 3 and 5 screen the 32 completion words of
    an endpoint in one warp pass. They need A <= 10 and B <= 37, and refuse
    anything else.
  * At A = 11, the chunk layouts 29 and 30 screen each 32-word chunk of the
    endpoint's range: chunk 1 for the top half.
* **The A = 11 full range** (chunks 0 and 1, `--range full` at A = 11) is
  supported and validated, but it measured 1.033x the A = 10 full range, so
  the full range uses A = 10.
* **Staging.** Running the top half now and the bottom half later at A = 11
  (`--range bottom`, chunk 0) was not timed.
* **Recorded.** The scheduler specs have no semantic effect, so the work
  tuples equal those of any other scheduler on the same codes and screen. They
  are recorded in `config.warp_queue`, together with `rows_from` 0 and the
  codes.

The production sampler applies the bounded K38 walkback
(`--walkback-depth 38 --walkback-states 65536`). The tree applies the same
`walkback_certificate` at depth 38. The equivalence of the two in `cv verify`
was not measured here.

`--range bottom`, or an explicit `--h-range LO..HI` (multiples of 64), covers
other completion ranges.

The round-2 configuration (`fxr2`) remains available and SASS-identical. Use
codes `93,91,93,90,90,90,90,111x9`, `--gpu-wq=-,v43.k8,-x14` and
`--rows-from 2`.

### Commands

    cv plan $SHAPE --costs COSTS.json --target-hours 1 --out plan72.json
    cv verify $SHAPE $SEM --gpu --plan plan72.json --package I..J --slices 64 --json pkg_I_J.json

* `--package I..J` runs packages I to J-1 in one process, building the arrays
  once (about 77 s at F24). Each package is run separately and split into
  `--slices` contiguous slices of its node list. The report holds the
  combined totals and a `parts` entry per (package, slice).
* `cv verify` refuses a package run whose parity rules differ from the
  plan's. `cv plan` splits the tree with the parity part of its `--rules`
  (pm_inf, mix and walkback stripped) and records that label. The run's own
  value rules then apply below the plan's nodes, and the report records
  both.
* It exits non-zero on any refused configuration, on unresolved flags (a
  256-bit recheck failure), and on a measurement-only run. The GPU path also
  fails loudly when a launch would hold more than 65,536 flags; the
  messages say to raise `--cutoff-bits`.

### What the report records (`--json`, schema 2)

Top level (unchanged fields): `n a b num_bv bv_mode class_b window rules
backend cutoff_bits tail_oee units tree_nodes endpoints cases iters hash
flagged flagged_rechecked seconds tree tail gpu`. New and changed fields:

* `verification_scope`: `full_range`, `top_half`, `bottom_half`,
  `h_range[lo,hi)`, `selected_work_units[+<range>]`, or
  `measurement_only_not_a_verification`. Round 2 always reported
  `full_range`.
* `measurement_only`: false for every run `cv verify` accepts.
* `config` (`runspec::RunRecord`). The CPU and GPU paths build it identically
  from the same `VerifyConfig`; the GPU test `session_refuses_...` checks
  equality. It holds:
  * `schema` and the shape: `n`, `a`, `m`, `rules`;
  * `h_range` {`kind`, `h_lo`, `h_hi`, `completions`, `covers`}, where
    `covers` is a sentence stating what a successful run establishes;
  * `units` {`selection` all|selected, `count`, `digest` (FNV-1a 64 of the
    units' (n0, k)), `plan` {`file`, `file_digest`, `packages`, `plan_n`,
    `plan_a`, `plan_b`, `plan_rules`, `slices`, `slice`}};
  * `screen` {`rows`, `rows_from`, `label`};
  * `bitvector` {`mode`, `b`, `num_bv`, `pm_inf`, `mix_j`, `label`,
    `class_windows`};
  * `tail_codes` (one per class) and `cpu_only_classes_code` (0, the
    original loop);
  * `warp_queue` (one spec per class, `-` = per-thread);
  * `cutoff_bits` and `tail_oee`;
  * `work_definition` {`version`: `cvwork-2`, `cases`, `codes`: for each
    code used, its classes, `tag` (for example `jd-W11-WD11-Q32-KA0`),
    `iterations`, `hash`, `stop` and `flag` definitions};
  * `measurement_only`.
* `parts` (plan runs). For every (package, slice): `units`, `units_digest`,
  `tree_nodes`, `endpoints`, `cases`, `iters`, `hash`, `flagged`, and
  `flagged_digest` (FNV-1a 64 of the sorted flagged starts).
* `cv plan` writes `schema_version: 2` plus the cost layout (`bv_mode`,
  `class_windows`, `window`). Old plans still load.
* `cv export-paths` copies `verification_scope`, `measurement_only` and
  `config` into the artifact.

## 3. Spot-checking a finished package on the CPU

A GPU package report fixes everything needed to re-run any part of it: the
configuration (`config`), the plan file (`config.units.plan.file_digest`),
the package and slice layout, and each part's work tuple (`parts`). To check
part (package P, slice K of S) on the CPU, run the same command without
`--gpu`, restricted to that part:

    cv verify $SHAPE $SEM --plan plan72.json --package P --slices S --slice K --json spot.json

Then compare:

* `spot.config` must equal `pkg.config`, except `config.units`, which names
  the part (its `digest` must equal the part's `units_digest`);
* `spot.parts[0]` must equal the matching entry of `pkg.parts`: tree nodes,
  endpoints, cases, iterations, XOR hash, flag count and `flagged_digest`.

`$F/scripts/spotcheck.py PKG.json SPOT.json` (c3-hyg's tool, copied unchanged) does the comparison. The CPU computes
the same per-case semantics from the same `TailParams` (codes, screen, h
range), so any difference is a fault in one of the two runs.

Cost: a whole 1-GPU-hour package is far too expensive on the CPU. The N40
CPU run here did 2.0e7 cases per core-second, tree included (measured-cpu).
The round-3 final GPU configuration does about 7.7e10 cases per second
(c3-final's confirmation: 1.78e11 cases per 2.32 s cohort; round 2: 6.7e10).
One GPU-hour is therefore roughly 3,500-5,500 CPU core-hours (modeled; N72
cases take about 30% more jumps than N40 cases). Spot-check
slices instead: with `--slices 64`, one slice of a 1-GPU-hour package is
roughly 50-80 core-hours, or 3-5 hours on 16 cores. A repeat run of the same package on a second GPU host is the cheap
complement: the tuples must match bit for bit. It does not catch a
deterministic GPU implementation error, which the CPU spot check does.

## 4. Limits, and what is missing for a production run

### Fixed limits (enforced by the code)

* **cutoff >= 96 + K.** That is 103 for W11 codes and 104 for W12 codes, so
  this mix needs `--cutoff-bits >= 104`. Both paths refuse less
  (`TailCodes::check`, with the GPU launch guard as a backstop). Every cutoff
  must be <= 123.
* **Path-record runs at cutoff 97 must keep the original kernel.** Run them
  without `--tail-codes`; the jd codes refuse cutoff 97. A jd run's flags are
  the cases with x >= 2^(cutoff - K) at an executed jump boundary before the
  stop. Values inside a certified lookahead span are never examined, so a jd
  run establishes convergence, not path-record completeness.
* **Iterations and hash are redefined.** Iterations count W-step jumps; the
  hash XORs x mod 2^32 after every jump; jd stops are window-dip
  certificates. The case set depends on the windows, the bitvector rules, the
  row screen and the h range. Reports carry `work_definition.version =
  cvwork-2` and the per-code definitions. Work tuples from runs with other
  codes, windows or screens are not comparable, and neither are
  FINDINGS/COST_MODEL numbers, which use the original loop.
* **Other limits:**
  * M = N - A <= 80; 6 <= A <= B <= 39.
  * Top and bottom halves need A >= 7. Rows need M >= 12 and A <= 16.
  * Per-class codes cover at most 32 classes.
  * Warp-queue specs only on classes with codes 49..53 or 91/92/94/95; warp
    queue words32 <= 2048.
* **Two-phase specs** (`t<layout>`) only on classes with the jd FD3 codes 90,
  91, 92, 93, 95 or 111.
  * The batched layouts 3 and 5 need A <= 10 and B <= 37.
  * The chunk layouts 29-31 need A <= 13 and B <= 37.
  * A configuration a layout cannot run is refused at launch, and the run
    fails.

### Missing for a production N72 run (not addressed here)

1. **Transport.** `cv verify` expands every plan node to depth 62 on the CPU
   (`DfsExpander`). On the GPU it sends Affine32 records for screened runs,
   which the production configuration always is. The measured throughput
   (FINAL3.md: about 0.156 GPU-years for the full range) assumes
   fullrange_bench's transport: the Affine
   terminal-table expander, K52 parents, and device expansion
   (`GpuTransport::Parents`, `cv_expand_parents`). That code lives in
   `scripts/expansion_bench/{affine,terminal,ternary}.rs`, not in the library.
   With the DFS expander the step-1 CPU work per GPU is much higher. The
   original README estimates 5.6 cores per GPU at the original kernel's
   0.85 GPU-years, so roughly 30 cores per GPU at the round-3 final's speed
   (modeled).
   A production driver should run plan parts through
   `GpuSession::run_transport_tuned(cfg, &affine, GpuTransport::Parents,
   words, tuning)`, which is what fullrange_bench does. Its CPU spot check
   must then use the same expander (`verify_cpu_with_expander(cfg, &affine)`),
   because the Affine expander applies the non-parity rules only up to its
   cutoff (K50). Its work tuple is a different, equally sound, definition from
   `DfsExpander`'s.
2. **Frontier.** The pipeline was measured from K38 roots. Splitting the tree
   to K38 for a full run, and storing that frontier, is not implemented.
3. **Costs for planning.** `cv plan` balances packages by `cv measure` costs,
   which describe the original kernel. Per-class costs of the final kernels
   exist only as screenbench output, in ns per endpoint: the two-phase
   window sweeps `c3-final/runs/ws0-4.jsonl` (A = 10) and `ws11_*.jsonl`
   (A = 11). They would need converting to the `cv measure` JSON format.
   Without that, package GPU times are mispredicted by about 3x (balance
   only; coverage is unaffected).
4. **Aggregation.** No tool sums package reports into a run total or checks
   that every package of a plan was run exactly once. Plan, package ids, unit
   digests and per-part tuples are now recorded, so this is a straightforward
   script.
5. **Multi-GPU in one process.** It is not supported: one process per GPU,
   with `CUDA_VISIBLE_DEVICES`. A context used from another device, or after
   a device reset, is refused.
