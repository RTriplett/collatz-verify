#!/usr/bin/env python3
"""Audit rare F56..62 CPU-tail samples and estimate a separate CPU allowance.

This conditional baseline sample applies no improved/checkpoint screening. Its
timings are scalar CPU timings, not GPU performance or a correction factor for
the independently measured main GPU pipeline.
"""

import argparse
from collections import Counter
from functools import reduce
import json
import math
import operator
from pathlib import Path
import re
import statistics

from summarize_fullrange import YEAR_SECONDS, estimate
from summarize_overall import digest, require

POPULATION = 49_063_601
DESIGN = "fixed_uniform_baseline_endpoints_with_replacement"
TIMINGS = ("wall_seconds", "process_cpu_seconds", "tail_wall_seconds",
           "tail_process_cpu_seconds", "wide_recheck_wall_seconds",
           "wide_recheck_process_cpu_seconds")


def nonnegative(value, name):
    require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
            f"invalid {name}")


def audit_work(work, roots, config):
    classes = list(range(config["f_min"], config["f_max"] + 1))
    require([row["f"] for row in work["by_f"]] == classes, "missing, duplicate or unordered F classes")
    for key in ("endpoints", "cases", "iters"):
        require(isinstance(work[key], int) and work[key] >= 0, f"invalid total {key}")
        require(all(isinstance(row[key], int) and row[key] >= 0 for row in work["by_f"]),
                f"invalid per-F {key}")
        require(sum(row[key] for row in work["by_f"]) == work[key], f"per-F/total {key} differ")
    require(work["endpoints"] == len(roots), "work endpoint count differs from saved draws")
    by_f = Counter(root["f"] for root in roots)
    require(all(row["endpoints"] == by_f[row["f"]] for row in work["by_f"]),
            "per-F endpoint counts differ from roots")
    require(isinstance(work["hash"], str) and re.fullmatch(r"[0-9A-F]{8}", work["hash"]) is not None,
            "malformed 32-bit checksum")
    flags = [int(origin) for origin in work["flagged"]]
    require(all(0 < origin < 1 << config["n"] for origin in flags), "flag origin outside target range")
    require(work["flagged_rechecked"] == len(flags) and work["unresolved_flags"] == 0,
            "flagged trajectories have unresolved rechecks")
    require(len(flags) <= work["cases"], "more flags than cases")
    require(isinstance(work["wide_recheck_steps"], int) and work["wide_recheck_steps"] >= 0,
            "invalid wide recheck step count")
    require(not flags or work["wide_recheck_steps"] > 0, "flags have no wide recheck steps")
    root_counts = Counter(int(root["n0"]) for root in roots)
    mask = (1 << config["m"]) - 1
    require(all(multiplicity <= root_counts[origin & mask]
                for origin, multiplicity in Counter(flags).items()),
            "flag origin/multiplicity is not represented by sampled endpoint roots")
    return {**work, "flagged": sorted(work["flagged"], key=int)}


def audit_timing(timing):
    for key in TIMINGS:
        nonnegative(timing[key], key)
    require(timing["wall_seconds"] > 0 and timing["process_cpu_seconds"] > 0,
            "total CPU/wall timing must be positive")
    require(timing["tail_wall_seconds"] + timing["wide_recheck_wall_seconds"]
            <= timing["wall_seconds"] + 1e-9, "component wall times exceed total")
    require(timing["tail_process_cpu_seconds"] + timing["wide_recheck_process_cpu_seconds"]
            <= timing["process_cpu_seconds"] + 1e-9, "component CPU times exceed total")
    require(math.isclose(timing["average_busy_cpu_cores"],
                         timing["process_cpu_seconds"] / timing["wall_seconds"], rel_tol=1e-9),
            "busy CPU core accounting differs")


def summarize(path):
    metadata = json.loads((path / "metadata.json").read_text())
    require(metadata.get("state") == "complete" and metadata.get("exit_status") == 0,
            f"{path}: incomplete or failed run")
    rows = [json.loads(line) for line in (path / "pipeline.jsonl").read_text().splitlines() if line.strip()]
    require(all(row.get("type") in ("config", "cohort", "warmup", "trial") for row in rows),
            "unknown measurement row")
    configs = [row for row in rows if row["type"] == "config"]
    require(len(configs) == 1, "missing or duplicate configuration")
    cfg = configs[0]
    require(cfg["experiment"] == "rare_cpu_tail" and cfg["sampling_design"] == DESIGN,
            "wrong experiment or sampling design")
    require((cfg["n"], cfg["a"], cfg["m"], cfg["f_min"], cfg["f_max"]) == (72, 10, 62, 56, 62),
            "unsupported target population")
    require(int(cfg["baseline_population"]) == POPULATION and cfg["baseline_population_forward_backward_equal"] is True,
            "exact population disagrees with the recorded DP check")
    populations = cfg["baseline_population_by_f"]
    require([row["f"] for row in populations] == list(range(56, 63)), "invalid per-F populations")
    require(all(int(row["endpoints"]) > 0 for row in populations)
            and sum(int(row["endpoints"]) for row in populations) == POPULATION,
            "per-F populations do not sum to exact conditional population")
    require(cfg["threads"] == 1 and cfg["num_bv"] == 16
            and cfg["lookahead_arrays_built"] is False and cfg["gpu_initialized"] is False,
            "probe did not use the expected scalar fallback path")
    count, draws, reps = cfg["cohort_count"], cfg["draws_per_cohort"], cfg["repetitions"]
    require(isinstance(count, int) and count >= 2 and isinstance(draws, int) and draws > 0
            and isinstance(reps, int) and reps > 0, "invalid fixed sample design")
    args = metadata.get("arguments", {})
    for field, target in (("draws", "draws_per_cohort"), ("cohorts", "cohort_count"),
                          ("repetitions", "repetitions"), ("seed", "seed")):
        if field in args:
            require(args[field] == cfg[target], f"metadata/config {field} differ")
    cohorts = [row for row in rows if row["type"] == "cohort"]
    require(sorted(row["cohort"] for row in cohorts) == list(range(count)), "missing or duplicate cohorts")
    cohorts.sort(key=lambda row: row["cohort"])
    require(len({row["seed"] for row in cohorts}) == count, "duplicate cohort seeds")
    for row in cohorts:
        roots = row["roots"]
        require(row["drawn"] == row["accepted"] == len(roots) == draws,
                "conditional baseline draws were rejected or omitted")
        require(all(root["k"] == cfg["m"] and cfg["f_min"] <= root["f"] <= cfg["f_max"]
                    and 0 < int(root["n0"]) < 1 << cfg["m"] for root in roots),
                "saved root outside the conditional endpoint population")
        require(row["duplicates_retained"] == draws - len({int(root["n0"]) for root in roots}),
                "retained-duplicate accounting differs")
    warmups = [row for row in rows if row["type"] == "warmup"]
    require(len(warmups) == 1, "missing or duplicate warmup")
    audit_timing(warmups[0]["timing"])
    audit_work(warmups[0]["work"], cohorts[0]["roots"][:min(draws, 64)], cfg)
    trials = [row for row in rows if row["type"] == "trial"]
    require(len(trials) == count * reps, "trial count differs from fixed design")
    by_trial, signatures = {}, {}
    for row in trials:
        c, rep = row["cohort"], row["repetition"]
        require(c in range(count) and rep in range(reps) and (c, rep) not in by_trial,
                "unconfigured or duplicate trial")
        require(row["drawn"] == draws, "trial draw denominator differs")
        require(row["order_position"] in range(count), "invalid trial order position")
        audit_timing(row["timing"])
        work = audit_work(row["work"], cohorts[c]["roots"], cfg)
        require(c not in signatures or signatures[c] == work, "cohort work changes between repetitions")
        signatures[c] = work
        by_trial[c, rep] = row
    require(set(by_trial) == {(c, rep) for c in range(count) for rep in range(reps)}, "missing trial")
    for rep in range(reps):
        require({by_trial[c, rep]["order_position"] for c in range(count)} == set(range(count)),
                "trial order positions do not cover each cohort once")
    per_cohort = {key: [statistics.mean(by_trial[c, rep]["timing"][key] for rep in range(reps)) / draws
                        for c in range(count)] for key in TIMINGS}
    seconds = {key: estimate(values, POPULATION) for key, values in per_cohort.items()}
    years = {key: estimate(values, POPULATION / YEAR_SECONDS) for key, values in per_cohort.items()}
    works = [signatures[c] for c in range(count)]
    work_fields = ("endpoints", "cases", "iters", "flagged_rechecked", "wide_recheck_steps")
    work_estimates = {key: estimate([work[key] / draws for work in works], POPULATION)
                      for key in work_fields}
    per_f = [{"f": f, "exact_baseline_endpoints": populations[i]["endpoints"],
              **{key: estimate([work["by_f"][i][key] / draws for work in works], POPULATION)
                 for key in ("endpoints", "cases", "iters")}}
             for i, f in enumerate(range(56, 63))]
    sample_counts = {key: sum(work[key] for work in works) for key in work_fields}
    sample_counts["flagged_trajectories"] = sum(len(work["flagged"]) for work in works)
    sample_counts["distinct_flagged_origins"] = len({int(origin) for work in works for origin in work["flagged"]})
    sample_counts["checksum_xor"] = f'{reduce(operator.xor, (int(work["hash"], 16) for work in works), 0):08X}'
    all_roots = [root for cohort in cohorts for root in cohort["roots"]]
    return {"directory": str(path), "config": cfg, "runner": metadata,
            "ordered_corpus_sha256": digest(all_roots),
            "cohort_corpus_sha256": [digest(cohort["roots"]) for cohort in cohorts],
            "cohort_work_signature_sha256": [digest(work) for work in works],
            "sampling_records": [{key: value for key, value in cohort.items() if key != "roots"} for cohort in cohorts],
            "one_pass_sample_work": sample_counts, "full_baseline_work_estimates": work_estimates,
            "full_baseline_work_by_f": per_f,
            "full_baseline_seconds": seconds, "full_baseline_years": years,
            "mean_busy_cpu_cores": statistics.mean(per_cohort["process_cpu_seconds"]) / statistics.mean(per_cohort["wall_seconds"]),
            "cohort_mean_timings": {key: [value * draws for value in values] for key, values in per_cohort.items()},
            "hypothetical_serial_gpu_allocation_addition_years": {
                **years["wall_seconds"],
                "scenario": "A pessimistic no-overlap accounting scenario in which one GPU is allocated but idle for the entire measured CPU-tail allowance; not measured GPU work, not an automatic correction to the main estimate.",
            }}


def markdown(result):
    run = result["run"]
    cfg = run["config"]
    lines = ["# Separate allowance for rare CPU tails", "", result["scope"], "", result["method"], "", result["limitations"], "",
             f'Exact conditional baseline population: {POPULATION:,} M62 endpoints with F56–62. '
             f'{cfg["cohort_count"]} cohorts × {cfg["draws_per_cohort"]} draws; '
             f'{cfg["repetitions"]} repetitions per cohort. Every draw is retained, including duplicates.', "",
             '| CPU stage | Full-population wall seconds [95% interval] | CPU core-years [95% interval] |',
             '|---|---:|---:|']
    for label, wall, cpu in (("Total tail and wide rechecks", "wall_seconds", "process_cpu_seconds"),
                             ("Scalar tail", "tail_wall_seconds", "tail_process_cpu_seconds"),
                             ("Wide rechecks", "wide_recheck_wall_seconds", "wide_recheck_process_cpu_seconds")):
        w, c = run["full_baseline_seconds"][wall], run["full_baseline_years"][cpu]
        lines.append(f'| {label} | {w["estimate"]:.3f} [{w["ci95"][0]:.3f}, {w["ci95"][1]:.3f}] | '
                     f'{c["estimate"]:.7f} [{c["ci95"][0]:.7f}, {c["ci95"][1]:.7f}] |')
    sample = run["one_pass_sample_work"]
    lines += ["", f'One pass through the sampled endpoints contains {sample["cases"]:,} cases and '
              f'{sample["iters"]:,} tail iterations. All {sample["flagged_trajectories"]:,} flagged trajectories '
              f'({sample["distinct_flagged_origins"]:,} distinct origins) finish wide rechecking, totaling '
              f'{sample["wide_recheck_steps"]:,} wide steps. Repetitions agree on the full work signature; '
              f'the one-pass XOR checksum is `{sample["checksum_xor"]}`.', ""]
    scenario = run["hypothetical_serial_gpu_allocation_addition_years"]
    lines += [f'If a GPU remained allocated and completely idle throughout this CPU work, the '
              f'**hypothetical no-overlap allocation addition** would be {scenario["estimate"]:.7f} GPU-years '
              f'[{scenario["ci95"][0]:.7f}, {scenario["ci95"][1]:.7f}]. This is a pessimistic accounting '
              'scenario, not GPU computation measured by this probe. No allowance is silently added to the main pipeline estimate.', "",
              f'Ordered corpus SHA-256: `{run["ordered_corpus_sha256"]}`. '
              f'Setup takes {cfg["setup_seconds"]:.6f} seconds and is outside the timed samples.', ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    try:
        result = {
            "scope": "A separately measured scalar CPU workload allowance for all baseline N72/A10 endpoints in F56–62, without improved ancestry or K38 walkback screening. It is not a GPU performance estimate or a correction factor for the main checkpoint sample.",
            "method": "Fixed conditional baseline endpoint draws with replacement. Average repetitions within each independent cohort, then multiply the mean per-draw cost by the exact population 49,063,601. Approximate Student-t95 intervals use cohort degrees of freedom; one year is 365.25 days. The total includes wide rechecking, with tail and recheck timings also reported separately.",
            "limitations": "Omitting improved screening makes the selected workload a conservative superset of the experimental pipeline's rare endpoints; the sampled runtime estimate is not a rigorous upper bound. It excludes endpoint generation, checkpoint preprocessing, storage, scheduling and GPU work. Rare/clustered flags and high-F strata may be sampled imprecisely; cohort intervals do not include systematic host performance differences or long-term load. CPU/GPU overlap is not measured here.",
            "run": summarize(args.run),
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.out.with_suffix(".md").write_text(markdown(result))
    print(f"Saved {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
