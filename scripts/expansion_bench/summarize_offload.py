#!/usr/bin/env python3
"""Audit CPU tail-offload experiments and summarize directly measured costs.

This uses the full-range runner's strict configuration, exact-work, queue, and
paired-cohort audits, but deliberately reports only measured comparative costs.
It does not promote small offload pilots to new full-range GPU-year estimates.
"""

import argparse
import json
from pathlib import Path
import statistics

from summarize_fullrange import paired_ratio, summarize


COUNTERS = ("deferred_attempts", "deferred_cases", "deferred_rejected",
            "deferred_iterations", "deferred_flags", "deferred_transfer_bytes")
TIMINGS = ("deferred_cpu_seconds", "deferred_queue_wait_seconds", "deferred_drain_seconds")


def divide(numerator, denominator):
    return numerator / denominator if denominator else None


def costs(rows):
    """Average repeated measurements of one fixed corpus, never add repeats."""
    result = {key: statistics.mean(row["report"]["gpu"].get(key, 0) for row in rows)
              for key in COUNTERS + TIMINGS}
    result.update({key: statistics.mean(row[key] for row in rows)
                   for key in ("wall_seconds", "process_cpu_seconds")})
    for key in ("cases", "iters", "endpoints"):
        result["gpu_origin_" + key] = statistics.mean(
            sum(item[key] for item in row["report"]["gpu"]["classes"]) for row in rows)
    result["input_transfer_bytes"] = statistics.mean(
        row["report"]["gpu"]["transfer_bytes"] for row in rows)
    return result


def derived(values):
    return {**values,
            "average_busy_cpu_cores": divide(values["process_cpu_seconds"], values["wall_seconds"]),
            "threshold_exceeding_case_fraction": divide(values["deferred_attempts"], values["gpu_origin_cases"]),
            "offloaded_case_fraction": divide(values["deferred_cases"], values["gpu_origin_cases"]),
            "cpu_remainder_iteration_fraction": divide(values["deferred_iterations"], values["gpu_origin_iters"]),
            "queue_rejection_fraction": divide(values["deferred_rejected"], values["deferred_attempts"]),
            "mean_remainder_iterations_per_offloaded_case": divide(values["deferred_iterations"], values["deferred_cases"]),
            "cpu_nanoseconds_per_remainder_iteration": divide(1e9 * values["deferred_cpu_seconds"], values["deferred_iterations"]),
            "gpu_executed_iterations": values["gpu_origin_iters"] - values["deferred_iterations"]}


def summarize_offload(path):
    audited = summarize(path)
    rows = [json.loads(line) for line in (path / "pipeline.jsonl").read_text().splitlines() if line.strip()]
    cfg = audited["config"]
    variants = []
    for variant in audited["variants"]:
        name = variant["name"]
        selected = [row for row in rows if row.get("variant") == name]
        cohorts = [costs([row for row in selected if row.get("type") == "trial"
                          and row["cohort"] == cohort]) for cohort in range(cfg["cohort_count"])]
        combined = {key: sum(cohort[key] for cohort in cohorts) for key in cohorts[0]}
        aggregate_rows = [row for row in selected if row.get("type") == "aggregate_trial"]
        variants.append({"name": name, "transport": variant["gpu_transport"],
                         "tuning": variant["gpu_tuning"],
                         "comparison_baseline": variant["comparison_baseline"],
                         "paired_wall_ratio": variant["paired_wall_ratio"],
                         "paired_cpu_ratio": variant["paired_cpu_ratio"],
                         "one_pass_cohort_totals": derived(combined),
                         "cohorts": [derived(cohort) for cohort in cohorts],
                         "continuous_aggregate": derived(costs(aggregate_rows)) if aggregate_rows else None})
    by_name = {variant["name"]: variant for variant in variants}
    for variant in variants:
        controls = {}
        ordinary = "compact66" if variant["transport"] == "compact_chain" else "novote_oee"
        for role, control in (("ordinary_transport", ordinary),
                              ("budget_instrumentation", "defer_count")):
            if control in by_name and by_name[control]["transport"] == variant["transport"]:
                controls[role] = {"variant": control,
                                  **paired_ratio([cohort["wall_seconds"] for cohort in variant["cohorts"]],
                                                 [cohort["wall_seconds"] for cohort in by_name[control]["cohorts"]])}
        variant["paired_wall_controls"] = controls
    return {"directory": str(path), "baseline_draws": cfg["draws_per_cohort"] * cfg["cohort_count"],
            "accepted_draws": audited["sampling"]["accepted"], "cohort_count": cfg["cohort_count"],
            "repetitions": cfg["repetitions"], "n": cfg["n"], "a": cfg["a"],
            "threads": cfg["threads"], "aggregate_corpus_sha256": audited["aggregate_corpus_sha256"],
            "exact_work_audit_passed": True, "variants": variants}


def markdown(result):
    lines = ["# CPU tail-offload experiment", "", result["scope"], "", result["accounting"], ""]
    for run in result["runs"]:
        lines.extend([f'## {run["directory"]}', "",
                      f'{run["cohort_count"]} independent cohorts, {run["baseline_draws"]:,} baseline draws '
                      f'({run["accepted_draws"]:,} accepted), {run["repetitions"]} timing repetitions. '
                      'All work signatures and accounting checks passed.', "",
                      '| Variant | Wall ratio [95% paired interval] | Wall s | CPU core-s | Offloaded cases | CPU remainder iterations | Queue rejected |',
                      '|---|---:|---:|---:|---:|---:|---:|'])
        for variant in run["variants"]:
            row, ratio = variant["one_pass_cohort_totals"], variant["paired_wall_ratio"]
            lines.append(f'| {variant["name"]} | {ratio["ratio"]:.4f} '
                         f'[{ratio["ci95"][0]:.4f}, {ratio["ci95"][1]:.4f}] | '
                         f'{row["wall_seconds"]:.3f} | {row["process_cpu_seconds"]:.3f} | '
                         f'{row["deferred_cases"]:,.0f} | {row["deferred_iterations"]:,.0f} | '
                         f'{row["deferred_rejected"]:,.0f} |')
        lines.extend(["", 'Rows sum cohort means, retaining each original draw once. '
                      'Detailed JSON includes continuation CPU time, queue wait, final drain, '
                      'threshold exceedance, and continuous aggregate measurements.', ""])
        if any(variant["continuous_aggregate"] for variant in run["variants"]):
            lines.extend(['Continuous aggregate: the same accepted roots concatenated; '
                          'this measures pipeline drain overhead and is not an independent sample.', "",
                          '| Variant | Wall s | CPU core-s | CPU continuation s | Queue wait s | Final drain s |',
                          '|---|---:|---:|---:|---:|---:|'])
            for variant in run["variants"]:
                row = variant["continuous_aggregate"]
                if row:
                    lines.append(f'| {variant["name"]} | {row["wall_seconds"]:.3f} | '
                                 f'{row["process_cpu_seconds"]:.3f} | {row["deferred_cpu_seconds"]:.3f} | '
                                 f'{row["deferred_queue_wait_seconds"]:.3f} | {row["deferred_drain_seconds"]:.3f} |')
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        if len({path.resolve() for path in args.runs}) != len(args.runs):
            raise ValueError("duplicate run directory")
        result = {
            "scope": "Directly measured CPU/GPU continuation experiments; no new absolute full-range GPU-year estimate.",
            "accounting": "GPU-origin cases, iterations, flags and checksums include exact CPU continuations. Deferred work is a subset of these totals. Budgets and iteration counts use the existing double-iteration loop, not individual Collatz steps. A full queue leaves the candidate on the GPU. GPU time comparisons use outer pipeline wall time, including queue backpressure, CPU continuation drain and wide rechecks. Timing repetitions are averaged within each independent cohort before paired inference.",
            "runs": [summarize_offload(path) for path in args.runs],
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.out.with_suffix(".md").write_text(markdown(result))
    print(f"Saved {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
