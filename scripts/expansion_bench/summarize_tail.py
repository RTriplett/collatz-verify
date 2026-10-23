#!/usr/bin/env python3
"""Summarize completed checkpoint-tail CPU and paired-work runs.

Usage: python3 scripts/expansion_bench/summarize_tail.py [RESULTS] [--out FILE]
Incomplete or failed runs remain visible with their status, but contribute no
performance estimates. Only explicitly equivalent policies are compared for
equal output; disabling checks intentionally changes the endpoint set.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[2]
METRICS = ("endpoints", "cases", "iters", "model_seconds")


def load_run(path, directory):
    """A growing last JSONL line is an incomplete run, not a script failure."""
    result = {"file": str(path.relative_to(directory))}
    problems = []
    rows = []
    try:
        text = path.read_text()
    except OSError as error:
        return result, rows, [f"cannot read results: {error}"]
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("record is not a JSON object")
            rows.append(row)
        except (ValueError, json.JSONDecodeError) as error:
            problems.append(f"unreadable JSON record at line {number}: {error}")
            break

    metadata_path = path.parent / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            result["runner"] = {
                key: metadata.get(key)
                for key in ("state", "exit_status", "started_at", "finished_at")
            }
            if metadata.get("state") != "complete" or metadata.get("exit_status") != 0:
                problems.append(
                    "runner has not completed successfully: "
                    f"state={metadata.get('state')!r}, exit_status={metadata.get('exit_status')!r}"
                )
        except (OSError, ValueError) as error:
            problems.append(f"cannot read runner metadata: {error}")
    return result, rows, problems


def skipped(result, problems):
    state = result.get("runner", {}).get("state")
    result.update({
        "status": state if state in ("failed", "interrupted") else "incomplete",
        "skipped": True,
        "reasons": problems,
    })
    return result


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def equivalence_groups(summaries):
    groups = [
        ("full", "screened"),
        ("baseline", "fast_baseline"),
        ("pm_only", "pm_screened"),
    ]
    groups.extend(
        (name.removeprefix("fast_"), name)
        for name in sorted(summaries)
        if name.startswith("fast_cutoff")
    )
    comparisons = []
    for group in groups:
        available = [name for name in group if name in summaries]
        if not available:
            continue
        comparison = {"policies": list(group), "available": available}
        if len(available) < 2:
            comparison["status"] = "not_comparable"
        else:
            first, second = (summaries[name] for name in available)
            comparison.update({
                "status": "compared",
                "digest_equal": first["digest"] == second["digest"],
                "tree_stats_equal": first["tree_stats"] == second["tree_stats"],
            })
            comparison["passed"] = (
                comparison["digest_equal"] and comparison["tree_stats_equal"]
            )
        comparisons.append(comparison)
    return comparisons


def summarize_cpu(path, directory):
    result, rows, problems = load_run(path, directory)
    configs = [row for row in rows if row.get("type") == "config"]
    if len(configs) != 1:
        problems.append(f"expected one config record, found {len(configs)}")
        return skipped(result, problems)
    config = configs[0]
    result["config"] = config
    names = config["variants"]
    reps = set(range(config["repetitions"]))
    trials = {name: {} for name in names}
    summaries = {}
    for row in rows:
        if row.get("type") == "trial":
            name, repetition = row["variant"], row["repetition"]
            if name not in trials:
                problems.append(f"trial for unconfigured variant {name}")
                continue
            if repetition in trials[name]:
                problems.append(f"duplicate trial {name}, repetition {repetition}")
            trials[name][repetition] = row
        elif row.get("type") == "summary":
            name = row["variant"]
            if name in summaries:
                problems.append(f"duplicate summary for {name}")
            summaries[name] = row
    if set(summaries) != set(names):
        problems.append(
            f"summaries missing={sorted(set(names) - set(summaries))}, "
            f"unexpected={sorted(set(summaries) - set(names))}"
        )
    for name in names:
        if set(trials[name]) != reps:
            problems.append(
                f"{name}: repetitions missing={sorted(reps - set(trials[name]))}, "
                f"unexpected={sorted(set(trials[name]) - reps)}"
            )
    validations = [row for row in rows if row.get("type") == "validation"]
    if not validations or not all(row.get("passed") is True for row in validations):
        problems.append("missing or unsuccessful endpoint audit")
    if problems:
        result["trial_counts"] = {name: len(values) for name, values in trials.items()}
        return skipped(result, problems)

    reference = "full" if "full" in summaries else "screened" if "screened" in summaries else None
    full = summaries.get("full")
    screened = summaries.get("screened")
    variants = []
    for name in names:
        summary = summaries[name]
        seconds = summary["median_seconds"]
        endpoints = summary["digest"]["endpoints"]
        generated = sum(summary["tree_stats"]["generated"])
        variant = {
            "variant": name,
            "median_seconds": seconds,
            "minimum_seconds": min(row["seconds"] for row in trials[name].values()),
            "maximum_seconds": max(row["seconds"] for row in trials[name].values()),
            "endpoints": endpoints,
            "endpoints_by_f": summary["digest"].get("by_f"),
            "endpoint_ratio_reference": reference,
            "endpoint_ratio": safe_ratio(endpoints, summaries[reference]["digest"]["endpoints"])
                if reference else None,
            "cpu_ratio_vs_full": safe_ratio(seconds, full["median_seconds"]) if full else None,
            "cpu_ratio_vs_screened": safe_ratio(seconds, screened["median_seconds"])
                if screened else None,
            "generated_nodes": generated,
            "generated_node_ratio_vs_full": safe_ratio(generated, sum(full["tree_stats"]["generated"]))
                if full else None,
        }
        if screened:
            paired = [
                trials[name][rep]["seconds"] / trials["screened"][rep]["seconds"]
                for rep in sorted(reps)
            ]
            variant.update({
                "paired_median_cpu_ratio_vs_screened": statistics.median(paired),
                "paired_cpu_ratio_range_vs_screened": [min(paired), max(paired)],
                "wins_vs_screened": sum(value < 1.0 for value in paired),
                "ties_vs_screened": sum(value == 1.0 for value in paired),
                "paired_repetitions": len(paired),
            })
        else:
            variant.update({
                "paired_median_cpu_ratio_vs_screened": None,
                "paired_cpu_ratio_range_vs_screened": None,
                "wins_vs_screened": None,
                "ties_vs_screened": None,
                "paired_repetitions": 0,
            })
        variants.append(variant)
    result.update({
        "status": "complete",
        "skipped": False,
        "variants": variants,
        "exact_equivalence_groups": equivalence_groups(summaries),
        "validation": validations,
    })
    return result


def summarize_work(path, directory):
    result, rows, problems = load_run(path, directory)
    configs = [row for row in rows if row.get("kind") == "config"]
    if len(configs) != 1:
        problems.append(f"expected one config record, found {len(configs)}")
        return skipped(result, problems)
    config = configs[0]
    result["config"] = config
    references = [row for row in rows if row.get("kind") == "reference"]
    outcomes = [row for row in rows if row.get("kind") == "result"]
    classes = [row for row in rows if row.get("kind") == "class"]
    schedules = config["cpu_schedules"]
    actual_schedules = [row["schedule"] for row in outcomes]
    if len(references) != 1:
        problems.append(f"expected one reference record, found {len(references)}")
    if len(actual_schedules) != len(schedules) or set(actual_schedules) != set(schedules):
        problems.append("not every configured work schedule has exactly one result")
    expected_classes = set(range(config["f_max"] - config["f_min"] + 1))
    actual_classes = [row["class"] for row in classes]
    if len(actual_classes) != len(expected_classes) or set(actual_classes) != expected_classes:
        problems.append("not every configured final-f stratum has exactly one result")
    if problems:
        result["classes_completed"] = len(classes)
        result["schedules_completed"] = actual_schedules
        return skipped(result, problems)

    percentages = []
    for outcome in outcomes:
        ratios = outcome["additional_fraction_of_paired_full"]
        percentages.append({
            "schedule": outcome["schedule"],
            "additional_percent_of_paired_full": {
                metric: None if ratios.get(metric) is None else {
                    "estimate_percent": 100.0 * ratios[metric]["estimate"],
                    "se_percentage_points": 100.0 * ratios[metric]["se"],
                }
                for metric in METRICS
            },
        })
    result.update({
        "status": "complete",
        "skipped": False,
        "reference": references[0],
        "results": outcomes,
        "percentages": percentages,
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=ROOT / "results/checkpoint_tail")
    parser.add_argument("--out", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args()
    directory = args.directory.resolve()
    output = {
        "notes": {
            "cpu_ratios": "Variant/reference elapsed time; smaller than one is faster. Medians come from the recorded summaries.",
            "paired_ratios": "Match rotated trials by repetition within a run; these are sequential measurements, not simultaneous executions. Ranges are observed extrema, not confidence intervals.",
            "endpoint_ratios": "Variant/reference endpoint count. Different pruning policies intentionally have different endpoints; equality is checked only within named equivalent policy groups.",
            "work_percentages": "Added work divided by the paired full-policy estimate, multiplied by 100. Standard errors are in percentage points and describe stratified sampling only, excluding GPU-model uncertainty.",
            "scope": "CPU timing excludes production endpoint transfer and GPU execution. Work-estimate scope and omitted high-f populations are preserved in each config. No GPU is timed.",
        },
        "cpu_runs": [summarize_cpu(path, directory) for path in sorted(directory.glob("*/cpu.jsonl"))],
        "work_runs": [summarize_work(path, directory) for path in sorted(directory.glob("*/work.jsonl"))],
    }
    text = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
