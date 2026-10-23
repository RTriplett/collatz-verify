#!/usr/bin/env python3
"""Summarize completed terminal-block benchmarks within each pruning policy.

Each directory contains one run with a common cutoff and endpoint sink. Failed,
incomplete, or output-mismatched runs are shown but supply no timing estimates.
No timings or endpoint sets are compared across directories.
"""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[2]


def summarize_run(path, directory):
    result = {"file": str(path.relative_to(directory))}
    problems = []
    rows = []
    try:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record is not an object")
                rows.append(row)
            except ValueError as error:
                problems.append(f"unreadable JSON at line {number}: {error}")
                break
    except OSError as error:
        problems.append(f"cannot read results: {error}")
    metadata_path = path.parent / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            result["runner"] = {
                key: metadata.get(key)
                for key in ("state", "exit_status", "started_at", "finished_at")
            }
            if metadata.get("state") != "complete" or metadata.get("exit_status") != 0:
                problems.append("runner has not completed successfully")
        except (OSError, ValueError) as error:
            problems.append(f"cannot read runner metadata: {error}")

    def skip():
        state = result.get("runner", {}).get("state")
        result.update({
            "status": state if state in ("failed", "interrupted") else "incomplete",
            "skipped": True,
            "reasons": problems,
        })
        return result

    configs = [row for row in rows if row.get("type") == "config"]
    if len(configs) != 1:
        problems.append(f"expected one config record, found {len(configs)}")
        return skip()
    config = configs[0]
    result["config"] = config  # Includes table sizes, setup time and all policy parameters.
    names = config.get("variants")
    repetitions = config.get("repetitions")
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        problems.append("missing, empty, or duplicate configured variants")
        return skip()
    if not isinstance(repetitions, int) or repetitions < 1:
        problems.append("configured repetitions must be positive")
        return skip()
    reps = set(range(repetitions))
    trials = {name: {} for name in names}
    summaries = {}
    for row in rows:
        if row.get("type") == "trial":
            name, rep, seconds = row.get("variant"), row.get("repetition"), row.get("seconds")
            if name not in trials:
                problems.append(f"trial for unconfigured variant {name!r}")
                continue
            if not isinstance(rep, int):
                problems.append(f"invalid repetition for {name}: {rep!r}")
                continue
            if rep in trials[name]:
                problems.append(f"duplicate trial for {name}, repetition {rep}")
            if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
                problems.append(f"nonpositive or invalid timing for {name}, repetition {rep}")
            trials[name][rep] = row
        elif row.get("type") == "summary":
            name = row.get("variant")
            if name in summaries:
                problems.append(f"duplicate summary for {name}")
            summaries[name] = row
    if set(summaries) != set(names):
        problems.append(
            f"summaries missing={sorted(set(names) - set(summaries))}, "
            f"unexpected={sorted(str(name) for name in set(summaries) - set(names))}"
        )
    for name in names:
        if set(trials[name]) != reps:
            problems.append(
                f"{name}: repetitions missing={sorted(reps - set(trials[name]))}, "
                f"unexpected={sorted(set(trials[name]) - reps)}"
            )
    validations = [row for row in rows if row.get("type") == "validation"]
    result["validation"] = validations
    result["additional_records"] = [
        row for row in rows if row.get("type") not in ("config", "trial", "summary", "validation")
    ]
    if not validations or not all(row.get("passed") is True for row in validations):
        problems.append("missing or unsuccessful endpoint audit")
    if problems:
        result["trial_counts"] = {name: len(values) for name, values in trials.items()}
        return skip()

    first = summaries[names[0]]
    for name in names:
        summary = summaries[name]
        if "digest" not in summary or "tree_stats" not in summary:
            problems.append(f"missing digest or tree statistics for {name}")
        elif summary["digest"] != first.get("digest") or summary["tree_stats"] != first.get("tree_stats"):
            problems.append(f"endpoint digest or tree statistics differ for {name}")
        elif any(row.get("endpoints") != summary["digest"].get("endpoints") for row in trials[name].values()):
            problems.append(f"trial endpoint count differs from summary for {name}")
    if problems:
        return skip()

    # Recompute medians so even repetition counts use the conventional midpoint.
    medians = {
        name: statistics.median(row["seconds"] for row in trials[name].values())
        for name in names
    }
    variants = []
    for name in names:
        timing = trials[name]
        variant = {
            "variant": name,
            "median_seconds": medians[name],
            "recorded_median_seconds": summaries[name].get("median_seconds"),
            "minimum_seconds": min(row["seconds"] for row in timing.values()),
            "maximum_seconds": max(row["seconds"] for row in timing.values()),
            "endpoints": summaries[name]["digest"]["endpoints"],
            "generated_nodes": sum(summaries[name]["tree_stats"]["generated"]),
        }
        for reference in ("compact", "production"):
            if reference not in trials:
                continue
            ratios = [timing[rep]["seconds"] / trials[reference][rep]["seconds"] for rep in sorted(reps)]
            variant.update({
                f"cpu_ratio_vs_{reference}": medians[name] / medians[reference],
                f"time_saved_fraction_vs_{reference}": 1.0 - medians[name] / medians[reference],
                f"paired_median_cpu_ratio_vs_{reference}": statistics.median(ratios),
                f"paired_cpu_ratio_range_vs_{reference}": [min(ratios), max(ratios)],
                f"wins_vs_{reference}": sum(ratio < 1.0 for ratio in ratios),
                f"ties_vs_{reference}": sum(ratio == 1.0 for ratio in ratios),
            })
        variants.append(variant)
    result.update({
        "status": "complete",
        "skipped": False,
        "all_digests_and_tree_stats_equal": True,
        "digest": first["digest"],
        "tree_stats": first["tree_stats"],
        "paired_repetitions": repetitions,
        "variants": variants,
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=ROOT / "results/terminal_blocks")
    parser.add_argument("--out", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args()
    directory = args.directory.resolve()
    output = {
        "notes": {
            "cpu_ratios": "Variant/reference elapsed time; smaller than one is faster. Medians are recomputed from recorded trials.",
            "paired_ratios": "Match sequential rotated trials by repetition within a run. Ranges are observed extrema, not confidence intervals.",
            "equivalence": "Every variant in a run must have identical endpoint digests and per-depth tree statistics. Runs with different cutoffs or sinks are never pooled.",
            "scope": "CPU benchmark only. Preserve each config for table size, table construction time, sink behavior and excluded costs. No GPU is timed.",
        },
        "runs": [summarize_run(path, directory) for path in sorted(directory.glob("*/cpu.jsonl"))],
    }
    text = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
