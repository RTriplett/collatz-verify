#!/usr/bin/env python3
"""Summarize completed runs, including ratios within each rotated repetition.

Usage: python3 scripts/expansion_bench/summarize.py [results/expansion_bench]
"""
import json
from pathlib import Path
import statistics
import sys

directory = Path(sys.argv[1] if len(sys.argv) > 1 else "results/expansion_bench")
runs = []
for path in sorted(directory.glob("*/*.jsonl")):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    config = next(row for row in rows if row["type"] == "config")
    trials = {}
    summaries = {}
    for row in rows:
        if row["type"] == "trial":
            trials.setdefault(row["variant"], {})[row["repetition"]] = row["seconds"]
        elif row["type"] == "summary":
            summaries[row["variant"]] = row
    assert set(summaries) == set(config["variants"]), f"Incomplete run: {path}"
    reference = trials["reference"]
    variants = []
    for name in config["variants"]:
        timing = trials[name]
        assert set(timing) == set(range(config["repetitions"])), path
        summary = summaries[name]
        assert summary["digest"] == summaries["reference"]["digest"], path
        ratios = [reference[rep] / value for rep, value in sorted(timing.items())]
        result = {
            "variant": name,
            "median_seconds": summary["median_seconds"],
            "minimum_seconds": min(timing.values()),
            "maximum_seconds": max(timing.values()),
            "ratio_of_medians_vs_reference": summary["speedup_vs_reference"],
            "paired_speedup_median_vs_reference": statistics.median(ratios),
            "paired_speedup_range_vs_reference": [min(ratios), max(ratios)],
            "repetitions_faster_than_reference": sum(ratio > 1 for ratio in ratios),
        }
        control_name = "pm_control" if "pm_control" in trials else "control"
        if control_name in trials:
            result["matched_control_variant"] = control_name
            result["paired_speedup_median_vs_control"] = statistics.median(
                trials[control_name][rep] / value for rep, value in timing.items()
            )
        variants.append(result)
    runs.append({
        "file": str(path.relative_to(directory)),
        "case": config["case"], "seed": config["seed"],
        "roots": config["roots"], "threads": config["threads"],
        "statistics": config.get("statistics", False),
        "repetitions": config["repetitions"],
        "endpoints": summaries["reference"]["digest"]["endpoints"],
        "variants": variants,
    })
print(json.dumps({
    "notes": "Speedup is reference/variant; greater than one is faster. Paired ratios use the same repetition, not simultaneous executions. Ranges are observed extrema, not confidence intervals.",
    "runs": runs,
}, indent=2))
