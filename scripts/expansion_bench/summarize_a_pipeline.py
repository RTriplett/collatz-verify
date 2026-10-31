#!/usr/bin/env python3
"""Compare A=8/9/10 live-pipeline runs on the same N72 checkpoint corpus.

Usage: summarize_a_pipeline.py --out SUMMARY.json [--variant combined50] RUN...
Each run must have completed successfully. Complete record/accounting checks
come from summarize_overall.summarize_run. Runs are separate timed blocks:
cross-A ratios compare medians and are never paired by repetition number.
"""

import argparse
import json
from pathlib import Path
import re

from summarize_overall import distribution, require, summarize_run


METRICS = (
    "wall_seconds", "process_cpu_seconds", "average_busy_cpu_cores",
    "transfer_gb_per_wall_second", "endpoints_per_wall_second",
    "cases_per_wall_second", "iterations_per_wall_second",
)
COUNTS = ("endpoints", "cases", "iters", "tree_nodes", "transfer_bytes")
COMPARISON_NOTE = (
    "Different A values ran in separate process blocks with resident arrays. "
    "Cross-A ratios are ratios of medians, not repetition-paired comparisons. "
    "Replicated blocks are preserved individually and pooled only within the "
    "same A/configuration. Ranges are observed minima and maxima, not confidence intervals."
)
WORK_NOTE = (
    "Changing A changes M=72-A, endpoint grouping, bitvector thresholds and tail "
    "work. Counts and hashes may differ across A. Within each A, every repetition "
    "and rerun must have the identical complete work signature. All A values use "
    "the same ordered K38 roots, full-check cutoff and terminal-entry depth."
)


def selected_variant(runs, requested):
    common = set.intersection(*[
        {item["name"] for item in run["variants"] if item["name"] in ("combined50", "combined54")}
        for run in runs
    ])
    if requested is not None:
        require(requested in common, f"variant {requested} is not present in every run")
        return requested
    require(len(common) == 1,
            "expected exactly one common combined50/combined54 variant; select it with --variant")
    return next(iter(common))


def metrics(trials):
    return {key: distribution(row[key] for row in trials) for key in METRICS}


def ratio_of_medians(row, reference):
    result = {
        "method": "ratio of medians from separate process blocks; unpaired across A",
        "reference_a": 10,
        "wall_ratio": row["wall_seconds"]["median"] / reference["wall_seconds"]["median"],
        "speedup": reference["wall_seconds"]["median"] / row["wall_seconds"]["median"],
        "cpu_ratio": row["process_cpu_seconds"]["median"] / reference["process_cpu_seconds"]["median"],
        "busy_core_ratio": row["average_busy_cpu_cores"]["median"] / reference["average_busy_cpu_cores"]["median"],
        "transfer_gb_per_second_ratio": row["transfer_gb_per_wall_second"]["median"]
            / reference["transfer_gb_per_wall_second"]["median"],
        "endpoint_rate_ratio": row["endpoints_per_wall_second"]["median"]
            / reference["endpoints_per_wall_second"]["median"],
    }
    result["wall_time_saved_fraction"] = 1 - result["wall_ratio"]
    result["cpu_time_saved_fraction"] = 1 - result["cpu_ratio"]
    for key in COUNTS:
        require(reference[key] > 0, f"zero A10 reference {key}")
        result[f"{key}_ratio"] = row[key] / reference[key]
    return result


def collect(directories, requested=None):
    require(bool(directories), "no run directories supplied")
    require(len({p.resolve() for p in directories}) == len(directories), "duplicate run directory")
    runs = [summarize_run(path) for path in directories]
    require(all(run["profile"] == "target" for run in runs), "A comparison requires target runs")
    variant_name = selected_variant(runs, requested)
    common_shape = None
    selected_runs = []
    work_by_a = {}
    engine_by_a = {}
    for directory, run in zip(directories, runs):
        cfg = run["config"]
        a = cfg["a"]
        require(cfg["n"] == 72 and a in (8, 9, 10), "expected N72 and A in {8,9,10}")
        require(cfg["m"] == 72 - a, "target depth does not equal 72-A")
        require(cfg["checkpoint"] == 38, "expected K38 checkpoint corpus")
        selected = next(item for item in run["variants"] if item["name"] == variant_name)
        match = re.fullmatch(r"terminal(50|54)x([1-9][0-9]*)", selected["engine"])
        require(match is not None, f"invalid combined engine {selected['engine']}")
        cutoff, width = map(int, match.groups())
        require(variant_name == f"combined{cutoff}", "variant name and engine cutoff differ")
        entry = cfg["m"] - width
        require(1 <= width <= 12 and entry >= cutoff, "terminal table bypasses full checks")
        require(selected["policy"] == f"cutoff{cutoff}", "combined pruning policy differs")
        runner = run["runner"]
        require(isinstance(runner["affinity"], list) and bool(runner["affinity"]),
                "missing CPU affinity")
        shape = {
            "n": 72, "checkpoint": 38, "corpus_sha256": run["corpus_sha256"],
            "root_count": run["root_count"], "threads": cfg["threads"],
            "affinity": sorted(runner["affinity"]), "class_b": cfg["class_b"],
            "num_bv": cfg["num_bv"], "window": cfg["window"], "rules": cfg["rules"],
            "variant": variant_name, "full_check_cutoff": cutoff,
            "terminal_entry_depth": entry, "device": selected["device"],
            "streams": selected["streams"],
        }
        if common_shape is None:
            common_shape = shape
        else:
            changed = [key for key in shape if shape[key] != common_shape[key]]
            require(not changed, f"{directory}: comparison shape differs in {', '.join(changed)}")
        if a in work_by_a:
            require(work_by_a[a] == selected["work_signature_sha256"],
                    f"A{a}: complete work/counts/hash differ between runs")
            require(engine_by_a[a] == selected["engine"], f"A{a}: terminal engine differs between runs")
        work_by_a[a] = selected["work_signature_sha256"]
        engine_by_a[a] = selected["engine"]

        trials = []
        for line in (directory / "pipeline.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") != "trial" or row["variant"] != variant_name:
                continue
            wall, report = row["wall_seconds"], row["report"]
            trials.append({
                "directory": str(directory), "repetition": row["repetition"],
                "order_position": row["order_position"], "wall_seconds": wall,
                "process_cpu_seconds": row["process_cpu_seconds"],
                "average_busy_cpu_cores": row["average_busy_cpu_cores"],
                "transfer_gb_per_wall_second": report["gpu"]["transfer_bytes"] / wall / 1e9,
                "endpoints_per_wall_second": report["endpoints"] / wall,
                "cases_per_wall_second": report["cases"] / wall,
                "iterations_per_wall_second": report["iters"] / wall,
            })
        require(len(trials) == cfg["repetitions"], "selected trial count differs")
        selected_runs.append({
            "directory": str(directory), "a": a, "m": cfg["m"],
            "terminal_width": width, "terminal_entry_depth": entry,
            "repetitions": len(trials), "engine": selected["engine"],
            "table_bytes": selected["table_bytes"],
            **{key: selected[key] for key in COUNTS},
            "hash": selected["hash"], "work_signature_sha256": selected["work_signature_sha256"],
            "cpu_fallback": selected["cpu_fallback"], "metrics": metrics(trials),
            "trials": trials, "setup": {
                "engine_seconds": cfg["engine_setup_seconds"],
                "gpu_wall_seconds": run["gpu_setup"]["wall_seconds"],
                "gpu_process_cpu_seconds": run["gpu_setup"].get("process_cpu_seconds"),
                "device_array_build_seconds": run["device_array_build_seconds"],
                "reported_session_setup_seconds": run["reported_session_setup_seconds"],
            },
            "verified_run": run,
        })

    require(10 in work_by_a, "A10 reference run is required")
    pooled = []
    for a in sorted(work_by_a):
        blocks = [run for run in selected_runs if run["a"] == a]
        first = blocks[0]
        trials = [row for block in blocks for row in block["trials"]]
        pooled.append({
            "a": a, "m": first["m"], "terminal_width": first["terminal_width"],
            "terminal_entry_depth": first["terminal_entry_depth"], "engine": first["engine"],
            "run_count": len(blocks), "trial_count": len(trials),
            "directories": [block["directory"] for block in blocks],
            "pooling": "all trials within identical A; separate process blocks retained above",
            **metrics(trials), **{key: first[key] for key in COUNTS},
            "transfer_gb_per_trial": first["transfer_bytes"] / 1e9,
            "hash": first["hash"], "work_signature_sha256": first["work_signature_sha256"],
            "cpu_fallback": first["cpu_fallback"],
            "per_run_median_wall_seconds": distribution(block["metrics"]["wall_seconds"]["median"]
                                                        for block in blocks),
            "per_run_median_process_cpu_seconds": distribution(block["metrics"]["process_cpu_seconds"]["median"]
                                                               for block in blocks),
        })
    reference = next(row for row in pooled if row["a"] == 10)
    for row in pooled:
        row["versus_a10"] = ratio_of_medians(row, reference)
    for run in selected_runs:
        run["versus_pooled_a10"] = ratio_of_medians({**run, **run["metrics"]}, reference)
    binary_hashes = sorted({run["verified_run"]["runner"]["binary_sha256"] for run in selected_runs})
    return {
        "comparison_note": COMPARISON_NOTE, "work_note": WORK_NOTE,
        "timing_scope": "Resident-array expansion, packing, queues, transfers, GPU execution and CPU fallback. "
                        "Checkpoint preparation, archive I/O, engine setup and array construction are excluded from trials.",
        "setup_note": "A changes M, terminal width/table size, mod-9 masks and lookahead cutoff parameters. "
                      "Each A receives its own session/array build; setup is recorded separately even though class window widths match.",
        "transfer_note": "GB is decimal (10^9 bytes). Transfer GB/s is payload bytes divided by full pipeline wall time, "
                         "not an isolated PCIe bandwidth measurement.",
        "common": common_shape, "same_binary_in_all_runs": len(binary_hashes) == 1,
        "binary_sha256": binary_hashes, "runs": selected_runs, "by_a": pooled,
    }


def interval(value, digits=3):
    return f'{value["median"]:.{digits}f} [{value["minimum"]:.{digits}f}, {value["maximum"]:.{digits}f}]'


def markdown(summary):
    common = summary["common"]
    lines = ["# A=8/9/10 pipeline comparison", "", summary["timing_scope"], "",
             summary["comparison_note"], "", summary["work_note"], "",
             f'N=72; {common["root_count"]} K38 roots; {common["threads"]} workers; affinity '
             f'{common["affinity"]}; {common["device"]}; {common["streams"]} streams. '
             f'Full checks end at {common["full_check_cutoff"]}; terminal tables start at '
             f'{common["terminal_entry_depth"]}. Corpus SHA-256: `{common["corpus_sha256"]}`.', "",
             f'Class windows: `{",".join(map(str, common["class_b"]))}`. Window rules: `{common["window"]}`.', "",
             "| A | M | Table width | Blocks / trials | Wall seconds, median [min, max] | CPU core-seconds, median [min, max] | Busy cores, median [min, max] | Wall ratio vs A10 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["by_a"]:
        lines.append(f'| {row["a"]} | {row["m"]} | {row["terminal_width"]} | {row["run_count"]} / {row["trial_count"]} | '
                     f'{interval(row["wall_seconds"])} | {interval(row["process_cpu_seconds"])} | '
                     f'{interval(row["average_busy_cpu_cores"])} | {row["versus_a10"]["wall_ratio"]:.4f} |')
    lines.extend(["", summary["transfer_note"], "",
                  "| A | Endpoints | Cases | Tail iterations | Transfer GB/trial | Transfer GB/s, median [min, max] | Endpoints/s, median [min, max] | Hash |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"])
    for row in summary["by_a"]:
        lines.append(f'| {row["a"]} | {row["endpoints"]:,} | {row["cases"]:,} | {row["iters"]:,} | '
                     f'{row["transfer_gb_per_trial"]:.3f} | {interval(row["transfer_gb_per_wall_second"])} | '
                     f'{interval(row["endpoints_per_wall_second"], 0)} | {row["hash"]} |')
    lines.extend(["", summary["setup_note"], "",
                  "| Run | A | Repetitions | Wall seconds, median [min, max] | CPU core-seconds, median [min, max] | GPU setup s | Device build s | Engine setup s | Table bytes |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for run in summary["runs"]:
        setup = run["setup"]
        lines.append(f'| {run["directory"]} | {run["a"]} | {run["repetitions"]} | '
                     f'{interval(run["metrics"]["wall_seconds"])} | {interval(run["metrics"]["process_cpu_seconds"])} | '
                     f'{setup["gpu_wall_seconds"]:.3f} | {setup["device_array_build_seconds"]:.3f} | '
                     f'{setup["engine_seconds"]:.3f} | {run["table_bytes"]:,} |')
    if not summary["same_binary_in_all_runs"]:
        lines.extend(["", "Runs used different binary hashes; source/binary provenance is retained in the JSON. "
                      "Any implementation differences must be considered when interpreting timing comparisons."])
    lines.extend(["", "Complete work signatures match between repetitions and reruns within each A. "
                  "The JSON preserves original validated run summaries, individual trial metrics and unpaired A10 ratios. "
                  "These samples do not constitute a complete 2^72 verification.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=("combined50", "combined54"))
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        summary = collect(args.runs, args.variant)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    markdown_path = args.out.with_suffix(".md")
    markdown_path.write_text(markdown(summary))
    print(f"Saved {args.out} and {markdown_path}")


if __name__ == "__main__":
    main()
