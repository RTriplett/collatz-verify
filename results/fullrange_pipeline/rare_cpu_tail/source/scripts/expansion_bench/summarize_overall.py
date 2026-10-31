#!/usr/bin/env python3
"""Summarize completed live-pipeline runs without pooling unlike workloads.

Usage: summarize_overall.py --out results/overall_pipeline/summary.json RUN...
Writes JSON and a sibling Markdown report. Every supplied run must have a
successful runner exit and complete, internally consistent records. Ratios are
paired by repetition within each run; different pruning policies may legitimately
have different endpoint counts and checksums. CUDA event totals are not GPU busy
wall time when streams overlap.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def distribution(values):
    values = list(values)
    require(bool(values) and all(math.isfinite(x) for x in values), "invalid numeric sample")
    return {"median": statistics.median(values), "minimum": min(values), "maximum": max(values)}


def policy(engine):
    if engine in ("full", "screened"):
        return "full"
    match = re.fullmatch(r"(?:reference|compact)(\d+)|terminal(\d+)x\d+", engine)
    require(match is not None, f"unknown engine policy: {engine}")
    return "cutoff" + next(group for group in match.groups() if group is not None)


def work_signature(report):
    return {
        "tree": report["tree"],
        "tail": report["tail"],
        "hash": report["hash"],
        "flagged": sorted(report["flagged"], key=int),
        "flagged_rechecked": report["flagged_rechecked"],
    }


def check_report(report, config=None):
    tail, gpu = report["tail"], report["gpu"]
    classes = report["num_bv"]
    require(report["backend"] == "gpu", "expected GPU pipeline report")
    require(len(gpu["classes"]) == classes, "GPU class count differs")
    for key in ("endpoints", "cases", "iters"):
        require(len(tail[key]) == classes + 1, f"tail {key} class count differs")
        require(sum(tail[key]) == report[key], f"total {key} differs from tail accounting")
        require([row[key] for row in gpu["classes"]] == tail[key][:-1],
                f"GPU/tail per-class {key} differ")
    require(report["tree_nodes"] == sum(report["tree"]["generated"]), "tree node total differs")
    gpu_endpoints = sum(tail["endpoints"][:-1])
    transport = gpu.get("transport", "replay16")
    if transport == "parents":
        require(gpu.get("payload_bytes_per_endpoint") is None, "parent payload is variable")
        require(0 < gpu["input_records"] <= gpu_endpoints, "invalid parent record count")
        expected_bytes = 64 * gpu["input_records"]
    else:
        sizes = {"replay16": 16, "packed8": 8, "affine32": 32}
        require(transport in sizes, "unknown GPU transport")
        require(gpu.get("payload_bytes_per_endpoint", 16) == sizes[transport], "payload size differs")
        require(gpu.get("input_records", gpu_endpoints) == gpu_endpoints, "endpoint record count differs")
        expected_bytes = sizes[transport] * gpu_endpoints
    require(gpu["transfer_bytes"] == expected_bytes, "transfer-byte count differs")
    require(report["hash"] == f'{tail["hash"]:08X}', "tail/report checksums differ")
    require(not report["flagged"], "benchmark reported flagged trajectories")
    require(gpu["flags"] == 0, "GPU reported flagged trajectories")
    if config is not None:
        for key in ("n", "a", "num_bv", "class_b", "window", "rules", "threads"):
            require(report[key] == config[key], f"trial/config {key} differ")
        require(report["n"] - report["a"] == config["m"], "trial target depth differs")
        require(report["units"] == len(config["roots"]), "trial does not cover configured corpus")
        require(report["bv_mode"] == "Shared", "target run changed bitvector mode")
        require(not gpu["setup_included_in_report_seconds"], "warm trial includes setup")


def summarize_validation(rows):
    require(bool(rows) and all(row.get("type") == "validation" for row in rows),
            "expected only complete validation records")
    policies = {}
    variants = []
    common_shape = None
    for row in rows:
        require(row.get("cpu_gpu_equal") is True and row.get("same_policy_equal") is True,
                "validation did not pass")
        report = row["report"]
        check_report(report)
        complete = "complete_range_bits" in row
        sampled = "sampled_range_bits" in row
        require(complete != sampled, "validation must declare exactly one complete or sampled range")
        range_key = "complete_range_bits" if complete else "sampled_range_bits"
        require(row[range_key] == report["n"], "validation range differs from report")
        a, m = report["a"], report["n"] - report["a"]
        require(row.get("a", a) == a and row.get("m", m) == m,
                "validation A/M differs from report")
        scope = {"validation_kind": "complete_range" if complete else "sampled_packages",
                 range_key: row[range_key], "a": a, "m": m}
        if sampled:
            roots = row["roots"]
            require(bool(roots) and len({int(root["n0"]) for root in roots}) == len(roots),
                    "sampled validation roots are empty or duplicated")
            require(report["units"] == len(roots), "sampled validation unit count differs")
            require(all(2 <= root["k"] <= m for root in roots), "sampled validation root depth is invalid")
            require(isinstance(row["validation_scope"], str) and bool(row["validation_scope"]),
                    "sampled validation scope is missing")
            coverage = row["coverage"]
            gpu_endpoints = sum(report["tail"]["endpoints"][:-1])
            require(coverage["gpu_endpoints"] == gpu_endpoints > 0,
                    "sampled validation GPU coverage differs from accounting")
            require(coverage["cpu_fallback_endpoints"] == report["tail"]["endpoints"][-1] > 0,
                    "sampled validation CPU fallback coverage differs from accounting")
            require(coverage["top_payload_bit"] == m - 1, "sampled validation top payload bit differs")
            for key in ("gpu_endpoints_with_top_payload_bit", "gpu_endpoints_with_iterate_above_u64"):
                require(0 < coverage[key] <= gpu_endpoints, f"invalid sampled validation coverage: {key}")
            scope.update(validation_scope=row["validation_scope"], roots=roots,
                         root_count=len(roots), corpus_sha256=digest(roots),
                         root_depths=sorted({root["k"] for root in roots}),
                         seed=row.get("seed"), coverage=coverage)
        shape = {key: report[key] for key in
                 ("n", "a", "num_bv", "class_b", "window", "rules", "bv_mode")}
        shape.update(validation_kind=scope["validation_kind"], corpus_sha256=scope.get("corpus_sha256"))
        if common_shape is None:
            common_shape = shape
        else:
            require(shape == common_shape, "validation scope/configuration differs between variants")
        group = policy(row["variant"])
        signature = work_signature(report)
        if group in policies:
            require(policies[group] == signature, f"validation differs within policy {group}")
        else:
            policies[group] = signature
        variants.append({
            "variant": row["variant"], "policy": group,
            **scope,
            "device_built_arrays": row.get("device_built_arrays", False),
            "endpoints": report["endpoints"], "cases": report["cases"],
            "iters": report["iters"], "hash": report["hash"],
            "cpu_fallback_endpoints": report["tail"]["endpoints"][-1],
            "cpu_gpu_equal": True, "same_policy_equal": True,
        })
    require(len({row["variant"] for row in variants}) == len(variants), "duplicate validation variant")
    return {"profile": "validation", "validation_kind": common_shape["validation_kind"],
            "variants": variants}


def summarize_target(rows, metadata):
    configs = [row for row in rows if row.get("type") == "config"]
    require(len(configs) == 1, "expected exactly one config")
    config = configs[0]
    roots = config["roots"]
    require(bool(roots) and len({int(root["n0"]) for root in roots}) == len(roots),
            "empty or duplicate root corpus")
    require(all(root["k"] == config["checkpoint"] for root in roots), "root checkpoint depths differ")
    args = metadata["arguments"]
    for key in ("seed", "threads", "repetitions"):
        require(config[key] == args[key], f"metadata/config {key} differ")
    require(len(roots) == args["roots"], "metadata/config root counts differ")
    variants = {row["name"]: row for row in config["variants"]}
    require(len(variants) == len(config["variants"]) and bool(variants), "duplicate or empty variants")
    repetitions = config["repetitions"]
    require(isinstance(repetitions, int) and repetitions > 0, "invalid repetition count")
    trial_map = {name: {} for name in variants}
    summaries = {}
    setup = [row for row in rows if row.get("type") == "gpu_setup"]
    require(len(setup) == 1, "expected exactly one GPU setup record")
    for row in rows:
        kind = row.get("type")
        if kind == "trial":
            name, rep = row["variant"], row["repetition"]
            require(name in trial_map, f"unconfigured trial variant {name}")
            require(isinstance(rep, int) and 0 <= rep < repetitions, "invalid trial repetition")
            require(rep not in trial_map[name], f"duplicate trial {name}/{rep}")
            require(math.isfinite(row["wall_seconds"]) and row["wall_seconds"] > 0, "invalid wall time")
            require(math.isfinite(row["process_cpu_seconds"]) and row["process_cpu_seconds"] > 0,
                    "invalid process CPU time")
            check_report(row["report"], config)
            require(math.isclose(row["average_busy_cpu_cores"],
                                 row["process_cpu_seconds"] / row["wall_seconds"], rel_tol=1e-9),
                    "average busy CPU accounting differs")
            trial_map[name][rep] = row
        elif kind == "summary":
            name = row["variant"]
            require(name not in summaries, f"duplicate recorded summary {name}")
            summaries[name] = row
        else:
            require(kind in ("config", "gpu_setup", "warmup"), f"unknown record type {kind}")
    require(set(summaries) == set(variants), "recorded summary variants differ")
    for name, trials in trial_map.items():
        require(set(trials) == set(range(repetitions)), f"missing trial repetitions for {name}")
    for rep in range(repetitions):
        require({trials[rep]["order_position"] for trials in trial_map.values()} == set(range(len(variants))),
                f"trial order positions differ at repetition {rep}")

    policies = {}
    summaries_out = []
    for name, variant in variants.items():
        group = policy(variant["engine"])
        trials = [trial_map[name][rep] for rep in range(repetitions)]
        first = trials[0]["report"]
        signature = work_signature(first)
        require(all(work_signature(trial["report"]) == signature for trial in trials),
                f"work changes between repetitions of {name}")
        if group in policies:
            require(policies[group]["signature"] == signature, f"same-policy work differs for {name}")
            policies[group]["variants"].append(name)
        else:
            policies[group] = {"signature": signature, "variants": [name]}
        gpu = first["gpu"]
        result = {
            **variant, "policy": group, "repetitions": repetitions,
            "wall_seconds": distribution(row["wall_seconds"] for row in trials),
            "process_cpu_seconds": distribution(row["process_cpu_seconds"] for row in trials),
            "average_busy_cpu_cores": distribution(row["average_busy_cpu_cores"] for row in trials),
            "endpoints": first["endpoints"], "cases": first["cases"], "iters": first["iters"],
            "tree_nodes": first["tree_nodes"], "hash": first["hash"],
            "transfer_bytes": gpu["transfer_bytes"], "streams": gpu["streams"],
            "device": gpu["device"], "cpu_fallback": {
                key: first["tail"][key][-1] for key in ("endpoints", "cases", "iters")
            },
            "work_signature_sha256": digest(signature),
            "per_class_work": [{key: cls[key] for key in ("endpoints", "cases", "iters")}
                               for cls in gpu["classes"]],
        }
        for key in ("pipeline_seconds", "producer_wall_seconds", "producer_pool_wait_seconds",
                    "producer_queue_send_seconds", "kernel_seconds"):
            result[key] = distribution(row["report"]["gpu"][key] for row in trials)
        result["launches"] = distribution(sum(cls["launches"] for cls in row["report"]["gpu"]["classes"])
                                          for row in trials)
        result["endpoints_per_wall_second"] = distribution(first["endpoints"] / row["wall_seconds"]
                                                           for row in trials)
        if "full" in trial_map:
            pairs = []
            for rep, row in enumerate(trials):
                reference = trial_map["full"][rep]
                pairs.append({
                    "repetition": rep,
                    "wall_ratio": row["wall_seconds"] / reference["wall_seconds"],
                    "speedup": reference["wall_seconds"] / row["wall_seconds"],
                    "cpu_ratio": row["process_cpu_seconds"] / reference["process_cpu_seconds"],
                })
            result["versus_full"] = {
                "paired": pairs,
                **{key: distribution(pair[key] for pair in pairs) for key in ("wall_ratio", "speedup", "cpu_ratio")},
                "wall_wins": sum(pair["wall_ratio"] < 1 for pair in pairs),
                "cpu_wins": sum(pair["cpu_ratio"] < 1 for pair in pairs),
                "median_wall_time_saved_fraction": 1 - statistics.median(pair["wall_ratio"] for pair in pairs),
                "median_cpu_time_saved_fraction": 1 - statistics.median(pair["cpu_ratio"] for pair in pairs),
            }
        summaries_out.append(result)
    first_gpu = next(iter(trial_map.values()))[0]["report"]["gpu"]
    return {
        "profile": "target", "config": config,
        "corpus_sha256": digest(roots), "root_count": len(roots),
        "gpu_setup": setup[0],
        "reported_session_setup_seconds": first_gpu["setup_seconds"],
        "device_array_build_seconds": first_gpu["build_seconds"],
        "same_policy_checks": {group: {
            "variants": item["variants"], "all_repetitions_equal": True,
            "multiple_implementations_compared": len(item["variants"]) > 1,
            "work_signature_sha256": digest(item["signature"]),
            "work": item["signature"],
        } for group, item in policies.items()},
        "variants": summaries_out,
    }


def summarize_run(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    require(metadata.get("state") == "complete" and metadata.get("exit_status") == 0,
            f"{directory}: runner has not completed successfully")
    rows = []
    for lineno, line in enumerate((directory / "pipeline.jsonl").read_text().splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            require(isinstance(row, dict), f"{directory}:{lineno}: record is not an object")
            rows.append(row)
    require(bool(rows), f"{directory}: no result records")
    if all(row.get("type") == "validation" for row in rows):
        result = summarize_validation(rows)
    else:
        result = summarize_target(rows, metadata)
    return {
        "directory": str(directory), "status": "complete", **result,
        "runner": {key: metadata.get(key) for key in (
            "state", "exit_status", "started_at", "finished_at", "elapsed_seconds",
            "command", "affinity", "cv_gpu_slots", "binary_sha256", "source_sha256",
        )},
    }


def markdown(summary):
    lines = ["# Overall pipeline benchmark", "", summary["timing_scope"], "",
             summary["kernel_seconds_note"], "", summary["comparison_note"], ""]
    for run in summary["runs"]:
        lines.extend([f'## {run["directory"]}', ""])
        if run["profile"] == "validation":
            sampled = run.get("validation_kind") == "sampled_packages"
            label = "sampled-package" if sampled else "complete-range"
            lines.extend([f'{len(run["variants"])} {label} variants passed CPU/GPU and same-policy checks.', ""])
            if sampled:
                first = run["variants"][0]
                coverage = first["coverage"]
                lines.extend([
                    f'{first["validation_scope"]}. A={first["a"]}, M={first["m"]}; '
                    f'{first["root_count"]} saved roots at depths {first["root_depths"]}. '
                    f'Corpus SHA-256: `{first["corpus_sha256"]}`.', "",
                    f'Coverage includes {coverage["gpu_endpoints_with_top_payload_bit"]:,} GPU endpoints '
                    f'with payload bit {coverage["top_payload_bit"]} set, '
                    f'{coverage["gpu_endpoints_with_iterate_above_u64"]:,} GPU endpoints with iterates above u64, '
                    f'and {coverage["cpu_fallback_endpoints"]:,} CPU-fallback endpoints. '
                    'This validates the sampled packages, not the complete range.', "",
                ])
            lines.extend([
                          "| Variant | Range | Endpoints | Cases | CPU fallback endpoints | Hash |",
                          "|---|---:|---:|---:|---:|---|"])
            for row in run["variants"]:
                range_label = (f'N={row["sampled_range_bits"]} sampled packages' if sampled
                               else f'2^{row["complete_range_bits"]}')
                lines.append(f'| {row["variant"]} | {range_label} | {row["endpoints"]:,} | '
                             f'{row["cases"]:,} | {row["cpu_fallback_endpoints"]:,} | {row["hash"]} |')
            lines.append("")
            continue
        cfg = run["config"]
        lines.extend([
            f'N={cfg["n"]}, A={cfg["a"]}, M={cfg["m"]}; {run["root_count"]} common K={cfg["checkpoint"]} roots; '
            f'{cfg["threads"]} workers; {cfg["repetitions"]} repetitions. Corpus SHA-256: `{run["corpus_sha256"]}`.', "",
            f'GPU setup: {run["gpu_setup"]["wall_seconds"]:.3f} s; device array build: '
            f'{run["device_array_build_seconds"]:.3f} s; expansion-engine setup: {cfg["engine_setup_seconds"]:.3f} s.', "",
            "| Variant | Wall median [min, max], s | CPU median, core-s | Busy CPU cores | Paired speedup vs full | Paired CPU reduction |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in run["variants"]:
            wall, versus = row["wall_seconds"], row.get("versus_full")
            speedup = f'{versus["speedup"]["median"]:.3f}×' if versus else "—"
            reduction = f'{100 * versus["median_cpu_time_saved_fraction"]:.2f}%' if versus else "—"
            lines.append(f'| {row["name"]} | {wall["median"]:.4f} [{wall["minimum"]:.4f}, {wall["maximum"]:.4f}] | '
                         f'{row["process_cpu_seconds"]["median"]:.4f} | {row["average_busy_cpu_cores"]["median"]:.3f} | '
                         f'{speedup} | {reduction} |')
        lines.extend(["", "| Variant | Endpoints | Cases | Iterations | Transfer bytes | CPU fallback endpoints | Hash |",
                      "|---|---:|---:|---:|---:|---:|---|"])
        for row in run["variants"]:
            lines.append(f'| {row["name"]} | {row["endpoints"]:,} | {row["cases"]:,} | {row["iters"]:,} | '
                         f'{row["transfer_bytes"]:,} | {row["cpu_fallback"]["endpoints"]:,} | {row["hash"]} |')
        lines.extend(["", "Counts, every tree counter, and hashes match across repetitions and within each pruning policy. "
                      "Policies with only one selected implementation have repetition consistency checks only.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        require(len(set(path.resolve() for path in args.runs)) == len(args.runs), "duplicate run directory")
        summary = {
            "timing_scope": "Resident-array CPU→GPU pipeline wall times include expansion, packing, queueing, transfers, "
                            "GPU execution and CPU fallback. Checkpoint preparation, archive I/O and table/array setup "
                            "are excluded from trials and recorded separately. N72 target runs are sampled work packages.",
            "kernel_seconds_note": "GPU kernel seconds sum CUDA stream event intervals. Concurrent streams overlap, "
                                   "so these totals are not GPU busy wall time.",
            "comparison_note": "Speedup and CPU ratios are paired by repetition against full within each run. "
                               "Different policies intentionally have different work and checksums. Runs with different "
                               "thread counts or corpora are kept separate. Reported ranges are observed extrema, "
                               "not confidence intervals.",
            "runs": [summarize_run(path) for path in args.runs],
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    markdown_path = args.out.with_suffix(".md")
    markdown_path.write_text(markdown(summary))
    print(f"Saved {args.out} and {markdown_path}")


if __name__ == "__main__":
    main()
