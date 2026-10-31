#!/usr/bin/env python3
"""Audit exact early-CPU replays and summarize repeated fixed-fixture timings.

Timing repetitions measure variability on one captured fixture. They are not
independent checkpoint samples and do not support a new full-range cost estimate.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


MODE_NAMES = ("scalar", "bucket_scalar", "table", "initial_group_table", "regrouped_table")
COUNTERS = ("cases", "iters", "hash", "flags", "buckets", "jumps",
            "jumped_iterations", "scalar_iterations", "table_lookups")
TIMINGS = ("wall_seconds", "cpu_seconds", "grouping_seconds",
           "verification_seconds", "table_setup_seconds")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"{label}: invalid integer")
    return value


def finite(value, label, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f"{label}: invalid timing")
    return value


def distribution(values):
    return {"min": min(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "max": max(values)}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def configuration_key(row):
    return row["mode"], row["width"], row["chunk_size"]


def fixture_identity(path, cache):
    path = path.resolve()
    if path not in cache:
        data = path.read_bytes()
        fingerprint = 0xcbf29ce484222325
        for byte in data:
            fingerprint = ((fingerprint ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
        cache[path] = {"path": str(path), "bytes": len(data),
                       "sha256": hashlib.sha256(data).hexdigest(),
                       "fnv1a64": f"{fingerprint:016x}"}
    return cache[path]


def audit_rows(rows):
    """Strict work/configuration checks; intentionally independent of timing ratios."""
    require(rows and all(isinstance(row, dict) for row in rows), "empty or non-object JSON rows")
    require(rows and rows[0].get("type") == "config", "missing initial configuration")
    require(rows[-1].get("type") == "complete", "replay did not finish")
    by_type = defaultdict(list)
    for row in rows:
        require(isinstance(row, dict), "non-object JSON row")
        by_type[row.get("type")].append(row)
    require(set(by_type) == {"config", "oracle", "table_setup", "trial", "complete"},
            "missing or unexpected record type")
    require(all(len(by_type[k]) == 1 for k in ("config", "oracle", "complete")),
            "duplicate config, oracle, or completion")
    config, oracle, complete = (by_type[k][0] for k in ("config", "oracle", "complete"))
    require(config["experiment"] == "early_cpu_replay", "unexpected experiment")
    states = integer(config["states"], "states", 1)
    available = integer(config["available_states"], "available states", 1)
    require(states <= available and config["input_bytes"] == available * 32,
            "fixture size or selected state count mismatch")
    require(config["workers"] == 1, "CPU replay requires one worker")
    require(type(config["cutoff_bits"]) is int and 97 <= config["cutoff_bits"] <= 123,
            "invalid cutoff")
    reps = integer(config["repetitions"], "repetitions", 3)
    modes, widths, chunks = (config[k] for k in ("modes", "widths", "chunks"))
    for name, values in (("modes", modes), ("widths", widths), ("chunks", chunks)):
        require(isinstance(values, list) and values and len(values) == len(set(values)),
                f"empty or duplicated {name}")
        for value in values:
            integer(value, name)
    require(0 in modes and set(modes) <= set(range(5)), "scalar control missing or invalid mode")
    require(set(widths) <= {8, 10, 12, 14} and min(chunks) > 0, "invalid width/chunk")
    expected_configs = [(mode, width, chunk) for chunk in chunks for mode in modes
                        for width in ([0] if mode == 0 else widths)]
    require(complete["exact_match"] is True and complete["configs"] == len(expected_configs)
            and complete["trials"] == len(expected_configs) * reps, "completion count mismatch")
    for key in ("cases", "iters", "hash", "flags", "unique_wide_rechecks"):
        integer(oracle[key], f"oracle {key}")
    require(oracle["cases"] == states and oracle["iters"] >= states
            and oracle["hash"] <= 0xffffffff and oracle["flags"] <= states,
            "invalid oracle signature")
    require(oracle["all_wide_rechecks_passed"] is True
            and oracle["unique_wide_rechecks"] <= oracle["flags"]
            and (oracle["flags"] == 0) == (oracle["unique_wide_rechecks"] == 0),
            "unresolved or inconsistent wide rechecks")
    setup = by_type["table_setup"]
    require(len(setup) == len(widths) and {r["width"] for r in setup} == set(widths),
            "missing or duplicated table setup")
    for row in setup:
        for key in ("wall_seconds", "cpu_seconds", "table_setup_seconds"):
            finite(row[key], f"setup {key}")
    trials = by_type["trial"]
    require(len(trials) == len(expected_configs) * reps, "missing or duplicated timing trial")
    seen = set()
    groups = defaultdict(list)
    for row in trials:
        key = configuration_key(row)
        repetition = integer(row["repetition"], "repetition")
        position = integer(row["order_position"], "order position")
        require(key in expected_configs and repetition < reps and position < len(expected_configs),
                "unexpected trial configuration/order")
        identity = (repetition, key)
        require(identity not in seen, "duplicate trial")
        seen.add(identity)
        order = [(i + 13 * repetition) % len(expected_configs) for i in range(len(expected_configs))]
        if repetition % 2:
            order.reverse()
        require(key == expected_configs[order[position]], "trial ordering differs from balanced rotation")
        require(row["mode_name"] == MODE_NAMES[key[0]] and row["exact_match"] is True,
                "mode label mismatch or failed comparison")
        for name in COUNTERS:
            integer(row[name], name)
        require(all(row[name] == oracle[name] for name in ("cases", "iters", "hash", "flags")),
                "trial work differs from scalar oracle")
        require(row["scalar_iterations"] + row["jumped_iterations"] == row["iters"],
                "iteration accounting mismatch")
        require(row["jumps"] <= row["jumped_iterations"] <= 3 * row["jumps"],
                "invalid affine jump lengths")
        for name in TIMINGS:
            finite(row[name], name, positive=name in ("wall_seconds", "cpu_seconds"))
        require(row["grouping_seconds"] + row["verification_seconds"]
                + row["table_setup_seconds"] <= row["wall_seconds"] + 1e-8,
                "phase timings exceed the enclosing wall time")
        for rate, count in (("cases_per_cpu_second", "cases"), ("iterations_per_cpu_second", "iters")):
            finite(row[rate], rate, positive=True)
            require(math.isclose(row[rate], row[count] / row["cpu_seconds"], rel_tol=1e-12),
                    "reported CPU throughput mismatch")
        mode = key[0]
        if mode < 2:
            require(row["jumps"] == row["jumped_iterations"] == row["table_lookups"] == 0,
                    "scalar mode reports table work")
        if mode in (0, 2):
            require(row["buckets"] == 0 and row["grouping_seconds"] == 0,
                    "ungrouped mode reports grouping")
        else:
            require(row["buckets"] > 0, "grouped mode did not visit a bucket")
        if mode == 3:
            require(row["table_lookups"] >= row["buckets"], "missing initial shared lookups")
        if mode == 4:
            require(row["table_lookups"] == row["buckets"], "regrouped lookup accounting mismatch")
        groups[key].append(row)
    require(set(groups) == set(expected_configs), "missing benchmark configuration")
    for key, group in groups.items():
        require({row["repetition"] for row in group} == set(range(reps)), "missing repetition")
        require(all(all(row[c] == group[0][c] for c in COUNTERS) for row in group),
                f"non-deterministic work or sharing counters for {key}")
        group.sort(key=lambda row: row["repetition"])
    return config, oracle, setup, groups


def summarize_run(path, fixture=None, metadata=None, cache=None):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    config, oracle, setup, groups = audit_rows(rows)
    fixture_path = fixture or Path(config["input"])
    if not fixture_path.exists() and not fixture:
        fixture_path = path.parent / fixture_path
    identity = fixture_identity(fixture_path, cache if cache is not None else {})
    require(identity["bytes"] == config["input_bytes"]
            and identity["fnv1a64"] == config["input_fingerprint_fnv1a64"],
            "fixture differs from the benchmark's input fingerprint")
    provenance = None
    if metadata:
        record = json.loads(metadata.read_text())
        declared = next((record[key] for key in ("fixture_sha256", "input_sha256", "capture_sha256")
                         if key in record), None)
        require(declared is not None, "runner metadata must declare fixture_sha256")
        require(declared == identity["sha256"], "fixture SHA-256 differs from runner metadata")
        provenance = {"path": str(metadata), "sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                      "fixture_sha256_verified": True, "record": record}
    variants = []
    for key, group in sorted(groups.items()):
        mode, width, chunk = key
        baseline = groups[(0, 0, chunk)]
        counters = {name: group[0][name] for name in COUNTERS}
        values = {name: distribution([row[name] for row in group]) for name in TIMINGS}
        controls = {}
        for metric in ("cpu_seconds", "wall_seconds"):
            ratios = [row[metric] / control[metric] for row, control in zip(group, baseline)]
            controls[metric] = {
                "mean_ratio": values[metric]["mean"] / statistics.mean(row[metric] for row in baseline),
                "median_ratio": values[metric]["median"] / statistics.median(row[metric] for row in baseline),
                "within_repetition_ratios": distribution(ratios),
            }
        table_control = groups.get((2, width, chunk))
        lookup_saving = None
        if mode >= 2 and table_control:
            denominator = table_control[0]["table_lookups"]
            lookup_saving = 1 - counters["table_lookups"] / denominator if denominator else None
        variants.append({
            "mode": mode, "name": MODE_NAMES[mode], "width": width, "chunk_size": chunk,
            "timing_repetitions": len(group), "timings": values, "counters": counters,
            "matched_scalar": controls,
            "grouping_fraction_of_wall": values["grouping_seconds"]["mean"] / values["wall_seconds"]["mean"],
            "jumped_iteration_fraction": counters["jumped_iterations"] / counters["iters"],
            "table_lookup_reduction_vs_independent": lookup_saving,
            "cases_per_mean_cpu_second": counters["cases"] / values["cpu_seconds"]["mean"],
            "iterations_per_mean_cpu_second": counters["iters"] / values["cpu_seconds"]["mean"],
        })
    def selection(variant):
        return {"name": variant["name"], "mode": variant["mode"], "width": variant["width"],
                "chunk_size": variant["chunk_size"], "mean_cpu_seconds": variant["timings"]["cpu_seconds"]["mean"],
                "mean_wall_seconds": variant["timings"]["wall_seconds"]["mean"],
                "cpu_ratio_to_matched_scalar": variant["matched_scalar"]["cpu_seconds"]["mean_ratio"]}
    best = min(variants, key=lambda v: v["timings"]["cpu_seconds"]["mean"])
    return {"input_jsonl": str(path), "input_jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fixture": identity, "runner_metadata": provenance,
            "config": config, "oracle": oracle, "table_setup": setup,
            "exact_work_audit_passed": True,
            "best_observed_by_mean_cpu": selection(best),
            "best_observed_per_method": [selection(min([v for v in variants if v["mode"] == mode],
                                                      key=lambda v: v["timings"]["cpu_seconds"]["mean"]))
                                         for mode in sorted(config["modes"])],
            "variants": variants}


def markdown(result):
    lines = ["# Early CPU verification replay", "", result["scope"], "", result["accounting"], ""]
    for run in result["runs"]:
        config = run["config"]
        lines.extend([f'## {run["input_jsonl"]}', "",
                      f'{config["states"]:,} captured states; {config["repetitions"]} timing repetitions '
                      f'on one CPU worker. All work and accounting checks passed. Fixture SHA-256: '
                      f'`{run["fixture"]["sha256"]}`.', ""])
        provenance = run.get("runner_metadata")
        if provenance:
            record = provenance["record"]
            if record.get("fixture_scope"):
                lines.extend([record["fixture_scope"], ""])
            if record.get("cpu_model"):
                lines.extend([f'CPU: {record["cpu_model"]}; affinity {record.get("cpu_affinity", "recorded by runner")}.', ""])
        lines.extend(["Best observed configuration per method (lowest mean CPU time):", "",
                      "| Method | Width | Chunk | CPU s | Wall s | CPU / matched scalar |",
                      "|---|---:|---:|---:|---:|---:|"])
        for best in run["best_observed_per_method"]:
            lines.append(f'| {best["name"]} | {best["width"] or "—"} | {best["chunk_size"]:,} | '
                         f'{best["mean_cpu_seconds"]:.6f} | {best["mean_wall_seconds"]:.6f} | '
                         f'{best["cpu_ratio_to_matched_scalar"]:.4f} |')
        lines.extend(["", "Every measured configuration:", "",
                      "| Method | Width | Chunk | CPU s min / mean / median | Wall s min / mean / median | CPU / scalar | Wall / scalar | Grouping / wall | Iterations jumped | Fewer table lookups |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for v in run["variants"]:
            def timings(metric):
                return " / ".join(f'{v["timings"][metric][k]:.6f}' for k in ("min", "mean", "median"))
            saving = v["table_lookup_reduction_vs_independent"]
            lookup_text = f"{saving:.2%}" if saving is not None else "—"
            lines.append(f'| {v["name"]} | {v["width"] or "—"} | {v["chunk_size"]:,} | '
                         f'{timings("cpu_seconds")} | {timings("wall_seconds")} | '
                         f'{v["matched_scalar"]["cpu_seconds"]["mean_ratio"]:.4f} | '
                         f'{v["matched_scalar"]["wall_seconds"]["mean_ratio"]:.4f} | '
                         f'{v["grouping_fraction_of_wall"]:.2%} | {v["jumped_iteration_fraction"]:.2%} | '
                         f'{lookup_text} |')
        lines.extend(["", "Ratios compare the scalar control with the same chunk size. Lookup reduction "
                      "compares independent table lookup at the same width and chunk size. "
                      "Initial grouping shares only the first transition; regrouping shares successive "
                      "transitions and includes each sorting pass. Setup times and raw counters are in JSON.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="early_cpu_bench JSONL outputs")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fixture", type=Path, help="override the fixture path stored in each run")
    parser.add_argument("--metadata", type=Path, help="runner metadata declaring fixture_sha256")
    args = parser.parse_args()
    try:
        require(len({p.resolve() for p in args.runs}) == len(args.runs), "duplicate replay output")
        cache = {}
        result = {"schema_version": 1,
                  "scope": "Repeated CPU-only timings of captured post-sieve states. Repetitions are "
                           "not independent root samples; no confidence interval or full-range GPU-year estimate is inferred.",
                  "accounting": "Single-worker CPU and wall times include state decoding, validation, allocation, "
                                "all bucket construction and exact verification. Resident tables are built separately. "
                                "Fixture loading, GPU filtering/transfers, oracle comparisons and wide rechecks are excluded. "
                                "Best configurations are descriptive selections from this fixture, not established population winners.",
                  "runs": [summarize_run(p, args.fixture, args.metadata, cache) for p in args.runs]}
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.out.with_suffix(".md").write_text(markdown(result))
    print(f"Saved {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
