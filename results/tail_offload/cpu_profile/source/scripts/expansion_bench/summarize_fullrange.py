#!/usr/bin/env python3
"""Audit fixed-draw checkpoint experiments and estimate full-range resource use.

Each independent cohort contains a predeclared number of *baseline* draws.
Rejected draws contribute zero downstream work; accepted duplicate roots remain
in the workload. Means use the baseline draw count, never the accepted count.
Repetitions share roots and are averaged within cohorts before inference.
"""

import argparse
from collections import Counter
from functools import reduce
import json
import math
import operator
from pathlib import Path
import statistics

from summarize_overall import digest, require

YEAR_SECONDS = 365.25 * 24 * 3600
DESIGN = "fixed_baseline_draws_with_replacement"
# Stable experiment labels describe these actual kernel settings. Keep this
# independent of the Rust dispatch so a mislabeled run cannot silently pass.
VARIANT_TUNING = {
    "replay16": (256, True, False, False),
    "parents": (256, True, False, False),
    "novote": (256, False, False, False),
    "readonly": (256, True, True, False),
    "both": (256, False, True, False),
    "both_oee": (256, False, True, True),
    "block128": (128, False, True, True),
    "block512": (512, False, True, True),
    "novote_oee": (256, False, False, True),
    "novote_oee128": (128, False, False, True),
    "novote_oee512": (512, False, False, True),
    "affine32": (256, False, False, True),
    "chain4": (256, False, False, True),
    "group4": (256, False, False, True),
    "chain6": (256, False, False, True),
    "group6": (256, False, False, True),
    "jump8": (256, False, False, True),
    "jump10": (256, False, False, True),
    "jump12": (256, False, False, True),
    "jump14": (256, False, False, True),
    "group4_jump12": (256, False, False, True),
    "early_parents": (256, False, False, True),
    "compact66": (256, False, False, True),
    "compact48": (256, False, False, True),
    "early_parents_1k": (256, False, False, True),
    "compact66_1k": (256, False, False, True),
    "compact48_1k": (256, False, False, True),
    "novote_oee_1k": (256, False, False, True),
}
# Tail budget, bounded per-batch GPU queue capacity, and CPU worker count.
# Keep this independent of the Rust variant dispatch so labels cannot hide
# changed thresholds, transport, or available CPU capacity.
TAIL_OFFLOAD_VARIANTS = {
    "defer1": (1, 65536, 1),
    "defer8": (8, 65536, 1),
    "defer16": (16, 65536, 1),
    "defer24": (24, 65536, 1),
    "defer32": (32, 65536, 1),
    "defer64": (64, 65536, 1),
    "defer128": (128, 65536, 1),
    "defer256": (256, 65536, 1),
    "defer512": (512, 65536, 1),
    "defer_count": ((1 << 32) - 1, 65536, 1),
    "defer1_cap1": (1, 1, 1),
    "defer16_w2": (16, 65536, 2),
    "defer24_w2": (24, 65536, 2),
    "defer24_cap262k_w2": (24, 262144, 2),
    "defer24_cap262k_w4": (24, 262144, 4),
    "defer16_cap262k_w4": (16, 262144, 4),
    "defer32_w2": (32, 65536, 2),
    "defer64_w2": (64, 65536, 2),
    "defer128_w2": (128, 65536, 2),
    "defer256_w2": (256, 65536, 2),
    "compact_defer24": (24, 65536, 1),
    "compact_defer24_cap262k_w2": (24, 262144, 2),
    "compact_defer32": (32, 65536, 1),
    "compact_defer128": (128, 65536, 1),
    "compact_defer256": (256, 65536, 1),
}
VARIANT_TUNING.update({name: (256, False, False, True)
                       for name in TAIL_OFFLOAD_VARIANTS})
# Two-sided 95% Student-t critical values for 1..30 degrees of freedom.
T975 = (0, 12.706204736, 4.302652730, 3.182446305, 2.776445105,
        2.570581836, 2.446911851, 2.364624252, 2.306004135, 2.262157163,
        2.228138852, 2.200985160, 2.178812830, 2.160368656, 2.144786688,
        2.131449546, 2.119905299, 2.109815578, 2.100922040, 2.093024054,
        2.085963447, 2.079613845, 2.073873068, 2.068657610, 2.063898562,
        2.059538553, 2.055529439, 2.051830516, 2.048407142, 2.045229642,
        2.042272456)


def critical95(df):
    require(isinstance(df, int) and df > 0, "confidence interval requires two cohorts")
    if df < len(T975):
        return T975[df]
    z = 1.959963984540054
    return (z + (z**3 + z) / (4 * df)
            + (5*z**5 + 16*z**3 + 3*z) / (96 * df**2)
            + (3*z**7 + 19*z**5 + 17*z**3 - 15*z) / (384 * df**3))


def estimate(values, scale=1.0):
    """Approximate t interval across independent, equally weighted cohorts."""
    values = list(values)
    require(len(values) >= 2 and all(math.isfinite(v) for v in values),
            "need at least two finite cohort observations")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(len(values))
    half = critical95(len(values) - 1) * se
    return {"estimate": mean * scale, "standard_error": se * scale,
            "ci95": [(mean - half) * scale, (mean + half) * scale],
            "relative_half_width95": half / abs(mean) if mean else None,
            "cohort_count": len(values), "degrees_of_freedom": len(values) - 1,
            "cohort_minimum": min(values) * scale,
            "cohort_maximum": max(values) * scale,
            "method": "Student-t interval on independent cohort means; approximate, not distribution-free"}


def paired_ratio(numerator, denominator):
    """Ratio of total resource costs, preserving pairing and cohort weights."""
    numerator, denominator = list(numerator), list(denominator)
    require(len(numerator) == len(denominator) >= 2, "invalid paired cohorts")
    denominator_mean = statistics.mean(denominator)
    require(denominator_mean > 0, "zero comparison denominator")
    ratio = statistics.mean(numerator) / denominator_mean
    residual = [a - ratio * b for a, b in zip(numerator, denominator)]
    se = statistics.stdev(residual) / math.sqrt(len(residual)) / denominator_mean
    half = critical95(len(residual) - 1) * se
    return {"ratio": ratio, "ci95": [ratio - half, ratio + half],
            "saved_fraction": 1 - ratio, "saved_fraction_ci95": [1 - ratio - half, 1 - ratio + half],
            "method": "Ratio of cohort means; paired delta-method t interval"}


def positive(value, label):
    require(isinstance(value, (int, float)) and math.isfinite(value) and value > 0,
            f"invalid {label}")


def work_signature(report):
    # Completion order can differ between GPU streams and CPU fallback. Preserve
    # every duplicate origin while comparing a canonical order. Tail flags are
    # currently skipped by Rust serialization; normalize them if later exposed.
    tail = report["tail"]
    if "flagged" in tail:
        tail = {**tail, "flagged": sorted(tail["flagged"], key=int)}
    return {"tree": report["tree"], "tail": tail, "hash": report["hash"],
            "flagged": sorted(report["flagged"], key=int),
            "flagged_rechecked": report["flagged_rechecked"]}


def audit_flags(report):
    gpu, tail = report["gpu"], report["tail"]
    flags = report["flagged"]
    require(isinstance(flags, list), "invalid flagged-origin list")
    origins = [int(value) for value in flags]
    require(all(0 < origin < (1 << report["n"]) for origin in origins),
            "flagged origin is outside the target range")
    require(isinstance(report["flagged_rechecked"], int)
            and report["flagged_rechecked"] == len(flags),
            "flagged trajectories have unresolved wide rechecks")
    require(isinstance(gpu["flags"], int) and 0 <= gpu["flags"] <= len(flags),
            "GPU flag count exceeds total flagged trajectories")
    require(gpu["flags"] <= sum(tail["cases"][:-1]), "GPU flags exceed GPU cases")
    require(len(flags) - gpu["flags"] <= tail["cases"][-1],
            "non-GPU flags exceed CPU-fallback cases")
    if "flagged" in gpu:
        gpu_origins = Counter(int(value) for value in gpu["flagged"])
        require(sum(gpu_origins.values()) == gpu["flags"], "GPU flag count/list differ")
        require(not (gpu_origins - Counter(origins)), "GPU flagged-origin multiset exceeds total flags")
    if "flagged" in tail:
        require(Counter(map(int, tail["flagged"])) == Counter(origins),
                "tail/report flagged-origin multisets differ")


def audit_deferred(gpu):
    """GPU class totals include the completed CPU continuation remainder.

    Deferred counters describe a subset of those exact totals, never additional
    verification cases. Archived reports lacking the counters mean no offload.
    """
    integers = ("deferred_attempts", "deferred_cases", "deferred_rejected",
                "deferred_iterations", "deferred_hash", "deferred_flags",
                "deferred_transfer_bytes")
    timings = ("deferred_cpu_seconds", "deferred_queue_wait_seconds",
               "deferred_drain_seconds")
    enabled = bool((gpu.get("tuning") or {}).get("tail_budget", 0))
    if enabled:
        require(all(key in gpu for key in integers + timings),
                "enabled tail offload is missing accounting fields")
    values = {key: gpu.get(key, 0) for key in integers + timings}
    require(all(isinstance(values[key], int) and not isinstance(values[key], bool)
                and values[key] >= 0 for key in integers), "invalid deferred count")
    require(all(isinstance(values[key], (int, float)) and math.isfinite(values[key])
                and values[key] >= 0 for key in timings), "invalid deferred timing")
    require(values["deferred_hash"] < (1 << 32), "invalid deferred checksum")
    require(values["deferred_attempts"] == values["deferred_cases"] + values["deferred_rejected"],
            "deferred queue attempts do not equal stored plus rejected records")
    require(values["deferred_transfer_bytes"] == 32 * values["deferred_cases"],
            "deferred transfer bytes differ from 32-byte continuation records")
    require(values["deferred_attempts"] <= sum(row["cases"] for row in gpu["classes"]),
            "deferred attempts exceed GPU-origin cases")
    require(values["deferred_iterations"] <= sum(row["iters"] for row in gpu["classes"]),
            "CPU continuation iterations exceed combined GPU-origin iterations")
    require(values["deferred_flags"] <= min(values["deferred_cases"], gpu["flags"]),
            "CPU continuation flags exceed deferred cases or GPU-origin flags")
    if not enabled:
        require(all(value == 0 for value in values.values()),
                "disabled tail offload has deferred work or timings")
    if not values["deferred_cases"]:
        require(values["deferred_iterations"] == values["deferred_hash"] == values["deferred_flags"] == 0,
                "no deferred cases but continuation work is recorded")


def check_report(report, config=None):
    """Full pipeline accounting, permitting only successfully rechecked flags.

    The earlier experiment auditor requires zero flags. Keep its counter,
    payload and configuration invariants here without modifying that auditor
    or hiding flags in a fabricated copy of the report.
    """
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
    if transport in ("parents", "compact_chain"):
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
    audit_flags(report)
    audit_deferred(gpu)
    if config is not None:
        for key in ("n", "a", "num_bv", "class_b", "window", "rules", "threads"):
            require(report[key] == config[key], f"trial/config {key} differ")
        require(report["n"] - report["a"] == config["m"], "trial target depth differs")
        require(report["units"] == len(config["roots"]), "trial does not cover configured corpus")
        require(report["bv_mode"] == "Shared", "target run changed bitvector mode")
        require(not gpu["setup_included_in_report_seconds"], "warm trial includes setup")


def audit_variant(name, gpu):
    require(name in VARIANT_TUNING, f"unrecognized variant label {name}")
    expected = dict(zip(("block_threads", "overflow_vote", "readonly_powers", "oee_mask"),
                        VARIANT_TUNING[name]))
    jump_width = {"jump8": 8, "jump10": 10, "jump12": 12,
                  "jump14": 14, "group4_jump12": 12}.get(name, 0)
    if jump_width:
        expected["tail_jump_bits"] = jump_width
    expected["group_chunk_records"] = 1024 if name.endswith("_1k") else 4096
    expected.update(zip(("tail_budget", "tail_queue_capacity", "tail_cpu_workers"),
                        TAIL_OFFLOAD_VARIANTS.get(name, (0, 65536, 1))))
    # Archived runs and default serialization omit this host-only field.
    # Omission always means 4096, so a reduced-chunk label must record 1024.
    recorded = dict(gpu.get("tuning") or {})
    recorded.setdefault("group_chunk_records", 4096)
    recorded.setdefault("tail_budget", 0)
    recorded.setdefault("tail_queue_capacity", 65536)
    recorded.setdefault("tail_cpu_workers", 1)
    require(recorded == expected and all(type(recorded[key]) is type(value)
                                        for key, value in expected.items()),
            f"variant {name} does not match recorded GPU tuning")
    base_name = name[:-3] if name.endswith("_1k") else name
    transport = "compact_chain" if base_name in ("compact66", "compact48", "compact_defer24", "compact_defer32", "compact_defer128", "compact_defer256", "compact_defer24_cap262k_w2") else "affine32" if name in ("affine32", "chain4", "group4", "chain6", "group6", "group4_jump12") else (
        "replay16" if name == "replay16" else "parents")
    require(gpu.get("transport") == transport,
            f"variant {name} does not match recorded GPU transport")


def audit_metadata(metadata, config):
    arguments = metadata.get("arguments", {})
    fields = {"roots": "draws_per_cohort", "cohorts": "cohort_count",
              "repetitions": "repetitions", "a": "a", "seed": "seed", "threads": "threads",
              "cutoff": "cutoff", "entry": "terminal_entry", "aggregate_runs": "aggregate_repetitions"}
    for argument, field in fields.items():
        if argument in arguments:
            require(arguments[argument] == config[field], f"runner/config {argument} differ")
    if "variants" in arguments:
        require(arguments["variants"].split(",") == config["variants"], "runner/config variants differ")
    if "class_b" in arguments:
        require([int(x) for x in arguments["class_b"].split(",")] == config["class_b"],
                "runner/config class windows differ")
    environment = metadata.get("fullrange_environment", {})
    for name, field in (("CV_FULLRANGE_COHORTS", "cohort_count"),
                        ("CV_FULLRANGE_CUTOFF", "cutoff"),
                        ("CV_FULLRANGE_ENTRY", "terminal_entry"),
                        ("CV_FULLRANGE_AGGREGATE", "aggregate_repetitions")):
        if name in environment:
            require(int(environment[name]) == config[field], f"runner environment/config {name} differ")


def audit_aggregate_draws(row, count, draws, accepted):
    require(row["drawn"] == count * draws, "aggregate trial draw denominator differs")
    require(row["accepted"] == accepted, "aggregate trial accepted count differs")


def audit_cohort(row, config):
    require(row["drawn"] == config["draws_per_cohort"], "cohort draw count differs from fixed design")
    for field in ("drawn", "cheap_rejected", "walk_rejected", "accepted"):
        require(isinstance(row[field], int) and row[field] >= 0, f"invalid cohort {field}")
    require(row["cheap_rejected"] + row["walk_rejected"] + row["accepted"] == row["drawn"],
            "draw accounting does not sum; rejected draws must remain in denominator")
    require(len(row["roots"]) == row["accepted"], "accepted count differs from saved roots")
    require(all(r["k"] == config["checkpoint"] for r in row["roots"]), "root depths differ")
    if "duplicates_retained" in row:
        require(row["duplicates_retained"] == len(row["roots"]) - len({int(r["n0"]) for r in row["roots"]}),
                "retained-duplicate accounting differs")
    # Deliberately do not reject duplicate roots: they are legitimate iid draws.
    return digest(row["roots"])


def aggregate_expected(reports):
    """Work signatures compose by addition except the trajectory checksum (XOR)."""
    signatures = [work_signature(r) for r in reports]
    def combine(items, key=None):
        first = items[0]
        if isinstance(first, dict):
            require(all(set(x) == set(first) for x in items), "work signature key mismatch")
            return {k: combine([x[k] for x in items], k) for k in first}
        if key == "hash":
            if isinstance(first, str):
                return f'{reduce(operator.xor, (int(x, 16) for x in items), 0):08X}'
            return reduce(operator.xor, items, 0)
        if isinstance(first, list):
            if key == "flagged":
                return sorted([v for x in items for v in x], key=int)
            require(all(len(x) == len(first) for x in items), "work signature array length mismatch")
            return [combine([x[i] for x in items]) for i in range(len(first))]
        require(isinstance(first, int), f"noninteger additive field {key}")
        return sum(items)
    return combine(signatures)


def summarize(path):
    metadata = json.loads((path / "metadata.json").read_text())
    require(metadata.get("state") == "complete" and metadata.get("exit_status") == 0,
            f"{path}: incomplete or failed run")
    rows = [json.loads(line) for line in (path / "pipeline.jsonl").read_text().splitlines() if line.strip()]
    configs = [r for r in rows if r.get("type") == "config"]
    require(len(configs) == 1, "missing or duplicate configuration")
    cfg = configs[0]
    require(cfg["sampling_design"] == DESIGN, "fixed accepted-count samples cannot use this estimator")
    audit_metadata(metadata, cfg)
    population = int(cfg["baseline_population"])
    positive(population, "baseline population")
    count, draws, reps = cfg["cohort_count"], cfg["draws_per_cohort"], cfg["repetitions"]
    require(isinstance(count, int) and count >= 2, "need two or more independent cohorts")
    require(isinstance(draws, int) and draws > 0 and isinstance(reps, int) and reps > 0,
            "invalid fixed draw or repetition count")
    names = cfg["variants"]
    require(bool(names) and len(names) == len(set(names)), "missing or duplicate variants")
    require(all(name in VARIANT_TUNING for name in names), "configuration has unknown variant labels")
    cohorts = [r for r in rows if r.get("type") == "cohort"]
    require(sorted(r["cohort"] for r in cohorts) == list(range(count)), "missing or duplicate cohorts")
    cohorts.sort(key=lambda r: r["cohort"])
    require(len({r["seed"] for r in cohorts}) == count, "cohort seeds are not independent")
    corpus_hashes = [audit_cohort(r, cfg) for r in cohorts]
    trials = [r for r in rows if r.get("type") == "trial"]
    require(len(trials) == count * reps * len(names), "trial count differs from predeclared design")
    by_trial = {}
    signatures = {}
    for row in trials:
        c, name, rep = row["cohort"], row["variant"], row["repetition"]
        require(c in range(count) and name in names and rep in range(reps), "unconfigured trial")
        key = c, name, rep
        require(key not in by_trial, "duplicate trial")
        by_trial[key] = row
        require(row["drawn"] == draws, "trial draw denominator differs from cohort")
        positive(row["wall_seconds"], "trial wall seconds")
        positive(row["process_cpu_seconds"], "trial CPU seconds")
        check_report(row["report"], {**cfg, "roots": cohorts[c]["roots"]})
        audit_variant(name, row["report"]["gpu"])
        sig = work_signature(row["report"])
        require(c not in signatures or signatures[c] == sig, "cohort work differs across repetitions or transports")
        signatures[c] = sig
    expected_keys = {(c, n, r) for c in range(count) for n in names for r in range(reps)}
    require(set(by_trial) == expected_keys, "missing trial")
    setups = [r for r in rows if r.get("type") == "gpu_setup"]
    require(len(setups) == 1, "missing or duplicate setup")
    warmups = [r for r in rows if r.get("type") == "warmup"]
    require(sorted(r["variant"] for r in warmups) == sorted(names), "missing or duplicate warmup")
    warm_sig = None
    for row in warmups:
        check_report(row["report"])
        audit_variant(row["variant"], row["report"]["gpu"])
        sig = work_signature(row["report"])
        require(warm_sig is None or sig == warm_sig, "warmup work differs")
        warm_sig = sig
    allowed = {"config", "cohort", "trial", "gpu_setup", "warmup", "aggregate_config", "aggregate_trial"}
    require(all(r.get("type") in allowed for r in rows), "unknown row type")
    reports = [by_trial[c, names[0], 0]["report"] for c in range(count)]
    work = {key: estimate([r[key] / draws for r in reports], population)
            for key in ("endpoints", "cases", "iters", "tree_nodes")}
    work["flagged_trajectories"] = estimate([len(r["flagged"]) / draws for r in reports], population)
    work["wide_rechecks"] = estimate([r["flagged_rechecked"] / draws for r in reports], population)
    observed_flags = [origin for report in reports for origin in report["flagged"]]
    rare_work = {
        "observed_flagged_trajectories": len(observed_flags),
        "observed_distinct_flagged_origins": len(set(map(int, observed_flags))),
        "cohorts_with_flags": sum(bool(report["flagged"]) for report in reports),
        "observed_gpu_flags": sum(report["gpu"]["flags"] for report in reports),
        "observed_successful_wide_rechecks": sum(report["flagged_rechecked"] for report in reports),
        "all_observed_flags_successfully_rechecked": True,
        "note": "Counts use one pass through the sampled cohorts, not repeated timings. Sparse or clustered overflow observations do not give a precise full-range frequency estimate; the t interval is only an approximate sampling diagnostic and may extend below zero.",
    }
    per_class = [{"class": i, **{key: estimate([r["tail"][key][i] / draws for r in reports], population)
                                for key in ("endpoints", "cases", "iters")}}
                 for i in range(cfg["num_bv"] + 1)]
    resource_values = {}
    variants = []
    for name in names:
        means = {key: [statistics.mean(by_trial[c, name, rep][key] for rep in range(reps)) / draws
                       for c in range(count)]
                 for key in ("wall_seconds", "process_cpu_seconds")}
        resource_values[name] = means
        transfer = [statistics.mean(by_trial[c, name, rep]["report"]["gpu"]["transfer_bytes"]
                                    for rep in range(reps)) / draws for c in range(count)]
        variants.append({"name": name,
                         "gpu_tuning": by_trial[0, name, 0]["report"]["gpu"]["tuning"],
                         "gpu_transport": by_trial[0, name, 0]["report"]["gpu"]["transport"],
                         "gpu_years": estimate(means["wall_seconds"], population / YEAR_SECONDS),
                         "cpu_core_years": estimate(means["process_cpu_seconds"], population / YEAR_SECONDS),
                         "transfer_bytes": estimate(transfer, population),
                         "mean_busy_cpu_cores": statistics.mean(means["process_cpu_seconds"]) / statistics.mean(means["wall_seconds"]),
                         "cohort_wall_seconds": [x * draws for x in means["wall_seconds"]],
                         "cohort_cpu_seconds": [x * draws for x in means["process_cpu_seconds"]]})
    baseline = "replay16" if "replay16" in names else names[0]
    for variant in variants:
        name = variant["name"]
        variant["comparison_baseline"] = baseline
        variant["paired_wall_ratio"] = paired_ratio(resource_values[name]["wall_seconds"], resource_values[baseline]["wall_seconds"])
        variant["paired_cpu_ratio"] = paired_ratio(resource_values[name]["process_cpu_seconds"], resource_values[baseline]["process_cpu_seconds"])
    aggregates = [r for r in rows if r.get("type") == "aggregate_trial"]
    aggregate_reps = cfg.get("aggregate_repetitions", 0)
    require(len(aggregates) == aggregate_reps * len(names), "aggregate calibration count differs")
    aggregate_keys = [(r["variant"], r["repetition"]) for r in aggregates]
    require(len(set(aggregate_keys)) == len(aggregate_keys), "duplicate aggregate calibration")
    require(set(aggregate_keys) == {(n, r) for n in names for r in range(aggregate_reps)}, "missing aggregate calibration")
    all_roots = [root for c in cohorts for root in c["roots"]]
    aggregate_configs = [r for r in rows if r.get("type") == "aggregate_config"]
    require(len(aggregate_configs) == int(aggregate_reps > 0), "aggregate configuration count differs")
    if aggregate_configs:
        aggregate_cfg = aggregate_configs[0]
        require(aggregate_cfg["roots"] == all_roots, "aggregate roots differ from ordered cohort concatenation")
        require(aggregate_cfg["accepted"] == len(all_roots) and aggregate_cfg["drawn"] == count * draws,
                "aggregate draw accounting differs")
    aggregate_sig = aggregate_expected(reports)
    aggregate_summary = []
    for row in aggregates:
        audit_aggregate_draws(row, count, draws, len(all_roots))
        check_report(row["report"], {**cfg, "roots": all_roots})
        audit_variant(row["variant"], row["report"]["gpu"])
        require(work_signature(row["report"]) == aggregate_sig, "aggregate work differs from cohort sum/XOR")
        positive(row["wall_seconds"], "aggregate wall seconds")
        name = row["variant"]
        sum_wall = draws * sum(resource_values[name]["wall_seconds"])
        aggregate_summary.append({"variant": name, "repetition": row["repetition"],
                                  "wall_seconds": row["wall_seconds"],
                                  "sum_cohort_wall_seconds": sum_wall,
                                  "aggregate_to_cohort_wall_ratio": row["wall_seconds"] / sum_wall,
                                  "direct_scaled_gpu_years": population * row["wall_seconds"] / (count * draws * YEAR_SECONDS),
                                  "work_equals_cohort_sum_and_hash_xor": True})
    sampling = {key: sum(r[key] for r in cohorts)
                for key in ("drawn", "cheap_rejected", "walk_rejected", "accepted")}
    baseline_audit = None
    if "baseline_endpoint_population" in cfg:
        exact = int(cfg["baseline_endpoint_population"])
        baseline_audit = {
            "exact_endpoint_population": str(exact),
            "sampled_endpoint_population": estimate(
                [int(c["baseline_descendants"]) / draws for c in cohorts], population),
            "purpose": "Exact baseline descendant total is known independently; this checks sampled root composition without changing the primary estimator.",
        }
        baseline_audit["relative_difference"] = baseline_audit["sampled_endpoint_population"]["estimate"] / exact - 1
    return {"directory": str(path), "config": cfg,
            "baseline_population": str(population), "seconds_per_year": YEAR_SECONDS,
            "sampling": sampling,
            "acceptance_fraction": sampling["accepted"] / sampling["drawn"],
            "cohort_corpus_sha256": corpus_hashes, "aggregate_corpus_sha256": digest(all_roots),
            "cohort_work_signature_sha256": [digest(signatures[c]) for c in range(count)],
            "baseline_descendant_audit": baseline_audit,
            "work_estimates": work, "per_class_work_estimates": per_class,
            "rare_work": rare_work,
            "variants": variants, "aggregate_calibration": aggregate_summary,
            "gpu_setup": setups[0],
            "sampling_records": [{k: v for k, v in c.items() if k != "roots"} for c in cohorts],
            "runner": metadata}


def formatted_ci(summary):
    lo, hi = summary["ci95"]
    return f'{summary["estimate"]:.4f} [{lo:.4f}, {hi:.4f}]'


def markdown(result):
    lines = ["# Full-range checkpoint-pipeline extrapolation", "", result["scope"], "", result["method"], "", result["limitations"], ""]
    for run in result["runs"]:
        cfg = run["config"]
        lines += [f'## {run["directory"]}', "",
                  f'N={cfg["n"]}, A={cfg["a"]}, K={cfg["checkpoint"]}; '
                  f'{cfg["cohort_count"]} independent cohorts × {cfg["draws_per_cohort"]} baseline draws, '
                  f'{cfg["repetitions"]} timing repetitions per cohort. '
                  f'Exact baseline population: {run["baseline_population"]}. '
                  f'{run["sampling"]["accepted"]:,}/{run["sampling"]["drawn"]:,} draws survive checkpoint screening.', "",
                  '| Variant | GPU-years [95% interval] | CPU core-years [95% interval] | Busy cores | Ratio to comparison baseline |',
                  '|---|---:|---:|---:|---:|']
        for v in run["variants"]:
            ratio = v["paired_wall_ratio"]
            lines.append(f'| {v["name"]} | {formatted_ci(v["gpu_years"])} | '
                         f'{formatted_ci(v["cpu_core_years"])} | {v["mean_busy_cpu_cores"]:.3f} | '
                         f'{ratio["ratio"]:.4f} [{ratio["ci95"][0]:.4f}, {ratio["ci95"][1]:.4f}] |')
        lines += ["", "Full-range work (sampling estimates, without window-density assumptions):", "",
                  '| Quantity | Estimate | 95% interval |', '|---|---:|---:|']
        for key, stat in run["work_estimates"].items():
            lines.append(f'| {key} | {stat["estimate"]:.7g} | [{stat["ci95"][0]:.7g}, {stat["ci95"][1]:.7g}] |')
        if run["baseline_descendant_audit"]:
            audit = run["baseline_descendant_audit"]
            lines += ["", f'Baseline descendant diagnostic: the same sampled roots predict '
                      f'{audit["sampled_endpoint_population"]["estimate"]:.7g} baseline endpoints versus '
                      f'the independently known exact total {audit["exact_endpoint_population"]} '
                      f'({100*audit["relative_difference"]:+.3f}%). This does not adjust the primary estimate.']
        rare = run["rare_work"]
        lines += ["", f'One pass through the sampled cohorts observed '
                  f'{rare["observed_flagged_trajectories"]} flagged trajectories '
                  f'({rare["observed_distinct_flagged_origins"]} distinct origins) in '
                  f'{rare["cohorts_with_flags"]} cohorts. Every observed flag completed a wide recheck. '
                  'These rechecks are included in the outer pipeline wall time. Sparse/clustered flags '
                  'do not support a precise full-range frequency estimate; their displayed t interval is only a diagnostic.']
        if run["aggregate_calibration"]:
            lines += ["", "Aggregate calibration joins the same roots into one continuous run. Its direct estimate is a correlated check, not an additional independent sample.", "",
                      '| Variant | Aggregate wall s | Aggregate / sum of cohorts | Direct scaled GPU-years |', '|---|---:|---:|---:|']
            for row in run["aggregate_calibration"]:
                lines.append(f'| {row["variant"]} | {row["wall_seconds"]:.3f} | '
                             f'{row["aggregate_to_cohort_wall_ratio"]:.4f} | {row["direct_scaled_gpu_years"]:.4f} |')
        lines += ["", f'One-time GPU setup: {run["gpu_setup"]["wall_seconds"]:.3f} seconds, outside trials. '
                  f'Ordered aggregate corpus SHA-256: `{run["aggregate_corpus_sha256"]}`.', ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        require(len({p.resolve() for p in args.runs}) == len(args.runs), "duplicate run directory")
        result = {
            "scope": "Full-range N-target extrapolation of measured resident-array CPU/GPU pipeline wall time. This is not a completed full-range verification.",
            "method": "Fixed baseline checkpoint draws with replacement; rejected draws contribute zero work. Full-range total = exact baseline population × mean cost per baseline draw. Timing repetitions are averaged within each independent cohort. Student-t intervals use cohort degrees of freedom. Paired performance ratios use ratios of means and a paired delta-method interval. One year is 365.25 days.",
            "limitations": "Intervals describe cohort sampling and observed timing variation only; they do not include systematic desktop clock drift, long-run availability, checkpoint construction or archive I/O, setup, scheduler overhead, rare unsampled fallback/overflow work, or extrapolation to another GPU. Pipeline wall time is used because overlapping CUDA event sums are not GPU busy wall time. Cohort drain/partial-batch effects are checked by a separately reported aggregate run; no correction factor is silently applied. CPU core-years mean process CPU seconds divided by seconds per year, without an SMT conversion.",
            "runs": [summarize(path) for path in args.runs],
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.out.with_suffix(".md").write_text(markdown(result))
    print(f"Saved {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
