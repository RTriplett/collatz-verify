#!/usr/bin/env python3
"""Audit early-screen pipeline runs and summarize measured costs, without extrapolation."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from summarize_fullrange import (DESIGN, aggregate_expected, audit_aggregate_draws,
                                 audit_cohort, audit_metadata, audit_variant,
                                 check_report, early_settings, paired_ratio,
                                 positive, work_signature)
from summarize_overall import digest, require


COUNTERS = ("regular_batches", "early_batches", "early_endpoints", "early_cases",
            "early_iterations", "early_flags", "early_transfer_bytes", "early_buckets",
            "early_jumps", "early_jumped_iterations", "early_scalar_iterations",
            "early_table_lookups")
TIMINGS = ("early_copy_seconds", "early_host_copy_seconds", "early_cpu_seconds",
           "early_decode_seconds", "early_group_seconds", "early_verification_seconds",
           "early_table_setup_seconds", "early_queue_wait_seconds", "early_drain_seconds")
CLASS_VECTORS = ("early_class_endpoints", "early_class_cases", "early_class_iterations")
NOTES = [
    "Directly measured costs only; no full-range GPU-year or rental-price extrapolation.",
    "Logical GPU-origin totals include the exact CPU-finished survivors; early work is a subset, never additional work.",
    "CPU workers, producer threads, CUDA compute and copies overlap. Do not add their times into an elapsed-time breakdown.",
    "early_copy_seconds sums CUDA stream event intervals around D2H copies; it is not exclusive device busy time.",
    "early_host_copy_seconds is host-thread CPU time scanning mask popcounts and copying compact jobs; early_cpu_seconds is summed worker CPU time. Decode/group/verification are summed worker elapsed intervals, not exclusive process CPU time.",
    "Every-Nth launch selection starts at launch zero and resets each run. Producer interleaving can change selected classes and actual CPU work between variants or repetitions. Compare recorded class mix and CPU counts; identical selected states are established only by separate fixture replay.",
    "Continuous aggregates concatenate the same cohort roots. Repeats share work; their min/mean/max describe measured variation, not independent sampling uncertainty.",
    "Warmup and setup are excluded. Every summarized trial must report zero captured states.",
]


def divide(numerator, denominator):
    return numerator / denominator if denominator else None


def spread(values):
    values = list(values)
    require(bool(values), "cannot summarize empty measurements")
    defined = [value for value in values if value is not None]
    counts = {"defined": len(defined), "undefined": len(values) - len(defined)}
    if not defined:
        return {"min": None, "mean": None, "max": None, **counts}
    return {"min": min(defined), "mean": statistics.mean(defined), "max": max(defined), **counts}


def measurement(row):
    gpu = row["report"]["gpu"]
    values = {key: gpu.get(key, 0) for key in COUNTERS + TIMINGS}
    values.update(wall_seconds=row["wall_seconds"], process_cpu_seconds=row["process_cpu_seconds"],
                  pipeline_seconds=row["report"]["seconds"], input_transfer_bytes=gpu["transfer_bytes"])
    for key in ("endpoints", "cases", "iters"):
        values["gpu_origin_" + key] = sum(item[key] for item in gpu["classes"])
    values.update(
        average_busy_cpu_cores=divide(values["process_cpu_seconds"], values["wall_seconds"]),
        cpu_tail_case_fraction=divide(values["early_cases"], values["gpu_origin_cases"]),
        cpu_tail_iteration_fraction=divide(values["early_iterations"], values["gpu_origin_iters"]),
        mean_cpu_iterations_per_case=divide(values["early_iterations"], values["early_cases"]),
        worker_cpu_ns_per_case=divide(values["early_cpu_seconds"] * 1e9, values["early_cases"]),
        table_jump_iteration_fraction=divide(values["early_jumped_iterations"], values["early_iterations"]),
        gpu_executed_iterations=values["gpu_origin_iters"] - values["early_iterations"]
            - gpu.get("deferred_iterations", 0),
    )
    return {"repetition": row["repetition"], "cohort": row.get("cohort"), "metrics": values,
            "class_mix": {key: gpu.get(key, [0] * len(gpu["classes"])) for key in CLASS_VECTORS},
            "early_hash": gpu.get("early_hash", 0),
            "work_signature_sha256": digest(work_signature(row["report"]))}


def summarize_measurements(rows):
    if not rows:
        return None
    samples = [measurement(row) for row in rows]
    keys = samples[0]["metrics"]
    classes = len(samples[0]["class_mix"][CLASS_VECTORS[0]])
    vectors = {}
    for key in CLASS_VECTORS:
        by_class = [spread(sample["class_mix"][key][i] for sample in samples) for i in range(classes)]
        vectors[key] = {stat: [value[stat] for value in by_class] for stat in ("min", "mean", "max")}
    return {"repetitions": len(samples),
            "metrics": {key: spread(sample["metrics"][key] for sample in samples) for key in keys},
            "class_mix": vectors, "measurements": samples}


def audit_run(path):
    """Reuse the full-range accounting audits without invoking its estimator."""
    path = path / "pipeline.jsonl" if path.is_dir() else path
    directory = path.parent
    metadata_path = directory / "metadata.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    require(metadata.get("state") == "complete" and metadata.get("exit_status") == 0,
            f"{directory}: incomplete or failed run; wait for runner completion")
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    allowed = {"config", "cohort", "trial", "gpu_setup", "warmup", "aggregate_config", "aggregate_trial"}
    require(all(row.get("type") in allowed for row in rows), "unknown pipeline row type")
    configs = [row for row in rows if row["type"] == "config"]
    require(len(configs) == 1, "missing or duplicate pipeline configuration")
    cfg = configs[0]
    require(cfg["sampling_design"] == DESIGN, "unexpected cohort sampling design")
    audit_metadata(metadata, cfg)
    count, draws, reps = cfg["cohort_count"], cfg["draws_per_cohort"], cfg["repetitions"]
    require(all(type(value) is int and value > 0 for value in (count, draws, reps)),
            "invalid cohort, draw or repetition count")
    names = cfg["variants"]
    require(bool(names) and len(names) == len(set(names)), "missing or duplicate variants")
    cohorts = sorted((row for row in rows if row["type"] == "cohort"), key=lambda row: row["cohort"])
    require([row["cohort"] for row in cohorts] == list(range(count)), "missing or duplicate cohorts")
    require(len({row["seed"] for row in cohorts}) == count, "cohort seeds are not independent")
    cohort_hashes = [audit_cohort(row, cfg) for row in cohorts]
    trials = [row for row in rows if row["type"] == "trial"]
    require(len(trials) == count * reps * len(names), "trial count differs from declared design")
    signatures, by_trial = {}, {}
    for row in trials:
        cohort, name, rep = row["cohort"], row["variant"], row["repetition"]
        require(cohort in range(count) and name in names and rep in range(reps), "unconfigured trial")
        key = cohort, name, rep
        require(key not in by_trial, "duplicate trial")
        by_trial[key] = row
        require(row["drawn"] == draws, "trial draw denominator differs")
        check_report(row["report"], {**cfg, "roots": cohorts[cohort]["roots"]})
        audit_variant(name, row["report"]["gpu"])
        require(row["report"]["gpu"].get("early_capture_cases", 0) == 0,
                "timed trial captured diagnostic states; exclude this run")
        for metric in ("wall_seconds", "process_cpu_seconds"):
            positive(row[metric], metric)
        signature = work_signature(row["report"])
        require(cohort not in signatures or signatures[cohort] == signature,
                "cohort work differs between methods or repeats")
        signatures[cohort] = signature
    require(set(by_trial) == {(c, name, rep) for c in range(count) for name in names for rep in range(reps)},
            "missing trial")
    setups = [row for row in rows if row["type"] == "gpu_setup"]
    require(len(setups) == 1, "missing or duplicate GPU setup")
    warmups = [row for row in rows if row["type"] == "warmup"]
    require(sorted(row["variant"] for row in warmups) == sorted(names), "missing or duplicate warmup")
    warm_signature = None
    for row in warmups:
        check_report(row["report"])
        audit_variant(row["variant"], row["report"]["gpu"])
        signature = work_signature(row["report"])
        require(warm_signature is None or warm_signature == signature, "warmup work differs")
        warm_signature = signature
    aggregate_reps = cfg.get("aggregate_repetitions", 0)
    require(type(aggregate_reps) is int and aggregate_reps >= 0, "invalid aggregate repetition count")
    aggregates = [row for row in rows if row["type"] == "aggregate_trial"]
    keys = [(row["variant"], row["repetition"]) for row in aggregates]
    require(len(keys) == len(set(keys)) and set(keys) == {(name, rep) for name in names for rep in range(aggregate_reps)},
            "missing, duplicate or unexpected aggregate trial")
    all_roots = [root for cohort in cohorts for root in cohort["roots"]]
    configs = [row for row in rows if row["type"] == "aggregate_config"]
    require(len(configs) == int(aggregate_reps > 0), "missing or duplicate aggregate configuration")
    if configs:
        require(configs[0]["roots"] == all_roots and configs[0]["drawn"] == count * draws
                and configs[0]["accepted"] == len(all_roots), "aggregate corpus differs")
    aggregate_signature = aggregate_expected([by_trial[c, names[0], 0]["report"] for c in range(count)])
    for row in aggregates:
        audit_aggregate_draws(row, count, draws, len(all_roots))
        check_report(row["report"], {**cfg, "roots": all_roots})
        audit_variant(row["variant"], row["report"]["gpu"])
        require(work_signature(row["report"]) == aggregate_signature, "aggregate work differs from cohort sum/XOR")
        require(row["report"]["gpu"].get("early_capture_cases", 0) == 0,
                "aggregate trial captured diagnostic states; exclude this run")
        for metric in ("wall_seconds", "process_cpu_seconds"):
            positive(row[metric], metric)
    return {"directory": str(directory), "jsonl": str(path), "metadata": metadata, "config": cfg,
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "jsonl_sha256": hashlib.sha256(raw).hexdigest(), "exact_work_audit_passed": True,
            "aggregate_corpus_sha256": digest(all_roots),
            "aggregate_work_signature_sha256": digest(aggregate_signature),
            "cohort_corpus_sha256": cohort_hashes,
            "cohort_work_signature_sha256": [digest(signatures[c]) for c in range(count)],
            "baseline_draws": count * draws, "accepted_draws": len(all_roots),
            "warmup_captured_states": sum(row["report"]["gpu"].get("early_capture_cases", 0) for row in warmups),
            "capture_requested": bool(cfg.get("early_capture_requested", False)),
            "setup": setups[0], "rows": rows}


def summarize_run(path, baseline=None, include_cohorts=False):
    run = audit_run(path)
    rows = run.pop("rows")
    cfg = run["config"]
    names = cfg["variants"]
    require(baseline is None or baseline in names, "requested baseline is absent from run")
    variants = []
    for name in names:
        trials = [row for row in rows if row["type"] == "trial" and row["variant"] == name]
        aggregates = sorted((row for row in rows if row["type"] == "aggregate_trial" and row["variant"] == name),
                            key=lambda row: row["repetition"])
        gpu = trials[0]["report"]["gpu"]
        cohorts = [summarize_measurements([row for row in trials if row["cohort"] == c])
                   for c in range(cfg["cohort_count"])]
        variants.append({"name": name, "transport": gpu["transport"], "tuning": gpu["tuning"],
                         "continuous_aggregate": summarize_measurements(aggregates), "cohorts": cohorts})
    by_name = {variant["name"]: variant for variant in variants}
    defaults = {"parents": "novote_oee", "compact_chain": "compact66", "affine32": "affine32", "replay16": "replay16"}
    for variant in variants:
        control_name = baseline or defaults.get(variant["transport"])
        control = by_name.get(control_name)
        require(control is None or control["transport"] == variant["transport"],
                "comparison baseline uses a different transport")
        require(control is None or not early_settings(control_name), "comparison baseline itself enables early offload")
        variant["comparison_baseline"] = control_name if control else None
        variant["continuous_ratios"] = None
        variant["paired_cohort_ratios"] = None
        if control:
            variant["paired_cohort_ratios"] = {}
            for metric in ("wall_seconds", "process_cpu_seconds"):
                numerator = [cohort["metrics"][metric]["mean"] for cohort in variant["cohorts"]]
                denominator = [cohort["metrics"][metric]["mean"] for cohort in control["cohorts"]]
                variant["paired_cohort_ratios"][metric] = (paired_ratio(numerator, denominator)
                    if len(numerator) >= 2 else {"ratio": numerator[0] / denominator[0], "ci95": None,
                                                "method": "One cohort; no sampling interval"})
            if variant["continuous_aggregate"] and control["continuous_aggregate"]:
                variant["continuous_ratios"] = {
                    metric: variant["continuous_aggregate"]["metrics"][metric]["mean"]
                    / control["continuous_aggregate"]["metrics"][metric]["mean"]
                    for metric in ("wall_seconds", "process_cpu_seconds")}
        variant["cohort_means"] = [{key: cohort["metrics"][key]["mean"]
                                    for key in ("wall_seconds", "process_cpu_seconds", "early_cases", "early_iterations")}
                                   for cohort in variant["cohorts"]]
    if not include_cohorts:
        for variant in variants:
            del variant["cohorts"]
    run["capture_warning"] = ("Diagnostic capture was requested or occurred during warmup. Warmup is excluded; all summarized trials captured zero states. Keep the capture context when interpreting this run."
        if run["capture_requested"] or run["warmup_captured_states"] else None)
    run["variants"] = variants
    return run


def markdown(result):
    lines = ["# Early GPU screening and CPU verification", ""]
    lines.extend(note + "\n" for note in NOTES)
    for run in result["runs"]:
        lines.extend([f'## {run["directory"]}', "",
                      f'{run["baseline_draws"]:,} baseline draws, {run["accepted_draws"]:,} accepted; '
                      'all exact-work and accounting audits passed.', ""])
        if run["capture_warning"]:
            lines.extend([run["capture_warning"], ""])
        lines.extend(["Continuous aggregate means; wall range is min–max across repeated timings.", "",
                      "| Variant | Wall s [range] | Wall / baseline | Process CPU core-s | CPU tail cases | CPU iteration share | D2H GiB |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for variant in run["variants"]:
            aggregate = variant["continuous_aggregate"]
            if not aggregate:
                continue
            m = aggregate["metrics"]
            ratio = variant["continuous_ratios"]
            ratio_text = f'{ratio["wall_seconds"]:.4f}' if ratio else "—"
            lines.append(f'| {variant["name"]} | {m["wall_seconds"]["mean"]:.4f} '
                         f'[{m["wall_seconds"]["min"]:.4f}–{m["wall_seconds"]["max"]:.4f}] | '
                         f'{ratio_text} | {m["process_cpu_seconds"]["mean"]:.3f} | '
                         f'{m["early_cases"]["mean"]:,.0f} | '
                         f'{100 * (m["cpu_tail_iteration_fraction"]["mean"] or 0):.5f}% | '
                         f'{m["early_transfer_bytes"]["mean"] / 2**30:.4f} |')
        lines.extend(["", "Separate timing measurements below overlap and must not be added together.", "",
                      "| Variant | D2H event s | Host scan/copy CPU s | Worker CPU s | Decode elapsed s | Group elapsed s | Verify elapsed s | Queue send s | Drain s |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for variant in run["variants"]:
            aggregate = variant["continuous_aggregate"]
            if aggregate:
                keys = ("early_copy_seconds", "early_host_copy_seconds", "early_cpu_seconds", "early_decode_seconds",
                        "early_group_seconds", "early_verification_seconds", "early_queue_wait_seconds", "early_drain_seconds")
                lines.append('| ' + variant["name"] + ' | ' + ' | '.join(
                    f'{aggregate["metrics"][key]["mean"]:.4f}' for key in keys) + ' |')
        lines.extend(["", 'JSON retains all min/mean/max metrics, individual repetitions, selected per-class work, '
                      'paired cohort ratios, corpus/work fingerprints, configuration and raw runner metadata. '
                      'Different CPU methods can receive different selected trajectories in these pipeline runs; '
                      'use the separate fixed-state CPU replay to isolate their algorithmic costs.', ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories or pipeline.jsonl files")
    parser.add_argument("--out", required=True, type=Path, help="JSON path; Markdown uses the same stem")
    parser.add_argument("--baseline", help="Override the matched transport baseline label")
    parser.add_argument("--include-cohorts", action="store_true", help="Include detailed per-cohort metrics")
    args = parser.parse_args()
    try:
        paths = [(path / "pipeline.jsonl" if path.is_dir() else path).resolve() for path in args.runs]
        require(len(paths) == len(set(paths)), "duplicate pipeline input")
        result = {"schema_version": 1, "scope": NOTES[0], "notes": NOTES,
                  "runs": [summarize_run(path, args.baseline, args.include_cohorts) for path in paths]}
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.out.with_suffix(".md").write_text(markdown(result))
    print(f"Saved {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
