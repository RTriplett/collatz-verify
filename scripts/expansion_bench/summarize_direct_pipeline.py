#!/usr/bin/env python3
"""Audit and summarize CPU routing before GPU dispatch, without extrapolation."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from summarize_fullrange import (aggregate_expected, audit_cohort, audit_metadata,
                                 audit_variant, check_report, work_signature)
from summarize_overall import digest, require

SLOTS = 81
MASK64 = (1 << 64) - 1
TIMINGS = ("wall_seconds", "process_cpu_seconds", "cpu_worker_seconds",
           "cpu_reconstruction_seconds", "cpu_verification_seconds", "cpu_drain_seconds",
           "cpu_wide_recheck_seconds")
NOTES = [
    "Measured continuous-corpus timings only; no GPU-year or rental-price extrapolation.",
    "GPU reports exclude diverted CPU work. Original-depth source endpoints are conserved separately from CPU-native descendants, cases and checksums.",
    "Different CPU A/B policies can remove different candidates and change checksums. Exact native signatures are compared only for the same policy and the same recorded admitted descriptor identity.",
    "Admission is nonblocking and depends on worker availability, so admitted groups can differ between repetitions. Descriptor count/XOR/sum identifiers are consistency checks, not cryptographic proofs of identical sets.",
    "The startup guard counts endpoints initially forwarded to the ordinary sink; it is not a live measurement of GPU queue occupancy.",
    "Process CPU, summed worker elapsed intervals, CUDA event sums and elapsed pipeline time overlap; they cannot be added into a wall-time breakdown.",
    "GPU case/iteration reduction is measured against the same corpus using the unchanged GPU policy. CPU-native iterations are not a substitute for that reduction.",
    "Warmup, source sampling and resident-table construction are excluded. Continuous repetitions reuse the same roots; observed ranges are not sampling confidence intervals.",
    "Target trials must disable CPU source capture and GPU diagnostic capture. Validation replay may capture sources outside the performance comparison.",
]


def integer(value, label, minimum=0, maximum=None):
    require(type(value) is int and value >= minimum and (maximum is None or value <= maximum),
            f"invalid {label}")
    return value


def finite(value, label, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f"invalid {label}")
    return value


def vector(value, label, length=SLOTS):
    require(isinstance(value, list) and len(value) == length, f"invalid {label} length")
    for entry in value:
        integer(entry, label)
    return value


def fingerprint(value, label):
    require(isinstance(value, dict) and set(value) == {"count", "xor", "sum"}, f"invalid {label}")
    integer(value["count"], label + " count")
    for key in ("xor", "sum"):
        integer(value[key], label + " " + key, maximum=MASK64)
    if not value["count"]:
        require(value["xor"] == value["sum"] == 0, f"nonempty {label} without descriptors")
    return value


def add_fingerprints(values):
    result = {"count": 0, "xor": 0, "sum": 0}
    for value in values:
        result["count"] += value["count"]
        result["xor"] ^= value["xor"]
        result["sum"] = (result["sum"] + value["sum"]) & MASK64
    return result


def flags(values, n, label):
    require(isinstance(values, list), f"invalid {label}")
    for value in values:
        require(isinstance(value, str) and re.fullmatch(r"[0-9]+", value)
                and 3 <= int(value) < 1 << n, f"invalid {label} origin")
    return sorted(values, key=int)


def variant_settings(value, gpu_a):
    name = value["name"]
    if name in ("gpu", "route_control"):
        expected = dict(name=name, cpu_a=gpu_a, cpu_b=None, stride=0,
                        workers=0, wrapper=name == "route_control")
    else:
        match = re.fullmatch(r"direct_a([0-9]+)_b(none|[0-9]+)_s([0-9]+)_w([0-9]+)", name)
        require(match is not None, "unknown routing variant")
        a, b, stride, workers = match.groups()
        expected = dict(name=name, cpu_a=int(a), cpu_b=None if b == "none" else int(b),
                        stride=int(stride), workers=int(workers), wrapper=True)
        require(0 <= expected["cpu_a"] <= gpu_a and expected["stride"] > 0
                and 1 <= expected["workers"] <= 8, "invalid routing policy")
        require(expected["cpu_b"] is None or max(6, expected["cpu_a"]) <= expected["cpu_b"] <= 24,
                "invalid CPU lookahead width")
    require(value == expected and all(type(value[k]) is type(v) for k, v in expected.items()),
            "variant label does not match its recorded policy")
    return expected


def common_config(cfg, roots):
    return {**cfg, "a": cfg["gpu_a"], "m": cfg["gpu_m"],
            "class_b": cfg["gpu_class_b"], "roots": roots}


def audit_root(root, depth):
    require(root["k"] == depth and isinstance(root["n0"], str), "root depth/residue differs")
    n = int(root["n0"])
    require(3 <= n < 1 << depth and n % 4 == 3, "invalid saved root")
    x, f = n, 0
    for _ in range(depth):
        if x % 2:
            x = (3 * x + 1) // 2
            f += 1
        else:
            x //= 2
    require(root["m"] == str(x) and root["f"] == f and root["m3"] == x % (3 ** 16),
            "saved root has inconsistent affine state")
    integer(root["oee"], "root OEE", maximum=2)
    integer(root["ne"], "root even run", maximum=3)


def audit_result(result, cfg, variant, roots, timed=True):
    report = result["report"]
    check_report(report, common_config(cfg, roots))
    audit_variant("novote_oee", report["gpu"])
    require(report["cutoff_bits"] == cfg["cutoff_bits"] and report["tail_oee"] is True,
            "tail cutoff/OEE differs")
    for key in ("endpoints", "cases", "iters", "tree_nodes", "units"):
        integer(report[key], "report " + key)
    for key in ("endpoints", "cases", "iters"):
        vector(report["tail"][key], "tail " + key, cfg["num_bv"] + 1)
    tree = report["tree"]
    vector(tree["generated"], "tree generated", cfg["gpu_m"] + 1)
    vector(tree["survived"], "tree survived", cfg["gpu_m"] + 1)
    require(len(tree["culls"]) == cfg["gpu_m"] + 1 and bool(tree["culls"][0]), "invalid tree culls")
    for row in tree["culls"]:
        vector(row, "tree culls", len(tree["culls"][0]))
    for key in TIMINGS:
        finite(result[key], key, positive=key in ("wall_seconds", "process_cpu_seconds"))
    finite(result["average_busy_cpu_cores"], "average busy CPU cores")
    require(math.isclose(result["average_busy_cpu_cores"], result["process_cpu_seconds"] / result["wall_seconds"],
                         rel_tol=1e-9, abs_tol=1e-9), "average CPU use differs")
    require(result["cpu_worker_seconds"] <= result["process_cpu_seconds"] + 0.01,
            "worker CPU exceeds total process CPU")
    require(result["cpu_drain_seconds"] + result["cpu_wide_recheck_seconds"] <= result["wall_seconds"] + 1e-8,
            "CPU completion phases exceed pipeline wall")
    require(result["partition_equal"] is True, "source partition failed")
    require(type(result["cpu_source_capture_enabled"]) is bool, "missing CPU capture policy")
    integer(result["captured_cpu_sources"], "captured CPU sources")
    if timed:
        require(not result["cpu_source_capture_enabled"] and result["captured_cpu_sources"] == 0,
                "performance trial captured CPU sources")
        require(report["gpu"].get("early_capture_cases", 0) == 0, "performance trial captured GPU states")
    work = result["cpu_native"]
    for key in ("source_endpoints", "endpoints", "cases", "iters"):
        integer(work[key], "CPU " + key)
    require(isinstance(work["hash"], str) and re.fullmatch(r"[0-9A-F]{8}", work["hash"]), "invalid CPU checksum")
    flag_list = flags(work["flagged"], cfg["n"], "CPU flags")
    require(flag_list == work["flagged"] and len(flag_list) <= work["cases"], "invalid CPU flag multiset")
    source_residues = {int(root["n0"]) for root in roots}
    source_mask = (1 << cfg["checkpoint"]) - 1
    require(all(int(origin) & source_mask in source_residues
                for origin in flag_list + report["flagged"]), "flagged origin is outside the assigned root families")
    integer(result["cpu_flagged_rechecked"], "CPU flagged rechecks")
    require(result["cpu_flagged_rechecked"] == len(flag_list), "unresolved diverted CPU flags")
    source_counts = vector(result["cpu_original_by_f"], "CPU original endpoint classes")
    require(sum(source_counts) == work["source_endpoints"], "CPU sources differ from completed class counts")
    require(work["endpoints"] <= work["source_endpoints"] * (1 << (cfg["gpu_a"] - variant["cpu_a"])),
            "too many CPU descendants")
    require(work["cases"] <= work["endpoints"] * (1 << variant["cpu_a"]), "too many CPU candidate cases")
    require(work["iters"] + len(flag_list) >= work["cases"], "CPU verification dropped cases")
    if not work["cases"]:
        require(work["iters"] == 0 and work["hash"] == "00000000" and not flag_list,
                "CPU tail work exists without candidate cases")
    if not work["source_endpoints"]:
        require(work == dict(source_endpoints=0, endpoints=0, cases=0, iters=0, hash="00000000", flagged=[]),
                "CPU work exists without source endpoints")
    if result["cpu_source_capture_enabled"]:
        require(result["captured_cpu_sources"] == work["source_endpoints"], "incomplete diagnostic capture")
    completed = fingerprint(result["cpu_groups"], "completed CPU groups")
    require(completed["count"] <= work["source_endpoints"], "CPU groups exceed source endpoints")
    integer(result["cpu_max_inflight_groups"], "maximum in-flight CPU groups")
    require(result["cpu_inflight_capacity_groups"] == cfg["queue_capacity_groups"]
            and result["cpu_max_inflight_groups"] <= result["cpu_inflight_capacity_groups"],
            "CPU in-flight capacity exceeded or changed")
    require((completed["count"] == 0) == (result["cpu_max_inflight_groups"] == 0),
            "CPU in-flight telemetry differs from admitted work")
    integer(result["startup_gpu_endpoints_forwarded"], "startup forwarded endpoints")
    routing = result["routing"]
    require((routing is not None) == variant["wrapper"], "routing wrapper differs from variant")
    if routing is None:
        require(work["source_endpoints"] == 0 and completed["count"] == 0, "unwrapped variant diverted CPU work")
        require(report["endpoints"] == tree["survived"][-1], "baseline lost original endpoints")
        return
    original = vector(routing["original_by_f"], "original endpoint classes")
    forwarded = vector(routing["gpu_by_f"], "forwarded endpoint classes")
    diverted = vector(routing["cpu_by_f"], "diverted endpoint classes")
    require(diverted == source_counts, "admitted CPU work differs from completed workers")
    require(original == [g+c for g,c in zip(forwarded, diverted)], "original endpoint partition differs")
    require(sum(original) == tree["survived"][-1] and sum(forwarded) == report["endpoints"],
            "original source totals differ from tree/report")
    all_groups = fingerprint(routing["groups"], "all groups")
    gpu_groups = fingerprint(routing["gpu_groups"], "forwarded groups")
    cpu_groups = fingerprint(routing["cpu_groups"], "admitted CPU groups")
    require(cpu_groups == completed, "CPU descriptor completion differs")
    require(add_fingerprints([gpu_groups, cpu_groups]) == all_groups, "descriptor partition count/XOR/sum differs")
    require(all_groups["count"] <= sum(original) and gpu_groups["count"] <= sum(forwarded), "too many descriptors")
    for key in ("eligible_groups", "selected_groups", "queue_full_groups", "startup_blocked_groups", "copied_descriptor_bytes",
                "successful_descriptor_bytes", "rejected_descriptor_bytes"):
        integer(routing[key], key)
    require(routing["selected_groups"] == cpu_groups["count"] + routing["queue_full_groups"] + routing["startup_blocked_groups"],
            "selected descriptor disposition differs")
    require(routing["selected_groups"] <= routing["eligible_groups"] <= all_groups["count"], "invalid descriptor eligibility")
    if variant["stride"] == 1:
        require(routing["selected_groups"] == routing["eligible_groups"], "stride-one routing omitted eligible candidates")
    # Node uses Rust's native layout, not the 64-byte GPU descriptor ABI.
    # Current Rust reorders its fields into 48 bytes; a declaration-order
    # layout would occupy 64. Infer and check that one supported size is used
    # consistently across the experiment, rather than relabel GPU bytes.
    parent_bytes = routing["successful_descriptor_bytes"] - 32*sum(diverted)
    require((not cpu_groups["count"] and parent_bytes == 0)
            or (cpu_groups["count"] > 0 and parent_bytes in (48*cpu_groups["count"],64*cpu_groups["count"])),
            "successful CPU descriptor bytes are inconsistent")
    require(routing["copied_descriptor_bytes"] == routing["successful_descriptor_bytes"] + routing["rejected_descriptor_bytes"]
            and routing["rejected_descriptor_bytes"] % 16 == 0, "copied CPU descriptor bytes are inconsistent")
    offset = cfg["gpu_m"] * 306 // 485 + 1
    for i, row in enumerate(report["gpu"]["classes"]):
        require(row["endpoints"] == forwarded[offset+i], "GPU class differs from original-depth forwarding")
    fallback = sum(value for f,value in enumerate(forwarded) if not offset <= f < offset+cfg["num_bv"])
    require(fallback == report["tail"]["endpoints"][-1], "original CPU fallback partition differs")
    require(all(not value or f in cfg["selected_original_f"] for f,value in enumerate(diverted)),
            "CPU received an ineligible original F class")
    require(all(not value or offset <= f < offset+cfg["num_bv"] for f,value in enumerate(diverted)),
            "CPU diversion changed the existing fallback ownership")
    if not variant["workers"]:
        require(sum(diverted) == 0 and routing["selected_groups"] == 0 and routing["copied_descriptor_bytes"] == 0,
                "route control performs offload work")


def exact_selected_signature(result):
    return {"remaining": work_signature(result["report"]), "cpu_native": result["cpu_native"]}


def audit_scope(rows, cfg, variants, roots):
    require(any(row["variant"] == "gpu" for row in rows), "scope lacks GPU baseline")
    baseline = next(row["result"] for row in rows if row["variant"] == "gpu")
    expected_tree = baseline["report"]["tree"]
    baseline_signature = work_signature(baseline["report"])
    original = None
    groups = None
    eligible_groups = None
    selected_by_stride = {}
    host_parent_size = None
    repeat_signatures = {}
    for row in rows:
        result = row["result"]
        name = row["variant"]
        variant = variants[name]
        audit_result(result, cfg, variant, roots)
        require(result["report"]["tree"] == expected_tree, "source tree differs across policies/repeats")
        if name in ("gpu", "route_control"):
            require(work_signature(result["report"]) == baseline_signature, "baseline/route-control exact work differs")
        route = result["routing"]
        if route is not None:
            require(original is None or original == route["original_by_f"], "original per-F corpus differs")
            require(groups is None or groups == route["groups"], "original descriptor corpus differs")
            require(eligible_groups is None or eligible_groups == route["eligible_groups"], "eligible descriptor corpus differs")
            original, groups = route["original_by_f"], route["groups"]
            eligible_groups = route["eligible_groups"]
            stride = variant["stride"]
            require(stride not in selected_by_stride or selected_by_stride[stride] == route["selected_groups"],
                    "deterministic descriptor candidate selection differs")
            selected_by_stride[stride] = route["selected_groups"]
            if route["cpu_groups"]["count"]:
                observed_size = (route["successful_descriptor_bytes"]-32*sum(route["cpu_by_f"])) // route["cpu_groups"]["count"]
                require(host_parent_size is None or host_parent_size == observed_size, "host parent layout differs within run")
                host_parent_size = observed_size
        offset = cfg["gpu_m"] * 306 // 485 + 1
        for i, (base, current) in enumerate(zip(baseline["report"]["gpu"]["classes"], result["report"]["gpu"]["classes"])):
            require(base["endpoints"] == current["endpoints"] + result["cpu_original_by_f"][offset+i],
                    "baseline != remaining GPU + diverted original endpoints")
            require(current["cases"] <= base["cases"] and current["iters"] <= base["iters"],
                    "remaining GPU work exceeds baseline")
        for key in ("endpoints", "cases", "iters"):
            require(result["report"]["tail"][key][-1] == baseline["report"]["tail"][key][-1],
                    "existing CPU fallback work changed")
        identity = digest({"a":variant["cpu_a"], "b":variant["cpu_b"],
                           "groups":result["cpu_groups"], "counts":result["cpu_original_by_f"]})
        signature = exact_selected_signature(result)
        require(identity not in repeat_signatures or repeat_signatures[identity] == signature,
                "same CPU policy/admitted descriptor identity changed exact work")
        repeat_signatures[identity] = signature
    return baseline, original, groups


def audit_rows(rows):
    require(rows and rows[0].get("type") == "config" and rows[-1].get("type") == "complete",
            "missing configuration or completion marker")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("type")].append(row)
    require(set(grouped) == {"config", "cohort", "gpu_setup", "warmup", "trial", "aggregate_config", "aggregate_trial", "complete"},
            "missing or unexpected pipeline record type")
    require(len(grouped["config"]) == len(grouped["complete"]) == len(grouped["gpu_setup"]) == len(grouped["aggregate_config"]) == 1,
            "duplicate pipeline configuration/completion/setup")
    cfg = grouped["config"][0]
    require(cfg["experiment"] == "direct_pipeline" and cfg["profile"] == "target", "expected target direct pipeline experiment")
    require(cfg["n"] == 72 and cfg["gpu_m"] == cfg["n"]-cfg["gpu_a"] and cfg["checkpoint"] == 38,
            "unexpected target/source configuration")
    for key in ("threads", "draws_per_cohort", "cohort_count", "repetitions", "aggregate_repetitions", "num_bv", "queue_capacity_groups"):
        integer(cfg[key], key, minimum=1)
    integer(cfg["startup_gpu_endpoint_guard"], "startup guard")
    integer(cfg["seed"], "seed", maximum=MASK64)
    require(cfg["gpu_a"] in (8,9,10) and cfg["cutoff"] >= cfg["checkpoint"]
            and cfg["cutoff"] <= cfg["terminal_entry"] < cfg["gpu_m"], "invalid expansion policy")
    require(len(cfg["gpu_class_b"]) == cfg["num_bv"], "invalid GPU windows")
    for b in cfg["gpu_class_b"]:
        integer(b, "GPU B", minimum=cfg["gpu_a"], maximum=39)
    require(cfg["selected_original_f"] and len(set(cfg["selected_original_f"])) == len(cfg["selected_original_f"]), "invalid selected F classes")
    offset = cfg["gpu_m"] * 306 // 485 + 1
    for f in cfg["selected_original_f"]:
        integer(f, "selected F", minimum=offset, maximum=offset+cfg["num_bv"]-1)
    variants = {value["name"]:variant_settings(value, cfg["gpu_a"]) for value in cfg["variants"]}
    require(len(variants) == len(cfg["variants"]) and {"gpu", "route_control"} <= variants.keys(),
            "missing controls or duplicate variant names")
    names = list(variants)
    complete = grouped["complete"][0]
    require(complete == dict(type="complete", cohorts=cfg["cohort_count"], variants=len(names),
                            repetitions=cfg["repetitions"], aggregate_repetitions=cfg["aggregate_repetitions"]),
            "completion marker differs from declared experiment")
    cohorts = sorted(grouped["cohort"], key=lambda row:row["cohort"])
    require([row["cohort"] for row in cohorts] == list(range(cfg["cohort_count"])), "missing/duplicate cohorts")
    require(len({row["seed"] for row in cohorts}) == len(cohorts), "cohort seeds repeat")
    cohort_hashes = []
    for row in cohorts:
        cohort_hashes.append(audit_cohort(row, cfg))
        for root in row["roots"]:
            audit_root(root, cfg["checkpoint"])
    all_roots = [root for row in cohorts for root in row["roots"]]
    aggregate_cfg = grouped["aggregate_config"][0]
    require(aggregate_cfg == dict(type="aggregate_config", drawn=cfg["draws_per_cohort"]*len(cohorts),
                                 accepted=len(all_roots), roots=all_roots), "aggregate corpus differs from saved cohort concatenation")
    finite(grouped["gpu_setup"][0]["wall_seconds"], "GPU setup")
    warmups = grouped["warmup"]
    require(sorted(row["variant"] for row in warmups) == sorted(names), "missing/duplicate warmup")
    audit_scope(warmups, cfg, variants, all_roots[:32])
    trial_keys = {(c,n,r) for c in range(len(cohorts)) for n in names for r in range(cfg["repetitions"])}
    actual_keys = [(row["cohort"], row["variant"], row["repetition"]) for row in grouped["trial"]]
    require(len(actual_keys) == len(set(actual_keys)) and set(actual_keys) == trial_keys, "missing/duplicate/unconfigured trial")
    cohort_baselines, cohort_original, cohort_groups = [], [], []
    for cohort in cohorts:
        selected = [row for row in grouped["trial"] if row["cohort"] == cohort["cohort"]]
        for row in selected:
            require(row["drawn"] == cfg["draws_per_cohort"], "trial changed draw denominator")
            integer(row["order_position"], "trial order position", maximum=len(names)-1)
        for rep in range(cfg["repetitions"]):
            require(sorted(row["order_position"] for row in selected if row["repetition"] == rep) == list(range(len(names))), "duplicate/missing variant order slot")
        baseline, original, descriptors = audit_scope(selected, cfg, variants, cohort["roots"])
        cohort_baselines.append(baseline["report"])
        cohort_original.append(original)
        cohort_groups.append(descriptors)
    agg_keys = [(row["variant"], row["repetition"]) for row in grouped["aggregate_trial"]]
    require(len(agg_keys) == len(set(agg_keys)) and set(agg_keys) == {(n,r) for n in names for r in range(cfg["aggregate_repetitions"])},
            "missing/duplicate/unconfigured aggregate trial")
    for row in grouped["aggregate_trial"]:
        require(row["drawn"] == len(cohorts)*cfg["draws_per_cohort"], "aggregate changed draw denominator")
        integer(row["order_position"], "aggregate order position", maximum=len(names)-1)
    for rep in range(cfg["aggregate_repetitions"]):
        require(sorted(row["order_position"] for row in grouped["aggregate_trial"] if row["repetition"] == rep) == list(range(len(names))), "duplicate/missing aggregate order slot")
    baseline, original, descriptors = audit_scope(grouped["aggregate_trial"], cfg, variants, all_roots)
    require(work_signature(baseline["report"]) == aggregate_expected(cohort_baselines), "aggregate baseline differs from cohort count sums/checksum XOR")
    require(original == [sum(values) for values in zip(*cohort_original)], "aggregate original endpoint classes differ from cohort sum")
    require(descriptors == add_fingerprints(cohort_groups), "aggregate original descriptors differ from cohort count/sum/XOR")
    return {"config":cfg, "variants":variants, "rows":rows,
            "cohort_corpus_sha256":cohort_hashes, "aggregate_corpus_sha256":digest(all_roots),
            "baseline_signature_sha256":digest(work_signature(baseline["report"])),
            "baseline_draws":len(cohorts)*cfg["draws_per_cohort"], "accepted_draws":len(all_roots)}


def spread(values):
    values = list(values)
    return dict(min=min(values), mean=statistics.mean(values), median=statistics.median(values), max=max(values))


def summarize_measurements(rows, baseline):
    records = []
    base_gpu = baseline["report"]["gpu"]["classes"]
    base_totals = {k:sum(c[k] for c in base_gpu) for k in ("endpoints", "cases", "iters")}
    for row in rows:
        result = row["result"]
        gpu = result["report"]["gpu"]
        metrics = {key:result[key] for key in TIMINGS}
        metrics.update(average_busy_cpu_cores=result["average_busy_cpu_cores"],
                       cpu_original_endpoints=result["cpu_native"]["source_endpoints"],
                       cpu_descendant_endpoints=result["cpu_native"]["endpoints"],
                       cpu_native_cases=result["cpu_native"]["cases"], cpu_native_iterations=result["cpu_native"]["iters"],
                       cpu_flags=len(result["cpu_native"]["flagged"]), gpu_kernel_event_seconds=gpu["kernel_seconds"],
                       gpu_input_records=gpu["input_records"], gpu_transfer_bytes=gpu["transfer_bytes"])
        route = result["routing"]
        for key in ("eligible_groups", "selected_groups", "queue_full_groups", "startup_blocked_groups", "copied_descriptor_bytes",
                    "successful_descriptor_bytes", "rejected_descriptor_bytes"):
            metrics[key] = route[key] if route else 0
        metrics["admitted_cpu_groups"] = result["cpu_groups"]["count"]
        metrics["cpu_max_inflight_groups"] = result["cpu_max_inflight_groups"]
        metrics["average_busy_cpu_worker_cores"] = result["cpu_worker_seconds"] / result["wall_seconds"]
        selected = route["selected_groups"] if route else 0
        for label,key in (("admitted", "admitted_cpu_groups"), ("queue_full", "queue_full_groups"),
                          ("startup_blocked", "startup_blocked_groups")):
            metrics["cpu_"+label+"_fraction_of_selected_groups"] = metrics[key]/selected if selected else 0
        for key in ("endpoints", "cases", "iters"):
            remaining = sum(c[key] for c in gpu["classes"])
            removed = base_totals[key] - remaining
            metrics["gpu_remaining_"+key] = remaining
            metrics["gpu_original_"+key+"_removed"] = removed
            metrics["gpu_original_"+key+"_removed_fraction"] = removed/base_totals[key] if base_totals[key] else 0
        offset = (result["report"]["n"]-result["report"]["a"])*306//485+1
        class_changes = [dict(f=offset+i,
                              original={key:base[key] for key in ("endpoints","cases","iters")},
                              remaining={key:current[key] for key in ("endpoints","cases","iters")},
                              removed={key:base[key]-current[key] for key in ("endpoints","cases","iters")},
                              kernel_event_seconds=current["kernel_seconds"])
                         for i,(base,current) in enumerate(zip(base_gpu,gpu["classes"]))]
        records.append({"repetition":row["repetition"], "cohort":row.get("cohort"), "metrics":metrics,
                        "cpu_original_by_f":result["cpu_original_by_f"], "cpu_groups":result["cpu_groups"],
                        "gpu_class_changes":class_changes,
                        "native_signature_sha256":digest(exact_selected_signature(result))})
    return {"samples":len(records), "metrics":{key:spread(r["metrics"][key] for r in records) for key in records[0]["metrics"]}, "measurements":records}


def summarize_run(path):
    path = path / "pipeline.jsonl" if path.is_dir() else path
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    result = audit_rows(rows)
    cfg = result["config"]
    metadata_path = path.parent / "metadata.json"
    metadata_raw = metadata_path.read_bytes()
    metadata = json.loads(metadata_raw)
    require(metadata.get("state") == "complete" and metadata.get("exit_status") == 0, "runner has not completed successfully")
    alias = common_config(cfg, [])
    alias["variants"] = list(result["variants"])
    audit_metadata(metadata, alias)
    require(metadata.get("arguments", {}).get("profile", "target") == "target", "runner profile differs")
    for env,key in (("CV_DIRECT_QUEUE","queue_capacity_groups"), ("CV_DIRECT_STARTUP_ENDPOINTS","startup_gpu_endpoint_guard")):
        if env in metadata.get("direct_environment", {}):
            require(int(metadata["direct_environment"][env]) == cfg[key], "runner direct environment differs")
    if "CV_DIRECT_CLASSES" in metadata.get("direct_environment", {}):
        require([int(x) for x in metadata["direct_environment"]["CV_DIRECT_CLASSES"].split(",")] == cfg["selected_original_f"], "runner selected F classes differ")
    for relative, expected in metadata.get("source_sha256", {}).items():
        source = path.parent / "source" / relative
        require(source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == expected, f"source snapshot hash differs: {relative}")
    require(re.fullmatch(r"[0-9a-f]{64}", metadata.get("binary_sha256", "")) is not None, "missing binary identity")
    aggregates = [row for row in rows if row["type"] == "aggregate_trial"]
    baseline = next(row["result"] for row in aggregates if row["variant"] == "gpu")
    summaries = []
    for name, policy in result["variants"].items():
        continuous = summarize_measurements([row for row in aggregates if row["variant"] == name], baseline)
        cohort_summaries = []
        for c in range(cfg["cohort_count"]):
            trials = [row for row in rows if row["type"] == "trial" and row["cohort"] == c]
            control = next(row["result"] for row in trials if row["variant"] == "gpu")
            cohort_summaries.append(summarize_measurements([row for row in trials if row["variant"] == name], control))
        summaries.append(dict(name=name, policy=policy, continuous=continuous, cohorts=cohort_summaries))
    by_name = {row["name"]:row for row in summaries}
    for row in summaries:
        row["continuous_ratio"] = {control:row["continuous"]["metrics"]["wall_seconds"]["mean"] / by_name[control]["continuous"]["metrics"]["wall_seconds"]["mean"] for control in ("gpu", "route_control")}
        row["continuous_median_ratio"] = {control:row["continuous"]["metrics"]["wall_seconds"]["median"] / by_name[control]["continuous"]["metrics"]["wall_seconds"]["median"] for control in ("gpu", "route_control")}
        row["paired_continuous_wall_ratios"] = {}
        measured = {sample["repetition"]:sample["metrics"]["wall_seconds"] for sample in row["continuous"]["measurements"]}
        for control in ("gpu", "route_control"):
            control_times = {sample["repetition"]:sample["metrics"]["wall_seconds"] for sample in by_name[control]["continuous"]["measurements"]}
            row["paired_continuous_wall_ratios"][control] = [dict(repetition=rep,ratio=measured[rep]/control_times[rep]) for rep in sorted(measured)]
    return {key:value for key,value in result.items() if key not in ("rows", "variants")} | {
        "directory":str(path.parent), "jsonl_sha256":hashlib.sha256(raw).hexdigest(),
        "metadata_sha256":hashlib.sha256(metadata_raw).hexdigest(), "metadata":metadata,
        "audited":True, "captured_sources_in_timed_trials":0,
        "variants":summaries, "notes":NOTES,
        "telemetry":str(path.parent / "gpu.csv"),
        "source_snapshot_hashes_checked":len(metadata.get("source_sha256", {})),
    }


def markdown(result):
    lines = ["# CPU routing before GPU dispatch", "", *[note+"\n" for note in NOTES]]
    for run in result["runs"]:
        lines += [f"## {run['directory']}", "", f"{run['baseline_draws']} baseline draws, {run['accepted_draws']} accepted; all source-partition and work-accounting audits passed.", "",
                  "| Variant | Continuous wall s, mean [min,max] | vs GPU | vs route control | CPU core-s | CPU worker core-s | Drain ms | Original endpoints diverted | GPU iterations removed |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in run["variants"]:
            m = row["continuous"]["metrics"]
            wall = m["wall_seconds"]
            lines.append(f"| {row['name']} | {wall['mean']:.4f} [{wall['min']:.4f}, {wall['max']:.4f}] | {row['continuous_ratio']['gpu']:.4f} | {row['continuous_ratio']['route_control']:.4f} | {m['process_cpu_seconds']['mean']:.3f} | {m['cpu_worker_seconds']['mean']:.3f} | {1000*m['cpu_drain_seconds']['mean']:.3f} | {m['cpu_original_endpoints']['mean']:,.0f} | {100*m['gpu_original_iters_removed_fraction']['mean']:.5f}% |")
        lines += [""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = {"runs":[summarize_run(path) for path in args.paths]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.out.with_suffix(".md").write_text(markdown(result))


if __name__ == "__main__":
    main()
