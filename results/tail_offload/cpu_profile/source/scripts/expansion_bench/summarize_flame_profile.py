#!/usr/bin/env python3
"""Extract additive device-stage trees from aggregate-only Nsight traces.

This is measured activity attribution, not sampled call stacks. Device event
durations overlap across streams; their sum is not pipeline elapsed time.
CPU core-seconds are reported separately and include tracing overhead.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

from summarize_fullrange import summarize, work_signature


def require(condition, message):
    if not condition:
        raise ValueError(message)


def node(name, children=None, **extra):
    children = children or []
    result = {"name": name, **extra}
    if children:
        result["children"] = children
        result["seconds"] = sum(child["seconds"] for child in children)
        result["operation_count"] = sum(child.get("operation_count", 0) for child in children)
    return result


def stream_leaves(rows):
    grouped = defaultdict(list)
    for row in rows:
        require(row["end"] >= row["start"], "negative CUDA activity duration")
        grouped[row["streamId"]].append(row)
    return [node(f"Stream {stream}", seconds=sum(row["end"] - row["start"] for row in group) / 1e9,
                 operation_count=len(group), stream_id=stream,
                 transfer_bytes=sum(row.get("bytes", 0) for row in group))
            for stream, group in sorted(grouped.items())]


def union_seconds(rows):
    spans = sorted((row["start"], row["end"]) for row in rows)
    total = 0
    left = right = None
    for start, end in spans:
        if right is None:
            left, right = start, end
        elif start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    if right is not None:
        total += right - left
    return total / 1e9


def extract_trace(path):
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    def rows(table):
        return [dict(row) for row in db.execute(f"SELECT * FROM {table}")] if table in tables else []
    kernels = rows("CUPTI_ACTIVITY_KIND_KERNEL")
    copies = rows("CUPTI_ACTIVITY_KIND_MEMCPY")
    clears = rows("CUPTI_ACTIVITY_KIND_MEMSET")
    require(kernels, f"{path}: no device kernels in captured range")
    grouped = defaultdict(list)
    for row in kernels:
        raw = strings[row.get("demangledName", row["shortName"]) ]
        if "cv_expand_parents" in raw:
            label = "Parent expansion"
        elif "cv_expand_chain" in raw:
            label = "Compact-chain expansion"
        elif "cv_kernel" in raw:
            label = "Candidate screening and tail verification"
        else:
            label = raw
        grouped[label].append({**row, "raw_name": raw})
    compute = node("GPU compute", [node(label, stream_leaves(group),
                    raw_kernel_names=sorted({row["raw_name"] for row in group}))
                    for label, group in sorted(grouped.items())])
    directions = {1: "H2D", 2: "D2H", 3: "H2A", 4: "A2H", 5: "A2A",
                  6: "A2D", 7: "D2A", 8: "D2D", 9: "H2H", 10: "P2P",
                  11: "Unified H2D", 12: "Unified D2H", 13: "Unified D2D"}
    grouped = defaultdict(list)
    for row in copies:
        grouped[row["copyKind"]].append(row)
    copy_nodes = [node(directions.get(kind, f"Copy kind {kind}"), stream_leaves(group),
                       transfer_direction=directions.get(kind, f"kind_{kind}"),
                       copy_kind=kind, transfer_bytes=sum(row["bytes"] for row in group))
                  for kind, group in sorted(grouped.items())]
    children = [compute]
    if copies:
        children.append(node("Copies", copy_nodes))
    if clears:
        children.append(node("Buffer clears", stream_leaves(clears),
                             transfer_bytes=sum(row["bytes"] for row in clears)))
    all_rows = kernels + copies + clears
    result = node("Device operations", children,
                  metric="sum_device_operation_seconds",
                  activity_union_seconds=union_seconds(all_rows),
                  kernel_union_seconds=union_seconds(kernels),
                  trace_span_seconds=(max(row["end"] for row in all_rows) -
                                      min(row["start"] for row in all_rows)) / 1e9)
    require(abs(result["seconds"] - sum(row["end"] - row["start"] for row in all_rows) / 1e9) < 1e-8,
            "tree duration differs from raw device activities")
    db.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, default=Path("results/tail_offload/flame_profile"))
    parser.add_argument("--confirmation", type=Path, default=Path("results/tail_offload/confirmation"))
    args = parser.parse_args()
    directory = args.profile_dir
    # Existing strict benchmark auditor checks all per-cohort/aggregate work,
    # flags, variant settings, transfer counts, and bounded-queue accounting.
    audited = summarize(directory)
    previous = summarize(args.confirmation)
    require(audited["aggregate_corpus_sha256"] == previous["aggregate_corpus_sha256"],
            "profiled corpus differs from the saved confirmation corpus")
    rows = [json.loads(line) for line in (directory / "pipeline.jsonl").read_text().splitlines()]
    previous_rows = [json.loads(line) for line in (args.confirmation / "pipeline.jsonl").read_text().splitlines()]
    aggregates = [row for row in rows if row["type"] == "aggregate_trial"]
    traces = sorted(directory.glob("direct_trace*.sqlite"))
    require(len(traces) == len(aggregates) == 2, "expected one captured aggregate per before/after variant")
    variants = []
    for index, (trace, row) in enumerate(zip(traces, aggregates)):
        prior = [item for item in previous_rows if item.get("type") == "aggregate_trial"
                 and item["variant"] == row["variant"]]
        require(prior and all(work_signature(item["report"]) == work_signature(row["report"])
                              for item in prior), "profiled aggregate work differs from confirmation")
        gpu = extract_trace(trace)
        report_gpu = row["report"]["gpu"]
        launches = sum(item["launches"] for item in report_gpu["classes"])
        compute = next(child for child in gpu["children"] if child["name"] == "GPU compute")
        require(len(compute["children"]) == 2 and all(child["operation_count"] == launches
                for child in compute["children"]), "capture omitted or added verifier/expansion launches")
        copies = next(child for child in gpu["children"] if child["name"] == "Copies")
        copies = {child["transfer_direction"]: child for child in copies["children"]}
        require(set(copies) == {"H2D", "D2H"}, "unexpected device-copy direction")
        require(copies["H2D"]["transfer_bytes"] == report_gpu["transfer_bytes"],
                "captured inputs differ from benchmark input bytes")
        require(report_gpu["flags"] == 0, "flag-copy accounting is outside this profile comparison")
        # One 24-byte KOut per launch; offload also returns a 4-byte count.
        expected_d2h = launches * (28 if report_gpu["tuning"].get("tail_budget", 0) else 24)
        expected_d2h += report_gpu["deferred_transfer_bytes"]
        require(copies["D2H"]["transfer_bytes"] == expected_d2h,
                "captured output bytes differ from results/counts/continuations")
        deferred = report_gpu["deferred_cpu_seconds"]
        other = row["process_cpu_seconds"] - deferred
        require(other >= 0, "continuation CPU time exceeds process CPU")
        cpu = node("CPU busy time", [node("Tail continuations", seconds=deferred),
                                    node("Other host work", seconds=other)],
                   metric="cpu_core_seconds")
        variants.append({"variant": row["variant"], "label": "Before" if index == 0 else "After",
                         "trace_file": trace.name, "wall_seconds": row["wall_seconds"],
                         "process_cpu_seconds": row["process_cpu_seconds"],
                         "deferred_cpu_seconds": deferred, "gpu": gpu, "cpu": cpu,
                         "trace_accounting": {"verifier_launches": launches,
                              "input_transfer_bytes": report_gpu["transfer_bytes"],
                              "output_transfer_bytes": expected_d2h,
                              "device_activity_accounting_matches_report": True}})
    result = {"schema_version": 1, "metric": "sum_device_operation_seconds",
              "notes": ["Measured stages, not sampled call stacks.",
                        "Device-operation durations overlap across streams and with CPU execution; their sum is not elapsed time.",
                        "CPU core-seconds are separate from device-operation seconds and include activity-tracing overhead.",
                        "CUDA kernel internals are fused: screening, stepping and handoff bookkeeping are not timed separately.",
                        "One warm aggregate capture per variant on the identical saved 512-draw, 484-accepted-root corpus.",
                        "These captures explain attribution and do not replace unprofiled performance measurements."],
              "aggregate_corpus_sha256": audited["aggregate_corpus_sha256"],
              "correctness_audit_passed": True, "matches_confirmation_work": True,
              "variants": variants,
              "trace_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in traces}}
    destination = directory / "trace_trees.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    for variant in variants:
        print(variant["label"], variant["variant"], "wall", variant["wall_seconds"],
              "device-operation seconds", variant["gpu"]["seconds"],
              "CPU core-seconds", variant["cpu"]["seconds"])
    print(destination)


if __name__ == "__main__":
    main()
