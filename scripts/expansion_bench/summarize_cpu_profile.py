#!/usr/bin/env python3
"""Audit warm CPU samples and attribute real stacks to algorithm stages.

Nsight frame-pointer stacks are enriched with DWARF inline frames from the
captured executable. Each running-state sample is assigned once to its most
specific recognized algorithm owner. Widths remain samples, not seconds.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess

from summarize_fullrange import summarize, work_signature
from summarize_flame_profile import extract_trace


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_trace(path):
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    events = [dict(row) for row in db.execute("SELECT * FROM COMPOSITE_EVENTS ORDER BY id")]
    frames = defaultdict(list)
    for row in db.execute("SELECT * FROM SAMPLING_CALLCHAINS ORDER BY id,stackDepth"):
        frame = dict(row)
        frame.update(raw_function=strings[frame["symbol"]], module=strings[frame["module"]])
        frames[frame["id"]].append(frame)
    tids = {row["globalTid"] for row in events}
    names = {row["globalTid"]: strings[row["nameId"]]
             for row in db.execute("SELECT * FROM ThreadNames") if row["globalTid"] in tids}
    db.close()
    return events, frames, names


def resolve_inline(directory, metadata, traces):
    binary = Path(metadata["binary"])
    cache = directory / "inline_frames.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        require(saved["binary_sha256"] == metadata["binary_sha256"], "inline cache binary differs")
        return saved["frames"]
    require(hashlib.sha256(binary.read_bytes()).hexdigest() == metadata["binary_sha256"],
            "symbolized executable hash differs from profiled binary")
    bases = [int(line.split('-')[0], 16)
             for line in (directory / "process_maps.txt").read_text().splitlines()
             if str(binary) in line and line.split()[2] == "00000000"]
    require(len(bases) == 1, "missing or ambiguous executable load base")
    base = bases[0]
    addresses = set()
    for _, stacks, _ in traces:
        for stack in stacks.values():
            for frame in stack:
                if frame["module"] == str(binary) and frame["originalIP"] is not None:
                    # Sampled IP is exact; ancestor IP is a return address.
                    addresses.add(frame["originalIP"] - base - (frame["stackDepth"] != 0))
    addresses = sorted(addresses)
    require(addresses and addresses[0] >= 0, "invalid executable-relative sample addresses")
    command = ["addr2line", "-a", "-f", "-C", "-i", "-e", str(binary)]
    output = subprocess.check_output(command, input=''.join(hex(ip) + '\n' for ip in addresses), text=True)
    (directory / "addr2line.stdout.txt").write_text(output)
    resolved = {}
    blocks = re.split(r"(?m)^(0x[0-9a-f]+)\n", output)
    for index in range(1, len(blocks), 2):
        address, body = int(blocks[index], 16), blocks[index + 1].splitlines()
        require(len(body) % 2 == 0, "malformed addr2line output")
        resolved[str(address)] = [{"raw_function": body[i], "source": body[i + 1],
                                    "inline": i + 2 < len(body)}
                                   for i in range(0, len(body), 2) if body[i] != "??"]
    saved = {"binary_sha256": metadata["binary_sha256"], "executable_load_base": base,
             "command": command, "ancestor_return_address_adjustment": -1,
             "frames": {}}
    for _, stacks, _ in traces:
        for stack in stacks.values():
            for frame in stack:
                if frame["module"] == str(binary) and frame["originalIP"] is not None:
                    adjusted = frame["originalIP"] - base - (frame["stackDepth"] != 0)
                    key = f'{frame["originalIP"]}:{frame["stackDepth"] == 0}'
                    saved["frames"][key] = resolved[str(adjusted)]
    cache.write_text(json.dumps(saved, indent=2) + '\n')
    return saved["frames"]


def readable(function):
    function = re.sub(r"::h[0-9a-f]{16}$", "", function)
    return function.replace("_$LT$", "<").replace("$GT$", ">").replace("$u20$", " ").replace("..", "::")


def enrich(stack, resolved):
    result = []
    for frame in stack:
        key = f'{frame["originalIP"]}:{frame["stackDepth"] == 0}'
        inline = resolved.get(key) or []
        if inline:
            for location in inline:
                result.append({**location, "module": frame["module"], "original_ip": frame["originalIP"],
                               "attribution_method": "DWARF inline frame" if location["inline"]
                               else "frame-pointer stack, symbol resolved with DWARF"})
        else:
            result.append({"raw_function": frame["raw_function"], "module": frame["module"],
                           "original_ip": frame["originalIP"], "inline": False,
                           "unresolved": bool(frame["unresolved"]),
                           "attribution_method": "Nsight frame-pointer sample"})
    return result


def category(frames, thread_name):
    """Disjoint sample attribution; inspect leaf-to-root to find its owner."""
    if thread_name in ("[NSys]", "[NSys Comms]", "CUPTI worker thread"):
        return "Profiler overhead", "Injected profiling workers"
    if any("DeferredWorkers::new" in frame["raw_function"] for frame in frames):
        arithmetic = any("cv::tail::" in frame["raw_function"] or "/src/tail.rs:" in frame.get("source", "")
                         for frame in frames[:4])
        return "CPU tail continuations", "Exact integer tail arithmetic" if arithmetic else "Continuation worker bookkeeping"
    for frame in frames:
        name = readable(frame["raw_function"])
        if "libToolsInjection" in frame["module"] or "libcupti" in frame["module"]:
            return "Profiler overhead", "Trace interception / collection"
        if any(token in name for token in ("cv::tail::endpoint_work", "cv::tail::process_endpoint",
                                          "cv::tail::recheck_wide", "cv::tail::step_until_below")):
            return "CPU fallback and wide rechecks", "Integer verification"
        if "cv::gpu::Shared::push" in name or "cv::gpu::Pool::" in name:
            return "Host packing and queues", "Batch buffers and synchronization"
        if "cv::gpu::GpuSink" in name or "cv::tree::Sink" in name and "terminal_group" in name:
            return "Host packing and queues", "Parent descriptor construction"
        if "::terminal::Table::" in name or "::compact_chain::" in name:
            return "CPU tree expansion and pruning", "Reusable suffix tables"
        if "pm_inf" in name or "pm_threshold" in name:
            return "CPU tree expansion and pruning", "pm∞ certificate screening"
        if "MixTables" in name or "culls_r" in name:
            return "CPU tree expansion and pruning", "Mixed-path residue screening"
        if "::ternary::" in name or "cv::rules::" in name:
            return "CPU tree expansion and pruning", "Ternary and path-merge screening"
        if "cv::tree::Node::children" in name:
            return "CPU tree expansion and pruning", "Child construction and OEE pruning"
        if "::Affine::baseline_prefix" in name:
            return "CPU tree expansion and pruning", "Baseline tree extension K50–K52"
        if "::Affine::prefix" in name:
            return "CPU tree expansion and pruning", "Improved tree traversal K38–K50"
        if "cv_gpu_" in name or "cv::gpu::gpu_run" in name or "cv::gpu::GpuSession" in name:
            return "GPU dispatch and result collection", "Host launch / synchronization / result copies"
        if "libcuda" in frame["module"] or "libcudart" in frame["module"]:
            return "GPU dispatch and result collection", "CUDA driver / runtime"
    if frames and any(token in frames[0]["raw_function"] for token in ("memcpy", "memmove")):
        return "Unattributed host memory copies", "memcpy / memmove; caller unavailable"
    if not frames or all(frame.get("unresolved") or frame["raw_function"].startswith('[') for frame in frames):
        return "Unresolved or truncated stacks", "No recognized algorithm owner"
    return "Other host runtime", "Scheduling, allocation and runtime"


def empty_node(name, **details):
    return {"name": name, "samples": 0, "self_samples": 0, "_children": {}, **details}


def add_stack(root, stages, frames):
    current = root
    current["samples"] += 1
    for name in stages:
        key = (name, "stage")
        current = current["_children"].setdefault(key, empty_node(name, attribution_method="Disjoint algorithm-owner classification"))
        current["samples"] += 1
    # The captured stack is leaf-first; flame frames run caller to callee.
    for frame in reversed(frames):
        name = readable(frame["raw_function"])
        key = (name, frame["module"], frame.get("source", ""), frame.get("inline", False))
        detail = {key: value for key, value in frame.items() if key != "original_ip"}
        current = current["_children"].setdefault(key, empty_node(name, **detail))
        current["samples"] += 1
    current["self_samples"] += 1


def finalize(node):
    children = [finalize(child) for child in node.pop("_children").values()]
    require(node["samples"] == node["self_samples"] + sum(child["samples"] for child in children),
            "CPU stack tree samples are not conserved")
    if children:
        # Add explicit exclusive samples so every visible child's width is additive.
        if node["self_samples"]:
            children.append({"name": "[self]", "samples": node["self_samples"],
                             "self_samples": node["self_samples"], "attribution_method": "Sample stopped in this function"})
        node["children"] = sorted(children, key=lambda child: (-child["samples"], child["name"]))
    return node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, default=Path("results/tail_offload/cpu_profile"))
    parser.add_argument("--confirmation", type=Path, default=Path("results/tail_offload/confirmation"))
    args = parser.parse_args()
    directory = args.profile_dir
    metadata = json.loads((directory / "metadata.json").read_text())
    audit = summarize(directory)
    previous = summarize(args.confirmation)
    require(audit["aggregate_corpus_sha256"] == previous["aggregate_corpus_sha256"], "corpus differs from confirmation")
    rows = [json.loads(line) for line in (directory / "pipeline.jsonl").read_text().splitlines()]
    prior = [json.loads(line) for line in (args.confirmation / "pipeline.jsonl").read_text().splitlines()]
    aggregates = [row for row in rows if row["type"] == "aggregate_trial"]
    paths = sorted(directory.glob("cpu_trace.*.sqlite"))
    require(len(paths) == len(aggregates) == 2, "expected two aggregate CPU profiles")
    traces = [read_trace(path) for path in paths]
    resolved = resolve_inline(directory, metadata, traces)
    pid = int((directory / "process_pid.txt").read_text())
    variants = []
    for index, (path, row, (events, stacks, threads)) in enumerate(zip(paths, aggregates, traces)):
        expected = [item for item in prior if item.get("type") == "aggregate_trial" and item["variant"] == row["variant"]]
        require(expected and all(work_signature(item["report"]) == work_signature(row["report"]) for item in expected),
                "profiled work differs from confirmation")
        cpu = empty_node("CPU algorithm work", metric="cpu_samples")
        coverage = Counter()
        stages = Counter()
        folded = Counter()
        sampled_threads = Counter()
        for event in events:
            require((event["globalTid"] >> 24) & 0xffffff == pid, "sample outside launched benchmark process")
            if event["threadState"] != 1:
                coverage["excluded_nonrunning_samples"] += 1
                continue
            raw_stack = stacks[event["id"]]
            frames = enrich(raw_stack, resolved)
            stage = category(frames, threads.get(event["globalTid"], ""))
            add_stack(cpu, stage, frames)
            stages[stage[0]] += 1
            sampled_threads[threads.get(event["globalTid"], "unknown")] += 1
            folded[';'.join(stage + tuple(readable(frame["raw_function"]).replace(';', ',') for frame in reversed(frames)))] += 1
            coverage["running_samples"] += 1
            coverage["samples_with_stack"] += bool(raw_stack)
            coverage["samples_with_inline_frames"] += any(frame.get("inline") for frame in frames)
            coverage["samples_with_broken_backtrace"] += any("Broken backtraces" in frame["raw_function"] for frame in frames)
            coverage["samples_reaching_thread_entry"] += any(
                "start_thread" in frame["raw_function"] or "Thread::new::thread_start" in frame["raw_function"]
                for frame in frames)
            coverage["samples_with_unresolved_leaf"] += bool(raw_stack and raw_stack[0]["unresolved"])
            coverage["samples_with_kernel_leaf"] += bool(raw_stack and raw_stack[0]["kernelMode"])
        cpu = finalize(cpu)
        require(cpu["samples"] == sum(stages.values()) == coverage["running_samples"], "CPU sample totals differ")
        label = "before" if index == 0 else "after"
        (directory / f"{label}.folded").write_text(''.join(f'{stack} {count}\n' for stack, count in sorted(folded.items())))
        gpu = extract_trace(path)
        reported_gpu = row["report"]["gpu"]
        launches = sum(item["launches"] for item in reported_gpu["classes"])
        compute = next(child for child in gpu["children"] if child["name"] == "GPU compute")
        require(len(compute["children"]) == 2 and all(child["operation_count"] == launches
                for child in compute["children"]), "capture omitted or added verifier/expansion launches")
        copies = next(child for child in gpu["children"] if child["name"] == "Copies")
        copies = {child["transfer_direction"]: child for child in copies["children"]}
        require(set(copies) == {"H2D", "D2H"}, "unexpected device-copy direction")
        require(copies["H2D"]["transfer_bytes"] == reported_gpu["transfer_bytes"],
                "captured inputs differ from benchmark input bytes")
        require(reported_gpu["flags"] == 0, "flag-copy accounting is outside this profile comparison")
        expected_d2h = launches * (28 if reported_gpu["tuning"].get("tail_budget", 0) else 24)
        expected_d2h += reported_gpu["deferred_transfer_bytes"]
        require(copies["D2H"]["transfer_bytes"] == expected_d2h,
                "captured output bytes differ from results/counts/continuations")
        variants.append({"variant": row["variant"], "label": label.capitalize(), "trace_file": path.name,
                         "wall_seconds": row["wall_seconds"], "process_cpu_seconds": row["process_cpu_seconds"],
                         "deferred_cpu_seconds": row["report"]["gpu"]["deferred_cpu_seconds"],
                         "cpu": cpu, "gpu": gpu, "cpu_sampling_coverage": dict(coverage),
                         "cpu_stage_samples": dict(stages), "sampled_thread_names": dict(sampled_threads),
                         "trace_accounting": {"verifier_launches": launches,
                              "input_transfer_bytes": reported_gpu["transfer_bytes"],
                              "output_transfer_bytes": expected_d2h,
                              "device_activity_accounting_matches_report": True}})
    result = {"schema_version": 1, "metric": "separate_cpu_samples_and_device_operation_seconds",
              "notes": ["CPU widths are actual running-state instruction-pointer samples; no conversion to seconds.",
                        "Algorithm categories assign each sample once using its most specific recognized application owner.",
                        "Expandable stacks are measured frame-pointer backtraces enriched with separately tagged DWARF inline frames.",
                        "Nsight marks many stacks broken beyond libc thread entry; coverage separately reports samples that reach thread entry and unresolved leaves.",
                        "GPU widths are summed operation durations from the same captures; concurrent streams overlap.",
                        "This is one instrumented capture per variant, not a new performance estimate.",
                        "Warm aggregate scope excludes checkpoint/frontier construction, sampling, resident lookahead arrays and table preparation."],
              "sampling_configuration": metadata["sampling"],
              "aggregate_corpus_sha256": audit["aggregate_corpus_sha256"],
              "correctness_audit_passed": True, "matches_confirmation_work": True,
              "setup_context": next(row for row in rows if row["type"] == "gpu_setup"),
              "variants": variants}
    # Deep real/inline stacks make indentation much larger than the data.
    (directory / "algorithm_trees.json").write_text(json.dumps(result, separators=(',', ':')) + '\n')
    for variant in variants:
        print(variant["label"], variant["cpu"]["samples"], "samples", variant["cpu_stage_samples"])
    print(directory / "algorithm_trees.json")


if __name__ == "__main__":
    main()
