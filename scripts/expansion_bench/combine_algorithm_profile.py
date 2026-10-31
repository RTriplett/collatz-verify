#!/usr/bin/env python3
"""Combine audited CPU/CUDA traces and a separate GPU phase diagnostic.

Each resource retains its original units and measurement source. In particular,
instrumented lane clocks never become subdivisions of ordinary kernel seconds.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from flame_offload import validate


def read_audited(path):
    data = json.loads(path.read_text())
    for field in ("correctness_audit_passed", "matches_confirmation_work"):
        if data.get(field) is not True:
            raise ValueError(f"{path}: missing successful {field}")
    if not data.get("aggregate_corpus_sha256"):
        raise ValueError(f"{path}: missing corpus identity")
    return data


def label_gpu_stages(node):
    labels = {
        "Parent expansion": "Step 1 · GPU endpoint expansion",
        "Compact-chain expansion": "Step 1 · GPU grouped expansion",
        "Candidate screening and tail verification": "Steps 2 + 3 · fused verifier",
        "Copies": "Data transfers",
        "H2D": "Host → GPU",
        "D2H": "GPU → host",
    }
    node["name"] = labels.get(node["name"], node["name"])
    for child in node.get("children", []):
        label_gpu_stages(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path,
                        default=Path("results/tail_offload/cpu_profile/algorithm_trees.json"))
    parser.add_argument("--phases", type=Path,
                        default=Path("results/tail_offload/gpu_phase_profile/phase_trees.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/tail_offload/algorithm_flamegraph/data.json"))
    args = parser.parse_args()
    result = copy.deepcopy(read_audited(args.cpu))
    phases = read_audited(args.phases)
    if result["aggregate_corpus_sha256"] != phases["aggregate_corpus_sha256"]:
        raise ValueError("CPU trace and GPU diagnostic use different input corpora")
    by_variant = {item["variant"]: item for item in phases["variants"]}
    if len(by_variant) != len(phases["variants"]):
        raise ValueError("duplicate diagnostic variant")
    if set(by_variant) != {item["variant"] for item in result["variants"]}:
        raise ValueError("CPU trace and GPU diagnostic compare different variants")
    for item in result["variants"]:
        label_gpu_stages(item["gpu"])
        for stage in item["cpu"].get("children", []):
            if stage["name"] == "CPU tree expansion and pruning":
                stage["name"] = "Step 1 · CPU tree expansion and pruning"
            elif stage["name"] == "CPU tail continuations":
                stage["name"] = "Step 3 · CPU tail continuations"
        diagnostic = by_variant[item["variant"]]
        item["gpu_phases"] = diagnostic["gpu_phases"]
        captures = diagnostic.get("captures", [])
        description = item["gpu_phases"].get("description", "")
        description += (
            f" Combined across {len(captures)} warm aggregate repetitions; the CPU and CUDA activity rows use one capture per variant."
        )
        ratio = diagnostic.get("wall_ratio_to_fresh_control")
        if ratio is not None:
            description += f" Instrumented elapsed runtime was {(ratio - 1) * 100:.1f}% higher than the fresh uninstrumented control."
        item["gpu_phases"]["description"] = description.strip()
        item["gpu_phase_diagnostic"] = {
            key: value for key, value in diagnostic.items()
            if key not in ("variant", "gpu_phases")
        }
    result["notes"] = list(dict.fromkeys(result.get("notes", []) + [
        "Step 1 expands and prunes low-bit residue classes from the stored K38 frontier; its terminal expansion runs on the GPU.",
        "Step 2 screens the remaining completions using B-step lookahead bitvectors, mod-9 masks and carried OEE state.",
        "Step 3 iterates surviving integer trajectories until descent, with CPU continuation only in the after experiment.",
        "CPU sample counts, summed CUDA-operation seconds and diagnostic GPU lane clocks use separate scales. None is added to elapsed runtime.",
        "GPU phase clocks come from a separate instrumented build. They include stalls, scheduling and instrumentation, and are not a decomposition of the ordinary kernel's measured seconds.",
        "This is steady-state sampled N72 work after K38; frontier construction, checkpoint walkback and one-time lookup-table construction are outside the captured range.",
        "The GPU's pm-infinity and mixed-word lookahead credits are precomputed in the bitvectors, so they have no separate live GPU-check timing.",
    ]))
    result["glossary"] = {
        "A": "Remaining high bits: each depth-M endpoint represents 2^A completions, with M = N - A. Here A = 10 and M = 62.",
        "B": "Lookahead depth and log2 bitvector size; varies by odd-step class in this comparison.",
        "pm-infinity": "A merge test using the maximal valid chain of inverse odd steps.",
        "Mixed-word checks": "Precomputed inverse parity-word certificates used to reject classes that merge with a smaller start.",
        "OEE": "An odd run followed by two even steps can certify a merge with a smaller start.",
    }
    result["combined_sources"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.cpu, args.phases)
    }
    validate(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Deep sampled stacks make indented JSON mostly whitespace; the interactive
    # viewer exposes the readable hierarchy without bloating the saved artifact.
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    temporary.replace(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
