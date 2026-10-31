#!/usr/bin/env python3
"""Audit and summarize a completed class_split_bench run.

Usage: summarize_class_split.py --out SUMMARY.json RUN_DIRECTORY
The Markdown companion is written beside SUMMARY.json. Timing ratios compare
separate blocks; repetition indices are not paired across A values.
"""

import argparse
import json
import math
from pathlib import Path
import re
from statistics import median


SHAPES = (("a10", 10), ("a9", 9), ("a8", 8), ("a10_end", 10))
KINDS = {"config", "validation", "warmup", "class_trial", "shape_summary", "probe_summary"}
COUNTS = ("endpoints", "cases", "iters", "flagged", "launches")
SCOPE = (
    "Serialized GPU kernel event seconds, including endpoint reconstruction and tail work. "
    "Host expansion, packing, transfer and array construction are excluded. This is a "
    "conditional M62/F40 sample with improved checks through K50 and no K38 walkback, "
    "not the earlier saved K38 corpus or an end-to-end mixed-A pipeline."
)
COMPARISON = (
    "A10, A9, A8 and the closing A10 control ran in separate blocks. Ratios compare "
    "medians, not paired repetitions. Each trial total sums the corresponding separately "
    "timed class runs. Observed timing ranges are not confidence intervals."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid {name}: {value!r}")
    return value


def seconds(value, name, positive=False):
    require(isinstance(value, (int, float)) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f"invalid {name}: {value!r}")
    return value


def close(actual, expected, name):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
            f"{name} differs: {actual!r} versus {expected!r}")


def distribution(values):
    values = list(values)
    require(bool(values), "empty timing distribution")
    return {"median": median(values), "minimum": min(values), "maximum": max(values),
            "count": len(values), "values": values}


def work_signature(work):
    return tuple(work[key] for key in (*COUNTS, "hash"))


def checked_work(work, name, one_class=False):
    for key in COUNTS:
        integer(work[key], f"{name}.{key}")
    require(isinstance(work["hash"], str) and re.fullmatch(r"[0-9A-F]{8}", work["hash"]),
            f"invalid {name}.hash")
    require(work["flagged"] == 0, f"flagged trajectories in {name}")
    seconds(work["kernel_seconds"], f"{name}.kernel_seconds", positive=work["endpoints"] > 0)
    if one_class:
        require(work["launches"] == (work["endpoints"] + (1 << 20) - 1) // (1 << 20),
                f"batch count differs in {name}")
    return work


def merge_work(parts):
    total = {key: sum(row[key] for row in parts) for key in COUNTS}
    checksum = 0
    for part in parts:
        checksum ^= int(part["hash"], 16)
    total["hash"] = f"{checksum:08X}"
    total["kernel_seconds"] = sum(row["kernel_seconds"] for row in parts)
    return total


def same_work(actual, expected, name, timing=False):
    require(work_signature(actual) == work_signature(expected), f"work differs: {name}")
    if timing:
        close(actual["kernel_seconds"], expected["kernel_seconds"], f"{name}.kernel_seconds")


def single(records, kind):
    selected = [row for row in records if row["type"] == kind]
    require(len(selected) == 1, f"expected exactly one {kind}, got {len(selected)}")
    return selected[0]


def summarize(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    require(metadata.get("state") == "complete" and metadata.get("exit_status") == 0,
            "run must have completed with exit status zero")
    records = [json.loads(line) for line in (directory / "pipeline.jsonl").read_text().splitlines()
               if line.strip()]
    require(bool(records), "empty log")
    require(all(row.get("type") in KINDS for row in records), "unexpected record type")
    config = single(records, "config")
    final = single(records, "probe_summary")
    require(records[0] == config and records[-1] == final, "config/final record order differs")
    require(config["experiment"] == "f40_serialized_gpu_probe", "wrong experiment")
    require((config["n"], config["source_m"], config["source_f"]) == (72, 62, 40),
            "expected N72 and original M62/F40")
    require(config["a_order"] == [10, 9, 8, 10], "A block order differs")
    require(config["class_b"] == [32, 35] and config["bv_mode"] == "Shared",
            "window configuration differs")
    require(config["window"] == "descent+pm1+oee+pm_inf+mix(3^13)", "window rules differ")
    require(config["cutoff_bits"] == 120 and config["tail_oee"] is True, "tail rules differ")
    count = integer(config["count"], "source count", 16)
    reps = integer(config["repetitions"], "repetitions", 1)
    integer(config["threads"], "threads", 1)
    args = metadata["arguments"]
    require(args["profile"] == "target", "wrong runner profile")
    for arg, key in (("roots", "count"), ("repetitions", "repetitions"),
                     ("threads", "threads"), ("seed", "seed")):
        require(args[arg] == config[key], f"runner/config disagreement: {arg}")
    corpus = config["corpus"]
    require(corpus["different_from_k38_corpus"] is True, "conditional corpus not identified")
    require((corpus["source_m"], corpus["source_f"], corpus["cutoff"]) == (62, 40, 50),
            "sampling policy differs")
    require(corpus["accepted"] == count and corpus["seed"] == config["seed"],
            "corpus count or seed differs")
    require(corpus["draws_consumed"] == count + corpus["rejected_through_cutoff"] + corpus["duplicates"],
            "sampling accounting does not add up")
    require(re.fullmatch(r"[0-9A-F]{16}", corpus["ordered_n0_fnv1a64_le"]) is not None,
            "invalid source FNV identifier")
    examples = corpus["first_16_sources"]
    require(len(examples) == 16 and len({node["n0"] for node in examples}) == 16,
            "expected 16 distinct audit sources")
    require(all(node["k"] == 62 and node["f"] == 40 and 0 <= int(node["n0"]) < (1 << 62)
                for node in examples), "invalid audit source state")

    shapes = [row for row in records if row["type"] == "shape_summary"]
    require(len(shapes) == 4, "expected four shape summaries")
    require(final["blocks"] == shapes, "final embedded summaries differ from emitted shapes")
    active = [row for row in records if row["type"] not in ("config", "probe_summary")]
    require([row["block"] for row in active] == sorted(row["block"] for row in active),
            "shape records are interleaved or out of order")
    require(set(row["block"] for row in active) == set(range(4)), "unexpected shape block")
    blocks = []
    validation_count = 0
    for block, (label, a) in enumerate(SHAPES):
        m = 72 - a
        offset = m * 306 // 485 + 1
        cutoffs = [485 * offset - 306 * m + 485 * idx for idx in range(2)]
        selected = [row for row in active if row["block"] == block]
        shape = single(selected, "shape_summary")
        require(selected[-1] == shape, f"{label}: shape summary is not last")
        for row in selected:
            require((row["label"], row["a"], row["m"]) == (label, a, m),
                    f"{label}: inconsistent shape identity")
        require(shape["offset"] == offset and shape["class_b"] == [32, 35]
                and shape["cutoffs"] == cutoffs, f"{label}: shape window parameters differ")
        require(shape["source_count"] == count, f"{label}: source count differs")
        expected_classes = {0} if a == 10 else {0, 1}
        class_records = [row for row in selected if row["type"] != "shape_summary"]
        require(set(row["class"] for row in class_records) == expected_classes,
                f"{label}: class coverage differs")
        require([row["class"] for row in class_records]
                == sorted(row["class"] for row in class_records), f"{label}: class order differs")
        per_class = []
        raw_by_class = {}
        for idx in sorted(expected_classes):
            rows = [row for row in class_records if row["class"] == idx]
            require(all(row["f"] == offset + idx for row in rows), f"{label}: final F differs")
            validation = single(rows, "validation")
            warmup = single(rows, "warmup")
            trials = [row for row in rows if row["type"] == "class_trial"]
            require([row["repetition"] for row in trials] == list(range(reps)),
                    f"{label} class {idx}: missing/duplicate/reordered repetitions")
            require(rows == [validation, warmup, *trials], f"{label}: class phase order differs")
            require(validation["cpu_gpu_equal"] is True and validation["source_endpoints"] == 16,
                    f"{label}: CPU/GPU validation failed or scope differs")
            audit_work = checked_work(validation["work"], f"{label} class {idx} audit", True)
            require(0 < audit_work["endpoints"] <= 16 * (1 << (m - 62)),
                    f"{label}: class audit has no endpoints or too many")
            validation_count += 1
            warm_work = checked_work(warmup["work"], f"{label} class {idx} warmup", True)
            require(warm_work["endpoints"] > 0, f"{label}: empty warmup class")
            for trial in trials:
                require(trial["b"] == config["class_b"][idx]
                        and trial["window_cutoff"] == cutoffs[idx], f"{label}: trial window differs")
                work = checked_work(trial["work"], f"{label} class {idx} trial", True)
                same_work(work, warm_work, f"{label} class {idx} warmup/trial")
            raw_by_class[idx] = trials
            per_class.append({"class": idx, "f": offset + idx, "b": config["class_b"][idx],
                              "window_cutoff": cutoffs[idx], "validation": validation,
                              "warmup": warmup, "trials": trials,
                              "kernel_seconds": distribution(row["work"]["kernel_seconds"]
                                                             for row in trials)})
        require(len(shape["trials"]) == reps and len(shape["kernel_seconds_by_repetition"]) == reps,
                f"{label}: summary repetition count differs")
        totals = []
        for rep in range(reps):
            merged = merge_work([raw_by_class[idx][rep]["work"] for idx in sorted(expected_classes)])
            summary_work = checked_work(shape["trials"][rep], f"{label} summary trial")
            same_work(merged, summary_work, f"{label} class sum, rep {rep}", timing=True)
            same_work(merged, shape["work"], f"{label} total work")
            close(merged["kernel_seconds"], shape["kernel_seconds_by_repetition"][rep],
                  f"{label} recorded total seconds")
            totals.append(merged)
        same_work(shape["work"], totals[0], f"{label} first trial summary", timing=True)
        times = distribution(work["kernel_seconds"] for work in totals)
        close(times["median"], shape["median_kernel_seconds"], f"{label} median")
        close(shape["endpoint_count_factor"], totals[0]["endpoints"] / count,
              f"{label} endpoint factor")
        if a == 10:
            require(totals[0]["endpoints"] == count, f"{label}: A10 endpoint count differs")
        for key in ("expansion_wall_seconds", "gpu_initialization_wall_seconds", "array_build_seconds"):
            seconds(shape[key], f"{label}.{key}")
        blocks.append({"block": block, "label": label, "a": a, "m": m,
                       "kernel_seconds": times, "work": totals[0], "trials": totals,
                       "classes": per_class, "endpoint_count_factor": shape["endpoint_count_factor"],
                       "candidate_retained_fraction": totals[0]["endpoints"] / count / (1 << (10 - a)),
                       "setup": {key: shape[key] for key in
                                 ("expansion_wall_seconds", "gpu_initialization_wall_seconds", "array_build_seconds")}})
    opening, closing = blocks[0], blocks[-1]
    same_work(opening["work"], closing["work"], "opening/closing A10 work")
    same_work(opening["classes"][0]["validation"]["work"], closing["classes"][0]["validation"]["work"],
              "opening/closing A10 audit work")
    pooled = distribution(opening["kernel_seconds"]["values"] + closing["kernel_seconds"]["values"])
    for block, shape in zip(blocks, shapes):
        measured = block["kernel_seconds"]["median"]
        comparisons = {}
        for name, reference in (("opening_a10", opening["kernel_seconds"]),
                                ("closing_a10", closing["kernel_seconds"]), ("pooled_a10", pooled)):
            ratio = measured / reference["median"]
            comparisons[name] = {"kernel_seconds_ratio": ratio, "kernel_seconds_saved_fraction": 1 - ratio}
        block["versus"] = comparisons
        close(shape["kernel_seconds_ratio_to_initial_a10"],
              comparisons["opening_a10"]["kernel_seconds_ratio"], f"{block['label']} reported ratio")
        block["cases_ratio_to_a10"] = block["work"]["cases"] / opening["work"]["cases"]
        block["iterations_ratio_to_a10"] = block["work"]["iters"] / opening["work"]["iters"]
    return {
        "directory": str(directory), "scope": SCOPE, "comparison_note": COMPARISON,
        "config": config, "validation_count": validation_count, "blocks": blocks,
        "a10_controls": {"opening": opening["kernel_seconds"], "closing": closing["kernel_seconds"],
                         "pooled": pooled,
                         "closing_to_opening_median_ratio": closing["kernel_seconds"]["median"] / opening["kernel_seconds"]["median"],
                         "observed_range_fraction_of_pooled_median": (pooled["maximum"] - pooled["minimum"]) / pooled["median"]},
        "provenance": {key: metadata.get(key) for key in
                       ("binary_sha256", "source_sha256", "affinity", "started_at", "finished_at", "command")},
        "runner_scope_note": "The executable config defines this microprobe's corpus and timing scope; generic runner descriptions of K38 pipeline runs do not apply.",
    }


def markdown(result):
    cfg = result["config"]
    lines = ["# F40 selective-expansion GPU microprobe", "", SCOPE, "", COMPARISON, "",
             f"Sources: {cfg['count']:,} unique M62/F40 endpoints, seed {cfg['seed']}; "
             f"ordered FNV-1a64 `{cfg['corpus']['ordered_n0_fnv1a64_le']}`. "
             "This identifier is not a cryptographic hash. Shared class windows: 32 and 35 bits.", "",
             "| Block | M | Kernel median (ms) | Observed range (ms) | Ratio to opening A10 | Ratio to closing A10 |",
             "|:---|---:|---:|:---|---:|---:|"]
    for block in result["blocks"]:
        time = block["kernel_seconds"]
        lines.append(f"| {block['label']} | {block['m']} | {time['median'] * 1000:.3f} | "
                     f"{time['minimum'] * 1000:.3f}–{time['maximum'] * 1000:.3f} | "
                     f"{block['versus']['opening_a10']['kernel_seconds_ratio']:.4f} | "
                     f"{block['versus']['closing_a10']['kernel_seconds_ratio']:.4f} |")
    controls = result["a10_controls"]
    pooled = controls["pooled"]
    lines += ["", f"Pooled A10 controls: median {pooled['median'] * 1000:.3f} ms, "
              f"observed range {pooled['minimum'] * 1000:.3f}–{pooled['maximum'] * 1000:.3f} ms. "
              f"Closing/opening median ratio: {controls['closing_to_opening_median_ratio']:.4f}.", "",
              "| Block | Endpoints | Endpoint factor | Baseline candidate fraction retained | Tail cases | Tail iterations | Hash |",
              "|:---|---:|---:|---:|---:|---:|:---|"]
    for block in result["blocks"]:
        work = block["work"]
        lines.append(f"| {block['label']} | {work['endpoints']:,} | {block['endpoint_count_factor']:.4f} | "
                     f"{block['candidate_retained_fraction']:.6f} | {work['cases']:,} | "
                     f"{work['iters']:,} | `{work['hash']}` |")
    lines += ["", "Candidate fractions divide the endpoint factor by 2 or 4 after one or two extra bits. "
              "They describe baseline candidates before lookahead, so they differ from tail-case and iteration ratios.", "",
              f"All {result['validation_count']} class-level CPU/GPU audits passed on descendants of the first 16 sources. "
              "All warmups, repetitions and same-A control work signatures agree, with zero flagged trajectories. "
              "Class counts, XOR hashes and kernel seconds were independently summed and checked against each shape summary.", "",
              "These measurements test GPU work for a conditional F40 corpus. They do not establish the throughput of "
              "a mixed-depth queue or predict the benefit for the saved K38 packages. The JSON retains each class trial, "
              "setup timing, source identifiers and validation record."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.directory)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.out.with_suffix(".md").write_text(markdown(result))
    print(f"Wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
