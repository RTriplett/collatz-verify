#!/usr/bin/env python3
"""Audit and compare CPU ownership policies on fixed conditional M62 cohorts.

Each A/B policy may prove away different starts and uses a different checksum
boundary. Exact work signatures are compared only within the same policy.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


TIMINGS = ("wall_seconds", "thread_cpu_seconds", "expansion_screen_tail_wall_seconds",
           "expansion_screen_tail_thread_cpu_seconds", "wide_recheck_wall_seconds")
COUNTERS = ("source_endpoints", "endpoints", "cases", "iters")


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


def fingerprint(values):
    payload = b"".join(value.to_bytes(16, "little") for value in values)
    fnv = 0xcbf29ce484222325
    for byte in payload:
        fnv = ((fnv ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
    return f"{fnv:016X}", hashlib.sha256(payload).hexdigest()


def distribution(values):
    return {"min": min(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "max": max(values)}


def iterate_source(source):
    value, odds = source, 0
    for _ in range(62):
        if value % 2:
            value = (3 * value + 1) // 2
            odds += 1
        else:
            value //= 2
    return value, odds


def key(row):
    return row["a"], row["b"], row["source_f"]


def audit_work(work, a, count, sources):
    for name in COUNTERS:
        integer(work[name], name)
    require(work["source_endpoints"] == count, "source endpoint count mismatch")
    require(work["endpoints"] <= count * (1 << (10 - a)), "too many descendant endpoints")
    require(work["cases"] <= work["endpoints"] * (1 << a), "too many screened cases")
    require(work["iters"] >= work["cases"], "missing scalar tail iterations")
    require(isinstance(work["hash"], str) and re.fullmatch(r"[0-9A-F]{8}", work["hash"]),
            "invalid native checksum")
    require(isinstance(work["flagged"], list), "invalid flagged-start list")
    flags = []
    for value in work["flagged"]:
        require(isinstance(value, str) and re.fullmatch(r"[0-9]+", value), "invalid flagged start")
        flag = int(value)
        require(3 <= flag < (1 << 72) and flag % (1 << 62) in sources,
                "flag lies outside the assigned source families")
        flags.append(flag)
    require(flags == sorted(flags) and len(flags) <= work["cases"], "flag multiset is invalid")


def audit_timing(timing, work):
    for name in TIMINGS:
        finite(timing[name], name, positive=name in ("wall_seconds", "thread_cpu_seconds"))
    integer(timing["wide_recheck_steps"], "wide recheck steps")
    require(timing["expansion_screen_tail_wall_seconds"] + timing["wide_recheck_wall_seconds"]
            <= timing["wall_seconds"] + 1e-8, "phase times exceed total wall time")
    require(timing["expansion_screen_tail_thread_cpu_seconds"]
            <= timing["thread_cpu_seconds"] + 1e-8, "kernel CPU time exceeds total CPU time")
    require((timing["wide_recheck_steps"] == 0) == (len(work["flagged"]) == 0),
            "wide recheck accounting mismatch")


def audit_rows(rows):
    require(rows and all(isinstance(row, dict) for row in rows), "empty or invalid JSONL")
    require(rows[0].get("type") == "config" and rows[-1].get("type") == "complete",
            "run is missing its configuration or completion marker")
    by_type = defaultdict(list)
    for row in rows:
        by_type[row.get("type")].append(row)
    require(set(by_type) == {"config", "setup", "validation", "warmup", "trial", "summary", "complete"},
            "missing or unexpected record type")
    require(len(by_type["config"]) == len(by_type["complete"]) == 1,
            "duplicate configuration or completion")
    config, complete = by_type["config"][0], by_type["complete"][0]
    require(config["experiment"] == "direct_cpu_complete_source_families", "wrong experiment")
    require((config["n"], config["source_m"], config["source_a"]) == (72, 62, 10),
            "unexpected source-family domain")
    require(config["timing_threads"] == 1 and config["bv_mode"] == "Shared"
            and config["cutoff_bits"] == 120, "unexpected execution policy")
    integer(config["setup_threads"], "setup threads", 1)
    integer(config["seed"], "seed")
    count = integer(config["count_per_class"], "count per class", 1)
    reps = integer(config["repetitions"], "repetitions", 1)
    audit_count = min(count, integer(config["audit_sources_per_class"], "audited sources", 1))
    classes, aa, bb = (config[name] for name in ("classes", "a_values", "b_values"))
    for name, values in (("classes", classes), ("A", aa), ("B", bb)):
        require(isinstance(values, list) and values and len(values) == len(set(values)),
                f"empty or duplicate {name} values")
        for value in values:
            integer(value, name)
    require(all(40 <= f <= 62 for f in classes) and all(0 <= a <= 10 for a in aa),
            "invalid source class or A")
    require(all(b == 0 or max(6, max(aa)) <= b <= 24 for b in bb), "invalid resident B")
    require(10 in aa, "A10 source-depth control missing")
    expected = [(a, b, f) for a in aa for b in bb for f in classes]
    require(complete["exact_equal"] is True and complete["configurations"] == len(expected)
            and complete["trials"] == len(expected) * reps, "completion count mismatch")

    cohorts = config["cohorts"]
    require(len(cohorts) == len(classes) and [c["source_f"] for c in cohorts] == classes,
            "cohort classes differ from configuration")
    source_sets, identities = {}, {}
    for cohort in cohorts:
        f = cohort["source_f"]
        require(cohort["count"] == count and len(cohort["source_n0"]) == count,
                "cohort source count mismatch")
        sources = []
        for value in cohort["source_n0"]:
            require(isinstance(value, str) and re.fullmatch(r"[0-9]+", value), "invalid source encoding")
            source = int(value)
            require(3 <= source < (1 << 62) and source % 4 == 3, "invalid source residue")
            sources.append(source)
        source_sets[f] = set(sources)
        exact = {source: iterate_source(source) for source in source_sets[f]}
        require(all(odds == f for _, odds in exact.values()), "source belongs to another F class")
        fnv, sha = fingerprint(sources)
        require(cohort["ordered_n0_fnv1a64_le128"] == fnv, "source fingerprint mismatch")
        require(cohort["duplicates_retained"] == count - len(source_sets[f]), "source duplicate count mismatch")
        require(cohort["draws"] == count + integer(cohort["rejected_before_or_at_k50"], "rejected draws"),
                "source draw accounting mismatch")
        require(isinstance(cohort["baseline_population"], str)
                and re.fullmatch(r"[0-9]+", cohort["baseline_population"])
                and int(cohort["baseline_population"]) > 0, "invalid conditional population")
        finite(cohort["sampling_wall_seconds"], "sampling time")
        require([int(node["n0"]) for node in cohort["first_8_sources"]] == sources[:8],
                "source preview differs from full source ledger")
        require(all(node["f"] == f and node["k"] == 62 for node in cohort["first_8_sources"]),
                "source preview class/depth mismatch")
        require(all(int(node["m"]) == exact[int(node["n0"])][0] for node in cohort["first_8_sources"]),
                "source preview affine iterate mismatch")
        identities[f] = {"count": count, "duplicates_retained": cohort["duplicates_retained"],
                         "ordered_n0_fnv1a64_le128": fnv, "ordered_n0_sha256_le128": sha}

    setups = {}
    for row in by_type["setup"]:
        policy = (row["a"], row["b"])
        require(policy in [(a, b) for a in aa for b in bb] and policy not in setups,
                "unexpected or duplicate setup")
        require(row["m"] == 72 - row["a"], "setup depth mismatch")
        finite(row["wall_seconds"], "setup wall time")
        integer(row["table_bytes"], "suffix table bytes", 1)
        integer(row["lookahead_bytes"], "lookahead bytes", 1)
        setups[policy] = row
    require(len(setups) == len(aa) * len(bb), "missing setup")

    indexed = {}
    for kind in ("validation", "warmup", "summary"):
        table = {}
        for row in by_type[kind]:
            identity = key(row)
            require(identity in expected and identity not in table, f"unexpected or duplicate {kind}")
            table[identity] = row
        require(set(table) == set(expected), f"missing {kind} configuration")
        indexed[kind] = table

    for identity in expected:
        a, b, f = identity
        validation = indexed["validation"][identity]["audit"]
        require(validation["exact_equal"] is True and validation["sources"] == audit_count,
                "validation did not compare the expected source count")
        parts = ("prefix_culled", "mod9_culled", "oee_culled", "b_culled", "verified")
        for name in parts + ("original_starts", "all_starts_independent_wide_steps"):
            integer(validation[name], f"validation {name}")
        require(sum(validation[name] for name in parts) == validation["original_starts"]
                == audit_count * 1024, "validation original-start ledger mismatch")
        require(validation["all_starts_independent_wide_steps"] >= validation["original_starts"],
                "validation missing original-start wide trajectories")
        require(validation["verified"] == validation["work"]["cases"], "validation case mismatch")
        audit_work(validation["work"], a, audit_count, source_sets[f])
        warmup = indexed["warmup"][identity]
        audit_work(warmup["work"], a, count, source_sets[f])
        audit_timing(warmup["timing"], warmup["work"])
        if audit_count == count:
            require(validation["work"] == warmup["work"], "whole-cohort oracle mismatch")
        if a == 10:
            require(warmup["work"]["endpoints"] == count and validation["prefix_culled"] == 0,
                    "A10 unexpectedly changed the source frontier")
        if b == 0:
            require(validation["b_culled"] == 0, "no-B policy reports B culls")

    groups = defaultdict(list)
    seen = set()
    for row in by_type["trial"]:
        identity = key(row)
        rep = integer(row["repetition"], "repetition")
        position = integer(row["position"], "position")
        require(identity in expected and rep < reps and position < len(expected), "unexpected trial")
        require((rep, identity) not in seen, "duplicate trial")
        seen.add((rep, identity))
        order = expected[(rep * 11) % len(expected):] + expected[:(rep * 11) % len(expected)]
        if rep % 2:
            order.reverse()
        require(identity == order[position], "trial differs from prescribed rotation/reversal")
        require(row["m"] == 72 - row["a"], "trial depth mismatch")
        control = indexed["warmup"][identity]
        require(row["work"] == control["work"], "native work changed within a policy")
        audit_timing(row["timing"], row["work"])
        require(row["timing"]["wide_recheck_steps"] == control["timing"]["wide_recheck_steps"],
                "wide rechecks changed within a policy")
        groups[identity].append(row)
    require(len(seen) == len(expected) * reps and set(groups) == set(expected), "missing timing trials")
    for identity, group in groups.items():
        group.sort(key=lambda row: row["repetition"])
        require([row["repetition"] for row in group] == list(range(reps)), "missing repetition")
        summary = indexed["summary"][identity]
        require(summary["m"] == 72 - identity[0] and summary["repetitions"] == reps
                and summary["work"] == group[0]["work"], "summary policy or work mismatch")
        times = [row["timing"]["wall_seconds"] for row in group]
        for name, value in (("median_wall_seconds", statistics.median(times)),
                            ("min_wall_seconds", min(times)), ("max_wall_seconds", max(times)),
                            ("median_ns_per_source", statistics.median(times) * 1e9 / count)):
            finite(summary[name], name)
            require(math.isclose(summary[name], value, rel_tol=1e-12), "summary timing mismatch")
    for a in aa:
        for f in classes:
            require(len({indexed["warmup"][(a, b, f)]["work"]["endpoints"] for b in bb}) == 1,
                    "B screening changed the expanded endpoint count")
    return config, setups, identities, groups


def summarize(path, metadata=None):
    raw = path.read_bytes()
    config, setups, identities, groups = audit_rows([json.loads(line) for line in raw.splitlines() if line.strip()])
    provenance = None
    if metadata:
        record = json.loads(metadata.read_text())
        require(record["state"] == "complete" and record["exit_status"] == 0,
                "runner did not complete successfully")
        require(len(record["affinity"]) == 1, "runner did not pin a single CPU")
        require(re.fullmatch(r"[0-9a-f]{64}", record["binary_sha256"]), "invalid executable digest")
        for name, expected_hash in record["source_sha256"].items():
            source = metadata.parent / "source" / name
            require(source.resolve().is_relative_to((metadata.parent / "source").resolve()),
                    "source snapshot path escaped its directory")
            require(hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash,
                    f"source snapshot hash mismatch: {name}")
        provenance = {"path": str(metadata), "sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                      "source_snapshot_verified": True, "record": record}
    variants = []
    count = config["count_per_class"]
    for identity, group in sorted(groups.items()):
        a, b, f = identity
        timings = {name: distribution([row["timing"][name] for row in group]) for name in TIMINGS}
        baseline = groups[(10, b, f)]
        controls = {}
        for metric in ("wall_seconds", "thread_cpu_seconds"):
            controls[metric] = {
                "mean_ratio": timings[metric]["mean"] / statistics.mean(r["timing"][metric] for r in baseline),
                "within_repetition_ratios": distribution([r["timing"][metric] / c["timing"][metric]
                                                          for r, c in zip(group, baseline)]),
            }
        variants.append({"a": a, "m": 72-a, "b": b, "source_f": f,
                         "timing_repetitions": len(group), "work": group[0]["work"], "timings": timings,
                         "mean_ns_per_source": {metric: timings[metric]["mean"] * 1e9 / count
                                                for metric in ("wall_seconds", "thread_cpu_seconds")},
                         "same_b_a10_control": controls,
                         "wide_recheck_steps": group[0]["timing"]["wide_recheck_steps"]})
    def selection(v):
        return {name: v[name] for name in ("a", "m", "b", "source_f", "mean_ns_per_source", "same_b_a10_control")}
    best = [selection(min((v for v in variants if v["source_f"] == f),
                          key=lambda v: v["timings"]["thread_cpu_seconds"]["mean"])) for f in config["classes"]]
    return {"input_jsonl": str(path), "input_sha256": hashlib.sha256(raw).hexdigest(),
            "runner_metadata": provenance, "config": config,
            "source_identities": identities, "complete_run_audit_passed": True,
            "setups": list(setups.values()), "best_observed_per_source_class": best, "variants": variants}


def markdown(result):
    lines = ["# Direct CPU endpoint ownership", "", result["scope"], "", result["accounting"], ""]
    for run in result["runs"]:
        config = run["config"]
        lines += [f'## {run["input_jsonl"]}', "",
                  f'{config["count_per_class"]:,} fixed sources per F class, {config["repetitions"]} timing repetitions. '
                  f'All run, source-identity, work, rotation and accounting audits passed. '
                  f'Independent original-start validation covers {min(config["audit_sources_per_class"], config["count_per_class"])} '
                  'sources per class and policy.', "",
                  "| Source F | CPU A | CPU M | B | Mean CPU µs/source | Mean wall µs/source | CPU / same-B A10 | Cases/source |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for v in sorted(run["variants"], key=lambda v: (v["source_f"], v["b"], -v["a"])):
            ns = v["mean_ns_per_source"]
            lines.append(f'| {v["source_f"]} | {v["a"]} | {v["m"]} | {v["b"] or "none"} | '
                         f'{ns["thread_cpu_seconds"]/1000:.3f} | {ns["wall_seconds"]/1000:.3f} | '
                         f'{v["same_b_a10_control"]["thread_cpu_seconds"]["mean_ratio"]:.4f} | '
                         f'{v["work"]["cases"]/config["count_per_class"]:.2f} |')
        lines += ["", "Best observed A/B per source class (by mean CPU time; descriptive selection):", ""]
        for v in run["best_observed_per_source_class"]:
            lines.append(f'- F={v["source_f"]}: A={v["a"]}, B={v["b"] or "none"}, '
                         f'{v["mean_ns_per_source"]["thread_cpu_seconds"]/1000:.3f} µs/source.')
        lines += ["", "Suffix-table and resident-array setup is recorded separately in JSON. "
                  "The source classes are not population-weighted; their timings must not be averaged into a full-range estimate.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, help="runner metadata; valid with one input run")
    args = parser.parse_args()
    try:
        require(not args.metadata or len(args.runs) == 1, "explicit metadata requires one run")
        require(len({path.resolve() for path in args.runs}) == len(args.runs), "duplicate input run")
        runs = []
        for path in args.runs:
            metadata = args.metadata or path.parent / "metadata.json"
            runs.append(summarize(path, metadata if metadata.exists() else None))
        result = {"schema_version": 1,
                  "scope": "CPU-only comparisons on fixed conditional M62/F source cohorts. Every policy owns the same "
                           "original source families; native endpoint/case/iteration counts and hashes can differ across A/B. "
                           "Repetitions measure timing variation, not independent root samples. No full-range cost or GPU saving is inferred.",
                  "accounting": "One pinned timing thread includes reusable suffix expansion, mod9/OEE and optional shared B screening, "
                                "scalar verification and all flagged wide rechecks. Sampling, table/array setup, independent validation "
                                "and JSON output are excluded. Different F classes are conditional strata, not equal population shares.",
                  "runs": runs}
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.out.with_suffix(".md").write_text(markdown(result))
    print(f"Saved {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
