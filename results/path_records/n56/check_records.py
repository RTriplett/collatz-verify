#!/usr/bin/env python3
"""Independently check the N56 path export; no verifier or GPU is invoked.

Run from any directory. Defaults are verification.json and candidates.json
beside this script; --report, --candidates, --reference and --out override them.
The oracle iterates the ordinary 3*n+1 / n/2 map using Python integers. It
extracts shortcut states after even-input transitions, independently of the
verifier's grouped odd/even arithmetic and fixed-width replay implementation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_URL = "https://pcbarina.fit.vut.cz/path-records.htm"
# Published shortcut-map maxima. IDs 73--76 precede 2^55; ID 77 is the
# only additional published record in [2^55, 2^56).
PUBLISHED_RECORDS = (
    (73, 1254251874774375, 1823036311464280263720932141024),
    (74, 5323048232813247, 1964730439297455725829478995944),
    (75, 8562235014026655, 13471057008351679202003944688336),
    (76, 10709980568908647, 175294593968539094415936960141122),
    (77, 49163256101584231, 301753104069007668258074264675786),
)
CONFIG_FIELDS = (
    "n", "a", "b", "num_bv", "bv_mode", "class_b", "window", "rules",
    "backend", "tail_oee", "verification_scope",
)
TREE_CONFIG = ("n", "a", "rules")
CASE_CONFIG = (
    "n", "a", "b", "num_bv", "bv_mode", "class_b", "window", "rules",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def decimal(value: object, label: str) -> int:
    require(isinstance(value, str) and value.isascii() and value.isdecimal(),
            f"{label} must be an exact decimal string")
    number = int(value)
    require(str(number) == value, f"{label} must use canonical decimal encoding")
    return number


def ordinary_oracle(start: int) -> dict:
    """Visit every ordinary-map state; retain the shortcut subsequence."""
    require(start > 0, "trajectory start must be positive")
    current = start
    standard_maximum = shortcut_maximum = start
    standard_peak_step = shortcut_peak_step = 0
    standard_steps = shortcut_steps = 0
    first_descent_step = None
    while current != 1:
        even_input = current % 2 == 0
        current = current // 2 if even_input else 3 * current + 1
        standard_steps += 1
        require(standard_steps <= 20_000_000, f"oracle step limit reached for {start}")
        if current > standard_maximum:
            standard_maximum = current
            standard_peak_step = standard_steps
        # The raw 3*n+1 intermediate is absent from the shortcut map.
        if even_input:
            shortcut_steps += 1
            if current > shortcut_maximum:
                shortcut_maximum = current
                shortcut_peak_step = shortcut_steps
            if first_descent_step is None and current < start:
                first_descent_step = shortcut_steps
    return {
        "start": str(start),
        "shortcut_maximum": str(shortcut_maximum),
        "standard_maximum": str(standard_maximum),
        "shortcut_peak_step": shortcut_peak_step,
        "standard_peak_step": standard_peak_step,
        "total_shortcut_steps": shortcut_steps,
        "total_standard_steps": standard_steps,
        "first_descent_step": first_descent_step,
    }


def source_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def field(data: dict, dotted: str) -> object:
    value = data
    for part in dotted.split("."):
        value = value[part]
    return value


def compare_reference(report: dict, reference: dict, path: Path) -> dict:
    """Compare only counters whose generating configuration still matches."""
    differences = {
        name: {"current": report[name], "reference": reference[name]}
        for name in CONFIG_FIELDS
        if name in report and name in reference and report[name] != reference[name]
    }
    unavailable = {
        name: [label for label, data in (("current", report), ("reference", reference))
               if name not in data]
        for name in (*CONFIG_FIELDS, "cutoff_bits")
        if name not in report or name not in reference
    }

    def same(names: tuple[str, ...]) -> bool:
        return all(name in report and name in reference and report[name] == reference[name]
                   for name in names)

    tree_matches = same(TREE_CONFIG)
    # A legacy report may omit these fields. Preserve the uncertainty rather
    # than interpreting absence as a default or proof of full-range coverage.
    scope_compatible = all(data.get("verification_scope") in (None, "full_range")
                           for data in (report, reference))
    tail_oee_compatible = ("tail_oee" not in report or "tail_oee" not in reference
                           or report["tail_oee"] == reference["tail_oee"])
    checks = []
    fields = [
        ("tree_nodes", tree_matches and scope_compatible),
        ("endpoints", tree_matches and scope_compatible),
        ("tree.generated", tree_matches and scope_compatible),
        ("tree.survived", tree_matches and scope_compatible),
        ("tree.culls", tree_matches and scope_compatible),
        ("tail.endpoints", tree_matches and same(("num_bv",)) and scope_compatible),
        ("cases", same(CASE_CONFIG) and scope_compatible and tail_oee_compatible),
        ("tail.cases", same(CASE_CONFIG) and scope_compatible and tail_oee_compatible),
    ]
    for name, comparable in fields:
        if not comparable:
            checks.append({"field": name, "status": "skipped_configuration_differs"})
            continue
        current, previous = field(report, name), field(reference, name)
        require(current == previous, f"reference invariant differs: {name}")
        check = {"field": name, "status": "matched"}
        if isinstance(current, int):
            check["value"] = current
        checks.append(check)
    return {
        "source_report": source_path(path),
        "configuration_differences": differences,
        "metadata_unavailable": unavailable,
        "checks": checks,
        "qualification": "Counter agreement is observed; missing legacy configuration metadata remains unknown.",
        "excluded": ["hash", "iters", "tail.hash", "tail.iters"],
        "exclusion_reason": "A lower arithmetic cutoff changes tail execution, iterations and checksum.",
    }


def validate(report: dict, artifact: dict, reference: dict, reference_path: Path) -> dict:
    require(report["n"] == 56, "this comparison expects N=56")
    require(report["verification_scope"] == "full_range", "source report must cover the full range")
    require(artifact["schema_version"] == 1, "unsupported artifact schema")
    require(artifact["artifact_type"] == "collatz_path_candidates", "unexpected artifact type")
    require(artifact["selection"] == "overflow_flags_only", "unexpected candidate selection")
    require(artifact["global_path_records_complete"] is False, "artifact must not claim a complete record search")
    require(artifact["all_input_flags_replayed"] is True, "artifact has incomplete replay")
    for name, value in artifact["verification"].items():
        require(value == report.get(name), f"artifact verification metadata differs: {name}")
    flags = [decimal(value, "flagged start") for value in report["flagged"]]
    require(all(0 < start < 1 << 56 for start in flags), "flagged start outside N56 range")
    counts = Counter(flags)
    require(report["flagged_rechecked"] == len(flags), "not every source flag was rechecked")
    require(artifact["flagged_occurrences"] == len(flags), "flag occurrence count differs")
    require(artifact["unique_candidates"] == len(counts), "unique candidate count differs")
    candidates = artifact["candidates"]
    starts = [decimal(candidate["start"], "candidate start") for candidate in candidates]
    require(starts == sorted(counts), "candidates must be numerically sorted, unique and equal to source flags")
    by_start = {}
    for start, candidate in zip(starts, candidates):
        expected = ordinary_oracle(start)
        expected["occurrences"] = counts[start]
        require(candidate == expected, f"independent trajectory metrics differ for start {start}: expected {expected}")
        by_start[start] = candidate

    published_matches = []
    for record_id, start, maximum in PUBLISHED_RECORDS:
        require(start in by_start, f"missing published record #{record_id}: {start}")
        candidate = by_start[start]
        require(candidate["shortcut_maximum"] == str(maximum), f"published peak differs for record #{record_id}")
        require(candidate["standard_maximum"] == str(2 * maximum), f"ordinary peak differs for record #{record_id}")
        published_matches.append({"record_id": record_id, **candidate})
    added = [row for row in published_matches if 1 << 55 <= int(row["start"]) < 1 << 56]
    require([row["record_id"] for row in added] == [77], "expected exactly published record #77 in [2^55,2^56)")
    candidate_peak_records = []
    highest_peak = -1
    for candidate in candidates:
        peak = int(candidate["shortcut_maximum"])
        if peak > highest_peak:
            candidate_peak_records.append(candidate["start"])
            highest_peak = peak
    require(candidate_peak_records == [str(start) for _, start, _ in PUBLISHED_RECORDS],
            "successive maximum setters among captured candidates differ from published records #73--77")
    return {
        "flagged_occurrences": len(flags),
        "unique_candidates": len(candidates),
        "duplicate_occurrences": len(flags) - len(candidates),
        "all_exported_metrics_match_independent_oracle": True,
        "published_records_matched": published_matches,
        "successive_maximum_setters_among_captured_candidates": candidate_peak_records,
        "additional_published_records_between_2_55_and_2_56": added,
        "reference_comparison": compare_reference(report, reference, reference_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=HERE / "verification.json")
    parser.add_argument("--candidates", type=Path, default=HERE / "candidates.json")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/validation/verify_n56_a12_cls.json")
    parser.add_argument("--out", type=Path, default=HERE / "comparison.json")
    args = parser.parse_args()
    require(args.out.resolve() not in {path.resolve() for path in (args.report, args.candidates, args.reference)},
            "comparison output must not overwrite an input")
    comparison = {
        "schema_version": 1,
        "status": "passed",
        "global_path_records_complete": False,
        "published_source": SOURCE_URL,
        "oracle": "Python arbitrary-precision integers, ordinary-map transitions, shortcut states after even-input transitions",
        "source_report": source_path(args.report),
        "source_candidates": source_path(args.candidates),
    }
    try:
        report = json.loads(args.report.read_text())
        artifact = json.loads(args.candidates.read_text())
        reference = json.loads(args.reference.read_text())
        comparison.update(validate(report, artifact, reference, args.reference))
    except (OSError, ValueError, KeyError, TypeError) as error:
        comparison.update(status="failed", error=str(error))
    args.out.write_text(json.dumps(comparison, indent=2) + "\n")
    if comparison["status"] != "passed":
        print(f"FAILED: {comparison['error']}", file=sys.stderr)
        return 1
    matched = sum(check["status"] == "matched" for check in comparison["reference_comparison"]["checks"])
    print(f"PASS: {comparison['unique_candidates']} unique candidates; all trajectory metrics match; "
          f"published records #73--77 present, including #77 above 2^55; {matched} reference count checks match.")
    print(f"Wrote {source_path(args.out)}. Captured candidates do not establish a complete global record census.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
