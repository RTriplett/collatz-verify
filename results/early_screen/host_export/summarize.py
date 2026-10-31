#!/usr/bin/env python3
"""Summarize the saved standalone host-export trials without rerunning them."""
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
raw = (HERE / "trials.jsonl").read_bytes()
rows = [json.loads(line) for line in raw.splitlines()]
assert rows[0]["type"] == "config"
config = rows[0]
trials = [row for row in rows[1:] if not row["warmup"]]
assert len(trials) == 8 * config["repetitions"]
assert all(row["checks_passed"] for row in rows[1:])
variants = []
metrics = (
    "wall_seconds", "cpu_seconds", "user_seconds", "system_seconds",
    "minor_faults", "major_faults", "voluntary_switches", "involuntary_switches",
)
for name in dict.fromkeys(row["variant"] for row in trials):
    samples = [row for row in trials if row["variant"] == name]
    assert sorted(row["repetition"] for row in samples) == list(range(config["repetitions"]))
    if name in {"software_popcount", "hardware_popcnt64", "software_scan_allocate_copy", "software_scan_reused_copy"}:
        assert all(row["count"] == config["expected_count"] for row in samples)
    variants.append({
        "variant": name,
        "sample_count": len(samples),
        "metrics": {
            metric: {
                "median": statistics.median(row[metric] for row in samples),
                "min": min(row[metric] for row in samples),
                "max": max(row[metric] for row in samples),
            }
            for metric in metrics
        },
    })
summary = {
    "schema_version": 1,
    "trials_sha256": hashlib.sha256(raw).hexdigest(),
    "config": config,
    "checks_passed": True,
    "warmup_trials_excluded": len(rows) - 1 - len(trials),
    "variants": variants,
    "caveats": [
        "Synthetic deterministic CPU-filled pinned sources; no GPU computation or DMA.",
        "One thread on logical CPU 6; no simultaneous workers or tree producer.",
        "Wall and thread CPU clocks are primary; getrusage user/system times are auxiliary with coarser accounting.",
        "Five order-rotated repetitions provide local sensitivity, not a confidence interval or a production speedup prediction.",
        "Destination allocation and first-touch behavior depend on the allocator and OS; frees occur outside the timed section.",
    ],
}
(HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"checks_passed": True, "measured_trials": len(trials), "summary": str(HERE / "summary.json")}))
