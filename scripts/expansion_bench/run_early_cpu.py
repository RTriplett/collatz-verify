#!/usr/bin/env python3
"""Record provenance and run one pinned CPU replay of GPU-screened states.

Run only while GPU performance experiments are idle: the replay intentionally
uses a CPU core that an integrated benchmark could otherwise need.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--core", type=int, default=6)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--cutoff", type=int, default=120)
    parser.add_argument("--chunks", default="16384,65536,262144")
    parser.add_argument("--widths", default="8,10,12,14")
    parser.add_argument("--modes", default="0,1,2,3,4")
    parser.add_argument("--fixture-scope", default=(
        "First up to 262144 exact states from the first selected GPU export during the "
        "first 32-root warmup. This is a conditional launch/class fixture, not an "
        "independent or representative sample of the full N72 survivor population."))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    fixture, binary, out = (p.resolve() for p in (args.fixture, args.binary, args.out))
    if not fixture.is_file() or not binary.is_file():
        parser.error("fixture and binary must exist")
    if fixture.stat().st_size == 0 or fixture.stat().st_size % 32:
        parser.error("fixture must contain complete 32-byte states")
    if args.core not in os.sched_getaffinity(0):
        parser.error("requested CPU core is outside the current affinity mask")
    if out.exists():
        parser.error("output directory already exists; choose a fresh directory")
    out.mkdir(parents=True)
    command = ["taskset", "-c", str(args.core), str(binary), "--input", str(fixture),
               "--cutoff", str(args.cutoff), "--repetitions", str(args.repetitions),
               "--chunks", args.chunks, "--widths", args.widths, "--modes", args.modes]
    source_paths = ["src/early_cpu.rs", "src/tail.rs", "src/arith.rs", "src/lib.rs",
                    "Cargo.toml", "Cargo.lock", "scripts/expansion_bench/Cargo.toml",
                    "scripts/expansion_bench/Cargo.lock", "scripts/expansion_bench/early_cpu_bench.rs",
                    "scripts/expansion_bench/run_early_cpu.py", "scripts/expansion_bench/summarize_early_cpu.py"]
    cpu_model = next((line.split(":", 1)[1].strip()
                      for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")), "unknown")
    capture_metadata = fixture.parent / "metadata.json"
    metadata = {"experiment": "early_cpu_replay", "command": command, "working_directory": str(root),
                "started_utc": utc_now(), "fixture": str(fixture), "fixture_sha256": sha256(fixture),
                "fixture_bytes": fixture.stat().st_size, "fixture_states": fixture.stat().st_size // 32,
                "fixture_scope": args.fixture_scope, "binary": str(binary), "binary_sha256": sha256(binary),
                "source_sha256": {name: sha256(root / name) for name in source_paths},
                "source_hash_scope": "Files present at execution; binary_sha256 identifies the executable.",
                "cpu_model": cpu_model, "cpu_affinity": [args.core],
                "inherited_cpu_affinity": sorted(os.sched_getaffinity(0)),
                "platform": platform.platform(), "arguments": {**vars(args), "fixture": str(fixture),
                    "binary": str(binary), "out": str(out)},
                "statistics_scope": "Repeated timings on one fixed conditional fixture; no root-sampling inference."}
    if capture_metadata.is_file():
        metadata["capture_metadata"] = {"path": str(capture_metadata), "sha256": sha256(capture_metadata)}
    metadata_path = out / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    start = time.monotonic()
    with (out / "replay.jsonl").open("w") as stdout, (out / "replay.stderr.txt").open("w") as stderr:
        process = subprocess.run(command, cwd=root, stdout=stdout, stderr=stderr)
    metadata.update(finished_utc=utc_now(), elapsed_wall_seconds=time.monotonic() - start,
                    returncode=process.returncode, fixture_unchanged=sha256(fixture) == metadata["fixture_sha256"])
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if process.returncode or not metadata["fixture_unchanged"]:
        sys.exit("CPU replay failed or fixture changed; see saved stderr and metadata")
    summarize = [sys.executable, str(root / "scripts/expansion_bench/summarize_early_cpu.py"),
                 "--out", str(out / "summary.json"), "--metadata", str(metadata_path),
                 "--fixture", str(fixture), str(out / "replay.jsonl")]
    subprocess.run(summarize, cwd=root, check=True)
    print(f"Saved CPU replay and provenance in {out}")


if __name__ == "__main__":
    main()
