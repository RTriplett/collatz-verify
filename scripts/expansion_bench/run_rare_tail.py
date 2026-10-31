#!/usr/bin/env python3
"""Record the isolated CPU fallback benchmark; run after GPU timing finishes."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from run_blocks import ROOT, read_optional, sha256, timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--draws", type=int, default=4096)
    parser.add_argument("--cohorts", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    if args.cpu not in os.sched_getaffinity(0):
        parser.error("requested CPU affinity unavailable")
    if args.draws < 1 or args.cohorts < 2 or args.repetitions < 1:
        parser.error("need positive draws/repetitions and at least two cohorts")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("metadata.json", "pipeline.jsonl", "pipeline.stderr.txt")):
        parser.error("measurement files already exist; use a fresh output directory")
    binary = args.binary.resolve()
    command = ["taskset", "-c", str(args.cpu), str(binary), str(args.draws),
               str(args.cohorts), str(args.repetitions), str(args.seed)]
    metadata = {
        "state": "preparing", "started_at": timestamp(), "command": command,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "working_directory": str(ROOT), "platform": platform.platform(),
        "affinity": [args.cpu], "cpuinfo": read_optional("/proc/cpuinfo"),
        "governor": read_optional(f"/sys/devices/system/cpu/cpu{args.cpu}/cpufreq/scaling_governor"),
        "source_sha256": {},
        "notes": "CPU-only conditional baseline F56–62 allowance. No GPU initialization. "
                 "Sources and binary hashes identify artifacts, not independent build provenance.",
    }
    def save():
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    save()
    started = time.monotonic()
    status = 1
    try:
        metadata["binary_sha256"] = sha256(binary)
        metadata["rustc"] = subprocess.check_output(["rustc", "--version"], text=True).strip()
        metadata["lscpu"] = json.loads(subprocess.check_output(["lscpu", "--json"], text=True))
        sources = set((ROOT / "src").glob("*.rs"))
        sources |= {ROOT / name for name in (
            "Cargo.toml", "Cargo.lock", "build.rs", "scripts/expansion_bench/Cargo.toml",
            "scripts/expansion_bench/Cargo.lock", "scripts/expansion_bench/rare_tail_bench.rs",
            "scripts/expansion_bench/run_rare_tail.py", "scripts/expansion_bench/run_blocks.py",
            "scripts/expansion_bench/summarize_rare_tail.py",
            "scripts/expansion_bench/summarize_fullrange.py", "scripts/expansion_bench/summarize_overall.py")}
        for source in sorted(sources):
            relative = source.relative_to(ROOT)
            destination = output / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            metadata["source_sha256"][str(relative)] = sha256(destination)
        metadata["state"] = "running"
        save()
        print(f"Running CPU fallback: {args.cohorts} × {args.draws} draws, {args.repetitions} repetitions", flush=True)
        with (output / "pipeline.jsonl").open("w") as stdout, (output / "pipeline.stderr.txt").open("w") as stderr:
            status = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False).returncode
        metadata["state"] = "complete" if status == 0 else "failed"
    except KeyboardInterrupt:
        status = 130
        metadata["state"] = "interrupted"
    except Exception as error:
        metadata["state"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        print(metadata["error"], file=sys.stderr)
    finally:
        metadata.update(exit_status=status, finished_at=timestamp(), elapsed_seconds=time.monotonic()-started)
        save()
    print(f"Exit {status}; results saved in {output}", flush=True)
    return status if status >= 0 else 128-status


if __name__ == "__main__":
    sys.exit(main())
