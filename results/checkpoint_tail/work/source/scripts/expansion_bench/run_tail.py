#!/usr/bin/env python3
"""Run one pinned checkpoint-policy CPU comparison and preserve its inputs.

This runner does not build the binary or time a GPU. CPU affinity 0-7 is the
eight-physical-core mapping used on this benchmark host.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
AFFINITY = "0-7"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--roots", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17395)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--target", type=int, default=62)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("/tmp/collatz-expansion-target/release/tail_bench"),
    )
    args = parser.parse_args()
    output = args.out.resolve()
    binary = args.binary.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Preserve existing measurements instead of silently replacing them.
    if any((output / name).exists() for name in ["metadata.json", "cpu.jsonl", "cpu.stderr.txt"]):
        parser.error(f"measurement files already exist in {output}; use a new --out directory")

    command = [
        "taskset", "-c", AFFINITY, str(binary),
        str(args.roots), str(args.repetitions), str(args.threads),
        str(args.seed), str(args.target), args.variants,
    ]
    metadata = {
        "started_at": timestamp(),
        "finished_at": None,
        "exit_status": None,
        "state": "preparing",
        "gpu_measured": False,
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "command": command,
        "working_directory": str(ROOT),
        "affinity": AFFINITY,
        "available_affinity": sorted(os.sched_getaffinity(0)),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "cpuinfo": read_optional("/proc/cpuinfo"),
        "governors": {
            str(cpu): read_optional(
                f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
            )
            for cpu in range(8)
        },
        "source_sha256": {},
        "outputs": {"stdout": "cpu.jsonl", "stderr": "cpu.stderr.txt"},
        "notes": (
            "Sources are copied relative to the repository root under source/. "
            "The caller must build the binary from those sources before running; "
            "the runner records hashes but does not establish build provenance. "
            "Affinity 0-7 follows this host's physical-core mapping."
        ),
    }
    metadata_path = output / "metadata.json"

    def save_metadata():
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    started = time.monotonic()
    exit_status = 1
    save_metadata()
    try:
        if args.roots < 1 or args.repetitions < 1 or args.threads < 1:
            raise ValueError("roots, repetitions, and threads must be positive")
        metadata["rustc"] = subprocess.check_output(
            ["rustc", "--version"], text=True, stderr=subprocess.STDOUT
        ).strip()
        metadata["binary_sha256"] = sha256(binary)
        lscpu = subprocess.run(
            ["lscpu", "--json"], capture_output=True, text=True, check=False
        )
        metadata["lscpu"] = json.loads(lscpu.stdout) if lscpu.returncode == 0 else None
        inputs = set((ROOT / "src").glob("*.rs"))
        inputs.update((ROOT / "scripts/expansion_bench").glob("*.rs"))
        inputs.update(
            ROOT / name for name in [
                "Cargo.toml", "Cargo.lock", "build.rs",
                "scripts/expansion_bench/Cargo.toml",
                "scripts/expansion_bench/Cargo.lock",
                "scripts/expansion_bench/run_tail.py",
            ]
        )
        for source in sorted(inputs):
            relative = source.relative_to(ROOT)
            destination = output / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            # Hash the saved bytes, which are the authoritative snapshot.
            metadata["source_sha256"][str(relative)] = sha256(destination)
        metadata["state"] = "running"
        metadata["benchmark_started_at"] = timestamp()
        save_metadata()
        print(
            f"Running {args.roots} roots to M={args.target}, {args.repetitions} repetitions, "
            f"{args.threads} workers on CPUs {AFFINITY}",
            flush=True,
        )
        with (output / "cpu.jsonl").open("w") as stdout, (output / "cpu.stderr.txt").open("w") as stderr:
            completed = subprocess.run(
                command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False
            )
        exit_status = completed.returncode
        metadata["state"] = "complete" if exit_status == 0 else "failed"
    except KeyboardInterrupt:
        exit_status = 130
        metadata["state"] = "interrupted"
        metadata["error"] = "KeyboardInterrupt"
    except Exception as error:
        exit_status = 1
        metadata["state"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        print(metadata["error"], file=sys.stderr)
    finally:
        metadata["exit_status"] = exit_status
        metadata["finished_at"] = timestamp()
        metadata["elapsed_seconds"] = time.monotonic() - started
        save_metadata()
    print(f"Exit status {exit_status}; results saved in {output}", flush=True)
    return exit_status if exit_status >= 0 else 128 - exit_status


if __name__ == "__main__":
    sys.exit(main())
