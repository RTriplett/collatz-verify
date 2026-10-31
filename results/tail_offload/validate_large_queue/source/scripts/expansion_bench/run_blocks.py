#!/usr/bin/env python3
"""Run one pinned terminal-block comparison and preserve its inputs.

Build terminal_bench separately before running this script. The default affinity
uses the eight physical cores of the benchmark host; override it when needed.
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


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def parse_affinity(value):
    """Expand taskset's CPU-list syntax for metadata and input validation."""
    cpus = set()
    for item in value.split(","):
        parts = item.split("-")
        if len(parts) == 1:
            first = last = int(parts[0])
        elif len(parts) == 2:
            first, last = map(int, parts)
        else:
            raise ValueError(f"invalid CPU range: {item!r}")
        if first < 0 or last < first:
            raise ValueError(f"invalid CPU range: {item!r}")
        cpus.update(range(first, last + 1))
    return sorted(cpus)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--roots", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17411)
    parser.add_argument("--target", type=int, default=62)
    parser.add_argument("--cutoff", type=int, default=50)
    parser.add_argument("--variants", default="all")
    parser.add_argument("--sink", choices=("digest", "packed"), default="digest")
    parser.add_argument("--affinity", default="0-7")
    parser.add_argument(
        "--binary", type=Path,
        default=Path("/tmp/collatz-expansion-target/release/terminal_bench"),
    )
    args = parser.parse_args()
    if min(args.roots, args.repetitions, args.threads) < 1:
        parser.error("roots, repetitions, and threads must be positive")
    if not 38 <= args.cutoff <= args.target <= 64:
        parser.error("expected 38 <= cutoff <= target <= 64")
    try:
        pinned_cpus = parse_affinity(args.affinity)
    except ValueError as error:
        parser.error(str(error))
    available_cpus = sorted(os.sched_getaffinity(0))
    if not set(pinned_cpus).issubset(available_cpus):
        parser.error(f"affinity {args.affinity!r} is outside available CPUs {available_cpus}")

    output = args.out.resolve()
    binary = args.binary.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("metadata.json", "cpu.jsonl", "cpu.stderr.txt")):
        parser.error(f"measurement files already exist in {output}; use a new --out directory")
    command = [
        "taskset", "-c", args.affinity, str(binary),
        str(args.roots), str(args.repetitions), str(args.threads),
        str(args.seed), str(args.target), str(args.cutoff), args.variants, args.sink,
    ]
    metadata = {
        "started_at": timestamp(),
        "finished_at": None,
        "exit_status": None,
        "state": "preparing",
        "gpu_measured": False,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "command": command,
        "working_directory": str(ROOT),
        "affinity": args.affinity,
        "pinned_cpus": pinned_cpus,
        "available_affinity": available_cpus,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "cpuinfo": read_optional("/proc/cpuinfo"),
        "governors": {
            str(cpu): read_optional(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
            for cpu in pinned_cpus
        },
        "source_sha256": {},
        "outputs": {"stdout": "cpu.jsonl", "stderr": "cpu.stderr.txt"},
        "notes": (
            "Sources are copied relative to the repository root under source/. "
            "Build the binary from these sources before running; hashes record "
            "the inputs but do not establish build provenance. The default "
            "affinity 0-7 follows this host's physical-core mapping. All timed "
            "variants in one run use the same pruning policy and endpoint sink."
        ),
    }
    metadata_path = output / "metadata.json"

    def save_metadata():
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    started = time.monotonic()
    exit_status = 1
    save_metadata()
    try:
        metadata["rustc"] = subprocess.check_output(
            ["rustc", "--version"], text=True, stderr=subprocess.STDOUT
        ).strip()
        metadata["binary_sha256"] = sha256(binary)
        lscpu = subprocess.run(["lscpu", "--json"], capture_output=True, text=True, check=False)
        metadata["lscpu"] = json.loads(lscpu.stdout) if lscpu.returncode == 0 else None
        inputs = set((ROOT / "src").glob("*.rs"))
        inputs.update((ROOT / "scripts/expansion_bench").glob("*.rs"))
        inputs.update(ROOT / name for name in (
            "Cargo.toml", "Cargo.lock", "build.rs",
            "scripts/expansion_bench/Cargo.toml",
            "scripts/expansion_bench/Cargo.lock",
            "scripts/expansion_bench/run_blocks.py",
            "scripts/expansion_bench/summarize_blocks.py",
        ))
        for source in sorted(inputs):
            relative = source.relative_to(ROOT)
            destination = output / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            metadata["source_sha256"][str(relative)] = sha256(destination)
        metadata["state"] = "running"
        metadata["benchmark_started_at"] = timestamp()
        save_metadata()
        print(
            f"Running {args.roots} roots to M={args.target}, cutoff {args.cutoff}, "
            f"{args.repetitions} repetitions, {args.threads} workers on CPUs "
            f"{args.affinity}, {args.sink} sink",
            flush=True,
        )
        with (output / "cpu.jsonl").open("w") as stdout, (output / "cpu.stderr.txt").open("w") as stderr:
            completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
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
