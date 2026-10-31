#!/usr/bin/env python3
"""Run the live GPU pipeline benchmark and preserve sources, inputs and telemetry.

Build first with CUDA_ARCH=sm_89 cargo build --release --offline --features cuda
--manifest-path scripts/expansion_bench/Cargo.toml --bin overall_bench
--target-dir /tmp/collatz-overall-target (adapt the architecture to your GPU).
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from run_blocks import ROOT, timestamp, sha256, read_optional, parse_affinity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--profile", choices=("validate", "validate-device", "target"), default="target")
    parser.add_argument("--roots", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--variants", default="full,screened,compact54,combined54,compact50,combined50")
    parser.add_argument("--affinity", default="0-7")
    parser.add_argument("--binary", type=Path, default=Path("/tmp/collatz-overall-target/release/overall_bench"))
    args = parser.parse_args()
    if min(args.roots, args.repetitions, args.threads) < 1:
        parser.error("roots, repetitions, and threads must be positive")
    cpus = parse_affinity(args.affinity)
    if not set(cpus).issubset(os.sched_getaffinity(0)):
        parser.error("requested affinity unavailable")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / f).exists() for f in ("metadata.json", "pipeline.jsonl", "pipeline.stderr.txt")):
        parser.error("measurement files already exist; use a new --out directory")
    binary = args.binary.resolve()
    command = ["taskset", "-c", args.affinity, str(binary), args.profile,
               str(args.roots), str(args.repetitions), str(args.threads), str(args.seed), args.variants]
    metadata = {
        "started_at": timestamp(), "state": "preparing", "command": command,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "working_directory": str(ROOT), "platform": platform.platform(),
        "affinity": cpus, "cpuinfo": read_optional("/proc/cpuinfo"),
        "governors": {str(c): read_optional(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor") for c in cpus},
        "cv_gpu_slots": os.environ.get("CV_GPU_SLOTS", "3 (default)"),
        "source_sha256": {},
        "notes": "Sources are copied under source/. Hashes identify files and binary, but do not independently establish build provenance. GPU kernel seconds sum overlapping stream events. Timed target runs share sampled, walkback-filtered K38 packages; they are not full 2^72 verification.",
    }
    metadata_file = output / "metadata.json"
    def save():
        metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
    save()
    timer = time.monotonic()
    monitor = None
    status = 1
    try:
        metadata["binary_sha256"] = sha256(binary)
        metadata["rustc"] = subprocess.check_output(["rustc", "--version"], text=True).strip()
        metadata["gpu_before"] = subprocess.check_output(["nvidia-smi"], text=True)
        metadata["lscpu"] = json.loads(subprocess.check_output(["lscpu", "--json"], text=True))
        sources = set((ROOT / "src").glob("*.rs")) | set((ROOT / "cuda").glob("*"))
        sources |= set((ROOT / "scripts/expansion_bench").glob("*.rs"))
        sources |= {ROOT / p for p in ("Cargo.toml", "Cargo.lock", "build.rs",
            "scripts/expansion_bench/Cargo.toml", "scripts/expansion_bench/Cargo.lock",
            "scripts/expansion_bench/run_overall.py", "scripts/expansion_bench/run_blocks.py")}
        for source in sorted(sources):
            if not source.is_file():
                continue
            relative = source.relative_to(ROOT)
            destination = output / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            metadata["source_sha256"][str(relative)] = sha256(destination)
        metadata["state"] = "running"
        save()
        print(f"Running {args.profile}: {args.roots} roots, {args.repetitions} repetitions, {args.threads} workers", flush=True)
        with (output / "gpu.csv").open("w") as telemetry, (output / "gpu.stderr.txt").open("w") as telemetry_error:
            monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,pstate", "--format=csv", "-l", "1"], stdout=telemetry, stderr=telemetry_error)
            with (output / "pipeline.jsonl").open("w") as stdout, (output / "pipeline.stderr.txt").open("w") as stderr:
                result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
            status = result.returncode
        metadata["state"] = "complete" if status == 0 else "failed"
    except KeyboardInterrupt:
        status = 130
        metadata["state"] = "interrupted"
    except Exception as error:
        metadata["state"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        print(metadata["error"], file=sys.stderr)
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait()
        metadata.update(exit_status=status, finished_at=timestamp(), elapsed_seconds=time.monotonic()-timer)
        save()
    print(f"Exit {status}; results saved in {output}", flush=True)
    return status if status >= 0 else 128-status


if __name__ == "__main__":
    sys.exit(main())
