#!/usr/bin/env python3
"""Run the paired checkpoint-policy work estimator with reproducible inputs.

The default all mode estimates both the full-policy denominator and added work
on the same draws. This is a CPU sampling probe, not a GPU timing benchmark.
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

from run_tail import AFFINITY, ROOT, read_optional, sha256, timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=17397)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--mode", choices=["all", "delta"], default="all")
    parser.add_argument("--model", type=Path, default=ROOT / "results/walkback_reuse/model.json")
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("/tmp/collatz-expansion-target/release/checkpoint_work"),
    )
    args = parser.parse_args()
    output = args.out.resolve()
    binary = args.binary.resolve()
    model = args.model.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ["metadata.json", "work.jsonl", "work.stderr.txt"]):
        parser.error(f"measurement files already exist in {output}; use a new --out directory")

    command = [
        "taskset", "-c", AFFINITY, str(binary), str(args.draws),
        str(args.seed), str(model), str(args.threads), args.mode,
    ]
    metadata = {
        "started_at": timestamp(),
        "finished_at": None,
        "exit_status": None,
        "state": "preparing",
        "gpu_measured": False,
        "paired_full_denominator": args.mode == "all",
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
        "outputs": {"stdout": "work.jsonl", "stderr": "work.stderr.txt"},
        "notes": (
            "Sources are copied relative to the repository root under source/. "
            "An external model is copied under source/model/. The probe reads "
            "the saved model snapshot. The runner records binary and source "
            "hashes but does not establish build provenance. All mode estimates "
            "the paired full-policy denominator; delta mode supplies only added "
            "work and the explicitly historical model denominator. No GPU is timed."
        ),
    }
    metadata_path = output / "metadata.json"

    def save_metadata():
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    started = time.monotonic()
    exit_status = 1
    save_metadata()
    try:
        if args.draws < 2 or args.threads < 1:
            raise ValueError("draws must be at least two and threads must be positive")
        metadata["rustc"] = subprocess.check_output(
            ["rustc", "--version"], text=True, stderr=subprocess.STDOUT
        ).strip()
        metadata["binary_sha256"] = sha256(binary)
        lscpu = subprocess.run(
            ["lscpu", "--json"], capture_output=True, text=True, check=False
        )
        metadata["lscpu"] = json.loads(lscpu.stdout) if lscpu.returncode == 0 else None
        inputs = {str(source.relative_to(ROOT)): source
                  for source in (ROOT / "src").glob("*.rs")}
        for name in [
            "Cargo.toml", "Cargo.lock", "build.rs",
            "scripts/expansion_bench/Cargo.toml",
            "scripts/expansion_bench/Cargo.lock",
            "scripts/expansion_bench/checkpoint_work.rs",
            "scripts/expansion_bench/run_work.py",
            "scripts/expansion_bench/run_tail.py",
        ]:
            inputs[name] = ROOT / name
        try:
            model_relative = model.relative_to(ROOT)
        except ValueError:
            model_relative = Path("model") / model.name
        inputs[str(model_relative)] = model
        for relative, source in sorted(inputs.items()):
            destination = output / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            metadata["source_sha256"][relative] = sha256(destination)
        model_snapshot = output / "source" / model_relative
        metadata["model_snapshot"] = str(model_snapshot.relative_to(output))
        command[6] = str(model_snapshot)
        metadata["state"] = "running"
        metadata["probe_started_at"] = timestamp()
        save_metadata()
        print(
            f"Sampling {args.draws} endpoints per class, mode={args.mode}, "
            f"{args.threads} workers on CPUs {AFFINITY}",
            flush=True,
        )
        with (output / "work.jsonl").open("w") as stdout, (output / "work.stderr.txt").open("w") as stderr:
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
