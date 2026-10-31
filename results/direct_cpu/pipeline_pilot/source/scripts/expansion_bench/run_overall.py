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
    parser.add_argument("--profile", choices=("validate", "validate-device", "validate-wide", "validate-jump", "target", "census"), default="target")
    parser.add_argument("--a", type=int, choices=(8, 9, 10), default=10,
                        help="CPU/GPU split: M=N-A; combined50 retains terminal entry K52")
    parser.add_argument("--roots", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--variants", default="full,screened,compact54,combined54,compact50,combined50")
    parser.add_argument("--affinity", default="0-7")
    parser.add_argument("--class-b", default="32,35,35,35,35,35,32,32,27,27,27,27,27,27,27,27")
    parser.add_argument("--binary", type=Path, default=Path("/tmp/collatz-overall-target/release/overall_bench"))
    parser.add_argument("--cohorts", type=int, default=16,
                        help="fullrange_bench: independent cohorts of --roots baseline draws")
    parser.add_argument("--cutoff", type=int, default=50,
                        help="fullrange_bench: last depth using improved tree checks")
    parser.add_argument("--entry", type=int, default=52,
                        help="fullrange_bench: terminal-table entry depth")
    parser.add_argument("--aggregate-runs", type=int, default=1,
                        help="fullrange_bench: concatenated-corpus calibration repetitions (0 disables)")
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
    cpu_only = args.profile == "census"
    if cpu_only:
        if binary.name != "fullrange_bench":
            parser.error("the census profile requires --binary pointing to fullrange_bench")
        if args.variants == "full,screened,compact54,combined54,compact50,combined50":
            args.variants = "chain4,group4,chain6,group6"
        if not set(args.variants.split(",")).issubset({"chain4", "group4", "chain6", "group6", "group4_jump12"}):
            parser.error("census variants must be chain4, group4, chain6, or group6")
    child_env = os.environ.copy()
    if binary.name in ("fullrange_bench", "direct_pipeline_bench"):
        if args.cohorts < 1 or args.aggregate_runs < 0:
            parser.error("cohorts must be positive; aggregate-runs must be nonnegative")
        child_env.update(CV_FULLRANGE_COHORTS=str(args.cohorts),
                         CV_FULLRANGE_CUTOFF=str(args.cutoff),
                         CV_FULLRANGE_ENTRY=str(args.entry),
                         CV_FULLRANGE_AGGREGATE=str(args.aggregate_runs))
    scope_note = (
        "Untimed CPU arithmetic-sharing census: fixed baseline K38 draws with replacement, "
        "at most sixteen accepted roots per cohort, no GPU initialization or tail execution. "
        "Reports shared table dispatch and affine-word setup, not throughput or a full verification."
        if cpu_only else
        "Full-range estimator: fixed uniform baseline K38 draws with replacement; rejected roots contribute zero downstream work. "
        "Independent cohort timings and a concatenated-corpus calibration support an N72 extrapolation, not a completed verification."
        if binary.name == "fullrange_bench" else
        "Direct CPU routing experiment: fixed baseline K38 packages, CPU and GPU may use different completion widths and windows. "
        "Validate original endpoint coverage and each CPU policy independently; cross-policy tail checksums can differ. No full-range extrapolation."
        if binary.name == "direct_pipeline_bench" else
        "Class-split probe: conditional M62/F40 endpoint samples, improved checks through K50, no K38 walkback. "
        "Serialized kernel times exclude host expansion, transfers and array construction; not a mixed-A pipeline measurement."
        if binary.name == "class_split_bench" else
        "Timed target runs share sampled, walkback-filtered K38 packages; they are not full 2^72 verification."
    )
    command = ["taskset", "-c", args.affinity, str(binary), args.profile,
               str(args.roots), str(args.repetitions), str(args.threads), str(args.seed), args.variants, args.class_b, str(args.a)]
    metadata = {
        "started_at": timestamp(), "state": "preparing", "command": command,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "working_directory": str(ROOT), "platform": platform.platform(),
        "affinity": cpus, "cpuinfo": read_optional("/proc/cpuinfo"),
        "governors": {str(c): read_optional(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor") for c in cpus},
        "cv_gpu_slots": os.environ.get("CV_GPU_SLOTS", "3 (default)"),
        "fullrange_environment": {k: v for k, v in child_env.items() if k.startswith("CV_FULLRANGE_")},
        "early_environment": {k: v for k, v in child_env.items() if k.startswith("CV_EARLY_")},
        "direct_environment": {k: v for k, v in child_env.items() if k.startswith("CV_DIRECT_")},
        "source_sha256": {},
        "notes": "Sources are copied under source/. Hashes identify files and binary, but do not independently establish build provenance. " + scope_note,
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
        metadata["gpu_before"] = None if cpu_only else subprocess.check_output(["nvidia-smi"], text=True)
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
        print(f"Running {args.profile}, A={args.a}: {args.roots} roots, {args.repetitions} repetitions, {args.threads} workers", flush=True)
        with (output / "gpu.csv").open("w") as telemetry, (output / "gpu.stderr.txt").open("w") as telemetry_error:
            if not cpu_only:
                monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,pstate", "--format=csv", "-l", "1"], stdout=telemetry, stderr=telemetry_error)
            with (output / "pipeline.jsonl").open("w") as stdout, (output / "pipeline.stderr.txt").open("w") as stderr:
                result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, env=child_env, check=False)
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
