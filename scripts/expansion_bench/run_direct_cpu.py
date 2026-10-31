#!/usr/bin/env python3
"""Record CPU completion-width experiments; run separately from GPU timings."""
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--core", type=int, default=6)
    p.add_argument("--count", type=int, default=2048)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--classes", default="40,41,42")
    p.add_argument("--a", default="10,8,6,4,2,0")
    p.add_argument("--b", default="0,16,20")
    p.add_argument("--audit", type=int, default=4)
    args = p.parse_args()
    if args.core not in os.sched_getaffinity(0):
        p.error("requested CPU affinity unavailable")
    if min(args.count, args.repetitions, args.audit) < 1:
        p.error("count, repetitions and audit must be positive")
    out = args.out.resolve()
    if out.exists():
        p.error("use a fresh output directory")
    out.mkdir(parents=True)
    binary = args.binary.resolve()
    command = ["taskset", "-c", str(args.core), str(binary), "--count", str(args.count),
               "--repetitions", str(args.repetitions), "--seed", str(args.seed),
               "--classes", args.classes, "--a", args.a, "--b", args.b,
               "--setup-threads", "1", "--audit", str(args.audit)]
    metadata = {
        "state": "preparing", "started_at": timestamp(), "command": command,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "working_directory": str(ROOT), "platform": platform.platform(),
        "affinity": [args.core], "cpuinfo": read_optional("/proc/cpuinfo"),
        "governor": read_optional(f"/sys/devices/system/cpu/cpu{args.core}/cpufreq/scaling_governor"),
        "source_sha256": {},
        "notes": "CPU-only conditional M62 endpoint experiment. Compare CPU completion widths on the same source families. "
                 "Different policies can have different tail counts and checksums. No full-range throughput extrapolation. "
                 "Hashes identify saved sources and executable, not independent build provenance.",
    }
    def save():
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    save()
    timer = time.monotonic()
    status = 1
    try:
        metadata["binary_sha256"] = sha256(binary)
        metadata["rustc"] = subprocess.check_output(["rustc", "--version"], text=True).strip()
        metadata["lscpu"] = json.loads(subprocess.check_output(["lscpu", "--json"], text=True))
        sources = set((ROOT / "src").glob("*.rs"))
        sources |= {ROOT / name for name in ("Cargo.toml", "Cargo.lock", "build.rs",
            "scripts/expansion_bench/Cargo.toml", "scripts/expansion_bench/Cargo.lock",
            "scripts/expansion_bench/direct_cpu_bench.rs", "scripts/expansion_bench/run_direct_cpu.py",
            "scripts/expansion_bench/run_blocks.py")}
        for source in sorted(sources):
            rel = source.relative_to(ROOT)
            dest = out / "source" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            metadata["source_sha256"][str(rel)] = sha256(dest)
        metadata["state"] = "running"
        save()
        print(f"Running direct CPU probe: {args.count} endpoints per class, {args.repetitions} repetitions", flush=True)
        with (out / "pipeline.jsonl").open("w") as stdout, (out / "pipeline.stderr.txt").open("w") as stderr:
            status = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False).returncode
        metadata["state"] = "complete" if status == 0 else "failed"
    except Exception as error:
        metadata.update(state="failed", error=f"{type(error).__name__}: {error}")
        print(metadata["error"], file=sys.stderr)
    finally:
        metadata.update(exit_status=status, finished_at=timestamp(), elapsed_seconds=time.monotonic()-timer)
        save()
    print(f"Exit {status}; results saved in {out}", flush=True)
    return status if status >= 0 else 128-status


if __name__ == "__main__":
    sys.exit(main())
