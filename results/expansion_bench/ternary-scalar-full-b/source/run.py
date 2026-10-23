#!/usr/bin/env python3
"""Run serial, affinity-pinned expansion comparisons and preserve raw trials.

Usage: python3 scripts/expansion_bench/run.py BINARY OUTPUT_DIR
Optional flags select corpus size/repetitions/cases/threads. No GPU is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("binary", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--roots", type=int, default=8192)
parser.add_argument("--reps", type=int, default=7)
parser.add_argument("--seed", type=int, default=17391)
parser.add_argument("--cases", nargs="+", default=["early", "late"])
parser.add_argument("--threads", nargs="+", type=int, default=[1, 8])
parser.add_argument("--rules", default="improved")
parser.add_argument("--variants", default="all")
parser.add_argument("--stats", action="store_true")
parser.add_argument("--profile-roots", type=int, default=0)
args = parser.parse_args()
binary = args.binary.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
sources = output / "source"
sources.mkdir(exist_ok=True)
hashes = {}
for name in ["main.rs", "gating.rs", "blocks.rs", "ternary.rs", "carry.rs", "profile.rs", "reuse.rs", "tests.rs", "run.py", "summarize.py", "Cargo.toml", "Cargo.lock"]:
    file = ROOT / "scripts/expansion_bench" / name
    if not file.exists():
        continue
    hashes[name] = hashlib.sha256(file.read_bytes()).hexdigest()
    shutil.copyfile(file, sources / name)
for file in sorted((ROOT / "src").glob("*.rs")):
    hashes[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
metadata = {
    "cpu": "AMD Ryzen 7 7700X",
    "physical_cores": 8,
    "logical_threads": 16,
    "affinity": {"1": "0", "8": "0-7"},
    "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
    "governor": Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip(),
    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    "source_sha256": hashes,
    "roots": args.roots, "repetitions": args.reps, "seed": args.seed,
    "production_statistics": args.stats, "gpu_measured": False,
    "notes": "Commands run serially; snapshot Rust files can replace corresponding experimental package files to reproduce this revision. Root cv dependency is unchanged.",
    "commands": [],
}
for case in args.cases:
    for threads in args.threads:
        assert threads in (1, 8), "Affinity mapping is specified for 1 or 8 threads"
        affinity = "0" if threads == 1 else "0-7"
        command = ["taskset", "-c", affinity, str(binary), str(args.roots),
                   str(args.reps), str(threads), str(args.seed), case, args.rules,
                   "64", args.variants, str(int(args.stats)), str(args.profile_roots)]
        destination = output / f"{case}-t{threads}.jsonl"
        metadata["commands"].append({"argv": [*command[:3], "BINARY", *command[4:]], "output": destination.name})
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Running {case}, {threads} threads, {args.roots} roots", flush=True)
        with destination.open("w") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, check=True)
        rows = [json.loads(line) for line in destination.read_text().splitlines()]
        for row in rows:
            if row["type"] == "summary":
                print(f"  {row['variant']}: {row['median_seconds']:.4f}s, "
                      f"{row['speedup_vs_reference']:.3f}x reference", flush=True)
