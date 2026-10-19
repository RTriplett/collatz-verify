#!/usr/bin/env python3
"""Resource tables from `cv measure` sweeps (results/sweep/*.json).

GPU time: sum over bitvector classes of (exact endpoint count) x (seconds per
endpoint), where seconds per endpoint come from the per-operation fit of
scripts/gpu_fit.py (overhead per endpoint + per case + per double iteration,
fitted to every RTX 4090 kernel timing in the sweep) applied to each class's
measured operation counts.  The raw per-run timing is reported alongside.
CPU time: exact (baseline) or
sampled (improved) step-1 node count x measured ns per node on a Ryzen 7
7700X with all 16 SMT threads busy, expressed per physical core.

Writes results/cost_table.csv and results/cost_table.md.
"""
import csv
import glob
import json
import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")
YEAR = 365.25 * 86400

# ns per node per hardware thread with 16 threads busy (results/cpu_bench.csv)
def cpu_rates():
    rates = {}
    with open(os.path.join(ROOT, "results/cpu_bench.csv")) as fh:
        for row in csv.DictReader(fh):
            if row["threads"] == "16":
                rates[row["rules"]] = float(row["ns_per_node_thread"])
    return rates


THREADS, CORES = 16, 8  # 7700X

CONFIGS = [
    # (label, file tag, totals key)
    ("Angeltveit reference (A10 B24, original bitvectors)", "ref-legacy", "base"),
    ("Sound baseline (A10 B24)", "a10b24", "base"),
    ("+ step-1 pm_inf + mix(3^13) (A10 B24)", "a10b24", "improved"),
    ("+ B28 + window pm_inf/mix (A10 B28)", "a10b28w", "base"),
    ("All improvements (A10 B28)", "a10b28w", "improved"),
    ("All improvements, CPU-light (A12 B28)", "a12b28w", "improved"),
]


def load():
    runs = {}
    for path in glob.glob(os.path.join(ROOT, "results/sweep/n*_*.json")):
        m = re.match(r"n(\d+)_(.+)\.json", os.path.basename(path))
        with open(path) as fh:
            runs[(int(m.group(1)), m.group(2))] = json.load(fh)
    return runs


FIT = json.load(open(os.path.join(ROOT, "results/gpu_fit.json")))


def fitted_gpu_seconds(r, key):
    a = FIT["overhead"][f"{r['a']},{r['b']}"]
    total = 0.0
    for c in r["classes"]:
        if c["class"] >= r["num_bv"]:
            continue  # finished on the CPU
        if key == "base":
            eps, cases, iters = 1.0, c["cases"]["mean"], c["iters"]["mean"]
        else:
            eps, cases, iters = c["survive"]["mean"], c["cases_kept"]["mean"], c["iters_kept"]["mean"]
        total += c["endpoints"] * (a * eps + FIT["per_case"] * cases + FIT["per_iter"] * iters)
    return total


def row(n, label, tag, key, runs, rates):
    r = runs.get((n, tag))
    if r is None:
        return None
    t = r[key] if key == "base" else r["improved"]
    if t is None or t.get("gpu_seconds") is None:
        return None
    rules = r["baseline"] if key == "base" else r["extra"]
    nodes = r["tree_nodes"] if key == "base" else r["tree_nodes_improved"]
    ns = rates[rules]
    # nodes * ns is thread-seconds with all SMT threads busy; a core runs two
    core_seconds = nodes * ns * 1e-9 * CORES / THREADS
    gpu = fitted_gpu_seconds(r, key)
    return {
        "N": n,
        "config": label,
        "A": r["a"],
        "B": r["b"],
        "endpoints": t["endpoints"],
        "cases": t["cases"],
        "iters": t["iters"],
        "tree_nodes": nodes,
        "gpu_years_4090": gpu / YEAR,
        "gpu_years_4090_raw": t["gpu_seconds"] / YEAR,
        "cpu_core_years_7700x": core_seconds / YEAR,
        "cores_per_gpu": core_seconds / gpu,
    }


def main():
    runs = load()
    rates = cpu_rates()
    ns = sorted({n for n, _ in runs})
    rows = []
    for n in ns:
        for label, tag, key in CONFIGS:
            x = row(n, label, tag, key, runs, rates)
            if x:
                rows.append(x)
    with open(os.path.join(ROOT, "results/cost_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ref = {x["N"]: x["gpu_years_4090"] for x in rows if x["config"].startswith("Sound baseline")}
    out = ["| N | configuration | RTX 4090-years | (raw timing) | vs sound baseline | CPU core-years (7700X) | cores per GPU | cases |",
           "|---|---|---|---|---|---|---|---|"]
    for x in rows:
        rel = (x["gpu_years_4090"] / ref[x["N"]] - 1) * 100 if x["N"] in ref else float("nan")
        out.append(
            f"| {x['N']} | {x['config']} | {x['gpu_years_4090']:.3g} | {x['gpu_years_4090_raw']:.3g} | {rel:+.1f}% | "
            f"{x['cpu_core_years_7700x']:.3g} | {x['cores_per_gpu']:.1f} | {x['cases']:.3e} |"
        )
    with open(os.path.join(ROOT, "results/cost_table.md"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
