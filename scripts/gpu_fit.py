#!/usr/bin/env python3
"""Fit a per-operation GPU cost model to every kernel timing in a sweep.

Each `cv measure --gpu` run (results/sweep: B <= 28 with host arrays;
results/bgpu: B up to 36 with device-built arrays) times 2^21 uniform
endpoints per bitvector class.
Individual timings drift with GPU clocks (desktop card, thermal state), but the
operation counts behind them are known to <0.01%.  Fit

    seconds/endpoint = a_(A,B) + b * cases/endpoint + c * iterations/endpoint

(`a_(A,B)`: per-endpoint overhead: recompute T^M, scan 2^A bits of a 2^B-bit
array whose cache behaviour depends on B)
by least squares over all (run, class) points, weighted by each class's share
of the run's GPU time.  Prints the fit, its residuals, and writes
results/gpu_fit.json for scripts/cost_table.py.
"""
import glob
import json
import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")


def points():
    pts = []
    paths = glob.glob(os.path.join(ROOT, "results/sweep/n*_*.json")) + glob.glob(os.path.join(ROOT, "results/bgpu/n*_*.json"))
    for path in sorted(paths):
        tag = re.match(r"n(\d+)_(.+)\.json", os.path.basename(path))
        r = json.load(open(path))
        tot = sum(c["endpoints"] * c["gpu_s"] for c in r["classes"] if c.get("gpu_s"))
        windows = r.get("class_b") or ([r["b_hi"][0]] * r["b_hi"][1] if r.get("b_hi") else [])
        for c in r["classes"]:
            if not c.get("gpu_s") or c["endpoints"] == 0:
                continue
            w = c["endpoints"] * c["gpu_s"] / tot
            b = windows[c["class"]] if c["class"] < len(windows) else r["b"]
            pts.append(dict(run=os.path.relpath(path, ROOT), n=int(tag.group(1)), a=r["a"], b=b, cls=c["class"],
                            y=c["gpu_s"], cases=c["cases"]["mean"], iters=c["iters"]["mean"], w=w))
    return pts


def solve(rows, ys, ws):
    # weighted normal equations, tiny dense solve
    k = len(rows[0])
    ata = [[0.0] * k for _ in range(k)]
    aty = [0.0] * k
    for x, y, w in zip(rows, ys, ws):
        for i in range(k):
            aty[i] += w * x[i] * y
            for j in range(k):
                ata[i][j] += w * x[i] * x[j]
    # Gaussian elimination
    for i in range(k):
        p = max(range(i, k), key=lambda r: abs(ata[r][i]))
        ata[i], ata[p] = ata[p], ata[i]
        aty[i], aty[p] = aty[p], aty[i]
        for r in range(k):
            if r != i:
                f = ata[r][i] / ata[i][i]
                for j in range(i, k):
                    ata[r][j] -= f * ata[i][j]
                aty[r] -= f * aty[i]
    return [aty[i] / ata[i][i] for i in range(k)]


def main():
    pts = points()
    shapes = sorted({(p["a"], p["b"]) for p in pts})
    def feats(p):
        return [1.0 if (p["a"], p["b"]) == s else 0.0 for s in shapes] + [p["cases"], p["iters"]]
    coef = solve([feats(p) for p in pts], [p["y"] for p in pts], [p["w"] / p["y"] ** 2 for p in pts])
    fit = {"overhead": {f"{a},{b}": coef[i] for i, (a, b) in enumerate(shapes)}, "per_case": coef[-2], "per_iter": coef[-1]}
    print("overhead per endpoint:", {k: f"{v*1e9:.3f} ns" for k, v in fit["overhead"].items()})
    print(f"per case: {fit['per_case']*1e12:.2f} ps   per double iteration: {fit['per_iter']*1e12:.3f} ps")
    # residuals: per run, GPU-time-weighted total vs fitted total
    runs = {}
    for p in pts:
        pred = sum(c * x for c, x in zip(coef, feats(p)))
        runs.setdefault(p["run"], [0.0, 0.0])
        runs[p["run"]][0] += p["w"]
        runs[p["run"]][1] += p["w"] * pred / p["y"]
    devs = sorted((v[1] / v[0] - 1) * 100 for v in runs.values())
    print(f"per-run fitted/measured: median {devs[len(devs)//2]:+.2f}%  range {devs[0]:+.2f}% .. {devs[-1]:+.2f}%  ({len(devs)} runs)")
    big = [p for p in pts if p["w"] > 0.02]
    worst = max(abs(sum(c * x for c, x in zip(coef, feats(p))) / p["y"] - 1) for p in big)
    print(f"worst class residual among classes with >2% of a run's time: {worst*100:.1f}%")
    fit["runs"] = {k: v[1] / v[0] for k, v in runs.items()}
    json.dump(fit, open(os.path.join(ROOT, "results/gpu_fit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
