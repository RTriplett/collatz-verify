#!/usr/bin/env python3
"""Which GPU to rent for a 2^N verification, given that VRAM buys lookahead.

Per-class GPU time t_i(B) at N = 72 (A = 10, step-1 pm_inf + mix(3^13), window
credits, shared bitvectors) comes from exact operation counts for each window
B (results/bsweep, direct window evaluation) times the per-operation GPU cost
fitted to every measured kernel timing (results/gpu_fit.json; real-kernel
runs up to B = 36 agree within a few percent).  For each card, an exact DP
over classes picks per-class windows B_i minimising sum_i t_i(B_i) subject to
sum_i (2^B_i + 2^A) bits <= VRAM - 2 GiB.  The chosen windows can be passed
to `cv verify --class-b`.

Speed relative to a 4090 and prices come from the verified research in
results/hardware/gpus.json (SMs x clock with NVIDIA's per-SM integer rates,
which are equal from cc 7.5 to 12.x; +/-10-15%, only the 4090 is measured).
"""
import json
import math
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
YEAR_H = 365.25 * 24
SHARED_COST = 1.0025  # shared vs split bitvectors: +0.20..0.28% iterations (measured)
A = 10
RESERVE_GIB = 2.0
UNIT = 2**25  # DP granularity in bytes (32 MiB = one array at B = 28)
# Step 1 at A = 10: improved-rule tree nodes at N = 72, CPU cost per node
# (16-thread Ryzen 7700X census rate + 10% production overhead, measured at N = 54).
TREE_NODES = 9.248e15
NS_PER_NODE_THREAD = 32.93 * 1.10
# Device bitvector build on an RTX 4090: seconds per 2^30 window patterns.
BUILD_S_PER_GPAT = 0.51


def per_class_times():
    fit = json.load(open(os.path.join(ROOT, "results/gpu_fit.json")))
    pc, pi = fit["per_case"], fit["per_iter"]
    measured = sorted(int(k.split(",")[1]) for k in fit["overhead"] if k.startswith(f"{A},"))

    def overhead(b):
        return fit["overhead"][f"{A},{max([x for x in measured if x <= b] or [measured[0]])}"]

    t = {}
    for b in range(24, 40):
        p = os.path.join(ROOT, f"results/bsweep/n72_a{A}_b{b}.json")
        if not os.path.exists(p):
            continue
        run = json.load(open(p))
        penalty = 1.0 if run.get("bv_mode") == "Shared" else SHARED_COST  # split-mode runs: add the shared penalty
        for c in run["classes"]:
            if c["class"] >= 16:
                continue
            secs = c["endpoints"] * (overhead(b) * c["survive"]["mean"] + pc * c["cases_kept"]["mean"] + pi * c["iters_kept"]["mean"])
            t[(c["class"], b)] = secs * penalty / (YEAR_H * 3600)
    return t


def best_windows(t, vram_gib):
    """Exact DP over classes: windows minimising GPU-years within the budget."""
    classes = sorted({i for i, _ in t})
    bs = sorted({b for _, b in t})
    budget = int((vram_gib - RESERVE_GIB) * 2**30 // UNIT)
    units = {b: max(1, math.ceil((2**b + 2**A) / 8 / UNIT)) for b in bs}
    INF = float("inf")
    best = [0.0] + [INF] * budget  # best[u]: least time using exactly u units so far
    picks = []
    for i in classes:
        nxt = [INF] * (budget + 1)
        pick = [None] * (budget + 1)
        for u, cur in enumerate(best):
            if cur == INF:
                continue
            for b in bs:
                v = u + units[b]
                if (i, b) in t and v <= budget and cur + t[(i, b)] < nxt[v]:
                    nxt[v] = cur + t[(i, b)]
                    pick[v] = (u, b)
        picks.append(pick)
        best = nxt
    u = min(range(budget + 1), key=lambda x: best[x])
    total, windows = best[u], []
    for pick in reversed(picks):
        u, b = pick[u]
        windows.append(b)
    return total, windows[::-1]


def interrupt_price(name, table):
    key = name.replace("GeForce ", "").split(" (")[0].replace(" 80GB", "").replace(" SXM5", " SXM").strip()
    for k, v in table.items():
        kk = k.split(" (")[0].strip()
        if kk and (kk == key or kk in key or key in kk):
            return v
    return None


def main():
    t = per_class_times()
    hw = json.load(open(os.path.join(ROOT, "results/hardware/gpus.json")))
    interruptible = {p["gpu"]: p["usd_per_gpu_hour_low"] for p in hw["vast_prices"] if "interrupt" in p["pricing_kind"]}
    out = [
        "| GPU | VRAM | windows, classes 0-7 | 4090-eq years | speed vs 4090 | GPU-years | array build per process | CPU threads per GPU | $ per 2^72, on-demand | interruptible (lowest bid) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows = []
    for g in hw["gpus"]:
        rel, vram, price = g["int32_rel_4090_est"], g["vram_gb"], g["vast_usd_hr"]
        if not rel or not vram or not price:
            continue
        gib = vram if g["gpu"].startswith(("GeForce", "RTX")) else vram * 1e9 / 2**30  # GeForce/RTX quote GiB
        w, windows = best_windows(t, gib)
        years = w / rel
        build_s = sum(2**b for b in windows) / 2**30 * BUILD_S_PER_GPAT / rel
        threads = TREE_NODES * NS_PER_NODE_THREAD * 1e-9 / (years * YEAR_H * 3600)
        ip = interrupt_price(g["gpu"], interruptible)
        rows.append((years * YEAR_H * price, g["gpu"], vram, windows, w, rel, years, build_s, threads, price, ip))
    for cost, name, vram, windows, w, rel, years, build_s, threads, price, ip in sorted(rows):
        icost = f"${years * YEAR_H * ip:,.0f} at ${ip:.2f}/h" if ip else "-"
        out.append(
            f"| {name[:34]} | {vram:g} GB | {','.join(map(str, windows[:8]))} | {w:.3f} | {rel:.2f} | {years:.2f} | "
            f"{build_s / 60:.1f} min | {threads:.0f} | ${cost:,.0f} at ${price:.2f}/h | {icost} |"
        )
    print("\n".join(out))
    open(os.path.join(ROOT, "results/hardware/cost_2p72.md"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
