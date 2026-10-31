#!/usr/bin/env python3
"""Export CPU/GPU algorithm flame comparisons as PNG/SVG (requires matplotlib).

CPU samples, GPU operation seconds and diagnostic lane-clock cycles retain
separate scales. Sampled CPU stacks are summarized at the algorithm category
level in the static figure; the interactive viewer preserves the full stacks.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from flame_offload import validate


METRICS = {
    "sum_device_operation_seconds": ("seconds", "GPU operation-seconds"),
    "cpu_core_seconds": ("seconds", "CPU core-seconds"),
    "cpu_samples": ("samples", "CPU cycle samples"),
    "gpu_cycles": ("cycles", "sampled lane elapsed cycles"),
    "gpu_phase_samples": ("samples", "GPU phase samples"),
}


def metric(root, resource):
    return root.get("metric", "sum_device_operation_seconds" if resource == "gpu" else "cpu_core_seconds")


def value(node, kind):
    return node.get(METRICS[kind][0], node.get("sample_count", 0))


def format_value(number, kind):
    return f"{number:,.0f}" if kind in ("cpu_samples", "gpu_cycles", "gpu_phase_samples") else f"{number:.3f}"


def depth(node):
    return 1 + max((depth(child) for child in node.get("children", [])), default=0)


def color(node, level):
    name = node["name"].lower()
    if any(word in name for word in ("unsplit", "unattributed", "other host", "unknown", "unresolved")):
        return "#ded6ca"
    if level == 0:
        return "#e9a54c"
    palette = ["#f4bd69", "#f2a35b", "#ee8e58", "#ed7956", "#e96e50", "#f4ce7b"]
    return palette[hashlib.sha256(name.encode()).digest()[0] % len(palette)]


def draw(axis, root, kind, domain, levels, max_depth=None):
    total = value(root, kind)
    axis.set_xlim(0, 1)
    axis.set_ylim(-.1, levels + .05)
    axis.axis("off")

    def visit(node, start, level):
        weight = value(node, kind)
        width = weight / domain if domain else 0
        box = Rectangle((start, level), width, .91, facecolor=color(node, level),
                        edgecolor="white", linewidth=.8)
        axis.add_patch(box)
        if width > .037:
            number = format_value(weight, kind) if level == 0 else f"{weight / total * 100:.1f}%" if total else "0%"
            label = f"{node['name']} · {number}"
            limit = int(width * 88)
            if len(label) > limit:
                label = label[:max(0, limit - 1)] + "…"
            text = axis.text(start + .006, level + .45, label, va="center",
                             ha="left", fontsize=8.5, color="#2e2422", clip_on=True)
            text.set_clip_path(box)
        offset = start
        if max_depth is None or level + 1 < max_depth:
            for child in node.get("children", []):
                visit(child, offset, level + 1)
                offset += value(child, kind) / domain if domain else 0

    visit(root, 0, 0)


def descendants(node):
    return [node] + [child for item in node.get("children", []) for child in descendants(item)]


def gpu_summary(root):
    nodes = descendants(root)
    expansion = next((n for n in nodes if any("cv_expand_parents" in raw for raw in n.get("raw_kernel_names", []))), None)
    if expansion is None:
        expansion = next((n for n in nodes if "endpoint expansion" in n["name"].lower() or "parent expansion" in n["name"].lower()), None)
    h2d = next((n for n in nodes if n.get("transfer_direction") == "H2D"), None)
    d2h = next((n for n in nodes if n.get("transfer_direction") == "D2H"), None)
    lines = []
    if expansion:
        lines.append(f"Endpoint expansion: {expansion['seconds']:.4f} GPU operation-s")
    transfers = []
    if h2d:
        transfers.append(f"Host → GPU: {h2d['seconds']:.5f}")
    if d2h:
        transfers.append(f"GPU → host: {d2h['seconds']:.5f}")
    if transfers:
        lines.append(" · ".join(transfers) + " GPU operation-s")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/tail_offload/flame_profile/trace_trees.json"))
    parser.add_argument("--out", type=Path, default=Path("results/tail_offload/flamegraph/comparison"))
    parser.add_argument("--scale", choices=("absolute", "relative"), default="absolute")
    parser.add_argument("--cpu-depth", type=int, default=3,
                        help="Sampled CPU levels in static view; 3 shows algorithms and substeps, 0 shows all frames")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    validate(data)
    variants = data["variants"]
    if len(variants) != 2:
        raise ValueError("exactly two matched variants are required")
    has_phases = all("gpu_phases" in item for item in variants)
    sampled = any(metric(item["cpu"], "cpu") == "cpu_samples" for item in variants)
    resources = ["gpu", "gpu_phases", "cpu"] if has_phases else ["gpu", "cpu"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#192e40",
                         "svg.fonttype": "none", "svg.hashsalt": "collatz-algorithm-flame"})
    fig = plt.figure(figsize=(16, 12.5 if has_phases else 10), facecolor="white")
    fig.text(.04, .951, "Before / after: which algorithms did the work", fontsize=24, weight="bold")
    scale_text = "Equal widths mean equal amounts within each measurement type." if args.scale == "absolute" else "Each tree fills its panel; width shows share of that tree."
    fig.text(.04, .918, scale_text + " CPU samples, GPU seconds and lane clocks stay separate.",
             fontsize=11, color="#536879")
    positions = [("gpu", .610, .160), ("gpu_phases", .390, .125), ("cpu", .180, .125)] if has_phases else [("gpu", .485, .230), ("cpu", .190, .180)]
    titles = {"gpu": "GPU operations", "gpu_phases": "GPU verifier internals · diagnostic", "cpu": "CPU algorithms · sampled stacks" if sampled else "CPU work"}
    for i, item in enumerate(variants):
        x = .04 + .49 * i
        fig.text(x, .865, "Before" if i == 0 else "After · CPU tail offload", fontsize=19, weight="bold")
        wall = item.get("wall_seconds")
        cpu = item.get("process_cpu_seconds")
        facts = []
        if wall is not None:
            facts.append(f"Profiled wall: {wall:.3f} s")
        if cpu is not None:
            facts.append(f"Measured CPU: {cpu:.3f} core-s")
        fig.text(x, .838, " · ".join(facts), fontsize=10, color="#536879")
        for resource, y, height in positions:
            root = item[resource]
            kind = metric(root, resource)
            other = variants[1-i][resource]
            if metric(other, resource) != kind:
                raise ValueError("before and after must share each resource metric")
            total = value(root, kind)
            domain = max(total, value(other, kind)) if args.scale == "absolute" else total
            max_depth = args.cpu_depth or None if kind == "cpu_samples" else None
            levels = max(depth(v[resource]) for v in variants)
            if max_depth:
                levels = min(levels, max_depth)
            fig.text(x, y + height + .040, titles[resource], fontsize=12, weight="bold")
            fig.text(x, y + height + .021,
                     f"{format_value(total, kind)} {METRICS[kind][1]}", fontsize=10, color="#536879")
            draw(fig.add_axes([x, y, .43, height]), root, kind, domain, levels, max_depth)
            if resource == "gpu":
                for line, text in enumerate(gpu_summary(root)):
                    fig.text(x, y-.017-line*.014, text, fontsize=8.2, color="#536879")
            if resource == "gpu_phases":
                names = [("Step 2", "Screening"), ("Step 3", "Exact tails"), ("Warp", "Warp/checksum")]
                parts = []
                for prefix, label in names:
                    node = next((n for n in root.get("children", []) if n["name"].startswith(prefix)), None)
                    if node and total:
                        parts.append(f"{label}: {100*value(node,kind)/total:.2f}%")
                fig.text(x, y-.017, " · ".join(parts), fontsize=8.2, color="#536879")
            if resource == "cpu" and sampled:
                biggest = sorted(root.get("children", []), key=lambda n: value(n, kind), reverse=True)[:3]
                if not total:
                    fig.text(x, y-.017, "No CPU samples", fontsize=8.2, color="#536879")
                for line, node in enumerate(biggest):
                    overview = f"{node['name']}: {100*value(node,kind)/total:.1f}% ({value(node,kind):,} samples)"
                    fig.text(x, y-.017-line*.014, overview, fontsize=8.2, color="#536879")
    footer = [
        "CPU categories partition actual samples; open the interactive view to expand sampled functions." if sampled else "Measured CPU stages; unsplit host work remains explicit.",
        "GPU lane clocks include stalls, scheduling and instrumentation; they are not SM busy cycles or a split of kernel seconds." if has_phases else "GPU operation durations overlap across streams and are not exclusive GPU busy time.",
        "Live work from stored K38 onward; frontier and table precomputation excluded. Profiling captures do not replace repeated benchmark timings.",
    ]
    fig.text(.04, .092, footer[0], fontsize=10.5, weight="bold")
    fig.text(.04, .064, footer[1], fontsize=9.5, color="#536879")
    fig.text(.04, .039, footer[2], fontsize=9.5, color="#536879")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "svg"]:
        fig.savefig(args.out.with_suffix("." + extension), dpi=150,
                    metadata={"Title": "CPU/GPU algorithm flame comparison"})
    plt.close(fig)
    print(f"Saved {args.out}.png and .svg")


if __name__ == "__main__":
    main()
