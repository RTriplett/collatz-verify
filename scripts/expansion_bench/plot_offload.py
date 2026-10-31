#!/usr/bin/env python3
"""Render audited before/after CPU-offload comparisons (requires matplotlib).

Produces PNG/SVG figures and a standalone HTML comparison switcher. Elapsed
seconds and CPU core-seconds deliberately have separate axes: GPU and CPU
execution overlap, and CUDA stream-event sums are not exclusive GPU time.
"""

import argparse
import html
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch

from summarize_offload import summarize_offload


GROUPS = {
    "parents": {
        "title": "K52 parent dispatch",
        "detail": "16 double iterations · 262,144 records per launch · 4 CPU workers",
        "variants": ["novote_oee", "defer_count", "defer16_cap262k_w4"],
        "labels": ["Before", "Control\nBookkeeping only", "After\n4 CPU workers"],
    },
    "compact": {
        "title": "Compact 6+6 dispatch",
        "detail": "24 double iterations · 262,144 records per launch · 2 CPU workers",
        "variants": ["compact66", "compact_defer24_cap262k_w2"],
        "labels": ["Before", "After\n2 CPU workers"],
    },
}

INK = "#162b3d"
MUTED = "#536879"
BLUE = "#34749a"
ORANGE = "#c96930"
GRAY = "#8997a3"


def figure(group, variants, raw, run, path):
    names = group["variants"]
    values = [variants[name]["continuous_aggregate"] for name in names]
    before, after = values[0], values[-1]
    repeat_count = len(raw[names[0]])
    if not repeat_count or any(len(raw[name]) != repeat_count for name in names):
        raise ValueError("matched continuous repetitions are required")
    wall = [v["wall_seconds"] for v in values]
    cpu = [v["process_cpu_seconds"] for v in values]
    tail = [v["deferred_cpu_seconds"] for v in values]
    other = [total - continuation for total, continuation in zip(cpu, tail)]
    if min(other) < 0:
        raise ValueError("worker CPU time exceeds process CPU time")
    slowdown = 100 * (wall[-1] / wall[0] - 1)
    cpu_ratio = cpu[-1] / cpu[0]
    shifted = 100 * after["cpu_remainder_iteration_fraction"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": INK,
        "svg.fonttype": "none", "svg.hashsalt": "collatz-offload",
    })
    fig = plt.figure(figsize=(14, 8.6), facecolor="#ffffff")
    fig.text(.06, .942, "CPU tail offload: where the time went", fontsize=24, weight="bold")
    fig.text(.06, .903, group["title"] + "  |  " + group["detail"], fontsize=11.5, color=MUTED)
    for x, number, description in [
        (.06, f"+{slowdown:.1f}%", "elapsed time"),
        (.37, f"{cpu_ratio:.1f}×", "CPU core-seconds"),
        (.68, f"{shifted:.4f}%", "GPU-origin tail iterations moved to CPU"),
    ]:
        fig.text(x, .822, number, fontsize=27, color=ORANGE, weight="bold")
        fig.text(x, .788, description, fontsize=10.5, color=MUTED)

    ax_wall = fig.add_axes([.18, .325, .31, .335])
    ax_cpu = fig.add_axes([.62, .325, .32, .335])
    y = list(range(len(values)))
    colors = [BLUE] + ([GRAY] if len(values) == 3 else []) + [ORANGE]
    ax_wall.barh(y, wall, height=.44, color=colors, zorder=3)
    ax_cpu.barh(y, other, height=.44, color=BLUE, zorder=3)
    ax_cpu.barh(y, tail, left=other, height=.44, color=ORANGE, zorder=3)
    for axis, limit, ticks, title, unit in [
        (ax_wall, 12.6, [0, 3, 6, 9, 12], "Elapsed runtime", "Seconds to complete the pipeline"),
        (ax_cpu, 21.5, [0, 5, 10, 15, 20], "Where CPU time was spent", "CPU core-seconds, summed across threads"),
    ]:
        axis.set_xlim(0, limit)
        axis.set_ylim(len(values) - .48, -.63)
        axis.set_xticks(ticks)
        axis.set_yticks(y)
        axis.tick_params(axis="both", length=0, pad=9)
        axis.set_yticklabels(group["labels"] if axis == ax_wall else [""] * len(y))
        axis.set_xlabel(unit, labelpad=13, fontsize=10)
        axis.set_title(title, loc="left", pad=30, fontsize=14, weight="bold")
        axis.grid(axis="x", color="#e7edf1", zorder=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    ax_cpu.legend(handles=[Patch(color=BLUE, label="Other CPU work"),
                           Patch(color=ORANGE, label="Tail continuations")],
                  loc="lower left", bbox_to_anchor=(-.02, 1.025), frameon=False,
                  ncol=2, fontsize=9, borderaxespad=0, handlelength=1.2,
                  columnspacing=1.0)

    for i, name in enumerate(names):
        for axis, mean, key, suffix in [
            (ax_wall, wall[i], "wall_seconds", " s"),
            (ax_cpu, cpu[i], "process_cpu_seconds", ""),
        ]:
            observations = [row[key] for row in raw[name]]
            # The summary and displayed raw observations must refer to the same mean.
            if abs(statistics.mean(observations) - mean) > 1e-9:
                raise ValueError("summary/raw timing disagreement")
            axis.errorbar(mean, i, xerr=[[mean - min(observations)],
                                        [max(observations) - mean]],
                          fmt="none", ecolor=INK, capsize=5, linewidth=1.2, zorder=4)
            axis.text(mean + (0.22 if axis == ax_wall else .32), i,
                      f"{mean:.2f}{suffix}", va="center", fontsize=12, weight="bold")
        if tail[i]:
            ax_cpu.text(other[i] + tail[i] / 2, i, f"{tail[i]:.2f}",
                        ha="center", va="center", color="white", weight="bold", fontsize=12)

    fig.add_artist(FancyBboxPatch((.055, .145), .89, .086,
                   boxstyle="round,pad=0.010,rounding_size=0.012",
                   transform=fig.transFigure, linewidth=0, facecolor="#f0f5f8"))
    fig.text(.07, .194,
             f"After: {other[-1]:.2f} core-s other CPU work + {tail[-1]:.2f} core-s tail continuations"
             f" = {cpu[-1]:.2f} core-s", fontsize=12, weight="bold")
    fig.text(.07, .160,
             "CPU and GPU work overlap. CPU core-seconds are not added to elapsed seconds.",
             fontsize=10.5, color=MUTED)
    fig.text(.06, .087,
             f"Same {run['accepted_draws']} accepted roots from {run['baseline_draws']} K38 draws; sampled N72 work on RTX 4090 + Ryzen 7700X.\n"
             f"Means of {repeat_count} continuous repetitions. Whiskers show observed min–max, not confidence intervals.",
             fontsize=9, color=MUTED, linespacing=1.7)
    fig.text(.06, .031,
             "Other CPU work includes expansion, dispatch, bookkeeping, fallback and worker overhead. One-time setup is excluded.\n"
             "GPU stream timers overlap and do not separate screening from tail stepping; no exclusive GPU-time split is inferred.",
             fontsize=8.5, color=MUTED, linespacing=1.6)
    for extension in ["png", "svg"]:
        fig.savefig(path.with_suffix("." + extension), dpi=160,
                    metadata={"Title": "CPU tail offload: where the time went"})
    plt.close(fig)


def make_html(out, run, variants):
    panels = []
    for name, group in GROUPS.items():
        svg = (out / f"time_{name}.svg").read_text()
        svg = svg[svg.index("<svg"):]
        # Keep the two embedded SVGs' clip-path IDs independent.
        import re
        ids = re.findall(r'\bid="([^"]+)"', svg)
        for old in ids:
            svg = svg.replace(f'id="{old}"', f'id="{name}_{old}"')
            svg = svg.replace(f'#{old}"', f'#{name}_{old}"')
            svg = svg.replace(f'#{old})', f'#{name}_{old})')
        rows = []
        for variant, label in zip(group["variants"], group["labels"]):
            value = variants[variant]["continuous_aggregate"]
            total, continuation = value["process_cpu_seconds"], value["deferred_cpu_seconds"]
            rows.append(f"<tr><th scope='row'>{html.escape(label.replace(chr(10), ': '))}</th>"
                        f"<td>{value['wall_seconds']:.3f}</td><td>{total - continuation:.3f}</td>"
                        f"<td>{continuation:.3f}</td><td>{total:.3f}</td>"
                        f"<td>{100 * value['cpu_remainder_iteration_fraction']:.5f}%</td></tr>")
        panels.append(f"<section id='{name}' aria-label='{html.escape(group['title'])}'"
                      + (" hidden" if name != "parents" else "") + f">{svg}"
                      f"<p class='downloads'>Export: <a href='time_{name}.png'>PNG</a> · "
                      f"<a href='time_{name}.svg'>SVG</a></p>"
                      "<details><summary>Measured values</summary><div class='table-wrap'><table>"
                      "<thead><tr><th>Configuration</th><th>Elapsed s</th><th>Other CPU core-s</th>"
                      "<th>Continuation core-s</th><th>Total CPU core-s</th><th>Iterations moved</th></tr></thead>"
                      f"<tbody>{''.join(rows)}</tbody></table></div></details></section>")
    body = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CPU tail offload — before and after</title>
<style>
* { box-sizing: border-box; } body { margin: 0; background: #eff3f6; color: #162b3d;
font: 15px/1.55 system-ui, sans-serif; } main { max-width: 1320px; margin: 24px auto;
background: white; border-radius: 16px; padding: 22px; } nav { display: flex; gap: 10px;
align-items: center; flex-wrap: wrap; } button { border: 1px solid #c4d2db; border-radius: 8px;
padding: 9px 16px; background: white; color: #162b3d; cursor: pointer; font: inherit; }
button[aria-pressed="true"] { background: #162b3d; color: white; }
button:focus-visible, a:focus-visible { outline: 3px solid #c96930; outline-offset: 3px; }
section[hidden] { display: none; } svg { width: 100%; height: auto; display: block; }
a { color: #276c94; } .downloads, details, footer { margin: 12px 4%; }
footer { font-size: 13px; color: #536879; } summary { cursor: pointer; }
.table-wrap { overflow-x: auto; } table { border-collapse: collapse; width: 100%; margin-top: 12px; }
th, td { padding: 10px; border-bottom: 1px solid #dfe7ed; text-align: right; white-space: nowrap; }
th:first-child { text-align: left; } @media (max-width: 650px) { main { margin: 0; padding: 10px; }
section { overflow-x: auto; } svg { min-width: 850px; } }
@media print { nav, .downloads, details { display: none; } main { padding: 0; margin: 0; } }
</style><main><nav aria-label="Choose matched before/after comparison">
<strong>Comparison:</strong><button type="button" data-target="parents" aria-pressed="true">K52 parents · 4 workers</button>
<button type="button" data-target="compact" aria-pressed="false">Compact dispatch · 2 workers</button>
</nav>__PANELS__
<footer>Source: <a href="../confirmation_summary.json">audited summary</a> and
<a href="../confirmation/pipeline.jsonl">raw continuous timings</a>.
<a href="../README.md">Full experiment report</a>. This is sampled work, not a completed 2<sup>72</sup> verification.
All data and charts are local; this page needs no network connection.</footer></main>
<script>
const buttons = document.querySelectorAll('button[data-target]');
function select(name) { buttons.forEach(button => {
const active = button.dataset.target === name;
button.setAttribute('aria-pressed', String(active));
document.getElementById(button.dataset.target).hidden = !active;
}); }
buttons.forEach(button => button.addEventListener('click', () => select(button.dataset.target)));
</script></html>"""
    (out / "index.html").write_text(body.replace("__PANELS__", "\n".join(panels)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/tail_offload/confirmation"))
    parser.add_argument("--out", type=Path, default=Path("results/tail_offload/visualization"))
    args = parser.parse_args()
    run = summarize_offload(args.run)
    variants = {v["name"]: v for v in run["variants"]}
    rows = [json.loads(line) for line in (args.run / "pipeline.jsonl").read_text().splitlines() if line.strip()]
    raw = {name: [row for row in rows if row.get("type") == "aggregate_trial"
                  and row["variant"] == name] for group in GROUPS.values() for name in group["variants"]}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, group in GROUPS.items():
        figure(group, variants, raw, run, args.out / f"time_{name}")
    make_html(args.out, run, variants)
    print(f"Saved PNG, SVG and interactive HTML comparisons in {args.out}")


if __name__ == "__main__":
    main()
