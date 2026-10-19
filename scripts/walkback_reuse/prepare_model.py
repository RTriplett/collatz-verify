"""Freeze the existing 24 GB / N72 / A10 cost model for the reuse probes.

Run from any directory; JSON goes to stdout. This does not time a GPU.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import hardware_cost

windows = [32, 35, 35, 35, 35, 35, 32, 32] + [27] * 8
times = hardware_cost.per_class_times()
model = {
    "windows": windows,
    "baseline_seconds": sum(times[i, b] for i, b in enumerate(windows))
    * hardware_cost.YEAR_H * 3600,
    "fit": json.loads((ROOT / "results/gpu_fit.json").read_text()),
}
print(json.dumps(model, indent=2))
