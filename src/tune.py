"""Safe range of each threshold on B vs C, the no-change control. Never loads A vs B.

A value is safe if B vs C still reports zero changes with it, all other thresholds unchanged.
Usage: python src/tune.py
"""
import numpy as np
import yaml

from classify import analyse, count_changes, naive, states

SWEEPS = {
    # Thresholds that change the objects or the visibility test: rerun everything.
    "confidence_cutoff": [1.0, 1.25, 1.5, 2.0, 3.0, 4.0],
    "mask_erosion_px": [0, 1, 3, 5, 8],
    "merge_radius_m": [0.005, 0.01, 0.02, 0.04, 0.08],
    "merge_overlap": [0.1, 0.3, 0.5, 0.7, 0.9],
    "min_views": [1, 2, 3, 5, 8],
    "depth_tolerance": [0.01, 0.02, 0.05, 0.1, 0.2],
    "size_tolerance_m": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
    # Thresholds that only change the final rules: reuse the visibility results.
    "c_min": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "mostly": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "r_naive": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
}
RULES_ONLY = {"c_min", "mostly", "r_naive"}

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)
cache = np.load("cache/BC.npz")
base_records, _ = analyse(cache, config)

for name, values in SWEEPS.items():
    changes = {}
    for value in values:
        trial = {**config, name: value}
        records = base_records if name in RULES_ONLY else analyse(cache, trial)[0]
        method = naive if name == "r_naive" else states
        changes[value] = count_changes(method([dict(r) for r in records], trial))
    # The naive baseline has no "not checked" state, so it can never reach zero on B vs C.
    # Its safe values are the ones with the fewest changes.
    best = min(changes.values()) if name == "r_naive" else 0
    safe = [value for value, n in changes.items() if n == best]
    print(f"{name:18s} now {config[name]!s:6s} safe on BC ({best} changes): {safe}")
