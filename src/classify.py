"""Five states per object, plus the naive baseline that skips the visibility test.

Usage: python src/classify.py --pair BC
"""
import argparse

import numpy as np
import yaml

from objects import build_objects, footprint
from visibility import check

CHANGES = {"removed", "added", "moved"}


def analyse(cache, config):
    """Build the objects of both videos and run the visibility test on each. Also returns the scale."""
    objects, scale_check = build_objects(cache, config)
    for obj in objects:
        obj["size"] = footprint(obj["points"])
    records = []
    for obj in objects:
        others = [o for o in objects if o["video"] != obj["video"]]
        result = check(obj, others, cache, scale_check["scale"], config)
        records.append({"id": len(records), "label": obj["label"], "video": obj["video"],
                        "centre": obj["centre"], "size": obj["size"], "points": obj["points"], **result})
    return records, scale_check["scale"]


def pair_moves(records):
    """A removed and an added object with the same label become one move."""
    for removed in [r for r in records if r["state"] == "removed"]:
        added = [r for r in records if r["state"] == "added" and r["label"] == removed["label"]]
        if added:
            nearest = min(added, key=lambda r: np.linalg.norm(r["centre"] - removed["centre"]))
            removed["state"], nearest["state"] = "moved", "moved to"
            removed["moved_cm"] = 100 * float(np.linalg.norm(nearest["centre"] - removed["centre"]))
            removed["moved_to"] = nearest["id"]
    return records


def states(records, config):
    """Our method: not checked, unchanged, removed, added or moved."""
    for r in records:
        if r["coverage"] < config["c_min"]:
            r["state"] = "not checked"
        elif r["present_fraction"] >= config["mostly"]:
            # A video 2 object that is present in video 1 is the same object, reported once under video 1.
            r["state"] = "unchanged" if r["video"] == 1 else "same as video 1"
        else:
            r["state"] = "removed" if r["video"] == 1 else "added"
    return pair_moves(records)


def same_size(a, b, tolerance):
    return abs(a["size"][0] - b["size"][0]) <= tolerance and abs(a["size"][1] - b["size"][1]) <= tolerance


def naive(records, config):
    """The baseline: an object with no same-size object nearby in the other video has changed."""
    result = []
    for r in records:
        match = any(o["video"] != r["video"] and same_size(r, o, config["size_tolerance_m"])
                    and np.linalg.norm(o["centre"] - r["centre"]) <= config["r_naive"] for o in records)
        state = ("unchanged" if r["video"] == 1 else "same as video 1") if match else \
                ("removed" if r["video"] == 1 else "added")
        result.append({**r, "state": state})
    return pair_moves(result)


def count_changes(records):
    return sum(r["state"] in CHANGES for r in records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, help="two letters, e.g. AB or BC")
    args = parser.parse_args()
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    records, _ = analyse(np.load(f"cache/{args.pair}.npz"), config)
    for name, method in [("ours", states), ("naive", naive)]:
        result = method([dict(r) for r in records], config)
        print(f"{name}: {count_changes(result)} changes")
        for r in sorted(result, key=lambda r: (r["video"], r["label"])):
            moved = f"  {r['moved_cm']:.0f} cm" if "moved_cm" in r else ""
            print(f"  {args.pair[r['video'] - 1]}: {r['label']:12s} {r['state']}{moved}")
