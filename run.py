"""Everything after Colab, for one pair: objects, visibility test, states, then the outputs.

Usage: python run.py --pair AB [--gif]
Reads cache/<pair>.npz. Writes results/<pair>/results.json, changes.md, viewer.html (and viewer.gif).
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, "src")
from classify import analyse, states  # noqa: E402
from describe import describe  # noqa: E402
from viewer import figure, write_gif  # noqa: E402


def row(r, pair):
    """One object as it appears in results.json. Rounded the same way the sentences round."""
    out = {"id": r["id"], "label": r["label"], "video": pair[r["video"] - 1], "state": r["state"],
           "centre_m": [round(float(c), 3) for c in r["centre"]],
           "size_cm": [round(100 * float(s)) for s in r["size"]],
           "coverage": round(float(r["coverage"]), 2), "present_fraction": round(float(r["present_fraction"]), 2),
           "checked_by_frames": r["checked_by"]}
    if "moved_to" in r:
        out.update(moved_to=r["moved_to"], moved_cm=round(r["moved_cm"]))
    return out


parser = argparse.ArgumentParser()
parser.add_argument("--pair", required=True, help="two letters, e.g. AB or BC")
parser.add_argument("--out", default="results")
parser.add_argument("--gif", action="store_true", help="also render a turning GIF of the viewer (about 4 minutes)")
args = parser.parse_args()
with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

cache = np.load(f"cache/{args.pair}.npz")
records, scale = analyse(cache, config)
records = states(records, config)
folder = os.path.join(args.out, args.pair)
os.makedirs(folder, exist_ok=True)

with open(os.path.join(folder, "results.json"), "w") as f:
    json.dump({"pair": args.pair, "objects": [row(r, args.pair) for r in records]}, f, indent=2)

lines = describe(records, args.pair, config)
with open(os.path.join(folder, "changes.md"), "w") as f:
    f.write(f"# Video {args.pair[0]} vs video {args.pair[1]}\n\n")
    f.writelines(f"- {text} (object {i})\n" for i, text in lines)
print(open(os.path.join(folder, "changes.md")).read())

fig, normal, along = figure(records, cache, scale, args.pair, config)
fig.write_html(os.path.join(folder, "viewer.html"), include_plotlyjs=True)
if args.gif:
    write_gif(fig, normal, along, os.path.join(folder, "viewer.gif"))
print(f"wrote {folder}/")
