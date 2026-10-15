"""Truth, naive baseline and ours, side by side, from data/ground_truth.yaml.

Usage: python src/evaluate.py
Writes results/evaluation.md and puts the same text into README.md between the results markers.
"""
import os

import numpy as np
import yaml

from classify import analyse, count_changes, naive, states

# The ground truth names objects in plain words. The detector named them from the prompt list.
# Written after the A vs B run, from the labels the system printed. It changes no result, only how they are read.
DETECTOR_NAME = {"flashcards": "box", "water bottle": "bottle"}
PAIRS = {"AB": "A_vs_B", "BC": "B_vs_C"}
START, END = "<!-- results start -->", "<!-- results end -->"


def answer(records, label):
    """Every state the method gave objects of this label, e.g. "not checked + added"."""
    found = [r["state"] for r in sorted(records, key=lambda r: r["video"])
             if r["label"] == label and r["state"] not in ("same as video 1", "moved to")]
    return " + ".join(found) or "not found"


def changes(records):
    n = count_changes(records)
    return f"{n} change" if n == 1 else f"{n} changes"


def evaluate(config, truth):
    lines = []
    for pair, key in PAIRS.items():
        records, _ = analyse(np.load(f"cache/{pair}.npz"), config)
        ours = states([dict(r) for r in records], config)
        base = naive([dict(r) for r in records], config)
        right = {"naive": 0, "ours": 0}
        lines += [f"### Video {pair[0]} vs video {pair[1]}", "",
                  "| Object | Truth | Naive | Ours |", "|---|---|---|---|"]
        for name, expected in truth["pairs"][key].items():
            label = DETECTOR_NAME.get(name, name)
            got = {"naive": answer(base, label), "ours": answer(ours, label)}
            cells = []
            for method in ("naive", "ours"):
                right[method] += got[method] == expected
                cells.append(got[method] if got[method] == expected else f"**{got[method]}** (wrong)")
            lines.append(f"| {name} | {expected} | {cells[0]} | {cells[1]} |")
        total = len(truth["pairs"][key])
        lines += ["", f"Naive gets {right['naive']} of {total} right and reports {changes(base)}. "
                      f"Ours gets {right['ours']} of {total} right and reports {changes(ours)}.", ""]
    lines.append("Video B vs video C has no real changes, so every change reported there is a false alarm.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    with open("data/ground_truth.yaml") as f:
        truth = yaml.safe_load(f)

    text = evaluate(config, truth)
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation.md", "w") as f:
        f.write(text)
    print(text)

    if os.path.exists("README.md"):
        readme = open("README.md").read()
        if START in readme and END in readme:
            before, rest = readme.split(START)
            after = rest.split(END)[1]
            with open("README.md", "w") as f:
                f.write(f"{before}{START}\n{text}{END}{after}")
            print("README.md table updated")
