"""The core test: could the other video see an object's spot, and what was there?

Usage: python src/visibility.py --pair AB
"""
import argparse

import numpy as np
import yaml

from objects import build_objects, footprint

NOT_VISIBLE, EMPTY, PRESENT, OTHER = 0, 1, 2, 3


def voxel_points(points, size):
    """Keep one point per cube of this size, so a patch filmed 15 times counts once."""
    _, first = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_index=True)
    return points[first]


def camera_answers(points, extrinsic, intrinsic, depth, confidence, owner, same, config):
    """What one camera says about each point: NOT_VISIBLE, EMPTY, PRESENT or OTHER.

    owner: for each pixel, the index of the other video's object whose mask covers it, or -1.
    same: for each of those objects, True if its footprint matches the object being tested.
    """
    camera = points @ extrinsic[:, :3].T + extrinsic[:, 3]
    z = camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intrinsic[0, 0] * camera[:, 0] / z + intrinsic[0, 2]
        v = intrinsic[1, 1] * camera[:, 1] / z + intrinsic[1, 2]
    height, width = depth.shape
    inside = (z > 0) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)

    answers = np.full(len(points), NOT_VISIBLE)
    index = np.flatnonzero(inside)
    u, v = np.round(u[index]).astype(int), np.round(v[index]).astype(int)
    ratio = depth[v, u] / z[index]  # camera's depth over the point's distance
    trusted = confidence[v, u] > config["confidence_cutoff"]
    blocked = ratio < 1 - config["depth_tolerance"]
    agrees = np.abs(ratio - 1) <= config["depth_tolerance"]
    # Trusted depth: seen unless something is in front. Low-confidence depth (plain white
    # table): seen only if it puts the surface where the point is.
    seen = np.where(trusted, ~blocked, agrees)

    pixel_owner = owner[v[seen], u[seen]]
    is_same = np.append(same, False)[pixel_owner]  # -1 picks the appended False
    answers[index[seen]] = np.where(pixel_owner < 0, EMPTY, np.where(is_same, PRESENT, OTHER))
    return answers


def summarise(answers):
    """Coverage and present fraction from every camera's answers, shape (cameras, points)."""
    covered = (answers != NOT_VISIBLE).any(axis=0)
    present_votes = (answers == PRESENT).sum(axis=0)
    absent_votes = ((answers == EMPTY) | (answers == OTHER)).sum(axis=0)
    # Majority among the cameras that saw the point. A tie counts as present, the safe side.
    present = covered & (present_votes >= absent_votes)
    coverage = covered.mean()
    present_fraction = present.sum() / covered.sum() if covered.any() else 0.0
    return coverage, present_fraction


def owner_maps(cache, others, frames):
    """For each frame, which of the other video's objects covers each pixel (-1 for none)."""
    maps = {frame: np.full(cache["depth"].shape[1:], -1) for frame in frames}
    for k, obj in enumerate(others):
        for m in obj["masks"]:
            maps[cache["mask_frame"][m]][cache["mask"][m]] = k
    return maps


def check(obj, others, cache, scale, config):
    """Test one object against every camera of the other video."""
    points = voxel_points(obj["points"], config["voxel_m"]) / scale  # back to VGGT units
    frames = np.flatnonzero(cache["video"] != obj["video"])
    maps = owner_maps(cache, others, frames)
    long_side, short_side = footprint(obj["points"])
    same = np.array([abs(o["size"][0] - long_side) <= config["size_tolerance_m"]
                     and abs(o["size"][1] - short_side) <= config["size_tolerance_m"] for o in others], bool)

    answers = np.array([camera_answers(points, cache["extrinsic"][f], cache["intrinsic"][f], cache["depth"][f],
                                       cache["confidence"][f], maps[f], same, config) for f in frames])
    coverage, present_fraction = summarise(answers)
    checked_by = [int(cache["frame_number"][f]) for f, a in zip(frames, answers) if (a != NOT_VISIBLE).any()]
    return {"coverage": coverage, "present_fraction": present_fraction, "checked_by": checked_by}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, help="two letters, e.g. AB or BC")
    args = parser.parse_args()
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    cache = np.load(f"cache/{args.pair}.npz")
    objects, scale_check = build_objects(cache, config)
    for obj in objects:
        obj["size"] = footprint(obj["points"])
    for obj in sorted(objects, key=lambda o: (o["video"], o["label"])):
        others = [o for o in objects if o["video"] != obj["video"]]
        result = check(obj, others, cache, scale_check["scale"], config)
        name = f"{args.pair[obj['video'] - 1]}: {obj['label']}"
        print(f"{name:16s} coverage {result['coverage']:4.0%}  present {result['present_fraction']:4.0%}  "
              f"checked from {len(result['checked_by'])} of 15 views of {args.pair[2 - obj['video']]}")
