"""Turn masks and depth into 3D objects, one set per video, in metres.

Usage: python src/objects.py --pair AB
Prints the objects per video and writes cache/objects_<pair>.html to look at.
"""
import argparse
import re
from collections import Counter

import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree


def unproject(depth, extrinsic, intrinsic):
    """Every pixel of one frame back into 3D world coordinates, shape (H, W, 3)."""
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    # Pixel to camera coordinates, using the focal length and image centre.
    x = (u - intrinsic[0, 2]) * depth / intrinsic[0, 0]
    y = (v - intrinsic[1, 2]) * depth / intrinsic[1, 1]
    camera_points = np.stack([x, y, depth], axis=-1)
    # Camera to world. The extrinsic maps world to camera, so apply its inverse.
    rotation, translation = extrinsic[:, :3], extrinsic[:, 3]
    return (camera_points - translation) @ rotation


def detections(cache, config):
    """One entry per mask: its index, label, video, frame and 3D points (not yet scaled)."""
    kernel = np.ones((2 * config["mask_erosion_px"] + 1,) * 2, np.uint8)
    world = [unproject(d, e, k) for d, e, k in zip(cache["depth"], cache["extrinsic"], cache["intrinsic"])]
    found = []
    for index, (mask, frame, label) in enumerate(zip(cache["mask"], cache["mask_frame"], cache["mask_label"])):
        keep = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
        keep &= cache["confidence"][frame] > config["confidence_cutoff"]
        if keep.sum() == 0:
            continue
        found.append({"mask": index, "label": str(label), "video": int(cache["video"][frame]),
                      "frame": int(cache["frame_number"][frame]), "points": world[frame][keep]})
    return found


def footprint(points):
    """Long and short side of a flat object seen from above, ignoring the 2% most extreme points."""
    centred = points - np.median(points, axis=0)
    # The two main directions of a flat object span its top surface.
    _, _, axes = np.linalg.svd(centred[::10], full_matrices=False)
    flat = (centred @ axes[:2].T).astype(np.float32)
    (_, _), (_, _), angle = cv2.minAreaRect(flat)
    turn = np.radians(angle)
    rotated = flat @ np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    sides = np.percentile(rotated, 99, axis=0) - np.percentile(rotated, 1, axis=0)
    return max(sides), min(sides)


def prompt_words(label, prompts):
    """The prompt words inside a label. Grounding DINO sometimes joins them, as in "pencil case bag"."""
    return [prompt for prompt in prompts if re.search(rf"\b{prompt}\b", label)] or [label]


def merge(found, radius, overlap, prompts):
    """Detections in one video whose 3D points overlap become one object, whatever their label.

    A detection joins an object if at least `overlap` of its points lie within `radius`
    of the object's points. The object's label is the prompt word most of its detections gave.
    """
    objects = []
    # Largest detections first, so partial views attach to an object seen whole.
    for detection in sorted(found, key=lambda d: -len(d["points"])):
        for obj in objects:
            if obj["video"] != detection["video"]:
                continue
            distances, _ = obj["tree"].query(detection["points"][::5])
            if np.mean(distances < radius) >= overlap:
                obj["points"] = np.vstack([obj["points"], detection["points"]])
                obj["tree"] = cKDTree(obj["points"])
                obj["frames"].add(detection["frame"])
                obj["masks"].append(detection["mask"])
                obj["votes"].update(prompt_words(detection["label"], prompts))
                break
        else:
            objects.append({"video": detection["video"], "frames": {detection["frame"]}, "masks": [detection["mask"]],
                            "points": detection["points"], "tree": cKDTree(detection["points"]),
                            "votes": Counter(prompt_words(detection["label"], prompts))})
    for obj in objects:
        obj["label"] = obj["votes"].most_common(1)[0][0]
        obj["centre"] = np.median(obj["points"], axis=0)
        del obj["tree"]
    return objects


def build_objects(cache, config):
    """Return the 3D objects of both videos in metres, and the scale check."""
    found = detections(cache, config)

    # Scale: the laptop's real width over its measured width, from both videos together.
    laptop = np.vstack([d["points"] for d in found if d["label"] == config["scale_object"]])
    width, depth = footprint(laptop)
    scale = config["scale_width_m"] / width
    for detection in found:
        detection["points"] = detection["points"] * scale

    objects = merge(found, config["merge_radius_m"], config["merge_overlap"], config["prompts"])
    objects = [obj for obj in objects if len(obj["frames"]) >= config["min_views"]]
    return objects, {"scale": scale, "laptop_depth_m": depth * scale}


if __name__ == "__main__":
    import plotly.graph_objects as go

    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, help="two letters, e.g. AB or BC")
    args = parser.parse_args()
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    objects, check = build_objects(np.load(f"cache/{args.pair}.npz"), config)
    print(f"{args.pair}: laptop depth {check['laptop_depth_m'] * 100:.1f} cm (real {config['scale_depth_m'] * 100:.1f})")
    figure = go.Figure()
    for obj in sorted(objects, key=lambda o: (o["video"], o["label"])):
        long_side, short_side = footprint(obj["points"])
        name = f"{args.pair[obj['video'] - 1]}: {obj['label']}"
        votes = ", ".join(f"{label} {n}" for label, n in obj["votes"].most_common())
        print(f"  {name:16s} {len(obj['frames']):2d} views  centre {np.round(obj['centre'], 2)}  "
              f"size {long_side * 100:.0f} x {short_side * 100:.0f} cm  votes: {votes}")
        shown = obj["points"][:: max(1, len(obj["points"]) // 3000)]
        figure.add_trace(go.Scatter3d(x=shown[:, 0], y=shown[:, 1], z=shown[:, 2], mode="markers",
                                      marker=dict(size=1.5), name=name))
    figure.update_layout(scene=dict(aspectmode="data"))
    figure.write_html(f"cache/objects_{args.pair}.html")
