"""Run the GPU models for one pair and save cache/<pair>.npz (Colab).

Usage: python src/gpu.py --pair AB
AB means A is video 1 and B is video 2.
"""
import argparse
import os

import numpy as np
import yaml

from frames import pick_frames, read_frames
from reconstruct import reconstruct

parser = argparse.ArgumentParser()
parser.add_argument("--pair", required=True, help="two letters, e.g. AB or BC")
args = parser.parse_args()

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

frames, video_tags, frame_numbers = [], [], []
for tag, letter in [(1, args.pair[0]), (2, args.pair[1])]:
    video_path = f"data/{config['videos'][letter]}"
    indices, _ = pick_frames(video_path, config["frames_per_video"])
    frames += read_frames(video_path, indices)
    video_tags += [tag] * len(indices)
    frame_numbers += indices

cache = reconstruct(frames)
cache["video"] = np.array(video_tags)          # 1 or 2, per frame
cache["frame_number"] = np.array(frame_numbers)  # position in the original video

os.makedirs("cache", exist_ok=True)
out_path = f"cache/{args.pair}.npz"
np.savez_compressed(out_path, **cache)
print(f"{args.pair}: {len(frames)} frames, image size {cache['image'].shape[1:3]} -> {out_path}")
