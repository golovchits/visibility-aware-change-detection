"""Pick sharp frames spread across a video.

The video is split into equal time slices and the sharpest frame of each
slice is kept. Same video in, same frames out, so the Mac and Colab agree.
"""
import os

import cv2
import numpy as np
import yaml


def sharpness(frame):
    # Variance of the Laplacian: crisp edges give a large spread, blur a small one.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def pick_frames(video_path, n_frames):
    """Return the indices and sharpness scores of the chosen frames."""
    video = cv2.VideoCapture(video_path)
    scores = []
    while True:
        ok, frame = video.read()
        if not ok:
            break
        scores.append(sharpness(frame))
    scores = np.array(scores)

    indices = []
    for time_slice in np.array_split(np.arange(len(scores)), n_frames):
        indices.append(int(time_slice[np.argmax(scores[time_slice])]))
    return indices, scores[indices]


def read_frames(video_path, indices):
    """Return the frames at these indices, as BGR images."""
    # Read straight through rather than seeking. Seeking in H.264 can land a frame off.
    wanted = set(indices)
    video = cv2.VideoCapture(video_path)
    frames = {}
    index = 0
    while len(frames) < len(wanted):
        ok, frame = video.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = frame
        index += 1
    return [frames[i] for i in indices]


def contact_sheet(frames, indices, scores, columns=5, width=384):
    """Tile the frames into one image, each labelled with its index and sharpness."""
    tiles = []
    for frame, index, score in zip(frames, indices, scores):
        tile = cv2.resize(frame, (width, width * frame.shape[0] // frame.shape[1]))
        label = f"frame {index}  sharp {score:.0f}"
        cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(tile)
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    return np.vstack(rows)


if __name__ == "__main__":
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    os.makedirs("cache", exist_ok=True)

    for name in ["before", "after", "control"]:
        video_path = f"data/{name}.MOV"
        indices, scores = pick_frames(video_path, config["frames_per_video"])
        frames = read_frames(video_path, indices)
        out_path = f"cache/contact_{name}.jpg"
        cv2.imwrite(out_path, contact_sheet(frames, indices, scores))
        print(f"{name}: frames {indices}, sharpness {scores.min():.0f} to {scores.max():.0f} -> {out_path}")
