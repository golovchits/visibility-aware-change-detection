"""The 3D view: the table in its real colours, each object coloured by its state.

viewer.html is one self-contained file that opens on a phone. The GIF turns the same view once around the table.
"""
import textwrap

import numpy as np
import plotly.graph_objects as go

from describe import sentence
from objects import unproject

COLOURS = {"unchanged": "#8c8c8c", "removed": "#d62728", "added": "#2ca02c", "moved": "#ff7f0e",
           "moved to": "#ff7f0e", "not checked": "#9467bd"}
TAGS = {"moved": "moved from here", "moved to": "moved here"}  # other states are shown as they are
STEP = 6  # keep every 6th pixel of each frame for the table
MARGIN_M = 0.15  # table shown up to 15 cm around the objects. Farther depth is mostly noise


def background(cache, scale, config):
    """Confident depth from every frame, thinned, in metres, with each pixel's colour."""
    points, colours = [], []
    for depth, extrinsic, intrinsic, confidence, image in zip(
            cache["depth"], cache["extrinsic"], cache["intrinsic"], cache["confidence"], cache["image"]):
        keep = confidence[::STEP, ::STEP] > config["confidence_cutoff"]
        points.append(unproject(depth, extrinsic, intrinsic)[::STEP, ::STEP][keep] * scale)
        colours.append(image[::STEP, ::STEP][keep])
    return np.vstack(points), np.vstack(colours)


def table_axes(points):
    """The table's normal and two directions along it, from the best-fit plane of the table points."""
    _, _, axes = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    along_1, along_2, normal = axes
    return normal, along_1, along_2


def figure(records, cache, scale, pair, config):
    points, colours = background(cache, scale, config)
    objects = np.vstack([r["points"] for r in records])
    near = np.all((points > objects.min(axis=0) - MARGIN_M) & (points < objects.max(axis=0) + MARGIN_M), axis=1)
    points, colours = points[near], colours[near]
    fig = go.Figure(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", name="table", hoverinfo="skip",
        marker=dict(size=1.5, color=[f"rgb({r},{g},{b})" for r, g, b in colours], opacity=0.4)))
    for r in records:
        if r["state"] == "same as video 1":
            continue  # drawn once, as its video 1 object
        shown = r["points"][:: max(1, len(r["points"]) // 2000)]
        name = f"{r['label']} ({pair[r['video'] - 1]}): {r['state']}"
        # Hovering shows the object's sentence. A "moved to" object has none, so it shows its partner's.
        partner = next((o for o in records if o.get("moved_to") == r["id"]), r)
        hover = "<br>".join(textwrap.wrap(sentence(partner, records, pair, config), 60)) + "<extra></extra>"
        fig.add_trace(go.Scatter3d(x=shown[:, 0], y=shown[:, 1], z=shown[:, 2], mode="markers", name=name,
                                   marker=dict(size=2, color=COLOURS[r["state"]]), hovertemplate=hover))
        if r["state"] == "moved":
            line = np.array([r["centre"], records[r["moved_to"]]["centre"]])
            fig.add_trace(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2], mode="lines", showlegend=False,
                                       line=dict(color=COLOURS["moved"], width=6)))

    normal, along, _ = table_axes(points)
    # Point the normal toward the cameras, so "up" is above the table, not below it.
    cameras = np.array([-e[:, :3].T @ e[:, 3] for e in cache["extrinsic"]]) * scale
    if (cameras - points.mean(axis=0)).mean(axis=0) @ normal < 0:
        normal = -normal
    # Tags 3 cm above each object with its verdict, so the view reads without the legend.
    tagged = [r for r in records if r["state"] != "same as video 1"]
    tags = np.array([r["centre"] + 0.03 * normal for r in tagged])
    fig.add_trace(go.Scatter3d(x=tags[:, 0], y=tags[:, 1], z=tags[:, 2], mode="text", showlegend=False,
                               text=[f"{r['label']}: {TAGS.get(r['state'], r['state'])}" for r in tagged],
                               textfont=dict(size=13), hoverinfo="skip"))
    hidden = dict(visible=False)
    fig.update_layout(scene=dict(aspectmode="data", xaxis=hidden, yaxis=hidden, zaxis=hidden,
                                 camera=view(normal, along, 0)),
                      margin=dict(l=0, r=0, t=30, b=0), title=f"Video {pair[0]} vs video {pair[1]}",
                      legend=dict(itemsizing="constant", orientation="h", y=0))
    return fig, normal, along


def view(normal, along, angle):
    """A camera above the table at 40 degrees, turned by angle around the table normal."""
    side = np.cross(normal, along)
    eye = 1.6 * (np.cos(angle) * along + np.sin(angle) * side) + 1.4 * normal
    return dict(up=dict(zip("xyz", normal)), eye=dict(zip("xyz", eye)))


def write_gif(fig, normal, along, path, frames=24):
    import imageio.v3 as imageio

    images = []
    for angle in np.linspace(0, 2 * np.pi, frames, endpoint=False):
        fig.update_layout(scene_camera=view(normal, along, angle))
        images.append(imageio.imread(fig.to_image(format="png", width=800, height=600)))
    fig.update_layout(scene_camera=view(normal, along, 0))
    imageio.imwrite(path, np.stack(images), duration=150, loop=0)
