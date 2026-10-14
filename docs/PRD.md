# PRD: Visibility-aware change detection

What we build, why it matters, and how we know it works.

## 01 Product overview
| | |
|---|---|
| **Name** | Visibility-aware change detection |
| **Tagline** | It only says something is gone if it looked. |
| **Description** | Two phone videos of the same small area go in. A 3D map of what moved, appeared or disappeared comes out, with one plain sentence per change. Anything the second video never looked at is marked "not checked". |

## 02 Problem
A robot that comes back to a room sees it from a new angle. Parts it can't see this time look like they are missing. A simple before-and-after comparison reports them as removed, and the robot plans around a world that isn't real.

The challenge asks for exactly this distinction: "distinguishing actual changes from differences in camera viewpoint, occlusion, or incomplete observations."

## 03 Goal
For each object, answer one of five things, and say where it is in 3D.

| State | Meaning |
|---|---|
| Unchanged | Seen in both videos, same place |
| Removed | Seen in the first video. The second video looked at that spot and found it empty |
| Added | Seen in the second video. The first video looked at that spot and found it empty |
| Moved | Removed from one spot and added at another, same object |
| Not checked | The other video never looked at that spot, so we can't tell |

## 04 Target users
- The Humanoid engineer who reviews the submission. They want to see the result in seconds and judge the thinking.
- The stand-in user: a robot's mapping system revisiting a space.

> **User need:** "Tell me what really changed. Don't tell me something is gone just because I'm standing somewhere else."

## 05 Core features (must have)
| # | Feature | What it does |
|---|---|---|
| 1 | Shared 3D scene | Both videos in one 3D frame, at real scale |
| 2 | Objects | Each object found, labelled and placed in 3D, once per video |
| 3 | Visibility test | For each object, was its spot visible in the other video, and what was there |
| 4 | Five states | Unchanged, removed, added, moved, not checked |
| 5 | 3D viewer | One HTML file. Colour per state, an arrow for each move |
| 6 | Sentences | One plain sentence per change, built from the numbers |
| 7 | Evaluation | Ground truth, a no-change control pair, and a naive baseline without the visibility test |
| 8 | Easy to run | One command locally, plus a Colab notebook for the GPU part |

## 06 Later (only if time is left)
- Appearance check for "moved" (same object, not just same label)
- A second scene
- Changes inside one object, such as a book missing from a stack

## 07 Out of scope
- Real time
- Whole rooms or houses
- People or things moving during a recording
- Training any model

## 08 Success criteria
- The system takes exactly two videos.
- A vs B matches the ground truth in `data/ground_truth.yaml`.

| Object | Expected |
|---|---|
| Notebook | Removed |
| Pencil case | Moved, from left of the laptop to right of it |
| Flashcards | Added |
| Water bottle | Not checked (never in view in video B) |
| Laptop | Unchanged |

- B vs C reports zero changes. The bottle comes out "not checked", because B never saw it.
- The naive baseline gets the bottle wrong (calls it removed). Ours does not.
- A reviewer sees the result without installing anything (HTML viewer and a GIF in the README).
- A fresh clone runs from the README.
