# Architecture

How the parts fit together. Every part exists because the challenge asks for it. Items marked **Open** are decided during the build, at the step named.

## 01 System overview
```
 video 1 + video 2 (phone, .MOV)
        |
        v
 frames.py         pick ~15 sharp frames per video    [Mac + Colab]
        |
 ------ Colab GPU -----------------------------------------
 reconstruct.py    VGGT, one joint run on both videos'
                   frames: cameras, depth, confidence
 detect.py         Grounding DINO boxes + SAM masks on
                   the same frames VGGT saw
        |  saves cache/<pair>.npz
 ----------------------------------------------------------
        v
 objects.py        masks + depth -> 3D objects, one set     [Mac]
                   per video, in metres
 visibility.py     each object vs the other video's
                   cameras: visible? something there?
 classify.py       five states + naive baseline
 describe.py       one sentence per change
 viewer.py         Plotly HTML
 evaluate.py       compare with data/ground_truth.yaml
        |
        v
 results/<pair>/   results.json, viewer.html, changes.md
```
A pair is written first video, second video. AB means A is video 1 and B is video 2. BC means B is video 1 and C is video 2.

The GPU part runs once per pair and saves a cache. Everything after it runs on the Mac in seconds, so tuning and fixing never wait on Colab.

## 02 Why each part exists
| Need (from the brief) | Part | Why this one |
|---|---|---|
| A shared 3D representation of both recordings | VGGT, one joint run | One forward pass gives cameras and depth for every frame, already in one coordinate frame. No separate alignment step to write or debug |
| Know which pixels are which object | Grounding DINO + SAM | Grounding DINO finds objects from text prompts, SAM turns each box into a mask. Both are off the shelf and run in `transformers` |
| Tell real change from viewpoint, occlusion or missing views | Visibility test | The core of the project. See section 07 |
| Moved, added or removed, in 3D | classify.py + viewer | Rules, not a model. Each rule can be explained in one line |
| Describe changes in words | Templates | Every word traces back to a number in results.json |
| Show it works | Ground truth, control pair, naive baseline | A clean demo proves little. A measured comparison proves more |

Rejected (reasons in the design note): combining evidence across frames with Dempster's rule, an evidential network, video tracking with SAM 2, COLMAP.

## 03 Tech stack
| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Frames | OpenCV (reads the iPhone .MOV, measures sharpness) |
| 3D | VGGT-1B (`facebookresearch/vggt`), fp16 on a Colab T4 |
| Objects | Grounding DINO (`IDEA-Research/grounding-dino-base`) and SAM (`facebook/sam-vit-base`), both through Hugging Face `transformers` |
| Maths | numpy, scipy |
| Viewer | Plotly, one self-contained HTML file. A turning GIF of it through kaleido (Plotly's image export, uses Chrome) and imageio |
| Config | one YAML file |
| Tests | pytest, for the visibility test and the classification rules |
| GPU | Google Colab, free T4 |

## 04 Project structure
```
visibility-aware-change-detection/
├── run.py                  # local stages: --pair AB --out results/
├── requirements.txt        # Mac side packages
├── configs/default.yaml    # thresholds, frame count, prompt list
├── src/
│   ├── frames.py           # video -> sharp frames
│   ├── reconstruct.py      # VGGT joint run (Colab)
│   ├── detect.py           # boxes + masks (Colab)
│   ├── gpu.py              # runs the two above, writes cache/<pair>.npz
│   ├── objects.py          # masks + depth -> 3D objects
│   ├── visibility.py       # the core test
│   ├── classify.py         # five states + naive baseline
│   ├── describe.py         # sentences
│   ├── viewer.py           # Plotly HTML
│   └── evaluate.py         # vs ground truth
├── notebooks/gpu.ipynb     # Colab: clone repo, run src/gpu.py, download cache
├── tests/                  # test_visibility.py, test_classify.py
├── data/                   # the three videos + ground_truth.yaml (never edited)
├── cache/                  # GPU outputs (git-ignored)
├── results/                # what the reviewer sees
├── docs/
├── CLAUDE.md
└── README.md
```

## 05 Data flow
1. `frames.py` reads each video, scores every frame's sharpness, and keeps about 15 sharp frames spread across the whole video.
2. `reconstruct.py` runs VGGT once on both videos' frames together. Each frame gets a camera pose, intrinsics, a depth map and a depth confidence map. All of it is saved. Low-confidence depth is dropped later, on the Mac, so the cut-off can be tuned without rerunning Colab.
3. `detect.py` runs Grounding DINO with a fixed prompt list, then SAM on each box, on the exact images VGGT saw, so pixels line up.
4. `gpu.py` saves everything for the pair in `cache/<pair>.npz`.
5. `objects.py` drops depth below the confidence cut-off, turns each mask into 3D points using the depth and the camera, then merges detections across frames of one video. A detection joins an object when its 3D points overlap the object's points, whatever its label. The object's label is the prompt word most of its detections gave. Scale comes from the laptop's known width (30.41 cm). Its depth (21.5 cm) is the independent check.
6. `visibility.py` tests every object against the other video's cameras (section 07).
7. `classify.py` gives each object one of the five states, then pairs a removed and an added object of the same label into "moved".
8. `describe.py`, `viewer.py` and `evaluate.py` write the outputs.

## 06 Data formats
| File | Holds |
|---|---|
| `cache/<pair>.npz` | per frame: image (RGB, as VGGT saw it), video tag (1 or 2), frame number in the original video, extrinsic (camera from world), intrinsic, depth, depth confidence, masks with labels and scores |
| `results/<pair>/results.json` | per object: id, label, video, state, centroid in metres (both videos where seen), coverage, present fraction, views that checked it |
| `results/<pair>/changes.md` | the sentences |
| `results/<pair>/viewer.html` | the 3D view |
| `results/<pair>/viewer.gif` | the same view turning once, with `run.py --gif` |

## 07 The visibility test
For an object from video 1, take each of its 3D points and project it into every camera of video 2.

**Step 1. Could that camera see the point?**
| Case | Result |
|---|---|
| The point lands outside the image, or behind the camera | Not visible |
| The camera's depth there is clearly shorter than the point's distance (something is in front) | Not visible, occluded |
| The depth there has low confidence and is more than the tolerance off the point's distance, either way | Not visible, no evidence |
| Otherwise | Visible |

Why low-confidence depth can still count. VGGT gives plain white table its lowest confidence. In A vs B the notebook's old spot is bare table, in frame in 10 of B's frames, and B's depth there matches the notebook spot within 2%. Throwing that away would turn a filmed removal into "not checked".

Points are first thinned to one per 1 cm cube, so a patch filmed in 15 frames counts once.

"Clearly shorter" uses a tolerance that grows with distance, a few percent of the depth. **Open, tune at step 6.**

**Step 2. If visible, what is there now?**
Position decides whether the spot is occupied. Look at video 2's masks at that pixel. If no mask covers it, the spot is empty. If one does, size tells us whose object it is. Labels are not used here.
| At that pixel | Result |
|---|---|
| No mask, background | Empty |
| A mask of a video 2 object with about the same footprint (long and short side within `size_tolerance`) | Present |
| A mask of a video 2 object with a different footprint | Something else is there now |

Why something beyond position is needed. In A vs B the flashcards cover about half of the pencil case's old spot. By position alone the pencil case would be about 50% present, and its state would hinge on the cut-off.

Why size and not the label. Grounding DINO names one object differently from video to video. The flashcards are "box" in B and "notebook" in C. Their size is 10 by 7 cm in both. The pencil case is 15 by 13 cm, so size still keeps it apart from the flashcards.

Why masks and not depth for this step. A notebook lies about 1 cm above the table. Depth from a phone at 1 m is not that precise, so bare table and notebook look the same in depth. Depth answers "could it see the spot", masks answer "what is on the spot".

**Step 3. Sum up per object.**
- **Coverage** is the share of the object's points that at least one camera of video 2 could see. A point seen from 20 near-identical frames counts once, so repeated frames never inflate confidence.
- **Present fraction** is the share of covered points that come out present. When the video 2 cameras that can see a point disagree, the majority wins. A tie counts as present, the safe side, because "removed" needs positive evidence.

## 08 Classification rules
| Object from | Condition | State |
|---|---|---|
| Either video | coverage below `c_min` | Not checked |
| Video 1 | covered, mostly present | Unchanged |
| Video 1 | covered, mostly empty or something else | Removed |
| Video 2 | covered in video 1, mostly empty or something else there | Added |
| Video 2 | covered in video 1, mostly present there | Same object as its video 1 match. Reported once, under the video 1 object |
| A removed + an added object | same label | Moved, with the distance in cm |

Thresholds: depth confidence cut-off, mask erosion, merge radius and overlap, minimum views, depth tolerance, `size_tolerance`, `c_min`, the "mostly" cut-off, and the baseline radius `r_naive`. All set on B vs C only, then frozen.

**Naive baseline.** The same pipeline without section 07 and without the "not checked" state. A video 1 object with no video 2 object of about the same footprint (within `size_tolerance`) and within `r_naive` is removed, and the reverse is added. Same size rule as ours, so the visibility test is the only difference. A removed and an added object of the same label become "moved", the same rule as ours. `r_naive` is tuned on B vs C the same way as our thresholds, then frozen. It cannot reach zero changes there (it has no "not checked"), so its safe range is the fewest changes. It isolates what the visibility test adds.

## 09 Limits and failure modes
| Case | Behaviour |
|---|---|
| Flat objects | Presence comes from masks, so a missed mask can look like "empty". Mask erosion and the "mostly" cut-off limit this |
| Same label twice (two books) | "Moved" pairing can match the wrong pair. Not in our scene. Later: appearance check |
| Wrong label from Grounding DINO | The sentence uses the wrong word. The state is still right, because size, not the label, decides "same object". "Moved" pairing still uses the label, so a label that differs between the videos can miss a move |
| Two objects of similar size swap spots | Read as "same object", so both look unchanged. Not in our scene. Later: appearance check |
| A new object on part of an old one's spot | Pencil case and flashcards in A vs B. Their sizes (15 by 13 cm, 10 by 7 cm) keep them apart |
| Plain white table | Few features, so registration can drift. Checked visually at step 3 |
| Objects at the frame edge | Partly seen objects get low coverage and lean towards "not checked", which is the safe side |
