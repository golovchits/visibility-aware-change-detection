"""The visibility test on a toy scene: one box, cameras, and a wall in between.

The box sits 5 m in front of a camera at the origin that looks along +z.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from visibility import EMPTY, NOT_VISIBLE, OTHER, PRESENT, camera_answers, summarise, voxel_points  # noqa: E402

CONFIG = {"confidence_cutoff": 1.5, "depth_tolerance": 0.05}
SIZE = 100
INTRINSIC = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
LOOK_ALONG_Z = np.hstack([np.eye(3), np.zeros((3, 1))])
BOX = np.random.default_rng(0).uniform([-0.2, -0.2, 4.9], [0.2, 0.2, 5.1], (500, 3))


def answers(depth=5.0, confidence=3.0, owner=-1, same=(True,), extrinsic=LOOK_ALONG_Z):
    """One camera's answers for the box, with flat depth, confidence and mask owner."""
    return camera_answers(BOX, extrinsic, INTRINSIC, np.full((SIZE, SIZE), depth),
                          np.full((SIZE, SIZE), confidence), np.full((SIZE, SIZE), owner),
                          np.array(same), CONFIG)


def test_spot_seen_and_no_mask_is_empty():
    assert (answers(owner=-1) == EMPTY).all()


def test_mask_of_same_size_object_is_present():
    assert (answers(owner=0, same=(True,)) == PRESENT).all()


def test_mask_of_different_size_object_is_something_else():
    assert (answers(owner=0, same=(False,)) == OTHER).all()


def test_wall_in_front_occludes():
    assert (answers(depth=2.0) == NOT_VISIBLE).all()


def test_camera_facing_away_sees_nothing():
    facing_away = np.hstack([np.diag([1.0, -1.0, -1.0]), np.zeros((3, 1))])
    assert (answers(extrinsic=facing_away) == NOT_VISIBLE).all()


def test_box_outside_the_image_is_not_visible():
    shifted = np.hstack([np.eye(3), np.array([[10.0], [0], [0]])])  # box lands far to the right
    assert (answers(extrinsic=shifted) == NOT_VISIBLE).all()


def test_low_confidence_depth_that_agrees_counts_as_seen():
    assert (answers(confidence=1.0, depth=5.0) == EMPTY).all()


def test_low_confidence_depth_that_disagrees_gives_no_evidence():
    assert (answers(confidence=1.0, depth=6.0) == NOT_VISIBLE).all()  # 20% too far
    assert (answers(confidence=1.0, depth=2.0) == NOT_VISIBLE).all()  # wall in front


def test_tie_counts_as_present():
    coverage, present = summarise(np.array([[PRESENT, EMPTY]]).T)
    assert coverage == 1.0 and present == 1.0


def test_repeated_cameras_do_not_raise_coverage():
    one_camera = np.array([[EMPTY, NOT_VISIBLE]])
    twenty_cameras = np.repeat(one_camera, 20, axis=0)
    assert summarise(one_camera) == summarise(twenty_cameras) == (0.5, 0.0)


def test_voxels_keep_one_point_per_cube():
    points = np.array([[0.001, 0.001, 0.001], [0.002, 0.002, 0.002], [0.5, 0.5, 0.5]])
    assert len(voxel_points(points, 0.01)) == 2
