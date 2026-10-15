"""Run VGGT once on all frames of a pair (Colab, GPU).

One joint run puts every camera of both videos in the same 3D frame,
so no alignment step is needed afterwards.
"""
import os
import tempfile

import cv2
import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def reconstruct(frames):
    """Take BGR frames, return per-frame arrays: image, extrinsic, intrinsic, depth, confidence."""
    # VGGT's loader reads files. PNG is lossless, so the pixels stay exactly as read.
    folder = tempfile.mkdtemp()
    paths = []
    for i, frame in enumerate(frames):
        path = os.path.join(folder, f"{i:03d}.png")
        cv2.imwrite(path, frame)
        paths.append(path)

    model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()
    images = load_and_preprocess_images(paths).to("cuda")  # (frames, 3, H, W), RGB, 0 to 1

    # fp16 because the T4 has no bf16 support.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        predictions = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    # Everything is kept, including low-confidence depth. The cut-off is applied on the Mac.
    return {
        "image": (images.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8),
        "extrinsic": extrinsic[0].float().cpu().numpy(),             # (frames, 3, 4), camera from world
        "intrinsic": intrinsic[0].float().cpu().numpy(),             # (frames, 3, 3)
        "depth": predictions["depth"][0, ..., 0].float().cpu().numpy(),  # (frames, H, W)
        "confidence": predictions["depth_conf"][0].float().cpu().numpy(),  # (frames, H, W)
    }
