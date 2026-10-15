"""Find objects and their masks on the frames VGGT saw (Colab, GPU).

Grounding DINO finds boxes from a text prompt. SAM turns each box into a mask.
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor


def detect(images, prompts, box_threshold, text_threshold):
    """Take RGB images (frames, H, W, 3), return every mask with its frame, label and score."""
    dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    dino = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to("cuda").eval()
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam = SamModel.from_pretrained("facebook/sam-vit-base").to("cuda").eval()

    # Grounding DINO wants lower case phrases, each ending in a full stop.
    text = " ".join(f"{prompt}." for prompt in prompts)

    masks, frames, labels, scores = [], [], [], []
    for i, image in enumerate(images):
        picture = Image.fromarray(image)
        inputs = dino_processor(images=picture, text=text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = dino(**inputs)
        found = dino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[image.shape[:2]])[0]
        if len(found["boxes"]) == 0:
            continue

        sam_inputs = sam_processor(picture, input_boxes=[found["boxes"].tolist()], return_tensors="pt").to("cuda")
        with torch.no_grad():
            sam_outputs = sam(**sam_inputs, multimask_output=False)
        frame_masks = sam_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu())[0]  # (boxes, 1, H, W)

        masks.append(frame_masks[:, 0].numpy().astype(bool))
        frames += [i] * len(found["boxes"])
        labels += found["text_labels"]
        scores += found["scores"].tolist()

    return {
        "mask": np.concatenate(masks),           # (masks, H, W)
        "mask_frame": np.array(frames),          # which frame each mask is on
        "mask_label": np.array(labels),          # the prompt word Grounding DINO matched
        "mask_score": np.array(scores, dtype=np.float32),
    }
