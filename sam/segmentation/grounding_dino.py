import os
import sys
import numpy as np
from PIL import Image
import supervision as sv
import sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(CUR_DIR, "..", ".."))
sys.path.insert(0, CUR_DIR)

grounding_dino_root = os.path.join(PROJECT_ROOT, "GroundingDINO")
if grounding_dino_root not in sys.path:
    sys.path.append(os.path.abspath(grounding_dino_root))
import cv2
import torch
import torchvision
from torchvision.ops import box_convert
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import GroundingDINO.groundingdino.datasets.transforms as T

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# GroundingDINO config and checkpoint
# GROUNDING_DINO_CONFIG_PATH = "segmentation/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# GROUNDING_DINO_CHECKPOINT_PATH = "checkpoints/groundingdino_swint_ogc.pth"
GROUNDING_DINO_CONFIG_PATH = os.path.join(
    grounding_dino_root,
    "groundingdino",
    "config",
    "GroundingDINO_SwinB_cfg.py",
)
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(
    grounding_dino_root,
    "weights",
    "groundingdino_swinb_cogcoor.pth",
)


def get_model():
    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                 model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    return grounding_dino_model


def get_detections(image, object_list, grounding_dino_model, output_folder="output",
                   box_threshold=0.25, text_threshold=0.25, nms_threshold=0.8, single=False):

    image = np.array(image)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=object_list,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    if len(detections) == 0:
        print("No objects detected. Try lowering the box_threshold and text_threshold.")
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=object_list,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

    # NMS post process
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    print(f"After NMS: {len(detections.class_id)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy),
        torch.from_numpy(detections.confidence),
        nms_threshold
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    print(f"After NMS: {len(detections.xyxy)} boxes")
    print(f"After NMS: {len(detections.class_id)} boxes")

    # annotate image with detections
    box_annotator = sv.BoxAnnotator(
        thickness=1
    )
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)

    # save the annotated grounding dino image
    annotated_frame = Image.fromarray(annotated_frame)
    if output_folder.endswith(".jpg"):
        annotated_frame.save(output_folder)
    else:
        annotated_frame.save(os.path.join(output_folder, "groundingdino_annotated_image.jpg"))

    if single:
        # Filter detections to only include highest confidence detections per object
        print(f"Before Selection: {len(detections.xyxy)} boxes")

        # Select highest confidence detection for each object in object_list
        highest_confidence_detections = {}
        for i, class_id in enumerate(detections.class_id):
            confidence = detections.confidence[i]
            if class_id not in highest_confidence_detections or confidence > highest_confidence_detections[class_id][1]:
                highest_confidence_detections[class_id] = (i, confidence)

        # Filter detections to only include highest confidence detections per object
        selected_indices = [idx for idx, _ in highest_confidence_detections.values()]
        detections.xyxy = detections.xyxy[selected_indices]
        detections.confidence = detections.confidence[selected_indices]
        detections.class_id = detections.class_id[selected_indices]

        print(f"After Selection: {len(detections.xyxy)} boxes")

    return detections