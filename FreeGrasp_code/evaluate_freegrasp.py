import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.new_utils import load_image_as_base64, process_grasping_result
from utils.new_config import client

# GroundingDINO
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAM_ROOT = PROJECT_ROOT / "sam"
if str(SAM_ROOT) not in sys.path:
    sys.path.append(str(SAM_ROOT))
try:
    from segmentation.grounding_dino import get_model as get_dino_model
except ImportError:
    print("Warning: GroundingDINO not available; DINO-based IoU will be skipped.")
    get_dino_model = None


def load_dataset_index(index_path: str) -> List[Dict[str, Any]]:
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_dataset_index_paths(dataset_index: List[Dict[str, Any]], dataset_root: str) -> List[Dict[str, Any]]:
    dataset_root_path = Path(dataset_root).expanduser().resolve()

    def resolve_path(path_str: Optional[str]) -> Optional[str]:
        if not path_str:
            return path_str
        p = Path(path_str)
        if p.is_absolute():
            return str(p)
        return str(dataset_root_path / p)

    normalized: List[Dict[str, Any]] = []
    for entry in dataset_index:
        fixed = dict(entry)
        fixed["scene_dir"] = resolve_path(entry.get("scene_dir"))
        fixed["samples_path"] = resolve_path(entry.get("samples_path"))
        fixed["image_path"] = resolve_path(entry.get("image_path"))
        normalized.append(fixed)
    return normalized


def load_samples(samples_path: str) -> List[Dict[str, Any]]:
    with open(samples_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_coordinates(coords_path: str) -> List[Dict[str, Any]]:
    with open(coords_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_molmo_id_file(id_txt_path: str) -> Dict[int, Tuple[float, float]]:
    """Parse annotated/final_rgb_id.txt → {molmo_id: (x, y)}.

    Expected format (from annotate_pngs.py):
        Prompt: ...
        Generated Text: ...

        Molmo_ID X Y
        1 134 151
        2 273 225
        ...
    """

    if not os.path.exists(id_txt_path):
        raise FileNotFoundError(f"Missing id file: {id_txt_path}")

    id_to_xy: Dict[int, Tuple[float, float]] = {}
    with open(id_txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Find header line starting with Molmo_ID
    header_idx = -1
    for idx, line in enumerate(lines):
        if line.lower().startswith("molmo_id"):
            header_idx = idx
            break

    if header_idx == -1:
        raise ValueError(f"Molmo_ID header not found in {id_txt_path}")

    for line in lines[header_idx + 1 :]:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            molmo_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            id_to_xy[molmo_id] = (x, y)
        except ValueError:
            continue

    return id_to_xy


def point_in_bbox(pt: Tuple[float, float], bbox: Tuple[float, float, float, float]) -> bool:
    x, y = pt
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def find_gt_bbox_by_center(gt_center: List[float], coords: List[Dict[str, Any]]) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[int]]:
    """Find the bbox whose center is nearest to gt_center.

    Returns (bbox, obj_id) where bbox is (x1, y1, x2, y2) or (None, None) if not found.
    """

    if not gt_center or not coords:
        return None, None

    best = None
    best_dist = float("inf")
    best_id = None
    for item in coords:
        cx, cy = item.get("center", [None, None])
        if cx is None or cy is None:
            continue
        dist = math.hypot(cx - gt_center[0], cy - gt_center[1])
        if dist < best_dist:
            best_dist = dist
            bbox = item.get("bbox")
            if bbox and len(bbox) == 4:
                best = tuple(bbox)
                best_id = item.get("id")
    return best, best_id


def draw_and_save(image_path: str, gt_center: Optional[List[float]], pred_xy: Optional[Tuple[float, float]], gt_bbox: Optional[Tuple[float, float, float, float]], out_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return

    if gt_bbox:
        x1, y1, x2, y2 = map(int, gt_bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "gt bbox", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    if gt_center:
        gx, gy = int(gt_center[0]), int(gt_center[1])
        cv2.circle(img, (gx, gy), 5, (255, 255, 0), -1)
        cv2.putText(img, "gt center", (gx + 6, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    if pred_xy:
        px, py = int(pred_xy[0]), int(pred_xy[1])
        cv2.circle(img, (px, py), 5, (0, 0, 255), -1)
        cv2.putText(img, "pred", (px + 6, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)


def call_llm_pick_id(image_path: str, instruction: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Call GPT to pick an object id on the annotated image and return token usage."""

    base64_image = load_image_as_base64(image_path)
    input_text = f"{instruction}"

    messages = [
        {
            "role": "system",
                "content": (
               "You are a robotic system for bin picking, using a parallel gripper. I labeled all objects id in the image."

                "You have two possible actions:"

                "1. remove obstacle, object_id: This action moves the specified object out of the way so it does not interfere with grasping the desired target object. This action can only be performed if the specified object is free of obstacles (not occluded by any other object)."
                "2. pick object, object_id: This action picks up the specified object. It can only be performed if the object is free of obstacles."
                "An object is considered an obstacle if it occludes another object."

                "Task:"
                "Given a target object description as input, determine the first object that needs to be grasped to enable picking the target object. If the target object is free of obstacles, return the target object ID itself. Otherwise, identify an object that is occluding the target and is itself free of obstacles. If multiple objects could be removed, return any one valid option."

                "Output Format:"
                    "The output should only be the object ID of the first object to grasp, must formatted as: [object_id, color class_name]\n"
                )
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": input_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            ],
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        # model = "qwen3-vl-plus-2025-12-19",
        messages=messages,
        temperature=0,
        max_tokens=713,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        seed=0,
    )

    usage_raw = getattr(response, "usage", None)
    usage: Dict[str, int] = {}
    if usage_raw:
        usage = {
            "prompt_tokens": getattr(usage_raw, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage_raw, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage_raw, "total_tokens", 0) or 0,
        }

    output = response.choices[0].message.content
    return process_grasping_result(output, instruction), usage


def evaluate_scene(
    scene_entry: Dict[str, Any],
    dist_thresh: float,
    use_bbox: bool,
    vis_dir: str,
    token_acc: Dict[str, int],
) -> List[Dict[str, Any]]:
    scene_dir = scene_entry["scene_dir"]
    annotated_dir = os.path.join(scene_dir, "annotated")
    annotated_img = os.path.join(annotated_dir, "final_rgb_annotated.png")
    id_txt = os.path.join(annotated_dir, "final_rgb_id.txt")
    samples_path = scene_entry["samples_path"]
    coords_path = os.path.join(scene_dir, "coordinates.json")

    id_to_xy = parse_molmo_id_file(id_txt)
    samples = load_samples(samples_path)
    coords = load_coordinates(coords_path)

    scene_results = []

    image_raw_path = os.path.join(scene_dir, "final_rgb.png")

    for sample_idx, sample in enumerate(samples):
        gt_center = sample.get("ground_truth", {}).get("center")
        instruction = sample.get("instruction", "")
        sample_type = sample.get("type", 0)

        pred_info, usage = call_llm_pick_id(annotated_img, instruction)
        print(usage)
        pred_id = pred_info.get("selected_object_id")

        token_acc["prompt_tokens"] += usage.get("prompt_tokens", 0)
        token_acc["completion_tokens"] += usage.get("completion_tokens", 0)
        token_acc["total_tokens"] += usage.get("total_tokens", 0)

        pred_xy = id_to_xy.get(pred_id) if pred_id is not None else None

        dist = math.inf
        success = False
        gt_bbox = None

        if gt_center and coords:
            gt_bbox, _ = find_gt_bbox_by_center(gt_center, coords)

        if pred_xy and gt_center:
            dist = math.hypot(pred_xy[0] - gt_center[0], pred_xy[1] - gt_center[1])
            print(dist)

        if pred_xy:
            if gt_bbox:
                success = point_in_bbox(pred_xy, gt_bbox)
            elif gt_center:
                success = dist <= dist_thresh

        scene_results.append(
            {
                "scene_id": scene_entry.get("scene_id"),
                "sample_type": sample_type,
                "instruction": instruction,
                "pred_id": pred_id,
                "pred_xy": pred_xy,
                "gt_center": gt_center,
                "gt_bbox": gt_bbox,
                "distance": dist,
                "use_bbox": use_bbox,
                "success": success,
                "usage": usage,
            }
        )

        vis_path = os.path.join(vis_dir, f"scene{scene_entry.get('scene_id')}_sample{sample_idx}.jpg")
        draw_and_save(image_raw_path, gt_center, pred_xy, gt_bbox, vis_path)

    return scene_results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    success = sum(1 for r in results if r.get("success"))
    overall_sr = success / total if total else 0.0

    per_type: Dict[Any, List[bool]] = {}
    for r in results:
        t = r.get("sample_type")
        per_type.setdefault(t, []).append(r.get("success", False))

    type_stats = {
        str(t): {
            "count": len(flags),
            "success_rate": round((sum(1 for f in flags if f) / len(flags)), 6) if flags else 0.0,
        }
        for t, flags in per_type.items()
    }

    return {"total_samples": total, "overall_sr": round(overall_sr, 6), "type_breakdown": type_stats}


def main():
    parser = argparse.ArgumentParser(description="Center-distance evaluation on annotated Molmo IDs.")
    parser.add_argument(
        "benchmark_folder",
        type=str,
        help="Benchmark folder name under data/, e.g. benchmark-200",
    )
    parser.add_argument("--dist_thresh", type=float, default=25.0, help="Pixel distance threshold for success.")
    parser.add_argument("--output_prefix", type=str, default="eval_results_freeGrasp", help="Prefix for output JSON files.")
    parser.add_argument("--use_bbox",type=float, default=True, help="Use GT bbox containment instead of distance threshold.")
    args = parser.parse_args()

    data_root = PROJECT_ROOT / "data"
    benchmark_path = data_root / args.benchmark_folder
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark folder not found: {benchmark_path}")

    dataset_root = str(benchmark_path.resolve())
    index_path = os.path.join(dataset_root, "dataset_index.json")
    dataset_index = load_dataset_index(index_path)
    dataset_index = normalize_dataset_index_paths(dataset_index, dataset_root)

    all_results: List[Dict[str, Any]] = []
    token_acc = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for scene_entry in tqdm(dataset_index, desc="Scenes"):
        try:
            vis_dir = os.path.join(scene_entry["scene_dir"], f"{args.output_prefix}_vis")
            scene_results = evaluate_scene(scene_entry, args.dist_thresh, args.use_bbox, vis_dir, token_acc)
            all_results.extend(scene_results)
        except Exception as exc:
            print(f"Scene {scene_entry.get('scene_id')} failed: {exc}")
            continue

    metrics = summarize(all_results)
    metrics["token_usage"] = token_acc

    out_dir = os.path.join(dataset_root, "benchmark_runs", "FreeGrasp")
    os.makedirs(out_dir, exist_ok=True)
    details_path = os.path.join(out_dir, f"{args.output_prefix}_details.json")
    metrics_path = os.path.join(out_dir, f"{args.output_prefix}_metrics.json")

    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Saved details to: {details_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Overall SR: {metrics['overall_sr']:.2%}")


if __name__ == "__main__":
    main()