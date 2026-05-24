import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

AMBRES_ROOT = Path(__file__).resolve().parents[1]
if str(AMBRES_ROOT) not in sys.path:
    sys.path.insert(0, str(AMBRES_ROOT))

from ambres.ambres_model import AmbresFSPrompt, AmbresFineTuned
from ambres import CKPT
from ambres.molmo_chat import extract_coordinates


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def point_in_bbox(pt: Tuple[float, float], bbox: Tuple[float, float, float, float]) -> bool:
    x, y = pt
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def find_gt_bbox_by_center(gt_center: List[float], coords: List[Dict[str, Any]]) -> Tuple[Tuple[float, float, float, float] | None, int | None]:
    """Find the bbox whose center is nearest to gt_center."""

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


def load_benchmark(dataset_root: str) -> List[Dict[str, Any]]:
    """Load benchmark dataset into a flat list of samples.

    Each entry carries scene metadata, instruction, target category, and the RGB path.
    """

    dataset_root_path = Path(dataset_root).expanduser().resolve()

    def resolve_dataset_path(path_str: str | None) -> str | None:
        if not path_str:
            return None
        p = Path(path_str)
        if p.is_absolute():
            return str(p)
        return str(dataset_root_path / p)

    index_path = dataset_root_path / "dataset_index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    samples: List[Dict[str, Any]] = []
    for entry in index:
        scene_id = entry.get("scene_id")
        image_path = resolve_dataset_path(entry.get("image_path"))
        samples_path = resolve_dataset_path(entry.get("samples_path"))
        if not (image_path and samples_path):
            continue
        with open(samples_path, "r", encoding="utf-8") as f:
            scene_samples = json.load(f)
        for i, s in enumerate(scene_samples):
            gt = s.get("ground_truth", {})
            target_cat = gt.get("target_category")
            gt_center = gt.get("center")
            if isinstance(gt_center, (list, tuple)) and len(gt_center) == 2:
                gt_center = (float(gt_center[0]), float(gt_center[1]))
            else:
                gt_center = None
            gt_bbox = gt.get("bbox")
            if isinstance(gt_bbox, (list, tuple)) and len(gt_bbox) == 4:
                gt_bbox = tuple(float(x) for x in gt_bbox)
            else:
                gt_bbox = None
            samples.append(
                {
                    "scene_id": scene_id,
                    "sample_idx": i,
                    "instruction": s.get("instruction", ""),
                    "target_category": target_cat,
                    "target_description": gt.get("target_description"),
                    "gt_center": gt_center,
                    "gt_bbox": gt_bbox,
                    "image_path": image_path,
                    "sample_type": s.get("type", 0),
                }
            )
    return samples


def normalize_category(cat: str) -> str:
    return cat.strip().lower() if isinstance(cat, str) else ""


def pick_pred_index(pred_names: List[str], target_cat: str) -> int | None:
    """Pick prediction index with relaxed matching.

    Preference order:
    1) exact match after normalization
    2) substring containment either direction
    3) fallback to the first prediction if nothing matches
    """

    if not pred_names:
        return None

    norm_preds = [normalize_category(x) for x in pred_names]
    # exact match
    for idx, name in enumerate(norm_preds):
        if target_cat and name == target_cat:
            return idx
    # substring match (either direction)
    for idx, name in enumerate(norm_preds):
        if target_cat and (target_cat in name or name in target_cat):
            return idx
    # fallback: first prediction
    return 0


def generate_clarifying_response(target_description: str) -> str:
    """Lightweight clarifying answer using the GT description.

    This mirrors the idea of using an LLM to respond with the intended target.
    """

    if not target_description:
        return "No more information."
    return target_description


def extract_pred_coord(
    obj_detection_messages: List[Dict[str, str]],
    obj_index: int,
    img_w: int,
    img_h: int,
    expected_objects: int | None = None,
) -> Tuple[float, float] | None:
    """Extract predicted (x, y) from detection messages for the selected object index.

    The original assumption that detection messages are stored strictly as pairs
    ([user, assistant] per object) breaks because `obj_detection_messages` now
    carries the *entire* chat history (prompts, ambiguity questions, clarifying
    turns, etc.). We instead collect assistant messages that actually contain
    point markup and, if possible, take the last block that corresponds to the
    current object list.
    """

    if not obj_detection_messages or img_w == 0 or img_h == 0:
        return None

    # Collect all assistant messages that include point markup.
    point_msgs: List[str] = []
    for msg in obj_detection_messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if "<point" in content or "<points" in content:
            point_msgs.append(content)

    # If we expect N objects (from the latest grounding), use the last N point
    # messages to align with the most recent detection pass.
    if expected_objects is not None and expected_objects > 0 and len(point_msgs) >= expected_objects:
        point_msgs = point_msgs[-expected_objects:]

    if obj_index < 0 or obj_index >= len(point_msgs):
        return None

    coords = extract_coordinates(point_msgs[obj_index], img_h, img_w)
    if coords and coords[0]:
        return float(coords[0][0]), float(coords[0][1])
    return None


def save_pred_visualization(
    image_path: str,
    pred_xy: Tuple[float, float] | None,
    gt_center: Tuple[float, float] | None,
    gt_bbox: Tuple[float, float, float, float] | None,
    out_path: str,
):
    """Draw prediction (red), GT center (green), and GT bbox (green box) on the image."""

    if not pred_xy and not gt_center and not gt_bbox:
        return

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return

    draw = ImageDraw.Draw(image)
    r_pred = 6
    r_gt = 6

    if gt_bbox:
        x1, y1, x2, y2 = gt_bbox
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=2)
        draw.text((x1, max(0, y1 - 12)), "gt bbox", fill="lime")

    if pred_xy:
        x, y = pred_xy
        draw.ellipse((x - r_pred, y - r_pred, x + r_pred, y + r_pred), fill="red", outline="red", width=2)
        draw.text((x + 8, y), "pred", fill="red")

    if gt_center:
        gx, gy = gt_center
        draw.ellipse((gx - r_gt, gy - r_gt, gx + r_gt, gy + r_gt), fill="cyan", outline="cyan", width=2)
        draw.text((gx + 8, gy), "gt center", fill="cyan")

    if pred_xy and gt_center:
        draw.line((pred_xy[0], pred_xy[1], gt_center[0], gt_center[1]), fill="yellow", width=2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    image.save(out_path)


def run_once(args, run_id: int) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    coords_cache = {}
    if args.model_type == "prompt":
        model = AmbresFSPrompt(use_detection=True)
    else:
        ckpt_name = CKPT.get_ckpt_name(args.env)
        model = AmbresFineTuned(adapter_ckpt=ckpt_name, use_detection=True)

    dataset = load_benchmark(args.dataset)
    print(f"[Run {run_id}] Loaded {len(dataset)} samples from {args.dataset}")

    details = []
    correct = 0
    type_stats: Dict[Any, List[bool]] = {}

    for sample in tqdm(dataset, desc=f"Evaluating run {run_id}"):
        model.reset_chat()
        inp = {
            "task_description": sample["instruction"],
            "image_path": sample["image_path"],
        }
        print("输入",inp)
        # First turn
        output_query = model.handle_query_dict(inp)
        task_objects = output_query.get("task_objects", [])
        print("预测物体",task_objects)
        task_ambiguous = output_query.get("task_ambiguous", False)
        obj_detection_messages = output_query.get("obj_detection_messages", [])
        img_w, img_h = model.images[0].size if model.images else (0, 0)
        with Image.open(sample["image_path"]) as img_orig:
            orig_w, orig_h = img_orig.size

        final_task_objects = task_objects
        final_detection_messages = obj_detection_messages

        if bool(task_ambiguous):
            # Use GT description to craft clarifying response via lightweight generator
            # (mirrors LLM-based clarification in full_sft)
            target_desc = sample.get("target_description") or sample.get("target_category") or ""
            # response_text = generate_clarifying_response(target_desc)
            response_text = f"I want {target_desc}"
            output_reply = model.handle_response(response_text)
            final_task_objects = output_reply.get("task_objects", task_objects)
            final_detection_messages = output_reply.get("obj_detection_messages", obj_detection_messages)

        target_cat = normalize_category(sample.get("target_category"))
        pred_idx = pick_pred_index(final_task_objects, target_cat)

        pred_xy_down = None
        pred_xy_full = None
        if pred_idx is not None:
            pred_xy_down = extract_pred_coord(
                final_detection_messages, pred_idx, img_w, img_h, expected_objects=len(final_task_objects)
            )

        if pred_xy_down and img_w and img_h:
            scale_x = orig_w / img_w
            scale_y = orig_h / img_h
            pred_xy_full = (pred_xy_down[0] * scale_x, pred_xy_down[1] * scale_y)
        gt_center = sample.get("gt_center")
        gt_bbox = sample.get("gt_bbox")

        if args.use_bbox and gt_center:
            scene_dir = os.path.dirname(sample["image_path"])
            if scene_dir not in coords_cache:
                coords_path = os.path.join(scene_dir, "coordinates.json")
                coords_cache[scene_dir] = []
                if os.path.exists(coords_path):
                    try:
                        with open(coords_path, "r", encoding="utf-8") as f:
                            coords_cache[scene_dir] = json.load(f)
                    except:
                        pass
            coords = coords_cache[scene_dir]
            coords_bbox, _ = find_gt_bbox_by_center(gt_center, coords)
            if coords_bbox:
                gt_bbox = coords_bbox

        success = False
        dist = math.inf
        if pred_xy_full and gt_center:
            dist = math.hypot(pred_xy_full[0] - gt_center[0], pred_xy_full[1] - gt_center[1])
            print(dist)

        if pred_xy_full:
            px, py = pred_xy_full
            if gt_bbox:
                success = point_in_bbox((px, py), gt_bbox)
            elif gt_center:
                success = dist <= args.dist_thresh

        sample_type = sample.get("sample_type", 0)

        # Save visualization with pred/GT points for inspection.
        vis_dir = os.path.join(os.path.dirname(sample["image_path"]), "eval_results_AmbresVLM")
        vis_name = f"pred_gt_run{run_id + 1}_scene{sample['scene_id']}_type{sample_type}_idx{sample['sample_idx']}.jpg"
        save_pred_visualization(sample["image_path"], pred_xy_full, gt_center, gt_bbox, os.path.join(vis_dir, vis_name))

        if success:
            correct += 1
        type_stats.setdefault(sample_type, []).append(success)
        details.append(
            {
                "run": run_id,
                "scene_id": sample["scene_id"],
                "sample_type": sample_type,
                "sample_idx": sample["sample_idx"],
                "instruction": sample["instruction"],
                "target_category": sample.get("target_category"),
                "target_description": sample.get("target_description"),
                "task_ambiguous_pred": task_ambiguous,
                "task_objects_pred": final_task_objects,
                "pred_xy_downsampled": pred_xy_down,
                "pred_xy": pred_xy_full,
                "gt_center": gt_center,
                "gt_bbox": gt_bbox,
                "distance": dist,
                "success": success,
                "obj_detection_messages": final_detection_messages,
            }
        )

    total = len(dataset)
    overall_sr = correct / total if total else 0.0
    type_breakdown = {
        str(t): {
            "count": len(flags),
            "success_rate": (sum(1 for f in flags if f) / len(flags)) if flags else 0.0,
        }
        for t, flags in type_stats.items()
    }
    metrics = {"total_samples": total, "overall_sr": overall_sr, "type_breakdown": type_breakdown}
    return details, metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate AmbRes models on benchmark dataset with coordinate matching.")
    parser.add_argument(
        "benchmark_folder",
        type=str,
        help="Benchmark folder name under data/, e.g. benchmark-200",
    )
    parser.add_argument("--model_type", choices=["prompt", "finetune"], default="finetune")
    parser.add_argument("--env", choices=["sim", "real"], default="sim", help="Used only to pick finetuned checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save eval outputs; defaults to dataset/benchmark_runs/ambres_vlm")
    parser.add_argument("--run_tag", type=str, default=None, help="Optional tag to suffix output filenames, e.g., run1")
    parser.add_argument("--dist_thresh", type=float, default=25.0, help="Pixel threshold for center distance")
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated runs to average")
    parser.add_argument("--use_bbox", action="store_true", help="Use bounding box for success check if available")
    args = parser.parse_args()

    data_root = PROJECT_ROOT / "data"
    benchmark_path = data_root / args.benchmark_folder
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark folder not found: {benchmark_path}")
    args.dataset = str(benchmark_path)

    output_dir = args.output_dir or os.path.join(str(benchmark_path), "benchmark_runs", "ambres_vlm")
    os.makedirs(output_dir, exist_ok=True)

    all_details: List[Dict[str, Any]] = []
    metrics_list: List[Dict[str, float]] = []

    for r in range(args.runs):
        details, metrics = run_once(args, r)
        all_details.extend(details)
        metrics_list.append(metrics)

        tag = f"{args.run_tag}_run{r+1}" if args.run_tag else f"run{r+1}"
        metrics_name = f"eval_metrics_ambres_vlm_{tag}.json"
        details_name = f"eval_details_ambres_vlm_{tag}.json"
        with open(os.path.join(output_dir, metrics_name), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, details_name), "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)

    mean_sr = float(np.mean([m["overall_sr"] for m in metrics_list])) if metrics_list else 0.0
    summary = {
        "runs": metrics_list,
        "mean_overall_sr": mean_sr,
        "total_samples": metrics_list[0].get("total_samples", 0) if metrics_list else 0,
        "runs_count": len(metrics_list),
    }

    metrics_avg_name = "eval_metrics_ambres_vlm_avg.json" if not args.run_tag else f"eval_metrics_ambres_vlm_avg_{args.run_tag}.json"
    metrics_avg_path = os.path.join(output_dir, metrics_avg_name)

    with open(metrics_avg_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved per-run outputs under {output_dir}")
    print(f"Saved averaged metrics to {metrics_avg_path}")
    print(f"Mean success rate over {args.runs} runs: {mean_sr:.2%}")


if __name__ == "__main__":
    main()