import json
import os
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

from PIL import Image

from ambiguity.utils.api_wrappers import OpenAIWrapper
from ambiguity.utils.dino_client import DinoServiceClient
from smolVLM.client import SmolVLMClient


class ScenePerceptionImageOnly:
    """Lightweight perception pipeline that relies on RGB images only."""

    def __init__(self, config_path='ambiguity/configs/models.yaml',
                 smol_url="http://localhost:1234",
                 dino_url="http://localhost:5678"):
        self.openai_wrapper = OpenAIWrapper(config_path)
        self.smol_client = SmolVLMClient(smol_url)
        self.dino_client = DinoServiceClient(base_url=dino_url)
        project_root = Path(__file__).resolve().parents[2]
        prompt_dir = project_root / "prompts" / "repro_release"
        self.prompts = {
            "describe_scene": (prompt_dir / "perception_describe_scene.txt").read_text(encoding="utf-8").strip(),
            "oracle_scene_nouns": (prompt_dir / "perception_oracle_scene_nouns.txt").read_text(encoding="utf-8").strip(),
            "attribute": (prompt_dir / "perception_attribute.txt").read_text(encoding="utf-8").strip(),
        }

    def analyze_scene(self, image_path: str, results_dir: str,
                      use_oracle: bool = False, output_format: str = 'v2', return_masks: bool = False) -> Dict:
        """Return a scene graph estimated purely from the RGB image."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        os.makedirs(results_dir, exist_ok=True)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Stage 1: describe scene to assemble candidate nouns
        initial_description = self._describe_scene(image_path, use_oracle)
        noun_candidates = self._extract_nouns(initial_description)
        if not noun_candidates:
            noun_candidates = ["object"]

        detections, mask, obj_names = self.dino_client.get_object_localizations(
            image=image,
            object_nouns=noun_candidates,
            results_dir=results_dir
        )

        boxes = self._normalize_boxes(detections)
        if not boxes:
            return {"Objects": [], "relationships": [], "bbox": []}

        detailed_names = self._collect_attributes(image, boxes, obj_names, results_dir, use_oracle)
        objects, bbox_list = self._build_objects(detailed_names, boxes, mask, width, height)
        relationships = self._build_relationships(objects)

        scene_graph = {
            "Objects": objects,
            "relationships": relationships,
            "bbox": bbox_list
        }

        # Optional save for debugging (exclude masks from JSON)
        with open(os.path.join(results_dir, "image_only_scene_graph.json"), 'w', encoding='utf-8') as f:
            json.dump(scene_graph, f, indent=2, ensure_ascii=False)

        if return_masks:
            scene_graph['masks'] = mask

        return scene_graph

    # ------------------------------------------------------------------
    def _describe_scene(self, image_path: str, use_oracle: bool=False) -> str:
        prompt = self.prompts["describe_scene"]
        if use_oracle:
            return self.openai_wrapper.get_vl_completion(
                prompt=self.prompts["oracle_scene_nouns"],
                message="Here is image:",
                image_path=image_path
            ) or ""
        return self.smol_client.get_smolvlm_response(
            prompt=prompt,
            image_path=image_path,
            max_tokens=200
        ) or ""

    def _extract_nouns(self, description: str) -> List[str]:
        nouns = []
        seen = set()
        for item in description.replace('，', ',').replace('.', '').split(','):
            noun = item.strip().lower()
            if not noun:
                continue
            last_word = noun.split()[-1]
            if last_word and last_word not in seen:
                seen.add(last_word)
                nouns.append(last_word)
        return nouns

    def _normalize_boxes(self, detections) -> List[List[float]]:
        if detections is None:
            return []
        if isinstance(detections, dict) and 'xyxy' in detections:
            return detections['xyxy']
        if isinstance(detections, list):
            return detections
        if hasattr(detections, 'xyxy'):
            return detections.xyxy
        if hasattr(detections, 'tolist'):
            return detections.tolist()
        return []

    def _collect_attributes(self, image: Image.Image, boxes: List[List[float]],
                             obj_names: List[str], results_dir: str,
                             use_oracle: bool) -> List[str]:
        if not boxes or not obj_names:
            return obj_names or []

        width, height = image.size
        crops_dir = os.path.join(results_dir, "eval_results_Full")
        os.makedirs(crops_dir, exist_ok=True)
        detailed = []
        for idx, (box, base_name) in enumerate(zip(boxes, obj_names)):
            try:
                x1, y1, x2, y2 = box
                if x2 <= 1.0 and y2 <= 1.0:
                    x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
                crop = image.crop((x1, y1, x2, y2))
                crop_path = os.path.join(crops_dir, f"crop_{idx}_{base_name.replace(' ', '_')}.png")
                crop.save(crop_path)
                attr_prompt = self.prompts["attribute"]
                if use_oracle:
                    attributes = self.openai_wrapper.get_vl_completion(
                        prompt=attr_prompt,
                        message="Here is image:",
                        image_path=crop_path
                    )
                else:
                    attributes = self.smol_client.get_smolvlm_response(
                        prompt=attr_prompt,
                        image_path=crop_path,
                        max_tokens=80
                    )
                attributes = (attributes or "").strip().replace('.', ' ').replace(',', ' ')
                if attributes:
                    detailed.append(f"{attributes} {base_name}")
                else:
                    detailed.append(base_name)
            except Exception:
                detailed.append(base_name)
        return detailed

    def _build_objects(self, names: List[str], boxes: List[List[float]], masks,
                       width: int, height: int) -> Tuple[List[Dict], List[List[float]]]:
        objects = []
        bbox_list = []

        if masks is None:
            masks = [None] * len(boxes)

        for idx, (name, box, mask_item) in enumerate(zip(names, boxes, masks), start=1):
            x1, y1, x2, y2 = [float(v) for v in box]
            if x2 <= 1.0 and y2 <= 1.0:
                x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            norm_x = center_x / width
            norm_y = center_y / height
            area = (box_w * box_h) / float(width * height)

            if mask_item is not None:
                try:
                    if hasattr(mask_item, 'sum'):
                        mask_pixel_count = float(mask_item.sum())
                    else:
                        mask_pixel_count = float(np.sum(np.array(mask_item)))
                    area = mask_pixel_count / float(width * height)
                except Exception:
                    pass

            parts = name.split()
            category = parts[-1] if parts else name
            attributes = " ".join(parts[:-1]) if len(parts) > 1 else ""

            objects.append({
                "id": idx,
                "name": name,
                "category": category,
                "attributes": attributes,
                "size": int(area*10000), #放大 不要小数
                "position": {
                    "x": round(norm_x, 4),
                    "y": round(norm_y, 4)
                }
            })
            bbox_list.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
        return objects, bbox_list

    def _build_relationships(self, objects: List[Dict]) -> List[str]:
        relationships = []
        if len(objects) < 2:
            return relationships

        tolerance = 0.00001
        for obj_a in objects:
            for obj_b in objects:
                if obj_a['id'] == obj_b['id']:
                    continue
                dx = obj_a['position']['x'] - obj_b['position']['x']
                dy = obj_a['position']['y'] - obj_b['position']['y']

                if abs(dx) > tolerance:
                    if dx < 0:
                        relationships.append(f"{obj_a['id']} is left of {obj_b['id']}")
                    else:
                        relationships.append(f"{obj_a['id']} is right of {obj_b['id']}")

                if abs(dy) > tolerance:
                    if dy < 0:
                        relationships.append(f"{obj_a['id']} is behind of {obj_b['id']}")
                    else:
                        relationships.append(f"{obj_a['id']} is in front {obj_b['id']}")
        return relationships
