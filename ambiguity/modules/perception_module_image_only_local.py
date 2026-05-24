import json
import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
try:
    from peft import PeftModel
except ImportError:  # peft is optional when no LoRA is provided
    PeftModel = None

from ambiguity.utils.api_wrappers import OpenAIWrapper, DinoWrapper


DEFAULT_SMOL_MODEL_PATH = \
    os.environ.get("SMOL_MODEL_PATH", "/data2/fdy/smolVLM/finetunes_vlm/output/Mytasks_full_real/checkpoint-150")
DEFAULT_SMOL_PROCESSOR_PATH = \
    os.environ.get("SMOL_PROCESSOR_PATH", "/data2/fdy/smolVLM/model_256M")

# DEFAULT_SMOL_MODEL_PATH = \
#     os.environ.get("SMOL_MODEL_PATH", "/home/zty_group/fdy/smolVLM/real")
# DEFAULT_SMOL_PROCESSOR_PATH = \
#     os.environ.get("SMOL_PROCESSOR_PATH", "/home/zty_group/fdy/smolVLM/real")

DEFAULT_SMOL_LORA_PATH = os.environ.get("SMOL_LORA_PATH", "")


class SmolVLMLocal:
    """Minimal in-process SmolVLM wrapper (no HTTP)."""

    def __init__(
        self,
        model_path: str = DEFAULT_SMOL_MODEL_PATH,
        processor_path: Optional[str] = DEFAULT_SMOL_PROCESSOR_PATH,
        lora_path: str = DEFAULT_SMOL_LORA_PATH,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.processor_path = processor_path or model_path
        self.lora_path = lora_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(self.processor_path)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            _attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",
        ).to(self.device)

        if self.lora_path and PeftModel is not None:
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
        self.model.eval()

    def get_response(self, prompt: str, image_path: str, max_tokens: int = 200) -> str:
        image = Image.open(image_path).convert("RGB")
        # Follow the same pattern as smolVLM model_server: build chat template with image token.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=prompt_text,
            images=[image],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded[0] if decoded else ""


def _extract_answer_only(text: str) -> str:
    """Strip chat template markers like 'Assistant:' and return answer only."""
    if not text:
        return ""
    if "Assistant:" in text:
        # take the last Assistant: block
        text = text.split("Assistant:")[-1]
    if "assistant:" in text:
        text = text.split("assistant:")[-1]
    # Drop possible 'User:' prefix pieces
    if "User:" in text:
        parts = text.split("User:")
        text = parts[-1]
    return text.strip()


class ScenePerceptionImageOnlyLocal:
    """Perception pipeline that keeps DINO + SmolVLM in process (no HTTP services)."""

    def __init__(
        self,
        config_path: str = "ambiguity/configs/models.yaml",
        smol_model_path: str = DEFAULT_SMOL_MODEL_PATH,
        smol_processor_path: Optional[str] = DEFAULT_SMOL_PROCESSOR_PATH,
        smol_lora_path: str = DEFAULT_SMOL_LORA_PATH,
    ) -> None:
        self.openai_wrapper = OpenAIWrapper(config_path)
        self.smol_local = SmolVLMLocal(
            model_path=smol_model_path,
            processor_path=smol_processor_path,
            lora_path=smol_lora_path,
        )
        self.dino_wrapper = DinoWrapper(config_path)

    def analyze_scene(
        self,
        image_path: str,
        results_dir: str,
        use_oracle: bool = False,
        output_format: str = "v2",
        return_masks: bool = False,
    ) -> Dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        os.makedirs(results_dir, exist_ok=True)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        initial_description = self._describe_scene(image_path, use_oracle)
        noun_candidates = self._extract_nouns(initial_description)
        if not noun_candidates:
            noun_candidates = ["object"]

        detections, mask, obj_names = self.dino_wrapper.get_object_localizations(
            image=image,
            object_nouns=noun_candidates,
            results_dir=results_dir,
        )

        boxes = self._normalize_boxes(detections)
        if len(boxes) == 0:
            return {"Objects": [], "relationships": [], "bbox": []}

        detailed_names = self._collect_attributes(image, boxes, obj_names, results_dir, use_oracle)
        objects, bbox_list = self._build_objects(detailed_names, boxes, mask, width, height)
        relationships = self._build_relationships(objects)

        scene_graph = {
            "Objects": objects,
            "relationships": relationships,
            "bbox": bbox_list,
        }

        with open(os.path.join(results_dir, "image_only_scene_graph.json"), "w", encoding="utf-8") as f:
            json.dump(scene_graph, f, indent=2, ensure_ascii=False)

        if return_masks:
            scene_graph["masks"] = mask

        return scene_graph

    # ------------------------------------------------------------------
    def _describe_scene(self, image_path: str, use_oracle: bool = False) -> str:
        prompt = "Describe all objects in the image."
        if use_oracle:
            return self.openai_wrapper.get_vl_completion(
                prompt="List every distinct object category you can see in English nouns only.",
                message="Here is image:",
                image_path=image_path,
            ) or ""
        return self.smol_local.get_response(prompt=prompt, image_path=image_path, max_tokens=200) or ""

    def _extract_nouns(self, description: str) -> List[str]:
        nouns: List[str] = []
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
            arr = detections['xyxy']
            return arr.tolist() if hasattr(arr, 'tolist') else list(arr)
        if isinstance(detections, list):
            return detections
        if hasattr(detections, 'xyxy'):
            arr = detections.xyxy
            return arr.tolist() if hasattr(arr, 'tolist') else list(arr)
        if hasattr(detections, 'tolist'):
            return detections.tolist()
        return []

    def _collect_attributes(
        self,
        image: Image.Image,
        boxes: List[List[float]],
        obj_names: List[str],
        results_dir: str,
        use_oracle: bool,
    ) -> List[str]:
        if not boxes or not obj_names:
            return obj_names or []

        width, height = image.size
        detailed = []
        os.makedirs(os.path.join(results_dir, "eval_results_Full"), exist_ok=True)
        for idx, (box, base_name) in enumerate(zip(boxes, obj_names)):
            try:
                x1, y1, x2, y2 = box
                if x2 <= 1.0 and y2 <= 1.0:
                    x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
                crop = image.crop((x1, y1, x2, y2))
                crop_path = os.path.join(results_dir, "eval_results_Full", f"crop_{idx}_{base_name.replace(' ', '_')}.png")
                crop.save(crop_path)

                attr_prompt = "Describe the color and appearance of the object."
                if use_oracle:
                    attributes = self.openai_wrapper.get_vl_completion(
                        prompt=attr_prompt,
                        message="Here is image:",
                        image_path=crop_path,
                    )
                else:
                    attributes = self.smol_local.get_response(
                        prompt=attr_prompt,
                        image_path=crop_path,
                        max_tokens=80,
                    )
                attr_clean = _extract_answer_only(attributes).replace('.', ' ').replace(',', ' ').strip()
                attr_clean = " ".join(attr_clean.split())  # normalize spaces
                if attr_clean:
                    detailed.append(f"{attr_clean} {base_name}")
                else:
                    detailed.append(base_name)
            except Exception:
                detailed.append(base_name)
        return detailed

    def _build_objects(
        self,
        names: List[str],
        boxes: List[List[float]],
        masks,
        width: int,
        height: int,
    ) -> Tuple[List[Dict], List[List[float]]]:
        objects: List[Dict] = []
        bbox_list: List[List[float]] = []

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
                "size": int(area * 10000),
                "position": {
                    "x": round(norm_x, 4),
                    "y": round(norm_y, 4),
                },
            })
            bbox_list.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
        return objects, bbox_list

    def _build_relationships(self, objects: List[Dict]) -> List[str]:
        relationships: List[str] = []
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

