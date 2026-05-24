import argparse
from tqdm import tqdm
from PIL import Image
import os
import sys
import json
import re
import cv2
import yaml
import time
import base64
from pathlib import Path

# Add project root to path
# We are in evaluate/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports from project
from ambiguity.modules.language_module_image_only import LanguageProcessorFullSFT as BaseLanguageProcessorFullSFT
from ambiguity.modules.perception_module_image_only import ScenePerceptionImageOnly


class LanguageProcessorFullSFT(BaseLanguageProcessorFullSFT):
    """复用全量 SFT 语言模块，仅保留无场景图流程。"""

    def resolve_ambiguity(self, user_command, candidates, history=None):
        """歧义消解"""
        print("\n--- 歧义消解 (Stage 2) ---")
        sys_prompt = "You are a robotic-arm assistant. Given the user instruction and a JSON list of candidate objects (detected from the scene), select exactly one object ID. If there is ambiguity, first ask a clarifying question based on object attribute differences, then make the choice."

        if history is None:
            candidates_json = json.dumps(candidates, ensure_ascii=False)
            user_input = f"Instruction: {user_command}\nCandidates:\n{candidates_json}"
            self.conversation_log.append({"role": "user", "content": user_input})
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_input}
            ]
        else:
            messages = history
        
        llm_response = self._call_llm(messages=messages)
        self.conversation_log.append({"role": "robot", "content": llm_response})
        content = llm_response.get('content', '').strip() if llm_response else "{}"
        think = llm_response.get('thinking', '').strip() if llm_response else ""
        print(f"消歧模型输出: {content}")
        
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Try to fix common JSON errors
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(0))
                except:
                    print(f"JSON解析失败: {content}")
                    result = {}
            else:
                result = {}
            
        return result, think

    def process_command(self, user_command, image_path, perception_module, output_dir):
        """
        New Flow:
        1. Extract category from instruction (No SceneGraph needed).
        2. DINO Detect using category.
        3. Check count:
           - 1: Success/Return.
           - >1: Collect attributes -> Resolve Ambiguity.
        """
        print("=" * 50)
        print("开始处理用户指令 (No SceneGraph Flow)")
        print(user_command)
        print("=" * 50)
        self.conversation_log = []
        self.intermediate_results = {}
        
        # Step 1: LLM Extract Target Object Category
        print("\n--- Stage 1: Extract Category (Agent) ---")
        # We pass an empty list for obj_list, relying on the Agent/VLM to identify the object.
        # Original code used: self.extract_target_object(user_command, obj_list, image_path)
        obj_list = perception_module._describe_scene(image_path)
        obj_list = perception_module._extract_nouns(obj_list)
        target_str, _ = self.extract_target_object(user_command, obj_list, image_path)
        
        print(f"Extracted Target Category: {target_str}")
        # self.intermediate_results['target_objs_split'] = [target_str] # For judge compatibility
        
        if not target_str:
            print("Failed to extract target.")
            return []

        # Step 2: DINO Detection
        print("\n--- DINO Detection ---")
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        # Use perception module's client
        noun_list = [n.strip() for n in target_str.split(',')] # Handle potential multiple
        self.intermediate_results['target_objs_split'] = [noun_list]
        detections, mask, obj_names = perception_module.dino_client.get_object_localizations(
            image=image,
            object_nouns=noun_list,
            results_dir=output_dir+f"/eval_results_No_SceneGraph/{target_str}.jpg"
        )
        # Normalize boxes
        boxes = perception_module._normalize_boxes(detections)
        print(f"DINO detected {len(boxes)} instances.")
        print(boxes)
        if len(boxes) == 0:
            print("No objects detected.")
            return []
            
        # Check count
        if len(boxes) == 1:
            # Only 1 object. Finish.
            # Build basic object dict
            # We reuse _build_objects to get standard format (ID, etc)
            detailed_names = obj_names
            objects, bbox_list = perception_module._build_objects(detailed_names, boxes, mask, width, height)
            
            # Inject bbox into the object dict because evaluation needs it
            if objects and bbox_list:
                objects[0]['bbox'] = bbox_list[0]
                
            print(f"Single object detected. Selection: ID={objects[0]['id']}")
            self.intermediate_results['final_results'] = objects
            return objects
            
        else:
            # >= 2 objects. Enter Stage 2 (Ambiguity).
            print(f"Multiple objects ({len(boxes)}). Entering Stage 2.")
            
            # No attributes collection as requested
            
            # Build simplified candidates list with id, category, bbox
            candidates_list = []
            for idx, (name, box) in enumerate(zip(obj_names, boxes), start=1):
                # Ensure box is a list
                box_list = box.tolist() if hasattr(box, 'tolist') else box
                candidates_list.append({
                    "id": idx,
                    "category": name,
                    "bbox": box_list
                })
            
            # Now resolve ambiguity
            self.intermediate_results.setdefault('candidates', []).append({'objects': candidates_list})
            
            # Use instruction for ambiguity resolution if target extraction was just a category
            current_command = user_command
            history = None
            i = 3
            selected_candidate = None
            
            while i > 0:
                i -= 1
                res, think = self.resolve_ambiguity(current_command, candidates_list, history)
                self.intermediate_results.setdefault('resolve_ambiguity_results', []).append(res)
                
                if history is None:
                    # Init history
                    sys_prompt = "You are a robotic-arm assistant. Given the user instruction and a JSON list of candidate objects, select exactly one object ID. If there is ambiguity, first ask a clarifying question based on object attribute differences, then make the choice."
                    candidates_json = json.dumps(candidates_list, ensure_ascii=False)
                    user_input = f"Instruction: {current_command}\nCandidates:\n{candidates_json}"
                    history = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_input}
                    ]
                
                history.append({"role": "assistant", "content": json.dumps(res, ensure_ascii=False)})

                if res.get("action") == "select":
                    target_id = res.get("target_id")
                    selected_candidate = next((c for c in candidates_list if c["id"] == target_id), None)
                    if selected_candidate:
                        print(f"Ambiguity Resolved. Selected: ID={selected_candidate['id']}")
                    else:
                        print(f"Error: Selected ID {target_id} not found.")
                    break
                elif res.get("action") == "ask":
                    question = res.get("question")
                    ans = self.get_user_input(question) # This uses the mocked input
                    res['user_answer'] = ans
                    history.append({"role": "user", "content": ans})
                else:
                    print(f"Unknown action/flow: {res}")
                    # If just thinking or failed, break or continue?
                    # Assuming if no action, we might stop or it's a failure.
                    break
            
            final = [selected_candidate] if selected_candidate else []
            self.intermediate_results['final_results'] = final
            return final

class ImageOnlyEvaluatorFullSFT:
    """Evaluation pipeline variant that uses RGB images and Full SFT Qwen3."""

    def __init__(self, config_path, model_name=None, use_cached_graph:bool = False):
        # Initialize the new Language Processor
        self.language = LanguageProcessorFullSFT(config_path=config_path, model_name=model_name)
        # Perception module initialized but NOT used for full scene analysis
        self.perception = ScenePerceptionImageOnly(config_path=config_path)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.api_key = os.getenv("DASHSCOPE_API_KEY", config['openai']['api_key'])
        self.base_url = config['openai']['base_url']
        # Judge configs
        self.llm_model = "qwen3-235b-a22b-instruct-2507" 
        self.vl_model = config['openai'].get('qwen_vl_model', 'qwen-vl-max')

        from openai import OpenAI
        self.llm = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.vlm = OpenAI(api_key=self.api_key, base_url=self.base_url)

    # ... Copied Judge Methods ...
    def _ask_text_judge(self, prompt):
        try:
            response = self.llm.chat.completions.create(
                model=self.llm_model, messages=[{"role": "user", "content": prompt}], temperature=0
            )
            content = response.choices[0].message.content.strip().upper()
            return "YES" in content
        except Exception as exc:
            print(f"Text Judge Error: {exc}")
            return False

    def judge_stage1_category(self, instruction, online_obj_list, predicted_categories, target_description):
        if not predicted_categories: return False
        # online_obj_list might be just the target or empty since we don't have scene graph
        # But we still check if the prediction is reasonable given the instruction + GT.
        prompt = f"""
        You are evaluation whether a robotic system picked the correct object CATEGORY after parsing a user instruction.
        User Instruction: {instruction}
        Visual Target Description (from benchmark GT): {target_description}
        Detected Object Categories (Target Candidates): {online_obj_list}
        Categories predicted by the system: {predicted_categories}
        Decide if the predicted categories correspond to the user's intended object (synonyms count as correct).
        Respond with ONLY "YES" or "NO".
        """
        return self._ask_text_judge(prompt)

    def judge_stage2_instance(self, instruction, candidates, conversation_log, selected_obj, target_description, user_answer_hint=None):
        if not selected_obj: return False
        conv_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_log]) if conversation_log else "(no questions asked)"
        prompt = f"""
        A robot asked clarification questions to pick a specific OBJECT INSTANCE.
        User Instruction: {instruction}
        Ground Truth Visual Target: {target_description}
        Candidate objects (id, category, attributes): {candidates}
        Dialogue History:\n{conv_text}
        Final selection by the robot: id={selected_obj.get('id')}, category={selected_obj.get('category')}
        Judge if the final selection satisfies the user's answers and the intent. Reply ONLY with "YES" or "NO".
        """
        return self._ask_text_judge(prompt)

    def judge_by_center(self, selected_bbox, gt_center, tolerance=30):
        if not selected_bbox or not gt_center: return False
        x1, y1, x2, y2 = selected_bbox
        pred_cx, pred_cy = (x1 + x2) / 2, (y1 + y2) / 2
        gt_cx, gt_cy = gt_center
        distance = ((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)**0.5
        return distance < tolerance

    def save_visualization(self, image_path, selected_bbox, output_path, gt_center=None):
        if not selected_bbox: return
        img = cv2.imread(image_path)
        if img is None: return
        x1, y1, x2, y2 = selected_bbox
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        if gt_center:
            cx, cy = int(gt_center[0]), int(gt_center[1])
            cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)
        if output_path:
            cv2.imwrite(output_path, img)

    def vlm_judge(self, image_path, instruction, selected_bbox, target_description, conversation_text=None, selection_text=None, output_path=None):
        if not selected_bbox: return False
        img = cv2.imread(image_path)
        if img is None: return False
        x1, y1, x2, y2 = selected_bbox
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        if output_path: cv2.imwrite(output_path, img)
        _, buffer = cv2.imencode('.jpg', img)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        prompt = f"""
        You are a judge for a robotic manipulation task.
        User Instruction: "{instruction}"
        Ground Truth Target Description: "{target_description}"
        Framework selection summary: {selection_text or '(not provided)'}
        Dialogue (if any):\n{conversation_text or '(no dialogue)'}
        The robot has selected the object inside the RED bounding box in the image.
        Task:
        1. Check if the selected object matches the "Ground Truth Target Description".
        2. Check if the selection makes sense for the "User Instruction".
        Is the robot's selection correct? Answer ONLY with "YES" or "NO".
        """
        try:
            response = self.vlm.chat.completions.create(
                model=self.vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }]
            )
            content = response.choices[0].message.content.strip().upper()
            return "YES" in content
        except Exception as exc:
            print(f"VLM Judge Error: {exc}")
            return False

    def run_evaluation(self, dataset_path, output_dir=None, run_tag=None):
        dataset_root = Path(dataset_path)
        with open(dataset_root / "dataset_index.json", 'r') as f:
            dataset_index = json.load(f)

        all_results = []
        print(f"Starting evaluations on {len(dataset_index)} scenes with Full SFT model (No SceneGraph)...")

        for scene_data in tqdm(dataset_index):
            scene_dir = str(dataset_root / scene_data['scene_dir'])
            os.makedirs(scene_dir+"/eval_results_No_SceneGraph",exist_ok=True)
            image_path = str(dataset_root / scene_data['image_path'])
            samples_path = dataset_root / scene_data['samples_path']
            with open(samples_path, 'r') as f:
                samples = json.load(f)
            
            # Note: We do NOT generate scene graph anymore.
            # But we might need an 'results_dir' for dino crops.
            results_dir = scene_dir

            for sample in samples:
                task_type = sample.get('task_type', 'pick')
                if task_type == 'pick_place':
                    # Simplified logic: just focus on 'pick' or primary target for now as request didn't specify multi-step change specifically
                    # except "stage 1 -> dino -> if 2...". 
                    # Assuming instruction handles it or we process once. 
                    pick_gt = sample['ground_truth'].get('pick', {})
                    place_gt = sample['ground_truth'].get('place', {})
                    target_description = f"Pick target: {pick_gt.get('target_description')}. Place target: {place_gt.get('target_description')}"
                    user_answer_hint = f"{pick_gt.get('user_answer_hint', '')} {place_gt.get('user_answer_hint', '')}".strip()
                else:
                    target_description = sample['ground_truth']['target_description']
                    user_answer_hint = sample['ground_truth'].get('user_answer_hint', "")

                question_asked = False

                def mock_input(question):
                    nonlocal question_asked
                    question_asked = True
                    prompt = f"""
                     Role
You are a precision-guidance assistant for a vision-capable robot. Your sole task is to command object selection using ONLY explicit attributes from the ground-truth description.

Inputs
- Robot's Question: "{question}" (Multiple-choice query about ONE attribute: color/material/shape)
- Target Description: "{target_description}" (Ground-truth attributes ONLY)

Execution Protocol
1. OPTION MATCH CHECK: 
   → Scan "{target_description}" for EXACT WORD MATCHES with options in "{question}"
   → IF found: Respond "Pick up the [exact matching option] one." (e.g., "Pick up the ceramic one")
   
2. NO-MATCH FALLBACK:
   → IF no options match: Extract the SINGLE MOST DISTINCTIVE attribute explicitly stated in "{target_description}"
   → Respond "Pick up the [attribute] one." (e.g., "Pick up the octagonal one")

Hard Constraints
- 🚫 ZERO INFERENCE: Never use attributes absent from "{target_description}"

- ✅ ONE SENTENCE ONLY: Strictly 5-12 word response


Output Goal
A self-contained command that uniquely identifies the target using ground-truth attributes ONLY.
                    """
                    try:
                        response = self.llm.chat.completions.create(
                            model=self.llm_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.1
                        )
                        ans = response.choices[0].message.content.strip()
                    except:
                        ans = "No more information."
                    self.language.conversation_log.append({"role": "robot", "content": question})
                    self.language.conversation_log.append({"role": "user", "content": ans})
                    return ans

                self.language.get_user_input = mock_input # Override input

                start_time = time.time()
                final_results = []
                stage_1_success = False
                stage_2_success = False
                
                try:
                    # Run Full SFT Pipeline
                    # Pass self.perception instead of scene graph
                    final_results = self.language.process_command(
                        sample['instruction'], image_path, self.perception, results_dir
                    )
                    
                    predicted_cats = self.language.intermediate_results.get('target_objs_split', [])
                    # We pass predicted_cats as online_obj_list because that's what we found
                    online_obj_list = predicted_cats 
                    
                    # stage_1_success = self.judge_stage1_category(
                    #     sample['instruction'], online_obj_list, predicted_cats, target_description
                    # )
                except Exception as exc:
                    print(f"Inference error: {exc}")
                    import traceback
                    traceback.print_exc()
                    final_results = []

                latency = (time.time() - start_time) * 1000
                
                candidates_all = self.language.intermediate_results.get('candidates', []) if hasattr(self.language, 'intermediate_results') else []
                # candidates is now [{'objects': [...]}] struct
                primary_candidates = candidates_all[0]['objects'] if candidates_all and 'objects' in candidates_all[0] else []
                
                selected_obj = final_results[0] if final_results else None
                
                # if final_results:
                #     # Only judge Stage 2 if we actually had candidates (meaning >1 objects)
                #     if len(primary_candidates) > 1:
                #         stage_2_success = self.judge_stage2_instance(
                #             sample['instruction'], primary_candidates, self.language.conversation_log, selected_obj,
                #             target_description, user_answer_hint
                #         )
                #     else:
                #         # 1 object case, if stage 1 (detection) matches and we selected it, implicitly good for stage 2?
                #         # Or stage 2 is N/A. We usually set it false if flow didn't happen, or true if success.
                #         # Let's align with "if success then s1=True, s2=True" logic at end.
                #         stage_2_success = False 
                # else:
                #     stage_2_success = False

                success = False
                pick_success = False
                place_success = False
                # bbox list is not from scene graph anymore. We can get it from selected obj.
                
                if task_type == 'pick_place':
                     # ... (Keep existing logic if applicable, but handle missing bbox list)
                     pass 

                elif final_results and len(final_results) == 1:
                    selected_obj = final_results[0]
                    bbox = selected_obj.get('bbox')
                    if bbox:
                        output_path = os.path.join(scene_dir+"/eval_results_No_SceneGraph", f"bbox_{scene_data['scene_id']}_{sample['type']}_no_scenegraph.jpg")
                        gt_center = sample["ground_truth"].get('center')
                        if gt_center:
                            print("===Judge by Center====")
                            success = self.judge_by_center(bbox, gt_center)
                            self.save_visualization(image_path, bbox, output_path, gt_center=gt_center)
                        else:
                            print("===Judge by VLM====")
                            conversation_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.language.conversation_log])
                            selection_text = json.dumps(final_results, ensure_ascii=False)
                            success = self.vlm_judge(image_path, sample['instruction'], bbox, target_description, conversation_text, selection_text, output_path)

                if success:
                    stage_1_success = True
                    stage_2_success = True

                result_entry = {
                    "scene_id": scene_data['scene_id'],
                    "sample_type": sample['type'],
                    "instruction": sample['instruction'],
                    "target_description": target_description,
                    "question_asked": question_asked,
                    "success": success,
                    "pick_success": pick_success,
                    "place_success": place_success,
                    "stage_1_success": stage_1_success,
                    "stage_2_success": stage_2_success,
                    "latency": latency,
                    "conversation_log": self.language.conversation_log,
                }
                all_results.append(result_entry)

        return self.calculate_metrics(all_results, output_dir or dataset_path, run_tag=run_tag)

    def calculate_metrics(self, results, output_dir, run_tag=None):
        if not results: return
        metrics = {
            "total_samples": len(results),
            "overall_sr": 0,
            "avg_latency": 0,
            "category_subset_metrics": {"count": 0, "stage_1_accuracy": 0, "success_rate": 0},
            "instance_subset_metrics": {"count": 0, "clarification_recall": 0, "success_rate": 0},
            "no_instance_ambiguity_metrics": {"count": 0, "clarification_fpr": 0},
            "type_breakdown": {}
        }
        type_stats = {t: {"total": 0, "correct": 0, "s1_correct": 0, "s2_correct": 0, "asked": 0, "latency": 0} for t in [1, 2, 3, 4,5]}
        overall_correct = 0
        total_latency = 0
        cat_subset_total = cat_subset_s1 = cat_subset_final = 0
        inst_subset_total = inst_subset_asked = inst_subset_final = 0
        no_inst_total = no_inst_asked = 0

        for res in results:
            t = res['sample_type'] if isinstance(res['sample_type'], int) else int(''.join([c for c in res['sample_type'] if c.isdigit()]) or 0)
            is_correct = res['success']
            s1_correct = res['stage_1_success']
            s2_correct = res.get('stage_2_success', False)
            asked = res['question_asked']
            latency = res['latency']

            if t in type_stats:
                type_stats[t]["total"] += 1
                type_stats[t]["latency"] += latency
                if is_correct: type_stats[t]["correct"] += 1
                if s1_correct: type_stats[t]["s1_correct"] += 1
                if s2_correct: type_stats[t]["s2_correct"] += 1
                if asked: type_stats[t]["asked"] += 1
            
            if is_correct: overall_correct += 1
            total_latency += latency

            # Category Ambiguity (2 + 4)
            if t in [2, 4]:
                cat_subset_total += 1
                if s1_correct: cat_subset_s1 += 1
                if is_correct: cat_subset_final += 1

            # Instance Ambiguity (3 + 4)
            if t in [3, 4]:
                inst_subset_total += 1
                if asked: inst_subset_asked += 1
                if is_correct: inst_subset_final += 1
            
            # No Instance Ambiguity (1 + 2)
            if t in [1, 2]:
                no_inst_total += 1
                if asked: no_inst_asked += 1

        metrics['overall_sr'] = overall_correct / metrics['total_samples'] if metrics['total_samples'] else 0
        metrics['avg_latency'] = total_latency / metrics['total_samples'] if metrics['total_samples'] else 0

        if cat_subset_total:
            metrics['category_subset_metrics'] = {
                "count": cat_subset_total,
                "stage_1_accuracy": cat_subset_s1 / cat_subset_total,
                "success_rate": cat_subset_final / cat_subset_total
            }
        if inst_subset_total:
            metrics['instance_subset_metrics'] = {
                "count": inst_subset_total,
                "clarification_recall": inst_subset_asked / inst_subset_total,
                "success_rate": inst_subset_final / inst_subset_total
            }
        if no_inst_total:
            metrics['no_instance_ambiguity_metrics'] = {
                "count": no_inst_total,
                "clarification_fpr": no_inst_asked / no_inst_total
            }

        for t in [1, 2, 3, 4,5]:
            stats = type_stats[t]
            total = stats['total']
            metrics['type_breakdown'][f"type_{t}"] = {
                "count": total,
                "success_rate": stats['correct'] / total if total else 0,
                "stage_1_acc": stats['s1_correct'] / total if total else 0,
                "stage_2_acc": stats['s2_correct'] / total if total else 0,
                "ask_rate": stats['asked'] / total if total else 0,
                "avg_latency": stats['latency'] / total if total else 0
            }
        print("\n" + "=" * 30)
        print("HADR Evaluation Metrics (Full SFT / Image-Only / No SG)")
        print("=" * 30)
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"Overall Success Rate: {metrics['overall_sr']:.6%}")
        print("-" * 20)
        os.makedirs(output_dir, exist_ok=True)
        metrics_name = "eval_metrics_image_only_no_SceneGraph.json" if not run_tag else f"eval_metrics_image_only_no_SceneGraph_{run_tag}.json"
        details_name = "eval_details_image_only_no_SceneGraph.json" if not run_tag else f"eval_details_image_only_no_SceneGraph_{run_tag}.json"
        with open(os.path.join(output_dir, metrics_name), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        with open(os.path.join(output_dir, details_name), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        return metrics

def _average_metrics(metrics_list):
    if not metrics_list:
        return {}
    def is_number(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    def avg_dict(dicts):
        keys = dicts[0].keys()
        out = {}
        for k in keys:
            values = [d.get(k) for d in dicts]
            if all(isinstance(v, dict) for v in values):
                out[k] = avg_dict(values)
            elif all(is_number(v) for v in values):
                out[k] = sum(values) / len(values)
            else:
                out[k] = values[0]
        return out
    return avg_dict(metrics_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image-only Full SFT benchmark (No SceneGraph).")
    parser.add_argument(
        "benchmark_folder",
        type=str,
        help="Benchmark folder name under data/, e.g. benchmark-200",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="How many repeated runs to execute (default: 1).",
    )
    args = parser.parse_args()

    data_root = Path(project_root) / "data"
    benchmark_path = data_root / args.benchmark_folder
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark folder not found: {benchmark_path}")

    output_dir = benchmark_path / "benchmark_runs" / "no_scenegraph"
    use_cached = os.getenv("USE_CACHED_SCENE_GRAPH", "0") == "1"
    config_path = Path(project_root) / "ambiguity" / "configs" / "models.yaml"

    evaluator = ImageOnlyEvaluatorFullSFT(
        config_path=str(config_path),
        model_name=None,
        use_cached_graph=use_cached,
    )

    run_metrics = []
    for i in range(args.runs):
        print("\n" + "=" * 40)
        print(f"Running benchmark: {i + 1}/{args.runs}")
        print("=" * 40)
        metrics = evaluator.run_evaluation(str(benchmark_path), output_dir=str(output_dir), run_tag=f"run{i + 1}")
        if metrics:
            run_metrics.append(metrics)

    if run_metrics:
        avg_metrics = _average_metrics(run_metrics)
        print("\n" + "=" * 30)
        print(f"Average Metrics ({args.runs} runs)")
        print("=" * 30)
        print(f"Total Samples: {avg_metrics.get('total_samples', 0)}")
        print(f"Overall Success Rate: {avg_metrics.get('overall_sr', 0):.6%}")
        print("-" * 20)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir / "eval_metrics_image_only_no_SceneGraph_avg.json", 'w', encoding='utf-8') as f:
            json.dump(avg_metrics, f, indent=4, ensure_ascii=False)
