import argparse
import os
import sys
import json
import re
import cv2
import yaml
import time
import base64
from pathlib import Path
from tqdm import tqdm

# Add project root to path
# We are in evaluate/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports from project
from ambiguity.modules.language_module_image_only import LanguageProcessorFullSFT as BaseLanguageProcessorFullSFT
from ambiguity.modules.perception_module_image_only import ScenePerceptionImageOnly


class LanguageProcessorEnd2End(BaseLanguageProcessorFullSFT):
    """
    Handles the language processing pipeline for End-to-End Qwen3.
    Simplified pipeline: User Instruction + Scene Graph (no bbox) -> Agent -> JSON Action.
    """
    def __init__(self, config_path='./configs/models.yaml', model_name=None):
        super().__init__(config_path=config_path, model_name=model_name)
        self.system_instruction = (
            "You are a robotic arm assistant. Given user instructions and a JSON scene graph, you need to select the ID of the object the user needs. You can selectively ask questions to user or optionally call the vision tool 'smolvlm' to resolve visual cues, finally make selections.Finish with selection(s): {\"action\": \"select\", \"target_id\": <id>}."
        )

    def clean_scene_graph(self, scene_graph):
        """Clean and format scene graph for LLM input."""
        # Handle dict input (expected)
        if not isinstance(scene_graph, dict):
            return scene_graph
            
        new_sg = {}
        
        # transform objects
        # Input keys might be 'Objects' (from some sources) or 'objects'
        objs = scene_graph.get('Objects', scene_graph.get('objects', []))
        new_objs = []
        
        for o in objs:
            new_o = {}
            # Keep only specific fields: id, category, attributes, size
            if 'id' in o:
                new_o['id'] = o['id']
            if 'category' in o:
                new_o['category'] = o['category']
                
            # Handle attributes: ensure list format
            if 'attributes' in o:
                attr = o['attributes']
                if isinstance(attr, str):
                    # If it's a string, try to split by space as they seem to be space-separated adjectives
                    # or just wrap in list if it's a single unit. 
                    # Looking at "black red cylindrical", splitting seems appropriate.
                    # But verifying if valid json attributes are usually single words.
                    new_o['attributes'] = attr.split() if attr else []
                elif isinstance(attr, list):
                    new_o['attributes'] = attr
                else:
                    new_o['attributes'] = []
            
            # Handle size: ensure string format
            if 'size' in o:
                new_o['size'] = str(o['size'])
                
            new_objs.append(new_o)
            
        new_sg['objects'] = new_objs
        
        # transform relationships
        rels = scene_graph.get('relationships', scene_graph.get('spatial_relations', []))
        new_sg['spatial_relations'] = rels
        
        return new_sg

    def process_command(self, user_command, scene_graph, image_path, obj_list=None):
        """End-to-end processing."""
        print("=" * 50)
        print("开始处理用户指令 (End2End)")
        print(user_command)
        print("=" * 50)
        
        self.intermediate_results = {}
        
        # Clean graph
        cleaned_sg = self.clean_scene_graph(scene_graph)
        sg_json = json.dumps(cleaned_sg, ensure_ascii=False)
        # self.intermediate_results['cleaned_sg'] = cleaned_sg
        print(sg_json)
        # Prepare Input
        first_input = f"Instruction: {user_command}\nScene graph:\n{sg_json}"
        
        messages = [{'role': 'user', 'content': first_input}]
        
        final_results = []
        max_turns = 6
        turn_count = 0
        
        while turn_count < max_turns:
            print(f"--- Turn {turn_count + 1} ---")
            llm_response = self._call_qwen_server(
                messages=messages,
                use_tools=True,
                image_path=image_path,
            )

            content = llm_response.get('content', '').strip() if llm_response else ""
            if not content:
                print("Error: Empty response from agent.")
                break

            for msg in llm_response.get('trace', []):
                if isinstance(msg, dict) and msg.get("role") and msg.get("content") is not None:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            if not messages or messages[-1].get("role") != "assistant" or messages[-1].get("content") != content:
                messages.append({'role': 'assistant', 'content': content})
            
            print(f"Model Output: {content}")
            
            # Check for JSON Action: {"action": "select", "target_id": ...}
            select_pattern = r'\{\s*"action":\s*"select",\s*"target_id":\s*(\d+)\s*\}'
            matches = re.findall(select_pattern, content)
            
            if matches:
                print(f"Found Select Actions with IDs: {matches}")
                ids = [int(mid) for mid in matches]
                
                # Retrieve objects from ORIGINAL scene graph based on IDs
                all_objects = scene_graph.get("Objects", [])
                for target_id in ids:
                    obj = next((o for o in all_objects if o.get("id") == target_id), None)
                    if obj:
                        final_results.append(obj)
                    else:
                        print(f"Warning: Model selected ID {target_id} which does not exist in original scene graph.")
                
                # If we found selections, we assume the task is done (even for pick-place with 2 jsons in one output)
                break 
            
            # Check for explicitly JSON "ask" action (legacy support)
            ask_pattern = r'\{\s*"action":\s*"ask",\s*"question":\s*"(.*?)"\s*\}'
            ask_match = re.search(ask_pattern, content, re.DOTALL)
            
            question = None
            if ask_match:
                question = ask_match.group(1)
            else:
                # If no select and no json ask, treat text as question?
                # Need to be careful not to loop on thinking tokens
                clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                # If it ended with tool call, qwen-agent usually appends the tool result and continues.
                # If we are here, the agent PAUSED. Usually meaning it output text and is waiting.
                if clean_content:
                    question = clean_content
            
            if question:
                print(f"Detected Question: {question}")
                user_ans = self.get_user_input(question)
                # Append answer to messages so agent sees it next turn
                messages.append({'role': 'user', 'content': user_ans})
            else:
                print("Model stopped but no selection and no clear question found. Stop.")
                break
                
            turn_count += 1
            
        self.intermediate_results['final_results'] = final_results
        self.intermediate_results['conversation_history'] = messages
        return final_results


class ImageOnlyEvaluatorEnd2End:
    """Evaluation pipeline variant for End-to-End Qwen3."""

    def __init__(self, config_path, model_name=None, use_cached_graph:bool = False):
        self.language = LanguageProcessorEnd2End(config_path=config_path, model_name=model_name)
        self.perception = ScenePerceptionImageOnly(config_path=config_path)
        self.use_cached_graph = use_cached_graph
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

    def judge_stage1_category(self, instruction, online_obj_list, predicted_categories, target_description, target_category=None):
        if not predicted_categories: return False
        
        if target_category:
            # Rule-based judgment using target_category (avoids LLM)
            # Supports string or list of strings
            targets = target_category if isinstance(target_category, list) else [target_category]
            
            # Simple fuzzy match helper
            def is_match(pred, tgt):
                if not pred or not tgt: return False
                p = str(pred).lower()
                t = str(tgt).lower()
                return t in p or p in t
                
            # Check if all targets are covered by at least one prediction
            # This handles both single and multi-object selection reasonably well for Stage 1
            matched_count = 0
            for tgt in targets:
                if any(is_match(pred, tgt) for pred in predicted_categories):
                    matched_count += 1
            
            # If all targets are found in predictions
            return matched_count == len(targets)

        prompt = f"""
        You are evaluating whether a robotic system picked the correct object CATEGORY after parsing a user instruction.
        User Instruction: {instruction}
        Visual Target Description (from benchmark GT): {target_description}
        Detected Object Categories in the scene: {online_obj_list}
        Categories predicted by the system for the target: {predicted_categories}
        Decide if the predicted categories correspond to the user's intended object (synonyms count as correct).
        If the required object category clearly does not exist in the detected list, answer NO.
        Respond with ONLY "YES" or "NO".
        """
        return self._ask_text_judge(prompt)

    def judge_stage2_instance(self, instruction, online_obj_list, conversation_log, selected_obj, target_description, user_answer_hint=None):
        if not selected_obj: return False
        conv_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_log]) if conversation_log else "(no questions asked)"
        prompt = f"""
        A robot asked clarification questions to pick a specific OBJECT INSTANCE.
        User Instruction: {instruction}
        Ground Truth Visual Target: {target_description}
        Detected Object Categories in the scene: {online_obj_list}
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
        print("distance")
        print(distance)
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
        print(f"Starting End-to-End evaluations on {len(dataset_index)} scenes...")

        for scene_data in tqdm(dataset_index):
            scene_dir = str(dataset_root / scene_data['scene_dir'])
            image_path = str(dataset_root / scene_data['image_path'])
            os.makedirs(scene_dir+"/eval_results_baseline1_end2end",exist_ok=True)
            samples_path = dataset_root / scene_data['samples_path']
            with open(samples_path, 'r') as f:
                samples = json.load(f)

            online_scene_graph_path = os.path.join(scene_dir, "online_scene_graph_image_only_sft.json")
            if self.use_cached_graph and os.path.exists(online_scene_graph_path):
                try:
                    with open(online_scene_graph_path, "r", encoding="utf-8") as f:
                        online_scene_graph = json.load(f)
                    print("成功加载场景图")
                except Exception as exc:
                    print(f"Load cached scene graph failed: {exc}, fallback.")
                    online_scene_graph = None

            if not self.use_cached_graph or online_scene_graph is None:
                try:
                    online_scene_graph = self.perception.analyze_scene(
                        image_path=image_path,
                        results_dir=scene_dir,
                        use_oracle=False,
                        output_format="v2",
                    )
                    with open(online_scene_graph_path, "w", encoding="utf-8") as f:
                        json.dump(online_scene_graph, f, indent=2, ensure_ascii=False)
                    print("在线生成场景图")
                except Exception as exc:
                    print(f"Perception failed: {exc}")
                    online_scene_graph = {"Objects": [], "relationships": [], "bbox": []}

            for sample in samples:
                task_type = sample.get('task_type', 'pick')
                target_category = None
                
                if task_type == 'pick_place':
                    pick_gt = sample['ground_truth'].get('pick', {})
                    place_gt = sample['ground_truth'].get('place', {})
                    target_description = f"Pick target: {pick_gt.get('target_description')}. Place target: {place_gt.get('target_description')}"
                    user_answer_hint = f"{pick_gt.get('user_answer_hint', '')} {place_gt.get('user_answer_hint', '')}".strip()
                    
                    # Extract target categories for pick_place
                    t_cat_pick = pick_gt.get('target_category')
                    t_cat_place = place_gt.get('target_category')
                    if t_cat_pick and t_cat_place:
                        target_category = [t_cat_pick, t_cat_place]
                        
                else:
                    target_description = sample['ground_truth']['target_description']
                    user_answer_hint = sample['ground_truth'].get('user_answer_hint', "")
                    target_category = sample['ground_truth'].get('target_category')

                conversation_log = []
                question_asked = False

                def mock_input(question):
                    nonlocal question_asked
                    question_asked = True
                    prompt = f"""
                    You are interacting with a robot that asks a specific choice-based question ("{question}"). Your goal is to instruct the robot to pick up the target object described in "{target_description}". 
                    Your answer must:
                    1. Strictly respond to the robot’s question.
                    2. Answer based on the information provided in "{target_description}".
                    3. If the attribute asked in the question is available in "{target_description}", answer with that attribute.
                    4. If the attribute asked is NOT available in "{target_description}", answer using other available information (e.g. spatial relation).
                    5. Do not invent details not mentioned.
                    6. Be concise, natural, and in English.
                    """
                    try:
                        response = self.llm.chat.completions.create(
                            model=self.llm_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7
                        )
                        ans = response.choices[0].message.content.strip()
                    except:
                        ans = "No more information."
                    conversation_log.append({"role": "assistant", "content": question})
                    conversation_log.append({"role": "user", "content": ans})
                    return ans

                self.language.get_user_input = mock_input 

                start_time = time.time()
                final_results = []
                
                try:
                    online_obj_list = list({
                        obj['category'].lower() for obj in online_scene_graph.get('Objects', []) if 'category' in obj
                    })
                    # Use End2End Process Command
                    final_results = self.language.process_command(
                        sample['instruction'], online_scene_graph, image_path, online_obj_list
                    )
                except Exception as exc:
                    print(f"Inference error: {exc}")
                    final_results = []

                latency = (time.time() - start_time) * 1000
                
                # Evaluation logic for Stage 1 & 2
                # In End-to-End, "Stage 1" (Category) is implicit.
                # We can check if the final selected object's category matches.
                
                selected_obj = final_results[0] if final_results else None
                predicted_cats = [obj.get('category') for obj in final_results] if final_results else []
                
                stage_1_success = self.judge_stage1_category(
                    sample['instruction'], online_obj_list, predicted_cats, target_description, target_category=target_category
                )
                
                # For Stage 2, using final_results as candidates
                candidates_for_judge = final_results 
                
                if final_results:
                    stage_2_success = self.judge_stage2_instance(
                        sample['instruction'], online_obj_list, conversation_log, selected_obj,
                        target_description
                    )
                else:
                    stage_2_success = False

                success = False
                pick_success = False
                place_success = False
                bbox_list = online_scene_graph.get("bbox", [])

                if task_type == 'pick_place':
                    if final_results and len(final_results) >= 2:
                        pick_obj = final_results[0]
                        place_obj = final_results[1]

                        pick_bbox = bbox_list[pick_obj['id'] - 1] if len(bbox_list) >= pick_obj['id'] else None
                        place_bbox = bbox_list[place_obj['id'] - 1] if len(bbox_list) >= place_obj['id'] else None

                        pick_center = pick_gt.get('center') if 'pick_gt' in locals() else None
                        place_center = place_gt.get('center') if 'place_gt' in locals() else None

                        if pick_bbox and pick_center:
                            pick_success = self.judge_by_center(pick_bbox, pick_center)
                            self.save_visualization(image_path, pick_bbox, os.path.join(scene_dir+"/eval_results_baseline1_end2end", f"bbox_{scene_data['scene_id']}_{sample['type']}_e2e_pick.jpg"), gt_center=pick_center)
                        if place_bbox and place_center:
                            place_success = self.judge_by_center(place_bbox, place_center)
                            self.save_visualization(image_path, place_bbox, os.path.join(scene_dir+"/eval_results_baseline1_end2end", f"bbox_{scene_data['scene_id']}_{sample['type']}_e2e_place.jpg"), gt_center=place_center)

                        success = pick_success and place_success
                        if not success:
                            print(f"  [Result] Pick: {'Correct' if pick_success else 'Fail'}, Place: {'Correct' if place_success else 'Fail'}")
                    else:
                        print(f"  [Result] Failed: Expected 2 objects for pick-place, got {len(final_results) if final_results else 0}")

                elif final_results and len(final_results) == 1:
                    selected_obj_id = final_results[0]['id']
                    bbox = bbox_list[selected_obj_id - 1] if len(bbox_list) >= selected_obj_id else None
                    if bbox:
                        output_path = os.path.join(scene_dir+"/eval_results_baseline1_end2end", f"bbox_{scene_data['scene_id']}_{sample['type']}_e2e.jpg")
                        gt_center = sample["ground_truth"].get('center')
                        if gt_center:
                            print("===Judge by Center====")
                            success = self.judge_by_center(bbox, gt_center)
                            self.save_visualization(image_path, bbox, output_path, gt_center=gt_center)
                        else:
                            print("===Judge by VLM====")
                            conversation_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_log])
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
                    "type": sample['type'],
                    "intermediate_results": self.language.intermediate_results
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
            "pick_place_metrics": {"count": 0, "pick_success_rate": 0, "place_success_rate": 0, "success_rate": 0},
            "type_breakdown": {}
        }
        type_stats = {t: {"total": 0, "correct": 0, "s1_correct": 0, "s2_correct": 0, "asked": 0, "latency": 0} for t in [1, 2, 3, 4,5]}
        pp_stats = {"total": 0, "pick_ok": 0, "place_ok": 0, "full_ok": 0}

        overall_correct = 0
        total_latency = 0
        cat_subset_total = cat_subset_s1 = cat_subset_final = 0
        inst_subset_total = inst_subset_asked = inst_subset_final = 0
        no_inst_total = no_inst_asked = 0

        for res in results:
            raw_type = res.get('type', 0)
            t = 0
            if isinstance(raw_type, int):
                t = raw_type
            elif isinstance(raw_type, str):
                digits = ''.join([c for c in raw_type if c.isdigit()])
                t = int(digits) if digits else 0
            
            if t == 0 and raw_type == 'pick_place': t = 4
            elif t == 0 and raw_type == 'pick': t = 1

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
            
            # Pick Place Check
            if t == 4:
                pp_stats["total"] += 1
                if res.get("pick_success", False): pp_stats["pick_ok"] += 1
                if res.get("place_success", False): pp_stats["place_ok"] += 1
                if is_correct: pp_stats["full_ok"] += 1

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
        
        if pp_stats["total"] > 0:
            metrics['pick_place_metrics'] = {
                "count": pp_stats["total"],
                "pick_success_rate": pp_stats["pick_ok"] / pp_stats["total"],
                "place_success_rate": pp_stats["place_ok"] / pp_stats["total"],
                "success_rate": pp_stats["full_ok"] / pp_stats["total"]
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
        print("HADR Evaluation Metrics (End-to-End)")
        print("=" * 30)
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"Overall Success Rate: {metrics['overall_sr']:.2%}")
        print("-" * 20)
        os.makedirs(output_dir, exist_ok=True)
        metrics_name = "eval_metrics_baseline1_end2end.json" if not run_tag else f"eval_metrics_baseline1_end2end_{run_tag}.json"
        details_name = "eval_details_baseline1_end2end.json" if not run_tag else f"eval_details_baseline1_end2end_{run_tag}.json"
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
    parser = argparse.ArgumentParser(description="Evaluate image-only End2End benchmark.")
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

    output_dir = benchmark_path / "benchmark_runs" / "end2end"
    use_cached = os.getenv("USE_CACHED_SCENE_GRAPH", "0") == "1"
    config_path = Path(project_root) / "ambiguity" / "configs" / "models.yaml"

    evaluator = ImageOnlyEvaluatorEnd2End(
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
        print(f"Overall Success Rate: {avg_metrics.get('overall_sr', 0):.2%}")
        print("-" * 20)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir / "eval_metrics_baseline1_end2end_avg.json", 'w', encoding='utf-8') as f:
            json.dump(avg_metrics, f, indent=4, ensure_ascii=False)
