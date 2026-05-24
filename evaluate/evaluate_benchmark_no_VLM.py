import argparse
import os
import sys
import json
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


class LanguageProcessorFullSFT(BaseLanguageProcessorFullSFT):
    """复用全量 SFT 语言模块，仅保留本脚本原有流程。"""

    def process_command(self, user_command, scene_graph, image_path, obj_list=None):
        """处理流程：提取->(动作分解)->匹配->消歧->确认"""
        print("=" * 50)
        print("开始处理用户指令 (Full SFT)")
        print(user_command)
        print("=" * 50)
        self.conversation_log = []
        self.conversation_log.append({"role": "user", "content": user_command})
        self.intermediate_results = {}
        
        # Step 1: 提取场景中的物体列表
        sg_obj_list = self.extract_object_names(scene_graph)
        self.intermediate_results['sg_obj_list'] = sg_obj_list
        print(f"场景物体类别: {obj_list}")
        
        # Step 2: LLM提取目标物体名称 (List)
        print("\n--- Stage 1 提取目标物体 ---")
        target_objs, agent_msg = self.extract_target_object(user_command, obj_list, image_path)
        self.intermediate_results['target_objs'] = target_objs
        self.intermediate_results['agent_msg'] = agent_msg
        
        if not target_objs:
            print("未提取到目标物体")
            return None
        target_objs = target_objs.split(',')
        self.intermediate_results['target_objs_split'] = target_objs
        print(f"目标物体类别: {target_objs}")
        
        final_results = []
        
        # Step 3: Action Decomposition
        actions = {}
        if len(target_objs) > 1:
             print("分解动作")
             actions = self.decompose_action(user_command, target_objs)
             print(f"动作分解结果: {actions}")
        self.intermediate_results['actions'] = actions
        
        # Step 4: Process each target
        for target in target_objs:
            print(f"\n处理目标物体: {target}")
            candidates = self.match_objects_by_name(target, scene_graph, use_graph=True)
            print("匹配结果", candidates["objects"])
            
            self.intermediate_results.setdefault('candidates', []).append(candidates)
            
            if len(candidates["objects"]) == 0:
                print(f"错误：未找到匹配的物体 {target}")
                continue
            
            selected_candidate = None
            
            if len(candidates['objects']) == 1:
                print(f"唯一匹配：ID={candidates['objects'][0]['id']}")
                selected_candidate = candidates['objects'][0]
            else:
                print(f"发现 {len(candidates['objects'])} 个候选物体，进入消歧流程")
                current_command = actions.get(target, user_command) if len(target_objs) > 1 else user_command
                print(f"用于消歧的指令: {current_command}")
                
                history = None
                i = 3
                while i > 0:
                    i -= 1
                    res, think = self.resolve_ambiguity(current_command, candidates, history)
                    self.intermediate_results.setdefault('resolve_ambiguity_results', []).append(res)
                    self.intermediate_results.setdefault('resolve_ambiguity_results', []).append(think)
                    
                    if history is None:
                        sys_prompt = "You are a robotic-arm assistant. Given the user instruction and a JSON scene graph, select exactly one object ID. If there is ambiguity, first ask a clarifying question based on object attribute differences, then make the choice."
                        candidates_json = json.dumps(candidates, ensure_ascii=False)
                        user_input = f"指令：{current_command}\n候选列表：\n{candidates_json}"
                        history = [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_input}
                        ]
                    
                    history.append({"role": "assistant", "content": json.dumps(res, ensure_ascii=False)})

                    if res.get("action") == "select":
                        target_id = res.get("target_id")
                        selected_candidate = next((c for c in candidates["objects"] if c["id"] == target_id), None)
                        if selected_candidate:
                            print(f"消歧完成，选择：ID={selected_candidate['id']}")
                        else:
                            print(f"错误：消歧返回的ID {target_id} 不在候选列表中")
                        break
                    elif res.get("action") == "ask":
                        question = res.get("question")
                        # Use self.get_user_input (evaluated acts as mock_input)
                        ans = self.get_user_input(question)
                        res['user_answer'] = ans
                        history.append({"role": "user", "content": ans})
                    else:
                        print(f"未知动作: {res}")
                        break
            
            if selected_candidate:
                final_results.append(selected_candidate)

        self.intermediate_results['final_results'] = final_results
        return final_results


class ImageOnlyEvaluatorFullSFT:
    """Evaluation pipeline variant that uses RGB images and Full SFT Qwen3."""

    def __init__(self, config_path, model_name=None, use_cached_graph:bool = False):
        # Initialize the new Language Processor
        self.language = LanguageProcessorFullSFT(config_path=config_path, model_name=model_name)
        # Perception remains the same
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

    def judge_stage2_instance(self, instruction, candidates, conversation_log, selected_obj, target_description, user_answer_hint=None):
        if not selected_obj: return False
        conv_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_log]) if conversation_log else "(no questions asked)"
        prompt = f"""
        A robot asked clarification questions to pick a specific OBJECT INSTANCE.
        User Instruction: {instruction}
        Ground Truth Visual Target: {target_description}
        Candidate objects (id, category, attributes if present): {candidates}
        Dialogue History:\n{conv_text}
        Final selection by the robot: id={selected_obj.get('id')}, category={selected_obj.get('category')}
        Judge if the final selection satisfies the user's answers and the intent. Reply ONLY with "YES" or "NO".
        """
        return self._ask_text_judge(prompt)

    def judge_by_center(self, selected_bbox, gt_center, tolerance=20):
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
        print(f"Starting evaluations on {len(dataset_index)} scenes with Full SFT model...")

        for scene_data in tqdm(dataset_index):
            scene_dir = str(dataset_root / scene_data['scene_dir'])
            save_img_dir = scene_dir+"/eval_results_NoVLM"
            os.makedirs(save_img_dir,exist_ok=True)
            image_path = str(dataset_root / scene_data['image_path'])
            samples_path = dataset_root / scene_data['samples_path']
            with open(samples_path, 'r') as f:
                samples = json.load(f)

            # 选择使用缓存场景图或在线感知
            online_scene_graph_path = os.path.join(scene_dir, "online_scene_graph.json")
            if self.use_cached_graph and os.path.exists(online_scene_graph_path):
                try:
                    with open(online_scene_graph_path, "r", encoding="utf-8") as f:
                        online_scene_graph = json.load(f)
                    print("成功加载场景图")
                except Exception as exc:
                    print(f"Load cached scene graph failed for scene {scene_data['scene_id']}: {exc}, fallback to perception.")
                    online_scene_graph = None

            else:
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
                    print(f"Perception failed for scene {scene_data['scene_id']}: {exc}")
                    online_scene_graph = {"Objects": [], "relationships": [], "bbox": []}

            for sample in samples:
                task_type = sample.get('task_type', 'pick')
                if task_type == 'pick_place':
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
                    online_obj_list = list({
                        obj['category'].lower() for obj in online_scene_graph.get('Objects', []) if 'category' in obj
                    })
                    # Run Full SFT Pipeline
                    final_results = self.language.process_command(
                        sample['instruction'], online_scene_graph, image_path, online_obj_list
                    )
                    predicted_cats = self.language.intermediate_results.get('target_objs_split', [])
                    # stage_1_success = self.judge_stage1_category(
                    #     sample['instruction'], online_obj_list, predicted_cats, target_description
                    # )
                except Exception as exc:
                    print(f"Inference error: {exc}")
                    final_results = []

                latency = (time.time() - start_time) * 1000
                
                candidates_all = self.language.intermediate_results.get('candidates', []) if hasattr(self.language, 'intermediate_results') else []
                primary_candidates = candidates_all[0] if candidates_all else []
                selected_obj = final_results[0] if final_results else None
                
                # if final_results:
                #     stage_2_success = self.judge_stage2_instance(
                #         sample['instruction'], primary_candidates, self.language.conversation_log, selected_obj,
                #         target_description, user_answer_hint
                #     )
                # else:
                #     stage_2_success = False

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
                            self.save_visualization(image_path, pick_bbox, os.path.join(save_img_dir, f"bbox_{scene_data['scene_id']}_{sample['type']}_sft_pick.jpg"), gt_center=pick_center)
                        if place_bbox and place_center:
                            place_success = self.judge_by_center(place_bbox, place_center)
                            self.save_visualization(image_path, place_bbox, os.path.join(save_img_dir, f"bbox_{scene_data['scene_id']}_{sample['type']}_sft_place.jpg"), gt_center=place_center)

                        success = pick_success and place_success
                        if not success:
                            print(f"  [Result] Pick: {'Correct' if pick_success else 'Fail'}, Place: {'Correct' if place_success else 'Fail'}")
                    else:
                        print(f"  [Result] Failed: Expected 2 objects for pick-place, got {len(final_results) if final_results else 0}")

                elif final_results and len(final_results) == 1:
                    selected_obj_id = final_results[0]['id']
                    bbox = bbox_list[selected_obj_id - 1] if len(bbox_list) >= selected_obj_id else None
                    if bbox:
                        output_path = os.path.join(save_img_dir, f"bbox_{scene_data['scene_id']}_{sample['type']}_sft.jpg")
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
        print("HADR Evaluation Metrics (Full SFT / Image-Only / No VLM)")
        print("=" * 30)
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"Overall Success Rate: {metrics['overall_sr']:.2%}")
        print("-" * 20)
        os.makedirs(output_dir, exist_ok=True)
        metrics_name = "eval_metrics_image_only_no_VLM.json" if not run_tag else f"eval_metrics_image_only_no_VLM_{run_tag}.json"
        details_name = "eval_details_image_only_no_VLM.json" if not run_tag else f"eval_details_image_only_no_VLM_{run_tag}.json"
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
    parser = argparse.ArgumentParser(description="Evaluate image-only Full SFT benchmark (No VLM).")
    parser.add_argument(
        "benchmark_folder",
        type=str,
        help="Benchmark folder name under data/, e.g. benchmark-200",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="How many repeated runs to execute (default: 3).",
    )
    args = parser.parse_args()

    data_root = Path(project_root) / "data"
    benchmark_path = data_root / args.benchmark_folder
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark folder not found: {benchmark_path}")

    output_dir = benchmark_path / "benchmark_runs" / "no_vlm"
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
        print(f"Overall Success Rate: {avg_metrics.get('overall_sr', 0):.2%}")
        print("-" * 20)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir / "eval_metrics_image_only_no_VLM_avg.json", 'w', encoding='utf-8') as f:
            json.dump(avg_metrics, f, indent=4, ensure_ascii=False)
