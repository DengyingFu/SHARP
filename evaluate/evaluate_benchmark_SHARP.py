import os
import sys
import json
import argparse
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

PROMPT_DIR = Path(project_root) / "prompts" / "repro_release"


def _load_prompt(prompt_name: str) -> str:
    prompt_path = PROMPT_DIR / prompt_name
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _format_prompt(template: str, **kwargs) -> str:
    safe_values = {k: ("" if v is None else str(v)) for k, v in kwargs.items()}
    return template.format(**safe_values)

# Imports from project
from ambiguity.modules.perception_module_image_only import ScenePerceptionImageOnly
from ambiguity.modules.language_module_image_only import LanguageProcessorFullSFT


class ImageOnlyEvaluatorFullSFT:
    """Evaluation pipeline variant that uses RGB images and Full SFT Qwen3."""

    def __init__(self, config_path, model_name=None, use_cached_graph:bool = False, eval_metric='center'):
        # Initialize the new Language Processor
        self.language = LanguageProcessorFullSFT(config_path=config_path, model_name=model_name)
        # Perception remains the same
        self.perception = ScenePerceptionImageOnly(config_path=config_path)
        self.use_cached_graph = use_cached_graph
        self.eval_metric = eval_metric
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.api_key = os.getenv("DASHSCOPE_API_KEY", config['openai']['api_key'])
        self.base_url = config['openai']['base_url']
        # Judge configs
        self.llm_model = "qwen3-235b-a22b-instruct-2507" 
        self.vl_model = config['openai'].get('qwen_vl_model', 'qwen-vl-max')
        self.prompt_templates = {
            "judge_stage1_category": _load_prompt("judge_stage1_category.txt"),
            "judge_stage2_instance": _load_prompt("judge_stage2_instance.txt"),
            "vlm_judge": _load_prompt("vlm_judge.txt"),
            "mock_user_answer": _load_prompt("mock_user_answer.txt"),
        }

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
        prompt = _format_prompt(
            self.prompt_templates["judge_stage1_category"],
            instruction=instruction,
            target_description=target_description,
            online_obj_list=online_obj_list,
            predicted_categories=predicted_categories,
        )
        return self._ask_text_judge(prompt)

    def judge_stage2_instance(self, instruction, candidates, conversation_log, selected_obj, target_description, user_answer_hint=None):
        if not selected_obj: return False
        conv_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation_log]) if conversation_log else "(no questions asked)"
        prompt = _format_prompt(
            self.prompt_templates["judge_stage2_instance"],
            instruction=instruction,
            target_description=target_description,
            candidates=candidates,
            conv_text=conv_text,
            selected_id=selected_obj.get('id'),
            selected_category=selected_obj.get('category'),
        )
        return self._ask_text_judge(prompt)

    def judge_by_center(self, selected_bbox, gt_center, tolerance=30):
        if not selected_bbox or not gt_center: return False
        x1, y1, x2, y2 = selected_bbox
        pred_cx, pred_cy = (x1 + x2) / 2, (y1 + y2) / 2
        gt_cx, gt_cy = gt_center
        distance = ((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)**0.5
        return distance < tolerance

    def judge_by_iou(self, selected_bbox, gt_bbox, iou_threshold=0.5):
        if not selected_bbox or not gt_bbox: return False
        
        pred_x1, pred_y1, pred_x2, pred_y2 = selected_bbox
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox

        x_left = max(pred_x1, gt_x1)
        y_top = max(pred_y1, gt_y1)
        x_right = min(pred_x2, gt_x2)
        y_bottom = min(pred_y2, gt_y2)

        if x_right < x_left or y_bottom < y_top:
            return False

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        bb2_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)

        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        return iou >= iou_threshold

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
        prompt = _format_prompt(
            self.prompt_templates["vlm_judge"],
            instruction=instruction,
            target_description=target_description,
            selection_text=selection_text or '(not provided)',
            conversation_text=conversation_text or '(no dialogue)',
        )
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

    def run_evaluation(self, dataset_path,output_dir, run_tag=None):
        dataset_root = Path(dataset_path)
        with open(dataset_root / "dataset_index.json", 'r', encoding='utf-8') as f:
            dataset_index = json.load(f)

        all_results = []
        print(f"Starting evaluations on {len(dataset_index)} scenes with Full SFT model...")

        for scene_data in tqdm(dataset_index):
            scene_dir = dataset_root / scene_data['scene_dir']
            save_img_dir = scene_dir / "eval_results_Full"
            os.makedirs(save_img_dir, exist_ok=True)
            image_path = dataset_root / scene_data['image_path']
            samples_path = dataset_root / scene_data['samples_path']
            with open(samples_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)

            # 选择使用缓存场景图或在线感知
            online_scene_graph_path = scene_dir / "online_scene_graph.json"
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
                        image_path=str(image_path),
                        results_dir=str(scene_dir),
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
                    prompt = _format_prompt(
                        self.prompt_templates["mock_user_answer"],
                        question=question,
                        target_description=target_description,
                    )
                    try:
                        response = self.llm.chat.completions.create(
                            model=self.llm_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.1
                        )
                        ans = response.choices[0].message.content.strip()
                    except Exception:
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
                        sample['instruction'], online_scene_graph, str(image_path), online_obj_list, task_type=task_type
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
                        pick_gt_bbox = pick_gt.get('bbox')
                        place_gt_bbox = place_gt.get('bbox')

                        if pick_bbox:
                            if self.eval_metric == 'iou' and pick_gt_bbox:
                                pick_success = self.judge_by_iou(pick_bbox, pick_gt_bbox)
                                self.save_visualization(str(image_path), pick_bbox, os.path.join(str(save_img_dir), f"bbox_{scene_data['scene_id']}_{sample['type']}_sft_pick.jpg"), gt_center=None)
                            elif pick_center:
                                pick_success = self.judge_by_center(pick_bbox, pick_center)
                                self.save_visualization(str(image_path), pick_bbox, os.path.join(str(save_img_dir), f"bbox_{scene_data['scene_id']}_{sample['type']}_sft_pick.jpg"), gt_center=pick_center)

                        if place_bbox:
                            if self.eval_metric == 'iou' and place_gt_bbox:
                                place_success = self.judge_by_iou(place_bbox, place_gt_bbox)
                                self.save_visualization(str(image_path), place_bbox, os.path.join(str(save_img_dir), f"bbox_{scene_data['scene_id']}_{sample['type']}_sft_place.jpg"), gt_center=None)
                            elif place_center:
                                place_success = self.judge_by_center(place_bbox, place_center)
                                self.save_visualization(str(image_path), place_bbox, os.path.join(str(save_img_dir), f"bbox_{scene_data['scene_id']}_{sample['type']}_sft_place.jpg"), gt_center=place_center)

                        success = pick_success and place_success
                        if not success:
                            print(f"  [Result] Pick: {'Correct' if pick_success else 'Fail'}, Place: {'Correct' if place_success else 'Fail'}")
                    else:
                        print(f"  [Result] Failed: Expected 2 objects for pick-place, got {len(final_results) if final_results else 0}")

                elif final_results and len(final_results) == 1:
                    selected_obj_id = final_results[0]['id']
                    bbox = bbox_list[selected_obj_id - 1] if len(bbox_list) >= selected_obj_id else None
                    if bbox:
                        output_path = os.path.join(str(save_img_dir), f"bbox_{scene_data['scene_id']}_{sample['type']}_sft.jpg")
                        gt_center = sample["ground_truth"].get('center')
                        gt_bbox = sample["ground_truth"].get('bbox')
                        
                        if self.eval_metric == 'iou' and gt_bbox:
                            print("===Judge by IOU====")
                            success = self.judge_by_iou(bbox, gt_bbox)
                            self.save_visualization(str(image_path), bbox, output_path, gt_center=None)
                        elif gt_center:
                            print("===Judge by Center====")
                            success = self.judge_by_center(bbox, gt_center)
                            self.save_visualization(str(image_path), bbox, output_path, gt_center=gt_center)
                        else:
                            print("===Judge by VLM====")
                            conversation_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.language.conversation_log])
                            selection_text = json.dumps(final_results, ensure_ascii=False)
                            success = self.vlm_judge(str(image_path), sample['instruction'], bbox, target_description, conversation_text, selection_text, output_path)

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
        print("HADR Evaluation Metrics (Full SFT / Image-Only)")
        print("=" * 30)
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"Overall Success Rate: {metrics['overall_sr']:.2%}")
        print("-" * 20)
        os.makedirs(output_dir, exist_ok=True)
        metrics_name = "eval_metrics_image_only_sft.json" if not run_tag else f"eval_metrics_image_only_sft_{run_tag}.json"
        details_name = "eval_details_image_only_sft.json" if not run_tag else f"eval_details_image_only_sft_{run_tag}.json"
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
    parser = argparse.ArgumentParser(description="Evaluate image-only Full SFT benchmark.")
    parser.add_argument(
        "benchmark_folder",
        type=str,
        help="Benchmark folder name under data/, e.g. benchmark-200",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="How many repeated runs to execute (default: 3).",
    )
    parser.add_argument(
        "--eval-metric",
        type=str,
        default="iou",
        choices=["iou", "center"],
        help="Evaluation metric for bbox matching.",
    )
    args = parser.parse_args()

    data_root = Path(project_root) / "data"
    benchmark_path = data_root / args.benchmark_folder
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark folder not found: {benchmark_path}")

    output_dir = benchmark_path / "benchmark_runs" / "full-test"
    use_cached = os.getenv("USE_CACHED_SCENE_GRAPH", "0") == "1"
    config_path = Path(project_root) / "ambiguity" / "configs" / "models.yaml"

    evaluator = ImageOnlyEvaluatorFullSFT(
        config_path=str(config_path),
        model_name=None,
        use_cached_graph=use_cached,
        eval_metric=args.eval_metric,
    )

    run_metrics = []
    for i in range(args.runs):
        print("\n" + "=" * 40)
        print(f"Running benchmark: {i + 1}/{args.runs}")
        print("=" * 40)
        metrics = evaluator.run_evaluation(str(benchmark_path), str(output_dir), run_tag=f"run{i + 1}")
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
        with open(output_dir / "eval_metrics_image_only_sft_avg.json", 'w', encoding='utf-8') as f:
            json.dump(avg_metrics, f, indent=4, ensure_ascii=False)
