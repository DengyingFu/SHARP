import json
import os
import re
import string
from urllib import error as urllib_error
from urllib import request as urllib_request
from pathlib import Path

from ambiguity.utils.api_wrappers import OpenAIWrapper


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = PROJECT_ROOT / "prompts" / "repro_release"


def _load_prompt(prompt_name: str) -> str:
    with open(PROMPT_DIR / prompt_name, "r", encoding="utf-8") as f:
        return f.read().strip()


class LanguageProcessorFullSFT:
    """
    Handles the language processing pipeline for Full SFT Qwen3.
    Replaces separate stage models with a single full SFT model.
    """

    def __init__(self, config_path='./configs/models.yaml', model_name=None):
        self.llm_wrapper = OpenAIWrapper(config_path)
        self.intermediate_results = {}
        self.prompt_templates = {
            "language_system_instruction": _load_prompt("language_system_instruction.txt"),
            "action_decomposition_system": _load_prompt("action_decomposition_system.txt"),
            "resolve_ambiguity_system": _load_prompt("resolve_ambiguity_system.txt"),
        }

        self.model_name = model_name
        self.qwen_server_url = os.getenv("QWEN_SERVER_URL", "http://127.0.0.1:1235")
        self.qwen_server_timeout = float(os.getenv("QWEN_SERVER_TIMEOUT", "120"))

        self.system_instruction = self.prompt_templates["language_system_instruction"]

    def _call_qwen_server(self, messages=None, prompt=None, use_tools=False, image_path=None):
        if messages is None:
            if prompt is None:
                return {}
            messages = [{"role": "user", "content": prompt}]

        payload = {
            "messages": messages,
            "system_prompt": self.system_instruction,
            "use_tools": bool(use_tools),
            "image_path": image_path,
        }

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib_request.Request(
            f"{self.qwen_server_url.rstrip('/')}/infer",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib_request.urlopen(req, timeout=self.qwen_server_timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
        except (urllib_error.URLError, TimeoutError) as exc:
            print(f"Qwen server request failed: {exc}")
            return {}
        except Exception as exc:
            print(f"Qwen server response parse failed: {exc}")
            return {}

        if resp_data.get("status") != "success":
            print(f"Qwen server infer failed: {resp_data.get('error', 'unknown error')}")
            return {}

        return {
            "content": resp_data.get("result", ""),
            "thinking": resp_data.get("thinking", ""),
            "trace": resp_data.get("trace", []) or [],
        }

    def _call_llm(self, messages=None, prompt=None):
        """统一通过远端 Qwen API 调用（不在本进程重复加载模型）。"""
        return self._call_qwen_server(messages=messages, prompt=prompt, use_tools=False)

    def get_user_input(self, question):
        print(f"\n机器人提问: {question}")
        return input("用户回答: ")

    def extract_object_names(self, scene_graph):
        """从场景图提取物体名称列表"""
        obj_list = [obj["category"] for obj in scene_graph["Objects"] if "category" in obj]
        punctuation = string.punctuation
        object_nouns = []
        seen = set()
        for item in obj_list:
            clean_item = item.strip()
            if not clean_item:
                continue
            while clean_item and clean_item[-1] in punctuation:
                clean_item = clean_item[:-1]
            if not clean_item:
                continue
            normed_noun = clean_item.lower()
            if normed_noun not in seen:
                seen.add(normed_noun)
                object_nouns.append(clean_item)
        return object_nouns

    def choice_obj_full_sft(self, user_command, img_path):
        messages = [{'role': 'user', 'content': user_command}]
        llm_response = self._call_qwen_server(
            messages=messages,
            use_tools=True,
            image_path=img_path,
        )
        if llm_response:
            for msg in llm_response.get('trace', []):
                self.conversation_log.append(msg)
            return llm_response.get('content', '')
        return ""

    def extract_target_object(self, user_command, obj_list, image_path):
        """提取用户指令中的目标物体名称"""
        model_input = f"Instruction:{user_command}\nObject list:{obj_list}"
        self.conversation_log.append({"role": "user", "content": model_input})
        last_msg = self.choice_obj_full_sft(model_input, image_path)
        target_obj = re.sub(r'<think>.*?</think>', '', last_msg, flags=re.DOTALL).strip()
        target_obj = target_obj.strip()
        target_obj = target_obj.replace('\n', ',')
        return target_obj, last_msg

    def decompose_action(self, user_command, target_objs):
        """为多个物体分解动作"""
        print("\n--- 动作分解 ---")
        sys_prompt = self.prompt_templates["action_decomposition_system"]
        user_input = f"Instruction:{user_command}\nObject list:{target_objs}"
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_input}
        ]
        llm_response = self._call_llm(messages=messages)
        self.conversation_log.append({"role": "robot", "content": llm_response})
        content = llm_response.get('content', '').strip() if llm_response else "{}"

        try:
            actions = json.loads(content)
        except json.JSONDecodeError:
            print("动作分解返回非JSON格式，尝试修复或使用原始指令")
            actions = {obj: user_command for obj in target_objs}

        return actions

    def resolve_ambiguity(self, user_command, candidates, history=None):
        """歧义消解"""
        print("\n--- 歧义消解 ---")
        sys_prompt = self.prompt_templates["resolve_ambiguity_system"]

        if history is None:
            candidates_json = json.dumps(candidates, ensure_ascii=False)
            user_input = f"Instruction: {user_command}\nScene graph:\n{candidates_json}"
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
            print(f"JSON解析失败: {content}")
            result = {}

        return result, think

    def match_objects_by_name(self, target_obj, scene_graph, use_graph=False):
        """Python匹配：根据物体名称在场景图中查找候选ID"""
        print(f"\n--- Python物体匹配: {target_obj} ---")
        candidates = []
        target_lower = target_obj.lower()

        for obj in scene_graph["Objects"]:
            obj_name = obj.get("category", "").lower()
            if target_lower in obj_name or obj_name in target_lower:
                candidate = {
                    "id": obj["id"],
                    "category": obj["category"]
                }
                fields = ["area", "attributes", "size"]
                if not use_graph:
                    fields.append("position")
                for field in fields:
                    if field in obj:
                        candidate[field] = obj[field]
                candidates.append(candidate)

        if use_graph:
            candidate_ids = {c["id"] for c in candidates}
            filtered_relations = []
            added_relations = set()

            def normalize_relation(tokens):
                rel_text = " ".join(tokens).strip()
                if rel_text.endswith(" of"):
                    rel_text = rel_text[:-3].strip()

                canonical_map = {
                    "is left": "left_of",
                    "is left of": "left_of",
                    "is right": "right_of",
                    "is right of": "right_of",
                    "is in front": "in_front_of",
                    "is in front of": "in_front_of",
                    "is behind": "behind_of",
                    "is behind of": "behind_of",
                }
                return canonical_map.get(rel_text, rel_text)

            inverse_map = {
                "left_of": "right_of",
                "right_of": "left_of",
                "in_front_of": "behind_of",
                "behind_of": "in_front_of",
            }

            for rel in scene_graph.get("relationships", []):
                if isinstance(rel, str):
                    parts = rel.split()
                    if len(parts) >= 4:
                        try:
                            id1 = int(parts[0])
                            id2 = int(parts[-1])
                            if id1 in candidate_ids and id2 in candidate_ids:
                                relation_tokens = parts[1:-1]
                                canonical = normalize_relation(relation_tokens)
                                inverse = inverse_map.get(canonical)
                                if inverse and (id2, inverse, id1) in added_relations:
                                    continue
                                if (id1, canonical, id2) not in added_relations:
                                    filtered_relations.append(rel)
                                    added_relations.add((id1, canonical, id2))
                        except (ValueError, IndexError):
                            continue
                elif isinstance(rel, dict):
                    id1 = rel.get("subject_id")
                    id2 = rel.get("object_id")
                    if id1 in candidate_ids and id2 in candidate_ids:
                        filtered_relations.append(rel)

            graph = {
                "objects": candidates,
                "spatial_relations": filtered_relations
            }
        else:
            graph = {
                "objects": candidates
            }
        print(f"匹配到 {len(candidates)} 个候选物体")
        return graph

    def process_command(self, user_command, scene_graph, image_path, obj_list=None, task_type="pick"):
        """处理流程：提取->(动作分解)->匹配->消歧->确认

        task_type: "pick" for grasp-only; "pick_place" to enable action decomposition.
        """
        print("=" * 50)
        print("开始处理用户指令 (Full SFT)")
        print(user_command)
        print("=" * 50)
        self.conversation_log = []
        self.conversation_log.append({"role": "user", "content": user_command})
        self.intermediate_results = {}

        sg_obj_list = self.extract_object_names(scene_graph)
        self.intermediate_results['sg_obj_list'] = sg_obj_list
        print(f"场景物体类别: {obj_list}")

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

        actions = {}
        if task_type == "pick_place" and len(target_objs) > 1:
            print("分解动作 (pick_place)")
            actions = self.decompose_action(user_command, target_objs)
            print(f"动作分解结果: {actions}")
        self.intermediate_results['actions'] = actions

        if task_type == "pick" and len(target_objs) > 1:
            print("\n抓取模式：多个类别同时处理")
            merged_objects = []
            merged_relations = []
            merged_ids = set()
            for target in target_objs:
                print(f"\n处理目标物体: {target}")
                candidates = self.match_objects_by_name(target, scene_graph, use_graph=True)
                print("匹配结果", candidates["objects"])
                self.intermediate_results.setdefault('candidates', []).append(candidates)
                for obj in candidates.get("objects", []):
                    if obj["id"] not in merged_ids:
                        merged_ids.add(obj["id"])
                        merged_objects.append(obj)
                merged_relations.extend(candidates.get("spatial_relations", []))

            combined_candidates = {
                "objects": merged_objects,
                "spatial_relations": merged_relations
            }

            if len(combined_candidates["objects"]) == 0:
                print("错误：未找到匹配的物体")
            else:
                selected_candidate = None
                if len(combined_candidates["objects"]) == 1:
                    print(f"唯一匹配：ID={combined_candidates['objects'][0]['id']}")
                    selected_candidate = combined_candidates['objects'][0]
                    self.conversation_log.append({
                        "role": "system",
                        "content": json.dumps({"action": "select", "target_id": selected_candidate['id'], "category": selected_candidate['category']}, ensure_ascii=False)
                    })
                else:
                    print(f"发现 {len(combined_candidates['objects'])} 个候选物体，进入消歧流程")
                    current_command = user_command
                    print(f"用于消歧的指令: {current_command}")

                    history = None
                    i = 3
                    while i > 0:
                        i -= 1
                        res, think = self.resolve_ambiguity(current_command, combined_candidates, history)
                        self.intermediate_results.setdefault('resolve_ambiguity_results', []).append(res)
                        self.intermediate_results.setdefault('resolve_ambiguity_results', []).append(think)

                        if history is None:
                            sys_prompt = self.prompt_templates["resolve_ambiguity_system"]
                            candidates_json = json.dumps(combined_candidates, ensure_ascii=False)
                            user_input = f"指令：{current_command}\n候选列表：\n{candidates_json}"
                            history = [
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": user_input}
                            ]

                        history.append({"role": "assistant", "content": json.dumps(res, ensure_ascii=False)})

                        if res.get("action") == "select":
                            target_id = res.get("target_id")
                            selected_candidate = next((c for c in combined_candidates["objects"] if c["id"] == target_id), None)
                            if selected_candidate:
                                print(f"消歧完成，选择：ID={selected_candidate['id']}")
                            else:
                                print(f"错误：消歧返回的ID {target_id} 不在候选列表中")
                            break
                        elif res.get("action") == "ask":
                            question = res.get("question")
                            ans = self.get_user_input(question)
                            res['user_answer'] = ans
                            history.append({"role": "user", "content": ans})
                        else:
                            print(f"未知动作: {res}")
                            break

                if selected_candidate:
                    final_results.append(selected_candidate)
        else:
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
                    self.conversation_log.append({
                        "role": "system",
                        "content": json.dumps({"action": "select", "target_id": selected_candidate['id'], "category": selected_candidate['category']}, ensure_ascii=False)
                    })
                else:
                    print(f"发现 {len(candidates['objects'])} 个候选物体，进入消歧流程")
                    current_command = actions.get(target, user_command) if actions else user_command
                    print(f"用于消歧的指令: {current_command}")

                    history = None
                    i = 3
                    while i > 0:
                        i -= 1
                        res, think = self.resolve_ambiguity(current_command, candidates, history)
                        self.intermediate_results.setdefault('resolve_ambiguity_results', []).append(res)
                        self.intermediate_results.setdefault('resolve_ambiguity_results', []).append(think)

                        if history is None:
                            sys_prompt = self.prompt_templates["resolve_ambiguity_system"]
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