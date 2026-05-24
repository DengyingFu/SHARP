import os
import sys
import json
import re
import cv2
import yaml
import time
import base64
import json5
import string
import torch
from io import BytesIO
from typing import Any, Dict, List, Optional
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from tqdm import tqdm
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
# We are in evaluate/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports from project
from smolVLM.client import SmolVLMClient
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

# --- Tool Definition for Agent ---
# Using a global instance as in the original agent_server style, but properly scoped if possible.
smol_client_instance = SmolVLMClient()

@register_tool('smolvlm') 
class MyImageGenFullSFT(BaseTool):
    description = 'Identify object names from feature/position cues'
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': 'An English question to resolve an object by feature or position, e.g., What is the object on the left? or Which is the red object?',
        'required': True
    }]
    current_image = None
    _model = None
    _processor = None
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def _load_model(cls):
        if cls._model is None:
            print("Loading SmolVLM model locally...")
            model_path = "/data2/fdy/smolVLM/finetunes_vlm/output/Mytasks_full_real/checkpoint-150"
            processor_path = "/data2/fdy/smolVLM/model_256M"
            
            cls._processor = AutoProcessor.from_pretrained(processor_path)
            cls._model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                _attn_implementation="flash_attention_2" if cls._device == "cuda" else "eager",
            ).to(cls._device)
            cls._model.eval()
            print("SmolVLM model loaded.")

    def call(self, params: str, **kwargs) -> str:
        try:
            params_dict = json5.loads(params)
            if isinstance(params_dict, str):
                params_dict = json5.loads(params_dict)
        except:
            params_dict = params if isinstance(params, dict) else {}
            
        prompt = params_dict.get('prompt', '')
        image_input = MyImageGenFullSFT.current_image
        print(f"Tool Call: {prompt}")
        
        MyImageGenFullSFT._load_model()
        
        try:
            if image_input is None:
                raise ValueError("No image provided to SmolVLM tool")

            if isinstance(image_input, Image.Image):
                image = image_input
            elif isinstance(image_input, (bytes, bytearray)):
                image = Image.open(BytesIO(image_input)).convert("RGB")
            else:
                image = load_image(image_input)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                },
            ]
            
            processor = MyImageGenFullSFT._processor
            model = MyImageGenFullSFT._model
            device = MyImageGenFullSFT._device

            prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt_text, images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            generated_ids = model.generate(**inputs, max_new_tokens=500)
            
            # Retrieve only new tokens
            input_len = inputs["input_ids"].shape[1]
            generated_ids_new = generated_ids[:, input_len:]
            response = processor.batch_decode(generated_ids_new, skip_special_tokens=True)[0]
            
        except Exception as e:
            print(f"SmolVLM inference error: {e}")
            response = "Error during inference."

        return json5.dumps({'response': response}, ensure_ascii=False)


class LanguageProcessorFullSFT:
    """
    Handles the language processing pipeline for Full SFT Qwen3.
    Replaces separate stage models with a single full SFT model.
    """
    def __init__(self, config_path='./configs/models.yaml', model_name=None):
        self.intermediate_results = {}
        
        # Configure the Agent with local Transformers model
        self.llm_cfg = {
            'model_type': 'transformers',
            'model': '/data/ssd/fdy/saves/qwen3-0.6B-agent/full/stage1_2_3split_0123',
            'api_key': 'EMPTY',
            'device': 'cuda',
            'generate_cfg': {
                "top_p": 0.8,
                "top_k": 20,
                "temperature": 0.7,
                "repetition_penalty": 1.0
            }
        }
        self.system_instruction = "You are the reasoning module of a robotic arm. You receive: 1) a user instruction; 2) an unordered list of available object names. Your task is to decide which objects from the list are needed."
        
        # Create the agent
        self.bot = Assistant(
            llm=self.llm_cfg,
            system_message=self.system_instruction,
            function_list=['smolvlm'] 
        )

    def _call_llm(self, messages=None, prompt=None):
        """Internal helper to call the reused LLM model."""
        if messages is None:
            if prompt is None:
                return {}
            messages = [{"role": "user", "content": prompt}]
        
        # 尝试直接使用 bot 内部的底层 LLM 对象
        # 这样做可以绕过 Assistant 类默认的 system_message 和 function_list
        # 从而实现同一个模型加载多用途（带工具/不带工具，不同系统提示词）
        if hasattr(self.bot, 'llm'):
            try:
                # 显式传入 functions=None 以禁用工具
                response_generator = self.bot.llm.chat(messages=messages, functions=None)
                
                final_content = ""
                # qwen-agent 的 chat 接口通常返回一个生成器，生成完整回复或增量
                # 我们遍历以获取最终结果
                for response in response_generator:
                    if isinstance(response, list) and len(response) > 0:
                        # 部分实现返回 list of messages
                        final_content = response[-1].get('content', '')
                    elif isinstance(response, dict):
                        final_content = response.get('content', '')
                    else:
                        final_content = str(response)
                
                if not final_content:
                    # 如果生成器为空或未产生内容
                    return {}

                thinking = ""
                thinking_match = re.search(r'<think>(.*?)</think>', final_content, re.DOTALL)
                if thinking_match:
                    thinking = thinking_match.group(1).strip()

                content = final_content
                content_match = re.search(r'\{[\s\S]*\}$', final_content, re.DOTALL)
                if content_match:
                    content = content_match.group(0).strip()
                
                return {'content': content, 'thinking': thinking}

            except Exception as e:
                print(f"Direct LLM usage failed, falling back to bot.run: {e}")

        # Fallback: 如果无法直接访问 llm，只能使用 bot.run
        # 注意：这可能会受到 Agent 默认 system prompt 的影响
        response_content = ""
        try:
            res_list = []
            for res in self.bot.run(messages=messages):
                res_list = res
            
            if res_list:
                response_content = res_list[-1]['content']
        except Exception as e:
            print(f"LLM Call Error: {e}")
            return {}
            
        thinking = ""
        thinking_match = re.search(r'<think>(.*?)</think>', response_content, re.DOTALL)
        if thinking_match:
            thinking = thinking_match.group(1).strip()
            
        return {'content': response_content, 'thinking': thinking}

    def get_user_input(self, question):
        raise RuntimeError("Server mode: user input should be provided by client")

    def extract_object_names(self, scene_graph):
        """从场景图提取物体名称列表"""
        obj_list=[obj["category"] for obj in scene_graph["objects"] if "category" in obj]
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

    def choice_obj_full_sft(self, user_command, image_input):
        MyImageGenFullSFT.current_image = image_input
        messages = [{'role': 'user', 'content': user_command}]
        response = []
        # Run agent
        for res in self.bot.run(messages=messages, response_mode="safe"):
            response = res
        
        if response:
             for msg in response:
                 if msg['role'] == 'user':
                     continue
                 self.conversation_log.append(msg)
             return response[-1]['content']
        return ""

    def extract_target_object(self, user_command, obj_list, image_input):
        """提取用户指令中的目标物体名称"""
        # Call the agent with the Full SFT model
        input = f"Instruction:{user_command}\nObject list:{obj_list}"
        self.conversation_log.append({"role": "user", "content": input})
        last_msg = self.choice_obj_full_sft(input, image_input)
        # self.conversation_log.append({"role": "robot", "content": last_msg})
        # Clean up output (remove thinking tags if present)
        target_obj = re.sub(r'<think>.*?</think>', '', last_msg, flags=re.DOTALL).strip()
        target_obj = target_obj.strip()                                           # 去掉首尾空格/换行
        target_obj = target_obj.replace('\n', ',')
        return target_obj, last_msg

    def decompose_action(self, user_command, target_objs):
        """为多个物体分解动作"""
        print("\n--- 动作分解 ---")
        sys_prompt = "You are the action decomposition module of a robotic arm. The user provides an instruction and a list of objects involved. You need to generate short natural language action descriptions for the objects in the list based on the instruction. Each object corresponds to exactly one pick up action or one place action."
        user_input = f"Instruction:{user_command}\nObject list:{target_objs}"
        messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_input}
            ]
        # Use reused Full SFT model
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
        sys_prompt = "You are a robotic-arm assistant. Given the user instruction and a JSON scene graph, select exactly one object ID. If there is ambiguity, first ask a clarifying question based on object attribute differences, then make the choice."

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
        
        # Use reused Full SFT model
        llm_response = self._call_llm(messages=messages)
        print("LLM输出",llm_response)
        content = llm_response.get('content', '').strip() if llm_response else "{}"
        think = llm_response.get('thinking', '').strip() if llm_response else ""
        self.conversation_log.append({"role": "robot", "content": content, "thinking": think})
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
        
        for obj in scene_graph["objects"]:
            obj_name = obj.get("category", "").lower()
            if target_lower in obj_name or obj_name in target_lower:
                candidate = {
                    "id": obj["id"],
                    "category": obj["category"]
                }
                # Copy fields
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

    def process_command(
        self,
        user_command,
        scene_graph,
        image_input,
        obj_list=None,
        task_type="pick",
        clarification_history: Optional[List[Dict[str, Any]]] = None,
        clarification_answer: Optional[str] = None,
        return_on_ask: bool = True,
    ):
        """处理流程：提取->(动作分解)->匹配->消歧->确认

        task_type: "pick" for grasp-only; "pick_place" to enable action decomposition.
        """
        print("=" * 50)
        print("开始处理用户指令 (Full SFT)")
        print(user_command)
        print("=" * 50)
        self.conversation_log = []
        self.conversation_log.append({"role": "user", "content": user_command})
        self.intermediate_results = {
            "command": user_command,
            "sg_obj_list": [],
            "target_objs": [],
            "candidates": [],
            "resolve_ambiguity_results": [],
            "final_results": [],
            "history": []
        }
        
        # Step 1: 提取场景中的物体列表
        sg_obj_list = self.extract_object_names(scene_graph)
        self.intermediate_results['sg_obj_list'] = sg_obj_list
        print(f"场景物体类别: {obj_list}")
        
        # Step 2: LLM提取目标物体名称 (List)
        print("\n--- Stage 1 提取目标物体 ---")
        target_objs, agent_msg = self.extract_target_object(user_command, obj_list, image_input)
        # 先记录原始输出，之后会用 split 后的列表覆盖
        self.intermediate_results['target_objs'] = target_objs
        
        if not target_objs:
            print("未提取到目标物体")
            return None
        target_objs = target_objs.split(',')
        self.intermediate_results['target_objs'] = target_objs
        print(f"目标物体类别: {target_objs}")
        
        final_results = []
        
        # Step 3: Action Decomposition (only for pick_place)
        actions = {}
        if task_type == "pick_place" and len(target_objs) > 1:
            print("分解动作 (pick_place)")
            actions = self.decompose_action(user_command, target_objs)
            print(f"动作分解结果: {actions}")
        # actions 仍可用于内部处理，但不返回给外部
        
        # Step 4: Process targets
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

                    history = clarification_history if clarification_history else None
                    used_answer = False
                    
                    if history is not None and clarification_answer is not None:
                         history.append({"role": "user", "content": clarification_answer})
                         used_answer = True

                    i = 3
                    while i > 0:
                        i -= 1
                        res, think = self.resolve_ambiguity(current_command, combined_candidates, history)
                        entry = {
                            "target": "multiple",
                            "command": current_command,
                            "assistant_action": res.get("action"),
                            "model_content": res,
                            "model_think": think,
                            "question": res.get("question"),
                            "user_answer": None,
                            "selection": None,
                        }

                        if history is None:
                            sys_prompt = "You are a robotic-arm assistant. Given the user instruction and a JSON scene graph, select exactly one object ID. If there is ambiguity, first ask a clarifying question based on object attribute differences, then make the choice."
                            candidates_json = json.dumps(combined_candidates, ensure_ascii=False)
                            user_input = f"指令：{current_command}\n候选列表：\n{candidates_json}"
                            history = [
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": user_input}
                            ]
                        
                    
                        self.intermediate_results['history'] = history

                        history.append({
                            "role": "assistant",
                            "content": json.dumps(res, ensure_ascii=False),
                            "thinking": think
                        })

                        if res.get("action") == "select":
                            target_id = res.get("target_id")
                            selected_candidate = next((c for c in combined_candidates["objects"] if c["id"] == target_id), None)
                            if selected_candidate:
                                print(f"消歧完成，选择：ID={selected_candidate['id']}")
                                entry["selection"] = {
                                    "target_id": selected_candidate.get("id"),
                                    "category": selected_candidate.get("category")
                                }
                            else:
                                print(f"错误：消歧返回的ID {target_id} 不在候选列表中")
                            self.intermediate_results['resolve_ambiguity_results'].append(entry)
                            break
                        elif res.get("action") == "ask":
                            question = res.get("question")
                            entry["question"] = question
                            if clarification_answer and not used_answer:
                                used_answer = True
                                entry["user_answer"] = clarification_answer
                                self.intermediate_results['resolve_ambiguity_results'].append(entry)
                                history.append({"role": "user", "content": clarification_answer})
                                continue
                            if return_on_ask:
                                self.intermediate_results['resolve_ambiguity_results'].append(entry)
                                return {
                                    "action": "ask",
                                    "question": question,
                                    "history": history,
                                    "partial_results": final_results,
                                }
                            res['user_answer'] = None
                            self.intermediate_results['resolve_ambiguity_results'].append(entry)
                        else:
                            print(f"未知动作: {res}")
                            self.intermediate_results['resolve_ambiguity_results'].append(entry)
                            break

                if selected_candidate:
                    final_results.append(selected_candidate)
        else:
            # Original per-target handling
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
                    
                    history = clarification_history if clarification_history else None
                    used_answer = False
                    
                    if history is not None and clarification_answer is not None:
                         history.append({"role": "user", "content": clarification_answer})
                         used_answer = True

                    i = 3
                    while i > 0:
                        i -= 1
                        res, think = self.resolve_ambiguity(current_command, candidates, history)
                        entry = {
                            "target": target,
                            "command": current_command,
                            "assistant_action": res.get("action"),
                            "model_content": res,
                            "model_think": think,
                            "question": res.get("question"),
                            "user_answer": None,
                            "selection": None,
                        }
                        
                        if history is None:
                            sys_prompt = "You are a robotic-arm assistant. Given the user instruction and a JSON scene graph, select exactly one object ID. If there is ambiguity, first ask a clarifying question based on object attribute differences, then make the choice."
                            candidates_json = json.dumps(candidates, ensure_ascii=False)
                            user_input = f"指令：{current_command}\n候选列表：\n{candidates_json}"
                            history = [
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": user_input}
                            ]
                        
                        self.intermediate_results['history'] = history
                        
                        history.append({
                            "role": "assistant",
                            "content": json.dumps(res, ensure_ascii=False),
                            "thinking": think
                        })

                        if res.get("action") == "select":
                            target_id = res.get("target_id")
                            selected_candidate = next((c for c in candidates["objects"] if c["id"] == target_id), None)
                            if selected_candidate:
                                print(f"消歧完成，选择：ID={selected_candidate['id']}")
                                entry["selection"] = {
                                    "target_id": selected_candidate.get("id"),
                                    "category": selected_candidate.get("category")
                                }
                            else:
                                print(f"错误：消歧返回的ID {target_id} 不在候选列表中")
                            self.intermediate_results['resolve_ambiguity_results'].append(entry)
                            break
                        elif res.get("action") == "ask":
                            question = res.get("question")
                            entry["question"] = question
                            if clarification_answer and not used_answer:
                                used_answer = True
                                entry["user_answer"] = clarification_answer
                                self.intermediate_results['resolve_ambiguity_results'].append(entry)
                                history.append({"role": "user", "content": clarification_answer})
                                continue
                            if return_on_ask:
                                self.intermediate_results['resolve_ambiguity_results'].append(entry)
                                return {
                                    "action": "ask",
                                    "question": question,
                                    "history": history,
                                    "partial_results": final_results,
                                }
                            res['user_answer'] = None
                            self.intermediate_results['resolve_ambiguity_results'].append(entry)
                        else:
                            print(f"未知动作: {res}")
                            self.intermediate_results['resolve_ambiguity_results'].append(entry)
                            break
                
                if selected_candidate:
                    final_results.append(selected_candidate)

        self.intermediate_results['final_results'] = final_results
        return final_results


class ProcessCommandRequest(BaseModel):
    instruction: str = Field(..., description="用户指令")
    scene_graph: Dict[str, Any] = Field(..., description="场景图JSON")
    image_base64: str = Field(..., description="图像base64字符串")
    obj_list: Optional[List[str]] = Field(None, description="候选物体类别列表")
    task_type: str = Field("pick", description="pick 或 pick_place")
    clarification_history: Optional[List[Dict[str, Any]]] = Field(None, description="消歧历史消息")
    clarification_answer: Optional[str] = Field(None, description="消歧回答")


class ProcessCommandResponse(BaseModel):
    status: str
    results: Optional[List[Dict[str, Any]]] = None
    intermediate_results: Optional[Dict[str, Any]] = None


app = FastAPI(
    title="LanguageProcessorFullSFT Service",
    description="API for process_command with local models preloaded",
    version="1.0.0",
)

_language_processor: Optional[LanguageProcessorFullSFT] = None


@app.on_event("startup")
def _startup():
    global _language_processor
    _language_processor = LanguageProcessorFullSFT()
    # 预加载视觉模型
    MyImageGenFullSFT._load_model()
    # 预热语言模型
    try:
        _language_processor._call_llm(prompt="warmup")
    except Exception:
        pass


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "language_loaded": _language_processor is not None,
        "smolvlm_loaded": MyImageGenFullSFT._model is not None,
    }


@app.post("/process_command", response_model=ProcessCommandResponse)
def process_command_api(request: ProcessCommandRequest):
    if _language_processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        b64 = request.image_base64
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        image_bytes = base64.b64decode(b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}")

    result = _language_processor.process_command(
        request.instruction,
        request.scene_graph,
        image,
        obj_list=request.obj_list,
        task_type=request.task_type,
        clarification_history=request.clarification_history,
        clarification_answer=request.clarification_answer,
        return_on_ask=True,
    )

    if isinstance(result, dict) and result.get("action") == "ask":
        return ProcessCommandResponse(
            status="ask",
            results=None,
            intermediate_results=_language_processor.intermediate_results,
        )

    return ProcessCommandResponse(
        status="ok",
        results=result,
        intermediate_results=_language_processor.intermediate_results,
    )

if __name__ == '__main__':
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7412,
        log_level="info"
    )