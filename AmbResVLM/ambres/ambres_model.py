"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

import json
import warnings
import subprocess
import transformers
import outlines
import outlines.models
import outlines.processors
from PIL import Image
from peft import PeftModel
from pydantic import BaseModel
from ambres.molmo_chat import MolmoChat
from ambres import ASSETS_DIR
from ambres.dataset import Sample


class ObjGrounding(BaseModel):
    task_objects: list[str]


class AmbiguityReasoning(BaseModel):
    task_ambiguous: bool
    explanation: str
    clarifying_question: str


class AmbresStructured(MolmoChat):
    def __init__(self, use_detection: bool = False):
        super().__init__()
        self.outlines_tokenizer = outlines.models.TransformerTokenizer(
            self.molmo_processor.tokenizer
        )
        self.use_detection = use_detection
        return

    def add_message(self, role: str, text: str):
        super().add_message(role, text)
        return

    def handle_query_dict(self, input_data: dict) -> dict:
        task_description: str = input_data["task_description"]
        image = Image.open(input_data["image_path"])
        image = image.reduce(4)  # Downsample
        output = self.handle_query(task_description, image)
        return output

    def handle_query(self, task_description: str, image: Image.Image) -> dict:
        self.set_image([image], reset_chat=False)
        user_text = "<image>\nTASK DESCRIPTION: " + task_description
        self.add_message("user", user_text)
        obj_grounding_processor = transformers.LogitsProcessorList(
            [outlines.processors.JSONLogitsProcessor(ObjGrounding, self.outlines_tokenizer)]
        )
        text_out = self.run_molmo_np(
            self.messages, generate_kwargs={"logits_processor": obj_grounding_processor}
        )
        self.add_message("assistant", text_out)
        try:
            task_objects = json.loads(text_out).get("task_objects", [])
        except json.decoder.JSONDecodeError:
            warnings.warn("LLM Output broken (task_objects), defaulting to empty list.")
            task_objects = []
        if self.use_detection:
            obj_detection_messages = self.detect(task_objects)
            self.messages.extend(obj_detection_messages)
        else:
            obj_detection_messages = []
        next_text = "Is the task ambiguous?"
        self.add_message("user", next_text)
        ambiguity_processor = transformers.LogitsProcessorList(
            [outlines.processors.JSONLogitsProcessor(AmbiguityReasoning, self.outlines_tokenizer)]
        )
        text_out = self.run_molmo_np(
            self.messages, generate_kwargs={"logits_processor": ambiguity_processor}, do_sample=True
        )
        self.add_message("assistant", text_out)
        try:
            data_out = json.loads(text_out)
        except json.decoder.JSONDecodeError:
            warnings.warn("LLM Output broken, will output default response.")
            data_out = {"task_ambiguous": False, "clarifying_question": ""}
        task_ambiguous = data_out["task_ambiguous"]
        clarifying_question = data_out["clarifying_question"] if task_ambiguous else ""
        output = {
            "task_objects": task_objects,
            "obj_detection_messages": self.messages,
            "task_ambiguous": task_ambiguous,
            "clarifying_question": clarifying_question,
        }
        return output

    def handle_response(self, response: str) -> dict:
        self.add_message("user", response)
        obj_grounding_processor = transformers.LogitsProcessorList(
            [outlines.processors.JSONLogitsProcessor(ObjGrounding, self.outlines_tokenizer)]
        )
        text_out = self.run_molmo_np(
            self.messages, generate_kwargs={"logits_processor": obj_grounding_processor}
        )
        self.add_message("assistant", text_out)
        try:
            task_objects = json.loads(text_out).get("task_objects", [])
        except json.decoder.JSONDecodeError:
            warnings.warn("LLM Output broken (clarify task_objects), defaulting to empty list.")
            task_objects = []
        if self.use_detection:
            obj_detection_messages = self.detect(task_objects)
            self.messages.extend(obj_detection_messages)
        else:
            obj_detection_messages = []
        output = {"task_objects": task_objects, "obj_detection_messages": self.messages}
        return output


class AmbresFSPrompt(AmbresStructured):
    def __init__(self, use_detection: bool = False):
        self.use_detection = use_detection
        super().__init__(use_detection)
        return

    def reset_chat(self):
        """Add the first prompt message to the chat."""
        sample1 = Sample(
            id="",
            obj_list=[],
            task_description="Put the banana in the drawer.",
            ambiguity_bool=True,
            task_objects=["banana", "drawer"],
            task_objects_gt=["banana", "middle drawer"],
        )
        sample2 = Sample(
            id="",
            obj_list=[],
            task_description="Put the mug in the microwave.",
            ambiguity_bool=False,
            task_objects=["mug", "microwave"],
            task_objects_gt=["mug", "microwave"],
        )
        messages1 = sample1.as_messages(with_img_token=True)["messages"]
        messages2 = sample2.as_messages(with_img_token=False)["messages"]
        self.messages = messages1 + messages2
        return


class AmbresFineTuned(AmbresStructured):
    def __init__(self, adapter_ckpt: str, use_detection: bool = False):
        super().__init__(use_detection)
        assert adapter_ckpt != ""
        adapter_path = ASSETS_DIR.CKPT / adapter_ckpt
        if not adapter_path.exists():
            subprocess.run(
                [
                    "rsync",
                    "-hPrl",
                    f"chisari@rlgpu2:{adapter_path}",
                    f"{ASSETS_DIR.CKPT}/",
                ]
            )
        ckpt_paths = sorted(
            [f for f in adapter_path.iterdir() if f.is_dir()],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        latest_ckpt = ckpt_paths[-1]
        print("loading_ckpt from: ", latest_ckpt)
        self.molmo_model = PeftModel.from_pretrained(self.molmo_model, latest_ckpt)
        return
