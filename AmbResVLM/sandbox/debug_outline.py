"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

import outlines
import outlines.models
import outlines.processors
import transformers
from PIL import Image
from typing import Optional
from pydantic import BaseModel
from ambres.molmo_chat import MolmoChat
from ambres.ambres_model import AmbresFineTuned
from ambres import ASSETS_DIR, DATA_DIR, CKPT


# class AmbiguityReasoning(BaseModel):
#     task_ambiguous: bool
#     explanation: str
#     clarifying_question: str


class AmbiguityReasoning(BaseModel):
    task_ambiguous: bool
    explanation: Optional[str] = ""
    clarifying_question: Optional[str] = ""


image = Image.open(DATA_DIR.get_dir("sim", "train") / "bxJy6scnqGMrdVqgbiZayA.jpeg")
image = image.reduce(4)
# molmo = MolmoChat()
molmo = AmbresFineTuned(CKPT.SIM)

molmo.set_image([image])
# molmo.add_message(
#     "user",
#     '<image>\nOutput the following json data: {"task_ambiguous": true}',  # , "explanation": "Only one red mug"}',  # , "clarifying_question": ""}',
#     # "<image>\nIs the task ambiguous?",
# )
molmo.add_message("user", "<image>\nTASK DESCRIPTION: Place the yellow block in the red bowl.")
molmo.add_message("assistant", '{"task_objects": ["yellow block", "red bowl"]}')
molmo.add_message("user", "Is the task ambiguous?")

outlines_tokenizer = outlines.models.TransformerTokenizer(molmo.molmo_processor.tokenizer)
ambiguity_processor = transformers.LogitsProcessorList(
    [outlines.processors.JSONLogitsProcessor(AmbiguityReasoning, outlines_tokenizer)]
)

text_out = molmo.run_molmo_np(
    molmo.messages, generate_kwargs={"logits_processor": ambiguity_processor}
)

print(text_out)
print("done")
