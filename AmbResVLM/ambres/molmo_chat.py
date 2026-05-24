"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

import warnings
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from ambres import ASSETS_DIR
from ambres.training.data import process_msg_list

# from ambres.dataset import SYSTEM_PROMPT_FORMATTED


def extract_coordinates(location: str, img_h: int, img_w: int) -> list:
    soup = BeautifulSoup(location, features="html.parser")
    if soup.point:
        x = float(soup.point["x"])
        y = float(soup.point["y"])
        point = np.array([x, y])
        point[0] = (point[0] / 100 * img_w).astype(int)
        point[1] = (point[1] / 100 * img_h).astype(int)
        points = [list(point)]
    elif soup.points:
        points = []
        idx = 1
        while "x" + str(idx) in soup.points.attrs:
            x = float(soup.points["x" + str(idx)])
            y = float(soup.points["y" + str(idx)])
            point = np.array([x, y])
            point[0] = (point[0] / 100 * img_w).astype(int)
            point[1] = (point[1] / 100 * img_h).astype(int)
            points.append(list(point))
            idx += 1
    else:
        warnings.warn("Invalid location format")
        points = [[]]
    return points


class MolmoChat:
    def __init__(self):
        # Molmo
        model_id = "allenai/Molmo-7B-D-0924"
        self.molmo_processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="bfloat16",
            device_map="cuda",
        )
        self.molmo_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="bfloat16",
            device_map="cuda",
        )
        self.images = []
        self.reset_chat()
        return

    def set_image(self, images: list[Image.Image], reset_chat: bool = True):
        self.images = images
        if reset_chat:
            self.reset_chat()
        return

    def add_message(self, role: str, text: str):
        assert role in ["user", "assistant"]
        self.messages.append({"role": role, "content": text})
        return

    def reset_chat(self):
        self.messages = []
        return

    def run_molmo_np(self, messages, generate_kwargs={}, do_sample=False) -> str:
        # process the image and text
        inputs = process_msg_list(messages, self.images, self.molmo_processor)
        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(device=self.molmo_model.device).unsqueeze(0) for k, v in inputs.items()}
        inputs["images"] = inputs["images"].to(self.molmo_model.dtype)
        inputs["image_masks"] = inputs["image_masks"].to(self.molmo_model.dtype)

        # GenerationConfig DOCS:
        # https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig
        # https://huggingface.co/docs/transformers/v4.48.2/en/generation_strategies
        # https://huggingface.co/blog/how-to-generate
        generation_config = GenerationConfig(
            do_sample=do_sample,
            max_new_tokens=2000,
            stop_strings="<|endoftext|>",
        )

        # generate output; maximum new tokens; stop generation when <|endoftext|> is generated
        generation_output = self.molmo_model.generate_from_batch(
            inputs,
            generation_config,
            tokenizer=self.molmo_processor.tokenizer,
            return_dict_in_generate=True,
            output_logits=True,
            **generate_kwargs,
        )
        sequences = generation_output.sequences

        # only get generated tokens; decode them to text
        generated_tokens = sequences[0, inputs["input_ids"].size(1) :]
        generated_text = self.molmo_processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        return generated_text

    def detect(self, objects: list[str]) -> list[str]:
        out_messages = []
        for obj in objects:
            in_message = {"role": "user", "content": f"Find the {obj}."}
            text_out = self.run_molmo_np([in_message])
            out_messages.extend([in_message, {"role": "assistant", "content": text_out}])
        return out_messages

    def detect_pretty(self, objects: list[str]) -> dict:
        out = {}
        for obj in objects:
            in_message = {"role": "user", "content": f"Find the {obj}."}
            text_out = self.run_molmo_np([in_message])
            img_w, img_h = self.images[0].size
            coords = extract_coordinates(text_out, img_h, img_w)
            out[obj] = coords
        return out

    def step_chat(self, user_text: str):
        self.add_message("user", user_text)
        text_out = self.run_molmo_np(self.messages)
        self.add_message("assistant", text_out)
        return text_out


if __name__ == "__main__":
    adapter_ckpt = ""  # None
    image = Image.open(ASSETS_DIR.IMAGES / "5rhU25AdQW4jADxhp8EYuq.jpeg")
    image = image.reduce(4)

    molmo = MolmoChat()
    molmo.set_image([image])
    for _ in range(10):
        user_text = input("Type the next message: ")
        text_out = molmo.step_chat(user_text)
        print(text_out)
    out = molmo.detect_pretty(["mug", "red_marker"])
    print(out)

    print("done")
