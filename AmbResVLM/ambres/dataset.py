"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

import json
import jsonlines
from PIL import Image
from bs4 import BeautifulSoup
from dataclasses import dataclass
from functools import cached_property
from collections import defaultdict
from ambres import ASSETS_DIR, DATA_DIR


def _objects_from_locations(locations: str) -> list[str]:
    soup = BeautifulSoup(locations, features="html.parser")
    objects = [tag["alt"] for tag in soup.find_all()]
    return objects


@dataclass
class Sample:
    id: str
    obj_list: list[str]
    task_description: str
    ambiguity_bool: bool
    task_objects: list[str]
    task_objects_gt: list[str]

    @cached_property
    def img_fname(self) -> str:
        return f"{self.id}.jpeg"

    @cached_property
    def amb_idx(self) -> int:
        """
        Find the index of the first ambiguous object.
        """
        idx = -1
        for i in range(min(len(self.task_objects), len(self.task_objects_gt))):
            if self.task_objects[i] != self.task_objects_gt[i]:
                idx = i
                break
        return idx

    @property
    def ambiguous_obj(self) -> str:
        return self.task_objects[self.amb_idx]

    @property
    def ambiguous_obj_gt(self) -> str:
        return self.task_objects_gt[self.amb_idx]

    def as_messages(self, with_img_token=True) -> dict:
        img_token_str = "<image>\n" if with_img_token else ""
        messages = []

        def add_message(role: str, content: str):
            assert role in ["user", "assistant"]
            messages.append({"role": role, "content": content})

        add_message("user", f"{img_token_str}TASK DESCRIPTION: {self.task_description}")
        add_message("assistant", f'{{"task_objects": {json.dumps(self.task_objects)}}}')
        add_message("user", f"Is the task ambiguous?")
        if self.ambiguity_bool:
            add_message(
                "assistant",
                f'{{"task_ambiguous": true, "explanation": "There are multiple {self.ambiguous_obj}s", "clarifying_question": "Which {self.ambiguous_obj} do you mean?"}}',
            )
            add_message("user", f"I mean the {self.ambiguous_obj_gt}")
            add_message("assistant", f'{{"task_objects": {self.task_objects_gt}}}\n')
        else:
            add_message(
                "assistant",
                '{"task_ambiguous": false, "explanation": "All objects are uniquely identified", "clarifying_question": "Everything is clear"}',
            )
        out = {"image": self.img_fname, "messages": messages}
        return out


class DataHandler:
    def __init__(self, env: str = "real", mode: str = "train"):
        assert env in ["sim", "real"]
        assert mode in ["train", "test"]
        self.env = env
        self.mode = mode
        self.images_dir = DATA_DIR.get_dir(env, mode)
        self.file_path = ASSETS_DIR.get_dir(env, mode) / "data_raw.jsonl"
        self.file_path.touch()
        self.data_list = self._load_data()
        self.image_data_map = self._map_images_to_data()
        return

    def _load_data(self) -> list[dict]:
        with open(self.file_path, "r") as file:
            out = [json.loads(line) for line in file]
        return out

    def _map_images_to_data(self) -> defaultdict:
        data_map = dict()
        for data in self.data_list:
            id = data["id"]
            data_map[id] = {
                "obj_list": data["obj_list"],
                "ambiguity_map": data["ambiguity_map"],
                "tasks": data["tasks"],
            }
        return data_map

    def append_sample(self, sample_dict: dict):
        for x in ["id", "obj_list", "tasks", "ambiguity_map"]:
            assert x in sample_dict.keys()
        with jsonlines.open(self.file_path, mode="a") as writer:
            writer.write(sample_dict)
        return

    def get_data_for_image(self, image_id: str) -> list[dict]:
        return self.image_data_map[image_id] if image_id in self.image_data_map else {}

    def get_images(self, image_fname: str) -> list[Image]:  # type: ignore
        image = Image.open(DATA_DIR.get_dir(self.env, self.mode) / image_fname)
        image = image.reduce(4)  # Downsample
        return [image]


if __name__ == "__main__":
    data_handler = DataHandler()
    print("debug")
