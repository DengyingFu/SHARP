"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

import re
import json
import random
import argparse
from ambres import ASSETS_DIR


def sample_single_obj_task(obj_choices: list[str]) -> str:
    variants = ["Bring me the ", "Pick up the ", "Grasp the "]
    action = random.choice(variants)
    obj = random.choice(obj_choices)
    return action + "{" + obj + "}."


def main(env: str):
    assert env in ["sim", "real"]
    tasks_per_image = 20
    for mode in ["train", "test"]:
        raw_data_dir = ASSETS_DIR.DATA / env / mode
        with open(raw_data_dir / "data_raw.jsonl", "r") as file:
            raw_data_list = [json.loads(line) for line in file]
        out_data_list = []
        for raw_data in raw_data_list:
            id: str = raw_data["id"]
            obj_list: list[str] = raw_data["obj_list"]
            tasks: list[str] = raw_data["tasks"]
            ambiguity_map: dict = raw_data["ambiguity_map"]
            for _ in range(tasks_per_image):
                task_objects = []
                task_objects_gt = []
                ambiguity_bool = random.uniform(0, 1) > 0.5
                single_task_bool = random.uniform(0, 1) > 0.8
                if single_task_bool:
                    obj_choices = list(ambiguity_map.keys()) if ambiguity_bool else obj_list
                    task_description = sample_single_obj_task(obj_choices)
                else:
                    task_description = random.choice(tasks)
                placeholders_list = re.findall(r"\{[^}]*\}", task_description)
                task_objects_raw = [x[1:-1] for x in placeholders_list]
                task_objects_gt = [
                    (
                        random.choice(ambiguity_map[task_objects_raw[i]])
                        if task_objects_raw[i] in ambiguity_map.keys()
                        else task_objects_raw[i]
                    )
                    for i in range(len(task_objects_raw))
                ]
                task_objects = []

                # Case1: No ambiguity
                if not ambiguity_bool:
                    for i, ph in enumerate(placeholders_list):
                        task_objects.append(task_objects_gt[i])
                        task_description = task_description.replace(ph, task_objects_gt[i])

                # Case2: Ambiguity
                else:
                    amb_objects_indices = [
                        i for i, x in enumerate(task_objects_raw) if x in ambiguity_map
                    ]
                    amb_idx = (
                        random.choice(amb_objects_indices) if len(amb_objects_indices) > 0 else -1
                    )
                    for i, ph in enumerate(placeholders_list):
                        if i == amb_idx:
                            task_objects.append(task_objects_raw[i])
                            task_description = task_description.replace(ph, task_objects_raw[i])
                        else:
                            task_objects.append(task_objects_gt[i])
                            task_description = task_description.replace(ph, task_objects_gt[i])

                out_data = {
                    "id": f"{id}",
                    "obj_list": obj_list,
                    "task_description": task_description,
                    "ambiguity_bool": ambiguity_bool,
                    "task_objects": task_objects,
                    "task_objects_gt": task_objects_gt,
                }
                out_data_list.append(out_data)
        # Save to json
        with open(raw_data_dir / "data.json", "w") as file:
            json.dump(out_data_list, file, indent=2)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["sim", "real"], default="real")
    args = parser.parse_args()
    main(args.env)
