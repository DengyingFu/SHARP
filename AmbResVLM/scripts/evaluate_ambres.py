"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

import json
import wandb
import argparse
import numpy as np
from tqdm import tqdm
import wandb.integration
from ambres import ASSETS_DIR, DATA_DIR, CKPT
from ambres.ambres_model import AmbresFSPrompt, AmbresFineTuned
from ambres.utils import classification_metrics


def find_ambiguity_idx(data):
    """
    Find the index of the first ambiguous object.
    """
    idx = -1
    for i in range(min(len(data["task_objects"]), len(data["task_objects_gt"]))):
        if data["task_objects"][i] != data["task_objects_gt"][i]:
            idx = i
            break
    return idx


def objects_iou(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    iou = len(intersection) / len(union)
    return iou


def main(env: str, model_type: str):
    assert env in ["sim", "real"]
    assert model_type in ["prompt", "finetune"]
    log_wandb = True
    verbose = False

    mode = "test"
    data_path = ASSETS_DIR.DATA / env / mode / "data.json"
    print(data_path)
    if model_type == "prompt":
        model = AmbresFSPrompt(use_detection=False)
    elif model_type == "finetune":
        ckpt_name = CKPT.get_ckpt_name(env)
        model = AmbresFineTuned(adapter_ckpt=ckpt_name)
    else:
        raise ValueError("Invalid model type or environment.")
    with open(data_path, "r") as file:
        data_list = json.load(file)

    wandb_config = {
        "env": env,
        "mode": mode,
        "model": model.__class__.__name__,
        "ckpt": ckpt_name if model_type == "finetune" else "",
    }
    print(wandb_config)
    wandb.init(
        project="ambres-eval",
        entity="dingfu",
        config=wandb_config,
        mode="online" if log_wandb else "disabled",
    )
    wandb.define_metric("obj_grounding_iou", summary="mean")
    wandb.define_metric("ambiguity_detection_acc", summary="mean")
    wandb.define_metric("ambiguity_resolution", summary="mean")

    obj_grounding_iou_list = []
    ambiguity_pred_list = []
    ambiguity_gt_list = []
    ambiguity_resolution_list = []
    for i, data in enumerate(tqdm(data_list)):
        model.reset_chat()
        # img_path = DATA_DIR.get_dir(env, mode) / f"{data['id']}.jpeg"
        img_path = "/home/zty_group/fdy/AmbRes/ambres/sim/test/"+f"{data['id']}.jpeg"
        data["image_path"] = img_path
        output_query = model.handle_query_dict(data)
        print(output_query)
        ambiguity_gt = data["ambiguity_bool"]
        ambiguity_pred = output_query["task_ambiguous"]
        iou = objects_iou(data["task_objects"], output_query["task_objects"])
        obj_grounding_iou_list.append(iou)
        ambiguity_gt_list.append(ambiguity_gt)
        ambiguity_pred_list.append(ambiguity_pred)
        wandb.log(
            {
                "obj_grounding_iou": iou,
                "amb_gt": int(ambiguity_gt),
                "amb_pred": int(ambiguity_pred),
            },
            step=i,
        )
        if data["ambiguity_bool"]:
            ambiguity_idx = find_ambiguity_idx(data)
            response = f"I want the {data['task_objects_gt'][ambiguity_idx]}"
            output_reply = model.handle_response(response)
            print(output_reply)
            reply_iou = objects_iou(output_reply["task_objects"], data["task_objects_gt"])
            amb_res_score = 1 if reply_iou == 1.0 else 0
            ambiguity_resolution_list.append(amb_res_score)
            wandb.log({"ambiguity_resolution": amb_res_score}, step=i)

        if verbose:
            print(f"obj_grounding_iou: {iou}")
            print(f"ambiguity_gt: {ambiguity_gt} - ", f"ambiguity_pred: {ambiguity_pred}")
            if data["ambiguity_bool"]:
                print(f"ambiguity_resolution: {amb_res_score}")

    obj_grounding_iou_np = np.array(obj_grounding_iou_list)
    ambiguity_resolution_np = np.array(ambiguity_resolution_list)
    ambiguity_pred_np = np.array(ambiguity_pred_list)
    ambiguity_gt_np = np.array(ambiguity_gt_list)

    # Log summary values
    mean_obj_grounding_iou = obj_grounding_iou_np.mean()
    print(f"Mean Obj Grounding IoU: {mean_obj_grounding_iou}")
    wandb.run.summary["mean_obj_grounding_iou"] = mean_obj_grounding_iou

    mean_amb_resolution = ambiguity_resolution_np.mean()
    print(f"Mean Ambiguity Resolution Score: {mean_amb_resolution}")
    wandb.run.summary["mean_amb_resolution"] = mean_amb_resolution

    _ = classification_metrics(ambiguity_pred_np, ambiguity_gt_np, verbose, log_wandb)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["sim", "real"], default="real")
    parser.add_argument("--model_type", choices=["prompt", "finetune"], default="finetune")
    args = parser.parse_args()
    main(args.env, args.model_type)
