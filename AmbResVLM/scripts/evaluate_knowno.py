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
from ambres import ASSETS_DIR
from ambres.knowno_model import KnowNo
from ambres.utils import classification_metrics


def main(env: str):
    assert env in ["sim", "real"]
    log_wandb = True
    verbose = False

    mode = "test"
    data_path = ASSETS_DIR.DATA / env / mode / "data.json"
    with open(data_path, "r") as file:
        data_list = json.load(file)
    model = KnowNo(env, verbose)

    wandb_config = {"env": env, "mode": mode, "model": model.__class__.__name__}
    print(wandb_config)
    wandb.init(
        project="ambres-eval",
        entity="chisarie",
        config=wandb_config,
        mode="online" if log_wandb else "disabled",
    )

    ambiguity_pred_list = []
    ambiguity_gt_list = []
    for data in tqdm(data_list):
        task_description = data["task_description"]
        obj_list = data["obj_list"]
        ambiguity_gt = data["ambiguity_bool"]
        ambiguity_pred = model.predict_ambiguity(task_description, obj_list)
        ambiguity_gt_list.append(ambiguity_gt)
        ambiguity_pred_list.append(ambiguity_pred)
        wandb.log({"amb_gt": int(ambiguity_gt), "amb_pred": int(ambiguity_pred)})

    ambiguity_pred_np = np.array(ambiguity_pred_list, dtype=int)
    ambiguity_gt_np = np.array(ambiguity_gt_list, dtype=int)
    _ = classification_metrics(ambiguity_pred_np, ambiguity_gt_np, verbose, log_wandb)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["sim", "real"], default="real")
    args = parser.parse_args()
    main(args.env)
