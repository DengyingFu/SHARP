"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

import torch
import pathlib
import socket
from dataclasses import dataclass

# select the device for computation
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


@dataclass
class ASSETS_DIR:
    ROOT = pathlib.Path(__file__).parents[1] / "assets"
    CKPT = ROOT.parent / "ckpt"
    WEIGHTS = ROOT / "weights"
    IMAGES = ROOT / "images"
    DATA = ROOT / "data"
    URDF = ROOT / "urdf"

    @staticmethod
    def get_dir(env="real", mode="train"):
        assert env in ["sim", "real"]
        assert mode in ["train", "test"]
        return ASSETS_DIR.DATA / env / mode


def root():
    hostname = socket.gethostname()
    if hostname in ["rlgpu3", "rlgpu7", "rlgpu9"]:
        return pathlib.Path("/data/ws1/chisari-ambres_ws")
    elif hostname.startswith("rlgpu"):
        return pathlib.Path("/data2/ws1/chisari-ambres_ws")
    elif hostname.startswith("dlc2gpu"):
        return pathlib.Path("/work/dlclarge2/chisari-ambres_ws")
    else:
        return pathlib.Path.home() / "datasets"


@dataclass
class DATA_DIR:
    ROOT = root()
    AMBRES = ROOT / "ambres"

    @staticmethod
    def get_dir(env="real", mode="train"):
        assert env in ["sim", "real"]
        assert mode in ["train", "test"]
        return DATA_DIR.AMBRES / env / mode


@dataclass
class CKPT:
    REAL = "43qazb3XcrZF5rZWnjRPVm"
    SIM = "TtVVGPRoknCTELVSjH92xX"

    @staticmethod
    def get_ckpt_name(env="real"):
        assert env in ["sim", "real"]
        return CKPT.SIM if env == "sim" else CKPT.REAL
