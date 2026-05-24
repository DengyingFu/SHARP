"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

import io
import base64
import requests
from PIL import Image
from ambres import CKPT

BASE_URL = "http://localhost:8000/"


def check_server_reply(response):
    if response.status_code != 200:
        raise ValueError(f"Server returned {response.status_code=}")
    return


class AmbresRemote:
    def __init__(self, model_type: str = "finetune"):
        assert model_type in ["prompt", "finetune"], f"I don't recognize {model_type=}"
        ckpt = CKPT.REAL if model_type == "finetune" else ""
        ckpt_response = requests.get(BASE_URL + "get_ckpt/")
        check_server_reply(ckpt_response)
        assert ckpt == ckpt_response.json()["ckpt"], "Are you using the right ckpt?"
        return

    def reset_chat(self):
        response = requests.post(BASE_URL + "reset_chat/")
        check_server_reply(response)
        print(response.json())
        return

    def set_image(self, pil_image: Image.Image):
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        response = requests.post(BASE_URL + "set_image_molmo/", data={"image_b64": img_str})
        check_server_reply(response)
        print(response.json())
        return

    def handle_query(self, task_description: str, image: Image.Image) -> dict:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        response = requests.post(
            BASE_URL + "handle_query/",
            data={"task_description": task_description, "image_b64": img_str},
        )
        check_server_reply(response)
        return response.json()

    def handle_response(self, response: str) -> dict:
        out = requests.post(BASE_URL + "handle_response/", data={"response": response})
        check_server_reply(out)
        return out.json()

    def detect(self, objects: list[str]) -> dict:
        out = requests.post(BASE_URL + "detect/", data={"objects": objects})
        check_server_reply(out)
        return out.json()
