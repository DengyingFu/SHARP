"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

import io
import base64
import requests
import numpy as np
from PIL import Image

BASE_URL = "http://localhost:8000/"


def check_server_reply(response):
    if response.status_code != 200:
        raise ValueError(f"Server returned {response.status_code=}")
    return


class SamRemote:
    def __init__(self):
        return

    def set_image(self, pil_image: Image.Image):
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        response = requests.post(BASE_URL + "set_image_sam/", data={"image_b64": img_str})
        check_server_reply(response)
        print(response.json())
        return

    def query_mask(self, point: list[int]) -> np.ndarray:
        response = requests.post(BASE_URL + "query_mask/", data={"point": point})
        check_server_reply(response)
        mask = np.array(response.json())
        return mask
