"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

"""
To open an ssh tunnel to the server:
-------------------------------------
ssh -L 8000:localhost:8000 user@server_ip
-------------------------------------

Usage Example from terminal:
-------------------------------------
curl -X POST "http://localhost:8000/ping/"
curl -X POST "http://localhost:8000/handle_response/" -F "response=helloworld"
-------------------------------------
"""

import argparse
import base64
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from ambres import CKPT
from ambres.sam2_model import Sam
from ambres.ambres_model import AmbresFSPrompt, AmbresFineTuned

parser = argparse.ArgumentParser()
parser.add_argument("model_type")
args = parser.parse_args()
model_type = args.model_type

assert model_type in ["prompt", "finetune"], f"I don't recognize {model_type=}"
if model_type == "prompt":
    model = AmbresFSPrompt(use_detection=False)
elif model_type == "finetune":
    model = AmbresFineTuned(adapter_ckpt=CKPT.REAL)
else:
    raise ValueError("Invalid model type or environment.")

sam = Sam()
app = FastAPI()


@app.get("/ping/")
async def ping():
    return JSONResponse(content={"response": "pong"})


@app.get("/get_ckpt/")
async def get_ckpt():
    """
    Usage Example:
    --------------
    import requests
    url = "http://localhost:8000/get_ckpt/"
    response = requests.get(url)
    print(response.json())
    --------------
    """
    ckpt = CKPT.REAL if model_type == "finetune" else ""
    return JSONResponse(content={"ckpt": ckpt})


@app.post("/reset_chat/")
async def reset_chat():
    """
    Usage Example:
    --------------
    import requests
    url = "http://localhost:8000/reset_chat/"
    response = requests.post(url)
    print(response.json())
    --------------
    """
    model.reset_chat()
    return JSONResponse(content={"response": "Chat reset"})


@app.post("/set_image_molmo/")
async def set_image_molmo(image_b64: str = Form(...)):
    """
    Usage Example:
    --------------
    import requests
    url = "http://localhost:8000/set_image_molmo/"
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    response = requests.post(url, data={"image_b64": img_str})
    print(response.json())
    --------------
    """
    # Convert base64 string to PIL Image
    image_bytes = base64.b64decode(image_b64)
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    model.set_image([pil_image])
    return JSONResponse(content={"response": "Molmo: Image set"})


@app.post("/handle_query/")
async def handle_query(task_description: str = Form(...), image_b64: str = Form(...)):
    """
    Usage Example:
    --------------
    import requests
    url = "http://localhost:8000/handle_query/"
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    response = requests.post(url, data={"task_description": "Grasp the banana", "image_b64": img_str})
    print(response.json())
    --------------
    """
    # Convert base64 string to PIL Image
    image_bytes = base64.b64decode(image_b64)
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    output_query = model.handle_query(task_description, pil_image)
    return JSONResponse(content=output_query)


@app.post("/handle_response/")
async def handle_response(response: str = Form(...)):
    """
    Usage Example:
    --------------
    import requests
    url = "http://localhost:8000/handle_response/"
    out = requests.post(url, data={"response": "response"})
    print(out.json())
    --------------
    """
    output_query = model.handle_response(response)
    return JSONResponse(content=output_query)


@app.post("/detect/")
async def detect(objects: list[str] = Form(...)):
    """
    Usage Example:
    --------------
    import requests
    url = "http://localhost:8000/detect/"
    response = requests.post(url, data={"objects": ["apple", "banana"]})
    print(response.json())
    --------------
    """
    output = model.detect_pretty(objects)
    return JSONResponse(content=output)


@app.post("/set_image_sam/")
async def set_image_sam(image_b64: str = Form(...)):
    """
    Usage Example:
    --------------
    import requests
    url = "http://localhost:8000/set_image_sam/"
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    response = requests.post(url, data={"image_b64": img_str})
    print(response.json())
    --------------
    """
    # Convert base64 string to PIL Image to numpy
    image_bytes = base64.b64decode(image_b64)
    image_pil = Image.open(BytesIO(image_bytes))
    image_np = np.array(image_pil.convert("RGB"))
    sam.set_sam_image(image_np)
    return JSONResponse(content={"response": "SAM: Image set"})


@app.post("/query_mask/")
async def query_mask(point: list[int] = Form(...)):
    """
    Usage Example:
    --------------
    import requests
    url = "http://localhost:8000/query_mask/"
    response = requests.post(url, data={"point": [1384, 1060]})
    print(response.json())
    --------------
    """
    points = np.array([point])
    mask = sam.query_sam_points(points)
    return JSONResponse(content=mask.tolist())


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
