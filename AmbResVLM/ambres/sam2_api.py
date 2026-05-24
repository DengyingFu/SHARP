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
"""

import base64
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from ambres.sam2_model import Sam


sam = Sam()
app = FastAPI()


@app.get("/ping/")
async def ping():
    return JSONResponse(content={"response": "pong"})


@app.post("/set_image_sam/")
async def set_image_sam(image_b64: str = Form(...)):
    """
    Usage Example:
    --------------
    import requests
    url = "http://localhost:8000/set_image_base64/"
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
    return JSONResponse(content={"response": "Image set"})


@app.post("/query_mask/")
async def query_mask(points: list[list[int]] = Form(...)):
    """
    Usage Example:
    --------------
    import requests
    url = "http://localhost:8000/query_mask/"
    response = requests.post(url, data={"points": [[1384, 1060]]})
    print(response.json())
    --------------
    """
    points = np.array(points)
    mask = sam.query_sam_points(points)
    return JSONResponse(content=mask.asdtype(np.uint8).tolist())


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
