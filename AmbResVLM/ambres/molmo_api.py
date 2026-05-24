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
curl -X POST "http://localhost:8000/set_image_file/" -F "image=@path/to/image.jpg"
curl -X POST "http://localhost:8000/step_chat/" -F "user_text=example text"
-------------------------------------
"""

import base64
import uvicorn
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from ambres.molmo_chat import MolmoChat

app = FastAPI()
molmo = MolmoChat()


@app.post("/set_image_file/")
async def set_image_file(image: UploadFile = File(...)):
    """
    Usage Example:
    --------------
    import requests
    url = "http://localhost:8000/set_image_file/"
    files = {'image': open('path/to/image.jpg', 'rb')}
    response = requests.post(url, files=files)
    print(response.json())
    ------------
    """
    # Convert image to PIL Image
    pil_image = Image.open(image.file).convert("RGB")
    molmo.set_image([pil_image])
    return JSONResponse(content={"response": "Image set"})


@app.post("/set_image_base64/")
async def set_image_base64(image_b64: str = Form(...)):
    """
    Usage Example:
    --------------
    import requests
    url = "http://localhost:8000/set_image_base64/"
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    response = requests.post(url, json={"image_b64": img_str})
    print(response.json())
    --------------
    """
    # Convert base64 string to PIL Image
    image_bytes = base64.b64decode(image_b64)
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    molmo.set_image([pil_image])
    return JSONResponse(content={"response": "Image set"})


@app.post("/add_message/")
async def add_message(role: str = Form(...), text: str = Form(...)):
    """
    Usage Example:
    --------------
    import requests
    url = "http://localhost:8000/add_message/"
    data = {"role": "user", "text": "example text"}
    response = requests.post(url, data=data)
    print(response.json())
    --------------
    """
    response_text = molmo.add_message(role, text)
    return JSONResponse(content={"response": response_text})


@app.post("/reset_chat/")
async def reset_chat():
    molmo.reset_chat()
    return JSONResponse(content={"response": "Chat reset"})


@app.post("/step_chat/")
async def step_chat(user_text: str = Form(...)):
    response_text = molmo.step_chat(user_text)
    return JSONResponse(content={"response": response_text})


@app.post("/debug/")
async def debug(text: str = Form(...)):
    return JSONResponse(content={"response": "Received: " + text})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
