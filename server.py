#!/usr/bin/env python3
"""
@Filename:    app.py.py
@Author:      dulanj
@Time:        2021-09-23 11.46
"""

from io import BytesIO

from PIL import Image
from fastapi import Body
from fastapi import FastAPI
from pydantic import Required

from src.main import get_predicted_image

app = FastAPI()


@app.post("/image/predict")
def authenticate(image_byte_stream: bytes = Body(Required, media_type="application/octet-stream")):
    img = Image.open(BytesIO(image_byte_stream))
    pred_img = get_predicted_image(img)
    if pred_img is not None:
        pred_img.save("output.png")
        ret = {"status_code": "Image saved", "status": "200"}
    else:
        ret = {"status_code": "Image saving failed", "status": "400"}
    return ret
