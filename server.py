#!/usr/bin/env python3
"""
@Filename:    app.py.py
@Author:      dulanj
@Time:        2021-09-23 11.46
"""

import datetime
from io import BytesIO

import uvicorn
from PIL import Image
from fastapi import Body
from fastapi import FastAPI
from pydantic import Required

from src.main import get_predicted_image

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/image/predict")
def authenticate(image_byte_stream: bytes = Body(Required, media_type="application/octet-stream")):
    img = Image.open(BytesIO(image_byte_stream))
    pred_img = get_predicted_image(img)
    if pred_img is not None:
        pred_img.save(f"output/output_{datetime.datetime.now().isoformat()}.png")
        ret = {"status_code": "Image saved", "status": "200"}
    else:
        ret = {"status_code": "Image saving failed", "status": "400"}
    return ret


def start_server():
    server_ip = '0.0.0.0'
    server_port = 80
    print(f"Starting Deeplab Server IP: {server_ip} PORT: {server_port}")
    print(f"Documentation available at: {server_ip}:{server_port}/docs")
    uvicorn.run(app, port=server_port, host=server_ip)


if __name__ == '__main__':
    start_server()
