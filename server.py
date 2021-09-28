#!/usr/bin/env python3
"""
@Filename:    app.py.py
@Author:      dulanj
@Time:        2021-09-23 11.46
"""

import datetime
import io
from io import BytesIO

import uvicorn
from PIL import Image
from fastapi import Body
from fastapi import FastAPI
from pydantic import Required
from starlette.responses import StreamingResponse

from src.main import get_predicted_image

app = FastAPI()


@app.get("/")
def read_root():
    return {"status_code": "Deeplab Server Running", "status": "200"}


@app.post("/image/predict")
def authenticate(image_byte_stream: bytes = Body(Required, media_type="application/octet-stream")):
    img = Image.open(BytesIO(image_byte_stream))
    pred_img = get_predicted_image(img)
    if pred_img is not None:
        file_name = f"output/output_{datetime.datetime.now().isoformat()}.png"
        pred_img.save(file_name)
        ret = {"status_code": f"Image saved: {file_name}", "status": "200"}
    else:
        ret = {"status_code": "Image saving failed", "status": "400"}
    return ret


@app.post("/image/predict2")
def authenticate(image_byte_stream: bytes = Body(Required, media_type="application/octet-stream")):
    img = Image.open(BytesIO(image_byte_stream))
    pred_img = get_predicted_image(img)
    if pred_img is not None:
        file_name = f"output/output_{datetime.datetime.now().isoformat()}.png"
        pred_img.save(file_name)
        pred_img.seek(0)
        ret = StreamingResponse(io.BytesIO(pred_img.tobytes()), media_type="image/png")
    else:
        ret = {"status_code": "Image saving failed", "status": "400"}
    return ret


def start_server():
    docker_server = True
    if docker_server:
        server_ip = '0.0.0.0'
        server_port = 80
    else:
        server_ip = '127.0.0.1'
        server_port = 5000
    print(f"Starting Deeplab Server IP: {server_ip} PORT: {server_port}")
    print(f"Documentation available at: http://{server_ip}:{server_port}/docs")
    uvicorn.run(app, port=server_port, host=server_ip)


if __name__ == '__main__':
    start_server()
