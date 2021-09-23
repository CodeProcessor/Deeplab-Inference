#!/usr/bin/env python3
"""
@Filename:    inference.py
@Author:      dulanj
@Time:        2021-09-23 00.27
"""
import glob
import os

import numpy as np
import tensorflow as tf
from PIL import Image

from src.model import LOADED_MODEL, DATASET_INFO
from src.utils import vis_segmentation

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def predict(image_path):
    _name = os.path.basename(image_path).split('.')[0]
    output_dir = "data/output"
    with tf.io.gfile.GFile(image_path, 'rb') as f:
        im = np.array(Image.open(f))

    output = LOADED_MODEL(tf.cast(im, tf.uint8))

    return vis_segmentation(im, output['panoptic_pred'][0], DATASET_INFO, output_name=f"{output_dir}/output_{_name}")


def get_predicted_image(image):
    im = np.array(image)
    output = LOADED_MODEL(tf.cast(im, tf.uint8))
    path = vis_segmentation(im, output['panoptic_pred'][0], DATASET_INFO, output_name="image")
    if os.path.exists(path):
        return Image.open(path)
    else:
        return None


if __name__ == '__main__':
    for _file_name in glob.glob("src/data/MVD_research_samples/Asia/*.jpg"):
        predict(_file_name)
        print(f"Predicted for the image : {_file_name}")

