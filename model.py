#!/usr/bin/env python3
"""
@Filename:    model.py.py
@Author:      dulanj
@Time:        2021-09-23 00.29
"""
import os
import tensorflow as tf
from utils import cityscapes_dataset_information

DIR_NAME = '/home/dulanj/Learn/Deeplab/'
MODEL_NAME = 'resnet50_os32_panoptic_deeplab_cityscapes_crowd_trainfine_saved_model'

_MODELS = ('resnet50_os32_panoptic_deeplab_cityscapes_crowd_trainfine_saved_model',
           'resnet50_beta_os32_panoptic_deeplab_cityscapes_trainfine_saved_model',
           'wide_resnet41_os16_panoptic_deeplab_cityscapes_trainfine_saved_model',
           'swidernet_sac_1_1_1_os16_panoptic_deeplab_cityscapes_trainfine_saved_model',
           'swidernet_sac_1_1_3_os16_panoptic_deeplab_cityscapes_trainfine_saved_model',
           'swidernet_sac_1_1_4.5_os16_panoptic_deeplab_cityscapes_trainfine_saved_model',
           'axial_swidernet_1_1_1_os16_axial_deeplab_cityscapes_trainfine_saved_model',
           'axial_swidernet_1_1_3_os16_axial_deeplab_cityscapes_trainfine_saved_model',
           'axial_swidernet_1_1_4.5_os16_axial_deeplab_cityscapes_trainfine_saved_model',
           'max_deeplab_s_backbone_os16_axial_deeplab_cityscapes_trainfine_saved_model',
           'max_deeplab_l_backbone_os16_axial_deeplab_cityscapes_trainfine_saved_model')
_DOWNLOAD_URL_PATTERN = 'https://storage.googleapis.com/gresearch/tf-deeplab/saved_model/%s.tar.gz'

_MODEL_NAME_TO_URL_AND_DATASET = {
    model: (_DOWNLOAD_URL_PATTERN % model, cityscapes_dataset_information())
    for model in _MODELS
}

MODEL_URL, DATASET_INFO = _MODEL_NAME_TO_URL_AND_DATASET[MODEL_NAME]
LOADED_MODEL = tf.saved_model.load(os.path.join(DIR_NAME, "resnet50_os32_panoptic_deeplab_cityscapes_crowd_trainfine_saved_model"))

