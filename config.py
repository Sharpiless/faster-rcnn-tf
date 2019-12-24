# config.py
import numpy as np
import os

DATA_PATH = r'..\VOC2012'

MODEL_PATH = 'model.ckpt'

Annotations_PATH = os.path.join(DATA_PATH, 'Annotations')
ImageSets_PATH = os.path.join(DATA_PATH, 'ImageSets')
JPEGImages_PATH = os.path.join(DATA_PATH, 'JPEGImages')

IMAGE_WIDTH = 300

target_size = 300

MAX_STEP = 60

MAX_EPOCH = 32

LEARNING_RATE = 2e-5

CLASSES = ['background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',

           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',

           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',

           'train', 'tvmonitor']

target_size = 600

max_size = 1000

overlaps_max = 0.7

overlaps_min = 0.3

IMAGE_BATCH = 1

momentum = 0.9

map_width = [38, 19, 10, 5, 3, 1]

anchor_scales = np.array([128, 256, 512])

anchor_ratios = np.array([1., 0.5, 2.])

anchor_batch = 256

feat_strides = 16

POOLING_SIZE = 7

PIXEL_MEANS = np.array([[[122.7717, 115.9465, 102.9801]]])  # 三通道像素的均值

dect_fg_rate = 0.25

bbox_nor_mean = (0.0, 0.0, 0.0, 0.0)

bbox_nor_stdv = (0.1, 0.1, 0.2, 0.2)

# roi_input_inside_weight = (0.5, 0.5, 1., 1.)  # dx,dy,dw,dh 的权重
roi_input_inside_weight = (1., 1., 1., 1.)

pos_thresh = 0.7

test_thresh = 0.80
