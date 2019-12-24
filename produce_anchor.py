# produce_anchor.py
import numpy as np
import tensorflow as tf
import config as cfg


def generate_anchor(scales=cfg.anchor_scales, ratios=cfg.anchor_ratios):

    anchor_widths = np.array(scales)
    # array([[128, 128],[256, 256],[512, 512]])
    anchor_ratios = np.array(ratios)
    # array([0.5, 1, 2])

    anchor_size = get_9_anchor_size(anchor_widths, anchor_ratios)
    # 返回大小为9*2的矩阵，包含9个anchor的width和height
    # 例如[181., 90.],[128., 128.],[91., 182.]（对128取样）

    anchor_dev_to_ctr = get_conners_coord(anchor_size)
    # 返回大小为(9, 4)的矩阵，包含9个anchor两个对角坐标对应中间点的偏移

    return anchor_dev_to_ctr


def get_9_anchor_size(anchor_width, anchor_ratios):
    # anchor_scales = array([128, 256, 512])
    # anchor_ratios = array([0.5,1,2])
    anchor_areas = anchor_width * anchor_width
    # 三个框的面积不变

    # np.vstack 将两个矩阵在垂直方向上叠加
    anchor_size = np.vstack([get_1_anchor_size(
        anchor_areas[i], anchor_ratios) for i in range(anchor_areas.shape[0])])

    return anchor_size  # (9, 2) 包含9个anchor的width和height


def get_1_anchor_size(anchor_area, anchor_ratios):
    # anchor_area = 128*128 （或者256*256、512*512）
    # anchor_ratios = array([0.5,1,2])
    width = np.round(np.sqrt(anchor_area/anchor_ratios))
    # array([181., 128.,  91.])

    height = np.round(width * anchor_ratios)
    # array([ 90., 128., 182.])

    anchor_size = np.stack((width, height), axis=-1)
    # anchors = array([[181., 90.],[128., 128.],[91., 182.]])
    return anchor_size  # (3, 2)


def get_conners_coord(anchor_size):
    # anchor_size 大小为(9, 2)，包含9个anchor的width和height
    width = anchor_size[:, 0]  # 9个anchor的width
    height = anchor_size[:, 1]  # 9个anchor的height

    # 分别计算四点坐标
    x_min = np.round(0 - 0.5 * width)
    y_min = np.round(0 - 0.5 * height)
    x_max = np.round(0 + 0.5 * width)
    y_max = np.round(0 + 0.5 * height)

    anchor_conner = np.stack((x_min, y_min, x_max, y_max), axis=-1)

    # (9, 4)
    return anchor_conner


def all_anchor_conner(image_width, image_height, stride=cfg.feat_strides):

    bias_anchor_conner = generate_anchor(
        cfg.anchor_scales, cfg.anchor_ratios)
    # 大小为(9, 4)，生成feature_map第一个特征点对应的9个anchor的4个坐标

    stride = np.float(stride)  # vgg-16经过了4个步长为2的池化层，stride为2的4次方
    image_width = np.float(image_width)
    image_height = np.float(image_height)

    # np.ceil 返回大于等于该值的最小整数
    map_width = np.ceil(image_width/stride)  # feature_map 的宽度
    map_height = np.ceil(image_height/stride)  # feature_map 的长度

    # feature_map 的像素点个数（深度为1）
    total_pixel = (map_width*map_height).astype(np.int)

    offset_x = np.arange(map_width) * stride
    offset_y = np.arange(map_height) * stride

    x, y = np.meshgrid(offset_x, offset_y)
    x = np.reshape(x, -1) + int(stride/2)
    y = np.reshape(y, -1) + int(stride/2)

    center_point = np.stack((x, y, x, y), axis=-1)  # 每个anchor中心的坐标
    center_point = np.transpose(np.reshape(
        center_point, [1, total_pixel, 4]), (1, 0, 2))
    # (total_pixel, 4) -> (total_pixel, 1, 4)

    all_anchor_conners = center_point + bias_anchor_conner  # 对应anchor在缩放图的真实坐标
    all_anchor_conners = np.reshape(all_anchor_conners, [-1, 4])

    return np.array(all_anchor_conners).astype(np.float32)


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    image = np.ones(shape=(600, 800, 3), dtype=np.int)

    anchors = all_anchor_conner(600, 800)
    print(anchors.shape)

    for i, anchor in enumerate(anchors):
        cv2.rectangle(image, (anchor[0], anchor[1]),
                      (anchor[2], anchor[3]), color=(0, 120, 0))

        if i % 9 == 0:

            image = np.clip(image, 0, 255)

            plt.imshow(image)
            plt.show()
