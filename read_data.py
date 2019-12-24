# read_data.py
import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import config as cfg


# 定义了一个im_reader类，从voc2012中读取图片信息
# 主要函数有两个：
# 1. self.pre_process() 预处理图片，即将图片信息提取并保存为kpl文件
# 2. self.generate() 用于加载并迭代图片

# 最终generate结果为: sample = {'image': image(-1, cfg.target_size, 1000, 3), 'scale': scale, 'class': true_label,
#                    'box': true_box, 'image_path': image_path}

# image 为缩放后的图片，大小为(-1, width, heigth, 3)，缩放比例为 scale
# class 为真值框的分类（一个代表类别的数字）
# box 为真值框的坐标（注：是在原始图像上的坐标）

class im_reader(object):

    def __init__(self, is_training):

        self.data_path = cfg.DATA_PATH

        self.is_training = is_training

        self.batch_size = 1

        self.target_size = cfg.target_size  # 把最小边缩放成cfg.target_size

        self.max_size = cfg.max_size  # 最大边不超过1000

        self.classes = cfg.CLASSES  # 类别列表

        self.pixel_means = cfg.PIXEL_MEANS  # 训练集三通道的均值列表

        # np.array([[[122.7717, 115.9465, 102.9801]]])

        self.class_to_ind = dict(
            zip(self.classes, range(len(self.classes))))  # 每个类别对应一个数字（包括背景0）

        self.cursor = 0  # 当前游标（在训练集上滑动）

        self.epoch = 1  # 当前的epoch

        self.true_labels = None  # 稍后保存图片信息
        # boxes:[n_object,4]保存object的坐标
        # gt_class:保存字符串标识的类别
        # imname:保存图片路径
        # filpped:是否翻转
        # image_size:[2]width和height
        # image_index:图片标号（名称）

        self.pre_process()

    def image_read(self, image_path):
        image = cv2.imread(image_path)  # 默认为BGR格式
        # 转换为RGB格式
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float)

    def load_one_info(self, name):
        # 处理单张图片，返回包含图片信息的dict
        # name是图片名，从对应的.xml文件读取信息
        # 注：坐标是对角坐标（x_min, y_min, x_max, y_max）
        filename = os.path.join(self.data_path, 'Annotations', name + '.xml')

        tree = ET.parse(filename)  # 使用ElementTree解析xml文件，返回一个tree类

        # 寻找object标签（包含name，pose，truncated，difficult，bndbox）
        objs = tree.findall('object')

        image_size = tree.find('size')  # 寻找size标签（包含图片的width，height，depth）

        # array([0., 0.], dtype=float32)
        size_info = np.zeros((2,), dtype=np.float32)

        size_info[0] = float(image_size.find('width').text)  # '500'

        size_info[1] = float(image_size.find('height').text)

        num_objs = len(objs)  # object的数量

        # boxes 坐标 (num_objs,4)个 dtype=np.uint16
        boxes = np.zeros((num_objs, 4), dtype=np.float32)

        # class 的数量num_objs个 dtype=np.int32 应该是groundtruth中读到的class
        true_classes = np.zeros((num_objs), dtype=np.int32)

        for ix, obj in enumerate(objs):

            bbox = obj.find('bndbox')

            # 注：VOC数据标注是按1为起始点，所以坐标要减一
            x_min = float(bbox.find('xmin').text) - 1

            y_min = float(bbox.find('ymin').text) - 1

            x_max = float(bbox.find('xmax').text) - 1

            y_max = float(bbox.find('ymax').text) - 1

            cls_ = self.class_to_ind[obj.find(
                'name').text.lower().strip()]  # 找到class对应的类别信息（'boat'），在class_to_ind中转换成数字

            # 注意boxes是一个np类的矩阵 大小为[num_objs,4]
            boxes[ix, :] = [x_min, y_min, x_max, y_max]

            # 将class信息存入gt_classses中，注意gt_classes也是一个np类的矩阵 大小为[num_objs] 是int值 对应于name
            true_classes[ix] = cls_  # 不是one_hot，而是一个字符串标识的类别

            image_path = os.path.join(
                self.data_path, 'JPEGImages', name + '.jpg')  # 图片路径

        # boxes:[n_object,4]保存object的坐标
        # gt_class:保存字符串标识的类别
        # image_path:保存图片路径
        # filpped:是否翻转
        # image_size:[2]width和height
        # image_index:图片标号（名称）
        return {'boxes': boxes, 'class': true_classes, 'image_path': image_path, 'flipped': False, 'image_size': size_info, 'image_index': name}

    def load_labels(self):
        is_training = 'train' if self.is_training else 'test'

        if not os.path.exists('dataset'):
            os.makedirs('dataset')

        pkl_file = os.path.join('dataset', is_training+'_labels.pkl')

        if os.path.isfile(pkl_file):  # 如果文件存在，则不需要重新写入文件
            print('Load labels from: '+str(pkl_file))
            with open(pkl_file, 'rb') as f:
                labels = pickle.load(f)

            return labels

        # 如果不存在，则从ImageSet中加载并保存
        print('Load labels from: '+str(cfg.ImageSets_PATH))
        if self.is_training:
            txt_path = os.path.join(cfg.ImageSets_PATH, 'Main', 'trainval.txt')
            # 这是用来存放训练集和测试集的列表的txt文件
        else:
            txt_path = os.path.join(cfg.ImageSets_PATH, 'Main', 'val.txt')

        with open(txt_path, 'r') as f:
            self.image_name = [x.strip() for x in f.readlines()]

        labels = []

        for name in self.image_name:
            # 包括objet box坐标信息 以及类别信息(转换成dict后的)
            true_label = self.load_one_info(name)
            labels.append(true_label)

        with open(pkl_file, 'wb') as f:
            pickle.dump(labels, f)

        print('Successfully saving '+is_training+'data to '+pkl_file)

        return labels

    def pre_process(self):
        # 初始化的时候已经运行
        true_labels = self.load_labels()  # 返回一个列表，列表元素是包含图片信息的dict

        np.random.shuffle(true_labels)

        self.true_labels = true_labels

    def resize_image(self, image):

        image -= self.pixel_means  # 均值归零

        image_shape = image.shape

        size_min = np.min(image_shape[:2])
        size_max = np.max(image_shape[:2])

        scale = float(self.target_size) / \
            float(size_min)  # cfg.target_size / 最短边

        image = cv2.resize(image, None, None, fx=scale,
                           fy=scale, interpolation=cv2.INTER_LINEAR)

        # 返回缩放后的图片和缩放比例
        return image, scale

    def generate(self):

        count = 0

        value = {}

        while count < self.batch_size:
            # 取一个batch
            image_path = self.true_labels[self.cursor]['image_path']  # 图片路径
            image = self.image_read(image_path)  # 读取RGB图片
            image, scale = self.resize_image(image)

            # 原图片中ground truth的大小
            true_box = self.true_labels[self.cursor]['boxes'] * scale

            true_label = self.true_labels[self.cursor]['class']

            count += 1

            self.cursor += 1

            if self.cursor >= len(self.true_labels):
                # 如果取完一遍，则打乱顺序并且游标归零
                np.random.shuffle(self.true_labels)

                self.cursor = 0
                self.epoch += 1

        value = {'image': np.array([image]), 'box': true_box, 'image_path': image_path,
                 'im_info': [image.shape[0], image.shape[1], scale], 'class':true_label}

        # 返回的image.shape=[batch,size,size,3] image_scale, gt_box.shape=[num_objs,4]
        return value


if __name__ == "__main__":
    from show_result import sample_image

    reader = im_reader(is_training=True)

    sample = reader.generate()

    image = sample_image({'image': sample['image'][0],
                          'true_box': sample['box'],
                          'pred_box': sample['box'],
                          'class': sample['class'],
                          'pred_class': sample['class']})