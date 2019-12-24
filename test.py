import cv2
import os
import tensorflow as tf
import numpy as np
import config as cfg
import matplotlib.pyplot as plt
from read_data import im_reader
from network import Net
from rpn_target_process import coord_transform, clip_boxes
from cpu_nms import py_cpu_nms


class Test(object):
    def __init__(self):

        self.reader = im_reader(is_training=False)
        self.net = Net(is_training=False)
        self.saver = tf.train.Saver()
        self.num_classes = len(cfg.CLASSES)

    def show(self, num=1):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(os.getcwd())
            if ckpt and ckpt.model_checkpoint_path:
                # 如果保存过模型，则在保存的模型的基础上继续训练
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model Reload Successfully!')

            for _ in range(num):

                data = self.reader.generate()
                _, height, width, _ = data['image'].shape

                feed_dcit = {
                    self.net.images: data['image'],
                    self.net.image_height: height,
                    self.net.image_width: width
                }

                bk_score, rpn_bias, cls_score, roi_bias, anchors = sess.run(
                    [self.net.rpn_bk_score, self.net.rpn_coord_bias,
                     self.net.cls_pred, self.net.roi_bias, self.net.anchors],
                    feed_dict=feed_dcit
                )

                roi_bias = np.squeeze(roi_bias)
                cls_score = np.squeeze(cls_score)

                for value in [bk_score, rpn_bias, cls_score, roi_bias, anchors]:
                    print(value.shape)
            
                cls_boxes = self.sample_boxes(
                    bk_score, rpn_bias, cls_score, roi_bias,
                    anchors, width, height)

                image = self.draw_image(data['image'][0], cls_boxes)
                image += cfg.PIXEL_MEANS
                image = image.astype(np.int32)

                plt.imshow(image)
                plt.axis('off')
                plt.show()

    def draw_image(self, image, cls_boxes, color=(0, 0, 120), font=cv2.FONT_HERSHEY_SIMPLEX):

        for key, value in cls_boxes.items():

            keep = py_cpu_nms(value[0], value[1], 0.5)
            boxes = value[0][keep].astype(np.int)

            for box in boxes:

                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]

                cv2.rectangle(image, (x1, y1), (x2, y2), color=color)
                cv2.putText(image, key, (x1+20, y1+20),
                            font, 1, color=(255, 0, 255), thickness=1)

        return image

    def sample_boxes(self, bk_score, rpn_bias, cls_score, roi_bias, anchors, width, height):

        keep_index = np.where(bk_score == 1)[0]

        rpn_bias = rpn_bias[keep_index]
        anchors = anchors[keep_index]

        pred_boxes = coord_transform(anchors, rpn_bias)
        # pred_boxes = coord_transform(anchors, roi_bias)
        pred_boxes = clip_boxes(pred_boxes, width, height)

        cls_boxes = {}

        for cls_num in range(1, self.num_classes):

            single_cls = cls_score[:, cls_num]
            pos_index = np.where(single_cls > cfg.test_thresh)[0]

            if pos_index.size > 0:
                cls_boxes[cfg.CLASSES[cls_num]] = [
                    pred_boxes[pos_index], cls_score[pos_index, cls_num]]
                print(cfg.CLASSES[cls_num], pos_index.size)

        return cls_boxes

if __name__ == "__main__":
    
    test_obj = Test()
    test_obj.show()