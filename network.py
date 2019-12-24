from tensorflow.contrib.layers import xavier_initializer
import tensorflow as tf
import numpy as np
import config as cfg
from produce_anchor import all_anchor_conner
from rpn_target_process import rpn_labels_process
from roi_target_process import roi_labels_process


class Net(object):
    def __init__(self, is_training=True):

        self.is_training = is_training

        self.batch_num = cfg.IMAGE_BATCH

        self.feat_stride = cfg.feat_strides

        self.num_anchor = len(cfg.anchor_scales) * len(cfg.anchor_ratios)

        self.anchor_batch = cfg.anchor_batch  # 一张图片取得有效anchor的数量(256)
        self.num_classes = len(cfg.CLASSES)  # 总类别的数量

        self.images = tf.placeholder(
            tf.float32, shape=[cfg.IMAGE_BATCH, None, None, 3])
        self.true_boxes = tf.placeholder(tf.float32, shape=[None, 4])
        self.true_classes = tf.placeholder(tf.float32, shape=[None, ])
        self.image_width = tf.placeholder(tf.float32)
        self.image_height = tf.placeholder(tf.float32)

        self.rpn_predictions = {}

        self.roi_predictions = {}

        self.build_network()

    def build_network(self):

        self.anchors = tf.numpy_function(
            all_anchor_conner, [self.image_width, self.image_height],
            tf.float32
        )

        self.net_feature = self.vgg16(self.images)

        self.rpn_predictions = self.rpn_net(self.net_feature, self.num_anchor)

        self.rpn_bk_score = tf.nn.softmax(self.rpn_predictions['rpn_bk_score'])
        self.rpn_coord_bias = self.rpn_predictions['rpn_coord_bias']

        self.proposals = self.coord_transform(
            self.anchors, self.rpn_coord_bias)
        self.proposals = self.clip_boxes(
            self.proposals, self.image_width, self.image_height)

        if self.is_training:
            keep_anchors, rpn_labels, rpn_targets, rpn_inside_w, self.roi_index, self.rois = tf.numpy_function(
                rpn_labels_process,
                [self.true_boxes, self.anchors, self.proposals],
                [tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.int32, tf.float32]
            )

            self.rpn_predictions['rpn_bk_label'] = rpn_labels
            self.rpn_predictions['rpn_targets'] = rpn_targets
            self.rpn_predictions['rpn_inside_w'] = rpn_inside_w
            self.rpn_predictions['rpn_index'] = self.roi_index

            roi_labels, roi_targets, roi_inside_w = tf.numpy_function(
                roi_labels_process,
                [self.true_boxes, self.true_classes, self.rois, self.num_classes],
                [tf.float32, tf.float32, tf.float32]
            )

            self.roi_predictions['roi_cls_label'] = roi_labels
            self.roi_predictions['roi_targets'] = roi_targets
            self.roi_predictions['roi_inside_w'] = roi_inside_w
            batch_inds = tf.to_float(tf.zeros(shape=(cfg.anchor_batch, 1)))

        else:
            self.rpn_bk_score = tf.argmax(self.rpn_bk_score, axis=1)
            self.roi_index = tf.squeeze(
                tf.where(tf.equal(self.rpn_bk_score, 1)))
            self.rois = tf.gather(self.proposals, self.roi_index)
            self.rois_num = tf.shape(self.roi_index)[0]
            batch_inds = tf.to_float(tf.zeros(shape=(self.rois_num, 1)))

        blob = tf.concat([batch_inds, self.rois], axis=1)

        pool5 = self._crop_pool_layer(
            self.net_feature, blob)

        fc7 = self._head_to_tail(pool5, self.is_training)

        self.cls_pred, self.roi_bias = self._region_classification(
            fc7, self.is_training)

        roi_coord = self.coord_transform(self.rois, self.roi_bias)
        roi_coord = self.clip_boxes(
            roi_coord, self.image_width, self.image_height)

    def _region_classification(self, fc7, is_training):

        cls_scores = tf.layers.dense(fc7, self.num_classes,
                                     kernel_initializer=tf.random_normal_initializer(
                                         mean=0.0, stddev=0.01),
                                     trainable=is_training,
                                     activation=None)

        cls_scores = tf.reshape(
            cls_scores, shape=[self.batch_num, -1, self.num_classes])

        reg_pred = tf.layers.dense(fc7, self.num_classes*4,
                                   kernel_initializer=tf.random_normal_initializer(
                                       mean=0.0, stddev=0.01),
                                   trainable=is_training,
                                   activation=None)

        reg_pred = tf.reshape(
            reg_pred, shape=[self.batch_num, -1, self.num_classes*4])

        self.roi_predictions['cls_score'] = tf.squeeze(cls_scores)
        self.roi_predictions['reg_bias'] = tf.squeeze(reg_pred)

        return tf.nn.softmax(cls_scores), reg_pred

    def _crop_pool_layer(self, bottom, rois):

        batch_ids = tf.reshape(rois[:, 0], [-1])

        bottom_shape = tf.shape(bottom)  # (256, 19, 19, 512)

        height = (tf.to_float(
            bottom_shape[1]) - 1.) * np.float32(self.feat_stride)

        width = (tf.to_float(bottom_shape[2]) -
                 1.) * np.float32(self.feat_stride)

        x1 = tf.slice(rois, [0, 1], [-1, 1]) / width
        y1 = tf.slice(rois, [0, 2], [-1, 1]) / height
        x2 = tf.slice(rois, [0, 3], [-1, 1]) / width
        y2 = tf.slice(rois, [0, 4], [-1, 1]) / height

        bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
        bboxes = tf.reshape(bboxes, (-1, 4))

        pre_pool_size = cfg.POOLING_SIZE * 2  # 7*2

        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(
            batch_ids), [pre_pool_size, pre_pool_size])

        return max_pool2d(crops, k=2, strides=2)

    def _head_to_tail(self, pool5, is_training):

        pool5_flat = tf.layers.flatten(pool5)

        fc6 = tf.layers.dense(pool5_flat, 4096)

        if is_training:
            fc6 = tf.layers.dropout(fc6, rate=0.5)

        fc7 = tf.layers.dense(fc6, 4096)
        if is_training:
            fc7 = tf.layers.dropout(fc6, rate=0.5)

        return fc7

    def rpn_net(self, input_feature_map, num_anchor):

        rpn_feature = conv2d(input_feature_map, filters=512)

        bk_score = conv2d(rpn_feature, 2*num_anchor,
                          k_size=1, activation=None)
        bias = conv2d(rpn_feature, 4*num_anchor,
                      k_size=1, activation=None)

        bk_score = tf.reshape(bk_score, [-1, 2])
        bias = tf.reshape(bias, [-1, 4])

        return {'rpn_bk_score': bk_score, 'rpn_coord_bias': bias}

    def coord_transform(self, all_anchors, rpn_anchor):

        anchors = tf.cast(all_anchors, tf.float32)

        anchor_x = tf.add(anchors[:, 2], anchors[:, 0]) * 0.5
        anchor_y = tf.add(anchors[:, 3], anchors[:, 1]) * 0.5
        acnhor_w = tf.subtract(anchors[:, 2], anchors[:, 0])+1.0
        acnhor_h = tf.subtract(anchors[:, 3], anchors[:, 1])+1.0

        rpn_anchor = tf.squeeze(rpn_anchor)

        boxes_x = rpn_anchor[:, 0]*acnhor_w + anchor_x
        boxes_y = rpn_anchor[:, 1]*acnhor_h + anchor_y
        boxes_w = tf.exp(rpn_anchor[:, 2])*acnhor_w
        boxes_h = tf.exp(rpn_anchor[:, 3])*acnhor_h

        coord_x1 = boxes_x - boxes_w*0.5
        coord_y1 = boxes_y - boxes_h*0.5
        coord_x2 = boxes_x + boxes_w*0.5
        coord_y2 = boxes_y + boxes_h*0.5

        proposals = tf.stack([coord_x1, coord_y1, coord_x2, coord_y2], axis=1)

        return proposals

    def clip_boxes(self, boxes, img_width, img_height):

        img_width = tf.cast(img_width, tf.float32)
        img_height = tf.cast(img_height, tf.float32)

        b0 = tf.maximum(tf.minimum(boxes[:, 0], img_width - 1), 0.0)
        b1 = tf.maximum(tf.minimum(boxes[:, 1], img_height - 1), 0.0)
        b2 = tf.maximum(tf.minimum(boxes[:, 2], img_width - 1), 0.0)
        b3 = tf.maximum(tf.minimum(boxes[:, 3], img_height - 1), 0.0)

        return tf.stack([b0, b1, b2, b3], axis=1)

    def vgg16(self, input_image):

        net = conv2d(input_image, filters=64)
        net = conv2d(net, filters=64)

        net = max_pool2d(net, k=2, strides=2)

        net = conv2d(net, filters=128)
        net = conv2d(net, filters=128)

        net = max_pool2d(net, k=2, strides=2)

        net = conv2d(net, filters=256)
        net = conv2d(net, filters=256)
        net = conv2d(net, filters=256)

        net = max_pool2d(net, k=2, strides=2)

        net = conv2d(net, filters=512)
        net = conv2d(net, filters=512)
        net = conv2d(net, filters=512)

        net = max_pool2d(net, k=2, strides=2)

        net = conv2d(net, filters=512)
        net = conv2d(net, filters=512)
        net = conv2d(net, filters=512)

        return net

    def pred_reshape(self, pred, num_anchors, num_outs):
        pred = tf.reshape(pred, [self.batch_num, num_anchors, num_outs])
        return pred


def conv2d(x, filters, k_size=3, strides=1, padding='SAME', dilation=[1, 1], activation=tf.nn.leaky_relu):

    return tf.layers.conv2d(x, filters, kernel_size=[k_size, k_size], strides=[strides, strides],
                            dilation_rate=dilation, padding=padding, activation=activation,
                            kernel_initializer=xavier_initializer(), use_bias=True, data_format='channels_last')


def max_pool2d(x, k, strides, padding='SAME'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=k, strides=strides, padding=padding)


def average_pool2d(x, k, strides, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=k, strides=strides, padding=padding)


def batch_normal(x):
    return tf.layers.batch_normalization(x, training=True)


if __name__ == "__main__":

    from read_data import im_reader
    from rpn_target_process import coord_transform, clip_boxes
    from roi_target_process import _cls_bais_deprocess
    from show_result import sample_image
    import os

    reader = im_reader(is_training=True)
    net = Net()

    data = reader.generate()
    saver = tf.train.Saver()

    _, height, width, _ = data['image'].shape

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(os.getcwd())
        if ckpt and ckpt.model_checkpoint_path:
            # 如果保存过模型，则在保存的模型的基础上继续训练
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model Reload Successfully!')

        feed_dcit = {
            net.images: data['image'],
            net.image_height: height,
            net.image_width: width,
            net.true_boxes: data['box'],
            net.true_classes: data['class']
        }

        rpn, roi, proposals = sess.run(
            [net.rpn_predictions, net.roi_predictions, net.rois], feed_dcit)

    for key, value in rpn.items():
        print(key, value.shape)

    for key, value in roi.items():
        print(key, value.shape)

    print('target_num', np.sum(rpn['rpn_bk_label']))

    rpn_index = rpn['rpn_index']
    roi_targets = roi['reg_bias']
    roi_cls_label = roi['cls_score']

    # proposals = proposals[rpn_index]

    pos_index = np.where(roi['roi_cls_label'][:, 0] == 0)[0]

    roi_targets = roi_targets[pos_index]
    roi_cls_label = roi_cls_label[pos_index]
    print(pos_index)
    roi_targets = _cls_bais_deprocess(roi_targets, roi_cls_label)
    print(roi_targets)

    keep_proposals = proposals[pos_index]

    boxes = coord_transform(keep_proposals, roi_targets)
    print(boxes)
    boxes = clip_boxes(boxes, width, height)

    image = sample_image({'image': data['image'][0],
                          'true_box': keep_proposals,
                          'pred_box': boxes,
                          'class': np.zeros(keep_proposals.shape[0]),
                          'pred_class': roi_cls_label.argmax(axis=1)})
