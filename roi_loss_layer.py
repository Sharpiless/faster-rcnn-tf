import tensorflow as tf
import config as cfg


class ROI_loss(object):
    def __init__(self, pred_bias, pred_cls_score, targets, true_cls_labels,inside_w):

        self.pred_bias = pred_bias
        self.pred_cls_score = pred_cls_score
        self.targets = targets
        self.true_cls_labels = true_cls_labels
        self.inside_w = inside_w

    def add_loss(self):

        self.cls_loss = self._cls_loss(
            self.pred_cls_score, self.true_cls_labels)
        self.reg_loss = self._smooth_l1_loss(self.pred_bias, self.targets, self.inside_w)

        return self.cls_loss, self.reg_loss*10

    def _cls_loss(self, pred_cls_score, true_cls_labels):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=true_cls_labels, logits=pred_cls_score
        )
        return tf.reduce_mean(loss)

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = box_diff * weights
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(
            tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
            + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = in_loss_box
        loss = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss


if __name__ == "__main__":

    from network import Net
    from read_data import im_reader

    reader = im_reader(is_training=True)
    net = Net()

    data = reader.generate()

    _, height, width, _ = data['image'].shape

    roi_loss_obj = ROI_loss(
        net.roi_predictions['reg_bias'],
        net.roi_predictions['cls_score'],
        net.roi_predictions['roi_targets'],
        net.roi_predictions['roi_cls_label'],
        net.roi_predictions['roi_inside_w']
    )
    'pred_bias, pred_cls_score, targets, true_cls_labels,inside_w'

    cls_loss, reg_loss = roi_loss_obj.add_loss()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        feed_dcit = {
            net.images: data['image'],
            net.image_height: height,
            net.image_width: width,
            net.true_boxes: data['box'],
            net.true_classes: data['class']
        }

        loss = sess.run([cls_loss, reg_loss], feed_dcit)

    print(loss)
