import tensorflow as tf
import config as cfg


class RPN_loss(object):
    def __init__(self, pred_bias, pred_bk_score, targets, true_bk_labels, keep_index):

        self.pred_bias = pred_bias
        self.pred_bk_score = pred_bk_score
        self.targets = targets
        self.true_bk_labels = true_bk_labels
        self.keep_index = keep_index

        self.keep_pred_bias = tf.gather(self.pred_bias, self.keep_index)
        self.keep_pred_bk_score = tf.gather(
            self.pred_bk_score, self.keep_index)

    def add_loss(self):

        self.cls_loss = self._bk_loss(
            self.keep_pred_bk_score, self.true_bk_labels)
        self.reg_loss = self._smooth_l1_loss(self.keep_pred_bias, self.targets)

        return self.cls_loss, self.reg_loss

    def _bk_loss(self, pred_bk_score, true_bk_labels):
        true_bk_labels = tf.one_hot(tf.to_int32(true_bk_labels), depth=2)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=true_bk_labels, logits=pred_bk_score
        )
        return tf.reduce_mean(loss)

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = box_diff
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

    rpn_loss_obj = RPN_loss(
        net.rpn_predictions['rpn_coord_bias'],
        net.rpn_predictions['rpn_bk_score'],
        net.rpn_predictions['rpn_targets'],
        net.rpn_predictions['rpn_bk_label'],
        net.rpn_predictions['rpn_index']
    )
    'pred_bias, pred_bk_score, targets, true_bk_labels, keep_index'

    bk_loss, reg_loss = rpn_loss_obj.add_loss()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        feed_dcit = {
            net.images: data['image'],
            net.image_height: height,
            net.image_width: width,
            net.true_boxes: data['box'],
            net.true_classes: data['class']
        }

        loss = sess.run([bk_loss, reg_loss], feed_dcit)

    print(loss)