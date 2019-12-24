import tensorflow as tf
from A import Net
from rpn_loss_layer import RPN_loss
from roi_loss_layer import ROI_loss
from read_data import im_reader
import config as cfg
import os
import numpy as np


class Train(object):
    def __init__(self):

        self.reader = im_reader(is_training=True)

        self.net = Net(is_training=True)

        self.rpn_loss_obj = RPN_loss(
            self.net.rpn_predictions['rpn_coord_bias'],
            self.net.rpn_predictions['rpn_bk_score'],
            self.net.rpn_predictions['rpn_targets'],
            self.net.rpn_predictions['rpn_bk_label'],
            self.net.rpn_predictions['rpn_index']
        )

        self.roi_loss_obj = ROI_loss(
            self.net.roi_predictions['reg_bias'],
            self.net.roi_predictions['cls_score'],
            self.net.roi_predictions['roi_targets'],
            self.net.roi_predictions['roi_cls_label'],
            self.net.roi_predictions['roi_inside_w']
        )

        self.rpn_bk_loss, self.rpn_reg_loss = self.rpn_loss_obj.add_loss()
        self.roi_cls_loss, self.roi_reg_loss = self.roi_loss_obj.add_loss()

        self.total_loss = self.rpn_bk_loss + self.rpn_reg_loss + \
            self.roi_cls_loss+self.roi_reg_loss

        self.loss_dict = {
            'rpn_bk_loss': self.rpn_bk_loss,
            'rpn_reg_loss': self.rpn_reg_loss,
            'roi_cls_loss': self.roi_cls_loss,
            'roi_reg_loss': self.roi_reg_loss,
            'total_loss': self.total_loss
        }

        if os.path.exists('result.txt'):
            with open('result.txt') as f:
                self.loss = eval(f.read())
        else:
            self.loss = []

    def train_model(self):

        self.saver = tf.train.Saver()

        optimizer = tf.train.MomentumOptimizer(cfg.LEARNING_RATE, cfg.momentum)
        train_op = optimizer.minimize(self.total_loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(os.getcwd())
            if ckpt and ckpt.model_checkpoint_path:
                # 如果保存过模型，则在保存的模型的基础上继续训练
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model Reload Successfully!')

            for step in range(cfg.MAX_STEP+1):

                losses_dict = {
                    'rpn_bk_loss': [],
                    'rpn_reg_loss': [],
                    'roi_cls_loss': [],
                    'roi_reg_loss': [],
                    'total_loss': []
                }

                for epoch in range(cfg.MAX_EPOCH):

                    data = self.reader.generate()
                    print('epoch:{}'.format(epoch), end='\r')
                    _, height, width, _ = data['image'].shape

                    feed_dcit = {
                        self.net.images: data['image'],
                        self.net.image_height: height,
                        self.net.image_width: width,
                        self.net.true_boxes: data['box'],
                        self.net.true_classes: data['class']
                    }

                    _, losses = sess.run(
                        [train_op, self.loss_dict], feed_dict=feed_dcit)

                    for key in losses_dict.keys():
                        losses_dict[key].append(losses[key])

                self.saver.save(sess, cfg.MODEL_PATH)

                mean_total_loss = np.mean(np.array(losses_dict['total_loss']))
                mean_rpn_bk_loss = np.mean(
                    np.array(losses_dict['rpn_bk_loss']))
                mean_rpn_reg_loss = np.mean(
                    np.array(losses_dict['rpn_reg_loss']))
                mean_roi_cls_loss = np.mean(
                    np.array(losses_dict['roi_cls_loss']))
                mean_roi_reg_loss = np.mean(
                    np.array(losses_dict['roi_reg_loss']))

                print('step:{} loss:{} rpn_bk_loss:{} rpn_reg_loss:{} roi_cls_loss:{} roi_reg_loss:{}'.format(
                    step, mean_total_loss, mean_rpn_bk_loss, 
                    mean_rpn_reg_loss, mean_roi_cls_loss, mean_roi_reg_loss))
                
                self.loss.append([mean_total_loss, mean_rpn_bk_loss, 
                    mean_rpn_reg_loss, mean_roi_cls_loss, mean_roi_reg_loss])

                with open('result.txt', 'w') as f:
                    f.write(str(self.loss))
        
        self.show_loss()

    def show_loss(self):
        import matplotlib.pyplot as plt

        losses = np.array(self.loss)

        total_loss = losses[:, 0]
        rpn_bk_loss = losses[:, 1]
        rpn_reg_loss = losses[:, 2]
        roi_cls_loss = losses[:, 3]
        roi_reg_loss = losses[:, 4]

        plt.figure()

        plt.plot(total_loss, label='total_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid(alpha=0.4, linestyle=':')
        plt.legend(loc='upper left')
        plt.savefig('all_loss.png')

        plt.show()

        plt.plot(rpn_bk_loss, label='background_loss')
        plt.plot(rpn_reg_loss, label='rpn_reg_loss')
        plt.plot(roi_cls_loss, label='class_loss')
        plt.plot(roi_reg_loss, label='roi_reg_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid(alpha=0.4, linestyle=':')
        plt.legend(loc='upper left')
        plt.savefig('sub_loss.png')

        plt.show()
        

if __name__ == "__main__":

    train_obj = Train()

    train_obj.train_model()
