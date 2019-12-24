import tensorflow as tf
from network import Net
from rpn_loss_layer import RPN_loss
from roi_loss_layer import ROI_loss
from read_data import im_reader
import config as cfg
import os
import numpy as np


class Train(object):
    def __init__(self):

        self.reader = im_reader(is_training=True)

        self.net = Net()

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
