# show_result.py
from read_data import im_reader
import cv2
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import config as cfg


def sample_image(sample, show_num=20):

    # sample = {'image': image, 'class': true_label,
    #             'true_box': true_box, 'pred_box': pred_box}

    true_boxes = np.round(sample['true_box'])
    pred_boxes = np.round(sample['pred_box'])

    classes = sample['class']
    pred_cls = sample['pred_class']

    image = (sample['image'] + cfg.PIXEL_MEANS).astype(np.int)

    n_true_boxes = true_boxes.shape[0]
    n_pred_boxes = pred_boxes.shape[0]

    cfg_classes = cfg.CLASSES

    class_to_ind = dict(zip(cfg_classes, range(len(cfg_classes))))

    for index in range(n_true_boxes):

        x_min = int(true_boxes[index, 0])
        y_min = int(true_boxes[index, 1])
        x_max = int(true_boxes[index, 2])
        y_max = int(true_boxes[index, 3])

        cv2.rectangle(image, (x_min, y_min),
                      (x_max, y_max), (0, 255, 0), thickness=2)

        class_num = classes[index]

        for key, value in class_to_ind.items():
            if class_num == value:
                class_ = key
                break

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, class_, (x_min+20, y_min+20),
                    font, 1, (255, 0, 255), thickness=1)

    true_image = image.copy()

    i = 0
    pre_list = list(range(n_pred_boxes))
    shuffle(pre_list)
    for index in pre_list:
        i += 1
        '''if i > show_num:
            break'''

        x_min = int(pred_boxes[index, 0])
        y_min = int(pred_boxes[index, 1])
        x_max = int(pred_boxes[index, 2])
        y_max = int(pred_boxes[index, 3])

        cv2.rectangle(image, (x_min, y_min),
                      (x_max, y_max), (255, 0, 0), thickness=2)

        class_num = pred_cls[index]

        for key, value in class_to_ind.items():
            if class_num == value:
                class_ = key
                break

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, class_, (x_min+20, y_min+20),
                    font, 1, (0, 125, 0), thickness=1)

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(true_image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.axis('off')

    plt.show()


if __name__ == "__main__":

    reader = im_reader(is_training=True)

    sample = reader.generate()

    image = sample_image({'image': sample['image'],
                          'true_box': sample['box'],
                          'pred_box': sample['box'],
                          'class': sample['class'],
                          'pred_class': sample['class']})

