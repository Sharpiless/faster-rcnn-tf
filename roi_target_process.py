# label_anchor.py
import numpy as np
import config as cfg


def roi_labels_process(true_boxes, true_classes, rois, num_classes):

    labels, IOUs = labels_generate(
        true_boxes, true_classes, rois, num_classes)

    batch_targets, batch_labels = gain_targets(
        rois, labels, true_boxes, num_classes, true_classes, IOUs
    )

    batch_targets, inside_w = _cls_bais_process(
        batch_targets, batch_labels)

    batch_labels = batch_labels.astype(np.float32)
    batch_targets = batch_targets.astype(np.float32)
    inside_w = inside_w.astype(np.float32)

    return batch_labels, batch_targets, inside_w


def labels_generate(true_boxes, true_classes, holdon_anchor, num_classes):

    true_classes = true_classes.astype(np.int32)

    total_anchors_num = holdon_anchor.shape[0]

    labels = np.zeros((total_anchors_num, num_classes))

    IOUs = calculate_IOU(holdon_anchor, true_boxes)

    max_target = np.argmax(IOUs, axis=1)

    object_labels = true_classes[max_target]
    rows = np.arange(total_anchors_num, dtype=np.int32)
    object_labels = object_labels.astype(np.int32)
    labels[rows, object_labels] = 1
    labels[:, 0] = -1

    max_IOUs = IOUs.max(axis=1)
    neg_index = np.where(max_IOUs <= cfg.overlaps_min)[0]
    labels[neg_index, 0] = 1
    labels[neg_index, 1:] = 0

    pos_index = np.where(max_IOUs >= cfg.overlaps_max)[0]
    labels[pos_index, 0] = 0

    max_IOUs_arg = IOUs.argmax(axis=0)
    labels[max_IOUs_arg, 0] = 0
    labels[max_IOUs_arg, true_classes] = 1

    return labels, IOUs


def calculate_IOU(holdon_anchor, true_boxes):

    num_true = true_boxes.shape[0]  # 真值框的个数 m
    num_holdon = holdon_anchor.shape[0]  # 候选框的个数（已删去越界的样本）n

    IOU_s = np.zeros((num_true, num_holdon), dtype=np.float)  # (m, n)

    for i in range(num_true):
        # 注：每个gt_box的坐标是 x_min, y_min, x_max, y_max
        # 注：并且坐标是缩放图片上的坐标
        lx = true_boxes[i, 2] - true_boxes[i, 0]
        ly = true_boxes[i, 3] - true_boxes[i, 1]
        true_area = lx * ly  # 真值框的面积大小

        for j in range(num_holdon):
            len_w = min(true_boxes[i, 2], holdon_anchor[j, 2]) - \
                max(true_boxes[i, 0], holdon_anchor[j, 0])
            # 重叠区域的宽度（如果小于零，则一定不重合）
            if len_w > 0:
                len_h = min(true_boxes[i, 3], holdon_anchor[j, 3]) - \
                    max(true_boxes[i, 1], holdon_anchor[j, 1])
                # 重叠区域的高度（如果小于零，则一定不重合）
                if len_h > 0:
                    t_x = holdon_anchor[j, 2] - holdon_anchor[j, 0]
                    t_y = holdon_anchor[j, 3] - holdon_anchor[j, 1]

                    target_area = t_x*t_y
                    overlap_area = len_w*len_h
                    IOU = overlap_area / \
                        float(true_area + target_area - overlap_area)

                    IOU_s[i, j] = IOU

    return np.transpose(IOU_s)  # (n, m) 转置矩阵


def gain_targets(anchors, labels, true_boxes, num_classes, true_classes, IOUs):
    
    batch_targets= _sample(
        anchors, labels, true_boxes, num_classes, true_classes, IOUs
    )

    batch_labels = labels.reshape(-1, 21)
    batch_targets = batch_targets.reshape(-1, 4)

    return batch_targets, batch_labels


def _sample(anchors, labels, true_boxes, num_classes, true_classes, IOUs):

    true_ass = IOUs.argmax(axis=1)

    batch_targets = _compute_targets(
        anchors, true_boxes[true_ass, :], labels[:, 0])

    return batch_targets


def _compute_targets(anchors, true_boxes, obj_labels):
    targets = bbox_transform(anchors, true_boxes)
    return targets.astype(np.float32, copy=False)


def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def _get_targets_labels(bbox_target_data, num_classes):
    clss = bbox_target_data
    bbox_inside_weights = np.zeros((clss.size, 4), dtype=np.float32)

    pos_index = np.where(clss > 0)[0]

    for index in pos_index:

        bbox_inside_weights[index, :] = cfg.roi_input_inside_weight

    return bbox_inside_weights


def coord_transform(anchors, pred_bias):

    anchor_x = (anchors[:, 2]+anchors[:, 0]) * 0.5
    anchor_y = (anchors[:, 3]+anchors[:, 1]) * 0.5
    acnhor_w = anchors[:, 2]-anchors[:, 0]+1.0
    acnhor_h = anchors[:, 3]-anchors[:, 1]+1.0

    boxes_x = pred_bias[:, 0]*acnhor_w + anchor_x
    boxes_y = pred_bias[:, 1]*acnhor_h + anchor_y
    boxes_w = np.exp(pred_bias[:, 2])*acnhor_w
    boxes_h = np.exp(pred_bias[:, 3])*acnhor_h

    coord_x1 = boxes_x - boxes_w*0.5
    coord_y1 = boxes_y - boxes_h*0.5
    coord_x2 = boxes_x + boxes_w*0.5
    coord_y2 = boxes_y + boxes_h*0.5

    proposals = np.stack([coord_x1, coord_y1, coord_x2, coord_y2], axis=1)

    return proposals


def clip_boxes(boxes, img_width):

    b0 = np.maximum(np.minimum(boxes[:, 0], img_width - 1), 0.0)
    b1 = np.maximum(np.minimum(boxes[:, 1], img_width - 1), 0.0)
    b2 = np.maximum(np.minimum(boxes[:, 2], img_width - 1), 0.0)
    b3 = np.maximum(np.minimum(boxes[:, 3], img_width - 1), 0.0)

    return np.stack([b0, b1, b2, b3], axis=1)


def _cls_bais_process(batch_targets, batch_labels):

    num_classes = len(cfg.CLASSES)

    targets = np.zeros(shape=[batch_targets.shape[0], 4*num_classes])
    w = np.zeros(shape=[batch_targets.shape[0], 4*num_classes])
    keep_index = np.where(batch_labels[:, 0]==0)[0]

    for index in keep_index:

        begin = np.argmax(batch_labels[index])*4
        end = begin + 4

        targets[index, begin:end] = batch_targets[index]
        w[index, begin:end] = cfg.roi_input_inside_weight
    
    return targets, w

def _cls_bais_deprocess(batch_targets, batch_labels):
    
    num_classes = len(cfg.CLASSES)

    targets = np.zeros(shape=[batch_targets.shape[0], 4])

    for index in range(batch_targets.shape[0]):

        begin = np.argmax(batch_labels[index])*4
        end = begin + 4

        targets[index] = batch_targets[index, begin:end]
    
    return targets

if __name__ == "__main__":

    from read_data import im_reader
    import matplotlib.pyplot as plt
    from produce_anchor import all_anchor_conner
    from show_result import sample_image

    reader = im_reader(is_training=True)

    sample = reader.generate()

    image = sample['image']  # resize过的
    true_boxes = sample['box']

    _, image_height, image_width, _ = image.shape

    all_anchor_conners = all_anchor_conner(image_width, image_height)
    # [x_min, y_min, x_max, y_max]
    # sample = {'image': image, 'scale': scale, 'class': true_label,
    #             'box': true_box, 'image_path': image_path}

    batch_labels, batch_targets, inside_w = roi_labels_process(
        sample['box'], sample['class'], all_anchor_conners, len(cfg.CLASSES))

    print(batch_labels.shape)
    print(batch_targets.shape)
    print(inside_w.shape, np.sum(inside_w))
    # sample = {'image': image, 'class': true_label,
    #             'true_box': true_box, 'pred_box': pred_box}

    keep_index = np.where(batch_labels[:, 0] == 0)[0]
    keep_bias = batch_targets[keep_index]
    
    labels = batch_labels[keep_index].argmax(axis=1)
    keep_bias = _cls_bais_deprocess(keep_bias, batch_labels[keep_index])
    proposals = coord_transform(all_anchor_conners[keep_index], keep_bias)

    image = sample_image({'image': sample['image'][0],
                          'true_box': all_anchor_conners[keep_index],
                          'pred_box': proposals,
                          'class': np.zeros(all_anchor_conners[keep_index].shape[0]),
                          'pred_class': labels})