# label_anchor.py
import numpy as np
import config as cfg


# 0是负样本，1是正样本
def rpn_labels_process(true_boxes, anchors, proposals):

    labels, IOUs = labels_generate(
        true_boxes, anchors)

    batch_anchors, batch_labels, batch_targets, keep_index = gain_targets(
        anchors, labels, true_boxes, IOUs
    )

    batch_targets, inside_w = _cls_bais_process(
        batch_targets, batch_labels)

    batch_anchors = batch_anchors.astype(np.float32)
    batch_labels = batch_labels.astype(np.float32)
    batch_targets = batch_targets.astype(np.float32)
    inside_w = inside_w.astype(np.float32)
    keep_index = keep_index.astype(np.int32)
    rois = proposals[keep_index]

    return batch_anchors, batch_labels, batch_targets, inside_w, keep_index, rois


def labels_generate(true_boxes, holdon_anchor):

    total_anchors_num = holdon_anchor.shape[0]

    labels = np.empty((total_anchors_num, ))
    labels.fill(-1)

    IOUs = calculate_IOU(holdon_anchor, true_boxes)

    max_IOUs = IOUs.max(axis=1)

    neg_index = np.where(max_IOUs <= cfg.overlaps_min)[0]

    labels[neg_index] = 0

    pos_index = np.where(max_IOUs >= cfg.overlaps_max)[0]
    labels[pos_index] = 1

    max_IOUs_arg = IOUs.argmax(axis=0)
    labels[max_IOUs_arg] = 1

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


def gain_targets(anchors, labels, true_boxes, IOUs):

    pos_num = np.round(cfg.anchor_batch * cfg.dect_fg_rate)

    batch_labels, batch_anchors, batch_targets, keep_index = _sample(
        anchors, labels, true_boxes, pos_num, IOUs
    )

    batch_labels = batch_labels.reshape((-1, ))
    batch_anchors = batch_anchors.reshape((-1, 4))
    batch_targets = batch_targets.reshape((-1, 4))

    return batch_anchors, batch_labels, batch_targets, keep_index


def _sample(anchors, labels, true_boxes, pos_num, IOUs, anchor_batch=cfg.anchor_batch):

    bk_index = np.where(labels == 0)[0]
    pos_index = np.where(labels == 1)[0]
    true_ass = IOUs.argmax(axis=1)

    if pos_index.size > 0 and bk_index.size > 0:
        pos_num = min(pos_index.size, pos_num)
        pos_index = np.random.choice(
            pos_index, size=int(pos_num), replace=False)
        bk_num = anchor_batch - pos_num
        to_replace = bk_index.size < bk_num
        bk_index = np.random.choice(
            bk_index, size=int(bk_num), replace=to_replace)

    elif pos_index.size > 0:
        to_replace = pos_index.size < anchor_batch
        pos_index = np.random.choice(
            pos_index, size=int(anchor_batch), replace=to_replace)
        pos_num = anchor_batch

    elif bk_index.size > 0:
        to_replace = bk_index.size < anchor_batch
        bk_index = np.random.choice(bk_index, size=(
            anchor_batch), replace=to_replace)
        pos_num = 0

    keep_index = np.append(pos_index, bk_index)

    labels = labels[keep_index]
    labels[int(pos_num):] = 0
    anchors = anchors[keep_index]

    batch_targets = _compute_targets(
        anchors, true_boxes[true_ass[keep_index], :])

    return labels, anchors, batch_targets, keep_index


def _compute_targets(anchors, true_boxes):
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


def clip_boxes(boxes, img_width, img_height):

    b0 = np.maximum(np.minimum(boxes[:, 0], img_width - 1), 0.0)
    b1 = np.maximum(np.minimum(boxes[:, 1], img_height - 1), 0.0)
    b2 = np.maximum(np.minimum(boxes[:, 2], img_width - 1), 0.0)
    b3 = np.maximum(np.minimum(boxes[:, 3], img_height - 1), 0.0)

    return np.stack([b0, b1, b2, b3], axis=1)


def _cls_bais_process(batch_targets, batch_labels):

    num_classes = len(cfg.CLASSES)

    w = np.zeros(shape=[batch_targets.shape[0], 4])
    keep_index = np.where(batch_labels == 1)[0]

    for index in keep_index:

        w[index] = cfg.roi_input_inside_weight

    return batch_targets, w


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

    batch_anchors, batch_labels, batch_targets, inside_w, rois, keep_index = rpn_labels_process(
        sample['box'], all_anchor_conners, all_anchor_conners)

    print(batch_anchors.shape)
    print(batch_labels.shape)
    print(batch_targets.shape)
    print(inside_w.shape)
    # sample = {'image': image, 'class': true_label,
    #             'true_box': true_box, 'pred_box': pred_box}

    keep_index = np.where(batch_labels == 1)[0]
    keep_anchors = batch_anchors[keep_index]
    keep_bias = batch_targets[keep_index]
    keep_labels = batch_labels[keep_index]

    print(keep_index.size)

    proposals = coord_transform(keep_anchors, keep_bias)

    image = sample_image({'image': sample['image'][0],
                          'true_box': sample['box'],
                          'pred_box': proposals,
                          'class': sample['class'],
                          'pred_class': batch_labels})
