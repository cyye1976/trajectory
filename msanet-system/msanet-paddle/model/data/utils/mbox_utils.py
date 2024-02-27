import cv2
import numpy as np


def head_mbox2points(mboxes, heads):
    """
    将带有头部标注的五参数表示的定向边界框
    转化成点集表示
    :param mboxes: [((582.9349, 353.2006), (778.1303, 174.2541), -12.285979837613853)]
    :param heads: [[964.0, 290.0]]
    :return:[[cx, cy], [hx, hy], [x1, y2], ..., [x4, y4]]
    """
    if heads is None:
        heads = np.zeros([len(mboxes), 2])
    gt_points = []
    for mbox, head in zip(mboxes, heads):
        gt_center = np.array([[mbox[0][0], mbox[0][1]]])
        gt_mbox = cv2.boxPoints(mbox)
        gt_head = np.array([head])
        gt_points.append(np.concatenate([gt_center, gt_head, gt_mbox], axis=0).astype('float32'))

    return np.array(gt_points)


def check_mboxes(mboxes, size):
    """
    检查mboxes的合法性，不合法的要相应的删除
    :param mboxes: [N, 6, 2]
    :param size: [w, h]
    :return:
    """
    ind = []
    for i, mbox in enumerate(mboxes):
        center = mbox[0]
        if center[0] < 0 or center[1] < 0 or center[0] >= size[0] or center[1] >= size[1]:
            continue
        ind.append(i)

    new_mboxes = mboxes[ind]

    return new_mboxes
