import math

import cv2
import numpy as np
import paddle

from model.utils.rbox_iou import rbox_overlaps, rbox_poly_overlaps


def bbox2wh(bbox):
    """
    根据水平框左上和右下坐标计算水平框宽高
    :param bbox: [xmin, ymin, xmax, ymax]
    :return:
    """
    xmin, ymin, xmax, ymax = bbox

    w = (xmax - xmin) / 2
    h = (ymax - ymin) / 2

    return w, h


def xyxy2cxcy(bbox):
    """
    根据bbox左上点坐标和右下点坐标计算目标框中心位置
    :param bbox: 目标水平框左上点坐标和右下点坐标  #example: [xmin, ymin, xmax, ymax]
    :return: center: 目标中心点坐标  #example: [cx, cy]
    """

    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5
    center = [cx, cy]

    return center

def xyxy2points(bbox):
    """
    根据bbox左上点坐标和右下点坐标计算目标框四点位置
    :param bbox:
    :return:
    """
    xmin, ymin, xmax, ymax = bbox
    pts = [
        [xmin, ymin],
        [xmax, ymin],
        [xmin, ymax],
        [xmax, ymax]
    ]
    return pts

def decoder_bbox(center_xy, wh, R=4):
    """
    将模型输出转化为水平框表示
    :param center_xy: 中心点坐标
    :param wh: 水平框宽高
    :return:
    """
    x, y = center_xy[..., 0], center_xy[..., 1]
    w, h = wh[..., 0] * R, wh[..., 1] * R

    xmin = np.expand_dims(x - w, axis=1)
    ymin = np.expand_dims(y - h, axis=1)
    xmax = np.expand_dims(x + w, axis=1)
    ymax = np.expand_dims(y + h, axis=1)

    return np.concatenate([xmin, ymin, xmax, ymax], axis=1).astype('int32')

def decode_mbox(center_xy, hbb_wh, obb_wh, R=4):
    """
    将六参数形式转化为旋转框四点形式
    :param center_xy: 中心点坐标 [N, 2]
    :param hbb_wh: 外接矩形框宽高 [N, 2]
    :param obb_wh: 旋转框在外接矩形框方向位置(相对于宽高的比例) [N, 2]
    :param R: 下采样倍率
    :return:
    """

    x, y = center_xy[..., 0], center_xy[..., 1]
    w, h = hbb_wh[..., 0] * R, hbb_wh[..., 1] * R
    obb_w, obb_h = (2 * w * obb_wh[..., 0]), (2 * h * obb_wh[..., 1])

    p1 = np.expand_dims(np.concatenate([np.expand_dims(x+w-obb_w, axis=1), np.expand_dims(y-h, axis=1)], axis=1), axis=1)
    p2 = np.expand_dims(np.concatenate([np.expand_dims(x-w, axis=1), np.expand_dims(y+h-obb_h, axis=1)], axis=1), axis=1)
    p3 = np.expand_dims(np.concatenate([np.expand_dims(x-w+obb_w, axis=1), np.expand_dims(y+h, axis=1)], axis=1), axis=1)
    p4 = np.expand_dims(np.concatenate([np.expand_dims(x+w, axis=1), np.expand_dims(y-h+obb_h, axis=1)], axis=1), axis=1)

    obb = np.concatenate([p1, p2, p3, p4], axis=1)

    return obb

def bbox_iou(box1, box2, giou=False, diou=False, ciou=False, eps=1e-9):
    """calculate the iou of box1 and box2

    Args:
        box1 (list): [x1, y1, x2, y2], all have the shape [b, na, h, w, 1]
        box2 (list): [x1, y1, w2, h2], all have the shape [b, na, h, w, 1]
        giou (bool): whether use giou or not, default False
        diou (bool): whether use diou or not, default False
        ciou (bool): whether use ciou or not, default False
        eps (float): epsilon to avoid divide by zero

    Return:
        iou (Tensor): iou of box1 and box1, with the shape [b, na, h, w, 1]
    """
    px, py, pw, ph = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    gx, gy, gw, gh = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    px1, py1, px2, py2 = px-pw, py-ph, px+pw, py+ph
    gx1, gy1, gx2, gy2 = gx-gw, gy-gh, gx+gw, gy+gh
    x1 = paddle.maximum(px1, gx1)
    y1 = paddle.maximum(py1, gy1)
    x2 = paddle.minimum(px2, gx2)
    y2 = paddle.minimum(py2, gy2)

    overlap = ((x2 - x1).clip(0)) * ((y2 - y1).clip(0))

    area1 = (px2 - px1) * (py2 - py1)
    area1 = area1.clip(0)

    area2 = (gx2 - gx1) * (gy2 - gy1)
    area2 = area2.clip(0)

    union = area1 + area2 - overlap + eps
    iou = overlap / union

    if giou or ciou or diou:
        # convex w, h
        cw = paddle.maximum(px2, gx2) - paddle.minimum(px1, gx1)
        ch = paddle.maximum(py2, gy2) - paddle.minimum(py1, gy1)
        if giou:
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
        else:
            # convex diagonal squared
            c2 = cw**2 + ch**2 + eps
            # center distance
            rho2 = ((px1 + px2 - gx1 - gx2)**2 + (py1 + py2 - gy1 - gy2)**2) / 4
            if diou:
                return iou - rho2 / c2
            else:
                w1, h1 = px2 - px1, py2 - py1 + eps
                w2, h2 = gx2 - gx1, gy2 - gy1 + eps
                delta = paddle.atan(w1 / h1) - paddle.atan(w2 / h2)
                v = (4 / math.pi**2) * paddle.pow(delta, 2)
                alpha = v / (1 + eps - iou + v)
                alpha.stop_gradient = True
                return iou - (rho2 / c2 + v * alpha)
    else:
        return iou

def test_decode_mbox(ct_int, hbb_wh, obb_wh, R=4):
    x, y = ct_int[0], ct_int[1]
    w, h = hbb_wh[0], hbb_wh[1]
    obb_w, obb_h = (2 * w * obb_wh[0]), (2 * h * obb_wh[1])

    obb = [[x + w - obb_w, y - h],
            [x-w, y+h-obb_h],
            [x-w+obb_w, y+h],
            [x+w, y-h+obb_h]]

    return obb

def test_decode_bbox(center_xy, hbb_wh, R=4):
    x, y = center_xy
    hbb_w, hbb_h = hbb_wh[0], hbb_wh[1]

    hbb = [x-hbb_w, y-hbb_h, x+hbb_w, y+hbb_h]

    return np.int32(hbb)

def mbox2HBB(mbox):
    """
    根据mbox旋转目标框参数计算HBB外接水平框左上和右下点位置
    :param mbox: 旋转框参数  #example: [cx, cy, w, h, ang(弧度)]
    :return: HBB:  HBB外接水平框左上和右下点位置  #example: [xmin, ymin, xmax, ymax]
    """

    cx, cy, w, h, ang = mbox
    mmbox = np.int32(cv2.boxPoints(((cx, cy), (w, h), math.degrees(ang))))
    mmbox = cv2.boundingRect(mmbox)  # [xmin, ymin, w, h]
    HBB = [mmbox[0], mmbox[1], mmbox[0] + mmbox[2], mmbox[1] + mmbox[3]]

    return HBB


def py_cpu_nms(dets, scores, thresh=0.2):
    '''
    对多个旋转框非极大值抑制
    :param dets: 经过得分筛选后的多个旋转框  [N, 4, 2]
    :param scores: 每个旋转框的得分  [N]
    :param thresh: 旋转框iou阈值
    :return:
    '''
    """Pure Python NMS baseline."""
    dets = np.array(dets)
    if len(dets) == 0:
        return []

    order = scores.argsort()[::-1]  # [::-1]表示降序排序，输出为其对应序号

    keep = []  # 需要保留的bounding box
    while order.size > 0:
        i = order[0]  # 取置信度最大的（即第一个）框
        keep.append(i)  # 将其作为保留的框

        # 以下计算置信度最大的框（order[0]）与其它所有的框（order[1:]，即第二到最后一个）框的IOU，以下都是以向量形式表示和计算
        ovr = bbox_iou_rotate_calculate1(np.tile(np.expand_dims(dets[i], 0), [len(dets[order[1:]]), 1, 1]),
                                         dets[order[1:]])

        inds = np.where(ovr <= thresh)[0]  # 本轮，order仅保留IOU不大于阈值的下标
        order = order[inds + 1]  # 删除IOU大于阈值的框

    return keep

def mboxes_hbb_iou(mboxes, target_mboxes):
    """
    计算旋转矩形外接正矩形的iou
    :param mboxes: [OBJECT_NUM, N, 4, 2]
    :param target_mboxes: [OBJECT_NUM, N, 4, 2]
    :return:
    """

    mboxes_hbb = []
    target_mboxes_hbb = []
    for mbox_points, target_mbox_points in zip(mboxes, target_mboxes):
        mboxes_hbb.append(bbox_rotate2hbb(mbox_points))
        target_mboxes_hbb.append(bbox_rotate2hbb(target_mbox_points))
    mboxes_hbb = np.array(mboxes_hbb)
    target_mboxes_hbb = np.array(target_mboxes_hbb)

    iou = bboxes_iou(mboxes_hbb, target_mboxes_hbb)

    return iou

def bboxes_iou(boxes1, boxes2):
    '''
    cal IOU of two boxes or batch boxes
    such as: (1)
            boxes1 = np.asarray([[0,0,5,5],[0,0,10,10],[0,0,10,10]])
            boxes2 = np.asarray([[0,0,5,5]])
            and res is [1.   0.25 0.25]
            (2)
            boxes1 = np.asarray([[0,0,5,5],[0,0,10,10],[0,0,10,10]])
            boxes2 = np.asarray([[0,0,5,5],[0,0,10,10],[0,0,10,10]])
            and res is [1. 1. 1.]
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''

    # cal the box's area of boxes1 and boxess
    boxes1Area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # cal Intersection
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1Area + boxes2Area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def mboxes_obb_iou(mboxes, target_mboxes):
    """
    计算旋转矩形的iou
    :param mboxes: [OBJECT_NUM, N, 4, 2]
    :param target_mboxes: [OBJECT_NUM, N, 4, 2]
    :return:
    """
    ious = []
    for mbox, target_mbox in zip(mboxes, target_mboxes):
        iou = rbox_poly_overlaps(mbox.reshape(-1, 8), target_mbox.reshape(-1, 8))
        ious.append(iou)

    return np.array(ious)



def py_cpu_hbb_nms(dets, scores, thresh=0.5):
    """Pure Python NMS baseline."""

    if len(dets) == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # [::-1]表示降序排序，输出为其对应序号

    keep = []  # 需要保留的bounding box
    while order.size > 0:
        i = order[0]  # 取置信度最大的（即第一个）框
        keep.append(i)  # 将其作为保留的框

        # 以下计算置信度最大的框（order[0]）与其它所有的框（order[1:]，即第二到最后一个）框的IOU，以下都是以向量形式表示和计算
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 计算xmin的max,即overlap的xmin
        yy1 = np.maximum(y1[i], y1[order[1:]])  # 计算ymin的max,即overlap的ymin
        xx2 = np.minimum(x2[i], x2[order[1:]])  # 计算xmax的min,即overlap的xmax
        yy2 = np.minimum(y2[i], y2[order[1:]])  # 计算ymax的min,即overlap的ymax

        w = np.maximum(0.0, xx2 - xx1 + 1)  # 计算overlap的width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # 计算overlap的hight
        inter = w * h  # 计算overlap的面积
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  # 计算并，-inter是因为交集部分加了两次。

        inds = np.where(ovr <= thresh)[0]  # 本轮，order仅保留IOU不大于阈值的下标
        order = order[inds + 1]  # 删除IOU大于阈值的框

    return keep

def bbox_rotate2hbb(dets):
    hbbs = []

    for num in range(0, len(dets)):
        # 将四点坐标形式转化为外接正矩形四点参数形式
        h1 = cv2.boundingRect(dets[num])  # x, y, w, h
        hbb = [h1[0], h1[1], h1[0]+h1[2], h1[1]+h1[3]]
        hbbs.append(hbb)

    return np.array(hbbs)


def bbox_iou_rotate_calculate1(boxes1, boxes2):
    '''
    计算旋转面积iou
    :param boxes1: [N, 4, 2]
    :param boxes2: [N, 4, 2]
    :return:
    '''
    ious_total = []
    boxes1 = boxes1.astype('float32')
    boxes2 = boxes2.astype('float32')

    for num in range(0, len(boxes2)):
        # 将四点坐标形式转化为旋转矩形参数形式
        # 这里数据格式一定要是float32
        r1 = cv2.minAreaRect(boxes1[num])
        r2 = cv2.minAreaRect(boxes2[num])
        area1 = r1[1][0] * r1[1][1]
        area2 = r2[1][0] * r2[1][1]

        int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
        if int_pts is not None:
            order_pts = cv2.convexHull(int_pts, returnPoints=True)
            int_area = cv2.contourArea(order_pts)
            # 计算出iou
            division = (area1 + area2 - int_area)
            if division == 0:
                ious = 0
            else:
                ious = int_area * 1.0 / division

            ious_total.append(ious)
        else:
            ious = 0
            ious_total.append(ious)

    return np.array(ious_total, dtype=np.float32)

def clear_unqualified_box(mboxes, image_size):
    '''
    清除不合格的框
    :param mboxes: [N, 4, 2]
    :return:
    '''

    mask = []
    for mbox in mboxes:
        if len(np.nonzero(mbox[:, 0] < 0)[0]) > 0:
            mask.append(False)
            continue
        elif len(np.nonzero(mbox[:, 0] > image_size[0])[0]) > 0:
            mask.append(False)
            continue
        elif len(np.nonzero(mbox[:, 1] < 0)[0]) > 0:
            mask.append(False)
            continue
        elif len(np.nonzero(mbox[:, 1] > image_size[1])[0]) > 0:
            mask.append(False)
            continue
        else:
            mask.append(True)

    indexes = np.nonzero(mask)[0]
    new_mboxes = mboxes[indexes]

    return new_mboxes, indexes

def clear_unqualified_bbox(mboxes, image_size):
    '''
    清除不合格的水平框
    :param mboxes: [N, 4]
    :return:
    '''
    mask = []
    for mbox in mboxes:
        if len(np.nonzero(mbox < 0)[0]) > 0:
            mask.append(False)
            continue
        elif len(np.nonzero(mbox > image_size[0])[0]) > 0:
            mask.append(False)
            continue
        else:
            mask.append(True)

    indexes = np.nonzero(mask)[0]
    new_mboxes = mboxes[indexes]

    return new_mboxes, indexes

def ch_box(centers, headers):
    """
    根据头部关键点坐标与中心点的位置关系，生成相应的bbox
    :param centers: [N, 2]
    :param headers: [N, 2]
    :return: [N, 4] [xmin, ymin, xmax, ymax]
    """
    bboxes = []
    for center, header in zip(centers, headers):
        if center[0] >= header[0] and center[1] >= header[1]:
            # 第二象限（左上）
            bbox = paddle.concat([header[0], header[1], center[0], center[1]])
        elif center[0] <= header[0] and center[1] >= header[1]:
            # 第一象限（右上）
            bbox = paddle.concat([center[0], header[1], header[0], center[1]])
        elif center[0] >= header[0] and center[1] <= header[1]:
            # 第三象限（左下）
            bbox = paddle.concat([header[0], center[1], center[0], header[1]])
        else:
            # 第四象限（右下）
            bbox = paddle.concat([center[0], center[1], header[0], header[1]])
        bboxes.append(paddle.unsqueeze(bbox, axis=0))

    return paddle.concat(bboxes)

def mboxes_points2bboxes(mboxes):
    """
    将旋转框4角点坐标转化为外接正矩形表示
    :param points: [N, 4, 2]
    :return: [N, 4] [xmin, ymin, xmax, ymax]
    """
    mboxes = mboxes.numpy()

    bboxes = []
    for mbox in mboxes:
        bbox = cv2.boundingRect(mbox)
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        bboxes.append(bbox)

    return paddle.to_tensor(bboxes, dtype='float32')

def compute_mbox2obb_wh(mbox, hbb):
    """
    根据旋转框信息计算其在外接矩形框的方向
    :param mbox: [[x1, y1], [x2, y2], ..., [x4, y4]]
    :param hbb: [xmin, ymin, xmax, ymax]
    :return:
    """

    bb = [hbb[0]+hbb[2], hbb[1], hbb[0], hbb[1]+hbb[3]]  # 右上左下

    w_index = 0  # 上边
    h_index = 0  # 左边
    # 找到旋转框在外接矩形框左边上边的点
    for i in range(1, len(mbox)):
        if mbox[w_index][1] > mbox[i][1]:
            w_index = i
        if mbox[h_index][0] > mbox[i][0]:
            h_index = i

    obb_w = bb[0] - mbox[w_index][0]
    obb_h = bb[3] - mbox[h_index][1]

    return obb_w, obb_h
