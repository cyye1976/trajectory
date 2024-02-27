import math

import cv2
import numpy as np
from imgaug import BoundingBox, BoundingBoxesOnImage, ia
import paddle
from model.utils.bbox_tr import bbox_tr_2_4pt
from model.utils.bbox_util import bbox_iou_rotate_calculate1
import paddle.nn.functional as F
from numpy import ma


def mbox_ang2vector(cx, cy, h, ang):
    """
    将旋转框的角度根据h先正矩形表示然后“以中心点沿正y轴h的距离作为向量”
    根据ang旋转，得到最终向量
    :param cy: 中心点y坐标
    :param cx: 中心点x坐标
    :param h: 旋转框高
    :param ang: 旋转角度
    :return: vector [x, y]
    """
    # 计算出旋转前的向量
    vector = [0, h]

    # 根据旋转角度计算旋转后
    x, y = vector
    rad = np.math.radians(ang)
    theta_x = x * np.math.cos(rad) + y * np.math.sin(rad)
    theta_y = -x * np.math.sin(rad) + y * np.math.cos(rad)
    vector = [theta_x, theta_y]

    return vector


def mbox2target_seg_cls(mask, mbox_points):
    '''
    将旋转框内部填充生成分割mask图
    :param mbox:
    :return:
    '''
    new_mask = mask.copy()

    background_mask = new_mask[:, :, 0].copy()
    object_mask = new_mask[:, :, 1].copy()

    pts = mbox_points.astype('int')

    # 此处若不改名的话则会导致错误，需要另起名字
    b_mask = cv2.fillPoly(background_mask, pts, 0.)
    o_mask = cv2.fillPoly(object_mask, pts, 1.)

    # 可视化
    # im = object_mask * 255.
    # ia.imshow(im)

    new_mask = np.concatenate([np.expand_dims(b_mask, axis=2), np.expand_dims(o_mask, axis=2)], axis=2)

    return new_mask


def segmentation2mbox(segmentation):
    """
    将实例分割点集转化为旋转框形式
    :param segmentation: [x1, y1, x2, y2, ...]
    :return:
    """
    contour = []
    for i in range(len(segmentation) // 2):
        contour.append([segmentation[i * 2], segmentation[(i * 2) + 1]])
    mbox = cv2.minAreaRect(np.int32(contour))
    return mbox


def decode_mbox_by_tr(center_xy, off, uv, s, rho, R=4):
    """
    将tr形式的旋转框转化为四点形式
    :param center_xy: 中心点坐标 [N, 2]
    :param off: 中心点偏移  [N, 2]
    :param uv: 向量偏移 [N, 2]
    :param s: 同号异号标识 [N]
    :param rho: [N]
    :param R: 下采样倍率
    :return:
    """
    R_center_xy = center_xy / R + off
    trs = np.concatenate([R_center_xy, uv, s, rho], axis=1)
    mboxes = []
    for tr in trs:
        mbox = bbox_tr_2_4pt(tr)
        mboxes.append([mbox[:2], mbox[2:4], mbox[4:6], mbox[6:8]])
    mboxes = np.array(mboxes) * R

    return mboxes, center_xy


def decode_mbox_by_obb(center_xy, hbb_wh, obb_wh, off, R=4):
    """
    将六参数形式转化为旋转框四点形式
    :param center_xy: 中心点坐标 [N, 2]
    :param hbb_wh: 外接矩形框宽高 [N, 2]
    :param obb_wh: 旋转框在外接矩形框方向位置(相对于宽高的比例) [N, 2]
    :param R: 下采样倍率
    :return:
    """
    R_center_xy = center_xy / R + off
    x, y = R_center_xy[..., 0] * R, R_center_xy[..., 1] * R
    w, h = hbb_wh[..., 0] * R, hbb_wh[..., 1] * R
    obb_w, obb_h = (2 * w * obb_wh[..., 0]), (2 * h * obb_wh[..., 1])

    p1 = np.expand_dims(np.concatenate([np.expand_dims(x + w - obb_w, axis=1), np.expand_dims(y - h, axis=1)], axis=1),
                        axis=1)
    p2 = np.expand_dims(np.concatenate([np.expand_dims(x - w, axis=1), np.expand_dims(y + h - obb_h, axis=1)], axis=1),
                        axis=1)
    p3 = np.expand_dims(np.concatenate([np.expand_dims(x - w + obb_w, axis=1), np.expand_dims(y + h, axis=1)], axis=1),
                        axis=1)
    p4 = np.expand_dims(np.concatenate([np.expand_dims(x + w, axis=1), np.expand_dims(y - h + obb_h, axis=1)], axis=1),
                        axis=1)

    mboxes = np.concatenate([p1, p2, p3, p4], axis=1)
    mboxes_center = R_center_xy * R

    return mboxes, mboxes_center


def reorder_pts(tt, rr, bb, ll):
    pts = np.asarray([tt, rr, bb, ll], np.float32)
    l_ind = np.argmin(pts[:, 0])
    r_ind = np.argmax(pts[:, 0])
    t_ind = np.argmin(pts[:, 1])
    b_ind = np.argmax(pts[:, 1])
    tt_new = pts[t_ind, :]
    rr_new = pts[r_ind, :]
    bb_new = pts[b_ind, :]
    ll_new = pts[l_ind, :]
    return tt_new, rr_new, bb_new, ll_new


def generate_bbav_trbl(mbox_points, center_xy, theta):
    """
    将旋转框4点坐标生成trbl表示
    :param mbox_points: [4, 2]
    :param center_xy: 中心点 [x, y]
    :return:
    """
    bl = mbox_points[0, :]
    tl = mbox_points[1, :]
    tr = mbox_points[2, :]
    br = mbox_points[3, :]

    tt = (np.asarray(tl, np.float32) + np.asarray(tr, np.float32)) / 2
    rr = (np.asarray(tr, np.float32) + np.asarray(br, np.float32)) / 2
    bb = (np.asarray(bl, np.float32) + np.asarray(br, np.float32)) / 2
    ll = (np.asarray(tl, np.float32) + np.asarray(bl, np.float32)) / 2

    if theta in [-90.0, -0.0, 0.0]:  # (-90, 0]
        tt, rr, bb, ll = reorder_pts(tt, rr, bb, ll)

    return tt - center_xy, rr - center_xy, bb - center_xy, ll - center_xy

    # # 得到四边中点坐标
    # mid_points = np.array(mbox2four_mid_points(mbox_points))  # [4, 2]
    #
    # # 计算每个中点相对中心点的偏移量
    # reg_p = mid_points - np.array(center_xy)  # [4, 2]
    #
    # # 遍历判断哪个偏移量属于哪个trbl
    # trbl = []
    # for off in reg_p:
    #     if off[0] < 0 and off[1] < 0:
    #         trbl.append(off)
    #         continue
    #     elif off[0] > 0 and off[1] < 0:
    #         trbl.append(off)
    #         continue
    #     elif off[0] > 0 and off[1] > 0:
    #         trbl.append(off)
    #         continue
    #     else:
    #         trbl.append(off)
    #         continue
    # return trbl


def decode_mbox_by_bbav(center_xy, trblwh, cls_theta, off, R=4):
    """
    根据trblwh解码旋转框
    :param center_xy: 中心点坐标 [N, 2] （已经过下采样倍率还原）
    :param trblwh: 旋转框坐标系向量，及外接矩形框宽高 [N, 10]  （未已经过下采样倍率还原）
    :param cls_theta: HBB(0)/RBB(1)
    :param off: 中心点偏移 [N, 2]
    :param R: 下采样倍率
    :return:
    """
    # 先把中心点还原成下采样倍率
    R_center_xy = center_xy / R + off
    decode_theta = np.where(cls_theta > 0, 1, 0)

    # 根据中心点，宽高，角度，还原旋转框四顶点坐标
    mboxes = []
    for i in range(len(R_center_xy)):
        x, y = R_center_xy[i]
        tx, ty, rx, ry, bx, by, lx, ly, w, h = trblwh[i]  # 这里的w，h是相对于左上角顶点的，而不是中心点！！！
        single_cls_theta = decode_theta[i]
        if single_cls_theta == 0:
            # 则为水平框检测HBB
            mbox_points = [[x - w / 2, y - h / 2], [x + w / 2, y - h / 2], [x + w / 2, y + h / 2],
                           [x - w / 2, y + h / 2]]
            mboxes.append(mbox_points)
        else:
            # 则为旋转框检测OBB
            tt = np.asarray([tx, ty], np.float32)
            rr = np.asarray([rx, ry], np.float32)
            bb = np.asarray([bx, by], np.float32)
            ll = np.asarray([lx, ly], np.float32)
            tl = tt + ll - R_center_xy[i]
            bl = bb + ll - R_center_xy[i]
            tr = tt + rr - R_center_xy[i]
            br = bb + rr - R_center_xy[i]
            mid_points = np.asarray([-tr, -br, -bl, -tl], np.float32)
            # mbox_points = cv2.boxPoints(cv2.minAreaRect(mid_points))
            mboxes.append(mid_points.tolist())

    mboxes = np.array(mboxes).astype('float32')

    # 根据旋转框四顶点坐标计算下采样倍率前的结果
    mboxes = mboxes * R
    mboxes_center = center_xy

    return mboxes, mboxes_center


def compute_cls_theta(mbox_points, bbox):
    """
    判断旋转框目标是旋转形式还是接近水平框形式
    :param mbox_points: 旋转框四顶点坐标 [4, 2]
    :param bbox_points: 水平框四顶点坐标 [xmin, ymin, xmax, ymax]
    :return:
    """
    # 计算旋转面积IOU
    xmin, ymin, xmax, ymax = bbox
    bbox_points = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    riou = bbox_iou_rotate_calculate1(np.expand_dims(mbox_points, axis=0), np.expand_dims(bbox_points, axis=0))
    if riou < 0.95:
        # 则为RBB
        return 1
    else:
        # 则为HBB
        return 0


def decode_mbox_by_vector(center_xy, wh, off, vector, R=4):
    """
    根据角度向量解码旋转框
    :param center_xy: 中心点坐标 [N, 2] （已经过下采样倍率还原）
    :param wh: 旋转框宽高 [N, 2]  （未已经过下采样倍率还原）
    :param off: 中心点偏移量 [N, 2]
    :param vector: 旋转框角度 [N, 2]（未已经过下采样倍率还原）
    :param R: 下采样倍率
    :return:
    """

    # 把向量表示的角度转化为cv可认的角度
    cv_theta = np.degrees(np.arctan2(vector[:, 0], vector[:, 1]))

    # 先把中心点还原成下采样倍率
    R_center_xy = center_xy / R + off

    # 根据中心点，宽高，角度，还原旋转框四顶点坐标
    mboxes = []
    for i in range(len(R_center_xy)):
        mbox = ((R_center_xy[i][0], R_center_xy[i][1]), (wh[i][0], wh[i][1]), cv_theta[i])
        mbox = cv2.boxPoints(mbox)
        mboxes.append(mbox)
    mboxes = np.array(mboxes)

    # 根据旋转框四顶点坐标计算下采样倍率前的结果
    mboxes = mboxes * R
    mboxes_center = center_xy

    return mboxes, mboxes_center


def vector2rotated(center, wh, ang):
    """
    将旋转框的角度根据h先正矩形表示然后“以中心点沿正y轴h的距离作为向量”
    :param center: [N, 2]
    :param wh: [N, 2]
    :param ang: [N]
    :return: [N, 4, 2]
    """

    # 计算出旋转前的向量
    vector = np.transpose(np.array([[-wh[..., 0], wh[..., 1]],
                                    [wh[..., 0], wh[..., 1]],
                                    [wh[..., 0], -wh[..., 1]],
                                    [-wh[..., 0], -wh[..., 1]]]), [2, 0, 1]) / 2

    # 根据旋转角度计算旋转后
    x, y = vector[..., 0], vector[..., 1]

    rad = ang * (np.pi / 180.)
    theta_x = x * np.tile(np.expand_dims(np.cos(rad), axis=-1), [1, 4]) + y * np.tile(np.expand_dims(np.sin(rad), axis=-1), [1, 4])
    theta_y = -x * np.tile(np.expand_dims(np.sin(rad), axis=-1), [1, 4]) + y * np.tile(np.expand_dims(np.cos(rad), axis=-1), [1, 4])
    vector = np.concatenate([np.expand_dims(theta_x, axis=-1), np.expand_dims(theta_y, axis=-1)], axis=-1)

    te = np.tile(np.expand_dims(center, axis=1), [1, 4, 1])
    mboxes = np.concatenate(
        [np.expand_dims(te[..., 0] + vector[..., 0], axis=-1), np.expand_dims(te[..., 1] - vector[..., 1], axis=-1)],
        axis=-1).astype('float32')

    return mboxes, vector


def vector2rotated_paddle(wh, ang):
    # 计算出旋转前的向量
    vector1 = paddle.unsqueeze(
        paddle.concat([
            paddle.unsqueeze(-wh[:, 0], axis=-1),
            paddle.unsqueeze(wh[:, 1], axis=-1)], axis=-1), axis=1)
    vector2 = paddle.unsqueeze(
        paddle.concat([
            paddle.unsqueeze(wh[:, 0], axis=-1),
            paddle.unsqueeze(wh[:, 1], axis=-1)], axis=-1), axis=1)
    vector3 = paddle.unsqueeze(
        paddle.concat([
            paddle.unsqueeze(wh[:, 0], axis=-1),
            paddle.unsqueeze(-wh[:, 1], axis=-1)], axis=-1), axis=1)
    vector4 = paddle.unsqueeze(
        paddle.concat([
            paddle.unsqueeze(-wh[:, 0], axis=-1),
            paddle.unsqueeze(-wh[:, 1], axis=-1)], axis=-1), axis=1)
    vector = paddle.concat([vector1, vector2, vector3, vector4], axis=1)

    # 根据旋转角度计算旋转后
    x, y = vector[:, :, 0], vector[:, :, 1]

    rad = ang * (np.pi / 180.)
    theta_x = x * paddle.tile(paddle.unsqueeze(paddle.cos(rad), axis=-1), [1, 4]) + y * paddle.tile(paddle.unsqueeze(paddle.sin(rad), axis=-1), [1, 4])
    theta_y = -x * paddle.tile(paddle.unsqueeze(paddle.sin(rad), axis=-1), [1, 4]) + y * paddle.tile(paddle.unsqueeze(paddle.cos(rad), axis=-1), [1, 4])
    vector = paddle.concat([paddle.unsqueeze(theta_x, axis=-1), paddle.unsqueeze(theta_y, axis=-1)], axis=-1)

    return vector


def decode_mbox_by_vector_s(center_xy, wh, off, vector, s, R=4):
    """
    根据角度向量解码旋转框
    :param center_xy:  中心点坐标 [N, 2] （已经过下采样倍率还原）
    :param wh: 旋转框宽高 [N, 2]  （未已经过下采样倍率还原）
    :param off: 中心点偏移量 [N, 2]
    :param vector: 旋转框角度向量 [N, 2]（未已经过下采样倍率还原）
    :param s:  向量所处象限 [N, 1]（0， 1 一或二）
    :param R: 下采样倍率
    :return:
    """
    # 解码分类s
    decode_s = np.where(s > 0, -1, 1)

    # 把向量表示的角度转化为cv可认的角度
    cv_theta = np.degrees(np.arctan2(decode_s[:, 0] * vector[:, 0], vector[:, 1]))

    # 先把中心点还原成下采样倍率
    R_center_xy = center_xy / R + off

    mboxes, _ = vector2rotated(R_center_xy, wh, cv_theta)

    # 根据中心点，宽高，角度，还原旋转框四顶点坐标
    # mboxes = []
    # for i in range(len(R_center_xy)):
    #     mbox = ((R_center_xy[i][0], R_center_xy[i][1]), (wh[i][0], wh[i][1]), cv_theta[i])
    #     mbox = cv2.boxPoints(mbox)
    #     mboxes.append(mbox)
    # mboxes = np.array(mboxes)

    # 根据旋转框四顶点坐标计算下采样倍率前的结果
    mboxes = mboxes * R
    mboxes_center = center_xy

    return mboxes, mboxes_center


def decode_mbox_by_ang(center_xy, wh, ang, R=4):
    """
    根据角度解码旋转框
    :param center_xy: 中心点坐标 [N, 2] （已经过下采样倍率还原）
    :param wh: 旋转框宽高 [N, 2]  （未已经过下采样倍率还原）
    :param off: 中心点偏移量 [N, 2] （未已经过下采样倍率还原）
    :param ang: 旋转框角度 [N, 1]（未已经过下采样倍率还原）
    :param R: 下采样倍率
    :return:
    """
    # 把角度转化为cv可认
    cv_theta = ang

    # 先把中心点还原成下采样倍率
    R_center_xy = center_xy / R

    # 根据中心点，宽高，角度，还原旋转框四顶点坐标
    mboxes = []
    for i in range(len(R_center_xy)):
        mbox = ((R_center_xy[i][0], R_center_xy[i][1]), (wh[i][0], wh[i][1]), int(cv_theta[i]))
        mbox = cv2.boxPoints(mbox)
        mboxes.append(mbox)
    mboxes = np.array(mboxes)

    # 根据旋转框四顶点坐标计算下采样倍率前的结果
    mboxes = mboxes * R
    mboxes_center = center_xy

    return mboxes, mboxes_center


def header_match(header_xy, header_xy_match, R=4):
    '''
    将回归的到的头部关键点与热力图的得到的头部关键点做最近匹配
    :param header_xy: [2, N]  特征图尺寸
    :param header_xy_match: [2, N]  输入图片尺寸
    :return:
    '''
    xy = np.expand_dims(header_xy, axis=1)  # [N, 1, 2]
    xy = np.tile(xy, [1, len(header_xy_match), 1])  # [N, N, 2]

    xy_match = np.expand_dims(header_xy_match, axis=0) / R
    xy_match = np.tile(xy_match, [len(xy), 1, 1])  # [N, N, 2]

    # 计算距离取距离最小的索引
    distance = np.sqrt(
        (xy[:, :, 0] - xy_match[:, :, 0]) ** 2 + (xy[:, :, 1] - xy_match[:, :, 1]) ** 2)  # [N_xy, N_location]
    min_ind = np.argmin(distance, axis=1)

    xy = header_xy_match[min_ind] / R

    return xy


def decode_mbox_by_chp(center_xy, wh, off, head_xy, R=4):
    """
    CHPDet模型解码
    :param center_xy:  中心点坐标 [N, 2] （已经过下采样倍率还原）
    :param wh: 旋转框宽高 [N, 2]  （未已经过下采样倍率还原）
    :param off: 中心点偏移量 [N, 2] （未已经过下采样倍率还原）
    :param head_xy: 头部点位置 [N, 2] （未已经过下采样倍率还原）
    :param head_off: 头部点位置偏移量 [N, 2] （未已经过下采样倍率还原）
    :param R: 下采样倍率
    :return:
    """
    # 先把中心点还原成下采样倍率
    R_center_xy = center_xy / R + off

    # 还原旋转框顶点坐标
    mboxes = []
    for i in range(len(head_xy)):
        ang = azimuthAngle_cv2(R_center_xy[i][0], R_center_xy[i][1], head_xy[i][0], head_xy[i][1])
        mbox = ((R_center_xy[i][0], R_center_xy[i][1]), (wh[i][1], wh[i][0]), ang)
        mbox = cv2.boxPoints(mbox)
        mboxes.append(mbox)
    mboxes = np.array(mboxes)

    # 根据旋转框四顶点坐标计算下采样倍率前的结果
    mboxes = mboxes * R
    mboxes_center = center_xy

    return mboxes, mboxes_center


def course2ang(R_center_xy, head_xy):
    ang = getDegree(R_center_xy[0], R_center_xy[1], head_xy[0], head_xy[1])
    if head_xy[0] > 0 and head_xy[1] > 0:
        ang = -(360 - ang)
    elif head_xy[0] < 0 and head_xy[1] > 0:
        ang = -(270 - ang)
    elif head_xy[0] < 0 and head_xy[1] < 0:
        ang = -(180 - ang)
    else:
        ang = -(90 - ang)
    return ang


def bb_is_negative_number(points, min, max):
    """
    判断bbox是否超出界限
    :param bb:[x, y, w, h]
    :param min:
    :param max:
    :return:
    """
    points = np.array(points).flatten()
    for num in points:
        if num < min or num > max:
            return True
    return False


def points2mid(p1, p2):
    """
    将两点计算它们的中点坐标
    :param p1: [x, y]
    :param p2: [x, y]
    :return:
    """
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]


def mbox2four_mid_points(mbox_points):
    """
    根据旋转框四顶点坐标计算旋转框四边中点
    :param mbox_points: [N, 2]
    :return:
    """
    # 计算旋转框四边中点
    p1, p2, p3, p4 = mbox_points
    mid1 = points2mid(p1, p2)
    mid2 = points2mid(p2, p3)
    mid3 = points2mid(p3, p4)
    mid4 = points2mid(p4, p1)
    return [mid1, mid2, mid3, mid4]


def mbox2mid_bbox(mbox, image=None, R=4):
    """
    根据旋转框四点坐标计算旋转框四边中点，并根据计算出的四个中点
    转化为旋转框中目标的内部水平框
    :param mbox: 旋转框四点坐标 [[x1, y1], ..., [x4, y4]]
    :return:
    """
    # 计算旋转框四边中点
    p1, p2, p3, p4 = mbox
    mid1 = points2mid(p1, p2)
    mid2 = points2mid(p2, p3)
    mid3 = points2mid(p3, p4)
    mid4 = points2mid(p4, p1)

    # 计算四中点形成的内部水平框
    br = cv2.boundingRect(np.int32([mid1, mid2, mid3, mid4]))
    bbox = [br[0], br[1], br[0] + br[2], br[1] + br[3]]

    # 如果存在图片输入则需要进行水平框可视化
    if image is not None:
        image_bbox = np.array(bbox) * R
        box = [BoundingBox(x1=image_bbox[0], y1=image_bbox[1], x2=image_bbox[2], y2=image_bbox[3])]
        box = BoundingBoxesOnImage(box, shape=image.shape)
        img = box.draw_on_image(image, color=[255, 0, 255])
        ia.imshow(img)

    return bbox


def mbox2head_class(hbbs, heads):
    """
    根据旋转框外接矩形框和旋转框头部点坐标
    计算头部所属类
    :param hbbs: 旋转框外接矩形框 [xmin, ymin, xmax, ymax]:  [N, 4]
    :param heads: 旋转框头部点坐标 [x, y]:  [N, 2]
    :return:
    """

    head_classes = []
    for hbb, head in zip(hbbs, heads):
        xmin, ymin, xmax, ymax = hbb
        x, y = head

        # 求出与头部点最近的外接矩形框的顶点
        bbox = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]  # 外接矩形框四顶点（从左上角点开始顺时针标记0,1,2,3）
        ind = 0
        min_distance = 0
        for i in range(len(bbox)):
            distance = np.math.sqrt((x - bbox[i][0]) ** 2 + (y - bbox[i][1]) ** 2)
            if i == 0:
                min_distance = distance
            else:
                if min_distance > distance:
                    min_distance = distance
                    ind = i

        # 根据求出的顶点对应其头部分类
        head_cls = [0, 1, 2, 3]  # 外接矩形上两点为0, 下两点为1
        head_classes.append(head_cls[ind])

    return np.array(head_classes)


def decode_mbox_head_by_pts(center_xy, mboxes, head_offset, R=4):
    """
    根据中心点以及外接矩形宽高
    求出旋转框头部分类
    :param center_xy: [N, 2]
    :param mboxes: [N, 4, 2]
    :param head_offset: [N, 2] 头部相对中心点回归位置
    :param R:
    :return:
    """
    # 这里的wh是中心点到两边长
    hbb_wh = []
    for i in range(len(mboxes)):
        hbb = cv2.boundingRect(mboxes[i])
        hbb_wh.append([hbb[2] / 2, hbb[3] / 2])
    hbb_wh = np.array(hbb_wh, dtype='float32')

    x, y = center_xy[:, 0], center_xy[:, 1]
    w, h = hbb_wh[:, 0] * R, hbb_wh[:, 1] * R

    p0 = np.concatenate([np.expand_dims(x - w, axis=1), np.expand_dims(y - h, axis=1)], axis=1)
    p2 = np.concatenate([np.expand_dims(x + w, axis=1), np.expand_dims(y + h, axis=1)], axis=1)
    bbox = np.concatenate([p0, p2], axis=1)

    head = center_xy - head_offset * R

    head_classes = mbox2head_class(bbox, head)

    return np.array(head_classes)


def decode_mbox_head(center_xy, hbb_wh, head_offset, R=4):
    """
    根据中心点以及外接矩形宽高
    求出旋转框头部分类
    :param center_xy: [N, 2]
    :param hbb_wh: [N, 2] （这里的wh是中心点到两边长）
    :param head_offset:  [N, 2]
    :return:
    """
    x, y = center_xy[:, 0], center_xy[:, 1]
    w, h = hbb_wh[:, 0] * R, hbb_wh[:, 1] * R

    p0 = np.concatenate([np.expand_dims(x - w, axis=1), np.expand_dims(y - h, axis=1)], axis=1)
    p2 = np.concatenate([np.expand_dims(x + w, axis=1), np.expand_dims(y + h, axis=1)], axis=1)
    bbox = np.concatenate([p0, p2], axis=1)

    head = center_xy - head_offset * R

    head_classes = mbox2head_class(bbox, head)

    return np.array(head_classes)


def getDegree(cx, cy, hx, hy):
    """
    Args:
        point p1(cx, cy)
        point p2(hx, hy)
    Returns:
        bearing between the two GPS points,
        default: the basis of heading direction is north
    """
    # TODO：要注意这里指的是直角坐标系，而不是opencv以左上点为原点
    ang = np.degrees(np.arctan2(cy - hy, hx - cx))
    ang = np.where(ang < 0, abs(ang), 360 - ang)
    # if ang < 0:
    #     ang = abs(ang)
    # else:
    #     ang = 360 - ang

    return ang


# 计算cv2角度
def azimuthAngle_cv2(cx, cy, hx, hy):
    """
    中心点到头部点cv2可认角度，角度可直接用于旋转框角度
    :param cx:
    :param cy:
    :param hx:
    :param hy:
    :return:
    """
    ang = np.degrees(np.arctan2(cy - hy, hx - cx))
    if ang < 0:
        ang = abs(ang)
    else:
        ang = 360 - ang

    return ang


def correct_box_head_by_crf(center_xy, mboxes, mboxes_head, R=4):
    """
    基于外接矩形框的航向点修正方法
    :param center_xy: [N, 2] 定向边界框中心点
    :param mboxes: [N, 4, 2] 定向边界框
    :param mboxes_head: [N, 2] 航向点
    :param R: 下采样倍率
    :return:
    """
    # 这里的wh是中心点到两边长,得到外接矩形框四顶点位置信息
    hbb_wh = []
    for i in range(len(mboxes)):
        hbb = cv2.boundingRect(mboxes[i])
        hbb_wh.append([hbb[2] / 2, hbb[3] / 2])
    hbb_wh = np.array(hbb_wh, dtype='float32')
    x, y = center_xy[:, 0], center_xy[:, 1]
    w, h = hbb_wh[:, 0] * R, hbb_wh[:, 1] * R
    p0 = np.concatenate([np.expand_dims(x - w, axis=1), np.expand_dims(y - h, axis=1)], axis=1)
    p2 = np.concatenate([np.expand_dims(x + w, axis=1), np.expand_dims(y + h, axis=1)], axis=1)
    bbox = np.concatenate([p0, p2], axis=1)

    # 匹配航向点与外接矩形框顶点类
    head_classes = mbox2head_class(bbox, mboxes_head)

    # 根据分类计算具体航向角
    courses, head_points = decode_courses(mboxes, head_classes, center_xy)

    return np.array(courses), np.array(head_points)


def decode_courses(mboxes, mboxes_head, y_center_xy):
    """
    基于外接矩形框的航向点修正方法
    :param mboxes: [N, 4, 2]
    :param mboxes_head: [N]
    :param y_center_xy: [N, 2]
    :return:
    """
    courses = []
    head_points = []
    try:
        x, y = mboxes[..., 0], mboxes[..., 1]

        # 为旋转框四个顶点进行标记
        # 顶点处于最上边
        p0 = np.argmin(y, axis=1)
        # 顶点处于最右边
        p1 = np.argmax(x, axis=1)
        # 顶点处于最下边
        p2 = np.argmax(y, axis=1)
        # 顶点处于最左边
        p3 = np.argmin(x, axis=1)

        # 根据头部分类，选择两顶点，计算两点之间的中点即头部点坐标
        for i in range(len(mboxes)):
            if mboxes_head[i] == 0:
                # 则由3和0顶点组成
                mid_p = (mboxes[i, p3[i], ...] + mboxes[i, p0[i], ...]) / 2.
            elif mboxes_head[i] == 1:
                # 则由0和1顶点组成
                mid_p = (mboxes[i, p0[i], ...] + mboxes[i, p1[i], ...]) / 2.
            elif mboxes_head[i] == 2:
                # 则由1和2顶点组成
                mid_p = (mboxes[i, p1[i], ...] + mboxes[i, p2[i], ...]) / 2.
            elif mboxes_head[i] == 3:
                # 则由2和3顶点组成
                mid_p = (mboxes[i, p2[i], ...] + mboxes[i, p3[i], ...]) / 2.
            else:
                raise Exception(print("Head class is no exist."))

            # 根据中心点和头部点坐标计算航向
            course = getDegree(y_center_xy[i][0], y_center_xy[i][1], mid_p[0], mid_p[1])
            courses.append(course)
            head_points.append([mid_p[0], mid_p[1]])
    except:
        return courses, head_points

    return courses, head_points

if __name__ == '__main__':
    mboxes = np.array([[[223.43045043945312, 343.86096], [203.414306640625, 309.83618], [104.78903198242188, 430.2517],
                        [124.80517578125, 464.27652]],
                       [[223.43045043945312, 343.86096], [203.414306640625, 309.83618], [104.78903198242188, 430.2517],
                        [124.80517578125, 464.27652]]])
    mboxes_head = np.array([0, 2])
    y_center_xy = np.array([[41, 96], [41, 96]]) * 4
    courses = decode_courses(mboxes, mboxes_head, y_center_xy)
    print(courses)
