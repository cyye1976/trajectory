import numpy as np
import time
# get gt
from shapely.geometry import Polygon


def rbox2poly_single(rrect, get_best_begin_point=False):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x_ctr, y_ctr, width, height, angle = rrect[0], rrect[1], rrect[2], rrect[3], rrect[4]
    tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
    # rect 2x4
    rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    # poly
    poly = R.dot(rect)
    x0, x1, x2, x3 = poly[0, :4] + x_ctr
    y0, y1, y2, y3 = poly[1, :4] + y_ctr
    poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
    return poly


def intersection(g, p):
    """
    Intersection.
    """

    g = g[:8].reshape((4, 2))
    p = p[:8].reshape((4, 2))

    a = g
    b = p

    use_filter = True
    if use_filter:
        # step1:
        inter_x1 = np.maximum(np.min(a[:, 0]), np.min(b[:, 0]))
        inter_x2 = np.minimum(np.max(a[:, 0]), np.max(b[:, 0]))
        inter_y1 = np.maximum(np.min(a[:, 1]), np.min(b[:, 1]))
        inter_y2 = np.minimum(np.max(a[:, 1]), np.max(b[:, 1]))
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.
        x1 = np.minimum(np.min(a[:, 0]), np.min(b[:, 0]))
        x2 = np.maximum(np.max(a[:, 0]), np.max(b[:, 0]))
        y1 = np.minimum(np.min(a[:, 1]), np.min(b[:, 1]))
        y2 = np.maximum(np.max(a[:, 1]), np.max(b[:, 1]))
        if x1 >= x2 or y1 >= y2 or (x2 - x1) < 2 or (y2 - y1) < 2:
            return 0.

    g = Polygon(g)
    p = Polygon(p)
    #g = g.buffer(0)
    #p = p.buffer(0)
    if not g.is_valid or not p.is_valid:
        return 0

    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


# rbox_iou by python
def rbox_overlaps(anchors, gt_bboxes, use_cv2=False):
    """
    Args:
        anchors: [NA, 5]  x1,y1,x2,y2,angle
        gt_bboxes: [M, 5]  x1,y1,x2,y2,angle

    Returns:

    """
    assert anchors.shape[1] == 5
    assert gt_bboxes.shape[1] == 5

    gt_bboxes_ploy = [rbox2poly_single(e) for e in gt_bboxes]
    anchors_ploy = [rbox2poly_single(e) for e in anchors]

    num_gt, num_anchors = len(gt_bboxes_ploy), len(anchors_ploy)
    iou = np.zeros((num_gt, num_anchors), dtype=np.float32)

    for i in range(num_gt):
        for j in range(num_anchors):
            try:
                iou[i, j] = intersection(gt_bboxes_ploy[i], anchors_ploy[j])
            except Exception as e:
                print('cur gt_bboxes_ploy[i]', gt_bboxes_ploy[i],
                      'anchors_ploy[j]', anchors_ploy[j], e)
    iou = iou.T
    return iou

def rbox_poly_overlaps(anchors, gt_bboxes, use_cv2=False):
    """
    Args:
        anchors: [NA, 8]  [x0,y0,x1,y1,x2,y2,x3,y3]
        gt_bboxes: [M, 8]  [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:

    """
    assert anchors.shape[1] == 8
    assert gt_bboxes.shape[1] == 8

    gt_bboxes_ploy = gt_bboxes
    anchors_ploy = anchors

    num_gt, num_anchors = len(gt_bboxes_ploy), len(anchors_ploy)
    iou = np.zeros((num_gt, num_anchors), dtype=np.float32)

    for i in range(num_gt):
        for j in range(num_anchors):
            try:
                iou[i, j] = intersection(gt_bboxes_ploy[i], anchors_ploy[j])
            except Exception as e:
                print('cur gt_bboxes_ploy[i]', gt_bboxes_ploy[i],
                      'anchors_ploy[j]', anchors_ploy[j], e)
    iou = iou.T
    return iou

def np_rbox_poly_overlaps(anchors, gt_bboxes, use_cv2=False):
    """
    Args:
        anchors: [NA, 8]  [x0,y0,x1,y1,x2,y2,x3,y3]
        gt_bboxes: [M, 8]  [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:

    """
    assert anchors.shape[1] == 8
    assert gt_bboxes.shape[1] == 8

    gt_bboxes_ploy = gt_bboxes
    anchors_ploy = anchors

    num_gt, num_anchors = len(gt_bboxes_ploy), len(anchors_ploy)
    iou = np.zeros(num_gt, dtype=np.float32)

    for i in range(num_gt):
        try:
            iou[i] = intersection(gt_bboxes_ploy[i], anchors_ploy[i])
        except Exception as e:
            print('cur gt_bboxes_ploy[i]', gt_bboxes_ploy[i],
                  'anchors_ploy[i]', anchors_ploy[i], e)
    iou = iou.T
    return iou