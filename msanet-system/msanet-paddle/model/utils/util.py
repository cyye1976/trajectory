import math

import cv2
import numpy as np
import paddle.nn as nn
import paddle

def freeze_layers(layers):
    for param in layers.parameters():
        param.stop_gradient = True

def tensor_list2numpy(tensor_list):
    """
    将列表中的所有tensor元素转化为numpy
    :param tensor_list: dict or list
    :return:
    """
    if type(tensor_list) == dict:
        new_tensor_list = dict()
        for key in tensor_list:
            new_tensor_list[key] = tensor_list[key].numpy()
    else:
        new_tensor_list = []
        for i, tensor in enumerate(tensor_list):
            new_tensor_list.append(tensor.numpy())

    return new_tensor_list


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def pool_nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = hmax == heat
    return keep

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def compare_distance_get_min(point1, point2, value1, value2):
    x1, y1 = point1
    x2, y2 = point2

    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    if abs(value1-distance) <= abs(value2-distance):
        return value2
    else:
        return value1

def heatmap_overlay(image,heatmap):
    """
    将热力图叠加在原图上进行可视化
    :param image:
    :param heatmap:
    :return:
    """
    # 灰度化heatmap
    heatmap_g = (heatmap * 255).astype(np.uint8)
    # 热力图伪彩色
    heatmap_color = cv2.applyColorMap(heatmap_g, cv2.COLORMAP_JET)
    # overlay热力图
    merge_img = image.copy()
    heatmap_img = heatmap_color.copy()
    overlay = image.copy()
    alpha = 0.25  # 设置覆盖图片的透明度
    #cv2.rectangle(overlay, (0, 0), (merge_img.shape[1], merge_img.shape[0]), (0, 0, 0), -1) # 设置蓝色为热度图基本色
    # cv2.addWeighted(overlay, alpha, merge_img, 1-alpha, 0, merge_img) # 将背景热度图覆盖到原图
    cv2.addWeighted(heatmap_img, alpha, merge_img, 1-alpha, 0, merge_img) # 将热度图覆盖到原图
    return merge_img

def feat_locations(features, stride=4):
    """
    Arguments:
        features:  (N, C, H, W)
    Return:
        locations:  (2, H, W)
    """

    h, w = features.shape[2:]

    shifts_x = paddle.arange(
        0, w * stride,
        step=stride,
        dtype='float32'
    )
    shifts_y = paddle.arange(
        0, h * stride,
        step=stride,
        dtype='float32'
    )

    shift_y, shift_x = paddle.meshgrid(shifts_y, shifts_x)
    shift_x = paddle.flatten(shift_x, start_axis=0)
    shift_y = paddle.flatten(shift_y, start_axis=0)
    locations = paddle.stack((shift_x, shift_y), axis=1) + stride // 2

    locations = paddle.transpose(paddle.reshape(locations, shape=(h, w, 2)), perm=[2, 0, 1])

    return locations

def resize_kp_w(center, header, w):
    x1, y1 = center
    x2, y2 = header

    d1 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # 中心点到头部关键点的距离

def triangle_l_points(points, h):
    d1 = math.sqrt((points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2)
    d2 = math.sqrt((points[1][0] - points[2][0]) ** 2 + (points[1][1] - points[2][1]) ** 2)

    if d1 < d2:
        x = (points[1][0] + points[2][0]) / 2
        y = (points[1][1] + points[2][1]) / 2
    else:
        x = (points[0][0] + points[1][0]) / 2
        y = (points[0][1] + points[1][1]) / 2

    return (x, y)

def compute_mbox_4mid_points(centers, headers, ls):
    '''
    根据参数计算旋转框4点坐标, 从头部关键点算point1，。。。顺时针以此类推
    :param centers: 中心点坐标  [N, 2]
    :param headers: 头部关键点坐标  [N, 2]
    :param ls: 旋转框短边框  [N]
    :return:
    '''
    center_xy = paddle.transpose(centers, perm=[1, 0])  # [N, 2]

    point1 = paddle.transpose(headers, perm=[1, 0])  # [N, 2]

    point3_x = paddle.unsqueeze(2 * center_xy[:, 0] - point1[:, 0], axis=1)  # [N]
    point3_y = paddle.unsqueeze(2 * center_xy[:, 1] - point1[:, 1], axis=1)  # [N]
    point3 = paddle.concat([point3_x, point3_y], axis=1)  # [N, 2]

    point2_x = center_xy[:, 0] - ls * paddle.sin(paddle.atan((point1[:, 1]-center_xy[:, 1])/(point1[:, 0]-center_xy[:, 0])))
    point2_y = center_xy[:, 1] + ls * paddle.cos(paddle.atan((point1[:, 1]-center_xy[:, 1])/(point1[:, 0]-center_xy[:, 0])))
    point2 = paddle.concat([paddle.unsqueeze(point2_x, axis=1), paddle.unsqueeze(point2_y, axis=1)], axis=1)

    point4_x = center_xy[:, 0] + ls * paddle.sin(paddle.atan((point1[:, 1]-center_xy[:, 1])/(point1[:, 0]-center_xy[:, 0])))
    point4_y = center_xy[:, 1] - ls * paddle.cos(paddle.atan((point1[:, 1] - center_xy[:, 1]) / (point1[:, 0] - center_xy[:, 0])))
    point4 = paddle.concat([paddle.unsqueeze(point4_x, axis=1), paddle.unsqueeze(point4_y, axis=1)], axis=1)

    # 1-2
    mid_point1_x = (point1[:, 0] + point2[:, 0]) / 2
    mid_point1_y = (point1[:, 1] + point2[:, 1]) / 2
    box_point1_x = paddle.unsqueeze(2 * mid_point1_x - center_xy[:, 0], axis=1)
    box_point1_y = paddle.unsqueeze(2 * mid_point1_y - center_xy[:, 1], axis=1)
    box_point1 = paddle.concat([box_point1_x, box_point1_y], axis=1)

    # 2-3
    mid_point2_x = (point2[:, 0] + point3[:, 0]) / 2
    mid_point2_y = (point2[:, 1] + point3[:, 1]) / 2
    box_point2_x = paddle.unsqueeze(2 * mid_point2_x - center_xy[:, 0], axis=1)
    box_point2_y = paddle.unsqueeze(2 * mid_point2_y - center_xy[:, 1], axis=1)
    box_point2 = paddle.concat([box_point2_x, box_point2_y], axis=1)

    # 3-4
    mid_point3_x = (point3[:, 0] + point4[:, 0]) / 2
    mid_point3_y = (point3[:, 1] + point4[:, 1]) / 2
    box_point3_x = paddle.unsqueeze(2 * mid_point3_x - center_xy[:, 0], axis=1)
    box_point3_y = paddle.unsqueeze(2 * mid_point3_y - center_xy[:, 1], axis=1)
    box_point3 = paddle.concat([box_point3_x, box_point3_y], axis=1)

    # 4-1
    mid_point4_x = (point4[:, 0] + point1[:, 0]) / 2
    mid_point4_y = (point4[:, 1] + point1[:, 1]) / 2
    box_point4_x = paddle.unsqueeze(2 * mid_point4_x - center_xy[:, 0], axis=1)
    box_point4_y = paddle.unsqueeze(2 * mid_point4_y - center_xy[:, 1], axis=1)
    box_point4 = paddle.concat([box_point4_x, box_point4_y], axis=1)

    box_point1 = paddle.unsqueeze(box_point1, axis=1) # [N, 1, 2]
    box_point2 = paddle.unsqueeze(box_point2, axis=1)
    box_point3 = paddle.unsqueeze(box_point3, axis=1)
    box_point4 = paddle.unsqueeze(box_point4, axis=1)

    box_points = paddle.concat([box_point1, box_point2, box_point3, box_point4], axis=1)  # [N, 4, 2]

    return box_points

def convert_mboxes_corners(centers, headers, ls):
    """
    根据中心点坐标，头部坐标和旋转框宽计算最终旋转框4角点坐标
    :param centers: [N, 2]
    :param headers: [N, 2]
    :param ls: [N]
    :return:
    """

    point1 = headers

    point3 = np.zeros_like(centers)
    point3[:, 0] = 2 * centers[:, 0] - point1[:, 0]  # [N]
    point3[:, 1] = 2 * centers[:, 1] - point1[:, 1]  # [N]

    point2 = np.zeros_like(centers)
    point2[:, 0] = centers[:, 0] - ls * np.sin(
        np.arctan((point1[:, 1] - centers[:, 1]) / (point1[:, 0] - centers[:, 0])))
    point2[:, 1] = centers[:, 1] + ls * np.cos(
        np.arctan((point1[:, 1] - centers[:, 1]) / (point1[:, 0] - centers[:, 0])))

    point4 = np.zeros_like(centers)
    point4[:, 0] = centers[:, 0] + ls * np.sin(
        np.arctan((point1[:, 1] - centers[:, 1]) / (point1[:, 0] - centers[:, 0])))
    point4[:, 1] = centers[:, 1] - ls * np.cos(
        np.arctan((point1[:, 1] - centers[:, 1]) / (point1[:, 0] - centers[:, 0])))

    # 1-2
    mid_point1_x = (point1[:, 0] + point2[:, 0]) / 2
    mid_point1_y = (point1[:, 1] + point2[:, 1]) / 2
    box_point1 = np.zeros_like(centers)
    box_point1[:, 0] = 2 * mid_point1_x - centers[:, 0]
    box_point1[:, 1] = 2 * mid_point1_y - centers[:, 1]

    # 2-3
    mid_point2_x = (point2[:, 0] + point3[:, 0]) / 2
    mid_point2_y = (point2[:, 1] + point3[:, 1]) / 2
    box_point2 = np.zeros_like(centers)
    box_point2[:, 0] = 2 * mid_point2_x - centers[:, 0]
    box_point2[:, 1] = 2 * mid_point2_y - centers[:, 1]

    # 3-4
    mid_point3_x = (point3[:, 0] + point4[:, 0]) / 2
    mid_point3_y = (point3[:, 1] + point4[:, 1]) / 2
    box_point3 = np.zeros_like(centers)
    box_point3[:, 0] = 2 * mid_point3_x - centers[:, 0]
    box_point3[:, 1] = 2 * mid_point3_y - centers[:, 1]

    # 4-1
    mid_point4_x = (point4[:, 0] + point1[:, 0]) / 2
    mid_point4_y = (point4[:, 1] + point1[:, 1]) / 2
    box_point4 = np.zeros_like(centers)
    box_point4[:, 0] = 2 * mid_point4_x - centers[:, 0]
    box_point4[:, 1] = 2 * mid_point4_y - centers[:, 1]

    box_point1 = np.expand_dims(box_point1, axis=1)  # [N, 1, 2]
    box_point2 = np.expand_dims(box_point2, axis=1)
    box_point3 = np.expand_dims(box_point3, axis=1)
    box_point4 = np.expand_dims(box_point4, axis=1)

    box_points = np.concatenate([box_point1, box_point2, box_point3, box_point4], axis=1)  # [N, 4, 2]

    return box_points

def show_mbox(image, points, color=(0, 255, 255), thickness=2):
    le = cv2.line(image, (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1])), color, thickness=thickness)
    le = cv2.line(image, (int(points[1][0]), int(points[1][1])), (int(points[2][0]), int(points[2][1])), color, thickness=thickness)
    le = cv2.line(image, (int(points[2][0]), int(points[2][1])), (int(points[3][0]), int(points[3][1])), color, thickness=thickness)
    le = cv2.line(image, (int(points[3][0]), int(points[3][1])), (int(points[0][0]), int(points[0][1])), color, thickness=thickness)

    return le


def show_mboxes(image, mboxes, color=(0, 255, 255)):
    new_image = image.copy()
    for mbox in mboxes:
        show_mbox(new_image, np.int32(mbox), color)

    return new_image

def two_points_distance(pt1, pt2):
    '''
    批量计算两点距离 [x , y]
    :param pt1: [N, 2]
    :param pt2: [N, 2]
    :return:
    '''
    distance = np.sqrt((pt1[:, 0] - pt2[:, 0])**2 + (pt1[:, 1] - pt2[:, 1])**2)

    return distance

def azimuthAngle(pt1, pt2):
    '''
    计算gps方位角(pt1为起点， pt2为终点)
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    '''
    x1 = pt1[0, :]
    y1 = pt1[1, :]
    x2 = pt2[0, :]
    y2 = pt2[1, :]

    # angle = math.atan2((y2-y1), (x2-x1))
    angle = paddle.atan((y2-y1), (x2-x1))
    theta = angle*(180/math.pi)
    return theta

def scale_xy(xy, old_size, new_size):
    '''
    将bbox数值根据输入图像大小重新按比例resize到相应大小
    :param xy: [x, y]  [N, 2]
    :param new_size: [w, h]
    :param new_size: [w, h]
    :return:
    '''
    x, y = xy[:, 0], xy[:, 1]
    old_w, old_h = old_size
    new_w, new_h = new_size

    x_ratio = new_w / old_w
    y_ratio = new_h / old_h

    new_xy = np.zeros_like(xy)
    new_xy[:, 0] = x * x_ratio
    new_xy[:, 1] = y * y_ratio

    return new_xy

def header_match(xy, location):
    '''
    将回归的到的头部关键点与热力图的得到的头部关键点做最近匹配
    :param header_xy: [N, 2]  输入图片尺寸
    :param header_location: [N, 2]  特征图尺寸
    :return:
    '''

    xy = np.expand_dims(xy, axis=1)  # [N, 1, 2]
    xy = np.tile(xy, [1, len(location), 1])  # [N, N, 2]

    location = np.expand_dims(location, axis=0)
    location = np.tile(location, [len(xy), 1, 1])  # [N, N, 2]

    # 计算距离取距离最小的索引
    distance = np.sqrt((xy[..., 0] - location[..., 0])**2+(xy[..., 1] - location[..., 1])**2)  # [N_xy, N_location]
    min_ind = np.argmin(distance, axis=1)

    return min_ind

