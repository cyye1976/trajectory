import cv2
from imgaug import augmenters as iaa, SegmentationMapsOnImage, ia, KeypointsOnImage, Keypoint
import numpy as np

def show_mboxes(image, mboxes, color=(0, 255, 0), thickness=1):
    """
    可视化带头定向边界框
    :param mboxes: [N, 4, 2]
    :return:
    """
    draw_image = image.copy()
    for mbox in mboxes:
        box = mbox
        draw_image = draw_box(image, box, color, thickness)

    return draw_image

def draw_box(image, points, color=(0, 255, 0), thickness=1):
    le = cv2.line(image, (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1])), color, thickness=thickness)
    le = cv2.line(image, (int(points[1][0]), int(points[1][1])), (int(points[2][0]), int(points[2][1])), color, thickness=thickness)
    le = cv2.line(image, (int(points[2][0]), int(points[2][1])), (int(points[3][0]), int(points[3][1])), color, thickness=thickness)
    le = cv2.line(image, (int(points[3][0]), int(points[3][1])), (int(points[0][0]), int(points[0][1])), color, thickness=thickness)

    return le

def draw_mask_in_image(image, mask, alpha=1, beta=0.5, gamma=0):
    # alpha 为第一张图片的透明度
    # beta 为第二张图片的透明度
    mask_img = cv2.addWeighted(image, alpha, mask, beta, gamma)
    return mask_img


def vis_box(image, mbox, mbox_points, color=((255, 0, 0))):
    """
    验证定向边界框四顶点向量旋转
    :return:
    """

    def vector2rotated1(cx, cy, hx, hy, ang):
        """
        将旋转框的角度根据h先正矩形表示然后“以中心点沿正y轴h的距离作为向量”
        :param cx:
        :param cy:
        :param hx:
        :param hy:
        :return:
        """

        # 计算出旋转前的向量
        vector = [hx - cx, hy - cy]

        # 根据旋转角度计算旋转后
        x, y = vector
        import math
        rad = math.radians(ang)
        theta_x = x * math.cos(rad) + y * math.sin(rad)
        theta_y = -x * math.sin(rad) + y * math.cos(rad)
        vector = [theta_x + cx, theta_y + cy]

        return vector

    color_map = [
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (100, 149, 237)

    ]

    # 可视化
    vectors = []
    for i, point in enumerate(mbox_points):
        vector = np.array(vector2rotated1(mbox[0][0], mbox[0][1], point[0], point[1], 0), dtype='int64')
        draw_image = cv2.line(image, (mbox[0][0], mbox[0][1]), (vector[0], vector[1]), color_map[i], thickness=2)
        vectors.append(vector)

    vectors = np.array(vectors)
    kps = KeypointsOnImage([Keypoint(x=p[0], y=p[1]) for p in vectors], shape=image.shape)
    draw_image = show_mboxes(draw_image, mboxes=np.array([vectors]), color=color, thickness=2)
    draw_image = kps.draw_on_image(draw_image, size=6)
    ia.imshow(draw_image)