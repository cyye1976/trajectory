import numpy as np
import cv2
import paddle
from imgaug import augmenters as iaa, SegmentationMapsOnImage, ia, KeypointsOnImage, Keypoint, math

from model.data.utils.visualize import show_mboxes
from model.utils.bbox_util import bbox_iou_rotate_calculate1
from model.utils.iou import vector2rotated, cross_iou
from model.utils.rbox_iou import rbox_poly_overlaps

color_map = [
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (100, 149, 237)

]


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
    rad = math.radians(ang)
    theta_x = x * math.cos(rad) + y * math.sin(rad)
    theta_y = -x * math.sin(rad) + y * math.cos(rad)
    vector = [theta_x + cx, theta_y + cy]

    return vector


# def vector2rotated2(cx, cy, w, h, ang):
#     """
#     将旋转框的角度根据h先正矩形表示然后“以中心点沿正y轴h的距离作为向量”
#     :param cx:
#     :param cy:
#     :param hx:
#     :param hy:
#     :return:
#     """
#
#     # 计算出旋转前的向量
#     vector = [[-w, h], [w, h]]
#
#     # 根据旋转角度计算旋转后
#     x, y = vector
#     rad = math.radians(ang)
#     theta_x = x * math.cos(rad) + y * math.sin(rad)
#     theta_y = -x * math.sin(rad) + y * math.cos(rad)
#     vector = [theta_x + cx, theta_y + cy]
#
#     return vector

# def compute_cross_vectors(vectors, alpha):
#     """
#      计算向量沿四向坐标系分离的四个值
#      :param vectors:
#      :return:
#      """
#     vectors_x = paddle.tile(paddle.unsqueeze(vectors[:, :, 0], axis=-1), [1, 1, 4])
#     vectors_y = paddle.tile(paddle.unsqueeze(vectors[:, :, 1], axis=-1), [1, 1, 4])
#
#     xgt = paddle.concat([
#         paddle.tile(paddle.to_tensor([alpha]), [vectors.shape[0], vectors.shape[1], 1]),
#         paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.ones([vectors.shape[0], vectors.shape[1], 1])], axis=-1)
#     xlt = paddle.concat([
#         paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.tile(paddle.to_tensor([alpha]), [vectors.shape[0], vectors.shape[1], 1]),
#         paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.ones([vectors.shape[0], vectors.shape[1], 1])], axis=-1)
#     ygt = paddle.concat([
#         paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.tile(paddle.to_tensor([alpha]), [vectors.shape[0], vectors.shape[1], 1])], axis=-1)
#     ylt = paddle.concat([
#         paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.tile(paddle.to_tensor([alpha]), [vectors.shape[0], vectors.shape[1], 1]),
#         paddle.ones([vectors.shape[0], vectors.shape[1], 1])], axis=-1)
#
#     vectors_x_alpha = paddle.where(vectors_x >= 0, xgt, xlt)
#     vectors_y_alpha = paddle.where(vectors_y >= 0, ygt, ylt)
#     vectors_alpha = paddle.unsqueeze(vectors_y_alpha[:, :, 2], axis=-1), paddle.unsqueeze(vectors_x_alpha[:, :, 0],
#                                                                                           axis=-1), paddle.unsqueeze(
#         vectors_y_alpha[:, :, 3], axis=-1), paddle.unsqueeze(vectors_x_alpha[:, :, 1], axis=-1)
#     vectors_alpha = paddle.concat(vectors_alpha, axis=-1)
#
#     vectors_cross = paddle.concat(
#         [paddle.unsqueeze(vectors[:, :, 1], axis=-1), paddle.unsqueeze(vectors[:, :, 0], axis=-1)], axis=-1)
#     vectors_cross = paddle.tile(vectors_cross, [1, 2]) * vectors_alpha
#
#     return paddle.abs(vectors_cross)
#
#
# def cross_iou(vectors, target_vectors, alpha=0.2):
#     """
#     计算四边形iou（顶点向量计算方法）
#     :param vectors: [N, 4, 2]
#     :param target_vectors:  [N, 4, 2]
#     :param alpha: 系数
#     :return:
#     """
#
#     vectors_cross = compute_cross_vectors(vectors, alpha)
#     target_vectors_cross = compute_cross_vectors(target_vectors, alpha)
#
#     min_v = paddle.sum(paddle.minimum(vectors_cross, target_vectors_cross), axis=-1)
#     max_v = paddle.sum(paddle.maximum(vectors_cross, target_vectors_cross), axis=-1)
#     iou = paddle.mean(min_v / max_v)
#
#     return iou


def show(image, mbox, mbox_points, color=((255, 0, 0))):
    """
    验证定向边界框四顶点向量旋转
    :return:
    """

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


def main():
    # show()
    # 仿真定向边界框
    image = np.zeros([512, 512, 3])

    mbox1 = ((256, 256), (50, 100), 0)
    # mbox_points1 = cv2.boxPoints(mbox1)
    mbox_points1, vectors1 = vector2rotated(np.array([[256, 256]]) / 4, np.array([[100, 50]]) / 4, 0)
    mbox_points1 *= 4

    mbox2 = ((256, 256), (50, 100), 0)
    # mbox_points2 = cv2.boxPoints(mbox2)
    mbox_points2, vectors2 = vector2rotated(np.array([[256, 256]]) / 4, np.array([[100, 50]]) / 4, 0)
    mbox_points2 *= 4

    #
    show(image, mbox1, mbox_points1[0])
    show(image, mbox2, mbox_points2[0], color=((0, 0, 255)))
    #
    # # 计算两个旋转框的iou
    # iou1 = rbox_poly_overlaps(mbox_points1.reshape(-1, 8), mbox_points2.reshape(-1, 8))

    iou2 = paddle.to_tensor(
        bbox_iou_rotate_calculate1(
            np.expand_dims(mbox_points1, axis=0),
            np.expand_dims(mbox_points2, axis=0)))
    iou3 = cross_iou(
        paddle.to_tensor(vectors1),
        paddle.to_tensor(vectors2), alpha=0.2)

    # vectors = paddle.rand([3, 4, 2])
    # target_vectors = paddle.rand([3, 4, 2])
    #
    # iou = cross_iou(vectors, target_vectors)

    pass


if __name__ == '__main__':
    main()
