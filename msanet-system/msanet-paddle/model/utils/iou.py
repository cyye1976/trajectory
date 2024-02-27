
import numpy as np
import paddle

from model.data.utils.visualize import vis_box
from model.utils.bbox_util import bbox_iou_rotate_calculate1


def vector2rotated(center, wh, theta):
    """
    将旋转框的角度根据h先正矩形表示然后“以中心点沿正y轴h的距离作为向量”
    :param center: [N, 2]
    :param wh: [N, 2]
    :param theta: [N]
    :return: [N, 4, 2]
    要注意的是这里的vectors 是下采样4倍后的数值
    """
    ang = np.squeeze(theta, axis=-1)
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

def compute_cross_vectors(vectors, alpha=0.2):
    """
     计算向量沿四向坐标系分离的四个值
     :param vectors:
     :return:
     """
    vectors_x = paddle.tile(paddle.unsqueeze(vectors[:, :, 0], axis=-1), [1, 1, 4])
    vectors_y = paddle.tile(paddle.unsqueeze(vectors[:, :, 1], axis=-1), [1, 1, 4])

    xgt = paddle.concat([
        paddle.tile(paddle.to_tensor([alpha]), [vectors.shape[0], vectors.shape[1], 1]),
        paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
        paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
        paddle.ones([vectors.shape[0], vectors.shape[1], 1])], axis=-1)
    xlt = paddle.concat([
        paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
        paddle.tile(paddle.to_tensor([alpha]), [vectors.shape[0], vectors.shape[1], 1]),
        paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
        paddle.ones([vectors.shape[0], vectors.shape[1], 1])], axis=-1)
    ygt = paddle.concat([
        paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
        paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
        paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
        paddle.tile(paddle.to_tensor([alpha]), [vectors.shape[0], vectors.shape[1], 1])], axis=-1)
    ylt = paddle.concat([
        paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
        paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
        paddle.tile(paddle.to_tensor([alpha]), [vectors.shape[0], vectors.shape[1], 1]),
        paddle.ones([vectors.shape[0], vectors.shape[1], 1])], axis=-1)

    vectors_x_alpha = paddle.where(vectors_x >= 0, xgt, xlt)
    vectors_y_alpha = paddle.where(vectors_y >= 0, ygt, ylt)
    vectors_alpha = paddle.unsqueeze(vectors_y_alpha[:, :, 2], axis=-1), paddle.unsqueeze(vectors_x_alpha[:, :, 0],
                                                                                          axis=-1), paddle.unsqueeze(
        vectors_y_alpha[:, :, 3], axis=-1), paddle.unsqueeze(vectors_x_alpha[:, :, 1], axis=-1)
    vectors_alpha = paddle.concat(vectors_alpha, axis=-1)

    vectors_cross = paddle.concat(
        [paddle.unsqueeze(vectors[:, :, 1], axis=-1), paddle.unsqueeze(vectors[:, :, 0], axis=-1)], axis=-1)
    vectors_cross = paddle.tile(vectors_cross, [1, 2]) * vectors_alpha

    return paddle.abs(vectors_cross)


def cross_iou(vectors, target_vectors, reduction = 'mean'):
    """
    计算四边形iou（顶点向量计算方法）
    :param vectors: [N, 4, 4] 定向边界框顶点四分量预测值
    :param target_vectors:  [N, 4, 4] 定向边界框顶点四分量标签值
    :param alpha: 系数
    :return:
    """

    min_v = paddle.sum(paddle.minimum(vectors, target_vectors), axis=-1)
    max_v = paddle.sum(paddle.maximum(vectors, target_vectors), axis=-1)
    if reduction == 'none':
        iou = paddle.mean(min_v / max_v, axis=-1)
    else:
        iou = paddle.mean(min_v / max_v)

    return iou

if __name__ == '__main__':
    # 仿真定向边界框
    image = np.zeros([512, 512, 3])

    mbox1 = ((256, 256), (50, 100), 0)
    # mbox_points1 = cv2.boxPoints(mbox1)
    mbox_points1, vectors1 = vector2rotated(np.array([[256, 256]]) / 4, np.array([[100, 50]]) / 4, 0)
    mbox_points1 *= 4

    mbox2 = ((256, 256), (50, 100), 0)
    # mbox_points2 = cv2.boxPoints(mbox2)
    mbox_points2, vectors2 = vector2rotated(np.array([[256, 256]]) / 4, np.array([[100, 50]]) / 4, 2)
    mbox_points2 *= 4

    #
    vis_box(image, mbox1, mbox_points1[0])
    vis_box(image, mbox2, mbox_points2[0], color=((0, 0, 255)))
    #
    # # 计算两个旋转框的iou
    # iou1 = rbox_poly_overlaps(mbox_points1.reshape(-1, 8), mbox_points2.reshape(-1, 8))

    iou2 = paddle.to_tensor(
        bbox_iou_rotate_calculate1(
            np.expand_dims(mbox_points1, axis=0),
            np.expand_dims(mbox_points2, axis=0)))
    vectors1 = compute_cross_vectors(paddle.to_tensor(vectors1, dtype='float32'), 0.3)
    vectors2 = compute_cross_vectors(paddle.to_tensor(vectors2, dtype='float32'), 0.3)
    iou3 = cross_iou(vectors1, vectors2)
    pass