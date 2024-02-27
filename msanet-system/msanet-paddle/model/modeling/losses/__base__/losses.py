import paddle
import paddle.nn.functional as F

# 计算分量向量
# def compute_cross_vectors(vectors, alpha):
#     """
#     计算向量沿四向坐标系分离的四个值
#     :param vectors:
#     :return:
#     """
#     vectors_x = paddle.tile(paddle.unsqueeze(vectors[:, :, 0], axis=-1), [1, 1, 4])
#     vectors_y = paddle.tile(paddle.unsqueeze(vectors[:, :, 1], axis=-1), [1, 1, 4])
#
#     # TODO：如何去解决划分4向量另外两个向量的alpha乘积
#     xgt = paddle.concat([
#         paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.zeros([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.zeros([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.zeros([vectors.shape[0], vectors.shape[1], 1])], axis=-1)
#     xlt = paddle.concat([
#         paddle.zeros([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.zeros([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.zeros([vectors.shape[0], vectors.shape[1], 1])], axis=-1)
#     ygt = paddle.concat([
#         paddle.zeros([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.zeros([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.zeros([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.ones([vectors.shape[0], vectors.shape[1], 1])], axis=-1)
#     ylt = paddle.concat([
#         paddle.zeros([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.zeros([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.ones([vectors.shape[0], vectors.shape[1], 1]),
#         paddle.zeros([vectors.shape[0], vectors.shape[1], 1])], axis=-1)
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
#
#     vectors_cross = paddle.tile(vectors_cross, [1, 2])
#     vectors_cross = paddle.where(vectors_alpha == 1, vectors_cross * alpha, vectors_cross)
#
#     return paddle.abs(vectors_cross)
#
# # 计算分量向量iou
# def cross_iou(vectors, target_vectors, alpha=0.2):
#     """
#      计算四边形iou（顶点向量计算方法）
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

# smooth L1 loss
from model.utils.iou import cross_iou


def reg_l1_loss(pred, target, mask, weights=None):
    # --------------------------------#
    #   计算l1_loss
    # --------------------------------#
    pred = paddle.transpose(pred, perm=(0, 2, 3, 1))
    expand_mask = paddle.tile(paddle.unsqueeze(mask, axis=-1), (1, 1, 1, pred.shape[3]))
    if weights is not None:
        loss = F.smooth_l1_loss(pred * expand_mask, target * expand_mask, reduction='none')
        loss = paddle.sum(loss * weights)
    else:
        loss = F.smooth_l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (paddle.sum(mask) + 1e-4)
    return loss

def cross_iou_loss(pred, target, mask):

    new_pred = paddle.reshape(paddle.transpose(pred, [0, 2, 3, 1]), [-1, 4, 4])
    new_target = paddle.reshape(target, [-1, 4, 4])
    expand_mask = paddle.flatten(mask)

    loss = (1 - cross_iou(paddle.abs(new_pred), paddle.abs(new_target), reduction='none')) * expand_mask
    loss = paddle.sum(loss) / (paddle.sum(expand_mask) + 1e-4)

    return loss
