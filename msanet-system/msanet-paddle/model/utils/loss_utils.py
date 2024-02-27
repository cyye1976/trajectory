import math
import paddle
import numpy as np
import paddle.nn.functional as F

def compute_vector_iou_gpu(wha, target_wha):
    """
    计算旋转框iou（根据转化向量计算参考LSNET论文中的Cross-iou）
    :param wha: [N, C] C:[w, h, theta]
    :param target_wha: [N, C] C:[w, h, theta]
    :return:
    """
    # 角度转化为坐标系标准[-90, 0),(0 , 90]
    vectors = compute_vector_gpu(wha)  # [N, 3] [w, h, theta]
    target_vectors = compute_vector_gpu(target_wha)  # [N, 3] [w, h, theta]

    # # 旋转框转化为四个向量，每个向量单独求iou，最后求1-平均iou
    vectors = compute_theta_vector_gpu(vectors)  # [N, 4, 2]
    target_vectors = compute_theta_vector_gpu(target_vectors)  # [N, 4, 2]

    # 拆解向量到映射到坐标系
    vectors = convert_coordinate2vectors_gpu(vectors)  # [N, 4, 4]
    target_vectors = convert_coordinate2vectors_gpu(target_vectors)  # [N, 4, 4]

    # 计算每个向量iou
    min_v = paddle.minimum(vectors, target_vectors)  # [N, 4, 4]
    max_v = paddle.maximum(vectors, target_vectors)  # [N, 4, 4]
    ious = paddle.sum(paddle.sum(min_v, axis=-1) / paddle.sum(max_v, axis=-1), axis=1) / 4.  # [N]

    # TODO:用于DEBUG梯度回传是否成功
    # ious = F.smooth_l1_loss(min_v, max_v)

    return ious


def compute_theta_vector_gpu(vectors):
    """
    计算角度旋转后的向量
    :param vectors: [w, h, theta] [N, 3]
    :return:
    """
    w, h, theta = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    new_vectors = [[w, h], [-w, h], [-w, -h], [w, -h]]  # 按象限顺序 [N, 2]
    rad = np.pi / 180 * theta
    theta_vectors = []
    for vector in new_vectors:
        x, y = vector[0], vector[1]
        theta_x = x * paddle.cos(rad) + y * paddle.sin(rad)
        theta_y = -x * paddle.sin(rad) + y * paddle.cos(rad)
        theta_vectors.append(paddle.unsqueeze(paddle.concat([paddle.unsqueeze(theta_x, axis=1), paddle.unsqueeze(theta_y, axis=1)], axis=1), axis=1))

    new_vectors = paddle.concat([theta_vectors[0], theta_vectors[1], theta_vectors[2], theta_vectors[3]], axis=1)

    return new_vectors


def convert_coordinate2vectors_gpu(coordinates, alpha=0.2):
    """
    将向量投影到坐标系上
    :param coordinates: [N, 4, 2]
    :return: [N, 4, 4]
    """
    vectors = paddle.concat([paddle.unsqueeze(coordinates[:, :, 1], axis=-1), paddle.unsqueeze(coordinates[:, :, 0], axis=-1)], axis=-1)  # [y, x]
    vectors = paddle.tile(vectors, [1, 1, 2])  # [y, x, y, x]

    vp1 = paddle.unsqueeze(vectors[:, :, 0], axis=-1)
    vp2 = paddle.unsqueeze(vectors[:, :, 1], axis=-1)
    vp3 = paddle.unsqueeze(vectors[:, :, 2], axis=-1)
    vp4 = paddle.unsqueeze(vectors[:, :, 3], axis=-1)

    vp24 = paddle.where(paddle.tile(paddle.unsqueeze(coordinates[:, :, 0] > 0, axis=-1), [1, 2]), paddle.concat([vp2*alpha, vp4], axis=-1), paddle.concat([vp2, vp4*alpha], axis=-1))
    vp13 = paddle.where(paddle.tile(paddle.unsqueeze(coordinates[:, :, 1] > 0, axis=-1), [1, 2]), paddle.concat([vp1, vp3*alpha], axis=-1), paddle.concat([vp1*alpha, vp3], axis=-1))
    vp1 = paddle.unsqueeze(vp13[:, :, 0], axis=-1)
    vp2 = paddle.unsqueeze(vp24[:, :, 0], axis=-1)
    vp3 = paddle.unsqueeze(vp13[:, :, 1], axis=-1)
    vp4 = paddle.unsqueeze(vp24[:, :, 1], axis=-1)
    vectors = paddle.abs(paddle.concat([vp1, vp2, vp3, vp4], axis=-1))

    return vectors

def compute_vector_gpu(wha):
    """
    将旋转框的原始opencv角度转化为以坐标系为基准 并以[x, y, theta]向量表示
    :param wha: [N, C] C:[w, h, theta]
    :return:
    """
    new_wha = paddle.where(paddle.tile(paddle.unsqueeze(wha[:, 0] < wha[:, 1], axis=1), [1, 3]), wha, paddle.concat(
        [paddle.unsqueeze(wha[:, 1], axis=1), paddle.unsqueeze(wha[:, 0], axis=1),
         paddle.unsqueeze(-(90 - wha[:, -1]), axis=1)], axis=1))
    return new_wha

def convert_vector_mbox(mbox):
    """
    将旋转框的原始opencv角度转化为以坐标系为基准 并以[x, y, theta]向量表示
    :param mbox:[w, h, theta]
    :return:
    """
    if mbox[0] < mbox[1]:
        return mbox
    else:
        return [mbox[1], mbox[0], -(90-mbox[-1])]

def compute_vector_iou(mbox, target_mbox):
    """
    通过向量计算旋转框的iou
    :param mbox: [w, h, theta]
    :param target_mbox: [w, h, theta]
    :return:
    """
    # 角度转化为坐标系标准[-90, 0),(0 , 90]
    vector_mbox = convert_vector_mbox(mbox)
    target_vector_mbox = convert_vector_mbox(target_mbox)

    # 旋转框转化为四个向量，每个向量单独求iou，最后求1-平均iou
    coordinates = compute_theta_vector(vector_mbox)
    target_coordinates = compute_theta_vector(target_vector_mbox)

    # 拆解向量到映射到坐标系
    vectors = convert_coordinate2vectors(coordinates)
    target_vectors = convert_coordinate2vectors(target_coordinates)

    # 计算每个向量iou
    ious = []
    for i in range(len(vectors)):
        min_v = np.minimum(vectors[i], target_vectors[i])
        max_v = np.maximum(vectors[i], target_vectors[i])
        ious.append(np.sum(min_v) / np.sum(max_v))

    iou = np.array(ious).mean()

    return iou

def convert_coordinate2vectors(coordinates, alpha=0.2):
    """
    将向量投影到坐标系上
    :param coordinates: [N, 4]
    :return:
    """
    vectors = np.zeros([4, 4])
    for i, coordinate in enumerate(coordinates):
        if coordinate[0] > 0:
            vectors[i][3] = coordinate[0]
            vectors[i][1] = coordinate[0] * alpha
        else:
            vectors[i][1] = coordinate[0]
            vectors[i][3] = coordinate[0] * alpha
        if coordinate[1] > 0:
            vectors[i][0] = coordinate[1]
            vectors[i][2] = coordinate[1] * alpha
        else:
            vectors[i][2] = coordinate[1]
            vectors[i][0] = coordinate[1] * alpha
    return np.abs(vectors)


def compute_theta_vector(mbox):
    w, h, theta = mbox
    vectors = [[w, h], [-w, h], [-w, -h], [w, -h]]  # 按象限顺序
    # 计算旋转后的向量
    theta_vectors = []
    rad = math.radians(theta)
    for vector in vectors:
        x, y = vector
        theta_x = x * math.cos(rad) + y * math.sin(rad)
        theta_y = -x * math.sin(rad) + y * math.cos(rad)
        theta_vectors.append([theta_x, theta_y])

    return theta_vectors

if __name__ == '__main__':

    # mbox = [0, -2, -90]
    # theta_vector = compute_theta_vector(mbox)
    # print(theta_vector)
    # [w, h, theta]
    mbox = [1, 2, 0.1]
    target_mbox = [1, 2, 90]
    iou = compute_vector_iou(mbox, target_mbox)
    print(iou)

    mbox = paddle.to_tensor([[1, 2, 0.1]], dtype='float32')
    target_mbox = paddle.to_tensor([[1, 2, 90]], dtype='float32')
    ious = compute_vector_iou_gpu(mbox, target_mbox)
    print(ious)