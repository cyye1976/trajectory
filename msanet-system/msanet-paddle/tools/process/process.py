import os
from shutil import copy
from random import shuffle

import imageio
import numpy as np

from model.data.utils.color_map import color_list
from model.data.utils.transform import mask_to_onehot


class SSDDPath:
    """
    SSDD+ 雷达遥感图像船舶检测数据集
    """
    dataset_dir = 'E:/datasets/SSDD/'  # 数据集文件夹绝对路径
    images_dir = dataset_dir + 'JPEGImages/'  # 图片文件夹路径
    annotations_obb_dir = dataset_dir + 'Annotations_obb/Annotations/'  # 旋转框标注文件夹
    annotations_hbb_dir = dataset_dir + 'Annotations_hbb/Annotations/'  # 水平框标注文件夹
    
class HRSC2016DSPath:
    mode = 'train'
    dataset_dir = 'D:/dataset/main/HRSC2016-DS/'
    images_dir = 'D:/dataset/main/HRSC2016-DS/{}/seg_labels/'.format(mode)
    id_map_dir = 'D:/dataset/main/HRSC2016-DS/{}/id_map/'.format(mode)


def validate_dateset(path):
    """
    验证数据集
    :param path: 数据集路径类实例
    :return:
    """
    annotation_obb_files = os.listdir(path.annotations_obb_dir)
    annotation_hbb_files = os.listdir(path.annotations_hbb_dir)
    image_files = os.listdir(path.images_dir)

    invalid_datas = []
    for filename in annotation_obb_files:
        if filename[:-3] + 'jpg' not in image_files:
            invalid_datas.append(filename)
    for filename in annotation_hbb_files:
        if filename[:-3] + 'jpg' not in image_files:
            invalid_datas.append(filename)
    return invalid_datas


def split_train_val_test(path, ratios=None, is_random=True):
    """
    划分数据集
    :param ratios: 数据集划分比例（训练集，验证集，测试集）
    :param path: 数据集路径类实例
    :param is_random: 是否采取打乱读取数据
    :param
    :return:
    """
    if ratios is None:
        ratios = np.array([4, 2, 4])
    else:
        ratios = np.array(ratios)
    assert ratios.sum() == 10, 'Ratios error!'

    annotation_obb_files = os.listdir(path.annotations_obb_dir)
    if is_random:
        shuffle(annotation_obb_files)

    train_datas = annotation_obb_files[:int(len(annotation_obb_files)*(ratios[0]/ratios.sum()))]
    val_datas = annotation_obb_files[len(train_datas):len(train_datas)+int(len(annotation_obb_files)*(ratios[1]/ratios.sum()))]
    test_datas = annotation_obb_files[len(train_datas)+len(val_datas):]

    split_datas(path, train_datas, mode='train')
    split_datas(path, val_datas, mode='val')
    split_datas(path, test_datas, mode='test')

def seg_labels_to_id_map(path, mode):
    path.mode = mode

    image_list = os.listdir(path.images_dir)

    if not os.path.exists(path.id_map_dir):
        os.mkdir(path.id_map_dir)

    for i, file in enumerate(image_list):
        image = imageio.imread(path.images_dir + file)
        target_segmap = np.argmax(mask_to_onehot(image, color_list).astype('int64'), axis=-1)
        imageio.imwrite(path.id_map_dir + file, target_segmap)
        print('Complete {}/{}.'.format(i+1, len(image_list)))


def split_datas(path, datas, mode):
    """
    根据文件列表划分数据
    :param path: 数据路径类实例
    :param datas: 数据文件列表
    :param mode: 模式
    :return:
    """
    if not os.path.exists(path.dataset_dir+"new"):
        os.mkdir(path.dataset_dir+"new")
    if not os.path.exists(path.dataset_dir+"new/{}".format(mode)):
        os.mkdir(path.dataset_dir+"new/{}".format(mode))
    if not os.path.exists(path.dataset_dir+"new/{}/".format(mode)+"images"):
        os.mkdir(path.dataset_dir+"new/{}/".format(mode)+"images")
    if not os.path.exists(path.dataset_dir+"new/{}/".format(mode)+"annotations_hbb"):
        os.mkdir(path.dataset_dir+"new/{}/".format(mode)+"annotations_hbb")
    if not os.path.exists(path.dataset_dir+"new/{}/".format(mode)+"annotations_obb"):
        os.mkdir(path.dataset_dir+"new/{}/".format(mode)+"annotations_obb")

    new_image_dir = path.dataset_dir+"new/{}/".format(mode)+"images"
    new_annotation_obb_dir = path.dataset_dir+"new/{}/".format(mode)+"annotations_hbb"
    new_annotation_hbb_dir = path.dataset_dir+"new/{}/".format(mode)+"annotations_obb"

    for i, data in enumerate(datas):
        # 划分图片
        copy(path.images_dir + data[:-3]+"jpg", new_image_dir)
        # 划分标签
        copy(path.annotations_obb_dir + data[:-3]+"xml", new_annotation_obb_dir)
        copy(path.annotations_hbb_dir + data[:-3]+"xml", new_annotation_hbb_dir)
        print("complete {} datas for {}/{}".format(mode, i+1, len(datas)))

if __name__ == '__main__':
    # path = SSDDPath

    path = HRSC2016DSPath

    # 验证数据集
    # invalid_datas = validate_dateset(path)
    # print(invalid_datas)

    # 划分数据集
    # split_train_val_test(path)

    seg_labels_to_id_map(path, 'train')
