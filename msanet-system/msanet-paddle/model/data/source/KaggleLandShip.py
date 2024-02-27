import math
import imageio
import numpy as np
from configs.msanet_rbb import MSANetRBBConfig
from model.data.transform.msanet import MSANetTransform
from model.interface.dataset import BaseDataset
import os
from pycocotools.coco import COCO
from model.utils.mbox_util import segmentation2mbox


class KaggleLandShip(BaseDataset):
    def __init__(self, cfg, transform, mode="train"):
        super().__init__(cfg, transform)

        # 路径定义
        self.root_path = self.dataset_dir + mode
        self.images_path = self.root_path + "/images/"
        self.seg_label_path = self.root_path + "/labels/"
        self.det_label_path = self.root_path + "/kaggle_ship_label.json"

        # 数据读入
        coco = COCO(self.det_label_path)
        images_count = len(coco.imgs.values())
        self.images_name = []
        count = 0
        for img in coco.imgs.values():
            count += 1
            try:
                image = imageio.imread(self.images_path + img['file_name'])
            except:
                continue

            gt_seg = imageio.imread(self.seg_label_path + img['file_name'][:-4] + ".png")[..., :3]
            print("Complete {}/{}".format(count, images_count))
            target_annotations_id = coco.getAnnIds(imgIds=img['id'])
            objects = coco.loadAnns(ids=target_annotations_id)

            gt_mbox = []
            gt_bbox = []
            for object in objects:
                mbox = segmentation2mbox(object['segmentation'][0])
                gt_mbox.append(mbox)
                bbox = object['bbox']  # coco格式数据集bbox定义为[x, y, w, h] x和y为左上角顶点，而w和h为左上点距水平框宽高
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # 将其转化为[xmin, ymin, xmax, ymax]
                gt_bbox.append(bbox)

            # TODO:目标对象类别(目前只做一级分类即只检测舰船目标且不进行细粒度分类)
            gt_label = np.int64([0 for i in range(len(gt_mbox))])

            # 装载数据
            self.data.append([
                image,
                gt_label,
                gt_mbox,
                gt_bbox,
                gt_seg
            ])
            self.images_name.append(img['file_name'])

            if cfg.debug:
                # 用于控制训练验证测试的样本数量（防止爆内存时可开启）
                if len(self.data) == 20:
                    break

    def _getitem(self, idx):
        image = self.data[idx][0]
        target_mboxes = self.data[idx][2]
        target_seg = self.data[idx][4]

        input_data = self.transform(image, target_mboxes, None, target_seg)

        return input_data

if __name__ == '__main__':
    config = MSANetRBBConfig()
    trn = MSANetTransform(config, 'train')
    train_dataset = KaggleLandShip(mode="train", cfg=config, transform=trn)
    for i, item in enumerate(train_dataset):
        print(i)