import math
import os

import imageio
import xml.etree.ElementTree as ET

from configs.msanet import MSANetConfig
from configs.msanet_rbb import MSANetRBBConfig
from model.data.transform.msanet import MSANetTransform
from model.interface.dataset import BaseDataset

class HRSC2016DS(BaseDataset):
    def __init__(self, cfg, transform, mode='train'):
        super().__init__(cfg, transform)

        # 路径定义
        self.root_path = self.dataset_dir + mode
        self.images_path = self.root_path + "/images/"
        self.det_label_path = self.root_path + "/det_labels/"
        self.seg_label_path = self.root_path + "/seg_labels/"

        # 数据读入
        images_name = os.listdir(self.images_path)
        self.images_name = images_name
        self.data = []
        count = 0
        for image_name in images_name:
            count += 1
            # 加载图片
            image = imageio.imread(self.images_path + image_name)

            # 加载语义分割标签
            gt_seg = imageio.imread(self.seg_label_path + image_name[:-4] + ".png")

            # 加载目标检测标签
            tree = ET.parse(self.det_label_path + image_name[:-4] + ".xml")
            root = tree.getroot()
            hrsc_objects = root.find("HRSC_Objects")
            hrsc_object = hrsc_objects.findall("HRSC_Object")
            gt_mbox = []
            gt_head = []
            for object in hrsc_object:
                mbox = (
                    (float(object.find("mbox_cx").text),
                     float(object.find("mbox_cy").text)),
                    (float(object.find("mbox_w").text),
                     float(object.find("mbox_h").text)),
                    math.degrees(float(object.find("mbox_ang").text))
                )
                head = [float(object.find("header_x").text),
                        float(object.find("header_y").text)]
                gt_mbox.append(mbox)
                gt_head.append(head)

            # 装载数据
            self.data.append([
                image,
                gt_mbox,
                gt_head,
                gt_seg
            ])
            print("Complete {}/{}.".format(count, len(self.images_name)))
            if cfg.debug:
                # 用于控制训练验证测试的样本数量（防止爆内存时可开启）
                if len(self.data) == 20:
                    break

    def _getitem(self, idx):
        image = self.data[idx][0]
        target_mboxes = self.data[idx][1]
        target_heads = self.data[idx][2]
        target_seg = self.data[idx][3]

        input_data = self.transform(image, target_mboxes, target_heads, target_seg)

        return input_data


if __name__ == '__main__':
    config = MSANetRBBConfig()
    trn = MSANetTransform(config, 'train')
    train_dataset = HRSC2016DS(mode="train", cfg=config, transform=trn)
    for image, gt_label, gt_bbox in train_dataset:
        print(image.shape, gt_label.shape, gt_bbox.shape)
