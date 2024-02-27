import json
import os
import uuid
import random

import cv2
import numpy as np
from paddle.fluid.reader import DataLoader
from imgaug import imageio, ia, Keypoint, KeypointsOnImage

from api.loader.builder import ModelBuilder
from api.loader.config import LoaderConfig
from api.loader.dataloader import ApiDataset
from api.utils.color import RGB_to_Hex
from model.data.utils.color_map import select_color_list
from model.data.utils.transform import onehot_to_mask
from model.data.utils.visualize import draw_mask_in_image
from model.utils.util import show_mbox


class ApiModule(object):
    def __init__(self,
                 model_filename,
                 group_id,
                 model_name='MSANetRBB'):
        super().__init__()

        self.config = LoaderConfig()
        self.model_filename = model_filename

        # 模型配置
        self.builder = ModelBuilder(model_name).get_model()

        # 设置数据
        self.test_dataset = ApiDataset(dataset_dir=self.config.dataset_dir, group_id=group_id)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=1)

        # 解译结果
        self.results = []

    def inference(self):
        json_paths = []  # 存放解译结果路径

        # 装载模型
        self.builder['model'].prepare(optimizer=self.builder['optim'], loss=self.builder['loss_fn'])
        self.builder['model'].load(self.config.model_dir + self.model_filename)


        # pred: [N, batch_size, C, H, W]
        pred = self.builder['model'].predict(self.test_loader, batch_size=1)
        pred_dict = {}
        for i, head in enumerate(self.builder['config'].output_fields):
            pred_dict[head] = np.array(pred[:][i])
            if pred_dict[head].shape[1] == 1:
                pred_dict[head] = pred_dict[head].squeeze(axis=1)
            elif 'feat_location' in head:  # TODO:暂时这样，后续记得改！！！！！
                pred_dict[head] = pred_dict[head][0]

        batch = list(pred_dict.values())[0].shape[0]

        # 推理语义分割图
        if self.builder['config'].open_seg_head:
            mask_id_map = np.argmax(np.transpose(pred_dict['seg'], [0, 2, 3, 1]), axis=-1)
            seg_results = np.eye(self.builder['config'].seg_class_num)[mask_id_map]

        # 输入解码器换算出mboxes
        images_name = self.test_dataset.images_name
        for i in range(batch):
            det_results = self.builder['decoder'](i, {"pred": pred_dict}, False)
            result = {
                'image_name': images_name[i],
                'input_size': self.builder['config'].input_size,
                'det_results': det_results,
                'seg_results': seg_results[i]
            }
            self.results.append(result)
            for key in result['det_results']:
                result['det_results'][key] = result['det_results'][key].tolist()
            result['seg_results'] = result['seg_results'].tolist()

            uid = ''.join(str(uuid.uuid4()).split('-'))
            result_save_path = self.config.json_dir + uid + ".json"
            with open(result_save_path, 'w+') as f:
                json.dump(result, f)
            json_paths.append(result_save_path)
        return json_paths


    def visualize(self, options=[0,1,2,3]):
        # 0-原图，1-船舶检测，2-海陆分割，3-航向预测
        # 可视化
        save_paths = []
        images_name = self.test_dataset.images_name
        infer_image = np.zeros((512, 512, 3))
        for i in range(len(images_name)):
            box_color = []
            if 0 in options:
                infer_image = imageio.imread(self.test_dataset.images_path + "/" + images_name[i])
                infer_image = ia.imresize_single_image(infer_image, self.builder['config'].input_size)

            if 1 in options:
                kps = []
                h_kps = []
                center_points = self.results[i]['det_results']['mboxes_center']
                header_points = self.results[i]['det_results']['mboxes_head_points']
                for center_point, header_point in zip(center_points, header_points):
                    kps.append(Keypoint(x=center_point[0], y=center_point[1]))
                    h_kps.append(Keypoint(x=header_point[0], y=header_point[1]))
                kps = KeypointsOnImage(kps, shape=infer_image.shape)
                h_kps = KeypointsOnImage(h_kps, shape=infer_image.shape)
                infer_image = kps.draw_on_image(infer_image, size=6)  # 可视化中心点
                if 3 in options:
                    infer_image = h_kps.draw_on_image(infer_image, size=6, color=(255, 0, 0))  # 可视化头部点
                # 可视化旋转框
                for mbox in self.results[i]['det_results']['mboxes']:
                    # 修正平行四边形情况
                    boxPoints = mbox
                    random_color = np.random.randint(0, 255, 3).tolist()
                    box_color.append(RGB_to_Hex(random_color))
                    infer_image = show_mbox(infer_image, boxPoints, color=random_color)
            # ia.imshow(infer_image)
            if 2 in options:
                # 可视化原图和语义分割推理结果透明融合
                mask_prediction = onehot_to_mask(np.array(self.results[i]['seg_results'])[..., :2], select_color_list(self.builder['config'].dataset_name))
                infer_image = draw_mask_in_image(infer_image, mask_prediction)

            # 将可视化结果图保存并返回
            save_path = self.config.cache_dir + images_name[i][:-4] + ".jpg"
            imageio.imwrite(save_path, infer_image)
            print('save {}'.format(save_path + ' is complete.'))
            result = {
                'save_path': save_path,
                'box_color': box_color
            }

            save_paths.append(result)

        return save_paths

    def load_result(self):
        images_name = self.test_dataset.images_name
        json_results = []
        for i in range(len(images_name)):
            result_dicts = []
            results = self.results[i]['det_results']
            for j, rbox in enumerate(results['mboxes']):
                rbox = np.array(rbox, dtype='float32')
                rbox = cv2.minAreaRect(rbox)

                result_dict = {
                    'centerXY': str([round(value, 2) for value in self.results[i]['det_results']['mboxes_center'][j]]),
                    'WH': str([round(rbox[1][0], 2), round(rbox[1][1], 2)]),
                    'course': str(round(self.results[i]['det_results']['mboxes_courses'][j], 2))
                }
                result_dicts.append(result_dict)
            json_results.append(result_dicts)
        return json_results