import math
import numpy as np
import paddle
import cv2
from paddle.fluid.reader import DataLoader
import os
from imgaug import augmenters as iaa, imageio, BoundingBoxesOnImage, BoundingBox, ia, KeypointsOnImage, Keypoint
import time

from pycocotools.coco import COCO

from model.data.utils.color_map import select_color_list
from model.data.utils.transform import onehot_to_mask
from model.data.utils.visualize import draw_mask_in_image
from model.utils.util import feat_locations, pool_nms, show_mbox, scale_xy
import paddle.nn.functional as F

class InferModule(object):
    def __init__(self,
                 cfg,
                 test_dataset,
                 model,
                 optim,
                 loss_fn,
                 metrics,
                 decoder
                 ):
        super().__init__()

        self.config = cfg

        self.test_dataset = test_dataset
        self.test_loader = DataLoader(test_dataset, shuffle=False, batch_size=self.config.infer_batch_size)

        # 模型配置
        self.model = paddle.Model(model)

        # 设置优化器
        self.optim = optim

        # 设置损失函数
        self.loss_fn = loss_fn

        # 设置评估函数
        self.metrics = metrics

        # 设置解码器
        self.decoder = decoder

    def inference(self):
        self.model.prepare(optimizer=self.optim, loss=self.loss_fn, metrics=self.metrics)
        self.model.load(self.config.infer_model_path)

        # pred: [N, batch_size, C, H, W]
        pred = self.model.predict(self.test_loader, batch_size=self.config.infer_batch_size)
        pred_dict = {}
        for i, head in enumerate(self.config.output_fields):
            pred_dict[head] = np.array(pred[:][i])
            if pred_dict[head].shape[1] == self.config.infer_batch_size:
                pred_dict[head] = pred_dict[head].squeeze(axis=1)
            elif 'feat_location' in head:  # TODO:暂时这样，后续记得改！！！！！
                pred_dict[head] = pred_dict[head][0]

        batch = list(pred_dict.values())[0].shape[0]

        # 推理语义分割图
        if self.config.open_seg_head:
            seg_results = {}
            mask_id_map = np.argmax(np.transpose(pred_dict['seg'], [0, 2, 3, 1]), axis=-1)
            seg_results['mask_id_map'] = np.eye(self.config.seg_class_num)[mask_id_map]

        # 可视化
        images_name = self.test_dataset.images_name
        # 输入解码器换算出mboxes
        for i in range(batch):
            infer_image = imageio.imread(self.test_dataset.images_path + images_name[i])
            infer_image = ia.imresize_single_image(infer_image, self.config.input_size)
            results = self.decoder(i, {"pred": pred_dict}, False)
            if self.config.open_det_head:
                if 'mboxes_center' in results or 'mboxes' in results:
                    #  TODO: 暂时先可视化中心点
                    if self.config.open_box_head:
                        kps = []
                        h_kps = []
                        center_points = results['mboxes_center']
                        header_points = results['mboxes_head_points']
                        for center_point, header_point in zip(center_points, header_points):
                            kps.append(Keypoint(x=center_point[0], y=center_point[1]))
                            h_kps.append(Keypoint(x=header_point[0], y=header_point[1]))
                        kps = KeypointsOnImage(kps, shape=infer_image.shape)
                        h_kps = KeypointsOnImage(h_kps, shape=infer_image.shape)
                        infer_image = kps.draw_on_image(infer_image, size=6)  # 可视化中心点
                        infer_image = h_kps.draw_on_image(infer_image, size=6, color=(255, 0, 0))  # 可视化头部点
                    else:
                        kps = []
                        center_points = results['mboxes_center']
                        for center_point in center_points:
                            kps.append(Keypoint(x=center_point[0], y=center_point[1]))
                        kps = KeypointsOnImage(kps, shape=infer_image.shape)
                        infer_image = kps.draw_on_image(infer_image, size=6)  # 可视化中心点

                    # 可视化旋转框
                    for mbox in results['mboxes']:
                        # 修正平行四边形情况
                        boxPoints = mbox
                        try:
                            infer_image = show_mbox(infer_image, boxPoints)
                        except:
                            continue

                infer_save_path = 'inference/{}/det/'.format(self.config.model_name)
                if not os.path.exists(infer_save_path):
                    os.makedirs(infer_save_path)
                imageio.imwrite(infer_save_path + images_name[i][:-4]+".jpg", infer_image)
                print('save {}'.format(infer_save_path) + images_name[i] + ' det is complete.')

            if self.config.open_seg_head:
                # 可视化语义分割推理结果
                seg_image = np.zeros_like(infer_image)
                if 'mask_id_map' in seg_results:
                    mask_prediction = onehot_to_mask(seg_results['mask_id_map'][i], select_color_list(self.config.dataset_name))
                    seg_image = mask_prediction
                infer_save_path = 'inference/{}/seg/'.format(self.config.model_name)
                if not os.path.exists(infer_save_path):
                    os.makedirs(infer_save_path)
                imageio.imwrite(infer_save_path + images_name[i][:-4] + ".jpg", seg_image)
                print('save {}'.format(infer_save_path) + images_name[i] + ' seg is complete.')

            if self.config.open_det_head and self.config.open_seg_head:
                # 可视化原图和语义分割推理结果透明融合
                seg_image = np.zeros_like(infer_image)
                if 'mask_id_map' in seg_results:
                    mask_prediction = onehot_to_mask(seg_results['mask_id_map'][..., :2][i], select_color_list(self.config.dataset_name))
                    seg_image = draw_mask_in_image(infer_image, mask_prediction)
                infer_save_path = 'inference/{}/source_seg/'.format(self.config.model_name)
                if not os.path.exists(infer_save_path):
                    os.makedirs(infer_save_path)
                imageio.imwrite(infer_save_path + images_name[i][:-4] + ".jpg", seg_image)
                print('save {}'.format(infer_save_path) + images_name[i] + ' source_seg is complete.')

    def inference_source(self):
        self.model.prepare(optimizer=self.optim, loss=self.loss_fn, metrics=self.metrics)
        self.model.load(self.config.infer_model_path)

        pred = self.model.predict(self.test_loader, batch_size=self.config.infer_batch_size)
        pred_dict = {}
        for i, head in enumerate(self.config.output_fields):
            pred_dict[head] = np.array(pred[:][i])
            if pred_dict[head].shape[1] == self.config.infer_batch_size:
                pred_dict[head] = pred_dict[head].squeeze(axis=1)
            elif 'feat_location' in head:  # TODO:暂时这样，后续记得改！！！！！
                pred_dict[head] = pred_dict[head][0]

        batch = list(pred_dict.values())[0].shape[0]

        # 可视化
        images_name = os.listdir(self.test_dataset.images_path)
        # 输入解码器换算出mboxes
        for i in range(batch):
            source_image = imageio.imread(self.test_dataset.images_path + images_name[i])
            infer_image = source_image.copy()
            results = self.decoder(i, {"pred": pred_dict}, False)
            if self.config.open_det_head:
                if 'mboxes_center' in results or 'mboxes' in results:
                    #  TODO: 暂时先可视化中心点
                    if self.config.open_box_head:
                        kps = []
                        h_kps = []
                        center_points = results['mboxes_center']
                        header_points = results['mboxes_head_points']
                        for center_point, header_point in zip(center_points, header_points):
                            kps.append(Keypoint(x=center_point[0], y=center_point[1]))
                            h_kps.append(Keypoint(x=header_point[0], y=header_point[1]))
                        kps = KeypointsOnImage(kps, shape=infer_image.shape)
                        h_kps = KeypointsOnImage(h_kps, shape=infer_image.shape)
                        infer_image = kps.draw_on_image(infer_image, size=6)  # 可视化中心点
                        infer_image = h_kps.draw_on_image(infer_image, size=6, color=(255, 0, 0))  # 可视化头部点
                    else:
                        kps = []
                        center_points = scale_xy(results['mboxes_center'], self.config.input_size, (source_image.shape[1], source_image.shape[0]))
                        for center_point in center_points:
                            kps.append(Keypoint(x=center_point[0], y=center_point[1]))
                        kps = KeypointsOnImage(kps, shape=infer_image.shape)
                        infer_image = kps.draw_on_image(infer_image, size=6)  # 可视化中心点

                    # 可视化旋转框
                    for mbox in results['mboxes']:
                        # 修正平行四边形情况
                        boxPoints = scale_xy(mbox, self.config.input_size, (source_image.shape[1], source_image.shape[0]))
                        try:
                            infer_image = show_mbox(infer_image, boxPoints)
                        except:
                            continue

                infer_save_path = 'inference/{}/det/'.format(self.config.model_name)
                if not os.path.exists(infer_save_path):
                    os.makedirs(infer_save_path)
                imageio.imwrite(infer_save_path + images_name[i][:-4]+".jpg", infer_image)
                print('save {}'.format(infer_save_path) + images_name[i] + ' det is complete.')

            if self.config.open_seg_head:
                # 可视化语义分割推理结果
                seg_image = np.zeros_like(source_image)
                if 'mask_id_map' in results:
                    mask_prediction = ia.imresize_single_image(onehot_to_mask(results['mask_id_map'][i], select_color_list(self.config.dataset_name)), source_image.shape[:2])
                    seg_image = cv2.medianBlur(mask_prediction, 9)
                infer_save_path = 'inference/{}/seg/'.format(self.config.model_name)
                if not os.path.exists(infer_save_path):
                    os.makedirs(infer_save_path)
                imageio.imwrite(infer_save_path + images_name[i][:-4]+".jpg", seg_image)
                print('save {}'.format(infer_save_path) + images_name[i] + ' seg is complete.')

            if self.config.open_det_head and self.config.open_seg_head:
                # 可视化目标检测和语义分割融合结果
                seg_image = np.zeros_like(source_image)
                if 'mask_id_map' in results:
                    mask_prediction = ia.imresize_single_image(onehot_to_mask(results['mask_id_map'][i][..., :2], select_color_list(self.config.dataset_name)), source_image.shape[:2])
                    seg_image = draw_mask_in_image(infer_image, mask_prediction)
                infer_save_path = 'inference/{}/det_seg/'.format(self.config.model_name)
                if not os.path.exists(infer_save_path):
                    os.makedirs(infer_save_path)
                imageio.imwrite(infer_save_path + images_name[i][:-4]+".jpg", seg_image)
                print('save {}'.format(infer_save_path) + images_name[i] + ' det_seg is complete.')

    def get_fps(self):
        self.model.prepare(optimizer=self.optim, loss=self.loss_fn, metrics=self.metrics)
        self.model.load(self.config.infer_model_path)

        # pred: [N, batch_size, C, H, W]
        start = time.clock()  # 推理计时开始
        pred = self.model.predict(self.test_loader, batch_size=self.config.infer_batch_size)
        pred_dict = {}
        for i, head in enumerate(self.config.output_fields):
            pred_dict[head] = np.array(pred[:][i])
            if pred_dict[head].shape[1] == self.config.infer_batch_size:
                pred_dict[head] = pred_dict[head].squeeze(axis=1)
            elif 'feat_location' in head:  # TODO:暂时这样，后续记得改！！！！！
                pred_dict[head] = pred_dict[head][0]

        batch = list(pred_dict.values())[0].shape[0]

        # 输入解码器换算出mboxes
        for i in range(batch):
            results = self.decoder(i, {"pred": pred_dict}, False)

        # 推理语义分割图
        if self.config.open_seg_head:
            seg_results = {}
            mask_id_map = np.argmax(np.transpose(pred_dict['seg'], [0, 2, 3, 1]), axis=-1)
            seg_results['mask_id_map'] = np.eye(self.config.seg_class_num)[mask_id_map]
        end = time.clock()  # 计时结束
        print('infer_time:', end - start)