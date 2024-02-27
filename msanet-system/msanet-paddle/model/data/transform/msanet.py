import cv2
import paddle
from PIL import Image
from paddle.vision import Compose, ToTensor, Normalize
from imgaug import augmenters as iaa, SegmentationMapsOnImage, ia, KeypointsOnImage, Keypoint, math, imageio
import numpy as np
from model.data.utils.color_map import select_color_list
from model.data.utils.data_aug import image_aug
from model.data.utils.mbox_utils import head_mbox2points, check_mboxes
from model.data.utils.transform import mask_to_onehot, onehot_to_mask
from model.data.utils.visualize import show_mboxes, draw_mask_in_image
from model.decoders.msanet import MSANetDecoder
from model.decoders.msanet_rbb import MSANetRBBDecoder
from model.utils.iou import compute_cross_vectors, vector2rotated
from model.utils.mbox_util import mbox_ang2vector
from model.utils.util import gaussian_radius, draw_gaussian, feat_locations, pool_nms


class MSANetTransform(object):
    def __init__(self, cfg, mode):
        self.config = cfg
        self.mode = mode

        self.decoder = MSANetRBBDecoder(self.config)

        self.data_transform = Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.resize_transform = iaa.Sequential([
            iaa.Resize(self.config.input_size)
        ])

        self.target_transform = iaa.Sequential([
            iaa.Resize([self.config.input_size[0]//self.config.down_stride, self.config.input_size[1]//self.config.down_stride])
        ])

        if self.config.image_aug:
            self.aug_transform = image_aug()

    def __call__(self, image, gt_mboxes, gt_heads, gt_seg):
        if gt_heads is None:
            assert self.config.open_box_head is False, "This dataset is not accept predict head."

        # 统一图片和标签大小
        gt_mboxes = head_mbox2points(gt_mboxes, gt_heads)
        kps = KeypointsOnImage([Keypoint(x=point[0], y=point[1]) for point in gt_mboxes.reshape([-1, 2])], shape=image.shape)
        segmap = SegmentationMapsOnImage(gt_seg, shape=image.shape)

        new_image, segmap, kps = self.resize_transform(image=image, segmentation_maps=segmap, keypoints=kps.copy())

        # 数据增强
        if self.config.image_aug and self.mode == 'train':
            new_image, segmap, kps = self.aug_transform(image=new_image, segmentation_maps=segmap, keypoints=kps.copy())
        target_segmap = np.argmax(mask_to_onehot(segmap.get_arr(), select_color_list(self.config.dataset_name)).astype('int64'), axis=-1)

        _, _, kps = self.target_transform(image=new_image, segmentation_maps=segmap, keypoints=kps.copy())
        if len(gt_mboxes) == 0:
            target_mboxes = np.array([], dtype='float32')
        else:
            target_mboxes = np.reshape(np.concatenate([item.coords for item in kps.items], axis=0), [-1, 6, 2])
        # 检验target_mboxes中有没有存在不合格的标签目标，若有则需要去除
        target_mboxes = check_mboxes(target_mboxes, np.array(self.config.input_size) // self.config.down_stride)

        # 制作目标检测标签图
        target_maps = {}
        for head in self.config.det_heads:
            map = np.zeros((self.config.input_size[0] // self.config.down_stride,
                            self.config.input_size[1] // self.config.down_stride, self.config.det_heads[head]), dtype=np.float32)
            target_maps[head] = map
        target_maps['reg_mask'] = np.zeros((self.config.input_size[0] // self.config.down_stride,
                                            self.config.input_size[1] // self.config.down_stride), dtype=np.float32)

        for i in range(len(target_mboxes)):
            # TODO: 如果出现错误可能需要在此添加判断框是否超出图像边界
            bbox = cv2.minAreaRect(target_mboxes[i][2:])  # 定向边界框4顶点表示转5参数表示
            cx, cy, w, h, theta = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1], bbox[2]
            if self.config.long_side:
                # 使用长边表示法 并将 角度控制在[-90, 0] 或 [0, 90] 范围内
                if theta > 0:
                    if w > h:
                        w, h = h, w
                        theta = -(90 - theta)
                else:
                    if w > h:
                        w, h = h, w
                        theta = (90 + theta)
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = target_mboxes[i][0]
                ct_int = ct.astype(np.int32)
                target_maps['hm'][:, :, 0] = draw_gaussian(target_maps['hm'][:, :, 0], ct_int, radius)
                target_maps['reg_mask'][ct_int[1], ct_int[0]] = 1
                target_maps['reg'][ct_int[1], ct_int[0]] = 1. * (ct[0] - ct_int[0]), 1. * (ct[1] - ct_int[1])
                target_maps = self.generate_target_map(
                    self.config.transform_name,
                    i, target_maps, target_mboxes, ct, ct_int, cx, cy, w, h, theta
                )
                if 'iou' in self.config.det_heads:
                    # 开启Cross-IoU计算平滑处理分支
                    _, vectors = vector2rotated(np.expand_dims(ct_int, axis=0),
                                                np.expand_dims([1. * w, 1. * h], axis=0),
                                                np.expand_dims(np.array([1 * theta]), axis=0))
                    tes = compute_cross_vectors(paddle.to_tensor(vectors, dtype='float32'), alpha=0.2).numpy()
                    target_maps['iou'][ct_int[1], ct_int[0]] = np.squeeze(tes, axis=0).flatten()
                if 'head' in self.config.det_heads:
                    # 开启航向点预测
                    target_maps['head'][ct_int[1], ct_int[0]] = 1. * (target_mboxes[i, 0, 0] - target_mboxes[i, 1, 0]), 1. * (target_mboxes[i, 0, 1] - target_mboxes[i, 1, 1])

        # 验证解码器
        if self.config.debug:
            pred = dict()
            for head in self.config.det_heads:
                pred[head] = np.expand_dims(np.transpose(target_maps[head], [2, 0, 1]), axis=0)

            pred['feat_location'] = feat_locations(paddle.to_tensor(pred['hm'])).numpy()
            pred['center_mask'] = pool_nms(paddle.to_tensor(pred['hm']), kernel=self.config.center_pool_nms_size).numpy()
            pred['seg'] = np.expand_dims(np.transpose(mask_to_onehot(segmap.get_arr(), select_color_list(self.config.dataset_name)).astype('int64'), [2, 0, 1]), axis=0)
            results = self.decoder(0, {"pred": pred}, False)
            if self.config.open_det_head:
                # 验证器可视化
                # cet = results['mboxes_center']
                # kp_cet = KeypointsOnImage([Keypoint(x=p[0], y=p[1]) for p in cet], shape=new_image.shape)
                # draw_image = kp_cet.draw_on_image(new_image, size=6)
                draw_image = new_image
                if 'head' in self.config.det_heads:
                    kp_hed = KeypointsOnImage([Keypoint(x=p[0], y=p[1]) for p in results['mboxes_head_points']], shape=new_image.shape)
                    draw_image = kp_hed.draw_on_image(draw_image, size=6, color=(255, 0, 0))
                draw_image = show_mboxes(draw_image, results['mboxes'], color=(0, 255, 255))
                ia.imshow(draw_image)
            if self.config.open_seg_head:
                # TODO 不做船展示可开启
                sem_map = onehot_to_mask(np.squeeze(results['mask_id_map'], axis=0), select_color_list(self.config.dataset_name))
                arr = np.zeros_like(new_image)
                arr = draw_mask_in_image(arr, sem_map)
                ia.imshow(arr)
                # im = Image.fromarray(arr)
                # im.save("demo/test.jpg")

                mask_img = draw_mask_in_image(new_image, sem_map)
                ia.imshow(mask_img)
            # if self.config.open_seg_head and self.config.open_det_head:
            #     # 验证器可视化
            #     cet = results['mboxes_center']
            #     kp_cet = KeypointsOnImage([Keypoint(x=p[0], y=p[1]) for p in cet], shape=new_image.shape)
            #     draw_image = kp_cet.draw_on_image(new_image, size=6)
            #     if 'head' in self.config.det_heads:
            #         kp_hed = KeypointsOnImage([Keypoint(x=p[0], y=p[1]) for p in results['mboxes_head_points']],
            #                                   shape=new_image.shape)
            #         draw_image = kp_hed.draw_on_image(draw_image, size=6, color=(255, 0, 0))
            #     draw_image = show_mboxes(draw_image, results['mboxes'], color=(0, 255, 255))
            #     ia.imshow(draw_image)
            #     # TODO 不做船展示可开启
            #     sem_map = onehot_to_mask(np.squeeze(results['mask_id_map'][..., :2], axis=0), select_color_list(self.config.dataset_name))
            #     # sem_map = onehot_to_mask(np.squeeze(results['mask_id_map'], axis=0), color_list)
            #     mask_img = draw_mask_in_image(draw_image, sem_map)
            #     ia.imshow(mask_img)
            #     # im = Image.fromarray(mask_img)
            #     # im.save("demo/test.jpg")

        # 统一shape防止tensor无法组装
        object_num = len(target_mboxes)
        gt_mboxes = np.zeros([self.config.max_dets, 6, 2])
        if object_num != 0:
            gt_mboxes[:object_num] = target_mboxes * self.config.down_stride
        object_num = np.array([object_num]).astype('int64')

        new_image = self.data_transform(new_image)
        input_data = {
            'data': new_image,
            'target_segmap': paddle.to_tensor(target_segmap, dtype='int64'),
            'target_mboxes': paddle.to_tensor(gt_mboxes, dtype='float32'),
            'object_num': paddle.to_tensor(object_num, dtype='float32'),
            'target_map_reg_mask': paddle.to_tensor(target_maps['reg_mask'], dtype='float32')
        }
        for head in self.config.det_heads:
            input_data['target_map_{}'.format(head)] = paddle.to_tensor(target_maps[head], dtype='float32')

        return input_data

    def generate_target_map(self, name, i, target_maps, target_mboxes, ct, ct_int, cx, cy, w, h, theta):
        if name == 'vector':
            # TODO: 单类别
            target_maps['wh'][ct_int[1], ct_int[0]] = 1. * w, 1. * h
            vector = mbox_ang2vector(cx, cy, h / 2, theta)
            target_maps['ang'][ct_int[1], ct_int[0]] = abs(1. * vector[0]), 1. * vector[1]
            if vector[0] < 0:
                s = 1
            else:
                s = 0
            target_maps['s'][ct_int[1], ct_int[0]] = 1. * s

            target_maps['head'][ct_int[1], ct_int[0]] = 1. * (ct[0] - target_mboxes[i][1][0]), 1. * (
                        ct[1] - target_mboxes[i][1][1])
        elif name == 'rbb':
            target_maps['wh'][ct_int[1], ct_int[0]] = 1. * w, 1. * h
            target_maps['ang'][ct_int[1], ct_int[0]] = 1. * theta

        return target_maps
