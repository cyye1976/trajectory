import paddle

from model.interface.decoder import Decoder
import numpy as np

from model.utils.bbox_util import py_cpu_nms
from model.utils.mbox_util import decode_courses, decode_mbox_head_by_pts, decode_mbox_by_vector_s, getDegree, \
    decode_mbox_by_ang, correct_box_head_by_crf
import paddle.nn.functional as F


class MSANetRBBDecoder(Decoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, batch, inputs, is_metric=True):
        """
        解码批量输入数据
        :param batch: 遍历到了第几批次
        :param inputs: 批量的输入数据 {"pred":pred, "target": target} or {"pred":pred}
        :param is_metric: 是否处于评估状态（True or False） 评估状态需要输出target_mboxes
        :return:
        """
        results = dict()

        # 解码
        data = self.trans_inputs(batch, inputs)

        # 中心点热力图筛选及得分，坐标，偏移量匹配
        if self.config.open_det_head:
            center_nms_index = np.nonzero(data['center_mask'].flatten())[0]
            if len(center_nms_index) == 0:
                return results

            det_result = {}
            det_result['score'] = data['hm'].flatten()[center_nms_index]
            if self.config.open_seg_head and self.config.confidence_fusion:
                # 船舶检测和海陆分割置信度融合
                ship_map = F.sigmoid(
                    paddle.unsqueeze(paddle.unsqueeze(paddle.to_tensor(data['seg'][batch][2], dtype='float32'), axis=0),
                                     axis=0))
                seg_score = F.interpolate(
                    ship_map,
                    scale_factor=1. / self.config.down_stride,
                    mode='bilinear'
                ).numpy()
                det_result['score'] *= seg_score.flatten()[center_nms_index]

            cen_xy = np.transpose(data['feat_location'].reshape(2, -1), [1, 0])[center_nms_index, :]
            cen_off = np.transpose(data['reg'].reshape(2, -1), [1, 0])[center_nms_index, :]
            det_result['center_xy'] = cen_xy + cen_off * self.config.down_stride
            det_result['box_wh'] = np.transpose(data['wh'].reshape(2, -1), [1, 0])[center_nms_index, :]
            det_result['box_ang'] = np.expand_dims(data['ang'].flatten()[center_nms_index], axis=-1)
            if 'head' in self.config.det_heads:
                head_offset = np.transpose(data['head'].reshape(2, -1), [1, 0])[center_nms_index, :]
                det_result['box_head'] = det_result['center_xy'] - head_offset * self.config.down_stride

            # 前topk中心点得分筛选
            if len(det_result['score']) > self.config.nms_topk:
                center_score_nms_index = np.argsort(-det_result['score'])[:self.config.nms_topk]
                det_result = self.nms_result(center_score_nms_index, det_result)

            # 中心点得分筛选
            center_score_nms_index = np.nonzero(det_result['score'] > self.config.center_score_threshold)[0]
            if len(center_score_nms_index) == 0:
                return results
            det_result = self.nms_result(center_score_nms_index, det_result)

            # 得到旋转框4点
            mboxes, mboxes_center = decode_mbox_by_ang(det_result['center_xy'],
                                                       det_result['box_wh'],
                                                       det_result['box_ang'],
                                                       self.config.down_stride)

            mboxes_score = det_result['score']
            if self.config.open_box_head:
                mboxes_head = det_result['box_head']
            # 根据旋转框iou进行nms
            if self.config.obb_nms:
                mboxes_obb_nms_index = py_cpu_nms(mboxes, det_result['score'], self.config.obb_nms_threshold)
                if len(mboxes_obb_nms_index) == 0:
                    return results
                mboxes_score = mboxes_score[mboxes_obb_nms_index]
                mboxes_center = mboxes_center[mboxes_obb_nms_index]
                mboxes = mboxes[mboxes_obb_nms_index]
                if self.config.open_box_head:
                    mboxes_head = mboxes_head[mboxes_obb_nms_index]

            results['mboxes'] = mboxes
            results['mboxes_center'] = mboxes_center
            results['mboxes_score'] = mboxes_score

            # 具体航向角和具体头部位置
            if self.config.open_box_head:
                results['mboxes_head_points'] = mboxes_head
                if self.config.open_box_head_crf:
                    # 使用基于外接矩形框的航向点修正方法
                    results['mboxes_courses'], results['mboxes_head_points'] = correct_box_head_by_crf(mboxes_center,
                                                                                                        mboxes,
                                                                                                        mboxes_head,
                                                                                                        self.config.down_stride)
                else:
                    # 直接回归航向点
                    results['mboxes_courses'] = getDegree(mboxes_center[..., 0],
                                                          mboxes_center[..., 1],
                                                          mboxes_head[..., 0],
                                                          mboxes_head[..., 1])

        # 推理语义分割图
        if self.config.open_seg_head and self.config.debug:
            mask_id_map = np.argmax(np.transpose(data['seg'], [0, 2, 3, 1]), axis=-1)
            results['mask_id_map'] = np.eye(3)[mask_id_map]

        # 提取标签target_mboxes
        if is_metric and self.config.open_det_head:
            num = inputs['target']['object_num'][batch]
            results['object_num'] = int(num)
            results['target_mboxes'] = inputs['target']['target_mboxes'][batch][:int(num)][:, 2:6, :]
            if self.config.open_box_head:
                target_heads_points = inputs['target']['target_mboxes'][batch][:int(num)][:, 1, :]
                target_centers_points = inputs['target']['target_mboxes'][batch][:int(num)][:, 0, :]
                results['target_heads'] = getDegree(target_centers_points[..., 0], target_centers_points[..., 1],
                                                    target_heads_points[..., 0], target_heads_points[..., 1])

        # 最终返回结果要封装到字典中
        return results
