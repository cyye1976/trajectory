import paddle

from model.interface.decoder import Decoder
import numpy as np
import paddle.nn.functional as F
from model.utils.bbox_util import py_cpu_nms
from model.utils.mbox_util import decode_courses, decode_mbox_head_by_pts, decode_mbox_by_vector_s, getDegree


class MSANetDecoder(Decoder):
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
        hm, wh, off, head, ang, s, center_mask, feat_location, seg = inputs['pred']['hm'][batch], inputs['pred']['wh'][batch], \
                                                            inputs['pred']['reg'][batch], inputs['pred']['head'][batch],\
                                                            inputs['pred']['ang'][batch], inputs['pred']['s'][batch], inputs['pred']['center_mask'][batch], inputs['pred'][
                                                                'feat_location'], inputs['pred']['seg'][batch]

        center_nms_index = np.nonzero(center_mask.flatten())[0]

        ship_map = paddle.unsqueeze(paddle.unsqueeze(F.sigmoid(paddle.to_tensor(seg[2], dtype='float32')), axis=0), axis=0)
        seg_score = F.interpolate(
            ship_map,
            scale_factor=1. / self.config.down_stride,
            mode='bilinear'
        ).numpy()
        seg = inputs['pred']['seg']
        if len(center_nms_index) == 0:
            return results

        # 中心点热力图筛选及得分，坐标，偏移量匹配
        y_center_score = hm.flatten()[center_nms_index] * seg_score.flatten()[center_nms_index]
        # y_center_score = hm.flatten()[center_nms_index]
        y_center_xy = np.transpose(feat_location.reshape(2, -1), [1, 0])[center_nms_index, :]
        y_wh = np.transpose(wh.reshape(2, -1), [1, 0])[center_nms_index, :]
        y_center_off = np.transpose(off.reshape(2, -1), [1, 0])[center_nms_index, :]
        y_ang = np.transpose(ang.reshape(2, -1), [1, 0])[center_nms_index, :]
        y_s = np.transpose(s.reshape(1, -1), [1, 0])[center_nms_index, :]
        head_offset = np.transpose(head.reshape(2, -1), [1, 0])[center_nms_index, :]

        # 前topk中心点得分筛选
        if len(y_center_score) > self.config.nms_topk:
            center_score_nms_index = np.argsort(-y_center_score)[:self.config.nms_topk]
            y_center_score, y_center_xy, y_wh, y_center_off, y_ang, head_offset, y_s = self.y_nms_result(
                center_score_nms_index,
                y_center_score, y_center_xy, y_wh, y_center_off, y_ang, head_offset, y_s)

        # 中心点得分筛选
        center_score_nms_index = np.nonzero(y_center_score > self.config.center_score_threshold)[0]
        if len(center_score_nms_index) == 0:
            return results
        y_center_score, y_center_xy, y_wh, y_center_off, y_ang, head_offset, y_s = self.y_nms_result(
            center_score_nms_index,
            y_center_score, y_center_xy, y_wh, y_center_off, y_ang, head_offset, y_s)

        # 得到旋转框4点
        mboxes, mboxes_center = decode_mbox_by_vector_s(y_center_xy, y_wh, y_center_off, y_ang, y_s)
        mboxes_head = decode_mbox_head_by_pts(y_center_xy, mboxes, head_offset)
        mboxes_score = y_center_score

        # 根据旋转框iou进行nms
        if self.config.obb_nms:
            mboxes_obb_nms_index = py_cpu_nms(mboxes, mboxes_score, self.config.obb_nms_threshold)
            if len(mboxes_obb_nms_index) == 0:
                return results
            mboxes_score = mboxes_score[mboxes_obb_nms_index]
            mboxes = mboxes[mboxes_obb_nms_index]
            mboxes_head = mboxes_head[mboxes_obb_nms_index]

        results['mboxes'] = mboxes
        results['mboxes_center'] = mboxes_center
        results['mboxes_score'] = mboxes_score
        results["mboxes_head"] = mboxes_head

        # 具体角度和具体头部位置
        results['mboxes_courses'], results['mboxes_head_points'] = decode_courses(mboxes, mboxes_head, y_center_xy)

        # 推理语义分割图
        mask_id_map = np.argmax(np.transpose(seg, [0, 2, 3, 1]), axis=-1)
        results['mask_id_map'] = np.eye(3)[mask_id_map]

        # 提取标签target_mboxes
        if is_metric:
            num = inputs['target']['object_num'][batch]
            results['object_num'] = int(num)
            results['target_mboxes'] = inputs['target']['target_mboxes'][batch][:int(num)][:, 2:6, :]
            target_heads_points = inputs['target']['target_mboxes'][batch][:int(num)][:, 1, :]
            target_centers_points = inputs['target']['target_mboxes'][batch][:int(num)][:, 0, :]
            results['target_heads'] = getDegree(target_centers_points[..., 0], target_centers_points[..., 1], target_heads_points[..., 0], target_heads_points[..., 1])


        # 最终返回结果要封装到字典中
        return results

    def y_nms_result(self, nms_index, y_center_score, y_center_xy, y_wh, y_off, y_ang, head_offset, y_s):
        y_center_score = y_center_score[nms_index]
        y_center_xy = y_center_xy[nms_index, :]
        y_wh = y_wh[nms_index, :]
        y_off = y_off[nms_index, :]
        y_ang = y_ang[nms_index, :]
        head_offset = head_offset[nms_index, :]
        y_s = y_s[nms_index, :]

        return y_center_score, y_center_xy, y_wh, y_off, y_ang, head_offset, y_s