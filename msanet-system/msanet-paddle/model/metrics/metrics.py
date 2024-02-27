import itertools
import sys

import numpy as np
import paddle
import paddle.nn as nn
from paddleseg.utils.metrics import calculate_area, mean_iou, accuracy, kappa

from model.interface.metric import DetMetric
from model.utils.bbox_util import mboxes_obb_iou
from model.utils.rbox_iou import rbox_poly_overlaps
from model.utils.util import feat_locations, pool_nms
from ppdet.data import get_categories
from ppdet.metrics import VOCMetric
from ppdet.metrics.map_utils import prune_zero_padding, DetectionMAP, logger, draw_pr_curve
from tools.callbacks import voc_ap


"""
旋转框检测PR, MAP计算
"""


class PascalVOCMetric(VOCMetric):
    def __init__(self, cfg, decoder, head=False):
        super().__init__(cfg.dataset_dir + 'label_list', class_num=cfg.det_class_num)
        self.config = cfg
        self.decoder = decoder
        self.head = head

        self.detection_map = RoDetectionMAP(
            class_num=cfg.det_class_num,
            catid2name=self.catid2name,
            overlap_thresh=cfg.obb_nms_threshold,
            head=head)

        self.reset()

    def compute(self, pred, *target):
        # 删除不需要字段
        for key in self.config.del_output_fields:
            if key in list(pred.keys()):
                del pred[key]

        pred_len = paddle.to_tensor(len(pred))
        target_len = paddle.to_tensor(len(target))
        result = [pred_len, target_len]
        for value in list(pred.values()):
            result.append(value)
        for value in list(target):
            result.append(value)

        return result

    def convert_input_data(self, pred_len, target_len, inputs):
        pred = inputs[:int(pred_len)]
        target = inputs[int(pred_len):int(pred_len) + int(target_len)]

        convert_pred = {}
        convert_target = {}

        for i, field in enumerate(self.config.output_fields):
            convert_pred[field] = pred[i]
        for i, field in enumerate(self.config.input_fields[1:]):
            convert_target[field] = target[i]
        batch = pred[0].shape[0]
        return batch, convert_target, convert_pred

    def update(self, pred_len, target_len, *inputs):
        batch, target, pred = self.convert_input_data(pred_len, target_len, inputs)

        for batch_id in range(batch):
            outputs = {}
            inputs = {}

            results = self.decoder(batch_id, {"pred": pred, "target": target}, True)
            if 'mboxes' in results:
                outputs['bboxes'] = results['mboxes']
                outputs['scores'] = results['mboxes_score']
                outputs['labels'] = np.array([0 for _ in range(len(outputs['bboxes']))])  # TODO:目前为单类
            else:
                return

            if results['object_num'] != 0:
                inputs['gt_bbox'] = results['target_mboxes']
                inputs['gt_class'] = np.array([0 for _ in range(len(results['target_mboxes']))])  # TODO:目前为单类
            else:
                inputs['gt_bbox'] = []
                inputs['gt_class'] = []

            bboxes = outputs['bboxes']
            scores = outputs['scores']
            labels = outputs['labels']

            gt_boxes = inputs['gt_bbox']
            gt_labels = inputs['gt_class']
            difficult = None

            # 每张样本遍历
            gt_box = gt_boxes
            gt_label = gt_labels
            bbox = bboxes
            score = scores
            label = labels

            gt_head = None
            bbox_head = None
            if self.head:
                gt_head = results['target_heads']
                # bbox_head = results['mboxes_head']
                bbox_head = results['mboxes_courses']

            self.detection_map.update(bbox, score, label, gt_box, gt_label, bbox_head=bbox_head, gt_head=gt_head,
                                      difficult=difficult)  # 输入一张样本的数据

    def get_results(self):
        if self.head:
            return {'bbox': self.detection_map.get_map()}
        else:
            return {'bbox': [self.detection_map.get_map()]}

    def accumulate(self):
        self.detection_map.accumulate()
        ap = self.get_results()
        if self.head:
            return ap['bbox'][0], ap['bbox'][1]
        else:
            return ap['bbox'][0]

    def name(self):
        if self.head:
            return 'ap', 'head_acc'
        else:
            return 'ap'


class RoDetectionMAP(DetectionMAP):
    def __init__(self, class_num, catid2name, overlap_thresh=0.5, head=False):
        super().__init__(class_num=class_num, catid2name=catid2name, overlap_thresh=overlap_thresh)
        self.head = head
        self.head_tp = 0
        self.mhead_acc = None

    def update(self, bbox, score, label, gt_box, gt_label, bbox_head=None, gt_head=None, difficult=None):
        """
        Update metric statics from given prediction and ground
        truth infomations.
        """
        if difficult is None:
            difficult = np.zeros_like(gt_label)
        if self.head:
            assert bbox_head is not None and gt_head is not None, 'bbox_head or gt_head is None.'

        # record class gt count
        for gtl, diff in zip(gt_label, difficult):
            if self.evaluate_difficult or int(diff) == 0:
                self.class_gt_counts[int(np.array(gtl))] += 1

        # record class score positive
        visited = [False] * len(gt_label)
        index = 0
        for b, s, l in zip(bbox, score, label):
            pred = b
            max_idx = -1
            max_overlap = -1.0
            for i, gl in enumerate(gt_label):
                if int(gl) == int(l):
                    overlap = rbox_poly_overlaps(pred.reshape(-1, 8), gt_box[i].reshape(-1, 8))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        max_idx = i

            if max_overlap > self.overlap_thresh:
                if self.evaluate_difficult or \
                        int(np.array(difficult[max_idx])) == 0:
                    if not visited[max_idx]:
                        self.class_score_poss[int(l)].append([s, 1.0])
                        visited[max_idx] = True
                        if self.head:
                            if abs(bbox_head[index] - gt_head[max_idx]) < 10:
                                self.head_tp += 1
                    else:
                        self.class_score_poss[int(l)].append([s, 0.0])
            else:
                self.class_score_poss[int(l)].append([s, 0.0])
            index += 1

    def accumulate(self):
        """
        Accumulate metric results and calculate mAP
        """
        mAP = 0.
        valid_cnt = 0
        accum_tp = 0
        eval_results = []
        for score_pos, count in zip(self.class_score_poss,
                                    self.class_gt_counts):
            if count == 0: continue
            if len(score_pos) == 0:
                valid_cnt += 1
                continue

            accum_tp_list, accum_fp_list, accum_tp, accum_fp = \
                self._get_tp_fp_accum(score_pos)
            precision = []
            recall = []
            for ac_tp, ac_fp in zip(accum_tp_list, accum_fp_list):
                precision.append(float(ac_tp) / (ac_tp + ac_fp))
                recall.append(float(ac_tp) / count)

            one_class_ap = 0.0
            if self.map_type == '11point':
                max_precisions = [0.] * 11
                start_idx = len(precision) - 1
                for j in range(10, -1, -1):
                    for i in range(start_idx, -1, -1):
                        if recall[i] < float(j) / 10.:
                            start_idx = i
                            if j > 0:
                                max_precisions[j - 1] = max_precisions[j]
                                break
                        else:
                            if max_precisions[j] < precision[i]:
                                max_precisions[j] = precision[i]
                one_class_ap = sum(max_precisions) / 11.
                mAP += one_class_ap
                valid_cnt += 1
            elif self.map_type == 'integral':
                import math
                prev_recall = 0.
                for i in range(len(precision)):
                    recall_gap = math.fabs(recall[i] - prev_recall)
                    if recall_gap > 1e-6:
                        one_class_ap += precision[i] * recall_gap
                        prev_recall = recall[i]
                mAP += one_class_ap
                valid_cnt += 1
            else:
                logger.error("Unspported mAP type {}".format(self.map_type))
                sys.exit(1)
            eval_results.append({
                'class': self.classes[valid_cnt - 1],
                'ap': one_class_ap,
                'precision': precision,
                'recall': recall
            })
        self.eval_results = eval_results
        self.mAP = mAP / float(valid_cnt) if valid_cnt > 0 else mAP
        if self.head:
            self.mhead_acc = self.head_tp / accum_tp if accum_tp != 0 else 0.

    def reset(self):
        """
        Reset metric statics
        """
        self.class_score_poss = [[] for _ in range(self.class_num)]
        self.class_gt_counts = [0] * self.class_num
        self.mAP = None
        self.head_tp = 0
        self.mhead_acc = None

    def _get_tp_fp_accum(self, score_pos_list):
        """
        Calculate accumulating true/false positive results from
        [score, pos] records
        """
        sorted_list = sorted(score_pos_list, key=lambda s: s[0], reverse=True)
        accum_tp = 0
        accum_fp = 0
        accum_tp_list = []
        accum_fp_list = []
        for (score, pos) in sorted_list:
            accum_tp += int(pos)
            accum_tp_list.append(accum_tp)
            accum_fp += 1 - int(pos)
            accum_fp_list.append(accum_fp)
        return accum_tp_list, accum_fp_list, accum_tp, accum_fp

    def get_map(self):
        """
        Get mAP result
        """
        if self.mAP is None:
            logger.error("mAP is not calculated.")
        if self.mhead_acc is None and self.head:
            logger.error("mhead_acc is not calculated.")
        if self.classwise:
            # Compute per-category AP and PR curve
            try:
                from terminaltables import AsciiTable
            except Exception as e:
                logger.error(
                    'terminaltables not found, plaese install terminaltables. '
                    'for example: `pip install terminaltables`.')
                raise e
            results_per_category = []
            for eval_result in self.eval_results:
                results_per_category.append(
                    (str(eval_result['class']),
                     '{:0.3f}'.format(float(eval_result['ap']))))
                draw_pr_curve(
                    eval_result['precision'],
                    eval_result['recall'],
                    out_dir='voc_pr_curve',
                    file_name='{}_precision_recall_curve.jpg'.format(
                        eval_result['class']))

            num_columns = min(6, len(results_per_category) * 2)
            results_flatten = list(itertools.chain(*results_per_category))
            headers = ['category', 'AP'] * (num_columns // 2)
            results_2d = itertools.zip_longest(
                *[results_flatten[i::num_columns] for i in range(num_columns)])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            logger.info('Per-category of VOC AP: \n{}'.format(table.table))
            logger.info(
                "per-category PR curve has output to voc_pr_curve folder.")
        if self.head:
            return self.mAP, self.mhead_acc
        return self.mAP


"""
语义分割评估
"""
class SegMetric(paddle.metric.Metric):

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg

        self.class_iou = []
        self.miou = []
        self.class_acc = []
        self.macc = []
        self.kappa = []

    def reset(self):
        self.class_iou = []
        self.miou = []
        self.class_acc = []
        self.macc = []
        self.kappa = []

    def compute(self, pred, *args):
        target = {}
        for i, field in enumerate(self.config.input_fields[1:]):
            target[field] = args[i]

        pred_seg = paddle.argmax(paddle.transpose(pred['seg'], [0, 2, 3, 1]), axis=-1)
        target_seg = target['target_segmap']

        intersect_area, pred_area, label_area = calculate_area(pred_seg, target_seg, self.config.seg_class_num)

        class_iou, miou = mean_iou(intersect_area, pred_area, label_area)
        class_acc, macc = accuracy(intersect_area, pred_area)
        kap = kappa(intersect_area, pred_area, label_area)

        self.class_iou.append(class_iou)
        self.miou.append(miou)
        self.class_acc.append(class_acc)
        self.macc.append(macc)
        self.kappa.append(kap)

        return paddle.to_tensor([1.])

    def update(self, tag):
        pass

    def accumulate(self):
        class_iou = np.array(self.class_iou)
        class_acc = np.array(self.class_acc)
        class_iou = [class_iou[..., i].mean() for i in range(self.config.seg_class_num)]
        class_acc = [class_acc[..., i].mean() for i in range(self.config.seg_class_num)]

        miou = np.array(self.miou).mean()
        macc = np.array(self.macc).mean()
        kap = np.array(self.kappa).mean()

        return class_iou, miou, class_acc, macc, kap

    def name(self):
        return 'class_iou', 'miou', 'class_acc', 'macc', 'kappa'


"""
LOSS METRICS
"""

class LossMetric(paddle.metric.Metric):
    def __init__(self, cfg, criterion):
        super().__init__()
        self.config = cfg
        self.criterion = criterion
        self.loss_dict = self.criterion.get_loss_dict()

    def compute(self, pred, *target):
        self.loss_dict = self.criterion.get_loss_dict()
        return list(self.loss_dict.values())

    def update(self, *args):
        for i, loss_name in enumerate(self.loss_dict):
            self.__setattr__(loss_name, args[i])

    def reset(self):
        pass

    def accumulate(self):
        results = []
        for i, loss_name in enumerate(self.loss_dict):
            results.append(self.__getattribute__(loss_name).tolist())
        return results

    def name(self):
        results = list(self.loss_dict.keys())
        return results

