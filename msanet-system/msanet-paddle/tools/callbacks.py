import os
import paddle
import numpy as np

def save_best_model(save_dir, model):
    path = '{}/best'.format(save_dir)
    print('save best checkpoint at {}'.format(os.path.abspath(path)))
    model.save(path)

class ModelEval(paddle.callbacks.Callback):

    def __init__(self, cfg, log_dir, save_dir, monitor='loss', operator="gt", mode="train"):
        super(ModelEval, self).__init__()
        # 参数保存
        self.monitor = monitor
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.mode = mode
        self.operator = operator
        self.config = cfg

        # 记录参数
        self.epoch = 0

        self.eval_results_dict = dict()

        self.best_value = 0

    # def on_eval_batch_end(self, step, logs=None):
    #     for key in list(logs.keys()):
    #         # 先判断字典有没有存在该key，若没有先添加
    #         if key not in list(self.eval_results_dict.keys()):
    #             self.eval_results_dict[key] = []
    #         if "loss" in key:
    #             self.eval_results_dict[key].append(logs[key][0])

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_eval_end(self, logs=None):
        for key in list(logs.keys()):
            # 先判断字典有没有存在该key，若没有先添加
            if key not in list(self.eval_results_dict.keys()):
                self.eval_results_dict[key] = []
            if "loss" in key:
                self.eval_results_dict[key].append(logs[key][0])
            else:
                self.eval_results_dict[key].append(logs[key])

        # 输出所有评估标准
        epoch_eval_log = dict()
        epoch_eval_log['epoch'] = self.epoch
        for eval_log in self.eval_results_dict:
            epoch_eval_log[eval_log] = self.eval_results_dict[eval_log][0]
        del epoch_eval_log['step']
        del epoch_eval_log['batch_size']

        # 输出评估结果
        if self.config.open_det_head and self.config.open_seg_head:
            epoch_eval_log['f1_ap_iou'] = 2 * epoch_eval_log['ap'] * epoch_eval_log['miou'] / (epoch_eval_log['ap'] + epoch_eval_log['miou'])
        print(str(epoch_eval_log))

        # 训练评估流程保存模型
        if self.mode == "train":
            assert self.monitor in ['loss', 'f1_ap_iou', 'ap', 'miou'], 'Monitor no accept.'
            with open(self.log_dir, 'a+') as f:
                f.write(str(epoch_eval_log) + "\n")
            # 保存最好结果
            if self.epoch == 0:
                self.best_value = epoch_eval_log[self.monitor]
            else:
                assert self.operator in ['gt', 'lt'], 'Operator no accept.'
                if self.operator == "gt":
                    if self.best_value > epoch_eval_log[self.monitor]:
                        save_best_model(self.save_dir, self.model)
                        self.best_value = epoch_eval_log[self.monitor]
                elif self.operator == "lt":
                    if self.best_value < epoch_eval_log[self.monitor]:
                        save_best_model(self.save_dir, self.model)
                        self.best_value = epoch_eval_log[self.monitor]

        # 记录参数清空
        self.eval_results_dict = dict()


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    # 针对2007年VOC，使用的11个点计算AP，现在不使用
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))  #[0.  0.0666, 0.1333, 0.4   , 0.4666,  1.]
        mpre = np.concatenate(([0.], prec, [0.])) #[0.  1.,     0.6666, 0.4285, 0.3043,  0.]

        # compute the precision envelope
        # 计算出precision的各个断点(折线点)
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])  #[1.     1.     0.6666 0.4285 0.3043 0.    ]

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]  #precision前后两个值不一样的点
        print(mrec[1:], mrec[:-1])
        print(i) #[0, 1, 3, 4, 5]

        # AP= AP1 + AP2+ AP3+ AP4
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

