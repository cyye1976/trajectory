import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class NetLoss(nn.Layer):

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.loss_dict = {}
        if self.config.open_det_head:
            for key in self.config.det_heads:
                self.loss_dict[key] = None
        if self.config.open_seg_head:
            for key in self.config.seg_heads:
                self.loss_dict[key] = None

    def forward(self, pred, *target):
        target_dict = self.convert_target(target)
        loss = self.get_loss(pred, target_dict)
        return loss

    def get_loss_dict(self):
        if self.loss_dict is None:
            raise Exception(print("loss dict is none."))
        return self.loss_dict

    def get_loss(self, pred, target):
        raise NotImplementedError

    def convert_target(self, target):
        if len(target) == 1 and type(target) == tuple:
            target = target[0]
        results = {}
        for i, field in enumerate(self.config.input_fields[1:]):
             results[field] = target[i]
        return results
