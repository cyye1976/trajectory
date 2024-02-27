import paddle
import paddle.nn.functional as F
from model.interface.loss import NetLoss
from model.modeling.losses.__base__.losses import cross_iou, reg_l1_loss
from ppdet.modeling import CTFocalLoss
import numpy as np

class MSANetLoss(NetLoss):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.ce_loss = paddle.nn.CrossEntropyLoss()
        self.ct_loss = CTFocalLoss()

        self.loss_dict = {
            'hm': None,
            'seg': None,
            'wh': None,
            'ang': None,
            's': None,
            'head': None,
            'reg': None,
        }

    def get_loss(self, pred, target):
        self.loss_dict['seg'] = self.seg_loss(pred['seg'], target['target_segmap'])
        self.loss_dict['hm'] = self.config.loss_weights['hm'] * self.ctloss(pred['hm'], target['target_map_hm'])
        self.loss_dict['wh'] = self.config.loss_weights['wh'] * reg_l1_loss(pred['wh'], target['target_map_wh'], target['target_map_reg_mask'])
        self.loss_dict['reg'] = self.config.loss_weights['reg'] * reg_l1_loss(pred['reg'], target['target_map_reg'], target['target_map_reg_mask'])
        self.loss_dict['ang'] = self.config.loss_weights['ang'] * reg_l1_loss(pred['ang'], target['target_map_ang'], target['target_map_reg_mask'])
        self.loss_dict['s'] = self.config.loss_weights['s'] * self.sigmoid_bce_loss(pred['s'], target['target_map_s'], target['target_map_reg_mask'])
        self.loss_dict['head'] = self.config.loss_weights['head'] * reg_l1_loss(pred['head'], target['target_map_head'], target['target_map_reg_mask'])

        total_loss = 0
        for key in self.loss_dict:
            total_loss += self.loss_dict[key]

        return total_loss

    def sigmoid_bce_loss(self, pred, target, mask, weight=None):
        pred = paddle.transpose(pred, perm=(0, 2, 3, 1))
        expand_mask = paddle.tile(paddle.unsqueeze(mask, axis=-1), (1, 1, 1, pred.shape[3]))

        loss = F.binary_cross_entropy_with_logits(pred, target, weight=weight, reduction='none') * expand_mask
        loss = paddle.sum(loss) / (paddle.sum(mask) + 1e-4)

        return loss

    def seg_loss(self, pred, target):
        x = paddle.transpose(pred, perm=[0, 2, 3, 1])
        return self.ce_loss(x, target)

    def ctloss(self, pred, target):
        pred = paddle.transpose(pred, perm=(0, 2, 3, 1))

        # TODO: paddleDetection的ctloss
        loss = self.ct_loss(pred, target)

        return loss