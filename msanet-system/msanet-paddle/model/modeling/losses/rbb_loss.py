from model.interface.loss import NetLoss
import paddle

from model.modeling.losses.__base__.losses import reg_l1_loss, cross_iou_loss
from model.modeling.losses.__base__.multi_loss import CustomMultiLossLayer
from ppdet.modeling import CTFocalLoss


class MSANetRBBLoss(NetLoss):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.ce_loss = paddle.nn.CrossEntropyLoss()
        self.ct_loss = CTFocalLoss()

        # 不确定性损失计算层
        if self.config.open_custom_multi_loss_layer:
            # nb_outputs = 0
            # if self.config.open_det_head:
            #     for key in self.config.det_heads:
            #         if key in self.config.not_loss:
            #             continue
            #         nb_outputs += 1
            # if self.config.open_seg_head:
            #     for key in self.config.seg_heads:
            #         if key in self.config.not_loss:
            #             continue
            #         nb_outputs += 1
            self.multi_loss = CustomMultiLossLayer(2)

    def get_loss(self, pred, target):
        total_loss = 0

        train_types = self.config.muti_train_type.split("-")
        if 'd' in train_types[self.config.muti_train_step]:
            # 只训练目标检测分支
            self.loss_dict['hm'] = self.ctloss(pred['hm'], target['target_map_hm'])
            self.loss_dict['wh'] = reg_l1_loss(pred['wh'], target['target_map_wh'],
                                               target['target_map_reg_mask'])
            self.loss_dict['reg'] = reg_l1_loss(pred['reg'], target['target_map_reg'],
                                                target['target_map_reg_mask'])
            self.loss_dict['ang'] = reg_l1_loss(pred['ang'], target['target_map_ang'],
                                                target['target_map_reg_mask'])
            if 'head' in self.config.det_heads:
                self.loss_dict['head'] = reg_l1_loss(pred['head'], target['target_map_head'],
                                                     target['target_map_reg_mask'])
            # 冻结语义分割分支情况
            self.loss_dict['seg'] = paddle.to_tensor(0, dtype='float32')
        elif 's' in train_types[self.config.muti_train_step]:
            # 只训练语义分割分支
            self.loss_dict['seg'] = self.seg_loss(pred['seg'], target['target_segmap'])
            # 冻结编码器和目标检测分支的情况
            for key in self.loss_dict:
                if key in self.config.det_heads:
                    self.loss_dict[key] = paddle.to_tensor(0, dtype='float32')
        elif 'all' in train_types[self.config.muti_train_step] or self.config.muti_train_step == -1:
            # 全部分支联合训练
            if self.config.open_det_head:
                self.loss_dict['hm'] = self.ctloss(pred['hm'], target['target_map_hm'])
                self.loss_dict['wh'] = reg_l1_loss(pred['wh'], target['target_map_wh'],
                                                   target['target_map_reg_mask'])
                self.loss_dict['reg'] = reg_l1_loss(pred['reg'], target['target_map_reg'],
                                                    target['target_map_reg_mask'])
                self.loss_dict['ang'] = reg_l1_loss(pred['ang'], target['target_map_ang'],
                                                    target['target_map_reg_mask'])
            if 'head' in self.config.det_heads:
                self.loss_dict['head'] = reg_l1_loss(pred['head'], target['target_map_head'],
                                                     target['target_map_reg_mask'])
            if self.config.open_seg_head:
                self.loss_dict['seg'] = self.seg_loss(pred['seg'], target['target_segmap'])


        # if self.config.open_seg_head:
        #     if self.config.muti_train_type == 'ed-s-all':
        #         if self.config.muti_train_step != 0:
        #             self.loss_dict['seg'] = self.seg_loss(pred['seg'], target['target_segmap'])
        #         else:
        #             # 冻结语义分割分支情况
        #             self.loss_dict['seg'] = paddle.to_tensor(0, dtype='float32')
        #     elif self.config.muti_train_type == "es-d-all":
        #         if self.config.muti_train_step == 1:
        #             # 冻结语义分割分支情况
        #             self.loss_dict['seg'] = paddle.to_tensor(0, dtype='float32')
        #         else:
        #             self.loss_dict['seg'] = self.seg_loss(pred['seg'], target['target_segmap'])
        #
        # if self.config.open_det_head:
        #     if self.config.muti_train_type == "ed-s-all":
        #         if self.config.muti_train_step != 1:
        #             self.loss_dict['hm'] = self.ctloss(pred['hm'], target['target_map_hm'])
        #             self.loss_dict['wh'] = reg_l1_loss(pred['wh'], target['target_map_wh'],
        #                                                target['target_map_reg_mask'])
        #             self.loss_dict['reg'] = reg_l1_loss(pred['reg'], target['target_map_reg'],
        #                                                 target['target_map_reg_mask'])
        #             self.loss_dict['ang'] = reg_l1_loss(pred['ang'], target['target_map_ang'],
        #                                                 target['target_map_reg_mask'])
        #             if 'iou' in self.config.det_heads:
        #                 self.config.not_loss.append('ang')
        #                 self.loss_dict['iou'] = self.config.loss_weights['iou'] * cross_iou_loss(pred['iou'], target[
        #                     'target_map_iou'], target['target_map_reg_mask'])
        #                 self.loss_dict['ang'] *= self.config.loss_weights['ang']
        #                 self.loss_dict['l1'] = self.loss_dict['ang'] * self.loss_dict['iou']
        #                 total_loss += self.loss_dict['l1']
        #             if 'head' in self.config.det_heads:
        #                 self.loss_dict['head'] = reg_l1_loss(pred['head'], target['target_map_head'],
        #                                                      target['target_map_reg_mask'])
        #         else:
        #             # 冻结编码器和目标检测分支的情况
        #             for key in self.loss_dict:
        #                 if key in self.config.det_heads:
        #                     self.loss_dict[key] = paddle.to_tensor(0, dtype='float32')
        #     elif self.config.muti_train_type == "es-d-all":
        #         if self.config.muti_train_step == 0:
        #             # 冻结编码器和目标检测分支的情况
        #             for key in self.loss_dict:
        #                 if key in self.config.det_heads:
        #                     self.loss_dict[key] = paddle.to_tensor(0, dtype='float32')
        #         else:
        #             self.loss_dict['hm'] = self.ctloss(pred['hm'], target['target_map_hm'])
        #             self.loss_dict['wh'] = reg_l1_loss(pred['wh'], target['target_map_wh'],
        #                                                target['target_map_reg_mask'])
        #             self.loss_dict['reg'] = reg_l1_loss(pred['reg'], target['target_map_reg'],
        #                                                 target['target_map_reg_mask'])
        #             self.loss_dict['ang'] = reg_l1_loss(pred['ang'], target['target_map_ang'],
        #                                                 target['target_map_reg_mask'])
        #             if 'iou' in self.config.det_heads:
        #                 self.config.not_loss.append('ang')
        #                 self.loss_dict['iou'] = self.config.loss_weights['iou'] * cross_iou_loss(pred['iou'], target[
        #                     'target_map_iou'], target['target_map_reg_mask'])
        #                 self.loss_dict['ang'] *= self.config.loss_weights['ang']
        #                 self.loss_dict['l1'] = self.loss_dict['ang'] * self.loss_dict['iou']
        #                 total_loss += self.loss_dict['l1']
        #             if 'head' in self.config.det_heads:
        #                 self.loss_dict['head'] = reg_l1_loss(pred['head'], target['target_map_head'],
        #                                                      target['target_map_reg_mask'])

        if self.config.open_custom_multi_loss_layer and self.config.open_seg_head and self.config.open_det_head:
            # 利用不确定性计算损失(只分检测和分割分支)
            multi_loss_dict = {'seg': 0, 'det': 0}
            for key in self.config.seg_heads:
                multi_loss_dict['seg'] += self.config.loss_weights[key] * self.loss_dict[key]
            for key in self.config.det_heads:
                multi_loss_dict['det'] += self.config.loss_weights[key] * self.loss_dict[key]

            total_loss += self.multi_loss(multi_loss_dict)
        else:
            # 固定权重计算损失和
            for key in self.loss_dict:
                if key in self.config.not_loss:
                    continue
                self.loss_dict[key] *= self.config.loss_weights[key]
                total_loss += self.loss_dict[key]

        return total_loss

    def seg_loss(self, pred, target):
        x = paddle.transpose(pred, perm=[0, 2, 3, 1])
        return self.ce_loss(x, target)

    def ctloss(self, pred, target):
        pred = paddle.transpose(pred, perm=(0, 2, 3, 1))

        # TODO: paddleDetection的ctloss
        loss = self.ct_loss(pred, target)

        return loss