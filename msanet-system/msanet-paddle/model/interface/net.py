import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers

from model.modeling.backbones.resnet import ResNet
from model.modeling.necks.fpn import FPN
from model.utils.util import freeze_layers
from ppdet.modeling import HRNet


class Net(nn.Layer):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.stop_gradient = True

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.stop_gradient = False

    def multi_task_alter_training(self):
        """
        多任务交替训练
        :return:
        """
        if self.config.open_det_head and self.config.open_seg_head:
            assert self.config.muti_train_type in ["ed-s-all", "es-d-all", "ed-all", "es-all"], "muti train type not accept."
            if self.config.muti_train_type == "ed-s-all":
                if self.config.muti_train_step == 0:
                    # 先冻结语义分割分支
                    freeze_layers(self.head.seg_head)
                elif self.config.muti_train_step == 1:
                    # 冻结编码器和目标检测分支
                    freeze_layers(self.backbone)
                    freeze_layers(self.neck)
                    freeze_layers(self.head.det_head)
            elif self.config.muti_train_type == "es-d-all":
                if self.config.muti_train_step == 0:
                    # 先冻结目标检测分支
                    freeze_layers(self.head.det_head)
                elif self.config.muti_train_step == 1:
                    # 冻结编码器和语义分割分支
                    freeze_layers(self.backbone)
                    freeze_layers(self.neck)
                    freeze_layers(self.head.seg_head)
            elif self.config.muti_train_type == "ed-all":
                if self.config.muti_train_step == 0:
                    # 冻结语义分割分支
                    freeze_layers(self.head.seg_head)
            elif self.config.muti_train_type == "es-all":
                if self.config.muti_train_step == 0:
                    # 冻结目标检测分支
                    freeze_layers(self.head.det_head)

    def forward(self, x):
        pred = self._forward(x)
        output_fields = []
        del_output_fields = []
        tensor_type = type(paddle.to_tensor([1], dtype='float32'))
        for key in list(pred.keys()):
            if type(pred[key]) != tensor_type:
                del_output_fields.append(key)
                continue
            output_fields.append(key)
        self.config.output_fields = output_fields
        self.config.del_output_fields = del_output_fields
        if self.config.args.phase == 'infer':
            for key in self.config.del_output_fields:
                del pred[key]
            pred = list(pred.values())
        return pred

    def _forward(self, x):
        raise NotImplementedError

    def build_feat_enhance_layer(self):
        """
        特征增强模块选择
        :return:
        """
        layer = None

        if self.config.feat_enhance_layer is not None:
            assert self.config.feat_enhance_layer in ['spp', 'aspp'], 'This feat enhance layer is not accept'
            if self.config.feat_enhance_layer == 'spp':
                layer = layers.PPModule(
                    in_channels=self.config.backbone['fpn_channel'][-1],
                    out_channels=self.config.backbone['fpn_channel'][-1],
                    bin_sizes=(1, 2, 3, 6),
                    dim_reduction=True,
                    align_corners=False)
            elif self.config.feat_enhance_layer == 'aspp':
                layer = layers.ASPPModule((1, 6, 12, 18),
                                          self.config.backbone['fpn_channel'][-1],
                                          self.config.backbone['fpn_channel'][-1],
                                          False,
                                          use_sep_conv=True,
                                          image_pooling=True)
        return layer

    def build_backbone(self):
        """
        选择特征提取网络
        :return:
        """
        assert self.config.backbone['name'] in ['ResNet', 'HRNet'], "this backbone is not accept."
        backbone = None

        if self.config.backbone['name'] == 'ResNet':
            backbone = ResNet(
                norm_type='bn',
                freeze_norm=False,
                freeze_at=0,
                dcn_v2_stages=self.config.backbone['dcn_v2_stages'],
                depth=self.config.backbone['depth'])
        elif self.config.backbone['name'] == 'HRNet':
            backbone = HRNet(
                width=self.config.backbone['depth'],
                has_se=False,
                freeze_at=4,
                freeze_norm=False,
                norm_decay=0.,
                return_idx=[0, 1, 2, 3])

        return backbone

    def build_neck(self):
        """
        选择特征融合模块
        :return:
        """
        neck = None
        if self.config.neck['name'] is not None:
            assert self.config.neck['name'] in ['FPN'], "this neck is not accept."
            if self.config.neck['name'] == 'FPN':
                neck = FPN(in_channels=self.config.backbone['fpn_channel'],
                           out_channel=self.config.backbone['fpn_channel'][0],
                           extra_stage=0,
                           norm_type='bn',
                           use_dcn=self.config.neck['use_dcn'])
        return neck
