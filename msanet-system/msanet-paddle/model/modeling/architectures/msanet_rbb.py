from paddleseg.models import layers

from model.interface.net import Net
from model.modeling.heads.msahead import MSANetHead, MSANetRBBHead
from model.modeling.necks.fpn import FPN
from model.utils.util import freeze_layers
from ppdet.modeling import ResNet


class MSANetRBB(Net):
    def __init__(self, cfg):
        super().__init__(cfg)

        # 特征提取
        self.backbone = self.build_backbone()

        # 特征增强
        self.enhance_layer = self.build_feat_enhance_layer()

        # 特征融合
        self.neck = self.build_neck()

        # 预测器
        self.head = MSANetRBBHead(in_channels=self.config.backbone['fpn_channel'],
                                  out_channels=self.config.backbone['fpn_channel'][0],
                                  det_heads=self.config.det_heads,
                                  seg_heads=self.config.seg_heads,
                                  cfg=self.config)

        # 多任务训练策略
        self.multi_task_alter_training()

    def _forward(self, x):
        input = dict({
            'image': x
        })

        # 特征提取
        out = self.backbone(input)

        # 特征增强
        if self.enhance_layer:
            out[-1] = self.enhance_layer(out[-1])

        # 特征融合
        if self.neck:
            out = self.neck(out)

        # 输出头
        ret = self.head(out)

        return ret
