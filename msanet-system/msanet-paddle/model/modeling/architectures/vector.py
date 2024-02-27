from model.interface.net import Net
from model.modeling.heads.msahead import MSANetRBBHead
from ppdet.modeling import FPN, ResNet


class VectorNet(Net):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.backbone = ResNet(
            norm_type='bn',
            freeze_norm=False,
            freeze_at=0,
            depth=self.config.backbone['depth'])
        self.neck = FPN(in_channels=self.config.backbone['fpn_channel'],
                        out_channel=self.config.backbone['fpn_channel'][0],
                        extra_stage=0,
                        norm_type='bn')
        self.head = MSANetRBBHead(in_channels=self.config.backbone['fpn_channel'][0],
                                  out_channels=self.config.backbone['fpn_channel'][0],
                                  det_heads=self.config.det_heads,
                                  seg_heads=self.config.seg_heads,
                                  cfg=cfg)

    def _forward(self, x):
        input = dict({
            'image': x
        })

        out = self.backbone(input)
        out = self.neck(out)
        ret = self.head(out)

        return ret
