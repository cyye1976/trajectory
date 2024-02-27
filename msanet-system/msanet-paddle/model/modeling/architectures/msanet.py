from model.interface.net import Net
from model.modeling.heads.msahead import MSANetHead
from model.modeling.libs.layers import MutiTaskSelfAttention
from ppdet.modeling import ResNet, FPN


class MSANet(Net):
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
        self.head = MSANetHead(in_channels=self.config.backbone['fpn_channel'][0],
                               out_channels=self.config.backbone['fpn_channel'][0],
                               det_heads=self.config.det_heads,
                               seg_heads=self.config.seg_heads)
        self.attention1 = MutiTaskSelfAttention(in_channels=self.config.backbone['fpn_channel'],
                                                out_channels=self.config.backbone['fpn_channel'])
        self.attention2 = MutiTaskSelfAttention(in_channels=self.config.backbone['fpn_channel'],
                                                out_channels=self.config.backbone['fpn_channel'])

    def _forward(self, x):
        input = dict({
            'image': x
        })

        out = self.backbone(input)
        out1 = self.attention1(out)
        out2 = self.attention2(out)

        out1 = self.neck(out1)
        out2 = self.neck(out2)

        ret = self.head(out1, out2)

        return ret