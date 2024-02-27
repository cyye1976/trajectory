import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Constant

from model.interface.head import NetHead
from model.modeling.libs.layers import ConvBNReLU
from model.utils.util import feat_locations, pool_nms
import numpy as np

class MSADetHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads,
                 pool_size=3):
        super().__init__()
        self.pool_size = pool_size
        self.heads = heads
        for head in self.heads:
            num_output = self.heads[head]
            if 'hm' in head:
                fc = nn.Sequential(
                    ConvBNReLU(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
                        bias_attr=ParamAttr(initializer=Constant(0))),
                    nn.Conv2D(
                        out_channels,
                        num_output,
                        kernel_size=1,
                        padding=0,
                        weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
                        bias_attr=ParamAttr(initializer=Constant(-2.19))))
            else:
                fc = nn.Sequential(
                    ConvBNReLU(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
                        bias_attr=ParamAttr(initializer=Constant(0))),
                    nn.Conv2D(
                        out_channels,
                        num_output,
                        kernel_size=3,
                        padding=1,
                        weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
                        bias_attr=ParamAttr(initializer=Constant(0))))
            self.add_sublayer(head, fc)

    def forward(self, x):
        ret = {}
        for head in self.heads:
            if 'hm' in head:
                ret[head] = F.sigmoid(self.__getattr__(head)(x))
            else:
                ret[head] = self.__getattr__(head)(x)

        ret['feat_location'] = feat_locations(ret['hm'])
        ret['center_mask'] = pool_nms(ret['hm'], kernel=self.pool_size)

        return ret


class MSASegHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 num_classes,
                 is_neck=True):
        super().__init__()
        in_c = 0
        if is_neck:
            in_c = in_channels[0] * 4
        else:
            for i in in_channels:
                in_c += i

        # self.lateral_conv = ConvBNReLU(in_c,
        #                                in_channels[0],
        #                                kernel_size=3,
        #                                padding=1,
        #                                weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
        #                                bias_attr=ParamAttr(initializer=Constant(0)))
        self.out_conv = nn.Conv2D(
            in_c,
            num_classes,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
            bias_attr=ParamAttr(initializer=Constant(0))
        )

    def forward(self, x):
        h, w = x[0].shape[2], x[0].shape[3]

        laterals = []
        for item in x:
            laterals.append(F.interpolate(
                item,
                size=[w, h],
                mode='bilinear'))
        laterals = paddle.concat(laterals, axis=1)
        # laterals = self.lateral_conv(laterals)

        laterals = F.interpolate(
            self.out_conv(laterals),
            size=[w * 4, h * 4],
            mode='bilinear')

        return laterals

class MSANetHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 det_heads,
                 seg_heads):
        super().__init__()
        self.det_head = MSADetHead(in_channels=in_channels,
                                   out_channels=out_channels,
                                   heads=det_heads)
        self.seg_head = MSASegHead(in_channels=in_channels,
                                   out_channels=out_channels,
                                   heads=seg_heads)

    def forward(self, x1, x2):
        det = self.det_head(x1[0])
        seg = self.seg_head(x2)

        det['seg'] = seg

        return det

class MSANetRBBHead(NetHead):
    def __init__(self,
                 in_channels,
                 out_channels,
                 det_heads,
                 seg_heads,
                 cfg):
        super().__init__(cfg)
        if self.config.open_det_head:
            self.det_head = MSADetHead(in_channels=in_channels[0],
                                       out_channels=out_channels,
                                       heads=det_heads)
        if self.config.open_seg_head:
            self.seg_head = MSASegHead(in_channels=in_channels,
                                       num_classes=self.config.seg_class_num,
                                       is_neck=self.config.neck['name'])
    def forward(self, x):
        ret = {}
        if self.config.open_det_head:
            det = self.det_head(x[0])
            ret = det
        if self.config.open_seg_head:
            seg = self.seg_head(x)
            ret['seg'] = seg

        return ret