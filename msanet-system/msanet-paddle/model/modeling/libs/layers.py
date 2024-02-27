from abc import ABC

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant, Normal


class ConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        self._batch_norm = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = F.relu(x)
        return x


class MutiTaskSelfAttention(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.conv_list = []
        for i, channel in enumerate(in_channels):
            self.conv_list.append(SelfAttention(channel, out_channels[i]))

    def forward(self, x):
        outs = []
        for i, item in enumerate(x):
            outs.append(self.conv_list[i](item))

        return outs


class SelfAttention(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.conv_f = nn.Conv2D(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
                                bias_attr=ParamAttr(initializer=Constant(0)))
        self.conv_g = nn.Conv2D(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
                                bias_attr=ParamAttr(initializer=Constant(0)))
        self.conv_h = nn.Conv2D(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                weight_attr=ParamAttr(initializer=Normal(0.0, 0.01)),
                                bias_attr=ParamAttr(initializer=Constant(0)))

    def forward(self, x):
        fx = self.conv_f(x)
        gx = self.conv_g(x)
        hx = self.conv_h(x)

        out = F.softmax(paddle.matmul(fx, gx, transpose_x=True))
        out = paddle.matmul(out, hx)
        out += x

        return out
