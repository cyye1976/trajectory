import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import param_init

class CenterNetHead(nn.Layer):
    def __init__(self, heads, center_pool_nms_size=3):
        super().__init__()
        self.heads = heads
        self.center_pool_nms_size = center_pool_nms_size

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.01)
            elif isinstance(layer, (nn.BatchNorm2D, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
        # 再单独对中心热力图预测参数初始化
        hm_layer = self.hm.sublayers()
        param_init.constant_init(hm_layer[-1].bias, value=-2.19)

class NetHead(nn.Layer):
    def __init__(self, cfg):
        super(NetHead, self).__init__()
        self.config = cfg