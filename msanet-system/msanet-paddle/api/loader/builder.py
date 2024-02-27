import argparse

import paddle

from configs.msanet import MSANetConfig

from configs.vector import VectorNetConfig
from model.decoders.msanet import MSANetDecoder
from model.decoders.msanet_rbb import MSANetRBBDecoder
from model.modeling.architectures.msanet import MSANet
from model.modeling.architectures.msanet_rbb import MSANetRBB
from model.modeling.architectures.vector import VectorNet
from model.modeling.losses.msaloss import MSANetLoss
from model.modeling.losses.rbb_loss import MSANetRBBLoss
from api.loader.msanet_rbb import MSANetRBBConfig


class ModelBuilder(object):
    def __init__(self, model_name):
        super().__init__()

        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.phase = 'infer'

        if model_name == 'MSANet':
            config = MSANetConfig()
            # 模型配置
            model = MSANet(cfg=config)
            # 设置损失函数
            loss_fn = MSANetLoss(cfg=config)
            # 设置输出结果解码器
            decoder = MSANetDecoder(cfg=config)
        elif model_name == 'MSANetRBB':
            config = MSANetRBBConfig()
            # 模型配置
            model = MSANetRBB(cfg=config)
            # 设置损失函数
            loss_fn = MSANetRBBLoss(cfg=config)
            # 设置输出结果解码器
            decoder = MSANetRBBDecoder(cfg=config)
        elif model_name == 'VectorNet':
            config = VectorNetConfig()
            # 模型配置
            model = VectorNet(cfg=config)
            # 设置损失函数
            loss_fn = MSANetLoss(cfg=config)
            # 设置输出结果解码器
            decoder = MSANetDecoder(cfg=config)
        else:
            raise Exception(print('Model no accept!'))

        lr_scheduler = paddle.optimizer.lr.StepDecay(learning_rate=config.lr,
                                                     step_size=int(config.epochs / 3) * config.step_size, gamma=0.5,
                                                     verbose=False)

        parameters = model.parameters()
        if config.open_custom_multi_loss_layer:
            parameters += loss_fn.multi_loss.log_vars
        optim = paddle.optimizer.Adam(parameters=parameters, learning_rate=lr_scheduler, weight_decay=0.0001)

        config.args = args
        self.config = config
        self.model = paddle.Model(model)
        self.loss_fn = loss_fn
        self.decoder = decoder
        self.optim = optim
        self.decoder = decoder

    def get_model(self):
        model = dict()
        for key in list(self.__dict__.keys()):
            model[key] = self.__getattribute__(key)

        return model