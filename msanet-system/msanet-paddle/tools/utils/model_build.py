import paddle

from configs.msanet import MSANetConfig
from configs.msanet_rbb import MSANetRBBConfig
from configs.vector import VectorNetConfig
from model.data.source.KaggleLandShip import KaggleLandShip
from model.data.source.hrsc2016_ds import HRSC2016DS
from model.data.transform.msanet import MSANetTransform
from model.decoders.msanet import MSANetDecoder
from model.decoders.msanet_rbb import MSANetRBBDecoder
from model.metrics.metrics import LossMetric, PascalVOCMetric, SegMetric
from model.modeling.architectures.msanet import MSANet
from model.modeling.architectures.msanet_rbb import MSANetRBB
from model.modeling.architectures.vector import VectorNet
from model.modeling.losses.msaloss import MSANetLoss
from model.modeling.losses.rbb_loss import MSANetRBBLoss
from tools.callbacks import ModelEval
from tools.utils.util import calculate_params


class ModelBuilder(object):
    def __init__(self, args):
        super().__init__()
        model_name, dataset_name = args.model, args.dataset
        phase = args.phase

        config = None
        train_transform = None
        val_transform = None
        train_dataset = None
        val_dataset = None
        test_dataset = None
        model = None
        optim = None
        loss_fn = None
        metrics = None
        callbacks = None
        decoder = None

        if model_name == 'MSANet':
            config = MSANetConfig()
            train_transform = MSANetTransform(cfg=config, mode='train')
            val_transform = MSANetTransform(cfg=config, mode='val')
            # 模型配置
            model = MSANet(cfg=config)
            # 设置损失函数
            loss_fn = MSANetLoss(cfg=config)
            # 设置输出结果解码器
            decoder = MSANetDecoder(cfg=config)
        elif model_name == 'MSANetRBB':
            config = MSANetRBBConfig()
            train_transform = MSANetTransform(cfg=config, mode='train')
            val_transform = MSANetTransform(cfg=config, mode='val')
            # 模型配置
            model = MSANetRBB(cfg=config)
            # 设置损失函数
            loss_fn = MSANetRBBLoss(cfg=config)
            # 设置输出结果解码器
            decoder = MSANetRBBDecoder(cfg=config)
        elif model_name == 'VectorNet':
            config = VectorNetConfig()
            train_transform = MSANetTransform(cfg=config, mode='train')
            val_transform = MSANetTransform(cfg=config, mode='val')
            # 模型配置
            model = VectorNet(cfg=config)
            # 设置损失函数
            loss_fn = MSANetLoss(cfg=config)
            # 设置输出结果解码器
            decoder = MSANetDecoder(cfg=config)
        else:
            raise Exception(print('Model no accept!'))

        assert config is not None, "config is None."
        assert train_transform is not None, "train_transform is None."
        assert val_transform is not None, "val_transform is None."
        assert model is not None, "model is None."
        assert loss_fn is not None, "loss_fn is None."
        assert decoder is not None, "decoder is None."

        # 配置数据源
        if dataset_name == 'HRSC2016DS':
            if phase == 'train':
                train_dataset = HRSC2016DS(cfg=config, transform=train_transform)
                val_dataset = HRSC2016DS(cfg=config, mode='val', transform=val_transform)
            elif phase == 'eval':
                test_dataset = HRSC2016DS(cfg=config, mode='test', transform=val_transform)
            elif phase == 'infer':
                test_dataset = HRSC2016DS(cfg=config, mode='test', transform=val_transform)
            else:
                raise Exception(print('Phase error!'))
        elif dataset_name == 'KaggleLandShip':
            if phase == 'train':
                train_dataset = KaggleLandShip(cfg=config, transform=train_transform)
                val_dataset = KaggleLandShip(cfg=config, mode='val', transform=val_transform)
            elif phase == 'eval':
                test_dataset = KaggleLandShip(cfg=config, mode='test', transform=val_transform)
            elif phase == 'infer':
                test_dataset = KaggleLandShip(cfg=config, mode='test', transform=val_transform)
            else:
                raise Exception(print('Phase error!'))
        else:
            raise Exception(print('Dataset no accept!'))

        # 设置优化器
        lr_scheduler = paddle.optimizer.lr.StepDecay(learning_rate=config.lr,
                                                     step_size=int(config.epochs / 3) * config.step_size, gamma=0.5,
                                                     verbose=False)

        parameters = model.parameters()
        if config.open_custom_multi_loss_layer:
            parameters += loss_fn.multi_loss.log_vars
        optim = paddle.optimizer.Adam(parameters=parameters, learning_rate=lr_scheduler, weight_decay=0.0001)

        # 设置评估函数
        metrics = [LossMetric(config, criterion=loss_fn)]
        if config.open_det_head:
            metrics.append(PascalVOCMetric(config, decoder, head=config.open_box_head))
        if config.open_seg_head:
            metrics.append(SegMetric(config))

        # 设置日志记录
        visual_dl_callback = paddle.callbacks.VisualDL(log_dir=config.log_dir)
        callback = paddle.callbacks.ModelCheckpoint(save_dir=config.save_dir,
                                                    save_freq=config.save_freq)
        eval_callback = ModelEval(log_dir=config.log_dir + "/epoch_eval_log.txt",
                                  save_dir=config.save_dir,
                                  monitor=config.model_eval_monitor,
                                  operator=config.model_eval_operator,
                                  cfg=config)
        callbacks = [visual_dl_callback, callback, eval_callback]

        assert metrics is not None, "metrics is None."
        # assert train_dataset is not None, "train_dataset is None."
        # assert val_dataset is not None, "val_dataset is None."
        assert callbacks is not None, "callbacks is None."
        assert optim is not None, "optim is None."

        self.config = config
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.callbacks = callbacks
        self.decoder = decoder
        self.test_dataset = test_dataset

        # 保存输入配置
        self.config.args = args
        self.config.dataset_name = args.dataset

    def get_model(self):
        model = dict()
        for key in list(self.__dict__.keys()):
            model[key] = self.__getattribute__(key)

        return model


