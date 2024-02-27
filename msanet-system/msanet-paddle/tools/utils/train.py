from paddle.fluid.reader import DataLoader
import paddle

from ppdet.utils.checkpoint import load_pretrain_weight

from model.utils import model_zoo


class TrainModule(object):
    def __init__(self,
                 cfg,
                 train_dataset,
                 val_dataset,
                 model,
                 optim,
                 loss_fn,
                 metrics,
                 callbacks,
                 ):
        super().__init__()

        self.config = cfg
        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.config.train_batch_size)
        self.val_loader = DataLoader(val_dataset, shuffle=False, batch_size=self.config.eval_batch_size)

        # 模型配置
        if self.config.freeze_backbone:
            model.freeze_backbone()
        if self.config.resume_train_model_path == '':
            load_pretrain_weight(model, pretrain_weight=model_zoo.pretrain_weights_url["{}{}".format(self.config.backbone['name'], self.config.backbone['depth'])])
            print('Load model pretrained weight complete!')
        self.model = paddle.Model(model)

        # 设置优化器
        self.optim = optim

        # 设置损失函数
        self.loss_fn = loss_fn

        # 设置评估函数
        self.metrics = metrics

        # 设置日志记录
        self.callbacks = callbacks

    def train_network(self):
        self.model.prepare(optimizer=self.optim, loss=self.loss_fn, metrics=self.metrics)

        if self.config.resume_train_model_path != '':
            self.model.load(self.config.resume_train_model_path)

        self.model.fit(self.train_loader,
                       self.val_loader,
                       epochs=self.config.epochs,
                       callbacks=self.callbacks)
