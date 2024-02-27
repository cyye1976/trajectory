import os

import paddle
from paddle.fluid.reader import DataLoader

class EvalModule(object):
    def __init__(self,
                 cfg,
                 test_dataset,
                 model,
                 optim,
                 loss_fn,
                 metrics,
                 callbacks,
                 ):
        super().__init__()

        self.config = cfg
        self.val_loader = DataLoader(test_dataset, shuffle=False, batch_size=self.config.eval_batch_size)

        # 模型配置
        self.model = paddle.Model(model)

        # 设置优化器
        self.optim = optim

        # 设置损失函数
        self.loss_fn = loss_fn

        # 设置评估函数
        self.metrics = metrics

        # 设置日志记录
        self.callbacks = callbacks

    def eval_network(self):
        self.model.prepare(optimizer=self.optim, loss=self.loss_fn, metrics=self.metrics)

        self.model.load(self.config.eval_model_path)
        if not os.path.exists(self.config.eval_log_dir):
            os.makedirs(self.config.eval_log_dir)

        self.model.evaluate(self.val_loader, callbacks=self.callbacks)
