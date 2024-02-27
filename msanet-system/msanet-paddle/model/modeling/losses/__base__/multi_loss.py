import paddle
import paddle.nn.functional as F
import paddle.nn as nn
from paddle import ParamAttr


class CustomMultiLossLayer(nn.Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        super(CustomMultiLossLayer, self).__init__(**kwargs)
        self.nb_outputs = nb_outputs

        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_parameter(
                "log_var_{}".format(i),
                self.create_parameter(
                    shape=[1,],
                    attr=nn.initializer.Constant(value=0.)
                )
            )]

    def multi_loss(self, loss_dict):
        assert len(loss_dict) == self.nb_outputs, "len loss_dict not match len nb_outputs."
        loss = 0
        for key, log_var in zip(loss_dict, self.log_vars):
            precision = paddle.exp(-log_var)
            loss += paddle.sum(precision * loss_dict[key] + log_var, -1)
        return loss

    def forward(self, inputs):
        loss = self.multi_loss(inputs)
        return loss


