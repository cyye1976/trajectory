import paddle

class DetMetric(paddle.metric.Metric):
    def __init__(self, cfg, decoder):
        super().__init__()
        self.config = cfg
        self.decoder = decoder

    def reset(self):
        raise NotImplementedError

    def compute(self, pred, *target):
        pred_len = paddle.to_tensor(len(pred))
        target_len = paddle.to_tensor(len(target))
        result = [pred_len, target_len]
        for value in list(pred.values()):
            result.append(value)
        for value in list(target):
            result.append(value)

        return result

    def convert_input_data(self, pred_len, target_len, inputs):
        pred = inputs[:int(pred_len)]
        target = inputs[int(pred_len):int(pred_len) + int(target_len)]

        convert_pred = {}
        convert_target = {}

        for i, field in enumerate(self.config.output_fields):
            convert_pred[field] = pred[i]
        for i, field in enumerate(self.config.input_fields[1:]):
            convert_target[field] = target[i]

        batch = pred[0].shape[0]
        return batch, convert_pred, convert_target

    def update(self, pred_len, target_len, *inputs):
        batch, pred, target = self.convert_input_data(pred_len, target_len, inputs)
        self._update(batch, pred, target)

    def _update(self, batch, pred, target):
        raise NotImplementedError

    def accumulate(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError