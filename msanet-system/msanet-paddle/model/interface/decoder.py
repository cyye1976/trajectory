class Decoder(object):
    def __init__(self, cfg):
        self.config = cfg

    def __call__(self, batch, inputs, is_metric=True):
        """
        解码批量输入数据
        :param batch: 遍历到了第几批次
        :param inputs: 批量的输入数据 {"pred":pred, "target": target} or {"pred":pred}
        :param is_metric: 是否处于评估状态（True or False） 评估状态需要输出target_mboxes
        :return:
        """
        raise NotImplementedError

    def trans_inputs(self, batch, inputs):
        """
        转换模型输出结果
        :param batch: 批次
        :param inputs: 模型输出结果 dict
        :return:
        """
        ret = {}
        if self.config.open_det_head:
            for key in self.config.det_heads:
                ret[key] = inputs['pred'][key][batch]
            ret['center_mask'] = inputs['pred']['center_mask'][batch]
            ret['feat_location'] = inputs['pred']['feat_location']

        if self.config.open_seg_head:
            for key in self.config.seg_heads:
                ret[key] = inputs['pred'][key]

        return ret

    def nms_result(self, nms_index, result):
        """
        根据筛选条件筛选结果
        :param nms_index: 筛选索引
        :param result: 检测头结果
        :return:
        """
        for key in result:
            result[key] = result[key][nms_index]

        return result

