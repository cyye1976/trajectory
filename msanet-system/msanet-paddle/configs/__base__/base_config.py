
class BaseConfig:
    """
    全局设定
    """
    debug = True  # 开启调试

    """
    数据读取
    """
    input_size = [512, 512]  # 输入图像大小
    image_aug = False  # 数据增强
    train_batch_size = 4  # 训练样本批量大小
    eval_batch_size = 4  # 评估样本批量大小
    infer_batch_size = 1  # 推理样本批量大小
    freeze_backbone = True  # 是否冻结特征提取网络
    save_freq = 30  # 设定保存模型轮次
    epochs = 20  # 训练总轮次
    step_size = 55  # 一轮输入样本数量（设定衰减策略）
    long_side = True  # 是否开启定向边界框长边表示法

    """
    模型
    """
    model_name = 'base'  # 模型名称
    backbone = dict({
        "name": "ResNet",
        "depth": 34,
        "fpn_channel": [64, 128, 256, 512],
        "dcn_v2_stages": [-1]
    })
    neck = {
        'name': "FPN",
        'use_dcn': False
    }

    down_stride = 4  # 下采样倍率
    obb_nms_threshold = 0.5  # 目标检测定向边界框NMS阈值
    open_det_head = True  # 是否开启检测模块
    open_seg_head = True  # 是否开启分割模块
    open_box_head = False  # 是否开启航向点预测模块（需先开启检测模块）
    open_box_head_crf = False  # 是否开启基于外接矩形框的航向点修正方法
    feat_enhance_layer = None  # 特征增强模块
    confidence_fusion = False  # 置信度融合

    """
    日志路径
    """
    resume_train_model_path = ''  # 恢复训练模型路径（若不进行恢复训练则置为空字符串）
    log_dir = 'log/train/{}'.format(model_name)  # 训练日志保存路径
    eval_log_dir = 'log/eval/{}'.format(model_name)  # 评估日志保存路径
    eval_model_path = 'checkpoint/{}/best'.format(model_name)  # 评估模型路径
    save_dir = 'checkpoint/{}'.format(model_name)  # 训练模型保存路径
    infer_model_path = 'checkpoint/{}/best'.format(model_name)  # 推理模型路径

    """
    数据集路径
    """
    dataset_dir = ''  # 数据集路径

    """
    评估指标设定(model_eval)
    """
    model_eval_monitor = 'f1_ap_iou'  # 选用的评估指标名称
    model_eval_operator = 'lt'  # 评估指标更新等式（大于、小于）

    """
    其他配置
    """
    not_loss = []  # 不进行loss计算的字段
    muti_train_step = -1  # 交替训练步骤[-1(关闭), 0, 1, 2]
    muti_train_type = "ed-s-all"  # ed-s-all, es-d-all

    """
    利用不确定性学习多任务损失权重
    """
    open_custom_multi_loss_layer = False  # 开启不确定性损失计算层

