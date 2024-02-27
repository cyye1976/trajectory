from configs.__base__.base_config import BaseConfig
from model.data.source import source_datasets


class MSANetRBBConfig(BaseConfig):
    model_name = 'MSANetRBB'

    input_size = [512, 512]  # 输入图像大小

    train_batch_size = 2
    eval_batch_size = 1
    image_aug = False
    freeze_backbone = False
    debug = False
    save_freq = 30
    epochs = 600
    seg_class_num = 3  # 类别包含背景类
    det_class_num = 1  # 类别不包含背景类

    """
    模型配置
    """
    transform_name = 'rbb'  # TODO
    det_heads = {
        'hm': det_class_num,
        'wh': 2,
        'ang': 1,
        'head': 2,
        'reg': 2,
    }
    seg_heads = {
        'seg': seg_class_num
    }
    backbone = dict({
        "name": "ResNet",
        "depth": 101,
        "fpn_channel": [256, 512, 1024, 2048],
        "dcn_v2_stages": [-1]
    })
    neck = {
        'name': 'FPN',
        'use_dcn': False
    }

    center_pool_nms_size = 3
    nms_topk = 100  # 可取前topk个点
    max_dets = 50
    center_score_threshold = 0.1
    obb_nms = True
    open_det_head = True
    open_seg_head = True
    open_box_head = True
    open_box_head_crf = True
    feat_enhance_layer = None  # 是否开启特征增强模块
    confidence_fusion = True  # 置信度融合

    """
    模型训练
    """
    lr = 1e-4
    pretrained = False
    loss_weights = {
        'hm': 1,
        'seg': 1,
        'wh': 0.1,
        'ang': 0.1,
        'reg': 1,
        'iou': 1,
        'head': 0.1,
    }

    """
    数据集路径
    """
    dataset_name = "HRSC2016DS"
    dataset_dir = source_datasets[dataset_name]  # 数据集路径

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
    评估指标设定(model_eval)
    """
    model_eval_monitor = 'f1_ap_iou'
    model_eval_operator = 'lt'

    """
    其他配置
    """
    not_loss = ['iou', 'l1']
    muti_train_step = -1
    muti_train_type = "ed-all"     # ed-s-all(0,1,2), es-d-all(0,1,2), ed-all(0,1), es-all(0,1)

    """
    利用不确定性学习多任务损失权重
    """
    open_custom_multi_loss_layer = False  # 开启不确定性损失计算层