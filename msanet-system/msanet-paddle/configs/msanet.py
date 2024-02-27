from configs.__base__.base_config import BaseConfig
from model.utils.model_zoo import pretrain_weights_url


class MSANetConfig(BaseConfig):
    model_name = 'MSANet'

    train_batch_size = 2
    eval_batch_size = 1
    image_aug = False
    freeze_backbone = True
    debug = True
    save_freq = 30
    epochs = 300
    seg_class_num = 3  # 类别包含背景类
    det_class_num = 1  # 类别不包含背景类

    """
    模型配置
    """
    transform_name = 'vector'
    det_heads = {
        'hm': det_class_num,
        'wh': 2,
        'ang': 2,
        's': 1,
        'head': 2,
        'reg': 2
    }
    seg_heads = {
        'seg': seg_class_num
    }
    backbone = dict({
        "name": "ResNet",
        "depth": 34,
        "fpn_channel": [64, 128, 256, 512]
    })
    center_pool_nms_size = 3
    nms_topk = 100  # 可取前topk个点
    max_dets = 50
    center_score_threshold = 0.1
    obb_nms = True

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
        's': 1,
        'head': 0.1,
        'reg': 1,
        'iou': 1,
    }

    """
    数据集路径
    """
    dataset_dir = 'D:/dataset/main/HRSC2016-DS/'  # 数据集路径

    """
    日志路径
    """
    resume_train_model_path = ''  # 恢复训练模型路径（若不进行恢复训练则置为空字符串）
    log_dir = 'log/train/{}'.format(model_name)  # 训练日志保存路径
    eval_log_dir = 'log/eval/{}'.format(model_name)  # 评估日志保存路径
    eval_model_path = 'checkpoint/{}/best'.format(model_name)  # 评估模型路径
    save_dir = 'checkpoint/{}'.format(model_name)  # 训练模型保存路径
    infer_model_path = 'checkpoint/{}/best'.format(model_name)  # 推理模型路径
