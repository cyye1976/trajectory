import numpy as np

def calculate_params(model):
    """
    计算模型参数量
    :param model:
    :return:
    """
    n_train = 0
    n_non_train = 0
    for p in model.parameters():
        if p.trainable:
            n_train += np.prod(p.shape)
        else:
            n_non_train += np.prod(p.shape)
    return n_train + n_non_train, n_train, n_non_train