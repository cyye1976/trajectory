import numpy as np
import paddle
import paddle.nn.functional as F

if __name__ == '__main__':
    arr = [[0, 30, 0, 0, 0],
           [20, 0, 20, 0, 0],
           [0, 0, 0, 10, 0]]
    arr = np.array(arr)
    p_arr = paddle.unsqueeze(paddle.unsqueeze(paddle.to_tensor(arr), axis=0), axis=0)
    result = F.interpolate(p_arr,
                           mode='bilinear',
                           size=[10, 10])

    result = paddle.squeeze(paddle.squeeze(result, axis=0), axis=0).numpy()

    pass

