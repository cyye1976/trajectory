import os

import cv2
import paddle
import paddle.nn.functional as F
from configs.msanet_rbb import MSANetRBBConfig
from main import parse_args
from model.modeling.architectures.msanet_rbb import MSANetRBB
from model.modeling.losses.rbb_loss import MSANetRBBLoss
from tools.utils.model_build import ModelBuilder


# 以正确标签lebal获取gradcam热图
# 获取 Grad-CAM 类激活热图
def get_gradcam(bulider, data, label, class_dim=2):
    input = dict({
        'image': data
    })

    config = bulider['config']
    model = bulider['model']

    # TODO: 前向传播
    # 特征提取
    out = model.backbone(input)
    # 特征融合
    out = model.neck(out)
    # 输出头
    ret = {}
    if config.open_det_head:
        det = model.head.det_head(out[0])
        ret = det
    if config.open_seg_head:
        h, w = out[0].shape[2], out[0].shape[3]

        laterals = []
        for item in out:
            laterals.append(F.interpolate(
                item,
                size=[w, h],
                mode='bilinear'))
        laterals_1 = paddle.concat(laterals, axis=1)
        # laterals = self.lateral_conv(laterals)

        laterals = F.interpolate(
            model.head.seg_head.out_conv(laterals_1),
            size=[w * 4, h * 4],
            mode='bilinear')
        ret['seg'] = laterals

    loss_fn = bulider['loss_fn']
    loss = loss_fn(ret, label)

    loss.backward()  # 反向传播计算梯度
    grad_map = out[0].grad  # 得到目标类别的loss对最后一个卷积层输出的特征图的梯度
    grad = paddle.mean(paddle.to_tensor(grad_map), (2, 3), keepdim=True)  # 对特征图的梯度进行GAP（全局平局池化）
    gradcam = paddle.sum(grad * out[0], axis=1)  # 将最后一个卷积层输出的特征图乘上从梯度求得权重进行各个通道的加和
    gradcam = paddle.maximum(gradcam, paddle.to_tensor(0.))  # 进行ReLU操作，小于0的值设为0
    for j in range(gradcam.shape[0]):
        gradcam[j] = gradcam[j] / paddle.max(gradcam[j])  # 分别归一化至[0, 1]
    return gradcam


# 将 Grad-CAM 叠加在原图片上显示激活热图的效果
def show_gradcam(builder, data, label, image_id, output_path, show_cam=False):
    gradcams = get_gradcam(builder, data, label)
    images_dir = builder['test_dataset'].images_path
    images = builder['test_dataset'].images_name
    for i in range(data.shape[0]):
        img = cv2.resize(cv2.imread(images_dir + images[image_id]), (data.shape[2], data.shape[3]))
        # img = (data[i] * 255.).numpy().astype('uint8').transpose([1, 2, 0])  # 归一化至[0,255]区间，形状：[h,w,c]
        heatmap = cv2.resize(gradcams[i].numpy() * 255., (data.shape[2], data.shape[3])).astype(
            'uint8')  # 调整热图尺寸与图片一致、归一化
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热图转化为“伪彩热图”显示模式
        superimposed_img = cv2.addWeighted(heatmap, .3, img, .7, 0.)  # 将特图叠加到原图片上
        cv2.imwrite(output_path, superimposed_img)
        if show_cam:
            cv2.imshow("heatmap", superimposed_img)
            cv2.waitKey(0)


def update_model_builder(model_builder, key):
    # 根据联合任务独立任务改变配置模型
    config = model_builder['config']
    if config.model_name == 'MSANetRBB':
        model_builder['model'], model_builder['loss_fn'] = _update_model_builder(key, config, MSANetRBB, MSANetRBBLoss)
        model_builder['config'] = config
    return model_builder


def _update_model_builder(key, config, model_class, loss_fn_class):
    if key == 'det+seg':
        config.open_det_head = True
        config.open_seg_head = True
    elif key == 'det':
        config.open_det_head = True
        config.open_seg_head = False
    elif key == 'seg':
        config.open_det_head = False
        config.open_seg_head = True
    model = model_class(cfg=config)
    loss_fn = loss_fn_class(cfg=config)
    return model, loss_fn


def show_batch_gradcam(model_builder, load_model_path_dict, image_id, save_dir):
    if os.path.exists(save_dir) is not True:
        os.mkdir(save_dir)

    # 数据加载
    loader = model_builder['test_dataset']
    datas = loader[image_id]
    img = paddle.unsqueeze(datas[0], axis=0)

    for i, item in enumerate(datas):
        if i == 0:
            continue
        datas[i] = paddle.unsqueeze(item, axis=0)
    label = datas[1:]

    for key in load_model_path_dict:
        model_builder = update_model_builder(model_builder, key)
        # 模型读取
        paddle.Model(model_builder['model']).load(load_model_path_dict[key])
        # 计算特征图可视化，并保存结果
        show_gradcam(model_builder, img, label, image_id, save_dir + key + ".jpg")
        print('Complete {} grad_cam.'.format(key))
    print('Complete show and write grad-cam.')

if __name__ == '__main__':
    args = parse_args()
    phase = ['train', 'eval', 'infer']
    dataset = ['HRSC2016DS', 'KaggleLandShip']
    model = ['MSANet', 'MSANetRBB', 'VectorNet']
    args.phase = phase[2]
    args.dataset = dataset[0]
    args.model = model[1]

    config = MSANetRBBConfig()
    model_builder = ModelBuilder(args).get_model()

    # img_path = '../../demo/det_test.jpg'
    # out_dir = '../../demo/test'
    # size = (512, 512)

    # camImages(net, img_path, out_dir, size=size)
    # draw_CAM(net, img_path, out_dir)

    # 图像加载&预处理
    # img = Image.open(img_path).convert('RGB')

    # label = paddle.to_tensor(paddle.ones_like(img), dtype='int32')
    root_dir = "../../" + model_builder['config'].infer_model_path
    load_model_path_dict = {
        'det': root_dir + "/det",
        'seg': root_dir + "/seg",
        'det+seg': root_dir + "/det+seg"
    }

    # 参数配置
    image_id = 8

    show_batch_gradcam(model_builder, load_model_path_dict, image_id=image_id, save_dir='output/')
