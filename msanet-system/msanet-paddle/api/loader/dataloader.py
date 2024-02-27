import os

import imageio
from paddle.fluid.dataloader import Dataset
from paddle.vision import Compose, ToTensor, Normalize, Resize


class ApiDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 group_id):
        super().__init__()
        """
        dataset_dir: 数据集根路径，即分组号路径前一个目录
        group_id: 分组ID号
        """

        self.transform = Compose([
            ToTensor(),
            Resize(size=(512, 512)),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.images_path = dataset_dir + str(group_id)

        # 数据读入
        self.images_name = os.listdir(self.images_path)
        self.data = []
        count = 0
        for image_name in self.images_name:
            count += 1
            # 加载图片
            image = imageio.imread(self.images_path + "/" + image_name)
            # 装载数据
            self.data.append([image])
            print("Complete {}/{}.".format(count, len(self.images_name)))

    def __getitem__(self, idx):
        image = self.data[idx][0]

        input_data = self.transform(image)

        return input_data

    def __len__(self):
        return len(self.data)
