from paddle.fluid.dataloader import Dataset


class BaseDataset(Dataset):
    def __init__(self,
                 cfg,
                 transform):
        super().__init__()
        self.config = cfg
        self.transform = transform
        self.dataset_dir = cfg.dataset_dir
        self.data = []

    def __getitem__(self, idx):
        input_data = self._getitem(idx)
        self.config.input_fields = list(input_data.keys())
        return list(input_data.values())

    def __len__(self):
        return len(self.data)

    def _getitem(self, idx):
        raise NotImplementedError