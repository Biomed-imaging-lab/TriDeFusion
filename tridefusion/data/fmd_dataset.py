from .base_dataset import BaseDataset

class FmdDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)