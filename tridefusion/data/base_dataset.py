import torch.utils.data as data
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

class BaseDataset(data.Dataset, ABC):

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_cli_options(parser, is_train):
        return parser
    

def get_transform():
    pass