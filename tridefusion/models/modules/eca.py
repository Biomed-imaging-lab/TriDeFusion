import torch
from torch import nn


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  
        
        self.gap = nn.AdaptiveAvgPool2d(1) 
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.gap(x).view(b, 1, c)
        y = self.conv1d(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y