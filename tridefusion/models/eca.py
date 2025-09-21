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

class EfficientResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EfficientResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        self.eca = ECABlock(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.eca(self.conv2(self.conv1(x))) + self.skip(x)