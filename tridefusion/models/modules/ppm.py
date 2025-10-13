import torch
from torch import nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()
        self.pool_layers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=pool),
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels // 4),
                nn.SiLU()
            ) for pool in pool_sizes
        ])
        self.conv_final = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        pooled_features = [x] + [F.interpolate(layer(x), size=x.shape[2:], mode="bilinear", align_corners=False) for layer in self.pool_layers]
        return self.conv_final(torch.cat(pooled_features, dim=1))

class AttentionGate(nn.Module):
    def __init__(self, in_channels):
        super(AttentionGate, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, skip):
        attn = self.sigmoid(self.conv(x))
        return attn * skip