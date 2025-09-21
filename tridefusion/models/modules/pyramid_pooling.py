import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    """Pyramid Pooling Module for Multi-Scale Context Extraction."""
    def __init__(self, in_channels, out_channels):
        super(PyramidPoolingModule, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.final_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        p1 = F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode="bilinear", align_corners=True)
        p2 = F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode="bilinear", align_corners=True)
        p3 = F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode="bilinear", align_corners=True)
        p4 = F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode="bilinear", align_corners=True)
        return self.final_conv(torch.cat([x, p1, p2, p3, p4], dim=1))
