import torch
from torch import nn


class AttentionGate(nn.Module):
    def __init__(self, in_channels):
        super(AttentionGate, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, skip):
        attn = self.sigmoid(self.conv(x))
        return attn * skip