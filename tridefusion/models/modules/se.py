import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SelfAttention, self).__init__()
        self.q = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if x.shape[-1] > 128:
            return x
        b, c, h, w = x.shape
        q = self.q(x).view(b, -1, h * w).permute(0, 2, 1)  
        k = self.k(x).view(b, -1, h * w)  
        v = self.v(x).view(b, -1, h * w)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)  
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)  
        return self.gamma * out + x