import torch
from torch import nn
from .flash_att import FlashSelfAttention
from models.modules.eca import ECABlock 
from models.modules.eca import ECABlock # Lightweight Residual Block (Depthwise + Pointwise + ECA) 
class MobileResidualBlock(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__() 
        self.conv = nn.Sequential( 
            nn.Conv2d(
                in_channels, 
                in_channels, 
                kernel_size=3, 
                padding=1, 
                groups=in_channels, 
                bias=False
            ), 
            # depthwise 
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), # pointwise 
            nn.BatchNorm2d(out_channels), nn.SiLU() 
        ) 
        self.eca = ECABlock(out_channels) 
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity() 
    def forward(self, x): 
        return self.eca(self.conv(x)) + self.skip(x)