import torch 
from torch import nn 
import torch.nn.functional as F 
from flash_attn import flash_attn_func # core FlashAttention function 
from models.modules.eca import ECABlock 
from models.modules.ag import AttentionGate 
from models.modules.flash_att import FlashSelfAttention 
from models.modules.cbam import CBAM 


class EfficientResidualBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, dilation=1): 
        super().__init__() 
        mid_channels = max(in_channels, out_channels) 
        self.conv1 = nn.Sequential( 
            nn.Conv2d(
                in_channels, 
                mid_channels, 
                kernel_size=3, 
                padding=dilation, 
                groups=1, 
                dilation=dilation, 
                bias=False
            ), 
            nn.Conv2d(
                mid_channels, 
                out_channels, 
                kernel_size=1, 
                bias=False
            ), 
            nn.BatchNorm2d(
                out_channels
            ), nn.SiLU()
        ) 
        self.conv2 = nn.Sequential( 
            nn.Conv2d(out_channels, 
                      out_channels, 
                      kernel_size=3, 
                      padding=dilation, 
                      groups=1, 
                      dilation=dilation, 
                      bias=False
                ), 
                nn.Conv2d(
                    out_channels, 
                    out_channels, 
                    kernel_size=1, 
                    bias=False
                ), 
                nn.BatchNorm2d(out_channels), nn.SiLU() 
        ) 
        self.cbam = CBAM(out_channels) 
        self.skip = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            bias=False
            ) if in_channels != out_channels else nn.Identity() 
        
    def forward(self, x): 
        return self.cbam(self.conv2(self.conv1(x))) + self.skip(x)