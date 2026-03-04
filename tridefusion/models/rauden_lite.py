from torch import nn
import torch
from .modules.mrb import MobileResidualBlock
from .modules.flash_att import FlashSelfAttention

from .base_model import BaseModel

class RAUDenLite(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Encoder
        self.enc1 = MobileResidualBlock(in_channels, 32)
        self.enc2 = MobileResidualBlock(32, 64)
        self.enc3 = MobileResidualBlock(64, 128)
        # Bottleneck
        self.attention = FlashSelfAttention(in_channels=128, reduction=8)
        # Decoder (lighter than teacher)
        self.dec1 = MobileResidualBlock(128, 64)
        self.dec2 = MobileResidualBlock(64, 32)
        # Aggregation (mini version)
        self.agg = nn.Conv2d(32 + 128, 32, kernel_size=1)
        # Output
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        # Bottleneck
        attention = self.attention(e3)
        # Decoder
        d1 = self.dec1(attention + e3)
        d2 = self.dec2(d1 + e2)
        # Aggregation (bring back enc1 + bottleneck)
        agg = torch.cat([d2, e1], dim=1)
        agg = self.agg(agg)
        # Output
        out = self.final_conv(agg)
        return out

if __name__ == "__main__":
    model = RAUDenLite(in_channels=1, out_channels=1)
    dummy = torch.randn(1, 1, 128, 128)
    out = model(dummy)
    print("Output shape:", out.shape, "| Params:", sum(p.numel() for p in model.parameters())/1e6, "M")