from torch import nn
import torch

from modules.erb import EfficientResidualBlock
from modules.ag import AttentionGate
from tridefusion.models.old.flash_attention import FlashSelfAttention

from .base_model import BaseModel

class RAUDen(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__() 
        # Encoder 
        self.enc1 = EfficientResidualBlock(in_channels, 64)
        self.enc2 = EfficientResidualBlock(64, 128, dilation=2)
        self.enc3 = EfficientResidualBlock(128, 256, dilation=4)
        # Bottleneck attention 
        self.attention = FlashSelfAttention(in_channels=256, reduction=4) 
        # Attention gates 
        self.attn1 = AttentionGate(256) 
        self.attn2 = AttentionGate(128) 
        self.attn2_conv = nn.Conv2d(64, 128, kernel_size=1, bias=False) 
        # Decoder 
        self.dec1 = EfficientResidualBlock(256+256, 128) # input = bottleneck + attn1_out 
        self.dec2 = EfficientResidualBlock(128+128, 64) # input = dec1 + attn2_out # Aggregation 
        self.agg1 = nn.Conv2d(64+256+64, 128, kernel_size=1) # dec2 + bottleneck + enc1 
        self.agg2 = nn.Conv2d(128, 64, kernel_size=1) 
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1) 
    def forward(self, x): # Encoder 
        enc1 = self.enc1(x) 
        enc2 = self.enc2(enc1) 
        enc3 = self.enc3(enc2) 
        # Bottleneck 
        bottleneck = self.attention(enc3) 
        # Attention gates 
        attn1_out = self.attn1(enc3, bottleneck) 
        dec1_input = torch.cat([bottleneck, attn1_out], dim=1) 
        dec1 = self.dec1(dec1_input) 
        attn2_out = self.attn2(enc2, dec1) 
        attn2_out = attn2_out[:, :64, :, :] # crop if needed 
        attn2_out = self.attn2_conv(attn2_out) 
        dec2_input = torch.cat([dec1, attn2_out], dim=1) 
        dec2 = self.dec2(dec2_input) # Multi-scale aggregation 
        aggregated = torch.cat([enc1, dec2, bottleneck], dim=1) 
        aggregated = self.agg1(aggregated) 
        aggregated = self.agg2(aggregated) # Output 
        out = self.final_conv(aggregated) 
        return out