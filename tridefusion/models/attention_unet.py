import torch
import torch.nn as nn
import torch.nn.functional as F

from tridefusion.models.modules.old.attention import AttentionBlock
from modules.residual import ResidualBlock



class AttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUNet, self).__init__()

        # Encoder
        self.enc1 = ResidualBlock(in_channels, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)

        # Bottleneck Attention
        self.attention = AttentionBlock(256)

        # Decoder
        self.dec1 = ResidualBlock(256, 128)
        self.dec2 = ResidualBlock(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        # Attention
        bottleneck = self.attention(enc3)

        # Decoding
        dec1 = self.dec1(bottleneck + enc3)
        dec2 = self.dec2(dec1 + enc2)
        out = self.final_conv(dec2 + enc1)
        return out


if __name__ == "__main__":
    model = AttentionUNet(in_channels=1, out_channels=1)
    dummy_input = torch.randn(1, 1, 128, 128)
    output = model(dummy_input)
    print("Output shape:", output.shape)
