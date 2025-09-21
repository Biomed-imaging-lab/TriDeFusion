import torch
from torch import nn
import torch.nn.functional as F
from src.models.modules.residual import ResidualBlock
from src.models.modules.seblock import SEBlock
from torchmetrics.functional import structural_similarity_index_measure as ssim


class MultiScaleAttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleAttentionUNet, self).__init__()

        # Encoder
        self.enc1 = ResidualBlock(in_channels, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)

        # Bottleneck with deeper residual connections
        self.bottleneck1 = ResidualBlock(256, 256)
        self.bottleneck2 = ResidualBlock(256, 256)

        # Channel Attention in Bottleneck
        self.attention = SEBlock(256)

        # Decoder
        self.dec1 = ResidualBlock(256, 128)
        self.dec2 = ResidualBlock(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # Multi-Scale Feature Aggregation
        self.agg1 = nn.Conv2d(384, 128, kernel_size=1)  # Updated input channels
        self.agg2 = nn.Conv2d(128, 64, kernel_size=1)

    def forward(self, x):
        # Encoding
        enc1 = self.enc1(x)  # 64 channels
        enc2 = self.enc2(enc1)  # 128 channels
        enc3 = self.enc3(enc2)  # 256 channels

        # Bottleneck
        bottleneck = self.bottleneck1(enc3)
        bottleneck = self.bottleneck2(bottleneck)
        bottleneck = self.attention(bottleneck)
        

        # Decoding
        dec1 = self.dec1(bottleneck + enc3)  # Add skip connection from encoder
        dec2 = self.dec2(dec1 + enc2)

        # Multi-Scale Feature Aggregation
        aggregated = self.agg1(torch.cat([enc1, dec2, bottleneck], dim=1))  # Input now matches agg1
        aggregated = self.agg2(aggregated)

        out = self.final_conv(aggregated)
        return out


if __name__ == "__main__":
    model = MultiScaleAttentionUNet(in_channels=1, out_channels=1)
    dummy_input = torch.randn(1, 1, 128, 128)
    output = model(dummy_input)
    print("Output shape:", output.shape)
