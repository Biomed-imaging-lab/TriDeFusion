import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block to refine feature channels."""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.global_pool(x)  # Global average pooling
        scale = F.relu(self.fc1(scale))  # Reduce channels
        scale = self.sigmoid(self.fc2(scale))  # Restore channels
        return x * scale  # Scale the input feature
    
if __name__ == "__main__":
    se_block = SEBlock(channels=64, reduction=16)
    x = torch.randn(1, 64, 32, 32)
    print(se_block(x).shape)