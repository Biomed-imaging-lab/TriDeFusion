import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, dropout_prob=0.3):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        attention_weights = self.sigmoid(self.conv(x))
        attention_weights = self.dropout(attention_weights)
        return x * attention_weights
    
if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    model = AttentionBlock(3)
    print(model(x).shape)