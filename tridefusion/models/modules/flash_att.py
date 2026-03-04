import torch
from torch import nn
import torch.nn.functional as F
from flash_attn import flash_attn_func

class FlashSelfAttention(nn.Module):
    def __init__(self, in_channels, reduction=4, dropout=0.0):
        super(FlashSelfAttention, self).__init__()
        self.q_proj = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.shape[-1] > 128:  # optional condition for skipping attention on large maps
            return x

        b, c, h, w = x.shape
        seq_len = h * w
        q = self.q_proj(x).flatten(2).transpose(1, 2)  # [B, HW, Cq]
        k = self.k_proj(x).flatten(2).transpose(1, 2)  # [B, HW, Ck]
        v = self.v_proj(x).flatten(2).transpose(1, 2)  # [B, HW, Cv]

        # Add sequence dimension for FlashAttention: [B, Seq, HeadDim] -> [B, Seq, H, D]
        # Here we assume 1 attention head for simplicity
        q = q.unsqueeze(2)
        k = k.unsqueeze(2)
        v = v.unsqueeze(2)

        # Ensure FP16 or BF16 for FlashAttention inside AMP context
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        # FlashAttention forward
        # flash_attn_func expects [B, Seq, H, D] in fp16 or bf16
        out = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            causal=False
        )  # [B, Seq, H, D]

        out = out.squeeze(2).transpose(1, 2).view(b, c, h, w)
        return self.gamma * out + x