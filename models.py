"""Stronger single-trial MEG encoders; no audio, text or pretrained weights."""
import math

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * width, width),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class ConvolutionModule(nn.Module):
    """Conformer temporal convolution, using GroupNorm for small trial batches."""

    def __init__(self, width, kernel_size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(
            nn.Conv1d(width, 2 * width, 1), nn.GLU(dim=1),
            nn.Conv1d(width, width, kernel_size, padding=kernel_size // 2, groups=width),
            nn.GroupNorm(8, width), nn.SiLU(), nn.Conv1d(width, width, 1), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(self.norm(x).transpose(1, 2)).transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, width, heads, kernel_size, dropout):
        super().__init__()
        self.ff1 = FeedForward(width, dropout)
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.conv = ConvolutionModule(width, kernel_size, dropout)
        self.ff2 = FeedForward(width, dropout)
        self.final_norm = nn.LayerNorm(width)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        q = self.attention_norm(x)
        attended, _ = self.attention(q, q, q, need_weights=False)
        x = x + self.attention_dropout(attended)
        x = x + self.conv(x)
        return self.final_norm(x + 0.5 * self.ff2(x))


class MultiScaleConformer(nn.Module):
    """Sensor projection → multi-scale temporal CNN → Conformer → latency-aware head.

    Input is one (306, 250) normalized MEG trial. All attention is within that
    trial: no holdout context, label grouping, or cross-trial pooling is used.
    """

    def __init__(self, width=128, dropout=0.2, layers=4, heads=4, kernel_size=15):
        super().__init__()
        if width < 8 or width % 8 or heads < 1 or width % heads:
            raise ValueError("width must be divisible by 8 and by positive heads")
        if layers < 1 or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("layers must be positive; kernel_size must be positive and odd")
        self.projection = nn.Sequential(nn.Conv1d(306, width, 1), nn.GroupNorm(8, width), nn.GELU())
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(width, width, kernel, padding=kernel // 2, groups=width),
                          nn.GELU()) for kernel in (5, 15, 31)
        ])
        self.frontend = nn.Sequential(nn.Conv1d(3 * width, width, 1), nn.GroupNorm(8, width),
                                      nn.GELU(), nn.Conv1d(width, width, 5, stride=4, padding=2),
                                      nn.GroupNorm(8, width), nn.GELU(), nn.Dropout(dropout))
        # Stride-4 convolution maps 250 samples to 63 onset-relative tokens.
        positions = torch.arange(63, dtype=torch.float32)[:, None]
        frequencies = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        pe = torch.zeros(1, 63, width)
        pe[0, :, 0::2] = torch.sin(positions * frequencies)
        pe[0, :, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("position_encoding", pe)
        self.blocks = nn.Sequential(*[
            ConformerBlock(width, heads, kernel_size, dropout) for _ in range(layers)
        ])
        self.pool_scores = nn.Linear(width, 1)
        self.head = nn.Sequential(nn.LayerNorm(5 * width), nn.Dropout(dropout),
                                  nn.Linear(5 * width, width), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(width, 50))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (306, 250):
            raise ValueError("Expected batch x 306 channels x 250 samples")
        x = self.projection(x)
        x = self.frontend(torch.cat([branch(x) for branch in self.branches], dim=1))
        tokens = x.transpose(1, 2) + self.position_encoding
        tokens = self.blocks(tokens)
        weights = self.pool_scores(tokens).softmax(dim=1)
        attended = (weights * tokens).sum(dim=1)
        # Fixed bins preserve onset-relative latency. Plain means avoid CUDA's
        # unsupported deterministic adaptive-pooling backward operation here.
        bins = torch.cat([part.mean(dim=1) for part in torch.tensor_split(tokens, 4, dim=1)], dim=1)
        return self.head(torch.cat([attended, bins], dim=1))
