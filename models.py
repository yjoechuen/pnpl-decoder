"""Stronger single-trial MEG encoders; no audio, text or pretrained weights."""
import math

import torch
from torch import nn


class WideResidualBlock(nn.Module):
    """Expanded pointwise mixing and local dilated convolution, without attention."""

    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, width),
            nn.Conv1d(width, 2 * width, 1), nn.GELU(),
            nn.Conv1d(2 * width, 2 * width, 7, padding=3 * dilation,
                      dilation=dilation, groups=2 * width),
            nn.GroupNorm(8, 2 * width), nn.GELU(),
            nn.Conv1d(2 * width, width, 1), nn.Dropout(dropout))

    def forward(self, x):
        return x + 0.5 * self.net(x)


class TemporalSpatialCNN(nn.Module):
    """EEGNet-inspired temporal banks then bank-specific full-sensor spatial filters.

    A compact hypothesis for MEG; no claim of published competition performance.
    Shared temporal filters precede spatial mixing, unlike our baseline encoder.
    """

    def __init__(self, banks=32, depth=2, dropout=0.3, pooling_bins=8):
        super().__init__()
        if banks < 1 or depth < 1 or (banks*depth) % 8:
            raise ValueError('banks*depth must be a positive multiple of 8')
        width = banks*depth
        self.temporal = nn.Conv2d(1, banks, (1, 31), padding=(0,15), bias=False)
        self.spatial = nn.Sequential(nn.Conv2d(banks, width, (306,1), groups=banks, bias=False),
                                     nn.GroupNorm(8,width), nn.GELU())
        self.encoder = nn.Sequential(nn.AvgPool1d(2),
                                     WideResidualBlock(width,1,dropout), WideResidualBlock(width,2,dropout),
                                     WideResidualBlock(width,4,dropout))
        self.bins = pooling_bins
        if not 1 <= pooling_bins <=125:
            raise ValueError('Invalid pooling bins')
        self.head = nn.Sequential(nn.Flatten(),nn.Dropout(dropout),nn.Linear(width*pooling_bins,50))

    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(306,250):
            raise ValueError('Expected batch x 306 x 250')
        x=self.spatial(self.temporal(x[:,None])).squeeze(2)
        x=self.encoder(x)
        pooled=torch.stack([p.mean(-1) for p in torch.tensor_split(x,self.bins,dim=-1)],dim=-1)
        return self.head(pooled)


class MultiScaleSpatialCNN(nn.Module):
    """Same total bank count as spatial baseline; kernels 15/31/63, no context."""
    def __init__(self,dropout=.2,depth=2,pooling_bins=8):
        super().__init__()
        self.branches=nn.ModuleList([nn.Conv2d(1,b,(1,k),padding=(0,k//2),bias=False)
                                    for b,k in [(10,15),(12,31),(10,63)]])
        self.backbone=TemporalSpatialCNN(banks=32,depth=depth,dropout=dropout,pooling_bins=pooling_bins)
        self.backbone.temporal=nn.Identity()

    def forward(self,x):
        if x.shape[1:]!=(306,250):raise ValueError('Expected 306 x 250')
        temporal=torch.cat([branch(x[:,None]) for branch in self.branches],dim=1)
        x=self.backbone.encoder(self.backbone.spatial(temporal).squeeze(2))
        return self.backbone.head(torch.stack([p.mean(-1) for p in torch.tensor_split(x,self.backbone.bins,dim=-1)],dim=-1))


def dataset_channel_names(frame,data_dir):
    """Independently verify channel names/order across every indexed recording."""
    from pathlib import Path
    import h5py
    from decoder import reject_final_test_path
    expected=None
    for filename in frame.h5_file.unique():
        path=Path(filename)
        if not path.is_absolute():path=Path(data_dir)/path
        reject_final_test_path(path)
        with h5py.File(path,'r') as f:
            names=f.attrs.get('channel_names','')
            if isinstance(names,bytes):names=names.decode()
            names=[n.strip() for n in names.split(',')]
        if len(names)!=306 or len(set(names))!=306:
            raise ValueError('Missing or invalid H5 channel names')
        if expected is not None and names!=expected:
            raise ValueError('H5 channel order differs across recordings')
        expected=names
    if expected is None:raise ValueError('No recordings to verify')
    return expected


def load_sensor_geometry(path,expected):
    """Require coordinates and names in actual H5 order; no guessed sensor layout."""
    import json
    import numpy as np
    from pathlib import Path
    data=json.loads(Path(path).read_text())
    names=data['channel_names']
    xyz=np.asarray(data['positions'],dtype='float32')
    if names!=expected or len(names)!=306 or len(set(names))!=306 or xyz.shape!=(306,3) or not np.isfinite(xyz).all():
        raise ValueError('Geometry must match verified H5 channel order')
    if np.linalg.norm(xyz-xyz.mean(axis=0)) < 1e-6:
        raise ValueError('Degenerate sensor geometry')
    return xyz


class GeometryCNN(nn.Module):
    """Geometry-initialized spatial mixing + temporal CNN, BrainModule-inspired.

    Not an exact BrainModule reproduction. Geometry supplies a smooth initial
    attention prior; learned deviations can recover nonlocal sensor mixtures.
    """
    def __init__(self,width=64,dropout=.2):
        super().__init__()
        if width%8:raise ValueError('width must be divisible by 8')
        self.register_buffer('spatial_prior',torch.zeros(width,306))
        self.delta=nn.Parameter(torch.zeros(width,306))
        self.encoder=nn.Sequential(nn.AvgPool1d(2),*[WideResidualBlock(width,d,dropout) for d in [1,2,4,8]])
        self.attention=nn.Conv1d(width,1,1)
        self.head=nn.Linear(width*5,50)

    def initialize_geometry(self,xyz):
        xyz=torch.as_tensor(xyz,dtype=torch.float32,device=self.spatial_prior.device)
        scale=(xyz-xyz.mean(0)).square().sum(-1).mean().sqrt().clamp_min(1e-6)
        anchors=xyz[torch.linspace(0,305,self.spatial_prior.shape[0]).long()]
        self.spatial_prior.copy_(-torch.cdist(anchors,xyz).square()/(2*(.4*scale)**2))

    def forward(self,x):
        if x.shape[1:]!=(306,250):raise ValueError('Expected 306 x 250')
        x=torch.einsum('wc,bct->bwt',(self.spatial_prior+self.delta).softmax(-1),x)
        x=self.encoder(x)
        attended=(self.attention(x).softmax(-1)*x).sum(-1)
        bins=torch.cat([p.mean(-1) for p in torch.tensor_split(x,4,dim=-1)],dim=-1)
        return self.head(torch.cat([attended,bins],dim=1))


class WideResidualCNN(nn.Module):
    """Wider baseline-like sensor/temporal CNN retaining onset-relative latency.

    Independent single-trial inference. No attention, cross-trial pooling,
    pretrained features, or adaptive-pooling CUDA backward required.
    """

    def __init__(self, width=128, dropout=0.2, dilations=(1, 2, 4, 8, 1, 2), pooling_bins=8):
        super().__init__()
        if width < 8 or width % 8:
            raise ValueError("width must be a positive multiple of 8")
        if not dilations or any(not isinstance(d, int) or d < 1 for d in dilations):
            raise ValueError("dilations must be positive integers")
        if not isinstance(pooling_bins, int) or not 1 <= pooling_bins <= 125:
            raise ValueError("pooling_bins must be between 1 and 125")
        self.pooling_bins = pooling_bins
        self.encoder = nn.Sequential(
            nn.Conv1d(306, width, 1), nn.GroupNorm(8, width), nn.GELU(), nn.AvgPool1d(2),
            *(WideResidualBlock(width, d, dropout) for d in dilations),
            nn.GroupNorm(8, width), nn.GELU())
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(width * pooling_bins, 50))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (306, 250):
            raise ValueError("Expected batch x 306 channels x 250 samples")
        x = self.encoder(x)
        pooled = torch.stack([part.mean(dim=-1) for part in torch.tensor_split(x, self.pooling_bins, dim=-1)], dim=-1)
        return self.head(pooled)


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
