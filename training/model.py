#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MalNet-FocusAug v2
- Input: (B,1,H,W), grayscale
- Stem: Conv(1→32) + BN + ReLU
- 3 residual stages (ResNet-style) with AvgPool2d(2) downsampling after each stage
  Stage1: 32 → 32
  Stage2: 32 → 64 (projection skip 1x1)
  Stage3: 64 → 128 (projection skip 1x1)
- Attention branch on final 128-C feature map:
    Conv 128→32, ReLU, Conv 32→1 → spatial softmax → attention weights A(H×W)
    Attention vector = Σ_{h,w} A(h,w) * F[:, :, h, w]  → shape (B, 128)
- GAP on final map → (B, 128)
- Fuse [Attention || GAP] → (B, 256) → MLP 256→128→1 (single logit; BCEWithLogitsLoss upstream)

Notes:
- Attention is ON by default. You can disable by MalNetFocusAug(attention=False).
- No dropout, no maxpool; all spatial downsampling uses AvgPool2d.
- Size-agnostic via AdaptiveAvgPool2d and global attention reduction.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ Blocks ------------------

class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """
    Two 3×3 convs with BN+ReLU, skip connection.
    If channels differ, uses 1×1 projection conv on the skip path.
    No spatial downsampling here; an AvgPool2d(2) is applied *after* each stage outside the block.
    """
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(c_out)

        self.proj = None
        if c_in != c_out:
            self.proj = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        out = self.relu(out)
        return out

# ------------------ Model ------------------

class MalNetFocusAug(nn.Module):
    def __init__(self, attention: bool = True):
        """
        attention=True by default per project spec. If False, the attention vector is zeroed
        and only GAP contributes (still produces a 256-dim fused vector for a stable head).
        """
        super().__init__()
        self.attention_on = bool(attention)

        # Stem
        self.stem = ConvBNReLU(1, 32, k=3, s=1, p=1)

        # Residual stages (with external AvgPool2d between stages)
        self.block1 = ResidualBlock(32, 32)   # out: (32, H,   W)
        self.pool1  = nn.AvgPool2d(kernel_size=2)  # -> (32, H/2, W/2)

        self.block2 = ResidualBlock(32, 64)   # projection 1x1 on skip path
        self.pool2  = nn.AvgPool2d(kernel_size=2)  # -> (64, H/4, W/4)

        self.block3 = ResidualBlock(64, 128)  # projection 1x1 on skip path
        self.pool3  = nn.AvgPool2d(kernel_size=2)  # -> (128, H/8, W/8)

        # Attention branch on final 128-C map
        self.att_conv1 = nn.Conv2d(128, 32, kernel_size=3, padding=1, bias=True)
        self.att_relu  = nn.ReLU(inplace=True)
        self.att_conv2 = nn.Conv2d(32, 1, kernel_size=1, padding=0, bias=True)

        # GAP head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # -> (128, 1, 1)

        # Fusion + MLP head
        self.head = nn.Sequential(
            nn.Linear(256, 128),  # [attention(128) || gap(128)]
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)     # single logit
        )

        self._init_weights()

    def _init_weights(self):
        # Kaiming for convs; small init for linear
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _spatial_softmax(self, att_logits: torch.Tensor) -> torch.Tensor:
        # att_logits: (B,1,H,W) -> softmax over H*W
        B, _, H, W = att_logits.shape
        a = att_logits.view(B, -1)
        a = torch.softmax(a, dim=1)
        return a.view(B, 1, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x: (B,1,H,W)
        x = self.stem(x)

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.pool3(x)   # final feature map F: (B,128,H/8,W/8)

        # GAP vector
        gap_vec = self.gap(x).flatten(1)  # (B,128)

        # Attention vector
        if self.attention_on:
            a = self.att_conv1(x)
            a = self.att_relu(a)
            a = self.att_conv2(a)               # (B,1,H/8,W/8)
            a = self._spatial_softmax(a)        # (B,1,H/8,W/8), sums to 1 over spatial
            att_vec = (x * a).sum(dim=(2,3))    # (B,128)
        else:
            # Disabled attention: use zeros so fusion stays size 256 consistently
            att_vec = torch.zeros_like(gap_vec)

        fused = torch.cat([gap_vec, att_vec], dim=1)  # (B,256)
        logit = self.head(fused).squeeze(1)           # (B,)
        return logit
