#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MalNet-FocusAug v3
- Input: (B,1,H,W), grayscale
- Stem: Conv(1→32) + BN + Mish
- 4 residual stages (ResNet-style) with AvgPool2d(2) downsampling after each stage
  Stage1: 32 → 32 (kernel 3×3)
  Stage2: 32 → 64 (kernel 5×5, projection skip 1x1)
  Stage3: 64 → 128 (kernel 3×3, projection skip 1x1)
  Stage4: 128 → 256 (kernel 7×7, projection skip 1x1)
- Each residual block includes SE (Squeeze-and-Excitation) for channel recalibration
- Attention branch on final 256-C feature map:
    Conv 256→32, Mish, Conv 32→1 → spatial softmax → attention weights A(H×W)
    Attention vector = Σ_{h,w} A(h,w) * F[:, :, h, w]  → shape (B, 256)
- GAP on final map → (B, 256)
- Fuse [Attention || GAP] → (B, 512) → MLP 512→128→1 (single logit; BCEWithLogitsLoss upstream)

Notes:
- Attention is ON by default. You can disable by MalNetFocusAug(attention=False).
- No dropout, no maxpool; all spatial downsampling uses AvgPool2d.
- Size-agnostic via AdaptiveAvgPool2d and global attention reduction.
- Uses Mish activation throughout (stem, residual blocks, attention, MLP head).
- SE blocks positioned after 2nd conv+BN, before skip connection.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ Activation ------------------

class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# ------------------ Blocks ------------------

class ConvBNMish(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)
        self.mish = Mish()

    def forward(self, x):
        return self.mish(self.bn(self.conv(x)))

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel recalibration.
    GAP → FC(channels → channels//16) → ReLU → FC(channels//16 → channels) → Sigmoid → scale
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """
    Two convs with BN+Mish, SE block, skip connection.
    Kernel size is configurable (3, 5, or 7).
    If channels differ, uses 1×1 projection conv on the skip path.
    No spatial downsampling here; an AvgPool2d(2) is applied *after* each stage outside the block.
    SE block is positioned after 2nd conv+BN, before skip connection.
    """
    def __init__(self, c_in, c_out, kernel_size=3):
        super().__init__()
        # Compute padding to maintain spatial dimensions
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn1   = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2   = nn.BatchNorm2d(c_out)
        
        # SE block after 2nd conv+BN, before skip connection
        self.se = SEBlock(c_out)

        # Projection skip if channels differ
        self.proj = None
        if c_in != c_out:
            self.proj = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False)

        self.mish = Mish()

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mish(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply SE block after 2nd conv+BN, before skip connection
        out = self.se(out)

        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        out = self.mish(out)
        return out

# ------------------ Model ------------------

class MalNetFocusAug(nn.Module):
    def __init__(self, attention: bool = True):
        """
        attention=True by default per project spec. If False, the attention vector is zeroed
        and only GAP contributes (still produces a 512-dim fused vector for a stable head).
        """
        super().__init__()
        self.attention_on = bool(attention)

        # Stem
        self.stem = ConvBNMish(1, 32, k=3, s=1, p=1)

        # Residual stages (with external AvgPool2d between stages)
        # Block 1: 32 → 32, kernel 3×3
        self.block1 = ResidualBlock(32, 32, kernel_size=3)   # out: (32, H,   W)
        self.pool1  = nn.AvgPool2d(kernel_size=2)  # -> (32, H/2, W/2)

        # Block 2: 32 → 64, kernel 5×5
        self.block2 = ResidualBlock(32, 64, kernel_size=5)   # projection 1x1 on skip path
        self.pool2  = nn.AvgPool2d(kernel_size=2)  # -> (64, H/4, W/4)

        # Block 3: 64 → 128, kernel 3×3
        self.block3 = ResidualBlock(64, 128, kernel_size=3)  # projection 1x1 on skip path
        self.pool3  = nn.AvgPool2d(kernel_size=2)  # -> (128, H/8, W/8)

        # Block 4: 128 → 256, kernel 7×7
        self.block4 = ResidualBlock(128, 256, kernel_size=7)  # projection 1x1 on skip path
        self.pool4  = nn.AvgPool2d(kernel_size=2)  # -> (256, H/16, W/16)

        # Attention branch on final 256-C map
        self.att_conv1 = nn.Conv2d(256, 32, kernel_size=3, padding=1, bias=True)
        self.att_mish  = Mish()
        self.att_conv2 = nn.Conv2d(32, 1, kernel_size=1, padding=0, bias=True)

        # GAP head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # -> (256, 1, 1)

        # Fusion + MLP head
        self.head = nn.Sequential(
            nn.Linear(512, 128),  # [attention(256) || gap(256)]
            Mish(),
            nn.Linear(128, 1)     # single logit
        )

        self._init_weights()

    def _init_weights(self):
        # Kaiming for convs; xavier for linear
        # Note: Using "relu" mode for Mish (slight approximation, but standard practice)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
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
        x = self.pool3(x)

        x = self.block4(x)
        x = self.pool4(x)   # final feature map F: (B,256,H/16,W/16)

        # GAP vector
        gap_vec = self.gap(x).flatten(1)  # (B,256)

        # Attention vector
        if self.attention_on:
            a = self.att_conv1(x)
            a = self.att_mish(a)
            a = self.att_conv2(a)               # (B,1,H/16,W/16)
            a = self._spatial_softmax(a)        # (B,1,H/16,W/16), sums to 1 over spatial
            att_vec = (x * a).sum(dim=(2,3))    # (B,256)
        else:
            # Disabled attention: use zeros so fusion stays size 512 consistently
            att_vec = torch.zeros_like(gap_vec)

        fused = torch.cat([gap_vec, att_vec], dim=1)  # (B,512)
        logit = self.head(fused).squeeze(1)           # (B,)
        return logit
