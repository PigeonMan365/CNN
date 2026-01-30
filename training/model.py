#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline CNN for malware detection.

Architecture:
- Input: (B, 1, S, S) grayscale images, where S is divisible by 8
- Stem: Conv2d(1 → S//8, 3×3) → BN → ReLU
- 3 Residual Blocks with channel progression: S//8 → S//8 → S//4 → S//2
- Each block: 2× Conv(3×3) → BN → ReLU, with skip connection
- Downsampling: AvgPool2d(2) after each block
- Head: GAP → Linear(C → C//2) → ReLU → Linear(C//2 → 1)

Design principles:
- Simple, proven components only (ReLU, BatchNorm, standard convolutions)
- Channel counts scale with input size S
- No attention, SE blocks, or advanced mechanisms
- Compatible with existing training, export, and resume infrastructure
"""

from __future__ import annotations
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Simple residual block with two 3×3 convolutions.
    - Conv(3×3) → BN → ReLU
    - Conv(3×3) → BN
    - Skip connection (identity or 1×1 projection if channels differ)
    - ReLU
    No spatial downsampling (handled externally).
    """
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        
        # Projection skip if channels differ
        self.proj = None
        if c_in != c_out:
            self.proj = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class MalNetFocusAug(nn.Module):
    """
    Baseline CNN for malware detection.
    
    Channel scaling rule (derived from image size S):
    - Stem: S // 8
    - Block 1: S // 8
    - Block 2: S // 4
    - Block 3: S // 2
    
    Example for S=256: 32 → 32 → 64 → 128
    
    Args:
        attention: Ignored (kept for API compatibility). No attention is used.
    """
    def __init__(self, attention: bool = True):
        super().__init__()
        # Default to S=256 for initialization (channels: 32, 32, 64, 128)
        # Model will work with any S divisible by 8
        default_s = 256
        c_stem = default_s // 8
        c1 = default_s // 8
        c2 = default_s // 4
        c3 = default_s // 2
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, c_stem, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_stem),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks with downsampling
        self.block1 = ResidualBlock(c_stem, c1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.block2 = ResidualBlock(c1, c2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.block3 = ResidualBlock(c2, c3)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Head: GAP → Linear(C → C//2) → ReLU → Linear(C//2 → 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(c3, c3 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(c3 // 2, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights: Kaiming for convs, Xavier for linear, standard for BN."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 1, S, S) where S is divisible by 8.
        
        Returns:
            Logits of shape (B,) for binary classification.
        """
        x = self.stem(x)
        
        x = self.block1(x)
        x = self.pool1(x)
        
        x = self.block2(x)
        x = self.pool2(x)
        
        x = self.block3(x)
        x = self.pool3(x)
        
        # Global average pooling and classification
        x = self.gap(x).flatten(1)  # (B, C_final)
        logit = self.head(x).squeeze(1)  # (B,)
        return logit
