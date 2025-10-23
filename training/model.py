import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    Two 3x3 convs with BN+ReLU and a skip connection.
    Followed by AvgPool2d(2) to reduce spatial size (per spec: no max pooling).
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        # Spec requires Average Pooling (not Max)
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.proj is not None:
            identity = self.proj(identity)
        out = F.relu(out + identity, inplace=True)
        out = self.pool(out)
        return out


class MalNetFocusAug(nn.Module):
    """
    MalNet-FocusAug (strict spec):
      - Stem: Conv(1->c1) + BN + ReLU
      - Residual stages (BasicBlock) with AvgPool2d after each: c1, c2, c3
      - Mandatory spatial attention on final feature map
      - GAP + Attention fusion -> MLP -> single logit (BCEWithLogits)
      - Input-size agnostic via AdaptiveAvgPool2d((1,1))
    """
    def __init__(self, channels=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = channels

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        # Residual stages
        self.stage1 = BasicBlock(c1, c1)  # (B, c1, H/2,  W/2)
        self.stage2 = BasicBlock(c1, c2)  # (B, c2, H/4,  W/4)
        self.stage3 = BasicBlock(c2, c3)  # (B, c3, H/8,  W/8)

        # Mandatory attention branch (tiny)
        self.attn_conv1 = nn.Conv2d(c3, 32, kernel_size=1)
        self.attn_conv2 = nn.Conv2d(32, 1,  kernel_size=1)

        # GAP (size-agnostic)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Fusion head (concat GAP + Attn) -> MLP -> 1 logit
        self.fc1 = nn.Linear(c3 * 2, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, 1, H, W)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)   # (B, C, H', W')
        B, C, H, W = x.shape

        # GAP path
        gap_vec = self.gap(x).view(B, C)  # (B, C)

        # Attention path: spatial softmax over H*W, then weighted sum of features
        a = F.relu(self.attn_conv1(x), inplace=True)   # (B, 32, H, W)
        a = self.attn_conv2(a).view(B, 1, H * W)       # (B, 1, HW)
        a = F.softmax(a, dim=-1)                       # (B, 1, HW) weights
        x_flat = x.view(B, C, H * W)                   # (B, C, HW)
        attn_vec = torch.bmm(x_flat, a.transpose(1, 2)).squeeze(-1)  # (B, C)

        # Fusion + MLP head
        fused = torch.cat([gap_vec, attn_vec], dim=1)  # (B, 2C)
        z = F.relu(self.fc1(fused), inplace=True)
        logit = self.fc2(z).squeeze(1)                 # (B,)

        return logit  # single logit; use with BCEWithLogitsLoss
