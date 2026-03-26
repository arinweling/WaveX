"""
U-Net encoder-decoder for wavelet-domain mask generation.

The network takes a full-resolution image (B, C, H, W) and outputs four
per-subband binary masks (B, 1, H//2, W//2) — one for each Haar subband
{LL, LH, HL, HH}.  Subbands not in `active_subbands` are permanently
clamped to zero and contribute no learnable parameters in the output head.

Key design choices:
- 4 downsampling encoder blocks with skip connections
- Bottleneck at 256 channels
- 4 upsampling decoder blocks mirroring the encoder
- Final head outputs len(active_subbands) channels → avg-pooled → STE binarised
- STE (Straight-Through Estimator) allows gradients to flow through binarisation

Architecture (channel progression):
    Encoder:  3 → 32 → 64 → 128 → 256
    Bottleneck: 256 → 256
    Decoder:  256+256 → 128 → 128+128 → 64 → 64+64 → 32 → 32+32 → 32
    Head:     32 → len(active_subbands)  (sigmoid → avg_pool2d → STE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


_SUBBAND_ORDER = ["LL", "LH", "HL", "HH"]


# ---------------------------------------------------------------------------
# Straight-Through Estimator
# ---------------------------------------------------------------------------

class STE(torch.autograd.Function):
    """
    Binarises a tensor at threshold 0.5 in the forward pass.
    Passes gradients through unchanged in the backward pass.

    Usage:
        binary = STE.apply(continuous_mask)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return (x > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two Conv-IN-ReLU layers without spatial downsampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.ReLU(inplace=True),
    )


class _EncoderBlock(nn.Module):
    """One encoder stage: double conv then max-pool (returns features before pool too)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = _conv_block(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        features = self.conv(x)   # used for skip connection
        pooled = self.pool(features)
        return pooled, features


class _DecoderBlock(nn.Module):
    """One decoder stage: upsample (transposed conv) + skip concat + double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _conv_block(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle any off-by-one from odd spatial dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Lightweight U-Net that produces wavelet-domain binary masks.

    Input:  (B, C_in, H, W)  — typically C_in=3 for RGB, C_in=1 for grayscale
    Output: four tensors, each (B, 1, H//2, W//2), values in {0, 1} after STE.
            Subbands not in `active_subbands` are returned as all-zero tensors
            and have no corresponding learnable parameters in the output head.

    The network is resolution-agnostic as long as H and W are divisible by 16
    (4 pooling layers of stride 2).

    Args:
        in_channels:     number of input image channels (3 for RGB, 1 for grayscale)
        active_subbands: which subbands to learn masks for; subset of
                         ['LL', 'LH', 'HL', 'HH'].  Defaults to all four.
    """

    def __init__(self, in_channels: int = 3, active_subbands: List[str] = None):
        super().__init__()

        if active_subbands is None:
            active_subbands = list(_SUBBAND_ORDER)
        # Preserve canonical order regardless of how caller passed the list
        self.active_subbands = [s for s in _SUBBAND_ORDER if s in active_subbands]
        n_active = len(self.active_subbands)
        if n_active == 0:
            raise ValueError("active_subbands must contain at least one subband.")

        # Encoder
        self.enc1 = _EncoderBlock(in_channels, 32)
        self.enc2 = _EncoderBlock(32, 64)
        self.enc3 = _EncoderBlock(64, 128)
        self.enc4 = _EncoderBlock(128, 256)

        # Bottleneck
        self.bottleneck = _conv_block(256, 256)

        # Decoder  (in_ch, skip_ch, out_ch)
        self.dec4 = _DecoderBlock(256, 256, 128)
        self.dec3 = _DecoderBlock(128, 128, 64)
        self.dec2 = _DecoderBlock(64, 64, 32)
        self.dec1 = _DecoderBlock(32, 32, 32)

        # Output head: one channel per active subband
        self.final_conv = nn.Conv2d(32, n_active, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Use small-weight init for the final conv to avoid mask collapse early in training."""
        nn.init.xavier_uniform_(self.final_conv.weight, gain=0.1)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W) image tensor

        Returns:
            m_LL, m_LH, m_HL, m_HH — each (B, 1, H//2, W//2), values in {0, 1}.
                Disabled subband masks are all-zero tensors (no gradient).
            continuous_masks — tuple (m_LL_cont, m_LH_cont, m_HL_cont, m_HH_cont),
                pre-STE continuous values; disabled subbands are all-zero tensors.
        """
        B, _, H, W = x.shape
        subband_H, subband_W = H // 2, W // 2

        # Encoder
        x1, skip1 = self.enc1(x)    # skip1: (B, 32,  H,    W)
        x2, skip2 = self.enc2(x1)   # skip2: (B, 64,  H/2,  W/2)
        x3, skip3 = self.enc3(x2)   # skip3: (B, 128, H/4,  W/4)
        x4, skip4 = self.enc4(x3)   # skip4: (B, 256, H/8,  W/8)

        # Bottleneck
        b = self.bottleneck(x4)     # (B, 256, H/16, W/16)

        # Decoder
        d4 = self.dec4(b,  skip4)   # (B, 128, H/8,  W/8)
        d3 = self.dec3(d4, skip3)   # (B, 64,  H/4,  W/4)
        d2 = self.dec2(d3, skip2)   # (B, 32,  H/2,  W/2)
        d1 = self.dec1(d2, skip1)   # (B, 32,  H,    W)

        # Head: continuous masks for active subbands only
        logits = self.final_conv(d1)                        # (B, n_active, H, W)
        continuous = torch.sigmoid(logits)
        continuous_down = F.avg_pool2d(continuous, kernel_size=2, stride=2)
        # continuous_down: (B, n_active, H//2, W//2)

        # Reconstruct full 4-subband output; disabled subbands → zero tensors
        zeros = torch.zeros(B, 1, subband_H, subband_W, device=x.device)
        masks_bin  = {}
        masks_cont = {}
        active_idx = 0
        for name in _SUBBAND_ORDER:
            if name in self.active_subbands:
                cont = continuous_down[:, active_idx:active_idx + 1]
                masks_bin[name]  = STE.apply(cont)
                masks_cont[name] = cont
                active_idx += 1
            else:
                masks_bin[name]  = zeros
                masks_cont[name] = zeros

        continuous_masks = (
            masks_cont["LL"], masks_cont["LH"],
            masks_cont["HL"], masks_cont["HH"],
        )
        return (
            masks_bin["LL"], masks_bin["LH"],
            masks_bin["HL"], masks_bin["HH"],
            continuous_masks,
        )
