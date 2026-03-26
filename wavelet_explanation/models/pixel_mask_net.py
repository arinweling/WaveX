"""
Pixel-space binary mask network — explanation without wavelets.

PixelMaskNet takes a full-resolution image (B, C, H, W) and outputs a
single-channel binary mask (B, 1, H, W) at the same spatial resolution.
The explanation is constructed by direct pixel-space masking: e = m * x.

Architecture mirrors UNet but the final head keeps full spatial resolution
(no avg_pool to subband size, no multi-subband output).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import STE, _conv_block, _EncoderBlock, _DecoderBlock


class PixelMaskNet(nn.Module):
    """
    Lightweight U-Net that produces a single full-resolution binary pixel mask.

    Input:  (B, C_in, H, W)
    Output:
        m_bin  — (B, 1, H, W) binary mask in {0, 1}  (via STE, for constructing e)
        m_cont — (B, 1, H, W) continuous mask in [0,1] (pre-STE, for L_bin / L_tv)

    The resolution-agnostic encoder-decoder requires H and W divisible by 16.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # Encoder (same channel progression as UNet)
        self.enc1 = _EncoderBlock(in_channels, 32)
        self.enc2 = _EncoderBlock(32, 64)
        self.enc3 = _EncoderBlock(64, 128)
        self.enc4 = _EncoderBlock(128, 256)

        # Bottleneck
        self.bottleneck = _conv_block(256, 256)

        # Decoder
        self.dec4 = _DecoderBlock(256, 256, 128)
        self.dec3 = _DecoderBlock(128, 128, 64)
        self.dec2 = _DecoderBlock(64, 64, 32)
        self.dec1 = _DecoderBlock(32, 32, 32)

        # Head: 1×1 conv → sigmoid at full spatial resolution (no avg_pool)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # Encoder
        p1, s1 = self.enc1(x)
        p2, s2 = self.enc2(p1)
        p3, s3 = self.enc3(p2)
        p4, s4 = self.enc4(p3)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder with skip connections
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        # Head
        m_cont = torch.sigmoid(self.final_conv(d1))  # (B, 1, H, W) in [0, 1]
        m_bin  = STE.apply(m_cont)                    # (B, 1, H, W) in {0, 1}

        return m_bin, m_cont
