"""
Haar Discrete Wavelet Transform (DWT) and Inverse DWT.

Implements one-level 2D Haar decomposition using separable 1D filters
applied via strided convolutions. No external wavelet library is required.
If pytorch_wavelets is installed, it can be used as a faster backend.

Haar filters:
    Low-pass:  h_low  = [1, 1] / sqrt(2)
    High-pass: h_high = [1, -1] / sqrt(2)

2D subbands (separable application with stride 2):
    LL = h_low(rows)  ⊗ h_low(cols)   — coarse / low-frequency structure
    LH = h_low(rows)  ⊗ h_high(cols)  — horizontal edges
    HL = h_high(rows) ⊗ h_low(cols)   — vertical edges
    HH = h_high(rows) ⊗ h_high(cols)  — fine texture / diagonal edges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT(nn.Module):
    """
    One-level 2D Haar Wavelet Transform.

    Inputs and outputs are in channel-first format: (B, C, H, W).
    Subbands have spatial size (B, C, H//2, W//2).

    Reconstruction is exact (up to floating-point precision):
        ||x - IDWT(DWT(x))||_inf < 1e-5 for typical float32 tensors.
    """

    def __init__(self):
        super().__init__()
        # Haar filter coefficients (not learnable).
        # The 2D filter is the outer product of two 1D filters:
        #   h_low  = [1, 1] / sqrt(2)    → 1D coefficient = 1/sqrt(2)
        #   h_high = [1,-1] / sqrt(2)
        # 2D element = h_row[i] * h_col[j] = (1/sqrt(2))^2 = 0.5
        s = 0.5
        # Each filter is stored as a (out_channels, in_channels, kH, kW) tensor.
        ll = torch.tensor([[s, s], [s, s]])        # low ⊗ low
        lh = torch.tensor([[s, -s], [s, -s]])      # low ⊗ high
        hl = torch.tensor([[s, s], [-s, -s]])      # high ⊗ low
        hh = torch.tensor([[s, -s], [-s, s]])      # high ⊗ high

        # Shape: (4, 1, 2, 2)
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filters", filters)

        # Inverse filters — for Haar, the synthesis bank equals the analysis bank
        self.register_buffer("inv_filters", filters.clone())

    # ------------------------------------------------------------------
    # Forward transform
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Decompose x into four wavelet subbands.

        Args:
            x: (B, C, H, W) float tensor. H and W must be even.

        Returns:
            x_LL: (B, C, H//2, W//2)
            x_LH: (B, C, H//2, W//2)
            x_HL: (B, C, H//2, W//2)
            x_HH: (B, C, H//2, W//2)
        """
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, (
            f"Input spatial dims must be even, got ({H}, {W})"
        )

        # Treat each channel independently using grouped convolution.
        # Reshape to (B*C, 1, H, W) so we can apply the same filter to every channel.
        x_flat = x.reshape(B * C, 1, H, W)

        # filters: (4, 1, 2, 2)  →  stride 2 convolution produces (B*C, 4, H//2, W//2)
        out = F.conv2d(x_flat, self.filters, stride=2, padding=0)  # (B*C, 4, H//2, W//2)

        out = out.reshape(B, C, 4, H // 2, W // 2)

        x_LL = out[:, :, 0]
        x_LH = out[:, :, 1]
        x_HL = out[:, :, 2]
        x_HH = out[:, :, 3]

        return x_LL, x_LH, x_HL, x_HH

    # ------------------------------------------------------------------
    # Inverse transform
    # ------------------------------------------------------------------

    def inverse(
        self,
        x_LL: torch.Tensor,
        x_LH: torch.Tensor,
        x_HL: torch.Tensor,
        x_HH: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct the full-resolution image from four wavelet subbands.

        Args:
            x_LL, x_LH, x_HL, x_HH: each (B, C, H//2, W//2)

        Returns:
            x_rec: (B, C, H, W)
        """
        B, C, Hs, Ws = x_LL.shape
        H, W = Hs * 2, Ws * 2

        # Stack subbands: (B, C, 4, H//2, W//2)
        subbands = torch.stack([x_LL, x_LH, x_HL, x_HH], dim=2)

        # Reshape for grouped transpose convolution: (B*C, 4, H//2, W//2)
        subbands_flat = subbands.reshape(B * C, 4, Hs, Ws)

        # Apply each synthesis filter to the corresponding subband separately,
        # then sum the contributions.
        # inv_filters: (4, 1, 2, 2)
        rec = torch.zeros(B * C, 1, H, W, device=x_LL.device, dtype=x_LL.dtype)
        for i in range(4):
            sub = subbands_flat[:, i : i + 1]          # (B*C, 1, Hs, Ws)
            filt = self.inv_filters[i : i + 1]          # (1, 1, 2, 2)
            rec += F.conv_transpose2d(sub, filt, stride=2, padding=0)

        rec = rec.reshape(B, C, H, W)
        return rec
