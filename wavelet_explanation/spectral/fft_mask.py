"""
Radial Spectral Mask Network.

Instead of decomposing the image into a fixed set of wavelet subbands and
learning one binary mask per subband, this module learns a **radial frequency
profile** directly in the 2D DFT domain.

Architecture
------------
x (B, C, H, W)
  → lightweight CNN encoder  (B, 128)
  → FC head                  (B, num_radial_bins)    ← 1-D frequency profile
  → linear interpolation to rfft2 grid  (B, 1, H, W//2+1)
  → sigmoid + STE binarisation

The mask is radially symmetric: every frequency component (u, v) in the rfft2
output is assigned a weight that depends only on its magnitude
  r = sqrt((u/H)² + (v/W)²)
normalised to [0, 1] over the frequency grid.  This makes it easy to visualise
and interpret: plotting the 1-D profile reveals whether the classifier relies on
low-frequency shape information or high-frequency texture cues.

Gradient flow
-------------
STE is applied identically to `models/unet.py`: binarise at 0.5 in the forward
pass, pass gradients through unchanged in the backward pass.

Output
------
forward() returns:
  M_cont : (B, 1, H, W//2+1)  continuous mask in [0, 1]  (used for losses)
  M_bin  : (B, 1, H, W//2+1)  binary mask in {0, 1}       (used to mask FFT)
  radial_profile : (B, num_radial_bins)  raw logits per bin  (for visualisation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Straight-Through Estimator  (same as models/unet.py — duplicated here so
# the spectral package has no import dependency on the wavelet package)
# ---------------------------------------------------------------------------

class STE(torch.autograd.Function):
    """Binarise at 0.5 (forward), pass gradients unchanged (backward)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return (x > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


# ---------------------------------------------------------------------------
# Frequency grid helper
# ---------------------------------------------------------------------------

def make_radial_freq_grid(H: int, W: int) -> torch.Tensor:
    """
    Build a (H, W//2+1) tensor of normalised radial frequency magnitudes,
    matching the layout of torch.fft.rfft2(x) for input shape (*, H, W).

    Frequency index mapping (following numpy/torch FFT conventions):
      - Row axis (H):  index k maps to frequency k/H if k <= H//2,
                       else (k - H)/H  → range [-0.5, 0.5]
      - Column axis (W//2+1, rfft): index k maps to k/W  → range [0, 0.5]

    The returned value is r / r_max, so values lie in [0, 1] with 0 at DC
    (top-left of rfft output) and 1 at the highest represented frequency.

    Args:
        H, W: spatial dimensions of the *input* image tensor.

    Returns:
        freq_grid: (H, W//2+1) float tensor in [0, 1].
    """
    # Row frequencies: centre-adjusted
    u = torch.arange(H, dtype=torch.float32)
    u = torch.where(u <= H // 2, u / H, (u - H) / H)  # in [-0.5, 0.5]

    # Column frequencies: rfft keeps only non-negative half
    v = torch.arange(W // 2 + 1, dtype=torch.float32) / W  # in [0, 0.5]

    # 2-D outer-product grid
    uu, vv = torch.meshgrid(u, v, indexing="ij")  # (H, W//2+1) each
    r = torch.sqrt(uu ** 2 + vv ** 2)

    r_max = r.amax().clamp(min=1e-8)
    return r / r_max  # (H, W//2+1)  in [0, 1]


# ---------------------------------------------------------------------------
# Radial Spectral Mask Network
# ---------------------------------------------------------------------------

class RadialSpectralMaskNet(nn.Module):
    """
    Per-image radial frequency mask predictor.

    A compact CNN encoder maps the input image to a 1-D vector of
    `num_radial_bins` logits.  These are interpolated to the full rfft2
    frequency grid (H, W//2+1) and converted to a continuous mask via sigmoid.

    The mask is radially symmetric by construction — M[u,v] = f(r(u,v)) —
    which gives a compact, human-interpretable characterisation of which
    frequency band the classifier relies on.

    Args:
        in_channels:      number of input image channels (1 for grayscale, 3 for RGB)
        H, W:             spatial dimensions of the input image
        num_radial_bins:  number of 1-D frequency bins (default 64)
    """

    def __init__(self, in_channels: int, H: int, W: int, num_radial_bins: int = 64):
        super().__init__()

        self.H = H
        self.W = W
        self.num_radial_bins = num_radial_bins

        # ------------------------------------------------------------------
        # Lightweight CNN encoder → global feature vector
        # 3 conv blocks with stride-2 downsampling + global average pool
        # ------------------------------------------------------------------
        self.encoder = nn.Sequential(
            # Block 1: H → H/2
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            # Block 2: H/2 → H/4
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            # Block 3: H/4 → H/8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            # Global average pool → (B, 128, 1, 1)
            nn.AdaptiveAvgPool2d(1),
        )

        # ------------------------------------------------------------------
        # FC head: 128-dim feature → num_radial_bins logits
        # ------------------------------------------------------------------
        self.fc = nn.Sequential(
            nn.Flatten(),                     # (B, 128)
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_radial_bins),  # (B, num_radial_bins)  raw logits
        )

        # ------------------------------------------------------------------
        # Pre-computed frequency grid (not a learnable parameter)
        # ------------------------------------------------------------------
        freq_grid = make_radial_freq_grid(H, W)  # (H, W//2+1)
        self.register_buffer("freq_grid", freq_grid)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        """Small-gain Xavier init to avoid early mask collapse."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W) image tensor

        Returns:
            M_cont:          (B, 1, H, W//2+1)  continuous mask in [0, 1]
            M_bin:           (B, 1, H, W//2+1)  binary mask in {0, 1} via STE
            radial_profile:  (B, num_radial_bins)  sigmoid of per-bin logits
                             (useful for visualising the learned frequency curve)
        """
        B = x.shape[0]

        # --- encode image to per-bin logits ---
        feats = self.encoder(x)           # (B, 128, 1, 1)
        bin_logits = self.fc(feats)       # (B, num_radial_bins)

        # --- interpolate 1-D profile to 2-D freq grid ---
        # freq_grid: (H, W//2+1) in [0, 1]; map to bin index float
        Hf, Wf = self.freq_grid.shape     # H, W//2+1
        idx_float = self.freq_grid * (self.num_radial_bins - 1)  # (Hf, Wf)

        idx_lo = idx_float.long().clamp(0, self.num_radial_bins - 2)  # (Hf, Wf)
        idx_hi = (idx_lo + 1).clamp(0, self.num_radial_bins - 1)
        frac   = idx_float - idx_lo.float()                           # (Hf, Wf)

        # flatten spatial dims to use as gather indices
        lo_flat   = idx_lo.reshape(-1)   # (Hf*Wf,)
        hi_flat   = idx_hi.reshape(-1)   # (Hf*Wf,)
        frac_flat = frac.reshape(-1)     # (Hf*Wf,)

        # gather per-batch item: (B, num_bins) → (B, Hf*Wf)
        w_lo = bin_logits[:, lo_flat]    # (B, Hf*Wf)
        w_hi = bin_logits[:, hi_flat]    # (B, Hf*Wf)

        # linear interpolation between adjacent bins
        M_logits = w_lo * (1.0 - frac_flat) + w_hi * frac_flat  # (B, Hf*Wf)
        M_logits = M_logits.reshape(B, 1, Hf, Wf)               # (B, 1, Hf, Wf)

        M_cont = torch.sigmoid(M_logits)        # (B, 1, H, W//2+1)
        M_bin  = STE.apply(M_cont)              # (B, 1, H, W//2+1) in {0, 1}

        radial_profile = torch.sigmoid(bin_logits)  # (B, num_radial_bins)

        return M_cont, M_bin, radial_profile


# ---------------------------------------------------------------------------
# UNet Spectral Mask Network
# ---------------------------------------------------------------------------

class UNetSpectralMaskNet(nn.Module):
    """
    Generates a full 2D asymmetrical frequency mask using a full U-Net.
    Operates on the spatial image but output is interpreted as frequency weights.
    Output is cropped from (H, W) to (H, W//2+1) to match rfft2 frequency bins.
    """
    def __init__(self, in_channels: int, H: int, W: int):
        super().__init__()
        # Import dynamically to avoid circular dependencies if any
        from models.pixel_mask_net import PixelMaskNet
        self.unet = PixelMaskNet(in_channels=in_channels)
        
        # Precomputed frequency grid (used by loss functions in the trainer)
        freq_grid = make_radial_freq_grid(H, W)
        self.register_buffer("freq_grid", freq_grid)
        
    def forward(self, x: torch.Tensor):
        # PixelMaskNet returns m_bin, m_cont. We use the continuous one.
        _, m_cont_full = self.unet(x) # (B, 1, H, W)
        
        # Retain only the frequency components needed by rfft2
        W_half = x.shape[-1] // 2 + 1
        M_cont = m_cont_full[:, :, :, :W_half]
        M_bin = STE.apply(M_cont)
        
        # No 1D radial profile exists for a completely arbitrary 2D mask
        profile_dummy = torch.zeros(x.shape[0], 1, device=x.device)
        return M_cont, M_bin, profile_dummy


# ---------------------------------------------------------------------------
# Global FC Spectral Mask Network
# ---------------------------------------------------------------------------

class GlobalFCSpectralMaskNet(nn.Module):
    """
    Generates a full 2D asymmetrical frequency mask using a Convolutional
    Encoder followed by a massive Fully Connected layer mapping directly
    to the (H, W//2+1) frequency grid.
    """
    def __init__(self, in_channels: int, H: int, W: int):
        super().__init__()
        self.H = H
        self.W = W
        self.W_half = W // 2 + 1
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False), nn.InstanceNorm2d(32, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),  nn.InstanceNorm2d(64, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.InstanceNorm2d(128, affine=True), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            # Direct mapping from 512 embedding to H * W_half pixels
            nn.Linear(512, H * self.W_half)
        )
        
        freq_grid = make_radial_freq_grid(H, W)
        self.register_buffer("freq_grid", freq_grid)
        
    def forward(self, x: torch.Tensor):
        feats = self.encoder(x)
        logits = self.fc(feats)
        
        M_cont = torch.sigmoid(logits).view(-1, 1, self.H, self.W_half)
        M_bin = STE.apply(M_cont)
        
        profile_dummy = torch.zeros(x.shape[0], 1, device=x.device)
        return M_cont, M_bin, profile_dummy
