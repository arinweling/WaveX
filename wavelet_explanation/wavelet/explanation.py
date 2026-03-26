"""
Wavelet-domain explanation construction and perturbation.

construct_explanation: applies per-subband binary masks and reconstructs
    the explanation in pixel space via IDWT.

perturb_explanation: replaces masked-out subband coefficients with
    Gaussian noise (used for the robustness loss L_rob).
"""

import torch
import torch.nn.functional as F
from .dwt import HaarDWT


def construct_explanation(
    x: torch.Tensor,
    m_LL: torch.Tensor,
    m_LH: torch.Tensor,
    m_HL: torch.Tensor,
    m_HH: torch.Tensor,
    dwt: HaarDWT,
):
    """
    Build the wavelet-masked explanation image.

    Process:
        x → DWT → {x_LL, x_LH, x_HL, x_HH}
        e_sb = m_sb * x_sb  for each subband
        e   = IDWT(e_LL, e_LH, e_HL, e_HH)

    Args:
        x:    (B, C, H, W)  original image
        m_LL: (B, 1, H//2, W//2)  binary mask for LL subband
        m_LH: (B, 1, H//2, W//2)  binary mask for LH subband
        m_HL: (B, 1, H//2, W//2)  binary mask for HL subband
        m_HH: (B, 1, H//2, W//2)  binary mask for HH subband
        dwt:  HaarDWT instance

    Returns:
        e:           (B, C, H, W) — reconstructed explanation
        masked_subbands: dict with keys {'LL','LH','HL','HH'} → masked subband tensors
                         (B, C, H//2, W//2) — useful for visualization
    """
    x_LL, x_LH, x_HL, x_HH = dwt(x)

    # Masks are single-channel; broadcast over colour channels
    e_LL = m_LL * x_LL   # (B, C, H//2, W//2)
    e_LH = m_LH * x_LH
    e_HL = m_HL * x_HL
    e_HH = m_HH * x_HH

    e = dwt.inverse(e_LL, e_LH, e_HL, e_HH)

    masked_subbands = {"LL": e_LL, "LH": e_LH, "HL": e_HL, "HH": e_HH}
    return e, masked_subbands


def perturb_explanation(
    x: torch.Tensor,
    m_LL: torch.Tensor,
    m_LH: torch.Tensor,
    m_HL: torch.Tensor,
    m_HH: torch.Tensor,
    dwt: HaarDWT,
) -> torch.Tensor:
    """
    Build a perturbed explanation where masked-out subband coefficients are
    replaced by i.i.d. Gaussian noise (used to compute the robustness loss).

    Process for each subband sb:
        e_pert_sb = m_sb * x_sb + (1 - m_sb) * r_sb
        where r_sb ~ N(0, std(x_sb))

    Args:
        x:    (B, C, H, W)
        m_LL, m_LH, m_HL, m_HH: (B, 1, H//2, W//2) binary masks
        dwt:  HaarDWT instance

    Returns:
        e_perturbed: (B, C, H, W)
    """
    x_LL, x_LH, x_HL, x_HH = dwt(x)

    def _perturb_subband(x_sb, m_sb):
        std = x_sb.std().detach() + 1e-6
        noise = torch.randn_like(x_sb) * std
        return m_sb * x_sb + (1.0 - m_sb) * noise

    e_pert_LL = _perturb_subband(x_LL, m_LL)
    e_pert_LH = _perturb_subband(x_LH, m_LH)
    e_pert_HL = _perturb_subband(x_HL, m_HL)
    e_pert_HH = _perturb_subband(x_HH, m_HH)

    e_perturbed = dwt.inverse(e_pert_LL, e_pert_LH, e_pert_HL, e_pert_HH)
    return e_perturbed


def connectivity_filter(
    e: torch.Tensor,
    min_neighbors: int = 2,
    active_threshold: float = 0.0,
) -> torch.Tensor:
    """
    Zero out any pixel in the explanation that has fewer than `min_neighbors`
    active pixels in its 8-connected 3×3 neighbourhood.

    "Active" is defined as having per-pixel luminance (sum over channels)
    greater than `active_threshold * per-image-max-luminance`.  This lets
    you treat low-magnitude "grey blobs" (produced by the Haar IDWT when only
    the LL subband is kept) as inactive without affecting brighter regions.

    Args:
        e:                (B, C, H, W) explanation tensor
        min_neighbors:    minimum active neighbours required (default 2).
                          Set to 0 to disable.
        active_threshold: fraction of per-image peak luminance below which a
                          pixel is considered inactive (default 0.0 = strict >0).
                          A value like 0.05 ignores pixels whose luminance is
                          less than 5 % of the brightest pixel in that image.

    Returns:
        (B, C, H, W) filtered explanation
    """
    if min_neighbors <= 0:
        return e

    lum = e.detach().abs().sum(dim=1, keepdim=True)  # (B, 1, H, W)

    if active_threshold > 0.0:
        # Per-image max luminance, shape (B, 1, 1, 1)
        peak = lum.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        active = (lum > active_threshold * peak).float()
    else:
        active = (lum > 0).float()

    # Count 8-connected neighbours using a 3×3 all-ones kernel with centre = 0
    kernel = torch.ones(1, 1, 3, 3, device=e.device, dtype=e.dtype)
    kernel[0, 0, 1, 1] = 0.0
    neighbor_count = F.conv2d(active, kernel, padding=1)  # (B, 1, H, W)

    keep = (neighbor_count >= min_neighbors).float()  # (B, 1, H, W)
    return e * keep
