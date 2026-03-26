"""
Mask minimality priors: L_area and L_bin.

L_area — per-subband L1 sparsity penalty with different weights per subband.
         HH (fine texture) is penalised most; LL (coarse structure) least.
         This replaces total-variation loss — the wavelet decomposition already
         separates scales, so TV would be redundant.

L_bin  — binarisation regulariser ||m - m²||_1 that pushes soft mask values
         toward the extreme values {0, 1}.
"""

import torch


def area_loss(
    m_LL: torch.Tensor,
    m_LH: torch.Tensor,
    m_HL: torch.Tensor,
    m_HH: torch.Tensor,
    lambda_LL: float = 5.0,
    lambda_LH: float = 10.0,
    lambda_HL: float = 10.0,
    lambda_HH: float = 20.0,
) -> torch.Tensor:
    """
    Per-subband L1 area penalty.

    Encourages sparse masks.  HH (texture / diagonal edges) is penalised
    most aggressively because fine texture details are usually least
    diagnostic for classification.

    Args:
        m_LL, m_LH, m_HL, m_HH: per-subband mask tensors (B, 1, H', W')
                                  values in [0,1] (continuous before STE)
                                  or {0,1} (binary after STE)
        lambda_*: per-subband penalty weight

    Returns:
        scalar area loss
    """
    loss = (
        lambda_LL * m_LL.abs().mean()
        + lambda_LH * m_LH.abs().mean()
        + lambda_HL * m_HL.abs().mean()
        + lambda_HH * m_HH.abs().mean()
    )
    return loss


def binarization_loss(
    m_LL: torch.Tensor,
    m_LH: torch.Tensor,
    m_HL: torch.Tensor,
    m_HH: torch.Tensor,
) -> torch.Tensor:
    """
    Binarisation regulariser: ||m - m²||_1, summed across all four subbands.

    For a mask value v:
        v - v² = v(1-v) which is 0 at v=0 and v=1, and maximised at v=0.5.
    Minimising this pushes mask values away from 0.5 toward {0, 1}.

    Used on the *continuous* mask values (before STE), so gradients flow.

    Args:
        m_LL, m_LH, m_HL, m_HH: (B, 1, H', W') continuous masks in [0,1]

    Returns:
        scalar binarisation loss
    """
    def _bin_penalty(m):
        return (m - m ** 2).abs().mean()

    return _bin_penalty(m_LL) + _bin_penalty(m_LH) + _bin_penalty(m_HL) + _bin_penalty(m_HH)


def total_variation_loss(
    m_LL: torch.Tensor,
    m_LH: torch.Tensor,
    m_HL: torch.Tensor,
    m_HH: torch.Tensor,
) -> torch.Tensor:
    """
    Total variation (smoothness) loss: L_tv = sum_{i,j} |m_{i,j} - m_{i+1,j}|
                                                       + |m_{i,j} - m_{i,j+1}|

    Suppresses noisy, isolated activations and encourages the mask to form
    smooth, contiguous regions.  Applied to each active subband independently
    and summed.

    Args:
        m_LL, m_LH, m_HL, m_HH: (B, 1, H', W') mask tensors (binary or continuous)

    Returns:
        scalar TV loss (mean over spatial positions and batch)
    """
    def _tv(m: torch.Tensor) -> torch.Tensor:
        diff_h = (m[:, :, :-1, :] - m[:, :, 1:, :]).abs()   # vertical differences
        diff_w = (m[:, :, :, :-1] - m[:, :, :, 1:]).abs()   # horizontal differences
        return diff_h.mean() + diff_w.mean()

    return _tv(m_LL) + _tv(m_LH) + _tv(m_HL) + _tv(m_HH)


def explanation_area_loss(
    e: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Pixel-space area loss on the IDWT-reconstructed explanation.

    Measures the fraction of original image energy retained by the explanation:

        L_area_e = ||e||_1 / ||x||_1

    A value of 1.0 means the explanation preserves all pixel energy (no
    masking); minimising this loss drives the explanation toward a compact
    subset of the original image in pixel space, complementing the
    wavelet-domain area loss which operates on the binary masks.

    Args:
        e: (B, C, H, W) IDWT-reconstructed explanation
        x: (B, C, H, W) original input image (used only as a normaliser)

    Returns:
        scalar loss in [0, ∞)
    """
    return e.abs().mean() / (x.abs().mean() + 1e-8)


def explanation_tv_loss(e: torch.Tensor) -> torch.Tensor:
    """
    Total variation loss on the IDWT-reconstructed explanation in pixel space.

    Penalises abrupt pixel transitions in e directly, suppressing speckle /
    checkerboard artefacts that can appear in the reconstructed image even
    when the wavelet masks are smooth.

        L_tv_e = mean |e_{i,j} - e_{i+1,j}| + mean |e_{i,j} - e_{i,j+1}|

    Applied across all colour channels and averaged.

    Args:
        e: (B, C, H, W) IDWT-reconstructed explanation

    Returns:
        scalar TV loss
    """
    diff_h = (e[:, :, :-1, :] - e[:, :, 1:, :]).abs()
    diff_w = (e[:, :, :, :-1] - e[:, :, :, 1:]).abs()
    return diff_h.mean() + diff_w.mean()
