"""
Spectral-domain mask loss functions.

spectral_area_loss:
    Frequency-weighted L1 sparsity.  High-frequency components are penalised
    more than low-frequency ones, so the mask is encouraged to favour retaining
    coarse shape information over fine texture unless the classifier demands it.

spectral_smoothness_loss:
    Total variation on the continuous mask in frequency space.  Encourages the
    selected frequencies to form a smooth, connected region (typically a disc
    centred on DC) rather than a scattered set of isolated coefficients.

binarization_loss:
    ||M - M²||_1 — identical in purpose to the wavelet mask_priors version;
    pushes continuous mask values toward {0, 1}.
"""

import torch


def spectral_area_loss(
    M_cont: torch.Tensor,
    freq_grid: torch.Tensor,
    lambda_lf: float = 2.0,
    lambda_hf: float = 10.0,
) -> torch.Tensor:
    """
    Frequency-weighted L1 sparsity penalty.

    Each frequency coefficient (u, v) is penalised proportionally to its
    normalised radial frequency magnitude r ∈ [0, 1]:
        weight(u,v) = lambda_lf + (lambda_hf - lambda_lf) * r

    So keeping DC (r=0) costs lambda_lf, whereas keeping the highest
    representable frequency (r=1) costs lambda_hf.  Setting lambda_hf >>
    lambda_lf encourages a low-pass explanation (shape-first).

    Args:
        M_cont:    (B, 1, H, W//2+1)  continuous mask in [0, 1]
        freq_grid: (H, W//2+1)        pre-computed normalised radial freq grid
        lambda_lf: weight at lowest frequency (DC)
        lambda_hf: weight at highest frequency (Nyquist)

    Returns:
        scalar loss
    """
    # Expand freq_grid to match M_cont: (1, 1, H, W//2+1)
    freq = freq_grid.unsqueeze(0).unsqueeze(0).to(M_cont.device)
    weights = lambda_lf + (lambda_hf - lambda_lf) * freq  # (1, 1, H, W//2+1)
    return (weights * M_cont).mean()


def spectral_smoothness_loss(M_cont: torch.Tensor) -> torch.Tensor:
    """
    Total variation of the frequency-domain mask.

    Penalises abrupt transitions between adjacent frequency bins, encouraging the
    kept region to be a connected band (e.g. a low-pass disc) rather than
    scattered high/low frequency islands.

    Args:
        M_cont: (B, 1, H, W//2+1) continuous mask in [0, 1]

    Returns:
        scalar TV loss
    """
    diff_h = (M_cont[:, :, :-1, :] - M_cont[:, :, 1:, :]).abs()  # freq-row diff
    diff_w = (M_cont[:, :, :, :-1] - M_cont[:, :, :, 1:]).abs()  # freq-col diff
    return diff_h.mean() + diff_w.mean()


def binarization_loss(M_cont: torch.Tensor) -> torch.Tensor:
    """
    Binarisation regulariser: ||M - M²||_1.

    For a mask value v: v - v² = v(1-v), which is 0 at v∈{0,1} and maximal
    at v=0.5.  Minimising this pushes soft values toward binary extremes.

    Args:
        M_cont: (B, 1, H, W//2+1) continuous mask in [0, 1]

    Returns:
        scalar binarisation loss
    """
    return (M_cont - M_cont ** 2).abs().mean()
