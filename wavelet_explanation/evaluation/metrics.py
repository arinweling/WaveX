"""
Quantitative evaluation metrics for wavelet-domain explanations.

Metrics:
    compute_sparsity               — fraction of subband coefficients masked out
    compute_confidence_delta       — change in classifier confidence on explanation vs original
    compute_label_preservation_rate — top-1 agreement between f(x) and f(e)
    compute_subband_activity_profile — mean active fraction per subband (frequency profile)

All functions accept batched mask tensors; call them after accumulating a full
evaluation set for representative statistics.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sparsity(
    m_LL: torch.Tensor,
    m_LH: torch.Tensor,
    m_HL: torch.Tensor,
    m_HH: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute mask sparsity (fraction of coefficients set to zero = masked out).

    Args:
        m_LL, m_LH, m_HL, m_HH: (B, 1, H', W') binary masks (values in {0,1})

    Returns:
        dict with keys:
            'overall'          — fraction of all subband coefficients that are 0
            'LL', 'LH', 'HL', 'HH' — per-subband fraction zeroed
            'pixel_equivalent' — approximate fraction of original pixels masked
                                  (each subband coeff corresponds to a 2×2 pixel region)
    """
    def _zero_fraction(m: torch.Tensor) -> float:
        return (1.0 - m.float().mean()).item()

    per_subband = {
        "LL": _zero_fraction(m_LL),
        "LH": _zero_fraction(m_LH),
        "HL": _zero_fraction(m_HL),
        "HH": _zero_fraction(m_HH),
    }

    # Overall: all subband elements concatenated
    all_masks = torch.cat([m_LL.flatten(), m_LH.flatten(), m_HL.flatten(), m_HH.flatten()])
    overall = (1.0 - all_masks.float().mean()).item()

    # Pixel-equivalent sparsity:
    # A pixel is "kept" if the LL mask coefficient covering it is active (LL carries
    # the coarse approximation; other subbands add detail).
    # Simple approximation: proportion of LL coefficients kept.
    pixel_equivalent = per_subband["LL"]

    return {
        "overall": overall,
        "LL": per_subband["LL"],
        "LH": per_subband["LH"],
        "HL": per_subband["HL"],
        "HH": per_subband["HH"],
        "pixel_equivalent": pixel_equivalent,
    }


def compute_confidence_delta(
    classifier: nn.Module,
    x: torch.Tensor,
    e: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """
    Mean difference in classifier confidence: p_f(e)(y) - p_f(x)(y).

    Positive → explanation is more confidently classified than the original.
    Negative → explanation has lost discriminative information.

    Args:
        classifier: callable returning logits (B, C)
        x:  (B, C_img, H, W) original images
        e:  (B, C_img, H, W) explanation images
        y:  (B,) ground-truth / top-1 class indices

    Returns:
        mean confidence delta (float)
    """
    with torch.no_grad():
        prob_x = F.softmax(classifier(x), dim=1)
        prob_e = F.softmax(classifier(e), dim=1)

    # Gather confidence at the true class
    conf_x = prob_x[torch.arange(len(y)), y]  # (B,)
    conf_e = prob_e[torch.arange(len(y)), y]

    return (conf_e - conf_x).mean().item()


def compute_label_preservation_rate(
    classifier: nn.Module,
    x_batch: torch.Tensor,
    e_batch: torch.Tensor,
) -> float:
    """
    Fraction of examples where argmax f(x) == argmax f(e).

    Should be close to 1.0 for high-quality explanations.

    Args:
        classifier: callable returning logits (B, C)
        x_batch:   (B, C_img, H, W) original images
        e_batch:   (B, C_img, H, W) explanation images

    Returns:
        label preservation rate in [0, 1]
    """
    with torch.no_grad():
        pred_x = classifier(x_batch).argmax(dim=1)
        pred_e = classifier(e_batch).argmax(dim=1)

    return (pred_x == pred_e).float().mean().item()


def compute_subband_activity_profile(
    m_LL: torch.Tensor,
    m_LH: torch.Tensor,
    m_HL: torch.Tensor,
    m_HH: torch.Tensor,
) -> Dict[str, float]:
    """
    Mean active fraction per subband (opposite of sparsity — fraction of 1s).

    Reveals the frequency profile used by the classifier:
        - High LL activity → classifier is shape/structure-biased
        - High HH activity → classifier is texture-biased

    Args:
        m_LL, m_LH, m_HL, m_HH: (B, 1, H', W') binary masks

    Returns:
        dict {'LL', 'LH', 'HL', 'HH'} → mean active fraction (float)
    """
    return {
        "LL": m_LL.float().mean().item(),
        "LH": m_LH.float().mean().item(),
        "HL": m_HL.float().mean().item(),
        "HH": m_HH.float().mean().item(),
    }
