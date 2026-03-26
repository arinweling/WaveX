"""
Output fidelity losses: L_CE and L_KL.

L_CE  — cross-entropy between classifier output on explanation e and the
        ground-truth label y.  Ensures the explanation preserves the predicted class.

L_KL  — KL divergence from the classifier's output distribution on x to the
        distribution on e.  Encourages the full softmax distribution (not just
        top-1) to match, for a richer fidelity constraint.
"""

import torch
import torch.nn.functional as F


def cross_entropy_loss(logits_e: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Standard cross-entropy: -log p_f(e)(y).

    Args:
        logits_e: (B, num_classes) raw logits from classifier on explanation e
        y:        (B,) integer class labels (ground truth / top-1 of original)

    Returns:
        scalar mean cross-entropy loss
    """
    return F.cross_entropy(logits_e, y)


def kl_divergence_loss(logits_x: torch.Tensor, logits_e: torch.Tensor) -> torch.Tensor:
    """
    KL divergence: KL( softmax(f(x)) || softmax(f(e)) ).

    Penalises the explanation when its output distribution diverges from the
    original image's output distribution.

    KL(P||Q) = sum P * (log P - log Q)

    PyTorch's kl_div expects log-probabilities as the *input* (Q) and
    probabilities as the *target* (P):
        F.kl_div(log_Q, P)  →  sum P * (log P - log Q)

    Args:
        logits_x: (B, num_classes) logits from classifier on original image x
        logits_e: (B, num_classes) logits from classifier on explanation e

    Returns:
        scalar mean KL divergence
    """
    p = F.softmax(logits_x.detach(), dim=1)   # target distribution (from x)
    log_q = F.log_softmax(logits_e, dim=1)    # log of predicted distribution (from e)
    return F.kl_div(log_q, p, reduction="batchmean")
