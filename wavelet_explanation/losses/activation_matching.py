"""
Activation matching loss (L_act).

Measures how well the activated feature maps of the explanation e match
those of the original image x at every instrumented layer.

Distance metric per layer type:
    Conv2d / ReLU: mean squared error over spatial feature maps
    Linear:        1 - cosine_similarity  (0 = identical directions)

A weighted sum across all layers is returned.

Layer weights (α_l):
    - "uniform"       : all layers weight 1.0
    - "depth_weighted": linearly increasing weight from 1.0 to 2.0 with depth
      (deeper semantic features are emphasised more)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def activation_matching_loss(
    activations_x: Dict[str, Tuple[torch.Tensor, str]],
    activations_e: Dict[str, Tuple[torch.Tensor, str]],
    layer_weights: Optional[Dict[str, float]] = None,
    weight_scheme: str = "uniform",
) -> torch.Tensor:
    """
    Compute the activation matching loss between a reference image and its explanation.

    Args:
        activations_x:  {layer_name: (activation_tensor, layer_type_str)}
                        captured while running the classifier on x
        activations_e:  {layer_name: (activation_tensor, layer_type_str)}
                        captured while running the classifier on e
                        Must have the same keys as activations_x.
        layer_weights:  optional per-layer scalar weights.  If None, weights
                        are computed according to weight_scheme.
        weight_scheme:  "uniform" | "depth_weighted"

    Returns:
        total_loss: scalar tensor (weighted sum of per-layer distances)
    """
    common_layers = sorted(set(activations_x.keys()) & set(activations_e.keys()))
    if not common_layers:
        raise ValueError("No common layers found between activations_x and activations_e.")

    if layer_weights is None:
        layer_weights = _compute_weights(common_layers, weight_scheme)

    total = torch.tensor(0.0, device=_device_of(activations_e, common_layers[0]))

    for name in common_layers:
        act_x, layer_type = activations_x[name]
        act_e, _          = activations_e[name]

        dist = _layer_distance(act_x, act_e, layer_type)
        weight = layer_weights.get(name, 1.0)
        total = total + weight * dist

    return total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _layer_distance(
    act_x: torch.Tensor,
    act_e: torch.Tensor,
    layer_type: str,
) -> torch.Tensor:
    """
    Compute per-layer distance between activations of x and e.

    Conv/ReLU:  MSE averaged over all spatial and channel dimensions.
    Linear:     1 - mean_cosine_similarity (batch-wise).
    """
    if layer_type == "linear":
        # act_x, act_e: (B, D)
        cos_sim = F.cosine_similarity(act_x.detach(), act_e, dim=1)  # (B,)
        return (1.0 - cos_sim).mean()
    else:
        # Conv or ReLU: (B, C, H, W) or any shape
        return F.mse_loss(act_e, act_x.detach())


def _compute_weights(layer_names, scheme: str) -> Dict[str, float]:
    n = len(layer_names)
    if scheme == "depth_weighted":
        # Linear ramp from 1.0 (shallowest) to 2.0 (deepest)
        return {
            name: 1.0 + (i / max(n - 1, 1))
            for i, name in enumerate(layer_names)
        }
    # Default: uniform
    return {name: 1.0 for name in layer_names}


def _device_of(activations, layer_name):
    tensor, _ = activations[layer_name]
    return tensor.device
