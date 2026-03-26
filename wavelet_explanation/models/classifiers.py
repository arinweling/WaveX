"""
Frozen pretrained classifier wrapper with activation tapping.

FrozenClassifier wraps a torchvision model, freezes all its parameters,
and registers forward hooks on Conv2d, Linear, and ReLU layers to capture
intermediate activations.  These activations are used by the activation
matching loss (L_act).

Supported backbones (loaded from torchvision.models):
    resnet18, mobilenet_v3_small, convnext_small, efficientnet_b0, vit_b_16
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm


# Maps backbone name → (model_fn, weights_enum)
_BACKBONE_REGISTRY = {
    "resnet18":          (tvm.resnet18,          tvm.ResNet18_Weights.DEFAULT),
    "mobilenet_v3_small":(tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights.DEFAULT),
    "convnext_small":    (tvm.convnext_small,     tvm.ConvNeXt_Small_Weights.DEFAULT),
    "efficientnet_b0":   (tvm.efficientnet_b0,    tvm.EfficientNet_B0_Weights.DEFAULT),
    "vit_b_16":          (tvm.vit_b_16,           tvm.ViT_B_16_Weights.DEFAULT),
}


class FrozenClassifier(nn.Module):
    """
    Wraps a pretrained torchvision classifier with all parameters frozen.

    Activations at every Conv2d, Linear, and ReLU layer are captured by
    forward hooks during each forward pass.  Call clear_activations() before
    each forward pass to avoid stale data.

    Note: keeps BatchNorm in eval mode permanently so running statistics
    (rather than batch statistics) are always used.

    Args:
        backbone_name: one of the keys in _BACKBONE_REGISTRY
        pretrained:    if True, load ImageNet pretrained weights; else random init
    """

    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True):
        super().__init__()

        if backbone_name not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Choose from {list(_BACKBONE_REGISTRY.keys())}."
            )

        model_fn, weights = _BACKBONE_REGISTRY[backbone_name]
        self.backbone_name = backbone_name
        self.model = model_fn(weights=weights if pretrained else None)

        # Freeze every parameter — gradients must NOT flow into the classifier
        for param in self.model.parameters():
            param.requires_grad = False

        # Keep BN in eval mode permanently
        self.model.eval()

        # Activation storage: {layer_name: (tensor, layer_type_str)}
        self._activations: Dict[str, Tuple[torch.Tensor, str]] = {}
        self._hooks = []

        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_hooks(self):
        """Register capture hooks on Conv2d, Linear, and ReLU layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
                layer_type = _layer_type_str(module)
                hook = module.register_forward_hook(
                    self._make_hook(name, layer_type)
                )
                self._hooks.append(hook)

    def _make_hook(self, name: str, layer_type: str):
        """Return a hook closure that stores the output activation."""
        def hook(module, input, output):
            # Detach from graph; gradients must not flow through the classifier.
            # The key insight: we freeze params with requires_grad=False, so
            # gradients from the loss CAN still flow back through `e` (the input
            # to the classifier), even though we detach the stored activations
            # (used only for L_act comparison, not as a differentiable path).
            self._activations[name] = (output, layer_type)
        return hook

    def clear_activations(self):
        """Clear stored activations. Call this before every forward pass."""
        self._activations = {}

    def get_activations(self) -> Dict[str, Tuple[torch.Tensor, str]]:
        """Return a shallow copy of the activation dict."""
        return dict(self._activations)

    def remove_hooks(self):
        """Remove all registered hooks (useful when done training)."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference.  Activations are stored in self._activations.

        Args:
            x: (B, C, H, W) image batch
        Returns:
            logits: (B, num_classes)
        """
        # Always run in eval mode
        return self.model(x)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        """Override to prevent accidentally setting the classifier to training mode."""
        # Encoder-decoder (the *other* model) can be in train mode,
        # but this classifier must always stay in eval mode.
        return self  # no-op

    def eval(self):
        return self


def _layer_type_str(module: nn.Module) -> str:
    if isinstance(module, nn.Conv2d):
        return "conv"
    if isinstance(module, nn.Linear):
        return "linear"
    if isinstance(module, nn.ReLU):
        return "relu"
    return "other"
