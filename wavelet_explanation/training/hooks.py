"""
Forward hook management for capturing intermediate activations.

ActivationHookManager attaches PyTorch forward hooks to every Conv2d,
Linear, and ReLU layer in a given model.  After each forward pass the
captured activations are available via get_activations().

Usage pattern in the training loop:
    hook_manager = ActivationHookManager(classifier)
    ...
    hook_manager.clear()
    logits_x = classifier(x)
    activations_x = hook_manager.get_activations()

    hook_manager.clear()
    logits_e = classifier(e)
    activations_e = hook_manager.get_activations()

    loss = activation_matching_loss(activations_x, activations_e)
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class ActivationHookManager:
    """
    Registers and manages forward hooks on a model's Conv2d, Linear, and ReLU layers.

    Activations are stored as { layer_name: (output_tensor, layer_type_str) }.
    Call clear() before each forward pass to discard stale activations.
    Call remove() when training is complete to detach hooks.

    Args:
        model: the nn.Module to instrument (e.g. a FrozenClassifier.model)
    """

    def __init__(self, model: nn.Module):
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._activations: Dict[str, Tuple[torch.Tensor, str]] = {}
        self._register_hooks(model)

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self, model: nn.Module):
        """Attach a capture hook to every Conv2d, Linear, and ReLU layer."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
                layer_type = _classify_layer(module)
                hook = module.register_forward_hook(self._make_hook(name, layer_type))
                self._hooks.append(hook)

    def _make_hook(self, name: str, layer_type: str):
        """Return a hook that stores the layer output under `name`."""
        def hook(module: nn.Module, inp, output: torch.Tensor):
            # Store the output tensor and its layer type.
            # We do NOT detach here — the caller decides when to detach.
            self._activations[name] = (output, layer_type)
        return hook

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_activations(self) -> Dict[str, Tuple[torch.Tensor, str]]:
        """Return a shallow copy of the current activation dictionary."""
        return dict(self._activations)

    def clear(self):
        """Discard all stored activations.  Call before every forward pass."""
        self._activations = {}

    def remove(self):
        """Remove all registered hooks from the model."""
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_layer(module: nn.Module) -> str:
    if isinstance(module, nn.Conv2d):
        return "conv"
    if isinstance(module, nn.Linear):
        return "linear"
    return "relu"
