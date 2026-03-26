"""
ResNet-18 fine-tuned on STL-10 (96×96, 10 classes), loaded from
DuckyDuck123/resnet18-stl10 on HuggingFace.

Architecture:
    Standard ResNet-18 with the final FC layer replaced:
    Linear(512, 1000) → Linear(512, 10).

Classes (index order matches the HuggingFace checkpoint):
    0: airplane  1: bird  2: car   3: cat   4: deer
    5: dog       6: horse 7: monkey 8: ship  9: truck

Expected input: (B, 3, 96, 96) normalised with ImageNet mean/std.

Usage:
    from models.stl10_classifier import STL10Classifier, load_stl10_classifier
    model = load_stl10_classifier(save_path="stl10_classifier.pth", device="cuda")
"""

import os

import torch
import torch.nn as nn
import torchvision


class STL10Classifier(nn.Module):
    """
    ResNet-18 for STL-10 (10-class head).

    Input:  (B, 3, 96, 96)
    Output: (B, 10) logits
    """

    def __init__(self):
        super().__init__()
        # Start with random weights — actual weights loaded separately via
        # load_stl10_classifier() from the HuggingFace checkpoint.
        self.model = torchvision.models.resnet18(weights=None)
        # Replace the 1000-class head with a 10-class head
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_stl10_classifier(
    save_path: str = "stl10_classifier.pth",
    device: str = "cpu",
) -> STL10Classifier:
    """
    Download (once) and load the DuckyDuck123/resnet18-stl10 checkpoint.

    On first call the weights are fetched from HuggingFace Hub and cached
    locally by huggingface_hub.  Subsequent calls reuse the local cache.

    Args:
        save_path:  unused — weights are managed by hf_hub_download cache.
                    Kept for API consistency with other classifier loaders.
        device:     'cpu' or 'cuda'

    Returns:
        STL10Classifier with frozen parameters in eval mode.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download the STL-10 checkpoint.\n"
            "Install it with: pip install huggingface_hub"
        ) from exc

    model = STL10Classifier().to(device)

    ckpt_path = hf_hub_download(
        repo_id="DuckyDuck123/resnet18-stl10",
        filename="pytorch_model.pth",
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    # The HuggingFace checkpoint was saved from a bare ResNet18 (keys like
    # "conv1.weight"), but STL10Classifier wraps it as self.model, so we need
    # to add the "model." prefix to match the expected key names.
    if not any(k.startswith("model.") for k in state_dict):
        state_dict = {"model." + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"Loaded STL-10 classifier from HuggingFace (DuckyDuck123/resnet18-stl10)")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model
