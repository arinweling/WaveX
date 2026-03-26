"""
Standalone entry point for CartoonX-style explanation generation using custom losses.

In CartoonX, the wavelet coefficients are optimized DIRECTLY for a single image,
rather than training a neural network prior (U-Net) to predict them.

This file leverages the existing `WaveletExplanationTrainer` to reuse your entire 
suite of losses (Activation Matching, KL Divergence, CE, Robustness, etc.), 
but dynamically swaps the U-Net out for a `DirectWaveletMask` module.

Usage:
    python train_cartoonx.py --config configs/resnet18_cartoonx.yaml
"""

import argparse
import os
import sys

import yaml
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(__file__))

from training.trainer import WaveletExplanationTrainer
from models.unet import STE
from visualization.visualize import visualize_explanation, visualize_subband_attribution


# ---------------------------------------------------------------------------
# CartoonX Module: Direct Tensor Optimization
# ---------------------------------------------------------------------------

class DirectWaveletMask(nn.Module):
    """
    Directly optimizes the raw wavelet masking values for a single image,
    bypassing any Neural Network.
    """
    def __init__(self, H: int, W: int, active_subbands: list):
        super().__init__()
        
        self.active_subbands = active_subbands
        subband_H, subband_W = H // 2, W // 2
        
        # We initialize slightly above 0 so the sigmoid starts > 0.5 and the mask starts ON
        init_val = 0.5
        
        self.raw_LL = nn.Parameter(torch.full((1, 1, subband_H, subband_W), init_val))
        self.raw_LH = nn.Parameter(torch.full((1, 1, subband_H, subband_W), init_val))
        self.raw_HL = nn.Parameter(torch.full((1, 1, subband_H, subband_W), init_val))
        self.raw_HH = nn.Parameter(torch.full((1, 1, subband_H, subband_W), init_val))

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        masks_bin = {}
        masks_cont = {}
        zeros = torch.zeros(B, 1, H // 2, W // 2, device=x.device)
        
        raw_tensors = {
            "LL": self.raw_LL, "LH": self.raw_LH, 
            "HL": self.raw_HL, "HH": self.raw_HH
        }
        
        for name in ["LL", "LH", "HL", "HH"]:
            if name in self.active_subbands:
                # Sigmoid bounds backprop-able parameters between 0 and 1
                cont = torch.sigmoid(raw_tensors[name])
                cont = cont.expand(B, -1, -1, -1) 
                
                masks_bin[name] = STE.apply(cont)
                masks_cont[name] = cont
            else:
                masks_bin[name] = zeros
                masks_cont[name] = zeros
                
        continuous_masks = (masks_cont["LL"], masks_cont["LH"], masks_cont["HL"], masks_cont["HH"])
        return (masks_bin["LL"], masks_bin["LH"], masks_bin["HL"], masks_bin["HH"], continuous_masks)


# ---------------------------------------------------------------------------
# Custom Trainer inheriting your losses but replacing the network
# ---------------------------------------------------------------------------

class CartoonXTrainer(WaveletExplanationTrainer):
    def __init__(self, config: dict, device: str = "cpu"):
        super().__init__(config, device)
        
        # 1. Overwrite the U-Net with the Direct Tensor Optimizer
        H = config.get("image_size", 224)
        W = config.get("image_size", 224)
        active_subbands = [s for s in ["LL", "LH", "HL", "HH"] if s not in self.disabled_subbands]
        
        self.encoder_decoder = DirectWaveletMask(H, W, active_subbands).to(self.device)
        
        # 2. Re-initialize the optimizer because the parameters changed!
        # Direct Optimization usually requires a larger learning rate (e.g., 0.1) than a basic UNet
        self.optimizer = torch.optim.Adam(
            self.encoder_decoder.parameters(),
            lr=config.get("lr", 0.1),
        )


# ---------------------------------------------------------------------------
# Minimal Single-Image Training Loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run CartoonX with custom losses")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, default="outputs/cartoonx", help="Save directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load the single image directly
    image_path = config.get("single_image_path")
    if not image_path or not os.path.exists(image_path):
        raise ValueError("CartoonX requires 'single_image_path' inside the config! Point it to a valid image.")
        
    image_size = config.get("image_size", 224)
    transform = T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img = Image.open(image_path).convert("RGB")
    fixed_x = transform(img).unsqueeze(0).to(device)
    
    # Target label setup
    fixed_class_label = config.get("fixed_class_label", None)
    fixed_y = torch.tensor([fixed_class_label if fixed_class_label is not None else 0], dtype=torch.long, device=device)

    print(f"Loaded image '{image_path}' for CartoonX direct optimization.")

    # 2. Build the specialized Trainer
    trainer = CartoonXTrainer(config, device=device)

    # 3. Optimize the Mask Tensors!
    epochs = config.get("epochs", 500)
    for epoch in range(1, epochs + 1):
        # We pass the same image over and over
        losses = trainer.train_step(fixed_x, fixed_y)

        # Log periodic losses
        if epoch % 10 == 0 or epoch == 1:
            loss_str = " ".join(f"{k}={v:.4f}" for k, v in losses.items() if "total" in k or "ce" in k or "act" in k or "area" in k)
            
            with torch.no_grad():
                _, _, _, _, e, _ = trainer.predict_masks(fixed_x)
                logits_e = trainer.classifier(e)
                probs = torch.softmax(logits_e, dim=1)
                top_prob, top_class = probs[0].max(dim=0)
                
            print(f"Epoch {epoch:3d}/{epochs} | {loss_str} | Conf: {top_prob.item():.4f} (Class {top_class.item()})")

    # 4. Save Final Visualization
    with torch.no_grad():
        m_LL, m_LH, m_HL, m_HH, e, _ = trainer.predict_masks(fixed_x)
        
        # Denorm function
        t = fixed_x.clone().cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x_denorm = (t * std + mean).clamp(0, 1)
        e_denorm = (e.clone().cpu() * std + mean).clamp(0, 1)

        viz_path = os.path.join(args.output_dir, "cartoonx_explanation.png")
        visualize_explanation(
            x_denorm, m_LL, m_LH, m_HL, m_HH, e_denorm,
            save_path=viz_path,
        )
        print(f"\nOptimization Complete! Saved visualization to {viz_path}")


if __name__ == "__main__":
    main()
