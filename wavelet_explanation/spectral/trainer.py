"""
Standalone trainer for spectral-domain explanation generation.

SpectralExplanationTrainer is intentionally self-contained and does not
inherit from or depend on the wavelet WaveletExplanationTrainer.  It reuses
shared infrastructure (FrozenClassifier, ActivationHookManager, and the
existing loss functions from `losses/`) but has its own training loop built
around the radial FFT mask.

Training objective:
    L_EXP = λ_act   * L_act          (activation matching — shared with wavelet)
          + λ_CE    * L_CE           (cross-entropy on explanation)
          + λ_KL    * L_KL           (KL divergence on output distributions)
          + λ_rob   * L_rob          (robustness under freq-domain noise)
          + λ_area  * L_area         (frequency-weighted sparsity)
          + λ_tv    * L_tv           (smoothness of mask in freq space)
          + λ_bin   * L_bin          (binarisation regulariser)

Usage:
    from spectral.trainer import SpectralExplanationTrainer
    trainer = SpectralExplanationTrainer(config, device='cuda')
    trainer.train_epoch(dataloader)
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from spectral.fft_mask import RadialSpectralMaskNet
from spectral.explanation import construct_spectral_explanation, perturb_spectral_explanation
from spectral.losses import spectral_area_loss, spectral_smoothness_loss, binarization_loss

# Shared infrastructure re-used from the wavelet package
from models.classifiers import FrozenClassifier
from training.hooks import ActivationHookManager
from losses.activation_matching import activation_matching_loss
from losses.output_fidelity import cross_entropy_loss, kl_divergence_loss


class SpectralExplanationTrainer:
    """
    Trains a RadialSpectralMaskNet while keeping the target classifier frozen.

    The mask network is the only component with learnable parameters.  It takes
    the original image as input and outputs a binary radial frequency mask.
    Applying the mask in the FFT domain and inverting via IFFT gives the
    explanation image e.  All downstream losses operate on e in pixel space,
    identical in structure to the wavelet trainer.

    Args:
        config: dict loaded from a YAML config file (see configs/).
        device: 'cpu' or 'cuda'.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)

        # ------------------------------------------------------------------
        # Image geometry — RadialSpectralMaskNet needs H and W at init time
        # ------------------------------------------------------------------
        image_size = config.get("image_size", 224)
        # Support non-square images via 'image_height' / 'image_width' keys,
        # falling back to the square `image_size` for backward compatibility.
        H = config.get("image_height", image_size)
        W = config.get("image_width",  image_size)

        in_channels = 1 if config.get("dataset") == "mnist" else 3

        # ------------------------------------------------------------------
        # Spectral mask network (the only trained model)
        # ------------------------------------------------------------------
        arch = config.get("spectral_mask_arch", "radial")
        if arch == "radial":
            num_radial_bins = config.get("spectral_radial_bins", 64)
            self.mask_net = RadialSpectralMaskNet(
                in_channels=in_channels,
                H=H,
                W=W,
                num_radial_bins=num_radial_bins,
            ).to(self.device)
        elif arch == "unet":
            from spectral.fft_mask import UNetSpectralMaskNet
            self.mask_net = UNetSpectralMaskNet(
                in_channels=in_channels, H=H, W=W
            ).to(self.device)
        elif arch == "global_fc":
            from spectral.fft_mask import GlobalFCSpectralMaskNet
            self.mask_net = GlobalFCSpectralMaskNet(
                in_channels=in_channels, H=H, W=W
            ).to(self.device)
        else:
            raise ValueError(f"Unknown spectral_mask_arch: {arch}")

        # Store freq_grid reference for loss computation
        self.freq_grid = self.mask_net.freq_grid  # (H, W//2+1), on self.device

        # ------------------------------------------------------------------
        # Frozen classifier
        # ------------------------------------------------------------------
        backbone = config.get("backbone", "resnet18")
        if backbone == "custom_cnn":
            self.classifier = _wrap_custom_cnn(config, self.device)
        elif backbone == "resnet18_stl10":
            self.classifier = _wrap_resnet18_stl10(self.device)
        elif backbone == "resnet18_tiny":
            self.classifier = _wrap_resnet18_tiny(config, self.device)
        else:
            pretrained = config.get("pretrained", True)
            self.classifier = FrozenClassifier(backbone, pretrained).to(self.device)

        # ------------------------------------------------------------------
        # Activation hooks
        # ------------------------------------------------------------------
        classifier_model = (
            self.classifier.model
            if isinstance(self.classifier, FrozenClassifier)
            else self.classifier
        )
        self.hook_manager = ActivationHookManager(classifier_model)

        # ------------------------------------------------------------------
        # Optimizer (mask_net parameters only)
        # ------------------------------------------------------------------
        self.optimizer = torch.optim.Adam(
            self.mask_net.parameters(),
            lr=config.get("lr", 1e-4),
        )

        # ------------------------------------------------------------------
        # Loss weights
        # ------------------------------------------------------------------
        self.lambda_act         = config.get("lambda_act",              1.0)
        self.lambda_CE          = config.get("lambda_CE",               4.0)
        self.lambda_KL          = config.get("lambda_KL",               0.4)
        self.lambda_rob         = config.get("lambda_rob",              6.0)
        self.lambda_bin         = config.get("lambda_bin",              0.5)
        self.lambda_spectral_area_lf = config.get("lambda_spectral_area_lf", 2.0)
        self.lambda_spectral_area_hf = config.get("lambda_spectral_area_hf", 10.0)
        self.lambda_spectral_tv = config.get("lambda_spectral_tv",      1.0)

        self.fixed_class_label  = config.get("fixed_class_label", None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_target_labels(self, logits_x: torch.Tensor) -> torch.Tensor:
        """Return target class ids (fixed override or model top-1)."""
        if self.fixed_class_label is not None:
            return torch.full(
                (logits_x.shape[0],),
                int(self.fixed_class_label),
                dtype=torch.long,
                device=logits_x.device,
            )
        return logits_x.argmax(dim=1).detach()

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        One complete gradient update.

        Steps:
            1.  Forward through RadialSpectralMaskNet → M_cont, M_bin, profile
            2.  Construct explanation e via spectral masking (FFT → mask → IFFT)
            3.  Run x through frozen classifier → activations_x, logits_x
            4.  Run e through frozen classifier → activations_e, logits_e
            5.  Compute L_act (activation matching)
            6.  Compute L_CE, L_KL (output fidelity)
            7.  Construct perturbed explanation e_pert, run through classifier
            8.  Compute L_rob (robustness)
            9.  Compute L_area, L_tv, L_bin (spectral mask priors)
            10. Assemble L_EXP, backward, clip gradients, step optimizer

        Args:
            x: (B, C, H, W) image batch on self.device
            y: (B,) integer class labels on self.device (may be unused if
               fixed_class_label is set or using model top-1 as target)

        Returns:
            dict of individual loss values (Python floats) for logging
        """
        self.mask_net.train()

        # --- Step 1: generate masks ---
        M_cont, M_bin, _ = self.mask_net(x)
        # M_cont: (B, 1, H, W//2+1) continuous
        # M_bin:  (B, 1, H, W//2+1) binary via STE

        # --- Step 2: construct explanation ---
        e = construct_spectral_explanation(x, M_bin)  # (B, C, H, W)

        # --- Steps 3–4: run classifier on x and e, collect activations ---
        # x first (reference — detached when computing L_act)
        self.hook_manager.clear()
        logits_x = self.classifier(x)
        activations_x = self.hook_manager.get_activations()

        # e second (activations remain in the graph for gradient flow)
        self.hook_manager.clear()
        logits_e = self.classifier(e)
        activations_e = self.hook_manager.get_activations()

        # --- Step 5: L_act ---
        l_act = activation_matching_loss(activations_x, activations_e)

        # --- Step 6: L_CE, L_KL ---
        y_model = self._get_target_labels(logits_x)
        l_ce = cross_entropy_loss(logits_e, y_model)
        l_kl = kl_divergence_loss(logits_x, logits_e)

        # --- Steps 7–8: L_rob ---
        e_pert = perturb_spectral_explanation(x, M_bin)
        self.hook_manager.clear()
        logits_pert = self.classifier(e_pert)
        l_rob = F.cross_entropy(logits_pert, y_model)

        # --- Step 9: spectral mask priors ---
        l_area = spectral_area_loss(
            M_cont,
            self.freq_grid,
            lambda_lf=self.lambda_spectral_area_lf,
            lambda_hf=self.lambda_spectral_area_hf,
        )
        l_tv  = spectral_smoothness_loss(M_cont)
        l_bin = binarization_loss(M_cont)

        # --- Step 10: assemble and backprop ---
        l_exp = (
            self.lambda_act * l_act
            + self.lambda_CE  * l_ce
            + self.lambda_KL  * l_kl
            + self.lambda_rob * l_rob
            + l_area
            + self.lambda_spectral_tv * l_tv
            + self.lambda_bin         * l_bin
        )

        self.optimizer.zero_grad()
        l_exp.backward()
        nn.utils.clip_grad_norm_(self.mask_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss_total": l_exp.item(),
            "loss_act":   l_act.item(),
            "loss_ce":    l_ce.item(),
            "loss_kl":    l_kl.item(),
            "loss_rob":   l_rob.item(),
            "loss_area":  l_area.item(),
            "loss_tv":    l_tv.item(),
            "loss_bin":   l_bin.item(),
        }

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Run train_step for every batch in the dataloader.

        Args:
            dataloader: yields (images, labels) batches

        Returns:
            dict of average loss values across the epoch
        """
        accum: Dict[str, float] = {}
        n_batches = 0

        for x, y in tqdm(dataloader, desc="training", leave=False):
            x = x.to(self.device)
            y = y.to(self.device)
            losses = self.train_step(x, y)

            for key, val in losses.items():
                accum[key] = accum.get(key, 0.0) + val
            n_batches += 1

        return {key: val / n_batches for key, val in accum.items()}

    # ------------------------------------------------------------------
    # Inference (no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """
        Generate the spectral mask and explanation for a batch.

        Args:
            x: (B, C, H, W) on self.device

        Returns:
            M_bin:           (B, 1, H, W//2+1) binary frequency mask
            M_cont:          (B, 1, H, W//2+1) continuous frequency mask
            radial_profile:  (B, num_radial_bins) per-bin sigmoid weights
            e:               (B, C, H, W) explanation image
        """
        self.mask_net.eval()
        M_cont, M_bin, radial_profile = self.mask_net(x)
        e = construct_spectral_explanation(x, M_bin)
        return M_bin, M_cont, radial_profile, e

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str):
        """Save mask_net weights, optimizer state, and config."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "mask_net_state_dict":  self.mask_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config":               self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load mask_net weights (and optionally optimizer state)."""
        ckpt = torch.load(path, map_location=self.device)
        self.mask_net.load_state_dict(ckpt["mask_net_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])


# ---------------------------------------------------------------------------
# Classifier loader helpers  (mirrors training/trainer.py)
# ---------------------------------------------------------------------------

def _wrap_custom_cnn(config: dict, device: torch.device) -> nn.Module:
    from models.custom_cnn import MNISTClassifier
    model = MNISTClassifier().to(device)
    ckpt_path = config.get("classifier_checkpoint", None)
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded MNIST classifier from {ckpt_path}")
    else:
        print("Warning: no classifier checkpoint found; using random weights.")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def _wrap_resnet18_stl10(device: torch.device) -> nn.Module:
    from models.stl10_classifier import load_stl10_classifier
    return load_stl10_classifier(device=str(device))


def _wrap_resnet18_tiny(config: dict, device: torch.device) -> nn.Module:
    from models.tiny_imagenet_classifier import TinyImageNetClassifier
    model = TinyImageNetClassifier().to(device)
    ckpt_path = config.get("classifier_checkpoint", "tiny_imagenet_classifier.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Tiny-ImageNet classifier checkpoint not found: {ckpt_path}"
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model
