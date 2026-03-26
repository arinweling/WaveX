"""
Main training loop for wavelet-domain explanation generation.

WaveletExplanationTrainer assembles every component:
    - HaarDWT for wavelet decomposition / reconstruction
    - UNet encoder-decoder that outputs four subband masks
    - FrozenClassifier (pretrained, parameters frozen)
    - ActivationHookManager for tapping intermediate activations
    - All loss terms: L_act, L_CE, L_KL, L_rob, L_area, L_bin

Training objective (L_EXP):
    L_EXP = λ_act * L_act + λ_CE * L_CE + λ_KL * L_KL
           + λ_rob * L_rob + λ_bin * L_bin + L_area

Key design decisions:
    - Classifier parameters are frozen via requires_grad=False, NOT torch.no_grad(),
      so that gradients still flow back through the explanation e to the U-Net.
    - Activation tensors captured for L_act are detached from the graph for x
      (reference) but kept attached for e (so the comparison creates a gradient
      signal for the U-Net).
    - Gradient clipping at max_norm=1.0 prevents training instability.
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from wavelet.dwt import HaarDWT
from wavelet.explanation import construct_explanation, perturb_explanation, connectivity_filter
from models.unet import UNet
from models.classifiers import FrozenClassifier
from training.hooks import ActivationHookManager
from losses.activation_matching import activation_matching_loss
from losses.output_fidelity import cross_entropy_loss, kl_divergence_loss
from losses.mask_priors import area_loss, binarization_loss, total_variation_loss, explanation_area_loss, explanation_tv_loss


class WaveletExplanationTrainer:
    """
    Trains the wavelet mask U-Net while keeping the target classifier frozen.

    Args:
        config: dict loaded from a YAML config file (see configs/).
        device: torch device string, e.g. 'cpu' or 'cuda'.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)

        # Wavelet transform (no learnable parameters)
        self.dwt = HaarDWT().to(self.device)

        # Subbands listed here are clamped to all-zeros (disabled).
        # Set via config key 'disabled_subbands', e.g. [LH, HL]
        self.disabled_subbands = set(config.get("disabled_subbands", []))
        _all_subbands = ["LL", "LH", "HL", "HH"]
        active_subbands = [s for s in _all_subbands if s not in self.disabled_subbands]

        # Encoder-decoder (the only trained model)
        in_channels = 1 if config.get("dataset") == "mnist" else 3
        self.pixel_mask_mode = config.get("pixel_mask_mode", False)
        if self.pixel_mask_mode:
            from models.pixel_mask_net import PixelMaskNet
            self.encoder_decoder = PixelMaskNet(in_channels=in_channels).to(self.device)
        else:
            self.encoder_decoder = UNet(
                in_channels=in_channels, active_subbands=active_subbands
            ).to(self.device)

        # Frozen classifier
        backbone = config.get("backbone", "resnet18")
        if backbone == "custom_cnn":
            # Load the pre-trained MNIST CNN wrapped as a FrozenClassifier substitute
            self.classifier = _wrap_custom_cnn(config, self.device)
        elif backbone == "resnet18_tiny":
            # ResNet-18 trained from scratch on Tiny-ImageNet-200 (200-class head)
            self.classifier = _wrap_resnet18_tiny(config, self.device)
        elif backbone == "resnet18_stl10":
            # ResNet-18 fine-tuned on STL-10 (10-class head), loaded from HuggingFace
            self.classifier = _wrap_resnet18_stl10(self.device)
        elif backbone == "resnet18_tiatoolbox_pcam":
            # ResNet-18 trained by TIAToolbox on PCam, loaded from HuggingFace Hub via timm
            self.classifier = _wrap_resnet18_tiatoolbox_pcam(config, self.device)
        else:
            pretrained = config.get("pretrained", True)
            self.classifier = FrozenClassifier(backbone, pretrained).to(self.device)

        # Activation hooks (attached to the inner .model for FrozenClassifier,
        # or to the raw module for the custom CNN wrapper)
        classifier_model = (
            self.classifier.model
            if isinstance(self.classifier, FrozenClassifier)
            else self.classifier
        )
        self.hook_manager = ActivationHookManager(classifier_model)

        # Optimizer (only encoder-decoder parameters are optimised)
        self.optimizer = torch.optim.Adam(
            self.encoder_decoder.parameters(),
            lr=config.get("lr", 1e-4),
        )

        # Loss weights from config
        self.lambda_act = config.get("lambda_act", 1.0)
        self.lambda_CE  = config.get("lambda_CE",  4.0)
        self.lambda_KL  = config.get("lambda_KL",  0.4)
        self.lambda_rob = config.get("lambda_rob", 6.0)
        self.lambda_bin = config.get("lambda_bin", 0.3)
        self.lambda_tv        = config.get("lambda_tv",        1.0)
        self.lambda_area_e    = config.get("lambda_area_e",    1.0)
        self.lambda_tv_e      = config.get("lambda_tv_e",      1.0)
        self.lambda_area_pixel = config.get("lambda_area_pixel", 10.0)
        self.min_explanation_neighbors = config.get("min_explanation_neighbors", 0)
        self.explanation_active_threshold = config.get("explanation_active_threshold", 0.0)
        self.fixed_class_label = config.get("fixed_class_label", None)

        self.lambda_area_LL = config.get("lambda_area_LL", 5.0)
        self.lambda_area_LH = config.get("lambda_area_LH", 10.0)
        self.lambda_area_HL = config.get("lambda_area_HL", 10.0)
        self.lambda_area_HH = config.get("lambda_area_HH", 20.0)

    def _get_target_labels(self, logits_x: torch.Tensor) -> torch.Tensor:
        """Return the class ids the explanation should preserve."""
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

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """
        One complete gradient update.

        Steps:
            1.  Forward through U-Net → masks {m_LL, m_LH, m_HL, m_HH}
            2.  Construct explanation e via wavelet masking
            3.  Run x through frozen classifier → activations_x, logits_x
            4.  Run e through frozen classifier → activations_e, logits_e
            5.  Compute L_act (activation matching)
            6.  Compute L_CE, L_KL (output fidelity)
            7.  Construct perturbed explanation e_pert, run through classifier
            8.  Compute L_rob (robustness)
            9.  Compute L_area, L_bin (mask priors)
            10. Assemble L_EXP, backward, clip gradients, step optimizer

        Args:
            x: (B, C, H, W) image batch on self.device
            y: (B,) integer class labels on self.device

        Returns:
            dict of individual loss values (Python floats) for logging
        """
        self.encoder_decoder.train()

        if self.pixel_mask_mode:
            return self._pixel_train_step(x, y)

        # ------ Step 1: generate masks ------
        # Disabled subbands are already zero — handled inside UNet
        m_LL, m_LH, m_HL, m_HH, cont_masks = self.encoder_decoder(x)

        # ------ Step 2: construct explanation ------
        e, _ = construct_explanation(x, m_LL, m_LH, m_HL, m_HH, self.dwt)
        e = connectivity_filter(e, self.min_explanation_neighbors, self.explanation_active_threshold)

        # ------ Steps 3–4: run classifier on x and e, collect activations ------
        # x first (reference – activations will be detached when computing L_act)
        self.hook_manager.clear()
        logits_x = self.classifier(x)
        activations_x = self.hook_manager.get_activations()

        # e second (activations stay in the graph for gradient flow)
        self.hook_manager.clear()
        logits_e = self.classifier(e)
        activations_e = self.hook_manager.get_activations()

        # ------ Step 5: L_act ------
        l_act = activation_matching_loss(activations_x, activations_e)

        # ------ Step 6: L_CE, L_KL ------
        # By default we explain the model's own top-1 prediction. When a fixed
        # class label is provided in config, we preserve that class instead.
        y_model = self._get_target_labels(logits_x)
        l_ce = cross_entropy_loss(logits_e, y_model)
        l_kl = kl_divergence_loss(logits_x, logits_e)

        # ------ Steps 7–8: L_rob ------
        e_pert = perturb_explanation(x, m_LL, m_LH, m_HL, m_HH, self.dwt)
        self.hook_manager.clear()
        logits_pert = self.classifier(e_pert)
        # Robustness: explanation should preserve the model's original prediction
        l_rob = F.cross_entropy(logits_pert, y_model)

        # ------ Step 9: L_area, L_bin, L_tv, L_area_e ------
        # binarization_loss and tv_loss use continuous (pre-STE) values
        l_bin    = binarization_loss(*cont_masks)
        l_tv     = total_variation_loss(*cont_masks)
        l_area_e = explanation_area_loss(e, x)
        l_tv_e   = explanation_tv_loss(e)
        l_area = area_loss(
            m_LL, m_LH, m_HL, m_HH,
            self.lambda_area_LL, self.lambda_area_LH,
            self.lambda_area_HL, self.lambda_area_HH,
        )

        # ------ Step 10: assemble and backprop ------
        l_exp = (
            self.lambda_act * l_act
            + self.lambda_CE  * l_ce
            + self.lambda_KL  * l_kl
            + self.lambda_rob * l_rob
            + l_area
            + self.lambda_bin    * l_bin
            + self.lambda_tv     * l_tv
            + self.lambda_area_e * l_area_e
            + self.lambda_tv_e   * l_tv_e
        )

        self.optimizer.zero_grad()
        l_exp.backward()
        nn.utils.clip_grad_norm_(self.encoder_decoder.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss_total": l_exp.item(),
            "loss_act":   l_act.item(),
            "loss_ce":    l_ce.item(),
            "loss_kl":    l_kl.item(),
            "loss_rob":   l_rob.item(),
            "loss_area":  l_area.item(),
            "loss_bin":   l_bin.item(),
            "loss_tv":    l_tv.item(),
            "loss_area_e": l_area_e.item(),
            "loss_tv_e":   l_tv_e.item(),
        }

    def _pixel_train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Training step for pixel-space mask mode (no wavelets).

        PixelMaskNet outputs a binary mask m in {0,1} at full image resolution.
        Explanation: e = m * x  (keep selected pixels, zero the rest).
        All losses mirror the wavelet train_step semantics.
        """
        # ------ generate pixel mask ------
        m_bin, m_cont = self.encoder_decoder(x)   # both (B, 1, H, W)

        # ------ construct explanation ------
        e = m_bin * x   # broadcast over channels: (B, C, H, W)
        e = connectivity_filter(e, self.min_explanation_neighbors, self.explanation_active_threshold)

        # ------ run classifier on x and e ------
        self.hook_manager.clear()
        logits_x = self.classifier(x)
        activations_x = self.hook_manager.get_activations()

        self.hook_manager.clear()
        logits_e = self.classifier(e)
        activations_e = self.hook_manager.get_activations()

        # ------ L_act ------
        l_act = activation_matching_loss(activations_x, activations_e)

        # ------ L_CE, L_KL ------
        y_model = self._get_target_labels(logits_x)
        l_ce = cross_entropy_loss(logits_e, y_model)
        l_kl = kl_divergence_loss(logits_x, logits_e)

        # ------ L_rob: add noise to the masked-out region ------
        noise = torch.randn_like(x) * 0.1
        e_pert = m_bin * x + (1.0 - m_bin) * noise
        self.hook_manager.clear()
        logits_pert = self.classifier(e_pert)
        l_rob = F.cross_entropy(logits_pert, y_model)

        # ------ mask priors (single mask) ------
        l_area   = self.lambda_area_pixel * m_cont.abs().mean()
        l_bin    = (m_cont - m_cont ** 2).abs().mean()
        diff_h   = (m_cont[:, :, :-1, :] - m_cont[:, :, 1:, :]).abs()
        diff_w   = (m_cont[:, :, :, :-1] - m_cont[:, :, :, 1:]).abs()
        l_tv     = diff_h.mean() + diff_w.mean()

        # ------ pixel-space explanation losses ------
        l_area_e = explanation_area_loss(e, x)
        l_tv_e   = explanation_tv_loss(e)

        # ------ assemble and backprop ------
        l_exp = (
            self.lambda_act    * l_act
            + self.lambda_CE   * l_ce
            + self.lambda_KL   * l_kl
            + self.lambda_rob  * l_rob
            + l_area
            + self.lambda_bin    * l_bin
            + self.lambda_tv     * l_tv
            + self.lambda_area_e * l_area_e
            + self.lambda_tv_e   * l_tv_e
        )

        self.optimizer.zero_grad()
        l_exp.backward()
        nn.utils.clip_grad_norm_(self.encoder_decoder.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss_total":  l_exp.item(),
            "loss_act":    l_act.item(),
            "loss_ce":     l_ce.item(),
            "loss_kl":     l_kl.item(),
            "loss_rob":    l_rob.item(),
            "loss_area":   l_area.item(),
            "loss_bin":    l_bin.item(),
            "loss_tv":     l_tv.item(),
            "loss_area_e": l_area_e.item(),
            "loss_tv_e":   l_tv_e.item(),
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
    # Mask inference (no grad, for evaluation / visualisation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_masks(self, x: torch.Tensor):
        """
        Generate masks and explanation for a batch without computing gradients.

        Args:
            x: (B, C, H, W) on self.device

        Returns:
            m_LL, m_LH, m_HL, m_HH: binary masks (B, 1, H//2, W//2)
            e: explanation (B, C, H, W)
            masked_subbands: dict {'LL','LH','HL','HH'} → (B, C, H//2, W//2)
        """
        self.encoder_decoder.eval()
        if self.pixel_mask_mode:
            m_bin, _ = self.encoder_decoder(x)
            e = m_bin * x
            e = connectivity_filter(e, self.min_explanation_neighbors, self.explanation_active_threshold)
            zeros = torch.zeros_like(m_bin)
            return m_bin, zeros, zeros, zeros, e, {}
        # Disabled subbands are already zero — handled inside UNet
        m_LL, m_LH, m_HL, m_HH, _ = self.encoder_decoder(x)
        e, masked_subbands = construct_explanation(x, m_LL, m_LH, m_HL, m_HH, self.dwt)
        e = connectivity_filter(e, self.min_explanation_neighbors, self.explanation_active_threshold)
        return m_LL, m_LH, m_HL, m_HH, e, masked_subbands

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str):
        """Save U-Net weights and config to a .pth file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "encoder_decoder_state_dict": self.encoder_decoder.state_dict(),
                "optimizer_state_dict":       self.optimizer.state_dict(),
                "config":                     self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load U-Net weights (and optionally optimizer state) from a .pth file."""
        ckpt = torch.load(path, map_location=self.device)
        self.encoder_decoder.load_state_dict(ckpt["encoder_decoder_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_custom_cnn(config: dict, device: torch.device) -> nn.Module:
    """
    Load and freeze a MNISTClassifier from the path specified in the config.
    Falls back to an un-trained model if no checkpoint path is given.
    """
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


def _wrap_resnet18_tiny(config: dict, device: torch.device) -> nn.Module:
    """
    Load and freeze a TinyImageNetClassifier from the path specified in the config.
    Raises FileNotFoundError if the checkpoint is missing — call
    train_tiny_imagenet_classifier() first (train.py does this automatically).
    """
    from models.tiny_imagenet_classifier import TinyImageNetClassifier

    model = TinyImageNetClassifier().to(device)
    ckpt_path = config.get("classifier_checkpoint", "tiny_imagenet_classifier.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Tiny-ImageNet classifier checkpoint not found: {ckpt_path}\n"
            "Run train.py with backbone=resnet18_tiny and it will train one automatically."
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded Tiny-ImageNet classifier from {ckpt_path}")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def _wrap_resnet18_stl10(device: torch.device) -> nn.Module:
    """
    Download (once via HuggingFace Hub cache) and freeze the
    DuckyDuck123/resnet18-stl10 classifier (ResNet-18, 10-class head,
    90.4% val accuracy on STL-10).
    """
    from models.stl10_classifier import load_stl10_classifier
    return load_stl10_classifier(device=str(device))


def _wrap_resnet18_tiatoolbox_pcam(config: dict, device: torch.device) -> nn.Module:
    """
    Load and freeze a TIAToolbox PCam ResNet-18 checkpoint from HuggingFace Hub
    through timm. Default repo id: 1aurent/resnet18.tiatoolbox-pcam.
    """
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required for backbone=resnet18_tiatoolbox_pcam. "
            "Install with: pip install timm"
        ) from exc

    repo_id = config.get("classifier_hf_repo", "1aurent/resnet18.tiatoolbox-pcam")
    model = timm.create_model(f"hf-hub:{repo_id}", pretrained=True)
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print(f"Loaded PCam classifier from HuggingFace Hub: {repo_id}")
    return model
