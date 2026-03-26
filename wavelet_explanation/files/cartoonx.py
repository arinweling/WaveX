"""
CartoonX: Cartoon Explanations of Image Classifiers
Kolek et al. (2022) — with extended losses from Med-CAM paper

Extended losses added on top of the original:
  1. KL Divergence     — matches full softmax distribution (not just top class)
  2. Activation Matching — matches intermediate layer activations
  3. Cross-Entropy     — forces explanation to independently predict correct class

Requirements:
    pip install torch torchvision pytorch-wavelets

PyTorch Wavelets:
    pip install git+https://github.com/fbcotter/pytorch_wavelets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from dataclasses import dataclass, field, asdict
import yaml
import os
from pytorch_wavelets import DWTForward, DWTInverse


# ─────────────────────────────────────────────
#  0. CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class CartoonXConfig:
    """
    Single config object controlling every aspect of CartoonX.

    Loss weights:
        lambda_sparse   : L1 sparsity penalty (core CartoonX term)
                          Higher → sparser explanation (more blurred out)
                          Lower  → denser explanation (more detail preserved)
                          Paper uses λk ∈ [20, 80], here expressed as λ directly.
                          Typical range: 10.0 – 200.0

        lambda_kl       : Weight for KL divergence loss (full distribution matching)
                          0.0 = disabled (falls back to L2 on top class)
                          Typical range: 0.5 – 5.0

        lambda_act      : Weight for activation matching loss (layer-wise supervision)
                          0.0 = disabled
                          Typical range: 0.01 – 0.5
                          Keep small — activation magnitudes are large

        lambda_ce       : Weight for cross-entropy loss (class-discriminative)
                          0.0 = disabled
                          Typical range: 0.1 – 2.0

    Loss toggles:
        use_kl          : Enable KL divergence (replaces L2 distortion)
        use_act_match   : Enable activation matching
        use_ce          : Enable cross-entropy

    DWT parameters:
        wavelet         : Mother wavelet — 'db3' (paper default), 'db4', 'db5', 'haar'
                          Haar produces blocky artifacts (avoid)
                          Daubechies-3 gives clean piece-wise smooth results
        n_scales        : Number of DWT decomposition levels J
                          Higher J → coarser/smoother explanations
                          Paper uses J=5 for 256×256 images

    Optimization:
        n_steps         : SGD iterations (paper: 2000)
        n_samples       : Noise samples per step L (paper: 64)
                          More samples = more stable gradients, more compute
        lr              : Adam learning rate (paper: 0.001)
        image_size      : Input image will be resized to this

    Activation matching:
        act_layers      : Named layers to hook for activation matching
                          Use model.named_modules() to find valid names
                          Example for VGG16: ['model.features.10', 'model.features.20']
        act_weights     : Per-layer multipliers α_ℓ (must match len(act_layers))
                          Upweight deeper layers for semantic matching

    Quick presets — use CartoonXConfig.preset(name):
        'original'      : Exact paper setup, L2 distortion only
        'kl'            : Paper + KL divergence (recommended default)
        'full'          : KL + activation matching + cross-entropy
        'fast'          : Fewer steps/samples for quick testing
    """

    # ── Loss weights ───────────────────────────────────────────────
    lambda_sparse:  float = 50.0
    lambda_kl:      float = 1.0
    lambda_act:     float = 0.1
    lambda_ce:      float = 0.5

    # ── Loss toggles ───────────────────────────────────────────────
    use_kl:         bool  = True
    use_act_match:  bool  = False
    use_ce:         bool  = False

    # ── DWT ────────────────────────────────────────────────────────
    wavelet:        str   = 'db3'
    n_scales:       int   = 5

    # ── Optimization ───────────────────────────────────────────────
    n_steps:        int   = 2000
    n_samples:      int   = 64
    lr:             float = 1e-3
    image_size:     int   = 256

    # ── Activation matching ────────────────────────────────────────
    act_layers:     list  = field(default_factory=list)
    act_weights:    list  = field(default_factory=list)

    # ── Logging ────────────────────────────────────────────────────
    log_every:      int   = 100

    # ── Validation ────────────────────────────────────────────────
    def __post_init__(self):
        if self.use_act_match and not self.act_layers:
            raise ValueError(
                "use_act_match=True but act_layers is empty.\n"
                "Provide layer names, e.g. act_layers=['model.features.10'].\n"
                "Run: [name for name, _ in model.named_modules()] to list all."
            )
        if self.act_layers and len(self.act_weights) == 0:
            # Auto-fill uniform weights if not specified
            self.act_weights = [1.0] * len(self.act_layers)
        if self.act_layers and len(self.act_weights) != len(self.act_layers):
            raise ValueError(
                f"act_layers has {len(self.act_layers)} entries but "
                f"act_weights has {len(self.act_weights)}. Must match."
            )

    def summary(self) -> str:
        """Print a readable summary of active loss terms."""
        lines = [
            "─" * 45,
            "  CartoonX Configuration",
            "─" * 45,
            f"  DWT:       wavelet={self.wavelet}, scales={self.n_scales}",
            f"  Opt:       steps={self.n_steps}, samples={self.n_samples}, lr={self.lr}",
            "  Losses:",
        ]
        if not self.use_kl:
            lines.append(f"    ✓ L2 distortion (top-class only)")
        if self.use_kl:
            lines.append(f"    ✓ KL divergence        λ={self.lambda_kl}")
        if self.use_act_match:
            lines.append(f"    ✓ Activation matching  λ={self.lambda_act}")
            for name, w in zip(self.act_layers, self.act_weights):
                lines.append(f"        layer={name}  α={w}")
        if self.use_ce:
            lines.append(f"    ✓ Cross-entropy        λ={self.lambda_ce}")
        lines.append(    f"    ✓ Sparsity (L1)        λ={self.lambda_sparse}")
        lines.append("─" * 45)
        return "\n".join(lines)

    @classmethod
    def preset(cls, name: str) -> "CartoonXConfig":
        """
        Factory for common configurations.

        Args:
            name: one of 'original', 'kl', 'full', 'fast'

        Example:
            cfg = CartoonXConfig.preset('kl')
            cfg.lambda_sparse = 80.0   # override any field after
            explainer = CartoonX(model, cfg)
        """
        presets = {
            # ── Exact paper setup ──────────────────────────────────
            'original': dict(
                use_kl        = False,
                use_act_match = False,
                use_ce        = False,
                lambda_sparse = 50.0,
                n_steps       = 2000,
                n_samples     = 64,
            ),
            # ── Paper + KL (recommended, nearly free) ─────────────
            'kl': dict(
                use_kl        = True,
                use_act_match = False,
                use_ce        = False,
                lambda_kl     = 1.0,
                lambda_sparse = 50.0,
                n_steps       = 2000,
                n_samples     = 64,
            ),
            # ── Full extended losses (no act_layers — set manually) ─
            'full': dict(
                use_kl        = True,
                use_act_match = False,   # set act_layers then flip to True
                use_ce        = True,
                lambda_kl     = 1.0,
                lambda_ce     = 0.5,
                lambda_sparse = 50.0,
                n_steps       = 2000,
                n_samples     = 64,
            ),
            # ── Quick test (reduced compute) ───────────────────────
            'fast': dict(
                use_kl        = True,
                use_act_match = False,
                use_ce        = False,
                lambda_kl     = 1.0,
                lambda_sparse = 50.0,
                n_steps       = 300,
                n_samples     = 16,
            ),
        }

        if name not in presets:
            raise ValueError(f"Unknown preset '{name}'. Choose from: {list(presets)}")
        return cls(**presets[name])

    @classmethod
    def from_yaml(cls, path: str) -> "CartoonXConfig":
        """
        Load config from a YAML file.

        If the file contains a 'preset' key, that preset is loaded first,
        then any other fields in the file override the preset values.
        Fields not present in the file keep their preset/default values.

        Example:
            cfg = CartoonXConfig.from_yaml('cartoonx_config.yaml')
            explainer = CartoonX(model, cfg)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Config file not found: {path}\n"
                f"Generate a template with: CartoonXConfig().to_yaml('{path}')"
            )

        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        # Strip comments-only keys and None values for missing fields
        data = {k: v for k, v in data.items() if v is not None or k not in ['preset']}

        # If a preset is specified, start from it then apply overrides
        preset_name = data.pop('preset', None)
        if preset_name:
            base = cls.preset(preset_name)
            base_dict = asdict(base)
            base_dict.update(data)   # file values override preset
            data = base_dict

        # Convert lists to plain lists (yaml sometimes gives other iterables)
        if 'act_layers' in data and data['act_layers'] is None:
            data['act_layers'] = []
        if 'act_weights' in data and data['act_weights'] is None:
            data['act_weights'] = []

        return cls(**data)

    def to_yaml(self, path: str):
        """
        Save current config to a YAML file.

        Useful for recording exactly what config produced a given explanation,
        or generating a starting template to edit.

        Example:
            cfg = CartoonXConfig.preset('kl')
            cfg.to_yaml('my_experiment.yaml')
            # Edit my_experiment.yaml, then reload:
            cfg2 = CartoonXConfig.from_yaml('my_experiment.yaml')
        """
        d = asdict(self)
        d['preset'] = None   # explicit null — loading won't override fields

        header = (
            "# CartoonX Configuration\n"
            f"# Saved from CartoonXConfig.to_yaml()\n"
            "# Load with: cfg = CartoonXConfig.from_yaml(this_file)\n\n"
        )

        with open(path, 'w') as f:
            f.write(header)
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)

        print(f"Config saved to {path}")


# ─────────────────────────────────────────────
#  1. ADAPTIVE GAUSSIAN NOISE (per DWT scale)
# ─────────────────────────────────────────────

def adaptive_gaussian_noise(
    yl: torch.Tensor,           # coarse approx coefficients  [B, C, H, W]
    yh: list[torch.Tensor],     # detail coefficients per scale, each [B, C, 3, H, W]
    n_samples: int = 64
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Sample adaptive Gaussian noise with per-scale statistics.

    For each DWT scale, compute empirical mean and std of coefficients,
    then sample noise v ~ N(mu_scale, sigma_scale²).

    This is critical: using global noise stats would drown out fine-scale
    detail coefficients (small magnitude) with noise calibrated to coarse
    coefficients (large magnitude).

    Args:
        yl:        Coarse approximation coefficients  shape [1, C, H_J, W_J]
        yh:        List of detail coefficients, one per scale j=1..J
                   Each has shape [1, C, 3, H_j, W_j]  (3 = LH, HL, HH)
        n_samples: Number of noise samples L to draw

    Returns:
        (vl, vh_list): Same structure as input but with shape [L, ...]
    """

    def _sample(coeff: torch.Tensor) -> torch.Tensor:
        """Sample L noise vectors with same shape as coeff (minus batch dim)."""
        c = coeff.squeeze(0)                # remove batch dim  [..., H, W]
        mu    = c.mean()
        sigma = c.std() + 1e-8
        return torch.randn(n_samples, *c.shape, device=coeff.device) * sigma + mu

    vl = _sample(yl)                            # [L, C, H_J, W_J]
    vh = [_sample(d) for d in yh]               # each [L, C, 3, H_j, W_j]

    return vl, vh


# ─────────────────────────────────────────────
#  2. OBFUSCATION: build y from mask + noise
# ─────────────────────────────────────────────

def build_obfuscation(
    yl: torch.Tensor,
    yh: list[torch.Tensor],
    sl: torch.Tensor,
    sh: list[torch.Tensor],
    vl: torch.Tensor,
    vh: list[torch.Tensor],
    idwt: DWTInverse
) -> torch.Tensor:
    """
    Construct the obfuscated image y from masked wavelet coefficients.

    For each coefficient position i:
        h_obs_i = s_i * h_i  +  (1 - s_i) * v_i

    Then reconstruct to pixel space via inverse DWT.

    Args:
        yl, yh:     Original DWT coefficients (fixed, shape [1, ...])
        sl, sh:     Mask values in [0,1] (same spatial shape as yl, yh)
        vl, vh:     Noise samples (shape [L, ...])
        idwt:       Inverse DWT module

    Returns:
        y: Obfuscated images in pixel space, shape [L, C, H, W], clipped to [0,1]
    """
    L = vl.shape[0]

    # Broadcast original coefficients to [L, ...] for batch operation
    yl_exp = yl.expand(L, *yl.shape[1:])
    yh_exp = [d.expand(L, *d.shape[1:]) for d in yh]

    # sl, sh have shape matching yl, yh — expand to [L, ...]
    sl_exp = sl.expand(L, *sl.shape[1:])
    sh_exp = [m.expand(L, *m.shape[1:]) for m in sh]

    # Masked interpolation: keep selected, replace unselected with noise
    yl_obs = sl_exp * yl_exp + (1 - sl_exp) * vl
    yh_obs = [
        sh_exp[j] * yh_exp[j] + (1 - sh_exp[j]) * vh[j]
        for j in range(len(yh))
    ]

    # Reconstruct to pixel space
    y = idwt((yl_obs, yh_obs))       # [L, C, H, W]
    y = y.clamp(0.0, 1.0)
    return y


# ─────────────────────────────────────────────
#  3. LOSS FUNCTIONS
# ─────────────────────────────────────────────

class CartoonXLoss(nn.Module):
    """
    Composite loss for CartoonX mask optimization.

    Four components (each optional):

    ① Distortion (original CartoonX):
        d(Φ(x), Φ(y)) = (p_j*(x) - p_j*(y))²
        Matches top-class softmax probability. This is the original paper loss.

    ② KL Divergence (upgrade to ①):
        D_KL(softmax(Φ(x)) || softmax(Φ(y)))
        Matches the FULL probability distribution, not just top class.
        Strictly more informative than ① — detects when other class probs shift.
        Nearly free to add (same forward passes, different function of outputs).

    ③ Activation Matching:
        Σ_ℓ α_ℓ · ||φ_ℓ(x) - φ_ℓ(y)||²
        Matches internal layer activations. Ensures the explanation preserves
        the network's REASONING PATH, not just its final answer.
        Catches cases where two very different internal computations
        accidentally produce similar final outputs.

    ④ Cross-Entropy:
        -log p_{j*}(y)
        Forces the explanation to INDEPENDENTLY predict the correct class
        with high confidence. Makes explanations more class-discriminative
        but can over-constrain ambiguous images.

    Sparsity (always on):
        λ · ||s||₁
        The core CartoonX constraint — forces the mask to be sparse.

    Args:
        use_kl:          Replace/supplement L2 distortion with KL divergence
        use_act_match:   Use activation matching loss
        use_ce:          Use cross-entropy loss
        lambda_sparse:   Weight for sparsity penalty (λ in the paper)
        lambda_kl:       Weight for KL divergence term
        lambda_act:      Weight for activation matching term
        lambda_ce:       Weight for cross-entropy term
        act_layers:      Layer names to extract activations from (for act matching)
        act_weights:     Per-layer weights α_ℓ for activation matching
    """

    def __init__(self, cfg: CartoonXConfig):
        super().__init__()
        self.use_kl        = cfg.use_kl
        self.use_act_match = cfg.use_act_match
        self.use_ce        = cfg.use_ce
        self.lambda_sparse = cfg.lambda_sparse
        self.lambda_kl     = cfg.lambda_kl
        self.lambda_act    = cfg.lambda_act
        self.lambda_ce     = cfg.lambda_ce
        self.act_layers    = cfg.act_layers
        self.act_weights   = cfg.act_weights

        # Registered hooks for activation extraction
        self._hooks   = []
        self._acts_x  = {}   # activations for original image x
        self._acts_y  = {}   # activations for obfuscation y

    # ── Hook management ──────────────────────────────────────────────

    def register_hooks(self, model: nn.Module):
        """
        Register forward hooks on named layers to capture activations.
        Call this once before optimization begins.

        Example layer names for VGG16:
            ['features.10', 'features.20', 'features.30']
        """
        for name, module in model.named_modules():
            if name in self.act_layers:
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        """Create a forward hook that stores activations by layer name."""
        def hook(module, input, output):
            # During original pass: store in _acts_x
            # During obfuscation pass: store in _acts_y
            # We use a flag set externally to distinguish
            if self._capturing_x:
                self._acts_x[name] = output
            else:
                self._acts_y[name] = output
        return hook

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ── Forward ──────────────────────────────────────────────────────

    def forward(
        self,
        model:    nn.Module,
        x:        torch.Tensor,   # original image     [1, C, H, W]
        y_batch:  torch.Tensor,   # L obfuscations     [L, C, H, W]
        sl:       torch.Tensor,   # coarse mask
        sh:       list[torch.Tensor],  # detail masks per scale
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute the full composite loss.

        Returns:
            total_loss: scalar tensor (differentiable w.r.t. sl, sh)
            info:       dict of individual loss components for logging
        """
        L = y_batch.shape[0]

        # ── Run original image through model ─────────────────────────
        self._capturing_x = True
        with torch.no_grad():
            logits_x = model(x)                      # [1, num_classes]
        prob_x   = F.softmax(logits_x, dim=-1)       # [1, num_classes]
        j_star   = prob_x.argmax(dim=-1)             # predicted class index

        # ── Run all L obfuscations through model ─────────────────────
        self._capturing_x = False
        logits_y = model(y_batch)                    # [L, num_classes]
        prob_y   = F.softmax(logits_y, dim=-1)       # [L, num_classes]

        info = {}
        total_loss = torch.tensor(0.0, device=x.device)

        # ─────────────────────────────────────────────────────────────
        # ① Original distortion: squared L2 on top-class probability
        # ─────────────────────────────────────────────────────────────
        if not self.use_kl:
            p_x = prob_x[0, j_star].detach()
            p_y = prob_y[:, j_star]                  # [L]
            distortion = ((p_x - p_y) ** 2).mean()
            total_loss = total_loss + distortion
            info['distortion_l2'] = distortion.item()

        # ─────────────────────────────────────────────────────────────
        # ② KL Divergence: full distribution matching
        #    D_KL(p_x || p_y) = Σ p_x * log(p_x / p_y)
        #
        #    Why better than ①:
        #    - Penalizes any shift in the probability distribution
        #    - Not just the top-1 class, but all class relationships
        #    - Catches cases where p(dog) stays same but p(wolf) surges
        # ─────────────────────────────────────────────────────────────
        if self.use_kl:
            prob_x_exp = prob_x.expand(L, -1).detach()  # [L, num_classes]
            # F.kl_div expects log-probabilities as input
            kl = F.kl_div(
                prob_y.log().clamp(min=-100),        # log q(y) — predicted
                prob_x_exp,                          # p(x)     — target
                reduction='batchmean',
                log_target=False
            )
            total_loss = total_loss + self.lambda_kl * kl
            info['kl_div'] = kl.item()

        # ─────────────────────────────────────────────────────────────
        # ③ Activation Matching: intermediate layer supervision
        #    L_act = Σ_ℓ α_ℓ · ||φ_ℓ(x) - φ_ℓ(y)||²
        #
        #    Why useful:
        #    - Supervises the network's internal reasoning path
        #    - Two images with same final softmax can have completely
        #      different intermediate representations
        #    - Critical for faithful explanations of misclassifications
        #
        #    Cost: requires storing activations for L=64 obfuscations
        #    Recommendation: use only 2-3 key layers
        # ─────────────────────────────────────────────────────────────
        if self.use_act_match and self.act_layers:
            act_loss = torch.tensor(0.0, device=x.device)

            # Capture original activations (already done in forward pass above)
            # Now we need per-obfuscation activations — expensive!
            # To keep memory manageable, process obfuscations one at a time
            for layer_name, alpha in zip(self.act_layers, self.act_weights):
                act_x = self._acts_x.get(layer_name)
                act_y = self._acts_y.get(layer_name)

                if act_x is not None and act_y is not None:
                    # act_x: [1, C_ℓ, H_ℓ, W_ℓ]
                    # act_y: [L, C_ℓ, H_ℓ, W_ℓ]
                    act_x_exp = act_x.detach().expand(L, -1, -1, -1)
                    layer_loss = F.mse_loss(act_y, act_x_exp)
                    act_loss   = act_loss + alpha * layer_loss

            total_loss = total_loss + self.lambda_act * act_loss
            info['act_match'] = act_loss.item()

        # ─────────────────────────────────────────────────────────────
        # ④ Cross-Entropy: class-discriminative explanation
        #    L_CE = -log p_{j*}(y)
        #
        #    Why useful:
        #    - Forces explanation to INDEPENDENTLY classify correctly
        #    - Makes explanations more discriminative (what makes THIS class unique)
        #
        #    Why to use carefully:
        #    - Can over-constrain ambiguous images
        #    - May cause mask to include extra discriminative features
        #      beyond the minimal sufficient set
        # ─────────────────────────────────────────────────────────────
        if self.use_ce:
            targets = j_star.expand(L)               # [L] — same class for all
            ce = F.cross_entropy(logits_y, targets)
            total_loss = total_loss + self.lambda_ce * ce
            info['cross_entropy'] = ce.item()

        # ─────────────────────────────────────────────────────────────
        # Sparsity: L1 penalty on mask
        # Combines both coarse and all detail masks
        # ─────────────────────────────────────────────────────────────
        all_mask_values = [sl] + sh
        sparsity = sum(m.abs().mean() for m in all_mask_values) / len(all_mask_values)
        total_loss = total_loss + self.lambda_sparse * sparsity
        info['sparsity'] = sparsity.item()

        info['total'] = total_loss.item()
        return total_loss, info


# ─────────────────────────────────────────────
#  4. MAIN CARTOONX OPTIMIZER
# ─────────────────────────────────────────────

class CartoonX:
    """
    CartoonX: wavelet-domain explanation for image classifiers.

    Finds the minimal set of DWT coefficients that, when kept,
    preserves the classifier's output — producing a piece-wise smooth
    (cartoon-like) explanation image.

    Usage:
        model    = models.vgg16(pretrained=True).eval()
        explainer = CartoonX(model, device='cuda')
        explanation, mask = explainer.explain(image_tensor)
        explainer.visualize(image_tensor, explanation)
    """

    def __init__(
        self,
        model:  nn.Module,
        cfg:    CartoonXConfig = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Args:
            model:  Pretrained classifier (wrap with NormalizedModel for ImageNet)
            cfg:    CartoonXConfig controlling all hyperparameters.
                    Defaults to CartoonXConfig.preset('kl') if not provided.
            device: 'cuda' or 'cpu'

        Examples:
            # Quickstart with defaults
            explainer = CartoonX(model)

            # Use a preset
            explainer = CartoonX(model, CartoonXConfig.preset('original'))

            # Custom config
            cfg = CartoonXConfig(
                lambda_sparse = 80.0,
                lambda_kl     = 2.0,
                use_ce        = True,
                lambda_ce     = 0.3,
                n_steps       = 1000,
            )
            explainer = CartoonX(model, cfg)

            # Start from preset, tweak one thing
            cfg = CartoonXConfig.preset('kl')
            cfg.lambda_sparse = 100.0
            explainer = CartoonX(model, cfg)
        """
        if cfg is None:
            cfg = CartoonXConfig.preset('kl')

        self.cfg    = cfg
        self.model  = model.to(device).eval()
        self.device = device

        print(cfg.summary())

        # DWT / IDWT modules (from pytorch_wavelets, supports autograd)
        self.dwt  = DWTForward(J=cfg.n_scales, wave=cfg.wavelet, mode='zero').to(device)
        self.idwt = DWTInverse(wave=cfg.wavelet, mode='zero').to(device)

        self.n_steps   = cfg.n_steps
        self.n_samples = cfg.n_samples
        self.lr        = cfg.lr
        self.log_every = cfg.log_every

        # Build composite loss from config
        self.loss_fn = CartoonXLoss(cfg)

        # Register activation hooks if needed
        if cfg.use_act_match and cfg.act_layers:
            self.loss_fn.register_hooks(self.model)

    def explain(
        self,
        x: torch.Tensor   # input image, shape [1, C, H, W], values in [0, 1]
    ) -> tuple[torch.Tensor, tuple]:
        """
        Run CartoonX optimization for a single image.

        Returns:
            explanation: Piece-wise smooth explanation image [1, C, H, W]
            mask:        (sl, sh) — learned mask in wavelet domain
        """
        x = x.to(self.device)
        assert x.shape[0] == 1, "CartoonX processes one image at a time"

        # ── Step 1: Compute DWT of input image ───────────────────────
        # yl:  coarse approximation  [1, C, H_J, W_J]
        # yh:  list of J detail tensors, each [1, C, 3, H_j, W_j]
        #      axis 2 = 3 orientations: LH (horizontal), HL (vertical), HH (diagonal)
        with torch.no_grad():
            yl, yh = self.dwt(x)

        # ── Step 2: Initialize masks to 1 (keep everything) ──────────
        # sl: mask for coarse coefficients, shape [1, C, H_J, W_J]
        # sh: list of masks matching each detail subband
        sl = torch.ones_like(yl, requires_grad=True)
        sh = [torch.ones_like(d, requires_grad=True) for d in yh]

        # ── Step 3: Set up Adam optimizer on mask parameters ─────────
        optimizer = Adam([sl] + sh, lr=self.lr)

        # ── Step 4: Optimization loop ─────────────────────────────────
        print(f"Starting CartoonX optimization: {self.n_steps} steps, "
              f"{self.n_samples} samples/step")
        print(f"Loss config: KL={self.loss_fn.use_kl}, "
              f"ActMatch={self.loss_fn.use_act_match}, "
              f"CE={self.loss_fn.use_ce}")

        history = []

        for step in range(self.n_steps):
            optimizer.zero_grad()

            # ── Sample L adaptive Gaussian noise vectors ──────────────
            # Noise statistics computed per DWT scale (Section 4.1)
            with torch.no_grad():
                vl, vh = adaptive_gaussian_noise(yl, yh, n_samples=self.n_samples)

            # ── Build L obfuscated images ─────────────────────────────
            # y = IDWT(s⊙h + (1-s)⊙v)
            # sl/sh are clamped to [0,1] before use
            sl_clamped = sl.clamp(0, 1)
            sh_clamped = [m.clamp(0, 1) for m in sh]

            y_batch = build_obfuscation(
                yl, yh,
                sl_clamped, sh_clamped,
                vl, vh,
                self.idwt
            )   # [L, C, H, W]

            # ── Compute composite loss ────────────────────────────────
            loss, info = self.loss_fn(
                self.model, x, y_batch,
                sl_clamped, sh_clamped
            )

            # ── Backprop through IDWT → mask ──────────────────────────
            # Gradient path: loss → Φ(y) → y → IDWT → (s⊙h + (1-s)⊙v) → s
            # IDWT is linear so gradients flow cleanly
            loss.backward()
            optimizer.step()

            # Project mask back to [0, 1] after Adam step
            with torch.no_grad():
                sl.clamp_(0, 1)
                for m in sh:
                    m.clamp_(0, 1)

            history.append(info)

            if step % self.log_every == 0:
                log_str = f"Step {step:4d}/{self.n_steps} | "
                log_str += " | ".join(f"{k}: {v:.4f}" for k, v in info.items())
                print(log_str)

        # ── Step 5: Build explanation image ───────────────────────────
        # Apply learned mask to grayscale DWT coefficients, invert to pixel space
        explanation = self._build_explanation(x, sl, sh)

        return explanation, (sl.detach(), [m.detach() for m in sh]), history

    def _build_explanation(
        self,
        x:  torch.Tensor,
        sl: torch.Tensor,
        sh: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Reconstruct the CartoonX explanation image.

        Algorithm (from paper Section 5.1):
        1. Convert input to grayscale
        2. Compute DWT of grayscale image
        3. Multiply DWT coefficients by learned mask
        4. Invert back to pixel space with IDWT
        5. Clip to [0, 1]

        The grayscale visualization is used because the mask operates
        jointly across color channels — a single mask entry covers all
        channels simultaneously. Grayscale avoids color artifacts.
        """
        with torch.no_grad():
            # Convert to grayscale: standard luminance weights
            gray_weights = torch.tensor([0.299, 0.587, 0.114],
                                        device=self.device).view(1, 3, 1, 1)
            x_gray = (x * gray_weights).sum(dim=1, keepdim=True)  # [1, 1, H, W]

            # DWT of grayscale
            yl_gray, yh_gray = self.dwt(x_gray)

            # Apply mask to grayscale coefficients
            sl_clamped = sl.clamp(0, 1)
            sh_clamped = [m.clamp(0, 1) for m in sh]

            # Handle channel dimension mismatch (mask has C=3, gray has C=1)
            # Average mask across color channels
            sl_gray = sl_clamped.mean(dim=1, keepdim=True)
            sh_gray = [m.mean(dim=1, keepdim=True) for m in sh_clamped]

            # Mask the grayscale coefficients
            yl_masked = sl_gray * yl_gray
            yh_masked = [sh_gray[j] * yh_gray[j] for j in range(len(yh_gray))]

            # Inverse DWT → explanation in pixel space
            explanation = self.idwt((yl_masked, yh_masked))  # [1, 1, H, W]
            explanation = explanation.clamp(0, 1)

        return explanation

    def visualize(
        self,
        x:           torch.Tensor,
        explanation: torch.Tensor,
        mask:        Optional[tuple] = None,
        save_path:   Optional[str] = None
    ):
        """
        Side-by-side visualization of input and CartoonX explanation.

        Optionally shows the mask values per DWT scale.
        """
        x_np   = x[0].cpu().permute(1, 2, 0).numpy()
        exp_np = explanation[0, 0].cpu().numpy()

        if mask is not None:
            sl, sh = mask
            n_cols = 3 + len(sh)
            fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

            axes[0].imshow(x_np.clip(0, 1))
            axes[0].set_title('Input Image', fontsize=12)
            axes[0].axis('off')

            axes[1].imshow(exp_np, cmap='gray')
            axes[1].set_title('CartoonX Explanation', fontsize=12)
            axes[1].axis('off')

            # Visualize coarse mask
            sl_np = sl[0].mean(0).cpu().numpy()
            axes[2].imshow(sl_np, cmap='hot', vmin=0, vmax=1)
            axes[2].set_title(f'Coarse Mask\n(LL subband)\nsparsity={1-sl_np.mean():.2f}',
                              fontsize=9)
            axes[2].axis('off')

            # Visualize detail masks per scale
            for j, m in enumerate(sh):
                m_np = m[0].mean(0).mean(0).cpu().numpy()  # avg over channels and orientations
                axes[3 + j].imshow(m_np, cmap='hot', vmin=0, vmax=1)
                axes[3 + j].set_title(
                    f'Detail Mask\nScale {j+1}\nsparsity={1-m_np.mean():.2f}',
                    fontsize=9
                )
                axes[3 + j].axis('off')

        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(x_np.clip(0, 1))
            axes[0].set_title('Input Image', fontsize=14)
            axes[0].axis('off')

            axes[1].imshow(exp_np, cmap='gray')
            axes[1].set_title('CartoonX Explanation', fontsize=14)
            axes[1].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        plt.show()

    def plot_loss_history(self, history: list[dict]):
        """Plot training loss curves for all active components."""
        keys = [k for k in history[0].keys() if k != 'total']
        n    = len(keys) + 1

        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

        # Total loss
        axes[0].plot([h['total'] for h in history], color='black', linewidth=2)
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Step')
        axes[0].grid(True, alpha=0.3)

        # Individual components
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        for i, (key, color) in enumerate(zip(keys, colors)):
            axes[i + 1].plot(
                [h[key] for h in history],
                color=color, linewidth=1.5
            )
            axes[i + 1].set_title(key.replace('_', ' ').title())
            axes[i + 1].set_xlabel('Step')
            axes[i + 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def cleanup(self):
        """Remove activation hooks when done."""
        self.loss_fn.remove_hooks()


# ─────────────────────────────────────────────
#  5. PREPROCESSING UTILITIES
# ─────────────────────────────────────────────

def load_image(path: str, size: int = 256) -> torch.Tensor:
    """
    Load and preprocess an image for CartoonX.

    Returns tensor of shape [1, 3, size, size] with values in [0, 1].
    Note: CartoonX does NOT use ImageNet normalization — pixel values
    must remain in [0, 1] for the wavelet domain and clipping to work correctly.
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),          # → [0, 1], no normalization
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)  # [1, 3, H, W]


def imagenet_preprocess(x: torch.Tensor) -> torch.Tensor:
    """
    Apply ImageNet normalization ONLY for classifier forward passes.

    CartoonX stores the mask in [0,1] wavelet space, but most pretrained
    classifiers expect ImageNet-normalized inputs. This function normalizes
    on the fly during classifier calls.

    Usage: wrap your model forward call:
        logits = model(imagenet_preprocess(y_batch))
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


class NormalizedModel(nn.Module):
    """
    Wrapper that applies ImageNet normalization inside the model.

    Allows CartoonX to work with [0,1] pixel values throughout the
    optimization, while the underlying classifier receives normalized inputs.

    Usage:
        base_model     = models.vgg16(pretrained=True)
        model          = NormalizedModel(base_model)
        cartoonx       = CartoonX(model, ...)
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer('mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


# ─────────────────────────────────────────────
#  6. EXAMPLE USAGE
# ─────────────────────────────────────────────

def run_example(image_path: str):
    """
    Full CartoonX pipeline demonstrating the CartoonXConfig API.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x          = load_image(image_path).to(device)
    base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model      = NormalizedModel(base_model).to(device).eval()

    # ── Config A: Original CartoonX (preset) ─────────────────────────
    cfg_A = CartoonXConfig.preset('original')
    exp_A, mask_A, _ = CartoonX(model, cfg_A, device).explain(x)

    # ── Config B: KL divergence (preset, then tweak one weight) ──────
    cfg_B = CartoonXConfig.preset('kl')
    cfg_B.lambda_sparse = 80.0      # override a single field after preset
    exp_B, mask_B, _ = CartoonX(model, cfg_B, device).explain(x)

    # ── Config C: Full custom config ──────────────────────────────────
    cfg_C = CartoonXConfig(
        use_kl        = True,
        use_act_match = True,
        use_ce        = True,
        lambda_sparse = 50.0,
        lambda_kl     = 1.5,
        lambda_act    = 0.05,
        lambda_ce     = 0.3,
        act_layers    = ['model.features.10', 'model.features.20', 'model.features.29'],
        act_weights   = [0.5, 1.0, 1.0],
        n_steps       = 2000,
        n_samples     = 64,
        wavelet       = 'db3',
        n_scales      = 5,
    )
    explainer_C = CartoonX(model, cfg_C, device)
    exp_C, mask_C, hist_C = explainer_C.explain(x)
    explainer_C.plot_loss_history(hist_C)
    explainer_C.cleanup()

    # ── Config D: Fast test preset ────────────────────────────────────
    cfg_D = CartoonXConfig.preset('fast')
    exp_D, mask_D, _ = CartoonX(model, cfg_D, device).explain(x)

    # ── Compare all four ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    x_np = x[0].cpu().permute(1, 2, 0).numpy()
    axes[0].imshow(x_np.clip(0, 1)); axes[0].set_title('Input'); axes[0].axis('off')
    for i, (exp, title) in enumerate(zip(
        [exp_A, exp_B, exp_C, exp_D],
        ['Original', 'KL (λ_s=80)', 'Full Custom', 'Fast Preset']
    )):
        axes[i+1].imshow(exp[0,0].cpu().numpy(), cmap='gray')
        axes[i+1].set_title(title, fontsize=10)
        axes[i+1].axis('off')

    plt.suptitle('CartoonX Config Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cartoonx_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    return exp_A, exp_B, exp_C, exp_D


if __name__ == '__main__':
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'your_image.jpg'
    run_example(image_path)
