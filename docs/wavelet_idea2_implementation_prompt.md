# Implementation Prompt: Wavelet-Domain Mask Generation for Minimal Explanation Generation

## Your Task

You are to implement from scratch a complete, runnable PyTorch codebase for generating minimal and faithful explanations of pretrained image classifiers using **wavelet-domain mask generation**. This is a research implementation based on the paper "Minimalist Explanation Generation and Circuit Discovery" (Suhail et al., IIT Bombay), with a novel modification that replaces pixel-space masking with wavelet-domain masking for richer interpretability.

The full codebase must be well-structured, documented, and runnable end-to-end on ImageNet-pretrained classifiers (ResNet-18 as primary) as well as MNIST custom CNNs.

---

## Background: The Original Method

The original paper trains a lightweight U-Net autoencoder to produce a binary pixel mask `m` for an input image `x`. The explanation is `e = m ⊙ x`. The autoencoder is trained (while the classifier is frozen) using a composite loss with three groups:

**1. Activation Matching + Output Fidelity (L_AM):**
- `L_act`: Weighted MSE between post-ReLU feature maps φₗ(x) and φₗ(e) at each convolutional layer; cosine similarity for linear layers. Summed across all layers with per-layer weights αₗ.
- `L_CE`: Cross-entropy loss to preserve top-1 predicted class label y.
- `L_KL`: KL divergence between softmax(f(x)) and softmax(f(e)) to match full output distributions.

**2. Mask Minimality Priors (L_MIN):**
- `L_area = ||m||₁`: Penalizes number of active pixels.
- `L_bin = ||m - m²||₁`: Pushes mask values toward {0,1}. Uses Straight-Through Estimator (STE) for backprop through binarization.
- `L_tv`: Total variation loss for spatial coherence of mask.

**3. Robustness Constraint (L_ROB):**
- Perturbs background with Gaussian noise: `ẽ = m ⊙ x + (1-m) ⊙ r`, then enforces `L_rob = -log p_f(ẽ)(y)`.

**Full objective:** `L_EXP = L_AM + L_MIN + L_ROB`

---

## Your Modification: Idea 2 — Wavelet-Domain Mask Generation

Instead of learning one pixel-space mask, you will learn **four separate binary masks in the wavelet domain**, one per subband: {mLL, mLH, mHL, mHH}.

### Core Change in Explanation Construction

Instead of `e = m ⊙ x`:

```
x → DWT → {x_LL, x_LH, x_HL, x_HH}    (one level Haar decomposition)
           ↓
    Apply subband masks:
    e_LL = mLL ⊙ x_LL
    e_LH = mLH ⊙ x_LH
    e_HL = mHL ⊙ x_HL
    e_HH = mHH ⊙ x_HH
           ↓
    e = IDWT(e_LL, e_LH, e_HL, e_HH)    (reconstruct explanation in pixel space)
```

Everything downstream — activation matching, cross entropy, KL divergence, robustness — operates on `e` in pixel space, exactly as in the original. The only things that change are:
1. The encoder-decoder output head (1 channel → 4 channels)
2. How the explanation `e` is constructed (pixel multiply → wavelet mask + IDWT)
3. The mask priors (per-subband area loss with different weights; TV loss removed)
4. The robustness perturbation (subband-wise noise instead of pixel-space Gaussian)

---

## Detailed File Structure to Implement

```
wavelet_explanation/
├── models/
│   ├── __init__.py
│   ├── unet.py               # U-Net encoder-decoder with 4-channel output head
│   ├── classifiers.py        # Wrapper around frozen pretrained classifiers
│   └── custom_cnn.py         # 3-layer custom CNN for MNIST experiments
├── losses/
│   ├── __init__.py
│   ├── activation_matching.py  # L_act with MSE for conv, cosine for linear
│   ├── output_fidelity.py      # L_CE and L_KL
│   └── mask_priors.py          # L_area (per subband), L_bin, no TV
├── wavelet/
│   ├── __init__.py
│   ├── dwt.py                # DWT/IDWT using pytorch_wavelets or manual Haar
│   └── explanation.py        # construct_explanation() and perturb_explanation()
├── training/
│   ├── __init__.py
│   ├── trainer.py            # Main training loop
│   └── hooks.py              # Forward hooks to tap activations from frozen classifier
├── evaluation/
│   ├── __init__.py
│   └── metrics.py            # Sparsity, confidence delta, label preservation rate
├── visualization/
│   ├── __init__.py
│   └── visualize.py          # Plot original/masks/explanation, subband masks separately
├── configs/
│   ├── resnet18_imagenet.yaml
│   └── cnn_mnist.yaml
├── train.py                  # Entry point
├── evaluate.py               # Evaluation entry point
└── requirements.txt
```

---

## Detailed Specification for Each Component

### 1. `wavelet/dwt.py`

Implement a `HaarDWT` class with `forward()` and `inverse()` methods.

- Use **Haar wavelet** (simplest, most interpretable)
- One level of decomposition (J=1)
- Input: `(B, C, H, W)` image tensor
- `forward()` returns:
  - `x_LL`: shape `(B, C, H//2, W//2)` — low-low (coarse structure)
  - `x_LH`: shape `(B, C, H//2, W//2)` — low-high (horizontal edges)
  - `x_HL`: shape `(B, C, H//2, W//2)` — high-low (vertical edges)
  - `x_HH`: shape `(B, C, H//2, W//2)` — high-high (fine texture)
- `inverse()` takes `(x_LL, x_LH, x_HL, x_HH)` and returns `(B, C, H, W)`

Implement using the Haar filter bank manually with `F.conv2d` so there is no external wavelet library dependency (making it easier to run). The Haar filters are:

```
Low-pass:  h_low  = [1, 1] / sqrt(2)
High-pass: h_high = [1, -1] / sqrt(2)

2D decomposition = separable application of 1D filters:
  LL = h_low(rows)  ⊗ h_low(cols)   applied with stride 2
  LH = h_low(rows)  ⊗ h_high(cols)  applied with stride 2
  HL = h_high(rows) ⊗ h_low(cols)   applied with stride 2
  HH = h_high(rows) ⊗ h_high(cols)  applied with stride 2
```

For the inverse, use the corresponding transpose convolutions.

Also try to support `pytorch_wavelets` as an optional faster backend if installed, with a graceful fallback to the manual implementation.

---

### 2. `wavelet/explanation.py`

Implement two functions:

**`construct_explanation(x, m_LL, m_LH, m_HL, m_HH, dwt)`:**
```
- Decompose x via dwt.forward(x) → x_LL, x_LH, x_HL, x_HH
- Each mask is (B, 1, H//2, W//2), expand to (B, C, H//2, W//2) to match subbands
- Apply masks: e_LL = m_LL * x_LL, etc.
- Reconstruct: e = dwt.inverse(e_LL, e_LH, e_HL, e_HH)
- Return e (B,C,H,W) and also the individual masked subbands for visualization
```

**`perturb_explanation(x, m_LL, m_LH, m_HL, m_HH, dwt)`:**
```
- Decompose x → x_LL, x_LH, x_HL, x_HH
- Sample Gaussian noise per subband: r_LL, r_LH, r_HL, r_HH (same shapes)
- Perturb: e_pert_LL = m_LL * x_LL + (1 - m_LL) * r_LL, etc.
- Reconstruct: e_perturbed = dwt.inverse(e_pert_LL, e_pert_LH, e_pert_HL, e_pert_HH)
- Return e_perturbed
```

---

### 3. `models/unet.py`

Implement a lightweight U-Net with:

**Encoder:**
- 4 downsampling blocks
- Block structure: `Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → MaxPool2d`
- Channels: 3 → 32 → 64 → 128 → 256

**Bottleneck:**
- Two conv layers at 256 channels

**Decoder:**
- 4 upsampling blocks using `ConvTranspose2d` for upsampling
- Skip connections from encoder (U-Net style concatenation)
- Channels mirror encoder in reverse

**Output head (MODIFIED from original):**
```python
self.final_conv = nn.Conv2d(32, 4, kernel_size=1)  # 4 subbands
```

**Forward method:**
```python
def forward(self, x):
    # ... unet body ...
    out = torch.sigmoid(self.final_conv(features))  # (B, 4, H, W)
    
    # Split into 4 subband masks and downsample to H//2, W//2
    # since wavelet subbands are half spatial resolution
    out_down = F.avg_pool2d(out, 2)  # (B, 4, H//2, W//2)
    
    m_LL = out_down[:, 0:1, :, :]
    m_LH = out_down[:, 1:2, :, :]
    m_HL = out_down[:, 2:3, :, :]
    m_HH = out_down[:, 3:4, :, :]
    
    # Apply STE binarization to each
    m_LL_bin = STE.apply(m_LL)
    m_LH_bin = STE.apply(m_LH)
    m_HL_bin = STE.apply(m_HL)
    m_HH_bin = STE.apply(m_HH)
    
    return m_LL_bin, m_LH_bin, m_HL_bin, m_HH_bin
```

**STE (Straight-Through Estimator):**
```python
class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0.5).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # pass gradient through unchanged
```

---

### 4. `models/classifiers.py`

Implement a `FrozenClassifier` wrapper:

```python
class FrozenClassifier(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained=True):
        # Load pretrained model
        # Call model.eval() and freeze all parameters
        # Register forward hooks on ALL layers to capture activations
        
    def get_activations(self):
        # Return dict of {layer_name: activation_tensor}
        # Activations are captured by hooks during forward pass
        
    def forward(self, x):
        # Standard forward pass, hooks collect activations automatically
        # Return final logits
```

Support these backbones: `resnet18`, `mobilenet_v3_small`, `convnext_small`, `efficientnet_b0`, `vit_b_16`. All loaded from `torchvision.models` with `pretrained=True` weights.

For activation tapping, use `register_forward_hook` on:
- All `nn.Conv2d` layers (use MSE for matching)
- All `nn.Linear` layers (use cosine similarity for matching)
- All `nn.ReLU` layers to get post-ReLU activations

Store layer type alongside activation so loss computation knows which distance metric to use.

---

### 5. `models/custom_cnn.py`

Implement a simple 3-layer CNN for MNIST:

```python
class MNISTClassifier(nn.Module):
    def __init__(self):
        # Conv(1,32,3) → ReLU → MaxPool
        # Conv(32,64,3) → ReLU → MaxPool  
        # Flatten → Linear(64*5*5, 128) → ReLU → Linear(128, 10)
```

Also provide a training script `train_mnist_classifier()` to train this from scratch on MNIST and save weights.

---

### 6. `losses/activation_matching.py`

```python
def activation_matching_loss(activations_x, activations_e, layer_weights=None):
    """
    activations_x: dict of {layer_name: (tensor, layer_type)}
    activations_e: dict of {layer_name: (tensor, layer_type)}
    layer_weights: dict of {layer_name: float} or None (uniform)
    
    For conv layers: MSE between spatial feature maps
    For linear layers: 1 - cosine_similarity (so 0 = perfect match)
    
    Returns weighted sum across all layers.
    """
```

Default layer weights αₗ: use uniform weights as default, but support a `depth_weighted` option that increases weight linearly with layer depth (to emphasize deeper semantic features).

---

### 7. `losses/output_fidelity.py`

```python
def cross_entropy_loss(logits_e, y):
    """Standard CE loss: -log p_f(e)(y)"""

def kl_divergence_loss(logits_x, logits_e):
    """KL(softmax(f(x)) || softmax(f(e)))"""
    # Use F.kl_div with log_target=False
    # Remember KL div in PyTorch expects log-probabilities as input
```

---

### 8. `losses/mask_priors.py`

```python
def area_loss(m_LL, m_LH, m_HL, m_HH,
              lambda_LL=5.0, lambda_LH=10.0,
              lambda_HL=10.0, lambda_HH=20.0):
    """
    Per-subband L1 sparsity penalty.
    HH weighted most heavily (fine texture usually least important).
    LL weighted least (coarse structure often more important to preserve).
    """

def binarization_loss(m_LL, m_LH, m_HL, m_HH):
    """
    ||m - m^2||_1 summed across all four subbands.
    Pushes values toward 0 or 1.
    """

# NOTE: NO total variation loss. 
# Wavelet decomposition already separates scales.
# Heavy lambda_HH penalization replaces what TV was doing.
```

---

### 9. `training/hooks.py`

Implement the hook registration and management:

```python
class ActivationHookManager:
    def __init__(self, model):
        self.hooks = []
        self.activations = {}
        self._register_hooks(model)
    
    def _register_hooks(self, model):
        # Register hooks on all Conv2d, Linear, ReLU layers
        # Store layer type for distance metric selection
    
    def get_activations(self):
        return self.activations.copy()
    
    def clear(self):
        self.activations = {}
    
    def remove(self):
        for h in self.hooks:
            h.remove()
```

---

### 10. `training/trainer.py`

Implement the main training loop. This is the central piece that assembles everything:

```python
class WaveletExplanationTrainer:
    def __init__(self, config):
        self.config = config
        self.dwt = HaarDWT()
        self.encoder_decoder = UNet()
        self.classifier = FrozenClassifier(config.backbone)
        self.hook_manager = ActivationHookManager(self.classifier)
        self.optimizer = torch.optim.Adam(
            self.encoder_decoder.parameters(),
            lr=config.lr
        )
    
    def train_step(self, x, y):
        """
        Complete single training iteration.
        
        Steps:
        1. Forward pass through encoder-decoder → m_LL, m_LH, m_HL, m_HH
        2. Construct explanation e via wavelet masking
        3. Forward pass of x through frozen classifier (hooks capture activations_x)
        4. Forward pass of e through frozen classifier (hooks capture activations_e)
        5. Compute L_act from stored activations
        6. Compute L_CE, L_KL from classifier outputs
        7. Compute perturbed explanation e_pert
        8. Forward pass of e_pert through classifier
        9. Compute L_rob
        10. Compute L_area, L_bin from subband masks
        11. Assemble L_EXP, backward, optimizer step
        12. Return dict of all individual loss values for logging
        """
    
    def train_epoch(self, dataloader):
        """Run train_step for all batches, return average losses."""
    
    def save_checkpoint(self, path):
        """Save encoder-decoder weights + config."""
    
    def load_checkpoint(self, path):
        """Load encoder-decoder weights."""
```

**Important implementation details for the training loop:**

- Call `self.hook_manager.clear()` before each forward pass to avoid stale activations
- Run `x` through classifier first, copy activations, run `e` through classifier, copy activations, then compute L_act
- The classifier must be called with `torch.no_grad()` context for its parameters, but activations still need to be in the computational graph for backprop through `e`. Handle this carefully — freeze params via `requires_grad=False` on classifier parameters, NOT via `torch.no_grad()` context manager.
- Gradient clipping: clip encoder-decoder gradients to max norm 1.0

---

### 11. `configs/resnet18_imagenet.yaml`

```yaml
backbone: resnet18
pretrained: true
dataset: imagenet
image_size: 224
batch_size: 8
lr: 1e-4
epochs: 50

# Loss weights
lambda_act: 1.0
lambda_CE: 4.0
lambda_KL: 0.4
lambda_rob: 6.0
lambda_bin: 0.3

# Per-subband area loss weights
lambda_area_LL: 5.0
lambda_area_LH: 10.0
lambda_area_HL: 10.0
lambda_area_HH: 20.0

# Circuit discovery
top_k_channels: 10

# Wavelet
wavelet: haar
levels: 1
```

```yaml
# configs/cnn_mnist.yaml
backbone: custom_cnn
pretrained: false
dataset: mnist
image_size: 28
batch_size: 32
lr: 1e-3
epochs: 100

lambda_act: 0.6
lambda_CE: 4.0
lambda_KL: 0.54
lambda_rob: 10.0
lambda_bin: 1.2

lambda_area_LL: 20.0
lambda_area_LH: 40.0
lambda_area_HL: 40.0
lambda_area_HH: 100.0

wavelet: haar
levels: 1
```

---

### 12. `evaluation/metrics.py`

```python
def compute_sparsity(m_LL, m_LH, m_HL, m_HH):
    """
    Returns:
    - overall_sparsity: fraction of total subband coefficients set to zero
    - per_subband_sparsity: dict {LL, LH, HL, HH} with individual sparsities
    - pixel_equivalent_sparsity: approximate fraction of original pixels masked
      (since each subband coefficient corresponds to a 2x2 pixel region)
    """

def compute_confidence_delta(classifier, x, e, y):
    """
    Returns p_f(e)(y) - p_f(x)(y)
    Positive = explanation increased confidence (background was hurting)
    Negative = explanation decreased confidence (mask too aggressive)
    """

def compute_label_preservation_rate(classifier, x_batch, e_batch):
    """
    Fraction of examples where argmax(f(x)) == argmax(f(e)).
    Should be close to 1.0 for good explanations.
    """

def compute_subband_activity_profile(m_LL, m_LH, m_HL, m_HH):
    """
    Returns mean activity per subband as interpretable frequency profile.
    Used to characterize whether classifier is texture-biased vs shape-biased.
    """
```

---

### 13. `visualization/visualize.py`

Implement the following visualization functions:

```python
def visualize_explanation(x, m_LL, m_LH, m_HL, m_HH, e, save_path=None):
    """
    3-row figure:
    Row 1: Original image
    Row 2: Four subband masks side by side (mLL, mLH, mHL, mHH)
            with titles: "LL (Coarse)", "LH (Horiz. Edges)", 
                         "HL (Vert. Edges)", "HH (Texture)"
    Row 3: Final explanation e
    """

def visualize_subband_decomposition(x, dwt, save_path=None):
    """
    Show the raw wavelet decomposition of x:
    Original | x_LL | x_LH | x_HL | x_HH
    Useful for understanding what each subband captures.
    """

def visualize_frequency_profile(activity_profiles, class_names, save_path=None):
    """
    Bar chart showing mean LL/LH/HL/HH mask activity per class.
    Reveals per-class frequency biases of the classifier.
    """

def compare_with_pixel_mask(x, e_wavelet, e_pixel, save_path=None):
    """
    Side-by-side: Original | Wavelet Explanation | Pixel-space Explanation
    For direct visual comparison with the baseline method.
    """
```

---

### 14. `train.py` (Entry Point)

```python
import argparse
import yaml
from training.trainer import WaveletExplanationTrainer
from torch.utils.data import DataLoader
import torchvision

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup dataset (ImageNet or MNIST based on config)
    # Setup trainer
    # Training loop with periodic visualization and checkpoint saving
    # Final evaluation with metrics

if __name__ == '__main__':
    main()
```

---

## Implementation Notes and Gotchas

### Spatial Resolution Handling

The DWT halves spatial resolution: `(B, C, H, W) → (B, C, H//2, W//2)` per subband.

The U-Net outputs masks at full resolution `(B, 4, H, W)`. You must downsample masks to `(B, 4, H//2, W//2)` before applying to subbands. Use `F.avg_pool2d` with stride 2, applied after sigmoid but before STE.

For MNIST (28×28), the subbands will be 14×14. For ImageNet (224×224), subbands will be 112×112.

### Gradient Flow Verification

After implementing, verify gradient flow is working correctly by checking that `encoder_decoder` parameters have non-None `.grad` after `L_EXP.backward()`. Add an assertion in debug mode:

```python
for name, param in encoder_decoder.named_parameters():
    assert param.grad is not None, f"No gradient for {name}"
```

### Classifier Parameter Freezing

Freeze the classifier correctly:
```python
for param in classifier.parameters():
    param.requires_grad = False
```
Do NOT wrap in `torch.no_grad()` — you need activations to be part of the computational graph for gradients to flow back through `e` to the encoder-decoder.

### Batch Normalization in Frozen Classifier

Call `classifier.eval()` at initialization and never call `classifier.train()`. This ensures BatchNorm uses stored running statistics rather than batch statistics during explanation training.

### STE and Float Precision

The STE forward returns `(x > 0.5).float()`. Make sure the threshold 0.5 is appropriate. If masks are collapsing to all-zeros or all-ones early in training, try initializing the final conv layer with small weights or using a threshold-warmup schedule.

### Wavelet Reconstruction Artifacts

For natural images, one level of Haar decomposition and reconstruction should be near-lossless. Verify by checking `||x - IDWT(DWT(x))||` is below 1e-5. If not, your filter implementation has a bug.

---

## Testing Checklist

Write tests in a `tests/` directory covering:

1. `test_dwt.py`: Test that `IDWT(DWT(x)) ≈ x` for random tensors
2. `test_unet.py`: Test output shapes are correct for various input sizes
3. `test_losses.py`: Test each loss is non-negative and differentiable
4. `test_explanation.py`: Test explanation e has same shape as x
5. `test_trainer.py`: Test one training step runs without error and produces non-None gradients
6. `test_metrics.py`: Test metrics return sensible values on trivial inputs

---

## Expected Results to Verify Correctness

When training on ImageNet with ResNet-18:

- After training, `pixel_equivalent_sparsity` should be in the range **3–10%** (comparable to the paper's ~5%)
- `label_preservation_rate` should exceed **0.85**
- `confidence_delta` should be **positive on average** (explanation confidence > original)
- `mHH` mask should typically be **sparser than mLL** mask (texture less important than coarse structure for most natural image classes)
- For texture-heavy classes (e.g., fabrics, dog breeds), expect higher `mHH` activity relative to `mLL`
- For geometric classes (e.g., basketball, aircraft), expect higher `mLL` activity relative to `mHH`

---

## Requirements (`requirements.txt`)

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.23.0
matplotlib>=3.6.0
PyYAML>=6.0
tqdm>=4.65.0
Pillow>=9.0.0
pytorch_wavelets>=1.3.0   # optional, falls back to manual Haar if not installed
pytest>=7.0.0
```

---

## Deliverables

Produce all files described in the file structure above. Every file must be fully implemented — no placeholder `pass` statements in functional code. Include docstrings for all classes and public methods. The code must run end-to-end with the command:

```bash
python train.py --config configs/resnet18_imagenet.yaml --data_path /path/to/imagenet
```

and for MNIST:

```bash
python train.py --config configs/cnn_mnist.yaml --data_path /path/to/mnist
```
