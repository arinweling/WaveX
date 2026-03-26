"""
Visualisation utilities for wavelet-domain explanations.

All functions accept a single example (the first item in a batch) and
optionally save figures to disk.

Functions:
    visualize_explanation          — original | 4 subband masks | explanation
    visualize_subband_decomposition — raw wavelet decomposition of an image
    visualize_frequency_profile    — bar chart of per-class subband activity
    compare_with_pixel_mask        — side-by-side wavelet vs pixel-space explanation
"""

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; switch to TkAgg/Qt5Agg for interactive use
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """
    Convert a (C, H, W) or (1, H, W) tensor to a displayable numpy array.
    Handles both grayscale and RGB images.
    """
    t = t.detach().cpu().float()
    if t.ndim == 3 and t.shape[0] == 3:
        arr = t.permute(1, 2, 0).numpy()           # (H, W, 3)
    elif t.ndim == 3 and t.shape[0] == 1:
        arr = t.squeeze(0).numpy()                 # (H, W)
    else:
        arr = t.numpy()
    # Clip to [0, 1] for display
    return np.clip(arr, 0.0, 1.0)


def _show_image(ax, img_tensor, title: str, cmap=None):
    arr = _to_numpy(img_tensor)
    ax.imshow(arr, cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _save_or_show(fig, save_path: Optional[str]):
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_explanation(
    x: torch.Tensor,
    m_LL: torch.Tensor,
    m_LH: torch.Tensor,
    m_HL: torch.Tensor,
    m_HH: torch.Tensor,
    e: torch.Tensor,
    save_path: Optional[str] = None,
):
    """
    Three-row figure showing the original image, the four subband masks,
    and the reconstructed explanation.

    Row 1: Original image  (1 panel, centred)
    Row 2: Subband masks   (4 panels: LL, LH, HL, HH)
    Row 3: Explanation     (1 panel, centred)

    Args:
        x:    (B, C, H, W) or (C, H, W)  original image
        m_LL, m_LH, m_HL, m_HH: (B,1,H',W') or (1,H',W') binary masks
        e:    (B, C, H, W) or (C, H, W)  explanation
        save_path: if given, save figure there; else displays nothing (Agg backend)
    """
    # Take first element of batch if batched
    x    = x[0]    if x.ndim    == 4 else x
    m_LL = m_LL[0] if m_LL.ndim == 4 else m_LL
    m_LH = m_LH[0] if m_LH.ndim == 4 else m_LH
    m_HL = m_HL[0] if m_HL.ndim == 4 else m_HL
    m_HH = m_HH[0] if m_HH.ndim == 4 else m_HH
    e    = e[0]    if e.ndim    == 4 else e

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle("Wavelet Explanation", fontsize=12, fontweight="bold")

    # Row 1: original (span columns by hiding unused axes and using the leftmost two)
    for ax in axes[0]:
        ax.axis("off")
    # Overwrite the second subplot with the original image stretched across middle
    ax_orig = fig.add_subplot(3, 1, 1)
    arr = _to_numpy(x)
    ax_orig.imshow(arr)
    ax_orig.set_title("Original Image", fontsize=10)
    ax_orig.axis("off")
    # Hide row-0 individual axes (already replaced by the wide subplot above)
    for ax in axes[0]:
        ax.set_visible(False)

    # Row 2: four subband masks
    mask_data = [
        (m_LL, "LL  (Coarse)"),
        (m_LH, "LH  (Horiz. Edges)"),
        (m_HL, "HL  (Vert. Edges)"),
        (m_HH, "HH  (Texture)"),
    ]
    for col, (mask, title) in enumerate(mask_data):
        _show_image(axes[1][col], mask, title, cmap="gray")

    # Row 3: explanation (same approach as row 1)
    for ax in axes[2]:
        ax.set_visible(False)
    ax_exp = fig.add_subplot(3, 1, 3)
    arr_e = _to_numpy(e)
    ax_exp.imshow(arr_e)
    ax_exp.set_title("Explanation  e = IDWT(m ⊙ DWT(x))", fontsize=10)
    ax_exp.axis("off")

    _save_or_show(fig, save_path)


def visualize_subband_attribution(
    x: torch.Tensor,
    m_LL: torch.Tensor,
    m_LH: torch.Tensor,
    m_HL: torch.Tensor,
    m_HH: torch.Tensor,
    e: torch.Tensor,
    save_path: Optional[str] = None,
):
    """
    Colour-coded attribution map showing which subband each pixel's explanation
    contribution came from.

    Each pixel in the explanation is coloured by its dominant active subband:
        LL — blue   (coarse structure)
        LH — green  (horizontal edges)
        HL — red    (vertical edges)
        HH — yellow (fine texture)

    Pixels where no subband mask is active are shown as the greyed-out original.
    Pixel brightness is proportional to the explanation intensity.

    Args:
        x:    (B, C, H, W) or (C, H, W)  original image (denormalised, [0,1])
        m_LL, m_LH, m_HL, m_HH: (B,1,H',W') or (1,H',W') binary masks (subband resolution)
        e:    (B, C, H, W) or (C, H, W)  explanation image (denormalised, [0,1])
        save_path: optional file path
    """
    import torch.nn.functional as F

    # Take first element of batch
    x    = (x[0]    if x.ndim    == 4 else x).detach().cpu().float()
    m_LL = (m_LL[0] if m_LL.ndim == 4 else m_LL).detach().cpu().float()
    m_LH = (m_LH[0] if m_LH.ndim == 4 else m_LH).detach().cpu().float()
    m_HL = (m_HL[0] if m_HL.ndim == 4 else m_HL).detach().cpu().float()
    m_HH = (m_HH[0] if m_HH.ndim == 4 else m_HH).detach().cpu().float()
    e    = (e[0]    if e.ndim    == 4 else e).detach().cpu().float()

    H, W = x.shape[-2], x.shape[-1]

    # Upsample subband masks (H//2, W//2) → (H, W) using nearest neighbour
    # so each subband pixel maps to exactly a 2×2 block (no interpolation blur)
    def _up(m):
        return F.interpolate(m.unsqueeze(0), size=(H, W), mode="nearest").squeeze(0)  # (1, H, W)

    m_LL_up = _up(m_LL)  # (1, H, W) values in {0, 1}
    m_LH_up = _up(m_LH)
    m_HL_up = _up(m_HL)
    m_HH_up = _up(m_HH)

    masks_up = {"LL": m_LL_up, "LH": m_LH_up, "HL": m_HL_up, "HH": m_HH_up}

    # Build an activity map from deviation away from the explanation's
    # neutral baseline colour. Uniform grey regions stay black in the
    # attribution panels even if they are non-zero after denormalisation.
    baseline = e.reshape(e.shape[0], -1).median(dim=1).values.view(e.shape[0], 1, 1)
    e_activity = (e - baseline).abs().mean(dim=0)  # (H, W)
    e_activity = e_activity / (e_activity.max() + 1e-6)
    e_activity = torch.where(e_activity >= 0.05, e_activity, torch.zeros_like(e_activity))

    # ── Figure: 3 rows ────────────────────────────────────────────────────
    # Row 1: Original | Explanation
    # Row 2: one panel per subband (coloured)
    # Row 3: explanation with all masks overlaid
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Subband Attribution Map", fontsize=12, fontweight="bold")

    # Row 1: original and explanation span the first two and last two columns
    for ax in axes[0]:
        ax.axis("off")
    ax_orig = fig.add_subplot(3, 2, 1)
    ax_orig.imshow(_to_numpy(x));  ax_orig.set_title("Original", fontsize=10);    ax_orig.axis("off")
    ax_exp  = fig.add_subplot(3, 2, 2)
    ax_exp.imshow(_to_numpy(e));   ax_exp.set_title("Explanation", fontsize=10);  ax_exp.axis("off")

    # Row 2: 4 individual per-subband maps — black background, no overlap
    subband_info = [
        ("LL", m_LL_up, [0.00, 0.90, 0.90], "LL — coarse structure\n(cyan)"),
        ("LH", m_LH_up, [0.20, 0.80, 0.30], "LH — horiz. edges\n(green)"),
        ("HL", m_HL_up, [0.90, 0.20, 0.20], "HL — vert. edges\n(red)"),
        ("HH", m_HH_up, [0.95, 0.85, 0.10], "HH — fine texture\n(yellow)"),
    ]
    for col, (name, mask_up, colour, title) in enumerate(subband_info):
        c = torch.tensor(colour).view(3, 1, 1)
        panel = torch.zeros(3, H, W)                            # black background
        layer = c * mask_up * e_activity.unsqueeze(0)          # coloured active regions only
        panel = panel + layer                                   # add onto black
        panel = np.clip(panel.permute(1, 2, 0).numpy(), 0, 1)
        axes[1][col].imshow(panel)
        axes[1][col].set_title(title, fontsize=9)
        axes[1][col].axis("off")

    # Row 3: explanation with all masks overlaid as semi-transparent tints
    for ax in axes[2]:
        ax.set_visible(False)
    # Build a combined colour overlay.
    # Overlay: black background. For each active mask pixel paint colour * activity.
    # Neutral grey regions in e are forced to black regardless of mask activity.
    e_activity_np = e_activity.numpy()                      # (H, W) in [0, 1]
    overlay  = np.zeros((H, W, 3), dtype=np.float32)       # start from black
    for _, mask_up, colour, _ in subband_info:
        active   = (mask_up.squeeze(0).numpy() > 0.5) & (e_activity_np > 0.0)
        c_arr    = np.array(colour, dtype=np.float32)      # (3,)
        overlay[active] += c_arr * e_activity_np[active, None]

    ax_overlay = fig.add_subplot(3, 1, 3)
    ax_overlay.imshow(np.clip(overlay, 0, 1))
    ax_overlay.set_title(
        "Explanation + mask overlay  "
        "(cyan=LL  green=LH  red=HL  yellow=HH)",
        fontsize=10,
    )
    ax_overlay.axis("off")

    _save_or_show(fig, save_path)


def visualize_subband_decomposition(
    x: torch.Tensor,
    dwt,
    save_path: Optional[str] = None,
):
    """
    Visualise the raw Haar wavelet subbands of an image.

    Panels: Original | x_LL | x_LH | x_HL | x_HH

    Args:
        x:    (B, C, H, W) or (C, H, W)
        dwt:  HaarDWT instance
        save_path: optional file path for saving
    """
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x_in = x[0:1]   # single example

    with torch.no_grad():
        x_LL, x_LH, x_HL, x_HH = dwt(x_in)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle("Haar Wavelet Decomposition", fontsize=11)

    titles = ["Original", "x_LL  (Coarse)", "x_LH  (Horiz.)", "x_HL  (Vert.)", "x_HH  (Texture)"]
    tensors = [x_in[0], x_LL[0], x_LH[0], x_HL[0], x_HH[0]]

    for ax, t, title in zip(axes, tensors, titles):
        _show_image(ax, t, title)

    _save_or_show(fig, save_path)


def visualize_frequency_profile(
    activity_profiles: Dict[str, Dict[str, float]],
    class_names: List[str],
    save_path: Optional[str] = None,
):
    """
    Grouped bar chart of mean subband mask activity per class.

    Reveals per-class frequency biases:
        - Shape-biased classes → tall LL bars
        - Texture-biased classes → tall HH bars

    Args:
        activity_profiles: {class_name: {'LL': float, 'LH': float, 'HL': float, 'HH': float}}
        class_names:       list of class names to display (must be keys in activity_profiles)
        save_path:         optional save path
    """
    subbands = ["LL", "LH", "HL", "HH"]
    colors   = ["#4878d0", "#ee854a", "#6acc65", "#d65f5f"]

    n_classes = len(class_names)
    x_positions = np.arange(n_classes)
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=(max(8, n_classes * 1.2), 5))

    for i, (subband, color) in enumerate(zip(subbands, colors)):
        values = [activity_profiles.get(cls, {}).get(subband, 0.0) for cls in class_names]
        ax.bar(
            x_positions + i * bar_width,
            values,
            width=bar_width,
            label=f"Mask {subband}",
            color=color,
            alpha=0.85,
        )

    ax.set_xticks(x_positions + bar_width * 1.5)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean Active Fraction")
    ax.set_title("Per-Class Subband Activity Profile")
    ax.legend()
    ax.set_ylim(0, 1)

    _save_or_show(fig, save_path)


def compare_with_pixel_mask(
    x: torch.Tensor,
    e_wavelet: torch.Tensor,
    e_pixel: torch.Tensor,
    save_path: Optional[str] = None,
):
    """
    Side-by-side comparison: Original | Wavelet Explanation | Pixel-Space Explanation.

    Useful for ablation studies comparing the wavelet method to the baseline.

    Args:
        x:         (B,C,H,W) or (C,H,W) original image
        e_wavelet: (B,C,H,W) or (C,H,W) explanation from wavelet masking
        e_pixel:   (B,C,H,W) or (C,H,W) explanation from pixel-space masking
        save_path: optional save path
    """
    x         = x[0]        if x.ndim         == 4 else x
    e_wavelet = e_wavelet[0] if e_wavelet.ndim == 4 else e_wavelet
    e_pixel   = e_pixel[0]   if e_pixel.ndim   == 4 else e_pixel

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Wavelet vs Pixel-Space Explanation", fontsize=11)

    _show_image(axes[0], x,         "Original")
    _show_image(axes[1], e_wavelet, "Wavelet Explanation")
    _show_image(axes[2], e_pixel,   "Pixel-Space Explanation")

    _save_or_show(fig, save_path)
