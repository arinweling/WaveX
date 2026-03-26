"""
Visualization helpers for the spectral explanation module.

visualize_spectral_explanation:
    Shows the original image, the 2D frequency mask (in log-magnitude space),
    and the final explanation side by side.

visualize_radial_profile:
    Plots the learned 1D radial frequency curve — the key
    interpretability output of the radial spectral mask approach.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def visualize_spectral_explanation(
    x: torch.Tensor,
    M_bin: torch.Tensor,
    M_cont: torch.Tensor,
    e: torch.Tensor,
    save_path: str = None,
) -> None:
    """
    3-column figure per image in the batch (up to 4 images):
        Col 1: Original image
        Col 2: Continuous frequency mask M_cont (log-scaled magnitude,
                DC centred via fftshift for readability)
        Col 3: Explanation e

    Args:
        x:         (B, C, H, W) original images, values in [0, 1]
        M_bin:     (B, 1, H, W//2+1) binary mask (rfft layout)
        M_cont:    (B, 1, H, W//2+1) continuous mask (rfft layout)
        e:         (B, C, H, W) explanation images, values in [0, 1]
        save_path: if given, saves the figure to this path
    """
    B   = min(x.shape[0], 4)
    fig = plt.figure(figsize=(12, 3.5 * B), constrained_layout=True)
    fig.suptitle("Spectral Explanation", fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(B, 3, figure=fig)

    col_titles = ["Original", "Freq mask  M_cont  (DC-centred)", "Explanation  e = IFFT(M·FFT(x))"]

    for row in range(B):
        # --- Original ---
        ax0 = fig.add_subplot(gs[row, 0])
        img = x[row].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        ax0.imshow(img if img.shape[-1] == 3 else img[..., 0], cmap="gray")
        ax0.axis("off")
        if row == 0:
            ax0.set_title(col_titles[0], fontsize=10)

        # --- Frequency mask (continuous, DC-centred via fftshift) ---
        ax1 = fig.add_subplot(gs[row, 1])
        # M_cont is (B, 1, H, W//2+1) — rfft layout; reconstruct full-spectrum view
        m = M_cont[row, 0].cpu().numpy()   # (H, W//2+1)
        # Mirror the rfft half to get a (H, W) full-spectrum mask for display
        H_m, Wh = m.shape                   # H, W//2+1
        W_full   = (Wh - 1) * 2
        m_full          = np.zeros((H_m, W_full), dtype=np.float32)
        m_full[:, :Wh]  = m
        m_full[:, Wh:]  = m[:, -2:0:-1]    # mirror (rfft symmetry)
        m_shifted = np.fft.fftshift(m_full)  # DC to centre
        ax1.imshow(m_shifted, cmap="viridis", vmin=0, vmax=1)
        ax1.axis("off")
        if row == 0:
            ax1.set_title(col_titles[1], fontsize=10)

        # --- Explanation ---
        ax2 = fig.add_subplot(gs[row, 2])
        exp = e[row].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        ax2.imshow(exp if exp.shape[-1] == 3 else exp[..., 0], cmap="gray")
        ax2.axis("off")
        if row == 0:
            ax2.set_title(col_titles[2], fontsize=10)

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def visualize_radial_profile(
    radial_profile: torch.Tensor,
    save_path: str = None,
) -> None:
    """
    Plot the learned 1D radial frequency profile for each image in the batch.

    Each curve shows how much of each frequency magnitude bin is retained by
    the mask.  A curve that is high near 0 and drops off is a low-pass
    (shape-biased) explanation; a curve that is high near 1 indicates reliance
    on fine texture.

    Args:
        radial_profile: (B, num_radial_bins) tensor of sigmoid-activated
                        per-bin mask values in [0, 1]
        save_path:      if given, saves the figure to this path
    """
    B, num_bins = radial_profile.shape
    B_plot = min(B, 4)

    fig, ax = plt.subplots(figsize=(8, 4))
    x_axis = np.linspace(0, 1, num_bins)

    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i in range(B_plot):
        profile = radial_profile[i].detach().cpu().numpy()
        ax.plot(x_axis, profile, color=colours[i % len(colours)],
                linewidth=2, label=f"Image {i+1}")

    ax.set_xlabel("Normalised radial frequency  (0 = DC,  1 = Nyquist)", fontsize=11)
    ax.set_ylabel("Mask value  (1 = keep,  0 = remove)", fontsize=11)
    ax.set_title("Learned Radial Frequency Profile", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.axvline(x=0.0,  color="grey", linestyle=":", linewidth=1, alpha=0.5)
    ax.axvline(x=0.5,  color="grey", linestyle="--", linewidth=1, alpha=0.5, label="Nyquist/2")
    ax.axvline(x=1.0,  color="grey", linestyle=":", linewidth=1, alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
