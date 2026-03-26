"""
Spectral-domain explanation construction and perturbation.

construct_spectral_explanation:
    Applies a binary radial mask to the rfft2 of x and reconstructs via irfft2.
    The result is an artifact-free explanation because IFFT is the exact inverse
    of FFT — unlike Haar IDWT where independent subband masking introduces
    grey halos.

perturb_spectral_explanation:
    Replaces unmasked frequency components with Gaussian noise in the frequency
    domain (used to compute the robustness loss L_rob).
"""

import torch


def construct_spectral_explanation(
    x: torch.Tensor,
    M_bin: torch.Tensor,
) -> torch.Tensor:
    """
    Build the spectral-masked explanation image.

    Process:
        X_freq  = rfft2(x)             (B, C, H, W//2+1)  complex
        X_masked = M_bin * X_freq      zero out unselected frequencies
        e       = irfft2(X_masked)     (B, C, H, W)  exact reconstruction

    Args:
        x:     (B, C, H, W)           original image
        M_bin: (B, 1, H, W//2+1)      binary frequency-domain mask {0, 1}

    Returns:
        e: (B, C, H, W)  spectral explanation in pixel space
    """
    H, W = x.shape[-2], x.shape[-1]
    X_freq   = torch.fft.rfft2(x)                            # (B, C, H, W//2+1) complex
    X_masked = M_bin * X_freq                                 # broadcast over C channels
    e        = torch.fft.irfft2(X_masked, s=(H, W))          # (B, C, H, W)
    return e


def perturb_spectral_explanation(
    x: torch.Tensor,
    M_bin: torch.Tensor,
) -> torch.Tensor:
    """
    Build a perturbed explanation for the robustness loss.

    Unmasked (rejected) frequency components are replaced with Gaussian noise
    sampled from the same distribution as the original coefficients at that
    frequency.  This tests whether the classifier is confused when the
    rejected frequencies are filled with random energy.

    Process for each frequency (u, v):
        If M_bin[u,v] = 1:  keep X_freq[u,v]            (selected)
        If M_bin[u,v] = 0:  replace with complex noise   (rejected)

    Args:
        x:     (B, C, H, W)
        M_bin: (B, 1, H, W//2+1) binary mask

    Returns:
        e_perturbed: (B, C, H, W)
    """
    H, W = x.shape[-2], x.shape[-1]
    X_freq = torch.fft.rfft2(x)  # (B, C, H, W//2+1)

    # Sample complex Gaussian noise scaled to the typical magnitude of X_freq
    mag = X_freq.abs().mean().detach().clamp(min=1e-6)
    noise_real = torch.randn_like(X_freq.real) * mag
    noise_imag = torch.randn_like(X_freq.imag) * mag
    noise_freq = torch.complex(noise_real, noise_imag)

    # Replace unmasked frequencies with noise; keep masked ones intact
    X_perturbed = M_bin * X_freq + (1.0 - M_bin) * noise_freq

    e_perturbed = torch.fft.irfft2(X_perturbed, s=(H, W))
    return e_perturbed
