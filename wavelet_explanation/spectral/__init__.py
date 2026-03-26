"""
spectral — Radial Spectral Mask explanation package.

Public API:
    RadialSpectralMaskNet          — per-image radial freq-domain mask predictor
    SpectralExplanationTrainer     — standalone training loop
    construct_spectral_explanation — FFT → mask → IFFT reconstruction
    perturb_spectral_explanation   — frequency-domain noise perturbation
"""

from spectral.fft_mask import (
    RadialSpectralMaskNet,
    UNetSpectralMaskNet,
    GlobalFCSpectralMaskNet,
    make_radial_freq_grid,
    STE,
)
from spectral.explanation import construct_spectral_explanation, perturb_spectral_explanation
from spectral.trainer import SpectralExplanationTrainer

__all__ = [
    "RadialSpectralMaskNet",
    "UNetSpectralMaskNet",
    "GlobalFCSpectralMaskNet",
    "make_radial_freq_grid",
    "STE",
    "construct_spectral_explanation",
    "perturb_spectral_explanation",
    "SpectralExplanationTrainer",
]
