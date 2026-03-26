"""
Tests for the spectral attention mask module.

Run:
    cd wavelet_explanation && pytest tests/test_spectral.py -v
"""

import pytest
import torch

from spectral.fft_mask import RadialSpectralMaskNet, make_radial_freq_grid, STE
from spectral.explanation import construct_spectral_explanation, perturb_spectral_explanation
from spectral.losses import spectral_area_loss, spectral_smoothness_loss, binarization_loss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_batch():
    """Tiny (2, 3, 32, 32) batch for fast CPU tests."""
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def mask_net_small():
    """RadialSpectralMaskNet configured for 32×32 input."""
    return RadialSpectralMaskNet(in_channels=3, H=32, W=32, num_radial_bins=16)


# ---------------------------------------------------------------------------
# 1. FFT round-trip — IFFT(FFT(x)) ≈ x
# ---------------------------------------------------------------------------

def test_fft_round_trip(small_batch):
    x = small_batch
    H, W = x.shape[-2], x.shape[-1]
    X_freq = torch.fft.rfft2(x)
    x_rec  = torch.fft.irfft2(X_freq, s=(H, W))
    assert x_rec.shape == x.shape, "Round-trip shape mismatch"
    assert (x - x_rec).abs().max().item() < 1e-5, "Round-trip error too large — FFT not lossless"


# ---------------------------------------------------------------------------
# 2. Radial frequency grid
# ---------------------------------------------------------------------------

def test_radial_freq_grid_shape():
    H, W = 32, 32
    grid = make_radial_freq_grid(H, W)
    assert grid.shape == (H, W // 2 + 1), f"Expected ({H}, {W//2+1}), got {grid.shape}"


def test_radial_freq_grid_range():
    grid = make_radial_freq_grid(32, 32)
    assert grid.min().item() >= 0.0, "Grid contains negative values"
    assert grid.max().item() <= 1.0 + 1e-6, "Grid exceeds 1.0"
    # DC should be at or near 0
    assert grid[0, 0].item() < 0.05, "DC component (0,0) should be near 0"


# ---------------------------------------------------------------------------
# 3. RadialSpectralMaskNet — output shapes
# ---------------------------------------------------------------------------

def test_mask_net_output_shapes(mask_net_small, small_batch):
    M_cont, M_bin, radial_profile = mask_net_small(small_batch)

    B, C, H, W = small_batch.shape
    expected_freq_shape = (B, 1, H, W // 2 + 1)

    assert M_cont.shape == expected_freq_shape, \
        f"M_cont shape mismatch: expected {expected_freq_shape}, got {M_cont.shape}"
    assert M_bin.shape == expected_freq_shape, \
        f"M_bin shape mismatch: expected {expected_freq_shape}, got {M_bin.shape}"
    assert radial_profile.shape == (B, 16), \
        f"radial_profile shape mismatch: expected ({B}, 16), got {radial_profile.shape}"


def test_mask_net_cont_in_unit_interval(mask_net_small, small_batch):
    M_cont, _, _ = mask_net_small(small_batch)
    assert M_cont.min().item() >= 0.0, "M_cont has negative values"
    assert M_cont.max().item() <= 1.0 + 1e-6, "M_cont exceeds 1.0"


def test_mask_net_bin_is_binary(mask_net_small, small_batch):
    _, M_bin, _ = mask_net_small(small_batch)
    unique_vals = M_bin.unique().tolist()
    for v in unique_vals:
        assert v in (0.0, 1.0), f"M_bin contains non-binary value {v}"


# ---------------------------------------------------------------------------
# 4. construct_spectral_explanation — output shape and value range
# ---------------------------------------------------------------------------

def test_construct_explanation_shape(mask_net_small, small_batch):
    _, M_bin, _ = mask_net_small(small_batch)
    e = construct_spectral_explanation(small_batch, M_bin)
    assert e.shape == small_batch.shape, \
        f"Explanation shape mismatch: expected {small_batch.shape}, got {e.shape}"


def test_construct_explanation_all_mask_on(small_batch):
    """With a fully-on mask (all-ones), e should equal x (up to fp precision)."""
    H, W = small_batch.shape[-2], small_batch.shape[-1]
    M_all_on = torch.ones(small_batch.shape[0], 1, H, W // 2 + 1)
    e = construct_spectral_explanation(small_batch, M_all_on)
    assert (small_batch - e).abs().max().item() < 1e-4, \
        "All-ones mask did not reproduce original image"


def test_construct_explanation_all_mask_off(small_batch):
    """With a fully-off mask (all-zeros), e should be near zero."""
    H, W = small_batch.shape[-2], small_batch.shape[-1]
    M_all_off = torch.zeros(small_batch.shape[0], 1, H, W // 2 + 1)
    e = construct_spectral_explanation(small_batch, M_all_off)
    assert e.abs().max().item() < 1e-4, "All-zeros mask should give near-zero explanation"


# ---------------------------------------------------------------------------
# 5. perturb_spectral_explanation — shape only (stochastic output)
# ---------------------------------------------------------------------------

def test_perturb_explanation_shape(mask_net_small, small_batch):
    _, M_bin, _ = mask_net_small(small_batch)
    e_pert = perturb_spectral_explanation(small_batch, M_bin)
    assert e_pert.shape == small_batch.shape, \
        f"Perturbed explanation shape mismatch: {e_pert.shape}"


# ---------------------------------------------------------------------------
# 6. Losses — non-negative and differentiable
# ---------------------------------------------------------------------------

def test_spectral_area_loss_non_negative(mask_net_small, small_batch):
    M_cont, _, _ = mask_net_small(small_batch)
    freq_grid = mask_net_small.freq_grid
    loss = spectral_area_loss(M_cont, freq_grid)
    assert loss.item() >= 0.0, "spectral_area_loss should be non-negative"


def test_spectral_smoothness_loss_non_negative(mask_net_small, small_batch):
    M_cont, _, _ = mask_net_small(small_batch)
    loss = spectral_smoothness_loss(M_cont)
    assert loss.item() >= 0.0, "spectral_smoothness_loss should be non-negative"


def test_binarization_loss_non_negative(mask_net_small, small_batch):
    M_cont, _, _ = mask_net_small(small_batch)
    loss = binarization_loss(M_cont)
    assert loss.item() >= 0.0, "binarization_loss should be non-negative"


def test_losses_differentiable(mask_net_small, small_batch):
    """All losses must produce non-None gradients on mask_net parameters."""
    M_cont, M_bin, _ = mask_net_small(small_batch)
    freq_grid = mask_net_small.freq_grid

    l = (
        spectral_area_loss(M_cont, freq_grid)
        + spectral_smoothness_loss(M_cont)
        + binarization_loss(M_cont)
    )
    l.backward()

    for name, param in mask_net_small.named_parameters():
        assert param.grad is not None, f"No gradient for parameter '{name}'"


# ---------------------------------------------------------------------------
# 7. STE — gradient pass-through
# ---------------------------------------------------------------------------

def test_ste_forward_binarises():
    x = torch.tensor([0.3, 0.5, 0.7, 1.0])
    out = STE.apply(x)
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0])  # threshold at >0.5
    assert torch.allclose(out, expected), f"STE forward mismatch: {out}"


def test_ste_gradient_passthrough():
    x = torch.tensor([0.3, 0.7], requires_grad=True)
    out = STE.apply(x)
    out.sum().backward()
    assert x.grad is not None, "STE should pass gradients through"
    assert torch.allclose(x.grad, torch.ones_like(x)), "STE gradient should be all-ones"
