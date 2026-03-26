"""
Tests for construct_explanation and perturb_explanation.
"""

import torch
import pytest
from wavelet.dwt import HaarDWT
from wavelet.explanation import construct_explanation, perturb_explanation, connectivity_filter


@pytest.fixture
def dwt():
    return HaarDWT()


def _binary_masks(B, H, W):
    """Return four random binary masks at subband resolution (H//2, W//2)."""
    masks = []
    for _ in range(4):
        m = (torch.rand(B, 1, H // 2, W // 2) > 0.5).float()
        masks.append(m)
    return tuple(masks)


class TestConstructExplanation:

    def test_explanation_has_same_shape_as_input(self, dwt):
        B, C, H, W = 2, 3, 64, 64
        x = torch.randn(B, C, H, W)
        masks = _binary_masks(B, H, W)
        e, _ = construct_explanation(x, *masks, dwt)
        assert e.shape == x.shape

    def test_full_mask_recovers_input(self, dwt):
        """When all masks are 1, the explanation should equal x (up to numerical error)."""
        B, C, H, W = 2, 3, 32, 32
        x = torch.randn(B, C, H, W)
        ones = torch.ones(B, 1, H // 2, W // 2)
        e, _ = construct_explanation(x, ones, ones, ones, ones, dwt)
        max_err = (e - x).abs().max().item()
        assert max_err < 1e-4, f"Full mask reconstruction error: {max_err}"

    def test_zero_mask_gives_zero_explanation(self, dwt):
        """Zero masks → zero explanation (DWT of x masked to zero, IDWT of zeros = 0)."""
        B, C, H, W = 2, 3, 32, 32
        x = torch.randn(B, C, H, W)
        zeros = torch.zeros(B, 1, H // 2, W // 2)
        e, _ = construct_explanation(x, zeros, zeros, zeros, zeros, dwt)
        max_val = e.abs().max().item()
        assert max_val < 1e-5, f"Zero mask should give zero explanation, got max={max_val}"

    def test_masked_subbands_returned(self, dwt):
        """construct_explanation returns a dict of 4 named masked subbands."""
        B, C, H, W = 2, 3, 32, 32
        x = torch.randn(B, C, H, W)
        masks = _binary_masks(B, H, W)
        _, masked = construct_explanation(x, *masks, dwt)
        assert set(masked.keys()) == {"LL", "LH", "HL", "HH"}
        for sb in masked.values():
            assert sb.shape == (B, C, H // 2, W // 2)

    def test_gradients_flow_through_explanation(self, dwt):
        """Gradients from a downstream loss should flow back to the masks."""
        B, C, H, W = 2, 3, 32, 32
        x = torch.randn(B, C, H, W)
        # Use leaf tensors so that .grad is populated after backward.
        masks = [torch.rand(B, 1, H // 2, W // 2).requires_grad_(True) for _ in range(4)]
        e, _ = construct_explanation(x, *masks, dwt)
        e.sum().backward()
        for m in masks:
            assert m.grad is not None, "Gradient did not reach mask"


class TestPerturbExplanation:

    def test_perturbed_has_same_shape(self, dwt):
        B, C, H, W = 2, 3, 32, 32
        x = torch.randn(B, C, H, W)
        masks = _binary_masks(B, H, W)
        e_pert = perturb_explanation(x, *masks, dwt)
        assert e_pert.shape == x.shape

    def test_perturbed_differs_from_clean(self, dwt):
        """With partial masks (not all-ones), the perturbed explanation should differ from clean."""
        B, C, H, W = 2, 3, 32, 32
        x = torch.randn(B, C, H, W)
        # Make half the coefficients zero (masked out), so noise is injected
        zeros = torch.zeros(B, 1, H // 2, W // 2)
        ones  = torch.ones( B, 1, H // 2, W // 2)
        e_clean, _ = construct_explanation(x, ones, ones, ones, ones, dwt)
        e_pert = perturb_explanation(x, zeros, zeros, zeros, zeros, dwt)
        # Perturbation replaces all subbands with noise → result should differ
        assert not torch.allclose(e_clean, e_pert)


class TestConnectivityFilter:

    def test_disabled_when_min_neighbors_zero(self):
        e = torch.randn(1, 3, 8, 8)
        assert torch.equal(connectivity_filter(e, 0), e)

    def test_isolated_pixel_is_zeroed(self):
        # Single active pixel at (4, 4) with all surrounding pixels off
        e = torch.zeros(1, 3, 8, 8)
        e[0, :, 4, 4] = 1.0
        out = connectivity_filter(e, min_neighbors=2)
        assert out.abs().sum() == 0.0, "isolated pixel should be removed"

    def test_pixel_with_sufficient_neighbors_is_kept(self):
        # 3×3 block of active pixels — centre pixel has 8 neighbours, all kept
        e = torch.zeros(1, 3, 8, 8)
        e[0, :, 3:6, 3:6] = 1.0
        out = connectivity_filter(e, min_neighbors=2)
        # Centre of the block (4,4) must survive
        assert out[0, 0, 4, 4].item() == 1.0

    def test_output_shape_unchanged(self):
        e = torch.randn(2, 3, 16, 16)
        out = connectivity_filter(e, min_neighbors=1)
        assert out.shape == e.shape

    def test_gradients_flow_through_kept_pixels(self):
        # 3×3 block — gradients should reach the centre pixel
        e = torch.zeros(1, 3, 8, 8, requires_grad=False)
        e[0, :, 3:6, 3:6] = 1.0
        e = e.detach().requires_grad_(True)
        out = connectivity_filter(e, min_neighbors=2)
        loss = out.sum()
        loss.backward()
        assert e.grad is not None
        assert e.grad[0, 0, 4, 4].item() != 0.0, "gradient must flow through kept pixels"

    def test_threshold_treats_dim_pixels_as_inactive(self):
        # One bright isolated pixel (value 1.0) and one dim isolated pixel (value 0.02)
        # With threshold=0.05, the dim pixel is inactive → gets zeroed by neighbour check
        # The bright pixel is also isolated so it gets zeroed too
        e = torch.zeros(1, 1, 8, 8)
        e[0, 0, 1, 1] = 1.0    # bright, isolated
        e[0, 0, 6, 6] = 0.02   # dim (< 5% of peak = 0.05), isolated
        out = connectivity_filter(e, min_neighbors=1, active_threshold=0.05)
        # Both are isolated so both should be zero
        assert out.abs().sum() == 0.0

    def test_threshold_keeps_dim_pixels_when_they_have_bright_neighbours(self):
        # 3×3 block: centre bright, surround dim but above threshold (0.1 > 0.05 * 1.0)
        e = torch.zeros(1, 1, 8, 8)
        e[0, 0, 3:6, 3:6] = 0.1   # 10% of peak (peak = 0.1 here) → above 5% threshold
        out = connectivity_filter(e, min_neighbors=2, active_threshold=0.05)
        assert out[0, 0, 4, 4].item() == pytest.approx(0.1)

    def test_threshold_zero_matches_strict_nonzero(self):
        e = torch.zeros(1, 3, 8, 8)
        e[0, :, 4, 4] = 1.0
        assert torch.equal(
            connectivity_filter(e, min_neighbors=2, active_threshold=0.0),
            connectivity_filter(e, min_neighbors=2),
        )
