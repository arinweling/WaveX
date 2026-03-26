"""
Tests for all loss functions.

Each loss must:
  1. Return a non-negative scalar tensor.
  2. Be differentiable (gradient flows back to an upstream parameter).
"""

import torch
import torch.nn.functional as F
import pytest
from losses.activation_matching import activation_matching_loss
from losses.output_fidelity import cross_entropy_loss, kl_divergence_loss
from losses.mask_priors import area_loss, binarization_loss, total_variation_loss, explanation_area_loss, explanation_tv_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_activations(batch=2, channels=16, H=8, W=8):
    """Return a minimal activations dict with one conv and one linear layer."""
    conv_act  = torch.randn(batch, channels, H, W, requires_grad=True)
    lin_act   = torch.randn(batch, 32,        requires_grad=True)
    return {
        "conv1":  (conv_act, "conv"),
        "linear1":(lin_act,  "linear"),
    }


def _random_masks(B=2, H=14, W=14):
    """Continuous masks in (0,1) as leaf tensors with gradient tracking."""
    # Must be leaf tensors (not results of ops) so that .grad is populated.
    masks = [torch.rand(B, 1, H, W).requires_grad_(True) for _ in range(4)]
    return tuple(masks)


# ---------------------------------------------------------------------------
# Activation matching
# ---------------------------------------------------------------------------

class TestActivationMatchingLoss:

    def test_identical_activations_give_low_loss(self):
        acts = _random_activations()
        loss = activation_matching_loss(acts, acts)
        # MSE of identical tensors = 0; cosine sim of identical vectors = 1 → dist = 0
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_different_activations_give_positive_loss(self):
        acts_x = _random_activations()
        acts_e = _random_activations()
        loss = activation_matching_loss(acts_x, acts_e)
        assert loss.item() > 0.0

    def test_loss_is_scalar(self):
        acts_x = _random_activations()
        acts_e = _random_activations()
        loss = activation_matching_loss(acts_x, acts_e)
        assert loss.ndim == 0

    def test_gradients_flow_through_e_activations(self):
        acts_x = _random_activations()

        # Use a fresh tensor with grad for e activations
        conv_e  = torch.randn(2, 16, 8, 8, requires_grad=True)
        lin_e   = torch.randn(2, 32, requires_grad=True)
        acts_e = {"conv1": (conv_e, "conv"), "linear1": (lin_e, "linear")}

        loss = activation_matching_loss(acts_x, acts_e)
        loss.backward()
        assert conv_e.grad is not None
        assert lin_e.grad is not None

    def test_depth_weighted_scheme(self):
        acts_x = _random_activations()
        acts_e = _random_activations()
        loss = activation_matching_loss(acts_x, acts_e, weight_scheme="depth_weighted")
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# Output fidelity
# ---------------------------------------------------------------------------

class TestOutputFidelityLosses:

    def test_ce_loss_nonneg(self):
        logits = torch.randn(4, 10, requires_grad=True)
        y = torch.randint(0, 10, (4,))
        loss = cross_entropy_loss(logits, y)
        assert loss.item() >= 0.0
        loss.backward()
        assert logits.grad is not None

    def test_ce_loss_zero_on_perfect_prediction(self):
        """A very confident correct prediction should produce near-zero CE loss."""
        logits = torch.zeros(2, 5)
        logits[0, 2] = 100.0
        logits[1, 3] = 100.0
        y = torch.tensor([2, 3])
        loss = cross_entropy_loss(logits, y)
        assert loss.item() < 0.01

    def test_kl_loss_nonneg(self):
        logits_x = torch.randn(4, 10)
        logits_e = torch.randn(4, 10, requires_grad=True)
        loss = kl_divergence_loss(logits_x, logits_e)
        assert loss.item() >= 0.0
        loss.backward()
        assert logits_e.grad is not None

    def test_kl_loss_zero_when_equal(self):
        """KL divergence is 0 when both distributions are identical."""
        logits = torch.randn(3, 10)
        loss = kl_divergence_loss(logits, logits.clone().requires_grad_(True))
        assert loss.item() < 1e-5


# ---------------------------------------------------------------------------
# Mask priors
# ---------------------------------------------------------------------------

class TestMaskPriors:

    def test_area_loss_nonneg(self):
        m_LL, m_LH, m_HL, m_HH = _random_masks()
        loss = area_loss(m_LL, m_LH, m_HL, m_HH)
        assert loss.item() >= 0.0

    def test_area_loss_zero_on_all_zero_masks(self):
        """Zero masks → zero area loss."""
        masks = tuple(torch.zeros(2, 1, 14, 14, requires_grad=True) for _ in range(4))
        loss = area_loss(*masks)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_area_loss_differentiable(self):
        m_LL, m_LH, m_HL, m_HH = _random_masks()
        loss = area_loss(m_LL, m_LH, m_HL, m_HH)
        loss.backward()
        for m in (m_LL, m_LH, m_HL, m_HH):
            assert m.grad is not None

    def test_binarization_loss_nonneg(self):
        m_LL, m_LH, m_HL, m_HH = _random_masks()
        loss = binarization_loss(m_LL, m_LH, m_HL, m_HH)
        assert loss.item() >= 0.0

    def test_binarization_loss_zero_on_binary_masks(self):
        """For exactly binary masks the binarisation penalty is 0."""
        masks = tuple(
            (torch.randint(0, 2, (2, 1, 14, 14)).float().requires_grad_(True))
            for _ in range(4)
        )
        loss = binarization_loss(*masks)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_binarization_loss_differentiable(self):
        m_LL, m_LH, m_HL, m_HH = _random_masks()
        loss = binarization_loss(m_LL, m_LH, m_HL, m_HH)
        loss.backward()
        for m in (m_LL, m_LH, m_HL, m_HH):
            assert m.grad is not None


class TestTotalVariationLoss:

    def test_tv_loss_nonneg(self):
        m_LL, m_LH, m_HL, m_HH = _random_masks()
        loss = total_variation_loss(m_LL, m_LH, m_HL, m_HH)
        assert loss.item() >= 0.0

    def test_tv_loss_zero_on_constant_mask(self):
        """A spatially constant mask has zero TV loss."""
        masks = tuple(torch.full((2, 1, 14, 14), 0.7).requires_grad_(True) for _ in range(4))
        loss = total_variation_loss(*masks)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_tv_loss_higher_for_noisy_mask(self):
        """A noisy mask should have strictly higher TV loss than a smooth mask."""
        smooth = tuple(torch.ones(2, 1, 14, 14) * 0.5 for _ in range(4))
        noisy  = tuple(torch.randint(0, 2, (2, 1, 14, 14)).float() for _ in range(4))
        loss_smooth = total_variation_loss(*smooth)
        loss_noisy  = total_variation_loss(*noisy)
        assert loss_noisy.item() > loss_smooth.item()

    def test_tv_loss_differentiable(self):
        m_LL, m_LH, m_HL, m_HH = _random_masks()
        loss = total_variation_loss(m_LL, m_LH, m_HL, m_HH)
        loss.backward()
        for m in (m_LL, m_LH, m_HL, m_HH):
            assert m.grad is not None

    def test_tv_loss_scalar(self):
        m_LL, m_LH, m_HL, m_HH = _random_masks()
        loss = total_variation_loss(m_LL, m_LH, m_HL, m_HH)
        assert loss.ndim == 0


class TestExplanationAreaLoss:

    def _image_pair(self, B=2, C=3, H=32, W=32):
        x = torch.rand(B, C, H, W) + 0.1  # avoid zero denominator
        e = torch.rand(B, C, H, W, requires_grad=True)
        return x, e

    def test_nonneg(self):
        x, e = self._image_pair()
        loss = explanation_area_loss(e, x)
        assert loss.item() >= 0.0

    def test_zero_explanation_gives_zero_loss(self):
        x = torch.rand(2, 3, 32, 32) + 0.1
        e = torch.zeros(2, 3, 32, 32, requires_grad=True)
        loss = explanation_area_loss(e, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_full_explanation_gives_loss_near_one(self):
        """When e == x the ratio ||e||_1 / ||x||_1 equals 1."""
        x = torch.rand(2, 3, 32, 32) + 0.1
        e = x.clone().requires_grad_(True)
        loss = explanation_area_loss(e, x)
        assert loss.item() == pytest.approx(1.0, abs=1e-4)

    def test_differentiable(self):
        x, e = self._image_pair()
        loss = explanation_area_loss(e, x)
        loss.backward()
        assert e.grad is not None

    def test_scalar(self):
        x, e = self._image_pair()
        loss = explanation_area_loss(e, x)
        assert loss.ndim == 0


class TestExplanationTVLoss:

    def test_nonneg(self):
        e = torch.rand(2, 3, 32, 32, requires_grad=True)
        assert explanation_tv_loss(e).item() >= 0.0

    def test_zero_on_constant_image(self):
        e = torch.full((2, 3, 32, 32), 0.5, requires_grad=True)
        assert explanation_tv_loss(e).item() == pytest.approx(0.0, abs=1e-6)

    def test_higher_for_noisy_image(self):
        smooth = torch.ones(2, 3, 32, 32) * 0.5
        noisy  = torch.rand(2, 3, 32, 32)
        assert explanation_tv_loss(noisy).item() > explanation_tv_loss(smooth).item()

    def test_differentiable(self):
        e = torch.rand(2, 3, 32, 32, requires_grad=True)
        explanation_tv_loss(e).backward()
        assert e.grad is not None

    def test_scalar(self):
        e = torch.rand(2, 3, 32, 32, requires_grad=True)
        assert explanation_tv_loss(e).ndim == 0
