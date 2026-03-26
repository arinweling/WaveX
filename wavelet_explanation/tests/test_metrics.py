"""
Tests for evaluation metrics.

All metrics are validated on tractable synthetic inputs with known expected values.
"""

import torch
import torch.nn as nn
import pytest
from evaluation.metrics import (
    compute_sparsity,
    compute_confidence_delta,
    compute_label_preservation_rate,
    compute_subband_activity_profile,
)


# ---------------------------------------------------------------------------
# Toy classifier
# ---------------------------------------------------------------------------

class _ToyClassifier(nn.Module):
    """Always predicts class 0 with high confidence."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        B = x.shape[0]
        logits = torch.zeros(B, self.num_classes)
        logits[:, 0] = 10.0  # class 0 always most confident
        return logits


# ---------------------------------------------------------------------------
# Sparsity
# ---------------------------------------------------------------------------

class TestComputeSparsity:

    def test_all_zero_masks_give_full_sparsity(self):
        B, H, W = 2, 14, 14
        zeros = torch.zeros(B, 1, H, W)
        sp = compute_sparsity(zeros, zeros, zeros, zeros)
        assert sp["overall"] == pytest.approx(1.0)
        for k in ("LL", "LH", "HL", "HH"):
            assert sp[k] == pytest.approx(1.0)

    def test_all_one_masks_give_zero_sparsity(self):
        B, H, W = 2, 14, 14
        ones = torch.ones(B, 1, H, W)
        sp = compute_sparsity(ones, ones, ones, ones)
        assert sp["overall"] == pytest.approx(0.0, abs=1e-6)

    def test_half_masks_give_half_sparsity(self):
        B, H, W = 1, 4, 4
        m = torch.zeros(B, 1, H, W)
        m[:, :, :H//2, :] = 1.0  # top half active
        sp = compute_sparsity(m, m, m, m)
        assert sp["overall"] == pytest.approx(0.5, abs=0.01)

    def test_output_keys(self):
        m = torch.rand(2, 1, 14, 14)
        sp = compute_sparsity(m, m, m, m)
        assert {"overall", "LL", "LH", "HL", "HH", "pixel_equivalent"}.issubset(sp.keys())


# ---------------------------------------------------------------------------
# Confidence delta
# ---------------------------------------------------------------------------

class TestConfidenceDelta:

    def test_identical_x_and_e_give_zero_delta(self):
        clf = _ToyClassifier()
        x = torch.randn(4, 3, 32, 32)
        y = torch.zeros(4, dtype=torch.long)
        delta = compute_confidence_delta(clf, x, x, y)
        assert delta == pytest.approx(0.0, abs=1e-5)

    def test_returns_float(self):
        clf = _ToyClassifier()
        x = torch.randn(2, 3, 32, 32)
        e = torch.randn(2, 3, 32, 32)
        y = torch.zeros(2, dtype=torch.long)
        delta = compute_confidence_delta(clf, x, e, y)
        assert isinstance(delta, float)


# ---------------------------------------------------------------------------
# Label preservation rate
# ---------------------------------------------------------------------------

class TestLabelPreservationRate:

    def test_identical_inputs_give_rate_one(self):
        clf = _ToyClassifier()
        x = torch.randn(4, 3, 32, 32)
        lpr = compute_label_preservation_rate(clf, x, x)
        assert lpr == pytest.approx(1.0)

    def test_output_in_zero_one_range(self):
        clf = _ToyClassifier()
        x = torch.randn(4, 3, 32, 32)
        e = torch.randn(4, 3, 32, 32)
        lpr = compute_label_preservation_rate(clf, x, e)
        assert 0.0 <= lpr <= 1.0


# ---------------------------------------------------------------------------
# Subband activity profile
# ---------------------------------------------------------------------------

class TestSubbandActivityProfile:

    def test_all_ones_returns_one_per_subband(self):
        ones = torch.ones(2, 1, 14, 14)
        ap = compute_subband_activity_profile(ones, ones, ones, ones)
        for v in ap.values():
            assert v == pytest.approx(1.0)

    def test_all_zeros_returns_zero_per_subband(self):
        zeros = torch.zeros(2, 1, 14, 14)
        ap = compute_subband_activity_profile(zeros, zeros, zeros, zeros)
        for v in ap.values():
            assert v == pytest.approx(0.0, abs=1e-6)

    def test_output_keys(self):
        m = torch.rand(2, 1, 14, 14)
        ap = compute_subband_activity_profile(m, m, m, m)
        assert set(ap.keys()) == {"LL", "LH", "HL", "HH"}

    def test_activity_in_zero_one_range(self):
        m = torch.rand(2, 1, 14, 14)
        ap = compute_subband_activity_profile(m, m, m, m)
        for v in ap.values():
            assert 0.0 <= v <= 1.0
