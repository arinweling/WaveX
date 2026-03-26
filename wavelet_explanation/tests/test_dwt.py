"""
Tests for HaarDWT.

Key property verified: perfect reconstruction — IDWT(DWT(x)) ≈ x.
"""

import torch
import pytest
from wavelet.dwt import HaarDWT


@pytest.fixture
def dwt():
    return HaarDWT()


class TestHaarDWT:

    def test_forward_output_shapes_rgb(self, dwt):
        """Forward pass on an RGB image produces correctly shaped subbands."""
        x = torch.randn(2, 3, 64, 64)
        x_LL, x_LH, x_HL, x_HH = dwt(x)
        for subband in (x_LL, x_LH, x_HL, x_HH):
            assert subband.shape == (2, 3, 32, 32), f"Unexpected shape: {subband.shape}"

    def test_forward_output_shapes_grayscale(self, dwt):
        """Works on single-channel (grayscale) images."""
        x = torch.randn(4, 1, 28, 28)
        x_LL, x_LH, x_HL, x_HH = dwt(x)
        for subband in (x_LL, x_LH, x_HL, x_HH):
            assert subband.shape == (4, 1, 14, 14)

    def test_perfect_reconstruction_random(self, dwt):
        """IDWT(DWT(x)) == x to within floating-point precision."""
        x = torch.randn(3, 3, 32, 32)
        x_LL, x_LH, x_HL, x_HH = dwt(x)
        x_rec = dwt.inverse(x_LL, x_LH, x_HL, x_HH)
        max_err = (x - x_rec).abs().max().item()
        assert max_err < 1e-4, f"Max reconstruction error too large: {max_err}"

    def test_perfect_reconstruction_imagenet_size(self, dwt):
        """Reconstruction works at 224×224 (ImageNet input size)."""
        x = torch.randn(1, 3, 224, 224)
        x_rec = dwt.inverse(*dwt(x))
        max_err = (x - x_rec).abs().max().item()
        assert max_err < 1e-4, f"Max reconstruction error: {max_err}"

    def test_odd_spatial_dimensions_raises(self, dwt):
        """DWT requires even spatial dimensions."""
        x = torch.randn(1, 3, 33, 32)
        with pytest.raises(AssertionError):
            dwt(x)

    def test_subbands_not_identical(self, dwt):
        """The four subbands capture different frequency content."""
        x = torch.randn(1, 3, 32, 32)
        x_LL, x_LH, x_HL, x_HH = dwt(x)
        assert not torch.allclose(x_LL, x_LH)
        assert not torch.allclose(x_LL, x_HH)

    def test_forward_no_inplace_modification(self, dwt):
        """DWT does not modify the input tensor."""
        x = torch.randn(1, 3, 32, 32)
        x_copy = x.clone()
        dwt(x)
        assert torch.allclose(x, x_copy)
