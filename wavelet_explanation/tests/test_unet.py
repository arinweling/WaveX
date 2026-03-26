"""
Tests for the U-Net encoder-decoder.

Checks output shapes for several input sizes and verifies the STE
binarisation behaves correctly.
"""

import torch
import pytest
from models.unet import UNet, STE
from models.pixel_mask_net import PixelMaskNet


class TestSTE:

    def test_forward_binarises_at_half(self):
        x = torch.tensor([0.0, 0.3, 0.5, 0.51, 1.0])
        out = STE.apply(x)
        expected = torch.tensor([0., 0., 0., 1., 1.])
        assert torch.equal(out, expected)

    def test_backward_passes_gradient_unchanged(self):
        x = torch.tensor([0.2, 0.8], requires_grad=True)
        out = STE.apply(x)
        # The binary output is 0 and 1, but a sum should still flow gradients back
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        # Straight-through: dL/dx = dL/d_out = [1, 1]
        assert torch.allclose(x.grad, torch.ones_like(x.grad))


class TestUNet:

    @pytest.mark.parametrize("H,W", [(224, 224), (64, 64), (32, 32)])
    def test_output_shapes_rgb(self, H, W):
        """Four masks at half spatial resolution for RGB input."""
        net = UNet(in_channels=3)
        x = torch.randn(2, 3, H, W)
        m_LL, m_LH, m_HL, m_HH, _ = net(x)
        for mask in (m_LL, m_LH, m_HL, m_HH):
            assert mask.shape == (2, 1, H // 2, W // 2), f"Bad shape: {mask.shape}"

    def test_output_shapes_grayscale(self):
        """Works for single-channel (MNIST) input."""
        net = UNet(in_channels=1)
        x = torch.randn(3, 1, 28, 28)
        m_LL, m_LH, m_HL, m_HH, _ = net(x)
        for m in (m_LL, m_LH, m_HL, m_HH):
            assert m.shape == (3, 1, 14, 14), f"Bad shape: {m.shape}"

    def test_output_values_binary(self):
        """All mask values must be exactly 0.0 or 1.0 after STE."""
        net = UNet(in_channels=3)
        x = torch.randn(2, 3, 32, 32)
        m_LL, m_LH, m_HL, m_HH, _ = net(x)
        for m in (m_LL, m_LH, m_HL, m_HH):
            unique_vals = m.unique()
            assert set(unique_vals.tolist()).issubset({0.0, 1.0}), \
                f"Non-binary mask values: {unique_vals}"

    def test_gradients_flow_through_unet(self):
        """Gradients should reach all U-Net parameters from a scalar loss on masks."""
        net = UNet(in_channels=3)
        x = torch.randn(2, 3, 32, 32)
        m_LL, m_LH, m_HL, m_HH, cont = net(x)
        # Use sum including continuous masks so L_bin gradient path is also tested
        loss = (m_LL.float().sum() + m_LH.float().sum() +
                m_HL.float().sum() + m_HH.float().sum() +
                sum(c.sum() for c in cont))
        loss.backward()
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"

    def test_active_subbands_output_channels(self):
        """UNet with 2 active subbands should output zeros for disabled ones."""
        net = UNet(in_channels=3, active_subbands=["LL", "HH"])
        x = torch.randn(2, 3, 32, 32)
        m_LL, m_LH, m_HL, m_HH, cont = net(x)
        # LH and HL must be all zeros
        assert m_LH.sum() == 0.0, "Disabled LH mask should be all zeros"
        assert m_HL.sum() == 0.0, "Disabled HL mask should be all zeros"
        assert cont[1].sum() == 0.0, "Disabled LH continuous mask should be all zeros"
        assert cont[2].sum() == 0.0, "Disabled HL continuous mask should be all zeros"
        # LL and HH should be valid binary masks
        assert set(m_LL.unique().tolist()).issubset({0.0, 1.0})
        assert set(m_HH.unique().tolist()).issubset({0.0, 1.0})

    def test_active_subbands_final_conv_channels(self):
        """final_conv should have only as many output channels as active subbands."""
        net2 = UNet(in_channels=3, active_subbands=["LL", "HH"])
        net4 = UNet(in_channels=3)
        assert net2.final_conv.out_channels == 2
        assert net4.final_conv.out_channels == 4

    def test_active_subbands_gradients(self):
        """Gradients should only flow through active subband parameters."""
        net = UNet(in_channels=3, active_subbands=["LL", "HH"])
        x = torch.randn(2, 3, 32, 32)
        m_LL, _, _, m_HH, cont = net(x)
        loss = m_LL.float().sum() + m_HH.float().sum() + cont[0].sum() + cont[3].sum()
        loss.backward()
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"


# ---------------------------------------------------------------------------
# PixelMaskNet
# ---------------------------------------------------------------------------

class TestPixelMaskNet:

    def test_output_shapes(self):
        """m_bin and m_cont should be (B, 1, H, W) — same spatial size as input."""
        net = PixelMaskNet(in_channels=3)
        x = torch.randn(2, 3, 96, 96)
        m_bin, m_cont = net(x)
        assert m_bin.shape  == (2, 1, 96, 96)
        assert m_cont.shape == (2, 1, 96, 96)

    def test_m_bin_is_binary(self):
        """m_bin values must be exactly 0 or 1."""
        net = PixelMaskNet(in_channels=3)
        x = torch.randn(2, 3, 64, 64)
        m_bin, _ = net(x)
        unique = m_bin.unique()
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_m_cont_in_unit_interval(self):
        """Continuous mask must be in [0, 1]."""
        net = PixelMaskNet(in_channels=3)
        x = torch.randn(2, 3, 64, 64)
        _, m_cont = net(x)
        assert m_cont.min().item() >= 0.0
        assert m_cont.max().item() <= 1.0

    def test_gradients_flow(self):
        """Gradients must reach every parameter via the continuous mask."""
        net = PixelMaskNet(in_channels=3)
        x = torch.randn(2, 3, 32, 32)
        _, m_cont = net(x)
        m_cont.sum().backward()
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for: {name}"

    def test_grayscale_input(self):
        net = PixelMaskNet(in_channels=1)
        x = torch.randn(2, 1, 64, 64)
        m_bin, m_cont = net(x)
        assert m_bin.shape == (2, 1, 64, 64)
