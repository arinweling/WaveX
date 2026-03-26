"""
Integration test: one complete training step.

Verifies that:
  1. The training step runs without errors on a tiny synthetic batch.
  2. All U-Net parameters receive non-None gradients.
  3. The returned loss dict contains the expected keys.
  4. The classifier parameters do NOT accumulate gradients.
"""

import torch
import pytest

# Use a minimal config with tiny batch and no pretrained weights
MINI_CONFIG = {
    "backbone":       "resnet18",
    "pretrained":     False,   # random weights — no internet required
    "dataset":        "imagenet",
    "image_size":     32,      # tiny spatial size for speed
    "batch_size":     2,
    "lr":             1e-4,
    "lambda_act":     1.0,
    "lambda_CE":      4.0,
    "lambda_KL":      0.4,
    "lambda_rob":     6.0,
    "lambda_bin":     0.3,
    "lambda_area_LL": 5.0,
    "lambda_area_LH": 10.0,
    "lambda_area_HL": 10.0,
    "lambda_area_HH": 20.0,
}


@pytest.fixture(scope="module")
def trainer():
    from training.trainer import WaveletExplanationTrainer
    return WaveletExplanationTrainer(MINI_CONFIG, device="cpu")


@pytest.fixture
def batch():
    B, C, H, W = 2, 3, 32, 32
    x = torch.randn(B, C, H, W)
    y = torch.randint(0, 1000, (B,))   # ImageNet class range
    return x, y


class TestTrainerStep:

    def test_train_step_runs_without_error(self, trainer, batch):
        x, y = batch
        losses = trainer.train_step(x, y)
        assert losses is not None

    def test_loss_dict_has_expected_keys(self, trainer, batch):
        x, y = batch
        losses = trainer.train_step(x, y)
        expected = {"loss_total", "loss_act", "loss_ce", "loss_kl",
                    "loss_rob", "loss_area", "loss_bin"}
        assert expected.issubset(set(losses.keys()))

    def test_all_losses_are_finite(self, trainer, batch):
        x, y = batch
        losses = trainer.train_step(x, y)
        for key, val in losses.items():
            assert torch.isfinite(torch.tensor(val)), f"Loss '{key}' is not finite: {val}"

    def test_unet_parameters_have_gradients(self, trainer, batch):
        """After a train step, every U-Net parameter must have a gradient."""
        x, y = batch
        trainer.train_step(x, y)
        for name, param in trainer.encoder_decoder.named_parameters():
            assert param.grad is not None, f"No gradient for U-Net param: {name}"

    def test_classifier_parameters_have_no_gradients(self, trainer, batch):
        """Classifier parameters must remain gradient-free (frozen)."""
        x, y = batch
        trainer.train_step(x, y)
        model = (trainer.classifier.model
                 if hasattr(trainer.classifier, "model")
                 else trainer.classifier)
        for name, param in model.named_parameters():
            assert param.grad is None, f"Classifier param accumulated gradient: {name}"

    def test_predict_masks_shapes(self, trainer, batch):
        """predict_masks returns masks at half resolution and explanation at full."""
        x, _ = batch
        x = x[:2]
        B, C, H, W = x.shape
        m_LL, m_LH, m_HL, m_HH, e, subbands = trainer.predict_masks(x)
        for m in (m_LL, m_LH, m_HL, m_HH):
            assert m.shape == (B, 1, H // 2, W // 2)
        assert e.shape == (B, C, H, W)
        assert set(subbands.keys()) == {"LL", "LH", "HL", "HH"}

    def test_fixed_class_label_overrides_model_top1(self):
        from training.trainer import WaveletExplanationTrainer

        config = dict(MINI_CONFIG)
        config["fixed_class_label"] = 7
        trainer = WaveletExplanationTrainer(config, device="cpu")

        logits_x = torch.randn(3, 1000)
        targets = trainer._get_target_labels(logits_x)

        assert torch.equal(targets, torch.tensor([7, 7, 7]))
