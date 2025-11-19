
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from lightning.pytorch import Trainer

from stable_pretraining.callbacks.mae_reconstruction_viz import MAEReconstructionViz


@pytest.mark.unit
class TestMAEReconstructionViz:

    def test_callback_instantiation_default(self):
        callback = MAEReconstructionViz()

        assert callback.log_interval == 10
        assert callback.num_samples == 8
        assert callback.mask_ratio == 0.75
        assert callback.fixed_samples_path is None
        assert callback.name == "mae_reconstruction"
        assert callback._fixed_samples is None

    def test_callback_instantiation_custom(self):
        callback = MAEReconstructionViz(
            log_interval=5,
            num_samples=16,
            mask_ratio=0.5,
            fixed_samples_path="/path/to/samples.pt",
            name="custom_mae_viz",
        )

        assert callback.log_interval == 5
        assert callback.num_samples == 16
        assert callback.mask_ratio == 0.5
        assert callback.fixed_samples_path == "/path/to/samples.pt"
        assert callback.name == "custom_mae_viz"

    def test_denormalize_image(self):
        callback = MAEReconstructionViz()

        image = torch.randn(3, 224, 224).numpy()

        denorm_image = callback._denormalize_image(image)

        assert denorm_image.shape == (3, 224, 224)

        assert denorm_image.min() >= 0.0
        assert denorm_image.max() <= 1.0

    def test_load_fixed_samples_nonexistent_path(self):
        callback = MAEReconstructionViz(
            fixed_samples_path="/nonexistent/path/samples.pt"
        )

        samples = callback._load_fixed_samples()
        assert samples is None

    def test_load_fixed_samples_none_path(self):
        callback = MAEReconstructionViz(fixed_samples_path=None)

        samples = callback._load_fixed_samples()
        assert samples is None

    def test_load_fixed_samples_valid_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = Path(tmpdir) / "fixed_val_samples.pt"
            dummy_images = torch.randn(16, 3, 224, 224)
            dummy_labels = torch.randint(0, 10, (16,))

            torch.save(
                {
                    "images": dummy_images,
                    "labels": dummy_labels,
                    "num_samples": 16,
                    "seed": 42,
                },
                samples_path,
            )

            callback = MAEReconstructionViz(
                fixed_samples_path=str(samples_path), num_samples=8
            )

            samples = callback._load_fixed_samples()

            assert samples is not None
            assert samples.shape == (8, 3, 224, 224)
            assert torch.allclose(samples, dummy_images[:8])

    def test_load_fixed_samples_cached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = Path(tmpdir) / "fixed_val_samples.pt"
            dummy_images = torch.randn(16, 3, 224, 224)

            torch.save(
                {"images": dummy_images, "labels": torch.zeros(16), "num_samples": 16},
                samples_path,
            )

            callback = MAEReconstructionViz(fixed_samples_path=str(samples_path))

            samples1 = callback._load_fixed_samples()
            samples2 = callback._load_fixed_samples()

            assert samples1 is samples2

    def test_callback_with_trainer_no_wandb(self):
        callback = MAEReconstructionViz()
        trainer = Trainer(logger=None)

        class MockModule:
            pass

        pl_module = MockModule()

        callback.on_validation_epoch_end(trainer, pl_module)

    def test_normalization_stats(self):
        callback = MAEReconstructionViz()

        mean = callback._normalization_stats["mean"]
        std = callback._normalization_stats["std"]

        expected_mean = np.array([0.485, 0.456, 0.406])
        expected_std = np.array([0.229, 0.224, 0.225])

        np.testing.assert_array_almost_equal(mean, expected_mean)
        np.testing.assert_array_almost_equal(std, expected_std)

    def test_log_interval_behavior(self):
        callback = MAEReconstructionViz(log_interval=10)

        trainer = Trainer(logger=None)
        trainer.current_epoch = 5

        class MockModule:
            pass

        pl_module = MockModule()

        callback.on_validation_epoch_end(trainer, pl_module)

        trainer.current_epoch = 10
        callback.on_validation_epoch_end(trainer, pl_module)
