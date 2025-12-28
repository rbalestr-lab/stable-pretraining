# test_registry.py
import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from stable_pretraining.callbacks.registry import (
    ModuleRegistryCallback,
    get_module,
    _MODULE_REGISTRY,
)
from stable_pretraining import log


class DummyModel(pl.LightningModule):
    """Minimal model for testing."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)
        self.logged_values = []

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        # Test global log function
        log("test_metric", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


@pytest.fixture
def dummy_dataloader():
    """Create a minimal dataloader for testing."""
    x = torch.randn(20, 10)
    y = torch.randn(20, 1)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=4)


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is clean before and after each test."""
    _MODULE_REGISTRY.clear()
    yield
    _MODULE_REGISTRY.clear()


class TestModuleRegistryCallback:
    """Minimal model for testing."""

    def test_module_registered_on_setup(self, dummy_model, dummy_dataloader):
        """Test that module is registered when training starts."""
        callback = ModuleRegistryCallback()
        trainer = pl.Trainer(
            max_epochs=1,
            callbacks=[callback],
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )

        assert get_module() is None
        trainer.fit(dummy_model, dummy_dataloader)
        # After teardown, should be cleaned up
        assert get_module() is None

    def test_module_accessible_during_training(self, dummy_model, dummy_dataloader):
        """Test that module is accessible via get_module during training."""
        accessed_during_training = []

        class CheckAccessCallback(pl.Callback):
            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                accessed_during_training.append(get_module() is not None)

        trainer = pl.Trainer(
            max_epochs=1,
            callbacks=[ModuleRegistryCallback(), CheckAccessCallback()],
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )
        trainer.fit(dummy_model, dummy_dataloader)

        assert all(accessed_during_training)

    def test_log_function_works(self, dummy_model, dummy_dataloader):
        """Test that global log() function works during training."""
        trainer = pl.Trainer(
            max_epochs=1,
            callbacks=[ModuleRegistryCallback()],
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )
        # Should not raise
        trainer.fit(dummy_model, dummy_dataloader)

    def test_custom_registry_name(self, dummy_model, dummy_dataloader):
        """Test registration with custom name."""
        accessed_modules = {}

        class CheckNameCallback(pl.Callback):
            def on_train_start(self, trainer, pl_module):
                accessed_modules["default"] = get_module("default")
