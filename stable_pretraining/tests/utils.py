"""Test utilities for creating mock data and fixtures."""

from typing import Any, Optional, Tuple

import torch
import lightning as pl
from torch.utils.data import DataLoader, Dataset


class MockImageDataset(Dataset):
    """Mock dataset for testing without downloading real data."""

    def __init__(
        self,
        num_samples: int = 100,
        image_size: Tuple[int, int, int] = (3, 224, 224),
        num_classes: int = 10,
        transform: Optional[Any] = None,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform

        # Pre-generate random data to speed up tests
        self.images = torch.randn(num_samples, *image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Return dict format expected by stable-ssl
        sample = {
            "image": image,
            "label": label,
            "index": idx,
        }

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample


def create_mock_dataloader(
    batch_size: int = 32,
    num_samples: int = 100,
    num_workers: int = 0,  # Use 0 for faster tests
    **dataset_kwargs,
) -> DataLoader:
    """Create a mock dataloader for testing."""
    dataset = MockImageDataset(num_samples=num_samples, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )


def create_mock_model(input_dim: int = 512, output_dim: int = 128) -> torch.nn.Module:
    """Create a simple mock model for testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, output_dim),
    )


class MockTransform:
    """Mock transform that just returns the input."""

    def __call__(self, x):
        return x


# 'Boring' vs. 'Mock'
# Because we just need a valid, placeholder object to make the code run


class BoringDataset(Dataset):
    """A simple, minimal dataset for testing core Trainer/Manager logic."""

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return 0


class BoringModule(pl.LightningModule):
    """A minimal LightningModule for instantiating the Manager."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self.layer(torch.randn(1)).sum()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class BoringDataModule(pl.LightningDataModule):
    """A minimal LightningDataModule for instantiating the Manager."""

    def train_dataloader(self):
        return DataLoader(BoringDataset())


class BoringTrainer(pl.Trainer):
    """A mock Trainer that disables default callbacks for predictable testing environments."""

    def __init__(self, **kwargs):
        kwargs.setdefault("enable_progress_bar", False)
        kwargs.setdefault("enable_model_summary", False)
        super().__init__(**kwargs)
