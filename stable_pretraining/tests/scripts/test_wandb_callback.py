"""Unit tests for checkpointing functionality."""

import torch
from torch.utils.data import Dataset
from pathlib import Path


class MockDataset(Dataset):
    """Simple mock dataset for testing samplers."""

    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"index": idx, "data": torch.randn(3)}


def main():
    import lightning as pl
    import torch
    from lightning.pytorch.loggers import WandbLogger
    import wandb

    import stable_pretraining as ssl

    pl.seed_everything(0)

    # Increase batch size for multi-GPU training (64 per GPU * 2 GPUs = 128 total)
    train = torch.utils.data.DataLoader(
        dataset=MockDataset(), batch_size=20, num_workers=8
    )
    val = torch.utils.data.DataLoader(
        dataset=MockDataset(), batch_size=20, num_workers=8
    )
    data = ssl.data.DataModule(train=train, val=val)

    def forward(self, batch, stage):
        output = {}
        output["embedding"] = self.backbone(batch["data"])
        if self.training:
            output["loss"] = output["embedding"].square().mean()
        return output

    backbone = torch.nn.Linear(3, 3)

    module = ssl.Module(backbone=backbone, forward=forward)

    wandb_logger = WandbLogger(project="DEBUG")

    # Configure multi-GPU trainer
    trainer = pl.Trainer(
        max_epochs=10,
        logger=wandb_logger,
        enable_checkpointing=False,
        accelerator="cpu",
    )

    manager = ssl.Manager(trainer=trainer, module=module, data=data)
    manager()
    manager.save_checkpoint("test.ckpt")

    wandb.finish()

    del manager

    backbone = torch.nn.Linear(3, 3)

    module = ssl.Module(backbone=backbone, forward=forward)

    wandb_logger = WandbLogger(project="DEBUG")

    # Configure multi-GPU trainer
    trainer = pl.Trainer(
        max_epochs=20,
        logger=wandb_logger,
        enable_checkpointing=False,
        accelerator="cpu",
    )
    manager = ssl.Manager(
        trainer=trainer, module=module, data=data, ckpt_path="test.ckpt"
    )
    manager()
    # manager.load_checkpoint("test.ckpt")
    Path("test.ckpt").unlink()

    print("Training completed!")


if __name__ == "__main__":
    main()
