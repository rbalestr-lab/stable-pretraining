
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import lightning as pl
import pytest
import torch
import torch.nn.functional as F
import torchmetrics

import stable_pretraining as spt
from stable_pretraining.data import transforms

benchmarks_path = Path(__file__).parent.parent.parent.parent / "benchmarks" / "imagenette"
sys.path.insert(0, str(benchmarks_path))

from utils import create_mae_with_custom_decoder, parse_decoder_type


@pytest.mark.integration
class TestMAEReconstructionGapIntegration:

    def _create_mae_reconstruction_gap_module(self, model_name, decoder_type_str, alpha, num_classes=10):
        decoder_config = parse_decoder_type(decoder_type_str)

        backbone = create_mae_with_custom_decoder(model_name, decoder_config)

        def forward(self, batch, stage):
            imgs = batch["image"]
            latent, pred, mask = self.backbone(imgs, mask_ratio=self.mask_ratio)

            target = self.backbone.patchify(imgs)
            loss_rec = spt.losses.mae(target, pred, mask)

            batch["embedding"] = latent[:, 0]
            batch["loss_rec"] = loss_rec

            if hasattr(self, "classifier") and self.alpha != 0.0:
                logits = self.classifier(batch["embedding"])
                loss_cls = F.cross_entropy(logits, batch["label"])

                batch["loss"] = loss_rec + self.alpha * loss_cls
                batch["loss_cls"] = loss_cls
                batch["logits"] = logits

                if stage == "val":
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds == batch["label"]).float().mean()
                    batch["supervised_acc"] = acc
            else:
                batch["loss"] = loss_rec

            return batch

        module = spt.Module(backbone=backbone, forward=forward)
        module.mask_ratio = 0.75
        module.alpha = alpha
        module.model_name = model_name
        module.decoder_type = decoder_type_str

        if alpha != 0.0:
            encoder_dim = backbone.embed_dim
            module.classifier = torch.nn.Linear(encoder_dim, num_classes)

        return module

    @pytest.mark.gpu
    @pytest.mark.download
    @pytest.mark.slow
    def test_mae_reconstruction_gap_positive_alpha(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(mean=mean, std=std),
        )

        val_transform = transforms.Compose(
            transforms.RGB(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToImage(mean=mean, std=std),
        )

        train_dataset = spt.data.HFDataset(
            path="frgfm/imagenette",
            name="160px",
            split="train",
            transform=train_transform,
        )

        train = torch.utils.data.DataLoader(
            dataset=train_dataset,
            sampler=spt.data.sampler.RepeatedRandomSampler(train_dataset, n_views=1),
            batch_size=32,
            num_workers=4,
            drop_last=True,
        )

        val = torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                path="frgfm/imagenette",
                name="160px",
                split="validation",
                transform=val_transform,
            ),
            batch_size=64,
            num_workers=2,
        )

        data = spt.data.DataModule(train=train, val=val)

        module = self._create_mae_reconstruction_gap_module(
            model_name="vit_tiny",
            decoder_type_str="base-8b",
            alpha=1.0,
            num_classes=10,
        )

        optimizer_cfg = {
            "name": "AdamW",
            "config": {
                "lr": 1e-4,
                "betas": (0.9, 0.95),
                "weight_decay": 0.05,
            },
        }

        linear_probe = spt.callbacks.OnlineProbe(
            "linear_probe",
            module,
            "embedding",
            "label",
            probe=torch.nn.Linear(192, 10),
            loss_fn=torch.nn.CrossEntropyLoss(),
            metrics={
                "top1": torchmetrics.classification.MulticlassAccuracy(10),
            },
        )

        with TemporaryDirectory() as tmpdir:
            trainer = pl.Trainer(
                max_epochs=2,
                accelerator="auto",
                devices=1,
                precision="16-mixed",
                callbacks=[linear_probe],
                default_root_dir=tmpdir,
                enable_checkpointing=False,
                logger=False,
            )

            manager = spt.Manager(module, data, optimizer_cfg)
            manager.run(trainer)

            assert trainer.current_epoch == 2

            metrics = trainer.callback_metrics
            assert "val/loss_rec" in metrics
            assert "val/loss_cls" in metrics
            assert "val/linear_probe/top1" in metrics

    @pytest.mark.gpu
    @pytest.mark.download
    @pytest.mark.slow
    def test_mae_reconstruction_gap_negative_alpha(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224)),
            transforms.ToImage(mean=mean, std=std),
        )

        val_transform = transforms.Compose(
            transforms.RGB(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToImage(mean=mean, std=std),
        )

        train_dataset = spt.data.HFDataset(
            path="frgfm/imagenette",
            name="160px",
            split="train",
            transform=train_transform,
        )

        train = torch.utils.data.DataLoader(
            dataset=train_dataset,
            sampler=spt.data.sampler.RepeatedRandomSampler(train_dataset, n_views=1),
            batch_size=32,
            num_workers=4,
            drop_last=True,
        )

        val = torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                path="frgfm/imagenette",
                name="160px",
                split="validation",
                transform=val_transform,
            ),
            batch_size=64,
            num_workers=2,
        )

        data = spt.data.DataModule(train=train, val=val)

        module = self._create_mae_reconstruction_gap_module(
            model_name="vit_tiny",
            decoder_type_str="base-8b",
            alpha=-1.0,
            num_classes=10,
        )

        optimizer_cfg = {
            "name": "AdamW",
            "config": {
                "lr": 1e-4,
                "betas": (0.9, 0.95),
                "weight_decay": 0.05,
            },
        }

        linear_probe = spt.callbacks.OnlineProbe(
            "linear_probe",
            module,
            "embedding",
            "label",
            probe=torch.nn.Linear(192, 10),
            loss_fn=torch.nn.CrossEntropyLoss(),
            metrics={
                "top1": torchmetrics.classification.MulticlassAccuracy(10),
            },
        )

        with TemporaryDirectory() as tmpdir:
            trainer = pl.Trainer(
                max_epochs=2,
                accelerator="auto",
                devices=1,
                precision="16-mixed",
                callbacks=[linear_probe],
                default_root_dir=tmpdir,
                enable_checkpointing=False,
                logger=False,
            )

            manager = spt.Manager(module, data, optimizer_cfg)
            manager.run(trainer)

            assert trainer.current_epoch == 2

            metrics = trainer.callback_metrics
            assert "val/loss_rec" in metrics
            assert "val/loss_cls" in metrics

            assert metrics["val/loss_rec"].item() > 0
            assert metrics["val/loss_cls"].item() > 0

    @pytest.mark.gpu
    @pytest.mark.download
    @pytest.mark.slow
    def test_trajectory_saving(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224)),
            transforms.ToImage(mean=mean, std=std),
        )

        val_transform = transforms.Compose(
            transforms.RGB(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToImage(mean=mean, std=std),
        )

        train_dataset = spt.data.HFDataset(
            path="frgfm/imagenette",
            name="160px",
            split="train",
            transform=train_transform,
        )

        train = torch.utils.data.DataLoader(
            dataset=train_dataset,
            sampler=spt.data.sampler.RepeatedRandomSampler(train_dataset, n_views=1),
            batch_size=32,
            num_workers=4,
            drop_last=True,
        )

        val = torch.utils.data.DataLoader(
            dataset=spt.data.HFDataset(
                path="frgfm/imagenette",
                name="160px",
                split="validation",
                transform=val_transform,
            ),
            batch_size=64,
            num_workers=2,
        )

        data = spt.data.DataModule(train=train, val=val)

        module = self._create_mae_reconstruction_gap_module(
            model_name="vit_tiny",
            decoder_type_str="base-8b",
            alpha=0.5,
            num_classes=10,
        )

        optimizer_cfg = {
            "name": "AdamW",
            "config": {
                "lr": 1e-4,
                "betas": (0.9, 0.95),
                "weight_decay": 0.05,
            },
        }

        linear_probe = spt.callbacks.OnlineProbe(
            "linear_probe",
            module,
            "embedding",
            "label",
            probe=torch.nn.Linear(192, 10),
            loss_fn=torch.nn.CrossEntropyLoss(),
            metrics={
                "top1": torchmetrics.classification.MulticlassAccuracy(10),
            },
        )

        sys.path.insert(0, str(benchmarks_path))
        from mae_vit_reconstruction_gap import ReconstructionGapCallback

        gap_callback = ReconstructionGapCallback()

        with TemporaryDirectory() as tmpdir:
            logger = pl.loggers.CSVLogger(save_dir=tmpdir, name="test")

            trainer = pl.Trainer(
                max_epochs=2,
                accelerator="auto",
                devices=1,
                precision="16-mixed",
                callbacks=[linear_probe, gap_callback],
                default_root_dir=tmpdir,
                enable_checkpointing=False,
                logger=logger,
            )

            manager = spt.Manager(module, data, optimizer_cfg)
            manager.run(trainer)

            trajectory_path = Path(logger.log_dir) / "trajectory.json"
            assert trajectory_path.exists(), f"Trajectory file not found at {trajectory_path}"

            with open(trajectory_path, "r") as f:
                trajectory_data = json.load(f)

            assert "alpha" in trajectory_data
            assert trajectory_data["alpha"] == 0.5
            assert "model" in trajectory_data
            assert trajectory_data["model"] == "vit_tiny"
            assert "decoder" in trajectory_data
            assert trajectory_data["decoder"] == "base-8b"
            assert "trajectory" in trajectory_data
            assert len(trajectory_data["trajectory"]) > 0

            for entry in trajectory_data["trajectory"]:
                assert "epoch" in entry
                assert "mse" in entry
                assert "accuracy" in entry
                assert "alpha" in entry
