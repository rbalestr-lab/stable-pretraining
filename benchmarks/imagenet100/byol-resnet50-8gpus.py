import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining.data import transforms
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

byol_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=1.0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
    ]
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToImage(**spt.data.static.ImageNet),
)

data_dir = get_data_dir("imagenet100")

train_dataset = spt.data.HFDataset(
    "clane9/imagenet-100",
    split="train",
    cache_dir=str(data_dir),
    transform=byol_transform,
)
val_dataset = spt.data.HFDataset(
    "clane9/imagenet-100",
    split="validation",
    cache_dir=str(data_dir),
    transform=val_transform,
)

batch_size = 512  # Per GPU batch size (4096 / 8 GPUs)
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=spt.data.sampler.RepeatedRandomSampler(train_dataset, n_views=2),
    batch_size=batch_size,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)


def forward(self, batch, stage):
    if self.training:
        images = batch["image"]
        sample_idx = batch["sample_idx"]

        online_features = self.backbone.forward_student(images)
        online_proj = self.projector.forward_student(online_features)
        online_pred = self.predictor(online_proj)

        with torch.no_grad():
            target_features = self.backbone.forward_teacher(images)
            target_proj = self.projector.forward_teacher(target_features)

        online_pred_views = spt.data.fold_views(online_pred, sample_idx)
        target_proj_views = spt.data.fold_views(target_proj, sample_idx)

        loss_1 = self.byol_loss(online_pred_views[0], target_proj_views[1])
        loss_2 = self.byol_loss(online_pred_views[1], target_proj_views[0])
        batch["loss"] = (loss_1 + loss_2) / 2

        batch["embedding"] = online_features.detach()
    else:
        batch["embedding"] = self.backbone.forward_student(batch["image"])

    return batch


backbone = spt.backbone.from_torchvision("resnet50", low_resolution=False, weights=None)
backbone.fc = nn.Identity()

wrapped_backbone = spt.TeacherStudentWrapper(
    backbone,
    warm_init=True,
    base_ema_coefficient=0.996,
    final_ema_coefficient=1.0,
)

projector = nn.Sequential(
    nn.Linear(2048, 4096),
    nn.BatchNorm1d(4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, 256),
)
wrapped_projector = spt.TeacherStudentWrapper(
    projector,
    warm_init=True,
    base_ema_coefficient=0.996,
    final_ema_coefficient=1.0,
)

predictor = nn.Sequential(
    nn.Linear(256, 4096),
    nn.BatchNorm1d(4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, 256),
)

module = spt.Module(
    backbone=wrapped_backbone,
    projector=wrapped_projector,
    predictor=predictor,
    forward=forward,
    byol_loss=spt.losses.BYOLLoss(),
    optim={
        "optimizer": {
            "type": "LARS",
            "lr": 0.5 * 4096 / 256,  # Scale LR for global batch size of 4096
            "weight_decay": 1e-6,
            "clip_lr": True,
            "eta": 0.02,
            "exclude_bias_n_norm": True,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
        },
        "interval": "epoch",
    },
)

linear_probe = spt.callbacks.OnlineProbe(
    name="linear_probe",
    input="embedding",
    target="label",
    probe=nn.Linear(2048, 100),
    loss_fn=nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(100),
        "top5": torchmetrics.classification.MulticlassAccuracy(100, top_k=5),
    },
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(100)},
    input_dim=2048,
    k=20,
)

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="imagenet100-byol",
    name="byol-resnet50-8gpus",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=400,
    num_sanity_val_steps=0,
    callbacks=[linear_probe, knn_probe],
    precision="16-mixed",
    logger=wandb_logger,
    devices=8,  # Use 8 GPUs
    strategy="ddp_find_unused_parameters_true",  # DDP with unused parameters detection
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
