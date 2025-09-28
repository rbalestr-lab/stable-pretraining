"""DINO training on ImageNet-100 with 8 GPUs."""

import lightning as pl
import torch
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from torch import nn

import stable_pretraining as spt
from stable_pretraining.forward import dino_forward
from stable_pretraining.data import transforms
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

# DINO multi-crop transform: 2 global + 6 local crops for ImageNet
dino_transform = transforms.MultiViewTransform(
    [
        # First global crop (224x224)
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.4, 1.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=1.0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        # Second global crop (224x224)
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.4, 1.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        # Local crops (96x96) - 6 of them
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.4)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.4)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.4)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.4)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.4)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.4)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(p=0.5),
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
    transform=dino_transform,
)
val_dataset = spt.data.HFDataset(
    "clane9/imagenet-100",
    split="validation",
    cache_dir=str(data_dir),
    transform=val_transform,
)

# Per-GPU batch size for 8 GPUs
# DINO typically uses smaller batch size due to multi-crop (8 views per sample)
# Total effective batch size: 64 * 8 GPUs * 8 views = 4096 images
batch_size = 64
world_size = 8
total_batch_size = batch_size * world_size

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=spt.data.sampler.RepeatedRandomSampler(train_dataset, n_views=8),
    batch_size=batch_size,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

# Create backbone with teacher-student wrapper
backbone = spt.backbone.from_timm(
    "vit_small_patch16_224",
    pretrained=False,
    num_classes=0,  # Remove classification head
)

wrapped_backbone = spt.TeacherStudentWrapper(
    backbone,
    warm_init=True,
    base_ema_coefficient=0.996,
    final_ema_coefficient=1.0,
)

# Create projector with teacher-student wrapper
# DINO uses 3-layer MLP with larger dimensions for ImageNet
projector = nn.Sequential(
    nn.Linear(384, 2048),
    nn.BatchNorm1d(2048),
    nn.GELU(),
    nn.Linear(2048, 2048),
    nn.BatchNorm1d(2048),
    nn.GELU(),
    nn.Linear(2048, 65536),  # Large output dimension for ImageNet
)

wrapped_projector = spt.TeacherStudentWrapper(
    projector,
    warm_init=True,
    base_ema_coefficient=0.996,
    final_ema_coefficient=1.0,
)

# Learning rate scaled by batch size: base_lr * (batch_size / 256)
# With 8 GPUs and batch_size=64: total_batch_size=512
lr = 0.0005 * (total_batch_size / 256)

module = spt.Module(
    backbone=wrapped_backbone,
    projector=wrapped_projector,
    forward=dino_forward,
    dino_loss=spt.losses.DINOLoss(
        temperature_student=0.1,
        center_momentum=0.9,
    ),
    # DINO-specific temperature parameters
    warmup_temperature_teacher=0.04,
    temperature_teacher=0.04,  # No warmup for smaller dataset
    warmup_epochs_temperature_teacher=10,
    optim={
        "optimizer": {
            "type": "AdamW",
            "lr": lr,
            "weight_decay": 0.04,
            "betas": [0.9, 0.95],
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "warmup_steps": 10,
        },
        "interval": "epoch",
    },
)

# Teacher-Student callback for EMA updates
teacher_student_callback = spt.callbacks.TeacherStudentCallback(
    update_frequency=1,
    update_after_backward=False,
)

linear_probe = spt.callbacks.OnlineProbe(
    name="linear_probe",
    input="embedding",
    target="label",
    probe=nn.Linear(384, 100),
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
    input_dim=384,
    k=20,
)

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="imagenet100-dino",
    name="dino-resnet18-8gpus",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=100,
    num_sanity_val_steps=0,
    callbacks=[teacher_student_callback, linear_probe, knn_probe],
    precision="16-mixed",
    logger=wandb_logger,
    devices=8,  # Use 8 GPUs
    strategy="ddp_find_unused_parameters_true",  # DDP with unused parameters detection
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
