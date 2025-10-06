import lightning as pl
import torch
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch import nn

import stable_pretraining as spt
from stable_pretraining.data import transforms
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

# SimCLR augmentations, as described in the paper
simclr_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0), ratio=(3/4, 4/3)),
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(sigma=(0.1, 2.0), p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(**spt.data.static.ImageNet),
        ),  
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0), ratio=(3/4, 4/3)),
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILGaussianBlur(sigma=(0.1, 2.0), p=0.5),
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

# Using a standard ImageNet-1k dataset
train_dataset = spt.data.HFDataset(
    "imagenet-1k",
    split="train",
    cache_dir=str(get_data_dir()),
    transform=simclr_transform,
)
val_dataset = spt.data.HFDataset(
    "imagenet-1k",
    split="validation",
    cache_dir=str(get_data_dir()),
    transform=val_transform,
)

# Batch size from the paper (adjust if necessary)
total_batch_size, world_size, num_epochs = 4096, 8, 800
local_batch_size = total_batch_size // world_size

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=local_batch_size,
    num_workers=64,
    drop_last=True,
    persistent_workers=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=local_batch_size,
    num_workers=32,
    persistent_workers=True,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

# Using ResNet-50 as in the paper
backbone = spt.backbone.from_torchvision(
    "resnet50",
    low_resolution=False,
)
backbone.fc = torch.nn.Identity()

# Projector network as described in SimCLR paper
projector = nn.Sequential(
    nn.Linear(2048, 2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 128),
)

module = spt.Module(
    backbone=backbone,
    projector=projector,
    forward=spt.forward.simclr_forward,
    # Temperature can be tuned, 0.1 is a common value
    simclr_loss=spt.losses.NTXEntLoss(temperature=0.1),
    optim={
        "optimizer": {
            "type": "LARS",
            "lr": 0.3
            * (total_batch_size / 256),  # 256 is base batch size they use in SimCLR
            "weight_decay": 1e-6,
            "clip_lr": True,
            "eta": 1e-3,
            "exclude_bias_n_norm": True,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "peak_step": 10 / num_epochs,  # 10 epochs warmup
            "total_steps": num_epochs * (len(train_dataloader) // world_size),
        },
        "interval": "epoch",
    },
)

# Update probes for 1000 classes
linear_probe = spt.callbacks.OnlineProbe(
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(2048, 1000),
    loss_fn=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(1000),
        "top5": torchmetrics.classification.MulticlassAccuracy(1000, top_k=5),
    },
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(1000)},
    input_dim=2048,
    k=20,
)

wandb_logger = WandbLogger(
    entity="samibg",
    project="imagenet1k-simclr",
    name="simclr-resnet50-4gpu",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=num_epochs,
    num_sanity_val_steps=0,
    callbacks=[
        linear_probe,
        knn_probe,
    ],
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=True,
    accelerator="gpu",
    sync_batchnorm=world_size > 1,
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
