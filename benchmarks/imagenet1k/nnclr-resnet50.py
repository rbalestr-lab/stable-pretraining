import lightning as pl
import torch
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from torch import nn

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.forward import nnclr_forward
from stable_pretraining.callbacks.queue import OnlineQueue
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

nnclr_transform = transforms.MultiViewTransform(
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

train_dataset = spt.data.HFDataset(
    "imagenet-1k",
    split="train",
    cache_dir=str(get_data_dir()),
    transform=nnclr_transform,
)
val_dataset = spt.data.HFDataset(
    "imagenet-1k",
    split="validation",
    cache_dir=str(get_data_dir()),
    transform=val_transform,
)

total_batch_size, world_size, num_epochs = 4096, 4, 400
local_batch_size = total_batch_size // world_size
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=spt.data.sampler.RepeatedRandomSampler(train_dataset, n_views=2),
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

backbone = spt.backbone.from_torchvision(
    "resnet50",
    low_resolution=False,
)
backbone.fc = torch.nn.Identity()

projector = nn.Sequential(
    nn.Linear(2048, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 256),
)

predictor = nn.Sequential(
    nn.Linear(256, 4096),
    nn.BatchNorm1d(4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, 256),
)


module = spt.Module(
    backbone=backbone,
    projector=projector,
    predictor=predictor,
    forward=nnclr_forward,
    nnclr_loss=spt.losses.NTXEntLoss(temperature=0.1),
    optim={
        "optimizer": {
            "type": "LARS",
            "lr": 0.3 * (total_batch_size / 256),
            "weight_decay": 1e-6,
            "clip_lr": True,
            "eta": 0.001,
            "exclude_bias_n_norm": True,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "peak_step": 10 / num_epochs,
            "total_steps": num_epochs * (len(train_dataloader) // world_size),
        },
        "interval": "epoch",
    },
    hparams={
        "support_set_size": 98304,  # was 16384
        "projection_dim": 256,      # keep
    },
)

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

support_queue = OnlineQueue(
    key="nnclr_support_set",
    queue_length=module.hparams.support_set_size,
    dim=module.hparams.projection_dim,
)

wandb_logger = WandbLogger(
    entity="samibg",
    project="imagenet-1k-nnclr",
    name=f"nnclr-resnet50-{world_size}gpus",
    log_model=False,
)

# --- Trainer ---
trainer = pl.Trainer(
    max_epochs=num_epochs,
    num_sanity_val_steps=0,
    callbacks=[linear_probe, knn_probe, support_queue],
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=True,
    accelerator="gpu",
    devices=world_size,
    sync_batchnorm=world_size > 1,
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()