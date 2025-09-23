"""BYOL training on CIFAR-10."""

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.forward import byol_forward
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

byol_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
    ]
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((32, 32)),
    transforms.ToImage(**spt.data.static.CIFAR10),
)

data_dir = get_data_dir("cifar10")
cifar_train = torchvision.datasets.CIFAR10(
    root=str(data_dir), train=True, download=True
)
cifar_val = torchvision.datasets.CIFAR10(root=str(data_dir), train=False, download=True)

train_dataset = spt.data.FromTorchDataset(
    cifar_train, names=["image", "label"], transform=byol_transform, add_sample_idx=True
)
val_dataset = spt.data.FromTorchDataset(
    cifar_val, names=["image", "label"], transform=val_transform, add_sample_idx=True
)

batch_size = 256
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=spt.data.sampler.RepeatedRandomSampler(train_dataset, n_views=2),
    batch_size=batch_size,
    num_workers=8,
    drop_last=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=8,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)


backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True, weights=None)
backbone.fc = nn.Identity()

wrapped_backbone = spt.TeacherStudentWrapper(
    backbone,
    warm_init=True,
    base_ema_coefficient=0.996,
    final_ema_coefficient=1.0,
)

projector = nn.Sequential(
    nn.Linear(512, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 128),
)
wrapped_projector = spt.TeacherStudentWrapper(
    projector,
    warm_init=True,
    base_ema_coefficient=0.996,
    final_ema_coefficient=1.0,
)

predictor = nn.Sequential(
    nn.Linear(128, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 128),
)

module = spt.Module(
    backbone=wrapped_backbone,
    projector=wrapped_projector,
    predictor=predictor,
    forward=byol_forward,
    byol_loss=spt.losses.BYOLLoss(),
    optim={
        "optimizer": {
            "type": "LARS",
            "lr": 5,
            "weight_decay": 1e-6,
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
    probe=nn.Linear(512, 10),
    loss_fn=nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(10),
        "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
    },
)

knn_probe = spt.callbacks.OnlineKNN(
    name="knn_probe",
    input="embedding",
    target="label",
    queue_length=20000,
    metrics={"accuracy": torchmetrics.classification.MulticlassAccuracy(10)},
    input_dim=512,
    k=10,
)

wandb_logger = WandbLogger(
    entity="stable-ssl",
    project="cifar10-byol",
    name="byol-resnet18",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=1000,
    num_sanity_val_steps=0,
    callbacks=[linear_probe, knn_probe],
    precision="16-mixed",
    logger=wandb_logger,
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
