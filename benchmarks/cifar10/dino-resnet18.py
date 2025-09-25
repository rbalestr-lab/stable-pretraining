"""DINO training on CIFAR-10."""

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.forward import dino_forward
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import get_data_dir

# DINO transform: 2 global crops (same as SimCLR)
dino_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
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
    cifar_train, names=["image", "label"], transform=dino_transform, add_sample_idx=True
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
    batch_size=256,
    num_workers=8,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)


# Create backbone with teacher-student wrapper
backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True, weights=None)
backbone.fc = nn.Identity()

wrapped_backbone = spt.TeacherStudentWrapper(
    backbone,
    warm_init=True,
    base_ema_coefficient=0.99,  # Faster updates for CIFAR-10
    final_ema_coefficient=1.0,
)

# Create projector with teacher-student wrapper
# DINO uses 3-layer MLP with specific architecture
projector = nn.Sequential(
    nn.Linear(512, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 2048),  # Bottleneck to 256 dim
)

wrapped_projector = spt.TeacherStudentWrapper(
    projector,
    warm_init=True,
    base_ema_coefficient=0.99,  # Faster updates for CIFAR-10
    final_ema_coefficient=1.0,
)

module = spt.Module(
    backbone=wrapped_backbone,
    projector=wrapped_projector,
    forward=dino_forward,
    dino_loss=spt.losses.DINOLoss(
        temperature_student=0.1,
        center_momentum=0.9,
    ),
    warmup_temperature_teacher=0.04,
    temperature_teacher=0.07,
    warmup_epochs_temperature_teacher=30,
    # optim={
    #     "optimizer": {
    #         "type": "SGD",
    #         "lr": 0.03,  # Lower learning rate
    #         "momentum": 0.9,
    #         "weight_decay": 1e-4,  # Lower weight decay
    #     },
    #     "scheduler": {
    #         "type": "LinearWarmupCosineAnnealing",
    #     },
    #     "interval": "epoch",
    # },
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

# Teacher-Student callback for EMA updates
teacher_student_callback = spt.callbacks.TeacherStudentCallback(
    update_frequency=1,
    update_after_backward=False,
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
    project="cifar10-dino",
    name="dino-resnet18",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=1000,
    num_sanity_val_steps=0,
    callbacks=[teacher_student_callback, linear_probe, knn_probe],
    precision="16-mixed",
    logger=wandb_logger,
)

manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
