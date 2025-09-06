import lightning as pl
import torch
import torchmetrics
import torchvision
from torch import nn
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining.data import transforms

# Define augmentations for SimCLR (creates 2 views of each image)
simclr_transform = transforms.MultiViewTransform(
    [
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
        # Second view with slightly different augmentations
        transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.ToImage(**spt.data.static.CIFAR10),
        ),
    ]
)

# Load CIFAR-10 and wrap in dictionary format
cifar_train = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True)
cifar_val = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True)

train_dataset = spt.data.FromTorchDataset(
    cifar_train,
    names=["image", "label"],  # Convert tuple to dictionary
    transform=simclr_transform,
)

val_dataset = spt.data.FromTorchDataset(
    cifar_val,
    names=["image", "label"],
    transform=transforms.Compose(
        transforms.RGB(),
        transforms.Resize((32, 32)),
        transforms.ToImage(**spt.data.static.CIFAR10),
    ),
)

# Create dataloaders with view sampling for contrastive learning
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=spt.data.sampler.RepeatedRandomSampler(train_dataset, n_views=2),
    batch_size=256,
    num_workers=8,
    drop_last=True,
)

val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=256,
    num_workers=10,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

# Define the forward function (replaces training_step in PyTorch Lightning)


def forward(self, batch, stage):
    out = {}
    out["embedding"] = self.backbone(batch["image"])
    if self.training:
        # Project embeddings and compute contrastive loss
        proj = self.projector(out["embedding"])
        views = spt.data.fold_views(proj, batch["sample_idx"])
        out["loss"] = self.simclr_loss(views[0], views[1])
    return out


# Build model components
backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
backbone.fc = torch.nn.Identity()  # Remove classification head

projector = nn.Sequential(
    nn.Linear(512, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 256),
)

# Create the module with all components
module = spt.Module(
    backbone=backbone,
    projector=projector,
    forward=forward,
    simclr_loss=spt.losses.NTXEntLoss(temperature=0.5),
    optim={
        "optimizer": {"type": "LARS", "lr": 5, "weight_decay": 1e-6},
        "scheduler": {"type": "LinearWarmupCosineAnnealing"},
        "interval": "epoch",
    },
)

# Add callbacks for monitoring performance during training
linear_probe = spt.callbacks.OnlineProbe(
    name="linear_probe",
    input="embedding",
    target="label",
    probe=torch.nn.Linear(512, 10),
    loss_fn=torch.nn.CrossEntropyLoss(),
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

# Configure training
trainer = pl.Trainer(
    max_epochs=1000,
    callbacks=[knn_probe, linear_probe],  # Monitor SSL quality in real-time
    precision="16-mixed",
    logger=WandbLogger(project="cifar10-simclr"),
)

# Launch training
manager = spt.Manager(trainer=trainer, module=module, data=data)
manager()
