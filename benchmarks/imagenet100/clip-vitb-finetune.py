from functools import partial
from lightning.pytorch.loggers import WandbLogger
import torch
import torchmetrics
import torch.nn.functional as F
import lightning as pl
from transformers import (
    AutoTokenizer,
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
)

import stable_pretraining as spt

num_devices = 1
global_batch = 1024
batch_size = global_batch // num_devices  # per-GPU

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
vision_model = CLIPVisionModelWithProjection.from_pretrained(
    "openai/clip-vit-base-patch32", trust_remote_code=True
)
text_model = CLIPTextModelWithProjection.from_pretrained(
    "openai/clip-vit-base-patch32", trust_remote_code=True
)


def tokenize(text: str | list[str], tokenizer: AutoTokenizer) -> torch.Tensor:
    data = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    return data["input_ids"]


image_transform = spt.data.transforms.Compose(
    spt.data.transforms.Resize((224, 224)),
    spt.data.transforms.RGB(),
    spt.data.transforms.ToImage(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
)

val_dataset = spt.data.HFDataset(
    "clane9/imagenet-100",
    split="validation",
    transform=image_transform,
)
train_dataset = spt.data.HFDataset(
    "clane9/imagenet-100",
    split="train",
    transform=image_transform,
)
# just one batch for finetuning, and we set lr to 0, as a dummy
train_dataset = spt.data.Subset(train_dataset, range(global_batch))
classes = val_dataset.dataset.features["label"].names

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=32,
    shuffle=False,
    pin_memory=True,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)


def forward(self: spt.Module, batch: dict, stage: str) -> dict:
    out = {}
    vision_outputs = self.vision_model(pixel_values=batch["image"])
    image_embeds = F.normalize(vision_outputs.image_embeds, dim=-1)
    out["image_embeds"] = image_embeds

    if self.training:
        out["loss"] = 0.0 * self.clip_loss(image_embeds, image_embeds)

    return out


module = spt.Module(
    vision_model=vision_model,
    text_model=text_model,
    forward=forward,
    clip_loss=spt.losses.CLIPLoss(temperature=0.07),
    optim={
        "optimizer": {
            "type": "AdamW",
            "lr": 0.0,
            "weight_decay": 0.0,
            "betas": (0.9, 0.98),
        },
    },
)

wandb_logger = WandbLogger(
    entity="stable-pretraining",
    project="imagenet100-clip",
    name="imagenet100-finetune-clip-vit-b32",
    log_model=False,
)

zero_shot_callback = spt.callbacks.CLIPZeroShot(
    name="clip",
    image_key="image",
    class_key="label",
    class_names=classes,
    image_backbone=vision_model,
    text_backbone=text_model,
    tokenizer_fn=partial(tokenize, tokenizer=tokenizer),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(
            num_classes=len(classes)
        ),
        "top5": torchmetrics.classification.MulticlassAccuracy(
            num_classes=len(classes), top_k=5
        ),
        "top10": torchmetrics.classification.MulticlassAccuracy(
            num_classes=len(classes), top_k=10
        ),
    },
)

trainer = pl.Trainer(
    max_epochs=9,
    num_sanity_val_steps=0,
    callbacks=[zero_shot_callback],
    precision="bf16-mixed",
    logger=wandb_logger,
    enable_checkpointing=False,
    devices=num_devices,
    accelerator="gpu",
    strategy="ddp_find_unused_parameters_true",
)

# Run training (resume optional)
manager = spt.Manager(
    trainer=trainer,
    module=module,
    data=data,
    ckpt_path=None,
)
manager()
