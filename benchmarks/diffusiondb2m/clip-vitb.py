import torch
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from transformers import (
    AutoTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)
import stable_pretraining as spt
from functools import partial


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--num_devices", type=int, default=8)
parser.add_argument("--global_batch", type=int, default=4096)
parser.add_argument("--num_epochs", type=int, default=8)
parser.add_argument("--val_percent", type=float, default=0.10)
parser.add_argument("--resume_ckpt_path", type=str, default=None)
args = parser.parse_args()

lr = args.lr
num_devices = args.num_devices
global_batch = args.global_batch
batch_size = global_batch // num_devices
num_epochs = args.num_epochs
val_percent = args.val_percent
resume_ckpt_path = args.resume_ckpt_path

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
vision_model = CLIPVisionModelWithProjection.from_pretrained(
    "openai/clip-vit-base-patch32", trust_remote_code=True
)
text_model = CLIPTextModelWithProjection.from_pretrained(
    "openai/clip-vit-base-patch32", trust_remote_code=True
)


def tokenize(text: str, tokenizer: AutoTokenizer):
    data = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    return data["input_ids"].squeeze(0), data["attention_mask"].squeeze(0)


image_transform = spt.data.transforms.Compose(
    spt.data.transforms.Resize((224, 224)),
    spt.data.transforms.ToImage(
        mean=[0.481, 0.457, 0.408],
        std=[0.268, 0.261, 0.275],
    ),
    spt.data.transforms.LambdaTransform(
        fn=partial(tokenize, tokenizer=tokenizer),
        source="prompt",
        targets=("tokenized_prompt", "attention_mask"),
    ),
)


train_base = spt.data.HFDataset(
    "poloclub/diffusiondb",
    "2m_all",
    split="train",
    transform=image_transform,
    remove_columns=[
        "timestamp",
        "user_name",
        "prompt_nsfw",
        "image_nsfw",
        "sampler",
    ],
)

size = len(train_base)
val_n = int(size * val_percent)
val_dataset = spt.data.Subset(train_base, range(0, val_n))
train_dataset = spt.data.Subset(train_base, range(val_n, size))


train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=16,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=8,
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)


def forward(self: spt.Module, batch: dict, stage: str) -> dict:
    out = {}
    vision_outputs = self.vision_model(pixel_values=batch["image"])
    image_embeds = F.normalize(vision_outputs.image_embeds, dim=-1)

    text_outputs = self.text_model(
        input_ids=batch["tokenized_prompt"], attention_mask=batch["attention_mask"]
    )
    text_embeds = F.normalize(text_outputs.text_embeds, dim=-1)

    out["image_embeds"] = image_embeds
    out["text_embeds"] = text_embeds

    if self.training:
        out["loss"] = self.clip_loss(image_embeds, text_embeds)
    return out


class CLIPMonitor(pl.Callback):
    """PyTorch Lightning callback that logs CLIP-style training metrics.

    Computes retrieval (R@1), contrastive statistics (pos prob, margin, entropy),
    and embedding alignment (cosine sim, norms) from image/text embeddings, and
    logs them during training and validation.
    """

    def __init__(self, log_every_n_steps: int = 10):
        super().__init__()
        self.every = log_every_n_steps
        self.scale = None  # 1 / temperature

    @torch.no_grad()
    def on_fit_start(self, trainer: pl.Trainer, pl_module):
        T = pl_module.clip_loss.temperature
        T = float(T.item()) if torch.is_tensor(T) else float(T)
        self.scale = 1.0 / T
        trainer.logger.log_metrics(
            {"config/temperature": T, "config/logit_scale": self.scale},
            step=trainer.global_step,
        )

    @torch.no_grad()
    def _log(self, trainer: pl.Trainer, outputs: dict, stage: str):
        img = F.normalize(outputs["image_embeds"], dim=-1)
        txt = F.normalize(outputs["text_embeds"], dim=-1)

        logits = self.scale * (img @ txt.T)  # [B, B]
        B = logits.size(0)
        diag = torch.arange(B, device=logits.device)

        probs = logits.softmax(dim=1)
        pos_prob = probs[diag, diag]
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)

        neg = logits.masked_fill(
            torch.eye(B, dtype=torch.bool, device=logits.device), float("-inf")
        )
        top_neg = neg.max(dim=1).values
        margin = logits[diag, diag] - top_neg

        r1_i2t = (logits.argmax(dim=1) == diag).float().mean()
        r1_t2i = (logits.argmax(dim=0) == diag).float().mean()

        cos_pos = F.cosine_similarity(img, txt, dim=-1).mean()
        img_norm = img.norm(dim=-1).mean()
        txt_norm = txt.norm(dim=-1).mean()

        trainer.logger.log_metrics(
            {
                f"{stage}/retrieval/R@1_i2t": float(r1_i2t.cpu()),
                f"{stage}/retrieval/R@1_t2i": float(r1_t2i.cpu()),
                f"{stage}/contrast/pos_prob": float(pos_prob.mean().cpu()),
                f"{stage}/contrast/margin": float(margin.mean().cpu()),
                f"{stage}/contrast/entropy": float(entropy.mean().cpu()),
                f"{stage}/align/cos_pos": float(cos_pos.cpu()),
                f"{stage}/embed/img_norm": float(img_norm.cpu()),
                f"{stage}/embed/txt_norm": float(txt_norm.cpu()),
            },
            step=trainer.global_step,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every == 0:
            self._log(trainer, outputs, "train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._log(trainer, outputs, "val")


module = spt.Module(
    vision_model=vision_model,
    text_model=text_model,
    forward=forward,
    clip_loss=spt.losses.CLIPLoss(temperature=0.07),
    optim={
        "optimizer": {
            "type": "AdamW",
            "lr": lr,
            "weight_decay": (wd := 1.0e-6),
            "betas": (0.9, 0.98),
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealing",
            "total_steps": (len(train_dataloader) // num_devices) * num_epochs,
            "peak_step": 0.1,
        },
        "interval": "step",
    },
)

wandb_logger = WandbLogger(
    entity="stable-pretraining",
    project="diffusiondb2m-clip",
    name="clip-vit-b32-diffusiondb2m-32k",
    log_model=False,
)

trainer = pl.Trainer(
    max_epochs=num_epochs,
    num_sanity_val_steps=0,
    callbacks=[
        ModelCheckpoint(
            monitor="train/loss_step",  # this appears in your progress bar
            mode="min",
            every_n_epochs=1,
            save_top_k=-1,
            dirpath="/your/path/to/checkpoints",
        ),
        LearningRateMonitor(logging_interval="step"),
        CLIPMonitor(log_every_n_steps=10),
    ],
    precision="bf16-mixed",
    logger=wandb_logger,
    enable_checkpointing=True,
    devices=num_devices,
    accelerator="gpu",
    strategy="ddp",
)

# Run training (resume optional)
manager = spt.Manager(
    trainer=trainer,
    module=module,
    data=data,
    ckpt_path=resume_ckpt_path,
)

manager()
