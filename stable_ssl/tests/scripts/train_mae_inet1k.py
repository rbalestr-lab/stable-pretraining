import math

import lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision
from lightning.pytorch.loggers import WandbLogger
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from lightning.pytorch.strategies import DDPStrategy

import stable_ssl as ssl
from stable_ssl.data import transforms
from stable_ssl.data.utils import Dataset
from stable_ssl.utils.pos_embed import get_2d_sincos_pos_embed


train_batch_size = 128
val_batch_size = 128
num_workers = 32
num_classes = 1000

# TODO
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
height, width, patch_size = 256, 256, 16
crop_height, crop_width = 224, 224
num_patches = (crop_height // patch_size) * (crop_width // patch_size)
patch_channel_dim = 3 * patch_size * patch_size
mask_ratio = 0.75
num_visible_patches = int(num_patches * (1 - mask_ratio))

mask_transform_kwargs = dict(
    patch_size=patch_size,
    mask_ratio=mask_ratio,
    source="image",
    target_visible="mask_visible",
    target_masked="mask_masked",
)

train_transform = transforms.Compose(
    transforms.RandomResizedCrop((crop_height, crop_width), scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    transforms.RandomHorizontalFlip(),
    transforms.RandomMask(**mask_transform_kwargs),
    transforms.ToImage(mean=mean, std=std),
)

val_transform = transforms.Compose(
    transforms.RGB(),
    transforms.Resize((height, width)),
    transforms.CenterCrop((height, width)),
    transforms.ToImage(mean=mean, std=std),
)


inet1k_train = ssl.data.HFDataset(
    path="ILSVRC/imagenet-1k",
    split="train",
    transform=train_transform,
)

inet1k_val = ssl.data.HFDataset(
    path="ILSVRC/imagenet-1k",
    split="validation",
    transform=val_transform,
)


train = torch.utils.data.DataLoader(
    dataset=inet1k_train,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
    collate_fn=torch.utils.data.default_collate,
    pin_memory=True,
    persistent_workers=True,
)

val = torch.utils.data.DataLoader(
    dataset=inet1k_val,
    batch_size=val_batch_size,
    num_workers=num_workers,
    shuffle=False,
    collate_fn=torch.utils.data.default_collate,
    pin_memory=True,
    persistent_workers=True,
)

data = ssl.data.DataModule(train=train, val=val)


def pos_embed(patches: torch.Tensor, with_cls: bool = True) -> torch.Tensor:
    embed = (
        torch.from_numpy(
            get_2d_sincos_pos_embed(patches.shape[-1], int(math.sqrt(patches.shape[1])))
        )
        .to(patches.device)
        .float()
        .repeat(patches.shape[0], 1, 1)
    )
    if with_cls:
        embed = torch.cat([
            torch.zeros(embed.shape[0], 1, embed.shape[2], device=embed.device),
            embed
        ], dim=1)

    return embed


def apply_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply single mask to tensor"""
    B, N, D = x.shape
    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(x, dim=1, index=mask_expanded)


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    images: [B, C, H, W] -> [B, N, P*P*C]
    """
    B, C, H, W = images.shape
    P = patch_size
    assert H % P == 0 and W % P == 0
    h = H // P
    w = W // P
    x = images.reshape(B, C, h, P, w, P)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, P * P * C)
    return x


class MAE_Encoder(VisionTransformer):
    def __init__(self, *args, **kwargs):
        mae_in_dim = kwargs.pop('mae_in_dim', 3 * patch_size * patch_size)
        super().__init__(*args, **kwargs)
        # number of patch tokens (no cls)
        self.patch_size = kwargs.get('patch_size', 16)
        self.num_patches = self.patch_embed.num_patches
        self.mae_patch_project = nn.Linear(mae_in_dim, self.embed_dim)

    def project_patches(self, patches: torch.Tensor) -> torch.Tensor:
        return self.mae_patch_project(patches)


class MAE_Decoder(VisionTransformer):
    def __init__(self, *args, **kwargs):
        mae_enc_dim = kwargs.pop('mae_enc_dim', 768)
        super().__init__(*args, **kwargs)
        in_chans = kwargs.get('in_chans', 3)
        patch_size = kwargs.get('patch_size', 16)
        # tokens in the decoded grid (no cls)
        self.num_patches = self.patch_embed.num_patches
        self.decoder_embed = nn.Linear(mae_enc_dim, self.embed_dim)
        # mask token for missing positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # predict pixel values per token (P^2 * C)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.decoder_pred = nn.Linear(self.embed_dim, (patch_size ** 2) * in_chans)


def _forward_decoder(self, batch: dict, out: dict, stage) -> torch.Tensor:
    decoder: MAE_Decoder = self.decoder
    patches_visible = out["embeddings"]
    inverse_shuffle = batch["ids_restore"].unsqueeze(-1).expand(-1, -1, decoder.embed_dim)
    # project encoded patches to decoder's space
    decoder_patches = decoder.decoder_embed(patches_visible)

    patches_cls, patches_visible = torch.split(decoder_patches, [1, decoder_patches.shape[1] - 1], dim=1)
    batch_size  = patches_visible.shape[0]
    num_patches = decoder.num_patches
    num_visible = patches_visible.shape[1]

    mask_tokens = decoder.mask_token.expand(batch_size, num_patches - num_visible, -1)
    # combine visible patches and mask tokens then unshuffle into place
    patches = torch.cat([patches_visible, mask_tokens], dim=1)
    unshuffled_patches = torch.gather(patches, dim=1, index=inverse_shuffle)
    patches = torch.cat([patches_cls, unshuffled_patches], dim=1)

    pe = pos_embed(patches[:, 1:, :], with_cls=True)
    patches = patches + pe
    for blk in decoder.blocks:
        patches = blk(patches)
    patches = decoder.norm(patches)
    # remove cls token
    patches = patches[:, 1:, :]
    patches = decoder.decoder_pred(patches)
    return patches


def forward(self, batch: dict, stage):
    out = {}
    encoder: MAE_Encoder = self.encoder
    images = batch["image"]
    image_patches = patchify(images, patch_size)
    patches = encoder.project_patches(image_patches)
    posemb_cls, posemb_patches = torch.split(pos_embed(patches, with_cls=True), [1, patches.shape[1]], dim=1)
    patches = patches + posemb_patches
    cls_tok = encoder.cls_token + posemb_cls
    
    if self.training:
        indices_keep, indices_masked = batch["mask_visible"], batch["mask_masked"]
        patches_visible = apply_mask(patches, indices_keep)
        patches_visible = torch.cat([cls_tok, patches_visible], dim=1)
        for blk in encoder.blocks:
            patches_visible = blk(patches_visible)
        patches_visible = encoder.norm(patches_visible)
        out["embeddings"] = patches_visible
        out["reconstructed_pixel_patches"] = _forward_decoder(self, batch, out, stage)
        out["loss"] = self.loss_fn(
            apply_mask(out["reconstructed_pixel_patches"], indices_masked),
            apply_mask(image_patches, indices_masked)
        )
    else:
        patches = torch.cat([cls_tok, patches], dim=1)
        for blk in encoder.blocks:
            patches = blk(patches)
        patches = encoder.norm(patches)
        out["embeddings"] = patches
    
    # exclude cls token for rankme (flat_embedding) and linear probe (sum_embedding)
    out["flat_embedding"] = out["embeddings"][:, 1:, :].flatten(start_dim=1)
    out["sum_embedding"] = out["embeddings"][:, 1:, :].sum(dim=1)
    return out


encoder_kwargs = dict(
    img_size=(crop_height, crop_width),
    patch_size=patch_size,
    embed_dim=768,
    depth=16,
    num_heads=16,
    qkv_bias=True,  # MAE typically uses bias
    mae_in_dim=patch_channel_dim,
)

decoder_kwargs = dict(
    img_size=(crop_height, crop_width),
    patch_size=patch_size,
    mae_enc_dim=768,
    embed_dim=512,
    depth=8,
    num_heads=16,
)


module = ssl.Module(
    encoder=MAE_Encoder(**encoder_kwargs),
    decoder=MAE_Decoder(**decoder_kwargs),
    forward=forward,
    loss_fn=F.mse_loss,  # pixel MSE loss. we make implicit assumption that norm-pix-loss is False
)

# Note: Linear probe uses visible patches only during training
linear_probe = ssl.callbacks.OnlineProbe(
    "linear_probe",
    "sum_embedding",
    "label", 
    probe=torch.nn.Linear(768, num_classes),
    loss_fn=torch.nn.CrossEntropyLoss(),
    metrics={
        "top1": torchmetrics.classification.MulticlassAccuracy(num_classes),
        "top5": torchmetrics.classification.MulticlassAccuracy(num_classes, top_k=5),
    },
)

# RankMe on encoder outputs
rankme = ssl.callbacks.RankMe(
    name="rankme",
    target="flat_embedding", 
    queue_length=min(512, train_batch_size),
    target_shape=(num_visible_patches, 768), 
)

# Initialize W&B logger
wandb_logger = WandbLogger(
    project="mae-inet1k",
    entity="slightly-more-badass",
    name="mae-inet1k-run",
    log_model=False,
    offline=True,
)

trainer = pl.Trainer(
    max_epochs=6,
    num_sanity_val_steps=0,
    callbacks=[linear_probe, rankme],
    precision="16-mixed",
    logger=wandb_logger,
    enable_checkpointing=False,
    accelerator="gpu",
    devices=8,
    strategy=DDPStrategy(
        find_unused_parameters=True,
        static_graph=True,
        gradient_as_bucket_view=True,
    )
)

manager = ssl.Manager(trainer=trainer, module=module, data=data)
manager()