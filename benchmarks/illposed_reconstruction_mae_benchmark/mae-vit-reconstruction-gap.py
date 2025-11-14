import argparse
import json
import sys
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining.data import transforms

import importlib.util

local_utils_path = Path(__file__).parent / "utils.py"
spec = importlib.util.spec_from_file_location("imagenette_utils", local_utils_path)
imagenette_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imagenette_utils)
parse_decoder_type = imagenette_utils.parse_decoder_type
create_mae_with_custom_decoder = imagenette_utils.create_mae_with_custom_decoder
save_fixed_samples = imagenette_utils.save_fixed_samples
load_mae_dataset = imagenette_utils.load_mae_dataset
detect_image_properties_from_dataset = imagenette_utils.detect_image_properties_from_dataset

dataset_configs_path = Path(__file__).parent / "dataset_configs.py"
spec = importlib.util.spec_from_file_location("dataset_configs", dataset_configs_path)
dataset_configs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_configs)
get_dataset_config = dataset_configs.get_dataset_config

benchmarks_utils_path = Path(__file__).parent.parent / "utils.py"
spec = importlib.util.spec_from_file_location("benchmarks_utils", benchmarks_utils_path)
benchmarks_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmarks_utils)
get_data_dir = benchmarks_utils.get_data_dir


def mae_reconstruction_gap_forward(self, batch, stage):
    imgs = batch["image"]
    latent, pred, mask = self.backbone(imgs, mask_ratio=self.mask_ratio)

    target = self.backbone.patchify(imgs)
    loss_rec = spt.losses.mae(target, pred, mask)

    batch["embedding"] = latent[:, 0]
    batch["loss_rec"] = loss_rec

    log_stage = "train" if stage == "fit" else ("val" if stage == "validate" else stage)

    self.log(f"{log_stage}/loss_rec", loss_rec, on_step=False, on_epoch=True, sync_dist=True)

    if hasattr(self, "classifier") and self.classifier is not None:
        if self.alpha == 0.0:
            with torch.no_grad():
                logits = self.classifier(batch["embedding"].detach())
        else:
            logits = self.classifier(batch["embedding"])

        loss_cls = F.cross_entropy(logits, batch["label"])
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch["label"]).float().mean()

        if self.alpha != 0.0:
            batch["loss"] = loss_rec + self.alpha * loss_cls
        else:
            batch["loss"] = loss_rec

        batch["loss_cls"] = loss_cls
        batch["supervised_acc"] = acc
        self.log(f"{log_stage}/supervised_cls_loss", loss_cls, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{log_stage}/supervised_cls_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{log_stage}/loss", batch["loss"], on_step=False, on_epoch=True, sync_dist=True)
    else:
        batch["loss"] = loss_rec
        self.log(f"{log_stage}/loss", batch["loss"], on_step=False, on_epoch=True, sync_dist=True)

    return batch


class ReconstructionGapCallback(pl.Callback):

    def __init__(self, dataset_name="imagenette", log_dir=None):
        super().__init__()
        self.trajectory = []
        self.dataset_name = dataset_name
        self.log_dir = log_dir

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        mse = metrics.get("val/loss_rec", None)

        acc = metrics.get("val/supervised_cls_acc", None)

        if mse is not None and acc is not None:
            self.trajectory.append({
                "epoch": trainer.current_epoch,
                "mse": float(mse),
                "accuracy": float(acc) * 100,
                "alpha": float(pl_module.alpha),
                "lr": float(pl_module.lr),  # Track learning rate
            })

    def on_train_end(self, trainer, pl_module):
        alpha_str = str(pl_module.alpha).replace("-", "neg")
        lr_str = f"{pl_module.lr:.0e}".replace("-", "neg")
        if self.log_dir is not None:
            log_dir = Path(self.log_dir) / f"{alpha_str}_lr{lr_str}"
        else:
            log_dir = Path("outputs") / f"{self.dataset_name}-mae-reconstruction-gap" / f"{alpha_str}_lr{lr_str}"
        output_path = log_dir / "trajectory.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump({
                "alpha": float(pl_module.alpha),
                "lr": float(pl_module.lr),
                "model": pl_module.model_name,
                "decoder": pl_module.decoder_type,
                "trajectory": self.trajectory
            }, f, indent=2)

        print(f"saved: {output_path}")

        if trainer.logger is not None and hasattr(trainer.logger.experiment, 'dir'):
            wandb_path = Path(trainer.logger.experiment.dir) / "trajectory.json"
            wandb_path.parent.mkdir(parents=True, exist_ok=True)
            with open(wandb_path, "w") as f:
                json.dump({
                    "alpha": float(pl_module.alpha),
                    "lr": float(pl_module.lr),
                    "model": pl_module.model_name,
                    "decoder": pl_module.decoder_type,
                    "trajectory": self.trajectory
                }, f, indent=2)
            print(f"wandb: {wandb_path}")


class MAEVisualizationCallback(pl.Callback):

    def __init__(self, fixed_samples_path, viz_interval=10, num_samples=8, save_dir="reconstruction_viz"):
        super().__init__()
        self.fixed_samples_path = Path(fixed_samples_path)
        self.viz_interval = viz_interval
        self.num_samples = num_samples
        self.save_dir = Path(save_dir)
        self.fixed_images = None
        self.fixed_labels = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.viz_interval != 0:
            return

        if self.fixed_images is None:
            if not self.fixed_samples_path.exists():
                print(f"no fixed samples at {self.fixed_samples_path}")
                return

            data = torch.load(self.fixed_samples_path)
            self.fixed_images = data['images'][:self.num_samples]
            self.fixed_labels = data['labels'][:self.num_samples]

        device = pl_module.device
        imgs = self.fixed_images.to(device)

        with torch.no_grad():
            latent, pred, mask = pl_module.backbone(imgs, mask_ratio=pl_module.mask_ratio)
            reconstructed = pl_module.backbone.unpatchify(pred)

            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

            imgs_vis = imgs * std + mean
            reconstructed_vis = reconstructed * std + mean

            imgs_vis = torch.clamp(imgs_vis, 0, 1)
            reconstructed_vis = torch.clamp(reconstructed_vis, 0, 1)

            mask_expanded = mask.unsqueeze(-1).repeat(1, 1, pl_module.backbone.patch_embed.patch_size[0]**2 * 3)
            mask_expanded = pl_module.backbone.unpatchify(mask_expanded)
            imgs_masked = imgs_vis * (1 - mask_expanded)
            imgs_with_mask_overlay = imgs_vis * (1 - mask_expanded) + reconstructed_vis * mask_expanded

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            self.save_dir.mkdir(parents=True, exist_ok=True)

            n_samples = min(self.num_samples, imgs.shape[0])
            fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

            if n_samples == 1:
                axes = axes.reshape(1, -1)

            for i in range(n_samples):
                axes[i, 0].imshow(imgs_vis[i].cpu().permute(1, 2, 0).numpy())
                axes[i, 0].set_title(f"Original (label={self.fixed_labels[i]})" if i == 0 else "")
                axes[i, 0].axis('off')
                axes[i, 0].text(0.02, 0.98, f"Sample {i+1}", transform=axes[i, 0].transAxes,
                               verticalalignment='top', fontsize=10, color='white',
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

                axes[i, 1].imshow(imgs_masked[i].cpu().permute(1, 2, 0).numpy())
                axes[i, 1].set_title("Masked" if i == 0 else "")
                axes[i, 1].axis('off')

                axes[i, 2].imshow(imgs_with_mask_overlay[i].cpu().permute(1, 2, 0).numpy())
                axes[i, 2].set_title("Reconstruction with Mask" if i == 0 else "")
                axes[i, 2].axis('off')

                axes[i, 3].imshow(reconstructed_vis[i].cpu().permute(1, 2, 0).numpy())
                axes[i, 3].set_title("Full Reconstruction" if i == 0 else "")
                axes[i, 3].axis('off')

            fig.suptitle(f"Epoch {trainer.current_epoch} - MAE Reconstructions",
                        fontsize=16, fontweight='bold', y=0.995)

            plt.tight_layout()
            save_path = self.save_dir / f"epoch_{trainer.current_epoch:03d}.png"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            print(f"viz ep{trainer.current_epoch}: {save_path}")

            if trainer.logger is not None:
                try:
                    import wandb
                    alpha_str = f"alpha_{pl_module.alpha:.1f}".replace("-", "neg").replace(".", "p")
                    trainer.logger.experiment.log({
                        f"reconstructions/{alpha_str}/epoch_{trainer.current_epoch:03d}": wandb.Image(str(save_path))
                    })
                except Exception as e:
                    print(f"wandb log failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="MAE Reconstruction Gap Benchmark")

    parser.add_argument("--dataset", type=str, default="imagenette",
                        choices=["imagenette", "tiny-imagenet", "imagenet100", "imagenet"],
                        help="Dataset to use")

    parser.add_argument("--model", type=str, default="vit_tiny",
                        choices=["vit_tiny", "vit_small", "vit_base", "vit_large"],
                        help="Encoder model size")
    parser.add_argument("--decoder_type", type=str, default="base-8b",
                        help="Decoder configuration: 'linear' or '<size>-<depth>b' (e.g., 'base-8b', 'tiny-4b')")
    parser.add_argument("--mask_ratio", type=float, default=0.75,
                        help="Masking ratio (default: 0.75)")

    parser.add_argument("--encoder_embed_dim", type=int, default=None,
                        help="Override encoder embedding dimension")
    parser.add_argument("--encoder_depth", type=int, default=None,
                        help="Override encoder depth (number of layers)")
    parser.add_argument("--encoder_num_heads", type=int, default=None,
                        help="Override encoder number of attention heads")
    parser.add_argument("--decoder_embed_dim", type=int, default=None,
                        help="Override decoder embedding dimension")
    parser.add_argument("--decoder_depth", type=int, default=None,
                        help="Override decoder depth (number of layers)")
    parser.add_argument("--decoder_num_heads", type=int, default=None,
                        help="Override decoder number of attention heads")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="Patch size for Vision Transformer")

    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Lambda: supervised regularization weight (can be negative)")

    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epochs")

    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    parser.add_argument("--project", type=str, default=None,
                        help="WandB project name (default: {dataset}-mae-reconstruction-gap)")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory for trajectory logs (default: outputs/{dataset}-mae-reconstruction-gap)")

    parser.add_argument("--fixed_samples_dir", type=str, default="fixed_samples/",
                        help="Directory to save/load fixed validation samples")
    parser.add_argument("--viz_interval", type=int, default=10,
                        help="Log reconstruction visualizations every N epochs")

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    dataset_config = get_dataset_config(args.dataset)

    if args.project is None:
        args.project = f"{args.dataset}-mae-reconstruction-gap"

    decoder_config = parse_decoder_type(args.decoder_type)
    if args.decoder_embed_dim is not None:
        decoder_config["embed_dim"] = args.decoder_embed_dim
    if args.decoder_depth is not None:
        decoder_config["depth"] = args.decoder_depth
    if args.decoder_num_heads is not None:
        decoder_config["num_heads"] = args.decoder_num_heads

    data_dir = get_data_dir(args.dataset)

    DEFAULT_IMG_SIZE = 224
    default_resize_size = int(DEFAULT_IMG_SIZE * 256 / 224)

    train_transform = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToImage(**spt.data.static.ImageNet),
    )

    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((default_resize_size, default_resize_size)),
        transforms.CenterCrop((DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE)),
        transforms.ToImage(**spt.data.static.ImageNet),
    )

    train_dataset = load_mae_dataset(
        args.dataset,
        "train",
        train_transform,
        data_dir,
    )

    val_dataset = load_mae_dataset(
        args.dataset,
        "val",
        val_transform,
        data_dir,
    )

    print(f"train {len(train_dataset)}, val {len(val_dataset)}")

    img_properties = detect_image_properties_from_dataset(train_dataset)
    detected_img_size = img_properties['img_size']

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=True,
        persistent_workers=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    data = spt.data.DataModule(train=train_dataloader, val=val_dataloader)

    fixed_samples_dir_full = Path(args.fixed_samples_dir) / args.dataset / f"size_{detected_img_size}x{detected_img_size}"
    fixed_samples_path = fixed_samples_dir_full / "fixed_val_samples.pt"
    save_fixed_samples(
        val_dataloader,
        str(fixed_samples_dir_full),
        num_samples=16,
        seed=args.seed
    )

    encoder_overrides = {}
    encoder_overrides["img_size"] = detected_img_size

    if args.encoder_embed_dim is not None:
        encoder_overrides["embed_dim"] = args.encoder_embed_dim
    if args.encoder_depth is not None:
        encoder_overrides["depth"] = args.encoder_depth
    if args.encoder_num_heads is not None:
        encoder_overrides["num_heads"] = args.encoder_num_heads
    if args.patch_size != 16:
        encoder_overrides["patch_size"] = args.patch_size

    backbone = create_mae_with_custom_decoder(args.model, decoder_config, encoder_overrides)

    model_dims = {"vit_tiny": 192, "vit_small": 384, "vit_base": 768, "vit_large": 1024}
    model_depths = {"vit_tiny": 12, "vit_small": 12, "vit_base": 12, "vit_large": 24}
    model_heads = {"vit_tiny": 3, "vit_small": 6, "vit_base": 12, "vit_large": 16}

    actual_encoder_dim = encoder_overrides.get("embed_dim", model_dims[args.model])
    actual_encoder_depth = encoder_overrides.get("depth", model_depths[args.model])
    actual_encoder_heads = encoder_overrides.get("num_heads", model_heads[args.model])
    actual_patch_size = encoder_overrides.get("patch_size", 16)

    actual_decoder_dim = decoder_config.get("embed_dim", 512)
    actual_decoder_depth = decoder_config.get("depth", 8)
    actual_decoder_heads = decoder_config.get("num_heads", 16)
    decoder_type_name = decoder_config.get("type", "transformer")

    encoder_name = f"enc{actual_encoder_dim}d{actual_encoder_depth}h{actual_encoder_heads}p{actual_patch_size}"
    if decoder_type_name == "linear":
        decoder_name = "dec-linear"
    else:
        decoder_name = f"dec{actual_decoder_dim}d{actual_decoder_depth}h{actual_decoder_heads}"

    encoder_dim = actual_encoder_dim
    num_classes = dataset_config["num_classes"]

    classifier = nn.Linear(encoder_dim, num_classes)
    print(f"{encoder_name}/{decoder_name}, λ={args.alpha}")

    if args.alpha == 0.0:
        optim_config = {
            "modules": "backbone",
            "optimizer": {
                "type": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
        }
    else:
        optim_config = {
            "optimizer": {
                "type": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
        }

    module = spt.Module(
        backbone=backbone,
        classifier=classifier,
        forward=mae_reconstruction_gap_forward,
        alpha=args.alpha,
        mask_ratio=args.mask_ratio,
        model_name=encoder_name,
        decoder_type=decoder_name,
        lr=args.lr,
        optim=optim_config,
    )

    gap_callback = ReconstructionGapCallback(dataset_name=args.dataset, log_dir=args.log_dir)

    viz_callback = MAEVisualizationCallback(
        fixed_samples_path=fixed_samples_path,
        viz_interval=args.viz_interval,
        num_samples=8
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    model_config_name = f"{encoder_name}_{decoder_name}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.output_dir) / f"{args.dataset}-checkpoints" / f"{model_config_name}_lambda{args.alpha}_seed{args.seed}",
        filename="epoch{epoch:03d}",
        every_n_epochs=10,
        save_last=True,
    )


    callbacks = [gap_callback, viz_callback, lr_monitor, checkpoint_callback]

    wandb_logger = WandbLogger(
        entity=args.entity,
        project=args.project,
        name=f"{args.dataset}-{model_config_name}-lambda{args.alpha}-lr{args.lr}-seed{args.seed}",
        group=f"{args.dataset}-{encoder_name}-{decoder_name}-sweep",
        tags=[
            f"dataset_{args.dataset}",
            f"lambda_{args.alpha}",
            f"lr_{args.lr}",
            f"encoder_{encoder_name}",
            f"decoder_{decoder_name}",
        ],
        config={
            "dataset": args.dataset,
            "lambda": args.alpha,
            "lr": args.lr,
            "encoder": encoder_name,
            "decoder": decoder_name,
            "encoder_dim": actual_encoder_dim,
            "encoder_depth": actual_encoder_depth,
            "encoder_heads": actual_encoder_heads,
            "patch_size": actual_patch_size,
            "decoder_dim": actual_decoder_dim,
            "decoder_depth": actual_decoder_depth,
            "decoder_heads": actual_decoder_heads,
            "decoder_type": decoder_type_name,
            "epochs": args.epochs,
        },
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        precision="16-mixed",
        logger=wandb_logger,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
    )

    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()

    print("done")


if __name__ == "__main__":
    main()
