
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging


class MAEReconstructionViz(Callback):

    def __init__(
        self,
        log_interval: int = 10,
        num_samples: int = 8,
        mask_ratio: float = 0.75,
        fixed_samples_path: Optional[str] = None,
        name: str = "mae_reconstruction",
    ):
        super().__init__()
        self.log_interval = log_interval
        self.num_samples = num_samples
        self.mask_ratio = mask_ratio
        self.fixed_samples_path = fixed_samples_path
        self.name = name
        self._fixed_samples = None
        self._normalization_stats = {
            "mean": np.array([0.485, 0.456, 0.406]),
            "std": np.array([0.229, 0.224, 0.225]),
        }

    def _load_fixed_samples(self):
        if self._fixed_samples is not None:
            return self._fixed_samples

        if self.fixed_samples_path is None:
            return None

        fixed_path = Path(self.fixed_samples_path)
        if not fixed_path.exists():
            logging.warning(
                f"Fixed samples path does not exist: {fixed_path}. "
                "Will use random samples from dataloader."
            )
            return None

        try:
            data = torch.load(fixed_path, map_location="cpu")
            images = data["images"][: self.num_samples]
            logging.info(
                f"Loaded {len(images)} fixed validation samples from {fixed_path}"
            )
            self._fixed_samples = images
            return images
        except Exception as e:
            logging.error(f"Failed to load fixed samples: {e}")
            return None

    def _denormalize_image(self, image):
        mean = self._normalization_stats["mean"][:, None, None]
        std = self._normalization_stats["std"][:, None, None]
        image = image * std + mean
        image = np.clip(image, 0, 1)
        return image

    @torch.no_grad()
    def _create_reconstruction_grid(self, model, images, device):
        model.eval()
        images = images.to(device)

        outputs = model(images, mask_ratio=self.mask_ratio)

        if isinstance(outputs, dict):
            pred = outputs.get("pred", outputs.get("reconstruction"))
            mask = outputs.get("mask")
        elif isinstance(outputs, tuple):
            if len(outputs) >= 3:
                pred = outputs[1]
                mask = outputs[2]
            else:
                logging.error("Unexpected output format from MAE model")
                return None
        else:
            logging.error("Unexpected output type from MAE model")
            return None

        pred = pred.detach().cpu()
        mask = mask.detach().cpu()
        images_cpu = images.cpu()

        mask_patches = mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])

        im_original = images_cpu

        im_patches = model.patchify(images_cpu)
        im_masked_gt = im_patches * (1 - mask_patches)
        im_masked_gt = model.unpatchify(im_masked_gt)

        im_partial = im_patches * (1 - mask_patches) + pred * mask_patches
        im_partial = model.unpatchify(im_partial)

        im_full_recon = model.unpatchify(pred)

        try:
            import matplotlib.pyplot as plt

            num_samples = min(self.num_samples, images.shape[0])
            fig, axes = plt.subplots(
                num_samples, 4, figsize=(16, num_samples * 4)
            )

            if num_samples == 1:
                axes = axes.reshape(1, -1)

            titles = ["Original GT", "Masked GT", "Partial Recon", "Full Recon"]

            for i in range(num_samples):
                for j, (img_tensor, title) in enumerate(
                    zip(
                        [im_original[i], im_masked_gt[i], im_partial[i], im_full_recon[i]],
                        titles,
                    )
                ):
                    ax = axes[i, j]
                    img = self._denormalize_image(img_tensor.numpy())
                    img = np.transpose(img, (1, 2, 0))
                    ax.imshow(img)
                    if i == 0:
                        ax.set_title(title)
                    ax.axis("off")

            plt.tight_layout()

            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )
            plt.close(fig)

            return img_array

        except Exception as e:
            logging.error(f"Failed to create reconstruction grid: {e}")
            return None

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.current_epoch % self.log_interval != 0:
            return

        from lightning.pytorch.loggers import WandbLogger

        if not isinstance(trainer.logger, WandbLogger):
            logging.debug(
                f"WandB logger not found. Skipping {self.name} visualization."
            )
            return

        try:
            import wandb

            fixed_samples = self._load_fixed_samples()

            if fixed_samples is not None:
                images = fixed_samples
            else:
                dataloader = trainer.val_dataloaders
                if dataloader is None:
                    logging.warning("No validation dataloader found")
                    return

                if isinstance(dataloader, list):
                    dataloader = dataloader[0]

                batch = next(iter(dataloader))
                if isinstance(batch, (list, tuple)):
                    images = batch[0][: self.num_samples]
                else:
                    images = batch[: self.num_samples]

            grid = self._create_reconstruction_grid(
                pl_module.backbone, images, pl_module.device
            )

            if grid is None:
                logging.warning("Failed to create reconstruction grid")
                return

            wandb.log(
                {
                    f"{self.name}/reconstructions": wandb.Image(
                        grid,
                        caption=f"Epoch {trainer.current_epoch} - MAE Reconstructions (mask_ratio={self.mask_ratio})",
                    ),
                    f"{self.name}/epoch": trainer.current_epoch,
                }
            )
            logging.info(
                f"Logged {self.name} reconstruction visualization at epoch {trainer.current_epoch}"
            )

        except ImportError:
            logging.debug("WandB not installed, skipping visualization")
        except Exception as e:
            logging.error(f"Failed to log {self.name} reconstruction: {e}")
