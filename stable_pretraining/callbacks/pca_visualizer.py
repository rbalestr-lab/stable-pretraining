"""DINO-style PCA visualisation of ViT patch tokens.

:class:`PCATokenVisualizer` is a read-only evaluation callback in the same
family as :class:`~stable_pretraining.callbacks.OnlineProbe` /
:class:`~stable_pretraining.callbacks.LatentViz`: it acts on keys of the batch
dict. Given a patch-token feature key and a raw-image key it fits a PCA over
all patches, paints the top-3 components as RGB (the classic DINO/DINOv2
"feature PCA" picture), writes the result back into the batch dict under a
unique key, and — periodically — renders a labelled grid figure that is logged
through the trainer's logger (and thereby written to disk).

Nothing here trains: there is no optimizer and no gradient. The callback runs
under ``torch.no_grad`` on whatever device the features already live on.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
from lightning.pytorch import Callback, Trainer
from loguru import logger as logging

from . import _viz_utils as V
from .utils import get_data_from_batch_or_outputs, log_header, resolve_verbose


class PCATokenVisualizer(Callback):
    """Visualise ViT patch tokens as an RGB PCA image, DINO-style.

    The callback fires at the start (``batch_idx == 0``) of each configured
    stage, once every ``every_n_epochs`` epochs. On each firing it:

    1. reads ``features`` (``(B, N, D)`` tokens, or ``(B, D, H, W)`` maps) and
       ``image`` (``(B, C, H, W)``) from the batch/outputs dict,
    2. drops the ``num_prefix_tokens`` prefix tokens (CLS/registers),
    3. fits a joint PCA over every patch and maps the leading components to
       RGB (optionally using the DINOv2 foreground/background split),
    4. writes the upsampled ``(B, 3, H, W)`` RGB tensor to
       ``batch[output_key]`` (``output_key`` defaults to ``name``), and
    5. renders a ``[original | PCA]`` grid for up to ``max_images`` samples and
       logs it under ``{stage}/{name}`` via any media-capable logger.

    Args:
        name: Unique identifier — used as the log tag and the default dict key.
        features: Batch/outputs key holding patch tokens ``(B, N, D)`` or a
            conv feature map ``(B, D, H, W)``.
        image: Batch/outputs key holding the raw images ``(B, C, H, W)``.
        num_prefix_tokens: Prefix tokens (CLS + registers) to strip from
            token inputs. Ignored for ``(B, D, H, W)`` inputs.
        n_components: PCA components mapped to colour (3 → RGB).
        grid_size: Explicit ``(gh, gw)`` patch grid. Inferred from the token
            count / image aspect ratio when ``None``.
        normalize_features: L2-normalise tokens before PCA (recommended).
        per_image: Fit the PCA basis separately per image (``True``) or jointly
            over the whole batch (``False``, default). Joint keeps colours
            comparable across images (same material → same colour); per-image
            gives each sample its own basis and maximal colour contrast.
        foreground_threshold: If set (0–1), reproduce the DINOv2 foreground
            trick — background patches (low first component) are painted with
            ``background_color`` and the object is coloured by a second PCA fit
            on foreground patches only.
        background_color: RGB in ``[0, 1]`` for background patches.
        quantile: Tail fraction clipped when scaling components to colour.
        max_images: Max samples drawn in the figure.
        every_n_epochs: Epoch interval between figures.
        log_on: Stages to fire on — any of ``"train"``, ``"val"``, ``"test"``.
        image_mean, image_std: Per-channel normalisation stats to invert for
            display. When ``None`` images are min-max normalised per sample.
        upsample_mode: Interpolation used to bring the patch grid up to image
            resolution (``"nearest"`` keeps crisp patch boundaries).
        output_key: Dict key for the RGB tensor. Defaults to ``name``.
        cell_size, dpi: Figure geometry.
        verbose: Verbosity; ``None`` derives from the global log level.

    Example:
        >>> import stable_pretraining as spt
        >>> viz = spt.callbacks.PCATokenVisualizer(
        ...     name="pca",
        ...     features="patch_tokens",  # (B, N, D) from a ViT
        ...     image="image",
        ...     num_prefix_tokens=1,  # strip CLS
        ...     foreground_threshold=0.6,  # DINOv2 look
        ... )
    """

    def __init__(
        self,
        name: str,
        features: str,
        image: str = "image",
        num_prefix_tokens: int = 1,
        n_components: int = 3,
        grid_size: Optional[Union[int, Tuple[int, int]]] = None,
        normalize_features: bool = True,
        per_image: bool = False,
        foreground_threshold: Optional[float] = None,
        background_color: Sequence[float] = (0.0, 0.0, 0.0),
        quantile: float = 0.02,
        max_images: int = 8,
        every_n_epochs: int = 1,
        log_on: Sequence[str] = ("val",),
        image_mean: Optional[Sequence[float]] = None,
        image_std: Optional[Sequence[float]] = None,
        upsample_mode: str = "nearest",
        output_key: Optional[str] = None,
        cell_size: float = 2.2,
        dpi: int = 130,
        verbose: bool = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.features = features
        self.image = image
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.n_components = int(n_components)
        self.grid_size = grid_size
        self.normalize_features = normalize_features
        self.per_image = bool(per_image)
        self.foreground_threshold = foreground_threshold
        self.background_color = tuple(background_color)
        self.quantile = float(quantile)
        self.max_images = int(max_images)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.log_on = tuple(log_on)
        self.image_mean = image_mean
        self.image_std = image_std
        self.upsample_mode = upsample_mode
        self.output_key = output_key or name
        self.cell_size = float(cell_size)
        self.dpi = int(dpi)
        self.verbose = resolve_verbose(verbose)
        self._warned_missing = False

        log_header("PCATokenVisualizer")
        logging.info(f"  name: {name}")
        logging.info(f"  features key: {features!r}")
        logging.info(f"  image key: {image!r}")
        logging.info(f"  output key: {self.output_key!r}")
        logging.info(f"  num_prefix_tokens: {self.num_prefix_tokens}")
        logging.info(
            f"  pca basis: {'per-image' if self.per_image else 'per-batch (joint)'}"
        )
        logging.info(
            f"  foreground_threshold: {foreground_threshold}"
            + ("" if foreground_threshold is None else " (DINOv2 fg/bg split)")
        )
        logging.info(f"  every_n_epochs: {self.every_n_epochs}, log_on: {self.log_on}")

    # -- hooks ---------------------------------------------------------------
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if "train" in self.log_on:
            self._maybe_visualize(
                trainer, pl_module, outputs, batch, batch_idx, "train"
            )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if "val" in self.log_on:
            self._maybe_visualize(trainer, pl_module, outputs, batch, batch_idx, "val")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if "test" in self.log_on:
            self._maybe_visualize(trainer, pl_module, outputs, batch, batch_idx, "test")

    # -- core ----------------------------------------------------------------
    def _should_fire(self, trainer: Trainer, batch_idx: int) -> bool:
        if batch_idx != 0:
            return False
        if getattr(trainer, "sanity_checking", False):
            return False
        return (trainer.current_epoch % self.every_n_epochs) == 0

    @torch.no_grad()
    def _maybe_visualize(
        self, trainer, pl_module, outputs, batch, batch_idx, stage: str
    ) -> None:
        if not self._should_fire(trainer, batch_idx):
            return
        try:
            feats, images = get_data_from_batch_or_outputs(
                [self.features, self.image], batch, outputs, caller_name=self.name
            )
        except ValueError:
            if not self._warned_missing:
                logging.warning(
                    f"PCATokenVisualizer[{self.name}]: key {self.features!r} or "
                    f"{self.image!r} not found; skipping visualisation."
                )
                self._warned_missing = True
            return

        rgb = self._compute_rgb(feats, images)  # (B, 3, H, W) in [0, 1]
        # Expose to downstream callbacks (writers, overlays) on every rank.
        target = (
            outputs
            if isinstance(outputs, dict) and self.features in (outputs or {})
            else batch
        )
        target[self.output_key] = rgb

        if trainer.global_rank == 0:
            self._render_and_log(trainer, images, rgb, stage)

    def _compute_rgb(self, feats: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        H, W = images.shape[-2:]
        if feats.dim() == 4:  # (B, D, gh, gw) conv map
            B, D, gh, gw = feats.shape
            tokens = feats.flatten(2).transpose(1, 2)  # (B, gh*gw, D)
        elif feats.dim() == 3:  # (B, N, D) tokens
            tokens = feats
            if self.num_prefix_tokens:
                tokens = tokens[:, self.num_prefix_tokens :]
            B, N, D = tokens.shape
            gh, gw = V.infer_grid_size(N, self.grid_size, image_hw=(H, W))
        else:
            raise ValueError(
                f"PCATokenVisualizer[{self.name}]: features must be (B, N, D) or "
                f"(B, D, H, W); got {tuple(feats.shape)}."
            )
        rgb_grid = V.pca_tokens_to_rgb(
            tokens,
            grid_hw=(gh, gw),
            n_components=self.n_components,
            normalize_features=self.normalize_features,
            foreground_threshold=self.foreground_threshold,
            background_color=self.background_color,
            quantile=self.quantile,
            per_image=self.per_image,
        )  # (B, 3, gh, gw)
        return V.upsample_maps(rgb_grid, size=(H, W), mode=self.upsample_mode)

    def _render_and_log(self, trainer, images, rgb, stage: str) -> None:
        n = min(self.max_images, images.shape[0])
        disp = V.denormalize_images(
            images[:n], mean=self.image_mean, std=self.image_std
        )
        rows = [[disp[i], rgb[i]] for i in range(n)]
        fig = V.render_grid_figure(
            rows,
            col_titles=["input", "token PCA"],
            title=f"{self.name} — patch-token PCA (epoch {trainer.current_epoch})",
            cell_size=self.cell_size,
            dpi=self.dpi,
        )
        tag = f"{stage}/{self.name}"
        V.emit_figure(
            trainer,
            tag=tag,
            image_hwc_uint8=fig,
            step=getattr(trainer, "global_step", None),
            caption=f"epoch {trainer.current_epoch}",
        )
        if self.verbose:
            logging.info(f"  {tag}: logged PCA figure for {n} images")
