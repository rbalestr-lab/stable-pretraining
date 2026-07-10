"""DINO-style CLS self-attention visualisation for ViTs.

:class:`AttentionVisualizer` is the attention twin of
:class:`~stable_pretraining.callbacks.PCATokenVisualizer`. Given a
self-attention tensor and the raw images, it extracts the CLS token's
attention over the patch grid, optionally thresholds it into a crisp object
mask (DINO's cumulative-mass recipe), writes the maps back into the batch dict,
and periodically logs an eye-candy grid figure through the trainer's logger.

Read-only: no optimizer, no gradient, runs under ``torch.no_grad``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
from lightning.pytorch import Callback, Trainer
from loguru import logger as logging

from . import _viz_utils as V
from .utils import get_data_from_batch_or_outputs, log_header, resolve_verbose


class AttentionVisualizer(Callback):
    """Visualise and threshold ViT CLS self-attention, DINO-style.

    Fires at the start (``batch_idx == 0``) of each configured stage, once
    every ``every_n_epochs`` epochs. On each firing it:

    1. reads ``attention`` (``(B, heads, T, T)`` full matrix or ``(B, heads, T)``
       CLS row) and ``image`` (``(B, C, H, W)``) from the batch/outputs dict,
    2. extracts the CLS→patch attention, dropping ``num_prefix_tokens`` prefix
       columns, and reshapes it onto the ``(gh, gw)`` patch grid,
    3. optionally builds a binary object mask by keeping the most-attended
       patches whose cumulative mass reaches ``threshold`` (DINO recipe),
    4. writes the upsampled mean attention map to ``batch[output_key]`` and, if
       thresholding, the mask to ``batch[f"{output_key}_mask"]``, and
    5. renders a grid (input + per-head or mean heat maps + mask overlay) and
       logs it under ``{stage}/{name}``.

    Args:
        name: Unique identifier — log tag and default dict-key prefix.
        attention: Batch/outputs key holding attention weights.
        image: Batch/outputs key holding raw images ``(B, C, H, W)``.
        num_prefix_tokens: Prefix tokens (CLS + registers) in the key/query
            dimension. The CLS row is read at ``cls_index``; these columns are
            stripped before reshaping to the grid.
        cls_index: Row index of the CLS token in a full attention matrix.
        threshold: Cumulative-mass fraction in ``(0, 1)`` for the object mask;
            ``None`` disables the mask (heat maps only). DINO uses ``0.6``.
        head_reduction: ``"mean"`` (average heads), ``"all"`` (one column per
            head), or an ``int`` head index.
        grid_size: Explicit ``(gh, gw)`` patch grid; inferred when ``None``.
        max_images: Max samples drawn in the figure.
        every_n_epochs: Epoch interval between figures.
        log_on: Stages to fire on — any of ``"train"``, ``"val"``, ``"test"``.
        image_mean, image_std: Normalisation stats to invert for display.
        cmap: Matplotlib colormap for the attention heat maps.
        mask_color: RGB overlay colour for the thresholded object mask.
        mask_alpha: Overlay opacity in ``[0, 1]``.
        upsample_mode: Interpolation for heat maps (masks always use nearest).
        output_key: Dict-key prefix. Defaults to ``name``.
        cell_size, dpi: Figure geometry.
        verbose: Verbosity; ``None`` derives from the global log level.

    Example:
        >>> import stable_pretraining as spt
        >>> viz = spt.callbacks.AttentionVisualizer(
        ...     name="cls_attn",
        ...     attention="last_selfattention",  # (B, heads, T, T)
        ...     image="image",
        ...     num_prefix_tokens=1,
        ...     threshold=0.6,  # DINO object mask
        ...     head_reduction="all",  # show every head
        ... )
    """

    def __init__(
        self,
        name: str,
        attention: str,
        image: str = "image",
        num_prefix_tokens: int = 1,
        cls_index: int = 0,
        threshold: Optional[float] = 0.6,
        head_reduction: Union[str, int] = "mean",
        grid_size: Optional[Union[int, Tuple[int, int]]] = None,
        max_images: int = 8,
        every_n_epochs: int = 1,
        log_on: Sequence[str] = ("val",),
        image_mean: Optional[Sequence[float]] = None,
        image_std: Optional[Sequence[float]] = None,
        cmap: str = "inferno",
        mask_color: Sequence[float] = (1.0, 0.15, 0.15),
        mask_alpha: float = 0.45,
        upsample_mode: str = "bilinear",
        output_key: Optional[str] = None,
        cell_size: float = 2.2,
        dpi: int = 130,
        verbose: bool = None,
    ) -> None:
        super().__init__()
        if not (isinstance(head_reduction, int) or head_reduction in ("mean", "all")):
            raise ValueError(
                f"head_reduction must be 'mean', 'all', or an int; got {head_reduction!r}"
            )
        self.name = name
        self.attention = attention
        self.image = image
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.cls_index = int(cls_index)
        self.threshold = threshold
        self.head_reduction = head_reduction
        self.grid_size = grid_size
        self.max_images = int(max_images)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.log_on = tuple(log_on)
        self.image_mean = image_mean
        self.image_std = image_std
        self.cmap = cmap
        self.mask_color = tuple(mask_color)
        self.mask_alpha = float(mask_alpha)
        self.upsample_mode = upsample_mode
        self.output_key = output_key or name
        self.cell_size = float(cell_size)
        self.dpi = int(dpi)
        self.verbose = resolve_verbose(verbose)
        self._warned_missing = False

        log_header("AttentionVisualizer")
        logging.info(f"  name: {name}")
        logging.info(f"  attention key: {attention!r}")
        logging.info(f"  image key: {image!r}")
        logging.info(f"  output key: {self.output_key!r}")
        logging.info(f"  num_prefix_tokens: {self.num_prefix_tokens}")
        logging.info(f"  threshold: {threshold}, head_reduction: {head_reduction}")
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
            attn, images = get_data_from_batch_or_outputs(
                [self.attention, self.image], batch, outputs, caller_name=self.name
            )
        except ValueError:
            if not self._warned_missing:
                logging.warning(
                    f"AttentionVisualizer[{self.name}]: key {self.attention!r} or "
                    f"{self.image!r} not found; skipping visualisation."
                )
                self._warned_missing = True
            return

        H, W = images.shape[-2:]
        gh, gw = self._resolve_grid(attn, H, W)
        maps, mask = V.process_cls_attention(
            attn,
            grid_hw=(gh, gw),
            num_prefix_tokens=self.num_prefix_tokens,
            cls_index=self.cls_index,
            threshold=self.threshold,
        )  # maps: (B, heads, gh, gw); mask: same or None

        mean_map = V.upsample_maps(
            maps.mean(dim=1, keepdim=True), size=(H, W), mode=self.upsample_mode
        )  # (B, 1, H, W)
        target = (
            outputs
            if isinstance(outputs, dict) and self.attention in (outputs or {})
            else batch
        )
        target[self.output_key] = mean_map
        if mask is not None:
            mean_mask = V.upsample_maps(
                mask.mean(dim=1, keepdim=True), size=(H, W), mode="nearest"
            )
            target[f"{self.output_key}_mask"] = (mean_mask > 0.5).float()

        if trainer.global_rank == 0:
            self._render_and_log(trainer, images, maps, mask, (H, W), stage)

    def _resolve_grid(self, attn: torch.Tensor, H: int, W: int) -> Tuple[int, int]:
        # Number of patch tokens = key dim minus prefix.
        T = attn.shape[-1]
        n_patches = T - self.num_prefix_tokens
        try:
            return V.infer_grid_size(n_patches, self.grid_size, image_hw=(H, W))
        except ValueError:
            # Maybe the key dim already excludes prefix tokens.
            return V.infer_grid_size(T, self.grid_size, image_hw=(H, W))

    def _select_head_maps(self, maps: torch.Tensor, i: int):
        """Return list of (label, single-head/mean map) for image ``i``."""
        if self.head_reduction == "mean":
            return [("attention", maps[i].mean(dim=0))]
        if self.head_reduction == "all":
            return [(f"head {h}", maps[i, h]) for h in range(maps.shape[1])]
        h = int(self.head_reduction)
        return [(f"head {h}", maps[i, h])]

    def _render_and_log(self, trainer, images, maps, mask, size, stage: str) -> None:
        H, W = size
        n = min(self.max_images, images.shape[0])
        disp = V.denormalize_images(
            images[:n], mean=self.image_mean, std=self.image_std
        )
        rows = []
        col_titles = ["input"]
        for i in range(n):
            cells = [disp[i]]
            for label, hmap in self._select_head_maps(maps, i):
                up = V.upsample_maps(
                    hmap[None, None], size=(H, W), mode=self.upsample_mode
                )[0, 0]
                cells.append(V.apply_colormap(up, self.cmap))
                if i == 0:
                    col_titles.append(label)
            if mask is not None:
                m = mask[i].mean(dim=0)  # (gh, gw)
                up_m = V.upsample_maps(m[None, None], size=(H, W), mode="nearest")[0, 0]
                cells.append(
                    V.overlay_mask(disp[i], up_m, self.mask_color, self.mask_alpha)
                )
                if i == 0:
                    col_titles.append(f"mask @ {self.threshold}")
            rows.append(cells)

        fig = V.render_grid_figure(
            rows,
            col_titles=col_titles,
            title=f"{self.name} — CLS attention (epoch {trainer.current_epoch})",
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
            logging.info(f"  {tag}: logged attention figure for {n} images")
