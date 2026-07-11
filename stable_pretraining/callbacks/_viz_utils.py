"""Shared helpers for the DINO-style visualisation callbacks.

Everything here is deliberately framework-light: pure ``torch`` for the maths
(so it runs on whatever device the features already live on) and a lazy
``matplotlib`` import for the eye-candy figures (so importing the callbacks
package never drags matplotlib in).

The two public callbacks — :class:`~stable_pretraining.callbacks.PCATokenVisualizer`
and :class:`~stable_pretraining.callbacks.AttentionVisualizer` — both:

1. read a feature/attention key and a raw-image key from the batch dict,
2. turn them into an ``(B, 3, H, W)`` RGB visualisation tensor,
3. write that tensor back into the batch dict under a unique key, and
4. optionally render a labelled grid figure and log it through the trainer's
   logger (which, for ``RegistryLogger``/``WandbLogger``, also writes it to
   disk).

The functions below are the reusable building blocks for steps 2–4.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from loguru import logger as logging


# --------------------------------------------------------------------------- #
# image (de)normalisation
# --------------------------------------------------------------------------- #
def denormalize_images(
    images: torch.Tensor,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Bring a batch of images into a displayable ``[0, 1]`` range.

    Args:
        images: ``(B, C, H, W)`` float tensor.
        mean: Per-channel mean used at normalisation time. If given together
            with ``std`` the operation ``x * std + mean`` is inverted.
        std: Per-channel std used at normalisation time.

    Returns:
        ``(B, C, H, W)`` float tensor clamped to ``[0, 1]``. When ``mean``/``std``
        are ``None`` the batch is min-max normalised per image so that whatever
        range it came in (e.g. already-normalised tensors) still displays
        sensibly.
    """
    images = images.detach().float()
    if mean is not None and std is not None:
        m = torch.as_tensor(mean, device=images.device).view(1, -1, 1, 1)
        s = torch.as_tensor(std, device=images.device).view(1, -1, 1, 1)
        images = images * s + m
        return images.clamp_(0.0, 1.0)

    # No stats: robust per-image min-max so display never blows out.
    flat = images.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return ((images - lo) / (hi - lo).clamp_min(1e-6)).clamp_(0.0, 1.0)


# --------------------------------------------------------------------------- #
# grid-size inference
# --------------------------------------------------------------------------- #
def infer_grid_size(
    num_tokens: int,
    grid_size: Optional[Union[int, Tuple[int, int]]] = None,
    image_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    """Resolve the ``(gh, gw)`` patch grid for ``num_tokens`` tokens.

    Priority: explicit ``grid_size`` → aspect-ratio match against ``image_hw``
    → assume a square grid. Raises if none of these produce a grid whose
    product equals ``num_tokens``.
    """
    if grid_size is not None:
        if isinstance(grid_size, int):
            gh = gw = grid_size
        else:
            gh, gw = grid_size
        if gh * gw != num_tokens:
            raise ValueError(
                f"grid_size={(gh, gw)} has {gh * gw} cells but got {num_tokens} tokens."
            )
        return int(gh), int(gw)

    # Try to honour the image aspect ratio for non-square inputs.
    if image_hw is not None:
        H, W = image_hw
        if H > 0 and W > 0:
            ratio = H / W
            gw = round((num_tokens / ratio) ** 0.5)
            for cand_w in {gw, gw - 1, gw + 1}:
                if cand_w > 0 and num_tokens % cand_w == 0:
                    gh = num_tokens // cand_w
                    return int(gh), int(cand_w)

    root = int(round(num_tokens**0.5))
    if root * root == num_tokens:
        return root, root
    raise ValueError(
        f"Cannot infer a patch grid for {num_tokens} tokens; pass grid_size=(gh, gw)."
    )


# --------------------------------------------------------------------------- #
# robust normalisation (quantile clip) for turning components into colours
# --------------------------------------------------------------------------- #
def robust_minmax(
    x: torch.Tensor, quantile: float = 0.02, dim: int = 0
) -> torch.Tensor:
    """Scale ``x`` to ``[0, 1]`` clipping the ``quantile`` tails.

    Clipping the extreme quantiles before scaling keeps a couple of outlier
    patches from washing out the whole colour map — this is what gives the
    DINO PCA plots their punchy, saturated look.
    """
    if quantile and quantile > 0:
        lo = torch.quantile(x, quantile, dim=dim, keepdim=True)
        hi = torch.quantile(x, 1.0 - quantile, dim=dim, keepdim=True)
    else:
        lo = x.min(dim=dim, keepdim=True).values
        hi = x.max(dim=dim, keepdim=True).values
    return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp_(0.0, 1.0)


# --------------------------------------------------------------------------- #
# PCA → RGB  (DINO / DINOv2 style)
# --------------------------------------------------------------------------- #
def _pca_project(feats: torch.Tensor, k: int) -> torch.Tensor:
    """Project ``feats`` (M, D) onto its top-``k`` PCA components → (M, k)."""
    centered = feats - feats.mean(dim=0, keepdim=True)
    # pca_lowrank is fast and GPU-friendly; ask for a couple extra comps for a
    # more stable subspace, then keep the first ``k``.
    q = min(feats.shape[1], max(k + 2, k))
    _, _, v = torch.pca_lowrank(centered, q=q, center=False)
    return centered @ v[:, :k]


def _color_token_set(
    x: torch.Tensor,
    foreground_threshold: Optional[float],
    background_color: Sequence[float],
    quantile: float,
) -> torch.Tensor:
    """Colour one set of tokens ``(M, D)`` → ``(M, 3)`` RGB in ``[0, 1]``.

    Fits a PCA over exactly these tokens; applies the DINOv2 foreground trick
    when ``foreground_threshold`` is set. The caller decides what ``x`` spans
    (all batch patches → per-batch basis, or one image → per-image basis).
    """
    comps = _pca_project(x, 3)  # (M, 3)
    if foreground_threshold is None:
        return robust_minmax(comps, quantile=quantile, dim=0)

    first = robust_minmax(comps[:, :1], quantile=quantile, dim=0).squeeze(1)
    fg = first > foreground_threshold
    if int(fg.sum()) < 4:
        # Degenerate threshold (almost nothing kept): fall back to plain PCA so
        # we still return a sensible image instead of a blank canvas.
        logging.warning(
            "pca_tokens_to_rgb: foreground_threshold kept too few patches "
            f"({int(fg.sum())}); falling back to unthresholded PCA."
        )
        return robust_minmax(comps, quantile=quantile, dim=0)
    fg_rgb = robust_minmax(_pca_project(x[fg], 3), quantile=quantile, dim=0)
    rgb = comps.new_zeros(x.shape[0], 3)
    rgb[~fg] = torch.as_tensor(background_color, device=rgb.device, dtype=rgb.dtype)
    rgb[fg] = fg_rgb
    return rgb


@torch.no_grad()
def pca_tokens_to_rgb(
    tokens: torch.Tensor,
    grid_hw: Tuple[int, int],
    n_components: int = 3,
    normalize_features: bool = True,
    foreground_threshold: Optional[float] = None,
    background_color: Sequence[float] = (0.0, 0.0, 0.0),
    quantile: float = 0.02,
    per_image: bool = False,
) -> torch.Tensor:
    """Map ViT patch tokens to an RGB image via PCA, DINO-style.

    The top three PCA components become the R, G, B channels. The PCA basis is
    fit either jointly across the whole batch (``per_image=False``, the default
    — colours are then comparable across images: the same material lands on the
    same colour) or independently per image (``per_image=True`` — each image
    uses its own basis and stretches its own variance, but colours are not
    comparable between images).

    Optionally reproduces the DINOv2 foreground trick: the first component
    usually separates object from background, so patches whose (normalised)
    first component falls below ``foreground_threshold`` are painted with
    ``background_color`` and a *second* PCA is fit on the surviving
    foreground patches to colour the object itself. With ``per_image=True`` the
    split is computed per image (each image gets its own foreground).

    Args:
        tokens: ``(B, N, D)`` patch tokens (prefix tokens already removed).
        grid_hw: ``(gh, gw)`` such that ``gh * gw == N``.
        n_components: Kept for API stability; the RGB image always uses the
            first three components.
        normalize_features: L2-normalise tokens before PCA (recommended;
            matches the DINOv2 demo and stabilises the basis).
        foreground_threshold: If set (0–1), separate foreground/background on
            the normalised first component before colouring.
        background_color: RGB (each in ``[0, 1]``) painted onto background
            patches. Only used when ``foreground_threshold`` is set.
        quantile: Tail fraction clipped by :func:`robust_minmax` per channel.
        per_image: Fit a separate PCA per image instead of one over the batch.

    Returns:
        ``(B, 3, gh, gw)`` float tensor in ``[0, 1]``.
    """
    B, N, D = tokens.shape
    gh, gw = grid_hw
    if gh * gw != N:
        raise ValueError(f"grid {grid_hw} has {gh * gw} cells but {N} tokens given.")

    x = tokens.float()
    if normalize_features:
        x = F.normalize(x, dim=-1)

    if per_image:
        rgb = torch.stack(
            [
                _color_token_set(x[b], foreground_threshold, background_color, quantile)
                for b in range(B)
            ],
            dim=0,
        )  # (B, N, 3)
    else:
        rgb = _color_token_set(
            x.reshape(B * N, D), foreground_threshold, background_color, quantile
        ).reshape(B, N, 3)

    return rgb.reshape(B, gh, gw, 3).permute(0, 3, 1, 2).contiguous()


# --------------------------------------------------------------------------- #
# CLS attention → maps + threshold mask  (DINO style)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def process_cls_attention(
    attention: torch.Tensor,
    grid_hw: Tuple[int, int],
    num_prefix_tokens: int = 1,
    cls_index: int = 0,
    threshold: Optional[float] = 0.6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Turn raw attention into per-head CLS→patch maps (+ optional mask).

    Accepts either a full attention matrix ``(B, heads, T, T)`` (the CLS row is
    extracted) or an already-reduced ``(B, heads, T)`` CLS row. Prefix-token
    columns (CLS + registers) are dropped so only patch attention remains.

    The threshold reproduces DINO's ``visualize_attention``: per head, keep the
    smallest set of patches whose cumulative attention mass reaches
    ``threshold`` (i.e. the most-attended ``threshold`` fraction of the mass),
    yielding a crisp binary object mask.

    Args:
        attention: ``(B, heads, T, T)`` or ``(B, heads, T)``.
        grid_hw: ``(gh, gw)`` patch grid.
        num_prefix_tokens: Number of leading prefix tokens to strip.
        cls_index: Row index of the CLS token in the full matrix.
        threshold: Cumulative-mass fraction in ``(0, 1)``. ``None`` disables the
            mask (only heat maps returned).

    Returns:
        ``(maps, mask)`` where ``maps`` is ``(B, heads, gh, gw)`` in ``[0, 1]``
        (per-head min-max) and ``mask`` is ``(B, heads, gh, gw)`` binary, or
        ``None`` when ``threshold`` is ``None``.
    """
    if attention.dim() == 4:
        cls = attention[:, :, cls_index, :]  # (B, heads, T)
    elif attention.dim() == 3:
        cls = attention
    else:
        raise ValueError(
            f"attention must be (B, heads, T, T) or (B, heads, T); got shape "
            f"{tuple(attention.shape)}."
        )
    cls = cls.detach().float()
    B, H, T = cls.shape
    gh, gw = grid_hw
    n_patches = gh * gw

    if T == n_patches + num_prefix_tokens:
        cls = cls[:, :, num_prefix_tokens:]
    elif T == n_patches:
        pass
    else:
        raise ValueError(
            f"attention has {T} key tokens but grid {grid_hw} implies "
            f"{n_patches} patches (+{num_prefix_tokens} prefix); pass grid_size."
        )

    mask = None
    if threshold is not None:
        # DINO recipe: normalise to a distribution, sort, keep the top-mass set.
        probs = cls / cls.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        val, idx = probs.sort(dim=-1, descending=True)
        cum = val.cumsum(dim=-1)
        keep = cum <= threshold
        keep[..., 0] = True  # always keep the single most-attended patch
        flat_mask = torch.zeros_like(probs)
        flat_mask.scatter_(-1, idx, keep.to(probs.dtype))
        mask = flat_mask.reshape(B, H, gh, gw)

    # Per-head min-max for a clean heat map.
    lo = cls.amin(dim=-1, keepdim=True)
    hi = cls.amax(dim=-1, keepdim=True)
    maps = ((cls - lo) / (hi - lo).clamp_min(1e-8)).reshape(B, H, gh, gw)
    return maps, mask


def upsample_maps(maps: torch.Tensor, size: Tuple[int, int], mode: str) -> torch.Tensor:
    """Upsample ``(B, K, gh, gw)`` maps to image resolution ``size=(H, W)``."""
    align = False if mode in ("bilinear", "bicubic") else None
    return F.interpolate(maps, size=size, mode=mode, align_corners=align)


# --------------------------------------------------------------------------- #
# colour maps (lazy matplotlib)
# --------------------------------------------------------------------------- #
def apply_colormap(x: torch.Tensor, cmap: str = "inferno") -> torch.Tensor:
    """Map a ``(..., H, W)`` scalar field in ``[0, 1]`` to ``(..., 3, H, W)`` RGB."""
    import matplotlib

    x = x.detach().float().clamp(0.0, 1.0).cpu()
    lut = matplotlib.colormaps[cmap](x.numpy())[..., :3]  # (..., H, W, 3)
    out = torch.from_numpy(lut).to(torch.float32)
    return out.movedim(-1, -3).contiguous()


def overlay_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    color: Sequence[float] = (1.0, 0.1, 0.1),
    alpha: float = 0.45,
) -> torch.Tensor:
    """Blend a binary ``(H, W)`` mask over an ``(3, H, W)`` image in ``[0, 1]``."""
    color_t = torch.as_tensor(color, device=image.device, dtype=image.dtype).view(
        3, 1, 1
    )
    m = mask.unsqueeze(0).clamp(0.0, 1.0)
    return (image * (1 - alpha * m) + color_t * (alpha * m)).clamp_(0.0, 1.0)


# --------------------------------------------------------------------------- #
# eye-candy figure rendering
# --------------------------------------------------------------------------- #
def render_grid_figure(
    cells: List[List[torch.Tensor]],
    col_titles: Optional[Sequence[str]] = None,
    row_titles: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    cell_size: float = 2.2,
    dpi: int = 130,
    facecolor: str = "#111114",
    text_color: str = "#f5f5f7",
) -> "torch.Tensor":
    """Render a grid of ``(3, H, W)`` image tensors into one RGB uint8 array.

    Args:
        cells: ``cells[r][c]`` is a ``(3, H, W)`` float tensor in ``[0, 1]``.
        col_titles: Column headers (drawn above the first row).
        row_titles: Per-row labels (drawn left of the first column).
        title: Optional figure supertitle.
        cell_size: Inches per cell (both axes).
        dpi: Render resolution.
        facecolor: Figure/background colour.
        text_color: Title/label colour.

    Returns:
        ``(H, W, 3)`` uint8 numpy array (via torch) suitable for
        ``logger.log_image`` and PIL.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    nrows = len(cells)
    ncols = max(len(r) for r in cells) if nrows else 0
    fig = plt.figure(
        figsize=(ncols * cell_size, nrows * cell_size + (0.4 if title else 0.0)),
        dpi=dpi,
        facecolor=facecolor,
    )
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            ax.set_facecolor(facecolor)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c < len(cells[r]) and cells[r][c] is not None:
                img = cells[r][c].detach().float().clamp(0, 1).cpu()
                ax.imshow(img.permute(1, 2, 0).numpy(), interpolation="nearest")
            else:
                ax.set_visible(False)
            if r == 0 and col_titles is not None and c < len(col_titles):
                ax.set_title(col_titles[c], color=text_color, fontsize=11, pad=6)
            if c == 0 and row_titles is not None and r < len(row_titles):
                ax.set_ylabel(
                    row_titles[r],
                    color=text_color,
                    fontsize=10,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=12,
                )
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
    if title:
        fig.suptitle(title, color=text_color, fontsize=13)
    fig.subplots_adjust(
        wspace=0.04,
        hspace=0.04,
        left=0.02,
        right=0.99,
        top=0.93 if title else 0.97,
        bottom=0.02,
    )
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = torch.frombuffer(bytes(canvas.buffer_rgba()), dtype=torch.uint8)
    arr = buf.reshape(h, w, 4)[..., :3].clone()
    plt.close(fig)
    return arr


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def emit_figure(
    trainer,
    tag: str,
    image_hwc_uint8: "torch.Tensor",
    step: Optional[int] = None,
    caption: Optional[str] = None,
) -> bool:
    """Send a rendered figure through every media-capable logger.

    Routes the array through any ``trainer.logger(s)`` exposing ``log_image``
    and lets the logger own persistence: ``RegistryLogger`` writes it under
    ``media/`` on disk (and indexes it in ``media.jsonl``); ``WandbLogger``
    uploads it. The callback performs no manual file I/O of its own — if no
    media logger is attached, the figure is simply not persisted (a warning is
    emitted so the misconfiguration is visible).

    Returns:
        ``True`` if at least one logger accepted the image.
    """
    arr = image_hwc_uint8
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()

    logged = False
    for lgr in list(getattr(trainer, "loggers", []) or []):
        fn = getattr(lgr, "log_image", None)
        if fn is None:
            continue
        try:
            fn(key=tag, images=[arr], step=step, caption=[caption] if caption else None)
            logged = True
        except Exception as e:  # never take training down over a figure
            logging.warning(
                f"emit_figure: logger {type(lgr).__name__} rejected image: {e}"
            )

    if not logged:
        logging.warning(
            f"emit_figure[{tag}]: no logger exposes log_image; figure not "
            "persisted. Attach a media-capable logger (RegistryLogger/WandB)."
        )
    return logged
