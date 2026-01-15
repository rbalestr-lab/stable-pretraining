"""Patch masking strategies for masked image modeling."""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PatchMasking", "MaskingOutput"]


@dataclass
class MaskingOutput:
    """Output from patch masking operation.

    :ivar visible: Visible patch embeddings (B, N_keep, D)
    :ivar mask: Binary mask where 1 = masked, 0 = visible (B, N)
    :ivar ids_restore: Indices to restore original order (B, N)
    :ivar ids_keep: Indices of kept (visible) patches (B, N_keep)
    """

    visible: torch.Tensor
    mask: torch.Tensor
    ids_restore: torch.Tensor
    ids_keep: torch.Tensor


class PatchMasking(nn.Module):
    """Flexible patch masking module for masked image modeling.

    Supports three masking strategies that are selected stochastically:

    - **Random**: Uniformly random patch selection (when block_size=1)
    - **Block**: Square blocks of adjacent patches (when block_size > 1)
    - **Crop**: Rectangular crop region, remaining patches masked (when crop_ratio > 0)

    Strategy selection per sample:

    1. With probability ``crop_ratio``, use crop masking
    2. Otherwise, if ``block_size > 1``, use block masking
    3. Otherwise, use random masking

    :param mask_ratio: Fraction of patches to mask, in [0, 1)
    :param block_size: Size of square blocks for block masking (1 = random masking)
    :param crop_ratio: Probability of using crop masking vs block/random
    :param crop_aspect_ratio: (min, max) aspect ratio range for crop regions

    Example::

        masking = PatchMasking(mask_ratio=0.75, block_size=4)
        output = masking(patch_embeddings, grid_h=14, grid_w=14)

        visible_patches = output.visible  # (B, N_keep, D)
        mask = output.mask  # (B, N), 1=masked, 0=visible
        ids_keep = output.ids_keep  # (B, N_keep)
    """

    def __init__(
        self,
        mask_ratio: float = 0.75,
        block_size: int = 1,
        crop_ratio: float = 0.0,
        crop_aspect_ratio: tuple[float, float] = (0.75, 1.33),
    ):
        super().__init__()

        # Validation
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if not 0.0 <= crop_ratio <= 1.0:
            raise ValueError(f"crop_ratio must be in [0, 1], got {crop_ratio}")
        if len(crop_aspect_ratio) != 2:
            raise ValueError(
                f"crop_aspect_ratio must be a tuple of 2 floats, got {crop_aspect_ratio}"
            )
        if crop_aspect_ratio[0] <= 0 or crop_aspect_ratio[1] <= 0:
            raise ValueError(
                f"crop_aspect_ratio values must be positive, got {crop_aspect_ratio}"
            )
        if crop_aspect_ratio[0] > crop_aspect_ratio[1]:
            raise ValueError(
                f"crop_aspect_ratio[0] must be <= crop_aspect_ratio[1], "
                f"got {crop_aspect_ratio}"
            )

        self.mask_ratio = mask_ratio
        self.block_size = block_size
        self.crop_ratio = crop_ratio
        self.crop_aspect_ratio = crop_aspect_ratio

    def forward(
        self,
        x: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> MaskingOutput:
        """Apply masking to patch embeddings.

        :param x: Patch embeddings of shape (B, N, D) where N = grid_h * grid_w
        :param grid_h: Height of the patch grid
        :param grid_w: Width of the patch grid
        :return: MaskingOutput containing visible patches and mask information
        :raises ValueError: If x.shape[1] != grid_h * grid_w
        :raises ValueError: If input tensor has wrong number of dimensions
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (B, N, D), got {x.dim()}D tensor with shape {x.shape}"
            )

        B, N, D = x.shape

        if N != grid_h * grid_w:
            raise ValueError(
                f"Number of patches {N} doesn't match grid size "
                f"{grid_h} x {grid_w} = {grid_h * grid_w}"
            )

        if self.mask_ratio == 0 or not self.training:
            # No masking - return all patches as visible
            return MaskingOutput(
                visible=x,
                mask=torch.zeros(B, N, device=x.device),
                ids_restore=torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1),
                ids_keep=torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1),
            )

        num_mask = int(N * self.mask_ratio)
        num_keep = N - num_mask
        device = x.device

        # Determine which strategy to use per sample
        use_crop = torch.rand(B, device=device) < self.crop_ratio
        noise = torch.rand(B, N, device=device)

        # Apply crop masking where selected
        if use_crop.any():
            crop_noise = self._generate_crop_noise(B, grid_h, grid_w, num_keep, device)
            noise = torch.where(use_crop.view(B, 1), crop_noise, noise)

        # Apply block masking where selected (and not using crop)
        if self.block_size > 1 and (~use_crop).any():
            block_noise = self._generate_block_noise(
                B, grid_h, grid_w, num_mask, device
            )
            noise = torch.where((~use_crop).view(B, 1), block_noise, noise)

        # Convert noise to indices via sorting (lower noise = keep)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]

        # Gather visible patches
        visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Create binary mask (1 = masked, 0 = visible)
        mask = torch.ones(B, N, device=device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return MaskingOutput(
            visible=visible,
            mask=mask,
            ids_restore=ids_restore,
            ids_keep=ids_keep,
        )

    def _generate_block_noise(
        self, B: int, grid_h: int, grid_w: int, num_mask: int, device: torch.device
    ) -> torch.Tensor:
        """Generate noise that induces block-structured masking."""
        N = grid_h * grid_w
        mask = torch.zeros(B, grid_h, grid_w, device=device)
        half = self.block_size // 2
        patches_per_block = self.block_size * self.block_size
        num_blocks_needed = (num_mask // patches_per_block) + 5

        centers_y = torch.randint(0, grid_h, (B, num_blocks_needed), device=device)
        centers_x = torch.randint(0, grid_w, (B, num_blocks_needed), device=device)

        rows = torch.arange(grid_h, device=device).view(1, 1, grid_h, 1)
        cols = torch.arange(grid_w, device=device).view(1, 1, 1, grid_w)

        for i in range(num_blocks_needed):
            cy = centers_y[:, i].view(B, 1, 1)
            cx = centers_x[:, i].view(B, 1, 1)

            y_start = (cy - half).clamp(min=0)
            y_end = (cy - half + self.block_size).clamp(max=grid_h)
            x_start = (cx - half).clamp(min=0)
            x_end = (cx - half + self.block_size).clamp(max=grid_w)

            in_block = (
                (rows >= y_start.unsqueeze(-1))
                & (rows < y_end.unsqueeze(-1))
                & (cols >= x_start.unsqueeze(-1))
                & (cols < x_end.unsqueeze(-1))
            ).squeeze(1)
            mask = torch.maximum(mask, in_block.float())

            if (mask.view(B, -1).sum(dim=1) >= num_mask).all():
                break

        mask_flat = self._adjust_mask_count(mask.view(B, N), num_mask, device)
        return torch.rand(B, N, device=device) * 0.5 + mask_flat * 0.5

    def _generate_crop_noise(
        self, B: int, grid_h: int, grid_w: int, num_keep: int, device: torch.device
    ) -> torch.Tensor:
        """Generate noise that induces crop-style masking."""
        N = grid_h * grid_w
        target_area = float(num_keep)

        log_ratio_min = math.log(self.crop_aspect_ratio[0])
        log_ratio_max = math.log(self.crop_aspect_ratio[1])
        log_ratios = torch.empty(B, device=device).uniform_(
            log_ratio_min, log_ratio_max
        )
        aspect_ratios = log_ratios.exp()

        crop_h = (target_area / aspect_ratios).sqrt().round().clamp(1, grid_h).long()
        crop_w = (target_area * aspect_ratios).sqrt().round().clamp(1, grid_w).long()

        max_y = (grid_h - crop_h).clamp(min=0)
        max_x = (grid_w - crop_w).clamp(min=0)
        top = (
            (torch.rand(B, device=device) * (max_y.float() + 1)).long().clamp(max=max_y)
        )
        left = (
            (torch.rand(B, device=device) * (max_x.float() + 1)).long().clamp(max=max_x)
        )

        rows = torch.arange(grid_h, device=device).view(1, grid_h, 1)
        cols = torch.arange(grid_w, device=device).view(1, 1, grid_w)

        in_crop = (
            (rows >= top.view(B, 1, 1))
            & (rows < (top + crop_h).view(B, 1, 1))
            & (cols >= left.view(B, 1, 1))
            & (cols < (left + crop_w).view(B, 1, 1))
        )
        crop_mask = (~in_crop).float().view(B, N)
        crop_mask = self._adjust_crop_to_target(
            crop_mask, num_keep, grid_h, grid_w, device
        )

        return torch.rand(B, N, device=device) * 0.5 + crop_mask * 0.5

    def _adjust_mask_count(
        self, mask_flat: torch.Tensor, target_masked: int, device: torch.device
    ) -> torch.Tensor:
        """Adjust mask to have exactly target_masked patches masked per sample."""
        B, N = mask_flat.shape
        mask_flat = mask_flat.clone()
        current_masked = mask_flat.sum(dim=1)

        excess = (current_masked - target_masked).clamp(min=0).long()
        if excess.any():
            noise = torch.rand(B, N, device=device) + (1 - mask_flat) * 2
            sorted_idx = noise.argsort(dim=1)
            position_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            unmask_positions = position_idx < excess.unsqueeze(1)
            unmask_idx = torch.gather(sorted_idx, 1, position_idx)
            mask_flat.scatter_(
                1,
                unmask_idx,
                mask_flat.gather(1, unmask_idx) * (~unmask_positions).float(),
            )

        current_masked = mask_flat.sum(dim=1)
        deficit = (target_masked - current_masked).clamp(min=0).long()
        if deficit.any():
            noise = torch.rand(B, N, device=device) + mask_flat * 2
            sorted_idx = noise.argsort(dim=1)
            position_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            mask_positions = position_idx < deficit.unsqueeze(1)
            mask_idx = torch.gather(sorted_idx, 1, position_idx)
            mask_flat.scatter_(
                1, mask_idx, mask_flat.gather(1, mask_idx) + mask_positions.float()
            )

        return mask_flat.clamp(0, 1)

    def _adjust_crop_to_target(
        self,
        crop_mask: torch.Tensor,
        num_keep: int,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Adjust crop mask using morphological operations to hit target visible count."""
        B, N = crop_mask.shape
        crop_mask = crop_mask.clone()

        max_iterations = 20
        for _ in range(max_iterations):
            num_visible = (crop_mask == 0).sum(dim=1)
            diff = num_visible - num_keep

            if (diff == 0).all():
                break

            mask_2d = crop_mask.view(B, 1, grid_h, grid_w)

            need_erode = diff > 0
            if need_erode.any():
                visible = (mask_2d == 0).float()
                padded = F.pad(1 - visible, (1, 1, 1, 1), value=1)
                neighbor_masked = F.max_pool2d(padded, 3, stride=1, padding=0)
                boundary = (visible.squeeze(1) == 1) & (neighbor_masked.squeeze(1) > 0)

                boundary_noise = (
                    torch.rand(B, grid_h, grid_w, device=device) * boundary.float()
                )
                boundary_noise[~need_erode] = -1

                flat_noise = boundary_noise.view(B, N)
                boundary_count = boundary.view(B, -1).sum(dim=1)
                to_remove = torch.minimum(diff.clamp(min=0), boundary_count)
                max_k = int(to_remove.max().item())

                if max_k > 0:
                    _, top_idx = flat_noise.topk(max_k, dim=1)
                    position_idx = torch.arange(max_k, device=device).unsqueeze(0)
                    valid = position_idx < to_remove.unsqueeze(1)
                    crop_mask.scatter_(
                        1, top_idx, crop_mask.gather(1, top_idx) + valid.float()
                    )

            need_dilate = diff < 0
            if need_dilate.any():
                mask_2d = crop_mask.view(B, 1, grid_h, grid_w)
                visible = (mask_2d == 0).float()
                padded = F.pad(visible, (1, 1, 1, 1), value=0)
                neighbor_visible = F.max_pool2d(padded, 3, stride=1, padding=0)
                boundary = (mask_2d.squeeze(1) == 1) & (neighbor_visible.squeeze(1) > 0)

                boundary_noise = (
                    torch.rand(B, grid_h, grid_w, device=device) * boundary.float()
                )
                boundary_noise[~need_dilate] = -1

                flat_noise = boundary_noise.view(B, N)
                boundary_count = boundary.view(B, -1).sum(dim=1)
                to_add = torch.minimum((-diff).clamp(min=0), boundary_count)
                max_k = int(to_add.max().item())

                if max_k > 0:
                    _, top_idx = flat_noise.topk(max_k, dim=1)
                    position_idx = torch.arange(max_k, device=device).unsqueeze(0)
                    valid = position_idx < to_add.unsqueeze(1)
                    crop_mask.scatter_(
                        1, top_idx, crop_mask.gather(1, top_idx) * (~valid).float()
                    )

        return crop_mask.clamp(0, 1)

    def extra_repr(self) -> str:
        return (
            f"mask_ratio={self.mask_ratio}, block_size={self.block_size}, "
            f"crop_ratio={self.crop_ratio}, crop_aspect_ratio={self.crop_aspect_ratio}"
        )
