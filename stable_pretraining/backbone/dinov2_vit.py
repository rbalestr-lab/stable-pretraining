# Copyright (c) stable-pretraining contributors
# Licensed under the MIT License
#
# This file creates DINOv2-enhanced Vision Transformers using timm as the base.
# Inspired by concepts from DINOv2: https://github.com/facebookresearch/dinov2

"""DINOv2-enhanced Vision Transformers built on timm.

This module provides enhanced ViT models with DINOv2-specific features:
- Mask tokens for iBOT loss
- Enhanced positional encoding interpolation
- Support for sequence packing (when xFormers available)
- Compatible with stable-pretraining's TeacherStudentWrapper
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as logging

# Check for optional dependencies
try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logging.warning("timm not available - from_timm_dinov2 will not work")

try:
    from xformers.ops import fmha  # noqa: F401 - Will be used in forward_nested

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    logging.info(
        "xFormers not available - sequence packing disabled (will use fallback)"
    )


class DINOv2EnhancedViT(nn.Module):
    """Wrapper around timm ViT that adds DINOv2-specific features.

    This class wraps a timm VisionTransformer and adds:
    1. Mask tokens for iBOT masked prediction
    2. Enhanced positional encoding interpolation with offset
    3. Support for multi-view sequence packing (requires xFormers)

    The wrapper preserves timm's API for single-view processing while
    adding a new forward_nested() method for efficient multi-view processing.

    Args:
        timm_vit: Base VisionTransformer from timm
        enable_mask_tokens: Add mask token parameter for iBOT
        interpolate_offset: Offset for position encoding interpolation (DINOv2 trick)
        interpolate_antialias: Use antialiasing when interpolating position embeddings

    Example:
        >>> import timm
        >>> base_vit = timm.create_model("vit_small_patch16_224", num_classes=0)
        >>> enhanced_vit = DINOv2EnhancedViT(
        ...     base_vit, enable_mask_tokens=True, interpolate_offset=0.1
        ... )
        >>>
        >>> # Single view (standard)
        >>> output = enhanced_vit(images)
        >>>
        >>> # Multi-view with sequence packing (requires xFormers)
        >>> if XFORMERS_AVAILABLE:
        ...     output = enhanced_vit.forward_nested([view1, view2, ...])
    """

    def __init__(
        self,
        timm_vit: nn.Module,
        enable_mask_tokens: bool = True,
        interpolate_offset: float = 0.1,
        interpolate_antialias: bool = False,
    ):
        super().__init__()
        self.vit = timm_vit
        self.enable_mask_tokens = enable_mask_tokens
        self.interpolate_offset = interpolate_offset
        self.interpolate_antialias = interpolate_antialias

        # Extract config from timm ViT
        self.embed_dim = self.vit.embed_dim
        self.patch_size = self.vit.patch_embed.patch_size[0]  # Assume square patches
        self.num_patches = self.vit.patch_embed.num_patches

        # Add mask token for iBOT if enabled
        if enable_mask_tokens:
            self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim))
            nn.init.normal_(self.mask_token, std=0.02)
            logging.info("DINOv2EnhancedViT: Added mask token for iBOT")
        else:
            self.mask_token = None

        # Store whether sequence packing is available
        self.sequence_packing_available = XFORMERS_AVAILABLE

        if not XFORMERS_AVAILABLE:
            logging.info(
                "DINOv2EnhancedViT: xFormers not available. "
                "Multi-view processing will use fallback (slower but functional)."
            )

        logging.info(
            f"DINOv2EnhancedViT initialized: "
            f"embed_dim={self.embed_dim}, patch_size={self.patch_size}, "
            f"mask_tokens={enable_mask_tokens}, interpolate_offset={interpolate_offset}"
        )

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """Interpolate positional encodings for arbitrary image sizes.

        This implements DINOv2's enhanced interpolation with an offset trick
        to avoid floating point errors. The offset is added to both dimensions
        before computing the scale factor.

        Args:
            x: Input tensor (after patch embedding) [B, N+1, D]
            w: Image width in pixels
            h: Image height in pixels

        Returns:
            Interpolated positional encoding [1, N_new+1, D]
        """
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1  # Exclude CLS token
        N = self.vit.pos_embed.shape[1] - 1  # Original num patches

        # If size matches, return original
        if npatch == N and w == h:
            return self.vit.pos_embed

        # Separate CLS and patch position embeddings
        pos_embed = self.vit.pos_embed.float()
        class_pos_embed = pos_embed[:, 0:1]  # [1, 1, D]
        patch_pos_embed = pos_embed[:, 1:]  # [1, N, D]

        # Calculate grid dimensions
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Original grid size

        assert N == M * M, f"Position embedding size {N} is not a perfect square"

        # Prepare interpolation kwargs
        kwargs = {}
        if self.interpolate_offset:
            # DINOv2 trick: add small offset to avoid floating point errors
            # See: https://github.com/facebookresearch/dino/issues/8
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)

        # Interpolate: reshape to 2D grid, interpolate, reshape back
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )

        assert (w0, h0) == patch_pos_embed.shape[-2:], (
            f"Interpolation size mismatch: expected {(w0, h0)}, got {patch_pos_embed.shape[-2:]}"
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        # Concatenate CLS and patch embeddings
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(
        self, x: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Prepare input tokens with optional masking for iBOT.

        Args:
            x: Input images [B, C, H, W]
            masks: Boolean mask [B, N] where True = masked patch

        Returns:
            Token sequence with positional encodings [B, N+1, D]
        """
        B, C, H, W = x.shape

        # Patch embedding
        x = self.vit.patch_embed(x)  # [B, N, D]

        # Apply mask tokens if provided
        if masks is not None and self.mask_token is not None:
            # masks: [B, N], mask_token: [1, D]
            # Where mask is True, replace with mask_token
            x = torch.where(
                masks.unsqueeze(-1),  # [B, N, 1]
                self.mask_token.to(x.dtype).unsqueeze(0),  # [1, 1, D]
                x,  # [B, N, D]
            )

        # Add CLS token
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]

        # Add positional encoding (with enhanced interpolation)
        x = x + self.interpolate_pos_encoding(x, W, H)

        # Add register tokens if present in base ViT
        if hasattr(self.vit, "num_reg_tokens") and self.vit.num_reg_tokens > 0:
            x = torch.cat(
                (
                    x[:, :1],  # CLS token
                    self.vit.reg_token.expand(B, -1, -1),  # Register tokens
                    x[:, 1:],  # Patch tokens
                ),
                dim=1,
            )

        return x

    def forward(
        self, x: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Standard forward pass (single view).

        Args:
            x: Input images [B, C, H, W]
            masks: Optional masks for iBOT [B, N]

        Returns:
            Features from ViT [B, N+1, D] or [B, D] depending on head
        """
        # Prepare tokens with optional masking
        x = self.prepare_tokens_with_masks(x, masks)

        # Pass through transformer blocks
        x = self.vit.blocks(x)

        # Apply final norm
        x = self.vit.norm(x)

        return x

    def forward_nested(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Multi-view forward with sequence packing (requires xFormers).

        This method efficiently processes multiple views by packing them into
        a single sequence with block-diagonal attention masking. This is much
        faster than processing views separately.

        Args:
            x_list: List of image tensors, each [B, C, H, W]
                   Can have different spatial sizes.

        Returns:
            List of feature tensors, one per input view

        Note:
            If xFormers is not available, falls back to processing views
            separately (slower but functionally equivalent).
        """
        if not XFORMERS_AVAILABLE:
            # Fallback: process each view separately
            logging.warning(
                "forward_nested called but xFormers not available. "
                "Falling back to separate processing (slower)."
            )
            return [self.forward(x) for x in x_list]

        # TODO: Implement sequence packing with xFormers
        # This will be done in Phase 1C (dinov2_blocks.py)
        # For now, use fallback
        logging.warning(
            "forward_nested: Sequence packing not yet implemented. "
            "Using fallback (will be added in Phase 1C)."
        )
        return [self.forward(x) for x in x_list]


def from_timm_dinov2(
    model_name: str,
    pretrained: bool = False,
    enable_mask_tokens: bool = True,
    interpolate_offset: float = 0.1,
    interpolate_antialias: bool = False,
    **timm_kwargs,
) -> DINOv2EnhancedViT:
    """Create a DINOv2-enhanced ViT using timm as the base.

    This function creates a Vision Transformer from timm and wraps it
    with DINOv2-specific enhancements for efficient multi-crop training.

    Args:
        model_name: timm model name (e.g., 'vit_small_patch16_224')
        pretrained: Load pretrained weights from timm
        enable_mask_tokens: Add mask token for iBOT loss
        interpolate_offset: Offset for position encoding (DINOv2 trick, default 0.1)
        interpolate_antialias: Use antialiasing when interpolating
        **timm_kwargs: Additional arguments passed to timm.create_model

    Returns:
        DINOv2EnhancedViT model

    Example:
        >>> # Basic DINOv2 ViT
        >>> model = from_timm_dinov2("vit_small_patch16_224")
        >>>
        >>> # With pretrained weights
        >>> model = from_timm_dinov2("vit_small_patch16_224", pretrained=True)
        >>>
        >>> # Custom configuration
        >>> model = from_timm_dinov2(
        ...     "vit_base_patch16_224",
        ...     enable_mask_tokens=True,
        ...     interpolate_offset=0.1,
        ...     num_classes=0,  # Remove classification head
        ...     dynamic_img_size=True,  # Enable dynamic sizes
        ... )
    """
    if not TIMM_AVAILABLE:
        raise ImportError(
            "timm is required for from_timm_dinov2. Install with: pip install timm"
        )

    # Set sensible defaults for SSL training
    timm_defaults = {
        "num_classes": 0,  # Remove classification head by default
        "dynamic_img_size": True,  # Enable dynamic resolution
    }
    timm_defaults.update(timm_kwargs)

    # Create base ViT from timm
    logging.info(f"Creating base ViT from timm: {model_name}")
    base_vit = timm.create_model(model_name, pretrained=pretrained, **timm_defaults)

    # Wrap with DINOv2 enhancements
    enhanced_vit = DINOv2EnhancedViT(
        timm_vit=base_vit,
        enable_mask_tokens=enable_mask_tokens,
        interpolate_offset=interpolate_offset,
        interpolate_antialias=interpolate_antialias,
    )

    logging.info(
        f"Created DINOv2-enhanced ViT: {model_name} "
        f"(mask_tokens={enable_mask_tokens}, "
        f"sequence_packing={XFORMERS_AVAILABLE})"
    )

    return enhanced_vit
