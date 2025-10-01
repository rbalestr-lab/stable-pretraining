# Copyright (c) stable-pretraining contributors
# Licensed under the MIT License
#
# This file creates DINO-enhanced Vision Transformers using timm as the base.
# Inspired by concepts from DINO/DINOv2: https://github.com/facebookresearch/dinov2

"""DINO-enhanced Vision Transformers built on timm.

This module provides enhanced ViT models with DINO-specific features:
- Mask tokens for iBOT loss (DINOv2)
- Enhanced positional encoding interpolation (DINOv2)
- Sequence packing for efficient multi-crop training (DINOv2 trick, works with v1 losses)
- Compatible with stable-pretraining's TeacherStudentWrapper

These enhancements work with both DINOv1 and DINOv2 training objectives.
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
    logging.warning("timm not available - from_timm_dino will not work")

# Import our DINO components
from .attention import MemEffAttention
from .packing import get_attn_bias_and_cat, unpack_sequences


class DINOEnhancedViT(nn.Module):
    """Wrapper around timm ViT that adds DINO-specific features for efficient multi-crop training.

    This class wraps a timm VisionTransformer and adds DINO features:
    1. Mask tokens for iBOT masked prediction (DINOv2)
    2. Enhanced positional encoding interpolation (DINOv2)
    3. Sequence packing for efficient multi-crop training (DINOv2 trick)

    The wrapper preserves timm's API for single-view processing while
    adding forward_nested() for efficient multi-view processing with sequence packing.

    Works with both DINOv1 and DINOv2 training objectives.

    Args:
        timm_vit: Base VisionTransformer from timm
        enable_mask_tokens: Add mask token parameter for iBOT loss (DINOv2 only)
        interpolate_offset: Offset for position encoding interpolation (DINOv2 trick)
        interpolate_antialias: Use antialiasing when interpolating position embeddings

    Example:
        >>> import timm
        >>> base_vit = timm.create_model("vit_small_patch16_224", num_classes=0)
        >>> enhanced_vit = DINOEnhancedViT(
        ...     base_vit, enable_mask_tokens=True, interpolate_offset=0.1
        ... )
        >>>
        >>> # Single view (standard)
        >>> output = enhanced_vit(images)
        >>>
        >>> # Multi-view with sequence packing (faster with xFormers!)
        >>> output = enhanced_vit.forward_nested([view1, view2, ...])
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
            logging.info("DINOEnhancedViT: Added mask token for iBOT")
        else:
            self.mask_token = None

        # Create MemEffAttention for sequence packing
        # (has built-in fallback to PyTorch SDPA when xFormers unavailable)
        first_block = self.vit.blocks[0]
        self.mem_eff_attn = MemEffAttention(
            dim=self.embed_dim,
            num_heads=first_block.attn.num_heads,
            qkv_bias=first_block.attn.qkv.bias is not None,
            proj_bias=first_block.attn.proj.bias is not None,
        )
        logging.info("DINOEnhancedViT: Created MemEffAttention for sequence packing")

        logging.info(
            f"DINOEnhancedViT initialized: "
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
        # Step 1: Prepare tokens for each view
        tokens_list = [self.prepare_tokens_with_masks(x) for x in x_list]

        # Step 2: Pack sequences and create block-diagonal mask
        # Note: get_attn_bias_and_cat will raise ImportError if xFormers unavailable
        try:
            attn_bias, x_packed = get_attn_bias_and_cat(tokens_list)
        except ImportError:
            # Fallback: process each view separately
            logging.warning(
                "⚡ xFormers not available - falling back to separate forward passes. "
                "Install xFormers for faster multi-crop training: pip install xformers"
            )
            return [self.forward(x) for x in x_list]

        # Extract batch size and sequence lengths for unpacking
        batch_size = tokens_list[0].shape[0]
        sequence_lengths = [t.shape[1] for t in tokens_list]

        # Step 3: Process through transformer blocks with sequence packing
        for block in self.vit.blocks:
            # Attention path with our MemEffAttention + attn_bias
            attn_out = self.mem_eff_attn(block.norm1(x_packed), attn_bias=attn_bias)
            x_packed = x_packed + block.drop_path1(block.ls1(attn_out))

            # MLP path (standard, no attention)
            mlp_out = block.mlp(block.norm2(x_packed))
            x_packed = x_packed + block.drop_path2(block.ls2(mlp_out))

        # Step 4: Apply final norm
        x_packed = self.vit.norm(x_packed)

        # Step 5: Unpack back to list of tensors
        outputs = unpack_sequences(x_packed, sequence_lengths, batch_size)

        return outputs


def from_timm_dino(
    model_name: str,
    pretrained: bool = False,
    enable_mask_tokens: bool = True,
    interpolate_offset: float = 0.1,
    interpolate_antialias: bool = False,
    **timm_kwargs,
) -> DINOEnhancedViT:
    """Create a DINO-enhanced ViT using timm as the base.

    This function creates a Vision Transformer from timm and wraps it
    with DINO-specific enhancements for efficient multi-crop training.

    Works with both DINOv1 and DINOv2 training objectives.

    Args:
        model_name: timm model name (e.g., 'vit_small_patch16_224')
        pretrained: Load pretrained weights from timm
        enable_mask_tokens: Add mask token for iBOT loss (DINOv2 only, disable for v1)
        interpolate_offset: Offset for position encoding (DINOv2 trick, default 0.1)
        interpolate_antialias: Use antialiasing when interpolating
        **timm_kwargs: Additional arguments passed to timm.create_model

    Returns:
        DINOEnhancedViT model

    Example:
        >>> # Basic DINO ViT with sequence packing
        >>> model = from_timm_dino("vit_small_patch16_224")
        >>>
        >>> # With pretrained weights
        >>> model = from_timm_dino("vit_small_patch16_224", pretrained=True)
        >>>
        >>> # Custom configuration
        >>> model = from_timm_dino(
        ...     "vit_base_patch16_224",
        ...     enable_mask_tokens=True,
        ...     interpolate_offset=0.1,
        ...     num_classes=0,  # Remove classification head
        ...     dynamic_img_size=True,  # Enable dynamic sizes
        ... )
    """
    if not TIMM_AVAILABLE:
        raise ImportError(
            "timm is required for from_timm_dino. Install with: pip install timm"
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

    # Wrap with DINO enhancements
    enhanced_vit = DINOEnhancedViT(
        timm_vit=base_vit,
        enable_mask_tokens=enable_mask_tokens,
        interpolate_offset=interpolate_offset,
        interpolate_antialias=interpolate_antialias,
    )

    logging.info(
        f"Created DINO-enhanced ViT: {model_name} (mask_tokens={enable_mask_tokens})"
    )

    return enhanced_vit
