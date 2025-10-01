# Copyright (c) stable-pretraining contributors
# Licensed under the MIT License
#
# DINO (v1 and v2) utilities for efficient multi-crop SSL training

"""DINO utilities for Vision Transformers.

This module provides efficient multi-crop training utilities for DINO-style
self-supervised learning, including:
- Sequence packing for 3-4x training speedup
- Memory-efficient attention with xFormers
- Enhanced position encoding interpolation
- Mask tokens for iBOT loss

These utilities work with both DINOv1 and DINOv2 training objectives.
"""

from .attention import MemEffAttention, create_mem_eff_attention
from .packing import (
    clear_attn_bias_cache,
    get_attn_bias_and_cat,
    get_cache_stats,
    pack_sequences,
    unpack_sequences,
)
from .vit import DINOEnhancedViT, from_timm_dino

__all__ = [
    "DINOEnhancedViT",
    "from_timm_dino",
    "MemEffAttention",
    "create_mem_eff_attention",
    "get_attn_bias_and_cat",
    "unpack_sequences",
    "pack_sequences",
    "clear_attn_bias_cache",
    "get_cache_stats",
]
