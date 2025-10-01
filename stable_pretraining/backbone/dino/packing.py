# Copyright (c) stable-pretraining contributors
# Licensed under the MIT License
#
# This file implements sequence packing utilities for DINOv2.
# Inspired by concepts from DINOv2: https://github.com/facebookresearch/dinov2

"""Sequence packing utilities for efficient multi-view processing.

This module provides utilities to pack multiple image crops into a single
sequence with block-diagonal attention masking, enabling 3-4x training speedup
for DINOv2-style multi-crop training.

Key features:
- Pack/unpack multiple tensors into single sequence
- Create BlockDiagonalMask for preventing cross-view attention
- Cache masks for repeated shapes (efficiency)
- Support for variable-size crops
"""

from typing import Dict, List, Tuple

import torch
from loguru import logger as logging

# Check for xFormers availability
try:
    from xformers.ops import fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    logging.warning("xFormers not available - sequence packing will not work")


# Global cache for attention bias masks (avoid recreating same masks)
_attn_bias_cache: Dict[Tuple, object] = {}


def get_attn_bias_and_cat(x_list: List[torch.Tensor]) -> Tuple[object, torch.Tensor]:
    """Pack multiple tensors and create block-diagonal attention mask.

    This is the core sequence packing function. It:
    1. Concatenates multiple tensors along sequence dimension
    2. Creates (or retrieves from cache) a BlockDiagonalMask
    3. Returns both the mask and packed tensor

    Args:
        x_list: List of tensors, each [B, N, D] where:
                - B: batch size (must be same for all)
                - N: sequence length (can differ)
                - D: embedding dimension (must be same for all)

    Returns:
        Tuple of (attn_bias, cat_tensor):
            - attn_bias: BlockDiagonalMask for preventing cross-sequence attention
            - cat_tensor: Concatenated tensor [B, total_N, D]

    Example:
        >>> # 2 global crops (197 tokens) + 8 local crops (37 tokens)
        >>> x_global1 = torch.randn(64, 197, 384)
        >>> x_global2 = torch.randn(64, 197, 384)
        >>> x_local = [torch.randn(64, 37, 384) for _ in range(8)]
        >>> x_list = [x_global1, x_global2] + x_local
        >>>
        >>> attn_bias, x_packed = get_attn_bias_and_cat(x_list)
        >>> print(x_packed.shape)  # [64, 690, 384] (197+197+8*37)
        >>> print(type(attn_bias))  # BlockDiagonalMask

    Note:
        - Requires xFormers to be installed
        - Masks are cached based on (batch_size, seq_len) tuples
        - All tensors in x_list must have same batch size and embedding dim
    """
    if not XFORMERS_AVAILABLE:
        raise ImportError(
            "xFormers is required for sequence packing. "
            "Install with: pip install xformers"
        )

    # Extract batch sizes and sequence lengths
    batch_sizes = [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))

    # Check cache for this shape combination
    if all_shapes not in _attn_bias_cache:
        # Create sequence lengths list for BlockDiagonalMask
        # For batch_size=64 and 10 crops: [197, 197, 37, ...] repeated 64 times
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])

        # Create block-diagonal mask (non-materialized!)
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)

        # Store batch sizes for potential future use
        attn_bias._batch_sizes = batch_sizes

        # Cache for reuse
        _attn_bias_cache[all_shapes] = attn_bias

        logging.debug(
            f"Created BlockDiagonalMask for shapes {all_shapes}, "
            f"total seqlens: {len(seqlens)}"
        )

    # Concatenate tensors
    # Reshape each [B, N, D] -> [1, B*N, D] then concatenate
    tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
    cat_tensor = torch.cat(tensors_bs1, dim=1)

    return _attn_bias_cache[all_shapes], cat_tensor


def unpack_sequences(
    packed_tensor: torch.Tensor, sequence_lengths: List[int], batch_size: int
) -> List[torch.Tensor]:
    """Unpack concatenated tensor back into list of separate tensors.

    Args:
        packed_tensor: Concatenated tensor [1, B*total_N, D] from get_attn_bias_and_cat
        sequence_lengths: List of sequence lengths for each view (e.g., [197, 197, 37, ...])
        batch_size: Original batch size

    Returns:
        List of tensors, each [B, N_i, D] where N_i is sequence_lengths[i]

    Example:
        >>> # After packing: packed has shape [1, 64*690, 384] = [1, 44160, 384]
        >>> packed = torch.randn(1, 44160, 384)
        >>> seqlens = [197, 197] + [37] * 8  # Per-view lengths
        >>> x_list = unpack_sequences(packed, seqlens, batch_size=64)
        >>> print(len(x_list))  # 10
        >>> print(x_list[0].shape)  # [64, 197, 384]
        >>> print(x_list[2].shape)  # [64, 37, 384]
    """
    embed_dim = packed_tensor.shape[-1]
    total_seq_len = sum(sequence_lengths)

    # Input format: [1, B*total_seq, D]
    # Need to reshape to [B, total_seq, D] then split by views
    assert packed_tensor.shape[0] == 1, (
        f"Expected batch_size=1, got {packed_tensor.shape[0]}"
    )
    assert packed_tensor.shape[1] == batch_size * total_seq_len, (
        f"Expected seq_len={batch_size * total_seq_len}, got {packed_tensor.shape[1]}"
    )

    # Reshape [1, B*total_seq, D] → [B, total_seq, D]
    packed_tensor = packed_tensor.view(batch_size, total_seq_len, embed_dim)

    # Split along sequence dimension by view lengths
    tensors = []
    start_idx = 0
    for seq_len in sequence_lengths:
        end_idx = start_idx + seq_len
        tensors.append(packed_tensor[:, start_idx:end_idx, :])
        start_idx = end_idx

    return tensors


def pack_sequences(x_list: List[torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:
    """Simple concatenation of sequences (without attention mask).

    This is a simpler alternative when you want to handle masking separately.

    Args:
        x_list: List of tensors [B, N_i, D]

    Returns:
        Tuple of (packed_tensor, sequence_lengths):
            - packed_tensor: [B, total_N, D]
            - sequence_lengths: List of N_i values

    Example:
        >>> x_list = [torch.randn(64, 197, 384), torch.randn(64, 37, 384)]
        >>> packed, lens = pack_sequences(x_list)
        >>> print(packed.shape)  # [64, 234, 384]
        >>> print(lens)  # [197, 37]
    """
    # Check all have same batch size and embed dim
    batch_size = x_list[0].shape[0]
    embed_dim = x_list[0].shape[-1]

    for x in x_list:
        assert x.shape[0] == batch_size, "All tensors must have same batch size"
        assert x.shape[-1] == embed_dim, "All tensors must have same embed dim"

    # Concatenate along sequence dimension
    packed = torch.cat(x_list, dim=1)
    sequence_lengths = [x.shape[1] for x in x_list]

    return packed, sequence_lengths


def clear_attn_bias_cache():
    """Clear the attention bias cache.

    Useful when switching between different model configurations or
    to free memory after training.
    """
    global _attn_bias_cache
    _attn_bias_cache.clear()
    logging.debug("Cleared attention bias cache")


def get_cache_stats() -> Dict[str, int]:
    """Get statistics about the attention bias cache.

    Returns:
        Dictionary with cache statistics
    """
    return {
        "num_cached_masks": len(_attn_bias_cache),
        "cache_keys": list(_attn_bias_cache.keys()),
    }
