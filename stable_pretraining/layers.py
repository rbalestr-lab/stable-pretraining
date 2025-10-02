"""Utility layers for SSL models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Norm(nn.Module):
    """L2 normalization layer that normalizes input to unit length.

    Normalizes the input tensor along the last dimension to have unit L2 norm.
    Commonly used in DINO before the prototypes layer.

    Example:
        ```python
        projector = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, 256),
            L2Norm(),  # Normalize to unit length
            nn.Linear(256, 4096, bias=False),  # Prototypes
        )
        ```
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to unit L2 norm.

        Args:
            x: Input tensor [..., D]

        Returns:
            L2-normalized tensor [..., D] where each D-dimensional vector has unit length
        """
        return F.normalize(x, dim=-1, p=2)
