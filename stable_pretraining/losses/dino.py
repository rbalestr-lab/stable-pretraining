"""DINO self-distillation losses.

This module contains losses for DINO-style self-distillation including:
- DINOLoss: CLS token distillation
- iBOTPatchLoss: Masked patch prediction

Reference: DINOv2/v3 papers and https://github.com/facebookresearch/dinov3
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import all_gather
from .utils import sinkhorn_knopp


def cross_entropy_loss(t, s, temp):
    """Cross-entropy loss function for iBOT.

    Computes the cross-entropy loss between teacher probabilities and student logits.
    Standard definition: CE(t, s) = -Σ t[i] * log(softmax(s[i]))

    Args:
        t: Teacher predictions (probabilities) [*, D]
        s: Student predictions (logits) [*, D]
        temp: Temperature for student softmax

    Returns:
        Per-sample cross-entropy loss [*] (positive values to minimize)
    """
    return -torch.sum(t.float() * F.log_softmax(s.float() / temp, dim=-1), dim=-1)


class DINOLoss(torch.nn.Module):
    """DINO loss for self-distillation with cross-entropy :cite:`caron2021emerging`.

    This loss computes the cross-entropy between teacher and student predictions
    using a sharpening temperature for the student and a separate temperature for
    the teacher. The teacher predictions are centered to prevent mode collapse.

    Args:
        out_dim (int): Dimensionality of the DINO head output (number of prototypes). Default is 65536.
        temperature_student (float, optional): Temperature for student softmax. Default is 0.1.
        center_momentum (float, optional): Momentum for center update. Default is 0.9.
    """

    def __init__(
        self,
        out_dim: int = 65536,
        temperature_student: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.temperature_student = temperature_student
        self.center_momentum = center_momentum
        self.register_buffer("center", None)
        self.prototypes_layer = None  # Will be created on first forward pass

    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
        temperature_teacher: float = 0.04,
        update_center: bool = True,
    ) -> torch.Tensor:
        """Compute DINO loss.

        Args:
            student_embeddings: Student network output embeddings [N_views, batch_size, embed_dim].
            teacher_embeddings: Teacher network output embeddings [N_views, batch_size, embed_dim].
            temperature_teacher: Temperature for teacher softmax.
            update_center: Whether to update the center (should be True during training).

        Returns:
            torch.Tensor: Scalar DINO loss value.
        """
        # Initialize prototypes layer if needed
        if self.prototypes_layer is None:
            embed_dim = student_embeddings.shape[-1]
            self.prototypes_layer = nn.Linear(embed_dim, self.out_dim, bias=False).to(
                student_embeddings.device
            )

        # L2 normalize embeddings
        student_embeddings = F.normalize(student_embeddings, dim=-1, p=2)
        teacher_embeddings = F.normalize(teacher_embeddings, dim=-1, p=2)

        # Apply prototypes layer to get logits
        student_logits = self.prototypes_layer(student_embeddings)
        teacher_logits = self.prototypes_layer(teacher_embeddings)
        # Compute teacher probabilities with centering
        if self.center is not None:
            teacher_probs = F.softmax(
                (teacher_logits - self.center) / temperature_teacher,
                dim=-1,
            )
        else:
            teacher_probs = F.softmax(teacher_logits / temperature_teacher, dim=-1)

        # Compute student log probabilities
        student_log_probs = F.log_softmax(
            student_logits / self.temperature_student, dim=-1
        )

        # Compute cross-entropy loss following the original DINO implementation
        # Flatten batch and feature dimensions together: [n_views, batch_size * dim]
        teacher_probs_flat = teacher_probs.flatten(start_dim=1)
        student_log_probs_flat = student_log_probs.flatten(start_dim=1)

        # Compute cross-entropy matrix: [n_teacher_views, n_student_views]
        # Each element represents the total cross-entropy across all batch items and features
        loss_matrix = -teacher_probs_flat @ student_log_probs_flat.T

        # Zero out the diagonal (same view comparisons)
        loss_matrix.fill_diagonal_(0)

        # Normalize the loss
        # Total number of valid terms: all pairs minus diagonal
        n_terms = loss_matrix.numel() - loss_matrix.diagonal().numel()
        batch_size = student_logits.shape[1]

        # Average over valid terms and batch size
        loss = loss_matrix.sum() / (n_terms * batch_size)

        # Update center with EMA
        if update_center and self.training:
            with torch.no_grad():
                # Compute batch center from teacher logits
                batch_center = torch.mean(teacher_logits, dim=(0, 1))

                if self.center is None:
                    self.center = batch_center
                else:
                    # For distributed training, gather from all GPUs
                    if (
                        torch.distributed.is_available()
                        and torch.distributed.is_initialized()
                    ):
                        batch_center = torch.cat(all_gather(batch_center), dim=0).mean(
                            dim=0
                        )
                    self.center = self.center * self.center_momentum + batch_center * (
                        1 - self.center_momentum
                    )

        return loss


class iBOTPatchLoss(torch.nn.Module):
    """iBOT patch-level prediction loss.

    Implements masked patch prediction loss from iBOT paper, used on top of DINO.
    The student predicts teacher representations for masked patches.

    Reference: https://github.com/facebookresearch/dinov3

    Args:
        patch_out_dim (int): Output dimension of patch predictions (number of prototypes)
        student_temp (float, optional): Temperature for student softmax. Default: 0.1
        center_momentum (float, optional): EMA coefficient for center update. Default: 0.9
    """

    def __init__(
        self,
        patch_out_dim: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        # Initialize center with nan (will be zeroed in init_weights)
        self.register_buffer("center", torch.full((1, 1, patch_out_dim), math.nan))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_patch_tokens = None
        self.async_batch_center = None

    def init_weights(self) -> None:
        """Initialize center to zeros (called after model initialization)."""
        self.center.zero_()

    @torch.no_grad()
    def softmax_center_teacher(
        self, teacher_patch_tokens, teacher_temp, update_centers=True
    ):
        """Apply centering and sharpening to teacher patch predictions.

        Args:
            teacher_patch_tokens: Teacher patch predictions [N, D]
            teacher_temp: Temperature for teacher softmax
            update_centers: Whether to apply center update before centering

        Returns:
            Softmax probabilities after centering
        """
        if update_centers:
            self.apply_center_update()
        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(
        self,
        teacher_patch_tokens,
        teacher_temp,
        n_masked_patches_tensor,
        n_iterations=3,
    ):
        """Apply Sinkhorn-Knopp optimal transport normalization to teacher predictions.

        Alternative to centering that enforces exact uniform distribution across prototypes.
        More expensive but more principled. Used in SwAV and DINOv3.

        Args:
            teacher_patch_tokens: Teacher patch predictions [N, D]
            teacher_temp: Temperature for softmax
            n_masked_patches_tensor: Total number of masked patches (can be tensor for distributed)
            n_iterations: Number of Sinkhorn iterations (default: 3)

        Returns:
            Normalized probabilities [N, D] with uniform distribution across prototypes
        """
        return sinkhorn_knopp(
            teacher_output=teacher_patch_tokens,
            teacher_temp=teacher_temp,
            num_samples=n_masked_patches_tensor,
            n_iterations=n_iterations,
        )

    def forward(self, student_patch_tokens, teacher_patch_tokens, student_masks_flat):
        """Compute iBOT loss for all patches (standard forward).

        Cross-entropy between softmax outputs of the teacher and student networks.

        Args:
            student_patch_tokens: Student predictions [B, N, D]
            teacher_patch_tokens: Teacher predictions [B, N, D]
            student_masks_flat: Binary mask indicating which patches are masked [B, N]

        Returns:
            Scalar loss value
        """
        t = teacher_patch_tokens
        s = student_patch_tokens
        loss = cross_entropy_loss(t, s, self.student_temp)
        loss = torch.sum(
            loss * student_masks_flat.float(), dim=-1
        ) / student_masks_flat.sum(dim=-1).clamp(min=1.0)
        return loss.mean()

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
    ):
        """Compute iBOT loss for masked patches (memory-efficient version).

        This version processes only the masked patches, making it more memory-efficient
        for large models. Used in DINOv3.

        Args:
            student_patch_tokens_masked: Student predictions for masked patches [N_masked, D]
            teacher_patch_tokens_masked: Teacher predictions for masked patches [N_masked, D]
            student_masks_flat: Binary mask indicating which patches are masked [B, N]
            n_masked_patches: Number of masked patches to use (optional, for truncation)
            masks_weight: Per-patch weights [N_masked] (optional)

        Returns:
            Scalar loss value
        """
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        loss = cross_entropy_loss(t, s, self.student_temp)

        # Compute per-patch weights if not provided
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )

        # Truncate to n_masked_patches if specified
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
            masks_weight = masks_weight[:n_masked_patches]

        loss = loss * masks_weight
        return loss.sum() / student_masks_flat.shape[0]

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        """Update center with EMA from teacher patch tokens.

        Args:
            teacher_patch_tokens: Teacher patch predictions [N, D]
        """
        self.reduce_center_update(teacher_patch_tokens)

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens):
        """Prepare center update (async for distributed training).

        Args:
            teacher_patch_tokens: Teacher patch predictions [N, D]
        """
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        self.async_batch_center = torch.sum(
            teacher_patch_tokens.mean(1), dim=0, keepdim=True
        )

        # Async all-reduce for distributed training
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.reduce_handle = torch.distributed.all_reduce(
                self.async_batch_center, async_op=True
            )

    @torch.no_grad()
    def apply_center_update(self):
        """Apply the prepared center update with EMA."""
        if self.updated is False:
            world_size = (
                torch.distributed.get_world_size()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 1
            )

            if self.reduce_handle is not None:
                self.reduce_handle.wait()

            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)

            self.center = self.center * self.center_momentum + _t * (
                1 - self.center_momentum
            )

            self.updated = True
