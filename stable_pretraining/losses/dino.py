"""DINO self-distillation losses.

This module contains losses for DINO-style self-distillation including:
- DINOLoss: CLS token distillation
- iBOTPatchLoss: Masked patch prediction

Reference: DINOv2/v3 papers and https://github.com/facebookresearch/dinov3
"""

import torch
import torch.nn.functional as F

from .utils import sinkhorn_knopp


def cross_entropy_loss(t, s, temp):
    """Cross-entropy loss function for iBOT.

    Computes per-sample cross-entropy: -Σ t[i] * log_softmax(s[i]/temp)

    Args:
        t: Teacher predictions (probabilities) [*, D]
        s: Student predictions (logits) [*, D]
        temp: Temperature for student softmax

    Returns:
        Per-sample cross-entropy loss [*] (positive, lower is better)
    """
    return -torch.sum(t.float() * F.log_softmax(s.float() / temp, dim=-1), dim=-1)


class DINOv1Loss(torch.nn.Module):
    """DINOv1 loss for self-distillation with cross-entropy :cite:`caron2021emerging`.

    This loss computes cross-entropy between teacher and student logits after applying
    temperature scaling and normalization. The teacher uses either classical centering or
    Sinkhorn-Knopp normalization to prevent mode collapse.

    Usage:
        ```python
        dino_loss = DINOv1Loss()

        # Get logits from prototype layer
        student_logits = prototype_layer(student_embeddings)  # [n_views, B, out_dim]
        teacher_logits = prototype_layer(teacher_embeddings)  # [n_views, B, out_dim]

        # Approach 1: Classical centering (recommended, faster)
        teacher_probs = dino_loss.softmax_center_teacher(teacher_logits, temp=0.04)
        loss = dino_loss(student_logits, teacher_probs)
        dino_loss.update_center(teacher_logits)  # Queue async center update

        # Approach 2: Sinkhorn-Knopp (more principled, slower, no centering needed)
        n_views, batch_size, _ = teacher_logits.shape
        num_samples = n_views * batch_size  # Total samples across views
        teacher_probs = dino_loss.sinkhorn_knopp_teacher(
            teacher_logits, temp=0.04, num_samples=num_samples
        )
        loss = dino_loss(student_logits, teacher_probs)
        # No update_center() needed for Sinkhorn-Knopp!
        ```

    Args:
        temperature_student (float): Temperature for student softmax. Default is 0.1.
        center_momentum (float): EMA momentum for center update. Default is 0.9.
    """

    def __init__(
        self,
        temperature_student: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.temperature_student = temperature_student
        self.center_momentum = center_momentum
        self.center = None
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DINO cross-entropy loss.

        This is a pure loss computation with no side effects (no centering, no updates).
        Teacher probabilities should be pre-processed with softmax_center_teacher() or
        sinkhorn_knopp_teacher(). Center updates should be done separately with update_center().

        Args:
            student_logits: Student predictions [n_views, batch_size, out_dim]
            teacher_probs: Teacher probabilities (already normalized) [n_views, batch_size, out_dim]

        Returns:
            Scalar DINO loss value (cross-entropy averaged over view pairs, excluding diagonal)

        Shape:
            - student_logits: (S, B, K) where S = student views, B = batch size, K = out_dim
            - teacher_probs: (T, B, K) where T = teacher views
            - output: scalar
        """
        student_log_probs = F.log_softmax(
            student_logits.float() / self.temperature_student, dim=-1
        )

        # Compute cross-entropy: -sum over dim K of (teacher_probs * log(student_probs))
        # Using einsum : sum over batch B and prototypes K, keep view dims S, T
        loss_matrix = -torch.einsum(
            "s b k, t b k -> s t", student_log_probs, teacher_probs.float()
        )

        # Exclude diagonal (same view comparisons)
        min_views = min(student_logits.shape[0], teacher_probs.shape[0])
        loss_matrix = torch.diagonal_scatter(
            loss_matrix, loss_matrix.new_zeros(min_views)
        )

        # Average over all valid pairs
        batch_size = student_logits.shape[1]
        n_student_views = student_logits.shape[0]
        n_teacher_views = teacher_probs.shape[0]
        n_pairs = n_student_views * n_teacher_views - min_views

        return loss_matrix.sum() / (batch_size * n_pairs)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(
        self, teacher_logits, teacher_temp, num_samples=None, n_iterations=3
    ):
        """Apply Sinkhorn-Knopp optimal transport normalization to teacher logits.

        **FOR SINKHORN-KNOPP APPROACH ONLY. DOES NOT USE CENTER.**

        This method applies sinkhorn-knopp to enforce exact uniform distribution across
        prototypes without using centering. More principled than centering but more expensive.
        Used in SwAV and DINOv3 for better theoretical guarantees.

        Note: When using Sinkhorn-Knopp, you do NOT need to call update_center() or
        apply_center_update() since centering is not used.

        Args:
            teacher_logits: Teacher logits [*, out_dim]. Can be any shape as long as last dim is out_dim.
                           Common shapes: [batch, out_dim] or [n_views, batch, out_dim]
            teacher_temp: Temperature for softmax
            num_samples: Total number of samples across all GPUs (int or tensor).
                        If None, inferred from shape assuming [batch, out_dim] format.
                        For multi-view [n_views, batch, out_dim], pass n_views * batch explicitly.
            n_iterations: Number of Sinkhorn iterations (default: 3)

        Returns:
            Teacher probabilities [same shape as input] with uniform prototype distribution
        """
        # Infer num_samples if not provided
        if num_samples is None:
            # Assume shape is [batch, out_dim]
            batch_size = teacher_logits.shape[0]
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                num_samples = batch_size * world_size
            else:
                num_samples = batch_size

        # Flatten all dims except last (out_dim) for Sinkhorn-Knopp
        original_shape = teacher_logits.shape
        teacher_logits_flat = teacher_logits.view(-1, original_shape[-1])

        result = sinkhorn_knopp(
            teacher_output=teacher_logits_flat,
            teacher_temp=teacher_temp,
            num_samples=num_samples,
            n_iterations=n_iterations,
        )

        # Reshape back to original shape
        return result.view(original_shape)

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_logits, teacher_temp, update_centers=True):
        """Apply classical centering and sharpening to teacher logits.

        **FOR CLASSICAL CENTERING APPROACH ONLY.**

        This method subtracts the center (EMA of batch means) from teacher logits before
        applying softmax. This prevents mode collapse by ensuring balanced prototype usage.

        Args:
            teacher_logits: Teacher logits [*, out_dim]
            teacher_temp: Temperature for teacher softmax
            update_centers: Whether to apply queued center update before centering

        Returns:
            Teacher probabilities after centering [*, out_dim]
        """
        if update_centers:
            self.apply_center_update()
        if self.center is not None:
            return F.softmax((teacher_logits - self.center) / teacher_temp, dim=-1)
        else:
            return F.softmax(teacher_logits / teacher_temp, dim=-1)

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Queue async center update from teacher logits.

        **FOR CLASSICAL CENTERING APPROACH ONLY. NOT NEEDED FOR SINKHORN-KNOPP.**

        Starts an asynchronous all-reduce for distributed training. The update is
        applied later when softmax_center_teacher() is called with update_centers=True.
        This allows the all-reduce to overlap with backward pass for efficiency.

        Typical usage:
            teacher_probs = dino_loss.softmax_center_teacher(teacher_logits, temp)
            loss = dino_loss(student_logits, teacher_probs)
            dino_loss.update_center(teacher_logits)  # Start async update
            # ... backward pass happens here, overlapping with all-reduce ...
            # Next iteration: softmax_center_teacher() will call apply_center_update()

        Args:
            teacher_output: Teacher logits [n_views, batch_size, out_dim]
        """
        # Mark as not updated yet
        self.updated = False
        self.len_teacher_output = len(teacher_output)

        # Compute batch mean
        self.async_batch_center = torch.sum(teacher_output.mean(1), dim=0, keepdim=True)

        # Start async all-reduce across GPUs
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.reduce_handle = torch.distributed.all_reduce(
                self.async_batch_center, async_op=True
            )

    @torch.no_grad()
    def apply_center_update(self):
        """Apply the queued center update with EMA.

        **FOR CLASSICAL CENTERING APPROACH ONLY. NOT NEEDED FOR SINKHORN-KNOPP.**

        Waits for async all-reduce to complete and updates self.center with EMA.
        Automatically called by softmax_center_teacher() if update_centers=True.
        """
        if self.updated is False:
            world_size = (
                torch.distributed.get_world_size()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 1
            )

            if self.reduce_handle is not None:
                self.reduce_handle.wait()

            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            # Initialize center on first call
            if self.center is None:
                self.center = _t.clone()
            else:
                self.center = self.center * self.center_momentum + _t * (
                    1 - self.center_momentum
                )

            self.updated = True


class iBOTPatchLoss(torch.nn.Module):
    """iBOT patch-level prediction loss for masked patch prediction.

    This loss computes cross-entropy between teacher and student patch predictions
    for masked patches only. Like DINOv1Loss, it supports either classical
    centering or Sinkhorn-Knopp normalization to prevent mode collapse.

    Usage:
        ```python
        ibot_loss = iBOTPatchLoss(patch_out_dim=8192)

        # Get masked patch logits (already extracted, only masked patches)
        student_logits_masked = ...  # [n_masked, patch_out_dim]
        teacher_logits_masked = ...  # [n_masked, patch_out_dim]
        masks = ...  # [batch, n_patches] binary mask

        # Approach 1: Classical centering (recommended, faster)
        teacher_probs = ibot_loss.softmax_center_teacher(
            teacher_logits_masked, temp=0.04
        )
        loss = ibot_loss.forward_masked(student_logits_masked, teacher_probs, masks)
        ibot_loss.update_center(teacher_logits_masked)  # Queue async center update

        # Approach 2: Sinkhorn-Knopp (more principled, slower, no centering needed)
        n_masked = torch.tensor(student_logits_masked.shape[0])
        teacher_probs = ibot_loss.sinkhorn_knopp_teacher(
            teacher_logits_masked, temp=0.04, n_masked_patches_tensor=n_masked
        )
        loss = ibot_loss.forward_masked(student_logits_masked, teacher_probs, masks)
        # No update_center() needed for Sinkhorn-Knopp!
        ```

    Args:
        patch_out_dim (int): Dimensionality of patch prototypes (number of clusters).
        student_temp (float): Temperature for student softmax. Default is 0.1.
        center_momentum (float): EMA momentum for center update. Default is 0.9.
    """

    def __init__(
        self,
        patch_out_dim: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.patch_out_dim = patch_out_dim
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center = None
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_patch_tokens = None
        self.async_batch_center = None

    def forward(self, student_patch_logits, teacher_patch_probs, student_masks_flat):
        """Compute iBOT cross-entropy loss for all patches.

        This is a pure loss computation with no side effects (no centering, no updates).
        Teacher probabilities should be pre-processed with softmax_center_teacher() or
        sinkhorn_knopp_teacher(). Center updates should be done separately with update_center().

        This version processes all patches and masks out non-masked ones. Use forward_masked()
        for memory-efficient computation with only masked patches.

        Args:
            student_patch_logits: Student patch logits [batch, n_patches, patch_out_dim]
            teacher_patch_probs: Teacher probabilities (already normalized) [batch, n_patches, patch_out_dim]
            student_masks_flat: Binary mask (1 = masked, 0 = unmasked) [batch, n_patches]

        Returns:
            Scalar iBOT loss value (cross-entropy for masked patches only)
        """
        loss = cross_entropy_loss(
            teacher_patch_probs, student_patch_logits, self.student_temp
        )
        # Weight by mask and normalize by number of masked patches per sample
        loss = torch.sum(
            loss * student_masks_flat.float(), dim=-1
        ) / student_masks_flat.sum(dim=-1).clamp(min=1.0)
        return loss.mean()

    def forward_masked(
        self,
        student_patch_logits_masked,
        teacher_patch_probs_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
    ):
        """Compute iBOT cross-entropy loss for masked patches only (memory-efficient).

        This is a pure loss computation with no side effects (no centering, no updates).
        Teacher probabilities should be pre-processed with softmax_center_teacher() or
        sinkhorn_knopp_teacher(). Center updates should be done separately with update_center().

        This version processes only masked patches (more memory-efficient for large models).
        Used in DINOv3.

        Args:
            student_patch_logits_masked: Student logits for masked patches [n_masked, patch_out_dim]
            teacher_patch_probs_masked: Teacher probs for masked patches [n_masked, patch_out_dim]
            student_masks_flat: Binary mask (1 = masked, 0 = unmasked) [batch, n_patches]
            n_masked_patches: Number of patches to use (optional truncation)
            masks_weight: Per-patch weights [n_masked] (optional, computed if None)

        Returns:
            Scalar iBOT loss value
        """
        loss = cross_entropy_loss(
            teacher_patch_probs_masked, student_patch_logits_masked, self.student_temp
        )

        # Compute per-patch weights if not provided
        # Default: 1 / num_masked_patches_per_sample for balanced contribution
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )

        # Truncate to n_masked_patches if specified (for memory/speed)
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
            masks_weight = masks_weight[:n_masked_patches]

        loss = loss * masks_weight
        return loss.sum() / student_masks_flat.shape[0]

    @torch.no_grad()
    def sinkhorn_knopp_teacher(
        self,
        teacher_patch_tokens,
        teacher_temp,
        n_masked_patches_tensor,
        n_iterations=3,
    ):
        """Apply Sinkhorn-Knopp optimal transport normalization to teacher patch logits.

        **FOR SINKHORN-KNOPP APPROACH ONLY. DOES NOT USE CENTER.**

        This method applies optimal transport to enforce exact uniform distribution across
        prototypes without using centering. More principled than centering but more expensive.
        Used in SwAV and DINOv3 for better theoretical guarantees.

        Note: When using Sinkhorn-Knopp, you do NOT need to call update_center() since
        centering is not used.

        Args:
            teacher_patch_tokens: Teacher patch logits [n_masked, patch_out_dim]
            teacher_temp: Temperature for softmax
            n_masked_patches_tensor: Total number of masked patches across all GPUs (int or tensor)
            n_iterations: Number of Sinkhorn iterations (default: 3)

        Returns:
            Teacher probabilities [n_masked, patch_out_dim] with uniform prototype distribution
        """
        return sinkhorn_knopp(
            teacher_output=teacher_patch_tokens,
            teacher_temp=teacher_temp,
            num_samples=n_masked_patches_tensor,
            n_iterations=n_iterations,
        )

    @torch.no_grad()
    def softmax_center_teacher(
        self, teacher_patch_tokens, teacher_temp, update_centers=True
    ):
        """Apply classical centering and sharpening to teacher patch logits.

        **FOR CLASSICAL CENTERING APPROACH ONLY.**

        This method subtracts the center (EMA of batch means) from teacher logits before
        applying softmax. This prevents mode collapse by ensuring balanced prototype usage.

        Args:
            teacher_patch_tokens: Teacher patch logits [n_masked, patch_out_dim]
            teacher_temp: Temperature for teacher softmax
            update_centers: Whether to apply queued center update before centering

        Returns:
            Teacher probabilities after centering [n_masked, patch_out_dim]
        """
        if update_centers:
            self.apply_center_update()
        if self.center is not None:
            return F.softmax(
                (teacher_patch_tokens - self.center) / teacher_temp, dim=-1
            )
        else:
            return F.softmax(teacher_patch_tokens / teacher_temp, dim=-1)

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        """Queue async center update from teacher patch logits.

        **FOR CLASSICAL CENTERING APPROACH ONLY. NOT NEEDED FOR SINKHORN-KNOPP.**

        Starts an asynchronous all-reduce for distributed training. The update is
        applied later when softmax_center_teacher() is called with update_centers=True.
        This allows the all-reduce to overlap with backward pass for efficiency.

        Typical usage:
            teacher_probs = ibot_loss.softmax_center_teacher(teacher_logits, temp)
            loss = ibot_loss.forward_masked(student_logits, teacher_probs, masks)
            ibot_loss.update_center(teacher_logits)  # Start async update
            # ... backward pass happens here, overlapping with all-reduce ...
            # Next iteration: softmax_center_teacher() will call apply_center_update()

        Args:
            teacher_patch_tokens: Teacher patch logits [n_masked, patch_out_dim]
        """
        # Mark as not updated yet
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)

        # Compute mean across masked patches
        self.async_batch_center = teacher_patch_tokens.mean(dim=0, keepdim=True)

        # Start async all-reduce across GPUs
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.reduce_handle = torch.distributed.all_reduce(
                self.async_batch_center, async_op=True
            )

    @torch.no_grad()
    def apply_center_update(self):
        """Apply the queued center update with EMA.

        **FOR CLASSICAL CENTERING APPROACH ONLY. NOT NEEDED FOR SINKHORN-KNOPP.**

        Waits for async all-reduce to complete and updates self.center with EMA.
        Automatically called by softmax_center_teacher() if update_centers=True.
        """
        if self.updated is False:
            world_size = (
                torch.distributed.get_world_size()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 1
            )

            if self.reduce_handle is not None:
                self.reduce_handle.wait()

            # Average across GPUs
            _t = self.async_batch_center / world_size

            # Initialize center on first call
            if self.center is None:
                self.center = _t.clone()
            else:
                # EMA update
                self.center = self.center * self.center_momentum + _t * (
                    1 - self.center_momentum
                )

            self.updated = True
