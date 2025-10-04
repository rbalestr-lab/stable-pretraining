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
            student_logits: Student logits [n_views, batch_size, out_dim]
            teacher_probs: Teacher probabilities (already normalized) [n_views, batch_size, out_dim]

        Returns:
            Scalar DINO loss value (cross-entropy averaged over view pairs, excluding diagonal)

        Shape:
            - student_logits: (S, B, K) where S = student views, B = batch size, K = out_dim
            - teacher_probs: (T, B, K) where T = teacher views
            - output: scalar
        """
        # Apply temperature-scaled log-softmax to student
        student_log_probs = F.log_softmax(
            student_logits.float() / self.temperature_student, dim=-1
        )

        # Compute cross-entropy matrix: [S, T]
        # Sum over batch and features, keep view dimensions
        loss_matrix = -torch.einsum("sbk,tbk->st", student_log_probs, teacher_probs)

        # Zero out diagonal (same view comparisons)
        loss_matrix.fill_diagonal_(0)

        # Average over valid pairs and batch size
        n_terms = loss_matrix.numel() - loss_matrix.diagonal().numel()
        batch_size = student_logits.shape[1]
        return loss_matrix.sum() / (n_terms * batch_size)

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

        **FOR CLASSICAL CENTERING APPROACH ONLY. NOT NEEDED FOR SINKHORN-KNOPP.**

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
    for masked patches only. Uses Sinkhorn-Knopp normalization exclusively (as in DINOv2/v3)
    to prevent mode collapse.

    Args:
        student_temp (float): Temperature for student softmax. Default is 0.1.
    """

    def __init__(
        self,
        student_temp: float = 0.1,
    ):
        super().__init__()
        self.student_temp = student_temp

    def forward(self, student_patch_logits, teacher_patch_probs):
        """Compute iBOT cross-entropy loss for masked patches.

        Args:
            student_patch_logits: Student patch logits [n_masked_total, patch_out_dim]
            teacher_patch_probs: Teacher probabilities [n_masked_total, patch_out_dim]

        Returns:
            Scalar iBOT loss value
        """
        loss = cross_entropy_loss(
            teacher_patch_probs, student_patch_logits, self.student_temp
        )
        return loss.mean()

    @torch.no_grad()
    def sinkhorn_knopp_teacher(
        self,
        teacher_patch_tokens,
        teacher_temp,
        num_samples=None,
        n_iterations=3,
    ):
        """Apply Sinkhorn-Knopp optimal transport normalization to teacher patch logits.

        This method applies optimal transport to enforce exact uniform distribution across
        prototypes. Used exclusively in DINOv2/v3 for iBOT patch loss.

        Args:
            teacher_patch_tokens: Teacher patch logits [n_masked, patch_out_dim]
            teacher_temp: Temperature for softmax
            num_samples: Total number of masked patches across all GPUs (int or tensor).
                        If None, inferred from shape.
            n_iterations: Number of Sinkhorn iterations (default: 3)

        Returns:
            Teacher probabilities [n_masked, patch_out_dim] with uniform prototype distribution
        """
        return sinkhorn_knopp(
            teacher_output=teacher_patch_tokens,
            teacher_temp=teacher_temp,
            num_samples=num_samples,
            n_iterations=n_iterations,
        )


class DINOv2Loss(torch.nn.Module):
    """DINOv2 loss combining CLS token and masked patch losses.

    DINOv2 combines two losses:
    - DINOv1Loss: CLS token distillation (global views) - uses Sinkhorn-Knopp
    - iBOTPatchLoss: Masked patch prediction - uses Sinkhorn-Knopp

    Both losses use Sinkhorn-Knopp normalization in DINOv2.

    Args:
        dino_loss_weight (float): Weight for CLS token loss. Default is 1.0.
        ibot_loss_weight (float): Weight for iBOT patch loss. Default is 1.0.
        temperature_student (float): Temperature for student softmax in DINO. Default is 0.1.
        center_momentum (float): EMA momentum for DINO centering (not used by iBOT). Default is 0.9.
        student_temp (float): Temperature for student softmax in iBOT. Default is 0.1.
    """

    def __init__(
        self,
        dino_loss_weight: float = 1.0,
        ibot_loss_weight: float = 1.0,
        temperature_student: float = 0.1,
        center_momentum: float = 0.9,
        student_temp: float = 0.1,
    ):
        super().__init__()
        self.dino_loss_weight = dino_loss_weight
        self.ibot_loss_weight = ibot_loss_weight

        self.dino_loss = DINOv1Loss(
            temperature_student=temperature_student,
            center_momentum=center_momentum,
        )

        self.ibot_loss = iBOTPatchLoss(
            student_temp=student_temp,
        )

    def forward(
        self,
        student_cls_logits: torch.Tensor,
        teacher_cls_probs: torch.Tensor,
        student_patch_logits: torch.Tensor = None,
        teacher_patch_probs: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute combined DINOv2 loss.

        Args:
            student_cls_logits: Student CLS logits [n_views, batch, out_dim]
            teacher_cls_probs: Teacher CLS probs [n_views, batch, out_dim]
            student_patch_logits: Student patch logits [n_masked_total, patch_out_dim] or None
            teacher_patch_probs: Teacher patch probs [n_masked_total, patch_out_dim] or None

        Returns:
            Combined weighted loss
        """
        dino_loss = self.dino_loss(student_cls_logits, teacher_cls_probs)

        if student_patch_logits is None or teacher_patch_probs is None:
            return self.dino_loss_weight * dino_loss

        ibot_loss = self.ibot_loss(student_patch_logits, teacher_patch_probs)
        return self.dino_loss_weight * dino_loss + self.ibot_loss_weight * ibot_loss
