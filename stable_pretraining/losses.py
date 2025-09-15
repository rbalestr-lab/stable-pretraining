"""SSL losses."""

from typing import Optional
import torch
import torch.nn.functional as F
from loguru import logger as logging

from .utils import all_gather, all_reduce


def mae(target, pred, mask, norm_pix_loss=False):
    """Compute masked autoencoder loss.

    Args:
        target: [N, L, p*p*3] target images
        pred: [N, L, p*p*3] predicted images
        mask: [N, L], 0 is keep, 1 is remove
        norm_pix_loss: whether to normalize pixels

    Returns:
        loss: mean loss value
    """
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss


def off_diagonal(x):
    """Return a flattened view of the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m, logging.error("Input tensor must be square.")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class NegativeCosineSimilarity(torch.nn.Module):
    """Negative cosine similarity objective.

    This objective is used for instance in BYOL :cite:`grill2020bootstrap`
    or SimSiam :cite:`chen2021exploring`.
    """

    def forward(self, z_i, z_j):
        """Compute the loss of the BYOL model.

        Args:
            z_i (torch.Tensor): Latent representation of the first augmented view of the batch.
            z_j (torch.Tensor): Latent representation of the second augmented view of the batch.

        Returns:
            float: The computed loss.
        """
        sim = torch.nn.CosineSimilarity(dim=1)
        return -sim(z_i, z_j).mean()


class BYOLLoss(torch.nn.Module):
    """Normalized MSE objective used in BYOL :cite:`grill2020bootstrap`.

    Computes the mean squared error between L2-normalized online predictions
    and L2-normalized target projections.
    """

    def forward(
        self, online_pred: torch.Tensor, target_proj: torch.Tensor
    ) -> torch.Tensor:
        """Compute BYOL loss.

        Args:
            online_pred: Predictions from the online network predictor.
            target_proj: Projections from the target network (no gradient).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        online_pred = F.normalize(online_pred, dim=-1, p=2)
        target_proj = F.normalize(target_proj, dim=-1, p=2)
        loss = 2 - 2 * (online_pred * target_proj).sum(dim=-1)
        return loss.mean()


class VICRegLoss(torch.nn.Module):
    """SSL objective used in VICReg :cite:`bardes2021vicreg`.

    Args:
        sim_coeff (float, optional): The weight of the similarity loss (attractive term).
            Default is 25.
        std_coeff (float, optional): The weight of the standard deviation loss.
            Default is 25.
        cov_coeff (float, optional): The weight of the covariance loss.
            Default is 1.
        epsilon (float, optional): Small value to avoid division by zero.
            Default is 1e-4.
    """

    def __init__(
        self,
        sim_coeff: float = 25,
        std_coeff: float = 25,
        cov_coeff: float = 1,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.epsilon = epsilon

    def forward(self, z_i, z_j):
        """Compute the loss of the VICReg model.

        Args:
            z_i (torch.Tensor): Latent representation of the first augmented view of the batch.
            z_j (torch.Tensor): Latent representation of the second augmented view of the batch.

        Returns:
            float: The computed loss.
        """
        repr_loss = F.mse_loss(z_i, z_j)

        z_i = torch.cat(all_gather(z_i), 0)
        z_j = torch.cat(all_gather(z_j), 0)

        z_i = z_i - z_i.mean(dim=0)
        z_j = z_j - z_j.mean(dim=0)

        std_i = torch.sqrt(z_i.var(dim=0) + self.epsilon)
        std_j = torch.sqrt(z_j.var(dim=0) + self.epsilon)
        std_loss = torch.mean(F.relu(1 - std_i)) / 2 + torch.mean(F.relu(1 - std_j)) / 2

        cov_i = (z_i.T @ z_i) / (z_i.size(0) - 1)
        cov_j = (z_j.T @ z_j) / (z_i.size(0) - 1)
        cov_loss = off_diagonal(cov_i).pow_(2).sum().div(z_i.size(1)) + off_diagonal(
            cov_j
        ).pow_(2).sum().div(z_i.size(1))

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss


class BarlowTwinsLoss(torch.nn.Module):
    """SSL objective used in Barlow Twins :cite:`zbontar2021barlow`.

    Args:
        lambd (float, optional): The weight of the off-diagonal terms in the loss.
            Default is 5e-3.
    """

    def __init__(self, lambd: float = 5e-3):
        super().__init__()
        self.lambd = lambd
        self.bn = torch.nn.LazyBatchNorm1d()

    def forward(self, z_i, z_j):
        """Compute the loss of the Barlow Twins model.

        Args:
            z_i (torch.Tensor): Latent representation of the first augmented view of the batch.
            z_j (torch.Tensor): Latent representation of the second augmented view of the batch.

        Returns:
            float: The computed loss.
        """
        c = self.bn(z_i).T @ self.bn(z_j)  # normalize along the batch dimension
        c = c / z_i.size(0)
        all_reduce(c)

        on_diag = (torch.diagonal(c) - 1).pow(2).sum()
        off_diag = off_diagonal(c).pow(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


class ContrastiveLoss(torch.nn.Module):
    """A general-purpose engine for InfoNCE-style contrastive loss.

    This module computes the cross-entropy loss between anchor embeddings
    and a set of candidate embeddings, given the ground-truth targets. It
    forms the core mathematical operation for losses like those in CLIP
    and SimCLR.

    Args:
        temperature (float, optional): The temperature scaling factor.
            Default is 0.07.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def _compute(
        self,
        anchors: torch.Tensor,
        candidates: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        logit_scale: Optional[torch.Tensor | float] = None,
    ) -> torch.Tensor:
        logit_scale = self.temperature if logit_scale is None else logit_scale

        anchors = torch.cat(all_gather(F.normalize(anchors, dim=-1)), 0)
        candidates = torch.cat(all_gather(F.normalize(candidates, dim=-1)), 0)

        logits = (anchors @ candidates.T) / logit_scale

        if mask is not None:
            logits = logits.masked_fill(mask, -torch.inf)

        return F.cross_entropy(logits, targets)

    def forward(
        self,
        anchors: torch.Tensor,
        candidates: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        logit_scale: Optional[torch.Tensor | float] = None,
    ) -> torch.Tensor:
        """Computes the contrastive loss.

        Args:
            anchors (torch.Tensor): The primary set of embeddings (queries) of shape `[N, D]`.
            candidates (torch.Tensor): The set of embeddings to contrast against (keys)
                of shape `[M, D]`.
            targets (torch.Tensor): A 1D tensor of ground-truth indices of shape `[N]`,
                where `targets[i]` is the index of the positive candidate for `anchors[i]`.
            mask (torch.Tensor, optional): A boolean mask of shape `[N, M]` to exclude
                certain anchor-candidate pairs from the loss calculation. Values set to
                `True` will be ignored.
            logit_scale (torch.Tensor | float, optional): The temperature scaling factor.
                Default is `self.temperature`.

        Returns:
            torch.Tensor: A scalar loss value.
        """
        return self._compute(anchors, candidates, targets, mask, logit_scale)


class NTXEntLoss(ContrastiveLoss):
    """Normalized temperature-scaled cross entropy loss.

    Introduced in the SimCLR paper :cite:`chen2020simple`.
    Also used in MoCo :cite:`he2020momentum`.

    Args:
        temperature (float, optional): The temperature scaling factor.
            Default is 0.5.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__(temperature=temperature)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute the NT-Xent loss.

        Args:
            z_i (torch.Tensor): Latent representation of the first augmented view of the batch.
            z_j (torch.Tensor): Latent representation of the second augmented view of the batch.

        Returns:
            float: The computed contrastive loss.
        """
        anchors = torch.cat([z_i, z_j], dim=0)
        candidates = anchors

        N = z_i.size(0)
        targets = torch.cat(
            [
                torch.arange(N, 2 * N, device=z_i.device),
                torch.arange(N, device=z_i.device),
            ]
        )
        # prevent self-matching by masking diagonal
        mask = torch.eye(2 * N, dtype=torch.bool, device=z_i.device)

        return self._compute(anchors, candidates, targets, mask=mask)


class SymmetricContrastiveLoss(ContrastiveLoss):
    """Symmetric contrastive image-text loss, as used in CLIP :cite:`radford2021learningtransferablevisualmodels`.

    Computes symmetric cross-entropy over image-text and text-image logits.

    Args:
        temperature (float, optional): Softmax temperature. Default is 0.07.
            (If you use a learnable logit_scale in your model, pass it to
            forward(...) and this temperature will be ignored.)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__(temperature=temperature)

    def forward(
        self,
        feats_i: torch.Tensor,
        feats_j: torch.Tensor,
        logit_scale: Optional[torch.Tensor | float] = None,
    ) -> torch.Tensor:
        # for CLIP, targets are always the diagonal
        targets = torch.arange(feats_i.size(0), device=feats_i.device)

        # calculate loss in both directions
        loss_i = self._compute(
            anchors=feats_i,
            candidates=feats_j,
            targets=targets,
            logit_scale=logit_scale,
        )
        loss_j = self._compute(
            anchors=feats_j,
            candidates=feats_i,
            targets=targets,
            logit_scale=logit_scale,
        )

        return 0.5 * (loss_i + loss_j)
