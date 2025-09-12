"""SSL losses."""

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


class NTXEntLoss(torch.nn.Module):
    """Normalized temperature-scaled cross entropy loss.

    Introduced in the SimCLR paper :cite:`chen2020simple`.
    Also used in MoCo :cite:`he2020momentum`.

    Args:
        temperature (float, optional): The temperature scaling factor.
            Default is 0.5.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """Compute the NT-Xent loss.

        Args:
            z_i (torch.Tensor): Latent representation of the first augmented view of the batch.
            z_j (torch.Tensor): Latent representation of the second augmented view of the batch.

        Returns:
            float: The computed contrastive loss.
        """
        z_i = torch.cat(all_gather(z_i), 0)
        z_j = torch.cat(all_gather(z_j), 0)

        z = torch.cat([z_i, z_j], 0)
        N = z.size(0)

        features = F.normalize(z, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature

        sim_i_j = torch.diag(sim, N // 2)
        sim_j_i = torch.diag(sim, -N // 2)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0)

        mask = torch.eye(N, dtype=bool).to(z_i.device)
        negative_samples = sim[~mask].reshape(N, -1)

        attraction = -positive_samples.mean()
        repulsion = torch.logsumexp(negative_samples, dim=1).mean()

        return attraction + repulsion


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


class InfoNCELoss(torch.nn.Module):
    """Contrastive image-text loss used in CLIP :cite:`radford2021learningtransferablevisualmodels`.

    Computes symmetric cross-entropy over image-text and text-image logits.

    Args:
        temperature (float, optional): Softmax temperature. Default is 0.07.
            (If you use a learnable logit_scale in your model, pass it to
            forward(...) and this temperature will be ignored.)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_feats: torch.Tensor,  # [B, D]
        text_feats: torch.Tensor,  # [B, D]
        logit_scale: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Compute CLIP loss.

        Args:
            image_feats: Image embeddings for the batch.
            text_feats: Text embeddings for the batch (paired with images).
            logit_scale: Optional scalar (float or 0-dim tensor). If provided,
                logits = logit_scale * (img @ txt^T). If None, uses
                1/temperature.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Gather across devices (DDP) to form global batch
        img = torch.cat(all_gather(F.normalize(image_feats, dim=-1)), dim=0)
        txt = torch.cat(all_gather(F.normalize(text_feats, dim=-1)), dim=0)

        # Compute pairwise logits
        if logit_scale is None:
            scale = 1.0 / self.temperature
        else:
            # Accept float or 0-dim tensor; detach is not applied so scale can be learnable
            scale = (
                float(logit_scale) if not torch.is_tensor(logit_scale) else logit_scale
            )
        logits_per_image = scale * (img @ txt.T)  # [N, N]
        logits_per_text = logits_per_image.T  # [N, N]

        # Ground-truth matches are on the diagonal after all_gather
        N = logits_per_image.size(0)
        device = logits_per_image.device
        targets = torch.arange(N, device=device)

        loss_i = F.cross_entropy(logits_per_image, targets)
        loss_t = F.cross_entropy(logits_per_text, targets)
        return 0.5 * (loss_i + loss_t)
