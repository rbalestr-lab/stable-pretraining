"""Forward functions for self-supervised learning methods.

This module provides pre-defined forward functions for various SSL methods
that can be used with the Module class. These functions define the training
logic for each method and can be specified in YAML configs or Python code.

Example:
    Using in a YAML config::

        module:
          _target_: stable_pretraining.Module
          forward: stable_pretraining.forward.simclr_forward
          backbone: ...
          projector: ...

    Using in Python code::

        from stable_pretraining import Module
        from stable_pretraining.forward import simclr_forward

        module = Module(forward=simclr_forward, backbone=backbone, projector=projector)
"""

import torch
import stable_pretraining as spt


def simclr_forward(self, batch, stage):
    """Forward function for SimCLR (Simple Contrastive Learning of Representations).

    SimCLR learns representations by maximizing agreement between differently
    augmented views of the same image via a contrastive loss in the latent space.

    Args:
        self: Module instance (automatically bound) with required attributes:
            - backbone: Feature extraction network
            - projector: Projection head mapping features to latent space
            - simclr_loss: NT-Xent contrastive loss function
        batch: Input batch dictionary containing:
            - 'image': Tensor of augmented images [N*views, C, H, W]
            - 'sample_idx': Indices to identify views of same image
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from backbone
            - 'loss': NT-Xent contrastive loss (during training only)

    Note:
        Introduced in the SimCLR paper :cite:`chen2020simple`.
    """
    out = {}
    out["embedding"] = self.backbone(batch["image"])

    if self.training:
        proj = self.projector(out["embedding"])
        views = spt.data.fold_views(proj, batch["sample_idx"])
        out["loss"] = self.simclr_loss(views[0], views[1])

    return out


def byol_forward(self, batch, stage):
    """Forward function for BYOL (Bootstrap Your Own Latent).

    BYOL learns representations without negative pairs by using a momentum-based
    target network and predicting target projections from online projections.

    Args:
        self: Module instance (automatically bound) with required attributes:
            - backbone: Online network feature extractor
            - projector: Online network projection head
            - predictor: Online network predictor
            - target_backbone: Target network backbone (momentum encoder)
            - target_projector: Target network projection head
        batch: Input batch dictionary containing:
            - 'image': Tensor of augmented images [N*views, C, H, W]
            - 'sample_idx': Indices to identify views of same image
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from online backbone
            - 'loss': MSE loss between predictions and targets (during training)

    Note:
        Introduced in the BYOL paper :cite:`grill2020bootstrap`.
    """
    out = {}
    out["embedding"] = self.backbone(batch["image"])

    if self.training:
        online_proj = self.projector(out["embedding"])
        online_pred = self.predictor(online_proj)

        with torch.no_grad():
            target_embedding = self.target_backbone(batch["image"])
            target_proj = self.target_projector(target_embedding)

        online_views = spt.data.fold_views(online_pred, batch["sample_idx"])
        target_views = spt.data.fold_views(target_proj, batch["sample_idx"])

        loss1 = torch.nn.functional.mse_loss(online_views[0], target_views[1].detach())
        loss2 = torch.nn.functional.mse_loss(online_views[1], target_views[0].detach())
        out["loss"] = (loss1 + loss2) / 2

    return out


def vicreg_forward(self, batch, stage):
    """Forward function for VICReg (Variance-Invariance-Covariance Regularization).

    VICReg learns representations using three criteria: variance (maintaining
    information), invariance (to augmentations), and covariance (decorrelating
    features).

    Args:
        self: Module instance (automatically bound) with required attributes:
            - backbone: Feature extraction network
            - projector: Projection head for embedding transformation
            - vicreg_loss: VICReg loss with variance, invariance, covariance terms
        batch: Input batch dictionary containing:
            - 'image': Tensor of augmented images [N*views, C, H, W]
            - 'sample_idx': Indices to identify views of same image
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from backbone
            - 'loss': Combined VICReg loss (during training only)

    Note:
        Introduced in the VICReg paper :cite:`bardes2022vicreg`.
    """
    out = {}
    out["embedding"] = self.backbone(batch["image"])

    if self.training:
        proj = self.projector(out["embedding"])
        views = spt.data.fold_views(proj, batch["sample_idx"])
        out["loss"] = self.vicreg_loss(views[0], views[1])

    return out


def barlow_twins_forward(self, batch, stage):
    """Forward function for Barlow Twins.

    Barlow Twins learns representations by making the cross-correlation matrix
    between embeddings of augmented views as close to the identity matrix as
    possible, reducing redundancy while maintaining invariance.

    Args:
        self: Module instance (automatically bound) with required attributes:
            - backbone: Feature extraction network
            - projector: Projection head (typically with BN and high dimension)
            - barlow_loss: Barlow Twins loss function
        batch: Input batch dictionary containing:
            - 'image': Tensor of augmented images [N*views, C, H, W]
            - 'sample_idx': Indices to identify views of same image
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from backbone
            - 'loss': Barlow Twins loss (during training only)

    Note:
        Introduced in the Barlow Twins paper :cite:`zbontar2021barlow`.
    """
    out = {}
    out["embedding"] = self.backbone(batch["image"])

    if self.training:
        proj = self.projector(out["embedding"])
        views = spt.data.fold_views(proj, batch["sample_idx"])
        out["loss"] = self.barlow_loss(views[0], views[1])

    return out


def supervised_forward(self, batch, stage):
    """Forward function for standard supervised training.

    This function implements traditional supervised learning with labels,
    useful for baseline comparisons and fine-tuning pre-trained models.

    Args:
        self: Module instance (automatically bound) with required attributes:
            - backbone: Feature extraction network
            - classifier: Classification head (e.g., Linear layer)
        batch: Input batch dictionary containing:
            - 'image': Tensor of images [N, C, H, W]
            - 'label': Ground truth labels [N] (optional, for loss computation)
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from backbone
            - 'logits': Classification predictions
            - 'loss': Cross-entropy loss (if labels provided)

    Note:
        Unlike SSL methods, this function uses actual labels for training
        and is primarily used for evaluation or supervised baselines.
    """
    out = {}
    out["embedding"] = self.backbone(batch["image"])
    out["logits"] = self.classifier(out["embedding"])

    if "label" in batch:
        out["loss"] = torch.nn.functional.cross_entropy(out["logits"], batch["label"])

    return out
