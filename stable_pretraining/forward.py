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
from .callbacks.queue import find_or_create_queue_callback, OnlineQueue


def supervised_forward(self, batch, stage):
    """Forward function for standard supervised training.

    This function implements traditional supervised learning with labels,
    useful for baseline comparisons and fine-tuning pre-trained models.

    Args:
        self: Module instance (automatically bound) with required attributes:
            - backbone: Feature extraction network
            - classifier: Classification head (e.g., Linear layer)
            - supervised_loss: Loss function for supervised learning
        batch: Input batch dictionary containing:
            - 'image': Tensor of images [N, C, H, W]
            - 'label': Ground truth labels [N] (optional, for loss computation)
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from backbone
            - 'logits': Classification predictions
            - 'loss': Supervised loss (if labels provided)

    Note:
        Unlike SSL methods, this function uses actual labels for training
        and is primarily used for evaluation or supervised baselines.
    """
    out = {}
    out["embedding"] = self.backbone(batch["image"])
    out["logits"] = self.classifier(out["embedding"])

    if "label" in batch:
        if not hasattr(self, "supervised_loss"):
            raise ValueError(
                "supervised_forward requires 'supervised_loss' to be provided (e.g., nn.CrossEntropyLoss()). "
                "Pass it when constructing the Module: Module(..., supervised_loss=nn.CrossEntropyLoss(), ...)"
            )
        out["loss"] = self.supervised_loss(out["logits"], batch["label"])

    return out


def simclr_forward(self, batch, stage):
    """Forward function for SimCLR (Simple Contrastive Learning of Representations).

    SimCLR learns representations by maximizing agreement between differently
    augmented views of the same image via a contrastive loss in the latent space.

    Args:
        self: Module instance (automatically bound) with required attributes:
            - backbone: Feature extraction network
            - projector: Projection head mapping features to latent space
            - simclr_loss: NT-Xent contrastive loss function
        batch: Either a list of view dicts (from MultiViewTransform) or
            a single dict (for validation/single-view)
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from backbone
            - 'loss': NT-Xent contrastive loss (during training only)
            - 'label': Labels if present (for probes/callbacks)

    Note:
        Introduced in the SimCLR paper :cite:`chen2020simple`.
    """
    out = {}

    if isinstance(batch, list):
        # Multi-view training - batch is a list of view dicts
        embeddings = [self.backbone(view["image"]) for view in batch]
        out["embedding"] = torch.cat(embeddings, dim=0)

        # Concatenate labels for callbacks (probes need this)
        if "label" in batch[0]:
            out["label"] = torch.cat([view["label"] for view in batch], dim=0)

        if self.training:
            projections = [self.projector(emb) for emb in embeddings]
            out["loss"] = self.simclr_loss(projections[0], projections[1])
    else:
        # Single-view validation
        out["embedding"] = self.backbone(batch["image"])
        if "label" in batch:
            out["label"] = batch["label"]

    return out


def byol_forward(self, batch, stage):
    """Forward function for BYOL (Bootstrap Your Own Latent).

    BYOL learns representations without negative pairs by using a momentum-based
    target network and predicting target projections from online projections.

    Args:
        self: Module instance with required attributes:
            - backbone: TeacherStudentWrapper for feature extraction
            - projector: TeacherStudentWrapper for projection head
            - predictor: Online network predictor
            - byol_loss: BYOL loss function (optional, uses MSE if not provided)
        batch: Either a list of view dicts (from MultiViewTransform) or
            a single dict (for validation/single-view)
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from teacher backbone (EMA target)
            - 'loss': BYOL loss between predictions and targets (during training)
            - 'label': Labels if present (for probes/callbacks)

    Note:
        Introduced in the BYOL paper :cite:`grill2020bootstrap`.
    """
    out = {}

    if isinstance(batch, list):
        # Multi-view training - batch is a list of view dicts
        images = [view["image"] for view in batch]

        # Concatenate labels for callbacks
        if "label" in batch[0]:
            out["label"] = torch.cat([view["label"] for view in batch], dim=0)

        # Get online embeddings for both views
        online_features = [self.backbone.forward_student(img) for img in images]

        # Return early if not training
        if not self.training:
            with torch.no_grad():
                all_images = torch.cat(images, dim=0)
                target_only_features = self.backbone.forward_teacher(all_images)
            return {"embedding": target_only_features.detach(), **out}

        # Process online network
        online_proj = [self.projector.forward_student(feat) for feat in online_features]
        online_pred = [self.predictor(proj) for proj in online_proj]

        # Process target network
        with torch.no_grad():
            target_features = [self.backbone.forward_teacher(img) for img in images]
            target_proj = [
                self.projector.forward_teacher(feat) for feat in target_features
            ]

        if not hasattr(self, "byol_loss"):
            raise ValueError(
                "byol_forward requires 'byol_loss' to be provided (e.g., spt.losses.BYOLLoss()). "
                "Pass it when constructing the Module: Module(..., byol_loss=spt.losses.BYOLLoss(), ...)"
            )

        # Compute loss between view pairs
        loss = (
            self.byol_loss(online_pred[0], target_proj[1])
            + self.byol_loss(online_pred[1], target_proj[0])
        ) / 2

        out["embedding"] = torch.cat(target_features, dim=0).detach()
        out["loss"] = loss

    else:
        # Single-view validation
        images = batch["image"]

        if "label" in batch:
            out["label"] = batch["label"]

        # Just return embeddings for validation
        with torch.no_grad():
            target_only_features = self.backbone.forward_teacher(images)
        out["embedding"] = target_only_features.detach()

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
        batch: Either a list of view dicts (from MultiViewTransform) or
            a single dict (for validation/single-view)
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from backbone
            - 'loss': Combined VICReg loss (during training only)
            - 'label': Labels if present (for probes/callbacks)

    Note:
        Introduced in the VICReg paper :cite:`bardes2022vicreg`.
    """
    out = {}

    if isinstance(batch, list):
        # Multi-view training - batch is a list of view dicts
        embeddings = [self.backbone(view["image"]) for view in batch]
        out["embedding"] = torch.cat(embeddings, dim=0)

        # Concatenate labels for callbacks
        if "label" in batch[0]:
            out["label"] = torch.cat([view["label"] for view in batch], dim=0)

        if self.training:
            projections = [self.projector(emb) for emb in embeddings]
            out["loss"] = self.vicreg_loss(projections[0], projections[1])
    else:
        # Single-view validation
        out["embedding"] = self.backbone(batch["image"])
        if "label" in batch:
            out["label"] = batch["label"]

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
        batch: Either a list of view dicts (from MultiViewTransform) or
            a single dict (for validation/single-view)
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from backbone
            - 'loss': Barlow Twins loss (during training only)
            - 'label': Labels if present (for probes/callbacks)

    Note:
        Introduced in the Barlow Twins paper :cite:`zbontar2021barlow`.
    """
    out = {}

    if isinstance(batch, list):
        # Multi-view training - batch is a list of view dicts
        embeddings = [self.backbone(view["image"]) for view in batch]
        out["embedding"] = torch.cat(embeddings, dim=0)

        # Concatenate labels for callbacks
        if "label" in batch[0]:
            out["label"] = torch.cat([view["label"] for view in batch], dim=0)

        if self.training:
            projections = [self.projector(emb) for emb in embeddings]
            out["loss"] = self.barlow_loss(projections[0], projections[1])
    else:
        # Single-view validation
        out["embedding"] = self.backbone(batch["image"])
        if "label" in batch:
            out["label"] = batch["label"]

    return out


def _find_nearest_neighbors(query, support_set):
    """Find the nearest neighbor for each query embedding in the support set."""
    query_norm = torch.nn.functional.normalize(query, dim=1)
    support_norm = torch.nn.functional.normalize(support_set, dim=1)
    similarity = torch.mm(query_norm, support_norm.t())
    _, indices = similarity.max(dim=1)
    return support_set[indices]


def nnclr_forward(self, batch, stage):
    """Forward function for NNCLR (Nearest-Neighbor Contrastive Learning).

    NNCLR learns representations by using the nearest neighbor of an augmented
    view from a support set of past embeddings as a positive pair. This encourages
    the model to learn representations that are similar for semantically similar
    instances, not just for different augmentations of the same instance.

    Args:
        self: Module instance (automatically bound) with required attributes:
            - backbone: Feature extraction network
            - projector: Projection head for embedding transformation
            - predictor: Prediction head used for the online view
            - nnclr_loss: NTXent contrastive loss function
        batch: Either a list of view dicts (from MultiViewTransform) or
            a single dict (for validation/single-view)
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from backbone
            - 'loss': NTXent contrastive loss (during training only)
            - 'nnclr_support_set': Projections to be added to the support set queue
            - 'label': Labels if present (for probes/callbacks)

    Note:
        Introduced in the NNCLR paper :cite:`dwibedi2021little`.
    """
    out = {}

    if isinstance(batch, list):
        # Multi-view training - batch is a list of view dicts
        embeddings = [self.backbone(view["image"]) for view in batch]
        out["embedding"] = torch.cat(embeddings, dim=0)

        # Concatenate labels for callbacks
        if "label" in batch[0]:
            out["label"] = torch.cat([view["label"] for view in batch], dim=0)

        if self.training:
            if not hasattr(self, "_nnclr_queue_callback"):
                self._nnclr_queue_callback = find_or_create_queue_callback(
                    self.trainer,
                    key="nnclr_support_set",
                    queue_length=self.hparams.support_set_size,
                    dim=self.hparams.projection_dim,
                )
            queue_callback = self._nnclr_queue_callback

            projections = [self.projector(emb) for emb in embeddings]
            proj_q, proj_k = projections[0], projections[1]

            support_set = OnlineQueue._shared_queues.get(queue_callback.key).get()

            if support_set is not None and len(support_set) > 0:
                pred_q = self.predictor(proj_q)
                pred_k = self.predictor(proj_k)

                nn_k = _find_nearest_neighbors(proj_k, support_set).detach()
                nn_q = _find_nearest_neighbors(proj_q, support_set).detach()

                loss_a = self.nnclr_loss(pred_q, nn_k)
                loss_b = self.nnclr_loss(pred_k, nn_q)
                out["loss"] = (loss_a + loss_b) / 2.0
            else:
                # Fallback to SimCLR style loss if queue is empty
                out["loss"] = self.nnclr_loss(proj_q, proj_k)

            # The key here must match the `key` argument of the `OnlineQueue` callback
            out["nnclr_support_set"] = torch.cat(projections)

    else:
        # Single-view validation
        out["embedding"] = self.backbone(batch["image"])
        if "label" in batch:
            out["label"] = batch["label"]

    return out
