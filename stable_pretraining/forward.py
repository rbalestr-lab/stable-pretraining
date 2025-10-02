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


def _get_views_list(batch):
    """Convert multi-view batch to list of views, whether it's a list or dict."""
    if isinstance(batch, dict) and "image" not in batch:
        # Dict of named views - convert to list
        return list(batch.values())
    elif isinstance(batch, list):
        # Already a list
        return batch
    else:
        # Single view
        return None


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

    views = _get_views_list(batch)
    if views is not None:
        # Multi-view training - SimCLR requires exactly 2 views
        if len(views) != 2:
            raise ValueError(
                f"SimCLR requires exactly 2 views, got {len(views)}. "
                "For other configurations, please implement a custom forward function."
            )

        embeddings = [self.backbone(view["image"]) for view in views]
        out["embedding"] = torch.cat(embeddings, dim=0)

        # Concatenate labels for callbacks (probes need this)
        if "label" in views[0]:
            out["label"] = torch.cat([view["label"] for view in views], dim=0)

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

    views = _get_views_list(batch)
    if views is not None:
        # Multi-view training - BYOL requires exactly 2 views
        if len(views) != 2:
            raise ValueError(
                f"BYOL requires exactly 2 views, got {len(views)}. "
                "For other configurations, please implement a custom forward function."
            )

        images = [view["image"] for view in views]

        # Concatenate labels for callbacks
        if "label" in views[0]:
            out["label"] = torch.cat([view["label"] for view in views], dim=0)

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

    views = _get_views_list(batch)
    if views is not None:
        # Multi-view training - VICReg requires exactly 2 views
        if len(views) != 2:
            raise ValueError(
                f"VICReg requires exactly 2 views, got {len(views)}. "
                "For other configurations, please implement a custom forward function."
            )

        embeddings = [self.backbone(view["image"]) for view in views]
        out["embedding"] = torch.cat(embeddings, dim=0)

        # Concatenate labels for callbacks
        if "label" in views[0]:
            out["label"] = torch.cat([view["label"] for view in views], dim=0)

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

    views = _get_views_list(batch)
    if views is not None:
        # Multi-view training - Barlow Twins requires exactly 2 views
        if len(views) != 2:
            raise ValueError(
                f"Barlow Twins requires exactly 2 views, got {len(views)}. "
                "For other configurations, please implement a custom forward function."
            )

        embeddings = [self.backbone(view["image"]) for view in views]
        out["embedding"] = torch.cat(embeddings, dim=0)

        # Concatenate labels for callbacks
        if "label" in views[0]:
            out["label"] = torch.cat([view["label"] for view in views], dim=0)

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

    views = _get_views_list(batch)
    if views is not None:
        # Multi-view training - NNCLR requires exactly 2 views
        if len(views) != 2:
            raise ValueError(
                f"NNCLR requires exactly 2 views, got {len(views)}. "
                "For other configurations, please implement a custom forward function."
            )

        embeddings = [self.backbone(view["image"]) for view in views]
        out["embedding"] = torch.cat(embeddings, dim=0)

        # Concatenate labels for callbacks
        if "label" in views[0]:
            out["label"] = torch.cat([view["label"] for view in views], dim=0)

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


def dino_forward(self, batch, stage):
    """Forward function for DINO (self-DIstillation with NO labels).

    DINO learns representations through self-distillation where a student network
    is trained to match the output of a teacher network (EMA of student) on
    different augmented views. Global views are processed by both networks while
    local views are only processed by the student.

    Args:
        self: Module instance (automatically bound) with required attributes:
            - backbone: TeacherStudentWrapper for feature extraction
            - projector: TeacherStudentWrapper for projection head
            - dino_loss: DINOv1Loss instance (required, pass spt.losses.DINOv1Loss())
            - warmup_temperature_teacher (float): Starting teacher temperature
            - temperature_teacher (float): Final teacher temperature
            - warmup_epochs_temperature_teacher (int): Epochs to warm up temperature
        batch: Either a list of view dicts (from MultiViewTransform) or
            a single dict (for validation/single-view).
            For multi-crop: First 2 views should be global crops, rest are local crops
        stage: Training stage ('train', 'val', or 'test')

    Returns:
        Dictionary containing:
            - 'embedding': Feature representations from teacher backbone
            - 'loss': DINO distillation loss (during training only)
            - 'label': Labels if present (for probes/callbacks)

    Note:
        Introduced in the DINO paper :cite:`caron2021emerging`.
        Requires TeacherStudentWrapper for both backbone and projector,
        and assumes first 2 views in batch are global views.
    """
    out = {}

    # Check if batch is dict of named views
    if isinstance(batch, dict) and "image" not in batch:
        # Dict of named views - separate by "global" or "local" in key
        global_views = []
        local_views = []

        for key, view in batch.items():
            if "global" in key:
                global_views.append(view)
            elif "local" in key:
                local_views.append(view)

        n_global = len(global_views)
        n_local = len(local_views)
        all_views = global_views + local_views

    elif isinstance(batch, list):
        # List of views - assume first 2 are global
        all_views = batch
        n_global = min(2, len(all_views))
        n_local = len(all_views) - n_global
        global_views = all_views[:n_global]
        local_views = all_views[n_global:] if n_local > 0 else []

    else:
        # Single view validation
        images = batch["image"]
        if "label" in batch:
            out["label"] = batch["label"]

        with torch.no_grad():
            teacher_features = self.backbone.forward_teacher(images)
        out["embedding"] = teacher_features.detach()
        return out

    # Multi-view processing
    batch_size = all_views[0]["image"].shape[0]

    # Concatenate labels for callbacks - only from global views for probes
    if "label" in all_views[0]:
        # For training, only use labels from global views since we only return global embeddings
        if self.training:
            out["label"] = torch.cat([view["label"] for view in global_views], dim=0)
        else:
            # For validation, use all labels since we return all embeddings
            out["label"] = torch.cat([view["label"] for view in all_views], dim=0)

    if not self.training:
        # During validation, just process all views through teacher
        all_images = torch.cat([view["image"] for view in all_views], dim=0)
        with torch.no_grad():
            teacher_features = self.backbone.forward_teacher(all_images)
        out["embedding"] = teacher_features.detach()
        return out

    # Training: separate processing for global and local views
    global_images = torch.cat([view["image"] for view in global_views], dim=0)

    # Teacher processes only global views
    with torch.no_grad():
        teacher_features = self.backbone.forward_teacher(global_images)
        teacher_embeddings = self.projector.forward_teacher(teacher_features)
        teacher_embeddings = teacher_embeddings.view(n_global, batch_size, -1)

    # Student processes all views
    student_embeddings_list = []

    # Process global views through student
    student_features = self.backbone.forward_student(global_images)
    student_global_embeddings = self.projector.forward_student(student_features)
    student_global_embeddings = student_global_embeddings.view(n_global, batch_size, -1)
    student_embeddings_list.append(student_global_embeddings)

    # Process local views through student (if any)
    if n_local > 0:
        local_images = torch.cat([view["image"] for view in local_views], dim=0)
        student_features = self.backbone.forward_student(local_images)
        student_local_embeddings = self.projector.forward_student(student_features)
        student_local_embeddings = student_local_embeddings.view(
            n_local, batch_size, -1
        )
        student_embeddings_list.append(student_local_embeddings)

    # Concatenate student embeddings along the view dimension
    student_embeddings = torch.cat(
        student_embeddings_list, dim=0
    )  # [n_views, batch_size, dim]

    if not hasattr(self, "dino_loss"):
        raise ValueError(
            "dino_forward requires 'dino_loss' to be provided (e.g., spt.losses.DINOv1Loss()). "
            "Pass it when constructing the Module: Module(..., dino_loss=spt.losses.DINOv1Loss(), ...)"
        )

    if not hasattr(self, "projector"):
        raise ValueError(
            "dino_forward requires 'projector' to be provided. "
            "This should be a TeacherStudentWrapper containing the projector (MLP + normalize + prototypes). "
            "Pass it when constructing the Module: Module(..., projector=wrapped_projector, ...)"
        )

    # Apply projector to get logits (separate teacher and student projectors)
    teacher_logits = self.projector.forward_teacher(teacher_embeddings)
    student_logits = self.projector.forward_student(student_embeddings)

    # Temperature scheduling for teacher
    if (
        hasattr(self, "warmup_epochs_temperature_teacher")
        and hasattr(self, "warmup_temperature_teacher")
        and hasattr(self, "temperature_teacher")
    ):
        if self.current_epoch < self.warmup_epochs_temperature_teacher:
            # Linear warmup from warmup_temperature_teacher to temperature_teacher
            progress = self.current_epoch / self.warmup_epochs_temperature_teacher
            temperature_teacher = self.warmup_temperature_teacher + progress * (
                self.temperature_teacher - self.warmup_temperature_teacher
            )
        else:
            temperature_teacher = self.temperature_teacher
    else:
        # Default temperature if attributes not set
        temperature_teacher = getattr(self, "temperature_teacher", 0.07)

    # Process teacher logits with classical centering (DINOv1 approach)
    teacher_probs = self.dino_loss.softmax_center_teacher(
        teacher_logits, teacher_temp=temperature_teacher
    )

    # Compute DINO loss
    loss = self.dino_loss(student_logits, teacher_probs)

    # Queue async center update (will be applied next iteration)
    self.dino_loss.update_center(teacher_logits)

    out["embedding"] = teacher_features.detach()
    out["loss"] = loss

    return out
