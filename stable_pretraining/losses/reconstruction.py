"""Reconstruction-based SSL losses.

This module contains reconstruction-based self-supervised learning losses
such as Masked Autoencoder (MAE).
"""


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
