import torch
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy


class MultiHeadAttentiveProbe(torch.nn.Module):
    """A multi-head attentive probe for sequence representations.

    This module applies multiple attention heads to a sequence of embeddings,
    pools the sequence into a fixed-size representation per head, concatenates
    the results, and projects to a set of output classes.

    Args:
        embedding_dim (int): Dimensionality of the input embeddings.
        num_classes (int): Number of output classes.
        num_heads (int, optional): Number of attention heads. Default is 4.

    Attributes:
        ln (torch.nn.LayerNorm): Layer normalization applied to the input.
        attn_vectors (torch.nn.Parameter): Learnable attention vectors for each head, shape (num_heads, embedding_dim).
        fc (torch.nn.Linear): Final linear layer mapping concatenated head outputs to class logits.
    Forward Args:
        x (torch.Tensor): Input tensor of shape (N, T, D), where
            N = batch size,
            T = sequence length,
            D = embedding_dim.

    Returns:
        torch.Tensor: Output logits of shape (N, num_classes).

    Example:
        >>> probe = MultiHeadAttentiveProbe(
        ...     embedding_dim=128, num_classes=10, num_heads=4
        ... )
        >>> x = torch.randn(32, 20, 128)  # batch of 32, sequence length 20
        >>> logits = probe(x)  # shape: (32, 10)
    """

    def __init__(self, embedding_dim: int, num_classes: int, num_heads: int = 4):
        super().__init__()
        self.ln = torch.nn.LayerNorm(embedding_dim)
        self.attn_vectors = torch.nn.Parameter(torch.randn(num_heads, embedding_dim))
        self.fc = torch.nn.Linear(embedding_dim * num_heads, num_classes)

    def forward(self, x: torch.Tensor):
        # x: (N, T, D)
        x = self.ln(x)
        # Compute attention for each head: (N, num_heads, T)
        attn_scores = torch.einsum("ntd,hd->nht", x, self.attn_vectors)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (N, num_heads, T)
        # Weighted sum for each head: (N, num_heads, D)
        pooled = torch.einsum("ntd,nht->nhd", x, attn_weights)
        pooled = pooled.reshape(x.size(0), -1)  # (N, num_heads * D)
        out = self.fc(pooled)  # (N, num_classes)
        return out


class LinearProbe(torch.nn.Module):
    """Linear using either CLS token or mean pooling with configurable normalization layer.

    Args:
        embedding_dim (int): Dimensionality of the input embeddings.
        num_classes (int): Number of output classes.
        pooling (str): Pooling strategy, either 'cls' or 'mean'.
        norm_layer (callable or None): Normalization layer class (e.g., torch.nn.LayerNorm, torch.nn.BatchNorm1d),
            or None for no normalization. Should accept a single argument: normalized_shape or num_features.

    Attributes:
        norm (nn.Module or None): Instantiated normalization layer, or None.
        fc (nn.Linear): Linear layer mapping pooled representation to class logits.
    Forward Args:
        x (torch.Tensor): Input tensor of shape (N, T, D) or (N, D).
            If 3D, pooling and normalization are applied.
            If 2D, input is used directly (no pooling or normalization).

    Returns:
        torch.Tensor: Output logits of shape (N, num_classes).

    Example:
        >>> probe = LinearProbe(
        ...     embedding_dim=128,
        ...     num_classes=10,
        ...     pooling="mean",
        ...     norm_layer=torch.nn.LayerNorm,
        ... )
        >>> x = torch.randn(32, 20, 128)
        >>> logits = probe(x)  # shape: (32, 10)
        >>> x2 = torch.randn(32, 128)
        >>> logits2 = probe(x2)  # shape: (32, 10)
    """

    def __init__(self, embedding_dim, num_classes, pooling="cls", norm_layer=None):
        super().__init__()
        assert pooling in (
            "cls",
            "mean",
            None,
        ), "pooling must be 'cls' or 'mean' or None"
        self.pooling = pooling
        self.norm = norm_layer(embedding_dim) if norm_layer is not None else None
        self.fc = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x: (N, T, D) or (N, D)
        if x.ndim == 2:
            # (N, D): no pooling or normalization
            pooled = x
        elif self.pooling == "cls":
            pooled = x[:, 0, :]  # (N, D)
        elif self.pooling == "mean":  # 'mean'
            pooled = x.mean(dim=1)  # (N, D)
        else:
            pooled = x.flatten(1)
        out = self.fc(self.norm(pooled))  # (N, num_classes)
        return out


class AutoLinearClassifier(torch.nn.Module):
    """Linear using either CLS token or mean pooling with configurable normalization layer.

    Args:
        embedding_dim (int): Dimensionality of the input embeddings.
        num_classes (int): Number of output classes.
        pooling (str): Pooling strategy, either 'cls' or 'mean'.
        norm_layer (callable or None): Normalization layer class (e.g., torch.nn.LayerNorm, torch.nn.BatchNorm1d),
            or None for no normalization. Should accept a single argument: normalized_shape or num_features.

    Attributes:
        norm (nn.Module or None): Instantiated normalization layer, or None.
        fc (nn.Linear): Linear layer mapping pooled representation to class logits.
    Forward Args:
        x (torch.Tensor): Input tensor of shape (N, T, D) or (N, D).
            If 3D, pooling and normalization are applied.
            If 2D, input is used directly (no pooling or normalization).

    Returns:
        torch.Tensor: Output logits of shape (N, num_classes).

    Example:
        >>> probe = LinearProbe(
        ...     embedding_dim=128,
        ...     num_classes=10,
        ...     pooling="mean",
        ...     norm_layer=torch.nn.LayerNorm,
        ... )
        >>> x = torch.randn(32, 20, 128)
        >>> logits = probe(x)  # shape: (32, 10)
        >>> x2 = torch.randn(32, 128)
        >>> logits2 = probe(x2)  # shape: (32, 10)
    """

    def __init__(self, name, module, embedding_dim, num_classes, pooling=None):
        super().__init__()
        assert pooling in (
            "cls",
            "mean",
            None,
        ), "pooling must be 'cls' or 'mean' or None"
        self.name = name
        self.pooling = pooling
        self.num_classes = num_classes
        self.bn_norm = torch.nn.BatchNorm1d(embedding_dim)
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)
        self.fc = torch.nn.ModuleDict(
            {
                f"{name}_bn": torch.nn.Linear(embedding_dim, num_classes * 3),
                f"{name}_bn_drop": torch.nn.Sequential(
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(embedding_dim, num_classes * 3),
                ),
                f"{name}_norm": torch.nn.Linear(embedding_dim, num_classes * 3),
                f"{name}_norm_drop": torch.nn.Sequential(
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(embedding_dim, num_classes * 3),
                ),
            }
        )
        self.loss = torch.nn.ModuleList(
            [
                torch.nn.CrossEntropyLoss(label_smoothing=0),
                torch.nn.CrossEntropyLoss(label_smoothing=1 / num_classes),
                torch.nn.CrossEntropyLoss(label_smoothing=10 / num_classes),
            ]
        )
        self.metrics = torchmetrics.MetricCollection(
            {
                f"val/{sub}_{i}": MulticlassAccuracy(num_classes)
                for i in range(3)
                for sub in self.fc.keys()
            }
        )
        # self._module = module
        # self.log_dict = module.log_dict

    def forward(self, x, y=None, pl_module=None):
        # x: (N, T, D) or (N, D)
        if x.ndim == 2:
            # (N, D): no pooling or normalization
            pooled = x
        elif self.pooling == "cls":
            pooled = x[:, 0, :]  # (N, D)
        elif self.pooling == "mean":  # 'mean'
            pooled = x.mean(dim=1)  # (N, D)
        else:
            pooled = x.flatten(1)
        bn_pooled = self.bn_norm(pooled)
        norm_pooled = self.layer_norm(pooled)
        inputs = {
            f"{self.name}_bn": bn_pooled,
            f"{self.name}_bn_drop": bn_pooled,
            f"{self.name}_norm": norm_pooled,
            f"{self.name}_norm_drop": norm_pooled,
        }
        loss = {}
        for k, v in self.fc.items():
            yhats = v(inputs[k]).reshape(x.size(0), 3, self.num_classes).unbind(1)
            for i, yhat in enumerate(yhats):
                if not self.training:
                    self.metrics[f"val/{k}_{i}"].update(yhat, y)
                    continue
                loss[f"train/{self.name}_{k}"] = self.loss[i](yhat, y)
        if not self.training:
            pl_module.log_dict(
                self.metrics,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        else:
            pl_module.log_dict(loss, on_step=True, on_epoch=False, rank_zero_only=True)
        return sum(loss.values())
