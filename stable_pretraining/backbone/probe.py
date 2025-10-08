import torch


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
        if x.dim() == 2:
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
