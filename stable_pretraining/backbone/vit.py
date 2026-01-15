import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal, Union, Type
import timm
from timm.models.vision_transformer import Block
from timm.layers import trunc_normal_, Mlp
from .patch_masking import PatchMasking
from dataclasses import dataclass
from .pos_embed import (
    get_sincos_pos_embed,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    get_timestep_embed,
    interpolate_pos_embed,
)

__all__ = [
    "MAEDecoder",
    "TransformerPredictor",
    "AdaLNDecoderBlock",
    "MaskedEncoder",
    "MaskedEncoderOutput",
    "AdaLNCrossAttentionBlock",
    "CrossAttentionBlock",
    "CrossAttentionDecoder",
]


class CrossAttentionBlock(nn.Module):
    """Pure cross-attention block (no self-attention among queries).

    Faster than full decoder block when queries don't need to interact.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm1_ctx = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.cross_attn(self.norm1(x), self.norm1_ctx(context), context)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class AdaLNCrossAttentionBlock(nn.Module):
    """Pure cross-attention with AdaLN (no self-attention)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1_q = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm1_kv = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

        # AdaLN: (γ1, β1, α1, γ2, β2, α2)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaln[1].weight)
        nn.init.zeros_(self.adaln[1].bias)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        assert cond is not None
        γ1, β1, α1, γ2, β2, α2 = [
            m.unsqueeze(1) for m in self.adaln(cond).chunk(6, dim=-1)
        ]

        h = self.norm1_q(x) * (1 + γ1) + β1
        kv = self.norm1_kv(context)
        x = x + α1 * self.cross_attn(h, kv, kv, need_weights=False)[0]

        h = self.norm2(x) * (1 + γ2) + β2
        x = x + α2 * self.mlp(h)
        return x


@dataclass
class MaskedEncoderOutput:
    """Output from MaskedEncoder forward pass.

    :ivar encoded: Encoded token representations (B, num_prefix + N_visible, D)
    :ivar mask: Binary mask where 1 = masked, 0 = visible (B, N_patches)
    :ivar ids_keep: Indices of visible patches (B, N_visible)
    :ivar grid_size: Patch grid dimensions (height, width)
    """

    encoded: torch.Tensor
    mask: torch.Tensor
    ids_keep: torch.Tensor
    grid_size: Tuple[int, int]


class MaskedEncoder(nn.Module):
    """Vision Transformer encoder with optional masking support.

    Wraps a timm ViT model and adds flexible masking via :class:`PatchMasking`.
    Handles all ViT internals: patch embedding, positional embeddings, prefix
    tokens (CLS, registers), and transformer blocks.
    :param model_or_model_name: timm model name string or pre-instantiated nn.Module
    :param masking: PatchMasking instance. If None, no masking is applied.
    :param pretrained: Load pretrained weights (only when model_or_model_name is str)
    :param img_size: Override default image size
    :param patch_size: Override default patch size (will reinitialize patch_embed)
    :param dynamic_img_size: Enable dynamic image size support with pos_embed interpolation
    Example::
        from spt.backbone import PatchMasking, MaskedEncoder

        masking = PatchMasking(mask_ratio=0.75, block_size=4)
        encoder = MaskedEncoder(
            model_or_model_name="vit_base_patch16_224",
            masking=masking,
            pretrained=True,
        )
        images = torch.randn(4, 3, 224, 224)
        output = encoder(images)
        print(output.encoded.shape)  # (4, 1 + 49, 768) with 75% masking
        print(output.mask.shape)  # (4, 196)
        print(output.ids_keep.shape)  # (4, 49)
    """

    def __init__(
        self,
        model_or_model_name: Union[str, nn.Module] = "vit_base_patch16_224",
        masking: Optional[PatchMasking] = None,
        pretrained: bool = False,
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        dynamic_img_size: bool = False,
    ):
        super().__init__()
        self.dynamic_img_size = dynamic_img_size
        self.masking = masking
        # === Load or use provided encoder ===
        if isinstance(model_or_model_name, str):
            create_kwargs = {
                "pretrained": pretrained,
                "num_classes": 0,
                "dynamic_img_size": dynamic_img_size,
            }
            if img_size is not None:
                create_kwargs["img_size"] = img_size
            if patch_size is not None:
                create_kwargs["patch_size"] = patch_size
                if pretrained:
                    print(
                        f"Warning: Changing patch_size to {patch_size} will reinitialize "
                        f"patch_embed weights. Pretrained weights won't fully apply."
                    )
            self.vit = timm.create_model(model_or_model_name, **create_kwargs)
        else:
            self.vit = model_or_model_name
            if patch_size is not None:
                self._rebuild_patch_embed(patch_size, img_size)
            # Remove classification head if present
            if hasattr(self.vit, "head") and hasattr(self.vit.head, "in_features"):
                self.vit.head = nn.Identity()
        # === Cache encoder properties ===
        self.embed_dim = self.vit.embed_dim
        self.patch_embed = self.vit.patch_embed
        ps = self.patch_embed.patch_size
        self.patch_size_h, self.patch_size_w = (ps, ps) if isinstance(ps, int) else ps
        gs = self.patch_embed.grid_size
        self.default_grid_h, self.default_grid_w = (
            (gs, gs) if isinstance(gs, int) else gs
        )
        self.num_prefix_tokens = getattr(self.vit, "num_prefix_tokens", 1)
        self.has_class_token = getattr(self.vit, "has_class_token", True)
        self.num_reg_tokens = getattr(self.vit, "num_reg_tokens", 0)
        self.no_embed_class = getattr(self.vit, "no_embed_class", False)

    def _rebuild_patch_embed(
        self,
        patch_size: Union[int, Tuple[int, int]],
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> None:
        """Rebuild patch embedding with new patch size."""
        from timm.layers import PatchEmbed

        old = self.vit.patch_embed
        if img_size is None:
            og, op = old.grid_size, old.patch_size
            img_size = (
                (og[0] * op[0], og[1] * op[1]) if isinstance(og, tuple) else og * op
            )
        self.vit.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=old.proj.in_channels,
            embed_dim=old.proj.out_channels,
        )
        if old.num_patches != self.vit.patch_embed.num_patches:
            self._resize_pos_embed(self.vit.patch_embed.grid_size)

    def _resize_pos_embed(self, new_grid_size: Tuple[int, int]) -> None:
        """Resize positional embeddings to new grid size."""
        old_pos = self.vit.pos_embed
        num_prefix = self.num_prefix_tokens if not self.no_embed_class else 0
        src_patches = old_pos.shape[1] - num_prefix
        src_size = int(src_patches**0.5)
        new_pos = interpolate_pos_embed(
            old_pos, (src_size, src_size), new_grid_size, num_prefix
        )
        self.vit.pos_embed = nn.Parameter(new_pos)

    def _get_grid_size(self, images: torch.Tensor) -> Tuple[int, int]:
        """Compute patch grid size from image dimensions."""
        H, W = images.shape[-2:]
        return H // self.patch_size_h, W // self.patch_size_w

    def _get_pos_embed(
        self, grid_h: int, grid_w: int
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Get positional embeddings, interpolating if needed for dynamic size."""
        pos_embed = self.vit.pos_embed
        num_prefix = self.num_prefix_tokens if not self.no_embed_class else 0
        if self.dynamic_img_size and (
            grid_h != self.default_grid_h or grid_w != self.default_grid_w
        ):
            src_patches = pos_embed.shape[1] - num_prefix
            src_size = int(src_patches**0.5)
            pos_embed = interpolate_pos_embed(
                pos_embed, (src_size, src_size), (grid_h, grid_w), num_prefix
            )
        if self.no_embed_class:
            return None, pos_embed
        return (
            pos_embed[:, : self.num_prefix_tokens],
            pos_embed[:, self.num_prefix_tokens :],
        )

    def _get_prefix_tokens(self, B: int) -> Optional[torch.Tensor]:
        """Get CLS and register tokens expanded to batch size."""
        tokens = []
        if self.has_class_token:
            tokens.append(self.vit.cls_token.expand(B, -1, -1))
        if self.num_reg_tokens > 0:
            tokens.append(self.vit.reg_token.expand(B, -1, -1))
        return torch.cat(tokens, dim=1) if tokens else None

    def forward(self, images: torch.Tensor) -> MaskedEncoderOutput:
        """Encode images with optional masking.

        :param images: Input images (B, C, H, W)
        :return: MaskedEncoderOutput with encoded tokens and mask info
        """
        B = images.shape[0]
        device = images.device
        grid_h, grid_w = self._get_grid_size(images)
        num_patches = grid_h * grid_w
        # Patch embed + positional embed
        x = self.patch_embed(images)
        prefix_pos, patch_pos = self._get_pos_embed(grid_h, grid_w)
        x = x + patch_pos
        # Apply masking (training only)
        if self.training and self.masking is not None:
            mask_out = self.masking(x, grid_h, grid_w)
            x = mask_out.visible
            mask = mask_out.mask
            ids_keep = mask_out.ids_keep
        else:
            mask = torch.zeros(B, num_patches, device=device)
            ids_keep = (
                torch.arange(num_patches, device=device).unsqueeze(0).expand(B, -1)
            )
        # Prepend prefix tokens
        prefix = self._get_prefix_tokens(B)
        if prefix is not None:
            if prefix_pos is not None and not self.no_embed_class:
                prefix = prefix + prefix_pos
            x = torch.cat([prefix, x], dim=1)
        # Transformer blocks
        x = self.vit.pos_drop(x)
        x = self.vit.blocks(x) if hasattr(self.vit, "blocks") else self.vit.layers(x)
        x = self.vit.norm(x)
        return MaskedEncoderOutput(
            encoded=x,
            mask=mask,
            ids_keep=ids_keep,
            grid_size=(grid_h, grid_w),
        )

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Encode without masking (for inference)."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output = self.forward(images)
        if was_training:
            self.train()
        return output.encoded

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"patch_size=({self.patch_size_h}, {self.patch_size_w}), "
            f"num_prefix_tokens={self.num_prefix_tokens}, "
            f"has_masking={self.masking is not None}"
        )


class AdaLNDecoderBlock(nn.Module):
    """Decoder block with AdaLN-Zero conditioning (DiT-style)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

        # AdaLN: predicts (γ1, β1, α1, γ2, β2, α2, γ3, β3, α3)
        self.adaLN_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim),
        )
        nn.init.zeros_(self.adaLN_mlp[1].weight)
        nn.init.zeros_(self.adaLN_mlp[1].bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor, cond: torch.Tensor):
        mods = self.adaLN_mlp(cond).chunk(9, dim=-1)
        γ1, β1, α1, γ2, β2, α2, γ3, β3, α3 = [m.unsqueeze(1) for m in mods]

        h = self.norm1(x) * (1 + γ1) + β1
        x = x + α1 * self.self_attn(h, h, h, need_weights=False)[0]

        h = self.norm2(x) * (1 + γ2) + β2
        x = x + α2 * self.cross_attn(h, context, context, need_weights=False)[0]

        h = self.norm3(x) * (1 + γ3) + β3
        x = x + α3 * self.mlp(h)
        return x


class CrossAttentionDecoder(nn.Module):
    """Cross-attention decoder with pluggable block type.

    :param input_dim: Input dimension
    :param hidden_dim: Internal dimension
    :param output_dim: Output dimension
    :param num_patches: Total patches (for pos embed)
    :param depth: Number of blocks
    :param num_heads: Attention heads
    :param mlp_ratio: MLP expansion
    :param block_class: Block class to use (e.g., CrossAttentionBlock, AdaLNCrossAttentionBlock)
    :param pos_embed_type: 'sincos_1d', 'sincos_2d', or 'learned'
    :param grid_size: Grid size for 2D pos embed
    :param zero_init_output: Zero-init output projection
    :param num_prefix_tokens: extra tokens passed in forward
    Example::
        # Pure cross-attention (no conditioning)
        decoder = CrossAttentionDecoder(
            768, 384, 768, 196, depth=4, block_class=CrossAttentionBlock
        )
        # With AdaLN for flow matching
        decoder = CrossAttentionDecoder(
            768, 384, 768, 196, depth=4, block_class=AdaLNCrossAttentionBlock
        )
        # With your full decoder block (self + cross attn)
        decoder = CrossAttentionDecoder(
            768, 384, 768, 196, depth=4, block_class=AdaLNDecoderBlock
        )
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 384,
        output_dim: int = 768,
        num_patches: int = 196,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        block_class: Type[nn.Module] = AdaLNCrossAttentionBlock,
        pos_embed_type: Literal["sincos_1d", "sincos_2d", "learned"] = "sincos_2d",
        grid_size: Optional[int] = None,
        zero_init_output: bool = True,
        num_prefix_tokens: int = 1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_dim = hidden_dim
        # Projections
        self.context_proj = nn.Linear(input_dim, hidden_dim)
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        if zero_init_output:
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
        # Positional embeddings
        if pos_embed_type == "sincos_2d" and grid_size is None:
            grid_size = int(num_patches**0.5)
            assert grid_size**2 == num_patches
        if pos_embed_type == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
            trunc_normal_(self.pos_embed, std=0.02)
        else:
            mode = "2d" if pos_embed_type == "sincos_2d" else "1d"
            pe = get_sincos_pos_embed(
                hidden_dim, num_patches, mode=mode, grid_size=grid_size
            )
            self.register_buffer("pos_embed", pe.unsqueeze(0))
        # Time MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        # Blocks
        self.blocks = nn.ModuleList(
            [block_class(hidden_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        # Final norm
        self.final_norm = nn.LayerNorm(hidden_dim)
        # Learnable position embedding for CLS/prefix tokens
        self.num_prefix_tokens = num_prefix_tokens
        if num_prefix_tokens > 0:
            self.prefix_pos_embed = nn.Parameter(
                torch.zeros(1, num_prefix_tokens, hidden_dim)
            )
            nn.init.normal_(self.prefix_pos_embed, std=0.02)

    def _gather_pos(self, idx: torch.Tensor, num_prefix: int = 0) -> torch.Tensor:
        B = idx.shape[0]

        if num_prefix > 0:
            # Prefix tokens get learnable pos embed
            prefix_pos = self.prefix_pos_embed.expand(B, -1, -1)
            # Patch tokens get gathered pos embed
            patch_idx = idx[:, num_prefix:]
            patch_pos = torch.gather(
                self.pos_embed.expand(B, -1, -1),
                dim=1,
                index=patch_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
            )
            return torch.cat([prefix_pos, patch_pos], dim=1)
        else:
            idx = idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            return torch.gather(self.pos_embed.expand(B, -1, -1), 1, idx)

    def forward(
        self,
        context: torch.Tensor,
        queries: torch.Tensor,
        context_idx: torch.Tensor,
        query_idx: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        num_prefix: int = None,
    ) -> torch.Tensor:
        if num_prefix is None:
            num_prefix = self.num_prefix_tokens
        # Project + pos embed
        context = self.context_proj(context) + self._gather_pos(
            context_idx, self.num_prefix_tokens
        )
        queries = self.query_proj(queries) + self._gather_pos(query_idx)
        # Time embedding (passed to blocks, they decide whether to use it)
        cond = (
            self.time_mlp(get_timestep_embed(t, self.hidden_dim))
            if t is not None
            else None
        )
        # Blocks
        for block in self.blocks:
            queries = block(queries, context, cond)
        return self.output_proj(self.final_norm(queries))


class TransformerPredictor(nn.Module):
    """Lightweight transformer predictor with configurable positional embeddings.

    A flexible predictor module commonly used in masked image modeling (e.g., MAE,
    I-JEPA). Processes context tokens and optionally includes learnable register/query
    tokens for aggregation.

    Positional Embedding Modes:

    - ``None``: No internal pos_embed. User can optionally provide ``pos_embed`` in forward().
    - ``'sincos_1d'``: 1D sinusoidal, requires ``ids_keep`` in forward().
    - ``'sincos_2d'``: 2D sinusoidal, requires ``ids_keep`` and ``grid_size`` in forward().
    - ``'learned'``: Learnable, requires ``max_seq_len`` at init and ``ids_keep`` in forward().

    :param input_dim: Dimension of input context tokens
    :param hidden_dim: Internal dimension of transformer layers
    :param output_dim: Dimension of output tokens
    :param depth: Number of transformer layers
    :param num_heads: Number of attention heads
    :param num_registers: Number of learnable register/query tokens to prepend
    :param mlp_ratio: MLP hidden dimension multiplier
    :param dropout: Dropout rate
    :param pos_embed_type: Type of positional embedding (None, 'sincos_1d', 'sincos_2d', 'learned')
    :param max_seq_len: Maximum sequence length (required if pos_embed_type='learned')

    Example::

        predictor = TransformerPredictor(
            input_dim=768,
            hidden_dim=384,
            output_dim=768,
            depth=4,
            num_registers=1,
            pos_embed_type="sincos_2d",
        )
        output = predictor(visible_tokens, ids_keep=ids_keep, grid_size=(14, 14))
        mean_pred = output[:, 0]  # Extract register output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        num_heads: int = 6,
        num_registers: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pos_embed_type: Literal["sincos_1d", "sincos_2d", "learned"] | None = None,
        max_seq_len: int | None = None,
    ):
        super().__init__()

        # Validation
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        if num_registers < 0:
            raise ValueError(f"num_registers must be non-negative, got {num_registers}")
        if pos_embed_type not in (None, "sincos_1d", "sincos_2d", "learned"):
            raise ValueError(f"Invalid pos_embed_type: {pos_embed_type!r}")
        if pos_embed_type == "learned" and max_seq_len is None:
            raise ValueError("max_seq_len is required when pos_embed_type='learned'")
        if pos_embed_type == "sincos_2d" and hidden_dim % 4 != 0:
            raise ValueError(
                f"hidden_dim must be divisible by 4 for sincos_2d, got {hidden_dim}"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_registers = num_registers
        self.pos_embed_type = pos_embed_type
        self.max_seq_len = max_seq_len

        # Projections
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Register tokens
        if num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, num_registers, hidden_dim)
            )
            self.register_pos_embed = nn.Parameter(
                torch.zeros(1, num_registers, hidden_dim)
            )
            nn.init.normal_(self.register_tokens, std=0.02)
            nn.init.normal_(self.register_pos_embed, std=0.02)
        else:
            self.register_tokens = None
            self.register_pos_embed = None

        # Positional embeddings
        if pos_embed_type == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
            nn.init.normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=int(hidden_dim * mlp_ratio),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def _get_pos_embed(
        self, ids_keep: torch.Tensor, grid_size: tuple[int, int] | None
    ) -> torch.Tensor:
        """Generate or gather positional embeddings based on pos_embed_type."""
        B, N = ids_keep.shape
        device = ids_keep.device

        if self.pos_embed_type == "learned":
            pos = self.pos_embed.expand(B, -1, -1)
            return torch.gather(
                pos, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            )

        elif self.pos_embed_type == "sincos_1d":
            max_pos = int(ids_keep.max().item()) + 1
            pos_embed = get_1d_sincos_pos_embed(
                self.hidden_dim, max_pos, cls_token=False
            )
            pos_embed = pos_embed.to(device=device, dtype=self.input_proj.weight.dtype)
            pos_embed = pos_embed.unsqueeze(0).expand(B, -1, -1)
            return torch.gather(
                pos_embed, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            )

        elif self.pos_embed_type == "sincos_2d":
            pos_embed = get_2d_sincos_pos_embed(
                self.hidden_dim, grid_size, cls_token=False
            )
            pos_embed = pos_embed.to(device=device, dtype=self.input_proj.weight.dtype)
            pos_embed = pos_embed.unsqueeze(0).expand(B, -1, -1)
            return torch.gather(
                pos_embed, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            )

        raise RuntimeError(f"Unexpected pos_embed_type: {self.pos_embed_type}")

    def forward(
        self,
        context: torch.Tensor,
        pos_embed: torch.Tensor | None = None,
        ids_keep: torch.Tensor | None = None,
        grid_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Forward pass through the predictor.

        :param context: Context tokens (B, N, input_dim)
        :param pos_embed: External positional embeddings (B, N, input_dim). Only when pos_embed_type=None.
        :param ids_keep: Indices of kept positions (B, N). Required when pos_embed_type is not None.
        :param grid_size: Grid size (H, W). Required when pos_embed_type='sincos_2d'.
        :return: Output tokens (B, num_registers + N, output_dim)
        """
        if context.dim() != 3:
            raise ValueError(f"context must be 3D (B, N, D), got shape {context.shape}")

        B, N, D = context.shape
        if D != self.input_dim:
            raise ValueError(
                f"context dim {D} doesn't match input_dim {self.input_dim}"
            )

        # Validate pos_embed constraints
        if self.pos_embed_type is not None:
            if pos_embed is not None:
                raise ValueError(
                    f"Cannot provide pos_embed when pos_embed_type={self.pos_embed_type!r}. "
                    f"Provide ids_keep instead."
                )
            if ids_keep is None:
                raise ValueError(
                    f"ids_keep is required when pos_embed_type={self.pos_embed_type!r}"
                )
            if self.pos_embed_type == "sincos_2d" and grid_size is None:
                raise ValueError(
                    "grid_size is required when pos_embed_type='sincos_2d'"
                )

        if ids_keep is not None and (ids_keep.shape[0] != B or ids_keep.shape[1] != N):
            raise ValueError(
                f"ids_keep shape {ids_keep.shape} doesn't match context ({B}, {N})"
            )

        if pos_embed is not None and pos_embed.shape != context.shape:
            raise ValueError(
                f"pos_embed shape {pos_embed.shape} doesn't match context {context.shape}"
            )

        # Project to hidden dimension
        x = self.input_proj(context)

        # Add positional embeddings
        if self.pos_embed_type is not None:
            x = x + self._get_pos_embed(ids_keep, grid_size)
        elif pos_embed is not None:
            x = x + self.input_proj(pos_embed)

        # Prepend register tokens
        if self.register_tokens is not None:
            registers = self.register_tokens.expand(B, -1, -1) + self.register_pos_embed
            x = torch.cat([registers, x], dim=1)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # Project to output dimension
        return self.output_proj(x)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"output_dim={self.output_dim}, depth={self.depth}, "
            f"num_registers={self.num_registers}, pos_embed_type={self.pos_embed_type!r}"
        )


class MAEDecoder(nn.Module):
    """MAE-style ViT Decoder.

    :param embed_dim: Encoder embedding dimension (input D)
    :param decoder_embed_dim: Internal decoder dimension
    :param output_dim: Output dimension (e.g., patch_size² × in_chans for pixels)
    :param num_patches: Total sequence length T
    :param depth: Number of transformer blocks
    :param num_heads: Attention heads
    :param mlp_ratio: MLP expansion ratio
    :param pos_embed_type: 'sincos_1d', 'sincos_2d', or 'learned'
    :param grid_size: Grid size for 2D pos embed (auto-inferred if None)
    :param kwargs: Additional args passed to timm.Block
    """

    def __init__(
        self,
        embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        output_dim: int = 768,
        num_patches: int = 196,
        depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        pos_embed_type: Literal["sincos_1d", "sincos_2d", "learned"] = "sincos_2d",
        grid_size: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.num_patches = num_patches

        # Projection layers
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, output_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        trunc_normal_(self.mask_token, std=0.02)

        # Auto-infer grid_size if not provided
        if pos_embed_type == "sincos_2d" and grid_size is None:
            grid_size = int(num_patches**0.5)
            assert grid_size**2 == num_patches, (
                "num_patches must be a perfect square for 2D pos embed"
            )

        # Positional embeddings
        if pos_embed_type == "learned":
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, decoder_embed_dim)
            )
            trunc_normal_(self.pos_embed, std=0.02)
        else:
            mode = "2d" if pos_embed_type == "sincos_2d" else "1d"
            pe = get_sincos_pos_embed(
                decoder_embed_dim, num_patches, mode=mode, grid_size=grid_size
            )
            self.register_buffer("pos_embed", pe.unsqueeze(0))

        # Transformer blocks from timm (highly optimized)
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=decoder_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    **kwargs,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(decoder_embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Applies the decoder transform.

        :param x: Either visible tokens only (N, T', D) or all tokens (N, T, D).
                Automatically inferred from shape: if x.shape[1] == mask.shape[1],
                assumes full sequence with masked positions to be replaced by [MSK].
        :param mask: Binary mask (N, T), 0=kept, 1=masked
        :return: Reconstructed full sequence (N, T, D)
        """
        N, T = mask.shape
        full_sequence = x.shape[1] == T

        # Project to decoder dim
        x = self.decoder_embed(x)

        if full_sequence:
            # x is (N, T, decoder_embed_dim) - replace masked positions with mask token
            mask_tokens = self.mask_token.to(x.dtype).expand(N, T, -1)
            tokens = torch.where(mask.unsqueeze(-1).bool(), mask_tokens, x)
        else:
            # x is (N, T', decoder_embed_dim) with T' visible tokens
            tokens = self.mask_token.to(x.dtype).expand(N, T, -1).clone()
            visible_mask = ~mask.bool()

            batch_indices = torch.arange(N, device=x.device)[:, None].expand_as(
                visible_mask
            )
            seq_indices = torch.arange(T, device=x.device)[None, :].expand_as(
                visible_mask
            )

            tokens[batch_indices[visible_mask], seq_indices[visible_mask]] = x.reshape(
                -1, x.shape[-1]
            )

        # Add positional embeddings + transform
        tokens = tokens + self.pos_embed
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)

        return self.decoder_pred(tokens)


class PositionalEncoding2D(nn.Module):
    """Flexible 2D positional encoding for vision transformers."""

    def __init__(
        self,
        embed_dim: int,
        grid_size: Tuple[int, int],
        pos_type: Literal["learnable", "sinusoidal", "rope", "none"] = "learnable",
        num_prefix_tokens: int = 1,
        learnable: Optional[
            bool
        ] = None,  # Override: force learnable even for sinusoidal
    ):
        """Positional encoding for 2d input.

        :param embed_dim: Embedding dimension
        :param grid_size: (H, W) grid size in patches
        :param pos_type: Type of positional encoding
        :param num_prefix_tokens: Number of prefix tokens (CLS + registers)
        :param learnable: If True, make sinusoidal learnable; if None, use default

        """
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_h, self.grid_w = grid_size
        self.num_patches = self.grid_h * self.grid_w
        self.pos_type = pos_type
        self.num_prefix_tokens = num_prefix_tokens

        # Override learnable if specified
        if learnable is not None:
            self.is_learnable = learnable
        else:
            self.is_learnable = pos_type == "learnable"

        if pos_type == "none":
            # No positional encoding
            self.pos_embed = None

        elif pos_type == "learnable":
            # Learnable absolute positional embeddings
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_prefix_tokens + self.num_patches, embed_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        elif pos_type == "sinusoidal":
            # 2D sinusoidal positional embeddings
            pos_embed = self._build_sinusoidal_2d(embed_dim, self.grid_h, self.grid_w)

            # Add prefix token positions (zeros or learned separately)
            prefix_pos = torch.zeros(1, num_prefix_tokens, embed_dim)
            pos_embed = torch.cat([prefix_pos, pos_embed], dim=1)

            if self.is_learnable:
                self.pos_embed = nn.Parameter(pos_embed)
            else:
                self.register_buffer("pos_embed", pos_embed)

        elif pos_type == "rope":
            # RoPE doesn't use additive embeddings
            self.pos_embed = None
            # Precompute RoPE frequencies
            self.register_buffer(
                "freqs_h", self._build_rope_freqs(embed_dim // 4, self.grid_h)
            )
            self.register_buffer(
                "freqs_w", self._build_rope_freqs(embed_dim // 4, self.grid_w)
            )
        else:
            raise ValueError(f"Unknown pos_type: {pos_type}")

    def _build_sinusoidal_2d(
        self, embed_dim: int, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        """Build 2D sinusoidal positional embeddings."""
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sinusoidal"

        dim_h = embed_dim // 2
        dim_w = embed_dim // 2

        # Height positions
        pos_h = torch.arange(grid_h).unsqueeze(1)  # [H, 1]
        dim_t_h = torch.arange(0, dim_h, 2).float()  # [dim_h/2]
        omega_h = 1.0 / (10000 ** (dim_t_h / dim_h))

        pos_embed_h = torch.zeros(grid_h, dim_h)
        pos_embed_h[:, 0::2] = torch.sin(pos_h * omega_h)
        pos_embed_h[:, 1::2] = torch.cos(pos_h * omega_h)

        # Width positions
        pos_w = torch.arange(grid_w).unsqueeze(1)  # [W, 1]
        dim_t_w = torch.arange(0, dim_w, 2).float()
        omega_w = 1.0 / (10000 ** (dim_t_w / dim_w))

        pos_embed_w = torch.zeros(grid_w, dim_w)
        pos_embed_w[:, 0::2] = torch.sin(pos_w * omega_w)
        pos_embed_w[:, 1::2] = torch.cos(pos_w * omega_w)

        # Combine: [H, W, D]
        pos_embed_h = pos_embed_h.unsqueeze(1).expand(-1, grid_w, -1)  # [H, W, dim_h]
        pos_embed_w = pos_embed_w.unsqueeze(0).expand(grid_h, -1, -1)  # [H, W, dim_w]

        pos_embed = torch.cat([pos_embed_h, pos_embed_w], dim=-1)  # [H, W, D]
        pos_embed = pos_embed.reshape(1, grid_h * grid_w, embed_dim)  # [1, H*W, D]

        return pos_embed

    def _build_rope_freqs(
        self, dim: int, max_seq_len: int, base: float = 10000.0
    ) -> torch.Tensor:
        """Build RoPE frequency tensor."""
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(max_seq_len)
        freqs = torch.einsum("i,j->ij", pos, inv_freq)  # [seq_len, dim/2]
        freqs = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        return freqs

    def _apply_rope_2d(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """Apply 2D RoPE to patch tokens."""
        B, N, D = x.shape

        # Separate prefix and patch tokens
        prefix = x[:, : self.num_prefix_tokens, :]
        patches = x[:, self.num_prefix_tokens :, :]  # [B, H*W, D]

        # Reshape to 2D grid
        patches = patches.reshape(B, grid_h, grid_w, D)

        # Split embedding into 4 parts for 2D RoPE
        d_quarter = D // 4
        x1, x2, x3, x4 = patches.split(d_quarter, dim=-1)

        # Get frequencies (interpolate if needed)
        freqs_h = self.freqs_h[:grid_h, :d_quarter]  # [H, d_quarter]
        freqs_w = self.freqs_w[:grid_w, :d_quarter]  # [W, d_quarter]

        # Apply rotation to height dimension (x1, x2)
        cos_h = torch.cos(freqs_h).unsqueeze(1)  # [H, 1, d_quarter]
        sin_h = torch.sin(freqs_h).unsqueeze(1)  # [H, 1, d_quarter]
        x1_rot = x1 * cos_h - x2 * sin_h
        x2_rot = x1 * sin_h + x2 * cos_h

        # Apply rotation to width dimension (x3, x4)
        cos_w = torch.cos(freqs_w).unsqueeze(0)  # [1, W, d_quarter]
        sin_w = torch.sin(freqs_w).unsqueeze(0)  # [1, W, d_quarter]
        x3_rot = x3 * cos_w - x4 * sin_w
        x4_rot = x3 * sin_w + x4 * cos_w

        # Combine
        patches = torch.cat([x1_rot, x2_rot, x3_rot, x4_rot], dim=-1)
        patches = patches.reshape(B, grid_h * grid_w, D)

        # Recombine with prefix (prefix tokens don't get RoPE)
        return torch.cat([prefix, patches], dim=1)

    def forward(
        self, x: torch.Tensor, grid_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Apply positional encoding.

        :param x: [B, num_prefix + num_patches, D]
        :param grid_size: (H, W) if different from default (for dynamic size)
        :return: x with positional encoding applied
        """
        if self.pos_type == "none":
            return x

        grid_h = grid_size[0] if grid_size else self.grid_h
        grid_w = grid_size[1] if grid_size else self.grid_w

        if self.pos_type == "rope":
            return self._apply_rope_2d(x, grid_h, grid_w)

        # Additive positional embeddings (learnable or sinusoidal)
        pos_embed = self.pos_embed

        # Interpolate if dynamic size
        if grid_h != self.grid_h or grid_w != self.grid_w:
            pos_embed = self._interpolate(pos_embed, grid_h, grid_w)

        return x + pos_embed

    def _interpolate(
        self, pos_embed: torch.Tensor, target_h: int, target_w: int
    ) -> torch.Tensor:
        """Interpolate positional embeddings to new grid size."""
        prefix_pos = pos_embed[:, : self.num_prefix_tokens, :]
        patch_pos = pos_embed[:, self.num_prefix_tokens :, :]

        D = patch_pos.shape[-1]
        patch_pos = patch_pos.reshape(1, self.grid_h, self.grid_w, D).permute(
            0, 3, 1, 2
        )
        patch_pos = F.interpolate(
            patch_pos, size=(target_h, target_w), mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, target_h * target_w, D)

        return torch.cat([prefix_pos, patch_pos], dim=1)
