import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal, Union
import timm
from timm.layers import DropPath, Mlp, trunc_normal_
from .patch_masking import PatchMasking
from dataclasses import dataclass
from .pos_embed import (
    get_sincos_pos_embed,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    get_timestep_embed,
    interpolate_pos_embed,
)


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


# =============================================================================
# Efficient Attention Modules
# =============================================================================
class Attention(nn.Module):
    """Multi-head self-attention with efficient SDPA backend.

    Uses F.scaled_dot_product_attention which automatically selects:
    - Flash Attention (when available, fastest)
    - Memory-efficient attention (xformers-style)
    - Math fallback
    :param dim: Input dimension
    :param num_heads: Number of attention heads
    :param qkv_bias: Add bias to QKV projection
    :param attn_drop: Attention dropout rate
    :param proj_drop: Output projection dropout rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        # Fused QKV projection for efficiency
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Input tensor [B, N, D]
        :return: Output tensor [B, N, D]
        """
        B, N, C = x.shape
        # Fused QKV: [B, N, 3*D] -> [B, N, 3, H, head_dim] -> [3, B, H, N, head_dim]
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        # Efficient attention (Flash/Memory-efficient when available)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        # Reshape back: [B, H, N, head_dim] -> [B, N, D]
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """Multi-head cross-attention with efficient SDPA backend.

    Queries attend to key-value pairs from a separate context sequence.
    :param dim: Query dimension
    :param context_dim: Context dimension (defaults to dim)
    :param num_heads: Number of attention heads
    :param qkv_bias: Add bias to projections
    :param attn_drop: Attention dropout rate
    :param proj_drop: Output projection dropout rate
    """

    def __init__(
        self,
        dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        context_dim = context_dim or dim
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(context_dim, dim * 2, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Query tensor [B, N, D]
        :param context: Key-value tensor [B, M, context_dim]
        :return: Output tensor [B, N, D]
        """
        B, N, C = x.shape
        M = context.shape[1]
        # Query projection: [B, N, D] -> [B, H, N, head_dim]
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # KV projection: [B, M, D] -> [B, H, M, head_dim] x2
        kv = (
            self.kv(context)
            .reshape(B, M, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        # Efficient attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        # Reshape back
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# =============================================================================
# Transformer Block
# =============================================================================
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale) + shift


class TransformerBlock(nn.Module):
    """Unified transformer block with optional AdaLN-Zero conditioning.

    Supports three attention configurations:
    **Mode 1: Pure Cross-Attention** (`self_attn=False, cross_attn=True`)
        - Queries attend to context but not to each other
        - Use case: Lightweight decoder
    **Mode 2: Decoder-Style** (`self_attn=True, cross_attn=True`)
        - Self-attention on queries, then cross-attention to context
        - Use case: Standard decoder (IJEPA predictor, etc.)
    **Mode 3: Joint Attention** (`self_attn=True, cross_attn=False`)
        - All tokens attend to all tokens (caller concatenates context + queries)
        - Use case: Full bidirectional flow (DiT, high masking ratio)
    **Conditioning:**
        - `use_adaln=True`: AdaLN-Zero modulation (scale, shift, gate per operation)
        - `use_adaln=False`: Standard pre-norm transformer
    :param dim: Hidden dimension
    :param num_heads: Number of attention heads
    :param mlp_ratio: MLP hidden dim = dim * mlp_ratio
    :param self_attn: Enable self-attention
    :param cross_attn: Enable cross-attention
    :param use_adaln: Enable AdaLN-Zero conditioning
    :param drop_path: Stochastic depth rate
    :param attn_drop: Attention dropout rate
    :param proj_drop: Projection dropout rate
    :param act_layer: Activation layer for MLP
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        self_attn: bool = True,
        cross_attn: bool = True,
        use_adaln: bool = True,
        drop_path: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: type = nn.GELU,
    ):
        super().__init__()
        self.use_self_attn = self_attn
        self.use_cross_attn = cross_attn
        self.use_adaln = use_adaln
        if not self_attn and not cross_attn:
            raise ValueError("At least one of self_attn or cross_attn must be True")
        # Self-attention
        if self_attn:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=not use_adaln)
            self.attn = Attention(
                dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop
            )
            self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # Cross-attention
        if cross_attn:
            self.norm2_q = nn.LayerNorm(dim, elementwise_affine=not use_adaln)
            self.norm2_kv = nn.LayerNorm(dim, elementwise_affine=not use_adaln)
            self.cross_attn = CrossAttention(
                dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop
            )
            self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # MLP
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=not use_adaln)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # AdaLN modulation MLP
        if use_adaln:
            # 3 params (shift, scale, gate) per operation
            num_ops = int(self_attn) + int(cross_attn) + 1  # +1 for MLP
            self.num_mods = num_ops * 3
            self.adaLN_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim, self.num_mods * dim),
            )
            # Zero-init for identity initialization
            nn.init.zeros_(self.adaLN_mlp[1].weight)
            nn.init.zeros_(self.adaLN_mlp[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        :param x: Input tensor [B, N, D]
        :param context: Context for cross-attention [B, M, D] (required if cross_attn=True)
        :param cond: Conditioning tensor [B, D] (required if use_adaln=True)
        :return: Output tensor [B, N, D]
        """
        if self.use_cross_attn and context is None:
            raise ValueError("context required when cross_attn=True")
        if self.use_adaln and cond is None:
            raise ValueError("cond required when use_adaln=True")
        if self.use_adaln:
            # Get modulation parameters: [B, num_mods * D] -> list of [B, 1, D]
            mods = self.adaLN_mlp(cond).chunk(self.num_mods, dim=-1)
            mods = [m.unsqueeze(1) for m in mods]
            i = 0
            # Self-attention with AdaLN
            if self.use_self_attn:
                shift, scale, gate = mods[i], mods[i + 1], mods[i + 2]
                i += 3
                x = x + gate * self.drop_path1(
                    self.attn(modulate(self.norm1(x), shift, scale))
                )
            # Cross-attention with AdaLN
            if self.use_cross_attn:
                shift, scale, gate = mods[i], mods[i + 1], mods[i + 2]
                i += 3
                q = modulate(self.norm2_q(x), shift, scale)
                kv = self.norm2_kv(context)
                x = x + gate * self.drop_path2(self.cross_attn(q, kv))
            # MLP with AdaLN
            shift, scale, gate = mods[i], mods[i + 1], mods[i + 2]
            x = x + gate * self.drop_path3(
                self.mlp(modulate(self.norm3(x), shift, scale))
            )
        else:
            # Standard pre-norm transformer (no conditioning)
            if self.use_self_attn:
                x = x + self.drop_path1(self.attn(self.norm1(x)))
            if self.use_cross_attn:
                x = x + self.drop_path2(
                    self.cross_attn(self.norm2_q(x), self.norm2_kv(context))
                )
            x = x + self.drop_path3(self.mlp(self.norm3(x)))
        return x


# =============================================================================
# Full Transformer
# =============================================================================
class FlexibleTransformer(nn.Module):
    """Flexible transformer supporting multiple architectures.

    Unified backbone for:
    - **MAE decoder**: `self_attn=True, cross_attn=False, use_adaln=False`
    - **IJEPA predictor**: `self_attn=True, cross_attn=True, use_adaln=False`
    - **DiT / Flow**: `self_attn=True, cross_attn=True/False, use_adaln=True`
    - **MaskGIT**: `self_attn=True, cross_attn=False, use_adaln=True`
    :param input_dim: Input embedding dimension (from encoder)
    :param hidden_dim: Internal transformer dimension
    :param output_dim: Output dimension
    :param num_patches: Total number of patches (for positional embeddings)
    :param depth: Number of transformer blocks
    :param num_heads: Number of attention heads
    :param mlp_ratio: MLP hidden dim multiplier
    :param self_attn: Enable self-attention in blocks
    :param cross_attn: Enable cross-attention in blocks
    :param use_adaln: Enable AdaLN-Zero conditioning
    :param pos_embed_type: 'sincos_1d', 'sincos_2d', or 'learned'
    :param grid_size: Grid size for 2D positional embeddings
    :param drop_path_rate: Stochastic depth rate (linearly increases through layers)
    :param attn_drop: Attention dropout rate
    :param proj_drop: Projection dropout rate
    :param zero_init_output: Zero-initialize output projection
    :param num_prefix_tokens: Number of prefix tokens (e.g., CLS token)
    Example::
        # MAE decoder
        decoder = FlexibleTransformer(
            768,
            512,
            768,
            196,
            depth=8,
            self_attn=True,
            cross_attn=False,
            use_adaln=False,
        )
        out = decoder(context, queries, context_idx, query_idx)
        # IJEPA predictor
        predictor = FlexibleTransformer(
            768,
            384,
            768,
            196,
            depth=6,
            self_attn=True,
            cross_attn=True,
            use_adaln=False,
        )
        out = predictor(context, queries, context_idx, query_idx)
        # DiT-style flow matching
        flow = FlexibleTransformer(
            768,
            384,
            768,
            196,
            depth=12,
            self_attn=True,
            cross_attn=False,
            use_adaln=True,
        )
        out = flow(context, queries, context_idx, query_idx, t=timesteps)
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
        self_attn: bool = True,
        cross_attn: bool = True,
        use_adaln: bool = True,
        pos_embed_type: Literal["sincos_1d", "sincos_2d", "learned"] = "sincos_2d",
        grid_size: Optional[int | tuple[int, int]] = None,
        drop_path_rate: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        zero_init_output: bool = True,
        num_prefix_tokens: int = 1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_dim = hidden_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.use_cross_attn = cross_attn
        self.use_adaln = use_adaln
        # Input/output projections
        self.context_proj = nn.Linear(input_dim, hidden_dim)
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        if zero_init_output:
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
        # Positional embeddings
        if pos_embed_type == "sincos_2d":
            if grid_size is None:
                grid_size = int(num_patches**0.5)
                if grid_size**2 != num_patches:
                    raise ValueError(
                        f"num_patches ({num_patches}) must be a perfect square for sincos_2d"
                    )
            pe = get_sincos_pos_embed(
                hidden_dim, num_patches, mode="2d", grid_size=grid_size
            )
            self.register_buffer("pos_embed", pe.unsqueeze(0))
        elif pos_embed_type == "sincos_1d":
            pe = get_sincos_pos_embed(hidden_dim, num_patches, mode="1d")
            self.register_buffer("pos_embed", pe.unsqueeze(0))
        else:  # learned
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
            trunc_normal_(self.pos_embed, std=0.02)
        # Prefix token positional embeddings
        if num_prefix_tokens > 0:
            self.prefix_pos_embed = nn.Parameter(
                torch.zeros(1, num_prefix_tokens, hidden_dim)
            )
            nn.init.normal_(self.prefix_pos_embed, std=0.02)
        # Time embedding MLP (only needed for AdaLN)
        if use_adaln:
            self.time_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
        # Transformer blocks with linearly increasing drop path
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    self_attn=self_attn,
                    cross_attn=cross_attn,
                    use_adaln=use_adaln,
                    drop_path=dpr[i],
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for i in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def _gather_pos(self, idx: torch.Tensor, num_prefix: int = 0) -> torch.Tensor:
        """Gather positional embeddings for given indices."""
        B = idx.shape[0]
        if num_prefix > 0:
            prefix_pos = self.prefix_pos_embed.expand(B, -1, -1)
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
        num_prefix: Optional[int] = None,
        return_all: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        :param context: Context token embeddings [B, N_ctx, input_dim]
        :param queries: Query token embeddings [B, N_qry, input_dim]
        :param context_idx: Patch indices for context tokens [B, N_ctx]
        :param query_idx: Patch indices for query tokens [B, N_qry]
        :param t: Timesteps for conditioning [B] (required if use_adaln=True)
        :param num_prefix: Override for number of prefix tokens
        :param return_all: If True and using joint attention (cross_attn=False),
                          return all tokens in original position order
                          [B, N_ctx + N_qry, output_dim]. Ignored for cross-attention modes.
        :return: Output embeddings [B, N_qry, output_dim] or [B, T, output_dim] if return_all
        """
        if num_prefix is None:
            num_prefix = self.num_prefix_tokens
        # Project and add positional embeddings
        context = self.context_proj(context) + self._gather_pos(context_idx, num_prefix)
        queries = self.query_proj(queries) + self._gather_pos(query_idx)
        # Time conditioning (only for AdaLN mode)
        cond = None
        if self.use_adaln:
            if t is None:
                raise ValueError("Timestep t required when use_adaln=True")
            cond = self.time_mlp(get_timestep_embed(t, self.hidden_dim))
        n_context = context.shape[1]
        n_queries = queries.shape[1]
        if self.use_cross_attn:
            # Modes 1 & 2: queries attend to context (return_all ignored)
            for block in self.blocks:
                queries = block(queries, context=context, cond=cond)
            return self.output_proj(self.final_norm(queries))
        # Mode 3: joint attention
        x = torch.cat([context, queries], dim=1)
        for block in self.blocks:
            x = block(x, cond=cond)
        x = self.final_norm(x)
        if return_all:
            # Unshuffle to original positions
            B = context_idx.shape[0]
            T = n_context + n_queries
            out = torch.empty(B, T, self.hidden_dim, device=x.device, dtype=x.dtype)
            out.scatter_(
                dim=1,
                index=context_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
                src=x[:, :n_context],
            )
            out.scatter_(
                dim=1,
                index=query_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
                src=x[:, n_context:],
            )
            return self.output_proj(out)
        # Handle edge case: no queries
        if n_queries == 0:
            B = context.shape[0]
            return torch.empty(
                B, 0, self.output_proj.out_features, device=x.device, dtype=x.dtype
            )

        return self.output_proj(x[:, -n_queries:])


class TransformerPredictor(nn.Module):
    """Lightweight transformer predictor using TransformerBlock.

    A flexible predictor module commonly used in masked image modeling (e.g., MAE,
    I-JEPA). Processes context tokens and optionally includes learnable register/query
    tokens for aggregation.
    :param input_dim: Dimension of input context tokens
    :param hidden_dim: Internal dimension of transformer layers
    :param output_dim: Dimension of output tokens
    :param depth: Number of transformer layers
    :param num_heads: Number of attention heads
    :param num_registers: Number of learnable register/query tokens to prepend
    :param mlp_ratio: MLP hidden dimension multiplier
    :param drop_path_rate: Stochastic depth rate
    :param pos_embed_type: Type of positional embedding (None, 'sincos_1d', 'sincos_2d', 'learned')
    :param max_seq_len: Maximum sequence length (required if pos_embed_type='learned')
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
        drop_path_rate: float = 0.0,
        pos_embed_type: Literal["sincos_1d", "sincos_2d", "learned"] | None = None,
        max_seq_len: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_registers = num_registers
        self.pos_embed_type = pos_embed_type
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
        # Learned positional embeddings (sincos computed on-the-fly)
        if pos_embed_type == "learned":
            assert max_seq_len is not None, "max_seq_len required for learned pos_embed"
            self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
            nn.init.normal_(self.pos_embed, std=0.02)
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    self_attn=True,
                    cross_attn=False,
                    use_adaln=False,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def _get_pos_embed(
        self,
        ids_keep: torch.Tensor,
        grid_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        """Gather or generate positional embeddings."""
        B, N = ids_keep.shape
        device, dtype = ids_keep.device, self.input_proj.weight.dtype
        if self.pos_embed_type == "learned":
            return torch.gather(
                self.pos_embed.expand(B, -1, -1),
                dim=1,
                index=ids_keep.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
            )
        # Generate sincos on-the-fly
        if self.pos_embed_type == "sincos_1d":
            max_pos = int(ids_keep.max().item()) + 1
            pe = get_1d_sincos_pos_embed(self.hidden_dim, max_pos)
        else:  # sincos_2d
            pe = get_2d_sincos_pos_embed(self.hidden_dim, grid_size)
        pe = pe.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        return torch.gather(
            pe, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        )

    def forward(
        self,
        context: torch.Tensor,
        pos_embed: torch.Tensor | None = None,
        ids_keep: torch.Tensor | None = None,
        grid_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        :param context: Context tokens [B, N, input_dim]
        :param pos_embed: External positional embeddings [B, N, input_dim] (when pos_embed_type=None)
        :param ids_keep: Indices of kept positions [B, N] (when pos_embed_type is not None)
        :param grid_size: Grid size (H, W) for sincos_2d
        :return: Output tokens [B, num_registers + N, output_dim]
        """
        B = context.shape[0]
        # Project to hidden dim
        x = self.input_proj(context)
        # Add positional embeddings
        if self.pos_embed_type is not None:
            x = x + self._get_pos_embed(ids_keep, grid_size)
        elif pos_embed is not None:
            x = x + self.input_proj(pos_embed)
        # Prepend registers
        if self.num_registers > 0:
            registers = self.register_tokens.expand(B, -1, -1) + self.register_pos_embed
            x = torch.cat([registers, x], dim=1)
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        return self.output_proj(self.norm(x))


class MAEDecoder(nn.Module):
    """MAE-style ViT Decoder using FlexibleTransformer.

    Uses joint self-attention where visible tokens and mask tokens
    all attend to each other.
    :param embed_dim: Encoder embedding dimension (input D)
    :param decoder_embed_dim: Internal decoder dimension
    :param output_dim: Output dimension (e.g., patch_size² × in_chans for pixels)
    :param num_patches: Total number of patches T
    :param depth: Number of transformer blocks
    :param num_heads: Attention heads
    :param mlp_ratio: MLP expansion ratio
    :param pos_embed_type: 'sincos_1d', 'sincos_2d', or 'learned'
    :param grid_size: Grid size for 2D pos embed (auto-inferred if None)
    :param drop_path_rate: Stochastic depth rate
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
        grid_size: Optional[int] = None,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.num_patches = num_patches
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.mask_token, std=0.02)
        # Core transformer
        self.transformer = FlexibleTransformer(
            input_dim=embed_dim,
            hidden_dim=decoder_embed_dim,
            output_dim=output_dim,
            num_patches=num_patches,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            self_attn=True,
            cross_attn=False,
            use_adaln=False,
            pos_embed_type=pos_embed_type,
            grid_size=grid_size,
            drop_path_rate=drop_path_rate,
            zero_init_output=False,
            num_prefix_tokens=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        output_masked_only: bool = True,
    ) -> torch.Tensor:
        """Forward pass.

        :param x: Visible tokens [B, N_vis, D] or full sequence [B, T, D]
        :param mask: Binary mask [B, T], 0=kept, 1=masked
        :param output_masked_only: If True, return [B, N_mask, D].
                                If False, return [B, T, D].
        :return: Predictions
        """
        B, T = mask.shape
        mask_bool = mask.bool()  # Convert once, use everywhere

        N_vis = (~mask_bool).sum(dim=1)[0].int().item()
        N_mask = T - N_vis
        # Get indices (sort False/0 before True/1, so visible indices come first)
        visible_idx = torch.argsort(mask_bool.int(), dim=1, stable=True)[:, :N_vis]
        masked_idx = torch.argsort((~mask_bool).int(), dim=1, stable=True)[:, :N_mask]
        # Get visible tokens
        if x.shape[1] == T:
            visible_tokens = torch.gather(
                x, dim=1, index=visible_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            )
        else:
            visible_tokens = x
        # Mask tokens for masked positions
        mask_tokens = self.mask_token.expand(B, N_mask, -1)
        return self.transformer(
            context=visible_tokens,
            queries=mask_tokens,
            context_idx=visible_idx,
            query_idx=masked_idx,
            return_all=not output_masked_only,
        )


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
