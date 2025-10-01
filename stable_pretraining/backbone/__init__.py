# Try to import mae if timm is available
try:
    from . import mae

    _MAE_AVAILABLE = True
except ImportError:
    mae = None
    _MAE_AVAILABLE = False

# Try to import DINO enhancements
try:
    from .dino import (
        DINOEnhancedViT,
        MemEffAttention,
        clear_attn_bias_cache,
        create_mem_eff_attention,
        from_timm_dino,
        get_attn_bias_and_cat,
        pack_sequences,
        unpack_sequences,
    )

    _DINO_AVAILABLE = True
except ImportError:
    from_timm_dino = None
    DINOEnhancedViT = None
    MemEffAttention = None
    create_mem_eff_attention = None
    get_attn_bias_and_cat = None
    unpack_sequences = None
    pack_sequences = None
    clear_attn_bias_cache = None
    _DINO_AVAILABLE = False

from .convmixer import ConvMixer
from .mlp import MLP
from .resnet9 import Resnet9
from .utils import (
    EvalOnly,
    FeaturesConcat,
    TeacherStudentWrapper,
    from_timm,
    from_torchvision,
    set_embedding_dim,
)

__all__ = [
    MLP,
    TeacherStudentWrapper,
    Resnet9,
    from_timm,
    from_torchvision,
    EvalOnly,
    FeaturesConcat,
    set_embedding_dim,
    ConvMixer,
]

if _MAE_AVAILABLE:
    __all__.append("mae")

if _DINO_AVAILABLE:
    __all__.extend(
        [
            "from_timm_dino",
            "DINOEnhancedViT",
            "MemEffAttention",
            "create_mem_eff_attention",
            "get_attn_bias_and_cat",
            "unpack_sequences",
            "pack_sequences",
            "clear_attn_bias_cache",
        ]
    )
