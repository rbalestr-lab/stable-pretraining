# Try to import mae if timm is available
try:
    from . import mae

    _MAE_AVAILABLE = True
except ImportError:
    mae = None
    _MAE_AVAILABLE = False

# Try to import dinov2 enhancements
try:
    from .dinov2_vit import DINOv2EnhancedViT, from_timm_dinov2
    from .dinov2_attention import MemEffAttention, create_mem_eff_attention

    _DINOV2_AVAILABLE = True
except ImportError:
    from_timm_dinov2 = None
    DINOv2EnhancedViT = None
    MemEffAttention = None
    create_mem_eff_attention = None
    _DINOV2_AVAILABLE = False

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

if _DINOV2_AVAILABLE:
    __all__.extend(
        [
            "from_timm_dinov2",
            "DINOv2EnhancedViT",
            "MemEffAttention",
            "create_mem_eff_attention",
        ]
    )
