# Try to import mae if timm is available
try:
    from . import mae

    _MAE_AVAILABLE = True
except ImportError:
    mae = None
    _MAE_AVAILABLE = False

from .convmixer import ConvMixer
from .mlp import MLP
from .resnet9 import Resnet9
from .utils import (
    EvalOnly,
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
    set_embedding_dim,
    ConvMixer,
]

if _MAE_AVAILABLE:
    __all__.append("mae")
