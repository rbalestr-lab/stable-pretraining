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
from .probe import (
    MultiHeadAttentiveProbe,
    LinearProbe,
    AutoLinearClassifier,
    AutoTuneMLP,
)
from .patch_masking import PatchMasking
from .utils import (
    EvalOnly,
    FeaturesConcat,
    TeacherStudentWrapper,
    from_timm,
    from_torchvision,
    set_embedding_dim,
    vit_hf,
    EfficientMaskedTimmViT,
    register_lr_scale_hook,
    HiddenStateExtractor,
)
from .vit import (
    MAEDecoder,
    MaskedEncoder,
    MaskedEncoderOutput,
    TransformerPredictor,
    AdaLNDecoderBlock,
    AdaLNCrossAttentionBlock,
    CrossAttentionBlock,
    CrossAttentionDecoder,
)

from .aggregator import TensorAggregator

__all__ = [
    MLP,
    TensorAggregator,
    TeacherStudentWrapper,
    Resnet9,
    from_timm,
    from_torchvision,
    EvalOnly,
    FeaturesConcat,
    set_embedding_dim,
    ConvMixer,
    vit_hf,
    MultiHeadAttentiveProbe,
    LinearProbe,
    AutoLinearClassifier,
    EfficientMaskedTimmViT,
    register_lr_scale_hook,
    AutoTuneMLP,
    HiddenStateExtractor,
    PatchMasking,
    TransformerPredictor,
    MAEDecoder,
    MaskedEncoder,
    MaskedEncoderOutput,
    TransformerPredictor,
    AdaLNDecoderBlock,
    AdaLNCrossAttentionBlock,
    CrossAttentionBlock,
    CrossAttentionDecoder,
]

if _MAE_AVAILABLE:
    __all__.append("mae")
