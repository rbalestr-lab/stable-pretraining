import copy
import math
from typing import Union, Iterable, List, Optional, Any, Dict

import torch
import torchvision
from loguru import logger as logging
from torch import nn

# Try to import optional dependencies
try:
    from timm.layers.classifier import ClassifierHead

    _TIMM_AVAILABLE = True
except ImportError:
    ClassifierHead = None
    _TIMM_AVAILABLE = False

try:
    from transformers import TimmWrapperModel, ViTConfig, ViTModel

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    TimmWrapperModel = None
    ViTConfig = None
    ViTModel = None
    _TRANSFORMERS_AVAILABLE = False


def vit_hf(
    size: str = "tiny",
    patch_size: int = 16,
    image_size: int = 224,
    pretrained: bool = False,
    use_mask_token: bool = True,
    **kwargs,
) -> nn.Module:
    """Create a Vision Transformer using HuggingFace transformers.

    This provides a clean, well-maintained ViT implementation with native support for:
    - Masking via bool_masked_pos parameter
    - Learnable mask token
    - Easy access to CLS and patch tokens

    Args:
        size: Model size - "tiny", "small", "base", or "large"
        patch_size: Patch size (default: 16)
        image_size: Input image size (default: 224)
        pretrained: Load pretrained weights from HuggingFace Hub
        use_mask_token: Whether to include learnable mask token (needed for iBOT)
        **kwargs: Additional ViTConfig parameters

    Returns:
        HuggingFace ViTModel

    Example:
        >>> backbone = vit_hf("tiny", use_mask_token=True)
        >>> x = torch.randn(2, 3, 224, 224)
        >>>
        >>> # Without masking
        >>> output = backbone(x)
        >>> cls_token = output.last_hidden_state[:, 0, :]
        >>> patch_tokens = output.last_hidden_state[:, 1:, :]
        >>>
        >>> # With masking (for iBOT student)
        >>> masks = torch.zeros(2, 196, dtype=torch.bool)
        >>> masks[:, :59] = True  # Mask 30%
        >>> output = backbone(x, bool_masked_pos=masks)
    """
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers library is required for vit_hf. "
            "Install with: pip install transformers"
        )

    # ViT size configurations (matching timm/DINOv3)
    size_configs = {
        "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
        "small": {
            "hidden_size": 384,
            "num_hidden_layers": 12,
            "num_attention_heads": 6,
        },
        "base": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
        },
        "large": {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
        },
    }

    if size not in size_configs:
        raise ValueError(
            f"Invalid size '{size}'. Choose from {list(size_configs.keys())}"
        )

    config_params = size_configs[size]
    config_params["intermediate_size"] = config_params["hidden_size"] * 4
    config_params["image_size"] = image_size
    config_params["patch_size"] = patch_size
    config_params.update(kwargs)

    if pretrained:
        # Try to load pretrained model from HF Hub
        model_name = f"google/vit-{size}-patch{patch_size}-{image_size}"
        logging.info(f"Loading pretrained ViT from {model_name}")
        model = ViTModel.from_pretrained(
            model_name, add_pooling_layer=False, use_mask_token=use_mask_token
        )
    else:
        config = ViTConfig(**config_params)
        model = ViTModel(config, add_pooling_layer=False, use_mask_token=use_mask_token)
        logging.info(f"Created ViT-{size} from scratch with config: {config_params}")

    # IMPORTANT: Set model to always interpolate position encodings for dynamic input sizes
    # This allows processing images of different sizes (e.g., 224x224 global + 96x96 local views)
    # Must be set as instance attribute, not in config
    model.config.interpolate_pos_encoding = True

    return model


class EvalOnly(nn.Module):
    """Wrapper that forces a module to remain in evaluation mode."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.backbone.train(False)
        self.requires_grad_(False)
        assert not self.backbone.training

    def train(self, mode):
        return self

    def forward(self, *args, **kwargs):
        if self.backbone.training:
            raise RuntimeError("EvalOnly module is in training mode")
        return self.backbone.forward(*args, **kwargs)


class FeaturesConcat(nn.Module):
    """Aggregates and concatenates features from a dictionary input, then classifies.

    Args:
        names (List[str]): Keys to extract from the input dictionary.
            if not given then we aggregate everything from dict/list
    """

    def __init__(self, agg: callable, names: Union[str, Iterable[str]] = None):
        super().__init__()
        if type(names) is str:
            names = [names]
        self.names = names
        self.agg = agg

    def forward(self, inputs: Union[dict, Iterable]):
        if type(inputs) is dict:
            assert self.names is not None
            tensors = [inputs[n] for n in self.names]
        else:
            tensors = inputs
        reps = []
        for t in tensors:
            reps.append(self.agg(t))
        concat = torch.cat(reps, dim=1)
        return concat

    @staticmethod
    def get_output_shape(
        agg: callable, shapes: Union[list[str], Dict[str, Iterable[int]]]
    ):
        """Given a list of shapes (tuples), returns the expected concatenated shape.

        Assumes all shapes have the same batch size (shapes[0][0]).

        Args:
            shapes (List[Tuple[int]]): List of shapes after aggregation.
            agg (callable): How to aggregate, can be None.

        Returns:
            Tuple[int]: The concatenated shape.
        """
        if not shapes:
            raise ValueError("Shape list is empty.")
        if type(shapes) is dict:
            shapes = list(shapes.values())
        x = [torch.empty(shape, device="meta") for shape in shapes]
        obj = FeaturesConcat(agg)
        out = obj(x)
        return out.shape


class ReturnEmbedding(nn.Module):
    """Cache embedding from a module given their names.

    Example:
    stable_pretraining.backbone.utils.ReturnEmbedding(
        torchvision.models.swin_v2_s(),
        stable_pretraining.static.EMBEDDINGS["swin_v2_s"]
        )

    Args:
    module_names (list of str): List of module names to hook (e.g., ['layer1', 'encoder.block1']).
    add_to_forward_output (bool): If True, enables merging cached outputs into the dict returned by forward.
    """

    def __init__(self, backbone: nn.Module, module_names: list[str]):
        super().__init__()
        logging.info("Init of ReturnEmbedding module")
        logging.info(f"\t - {len(module_names)} module names")
        self.backbone = backbone
        self.module_names = module_names
        self.hooks = []
        self.embedding_cache = {}
        for name in self.module_names:
            module = self._get_module_by_name(backbone, name)
            if module is None:
                raise ValueError(f"Module '{name}' not found in backbone.")
            hook = module.register_forward_hook(self._make_hook(name, backbone))
            self.hooks.append(hook)

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs), self.embedding_cache

    def _make_hook(self, name, pl_module):
        def hook(module, input, output):
            self.embedding_cache[name] = output

        return hook

    def _get_module_by_name(self, pl_module, name):
        module = pl_module
        for attr in name.split("."):
            if not hasattr(module, attr):
                return None
            module = getattr(module, attr)
        return module


class TeacherStudentWrapper(nn.Module):
    """Backbone wrapper that implements teacher-student distillation via EMA.

    This is a wrapper for backbones that creates a teacher model as an exponential moving average (EMA) of the student model.
    It should be passed as the backbone to stable_pretraining.Module and accessed via
    forward_student() and forward_teacher() methods in your custom forward function.

    The teacher model is updated by taking a running average of the student's
    parameters and buffers. When `ema_coefficient == 0.0`, the teacher and student
    are literally the same object, saving memory but forward passes through the teacher
    will not produce any gradients.

    Usage example:
        backbone = ResNet18()
        wrapped_backbone = TeacherStudentWrapper(backbone)
        module = ssl.Module(
            backbone=wrapped_backbone,
            projector=projector,
            forward=forward_with_teacher_student,
            ...
        )

    Args:
        student (torch.nn.Module): The student model whose parameters will be tracked.
        warm_init (bool, optional): If True, performs an initialization step to match the student's parameters
            immediately. Default is True.
        base_ema_coefficient (float, optional): EMA decay factor at the start of training.
            This value will be updated following a cosine schedule.
            Should be in [0, 1]. A value of 0.0 means the teacher is fully
            updated to the student's parameters on every step, while a value of 1.0 means
            the teacher remains unchanged.
            Default is 0.996.
        final_ema_coefficient (float, optional): EMA decay factor at the end of training.
            Default is 1.
    """

    def __init__(
        self,
        student: nn.Module,
        warm_init: bool = True,
        base_ema_coefficient: float = 0.996,
        final_ema_coefficient: float = 1,
    ):
        if not (0.0 <= base_ema_coefficient <= 1.0) or not (
            0.0 <= final_ema_coefficient <= 1.0
        ):
            error_msg = (
                f"ema_coefficient must be in [0, 1]. Found: "
                f"base_ema_coefficient={base_ema_coefficient}, "
                f"final_ema_coefficient={final_ema_coefficient}."
            )
            logging.error(error_msg)
            raise ValueError(error_msg)

        super().__init__()
        self.student = student
        # Register EMA coefficients as buffers so they persist through checkpointing
        self.register_buffer("base_ema_coefficient", torch.tensor(base_ema_coefficient))
        self.register_buffer(
            "final_ema_coefficient", torch.tensor(final_ema_coefficient)
        )

        if self.base_ema_coefficient == 0.0 and self.final_ema_coefficient == 0.0:
            # No need to create a teacher network if the EMA coefficient is 0.0.
            self.teacher = student
            # Even when teacher == student, register the buffer for consistency
            self.register_buffer("ema_coefficient", self.base_ema_coefficient.clone())
        else:
            # Create a teacher network with the same architecture as the student.
            if isinstance(student, ReturnEmbedding):
                self.teacher = ReturnEmbedding(
                    copy.deepcopy(student.backbone), student.module_names
                )
            else:
                self.teacher = copy.deepcopy(student)
            self.teacher.requires_grad_(False)  # Teacher should not require gradients.

            if warm_init:  # Initialization step to match the student's parameters.
                # Temporarily set ema_coefficient to 0 for warm init
                self.register_buffer("ema_coefficient", torch.zeros(()))
                self.update_teacher()
                # Now set to base value after warm init
                self.ema_coefficient.copy_(self.base_ema_coefficient)
            else:
                self.register_buffer(
                    "ema_coefficient", self.base_ema_coefficient.clone()
                )

    @torch.no_grad
    def update_teacher(self):
        """Perform one EMA update step on the teacher’s parameters.

        The update rule is:
            teacher_param = ema_coefficient * teacher_param
            + (1 - ema_coefficient) * student_param

        This is done in a `no_grad` context to ensure the teacher’s parameters do
        not accumulate gradients, but the student remains fully trainable.

        Everything is updated, including buffers (e.g. batch norm running averages).
        """
        if self.ema_coefficient.item() == 0.0:
            return  # Nothing to update when the teacher is the student.
        elif self.ema_coefficient.item() == 1.0:
            return  # No need to update when the teacher is fixed.

        for teacher_group, student_group in [
            (self.teacher.parameters(), self.student.parameters()),
            (self.teacher.buffers(), self.student.buffers()),
        ]:
            for t, s in zip(teacher_group, student_group):
                ty = t.dtype
                t.mul_(self.ema_coefficient.to(dtype=ty))
                t.add_((1.0 - self.ema_coefficient).to(dtype=ty) * s)

    @torch.no_grad
    def update_ema_coefficient(self, epoch: int, total_epochs: int):
        """Update the EMA coefficient following a cosine schedule.

        The EMA coefficient is updated following a cosine schedule:
            ema_coefficient = final_ema_coefficient -
            0.5 * (final_ema_coefficient - base_ema_coefficient)
            * (1 + cos(epoch / total_epochs * pi))

        Args:
            epoch (int): Current epoch in the training loop.
            total_epochs (int): Total number of epochs in the training loop.
        """
        new_value = self.final_ema_coefficient - 0.5 * (
            self.final_ema_coefficient - self.base_ema_coefficient
        ) * (1 + math.cos(epoch / total_epochs * math.pi))
        # Update the buffer in-place to maintain persistence
        self.ema_coefficient.copy_(new_value)

    def forward_student(self, *args, **kwargs):
        """Forward pass through the student network. Gradients will flow normally."""
        return self.student(*args, **kwargs)

    def forward_teacher(self, *args, **kwargs):
        """Forward pass through the teacher network.

        By default, the teacher network does not require grad.
        If ema_coefficient == 0, then teacher==student,
        so we wrap in torch.no_grad() to ensure no gradients flow.
        """
        with torch.no_grad():
            return self.teacher(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass through either the student or teacher network.

        You can choose which model to run in the default forward.
        Commonly the teacher is evaluated, so we default to that.
        """
        return self.forward_teacher(*args, **kwargs)


def from_torchvision(model_name, low_resolution=False, **kwargs):
    """Load a backbone model.

    If num_classes is provided, the last layer is replaced by a linear layer of
    output size num_classes. Otherwise, the last layer is replaced by an identity layer.

    Args:
        model_name (str): Name of the backbone model. Supported models are:
            - Any model from torchvision.models
            - "Resnet9"
            - "ConvMixer"
        low_resolution (bool, optional): Whether to adapt the resolution of the model (for CIFAR typically).
            By default False.
        **kwargs: Additional keyword arguments for the model. Special handling:
            - in_channels (int): Number of input channels. If provided for ResNet models, the first
              conv layer will be modified to accept this many channels. Default is 3.

    Returns:
        torch.nn.Module: The neural network model.
    """
    # Extract in_channels before passing to torchvision (which doesn't accept it)
    in_channels = kwargs.pop("in_channels", 3)

    try:
        model = torchvision.models.__dict__[model_name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown model: {model_name}.")

    # Modify conv1 for custom number of input channels and/or low resolution
    if "resnet" in model_name and (in_channels != 3 or low_resolution):
        if low_resolution:
            # Low resolution: smaller kernel, stride=1, no maxpool (for CIFAR)
            model.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            )
            model.maxpool = nn.Identity()
        else:
            # Full resolution: keep original kernel/stride, just change in_channels
            model.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
    elif low_resolution and "resnet" not in model_name:
        logging.warning(f"Cannot adapt resolution for model: {model_name}.")

    # Handle num_classes parameter as documented
    num_classes = kwargs.get("num_classes", None)
    if num_classes is not None:
        # Replace the last layer with a linear layer of the specified size
        if hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, (nn.ModuleList, nn.Sequential)):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
    else:
        # Replace the last layer with an identity layer for feature extraction
        if hasattr(model, "fc"):
            model.fc = nn.Identity()
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, (nn.ModuleList, nn.Sequential)):
                model.classifier[-1] = nn.Identity()
            else:
                model.classifier = nn.Identity()

    return model


def from_huggingface(model_name, pretrained, attn_implementation="sdpa", **kwargs):
    """Loads a Hugging Face Transformers base model, optionally with pretrained weights, and returns the backbone model.

    This function wraps the Hugging Face `transformers` library to load a model specified by `model_name`.
    It supports loading either pretrained weights or initializing from configuration only. The returned object
    is the model's backbone (`model.base_model`), which is useful for extracting the core architecture
    without task-specific heads.

    Args:
        model_name (str): The Hugging Face model repository identifier or local path. Examples include
            "bert-base-uncased", "facebook/opt-1.3b", or a local directory containing model files.
        pretrained (bool): If True, loads pretrained weights via `AutoModel.from_pretrained`. If False,
            initializes the model from configuration only via `AutoConfig.from_pretrained` and
            `AutoModel.from_config`.
        attn_implementation (str, optional): The attention backend to use. Supported values include
            "sdpa" (default), "eager", "flash_attention_2", etc., as supported by the installed
            version of `transformers` and your hardware. This is forwarded to the underlying model
            constructor.
        **kwargs: Additional keyword arguments forwarded to `AutoModel.from_pretrained` or
            `AutoConfig.from_pretrained`. Common options include:
            - `revision` (str): Model version or branch to use.
            - `cache_dir` (str): Directory to cache downloaded models.
            - `trust_remote_code` (bool): Allow loading custom code from model repo.
            - `torch_dtype` (str or torch.dtype): Data type for model weights.
            - `device_map` (str or dict): Device placement for model parameters.
            - And others supported by Hugging Face Transformers.

    Returns:
        transformers.PreTrainedModel: The base (backbone) model instance, typically accessible via
        `model.base_model`. For some architectures, this may be the model itself.

    Raises:
        ImportError: If the `transformers` library is not installed.
        OSError: If the model or configuration cannot be found or downloaded.
        ValueError: If invalid arguments are provided.
        Exception: Propagates any other exceptions raised by Hugging Face Transformers.

    Notes:
        - The returned `base_model` may differ depending on the architecture. For some models,
          `base_model` is the same as the full model.
        - The availability of certain attention implementations (e.g., "flash_attention_2") depends
          on your hardware, installed libraries, and the version of `transformers`.
        - Ensure that your environment meets the requirements for the selected attention backend.

    Examples:
        >>> # Load a pretrained BERT model with default attention
        >>> model = from_huggingface("bert-base-uncased", pretrained=True)
        >>> # Initialize a model from config only, specifying a revision and device
        >>> model = from_huggingface(
        ...     "facebook/opt-1.3b",
        ...     pretrained=False,
        ...     revision="main",
        ...     device_map="auto",
        ... )
        >>> # Load a pretrained model using flash attention (if supported)
        >>> model = from_huggingface(
        ...     "meta-llama/Llama-2-7b-hf",
        ...     pretrained=True,
        ...     attn_implementation="flash_attention_2",
        ... )
    """
    from transformers import AutoModel, AutoConfig

    if pretrained:
        model = AutoModel.from_pretrained(
            model_name, attn_implementation=attn_implementation, **kwargs
        )
    else:
        config = AutoConfig.from_pretrained(model_name, **kwargs)
        model = AutoModel.from_config(
            config,
            attn_implementation=attn_implementation,
        )
    return model.base_model


def from_timm(model_name, low_resolution=False, **kwargs):
    import timm

    model = timm.create_model(model_name, **kwargs)
    if low_resolution:  # reduce resolution, for instance for CIFAR
        if "resnet" in model_name:
            in_channels = kwargs.get("in_channels", 3)
            model.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            )
            model.maxpool = nn.Identity()
        else:
            logging.warning(f"Cannot adapt resolution for model: {model_name}.")
    return model


def _map_shapes(obj: Any) -> Any:
    """Recursively maps a nested structure, replacing torch.Tensor objects with their .shape.

    We preserve the original structure for lists, tuples, dicts, sets, namedtuples, and dataclasses.
    Non-tensor objects are left unchanged.
    """
    import dataclasses

    if isinstance(obj, torch.Tensor):
        return obj.shape
    elif isinstance(obj, dict):
        return {k: _map_shapes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_map_shapes(v) for v in obj]
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
        return type(obj)(*(_map_shapes(v) for v in obj))
    elif isinstance(obj, tuple):
        return tuple(_map_shapes(v) for v in obj)
    elif isinstance(obj, set):
        return {_map_shapes(v) for v in obj}
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(
            obj,
            **{
                f.name: _map_shapes(getattr(obj, f.name))
                for f in dataclasses.fields(obj)
            },
        )
    else:
        return obj


def get_output_shape(model: torch.nn.Module, *inputs, **kwargs) -> Any:
    """Infers the output shapes of a PyTorch nn.Module by forwarding fake inputs on the 'meta' device using FakeTensorMode.

    Handles arbitrary nested output structures (lists, dicts, tuples, sets, namedtuples, dataclasses), preserving their
    structure but replacing torch.Tensor objects with their .shape.
    This function temporarily replaces the model's parameters and buffers with fake tensors on the 'meta' device,
    converts all tensor inputs and keyword arguments to 'meta', and runs the forward pass under FakeTensorMode.
    After execution, the original parameters and buffers are restored. No real computation or memory allocation occurs.

    Args:
        model (torch.nn.Module): The PyTorch module to evaluate. Must be on a real device (e.g., CPU).
        *inputs: Positional arguments to pass to the model's forward method. All torch.Tensor inputs are converted to 'meta'.
        **kwargs: Keyword arguments to pass to the model's forward method. All torch.Tensor values are converted to 'meta'.

    Returns:
        Any: The output structure from the model's forward pass, with all torch.Tensor objects replaced by their .shape.
             Non-tensor objects are left unchanged.

    Notes:
        - Supports nested output structures: dict, list, tuple, set, namedtuple, and dataclasses.
        - No real memory is allocated; all tensors are on the 'meta' device.
        - Not thread-safe: concurrent calls may interfere with parameter/buffer swapping.
        - Requires PyTorch 1.11+ for FakeTensorMode.
        - If the model contains custom buffers or state, ensure they are handled appropriately.
        - Raises exceptions if model forward fails or if parameters/buffers cannot be swapped.
        - Non-tensor outputs are returned unchanged.

    Example:
        shapes = get_output_shape_multi_input(model, input1, input2, key1=kwarg1)
        # shapes will have the same structure as the model's output, but with torch.Size in place of tensors.
    """
    from torch.nn.utils.stateless import functional_call
    import dataclasses

    # Try to use FakeTensorConverter if available (PyTorch 2.x+)
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensorConverter

        fake_mode = FakeTensorMode()
        converter = FakeTensorConverter()

        def to_fake(t):
            return converter.from_real_tensor(fake_mode, t)

    except ImportError:
        # Fallback: just use .to('meta') inside FakeTensorMode
        from torch._subclasses.fake_tensor import FakeTensorMode

        fake_mode = FakeTensorMode()

        def to_fake(t):
            return t.to("meta")

    # Prepare fake params and buffers
    params_and_buffers = dict(model.named_parameters())
    params_and_buffers.update(model.named_buffers())
    fake_params_and_buffers = {k: to_fake(v) for k, v in params_and_buffers.items()}

    # Recursively convert all tensor inputs/kwargs to fake/meta
    def convert_inputs(obj):
        if isinstance(obj, torch.Tensor):
            return to_fake(obj)
        elif isinstance(obj, dict):
            return {k: convert_inputs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_inputs(v) for v in obj]
        elif isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
            return type(obj)(*(convert_inputs(v) for v in obj))
        elif isinstance(obj, tuple):
            return tuple(convert_inputs(v) for v in obj)
        elif isinstance(obj, set):
            return {convert_inputs(v) for v in obj}
        elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.replace(
                obj,
                **{
                    f.name: convert_inputs(getattr(obj, f.name))
                    for f in dataclasses.fields(obj)
                },
            )
        else:
            return obj

    fake_inputs = [convert_inputs(inp) for inp in inputs]
    fake_kwargs = {k: convert_inputs(v) for k, v in kwargs.items()}
    with fake_mode:
        output = functional_call(
            model, fake_params_and_buffers, tuple(fake_inputs), fake_kwargs
        )
    return _map_shapes(output)


def set_embedding_dim(
    module,
    dim,
    bias=True,
    expected_input_shape: Optional[Union[tuple, list]] = None,
    expected_output_shape: Optional[Union[tuple, list]] = None,
):
    if isinstance(module, TimmWrapperModel):
        module = module.timm_model

    def embedder(in_features):
        return nn.Sequential(
            nn.Flatten(), nn.Linear(in_features, out_features=dim, bias=bias)
        )

    # For models like ResNet.
    if hasattr(module, "fc"):
        in_features = module.fc.in_features
        module.fc = embedder(in_features)
    # For modules like VGG or AlexNet.
    elif hasattr(module, "classifier"):
        if isinstance(module.classifier, nn.ModuleList) or isinstance(
            module.classifier, nn.Sequential
        ):
            in_features = module.classifier[-1].in_features
            module.classifier[-1] = embedder(in_features)
        else:
            in_features = module.classifier.in_features
            module.classifier = embedder(in_features)
    # For modules like ViT.
    elif hasattr(module, "heads"):
        in_features = module.heads.head.in_features
        module.heads.head = embedder(in_features)
    # For modules like Swin Transformer.
    elif hasattr(module, "head") and (
        ClassifierHead is None or not isinstance(module.head, ClassifierHead)
    ):
        in_features = module.head.in_features
        module.head = embedder(in_features)
    else:
        logging.warning(
            f"Unknown module structure for : '{module}'.\n\n"
            "We will use the default's output and attach a "
            "linear module on top."
        )
        if expected_input_shape is None:
            logging.error("Can't do that without `expected_input_shape`")
            raise ValueError("Can't do that without `expected_input_shape`")
        test_input = torch.empty(expected_input_shape, device="meta")
        out_shape = module.to("meta")(test_input)
        in_features = out_shape.flatten(1).size(1)
        embedder = nn.Sequential(
            nn.Flatten(), nn.Linear(in_features, out_features=dim, bias=bias)
        )
        return nn.Sequential(module, embedder)

    if expected_input_shape is None:
        logging.warning(
            "No `expected_input_shape` provided, can't verify"
            "the behavior of `set_emebdding_dim`"
        )
    else:
        assert expected_output_shape is not None
        x = torch.empty(expected_input_shape, device="meta")
        # Save original device before moving to meta
        original_device = next(module.parameters()).device
        out = module.to("meta")(x)
        if isinstance(out, tuple):
            assert out[0].shape == expected_output_shape
        elif hasattr(out, "logits"):
            assert out["logits"].shape == expected_output_shape
        else:
            assert out.shape == expected_output_shape
        # Move module back to original device
        # Use to_empty() for meta tensors which have no data
        module = module.to_empty(device=original_device)
    return module


def get_children_modules(
    model: nn.Module, parent_name: str, L: int = 1, partial_match: bool = False
) -> List[str]:
    """Extracts unique module names matching a given parent_name and L submodules.

    Args:
        model: The root nn.Module.
        parent_name: The string or path component to match (e.g., 'blocks').
        L: Number of levels after the parent_name to include in the result.
        partial_match: whether to check with == or in

    Returns:
        Sorted list of unique qualified module names at depth L after the parent_name.
    """
    result: List[str] = []
    for name, _ in model.named_modules():
        parts = name.split(".")
        matches = [
            i
            for i, p in enumerate(parts)
            if (parent_name in p if partial_match else parent_name == p)
        ]
        if not matches:
            continue
        for idx in matches:
            target_idx = idx + L
            if target_idx < len(parts):
                truncated = ".".join(parts[: target_idx + 1])
                if truncated in result:
                    continue
                # Ensure this is a valid submodule
                try:
                    model.get_submodule(truncated)
                    result.append(truncated)
                except AttributeError:
                    continue
            elif L == 0:
                truncated = ".".join(parts[: idx + 1])
                try:
                    model.get_submodule(truncated)
                    result.append(truncated)
                except AttributeError:
                    continue
    return result
