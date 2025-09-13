import copy
import math
from typing import Optional, Union

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
    from transformers import TimmWrapperModel
except ImportError:
    TimmWrapperModel = None


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
        self.base_ema_coefficient = torch.Tensor([base_ema_coefficient])[0]
        self.final_ema_coefficient = torch.Tensor([final_ema_coefficient])[0]

        if self.base_ema_coefficient == 0.0 and self.final_ema_coefficient == 0.0:
            # No need to create a teacher network if the EMA coefficient is 0.0.
            self.teacher = student
        else:
            # Create a teacher network with the same architecture as the student.
            self.teacher = copy.deepcopy(student)
            self.teacher.requires_grad_(False)  # Teacher should not require gradients.

            if warm_init:  # Initialization step to match the student’s parameters.
                self.ema_coefficient = torch.zeros(())
                self.update_teacher()

        self.ema_coefficient = self.base_ema_coefficient.clone()

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
        self.ema_coefficient = self.final_ema_coefficient - 0.5 * (
            self.final_ema_coefficient - self.base_ema_coefficient
        ) * (1 + math.cos(epoch / total_epochs * math.pi))

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
        **kwargs: Additional keyword arguments for the model.

    Returns:
        torch.nn.Module: The neural network model.
    """
    try:
        model = torchvision.models.__dict__[model_name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown model: {model_name}.")

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
