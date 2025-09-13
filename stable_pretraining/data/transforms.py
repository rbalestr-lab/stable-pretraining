from contextlib import contextmanager
from itertools import islice
from random import getstate, setstate
from random import seed as rseed
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torchvision
from PIL import ImageFilter
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2._utils import query_chw

from stable_pretraining.data.masking import multi_block_mask


class Transform(v2.Transform):
    """Base transform class extending torchvision v2.Transform with nested data handling."""

    def nested_get(self, v, name):
        if name == "":
            return v
        i = name.split(".")
        if i[0].isnumeric():
            i[0] = int(i[0])
        return self.nested_get(v[i[0]], ".".join(i[1:]))

    def nested_set(self, original, value, name):
        if "." not in name:
            if name.isnumeric():
                name = int(name)
            original[name] = value
        else:
            i = name.split(".")
            if i[0].isnumeric():
                i[0] = int(i[0])
            self.nested_set(original[i[0]], value, ".".join(i[1:]))

    def get_name(self, x):
        base = self.name
        assert "_" not in base
        if base not in x:
            return base
        ctr = 0
        while f"{base}_{ctr}" in base:
            ctr += 1
        return f"{base}_{ctr}"

    @property
    def name(self):
        return self.__class__.__name__


@torch.jit.unused
def to_image(
    input: Union[torch.Tensor, PIL.Image.Image, np.ndarray],
) -> tv_tensors.Image:
    """See :class:`~torchvision.transforms.v2.ToImage` for details."""
    if isinstance(input, np.ndarray):
        output = torch.from_numpy(np.atleast_3d(input)).transpose(-3, -1).contiguous()
    elif isinstance(input, PIL.Image.Image):
        output = torchvision.transforms.functional.pil_to_tensor(input)
    elif isinstance(input, torch.Tensor):
        output = input
    else:
        raise TypeError(
            f"Input can either be a pure Tensor, a numpy array, or a PIL image, but got {type(input)} instead."
        )
    return tv_tensors.Image(output)


class ToImage(Transform):
    """Convert input to image tensor with optional normalization."""

    def __init__(
        self,
        dtype=torch.float32,
        scale=True,
        mean=None,
        std=None,
        source: str = "image",
        target: str = "image",
    ):
        super().__init__()
        t = [to_image, v2.ToDtype(dtype, scale=scale)]
        if mean is not None and std is not None:
            t.append(v2.Normalize(mean=mean, std=std))
        self.t = v2.Compose(t)
        self.source = source
        self.target = target

    def __call__(self, x):
        self.nested_set(x, self.t(self.nested_get(x, self.source)), self.target)
        return x


class RandomGrayscale(Transform, v2.RandomGrayscale):
    """Randomly convert image to grayscale with given probability."""

    def __init__(self, p=0.1, source: str = "image", target: str = "image"):
        super().__init__(p)
        self.source = source
        self.target = target

    def _get_params(self, inp: List[Any]) -> Dict[str, Any]:
        num_input_channels, *_ = query_chw([inp])
        return dict(num_input_channels=num_input_channels)

    def __call__(self, x) -> Any:
        if self.p < 1 and torch.rand(1) >= self.p:
            x[self.get_name(x)] = False
            self.nested_set(x, self.nested_get(x, self.source), self.target)
            return x
        channels, *_ = query_chw([self.nested_get(x, self.source)])
        self.nested_set(
            x,
            F.rgb_to_grayscale(
                self.nested_get(x, self.source), num_output_channels=channels
            ),
            self.target,
        )
        x[self.get_name(x)] = True
        return x


class Lambda(Transform, v2.Lambda):
    """Applies a lambda callable to target key and store it in source."""

    def __init__(self, lambd, source: str = "image", target: str = "image"):
        super().__init__(lambd)
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        self.nested_set(x, super().__call__(x), self.target)
        return x


class RandomSolarize(Transform, v2.RandomSolarize):
    """Randomly solarize image by inverting pixel values above threshold."""

    def __init__(self, threshold, p=0.5, source: str = "image", target: str = "image"):
        super().__init__(threshold, p)
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        if self.p < 1 and torch.rand(1) >= self.p:
            x[self.get_name(x)] = False
            return x
        self.nested_set(
            x, F.solarize(self.nested_get(x, self.source), self.threshold), self.target
        )
        x[self.get_name(x)] = True
        return x


class GaussianBlur(Transform, v2.GaussianBlur):
    """Apply Gaussian blur to image with random sigma values."""

    _NAMES = ["sigma_x", "sigma_y"]

    def __init__(
        self,
        kernel_size,
        sigma=(0.1, 2.0),
        p=1,
        source: str = "image",
        target: str = "image",
    ):
        super().__init__(kernel_size, sigma)
        self.p = p
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        if self.p < 1 and torch.rand(1) >= self.p:
            x[self.get_name(x)] = torch.zeros((2,))
            return x
        params = self.make_params([])
        self.nested_set(
            x, self.transform(self.nested_get(x, self.source), params), self.target
        )
        x[self.get_name(x)] = torch.Tensor(params["sigma"])
        return x


class PILGaussianBlur(Transform):
    """PIL-based Gaussian blur transform with random sigma sampling."""

    _NAMES = ["sigma_x", "sigma_y"]

    def __init__(self, sigma=None, p=1, source: str = "image", target: str = "image"):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
            p (float): probability of applying the transform.
            source (str): source key in the data dictionary.
            target (str): target key in the data dictionary.
        """
        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma
        self.p = p
        self.source = source
        self.target = target

    def __call__(self, x):
        """Applies gaussian blur to an input image.

        Args:
            x (dict): Data dictionary containing the image to transform.

        Returns:
            dict: Data dictionary with blurred image.
        """
        if self.p < 1 and torch.rand(1) >= self.p:
            x[self.get_name(x)] = torch.zeros((1,))
            return x
        sigma = torch.rand((1,)) * (self.sigma[1] - self.sigma[0]) + self.sigma[0]
        x[self.get_name(x)] = sigma
        self.nested_set(
            x,
            self.nested_get(x, self.source).filter(
                ImageFilter.GaussianBlur(radius=sigma.item())
            ),
            self.target,
        )
        return x


class UniformTemporalSubsample(Transform):
    """``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.uniform_temporal_subsample``."""

    def __init__(
        self,
        num_samples: int,
        temporal_dim: int = -3,
        source: str = "video",
        target: str = "video",
    ):
        super().__init__(num_samples, temporal_dim)
        self.source = source
        self.target = target

    def forward(self, x: dict) -> torch.Tensor:
        self.nested_set(
            x, super().forward(self, self.nested_get(x, self.source)), self.target
        )
        return x


class RandomContiguousTemporalSampler(Transform):
    """Randomly sample contiguous frames from a video sequence."""

    def __init__(self, source, target, num_frames, frame_subsampling: int = 1):
        self.source = source
        self.target = target
        self.num_frames = num_frames
        self.frame_subsampling = frame_subsampling

    def __call__(self, x):
        metadata = self.nested_get(x, self.source).get_metadata()
        T = int(metadata["video"]["duration"][0] * metadata["video"]["fps"][0])
        covering = self.num_frames * self.frame_subsampling
        start = torch.randint(low=0, high=T - covering, size=(1,)).item()
        video_frames = []  # video frame buffer

        # Seek and return frames
        count = 0
        for frame in islice(
            self.nested_get(x, self.source).seek(start / metadata["video"]["fps"][0]),
            covering,
        ):
            if count % self.frame_subsampling == 0:
                video_frames.append(frame["data"])
            count += 1
        # Stack it into a tensor
        self.nested_set(x, torch.stack(video_frames, 0), self.target)
        x[self.get_name(x)] = start
        return x


class RGB(Transform, v2.RGB):
    """Convert image to RGB format."""

    def __init__(self, source: str = "image", target: str = "image"):
        super().__init__()
        self.source = source
        self.target = target

    def __call__(self, x):
        self.nested_set(
            x, F.grayscale_to_rgb(self.nested_get(x, self.source)), self.target
        )
        return x


class Resize(Transform, v2.Resize):
    """Resize image to specified size."""

    def __init__(
        self,
        size,
        interpolation=2,
        max_size=None,
        antialias=True,
        source="image",
        target="image",
    ) -> None:
        super().__init__(size, interpolation, max_size, antialias)
        self.source = source
        self.target = target

    def __call__(self, x):
        self.nested_set(
            x, self.transform(self.nested_get(x, self.source), []), self.target
        )
        return x


class ColorJitter(Transform, v2.ColorJitter):
    """Randomly change brightness, contrast, saturation, and hue of an image."""

    def __init__(
        self,
        brightness=None,
        contrast=None,
        saturation=None,
        hue=None,
        p=1,
        source: str = "image",
        target: str = "image",
    ):
        super().__init__(brightness, contrast, saturation, hue)
        self.p = p
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        if self.p < 1 and torch.rand(1) > self.p:
            self.nested_set(x, self.nested_get(x, self.source), self.target)
            x[self.get_name(x)] = torch.zeros(8)
            return x
        params = self.make_params([])
        self.nested_set(
            x, self.transform(self.nested_get(x, self.source), params), self.target
        )
        brightness_factor = params["brightness_factor"]
        contrast_factor = params["contrast_factor"]
        saturation_factor = params["saturation_factor"]
        hue_factor = params["hue_factor"]
        perm = params["fn_idx"].tolist()
        x[self.get_name(x)] = torch.Tensor(
            [brightness_factor, contrast_factor, saturation_factor, hue_factor] + perm
        )
        return x


class RandomRotation(Transform, v2.RandomRotation):
    """Rotate image by random angle within specified degrees range."""

    def __init__(
        self,
        degrees,
        interpolation=InterpolationMode.NEAREST,
        expand=False,
        center=None,
        fill=0,
        source: str = "image",
        target: str = "image",
    ):
        super().__init__(degrees, interpolation, expand, center, fill)
        self.source = source
        self.target = target

    def __call__(self, x):
        angle = self.make_params([])
        self.nested_set(
            x, self.transform(self.nested_get(x, self.source), angle), self.target
        )
        x[self.get_name(x)] = angle
        return x


class RandomChannelPermutation(Transform, v2.RandomChannelPermutation):
    """Randomly permute the channels of an image."""

    def __init__(self, source: str = "image", target: str = "image"):
        super().__init__()
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        num_channels, *_ = query_chw([self.nested_get(x, self.source)])
        perm = torch.randperm(num_channels)
        self.nested_set(
            x, F.permute_channels(self.nested_get(x, self.source), perm), self.target
        )
        x[self.get_name(x)] = perm
        return x


class RandomCrop(Transform, v2.RandomCrop):
    """Crop a random portion of image and resize it to given size."""

    _NAMES = ["needs_crop", "top", "left", "height", "width", "needs_pad", "padding"]

    def __init__(
        self,
        size,
        padding=None,
        pad_if_needed=False,
        fill=0,
        padding_mode="constant",
        source: str = "image",
        target: str = "image",
    ):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)
        self.source = source
        self.target = target

    def __call__(self, x):
        params = self.make_params([self.nested_get(x, self.source)])
        self.nested_set(
            x, self.transform(self.nested_get(x, self.source), params), self.target
        )
        values = []
        values.append(params["needs_crop"])
        values.append(params["top"])
        values.append(params["left"])
        values.append(params["height"])
        values.append(params["width"])
        values.append(params["needs_pad"])
        values.extend(params["padding"])
        x[self.get_name(x)] = torch.Tensor(values)
        return x


class RandomHorizontalFlip(Transform, v2.RandomHorizontalFlip):
    """Horizontally flip the given image randomly with a given probability."""

    def __init__(self, p=0.5, source: str = "image", target: str = "image"):
        super().__init__(p)
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        if self.p > 0 and torch.rand(1) < self.p:
            self.nested_set(
                x, F.horizontal_flip(self.nested_get(x, self.source)), self.target
            )
            x[self.get_name(x)] = True
        else:
            self.nested_set(x, self.nested_get(x, self.source), self.target)
            x[self.get_name(x)] = False
        return x


class RandomResizedCrop(Transform, v2.RandomResizedCrop):
    """Crop a random portion of image and resize it to given size."""

    _NAMES = ["top", "left", "height", "width"]

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
        source: str = "image",
        target: str = "image",
    ):
        super().__init__(size, scale, ratio, interpolation, antialias)
        self.source = source
        self.target = target

    def __call__(self, x):
        params = self.make_params([self.nested_get(x, self.source)])
        self.nested_set(
            x, self.transform(self.nested_get(x, self.source), params), self.target
        )
        values = []
        values.append(params["top"])
        values.append(params["left"])
        values.append(params["height"])
        values.append(params["width"])
        x[self.get_name(x)] = torch.Tensor(values)
        return x


class CenterCrop(Transform, v2.CenterCrop):
    """Crop the center of an image to the given size."""

    _NAMES = []

    def __init__(self, size, source: str = "image", target: str = "image"):
        super().__init__(size)
        self.source = source
        self.target = target

    def __call__(self, x):
        self.nested_set(
            x, self.transform(self.nested_get(x, self.source), []), self.target
        )
        return x


def set_seed(seeds):
    if hasattr(seeds[0], "__len__"):
        version, state, gauss = seeds[0]
        setstate((version, tuple(state), gauss))
    else:
        rseed(seeds[0])
    if hasattr(seeds[1], "__len__"):
        np.random.set_state(seeds[1])
    else:
        np.random.seed(seeds[1])
    if hasattr(seeds[2], "__len__"):
        torch.set_rng_state(seeds[2])
    else:
        torch.manual_seed(seeds[2])
    if len(seeds) == 4:
        if hasattr(seeds[3], "__len__"):
            torch.cuda.set_rng_state_all(seeds[3])
        else:
            torch.cuda.manual_seed(seeds[3])


@contextmanager
def random_seed(seed):
    seeds = [getstate(), np.random.get_state(), torch.get_rng_state()]
    if False:  # torch.cuda.is_available():
        seeds.append(torch.cuda.get_rng_state_all())
    new_seeds = [int(seed)] * len(seeds)
    set_seed(new_seeds)
    yield
    set_seed(seeds)


class ControlledTransform(Transform):
    """Face Landmarks dataset."""

    def __init__(
        self, transform: callable, seed_offset: int = 0, key: Optional[str] = "idx"
    ):
        super().__init__()
        self.seed_offset = seed_offset
        self._transform = transform
        self.key = key

    def __call__(self, x):
        with random_seed(x["idx"] + self.seed_offset):
            x = self._transform(x)
        return x


class Conditional(Transform):
    """Apply transform conditionally based on a data dictionary key."""

    def __init__(self, transform, condition_key, apply_on_true=True):
        super().__init__()
        self._transform = transform
        self.condition_key = condition_key
        self.apply_on_true = apply_on_true

    def __call__(self, x):
        if x[self.condition_key] and self.apply_on_true:
            return self._transform(x)
        elif not x[self.condition_key] and not self.apply_on_true:
            return self._transform(x)
        # if the transform is not applied we still inform the user
        # otherwise collate_fn will complain
        x[self._transform.get_name(x)] = self._transform.BYPASS_VALUE
        return x


class AdditiveGaussian(Transform):
    """Add Gaussian noise to input data."""

    BYPASS_VALUE = False

    def __init__(self, sigma, p=1):
        super().__init__()
        if not torch.is_tensor(sigma):
            sigma = torch.Tensor([sigma])[0]
        self.sigma = sigma
        self.p = p

    def __call__(self, x):
        if self.p == 0 or self.p < torch.rand(1):
            x[self.get_name(x)] = self.BYPASS_VALUE
            return x
        x[self.get_name(x)] = True
        out = torch.randn_like(x["image"]).mul_(self.sigma)
        x["image"] = x["image"].add_(out)
        return x


class Compose(v2.Transform):
    """Compose multiple transforms together in sequence."""

    def __init__(self, *args):
        super().__init__()
        self.args = args

    def __call__(self, sample):
        for a in self.args:
            sample = a(sample)
        return sample


class MultiViewTransform(v2.Transform):
    """Apply different transforms to different views of the same sample.

    When using RepeatedRandomSampler with n_views=2, this allows you to apply
    different augmentations to each view. It alternates between transforms
    based on a simple counter.

    Args:
        transforms: List of transforms, one for each view

    Example:
        transform = MultiViewTransform([
            strong_augmentation,  # Applied to first view
            weak_augmentation,    # Applied to second view
        ])
    """

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms
        self.n_transforms = len(transforms)
        self.counter = 0

    def __call__(self, sample):
        # Use round-robin to apply transforms
        transform_idx = self.counter % self.n_transforms
        self.counter += 1
        return self.transforms[transform_idx](sample)


class ContextTargetsMultiBlockMask(Transform):
    """Transform that adds multi-block masks to batch, with multiple target blocks and one disjoint context block.

    Args:
        patch_size: Size of the patch in patches
        num_blocks: Number of blocks to sample
        context_scale: Scale of the context block
        aspect_ratio: Aspect ratio of the blocks
        min_keep: Minimum number of patches that must be in the block

    """

    def __init__(
        self,
        patch_size=16,
        context_scale=(0.85, 1.0),
        context_aspect_ratio=(1.0, 1.0),
        target_scales=((0.15, 0.2),) * 4,
        target_aspect_ratios=((0.75, 1.5),) * 4,
        min_keep=10,
        source: str = "image",
        target_context: str = "mask_context",
        target_targets: str = "masks_target",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.context_scale = context_scale
        self.context_aspect_ratio = context_aspect_ratio
        self.target_scales = target_scales
        self.target_aspect_ratios = target_aspect_ratios
        self.source = source
        self.target_context = target_context
        self.target_targets = target_targets
        if len(target_scales) != len(target_aspect_ratios):
            raise ValueError(
                "Each scale must have its associated aspect ratio and vice versa.",
                "Received {len(target_scales)=} {len(target_aspect_ratios)=}",
            )

        self.min_keep = min_keep

    def __call__(self, x):
        source = self.nested_get(x, self.source)
        if isinstance(source, PIL.Image.Image):
            W, H = source.size  # PIL is W,H
        elif isinstance(source, torch.Tensor):
            # assumes H W
            H, W = source.shape[-2:]
        else:
            raise ValueError(
                f"Source must be a PIL.Image.Image or a torch.Tensor, but got {type(source)} instead."
            )

        scales = [self.context_scale, *self.target_scales]
        aspect_ratios = [self.context_aspect_ratio, *self.target_aspect_ratios]
        context_mask, *target_masks = multi_block_mask(
            H // self.patch_size,
            W // self.patch_size,
            block_scales=scales,
            aspect_ratios=aspect_ratios,
            min_keep=self.min_keep,
        )
        # makes targets disjoint with context
        for mask in target_masks:
            context_mask &= ~mask

        x[self.target_context] = torch.nonzero(context_mask.flatten()).squeeze()
        x[self.target_targets] = [
            torch.nonzero(mask.flatten()).squeeze() for mask in target_masks
        ]
        x[self.get_name(x)] = torch.tensor([scales, aspect_ratios])
        return x


class RandomMask(Transform):
    r"""Creates a random MAE-style mask for an image.

    This transform generates a random permutation of all patch indices for an
    input image. It then splits these indices into two disjoint sets:
    'visible' and 'masked', according to the specified `mask_ratio`.

    It also provides an `ids_restore` tensor, which can un-shuffle a sequence
    of patches back to its original 2D grid order. All outputs are added as
    new keys to the sample dictionary.

    Example:
        >>> # xdoctest: +SKIP
        >>> transform = RandomMask(patch_size=16, mask_ratio=0.75)
        >>> sample = {"image": torch.randn(3, 224, 224)}
        >>> result = transform(sample)
        >>> sorted(result.keys())
        ['image', 'ids_restore', 'len_keep', 'mask_masked', 'mask_visible']
        >>> result["len_keep"]
        49
        >>> result["mask_visible"].shape
        torch.Size([49])

    Args:
        patch_size (int): The height and width of each square patch.
        mask_ratio (float): The fraction of patches to be masked (e.g., 0.75).
        source (str): The key in the sample dict for the source image tensor.
        target_visible (str): The key to use when storing visible patch indices.
        target_masked (str): The key to use when storing masked patch indices.
        target_ids_restore (str): The key to use for the restoration indices.
        target_len_keep (str): The key to use for the count of visible patches.
    """

    def __init__(
        self,
        patch_size=16,
        mask_ratio=0.75,
        source: str = "image",
        target_visible: str = "mask_visible",
        target_masked: str = "mask_masked",
        target_ids_restore: str = "ids_restore",
        target_len_keep: str = "len_keep",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.source = source
        self.target_visible = target_visible
        self.target_masked = target_masked
        self.target_ids_restore = target_ids_restore
        self.target_len_keep = target_len_keep

    def __call__(self, x):
        source = self.nested_get(x, self.source)
        if isinstance(source, PIL.Image.Image):
            W, H = source.size  # PIL is W,H
        elif isinstance(source, torch.Tensor):
            # NOTE assumes _HW
            H, W = source.shape[-2:]
        else:
            raise ValueError(
                f"Source must be a PIL.Image.Image or a torch.Tensor, but got {type(source)} instead."
            )

        num_patches = (H // self.patch_size) * (W // self.patch_size)
        len_keep = int(num_patches * (1 - self.mask_ratio))

        # Generate random noise and shuffle indices (like MAE)
        noise = torch.rand(num_patches)
        ids_shuffle = torch.argsort(noise)
        ids_restore = torch.argsort(ids_shuffle)  # inverse permutation

        # Split into visible and masked
        mask_visible = ids_shuffle[:len_keep]  # first len_keep are visible
        mask_masked = ids_shuffle[len_keep:]  # rest are masked

        # Add to sample
        x[self.target_visible] = mask_visible
        x[self.target_masked] = mask_masked
        x[self.target_ids_restore] = (
            ids_restore  # NEW: for reconstructing full sequence
        )
        x[self.target_len_keep] = len_keep

        return x


# class RandomClassSwitch(v2.Transform):
#     def __init__(
#         self,
#         label_key: str,
#         new_key: str,
#         p: float,
#         low: int = -2147483648,
#         high: int = 0,
#     ):
#         super().__init__()
#         self.p = p
#         self.label_key = label_key
#         self.new_key = new_key
#         self.low = low
#         self.high = high

#     def __call__(self, sample: dict):
#         assert type(sample) is dict
#         assert self.label_key in sample
#         assert self.new_key not in sample
#         if self.p > 0 and torch.rand(1) < self.p:
#             if torch.is_tensor(sample[self.label_key]):
#                 sample[self.new_key] = torch.randint(
#                     low=self.low, high=self.high, size=()
#                 )
#             else:
#                 sample[self.new_key] = np.random.randint(low=self.low, high=self.high)
#         else:
#             sample[self.new_key] = sample[self.label_key]
#         return sample
