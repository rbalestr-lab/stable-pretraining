from contextlib import contextmanager
from itertools import islice
from random import getstate, setstate
from random import seed as rseed
import random
import re
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
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from PIL import Image

from stable_pretraining.data.masking import multi_block_mask


# ============================================================
# ===================== Images ===============================
# ============================================================


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


class Lambda(Transform):
    """Applies a lambda callable to target key and store it in source."""

    def __init__(self, lambd, source: str = "image", target: str = "image"):
        super().__init__()
        self.source = source
        self.target = target
        self.lambd = lambd

    def __call__(self, x) -> Any:
        self.nested_set(x, self.lambd(x), self.target)
        return x


class RoutingTransform(Transform):
    """Applies a routing callable to conditionally apply a transform from many candidates."""

    def __init__(self, router: callable, transforms: Union[list, tuple, dict]):
        self.router = router
        self.transforms = transforms

    def __call__(self, x) -> Any:
        route = self.router(x)
        return self.transforms[route](x)


class WrapTorchTransform(Transform, v2.Lambda):
    """Applies a lambda callable to target key and store it in source."""

    def __init__(self, transform, source: str = "image", target: str = "image"):
        super().__init__(transform)
        self.source = source
        self.target = target

    def __call__(self, x) -> Any:
        self.nested_set(
            x, super().__call__(self.nested_get(x, self.source)), self.target
        )
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


class PatchMasking(Transform):
    """Randomly masks square patches in an image, similar to patch masking used in Masked Signal Encoding (MSE) tasks.

    This transform operates on a dictionary input, applies patch masking to the image found at the specified `source` key,
    and writes the masked image to the `target` key. It also saves a boolean mask matrix (one entry per patch) to the
    `mask_key` in the dictionary, indicating which patches were masked (False) or kept (True).
    The output image remains in the same format as the input (PIL Image or Tensor), and the masking is performed efficiently
    for both input types.

    Args:
        patch_size (int): The size (in pixels) of each square patch to be masked.
        drop_ratio (float): The fraction of patches to randomly mask (set to the mask value).
        source (str): The key in the input dictionary from which to read the image.
        target (str): The key in the output dictionary to which the masked image will be written.
        mask_key (str): The key in the output dictionary to which the boolean patch mask will be written.
        mask_value (float, optional): The value to use for masked patches. If None, defaults to 0.0 for float tensors,
            and 128/255.0 for PIL images (mid-gray). Can be set to any float in [0,1] for normalized images.
    Input:
        A dictionary containing at least the key specified by `source`, whose value is a PIL Image or a torch.Tensor
        of shape (C, H, W) or (H, W).
    Output:
        The input dictionary, with two new keys:
            - `target`: The masked image (same type and shape as input).
            - `mask_key`: A boolean torch.Tensor of shape (num_patches_h, num_patches_w), where each entry is True if
              the patch is kept, False if it is masked.

    Example:
        >>> transform = PatchMasking(
        ...     patch_size=16,
        ...     drop_ratio=0.5,
        ...     source="image",
        ...     target="masked_image",
        ...     mask_key="patch_mask",
        ... )
        >>> sample = {"image": PIL_image_or_tensor}
        >>> out = transform(sample)
        >>> masked_img = out["masked_image"]
        >>> patch_mask = out["patch_mask"]
    """

    _NAMES = ["patch_mask"]

    def __init__(
        self,
        patch_size: int = 16,
        drop_ratio: float = 0.5,
        source: str = "image",
        target: str = "image",
        mask_value: float = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.drop_ratio = drop_ratio
        self.source = source
        self.target = target
        self.mask_value = mask_value

    def __call__(self, x):
        img = self.nested_get(x, self.source)
        img_tensor = self._to_tensor(img)
        C, H, W = img_tensor.shape

        # Compute number of patches
        n_patches_h = H // self.patch_size
        n_patches_w = W // self.patch_size
        total_patches = n_patches_h * n_patches_w

        # Generate mask
        mask = torch.rand(total_patches) > self.drop_ratio
        mask = mask.reshape(n_patches_h, n_patches_w)

        # Apply mask
        masked_img = img_tensor.clone()
        if self.mask_value is not None:
            mask_value = self.mask_value
        elif isinstance(img, Image.Image):
            mask_value = 128 / 255.0  # PIL images are converted to float [0,1]
        elif img_tensor.dtype == torch.uint8:
            mask_value = 128
        else:
            mask_value = 0.0
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                if not mask[i, j]:
                    h_start = i * self.patch_size
                    w_start = j * self.patch_size
                    masked_img[
                        :,
                        h_start : h_start + self.patch_size,
                        w_start : w_start + self.patch_size,
                    ] = mask_value

        # Convert back to PIL if needed
        if isinstance(img, Image.Image):
            masked_img_out = F.to_pil_image(masked_img)
        else:
            masked_img_out = masked_img

        self.nested_set(x, masked_img_out, self.target)
        x[self._NAMES[0]] = mask
        return x

    @staticmethod
    def _to_tensor(img):
        if isinstance(img, torch.Tensor):
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            if img.ndim == 3:
                return img
            elif img.ndim == 2:
                return img.unsqueeze(0)
        elif isinstance(img, Image.Image):
            return F.pil_to_tensor(img).float() / 255.0
        else:
            raise TypeError("Unsupported image type")


# Example usage:
# transform = PatchMasking(patch_size=16, drop_ratio=0.5, source="image", target="masked_image", mask_key="patch_mask")
# sample = {"image": PIL_image_or_tensor}
# out = transform(sample)
# out["masked_image"], out["patch_mask"]


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


class RoundRobinMultiViewTransform(v2.Transform):
    """Round-robin multi-view transform that cycles through transforms using a counter.

    IMPORTANT: This transform is designed to work with RepeatedRandomSampler, where
    each image index appears multiple times consecutively in the batch. It uses an
    internal counter to apply different augmentations to each repeated occurrence.

    BATCH SIZE NOTE: When using this with RepeatedRandomSampler, the batch_size
    parameter refers to the total number of augmented samples, NOT the number of
    unique images. For example, with batch_size=256 and n_views=2, you get 128
    unique images, each appearing twice with different augmentations.

    How it works:
    1. RepeatedRandomSampler produces indices like [0,0,1,1,2,2,...] (for n_views=2)
    2. DataLoader loads the same image multiple times
    3. This transform applies a different augmentation each time using round-robin

    Args:
        transforms: List of transforms, one for each view. The counter cycles
                   through these transforms in order.

    Example:
        # With RepeatedRandomSampler(dataset, n_views=2)
        transform = RoundRobinMultiViewTransform([
            strong_augmentation,  # Applied to 1st occurrence of each image
            weak_augmentation,    # Applied to 2nd occurrence of each image
        ])

    Warning: The internal counter makes this transform stateful and not thread-safe.
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


class MultiViewTransform(v2.Transform):
    """Creates multiple views from one sample by applying different transforms.

    Takes a single sample and applies different transforms to create multiple
    views, returning a list of complete sample dicts. Preserves all modifications
    each transform makes (masks, augmentation params, metadata, etc.).

    Implementation Note:
        This transform uses shallow copy (dict.copy()) for the input sample before
        applying each transform. This is efficient and safe because:
        - The shallow copy shares references to the original tensors/objects
        - Standard transforms create NEW tensors (e.g., through mul(), resize(),
          crop()) rather than modifying inputs in-place
        - The original sample remains unchanged

    Consequences of shallow copy:
        - Memory efficient: Original tensors are not duplicated unnecessarily
        - Safe with torchvision transforms: All torchvision transforms and our
          custom transforms follow the pattern of creating new tensors
        - Caution: If using custom transforms that modify tensors in-place (using
          operations like mul_(), add_() with underscore), views may interfere with
          each other. Always use non-in-place operations in custom transforms.

    Args:
        transforms: Either a list or dict of transforms.
                   - List: Returns a list of views in the same order
                   - Dict: Returns a dict of views with the same keys

    Returns:
        Union[List[dict], Dict[str, dict]]:
            - If transforms is a list: Returns a list of transformed sample dicts
            - If transforms is a dict: Returns a dict of transformed sample dicts with same keys
            Each dict contains NEW tensors, not references to the original.

    Example:
        # List input - returns list of views
        transform = MultiViewTransform([
            strong_augmentation,  # Creates first view with strong aug
            weak_augmentation,    # Creates second view with weak aug
        ])
        # Input: {"image": img, "label": 0}
        # Output: [{"image": img_strong, "label": 0}, {"image": img_weak, "label": 0}]

        # Dict input - returns dict of named views
        transform = MultiViewTransform({
            "student": strong_augmentation,
            "teacher": weak_augmentation,
        })
        # Input: {"image": img, "label": 0}
        # Output: {"student": {"image": img_strong, "label": 0},
        #          "teacher": {"image": img_weak, "label": 0}}
    """

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms
        self.return_dict = isinstance(transforms, dict)

    def __call__(self, sample):
        """Create multiple views by applying different transforms to the sample."""
        if self.return_dict:
            # Dict input - return dict of views
            views = {}
            for key, transform in self.transforms.items():
                # Copy to avoid transforms modifying the original
                sample_copy = sample.copy()
                # Apply transform to entire dict
                transformed = transform(sample_copy)
                views[key] = transformed
        else:
            # List input - return list of views
            views = []
            for transform in self.transforms:
                # Copy to avoid transforms modifying the original
                sample_copy = sample.copy()
                # Apply transform to entire dict
                transformed = transform(sample_copy)
                views.append(transformed)

        return views


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

# -------------------------------------------------------------------------------------------------------------
# Spurious Text Transforms


class AddSampleIdx(Transform):
    """Add an "idx" key each sample to allow for deterministic injection."""

    def __init__(self):
        super().__init__()
        self._counter = 0

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "idx" not in x:
            x["idx"] = self._counter
            self._counter += 1

        return x


class ClassConditionalInjector(Transform):
    """Applies transformations conditionally based on sample label.

    Args:
        transformation (Transform): Transform to apply to the image.
        label_key (str): Key for label in the sample dict.
        target_labels (Union[int, list[int]]): Which labels to modify.
        proportion (float): Fraction of samples with matching labels to modify (0-1).
        total_samples (int, optional): Dataset size (for deterministic mask).
        seed (int): Seed for randomization to determine which samples transformation is applied to
    """

    def __init__(
        self,
        transformation: Transform,
        label_key: str = "label",
        target_labels: Union[int, list[int]] = 0,
        proportion: float = 0.5,
        total_samples: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.transformation = transformation
        self.label_key = label_key
        self.target_labels = (
            [target_labels] if isinstance(target_labels, int) else target_labels
        )
        self.proportion = proportion
        self.total_samples = total_samples
        self.seed = seed

        # Precompute deterministic mask if dataset size known
        if total_samples is not None:
            num_to_transform = int(total_samples * proportion)
            rng = torch.Generator().manual_seed(seed)
            self.indices_to_transform = set(
                torch.randperm(total_samples, generator=rng)[:num_to_transform].tolist()
            )
        else:
            self.indices_to_transform = None

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        label = self.nested_get(x, self.label_key)

        # Determine if we apply the transformation
        should_transform = False
        idx = self.nested_get(x, "idx")
        if label in self.target_labels:
            if self.indices_to_transform is not None:
                should_transform = idx in self.indices_to_transform
            else:
                should_transform = random.random() < self.proportion

        if should_transform:
            x = self.transformation(x)

        return x


class SpuriousTextInjection(Transform):
    """Injects spurious tokens into text for specific target classes.

    Args:
        text_key (str): The name of the key representing the text in the dataset
        class_key (str): The name of the key representing the label in the dataset
        class_target (int): The label to have the spurious correlation injected into
        file_path (str): The path of the file to inject spurious correlations from
        p (float): The proportion of samples to inject the spurious token into
        location (str): The location of the text to inject the spurious token(s) into
        token_proportion (float): The proportion of the original tokens available in the dataset to inject as spurious tokens
            (used to determine the number injected per sample)
        seed (int): Seed for reproducibility
    """

    def __init__(
        self,
        text_key: str,
        file_path: str,
        location: str = "random",
        token_proportion: float = 0.1,
        seed: int = None,
    ):
        self.text_key = text_key
        self.location = location
        self.token_proportion = token_proportion
        self.base_seed = seed
        # store RNG per idx
        self.rngs = {}

        with open(file_path, "r", encoding="utf-8") as f:
            self.items = [line.strip() for line in f if line.strip()]

        assert self.items, f"No valid lines found in {file_path}"
        assert 0 <= self.token_proportion <= 1, "token_proportion must be in [0, 1]"
        assert self.location in {"beginning", "random", "end"}

    def _get_rng(self, idx):
        if idx not in self.rngs:
            seed = self.base_seed + idx if self.base_seed is not None else None
            self.rngs[idx] = random.Random(seed)
        return self.rngs[idx]

    def _inject(self, text: str, rng: random.Random) -> str:
        words = text.split()
        num_tokens = len(words)
        num_to_inject = max(1, int(num_tokens * self.token_proportion))
        injections = [rng.choice(self.items) for _ in range(num_to_inject)]

        if self.location == "beginning":
            words = injections + words
        elif self.location == "end":
            words = words + injections
        elif self.location == "random":
            for inj in injections:
                pos = rng.randint(0, len(words))
                words.insert(pos, inj)
        return " ".join(words)

    def __call__(self, x: dict) -> dict:
        text = x[self.text_key]

        # Deterministic RNG per call
        if self.base_seed is not None:
            idx = x.get("idx", 0)
            rng = self._get_rng(idx)
        else:
            rng = random.Random()

        x[self.text_key] = self._inject(text, rng)
        return x


class HTMLInjection(Transform):
    """Injects HTML-like tags into text fields (deterministically if 'idx' present).

    This transform adds artificial HTML tokens to text data, optionally at a specific
    HTML nesting level or a random position. Supports deterministic per-sample
    injection when used with AddSampleIdx and ClassConditionalInjector.

    Args:
        text_key (str): Key for the text field in the dataset sample.
        file_path (str): Path to file containing HTML tags (each line = one tag or tag pair).
        location (str): Where to inject tags ("beginning", "end", or "random").
        level (int, optional): Target HTML nesting level to inject within.
        token_proportion (float, optional): The proportion of the original tokens available in the dataset
             to inject as spurious tokens (used to determine the number injected per sample)
        seed (int, optional): Random seed for reproducibility.
    """

    def __init__(
        self,
        text_key: str,
        file_path: str,
        location: str = "random",
        level: Optional[int] = None,
        token_proportion: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.text_key = text_key
        self.location = location
        self.level = level
        self.token_proportion = token_proportion
        self.base_seed = seed if seed is not None else 0
        self.rng = random.Random(seed)

        with open(file_path, "r", encoding="utf-8") as f:
            self.tags = [line.strip() for line in f if line.strip()]

        assert self.tags, f"No valid tags found in {file_path}"
        if token_proportion is not None:
            assert 0 < token_proportion <= 1, "token_proportion must be between 0 and 1"
        assert self.location in {"beginning", "end", "random"}, "invalid location"

    # ---------- Internal helpers ----------
    def _choose_tag(self, rng):
        """Select an opening/closing tag pair."""
        line = rng.choice(self.tags)
        parts = line.split()
        if len(parts) >= 2:
            return parts[0], parts[1]
        else:
            return parts[0], None

    def _inject_with_tags(self, tokens, opening, closing, location, rng):
        """Inject the a single tag into the text."""
        new_tokens = tokens[:]
        if location == "beginning":
            new_tokens.insert(0, opening)
            if closing:
                pos = rng.randint(1, len(new_tokens))
                new_tokens.insert(pos, closing)
        elif location == "end":
            pos = rng.randint(0, len(new_tokens))
            new_tokens.insert(pos, opening)
            if closing:
                new_tokens.append(closing)
        elif location == "random":
            pos_open = rng.randint(0, len(new_tokens))
            new_tokens.insert(pos_open, opening)
            if closing:
                pos_close = rng.randint(pos_open + 1, len(new_tokens))
                new_tokens.insert(pos_close, closing)
        return new_tokens

    def _inject(self, text, rng):
        """Overall injection of all tags into the text."""
        tokens = text.split()
        if not tokens:
            return text

        if self.token_proportion is None:
            opening, closing = self._choose_tag(rng)
            tokens = self._inject_with_tags(
                tokens, opening, closing, self.location, rng
            )
        else:
            n = len(tokens)
            num_insertions = max(1, int(n * self.token_proportion))
            for _ in range(num_insertions):
                opening, closing = self._choose_tag(rng)
                tokens = self._inject_with_tags(
                    tokens, opening, closing, self.location, rng
                )
        return " ".join(tokens)

    def _inject_at_level(self, text, level, rng):
        """Inject tags inside a specific HTML nesting level."""
        tag_regex = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)[^>]*>")
        stack = []
        for match in tag_regex.finditer(text):
            tag_str = match.group(0)
            tag_name = match.group(1)
            if not tag_str.startswith("</"):
                stack.append((tag_name, match.end()))
            else:
                if stack:
                    open_tag, start_index = stack.pop()
                    if len(stack) == level - 1:
                        start, end = start_index, match.start()
                        target = text[start:end]
                        injected = self._inject(target, rng)
                        return text[:start] + injected + text[end:]
        return self._inject(text, rng)

    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Main call function for the transformation."""
        text = x[self.text_key]

        # Deterministic per-sample RNG if idx available
        if "idx" in x:
            seed = self.base_seed + int(x["idx"])
            rng = random.Random(seed)
        # fallback (non-deterministic but seeded globally)
        else:
            rng = self.rng

        if self.level is None:
            x[self.text_key] = self._inject(text, rng)
        else:
            x[self.text_key] = self._inject_at_level(text, self.level, rng)

        return x


# -------------------------------------------------------------------------------------------------------------
# Spurious Image Transforms


class AddPatch(Transform):
    """Add a solid color patch to an image at a fixed position.

    Args:
        patch_size (float): Fraction of image width/height for the patch (0 < patch_size ≤ 1).
        color (Tuple[float, float, float]): RGB values in [0, 1].
        position (str): Where to place the patch: 'top_left_corner', 'top_right_corner',
                        'bottom_left_corner', 'bottom_right_corner', 'center'.
    """

    def __init__(
        self,
        patch_size: float = 0.1,
        color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        position: str = "bottom_right_corner",
    ):
        super().__init__()

        # checking constraints
        if patch_size <= 0 or patch_size > 1:
            raise ValueError("patch_size must be between 0 and 1.")

        if len(color) != 3:
            raise ValueError(
                "color must be a tuple of size 3 in the form \
             Tuple[float, float, float]) with each representing RGB values in [0, 1]"
            )

        for value in color:
            if value > 1 or value < 0:
                raise ValueError("Each color value must be in [0, 1]")

        self.patch_size = patch_size
        self.color = color
        self.position = position

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        img = self.nested_get(x, "image")
        _, H, W = img.shape

        patch_h = int(H * self.patch_size)
        patch_w = int(W * self.patch_size)

        # Create a colored patch
        patch = torch.zeros((3, patch_h, patch_w), device=img.device)
        patch[0] = self.color[0]
        patch[1] = self.color[1]
        patch[2] = self.color[2]

        img = img.clone()
        if self.position == "top_left_corner":
            img[:, :patch_h, :patch_w] = patch
        elif self.position == "top_right_corner":
            img[:, :patch_h, -patch_w:] = patch
        elif self.position == "bottom_left_corner":
            img[:, -patch_h:, :patch_w] = patch
        elif self.position == "bottom_right_corner":
            img[:, -patch_h:, -patch_w:] = patch
        elif self.position == "center":
            center_y, center_x = H // 2, W // 2
            img[
                :,
                center_y - patch_h // 2 : center_y + patch_h // 2,
                center_x - patch_w // 2 : center_x + patch_w // 2,
            ] = patch
        else:
            raise ValueError(
                f"Invalid position: {self.position}, valid positions are: \
             top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner, center"
            )

        self.nested_set(x, img, "image")
        return x


class AddColorTint(Transform):
    """Adds a color tint to the overall image (additive tint).

    Args:
        tint (Tuple[float, float, float]): RGB representation of the tint that will be applied to the overall image
        alpha (Float): mixing ratio for how much to blend the new color with the existing image
    """

    def __init__(
        self, tint: Tuple[float, float, float] = (1.0, 0.8, 0.8), alpha: float = 0.3
    ):
        super().__init__()
        self.tint = torch.tensor(tint).view(3, 1, 1)
        self.alpha = alpha

    def __call__(self, x):
        img = self.nested_get(x, "image")
        img = torch.clamp(img * (1 - self.alpha) + self.tint * self.alpha, 0, 1)
        self.nested_set(x, img, "image")
        return x


class AddBorder(Transform):
    """Adds a border around an image.

    Args:
        thickness (Float): how thick the border around the image will be
        color (Tuple[float, float, float]): RGB representation of the color of the border
    """

    def __init__(
        self, thickness: float = 0.05, color: Tuple[float, float, float] = (0, 1, 0)
    ):
        super().__init__()
        self.thickness = thickness
        self.color = color

    def __call__(self, x):
        img = self.nested_get(x, "image").clone()
        _, H, W = img.shape

        # scale to match image size
        t = int(min(H, W) * self.thickness)
        color_tensor = torch.tensor(self.color, device=img.device).view(3, 1, 1)

        img[:, :t, :] = color_tensor
        img[:, -t:, :] = color_tensor
        img[:, :, :t] = color_tensor
        img[:, :, -t:] = color_tensor
        self.nested_set(x, img, "image")

        return x


class AddWatermark(Transform):
    """Overlay another image (logo, emoji, etc.) onto the base image.

    Args:
        watermark_path (str): Path to the watermark image (e.g. 'smile.png').
        size (float): Fraction of base image size to scale watermark.
        position (str): One of ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'center'].
        alpha (float): Opacity of watermark (0-1).
    """

    def __init__(self, watermark_path, size=0.2, position="bottom_right", alpha=0.8):
        super().__init__()
        # [C,H,W] tensor in [0,1]
        self.watermark = read_image(watermark_path).float() / 255.0
        self.size = size
        self.position = position
        self.alpha = alpha

    def __call__(self, x):
        img = self.nested_get(x, "image").clone()
        _, H, W = img.shape

        # Resize watermark
        w_h, w_w = self.watermark.shape[1:]
        target_h = int(H * self.size)
        target_w = int(w_w / w_h * target_h)
        wm = resize(self.watermark, [target_h, target_w])

        # Compute position
        if self.position == "top_left":
            y0, x0 = 0, 0
        elif self.position == "top_right":
            y0, x0 = 0, W - target_w
        elif self.position == "bottom_left":
            y0, x0 = H - target_h, 0
        elif self.position == "bottom_right":
            y0, x0 = H - target_h, W - target_w
        elif self.position == "center":
            y0, x0 = (H - target_h) // 2, (W - target_w) // 2
        else:
            raise ValueError(f"Unknown position: {self.position}")

        background_region = img[:, y0 : y0 + target_h, x0 : x0 + target_w]
        img[:, y0 : y0 + target_h, x0 : x0 + target_w] = (
            background_region * (1 - self.alpha) + wm * self.alpha
        )

        self.nested_set(x, img, "image")
        return x
