import pytest
import torch
from PIL import Image
import numpy as np

# Assume PatchMasking is defined in patch_masking.py
from stable_pretraining.data.transforms import PatchMasking


@pytest.mark.unit
@pytest.mark.parametrize("input_type", ["pil", "tensor_float", "tensor_uint8"])
@pytest.mark.parametrize("fill_value", [None, 0.5, 0.0, 1.0])
def test_patch_masking_transform(input_type, fill_value):
    # Create a dummy image (3x32x32)
    np_img = np.ones((32, 32, 3), dtype=np.uint8) * 255
    if input_type == "pil":
        img = Image.fromarray(np_img)
    elif input_type == "tensor_float":
        img = (
            torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
        )  # C, H, W, float
    elif input_type == "tensor_uint8":
        img = torch.from_numpy(np_img).permute(2, 0, 1)  # C, H, W, uint8
    sample = {"image": img}
    patch_size = 8
    drop_ratio = 0.5
    transform = PatchMasking(
        patch_size=patch_size,
        drop_ratio=drop_ratio,
        source="image",
        target="masked_image",
        fill_value=fill_value,
    )
    out = transform(sample)
    # Check output keys
    assert "masked_image" in out
    assert "patch_mask" in out
    # Check mask shape and dtype
    n_patches_h = 32 // patch_size
    n_patches_w = 32 // patch_size
    mask = out["patch_mask"]
    assert mask.shape == (n_patches_h, n_patches_w)
    assert mask.dtype == torch.bool
    # Check that masked_image is still an image of the same size and type
    masked_img = out["masked_image"]
    if input_type == "pil":
        assert isinstance(masked_img, Image.Image)
        assert masked_img.size == (32, 32)
        masked_img_tensor = (
            torch.from_numpy(np.array(masked_img)).permute(2, 0, 1).float() / 255.0
        )
    else:
        assert isinstance(masked_img, torch.Tensor)
        assert masked_img.shape[1:] == (32, 32)
        if masked_img.dtype == torch.uint8:
            masked_img_tensor = masked_img.float() / 255.0
        else:
            masked_img_tensor = masked_img
    # Determine expected mask value
    if fill_value is not None:
        expected_fill_value = fill_value
    elif input_type == "pil" or (
        input_type == "tensor_uint8" and masked_img_tensor.max() > 1.0
    ):
        expected_fill_value = 128 / 255.0
    else:
        expected_fill_value = 0.0
    # Check that at least one patch is masked and that masked patches have the correct value
    found_masked = False
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            h_start = i * patch_size
            w_start = j * patch_size
            patch = masked_img_tensor[
                :, h_start : h_start + patch_size, w_start : w_start + patch_size
            ]
            if not mask[i, j]:
                found_masked = True
                # All values in the patch should be close to the mask value
                assert torch.allclose(
                    patch, torch.full_like(patch, expected_fill_value), atol=1e-2
                )
            else:
                # Only check if the original image is not all fill_value
                if not np.isclose(expected_fill_value, 1.0, atol=1e-2):
                    assert not torch.allclose(
                        patch, torch.full_like(patch, expected_fill_value), atol=1e-2
                    )
    assert found_masked, "At least one patch should be masked"


def test_patch_masking_fill_value_mean(monkeypatch):
    # Test using the mean of the image as fill_value
    np_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
    sample = {"image": img}
    patch_size = 8
    drop_ratio = 1.0  # Mask all patches

    # Patch the transform to use the mean as fill_value
    class PatchMaskingMean(PatchMasking):
        def __call__(self, x):
            img = self.nested_get(x, self.source)
            img_tensor = self._to_tensor(img)
            mean_val = img_tensor.mean().item()
            self.fill_value = mean_val
            return super().__call__(x)

    transform = PatchMaskingMean(
        patch_size=patch_size,
        drop_ratio=drop_ratio,
        source="image",
        target="masked_image",
        fill_value=None,
    )
    out = transform(sample)
    masked_img = out["masked_image"]
    masked_img_tensor = (
        masked_img
        if isinstance(masked_img, torch.Tensor)
        else torch.from_numpy(np.array(masked_img)).permute(2, 0, 1).float() / 255.0
    )
    # All values should be close to the mean
    assert torch.allclose(
        masked_img_tensor,
        torch.full_like(masked_img_tensor, masked_img_tensor.mean()),
        atol=1e-2,
    )
