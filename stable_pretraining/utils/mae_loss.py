import torch


def unpatchify(patches, patch_size=16, channels=3, height=None, width=None):
    """Reconstruct image from patches (inverse of patchify).

    :param patches: Patch tokens, shape (N, T, patch_size² × C)
    :param patch_size: Size of each patch (default: 16)
    :param channels: Number of image channels (default: 3)
    :param height: Target image height. If None, assumes square image.
    :param width: Target image width. If None, assumes square image.
    :return: Reconstructed images, shape (N, C, H, W)
    """
    p = patch_size
    C = channels

    # =========================================================================
    # Input validation
    # =========================================================================

    assert patches.ndim == 3, (
        f"patches must be 3D (N, T, D), got {patches.ndim}D with shape {patches.shape}"
    )

    N, T, D = patches.shape
    pixels_per_patch = p * p * C

    assert D == pixels_per_patch, (
        f"patches last dim {D} != expected {pixels_per_patch} "
        f"(patch_size² × channels = {p}² × {C} = {pixels_per_patch})"
    )

    # Infer grid dimensions
    if height is None and width is None:
        # Assume square image
        grid_size = int(T**0.5)
        assert grid_size**2 == T, (
            f"Cannot infer square grid: num_patches {T} is not a perfect square. "
            f"Please provide height and width explicitly."
        )
        num_patches_h = num_patches_w = grid_size
    else:
        assert height is not None and width is not None, (
            "Must provide both height and width, or neither (for square images)"
        )
        assert height % p == 0, f"height {height} must be divisible by patch_size {p}"
        assert width % p == 0, f"width {width} must be divisible by patch_size {p}"
        num_patches_h = height // p
        num_patches_w = width // p
        assert num_patches_h * num_patches_w == T, (
            f"height {height} × width {width} with patch_size {p} gives "
            f"{num_patches_h}×{num_patches_w}={num_patches_h * num_patches_w} patches, "
            f"but got {T} patches"
        )

    H = num_patches_h * p
    W = num_patches_w * p

    # =========================================================================
    # Step 1: Reshape to grid of patches
    # =========================================================================
    # patches: (N, T, p*p*C)
    # → (N, num_patches_h, num_patches_w, p, p, C)
    imgs = patches.reshape(N, num_patches_h, num_patches_w, p, p, C)

    # =========================================================================
    # Step 2: Rearrange to image format
    # =========================================================================
    # (N, num_patches_h, num_patches_w, p, p, C)
    # → (N, C, num_patches_h, p, num_patches_w, p)
    # → (N, C, H, W)
    imgs = imgs.permute(0, 5, 1, 3, 2, 4)  # (N, C, nh, p, nw, p)
    imgs = imgs.reshape(N, C, H, W)

    # Sanity check
    assert imgs.shape == (N, C, H, W), (
        f"Output shape {imgs.shape} != expected {(N, C, H, W)}. "
        "This should never happen — please report this bug."
    )

    return imgs


def mae_loss(pred, imgs, mask, patch_size=16, mask_only=True, patch_normalize=True):
    """Compute MAE reconstruction loss with per-patch normalization.

    :param pred: Decoder predictions, shape (N, T, patch_size² × C)
    :param imgs: Original images, shape (N, C, H, W)
    :param mask: Binary mask, shape (N, T), where 1 = masked (compute loss), 0 = visible (ignore)
    :param patch_size: Size of each patch (default: 16)
    :param mask_only: If True, compute loss only on masked patches. If False, compute on all patches. (default: True)
    :param patch_normalize: If True, normalize each patch to zero mean/unit var. If False, use raw pixels. (default: True)
    :return: Scalar loss value
    """
    p = patch_size

    # =========================================================================
    # Input validation
    # =========================================================================

    # Check imgs shape
    assert imgs.ndim == 4, (
        f"imgs must be 4D (N, C, H, W), got {imgs.ndim}D with shape {imgs.shape}"
    )
    N, C, H, W = imgs.shape

    # Check image dimensions are divisible by patch_size
    assert H % p == 0, (
        f"Image height {H} must be divisible by patch_size {p}, got remainder {H % p}"
    )
    assert W % p == 0, (
        f"Image width {W} must be divisible by patch_size {p}, got remainder {W % p}"
    )

    # Compute expected number of patches
    num_patches_h = H // p
    num_patches_w = W // p
    T_expected = num_patches_h * num_patches_w
    pixels_per_patch = p * p * C

    # Check pred shape
    assert pred.ndim == 3, (
        f"pred must be 3D (N, T, D), got {pred.ndim}D with shape {pred.shape}"
    )
    assert pred.shape[0] == N, f"pred batch size {pred.shape[0]} != imgs batch size {N}"
    assert pred.shape[1] == T_expected, (
        f"pred num_patches {pred.shape[1]} != expected {T_expected} "
        f"(image {H}×{W} with patch_size {p} → {num_patches_h}×{num_patches_w} = {T_expected} patches)"
    )
    assert pred.shape[2] == pixels_per_patch, (
        f"pred last dim {pred.shape[2]} != expected {pixels_per_patch} "
        f"(patch_size² × channels = {p}² × {C} = {pixels_per_patch})"
    )

    # Check mask shape
    assert mask.ndim == 2, (
        f"mask must be 2D (N, T), got {mask.ndim}D with shape {mask.shape}"
    )
    assert mask.shape[0] == N, f"mask batch size {mask.shape[0]} != imgs batch size {N}"
    assert mask.shape[1] == T_expected, (
        f"mask num_patches {mask.shape[1]} != expected {T_expected}"
    )

    # Check mask values
    assert mask.min() >= 0 and mask.max() <= 1, (
        f"mask values must be in [0, 1], got min={mask.min()}, max={mask.max()}"
    )

    # Check at least one masked patch (only if mask_only=True)
    if mask_only:
        num_masked = mask.sum().item()
        assert num_masked > 0, (
            "mask has no masked patches (all zeros). "
            "Need at least one masked patch to compute loss when mask_only=True. "
            "Set mask_only=False to compute loss on all patches."
        )

    # Check devices match
    assert pred.device == imgs.device, (
        f"pred device {pred.device} != imgs device {imgs.device}"
    )
    assert mask.device == imgs.device, (
        f"mask device {mask.device} != imgs device {imgs.device}"
    )

    # =========================================================================
    # Step 1: Patchify the image
    # =========================================================================
    # imgs: (N, C, H, W)
    #
    # unfold(dim, size, step) extracts sliding windows along a dimension
    #   - dim=2 (height): extract patches of size p with step p (non-overlapping)
    #   - dim=3 (width):  extract patches of size p with step p (non-overlapping)
    #
    # After unfold(2, p, p): (N, C, H//p, W, p)     — extracted p-sized chunks along H
    # After unfold(3, p, p): (N, C, H//p, W//p, p, p) — extracted p-sized chunks along W
    target = imgs.unfold(2, p, p).unfold(3, p, p)

    # Rearrange to (N, num_patches, patch_pixels)
    #   (N, C, H//p, W//p, p, p)
    #   → permute to (N, H//p, W//p, p, p, C)  — move spatial grid first, channels last
    #   → reshape to (N, T, p*p*C) where T = (H//p)*(W//p) = num_patches
    target = target.permute(0, 2, 3, 4, 5, 1).reshape(N, -1, pixels_per_patch)

    # Sanity check after patchify
    assert target.shape == pred.shape, (
        f"target shape {target.shape} != pred shape {pred.shape} after patchify. "
        "This should never happen — please report this bug."
    )

    # =========================================================================
    # Step 2: Per-patch normalization (optional)
    # =========================================================================
    # Normalize each patch independently to zero mean and unit variance.
    # This forces the model to predict texture/structure rather than mean color.
    #
    # mean/var computed over last dim (the 768 pixel values within each patch)
    if patch_normalize:
        mean = target.mean(dim=-1, keepdim=True)  # (N, T, 1)
        var = target.var(dim=-1, keepdim=True)  # (N, T, 1)
        target = (target - mean) / (var + 1e-6).sqrt()

    # =========================================================================
    # Step 3: Compute MSE loss
    # =========================================================================
    # pred: (N, T, 768) — decoder predictions (normalized if patch_normalize=True)
    # target: (N, T, 768) — ground truth (normalized if patch_normalize=True)
    loss = (pred - target) ** 2  # (N, T, 768)
    loss = loss.mean(dim=-1)  # (N, T) — average over pixels within each patch

    # Apply mask or compute on all patches
    if mask_only:
        # Only compute loss where mask=1 (masked/reconstructed patches)
        # Visible patches (mask=0) are ignored — no loss signal from them
        loss = (loss * mask).sum() / mask.sum()  # scalar
    else:
        # Compute loss on all patches (ignores mask)
        loss = loss.mean()  # scalar

    # Final sanity check
    assert not torch.isnan(loss), (
        "Loss is NaN. Check for zero-variance patches or extreme values in inputs."
    )
    assert not torch.isinf(loss), (
        "Loss is Inf. Check for extreme values in pred or imgs."
    )

    return loss
