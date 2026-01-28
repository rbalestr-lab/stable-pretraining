import torch
import torch.nn as nn
from functools import partial
from pathlib import Path
import numpy as np
import logging
import importlib.util
import stable_pretraining as spt

dataset_configs_path = Path(__file__).parent / "dataset_configs.py"
spec = importlib.util.spec_from_file_location("dataset_configs_module", dataset_configs_path)
dataset_configs_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_configs_module)
get_dataset_config_fn = dataset_configs_module.get_dataset_config


DECODER_DIM_MAP = {
    "tiny": 192,
    "small": 384,
    "base": 512,
    "large": 768,
}


def parse_decoder_type(decoder_type_str):
    if decoder_type_str == "linear":
        return {"type": "linear"}

    parts = decoder_type_str.split("-")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid decoder_type: {decoder_type_str}. "
            f"Expected 'linear' or '<size>-<depth>b' (e.g., 'base-8b', 'tiny-4b')"
        )

    size, depth_str = parts
    if size not in DECODER_DIM_MAP:
        raise ValueError(
            f"Invalid decoder size: {size}. Choose from {list(DECODER_DIM_MAP.keys())}"
        )

    if not depth_str.endswith("b"):
        raise ValueError(
            f"Invalid depth format: {depth_str}. Expected '<number>b' (e.g., '8b')"
        )

    try:
        depth = int(depth_str[:-1])
    except ValueError:
        raise ValueError(f"Invalid depth number in: {depth_str}")

    embed_dim = DECODER_DIM_MAP[size]
    num_heads = max(embed_dim // 64, 1)

    return {
        "type": "transformer",
        "embed_dim": embed_dim,
        "depth": depth,
        "num_heads": num_heads,
    }


def create_linear_decoder(encoder_embed_dim, patch_size, in_chans=3):
    return nn.Linear(encoder_embed_dim, patch_size**2 * in_chans)


def create_mae_with_custom_decoder(model_name, decoder_config, encoder_overrides=None):
    encoder_defaults = {
        "vit_tiny": {"patch_size": 16, "embed_dim": 192, "depth": 12, "num_heads": 3},
        "vit_small": {"patch_size": 16, "embed_dim": 384, "depth": 12, "num_heads": 6},
        "vit_base": {"patch_size": 16, "embed_dim": 768, "depth": 12, "num_heads": 12},
        "vit_large": {"patch_size": 16, "embed_dim": 1024, "depth": 24, "num_heads": 16},
        "vit_huge": {"patch_size": 14, "embed_dim": 1280, "depth": 32, "num_heads": 16},
    }

    default_decoder = {"decoder_embed_dim": 512, "decoder_depth": 8, "decoder_num_heads": 16}

    if model_name not in encoder_defaults:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from {list(encoder_defaults.keys())}"
        )

    encoder_overrides = encoder_overrides or {}

    encoder_config = {**encoder_defaults[model_name], **encoder_overrides}

    if decoder_config["type"] == "linear":
        encoder_dim = encoder_config["embed_dim"]

        model = spt.backbone.mae.MaskedAutoencoderViT(
            **encoder_config,
            decoder_embed_dim=encoder_dim,
            decoder_depth=1,
            decoder_num_heads=max(encoder_dim // 64, 1),
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        patch_size = encoder_config["patch_size"]
        in_chans = model.patch_embed.proj.in_channels

        linear_decoder = create_linear_decoder(encoder_dim, patch_size, in_chans)
        model.linear_decoder = linear_decoder

        def forward_decoder_linear(self, x, ids_restore):
            N = x.shape[0]
            L_visible = x.shape[1] - 1
            L_full = ids_restore.shape[1]

            x = x[:, 1:, :]

            mask_token = self.decoder_embed(self.mask_token).expand(N, L_full - L_visible, -1)
            x_ = torch.cat([x, mask_token], dim=1)

            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]))

            pred = self.linear_decoder(x_)

            return pred

        import types
        model.forward_decoder = types.MethodType(forward_decoder_linear, model)

        return model

    else:
        decoder_kwargs = {**default_decoder}
        if "embed_dim" in decoder_config:
            decoder_kwargs["decoder_embed_dim"] = decoder_config["embed_dim"]
        if "depth" in decoder_config:
            decoder_kwargs["decoder_depth"] = decoder_config["depth"]
        if "num_heads" in decoder_config:
            decoder_kwargs["decoder_num_heads"] = decoder_config["num_heads"]

        model = spt.backbone.mae.MaskedAutoencoderViT(
            **encoder_config,
            **decoder_kwargs,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        return model


def save_fixed_samples(dataloader, output_path, num_samples=16, seed=42):
    output_path = Path(output_path)
    output_file = output_path / 'fixed_val_samples.pt'

    if output_file.exists():
        try:
            data = torch.load(output_file)
            return data['images'], data['labels']
        except Exception as e:
            logging.warning(f"regenerating: {e}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    images = []
    labels = []
    indices = []
    idx = 0

    for batch in dataloader:
        if isinstance(batch, dict):
            batch_images = batch["image"]
            batch_labels = batch.get("label", torch.zeros(batch_images.shape[0], dtype=torch.long))
        elif isinstance(batch, (list, tuple)):
            batch_images, batch_labels = batch[0], batch[1]
        else:
            batch_images = batch
            batch_labels = torch.zeros(batch_images.shape[0], dtype=torch.long)

        for i in range(batch_images.shape[0]):
            if len(images) >= num_samples:
                break
            images.append(batch_images[i].cpu())
            labels.append(batch_labels[i].item() if torch.is_tensor(batch_labels) else batch_labels[i])
            indices.append(idx)
            idx += 1

        if len(images) >= num_samples:
            break

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)

    output_path.mkdir(parents=True, exist_ok=True)

    torch.save({
        'images': images_tensor,
        'labels': labels_tensor,
        'indices': indices,
        'num_samples': num_samples,
        'seed': seed,
    }, output_file)

    return images_tensor, labels_tensor


def load_mae_dataset(dataset_name, split, transform, cache_dir):
    config = get_dataset_config_fn(dataset_name)

    hf_config = config["hf_config"]
    if hf_config:
        dataset = spt.data.HFDataset(
            config["hf_path"],
            hf_config,
            split=config["splits"][split],
            cache_dir=str(cache_dir),
            transform=transform,
            trust_remote_code=True,
        )
    else:
        dataset = spt.data.HFDataset(
            config["hf_path"],
            split=config["splits"][split],
            cache_dir=str(cache_dir),
            transform=transform,
            trust_remote_code=True,
        )

    return dataset


def detect_image_properties_from_dataset(dataset):
    sample = dataset[0]

    if isinstance(sample, dict):
        image = sample["image"]
    elif isinstance(sample, (list, tuple)):
        image = sample[0]
    else:
        image = sample

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected tensor, got {type(image)}")

    if image.ndim != 3:
        raise ValueError(f"Expected 3D tensor [C, H, W], got shape {image.shape}")

    channels, height, width = image.shape

    value_min = image.min().item()
    value_max = image.max().item()

    properties = {
        'img_size': height,
        'height': height,
        'width': width,
        'channels': channels,
        'dtype': image.dtype,
        'value_range': (value_min, value_max),
        'is_square': height == width,
    }

    logging.info(f"detected {height}x{width}, {channels}ch, {image.dtype}, range [{value_min:.3f}, {value_max:.3f}]")

    return properties
