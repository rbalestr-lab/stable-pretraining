import torch
import torch.nn as nn
from functools import partial
from pathlib import Path
import numpy as np
import logging
import stable_pretraining as spt


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


def create_mae_with_custom_decoder(model_name, decoder_config):
    model_map = {
        "vit_tiny": spt.backbone.mae.vit_tiny_patch16,
        "vit_small": spt.backbone.mae.vit_small_patch16,
        "vit_base": spt.backbone.mae.vit_base_patch16,
        "vit_large": spt.backbone.mae.vit_large_patch16,
        "vit_huge": spt.backbone.mae.vit_huge_patch14,
    }

    if model_name not in model_map:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from {list(model_map.keys())}"
        )

    if decoder_config["type"] == "linear":
        base_model = model_map[model_name](decoder_depth=1, decoder_embed_dim=128)

        encoder_dim = base_model.embed_dim
        patch_size = base_model.patch_embed.patch_size[0]
        in_chans = base_model.patch_embed.proj.in_channels

        linear_decoder = create_linear_decoder(encoder_dim, patch_size, in_chans)
        base_model.linear_decoder = linear_decoder

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
        base_model.forward_decoder = types.MethodType(forward_decoder_linear, base_model)

        return base_model

    else:
        model = model_map[model_name]()
        return model


def save_fixed_samples(dataloader, output_path, num_samples=16, seed=42):
    output_path = Path(output_path)
    output_file = output_path / 'fixed_val_samples.pt'

    if output_file.exists():
        logging.info(f"Fixed samples already exist at {output_file}")
        try:
            data = torch.load(output_file)
            return data['images'], data['labels']
        except Exception as e:
            logging.warning(f"Could not load existing samples: {e}. Regenerating...")

    torch.manual_seed(seed)
    np.random.seed(seed)

    logging.info(f"Generating {num_samples} fixed validation samples...")

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

            if len(images) % 4 == 0:
                logging.info(f"  Collected {len(images)}/{num_samples} samples...")

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

    logging.info(f"✓ Saved {num_samples} fixed validation samples to {output_file}")
    logging.info(f"  Unique labels: {len(set(labels))} classes")
    logging.info(f"  Image shape: {images_tensor.shape}")

    return images_tensor, labels_tensor
