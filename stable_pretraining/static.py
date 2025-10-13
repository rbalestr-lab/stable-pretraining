from pathlib import Path
from loguru import logger as logging
import threading
from typing import Dict, List, Optional
import json
import os


class TIMM_EMBEDDINGS:
    """Thread-safe, lazy-loaded registry for TIMM (PyTorch Image Models) embedding names, accessed via class-level indexing.

    This class provides a mapping from string keys to lists of embedding names, loaded on first access from a
    JSON file located at 'assets/static_timm.json' relative to the source file. The data is cached as a class
    attribute after the first load, and subsequent accesses are served from memory.
    The class is intended to be used as a static registry, e.g.:
        >>> names = TIMM_EMBEDDINGS["resnet50"]
        >>> print(names)  # List of embedding names for 'resnet50'
    Notes:
        - The data is loaded only once per process and is shared across all uses of the class.
        - Thread-safe: concurrent first-time access is protected by a class-level lock.
        - The class depends on the presence of the 'assets/static_timm.json' file two directories above the source file.
        - The class assumes the `__file__` attribute is available and points to the current file.
        - The class attribute `_data` is private and shared.
        - Logging and printing occur on first load for debugging.
        - File system access and JSON parsing are required at runtime.

    Raises:
        RuntimeError: If the assets file is missing.
        OSError, IOError: If the file cannot be read.
        json.JSONDecodeError: If the file is not valid JSON.
        KeyError: If the requested key is not present in the data.

    Example:
        >>> embeddings = TIMM_EMBEDDINGS["vit_base_patch16_224"]
        >>> print(embeddings)
    """

    _data: Optional[Dict[str, List[str]]] = None
    _lock = threading.RLock()

    data: dict[str, list[str]] = None

    @classmethod
    def __class_getitem__(cls, key):
        """Retrieve a copy of the list of embedding names for a given model key, loading the registry from disk if necessary.

        On first access, this method loads the JSON file 'assets/static_timm.json' located two directories above
        the current file, caches the result in the class attribute `_data`, and returns a copy of the value for the given key.
        Subsequent accesses use the cached data.
        Parameters:
            key (str): The model identifier for which to retrieve embedding names.

        Returns:
            list[str]: A copy of the list of embedding names associated with the given key.

        Raises:
            RuntimeError: If the assets file is missing.
            OSError, IOError: If the file cannot be read.
            json.JSONDecodeError: If the file is not valid JSON.
            KeyError: If the requested key is not present in the data.

        Notes:
            - Logging and printing of the resolved path occur on first load.
            - Thread-safe: concurrent first-time access is protected by a class-level lock.
            - The method assumes the `__file__` attribute is available.
            - The returned list is a copy; mutating it will not affect the cached data.

        Example:
            >>> names = TIMM_EMBEDDINGS["efficientnet_b0"]
            >>> print(names)
        """
        if cls._data is None:
            with cls._lock:
                logging.info("TIMM cache not loaded yet... loading!")
                path = Path(os.path.abspath(__file__))
                logging.info(
                    f"Loading TIMM embeddings from: {path.parent.parent / 'assets/static_timm.json'}"
                )
                asset_path = path.parent.parent / "assets/static_timm.json"
                if not asset_path.is_file():
                    raise RuntimeError("Did you manually delete the assets folder?")
                with open(asset_path, "r") as f:
                    cls._data = json.load(f)
        # Defensive: always return a copy to prevent mutation of the cached data
        value = cls._data[key]
        return list(value)


TORCHVISION_EMBEDDINGS = {
    "vit_b_16": [
        "encoder.layers.encoder_layer_0",
        "encoder.layers.encoder_layer_1",
        "encoder.layers.encoder_layer_2",
        "encoder.layers.encoder_layer_3",
        "encoder.layers.encoder_layer_4",
        "encoder.layers.encoder_layer_5",
        "encoder.layers.encoder_layer_6",
        "encoder.layers.encoder_layer_7",
        "encoder.layers.encoder_layer_8",
        "encoder.layers.encoder_layer_9",
        "encoder.layers.encoder_layer_10",
        "encoder.layers.encoder_layer_11",
    ],
    "vit_l_16": [
        "encoder.layers.encoder_layer_0",
        "encoder.layers.encoder_layer_1",
        "encoder.layers.encoder_layer_2",
        "encoder.layers.encoder_layer_3",
        "encoder.layers.encoder_layer_4",
        "encoder.layers.encoder_layer_5",
        "encoder.layers.encoder_layer_6",
        "encoder.layers.encoder_layer_7",
        "encoder.layers.encoder_layer_8",
        "encoder.layers.encoder_layer_9",
        "encoder.layers.encoder_layer_10",
        "encoder.layers.encoder_layer_11",
        "encoder.layers.encoder_layer_12",
        "encoder.layers.encoder_layer_13",
        "encoder.layers.encoder_layer_14",
        "encoder.layers.encoder_layer_15",
        "encoder.layers.encoder_layer_16",
        "encoder.layers.encoder_layer_17",
        "encoder.layers.encoder_layer_18",
        "encoder.layers.encoder_layer_19",
        "encoder.layers.encoder_layer_20",
        "encoder.layers.encoder_layer_21",
        "encoder.layers.encoder_layer_22",
        "encoder.layers.encoder_layer_23",
    ],
    "vit_h_14": [
        "encoder.layers.encoder_layer_0",
        "encoder.layers.encoder_layer_1",
        "encoder.layers.encoder_layer_2",
        "encoder.layers.encoder_layer_3",
        "encoder.layers.encoder_layer_4",
        "encoder.layers.encoder_layer_5",
        "encoder.layers.encoder_layer_6",
        "encoder.layers.encoder_layer_7",
        "encoder.layers.encoder_layer_8",
        "encoder.layers.encoder_layer_9",
        "encoder.layers.encoder_layer_10",
        "encoder.layers.encoder_layer_11",
        "encoder.layers.encoder_layer_12",
        "encoder.layers.encoder_layer_13",
        "encoder.layers.encoder_layer_14",
        "encoder.layers.encoder_layer_15",
        "encoder.layers.encoder_layer_16",
        "encoder.layers.encoder_layer_17",
        "encoder.layers.encoder_layer_18",
        "encoder.layers.encoder_layer_19",
        "encoder.layers.encoder_layer_20",
        "encoder.layers.encoder_layer_21",
        "encoder.layers.encoder_layer_22",
        "encoder.layers.encoder_layer_23",
        "encoder.layers.encoder_layer_24",
        "encoder.layers.encoder_layer_25",
        "encoder.layers.encoder_layer_26",
        "encoder.layers.encoder_layer_27",
        "encoder.layers.encoder_layer_28",
        "encoder.layers.encoder_layer_29",
        "encoder.layers.encoder_layer_30",
        "encoder.layers.encoder_layer_31",
    ],
    "resnet18": [
        "maxpool",
        "layer1.0",
        "layer1.1",
        "layer2.0",
        "layer2.1",
        "layer3.0",
        "layer3.1",
        "layer4.0",
        "layer4.1",
    ],
    "resnet34": [
        "maxpool",
        "layer1.0",
        "layer1.1",
        "layer1.2",
        "layer2.0",
        "layer2.1",
        "layer2.2",
        "layer2.3",
        "layer3.0",
        "layer3.1",
        "layer3.2",
        "layer3.3",
        "layer3.4",
        "layer3.5",
        "layer4.0",
        "layer4.1",
        "layer4.2",
    ],
    "resnet50": [
        "maxpool",
        "layer1.0",
        "layer1.1",
        "layer1.2",
        "layer2.0",
        "layer2.1",
        "layer2.2",
        "layer2.3",
        "layer3.0",
        "layer3.1",
        "layer3.2",
        "layer3.3",
        "layer3.4",
        "layer3.5",
        "layer4.0",
        "layer4.1",
        "layer4.2",
    ],
    "resnet101": [
        "maxpool",
        "layer1.0",
        "layer1.1",
        "layer1.2",
        "layer2.0",
        "layer2.1",
        "layer2.2",
        "layer2.3",
        "layer3.0",
        "layer3.1",
        "layer3.2",
        "layer3.3",
        "layer3.4",
        "layer3.5",
        "layer3.6",
        "layer3.7",
        "layer3.8",
        "layer3.9",
        "layer3.10",
        "layer3.11",
        "layer3.12",
        "layer3.13",
        "layer3.14",
        "layer3.15",
        "layer3.16",
        "layer3.17",
        "layer3.18",
        "layer3.19",
        "layer3.20",
        "layer3.21",
        "layer3.22",
        "layer4.0",
        "layer4.1",
        "layer4.2",
    ],
    "resnet152": [
        "maxpool",
        "layer1.0",
        "layer1.1",
        "layer1.2",
        "layer2.0",
        "layer2.1",
        "layer2.2",
        "layer2.3",
        "layer3.0",
        "layer3.1",
        "layer3.2",
        "layer3.3",
        "layer3.4",
        "layer3.5",
        "layer3.6",
        "layer3.7",
        "layer3.8",
        "layer3.9",
        "layer3.10",
        "layer3.11",
        "layer3.12",
        "layer3.13",
        "layer3.14",
        "layer3.15",
        "layer3.16",
        "layer3.17",
        "layer3.18",
        "layer3.19",
        "layer3.20",
        "layer3.21",
        "layer3.22",
        "layer3.23",
        "layer3.24",
        "layer3.25",
        "layer3.26",
        "layer3.27",
        "layer3.28",
        "layer3.29",
        "layer3.30",
        "layer3.31",
        "layer3.32",
        "layer3.33",
        "layer3.34",
        "layer3.35",
        "layer4.0",
        "layer4.1",
        "layer4.2",
    ],
    "swin_v2_t": [
        "features.0",
        "features.1.0",
        "features.1.1",
        "features.2",
        "features.3.0",
        "features.3.1",
        "features.5.0",
        "features.5.1",
        "features.5.2",
        "features.5.3",
        "features.5.4",
        "features.5.5",
        "features.6",
        "features.7.0",
        "features.7.1",
    ],
    "swin_v2_b": [
        "features.0",
        "features.1.0",
        "features.1.1",
        "features.2",
        "features.3.0",
        "features.3.1",
        "features.5.0",
        "features.5.1",
        "features.5.2",
        "features.5.3",
        "features.5.4",
        "features.5.5",
        "features.5.6",
        "features.5.7",
        "features.5.8",
        "features.5.9",
        "features.5.10",
        "features.5.11",
        "features.5.12",
        "features.5.13",
        "features.5.14",
        "features.5.15",
        "features.5.16",
        "features.5.17",
        "features.6",
        "features.7.0",
        "features.7.1",
    ],
    "convnext_tiny": [
        "features.0",
        "features.1.0",
        "features.1.1",
        "features.1.2",
        "features.2",
        "features.3.0",
        "features.3.1",
        "features.3.2",
        "features.4",
        "features.5.0",
        "features.5.1",
        "features.5.2",
        "features.5.3",
        "features.5.4",
        "features.5.5",
        "features.5.6",
        "features.5.7",
        "features.5.8",
        "features.6",
        "features.7.0",
        "features.7.1",
        "features.7.2",
    ],
    "convnext_large": [
        "features.0",
        "features.1.0",
        "features.1.1",
        "features.1.2",
        "features.2",
        "features.3.0",
        "features.3.1",
        "features.3.2",
        "features.4",
        "features.5.0",
        "features.5.1",
        "features.5.2",
        "features.5.3",
        "features.5.4",
        "features.5.5",
        "features.5.6",
        "features.5.7",
        "features.5.8",
        "features.5.9",
        "features.5.10",
        "features.5.11",
        "features.5.12",
        "features.5.13",
        "features.5.14",
        "features.5.15",
        "features.5.16",
        "features.5.17",
        "features.5.18",
        "features.5.19",
        "features.5.20",
        "features.5.21",
        "features.5.22",
        "features.5.23",
        "features.5.24",
        "features.5.25",
        "features.5.26",
        "features.6",
        "features.7.0",
        "features.7.1",
        "features.7.2",
    ],
}

TORCHVISION_EMBEDDINGS["convnext_base"] = TORCHVISION_EMBEDDINGS["convnext_large"]
TORCHVISION_EMBEDDINGS["swin_v2_s"] = TORCHVISION_EMBEDDINGS["swin_v2_b"]
TORCHVISION_EMBEDDINGS["vit_b_32"] = TORCHVISION_EMBEDDINGS["vit_b_16"]
TORCHVISION_EMBEDDINGS["vit_l_32"] = TORCHVISION_EMBEDDINGS["vit_l_16"]
TORCHVISION_EMBEDDINGS["wide_resnet50_2"] = TORCHVISION_EMBEDDINGS["resnet50"]


def _retreive_timm_modules(args):
    name, parent_name = args
    import timm
    from stable_pretraining.backbone.utils import get_children_modules

    return get_children_modules(
        timm.create_model(name), parent_name=parent_name, partial_match=True
    )


def _generate_timm_embeddings_factory():
    from tqdm import tqdm
    import os
    import json
    from multiprocessing import Pool

    timm_embedding = {}
    # name = "swin_large_patch4_window7_224.ms_in22k"
    # timm_embedding[name] = get_children_modules(timm.create_model(name), "blocks")
    # name = "hf_hub:timm/vit_base_patch8_224.dino"
    # timm_embedding[name] = get_children_modules(timm.create_model(name), "blocks")
    # Non pretrained ViTs
    names = [
        "vit_base_mci_224",
        "vit_base_patch8_224",
        "vit_base_patch14_dinov2",
        "vit_base_patch14_reg4_dinov2",
        "vit_base_patch16_18x2_224",
        "vit_base_patch16_224",
        "vit_base_patch16_224_miil",
        "vit_base_patch16_384",
        "vit_base_patch16_clip_224",
        "vit_base_patch16_clip_384",
        "vit_base_patch16_clip_quickgelu_224",
        "vit_base_patch16_gap_224",
        "vit_base_patch16_plus_240",
        "vit_base_patch16_plus_clip_240",
        "vit_base_patch16_reg4_gap_256",
        "vit_base_patch16_rope_224",
        "vit_base_patch16_rope_ape_224",
        "vit_base_patch16_rope_mixed_224",
        "vit_base_patch16_rope_mixed_ape_224",
        "vit_base_patch16_rope_reg1_gap_256",
        "vit_base_patch16_rpn_224",
        "vit_base_patch16_siglip_224",
        "vit_base_patch16_siglip_256",
        "vit_base_patch16_siglip_384",
        "vit_base_patch16_siglip_512",
        "vit_base_patch16_siglip_gap_224",
        "vit_base_patch16_siglip_gap_256",
        "vit_base_patch16_siglip_gap_384",
        "vit_base_patch16_siglip_gap_512",
        "vit_base_patch16_xp_224",
        "vit_base_patch32_224",
        "vit_base_patch32_384",
        "vit_base_patch32_clip_224",
        "vit_base_patch32_clip_256",
        "vit_base_patch32_clip_384",
        "vit_base_patch32_clip_448",
        "vit_base_patch32_clip_quickgelu_224",
        "vit_base_patch32_plus_256",
        "vit_base_patch32_siglip_256",
        "vit_base_patch32_siglip_gap_256",
        "vit_base_r26_s32_224",
        "vit_base_r50_s16_224",
        "vit_base_r50_s16_384",
        "vit_base_resnet26d_224",
        "vit_base_resnet50d_224",
        "vit_betwixt_patch16_gap_256",
        "vit_betwixt_patch16_reg1_gap_256",
        "vit_betwixt_patch16_reg4_gap_256",
        "vit_betwixt_patch16_reg4_gap_384",
        "vit_betwixt_patch16_rope_reg4_gap_256",
        "vit_betwixt_patch32_clip_224",
        "vit_giant_patch14_224",
        "vit_giant_patch14_clip_224",
        "vit_giant_patch14_dinov2",
        "vit_giant_patch14_reg4_dinov2",
        "vit_giant_patch16_gap_224",
        "vit_giantopt_patch16_siglip_256",
        "vit_giantopt_patch16_siglip_384",
        "vit_giantopt_patch16_siglip_gap_256",
        "vit_giantopt_patch16_siglip_gap_384",
        "vit_gigantic_patch14_224",
        "vit_gigantic_patch14_clip_224",
        "vit_gigantic_patch14_clip_quickgelu_224",
        "vit_huge_patch14_224",
        "vit_huge_patch14_clip_224",
        "vit_huge_patch14_clip_336",
        "vit_huge_patch14_clip_378",
        "vit_huge_patch14_clip_quickgelu_224",
        "vit_huge_patch14_clip_quickgelu_378",
        "vit_huge_patch14_gap_224",
        "vit_huge_patch14_xp_224",
        "vit_huge_patch16_gap_448",
        "vit_intern300m_patch14_448",
        "vit_large_patch14_224",
        "vit_large_patch14_clip_224",
        "vit_large_patch14_clip_336",
        "vit_large_patch14_clip_quickgelu_224",
        "vit_large_patch14_clip_quickgelu_336",
        "vit_large_patch14_dinov2",
        "vit_large_patch14_reg4_dinov2",
        "vit_large_patch14_xp_224",
        "vit_large_patch16_224",
        "vit_large_patch16_384",
        "vit_large_patch16_rope_224",
        "vit_large_patch16_rope_ape_224",
        "vit_large_patch16_rope_mixed_224",
        "vit_large_patch16_rope_mixed_ape_224",
        "vit_large_patch16_siglip_256",
        "vit_large_patch16_siglip_384",
        "vit_large_patch16_siglip_512",
        "vit_large_patch16_siglip_gap_256",
        "vit_large_patch16_siglip_gap_384",
        "vit_large_patch16_siglip_gap_512",
        "vit_large_patch32_224",
        "vit_large_patch32_384",
        "vit_large_r50_s32_224",
        "vit_large_r50_s32_384",
        "vit_little_patch16_reg1_gap_256",
        "vit_little_patch16_reg4_gap_256",
        "vit_medium_patch16_clip_224",
        "vit_medium_patch16_gap_240",
        "vit_medium_patch16_gap_256",
        "vit_medium_patch16_gap_384",
        "vit_medium_patch16_reg1_gap_256",
        "vit_medium_patch16_reg4_gap_256",
        "vit_medium_patch16_rope_reg1_gap_256",
        "vit_medium_patch32_clip_224",
        "vit_mediumd_patch16_reg4_gap_256",
        "vit_mediumd_patch16_reg4_gap_384",
        "vit_mediumd_patch16_rope_reg1_gap_256",
        "vit_pe_core_base_patch16_224",
        "vit_pe_core_gigantic_patch14_448",
        "vit_pe_core_large_patch14_336",
        "vit_pe_core_small_patch16_384",
        "vit_pe_core_tiny_patch16_384",
        "vit_pe_lang_gigantic_patch14_448",
        "vit_pe_lang_large_patch14_448",
        "vit_pe_spatial_base_patch16_512",
        "vit_pe_spatial_gigantic_patch14_448",
        "vit_pe_spatial_large_patch14_448",
        "vit_pe_spatial_small_patch16_512",
        "vit_pe_spatial_tiny_patch16_512",
        "vit_pwee_patch16_reg1_gap_256",
        "vit_relpos_base_patch16_224",
        "vit_relpos_base_patch16_cls_224",
        "vit_relpos_base_patch16_clsgap_224",
        "vit_relpos_base_patch16_plus_240",
        "vit_relpos_base_patch16_rpn_224",
        "vit_relpos_base_patch32_plus_rpn_256",
        "vit_relpos_medium_patch16_224",
        "vit_relpos_medium_patch16_cls_224",
        "vit_relpos_medium_patch16_rpn_224",
        "vit_relpos_small_patch16_224",
        "vit_relpos_small_patch16_rpn_224",
        "vit_small_patch8_224",
        "vit_small_patch14_dinov2",
        "vit_small_patch14_reg4_dinov2",
        "vit_small_patch16_18x2_224",
        "vit_small_patch16_36x1_224",
        "vit_small_patch16_224",
        "vit_small_patch16_384",
        "vit_small_patch16_rope_224",
        "vit_small_patch16_rope_ape_224",
        "vit_small_patch16_rope_mixed_224",
        "vit_small_patch16_rope_mixed_ape_224",
        "vit_small_patch32_224",
        "vit_small_patch32_384",
        "vit_small_r26_s32_224",
        "vit_small_r26_s32_384",
        "vit_small_resnet26d_224",
        "vit_small_resnet50d_s16_224",
        "vit_so150m2_patch16_reg1_gap_256",
        "vit_so150m2_patch16_reg1_gap_384",
        "vit_so150m2_patch16_reg1_gap_448",
        "vit_so150m_patch16_reg4_gap_256",
        "vit_so150m_patch16_reg4_gap_384",
        "vit_so150m_patch16_reg4_map_256",
        "vit_so400m_patch14_siglip_224",
        "vit_so400m_patch14_siglip_378",
        "vit_so400m_patch14_siglip_384",
        "vit_so400m_patch14_siglip_gap_224",
        "vit_so400m_patch14_siglip_gap_378",
        "vit_so400m_patch14_siglip_gap_384",
        "vit_so400m_patch14_siglip_gap_448",
        "vit_so400m_patch14_siglip_gap_896",
        "vit_so400m_patch16_siglip_256",
        "vit_so400m_patch16_siglip_384",
        "vit_so400m_patch16_siglip_512",
        "vit_so400m_patch16_siglip_gap_256",
        "vit_so400m_patch16_siglip_gap_384",
        "vit_so400m_patch16_siglip_gap_512",
        "vit_srelpos_medium_patch16_224",
        "vit_srelpos_small_patch16_224",
        "vit_tiny_patch16_224",
        "vit_tiny_patch16_384",
        "vit_tiny_r_s16_p8_224",
        "vit_tiny_r_s16_p8_384",
        "vit_wee_patch16_reg1_gap_256",
        "vit_xsmall_patch16_clip_224",
    ]
    names = [(n, "blocks") for n in names]
    results = list(
        tqdm(
            Pool(30).imap(_retreive_timm_modules, names), total=len(names), desc="vits"
        )
    )
    for name, result in zip(names, results):
        timm_embedding[name[0]] = result

    # non pretrained swin
    names = [
        "swin_base_patch4_window7_224",
        "swin_base_patch4_window12_384",
        "swin_large_patch4_window7_224",
        "swin_large_patch4_window12_384",
        "swin_s3_base_224",
        "swin_s3_small_224",
        "swin_s3_tiny_224",
        "swin_small_patch4_window7_224",
        "swin_tiny_patch4_window7_224",
        "swinv2_base_window8_256",
        "swinv2_base_window12_192",
        "swinv2_base_window12to16_192to256",
        "swinv2_base_window12to24_192to384",
        "swinv2_base_window16_256",
        "swinv2_cr_base_224",
        "swinv2_cr_base_384",
        "swinv2_cr_base_ns_224",
        "swinv2_cr_giant_224",
        "swinv2_cr_giant_384",
        "swinv2_cr_huge_224",
        "swinv2_cr_huge_384",
        "swinv2_cr_large_224",
        "swinv2_cr_large_384",
        "swinv2_cr_small_224",
        "swinv2_cr_small_384",
        "swinv2_cr_small_ns_224",
        "swinv2_cr_small_ns_256",
        "swinv2_cr_tiny_224",
        "swinv2_cr_tiny_384",
        "swinv2_cr_tiny_ns_224",
        "swinv2_large_window12_192",
        "swinv2_large_window12to16_192to256",
        "swinv2_large_window12to24_192to384",
        "swinv2_small_window8_256",
        "swinv2_small_window16_256",
        "swinv2_tiny_window8_256",
        "swinv2_tiny_window16_256",
    ]
    names = [(n, "blocks") for n in names]
    results = list(
        tqdm(
            Pool(30).imap(_retreive_timm_modules, names), total=len(names), desc="swin"
        )
    )
    for name, result in zip(names, results):
        timm_embedding[name[0]] = result
    # non pretrained convnext
    names = [
        "convnext_atto",
        "convnext_atto_ols",
        "convnext_atto_rms",
        "convnext_base",
        "convnext_femto",
        "convnext_femto_ols",
        "convnext_large",
        "convnext_large_mlp",
        "convnext_nano",
        "convnext_nano_ols",
        "convnext_pico",
        "convnext_pico_ols",
        "convnext_small",
        "convnext_tiny",
        "convnext_tiny_hnf",
        "convnext_xlarge",
        "convnext_xxlarge",
        "convnext_zepto_rms",
        "convnext_zepto_rms_ols",
        "convnextv2_atto",
        "convnextv2_base",
        "convnextv2_femto",
        "convnextv2_huge",
        "convnextv2_large",
        "convnextv2_nano",
        "convnextv2_pico",
        "convnextv2_small",
        "convnextv2_tiny",
    ]
    names = [(n, "blocks") for n in names]
    results = list(
        tqdm(
            Pool(30).imap(_retreive_timm_modules, names),
            total=len(names),
            desc="convnext",
        )
    )
    for name, result in zip(names, results):
        timm_embedding[name[0]] = result
    # non pretrained LeVIT
    names = [
        "levit_128",
        "levit_128s",
        "levit_192",
        "levit_256",
        "levit_256d",
        "levit_384",
        "levit_384_s8",
        "levit_512",
        "levit_512_s8",
        "levit_512d",
        "levit_conv_128",
        "levit_conv_128s",
        "levit_conv_192",
        "levit_conv_256",
        "levit_conv_256d",
        "levit_conv_384",
        "levit_conv_384_s8",
        "levit_conv_512",
        "levit_conv_512_s8",
        "levit_conv_512d",
    ]
    names = [(n, "blocks") for n in names]
    results = list(
        tqdm(
            Pool(30).imap(_retreive_timm_modules, names), total=len(names), desc="levit"
        )
    )
    for name, result in zip(names, results):
        timm_embedding[name[0]] = result
    # non pretrained maxvit
    names = [
        "maxvit_base_tf_224",
        "maxvit_base_tf_384",
        "maxvit_base_tf_512",
        "maxvit_large_tf_224",
        "maxvit_large_tf_384",
        "maxvit_large_tf_512",
        "maxvit_nano_rw_256",
        "maxvit_pico_rw_256",
        "maxvit_rmlp_base_rw_224",
        "maxvit_rmlp_base_rw_384",
        "maxvit_rmlp_nano_rw_256",
        "maxvit_rmlp_pico_rw_256",
        "maxvit_rmlp_small_rw_224",
        "maxvit_rmlp_small_rw_256",
        "maxvit_rmlp_tiny_rw_256",
        "maxvit_small_tf_224",
        "maxvit_small_tf_384",
        "maxvit_small_tf_512",
        "maxvit_tiny_pm_256",
        "maxvit_tiny_rw_224",
        "maxvit_tiny_rw_256",
        "maxvit_tiny_tf_224",
        "maxvit_tiny_tf_384",
        "maxvit_tiny_tf_512",
        "maxvit_xlarge_tf_224",
        "maxvit_xlarge_tf_384",
        "maxvit_xlarge_tf_512",
        "maxxvit_rmlp_nano_rw_256",
        "maxxvit_rmlp_small_rw_256",
        "maxxvit_rmlp_tiny_rw_256",
        "maxxvitv2_nano_rw_256",
        "maxxvitv2_rmlp_base_rw_224",
        "maxxvitv2_rmlp_base_rw_384",
        "maxxvitv2_rmlp_large_rw_224",
    ]
    names = [(n, "blocks") for n in names]
    results = list(
        tqdm(
            Pool(30).imap(_retreive_timm_modules, names),
            total=len(names),
            desc="maxvit",
        )
    )
    for name, result in zip(names, results):
        timm_embedding[name[0]] = result
    # Swin models
    names = [
        "swin_base_patch4_window7_224.ms_in1k",
        "swin_base_patch4_window7_224.ms_in22k",
        "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        "swin_base_patch4_window12_384.ms_in1k",
        "swin_base_patch4_window12_384.ms_in22k",
        "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
        "swin_large_patch4_window7_224.ms_in22k",
        "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
        "swin_large_patch4_window12_384.ms_in22k",
        "swin_large_patch4_window12_384.ms_in22k_ft_in1k",
        "swin_s3_base_224.ms_in1k",
        "swin_s3_small_224.ms_in1k",
        "swin_s3_tiny_224.ms_in1k",
        "swin_small_patch4_window7_224.ms_in1k",
        "swin_small_patch4_window7_224.ms_in22k",
        "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
        "swin_tiny_patch4_window7_224.ms_in1k",
        "swin_tiny_patch4_window7_224.ms_in22k",
        "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        "swinv2_base_window8_256.ms_in1k",
        "swinv2_base_window12_192.ms_in22k",
        "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
        "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
        "swinv2_base_window16_256.ms_in1k",
        "swinv2_cr_small_224.sw_in1k",
        "swinv2_cr_small_ns_224.sw_in1k",
        "swinv2_cr_tiny_ns_224.sw_in1k",
        "swinv2_large_window12_192.ms_in22k",
        "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
        "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
        "swinv2_small_window8_256.ms_in1k",
        "swinv2_small_window16_256.ms_in1k",
        "swinv2_tiny_window8_256.ms_in1k",
        "swinv2_tiny_window16_256.ms_in1k",
    ]
    names = [(n, "blocks") for n in names]
    results = list(
        tqdm(
            Pool(30).imap(_retreive_timm_modules, names), total=len(names), desc="swin"
        )
    )
    for name, result in zip(names, results):
        timm_embedding[name[0]] = result

    # ConvNext models
    names = [
        "convnext_atto_ols.a2_in1k",
        "convnext_base.clip_laion2b",
        "convnext_base.clip_laion2b_augreg",
        "convnext_base.clip_laion2b_augreg_ft_in1k",
        "convnext_base.clip_laion2b_augreg_ft_in12k",
        "convnext_base.clip_laion2b_augreg_ft_in12k_in1k",
        "convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384",
        "convnext_base.clip_laiona",
        "convnext_base.clip_laiona_320",
        "convnext_base.clip_laiona_augreg_320",
        "convnext_base.clip_laiona_augreg_ft_in1k_384",
        "convnext_base.fb_in1k",
        "convnext_base.fb_in22k",
        "convnext_base.fb_in22k_ft_in1k",
        "convnext_base.fb_in22k_ft_in1k_384",
        "convnext_femto.d1_in1k",
        "convnext_femto_ols.d1_in1k",
        "convnext_large.fb_in1k",
        "convnext_large.fb_in22k",
        "convnext_large.fb_in22k_ft_in1k",
        "convnext_large.fb_in22k_ft_in1k_384",
        "convnext_large_mlp.clip_laion2b_augreg",
        "convnext_large_mlp.clip_laion2b_augreg_ft_in1k",
        "convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384",
        "convnext_large_mlp.clip_laion2b_augreg_ft_in12k_384",
        "convnext_large_mlp.clip_laion2b_ft_320",
        "convnext_large_mlp.clip_laion2b_ft_soup_320",
        "convnext_large_mlp.clip_laion2b_soup_ft_in12k_320",
        "convnext_large_mlp.clip_laion2b_soup_ft_in12k_384",
        "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320",
        "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384",
        "convnext_nano.d1h_in1k",
        "convnext_nano.in12k",
        "convnext_nano.in12k_ft_in1k",
        "convnext_nano_ols.d1h_in1k",
        "convnext_pico.d1_in1k",
        "convnext_pico_ols.d1_in1k",
        "convnext_small.fb_in1k",
        "convnext_small.fb_in22k",
        "convnext_small.fb_in22k_ft_in1k",
        "convnext_small.fb_in22k_ft_in1k_384",
        "convnext_small.in12k",
        "convnext_small.in12k_ft_in1k",
        "convnext_small.in12k_ft_in1k_384",
        "convnext_tiny.fb_in1k",
        "convnext_tiny.fb_in22k",
        "convnext_tiny.fb_in22k_ft_in1k",
        "convnext_tiny.fb_in22k_ft_in1k_384",
        "convnext_tiny.in12k",
        "convnext_tiny.in12k_ft_in1k",
        "convnext_tiny.in12k_ft_in1k_384",
        "convnext_tiny_hnf.a2h_in1k",
        "convnext_xlarge.fb_in22k",
        "convnext_xlarge.fb_in22k_ft_in1k",
        "convnext_xlarge.fb_in22k_ft_in1k_384",
        "convnext_xxlarge.clip_laion2b_rewind",
        "convnext_xxlarge.clip_laion2b_soup",
        "convnext_xxlarge.clip_laion2b_soup_ft_in1k",
        "convnext_xxlarge.clip_laion2b_soup_ft_in12k",
        "convnextv2_atto.fcmae",
        "convnextv2_atto.fcmae_ft_in1k",
        "convnextv2_base.fcmae",
        "convnextv2_base.fcmae_ft_in1k",
        "convnextv2_base.fcmae_ft_in22k_in1k",
        "convnextv2_base.fcmae_ft_in22k_in1k_384",
        "convnextv2_femto.fcmae",
        "convnextv2_femto.fcmae_ft_in1k",
        "convnextv2_huge.fcmae",
        "convnextv2_huge.fcmae_ft_in1k",
        "convnextv2_huge.fcmae_ft_in22k_in1k_384",
        "convnextv2_huge.fcmae_ft_in22k_in1k_512",
        "convnextv2_large.fcmae",
        "convnextv2_large.fcmae_ft_in1k",
        "convnextv2_large.fcmae_ft_in22k_in1k",
        "convnextv2_large.fcmae_ft_in22k_in1k_384",
        "convnextv2_nano.fcmae",
        "convnextv2_nano.fcmae_ft_in1k",
        "convnextv2_nano.fcmae_ft_in22k_in1k",
        "convnextv2_nano.fcmae_ft_in22k_in1k_384",
        "convnextv2_pico.fcmae",
        "convnextv2_pico.fcmae_ft_in1k",
        "convnextv2_tiny.fcmae",
        "convnextv2_tiny.fcmae_ft_in1k",
        "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "convnextv2_tiny.fcmae_ft_in22k_in1k_384",
    ]
    names = [(n, "blocks") for n in names]
    results = list(
        tqdm(
            Pool(30).imap(_retreive_timm_modules, names),
            total=len(names),
            desc="convnext",
        )
    )
    for name, result in zip(names, results):
        timm_embedding[name[0]] = result

    # LeVIT
    names = [
        "levit_128.fb_dist_in1k",
        "levit_128s.fb_dist_in1k",
        "levit_192.fb_dist_in1k",
        "levit_256.fb_dist_in1k",
        "levit_384.fb_dist_in1k",
        "levit_conv_128.fb_dist_in1k",
        "levit_conv_128s.fb_dist_in1k",
        "levit_conv_192.fb_dist_in1k",
        "levit_conv_256.fb_dist_in1k",
        "levit_conv_384.fb_dist_in1k",
    ]
    names = [(n, "blocks") for n in names]
    results = list(
        tqdm(
            Pool(30).imap(_retreive_timm_modules, names), total=len(names), desc="levit"
        )
    )
    for name, result in zip(names, results):
        timm_embedding[name[0]] = result

    # MaxVIT
    names = [
        "maxvit_base_tf_224.in1k",
        "maxvit_base_tf_224.in21k",
        "maxvit_base_tf_384.in1k",
        "maxvit_base_tf_384.in21k_ft_in1k",
        "maxvit_base_tf_512.in1k",
        "maxvit_base_tf_512.in21k_ft_in1k",
        "maxvit_large_tf_224.in1k",
        "maxvit_large_tf_224.in21k",
        "maxvit_large_tf_384.in1k",
        "maxvit_large_tf_384.in21k_ft_in1k",
        "maxvit_large_tf_512.in1k",
        "maxvit_large_tf_512.in21k_ft_in1k",
        "maxvit_nano_rw_256.sw_in1k",
        "maxvit_rmlp_base_rw_224.sw_in12k",
        "maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k",
        "maxvit_rmlp_base_rw_384.sw_in12k_ft_in1k",
        "maxvit_rmlp_nano_rw_256.sw_in1k",
        "maxvit_rmlp_pico_rw_256.sw_in1k",
        "maxvit_rmlp_small_rw_224.sw_in1k",
        "maxvit_rmlp_tiny_rw_256.sw_in1k",
        "maxvit_small_tf_224.in1k",
        "maxvit_small_tf_384.in1k",
        "maxvit_small_tf_512.in1k",
        "maxvit_tiny_rw_224.sw_in1k",
        "maxvit_tiny_tf_224.in1k",
        "maxvit_tiny_tf_384.in1k",
        "maxvit_tiny_tf_512.in1k",
        "maxvit_xlarge_tf_224.in21k",
        "maxvit_xlarge_tf_384.in21k_ft_in1k",
        "maxvit_xlarge_tf_512.in21k_ft_in1k",
        "maxxvit_rmlp_nano_rw_256.sw_in1k",
        "maxxvit_rmlp_small_rw_256.sw_in1k",
        "maxxvitv2_nano_rw_256.sw_in1k",
        "maxxvitv2_rmlp_base_rw_224.sw_in12k",
        "maxxvitv2_rmlp_base_rw_224.sw_in12k_ft_in1k",
        "maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k",
    ]
    names = [(n, "blocks") for n in names]
    results = list(
        tqdm(
            Pool(30).imap(_retreive_timm_modules, names), total=len(names), desc="vits"
        )
    )
    for name, result in zip(names, results):
        timm_embedding[name[0]] = result

    # resnet
    names = [
        "resnet10t.c3_in1k",
        "resnet14t.c3_in1k",
        "resnet18.a1_in1k",
        "resnet18.a2_in1k",
        "resnet18.a3_in1k",
        "resnet18.fb_ssl_yfcc100m_ft_in1k",
        "resnet18.fb_swsl_ig1b_ft_in1k",
        "resnet18.gluon_in1k",
        "resnet18.tv_in1k",
        "resnet18d.ra2_in1k",
        "resnet26.bt_in1k",
        "resnet26d.bt_in1k",
        "resnet26t.ra2_in1k",
        "resnet32ts.ra2_in1k",
        "resnet33ts.ra2_in1k",
        "resnet34.a1_in1k",
        "resnet34.a2_in1k",
        "resnet34.a3_in1k",
        "resnet34.bt_in1k",
        "resnet34.gluon_in1k",
        "resnet34.tv_in1k",
        "resnet34d.ra2_in1k",
        "resnet50.a1_in1k",
        "resnet50.a1h_in1k",
        "resnet50.a2_in1k",
        "resnet50.a3_in1k",
        "resnet50.am_in1k",
        "resnet50.b1k_in1k",
        "resnet50.b2k_in1k",
        "resnet50.bt_in1k",
        "resnet50.c1_in1k",
        "resnet50.c2_in1k",
        "resnet50.d_in1k",
        "resnet50.fb_ssl_yfcc100m_ft_in1k",
        "resnet50.fb_swsl_ig1b_ft_in1k",
        "resnet50.gluon_in1k",
        "resnet50.ra_in1k",
        "resnet50.ram_in1k",
        "resnet50.tv2_in1k",
        "resnet50.tv_in1k",
        "resnet50_gn.a1h_in1k",
        "resnet50c.gluon_in1k",
        "resnet50d.a1_in1k",
        "resnet50d.a2_in1k",
        "resnet50d.a3_in1k",
        "resnet50d.gluon_in1k",
        "resnet50d.ra2_in1k",
        "resnet50s.gluon_in1k",
        "resnet51q.ra2_in1k",
        "resnet61q.ra2_in1k",
        "resnet101.a1_in1k",
        "resnet101.a1h_in1k",
        "resnet101.a2_in1k",
        "resnet101.a3_in1k",
        "resnet101.gluon_in1k",
        "resnet101.tv2_in1k",
        "resnet101.tv_in1k",
        "resnet101c.gluon_in1k",
        "resnet101d.gluon_in1k",
        "resnet101d.ra2_in1k",
        "resnet101s.gluon_in1k",
        "resnet152.a1_in1k",
        "resnet152.a1h_in1k",
        "resnet152.a2_in1k",
        "resnet152.a3_in1k",
        "resnet152.gluon_in1k",
        "resnet152.tv2_in1k",
        "resnet152.tv_in1k",
        "resnet152c.gluon_in1k",
        "resnet152d.gluon_in1k",
        "resnet152d.ra2_in1k",
        "resnet152s.gluon_in1k",
        "resnet200d.ra2_in1k",
        "resnetaa50.a1h_in1k",
        "resnetaa50d.d_in12k",
        "resnetaa50d.sw_in12k",
        "resnetaa50d.sw_in12k_ft_in1k",
        "resnetaa101d.sw_in12k",
        "resnetaa101d.sw_in12k_ft_in1k",
        "resnetblur50.bt_in1k",
        "resnetrs50.tf_in1k",
        "resnetrs101.tf_in1k",
        "resnetrs152.tf_in1k",
        "resnetrs200.tf_in1k",
        "resnetrs270.tf_in1k",
        "resnetrs350.tf_in1k",
        "resnetrs420.tf_in1k",
        "resnet10t",
        "resnet14t",
        "resnet18",
        "resnet18d",
        "resnet26",
        "resnet26d",
        "resnet26t",
        "resnet32ts",
        "resnet33ts",
        "resnet34",
        "resnet34d",
        "resnet50",
        "resnet50_clip",
        "resnet50_clip_gap",
        "resnet50_gn",
        "resnet50_mlp",
        "resnet50c",
        "resnet50d",
        "resnet50s",
        "resnet50t",
        "resnet50x4_clip",
        "resnet50x4_clip_gap",
        "resnet50x16_clip",
        "resnet50x16_clip_gap",
        "resnet50x64_clip",
        "resnet50x64_clip_gap",
        "resnet51q",
        "resnet61q",
        "resnet101",
        "resnet101_clip",
        "resnet101_clip_gap",
        "resnet101c",
        "resnet101d",
        "resnet101s",
        "resnet152",
        "resnet152c",
        "resnet152d",
        "resnet152s",
        "resnet200",
        "resnet200d",
        "resnetaa34d",
        "resnetaa50",
        "resnetaa50d",
        "resnetaa101d",
        "resnetblur18",
        "resnetblur50",
        "resnetblur50d",
        "resnetblur101d",
        "resnetrs50",
        "resnetrs101",
        "resnetrs152",
        "resnetrs200",
        "resnetrs270",
        "resnetrs350",
        "resnetrs420",
        "resnext26ts",
        "resnext50_32x4d",
        "resnext50d_32x4d",
        "resnext101_32x4d",
        "resnext101_32x8d",
        "resnext101_32x16d",
        "resnext101_32x32d",
        "resnext101_64x4d",
    ]
    names = [(n, "layer") for n in names]
    results = list(
        tqdm(
            Pool(30).imap(_retreive_timm_modules, names),
            total=len(names),
            desc="resnets",
        )
    )
    for name, result in zip(names, results):
        timm_embedding[name[0]] = result

    # Resnet v2
    names = [
        "resnetv2_50.a1h_in1k",
        "resnetv2_50d_evos.ah_in1k",
        "resnetv2_50d_gn.ah_in1k",
        "resnetv2_50x1_bit.goog_distilled_in1k",
        "resnetv2_50x1_bit.goog_in21k",
        "resnetv2_50x1_bit.goog_in21k_ft_in1k",
        "resnetv2_50x3_bit.goog_in21k",
        "resnetv2_50x3_bit.goog_in21k_ft_in1k",
        "resnetv2_101.a1h_in1k",
        "resnetv2_101x1_bit.goog_in21k",
        "resnetv2_101x1_bit.goog_in21k_ft_in1k",
        "resnetv2_101x3_bit.goog_in21k",
        "resnetv2_101x3_bit.goog_in21k_ft_in1k",
        "resnetv2_152x2_bit.goog_in21k",
        "resnetv2_152x2_bit.goog_in21k_ft_in1k",
        "resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k",
        "resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k_384",
        "resnetv2_152x4_bit.goog_in21k",
        "resnetv2_152x4_bit.goog_in21k_ft_in1k",
        "resnetv2_18",
        "resnetv2_18d",
        "resnetv2_34",
        "resnetv2_34d",
        "resnetv2_50",
        "resnetv2_50d",
        "resnetv2_50d_evos",
        "resnetv2_50d_frn",
        "resnetv2_50d_gn",
        "resnetv2_50t",
        "resnetv2_50x1_bit",
        "resnetv2_50x3_bit",
        "resnetv2_101",
        "resnetv2_101d",
        "resnetv2_101x1_bit",
        "resnetv2_101x3_bit",
        "resnetv2_152",
        "resnetv2_152d",
        "resnetv2_152x2_bit",
        "resnetv2_152x4_bit",
    ]
    names = [(n, "blocks") for n in names]
    results = list(
        tqdm(
            Pool(30).imap(_retreive_timm_modules, names),
            total=len(names),
            desc="resnets",
        )
    )
    for name, result in zip(names, results):
        timm_embedding[name[0]] = result

    path = Path(os.path.abspath(__file__))
    with open(path.parent.parent / "assets/static_timm.json", "w") as f:
        json.dump(timm_embedding, f, indent=2)


if __name__ == "__main__":
    _generate_timm_embeddings_factory()

    # last 3 blocks
    import stable_pretraining as spt
    import timm
    import torch

    model = timm.create_model("resnet34")
    # add last 3 blocks as separate output
    names = spt.static.TIMM_EMBEDDINGS["resnet34"][-3:]
    # names = ['layers.2.blocks.17', 'layers.3.blocks.0', 'layers.3.blocks.1']
    model = spt.backbone.utils.ReturnEmbedding(model, names)
    # if you need shapes e.g. for probing definition
    image = torch.zeros((10, 3, 224, 224))
    output_shape, embedding_shapes = spt.backbone.utils.get_output_shape(model, image)
    # embedding_shapes = {'layers.2.blocks.17': torch.Size([10, 14, 14, 768]),
    # 'layers.3.blocks.0': torch.Size([10, 7, 7, 1536]),
    # 'layers.3.blocks.1': torch.Size([10, 7, 7, 1536])}
    output, embeddings = model(image)
    # output = tensor([[ 1.1009 ...
    # embeddings = {'layers.3.blocks.1': tensor([[[[-0.6236, ...}
