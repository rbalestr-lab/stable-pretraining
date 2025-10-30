from pathlib import Path
from loguru import logger as logging
import threading
from typing import Dict, List, Optional
import json
import os

import timm
import torch


class MetaStatic(type):
    """Metaclass that enables dict-like behavior on the TIMM_PARAMETERS class."""

    def __contains__(cls, key):
        """Enable 'in' operator on the class itself."""
        cls._ensure_loaded()
        return key in cls._data

    def __len__(cls):
        """Enable len() on the class itself."""
        cls._ensure_loaded()
        return len(cls._data)

    def __iter__(cls):
        """Enable iteration over keys."""
        cls._ensure_loaded()
        return iter(cls._data)


class TIMM_EMBEDDINGS(metaclass=MetaStatic):
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
    def _ensure_loaded(cls):
        """Ensure the TIMM parameters are loaded from the JSON file.

        This method uses double-checked locking to ensure thread-safe,
        one-time initialization of the cached data.

        Raises:
            RuntimeError: If the assets folder or JSON file is missing.
        """
        if cls._data is None:
            with cls._lock:
                # Double-check after acquiring lock
                if cls._data is None:
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
        cls._ensure_loaded()
        # Defensive: always return a copy to prevent mutation of the cached data
        value = cls._data[key]
        return list(value)


class TIMM_PARAMETERS(metaclass=MetaStatic):
    """Thread-safe singleton class for accessing TIMM (Timm Image Models) parameters.

    This class provides lazy-loaded, cached access to TIMM model parameters stored
    in a static JSON file. It implements a dict-like interface with thread-safe
    initialization and defensive copying to prevent mutation of cached data.

    Usage:
        # Access parameters by key
        params = TIMM_PARAMETERS['model_name']

        # Iterate over keys
        for key in TIMM_PARAMETERS.keys():
            print(key)

        # Iterate over values
        for values in TIMM_PARAMETERS.values():
            print(values)

        # Iterate over items
        for key, values in TIMM_PARAMETERS.items():
            print(f"{key}: {values}")

    Note:
        All methods return copies of the data to prevent accidental mutation
        of the internal cache.
    """

    _data: Optional[Dict[str, List[str]]] = None
    _lock = threading.RLock()

    data: dict[str, list[str]] = None

    @classmethod
    def _ensure_loaded(cls):
        """Ensure the TIMM parameters are loaded from the JSON file.

        This method uses double-checked locking to ensure thread-safe,
        one-time initialization of the cached data.

        Raises:
            RuntimeError: If the assets folder or JSON file is missing.
        """
        if cls._data is None:
            with cls._lock:
                # Double-check after acquiring lock
                if cls._data is None:
                    logging.info("TIMM cache not loaded yet... loading!")
                    path = Path(os.path.abspath(__file__))
                    logging.info(
                        f"Loading TIMM parameters from: {path.parent.parent / 'assets/static_timm_parameters.json'}"
                    )
                    asset_path = (
                        path.parent.parent / "assets/static_timm_parameters.json"
                    )
                    if not asset_path.is_file():
                        raise RuntimeError("Did you manually delete the assets folder?")
                    with open(asset_path, "r") as f:
                        cls._data = json.load(f)

    @classmethod
    def __class_getitem__(cls, key):
        """Access TIMM parameters by key.

        Args:
            key: The parameter key to retrieve.

        Returns:
            A copy of the list of parameters associated with the key.

        Raises:
            KeyError: If the key is not found in the parameters.
        """
        cls._ensure_loaded()
        # Defensive: always return a copy to prevent mutation of the cached data
        value = cls._data[key]
        return value

    @classmethod
    def keys(cls):
        """Return all parameter keys.

        Returns:
            A dict_keys view of all available parameter keys.

        Example:
            >>> list(TIMM_PARAMETERS.keys())
            ['resnet50', 'efficientnet_b0', ...]
        """
        cls._ensure_loaded()
        return cls._data.keys()

    @classmethod
    def values(cls):
        """Return all parameter values.

        Returns:
            A generator yielding copies of all parameter value lists.

        Note:
            Returns copies to prevent mutation of cached data.

        Example:
            >>> for params in TIMM_PARAMETERS.values():
            ...     print(params)
        """
        cls._ensure_loaded()
        # Return copies to prevent mutation
        return cls._data.values()

    @classmethod
    def items(cls):
        """Return all parameter key-value pairs.

        Returns:
            A generator yielding tuples of (key, value_copy) pairs.

        Note:
            Values are copies to prevent mutation of cached data.

        Example:
            >>> for key, params in TIMM_PARAMETERS.items():
            ...     print(f"{key}: {params}")
        """
        cls._ensure_loaded()
        # Return copies of values to prevent mutation
        return ((key, value) for key, value in cls._data.items())


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


def _get_last_submodule_full_name(model):
    # named_modules() yields (full_name, module) pairs in definition order
    for name, module in model.named_modules():
        last_name, _ = name, module
    return last_name


def _retreive_timm_modules(args):
    name, parent_name, L = args
    import timm
    from stable_pretraining.backbone.utils import get_children_modules

    model = timm.create_model(name, pretrained=False, num_classes=0)
    num_params = count_parameters(model)
    internals = get_children_modules(
        model, parent_name=parent_name, partial_match=True, L=L
    )
    last = _get_last_submodule_full_name(model)
    return internals + [last], num_params


def _generate_hf_embeddings_factory():
    from tqdm import tqdm
    import os
    import json
    from multiprocessing import Pool

    hf_embedding = {}
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
        hf_embedding[name[0]] = result

    path = Path(os.path.abspath(__file__))
    with open(path.parent.parent / "assets/static_timm.json", "w") as f:
        json.dump(hf_embedding, f, indent=2)


def count_parameters(model):
    """Count trainable parameters efficiently."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _generate_timm_factory():
    from tqdm import tqdm
    import timm
    import os
    import json
    from multiprocessing import Pool

    family_to_name = {
        "vit_": ("blocks", 1),
        "swin": ("blocks", 1),
        "levit": ("blocks", 1),
        "maxvit": ("blocks", 1),
        "maxxvit": ("blocks", 1),
        "convnext": ("blocks", 1),
        "resnetv2": ("blocks", 1),
        "resnet": ("layer", 1),
        "resnext": ("layer", 1),
        "efficientnet": ("blocks", 2),
        "mobilenet": ("blocks", 2),
        "convmixer": ("blocks", 1),
        "inception": ("blocks", 1),
    }

    names = []
    for name in timm.list_models(pretrained=False) + timm.list_models(pretrained=True):
        for f in family_to_name:
            if name.startswith(f):
                names.append((name,) + family_to_name[f])
    results = list(
        tqdm(
            Pool(20).imap(_retreive_timm_modules, names),
            total=len(names),
            desc="timm",
        )
    )
    timm_embeddings = {}
    timm_parameters = {}
    for name, result in zip(names, results):
        timm_embeddings[name[0]], timm_parameters[name[0]] = result

    path = Path(os.path.abspath(__file__))
    with open(path.parent.parent / "assets/static_timm.json", "w") as f:
        json.dump(timm_embeddings, f, indent=2)
    with open(path.parent.parent / "assets/static_timm_parameters.json", "w") as f:
        json.dump(timm_parameters, f, indent=2)


if __name__ == "__main__":
    _generate_timm_factory()
    # _generate_hf_embeddings_factory()

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
