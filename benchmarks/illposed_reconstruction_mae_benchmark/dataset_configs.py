from typing import Dict, Any

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "imagenette": {
        "hf_path": "randall-lab/imagenette",
        "hf_config": "320px",
        "num_classes": 10,
        "splits": {"train": "train", "val": "test"},
    },
    "tiny-imagenet": {
        "hf_path": "randall-lab/tiny-imagenet",
        "hf_config": None,
        "num_classes": 200,
        "splits": {"train": "train", "val": "validation"},
    },
    "imagenet100": {
        "hf_path": "randall-lab/imagenet100",
        "hf_config": None,
        "num_classes": 100,
        "splits": {"train": "train", "val": "validation"},
    },
    "imagenet": {
        "hf_path": "randall-lab/imagenet",
        "hf_config": None,
        "num_classes": 1000,
        "splits": {"train": "train", "val": "validation"},
    },
}


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    if dataset_name not in DATASET_CONFIGS:
        available = ", ".join(DATASET_CONFIGS.keys())
        raise ValueError(f"unknown dataset: {dataset_name}. available: {available}")
    return DATASET_CONFIGS[dataset_name]


def list_available_datasets():
    return list(DATASET_CONFIGS.keys())
