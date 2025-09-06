#!/usr/bin/env python
"""Universal training script for stable-pretraining using Hydra configs.

This script provides a unified entry point for all training runs via configuration files.
It supports both single-file configs and modular Hydra composition.

Usage:
    # Run with a config file
    python -m stable_pretraining.train --config-path ../examples --config-name simclr_cifar10

    # Run with config and override parameters
    python -m stable_pretraining.train --config-path ../examples --config-name simclr_cifar10 \
        module.optimizer.lr=0.01 \
        trainer.max_epochs=200

    # Run hyperparameter sweep
    python -m stable_pretraining.train --multirun \
        --config-path ../examples --config-name simclr_cifar10 \
        module.optimizer.lr=0.001,0.01,0.1
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from .config import instantiate_from_config


@hydra.main(version_base="1.3", config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    """Main training function that instantiates components from config and runs training.

    Args:
        cfg: Hydra configuration dictionary containing all training parameters
    """
    # Print configuration for debugging
    print("=" * 80)
    print("Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Instantiate and run
    manager = instantiate_from_config(cfg)

    # Check if we got a Manager instance
    if hasattr(manager, "__call__"):
        # It's a Manager, run it
        manager()
    else:
        # It's something else, probably just instantiated components
        print("Warning: Config did not produce a Manager instance.")
        print(f"Got: {type(manager)}")
        return manager


if __name__ == "__main__":
    main()
