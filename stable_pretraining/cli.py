#!/usr/bin/env python
"""Command-line interface for Stable SSL training.

This module provides the 'spt' command for easily launching training runs.

Usage:
    spt <config> [hydra_overrides...]
    spt simclr_cifar10_config
    spt configs/my_config.yaml
    spt simclr_cifar10_config trainer.max_epochs=100 module.optim.lr=0.01
"""

import sys
import os
from pathlib import Path
import subprocess
import argparse


def find_config_file(config_spec):
    """Find config file from various sources.

    Args:
        config_spec: Either a path to a config file or a config name

    Returns:
        tuple: (config_path, config_name) or (None, None) if not found
    """
    # If it's already a path that exists
    if os.path.isfile(config_spec):
        config_path = Path(config_spec)
        return str(config_path.parent), config_path.stem

    # Add .yaml extension if not present
    if not config_spec.endswith(".yaml") and not config_spec.endswith(".yml"):
        config_spec_with_ext = f"{config_spec}.yaml"
    else:
        config_spec_with_ext = config_spec
        config_spec = config_spec.replace(".yaml", "").replace(".yml", "")

    # Search locations in order of priority
    search_locations = [
        Path.cwd(),  # Current directory
        Path.cwd() / "configs",  # ./configs/
        Path.cwd() / "examples",  # ./examples/
        Path(__file__).parent.parent / "examples",  # Package examples
        Path(__file__).parent.parent / "configs",  # Package configs
    ]

    for location in search_locations:
        config_path = location / config_spec_with_ext
        if config_path.exists():
            return str(location.absolute()), config_spec

    # Try without extension
    for location in search_locations:
        config_path = location / config_spec
        if config_path.exists():
            return str(location.absolute()), config_spec

    return None, None


def run_command(args):
    """Execute experiment with the specified config."""
    config_spec = args.config
    overrides = args.overrides

    # Find the config file
    config_path, config_name = find_config_file(config_spec)

    if config_path is None:
        print(f"Error: Could not find config file '{config_spec}'")
        print("\nSearched in:")
        print("  - Current directory")
        print("  - ./configs/")
        print("  - ./examples/")
        print("  - Package examples directory")
        sys.exit(1)

    # Build the command
    cmd = [
        sys.executable,
        "-m",
        "stable_pretraining.run",
        "--config-path",
        config_path,
        "--config-name",
        config_name,
    ]

    # Add any Hydra overrides
    if overrides:
        cmd.extend(overrides)

    # Print what we're running
    print(f"Running experiment with config: {config_name}")
    print(f"Config path: {config_path}")
    if overrides:
        print(f"Overrides: {' '.join(overrides)}")
    print("-" * 50)

    # Execute the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(130)


def main():
    """Main entry point for the spt CLI."""
    parser = argparse.ArgumentParser(
        prog="spt",
        description="Stable SSL Training CLI - Launch training with config files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  spt simclr_cifar10_config
  spt configs/my_config.yaml
  spt simclr_cifar10_config trainer.max_epochs=100
  spt ../my_configs/custom.yaml module.optim.lr=0.01

Config resolution:
  1. If a file path is provided, use it directly
  2. Otherwise, search for <name>.yaml in:
     - Current directory
     - ./configs/
     - ./examples/
     - Package examples directory
        """,
    )

    parser.add_argument(
        "config",
        help="Config file path or name (e.g., simclr_cifar10_config or path/to/config.yaml)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides (e.g., trainer.max_epochs=100 module.optim.lr=0.01)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Direct execution without subcommand
    run_command(args)


if __name__ == "__main__":
    main()
