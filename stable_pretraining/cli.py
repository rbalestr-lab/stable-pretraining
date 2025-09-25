#!/usr/bin/env python
"""Command-line interface for Stable SSL training."""

import sys
from pathlib import Path
import subprocess
import argparse


def find_config_file(config_spec):
    """Find config file from path or name."""
    config_path = Path(config_spec)

    if config_path.exists():
        config_path = config_path.resolve()
        return str(config_path.parent), config_path.stem

    if not config_spec.endswith((".yaml", ".yml")):
        config_spec = f"{config_spec}.yaml"

    config_path = Path.cwd() / config_spec
    if config_path.exists():
        return str(Path.cwd()), config_path.stem

    return None, None


def needs_multirun(overrides):
    """Detect if multirun mode is needed."""
    if not overrides:
        return False

    overrides_str = " ".join(overrides)

    return (
        "--multirun" in overrides
        or "-m" in overrides
        or "hydra/launcher=" in overrides_str
        or "hydra.sweep" in overrides_str
        or any("=" in o and "," in o.split("=", 1)[1] for o in overrides if "=" in o)
    )


def run_command(args):
    """Execute experiment with the specified config."""
    config_spec = args.config
    overrides = args.overrides

    config_path, config_name = find_config_file(config_spec)

    if config_path is None:
        print(f"Error: Could not find config file '{config_spec}'")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "stable_pretraining.run",
        "--config-path",
        config_path,
        "--config-name",
        config_name,
    ]

    if needs_multirun(overrides):
        cmd.append("-m")
        overrides = [o for o in overrides if o not in ["-m", "--multirun"]]
        if not any("hydra/launcher=" in o for o in overrides):
            overrides.append("hydra/launcher=submitit_slurm")
        print("Running in multirun mode")

    if overrides:
        cmd.extend(overrides)

    print(f"Config: {config_name} from {config_path}")
    print("-" * 50)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)


def main():
    """Main entry point for the spt CLI."""
    parser = argparse.ArgumentParser(
        prog="spt",
        description="Stable SSL Training CLI",
        epilog="Examples: spt config.yaml | spt config.yaml -m | spt ../path/to/config.yaml trainer.max_epochs=100",
    )

    parser.add_argument("config", help="Config file path or name")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides")

    args = parser.parse_args()
    run_command(args)


if __name__ == "__main__":
    main()
