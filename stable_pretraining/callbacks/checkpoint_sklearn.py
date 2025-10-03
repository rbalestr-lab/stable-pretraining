from typing import Optional

import numpy as np
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging
from tabulate import tabulate
from lightning.pytorch.loggers import WandbLogger

from .. import SKLEARN_AVAILABLE

if SKLEARN_AVAILABLE:
    from sklearn.base import ClassifierMixin, RegressorMixin
else:
    ClassifierMixin = None
    RegressorMixin = None


class SklearnCheckpoint(Callback):
    """Callback for saving and loading sklearn models in PyTorch Lightning checkpoints.

    This callback automatically detects sklearn models (Regressors and Classifiers)
    attached to the Lightning module and handles their serialization/deserialization
    during checkpoint save/load operations. This is necessary because sklearn models
    are not natively supported by PyTorch's checkpoint system.

    The callback will:
    1. Automatically discover sklearn models attached to the Lightning module
    2. Save them to the checkpoint dictionary during checkpoint saving
    3. Restore them from the checkpoint during checkpoint loading
    4. Log information about discovered sklearn modules during setup

    Note:
        - Only attributes that are sklearn RegressorMixin or ClassifierMixin instances are saved
        - Private attributes (starting with '_') are ignored
        - The callback will raise an error if a sklearn model name conflicts with existing checkpoint keys
    """

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None
    ) -> None:
        sklearn_modules = _get_sklearn_modules(pl_module)
        stats = []
        for name, module in sklearn_modules.items():
            stats.append((name, module.__str__(), type(module)))
        headers = ["Module", "Name", "Type"]
        logging.info("Setting up SklearnCheckpoint callback!")
        logging.info("Sklearn Modules:")
        logging.info(f"\n{tabulate(stats, headers, tablefmt='heavy_outline')}")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        logging.info("Checking for non PyTorch modules to save... 🔧")
        modules = _get_sklearn_modules(pl_module)
        for name, module in modules.items():
            if name in checkpoint:
                raise RuntimeError(
                    f"Can't pickle {name}, already present in checkpoint"
                )
            checkpoint[name] = module
            logging.info(f"Saving non PyTorch system: {name} 🔧")

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        logging.info("Checking for non PyTorch modules to load... 🔧")
        if not SKLEARN_AVAILABLE:
            return
        for name, item in checkpoint.items():
            if isinstance(item, RegressorMixin) or isinstance(item, ClassifierMixin):
                setattr(pl_module, name, item)
                logging.info(f"Loading non PyTorch system: {name} 🔧")


def _contains_sklearn_module(item):
    if not SKLEARN_AVAILABLE:
        return False
    if isinstance(item, RegressorMixin) or isinstance(item, ClassifierMixin):
        return True
    if isinstance(item, list):
        return np.any([_contains_sklearn_module(m) for m in item])
    if isinstance(item, dict):
        return np.any([_contains_sklearn_module(m) for m in item.values()])
    return False


def _get_sklearn_modules(module):
    modules = dict()
    for name, item in vars(module).items():
        if name[0] == "_":
            continue
        item = getattr(module, name)
        if _contains_sklearn_module(item):
            modules[name] = item
    return modules


class WandbCheckpoint(Callback):
    """Callback for saving and loading sklearn models in PyTorch Lightning checkpoints.

    This callback automatically detects sklearn models (Regressors and Classifiers)
    attached to the Lightning module and handles their serialization/deserialization
    during checkpoint save/load operations. This is necessary because sklearn models
    are not natively supported by PyTorch's checkpoint system.

    The callback will:
    1. Automatically discover sklearn models attached to the Lightning module
    2. Save them to the checkpoint dictionary during checkpoint saving
    3. Restore them from the checkpoint during checkpoint loading
    4. Log information about discovered sklearn modules during setup

    Note:
        - Only attributes that are sklearn RegressorMixin or ClassifierMixin instances are saved
        - Private attributes (starting with '_') are ignored
        - The callback will raise an error if a sklearn model name conflicts with existing checkpoint keys
    """

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None
    ) -> None:
        logging.info("Setting up WandbCheckpoint callback!")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        logging.info("Checking for Wandb params to save... 🔧")
        if isinstance(trainer.logger, WandbLogger):
            checkpoint["wandb"] = {"id": trainer.logger.version}
            # checkpoint["wandb_checkpoint_name"] = trainer.logger._checkpoint_name
            logging.info(f"Saving Wandb params {checkpoint['wandb']}")
        logging.info("Checking for Wandb params to save... Done!")

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        logging.info("Checking for Wandb init params... 🔧")
        if "wandb" in checkpoint:
            logging.info("Wandb info in checkpoint!!! Restoring same run... 🔧")
            if not hasattr(trainer, "logger"):
                logging.warning("Expected Trainer to have a logger, leaving...")
                return
            elif not isinstance(trainer.logger, WandbLogger):
                logging.warning(
                    f"Expected WandbLogger, got {trainer.logger}, leaving..."
                )
                return
            else:
                logging.info("Trainer has a WandbLogger!")
            import wandb

            if wandb.run is None and trainer.global_rank > 0:
                logging.info(
                    "Run not initialized yet, skipping since this is a slave process!"
                )
                return
            logging.info(
                f"Deleting current run {wandb.run.entity}/{wandb.run.project}/{wandb.run.id}... 🔧"
            )
            api = wandb.Api()
            run = api.run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
            wandb.finish()
            run.delete()
            trainer.logger._experiment = None
            wandb_id = checkpoint["wandb"]["id"]
            trainer.logger._wandb_init["id"] = wandb_id
            trainer.logger._id = wandb_id
            # to reset the run
            trainer.logger.experiment
            logging.info(
                f"New run {wandb.run.entity}/{wandb.run.project}/{wandb.run.id}... 🔧"
            )

            # trainer.logger._wandb_init = wandb_init
            # trainer.logger._project = trainer.logger._wandb_init.get("project")
            # trainer.logger._save_dir = trainer.logger._wandb_init.get("dir")
            # trainer.logger._name = trainer.logger._wandb_init.get("name")
            # trainer.logger._checkpoint_name = checkpoint["wandb_checkpoint_name"]
            # logging.info("Updated Wandb parameters: ")
            # logging.info(f"\t- project={trainer.logger._project}")
            # logging.info(f"\t- _save_dir={trainer.logger._save_dir}")
            # logging.info(f"\t- name={trainer.logger._name}")
            # logging.info(f"\t- id={trainer.logger._id}")
