from functools import partial
from typing import Optional, Union

import torch
import torchmetrics
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging

from ..optim.utils import create_optimizer, create_scheduler
from ..optim import LARS


class TrainableCallback(Callback):
    """Base callback class with optimizer and scheduler management.

    This base class handles the common logic for callbacks that need their own
    optimizer and scheduler, including automatic inheritance from the main module's
    configuration when not explicitly specified.

    Subclasses should:
    1. Call super().__init__() with appropriate parameters
    2. Store their module configuration in self._module_config
    3. Override _initialize_module() to create their specific module
    4. Access their module via self.module property after setup
    """

    def __init__(
        self,
        name: str,
        optimizer: Optional[Union[str, dict, partial, torch.optim.Optimizer]] = None,
        scheduler: Optional[
            Union[str, dict, partial, torch.optim.lr_scheduler.LRScheduler]
        ] = None,
        accumulate_grad_batches: int = 1,
    ):
        """Initialize base callback with optimizer/scheduler configuration.

        Args:
            name: Unique identifier for this callback instance.
            optimizer: Optimizer configuration. If None, inherits from main module.
            scheduler: Scheduler configuration. If None, inherits from main module.
            accumulate_grad_batches: Number of batches to accumulate gradients.
        """
        super().__init__()
        self.name = name
        self.accumulate_grad_batches = accumulate_grad_batches

        # Store configurations
        self._optimizer_config = optimizer
        self._scheduler_config = scheduler
        self._pl_module = None

        # Will be initialized in setup
        self.optimizer = None
        self.scheduler = None

    def _initialize_module(self, pl_module: LightningModule) -> torch.nn.Module:
        """Initialize the module for this callback.

        Subclasses must override this method to create their specific module.

        Args:
            pl_module: The Lightning module being trained.

        Returns:
            The initialized module.
        """
        raise NotImplementedError("Subclasses must implement _initialize_module")

    def setup_module(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize and store the module.

        This method handles module initialization and storage in _callbacks_modules.
        Subclasses can override this if they need custom module setup logic.
        """
        # Initialize module
        module = self._initialize_module(pl_module)

        # Ensure all parameters require gradients
        for param in module.parameters():
            param.requires_grad = True

        # Move to correct device
        module = module.to(pl_module.device)

        # Store module in pl_module._callbacks_modules
        logging.info(f"{self.name}: Storing module in _callbacks_modules")
        if not hasattr(pl_module, "_callbacks_modules"):
            pl_module._callbacks_modules = {}
        pl_module._callbacks_modules[self.name] = module

    def setup_optimizer(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize optimizer with inheritance from main module if needed."""
        if self._optimizer_config is None:
            # Try to inherit from main module's optimizer config
            if hasattr(pl_module, "optim") and pl_module.optim:
                if isinstance(pl_module.optim, dict):
                    if "optimizer" in pl_module.optim:
                        main_opt_config = pl_module.optim["optimizer"]
                        logging.info(
                            f"{self.name}: Inheriting optimizer config from main module"
                        )
                    else:
                        # Multi-optimizer config, use the first one
                        first_opt_key = next(iter(pl_module.optim.keys()))
                        main_opt_config = pl_module.optim[first_opt_key].get(
                            "optimizer", "LARS"
                        )
                        logging.info(
                            f"{self.name}: Inheriting optimizer config from '{first_opt_key}'"
                        )

                    self.optimizer = create_optimizer(
                        self.module.parameters(), main_opt_config
                    )
                else:
                    # Fallback to LARS
                    logging.info(
                        f"{self.name}: Main module optim format not recognized, using LARS"
                    )
                    self.optimizer = LARS(
                        self.module.parameters(),
                        lr=0.1,
                        clip_lr=True,
                        eta=0.02,
                        exclude_bias_n_norm=True,
                        weight_decay=0,
                    )
            else:
                # No main module config, use default LARS
                logging.info(
                    f"{self.name}: No main module optimizer config found, using default LARS"
                )
                self.optimizer = LARS(
                    self.module.parameters(),
                    lr=0.1,
                    clip_lr=True,
                    eta=0.02,
                    exclude_bias_n_norm=True,
                    weight_decay=0,
                )
        else:
            # Use explicitly provided optimizer config
            self.optimizer = create_optimizer(
                self.module.parameters(), self._optimizer_config
            )

    def setup_scheduler(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize scheduler with inheritance from main module if needed."""
        if self._scheduler_config is None:
            # Try to inherit from main module's scheduler config
            if hasattr(pl_module, "optim") and pl_module.optim:
                if isinstance(pl_module.optim, dict):
                    if "scheduler" in pl_module.optim:
                        main_sched_config = pl_module.optim.get(
                            "scheduler", "CosineAnnealingLR"
                        )
                        logging.info(
                            f"{self.name}: Inheriting scheduler config from main module"
                        )
                        self.scheduler = create_scheduler(
                            self.optimizer, main_sched_config, module=pl_module
                        )
                    else:
                        # Multi-optimizer config, use the first one
                        first_opt_key = next(iter(pl_module.optim.keys()))
                        main_sched_config = pl_module.optim[first_opt_key].get(
                            "scheduler", "CosineAnnealingLR"
                        )
                        logging.info(
                            f"{self.name}: Inheriting scheduler config from '{first_opt_key}'"
                        )
                        self.scheduler = create_scheduler(
                            self.optimizer, main_sched_config, module=pl_module
                        )
                else:
                    # Fallback to ConstantLR
                    logging.info(f"{self.name}: Using default ConstantLR scheduler")
                    self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                        self.optimizer, factor=1.0
                    )
            else:
                # No main module config, use default
                logging.info(
                    f"{self.name}: No main module scheduler config found, using ConstantLR"
                )
                self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                    self.optimizer, factor=1.0
                )
        else:
            # Use explicitly provided scheduler config
            self.scheduler = create_scheduler(
                self.optimizer, self._scheduler_config, module=pl_module
            )

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Setup module, optimizer, and scheduler."""
        logging.info(f"Setting up {self.state_key} callback!")
        if stage != "fit":
            return

        self._pl_module = pl_module
        self.setup_module(trainer, pl_module)
        self.setup_optimizer(trainer, pl_module)
        self.setup_scheduler(trainer, pl_module)

        logging.info(f"{self.name}: Setup complete")

    def optimizer_step(self, batch_idx: int, trainer: Trainer) -> None:
        """Perform optimizer step with gradient accumulation support."""
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

    @property
    def module(self):
        """Access module from pl_module._callbacks_modules.

        This property is only accessible after setup() has been called.
        The module is stored centrally in pl_module._callbacks_modules
        to avoid duplication in checkpoints.
        """
        if self._pl_module is None:
            raise AttributeError(
                f"{self.name}: module not accessible before setup(). "
                "The module is initialized during the setup phase."
            )
        return self._pl_module._callbacks_modules[self.name]

    @property
    def state_key(self) -> str:
        """Unique identifier for this callback's state during checkpointing."""
        return f"{self.__class__.__name__}[name={self.name}]"

    def state_dict(self) -> dict:
        """Save callback state - only optimizer and scheduler states.

        The module itself is saved via pl_module._callbacks_modules,
        so we don't duplicate it here.
        """
        return {
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load callback state - only optimizer and scheduler states.

        The module itself is loaded via pl_module._callbacks_modules,
        so we don't handle it here.
        """
        if "optimizer" in state_dict and self.optimizer:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        if "scheduler" in state_dict and self.scheduler:
            self.scheduler.load_state_dict(state_dict["scheduler"])


class EarlyStopping(torch.nn.Module):
    """Early stopping mechanism with support for metric milestones and patience.

    This module provides flexible early stopping capabilities that can halt training
    based on metric performance. It supports both milestone-based stopping (stop if
    metric doesn't reach target by specific epochs) and patience-based stopping
    (stop if metric doesn't improve for N epochs).

    Args:
        mode: Optimization direction - 'min' for metrics to minimize (e.g., loss),
            'max' for metrics to maximize (e.g., accuracy).
        milestones: Dict mapping epoch numbers to target metric values. Training
            stops if targets are not met at specified epochs.
        metric_name: Name of the metric to monitor if metric is a dict.
        patience: Number of epochs with no improvement before stopping.

    Example:
        >>> early_stop = EarlyStopping(mode="max", milestones={10: 0.8, 20: 0.9})
        >>> # Stops if accuracy < 0.8 at epoch 10 or < 0.9 at epoch 20
    """

    def __init__(
        self,
        mode: str = "min",
        milestones: dict[int, float] = None,
        metric_name: str = None,
        patience: int = 10,
    ):
        super().__init__()
        self.mode = mode
        self.milestones = milestones or {}
        self.metric_name = metric_name
        self.patience = patience
        self.register_buffer("history", torch.zeros(patience))

    def should_stop(self, metric, step):
        if self.metric_name is None:
            assert type(metric) is not dict
        else:
            assert self.metric_name in metric
            metric = metric[self.metric_name]
        if step in self.milestones:
            if self.mode == "min":
                return metric > self.milestones[step]
            elif self.mode == "max":
                return metric < self.milestones[step]
        return False


def format_metrics_as_dict(metrics):
    """Formats various metric input formats into a standardized dictionary structure.

    This utility function handles multiple input formats for metrics and converts
    them into a consistent ModuleDict structure with separate train and validation
    metrics. This standardization simplifies metric handling across callbacks.

    Args:
        metrics: Can be:
            - None: Returns empty train and val dicts
            - Single torchmetrics.Metric: Applied to validation only
            - Dict with 'train' and 'val' keys: Separated accordingly
            - Dict of metrics: All applied to validation
            - List/tuple of metrics: All applied to validation

    Returns:
        ModuleDict with '_train' and '_val' keys, each containing metric ModuleDicts.

    Raises:
        ValueError: If metrics format is invalid or contains non-torchmetric objects.
    """
    if metrics is None:
        train = {}
        eval = {}
    elif isinstance(metrics, torchmetrics.Metric):
        train = {}
        eval = torch.nn.ModuleDict({metrics.__class__.__name__: metrics})
    elif type(metrics) is dict and set(metrics.keys()) == set(["train", "val"]):
        if type(metrics["train"]) in [list, tuple]:
            train = {}
            for m in metrics["train"]:
                if not isinstance(m, torchmetrics.Metric):
                    raise ValueError(f"metric {m} is no a torchmetric")
                train[m.__class__.__name__] = m
        else:
            train = metrics["train"]
        if type(metrics["val"]) in [list, tuple]:
            eval = {}
            for m in metrics["val"]:
                if not isinstance(m, torchmetrics.Metric):
                    raise ValueError(f"metric {m} is no a torchmetric")
                eval[m.__class__.__name__] = m
        else:
            eval = metrics["eval"]
    elif type(metrics) is dict:
        train = {}
        for k, v in metrics.items():
            assert type(k) is str
            assert isinstance(v, torchmetrics.Metric)
        eval = metrics
    elif type(metrics) in [list, tuple]:
        train = {}
        for m in metrics:
            if not isinstance(m, torchmetrics.Metric):
                raise ValueError(f"metric {m} is no a torchmetric")
        eval = {m.__class__.__name__: m for m in metrics}
    else:
        raise ValueError(
            "metrics can only be a torchmetric of list/tuple of torchmetrics"
        )
    return torch.nn.ModuleDict(
        {
            "_train": torch.nn.ModuleDict(train),
            "_val": torch.nn.ModuleDict(eval),
        }
    )
