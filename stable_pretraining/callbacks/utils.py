from functools import partial
from typing import Optional, Union

import torch
import torchmetrics
from lightning.pytorch import Callback, LightningModule
from loguru import logger as logging
import types
from ..optim import create_optimizer, create_scheduler, LARS


class TrainableCallback(Callback):
    """Base callback class with optimizer and scheduler management.

    This base class handles the common logic for callbacks that need their own
    optimizer and scheduler, including automatic inheritance from the main module's
    configuration when not explicitly specified.

    Subclasses should:
    1. Call super().__init__() with appropriate parameters
    2. Store their module configuration in self._module_config
    3. Override configure_model() to create their specific module
    4. Access their module via self.module property after setup
    """

    def __init__(
        self,
        module: LightningModule,
        name: str,
        optimizer: Optional[Union[str, dict, partial, torch.optim.Optimizer]] = None,
        scheduler: Optional[
            Union[str, dict, partial, torch.optim.lr_scheduler.LRScheduler]
        ] = None,
        accumulate_grad_batches: int = 1,
    ):
        """Initialize base callback with optimizer/scheduler configuration.

        Args:
            module: spt.Module.
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
        self._pl_module = module
        self.setup_model(module)
        self.wrap_configure_optimizers(module)

    def setup_model(self, pl_module: LightningModule) -> None:
        """Initialize and store the module.

        This method handles module initialization and storage in callbacks_modules.
        Subclasses can override this if they need custom module setup logic.
        """
        # Initialize module
        module = self.configure_model(pl_module)

        # Move to correct device
        module = module.to(pl_module.device)

        # Store module in pl_module.callbacks_modules
        logging.info(f"{self.name}: Storing module in callbacks_modules")
        pl_module.callbacks_modules[self.name] = module

    def configure_model(self, pl_module: LightningModule) -> torch.nn.Module:
        """Initialize the module for this callback.

        Subclasses must override this method to create their specific module.

        Args:
            pl_module: The Lightning module being trained.

        Returns:
            The initialized module.
        """
        raise NotImplementedError("Subclasses must implement configure_model")

    def wrap_configure_optimizers(self, pl_module):
        fn = pl_module.configure_optimizers

        def new_configure_optimizers(self, callback=self, fn=fn):
            outputs = fn()
            if outputs is None:
                optimizers = []
                schedulers = []
            else:
                optimizers, schedulers = outputs
            assert callback.name not in self._optimizer_index_by_name
            assert callback.name not in self._optimizer_frequencies
            assert callback.name not in self._optimizer_names
            self._optimizer_index_by_name[callback.name] = len(self._optimizer_names)
            self._optimizer_names.append(callback.name)
            self._optimizer_frequencies[callback.name] = (
                callback.accumulate_grad_batches
            )
            optimizers.append(callback.setup_optimizer(self))
            schedulers.append(callback.setup_scheduler(optimizers[-1], self))
            return optimizers, schedulers

        # Bind the new method to the instance
        logging.info(f"{self.name}: We are wrapping up your `configure_optimizers`!")
        pl_module.configure_optimizers = types.MethodType(
            new_configure_optimizers, pl_module
        )

    def setup_optimizer(self, pl_module: LightningModule) -> None:
        """Initialize optimizer with inheritance from main module if needed."""
        if self._optimizer_config is None:
            # Try to inherit from main module's optimizer config
            logging.info(f"{self.name}: No optimizer given, using default LARS")
            return LARS(
                self.module.parameters(),
                lr=0.1,
                clip_lr=True,
                eta=0.02,
                exclude_bias_n_norm=True,
                weight_decay=0,
            )
        # Use explicitly provided optimizer config
        logging.info(f"{self.name}: Use explicitly provided optimizer")
        return create_optimizer(self.module.parameters(), self._optimizer_config)

    def setup_scheduler(self, optimizer, pl_module: LightningModule) -> None:
        """Initialize scheduler with inheritance from main module if needed."""
        if self._scheduler_config is None:
            # Fallback to ConstantLR
            logging.info(f"{self.name}:No optimizer given, using default ConstantLR")
            return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        logging.info(f"{self.name}: Use explicitly provided scheduler")
        return create_scheduler(optimizer, self._scheduler_config, module=pl_module)

    @property
    def module(self):
        """Access module from pl_module.callbacks_modules.

        This property is only accessible after setup() has been called.
        The module is stored centrally in pl_module.callbacks_modules
        to avoid duplication in checkpoints.
        """
        if self._pl_module is None:
            raise AttributeError(
                f"{self.name}: module not accessible before setup(). "
                "The module is initialized during the setup phase."
            )
        return self._pl_module.callbacks_modules[self.name]

    @property
    def state_key(self) -> str:
        """Unique identifier for this callback's state during checkpointing."""
        return f"{self.__class__.__name__}[name={self.name}]"


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
    # Handle OmegaConf types
    from omegaconf import ListConfig, DictConfig

    if isinstance(metrics, (ListConfig, DictConfig)):
        import omegaconf

        metrics = omegaconf.OmegaConf.to_container(metrics, resolve=True)

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
            eval = metrics["val"]
    elif type(metrics) is dict:
        train = {}
        for k, v in metrics.items():
            assert type(k) is str
            assert isinstance(v, torchmetrics.Metric)
        eval = metrics
    elif type(metrics) in [list, tuple]:
        train = {}
        eval = {}
        for m in metrics:
            if not isinstance(m, torchmetrics.Metric):
                raise ValueError(f"metric {m} is no a torchmetric")
            eval[m.__class__.__name__] = m
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
