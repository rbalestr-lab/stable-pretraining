from functools import partial
from typing import Dict, Optional, Union

import torch
import torchmetrics
from hydra.utils import instantiate
from lightning.pytorch import LightningModule, Trainer
from loguru import logger as logging

from ..utils import get_data_from_batch_or_outputs

from .utils import EarlyStopping, TrainableCallback, format_metrics_as_dict


class OnlineProbe(TrainableCallback):
    """Online probe for evaluating learned representations during self-supervised training.

    This callback implements the standard linear evaluation protocol by training a probe
    (typically a linear classifier) on top of frozen features from the main model. The probe
    is trained simultaneously with the main model but maintains its own optimizer, scheduler,
    and training loop. This allows monitoring representation quality throughout training
    without modifying the base model.

    Key features:
    - Automatic gradient detachment to prevent probe gradients affecting the main model
    - Independent optimizer and scheduler management
    - Support for gradient accumulation
    - Mixed precision training compatibility through automatic dtype conversion
    - Built-in early stopping support
    - Metric tracking and logging

    Args:
        name: Unique identifier for this probe instance. Used for logging and storing
            metrics/modules.
        input: Key in batch dict or outputs dict containing input features to probe.
        target: Key in batch dict containing ground truth target labels.
        probe: The probe module to train. Can be a nn.Module instance, callable that
            returns a module, or Hydra config to instantiate.
        loss_fn: Loss function for probe training (e.g., nn.CrossEntropyLoss()).
        optimizer: Optimizer configuration for the probe. Can be:
            - str: optimizer name (e.g., "AdamW", "SGD", "LARS")
            - dict: {"type": "AdamW", "lr": 1e-3, ...}
            - partial: pre-configured optimizer factory
            - optimizer instance or callable
            - None: inherits from main Module's optimizer config (default)
        scheduler: Learning rate scheduler configuration. Can be:
            - str: scheduler name (e.g., "CosineAnnealingLR", "StepLR")
            - dict: {"type": "CosineAnnealingLR", "T_max": 1000, ...}
            - partial: pre-configured scheduler factory
            - scheduler instance or callable
            - None: inherits from main Module's scheduler config (default)
        accumulate_grad_batches: Number of batches to accumulate gradients before
            optimizer step. Default is 1 (no accumulation).
        metrics: Metrics to track during training/validation. Can be dict, list, tuple,
            or single metric instance.
        early_stopping: Early stopping configuration to halt training if validation
            metric stops improving.

    Note:
        - The probe module is stored in pl_module._callbacks_modules[name]
        - Metrics are stored in pl_module._callbacks_metrics[name]
        - Predictions are stored in batch dict with key '{name}_preds'
        - Loss is logged as 'train/{name}_loss'
        - Metrics are logged with prefix 'train/{name}_' and 'eval/{name}_'
    """

    def __init__(
        self,
        name: str,
        input: str,
        target: str,
        probe: torch.nn.Module,
        loss_fn: callable,
        optimizer: Optional[Union[str, dict, partial, torch.optim.Optimizer]] = None,
        scheduler: Optional[
            Union[str, dict, partial, torch.optim.lr_scheduler.LRScheduler]
        ] = None,
        accumulate_grad_batches: int = 1,
        metrics: Optional[Union[dict, tuple, list, torchmetrics.Metric]] = None,
        early_stopping: Optional[EarlyStopping] = None,
    ) -> None:
        # Initialize base class
        super().__init__(
            name=name,
            optimizer=optimizer,
            scheduler=scheduler,
            accumulate_grad_batches=accumulate_grad_batches,
        )

        self.input = input
        self.target = target
        self.loss_fn = loss_fn
        self.early_stopping = early_stopping

        # Store probe configuration for later initialization
        self._probe_config = probe

        # These will be initialized in setup
        self._train_metrics = None
        self._val_metrics = None

        # Format metrics
        self.metrics_config = metrics

        logging.info(f"Initialized OnlineProbe callback: {name}")
        logging.info(f"  - Input: {input}")
        logging.info(f"  - Target: {target}")
        logging.info(f"  - Accumulate grad batches: {accumulate_grad_batches}")

    def _initialize_module(self, pl_module: LightningModule) -> torch.nn.Module:
        """Initialize the probe module from configuration."""
        if isinstance(self._probe_config, torch.nn.Module):
            probe_module = self._probe_config
        elif callable(self._probe_config):
            probe_module = self._probe_config()
        else:
            probe_module = instantiate(self._probe_config, _convert_="object")

        return probe_module

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Initialize optimizer, scheduler, and metrics."""
        # Call parent setup for module/optimizer/scheduler
        super().setup(trainer, pl_module, stage)

        if stage != "fit":
            return

        # Setup metrics
        logging.info(f"{self.name}: Setting up metrics")
        if not hasattr(pl_module, "_callbacks_metrics"):
            logging.info(
                "attaching a `_callbacks_metrics` to your LightningModule for callbacks"
            )
            pl_module._callbacks_metrics = {}
        pl_module._callbacks_metrics[self.name] = format_metrics_as_dict(
            self.metrics_config
        )

        self._train_metrics = pl_module._callbacks_metrics[self.name]["_train"]
        self._val_metrics = pl_module._callbacks_metrics[self.name]["_val"]
        pl_module.register_forward_hook(self.forward_hook_fn)
        logging.info(f"Main module forward hooks: {pl_module._forward_hooks}")

    def forward_hook_fn(self, pl_module, args, outputs) -> None:
        """Perform probe training step."""
        # Extract batch from args tuple (it's the first argument to forward)
        if isinstance(args, tuple) and len(args) > 0:
            batch = args[0]
        else:
            batch = args if not isinstance(args, tuple) else {}

        x = get_data_from_batch_or_outputs(
            self.input, batch, outputs, caller_name=self.name
        )
        y = get_data_from_batch_or_outputs(
            self.target, batch, outputs, caller_name=self.name
        )

        if x is None or y is None:
            logging.warning(f"Callback {self.name} missing x or y")
            return

        self.module.train()

        with torch.enable_grad():
            x = x.detach()

            probe_dtype = next(self.module.parameters()).dtype
            if x.dtype != probe_dtype:
                x = x.to(probe_dtype)

            preds = self.module(x)
            if pl_module.trainer.training:
                loss = self.loss_fn(preds, y)

                loss = loss / self.accumulate_grad_batches

                outputs["loss"] += loss
                logs = {
                    f"train/{self.name}_loss": loss.item()
                    * self.accumulate_grad_batches
                }
            else:
                logs = {}

        prediction_key = f"{self.name}_preds"
        if prediction_key not in batch:
            outputs[prediction_key] = preds.detach()

        for metric_name, metric in pl_module._callbacks_metrics[self.name][
            "_train"
        ].items():
            metric(preds.detach(), y)
            logs[f"train/{self.name}_{metric_name}"] = metric

        pl_module.log_dict(logs, on_step=True, on_epoch=True, sync_dist=True)
        return outputs

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
    ) -> None:
        # Optimizer step using parent class method
        self.optimizer_step(batch_idx, trainer)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute probe predictions during validation."""
        # Get input and target data
        x = get_data_from_batch_or_outputs(
            self.input, batch, outputs, caller_name=self.name
        )
        y = get_data_from_batch_or_outputs(
            self.target, batch, outputs, caller_name=self.name
        )

        if x is None or y is None:
            logging.warning(
                "OnlineProbe callback doesn't have access to its `x` or `y` tensor!"
            )
            return

        # Ensure probe is in eval mode
        self.module.eval()

        # Forward pass without gradients
        with torch.inference_mode():
            # Ensure input has same dtype as probe module
            # This handles mixed precision training where features might be float16
            probe_dtype = next(self.module.parameters()).dtype
            if x.dtype != probe_dtype:
                x = x.to(probe_dtype)

            preds = self.module(x)

        # Store predictions in batch
        prediction_key = f"{self.name}_preds"
        if prediction_key not in batch:
            batch[prediction_key] = preds

        # Update metrics and log
        logs = {}
        for metric_name, metric in pl_module._callbacks_metrics[self.name][
            "_val"
        ].items():
            metric(preds, y)
            logs[f"eval/{self.name}_{metric_name}"] = metric

        pl_module.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Handle early stopping if configured."""
        if self.early_stopping is None:
            return
        logging.info(f"{self.name} checking for early stopping condition")
        # Get the metric value for early stopping
        metric_name = f"eval/{self.name}_{self.early_stopping.monitor}"
        if metric_name not in trainer.callback_metrics:
            logging.warning(
                f"{self.name}: Early stopping metric {metric_name} not found"
            )
            return

        current_value = trainer.callback_metrics[metric_name]
        should_stop = self.early_stopping.should_stop(
            current_value, trainer.current_epoch
        )

        if should_stop:
            logging.info(
                f"{self.name}: Early stopping triggered at epoch {trainer.current_epoch} by {self.name}"
            )
            trainer.should_stop = True

    @property
    def probe_module(self):
        """Alias for self.module for backward compatibility."""
        return self.module
