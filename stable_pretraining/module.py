import re
import types
from functools import partial

import lightning as pl
import torch
import torchmetrics
from loguru import logger as logging
from tabulate import tabulate

from .optim.utils import create_optimizer, create_scheduler


class Module(pl.LightningModule):
    """PyTorch Lightning module using manual optimization with multi-optimizer support.

    Core usage
    - Provide a custom `forward(self, batch, stage)` via the `forward` argument at init.
    - During training, `forward` must return a dict with `state["loss"]` (a single joint loss).
      When multiple optimizers are configured, this joint loss is used for all optimizers.

    Optimizer configuration (`self.optim`)
    - Single optimizer:
      {"optimizer": str|dict|partial|Class, "scheduler": <see below>, "interval": "step"|"epoch", "frequency": int}
      - Optimizer accepted forms:
        * string name (e.g., "AdamW", "SGD") from torch.optim
        * dict: {"type": "AdamW", "lr": 1e-3, ...}
        * functools.partial: partial(torch.optim.AdamW, lr=1e-3)
        * optimizer class: torch.optim.AdamW
    - Multiple optimizers:
      {
        name: {
          "modules": "regex",                # assign params by module-name pattern (children inherit)
          "optimizer": str|dict|partial|Class, # optimizer factory (same accepted forms as above)
          "scheduler": str|dict|partial|Class, # flexible scheduler config (see below)
          "interval": "step"|"epoch",       # scheduler interval
          "frequency": int,                   # optimizer step frequency
          "monitor": str                      # (optional) for ReduceLROnPlateau; alternatively set inside scheduler dict
        }, ...
      }

    Parameter assignment (multi-optimizer)
    - Modules are matched by regex on their qualified name. Children inherit the parent's assignment
      unless they match a more specific pattern. Only direct parameters of each module are collected
      to avoid duplication.

    Schedulers (flexible)
    - Accepted forms: string name (e.g., "CosineAnnealingLR", "StepLR"), dict with {"type": "...", ...},
      functools.partial, or a scheduler class. Smart defaults are applied when params are omitted for
      common schedulers (CosineAnnealingLR, OneCycleLR, StepLR, ExponentialLR, ReduceLROnPlateau,
      LinearLR, ConstantLR). For ReduceLROnPlateau, a `monitor` key is added (default: "val_loss").
      You may specify `monitor` either alongside the optimizer config (top level) or inside the
      scheduler dict itself.
    - The resulting Lightning scheduler dict includes `interval` and `frequency` (or `scheduler_frequency`).

    Training loop behavior
    - Manual optimization (`automatic_optimization = False`).
    - Gradient accumulation: scales loss by 1/N where N = Trainer.accumulate_grad_batches and steps on the boundary.
    - Per-optimizer step frequency: each optimizer steps only when its frequency boundary is met (in addition to accumulation boundary).
    - Gradient clipping: uses Trainer's `gradient_clip_val` and `gradient_clip_algorithm` before each step.
    - Returns the `state` dict from `forward` unchanged for logging/inspection.
    """

    def __init__(self, *args, forward: callable, hparams: dict = None, **kwargs):
        super().__init__()
        logging.info("Initializing Module configuration...")

        # Manual optimization to support multiple optimizers and custom stepping
        self.automatic_optimization = False

        self._callbacks_modules = torch.nn.ModuleDict()
        self._callbacks_metrics = torch.nn.ModuleDict()

        if len(args) > 0:
            raise ValueError(
                "Module does not accept positional arguments (*args). Please use keyword arguments instead (e.g., Module(forward=my_forward, hparams=my_hparams))."
            )

        if hparams is None:
            logging.warning(
                "No hyperparameters provided - hyperparameter logging is disabled."
            )
        else:
            logging.info("Saving provided hyperparameters.")
            self.save_hyperparameters(hparams)

        logging.info("Setting custom forward method.")
        setattr(self, "forward", types.MethodType(forward, self))

        for key, value in kwargs.items():
            logging.info(f"Setting attribute: self.{key} = {type(value)}")
            setattr(self, key, value)

        headers = ["Stage", "Inputs", "Metric"]
        if hasattr(self, "metrics"):
            stats = []
            assert isinstance(self.metrics, torch.nn.ModuleDict)
            logging.info("Metrics:")
            for stage, metrics in self.metrics.items():
                assert (
                    isinstance(metrics, torch.nn.ModuleDict)
                    or isinstance(metrics, torch.nn.ModuleList)
                    or isinstance(metrics, torchmetrics.Metric)
                )
                for name, metric in metrics.items():
                    stats.append([stage, name, str(metric)])
            logging.info(f"\n{tabulate(stats, headers, tablefmt='heavy_outline')}")
        else:
            self.metrics = dict(train={}, validate={}, test={}, predict={})
            logging.info(
                "No metrics configuration provided - automatic metric tracking is disabled."
            )

        # Internal optimizer metadata filled in configure_optimizers
        self._optimizer_names = None
        self._optimizer_index_by_name = None
        self._optimizer_frequencies = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError("The forward() method must be implemented.")

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        """Override to globally exclude callback-related parameters.

        Excludes parameters that belong to `self._callbacks_modules` or `self._callbacks_metrics`.
        This prevents accidental optimization of callback/metric internals, even if external code
        calls `self.parameters()` or `self.named_parameters()` directly.
        """
        for name, param in super().named_parameters(prefix=prefix, recurse=recurse):
            if name.startswith("_callbacks_modules.") or name.startswith(
                "_callbacks_metrics."
            ):
                continue
            yield name, param

    def parameters(self, recurse: bool = True):
        """Override to route through the filtered `named_parameters` implementation."""
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def training_step(self, batch, batch_idx):
        """Manual optimization training step with support for multiple optimizers.

        Expected output from forward during training (stage="fit"):
        - state["loss"]: torch.Tensor - Single joint loss for all optimizers

        When multiple optimizers are configured, the same loss is used for all of them.
        Each optimizer updates its assigned parameters based on gradients from this joint loss.
        """
        state = self(batch, stage="fit")

        # Early exit if optimization disabled
        if getattr(self, "optim", None) is None or self.optim is False:
            return state

        if "loss" not in state:
            raise ValueError("Training step requires 'loss' in the output state.")

        # Resolve optimizers and schedulers (can be single or list)
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]

        schedulers = self.lr_schedulers()
        if schedulers is None:
            schedulers = []
        elif not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]

        # Get the joint loss
        loss = state["loss"]

        # Log training loss each step (and aggregate per epoch)
        log_value = loss.detach() if torch.is_tensor(loss) else float(loss)
        self.log(
            "train/loss",
            log_value,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Gradient accumulation factor
        # Check for transferred parameter first (from Lightning patch), then original
        accum = max(
            int(
                getattr(
                    self.trainer,
                    "accumulate_grad_batches_",
                    getattr(self.trainer, "accumulate_grad_batches", 1),
                )
            ),
            1,
        )
        scale = 1.0 / float(accum)

        # Compute gradients once for the joint loss
        self.manual_backward(loss * scale)

        # Stepping and gradient clipping at accumulation boundary
        if (batch_idx + 1) % accum == 0:
            for idx, opt in enumerate(optimizers):
                # Honor per-optimizer frequency if available
                step_freq = 1
                if self._optimizer_names and self._optimizer_frequencies:
                    name = self._optimizer_names[idx]
                    step_freq = int(self._optimizer_frequencies.get(name, 1))
                if step_freq < 1:
                    step_freq = 1

                if (batch_idx + 1) % step_freq != 0:
                    continue

                # Clip gradients for this optimizer then step
                # Check for transferred parameters first (from Lightning patch), then original
                clip_val = getattr(
                    self.trainer, "gradient_clip_val_", self.trainer.gradient_clip_val
                )
                clip_algo = getattr(
                    self.trainer,
                    "gradient_clip_algorithm_",
                    self.trainer.gradient_clip_algorithm,
                )

                if clip_val is not None and clip_val > 0:
                    self.clip_gradients(
                        opt,
                        gradient_clip_val=clip_val,
                        gradient_clip_algorithm=clip_algo,
                    )
                opt.step()
                opt.zero_grad(set_to_none=True)

                # Step matching scheduler if it exists
                if idx < len(schedulers) and schedulers[idx] is not None:
                    try:
                        schedulers[idx].step()
                    except Exception as e:
                        logging.warning(
                            f"Scheduler step failed for optimizer index {idx}: {e}"
                        )

        return state

    def validation_step(self, batch, batch_idx):
        state = self.forward(batch, stage="validate")
        return state

    def test_step(self, batch, batch_idx):
        state = self.forward(batch, stage="test")
        return state

    def predict_step(self, batch, batch_idx):
        state = self.forward(batch, stage="predict")
        return state

    def _get_scheduler_name(self, scheduler_config, scheduler_instance=None):
        """Extract scheduler name from various config formats."""
        if isinstance(scheduler_config, str):
            return scheduler_config
        elif isinstance(scheduler_config, dict):
            return scheduler_config.get("type", "CosineAnnealingLR")
        elif hasattr(scheduler_config, "func"):  # partial
            return scheduler_config.func.__name__
        elif scheduler_instance:
            return scheduler_instance.__class__.__name__
        else:
            return "Unknown"

    def _build_scheduler_config(self, scheduler, config, name=None):
        """Build scheduler config dict for Lightning."""
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": config.get("interval", "step"),
            "frequency": config.get("scheduler_frequency", config.get("frequency", 1)),
        }

        if name:
            scheduler_dict["name"] = name

        # Add monitor for ReduceLROnPlateau
        scheduler_cfg = config.get("scheduler", "CosineAnnealingLR")
        scheduler_name = self._get_scheduler_name(scheduler_cfg, scheduler)
        if scheduler_name == "ReduceLROnPlateau":
            # Prefer nested monitor inside scheduler dict, fallback to top-level
            nested_monitor = None
            if isinstance(scheduler_cfg, dict):
                nested_monitor = scheduler_cfg.get("monitor")
            scheduler_dict["monitor"] = nested_monitor or config.get(
                "monitor", "val_loss"
            )

        return scheduler_dict

    def _collect_parameters_by_optimizer_groups(self, optim_items):
        """Assign modules and collect parameters per optimizer group defined by regex.

        Args:
            optim_items: list of (name, config) where config contains a "modules" regex
                describing group membership.

        Returns:
            params_by_name: dict[name, List[nn.Parameter]]
            modules_by_name: dict[name, List[str]]
        """
        # Pre-compile regex with stable order from optim_items
        compiled = [
            (name, re.compile(config["modules"])) for name, config in optim_items
        ]

        # Initialize containers
        params_by_name = {name: [] for name, _ in compiled}
        modules_by_name = {name: [] for name, _ in compiled}

        # Map module -> group index with inheritance
        module_to_group = {}
        for qual_name, module in self.named_modules():
            if "_callbacks_modules" in qual_name or "_callbacks_metrics" in qual_name:
                continue

            # inherit parent's group if any
            if "." in qual_name:
                parent_name = qual_name.rsplit(".", 1)[0]
                group_idx = module_to_group.get(parent_name)
            else:
                group_idx = None

            # override if explicit match
            for idx, (_, regex) in enumerate(compiled):
                if regex.match(qual_name):
                    group_idx = idx
                    break

            module_to_group[qual_name] = group_idx

            if group_idx is not None:
                group_name = compiled[group_idx][0]
                # record module name
                modules_by_name[group_name].append(qual_name)
                # collect direct parameters only to avoid duplication
                direct_params = list(module.parameters(recurse=False))
                if direct_params:
                    params_by_name[group_name].extend(direct_params)

        # Logging summary
        rows = []
        for group_name, config in optim_items:
            pattern = config.get("modules", "")
            tensors = params_by_name[group_name]
            num_tensors = len(tensors)
            num_elements = sum(int(p.numel()) for p in tensors)
            num_requires_grad = sum(int(p.requires_grad) for p in tensors)
            rows.append(
                [
                    group_name,
                    pattern,
                    len(modules_by_name[group_name]),
                    num_tensors,
                    num_elements,
                    num_requires_grad,
                ]
            )

        if rows:
            headers = [
                "Optimizer",
                "Pattern",
                "Matched Modules",
                "Param Tensors",
                "Total Params",
                "RequiresGrad Tensors",
            ]
            logging.info(
                "\n" + tabulate(rows, headers=headers, tablefmt="heavy_outline")
            )

        return params_by_name, modules_by_name

    def configure_optimizers(self):
        """Configure optimizers and schedulers for manual optimization.

        Returns:
            dict or tuple: Optimizer configuration with optional learning rate scheduler.
            For single optimizer: Returns a dict with optimizer and lr_scheduler.
            For multiple optimizers: Returns a tuple of (optimizers, schedulers).

        Example:
            Multi-optimizer configuration with module pattern matching and schedulers:

            >>> # Simple single optimizer with scheduler
            >>> self.optim = {
            ...     "optimizer": partial(torch.optim.AdamW, lr=1e-3),
            ...     "scheduler": "CosineAnnealingLR",  # Uses smart defaults
            ...     "interval": "step",
            ...     "frequency": 1,
            ... }

            >>> # Multi-optimizer with custom scheduler configs
            >>> self.optim = {
            ...     "encoder_opt": {
            ...         "modules": "encoder",  # Matches 'encoder' and all children
            ...         "optimizer": {"type": "AdamW", "lr": 1e-3},
            ...         "scheduler": {
            ...             "type": "OneCycleLR",
            ...             "max_lr": 1e-3,
            ...             "total_steps": 10000,
            ...         },
            ...         "interval": "step",
            ...         "frequency": 1,
            ...     },
            ...     "head_opt": {
            ...         "modules": ".*head$",  # Matches modules ending with 'head'
            ...         "optimizer": "SGD",
            ...         "scheduler": {
            ...             "type": "ReduceLROnPlateau",
            ...             "mode": "max",
            ...             "patience": 5,
            ...             "factor": 0.5,
            ...         },
            ...         "monitor": "val_accuracy",  # Required for ReduceLROnPlateau
            ...         "interval": "epoch",
            ...         "frequency": 2,
            ...     },
            ... }

            With model structure:
            - encoder                 -> encoder_opt (matches "encoder")
            - encoder.layer1          -> encoder_opt (inherits from parent)
            - encoder.layer1.conv     -> encoder_opt (inherits from encoder.layer1)
            - classifier_head         -> head_opt (matches ".*head$")
            - classifier_head.linear  -> head_opt (inherits from parent)
            - decoder                 -> None (no match, no parameters collected)
        """
        logging.info("Configuring optimizers and learning rate schedulers...")

        # Early exit for disabled optimization
        if hasattr(self, "optim") and not self.optim:
            logging.info("Optimization disabled - skipping optimizer configuration.")
            return None

        if not hasattr(self, "optim"):
            logging.info(
                "Using default optimization setup: AdamW optimizer with CosineAnnealingLR scheduler."
            )
            self.optim = dict(optimizer=partial(torch.optim.AdamW))
        elif isinstance(self.optim, partial):
            self.optim = dict(optimizer=self.optim)

        # Single optimizer case
        optimizer_cfg = self.optim.get("optimizer")
        if isinstance(optimizer_cfg, (str, dict)) or hasattr(optimizer_cfg, "__call__"):
            logging.info("Configuring single optimizer.")

            # Direct parameter extraction - use globally filtered parameters
            params = list(self.parameters())

            opt = create_optimizer(params, optimizer_cfg or "AdamW")

            # Create scheduler
            default = dict(
                type="CosineAnnealingLR", T_max=self.trainer.estimated_stepping_batches
            )
            sched_config = self.optim.get("scheduler", default)
            sched = create_scheduler(opt, sched_config, module=self)
            sched_name = self._get_scheduler_name(sched_config, sched)

            logging.info(
                f"Configured {opt.__class__.__name__} optimizer with {sched_name} scheduler."
            )

            # Track names/frequencies for training_step
            self._optimizer_names = ["default"]
            self._optimizer_index_by_name = {"default": 0}
            self._optimizer_frequencies = {
                "default": int(self.optim.get("frequency", 1))
            }

            # Build scheduler config dict for Lightning
            scheduler_dict = self._build_scheduler_config(sched, self.optim)

            # Return in list/dict style compatible with lr_schedulers() access
            return [opt], [scheduler_dict]

        # Multiple optimizers case - check once
        if not isinstance(self.optim, dict):
            raise ValueError(
                "Optimizer must be either a partial function or a dict of optimizer configs"
            )

        # Verify all values are dicts
        optim_items = list(self.optim.items())
        if not all(isinstance(v, dict) for _, v in optim_items):
            raise ValueError("For multiple optimizers, all config values must be dicts")

        logging.info(
            f"\tOptimizer specified by Dict with keys {[k for k, _ in optim_items]}... 🔧"
        )

        # Build grouping with detailed logging
        params_by_name, modules_by_name = self._collect_parameters_by_optimizer_groups(
            optim_items
        )

        # Build optimizers and schedulers
        optimizers = []
        schedulers = []

        self._optimizer_names = []
        self._optimizer_index_by_name = {}
        self._optimizer_frequencies = {}

        for name, config in optim_items:
            params = params_by_name.get(name, [])
            if not params:
                logging.warning(f"No parameters matched for optimizer {name}")
                # skip registration when there are no parameters
                continue

            opt = create_optimizer(params, config["optimizer"])
            optimizers.append(opt)

            sched_config = config.get("scheduler", "CosineAnnealingLR")
            scheduler = create_scheduler(opt, sched_config, module=self)
            sched_name = self._get_scheduler_name(sched_config, scheduler)

            # Build scheduler config dict for Lightning
            scheduler_dict = self._build_scheduler_config(scheduler, config, name)
            schedulers.append(scheduler_dict)

            logging.info(
                f"Configured optimizer '{name}' (modules={len(modules_by_name.get(name, []))}, "
                f"param_tensors={len(params)}, total_params={sum(int(p.numel()) for p in params)}) "
                f"with {sched_name} scheduler."
            )

            # Track names and frequencies aligned to optimizer order
            self._optimizer_names.append(name)
            self._optimizer_index_by_name[name] = len(optimizers) - 1
            self._optimizer_frequencies[name] = int(config.get("frequency", 1))

        return optimizers, schedulers
