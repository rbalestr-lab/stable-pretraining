import math
from loguru import logger
from lightning.pytorch import Callback, Trainer, LightningModule


class WeightDecayUpdater(Callback):
    """PyTorch Lightning Callback to update optimizer's weight decay per batch.

    - Supports multiple schedules: 'constant', 'linear', 'cosine', 'exponential'
    - Optionally specify which optimizer param group(s) to update (by index)
    - Infers total steps from Trainer config (max_steps or max_epochs + dataloader)
    - Checkpointable: state is saved/restored with Trainer checkpoints
    - Extensive Loguru logging
    Args:
        schedule_type (str): One of 'constant', 'linear', 'cosine', 'exponential'
        start_value (float): Initial weight decay value
        end_value (float): Final weight decay value (for non-constant schedules)
        param_group_indices (list[int] or None): List of param group indices to update. If None, updates all.
    """

    def __init__(
        self,
        schedule_type: str = "linear",
        start_value: float = 0.01,
        end_value: float = 0.0,
        param_group_indices: list = None,
    ):
        super().__init__()
        self.schedule_type = schedule_type
        self.start_value = start_value
        self.end_value = end_value
        self.param_group_indices = param_group_indices
        self.total_steps = None  # Will be set in on_fit_start

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        # Prefer max_steps if set
        self.total_steps = trainer.num_training_batches
        logger.info(f"[WeightDecayUpdater] Using total_steps={self.total_steps}")

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer, opt_idx=0
    ):
        step = trainer.global_step
        if self.total_steps is None or self.total_steps <= 0:
            logger.warning(
                "[WeightDecayUpdater] total_steps is not set. Skipping weight decay update."
            )
            return
        new_weight_decay = self._compute_weight_decay(step)
        indices = (
            self.param_group_indices
            if self.param_group_indices is not None
            else range(len(optimizer.param_groups))
        )
        for i in indices:
            param_group = optimizer.param_groups[i]
            old_wd = param_group.get("weight_decay", None)
            param_group["weight_decay"] = new_weight_decay
            logger.debug(
                f"[WeightDecayUpdater] Step {step}: param_group {i} weight_decay {old_wd} -> {new_weight_decay}"
            )

    def _compute_weight_decay(self, step: int) -> float:
        progress = min(step, self.total_steps) / self.total_steps
        if self.schedule_type == "constant":
            return self.start_value
        elif self.schedule_type == "linear":
            return self.start_value + (self.end_value - self.start_value) * progress
        elif self.schedule_type == "cosine":
            return self.end_value + 0.5 * (self.start_value - self.end_value) * (
                1 + math.cos(math.pi * progress)
            )
        elif self.schedule_type == "exponential":
            # Exponential decay from start_value to end_value
            gamma = math.log(self.end_value / self.start_value) / self.total_steps
            return self.start_value * math.exp(gamma * step)
        else:
            logger.error(
                f"[WeightDecayUpdater] Unknown schedule_type: {self.schedule_type}"
            )
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

    def state_dict(self):
        return {
            "schedule_type": self.schedule_type,
            "start_value": self.start_value,
            "end_value": self.end_value,
            "param_group_indices": self.param_group_indices,
            "total_steps": self.total_steps,
        }

    def load_state_dict(self, state_dict):
        self.schedule_type = state_dict.get("schedule_type", self.schedule_type)
        self.start_value = state_dict.get("start_value", self.start_value)
        self.end_value = state_dict.get("end_value", self.end_value)
        self.param_group_indices = state_dict.get(
            "param_group_indices", self.param_group_indices
        )
        self.total_steps = state_dict.get("total_steps", self.total_steps)
        logger.info(
            f"[WeightDecayUpdater] State restored from checkpoint: {state_dict}"
        )
