import lightning as pl


class EpochMilestones(pl.Callback):
    """PyTorch Lightning callback to stop training if a monitored metric does not meet specified thresholds at given epochs.

    This callback allows you to define "milestones"—specific epochs at which a metric must surpass (or fall below) a given value.
    If the metric fails to meet the requirement at the milestone epoch, training is stopped early.

    Args:
        metric_name (str):
            The name of the metric to monitor (as logged in `trainer.callback_metrics`).
        milestones (dict[int, float]):
            A dictionary mapping epoch numbers (int) to required metric values (float).
            At each specified epoch, the metric is checked against the corresponding value.
        direction (str, optional):
            One of "max" or "min".
            - "max": Training stops if the metric is less than or equal to the milestone value.
            - "min": Training stops if the metric is greater than or equal to the milestone value.
            Default is "max".
        after_validation (bool, optional):
            If True (default), the metric is checked after validation (`on_validation_end`).
            If False, the metric is checked after training (`on_training_end`).

    Raises:
        ValueError: If the specified metric is not found in `trainer.callback_metrics` at the milestone epoch.

    Example:
        >>> milestones = {10: 0.2, 20: 0.5}
        >>> callback = EpochMilestones(
        ...     metric_name="eva/accuracy",
        ...     milestones=milestones,
        ...     direction="max",
        ...     after_validation=True,
        ... )
        >>> trainer = pl.Trainer(callbacks=[callback])

    """

    def __init__(
        self,
        metric_name: str,
        milestones: dict[int, float],
        direction: str = "max",
        after_validation: bool = True,
    ):
        super().__init__()

        self.metric_name = metric_name
        self.milestones = milestones
        self.direction = direction
        self.after_validation = after_validation

    def _check_condition(self, trainer):
        # Get the current epoch
        epoch = trainer.current_epoch
        if epoch not in self.milestones:
            return
        # Retrieve the metric from the logged metrics
        metrics = trainer.callback_metrics
        current_value = metrics.get(self.metric_name)
        # If the metric is not available, do nothing
        if current_value is None:
            raise ValueError(f"Desired metric {self.metric_name} is not available")

        # Stop training if the metric is not greater than min_value

        if (self.direction == "max" and current_value <= self.milestones[epoch]) or (
            self.direction == "min" and current_value >= self.milestones[epoch]
        ):
            trainer.should_stop = True
            print(
                f"Early stopping: {self.metric_name}={current_value:.4f} "
                f"at epoch {epoch} (<= {self.milestones[epoch]})"
            )

    def on_training_end(self, trainer, pl_module):
        if self.after_validation:
            return
        self._check_condition(trainer)

    def on_validation_end(self, trainer, pl_module):
        if not self.after_validation:
            return
        self._check_condition(trainer)
