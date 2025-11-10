from .checkpoint_sklearn import SklearnCheckpoint, WandbCheckpoint
from .trainer_info import LoggingCallback, ModuleSummary, TrainerInfo, SLURMInfo
from .env_info import EnvironmentDumpCallback


def default():
    """Factory function that returns default callbacks."""
    callbacks = [
        # RichProgressBar(),
        LoggingCallback(),
        EnvironmentDumpCallback(async_dump=True),
        TrainerInfo(),
        SklearnCheckpoint(),
        WandbCheckpoint(),
        ModuleSummary(),
        SLURMInfo(),
    ]

    return callbacks
