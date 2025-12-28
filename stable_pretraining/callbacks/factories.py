from .checkpoint_sklearn import SklearnCheckpoint, WandbCheckpoint
from .trainer_info import LoggingCallback, ModuleSummary, TrainerInfo, SLURMInfo
from .env_info import EnvironmentDumpCallback
from .registry import ModuleRegistryCallback


def default():
    """Factory function that returns default callbacks."""
    callbacks = [
        # RichProgressBar(),
        ModuleRegistryCallback(),
        LoggingCallback(),
        EnvironmentDumpCallback(async_dump=True),
        TrainerInfo(),
        SklearnCheckpoint(),
        WandbCheckpoint(),
        ModuleSummary(),
        SLURMInfo(),
    ]

    return callbacks
