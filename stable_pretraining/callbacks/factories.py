from .checkpoint_sklearn import SklearnCheckpoint, WandbCheckpoint
from .trainer_info import LoggingCallback, ModuleSummary, TrainerInfo, SLURMInfo
from .env_info import EnvironmentDumpCallback


def default():
    """Factory function that returns callbacks.

    Since Lightning doesn't pass pl_module when loading from entry points,
    we can't auto-detect TeacherStudent. Returns static callbacks only.

    For auto-detection to work, the TeacherStudentCallback must be added
    manually in the trainer config or script.
    """
    callbacks = [
        # RichProgressBar(),
        LoggingCallback(),
        EnvironmentDumpCallback(),
        TrainerInfo(),
        SklearnCheckpoint(),
        WandbCheckpoint(),
        ModuleSummary(),
        SLURMInfo(),
    ]

    return callbacks
