from typing import Optional, Dict, Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

_MODULE_REGISTRY: Dict[str, pl.LightningModule] = {}


def get_module(name: str = "default") -> Optional[pl.LightningModule]:
    """Retrieve a registered module."""
    return _MODULE_REGISTRY.get(name)


def log(name: str, value: Any, module_name: str = "default", **kwargs) -> None:
    """Log a metric using the registered module."""
    module = _MODULE_REGISTRY.get(module_name)
    if module is not None:
        module.log(name, value, **kwargs)


class ModuleRegistryCallback(Callback):
    """Callback that automatically registers the module for global logging access."""

    def __init__(self, name: str = "default"):
        self.name = name

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """Register module at the start of any stage (fit, validate, test, predict)."""
        _MODULE_REGISTRY[self.name] = pl_module

    def teardown(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """Clean up registry when done."""
        _MODULE_REGISTRY.pop(self.name, None)
