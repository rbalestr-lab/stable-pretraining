import os
import glob
import shutil
from typing import List, Tuple, Optional, Sequence
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.trainer import Trainer
from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only

try:
    from hydra.core.hydra_config import HydraConfig
except ImportError:
    HydraConfig = None


def human_size(nbytes: int) -> str:
    """Convert a file size in bytes to a human-readable string.

    Args:
        nbytes (int): File size in bytes.

    Returns:
        str: Human-readable file size (e.g., '1.2 MB').
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def _resolve_hydra_output_dir() -> str:
    """Resolve the Hydra output directory if available, else fallback to current working directory.

    Returns:
        str: Path to the Hydra output directory or current working directory.
    """
    if HydraConfig is not None:
        try:
            return HydraConfig.get().runtime.output_dir
        except Exception:
            pass
    return os.getcwd()


class CleanUpCallback(Callback):
    """PyTorch Lightning callback to monitor and clean up SLURM and Hydra files during and after training.

    At each epoch, prints the names and sizes of SLURM and Hydra files. At the end of successful training,
    deletes those files and prints a summary. Safe for DDP (only rank 0 prints/deletes).

    Args:
        slurm_patterns (Optional[Sequence[str]]): Glob patterns for SLURM files to monitor/delete.
            Defaults to ["slurm-*.out", "slurm-*.err"].
        search_paths (Optional[Sequence[str]]): Directories to search for SLURM files.
            Defaults to [os.getcwd(), $SLURM_SUBMIT_DIR if set].
        delete_hydra (bool): Whether to delete Hydra artifacts (.hydra directory and hydra.log).
            Defaults to True.
        dry_run (bool): If True, only print what would be deleted, do not actually delete.
            Defaults to False.

    Example:
        ```python
        from pytorch_lightning import Trainer

        cleanup_cb = CleanUpCallback()
        trainer = Trainer(callbacks=[cleanup_cb])
        ```

    Example (custom patterns, dry run):
        ```python
        cleanup_cb = CleanUpCallback(
            slurm_patterns=["slurm-*.out", "myjob-*.log"],
            search_paths=["/scratch/logs", "/tmp"],
            delete_hydra=False,
            dry_run=True,
        )
        ```
    """

    def __init__(
        self,
        slurm_patterns: Optional[Sequence[str]] = None,
        search_paths: Optional[Sequence[str]] = None,
        delete_hydra: bool = True,
        dry_run: bool = False,
    ) -> None:
        self.slurm_patterns: Sequence[str] = slurm_patterns or [
            "slurm-*.out",
            "slurm-*.err",
        ]
        self.search_paths: List[str] = (
            list(search_paths) if search_paths else [os.getcwd()]
        )
        slurm_submit_dir = os.environ.get("SLURM_SUBMIT_DIR")
        if slurm_submit_dir and slurm_submit_dir not in self.search_paths:
            self.search_paths.append(slurm_submit_dir)
        self.delete_hydra: bool = delete_hydra
        self.dry_run: bool = dry_run
        self._exception: bool = False
        self._files_to_delete: List[Tuple[str, str, int]] = []

    def _find_files(self) -> List[Tuple[str, str, int]]:
        """Find SLURM and Hydra files to monitor/delete.

        Returns:
            List[Tuple[str, str, int]]: List of (type, path, size) tuples.
        """
        files: List[Tuple[str, str, int]] = []
        # SLURM files
        for path in self.search_paths:
            for pattern in self.slurm_patterns:
                for f in glob.glob(os.path.join(path, pattern)):
                    if os.path.isfile(f):
                        files.append(("SLURM", f, os.path.getsize(f)))
        # Hydra files
        hydra_dir = _resolve_hydra_output_dir()
        hydra_log = os.path.join(hydra_dir, "hydra.log")
        if os.path.isfile(hydra_log):
            files.append(("Hydra", hydra_log, os.path.getsize(hydra_log)))
        hydra_dot_dir = os.path.join(hydra_dir, ".hydra")
        if os.path.isdir(hydra_dot_dir):
            total_size = 0
            for root, _, fs in os.walk(hydra_dot_dir):
                for f in fs:
                    fp = os.path.join(root, f)
                    try:
                        total_size += os.path.getsize(fp)
                    except Exception:
                        pass
            files.append(("Hydra", hydra_dot_dir, total_size))
        return files

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Print SLURM and Hydra file info at the end of each epoch.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer.
            pl_module (LightningModule): The LightningModule being trained.
        """
        files = self._find_files()
        self._files_to_delete = files
        print(f"\n[CleanUpCallback] Epoch {trainer.current_epoch}:")
        if not files:
            print("  No SLURM/Hydra files found.")
        for typ, f, sz in files:
            print(f"  [{typ}] {f} ({human_size(sz)})")

    @rank_zero_only
    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule, exception: BaseException
    ) -> None:
        """Mark that an exception occurred, so files will not be deleted at the end.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer.
            pl_module (LightningModule): The LightningModule being trained.
            exception (BaseException): The exception that was raised.
        """
        self._exception = True

    @rank_zero_only
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Delete SLURM and Hydra files at the end of training if no exception occurred.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer.
            pl_module (LightningModule): The LightningModule being trained.
        """
        if self._exception:
            print("[CleanUpCallback] Training failed, skipping file deletion.")
            return
        print("\n[CleanUpCallback] Cleaning up files after successful training:")
        for typ, f, sz in self._files_to_delete:
            if typ == "Hydra" and not self.delete_hydra:
                print(f"  Skipping Hydra artifact: {f}")
                continue
            if self.dry_run:
                print(f"  Dry run: would delete {f} ({human_size(sz)})")
                continue
            try:
                if os.path.isdir(f):
                    shutil.rmtree(f)
                    print(f"  Deleted directory: {f}")
                else:
                    os.remove(f)
                    print(f"  Deleted file: {f}")
            except Exception as e:
                print(f"  Failed to delete {f}: {e}")
        print("[CleanUpCallback] Cleanup complete.")
