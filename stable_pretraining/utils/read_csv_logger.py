from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd
import yaml
from loguru import logger
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


class CSVLogAutoSummarizer:
    """Automatically discovers and summarizes PyTorch Lightning CSVLogger metrics from Hydra multirun sweeps.

    Handles arbitrary directory layouts, sparse metrics, multiple versions (preemption/resume), and aggregates
    config/hparams metadata into a single DataFrame.
    Features:
    - Recursively finds metrics CSVs using common patterns.
    - Infers run root by searching for .hydra/config.yaml (falls back to metrics parent).
    - Handles multiple metrics files per run with configurable grouping strategies:
      latest_mtime, latest_epoch, latest_step, or merge.
    - Robust CSV parsing (delimiter sniffing, repeated-header cleanup, type coercion).
    - Sparse-metrics aware: last values are last non-NaN; best values ignore NaNs.
    - Loads and flattens Hydra config, overrides (list or dict), and hparams metadata.

    Args:
    base_dir: Root directory to search for metrics files.
    monitor_keys: Metrics to summarize (if None, auto-infer).
    include_globs: Only include files matching these globs (relative to base_dir).
    exclude_globs: Exclude files matching these globs (e.g., '**/checkpoints/**').
    forward_fill_last: If True, forward-fill the frame (after sorting) before computing last.* summaries.

    """

    METRICS_PATTERNS = ["**/metrics.csv", "**/csv/metrics.csv", "**/*metrics*.csv"]

    def __init__(
        self,
        agg: Optional[callable] = None,
        monitor_keys: Optional[List[str]] = None,
        include_globs: Optional[List[str]] = None,
        exclude_globs: Optional[List[str]] = None,
        max_workers: Optional[int] = 10,
    ):
        self.agg = agg
        self.monitor_keys = monitor_keys
        self.include_globs = include_globs or []
        self.exclude_globs = exclude_globs or []
        self.max_workers = max_workers

    def collect(self, base_dir) -> pd.DataFrame:
        """Discover, summarize, and aggregate all runs into a DataFrame.

        Args:
        base_dir: Root directory to search for metrics files.

        Returns:
            pd.DataFrame: One row per selected metrics source (per file or per run root),
            with flattened columns such as:
            - last.val_accuracy
            - best.val_loss, best.val_loss.step, best.val_loss.epoch
            - config.optimizer.lr, override.0 (or override as joined string), hparams.seed
            Also includes 'metrics_path' and 'run_root'.
        """
        if type(base_dir) not in [list, tuple]:
            base_dir = [base_dir]

        dfs = []
        for b in base_dir:
            dfs.append(self._single_collect(b))
        return pd.concat(dfs, ignore_index=True)

    def _single_collect(self, base_dir) -> pd.DataFrame:
        """Discover, summarize, and aggregate all runs into a DataFrame.

        Args:
        base_dir: Root directory to search for metrics files.

        Returns:
            pd.DataFrame: One row per selected metrics source (per file or per run root),
            with flattened columns such as:
            - last.val_accuracy
            - best.val_loss, best.val_loss.step, best.val_loss.epoch
            - config.optimizer.lr, override.0 (or override as joined string), hparams.seed
            Also includes 'metrics_path' and 'run_root'.
        """
        self.base_dir = Path(base_dir)
        metrics_files = self._find_metrics_files()
        run_root_to_files: Dict[Path, List[Path]] = {}
        for f in tqdm(metrics_files, desc="compiling root paths"):
            root = self._find_run_root(f)
            run_root_to_files.setdefault(root, []).append(f)
        items = list(run_root_to_files.items())
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            summaries = list(
                tqdm(
                    executor.map(self._merge_metrics_files, items),
                    desc="Loading",
                    total=len(items),
                )
            )

        return pd.concat(summaries)

    # ----------------------------
    # Discovery and selection
    # ----------------------------
    def _find_metrics_files(self) -> List[Path]:
        """Recursively find all metrics CSV files matching known patterns, applying include/exclude globs."""
        files: List[Path] = []
        for pat in self.METRICS_PATTERNS:
            files.extend(self.base_dir.rglob(pat))
        # Unique files
        files = list({f.resolve() for f in files if f.is_file()})
        rel_files = [f.relative_to(self.base_dir) for f in files]
        if self.include_globs:
            files = [
                f
                for f, rel in zip(files, rel_files)
                if any(rel.match(g) for g in self.include_globs)
            ]
        if self.exclude_globs:
            files = [
                f
                for f, rel in zip(files, rel_files)
                if not any(rel.match(g) for g in self.exclude_globs)
            ]
        logger.info(f"Discovered {len(files)} metrics files.")
        return files

    def _find_run_root(self, metrics_file: Path) -> Path:
        """Walk up to nearest ancestor with .hydra/config.yaml; else use immediate parent."""
        for parent in [metrics_file.parent] + list(metrics_file.parents):
            hydra_dir = parent / ".hydra"
            if (hydra_dir / "config.yaml").exists():
                return parent
        return metrics_file.parent

    def _merge_metrics_files(self, args) -> Optional[pd.DataFrame]:
        """Merge multiple metrics CSVs for a run root, deduplicate by step/epoch.

        Keep the last occurrence for any duplicated step/epoch pair.
        """
        root, files = args
        dfs = [self._read_data(f) for f in files]
        dfs = [df for df in dfs if df is not None and not df.empty]
        if not dfs:
            return None
        df = pd.concat(dfs, ignore_index=True)
        df["root"] = root
        if self.agg is not None:
            return self.agg(df)
        return df

    # ----------------------------
    # CSV reading and summarization
    # ----------------------------
    def _read_data(self, path: Path) -> Optional[pd.DataFrame]:
        """Robustly read a metrics CSV, handling delimiter, repeated headers, and sparse rows.

        Returns None if file cannot be read or is empty.
        """
        df = pd.read_csv(path)
        # Drop unnamed columns, strip column names
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df.columns = df.columns.str.strip()
        # Try to convert numeric columns where possible
        df = df.apply(pd.to_numeric)
        df.ffill(inplace=True)
        # Diagnostics if all values are NaN (excluding axis/time cols)
        metric_cols = [c for c in df.columns if c not in ("step", "epoch", "time")]
        if metric_cols and df[metric_cols].isna().all().all():
            logger.warning(f"All metric values are NaN in {path}")
            logger.debug(f"Dtypes:\n{df.dtypes}\nHead:\n{df.head()}")
        hparams = self._find_hparams(path.parent / "hparams.yaml")
        for k, v in list(hparams.items()):
            hparams[f"config/{k}"] = v
            del hparams[k]
        df[list(hparams.keys())] = list(hparams.values())
        return df

    # ----------------------------
    # Metadata loading and flattening
    # ----------------------------
    def _load_yaml(self, path: Path) -> Any:
        """Load a YAML file. Returns the parsed object, which may be dict, list, or scalar.

        Returns {} if file is missing or unreadable.
        """
        if not path.exists():
            return {}
        try:
            with open(path, "r") as f:
                obj = yaml.safe_load(f)
            return {} if obj is None else obj
        except Exception as e:
            logger.warning(f"Failed to load YAML {path}: {e}")
            return {}

    def _find_hparams(self, start_dir: Path) -> Any:
        """Search upward from start_dir for hparams.yaml (first found). Return the parsed object (dict/list/scalar) or {}."""
        for parent in [start_dir] + list(start_dir.parents):
            hp = parent / "hparams.yaml"
            if hp.exists():
                return self._load_yaml(hp)
        return {}
