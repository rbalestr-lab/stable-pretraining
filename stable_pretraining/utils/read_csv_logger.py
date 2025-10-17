from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Literal
import pandas as pd
import yaml
from loguru import logger


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
    monitor_modes: Dict of metric -> 'min'/'max' (if key missing, defaults to 'min' for '*loss*', else 'max').
    include_globs: Only include files matching these globs (relative to base_dir).
    exclude_globs: Exclude files matching these globs (e.g., '**/checkpoints/**').
    group_by_run_root: If True, group metrics files by run root and produce one summary per run root.
    group_strategy: How to select among multiple files per run root: latest_mtime | latest_epoch | latest_step | merge.
    forward_fill_last: If True, forward-fill the frame (after sorting) before computing last.* summaries.

    """

    METRICS_PATTERNS = ["**/metrics.csv", "**/csv/metrics.csv", "**/*metrics*.csv"]

    def __init__(
        self,
        base_dir: Union[str, Path],
        monitor_keys: Optional[List[str]] = None,
        monitor_modes: Optional[Dict[str, Literal["min", "max"]]] = None,
        include_globs: Optional[List[str]] = None,
        exclude_globs: Optional[List[str]] = None,
        group_by_run_root: bool = False,
        group_strategy: Literal[
            "latest_mtime", "latest_epoch", "latest_step", "merge"
        ] = "latest_mtime",
        forward_fill_last: bool = False,
    ):
        self.base_dir = Path(base_dir)
        self.monitor_keys = monitor_keys
        self.monitor_modes = monitor_modes or {}
        self.include_globs = include_globs or []
        self.exclude_globs = exclude_globs or []
        self.group_by_run_root = group_by_run_root
        self.group_strategy = group_strategy
        self.forward_fill_last = forward_fill_last

    def collect(self) -> pd.DataFrame:
        """Discover, summarize, and aggregate all runs into a DataFrame.

        Returns:
            pd.DataFrame: One row per selected metrics source (per file or per run root),
            with flattened columns such as:
            - last.val_accuracy
            - best.val_loss, best.val_loss.step, best.val_loss.epoch
            - config.optimizer.lr, override.0 (or override as joined string), hparams.seed
            Also includes 'metrics_path' and 'run_root'.
        """
        metrics_files = self._find_metrics_files()
        if self.group_by_run_root:
            run_root_to_files: Dict[Path, List[Path]] = {}
            for f in metrics_files:
                root = self._find_run_root(f)
                run_root_to_files.setdefault(root, []).append(f)
            summaries = []
            for run_root, files in run_root_to_files.items():
                try:
                    if self.group_strategy == "merge":
                        df = self._merge_metrics_files(files)
                        metrics_path_repr = " | ".join(str(p) for p in files)
                    else:
                        chosen = self._select_metrics_file(files)
                        if not chosen:
                            logger.warning(
                                f"No valid metrics file for run root {run_root}"
                            )
                            continue
                        df = self._read_metrics_csv(chosen)
                        metrics_path_repr = str(chosen)
                    if df is None or df.empty:
                        logger.warning(
                            f"No valid data in metrics for run root {run_root}"
                        )
                        continue
                    summary = self._summarize_run(df, metrics_path_repr, run_root)
                    summaries.append(summary)
                except Exception as e:
                    logger.warning(f"Failed to process run root {run_root}: {e}")
            return pd.DataFrame(summaries)
        # Ungrouped: one row per metrics file
        summaries = []
        for f in metrics_files:
            try:
                df = self._read_metrics_csv(f)
                if df is None or df.empty:
                    logger.warning(f"No valid data in metrics file {f}")
                    continue
                run_root = self._find_run_root(f)
                summary = self._summarize_run(df, str(f), run_root)
                summaries.append(summary)
            except Exception as e:
                logger.warning(f"Failed to process metrics file {f}: {e}")
        return pd.DataFrame(summaries)

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

    def _select_metrics_file(self, files: List[Path]) -> Optional[Path]:
        """Select a single metrics file per run root according to group_strategy (non-merge cases)."""
        if not files:
            return None
        if self.group_strategy == "latest_mtime":
            chosen = max(files, key=lambda f: f.stat().st_mtime)
            logger.debug(f"Selected by mtime: {chosen}")
            return chosen
        elif self.group_strategy in ["latest_epoch", "latest_step"]:
            col = "epoch" if self.group_strategy == "latest_epoch" else "step"
            best_file, best_val = None, float("-inf")
            for f in files:
                df = self._read_metrics_csv(f)
                if df is not None and col in df.columns:
                    val = pd.to_numeric(df[col], errors="coerce").max()
                    if pd.notnull(val) and val > best_val:
                        best_file, best_val = f, val
            if best_file:
                logger.debug(f"Selected by {col}: {best_file}")
                return best_file
            # Fallback to mtime
            chosen = max(files, key=lambda f: f.stat().st_mtime)
            logger.debug(f"Fallback to mtime: {chosen}")
            return chosen
        else:
            logger.warning(
                f"Unknown group_strategy '{self.group_strategy}', defaulting to latest_mtime"
            )
            return max(files, key=lambda f: f.stat().st_mtime)

    def _merge_metrics_files(self, files: List[Path]) -> Optional[pd.DataFrame]:
        """Merge multiple metrics CSVs for a run root, deduplicate by step/epoch.

        Keep the last occurrence for any duplicated step/epoch pair.
        """
        dfs = [self._read_metrics_csv(f) for f in files]
        dfs = [df for df in dfs if df is not None and not df.empty]
        if not dfs:
            return None
        df = pd.concat(dfs, ignore_index=True)
        axis_cols = [c for c in ["step", "epoch"] if c in df.columns]
        if axis_cols:
            df = df.sort_values(axis_cols)
            df = df.drop_duplicates(subset=axis_cols, keep="last")
        else:
            df = df.drop_duplicates(keep="last")
        return df

    # ----------------------------
    # CSV reading and summarization
    # ----------------------------
    def _read_metrics_csv(self, path: Path) -> Optional[pd.DataFrame]:
        """Robustly read a metrics CSV, handling delimiter, repeated headers, and sparse rows.

        Returns None if file cannot be read or is empty.
        """
        try:
            df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")
        except Exception as e:
            logger.warning(f"Failed to read CSV {path}: {e}")
            return None
        # Drop unnamed columns, strip column names
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df.columns = df.columns.str.strip()
        # Remove repeated header rows (e.g., when resuming or concatenation)
        # If any of the first few rows equals the column names, drop such rows.
        try:
            if any((df.iloc[i] == df.columns).all() for i in range(min(5, len(df)))):
                df = df[~(df.apply(lambda row: (row == df.columns).all(), axis=1))]
        except Exception:
            # Be cautious with mixed dtypes; if it fails, keep as-is
            pass
        # Try to convert numeric columns where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        # Diagnostics if all values are NaN (excluding axis/time cols)
        metric_cols = [c for c in df.columns if c not in ("step", "epoch", "time")]
        if metric_cols and df[metric_cols].isna().all().all():
            logger.warning(f"All metric values are NaN in {path}")
            logger.debug(f"Dtypes:\n{df.dtypes}\nHead:\n{df.head()}")
        return df

    def _auto_infer_monitor_keys(self, df: pd.DataFrame) -> List[str]:
        """Infer monitor keys heuristically.

        - numeric columns whose names start with 'val_' or contain one of ['acc', 'f1', 'auc', 'loss'].
        """
        candidates = [
            c
            for c in df.columns
            if any(k in c for k in ["val_", "acc", "f1", "auc", "loss"])
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        return candidates

    @staticmethod
    def _last_valid(series: pd.Series) -> Any:
        """Return the last non-NaN value in a Series, or NaN if none."""
        s = series.dropna()
        return s.iloc[-1] if len(s) else float("nan")

    def _summarize_run(
        self, df: pd.DataFrame, metrics_path: str, run_root: Path
    ) -> Dict[str, Any]:
        """Summarize metrics and metadata for a single run (or merged run root).

        Returns:
            Dict[str, Any]: flattened summary
        """
        # Determine axis and sort
        axis_cols = [c for c in ["step", "epoch"] if c in df.columns]
        if axis_cols:
            df = df.sort_values(axis_cols)
        # Optionally forward-fill for last.* summaries (best.* uses original)
        df_ffill = df.ffill() if self.forward_fill_last else df
        # Determine monitor keys/modes
        monitor_keys = self.monitor_keys or self._auto_infer_monitor_keys(df)
        monitor_modes = self.monitor_modes
        summary: Dict[str, Any] = {}
        # Last values: last non-NaN per numeric metric (excluding axis/time)
        for col in df.select_dtypes(include="number").columns:
            if not any(t in col for t in ["step", "epoch", "time"]):
                summary[f"last.{col}"] = (
                    df_ffill[col].iloc[-1]
                    if self.forward_fill_last
                    else self._last_valid(df[col])
                )
        # Best values for monitor keys, ignoring NaNs
        for key in monitor_keys:
            if key in df.columns and pd.api.types.is_numeric_dtype(df[key]):
                mode = monitor_modes.get(key, "min" if "loss" in key else "max")
                s = df[key].dropna()
                if len(s):
                    idx = s.idxmin() if mode == "min" else s.idxmax()
                    best_row = df.loc[idx]
                    summary[f"best.{key}"] = best_row[key]
                    for axis in axis_cols:
                        summary[f"best.{key}.{axis}"] = best_row.get(axis, None)
                else:
                    summary[f"best.{key}"] = float("nan")
                    for axis in axis_cols:
                        summary[f"best.{key}.{axis}"] = None
        summary["metrics_path"] = metrics_path
        summary["run_root"] = str(run_root)
        # Metadata: load Hydra config/overrides and hparams; flatten safely (supports dict/list/scalars)
        hydra_dir = run_root / ".hydra"
        config_obj = self._load_yaml(hydra_dir / "config.yaml")
        overrides_obj = self._load_yaml(hydra_dir / "overrides.yaml")
        hparams_obj = self._find_hparams(run_root)
        summary.update(self._flatten_obj(config_obj, "config."))
        summary.update(self._flatten_obj(overrides_obj, "override."))
        summary.update(self._flatten_obj(hparams_obj, "hparams."))
        return summary

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

    def _flatten_obj(self, obj: Any, prefix: str = "") -> Dict[str, Any]:
        """Flatten dicts/lists/scalars into a flat dict with dot-separated keys.

        - Dict: recurse into keys (prefix.key).
        - List: store both a joined string at the prefix (without trailing dot) and enumerate items:
          e.g., override: "a,b,c", override.0: "a", override.1: "b", ...
        - Scalars: store as prefix (without trailing dot).
        """
        out: Dict[str, Any] = {}
        key_here = prefix[:-1] if prefix.endswith(".") else prefix
        if isinstance(obj, dict):
            for k, v in obj.items():
                out.update(self._flatten_obj(v, f"{prefix}{k}."))
        elif isinstance(obj, list):
            # Joined representation for convenience
            joined = ", ".join(str(x) for x in obj)
            if key_here:
                out[key_here] = joined
            # Enumerate items to keep structure
            for i, v in enumerate(obj):
                out.update(self._flatten_obj(v, f"{prefix}{i}."))
        else:
            if key_here:
                out[key_here] = obj
        return out
