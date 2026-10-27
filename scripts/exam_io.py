"""Shared, read-only data loading and atomic result writing for the exam skill."""
from __future__ import annotations
import csv
import json
import math
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PATH_KEYS = ("train_path", "test_path", "labels_path", "sample_submission_path", "output_dir")

@dataclass
class DataBundle:
    X: pd.DataFrame
    y: np.ndarray
    ids: np.ndarray
    X_test: pd.DataFrame | None
    test_ids: np.ndarray | None
    groups: np.ndarray | None
    times: np.ndarray | None
    feature_columns: list[str]
    diagnostics: dict

def load_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a JSON object.")
    cfg["_config_path"] = str(config_path)
    for key in PATH_KEYS:
        value = cfg.get(key)
        if value:
            candidate = Path(value).expanduser()
            cfg[key] = str((config_path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve())
    if not cfg.get("train_path") or not cfg.get("target"):
        raise ValueError("Set train_path and the exact target column from the exam brief.")
    if cfg.get("task") not in ("classification", "regression"):
        raise ValueError("Set task explicitly to classification or regression.")
    cfg.setdefault("seed", 42)
    cfg.setdefault("threads", 2)
    cfg.setdefault("output_dir", str(config_path.parent / "artifacts"))
    cfg.setdefault("drop_columns", [])
    cfg.setdefault("split", {"method": "stratified" if cfg["task"] == "classification" else "kfold", "n_splits": 3})
    return cfg

def _separator(path: Path, delimiter: str | None = None) -> str:
    sep = delimiter or ("\t" if path.suffix.lower() in (".tsv", ".tab") else ",")
    if len(sep) != 1:
        raise ValueError("Only single-character CSV/TSV delimiters are supported by this loader.")
    return sep

def read_table(path: str | Path, id_column: str | None = None, delimiter: str | None = None) -> pd.DataFrame:
    """Read every column. Keep ID lexemes (including leading zeroes and 'NA') intact."""
    path = Path(path)
    sep = _separator(path, delimiter)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle, delimiter=sep), None)
    if not header:
        raise ValueError(f"Empty table: {path.name}")
    if len(set(header)) != len(header):
        raise ValueError(f"Duplicate column names in {path.name}; resolve them explicitly before analysis.")
    # pandas historically names a blank index header 'Unnamed: 0'. Make this deterministic.
    normalized = [name if name else f"Unnamed: {i}" for i, name in enumerate(header)]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Ambiguous blank/Unnamed column names in {path.name}.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.ParserWarning)
        frame = pd.read_csv(path, sep=sep, encoding="utf-8-sig", names=normalized,
                            header=0, low_memory=False, index_col=False)
        if id_column and id_column in normalized:
            # pandas may still turn a converter's literal "NA" result into a
            # missing value. Read only IDs with NA handling disabled, using the
            # same parser/blank-line rules; keep ordinary feature NA inference.
            raw_ids = pd.read_csv(path, sep=sep, encoding="utf-8-sig", names=normalized,
                                  header=0, usecols=[id_column], dtype=str, na_filter=False,
                                  low_memory=False, index_col=False)
            if len(raw_ids) != len(frame):
                raise ValueError(f"ID/data row counts differ in {path.name}; inspect the CSV structure.")
            frame[id_column] = raw_ids[id_column].to_numpy(dtype=object)
    if frame.empty:
        raise ValueError(f"No rows in {path.name}.")
    return frame

def _ids(frame: pd.DataFrame, column: str | None, context: str) -> np.ndarray:
    if column is None:
        return np.asarray([f"row_{i}" for i in range(len(frame))], dtype=str)
    if column not in frame:
        raise ValueError(f"{context}: missing ID column {column!r}.")
    ids = frame[column].astype(str)
    if frame[column].isna().any() or ids.str.strip().eq("").any():
        raise ValueError(f"{context}: missing/empty IDs are not allowed.")
    if ids.duplicated().any():
        raise ValueError(f"{context}: duplicate IDs; define a unique observation key. Do not drop rows automatically.")
    return ids.to_numpy(dtype=str)

def load_data(cfg: dict, include_test: bool = True) -> DataBundle:
    target = cfg["target"]
    id_col = cfg.get("id_column")
    delimiter = cfg.get("delimiter")
    train = read_table(cfg["train_path"], id_col, delimiter)
    ids = _ids(train, id_col, "train")
    diagnostics = {"label_alignment": "target_in_train", "test_alignment": None}
    if cfg.get("labels_path"):
        if target in train:
            raise ValueError("Target exists in train and labels_path was also supplied; choose one authoritative label source.")
        label_id = cfg.get("label_id_column") or id_col
        labels = read_table(cfg["labels_path"], label_id, cfg.get("labels_delimiter", delimiter))
        if target not in labels:
            raise ValueError(f"Label file is missing target {target!r}.")
        if id_col is not None:
            if label_id is None:
                raise ValueError("Set label_id_column to join external labels by ID.")
            label_ids = _ids(labels, label_id, "labels")
            if set(label_ids) != set(ids):
                raise ValueError(f"Train/label ID sets differ: {len(set(ids)-set(label_ids))} missing, "
                                 f"{len(set(label_ids)-set(ids))} extra labels.")
            lookup = pd.Series(labels[target].to_numpy(), index=label_ids)
            y = lookup.reindex(ids).to_numpy()
            diagnostics["label_alignment"] = "one_to_one_ID_join"
            diagnostics["labels_reordered"] = bool(not np.array_equal(ids, label_ids))
        else:
            if not cfg.get("allow_positional_labels", False):
                raise ValueError("External labels without a train ID require explicit allow_positional_labels=true and documented row order.")
            if len(labels) != len(train):
                raise ValueError("Positional labels and train have different row counts.")
            y = labels[target].to_numpy()
            diagnostics["label_alignment"] = "explicit_positional_alignment"
    else:
        if target not in train:
            raise ValueError(f"Train is missing target {target!r}; provide the correct target or labels_path.")
        y = train[target].to_numpy()
    if pd.isna(y).any():
        raise ValueError("Target contains missing values; resolve the task definition instead of imputing labels.")
    if cfg["task"] == "regression":
        y = pd.to_numeric(pd.Series(y), errors="raise").to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Regression target contains non-finite values.")
    split = cfg.get("split", {})
    group_col, time_col = split.get("group_column"), split.get("time_column")
    groups = times = None
    if group_col:
        if group_col not in train or train[group_col].isna().any():
            raise ValueError("The configured group column is missing or contains missing values.")
        groups = train[group_col].astype(str).to_numpy()
    if time_col:
        if time_col not in train or train[time_col].isna().any():
            raise ValueError("The configured time column is missing or contains missing values.")
        if pd.api.types.is_numeric_dtype(train[time_col]):
            times = train[time_col].to_numpy()
            if not np.isfinite(times).all():
                raise ValueError("Non-finite times.")
        else:
            parsed = pd.to_datetime(train[time_col], errors="raise", utc=True)
            times = parsed.astype("int64").to_numpy()
    drops = cfg.get("drop_columns", [])
    if not isinstance(drops, list):
        raise ValueError("drop_columns must be a list.")
    missing_drops = set(drops) - set(train.columns)
    if missing_drops:
        raise ValueError(f"Configured drop_columns absent from train: {sorted(missing_drops)}.")
    prohibited = {v for v in (target, id_col, group_col, time_col) if v is not None} | set(drops)
    chosen = cfg.get("feature_columns")
    if chosen is not None:
        if not isinstance(chosen, list) or len(chosen) != len(set(chosen)):
            raise ValueError("feature_columns must be a list with unique column names.")
        if set(chosen) & prohibited:
            raise ValueError("feature_columns includes the target, ID, grouping/time key, or a dropped column.")
        missing = set(chosen) - set(train.columns)
        if missing:
            raise ValueError(f"Configured feature columns are absent: {sorted(missing)[:10]}.")
        features = chosen
    else:
        features = [name for name in train.columns if name not in prohibited]
    if not features:
        raise ValueError("No predictor columns remain.")
    X = train.loc[:, features]
    test_X = test_ids = None
    if include_test and cfg.get("test_path"):
        test = read_table(cfg["test_path"], id_col, delimiter)
        test_ids = _ids(test, id_col, "test")
        if id_col and set(ids) & set(test_ids):
            raise ValueError("Train/test IDs overlap. Audit the split and ID semantics before proceeding.")
        missing = set(features) - set(test.columns)
        if missing:
            raise ValueError(f"Test is missing required feature columns: {sorted(missing)[:10]}.")
        test_X = test.loc[:, features]
        diagnostics["test_alignment"] = "features_reindexed_to_train_order"
        diagnostics["test_extra_columns_excluded"] = [c for c in test if c not in features and c != id_col]
        diagnostics["test_feature_order_changed"] = [c for c in test if c in features] != features
    diagnostics["excluded_train_columns"] = [name for name in train if name not in features]
    return DataBundle(X, y, ids, test_X, test_ids, groups, times, features, diagnostics)

def make_output_dir(cfg: dict) -> Path:
    output = Path(cfg["output_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    return output

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return None if isinstance(value, float) and not math.isfinite(value) else value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value

def atomic_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_safe(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise

def file_metadata(path: str | Path) -> dict:
    """Record a source's location and size without reading its contents."""
    path = Path(path)
    return {"path": str(path), "name": path.name, "size_bytes": path.stat().st_size}
