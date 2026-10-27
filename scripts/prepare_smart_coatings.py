#!/usr/bin/env python3
"""Convert official Smart Coatings NPZ inputs into the skill's CSV workspace.

Preserves measurement columns, specimen IDs and sample-submission bytes. No
features are fitted, scaled or selected, and no model is trained. Supply the
original competition files yourself; this helper does not download them.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


FEATURES = [f"cond_{i}" for i in range(6)] + [f"scan_{i:04d}" for i in range(6, 2048)]
TEMPLATE = Path(__file__).resolve().parents[1] / "assets" / "smart_coatings_2026_config.example.json"


def specimen_ids(values: np.ndarray, n_rows: int, context: str) -> list[str]:
    if values.ndim != 1 or len(values) != n_rows:
        raise ValueError(f"{context}: id must be one-dimensional with one entry per X row.")
    if values.dtype.kind not in "USiu":
        raise ValueError(f"{context}: id must contain strings or integers; object and floating-point IDs are unsupported.")
    ids = [value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values]
    if any(not value.strip() for value in ids):
        raise ValueError(f"{context}: specimen IDs must not be empty.")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{context}: duplicate specimen IDs; no rows will be dropped.")
    return ids


def read_archive(path: Path, training: bool) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"X", "id", "y"} if training else {"X", "id"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path.name}: missing NPZ arrays: {', '.join(sorted(missing))}.")
        if not training and "y" in archive.files:
            raise ValueError("The official test archive must be unlabeled; do not supply test labels.")
        X = archive["X"]
        if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] != len(FEATURES):
            raise ValueError(f"{path.name}: X must have shape (n, 2048), with n > 0; got {X.shape}.")
        if X.dtype.kind not in "biuf" or not np.isfinite(X).all():
            raise ValueError(f"{path.name}: X must contain finite real numeric measurements; correct invalid inputs explicitly.")
        ids = specimen_ids(archive["id"], len(X), path.name)
        y = None
        if training:
            y = archive["y"]
            if y.ndim != 1 or len(y) != len(X):
                raise ValueError(f"{path.name}: y must be one-dimensional with one label per X row.")
            if y.dtype.kind not in "biuf" or not np.isfinite(y).all() or set(np.unique(y).tolist()) != {0, 1}:
                raise ValueError(f"{path.name}: y must contain both binary classes 0 and 1, with no missing labels.")
            y = y.astype(np.int64)
    return X, ids, y


def check_sample(path: Path, test_ids: list[str]) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != ["id", "target"]:
            raise ValueError("Sample submission must have exactly the columns id,target in that order.")
        ids = []
        for number, row in enumerate(reader, 2):
            if len(row) != 2 or not row[0].strip():
                raise ValueError(f"Sample submission row {number}: expected a nonempty ID and one probability.")
            probability = float(row[1])
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Sample submission row {number}: target must be a finite probability in [0, 1].")
            ids.append(row[0])
    if len(ids) != len(set(ids)):
        raise ValueError("Sample submission contains duplicate specimen IDs.")
    if len(ids) != len(test_ids) or set(ids) != set(test_ids):
        raise ValueError("Sample submission IDs must match the test archive exactly; its row order may differ.")


def write_measurements(path: Path, X: np.ndarray, ids: list[str], y: np.ndarray | None) -> None:
    frame = pd.DataFrame(X, columns=FEATURES)
    frame.insert(0, "id", ids)
    if y is not None:
        frame["target"] = y
    # Nine significant digits round-trip float32; retain float64 precision too.
    frame.to_csv(path, index=False, float_format="%.9g" if X.dtype.itemsize <= 4 else "%.17g")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Original labeled train.npz.")
    parser.add_argument("--test", type=Path, required=True, help="Original unlabeled test.npz.")
    parser.add_argument("--sample-submission", type=Path, required=True, help="Original sample_submission.csv.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New workspace directory; existing paths are refused.")
    args = parser.parse_args()
    output = args.output_dir.expanduser().absolute()
    created = False
    try:
        if output.exists() or output.is_symlink():
            raise ValueError(f"Output already exists: {output}")
        train_path, test_path, sample_path = [path.expanduser().resolve() for path in
                                             (args.train, args.test, args.sample_submission)]
        X, ids, y = read_archive(train_path, training=True)
        X_test, test_ids, _ = read_archive(test_path, training=False)
        if set(ids).intersection(test_ids):
            raise ValueError("Train/test specimen IDs overlap; the skill's loader requires distinct specimen keys. Audit the split explicitly.")
        check_sample(sample_path, test_ids)
        config = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        config.update({"train_path": "data/train.csv", "test_path": "data/test.csv",
                       "sample_submission_path": "data/sample_submission.csv",
                       "output_dir": "artifacts", "student_id": "", "full_name": "", "deadline": None})
        output.mkdir(parents=True, exist_ok=False)
        created = True
        data_dir = output / "data"
        data_dir.mkdir()
        write_measurements(data_dir / "train.csv", X, ids, y)
        write_measurements(data_dir / "test.csv", X_test, test_ids, None)
        shutil.copyfile(sample_path, data_dir / "sample_submission.csv")
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"config": str(output / "config.json"), "train_rows": len(X),
                          "test_rows": len(X_test), "predictors": len(FEATURES),
                          "identity": "Set student_id and full_name in config.json before exporting a submission.",
                          "source_files_unchanged": True}, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        if created:
            shutil.rmtree(output)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
