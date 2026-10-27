"""Bounded profiling for wide CSV/TSV data; never change the source data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from exam_io import load_config, load_data, make_output_dir, atomic_json, file_metadata

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    max_bytes = int(float(cfg.get("max_input_mib", 1024)) * 1024**2)
    source_files = {}
    for key in ("train_path", "labels_path", "test_path", "sample_submission_path"):
        if cfg.get(key):
            path = Path(cfg[key])
            if path.stat().st_size > max_bytes:
                raise ValueError(f"{key} exceeds max_input_mib. Inspect available RAM and explicitly raise the limit or use a chunked adapter.")
            source_files[key] = {**file_metadata(path), "bytes": path.stat().st_size}
    data = load_data(cfg)
    X = data.X
    numeric_cols = list(X.select_dtypes(include=[np.number, "bool"]).columns)
    nonnumeric = [c for c in X if c not in numeric_cols]
    missing = X.isna().sum()
    unique = X.nunique(dropna=True)
    numeric = X.loc[:, numeric_cols]
    numeric_values = numeric.to_numpy(dtype=float, copy=False) if numeric_cols else np.empty((len(X), 0))
    infinities = np.isinf(numeric_values).sum(axis=0)
    rows = pd.DataFrame({"feature": X.columns, "dtype": X.dtypes.astype(str).values,
                         "missing_n": missing.values, "missing_fraction": missing.values / len(X),
                         "observed_unique_n": unique.values,
                         "all_missing": unique.values == 0,
                         "constant_observed": unique.values == 1})
    rows["infinite_n"] = rows["feature"].map(dict(zip(numeric_cols, infinities))).fillna(0).astype(int)
    # No p-by-p matrix, per-feature chart explosion, or raw row/identifier dump.
    if numeric_cols:
        num_stats = numeric.agg(["min", "median", "max"]).T
        for stat in ("min", "median", "max"):
            rows[stat] = rows["feature"].map(num_stats[stat])
    class_counts = None
    if cfg["task"] == "classification":
        if pd.Series(data.y).nunique() < 2:
            raise ValueError("Classification requires at least two observed target classes.")
        class_counts = {str(k): int(v) for k, v in pd.Series(data.y).value_counts().items()}
    duplicate_rows = int(X.duplicated().sum())
    excluded = data.diagnostics["excluded_train_columns"]
    target_equal_features = []
    # Equality is a red flag, not an automatic leakage diagnosis; inspect provenance.
    y_series = pd.Series(data.y).reset_index(drop=True)
    for column in X.columns:
        if X[column].reset_index(drop=True).equals(y_series):
            target_equal_features.append(column)
    n, p = X.shape
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_files": source_files,
        "task": cfg["task"], "target": cfg["target"],
        "shape": {"train_rows": n, "predictors": p, "test_rows": None if data.X_test is None else len(data.X_test)},
        "numeric_predictors": len(numeric_cols), "nonnumeric_predictors": len(nonnumeric),
        "nonnumeric_examples": nonnumeric[:20],
        "p_over_n": p / n,
        "dense_float64_predictor_bytes": n * p * 8,
        "dense_float64_p_by_p_matrix_bytes": p * p * 8,
        "missing_values": int(missing.sum()),
        "rows_with_missing_predictors": int(X.isna().any(axis=1).sum()),
        "features_with_missing": int((missing > 0).sum()),
        "all_missing_features": int((unique == 0).sum()),
        "constant_observed_features": int((unique == 1).sum()),
        "infinite_values": int(infinities.sum()),
        "duplicate_predictor_rows": duplicate_rows,
        "target_equal_feature_candidates": target_equal_features[:50],
        "target_equal_feature_candidate_count": len(target_equal_features),
        "class_counts": class_counts,
        "target_summary": None if cfg["task"] == "classification" else {
            "min": float(np.min(data.y)), "median": float(np.median(data.y)),
            "max": float(np.max(data.y)), "std": float(np.std(data.y))
        },
        "alignment": data.diagnostics,
        "excluded_column_count": len(excluded), "excluded_columns": excluded,
        "group_count": None if data.groups is None else int(pd.Series(data.groups).nunique()),
        "time_range": None if data.times is None else [str(np.min(data.times)), str(np.max(data.times))],
        "alerts": [],
        "limitations": [
            "No claim of absence of leakage: metadata provenance, collection time and train/test generation still need review.",
            "Exact predictor duplicates are flagged, not removed. A group-aware split may be needed.",
            "Numeric type and predictor/target equality alone do not establish whether a feature is valid at prediction time.",
            "This profiler reads each configured table into memory. It is not a distributed or streaming loader."
        ]
    }
    alerts = report["alerts"]
    if class_counts and min(class_counts.values()) < int(cfg["split"].get("n_splits", 3)):
        alerts.append("The smallest class has fewer rows than the requested folds; revise the split explicitly.")
    if report["infinite_values"]:
        alerts.append("Infinite predictors require an explicit data correction before modeling; do not silently treat them as ordinary values.")
    if nonnumeric:
        alerts.append("Nonnumeric predictors require fold-local encoding. Use leaderboard.py for mixed numeric/categorical tables; evaluate_models.py is numeric-only.")
    if duplicate_rows:
        alerts.append("Investigate duplicate predictors and repeated entities before choosing IID folds.")
    if target_equal_features:
        alerts.append("Audit predictors equal to the target; do not select a model until their provenance is understood.")
    if p > n:
        alerts.append("High-dimensional branch: regularize; avoid full p-by-p covariance/correlation and large recursive searches.")
    if report["all_missing_features"]:
        alerts.append("All-missing features must be handled consistently within each training fold.")
    if data.X_test is not None:
        train_rate, test_rate = X.isna().mean(), data.X_test.isna().mean()
        shifts = (test_rate - train_rate).abs().sort_values(ascending=False).head(20)
        report["largest_missingness_shifts"] = [
            {"feature": c, "train_missing_fraction": float(train_rate[c]),
             "test_missing_fraction": float(test_rate[c]), "absolute_difference": float(v)}
            for c, v in shifts.items()
        ]
    out = make_output_dir(cfg) / ("profile_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir()
    rows.to_csv(out / "feature_profile.csv", index=False)
    atomic_json(out / "profile.json", report)
    print(json.dumps({"profile": str(out / "profile.json"), "feature_profile": str(out / "feature_profile.csv"),
                      "shape": report["shape"], "missing_values": report["missing_values"],
                      "constant_observed_features": report["constant_observed_features"],
                      "alerts": alerts}, ensure_ascii=False))

if __name__ == "__main__":
    main()
