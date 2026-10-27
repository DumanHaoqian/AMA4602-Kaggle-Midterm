#!/usr/bin/env python3
"""Bounded, fold-safe model comparison for an explicitly configured exam task.

Original implementation. All learned transformations live inside each estimator.
The budget is checked before fits; it is not an estimator interruption mechanism.
"""
from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import uuid
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,
                             log_loss, matthews_corrcoef, mean_absolute_error,
                             mean_absolute_percentage_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold, StratifiedGroupKFold,
                                     StratifiedKFold, TimeSeriesSplit, train_test_split)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC
from threadpoolctl import threadpool_limits

from exam_io import load_config, load_data, atomic_json, make_output_dir, file_metadata


MULTICLASS_AUC = {"roc_auc_ovr": ("ovr", "macro"), "roc_auc_ovr_weighted": ("ovr", "weighted"),
                  "roc_auc_ovo": ("ovo", "macro")}
CLASS_METRICS = {"accuracy", "macro_f1", "weighted_f1", "f1", "balanced_accuracy", "mcc", "qwk",
                 "log_loss", "roc_auc"} | set(MULTICLASS_AUC)
REG_METRICS = {"rmse", "mae", "r2", "rmsle", "mape"}
PROBABILITY_METRICS = {"log_loss", "roc_auc"} | set(MULTICLASS_AUC)
POSITIVE_CLASS_METRICS = {"roc_auc", "f1"}  # binary metrics that need an explicit positive_class
LOWER_IS_BETTER = {"log_loss", "rmse", "mae", "rmsle", "mape"}
TARGET_TRANSFORMS = {None, "log1p"}


def positive_class_index(classes, positive):
    """Index of the configured positive class; 1 and 1.0 match, but no other guessing."""
    matches = [i for i, label in enumerate(classes) if str(label) == str(positive)]
    if not matches:
        try:
            matches = [i for i, label in enumerate(classes) if float(label) == float(positive)]
        except (TypeError, ValueError):
            matches = []
    if len(matches) != 1:
        raise ValueError(f"positive_class {positive!r} must match exactly one of {classes!r}.")
    return matches[0]


def classification_setup(metric, classes, cfg):
    """Validate metric/class compatibility; return the positive-class index or None."""
    if metric in ("roc_auc", "f1") and len(classes) != 2:
        raise ValueError(f"{metric} is binary; for {len(classes)} classes use "
                         f"{'roc_auc_ovr / roc_auc_ovo' if metric == 'roc_auc' else 'macro_f1 / weighted_f1'} as the rules specify.")
    if metric == "qwk" and not all(isinstance(c, (int, float, np.number)) and not isinstance(c, bool) for c in classes):
        raise ValueError("qwk needs numeric ordinal labels so that the sorted label order is the rating order.")
    if metric in POSITIVE_CLASS_METRICS:
        if cfg.get("positive_class") is None:
            raise ValueError(f"{metric} requires explicit positive_class from the exam brief.")
        return positive_class_index(classes, cfg["positive_class"])
    return None


def target_transform(cfg):
    """Regression target transform; RMSLE defaults to log1p (fit in log space, predict >= 0)."""
    value = cfg.get("target_transform", "log1p" if cfg.get("metric") == "rmsle" else None)
    if value not in TARGET_TRANSFORMS:
        raise ValueError(f"target_transform must be one of {sorted(map(str, TARGET_TRANSFORMS))}.")
    if value and cfg["task"] != "regression":
        raise ValueError("target_transform applies to regression only.")
    return value


def _expm1_nonnegative(z):
    # The target is >= 0 under log1p, so negative back-transformed values are projected to 0.
    return np.maximum(np.expm1(z), 0.0)


def wrap_target(model, transform):
    if transform == "log1p":
        return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=_expm1_nonnegative,
                                          check_inverse=False)
    return model


def check_regression_target(y, metric, transform):
    if (metric == "rmsle" or transform == "log1p") and np.min(y) < 0:
        raise ValueError("RMSLE / log1p target transform needs a nonnegative target.")


class CappedSelectKBest(TransformerMixin, BaseEstimator):
    """Resolve k after imputation/constant filtering, separately in every fold."""

    def __init__(self, max_features=1000):
        self.max_features = max_features

    def fit(self, X, y):
        self.selector_ = SelectKBest(f_classif, k=min(self.max_features, X.shape[1]))
        self.selector_.fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)


class CappedPCA(TransformerMixin, BaseEstimator):
    """Choose a feasible randomized PCA size from this training fold only."""

    def __init__(self, max_components=50, random_state=42):
        self.max_components = max_components
        self.random_state = random_state

    def fit(self, X, y=None):
        count = min(self.max_components, X.shape[1], X.shape[0] - 1)
        if count < 1:
            raise ValueError("PCA needs at least two training rows and one nonconstant feature.")
        self.pca_ = PCA(n_components=count, svd_solver="randomized",
                        random_state=self.random_state).fit(X)
        return self

    def transform(self, X):
        return self.pca_.transform(X)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if callable(value) and hasattr(value, "__qualname__"):
        return f"{value.__module__}.{value.__qualname__}"
    return repr(value)


def atomic_csv(path, frame):
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


CLASS_MODELS = ["select_logistic", "linear_svm", "pca_logistic", "random_forest", "select_rbf_svm", "select_mlp"]
REG_MODELS = ["ridge", "pca_ridge", "random_forest", "rbf_svr", "mlp"]
DEFAULT_MODELS = {"classification": CLASS_MODELS[:3], "regression": REG_MODELS[:2]}  # nonlinear ones are opt-in


def build_model(name, task, seed, transform=None, threads=1):
    if name == "dummy":
        return DummyClassifier(strategy="prior") if task == "classification" else DummyRegressor()
    if task == "regression" and transform:
        return wrap_target(build_model(name, task, seed, threads=threads), transform)
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
             ("constant_filter", VarianceThreshold())]
    if name == "select_logistic":
        steps += [("select", CappedSelectKBest()), ("scale", StandardScaler()),
                  ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=seed))]
    elif name == "linear_svm":
        steps += [("select", CappedSelectKBest()), ("scale", StandardScaler()),
                  ("model", LinearSVC(C=0.1, dual="auto", max_iter=5000, random_state=seed))]
    elif name == "pca_logistic":
        steps += [("scale", StandardScaler()), ("pca", CappedPCA(random_state=seed)),
                  ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=seed))]
    elif name == "ridge":
        steps += [("scale", StandardScaler()), ("model", Ridge(alpha=1.0, solver="lsqr"))]
    elif name == "pca_ridge":
        steps += [("scale", StandardScaler()), ("pca", CappedPCA(random_state=seed)),
                  ("model", Ridge(alpha=1.0, solver="lsqr"))]
    # Nonlinear challengers with fixed, recorded settings (no tuning): a check that a linear rule is not leaving
    # signal on the table, not a search.
    elif name == "random_forest":
        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        steps += [("model", cls(n_estimators=500, max_features="sqrt" if task == "classification" else 0.3,
                                min_samples_leaf=1, n_jobs=threads, random_state=seed))]
    elif name == "select_rbf_svm":
        steps += [("select", CappedSelectKBest()), ("scale", StandardScaler()),
                  ("model", SVC(C=1.0, gamma="scale", probability=True, random_state=seed))]
    elif name == "select_mlp":
        steps += [("select", CappedSelectKBest()), ("scale", StandardScaler()),
                  ("model", MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, early_stopping=True,
                                          max_iter=500, random_state=seed))]
    elif name == "rbf_svr":
        steps += [("scale", StandardScaler()), ("model", SVR(C=1.0, gamma="scale"))]
    elif name == "mlp":
        steps += [("scale", StandardScaler()),
                  ("model", MLPRegressor(hidden_layer_sizes=(64,), alpha=1e-3, early_stopping=True,
                                         max_iter=500, random_state=seed))]
    else:
        raise ValueError(f"Unknown model {name!r}.")
    return Pipeline(steps)


def feature_weights(model, columns):
    """Weights of a fitted pipeline on the original (standardised) features, or None.

    Linear models: coefficients on standardised features (rows = classes, or one row).
    PCA pipelines: coefficients back-projected through the PCA loadings to standardised features.
    Random forest: impurity importance. Returns (feature names, weights array rows x features, kind)."""
    if isinstance(model, TransformedTargetRegressor):
        model = model.regressor_
    if not isinstance(model, Pipeline):
        return None
    steps = model.named_steps
    est = steps["model"]
    names = np.asarray(columns)[steps["constant_filter"].get_support()]
    if hasattr(est, "coef_"):
        weights, kind = np.atleast_2d(np.asarray(est.coef_, dtype=float)), "standardized_coefficient"
        if "pca" in steps:
            weights, kind = weights @ steps["pca"].pca_.components_, "pca_back_projected_coefficient"
    elif hasattr(est, "feature_importances_"):
        weights, kind = np.asarray(est.feature_importances_, dtype=float)[None, :], "impurity_importance"
    else:
        return None
    if "select" in steps:
        names = names[steps["select"].selector_.get_support()]
    return names, weights, kind


def numeric_frame(frame, context):
    bad = [str(c) for c in frame if not pd.api.types.is_numeric_dtype(frame[c])]
    if bad:
        raise ValueError(f"{context}: nonnumeric columns {bad[:15]}. This runner is numeric-only; "
                         "build an explicit fold-safe ColumnTransformer for mixed types. No columns were dropped.")
    result = frame.astype(np.float64)
    if np.isinf(result.to_numpy(copy=False)).any():
        raise ValueError(f"{context}: infinite feature values must be resolved explicitly; NaNs can be imputed.")
    return result


def make_splits(cfg, bundle, y, reusable_path=None):
    method = cfg["split"]["method"]
    count = int(cfg["split"].get("n_splits", 3))
    if count < 2:
        raise ValueError("split.n_splits must be at least 2.")
    n = len(y)
    notes = []
    if reusable_path:
        saved = json.loads(Path(reusable_path).read_text(encoding="utf-8"))
        if saved.get("row_ids") != bundle.ids.tolist() or saved.get("n_rows") != n:
            raise ValueError("Reusable fold indices do not match the full ordered training ID sequence.")
        if saved.get("split_method") != method:
            raise ValueError("Reusable fold split_method differs from the configured method.")
        folds = [(np.asarray(f["train_indices"], dtype=int),
                  np.asarray(f["validation_indices"], dtype=int)) for f in saved["folds"]]
        if len(folds) != count:
            raise ValueError("Reusable fold count differs from split.n_splits.")
        notes.append("Loaded explicit fold indices; indices and group/time boundaries were rechecked.")
    elif method == "stratified":
        if cfg["task"] != "classification":
            raise ValueError("Stratified splitting requires classification; choose kfold/group/time for regression.")
        if np.bincount(y).min() < count:
            raise ValueError("Smallest class has fewer rows than n_splits; reduce folds or revise the validation plan.")
        folds = list(StratifiedKFold(count, shuffle=True, random_state=cfg["seed"]).split(bundle.X, y))
    elif method == "kfold":
        folds = list(KFold(count, shuffle=True, random_state=cfg["seed"]).split(bundle.X))
    elif method in {"group", "stratified_group"}:
        if bundle.groups is None:
            raise ValueError("Grouped splitting requires split.group_column.")
        if len(np.unique(bundle.groups)) < count:
            raise ValueError("Fewer independent groups than n_splits.")
        if method == "stratified_group":
            if cfg["task"] != "classification":
                raise ValueError("stratified_group requires classification.")
            splitter = StratifiedGroupKFold(count, shuffle=True, random_state=cfg["seed"])
        else:
            splitter = GroupKFold(count)
        folds = list(splitter.split(bundle.X, y, groups=bundle.groups))
    elif method == "time":
        if bundle.times is None:
            raise ValueError("Time splitting requires split.time_column.")
        order = np.argsort(bundle.times, kind="stable")
        splitter = TimeSeriesSplit(count, gap=int(cfg["split"].get("gap", 0)))
        folds = []
        for train, valid in splitter.split(order):
            tr, va = order[train], order[valid]
            strict = tr[bundle.times[tr] < bundle.times[va].min()]
            if len(strict) < len(tr):
                notes.append(f"Purged {len(tr)-len(strict)} training rows sharing the validation boundary time.")
            folds.append((strict, va))
    else:
        raise ValueError(f"Unknown split method {method!r}.")
    coverage = np.zeros(n, dtype=int)
    for number, (tr, va) in enumerate(folds):
        if not len(tr) or not len(va):
            raise ValueError(f"Fold {number} has an empty training or validation partition.")
        if np.any(tr < 0) or np.any(va < 0) or np.any(tr >= n) or np.any(va >= n):
            raise ValueError(f"Fold {number} contains out-of-range indices.")
        if len(np.unique(tr)) != len(tr) or len(np.unique(va)) != len(va) or np.intersect1d(tr, va).size:
            raise ValueError(f"Fold {number} has duplicate or overlapping train/validation indices.")
        coverage[va] += 1
        if method in {"group", "stratified_group"}:
            if np.intersect1d(bundle.groups[tr], bundle.groups[va]).size:
                raise ValueError(f"Fold {number} leaks an entity/group across partitions.")
        if method == "time" and bundle.times[tr].max() >= bundle.times[va].min():
            raise ValueError(f"Fold {number} is not strictly chronological.")
        if cfg["task"] == "classification" and len(np.unique(y[tr])) < 2:
            raise ValueError(f"Fold {number} training has only one class; revise the split before comparing models.")
        if cfg["metric"] == "roc_auc" and len(np.unique(y[va])) < 2:
            raise ValueError(f"Fold {number} validation has one class, so binary ROC AUC is undefined.")
        if cfg["metric"] in MULTICLASS_AUC and len(np.unique(y[va])) < len(np.unique(y)):
            raise ValueError(f"Fold {number} validation misses a class, so multiclass ROC AUC is undefined; use fewer folds.")
        if cfg["metric"] == "r2" and len(va) < 2:
            raise ValueError("R2 requires at least two validation rows per fold.")
    if np.any(coverage > 1):
        raise ValueError("OOF rows must be validated at most once; repeated CV is not supported by this runner.")
    if method != "time" and not np.all(coverage == 1):
        raise ValueError("Non-temporal CV must cover every row exactly once.")
    if method == "time":
        notes.append("Time CV leaves initial/history rows without OOF predictions; pooled metrics use covered rows only.")
    return folds, notes


def aligned_predict(model, X, task, n_classes):
    prediction = np.asarray(model.predict(X))
    probabilities = None
    if task == "classification":
        if not np.isin(prediction, np.arange(n_classes)).all():
            raise ValueError("Estimator returned an unknown encoded class.")
        prediction = prediction.astype(int)
        if hasattr(model, "predict_proba"):
            local = np.asarray(model.predict_proba(X))
            probabilities = np.zeros((len(X), n_classes), dtype=float)
            probabilities[:, np.asarray(model.classes_, dtype=int)] = local
            if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                    or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)):
                raise ValueError("Invalid classifier probabilities.")
    elif not np.isfinite(prediction.astype(float)).all():
        raise ValueError("Regressor returned non-finite predictions.")
    return prediction, probabilities


def score(metric, y, prediction, probabilities, n_classes, positive_index):
    if metric == "accuracy":
        result = accuracy_score(y, prediction)
    elif metric in ("macro_f1", "weighted_f1"):
        result = f1_score(y, prediction, labels=np.arange(n_classes), average=metric.split("_")[0], zero_division=0)
    elif metric == "f1":
        result = f1_score(np.asarray(y) == positive_index, np.asarray(prediction) == positive_index, zero_division=0)
    elif metric == "balanced_accuracy":
        result = balanced_accuracy_score(y, prediction)
    elif metric == "mcc":
        result = matthews_corrcoef(y, prediction)
    elif metric == "qwk":
        # Encoded labels follow sorted numeric label order, which is the rating order.
        result = cohen_kappa_score(y, prediction, labels=np.arange(n_classes), weights="quadratic")
    elif metric == "log_loss":
        result = log_loss(y, probabilities, labels=np.arange(n_classes))
    elif metric == "roc_auc":
        result = roc_auc_score(np.asarray(y) == positive_index, probabilities[:, positive_index])
    elif metric in MULTICLASS_AUC:
        if n_classes == 2:
            result = roc_auc_score(y, probabilities[:, 1])
        else:
            multi_class, average = MULTICLASS_AUC[metric]
            result = roc_auc_score(y, probabilities, multi_class=multi_class, average=average,
                                   labels=np.arange(n_classes))
    elif metric == "rmse":
        result = np.sqrt(mean_squared_error(y, prediction))
    elif metric == "mae":
        result = mean_absolute_error(y, prediction)
    elif metric == "rmsle":
        prediction = np.asarray(prediction, dtype=float)
        if np.any(prediction < 0):
            raise ValueError("RMSLE got negative predictions; use target_transform=log1p (the RMSLE default).")
        result = np.sqrt(np.mean((np.log1p(prediction) - np.log1p(np.asarray(y, dtype=float))) ** 2))
    elif metric == "mape":
        result = mean_absolute_percentage_error(y, prediction)
    else:
        result = r2_score(y, prediction, force_finite=False)
    if not np.isfinite(result):
        raise ValueError(f"Metric {metric} is not finite on this validation partition.")
    return float(result)


def fit_details(model):
    if isinstance(model, TransformedTargetRegressor):
        model = model.regressor_
    if not isinstance(model, Pipeline):
        return {}
    steps = model.named_steps
    detail = {"features_after_constant_filter": int(steps["constant_filter"].get_support().sum())}
    if "select" in steps:
        detail["selected_features"] = int(steps["select"].selector_.get_support().sum())
    if "pca" in steps:
        detail["pca_components"] = int(steps["pca"].pca_.n_components_)
        detail["pca_explained_variance"] = float(steps["pca"].pca_.explained_variance_ratio_.sum())
    return detail


def warning_records(captured):
    # Deduplicate within a fit, while retaining category and actual message.
    return [dict(category=c, message=m) for c, m in dict.fromkeys(
        (w.category.__name__, str(w.message)) for w in captured)]


def initial_fit_estimate(name, n, p):
    # Planning heuristic only: actual estimator execution is never force-interrupted.
    if name == "dummy":
        return 0.02
    base = max(0.15, n * p / 2_000_000)
    factor = {"random_forest": 6.0, "select_rbf_svm": 3.0, "select_mlp": 4.0, "rbf_svr": 3.0, "mlp": 4.0}
    return base * (4.0 if name.startswith("pca_") else factor.get(name, 1.5))


def write_importance(run_dir, name, columns, weight_sum, weight_count, n_folds, kind, classes, task):
    """importance_<model>.csv: fold-averaged weights (0 in folds where a feature was not selected) and selection
    frequency, sorted by the largest absolute weight. Returns the evidence fields describing it."""
    mean = weight_sum / n_folds
    if mean.shape[0] == len(classes) and len(classes) > 2:
        labels = [str(c) for c in classes]
    elif task == "classification" and kind != "impurity_importance":
        labels = [str(classes[1])]  # binary linear models: one row, positive direction = second sorted class
    else:
        labels = ["all"]
    frame = pd.DataFrame(mean.T, columns=[f"weight_{c}" for c in labels])
    frame.insert(0, "feature", np.asarray(columns))
    frame["max_abs_weight"] = np.abs(mean).max(axis=0)
    frame["selection_frequency"] = weight_count / n_folds
    frame = frame[weight_count > 0].sort_values("max_abs_weight", ascending=False)
    atomic_csv(run_dir / f"importance_{name}.csv", frame)
    return {"importance_file": f"importance_{name}.csv", "importance_kind": kind, "importance_classes": labels,
            "importance_note": ("Weights averaged over the fold models (0 where a fold did not select the feature). "
                                "Standardised-feature coefficients share weight among correlated features and are "
                                "associations, not causal effects." if kind != "impurity_importance" else
                                "Impurity importance averaged over the fold forests; biased towards high-variance "
                                "features and shared among correlated ones.")}


def holdout_split(cfg, bundle, y):
    """Untouched holdout carved out before any fold is drawn; None when not configured.

    Respects the split design: stratified for classification, whole groups for group splits, the latest
    rows for time splits. Returns (development indices, holdout indices, description)."""
    spec = cfg.get("holdout")
    if not spec:
        return None
    fraction = float(spec.get("fraction", 0.2)) if isinstance(spec, dict) else float(spec)
    if not 0 < fraction < 0.5:
        raise ValueError("holdout.fraction must lie in (0, 0.5).")
    n, seed, method = len(y), int(cfg["seed"]), cfg["split"]["method"]
    rows = np.arange(n)
    if method in ("group", "stratified_group"):
        if bundle.groups is None:
            raise ValueError("A grouped holdout requires split.group_column.")
        dev, hold = next(GroupShuffleSplit(1, test_size=fraction, random_state=seed).split(rows, groups=bundle.groups))
        how = "whole groups (GroupShuffleSplit)"
    elif method == "time":
        order = np.argsort(bundle.times, kind="stable")
        cut = n - max(1, int(math.ceil(fraction * n)))
        dev, hold = order[:cut], order[cut:]
        dev = dev[bundle.times[dev] < bundle.times[hold].min()]
        how = "latest rows by time (earlier rows only in development)"
    else:
        stratify = y if cfg["task"] == "classification" else None
        dev, hold = train_test_split(rows, test_size=fraction, random_state=seed, stratify=stratify)
        how = "stratified random" if stratify is not None else "random"
    return np.sort(dev), np.sort(hold), {"fraction": fraction, "method": how, "n_development": int(len(dev)),
                                         "n_holdout": int(len(hold)), "seed": seed}


def run_evaluation(cfg, args, evidence, run_dir, started, budget):
    task, metric = cfg["task"], cfg.get("metric")
    permitted = CLASS_METRICS if task == "classification" else REG_METRICS
    if metric not in permitted:
        raise ValueError(f"Set the exam metric explicitly; {task} supports {sorted(permitted)}.")
    seed, threads = int(cfg.get("seed", 42)), int(cfg.get("threads", 2))
    if threads < 1:
        raise ValueError("threads must be a positive integer.")
    cfg["seed"] = seed
    evidence["data_sources"] = {}
    for role, key in (("train", "train_path"), ("labels", "labels_path"), ("test", "test_path")):
        source = cfg.get(key)
        evidence["data_sources"][role] = file_metadata(source) if source else None
    bundle = load_data(cfg, include_test=True)
    prediction_config = cfg.get("prediction", {})
    if not isinstance(prediction_config, dict):
        raise ValueError("prediction must be an object with an explicit kind.")
    prediction_kind = prediction_config.get("kind")
    prediction_source = "configured"
    if prediction_kind is None:
        if bundle.X_test is not None:
            raise ValueError("Test data requires explicit prediction.kind: label, probability, or regression. Read the submission contract before exporting predictions.")
        prediction_kind = ("regression" if task == "regression" else
                           "probability" if metric in PROBABILITY_METRICS else "label")
        prediction_source = "inferred_from_metric_for_no_test_practice"
    allowed_kinds = {"label", "probability"} if task == "classification" else {"regression"}
    if prediction_kind not in allowed_kinds:
        raise ValueError(f"prediction.kind={prediction_kind!r} is incompatible with task={task!r}; choose from {sorted(allowed_kinds)}.")
    if bundle.X_test is not None and metric in PROBABILITY_METRICS and prediction_kind != "probability":
        raise ValueError(f"Metric {metric} requires probability predictions, but prediction.kind={prediction_kind!r}. Resolve the exam submission contract; no label CSV was produced.")
    needs_probability = metric in PROBABILITY_METRICS or prediction_kind == "probability"
    evidence["prediction_contract"] = {"kind": prediction_kind, "source": prediction_source,
                                        "probabilities_required_for_selection": needs_probability}
    X = numeric_frame(bundle.X, "train")
    X_test = numeric_frame(bundle.X_test, "test") if bundle.X_test is not None else None
    classes, encoder, positive_index = [], None, None
    if task == "classification":
        encoder = LabelEncoder().fit(bundle.y)
        classes = encoder.classes_.tolist()
        y = encoder.transform(bundle.y)
        if len(classes) < 2:
            raise ValueError("Classification needs at least two target classes.")
        positive_index = classification_setup(metric, classes, cfg)
    else:
        y = np.asarray(bundle.y, dtype=float)
    transform = target_transform(cfg)
    if task == "regression":
        check_regression_target(y, metric, transform)
    names = [s.strip() for s in args.models.split(",")] if args.models else list(DEFAULT_MODELS[task])
    available = CLASS_MODELS if task == "classification" else REG_MODELS
    if names == ["all"]:
        names = list(available)
    names = list(dict.fromkeys(["dummy"] + [s.strip() for s in names if s.strip()]))
    valid_models = {"dummy", *available}
    if set(names) - valid_models:
        raise ValueError(f"Unsupported models for {task}: {sorted(set(names)-valid_models)}; choose from {available} or 'all'.")
    holdout = holdout_split(cfg, bundle, y)
    if holdout is None:
        folds, split_notes = make_splits(cfg, bundle, y, args.folds)
        dev = np.arange(len(y))
    else:
        if args.folds:
            raise ValueError("--folds cannot be combined with a holdout; the holdout is redrawn from the config seed.")
        dev, hold, holdout_info = holdout
        sub = dataclasses.replace(bundle, X=bundle.X.iloc[dev], y=bundle.y[dev], ids=bundle.ids[dev],
                                  groups=None if bundle.groups is None else bundle.groups[dev],
                                  times=None if bundle.times is None else bundle.times[dev])
        dev_folds, split_notes = make_splits(cfg, sub, y[dev])
        folds = [(dev[tr], dev[va]) for tr, va in dev_folds]  # back to full-data row indices
        split_notes.append(f"CV uses the {len(dev)} development rows only; {len(hold)} holdout rows are never "
                           f"used for fitting or selection.")
    atomic_json(run_dir / "fold_indices.json", {
        "schema_version": 1, "n_rows": len(y), "row_ids": bundle.ids.tolist(),
        "split_method": cfg["split"]["method"], "split_config": cfg["split"], "seed": seed,
        "holdout_indices": None if holdout is None else hold.tolist(),
        "folds": [{"fold": i, "train_indices": tr.tolist(), "validation_indices": va.tolist()}
                  for i, (tr, va) in enumerate(folds)]})
    if holdout is not None:
        evidence["holdout"] = {**holdout_info, "status": "pending"}
    evidence.update({
        "task": task, "metric": metric, "metric_direction": "minimize" if metric in LOWER_IS_BETTER else "maximize",
        "evaluation_kind": "selection_cv", "selection_statistic": "fold_score_mean",
        "cv_interpretation": "These CV results compare/select candidates; they are not an unbiased independent test estimate. Fold SD is descriptive, not a CI.",
        "metric_details": {"macro_f1_labels": classes if metric == "macro_f1" else None,
                           "positive_class": classes[positive_index] if positive_index is not None else None,
                           "positive_probability_column": f"p_{positive_index}" if positive_index is not None else None,
                           "multiclass_auc": MULTICLASS_AUC.get(metric), "target_transform": transform},
        "class_order": classes, "n_rows": len(y), "n_features": X.shape[1],
        "n_test_rows": len(X_test) if X_test is not None else 0,
        "feature_columns": bundle.feature_columns, "data_diagnostics": bundle.diagnostics,
        "split_file": "fold_indices.json", "split_notes": split_notes,
        "models_requested": names, "threads": threads,
        "limitations": ["Numeric predictors only; mixed types require a ColumnTransformer adaptation.",
                        "No automatic grid search, nested CV, GPU, or hard interruption of estimators.",
                        "The budget check uses heuristic and observed runtime estimates, not a deadline guarantee."]})
    if cfg["split"]["method"] in {"kfold", "stratified"} and (bundle.groups is not None or bundle.times is not None):
        evidence["warnings"].append("A group/time key was supplied but the chosen split is row-random; verify that this matches the exam's independent prediction unit.")
    reserve = max(0.25, min(30.0, budget * 0.1))

    def remaining():
        return budget - (time.perf_counter() - started)

    def checkpoint():
        evidence["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(run_dir / "evidence.json", jsonable(evidence))

    for name in names:
        result = {"name": name, "status": "pending", "folds": [], "warnings": [],
                  "parameters": jsonable(build_model(name, task, seed, transform).get_params(deep=True))}
        evidence["experiments"].append(result)
        if name == "linear_svm" and needs_probability:
            result.update(status="skipped_metric_incompatibility" if metric in PROBABILITY_METRICS else "skipped_prediction_incompatibility",
                          reason="The metric or prediction.kind requires probabilities; LinearSVC has no calibrated predict_proba. No calibration is fitted implicitly.")
            checkpoint()
            continue
        if name.startswith("pca_") and X.shape[1] <= CappedPCA().max_components:
            result.update(status="skipped_redundant",
                          reason=f"Only {X.shape[1]} features: PCA would keep all of them, a rotation that leaves an "
                                 "L2-regularised model on standardised features unchanged. Equivalent to the non-PCA candidate.")
            checkpoint()
            continue
        estimated = initial_fit_estimate(name, len(folds[0][0]), X.shape[1])
        if remaining() < estimated * len(folds) + reserve:
            result.update(status="skipped_budget", estimated_cv_seconds=estimated * len(folds))
            checkpoint()
            continue
        oof_prediction = np.full(len(y), np.nan)
        oof_prob = np.full((len(y), len(classes)), np.nan) if task == "classification" and name != "linear_svm" else None
        oof_fold = np.full(len(y), -1, dtype=int)
        experiment_started = time.perf_counter()
        weight_sum = weight_count = weight_kind = None  # fold-averaged feature weights, filled lazily

        def save_oof():
            covered = oof_fold >= 0
            decoded = np.full(len(y), None, dtype=object) if task == "classification" else oof_prediction.copy()
            if task == "classification" and covered.any():
                decoded[covered] = encoder.inverse_transform(oof_prediction[covered].astype(int))
            frame = pd.DataFrame({"row_id": bundle.ids, "fold": oof_fold,
                                  "y_true": bundle.y, "prediction": decoded})
            if oof_prob is not None:
                for j in range(len(classes)):
                    frame[f"p_{j}"] = oof_prob[:, j]
            atomic_csv(run_dir / f"oof_{name}.csv", frame)
            result["oof_file"] = f"oof_{name}.csv"
            result["oof_covered_rows"] = int(covered.sum())
            result["oof_uncovered_rows"] = int((~covered).sum())

        for number, (tr, va) in enumerate(folds):
            if remaining() < estimated + reserve:
                result.update(status="partial_budget", estimated_next_fit_seconds=estimated)
                break
            model = build_model(name, task, seed, transform, threads)
            fold_started = time.perf_counter()
            captured = []
            try:
                with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=threads):
                    warnings.simplefilter("always")
                    fit_started = time.perf_counter()
                    model.fit(X.iloc[tr], y[tr])
                    fit_seconds = time.perf_counter() - fit_started
                    prediction, probabilities = aligned_predict(model, X.iloc[va], task, len(classes))
                    value = score(metric, y[va], prediction, probabilities, len(classes), positive_index)
                    # Same metric on the fold's own training rows: the train-validation gap is an overfitting
                    # diagnostic, never a selection criterion.
                    train_prediction, train_probabilities = aligned_predict(model, X.iloc[tr], task, len(classes))
                    train_value = score(metric, y[tr], train_prediction, train_probabilities, len(classes), positive_index)
                    weights = feature_weights(model, X.columns)
                oof_prediction[va] = prediction
                if oof_prob is not None:
                    oof_prob[va] = probabilities
                oof_fold[va] = number
                if weights is not None:
                    w_names, w_values, weight_kind = weights
                    if weight_sum is None:
                        weight_sum = np.zeros((w_values.shape[0], X.shape[1]))
                        weight_count = np.zeros(X.shape[1], dtype=int)
                    position = X.columns.get_indexer(w_names)
                    weight_sum[:, position] += w_values
                    weight_count[position] += 1
                elapsed = time.perf_counter() - fold_started
                details = {"fold": number, "score": value, "train_score": train_value,
                           "n_train": len(tr), "n_validation": len(va),
                           "fit_seconds": fit_seconds, "elapsed_seconds": elapsed,
                           "warnings": warning_records(captured), "fitted_preprocessing": fit_details(model)}
                result["folds"].append(details)
                estimated = max(0.02, max(f["elapsed_seconds"] for f in result["folds"]) * 1.5)
                result["estimated_next_fit_seconds"] = estimated
                save_oof()
                checkpoint()
            except Exception as exc:
                result.update(status="failed_fit_or_score", failed_fold=number,
                              error=f"{type(exc).__name__}: {exc}")
                result["warnings"].extend(warning_records(captured))
                break
        result["elapsed_seconds"] = time.perf_counter() - experiment_started
        save_oof()
        if len(result["folds"]) == len(folds):
            values = np.asarray([f["score"] for f in result["folds"]])
            covered = oof_fold >= 0
            result.update(status="complete", fold_score_mean=float(values.mean()),
                          fold_score_sd=float(values.std(ddof=0)), fold_sd_ddof=0,
                          train_score_mean=float(np.mean([f["train_score"] for f in result["folds"]])),
                          pooled_oof_score=score(metric, y[covered], oof_prediction[covered],
                                                 oof_prob[covered] if oof_prob is not None else None,
                                                 len(classes), positive_index))
            if weight_sum is not None:
                result.update(write_importance(run_dir, name, X.columns, weight_sum, weight_count, len(folds),
                                               weight_kind, classes, task))
        checkpoint()
        print(json.dumps({"model": name, "status": result["status"],
                          "mean_score": result.get("fold_score_mean"),
                          "seconds_remaining": round(remaining(), 2)}), flush=True)

    eligible = [r for r in evidence["experiments"] if r["status"] == "complete"]
    if not eligible:
        evidence["status"] = "no_complete_candidate"
        return 3
    winner = (min if metric in LOWER_IS_BETTER else max)(eligible, key=lambda r: r["fold_score_mean"])
    evidence["selection"] = {"model": winner["name"], "score": winner["fold_score_mean"],
                             "metric": metric, "evaluation_kind": "selection_cv",
                             "tie_break": "first candidate in declared order; dummy is a valid baseline candidate"}
    if holdout is not None:
        # The selected pipeline only, refitted on all development rows and scored once on the untouched holdout.
        holdout_model = build_model(winner["name"], task, seed, transform, threads)
        captured = []
        try:
            with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=threads):
                warnings.simplefilter("always")
                holdout_model.fit(X.iloc[dev], y[dev])
                prediction, probabilities = aligned_predict(holdout_model, X.iloc[hold], task, len(classes))
                value = score(metric, y[hold], prediction, probabilities, len(classes), positive_index)
            frame = pd.DataFrame({"row_id": bundle.ids[hold], "y_true": bundle.y[hold],
                                  "prediction": encoder.inverse_transform(prediction) if encoder is not None else prediction})
            if probabilities is not None:
                for j in range(len(classes)):
                    frame[f"p_{j}"] = probabilities[:, j]
            atomic_csv(run_dir / "holdout_predictions.csv", frame)
            evidence["holdout"].update(status="complete", model=winner["name"], metric=metric, score=value,
                                       predictions_file="holdout_predictions.csv",
                                       evaluation_kind="untouched_holdout_single_evaluation",
                                       warnings=warning_records(captured))
        except Exception as exc:
            evidence["holdout"].update(status="failed", error=f"{type(exc).__name__}: {exc}")
        checkpoint()
    evidence["final_refit"] = {"status": "pending", "n_train": len(y)}
    estimate = max(0.02, max(f["elapsed_seconds"] for f in winner["folds"]) * 1.5
                   * (len(y) / min(f["n_train"] for f in winner["folds"])) ** 2)
    evidence["final_refit"]["estimated_seconds"] = estimate
    if remaining() < estimate + 0.1:
        evidence["final_refit"]["status"] = "skipped_budget"
        evidence["status"] = "selected_without_final_refit"
        return 3
    final_model = build_model(winner["name"], task, seed, transform, threads)
    final_started = time.perf_counter()
    captured = []
    try:
        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=threads):
            warnings.simplefilter("always")
            final_model.fit(X, y)
            evidence["final_refit"].update(status="complete", fitted_preprocessing=fit_details(final_model))
            if X_test is not None:
                prediction, probabilities = aligned_predict(final_model, X_test, task, len(classes))
                output = pd.DataFrame({"row_id": bundle.test_ids})
                if prediction_kind == "probability":
                    if probabilities is None:
                        raise ValueError("The selected model cannot produce the configured probability output.")
                    for j in range(len(classes)):
                        output[f"p_{j}"] = probabilities[:, j]
                    evidence["prediction_kind"] = "probabilities"
                else:
                    output["prediction"] = encoder.inverse_transform(prediction) if encoder is not None else prediction
                    evidence["prediction_kind"] = "labels" if encoder is not None else "continuous"
                atomic_csv(run_dir / "test_predictions.csv", output)
                evidence["test_predictions_file"] = "test_predictions.csv"
                evidence["test_predictions_rows"] = len(output)
        evidence["status"] = "complete" if X_test is not None else "complete_no_test"
    except Exception as exc:
        evidence["final_refit"].update(status="failed", error=f"{type(exc).__name__}: {exc}")
        evidence["status"] = "selected_final_fit_or_prediction_failed"
        return 3
    finally:
        evidence["final_refit"].update(elapsed_seconds=time.perf_counter() - final_started,
                                       warnings=warning_records(captured))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Exam configuration JSON.")
    parser.add_argument("--models", help="Comma-separated candidates; dummy is always included.")
    parser.add_argument("--budget-seconds", type=float, help="Wall-time planning budget, checked before each fit.")
    parser.add_argument("--folds", help="Reuse a previous fold_indices.json with the identical ordered training IDs.")
    args = parser.parse_args()
    started = time.perf_counter()
    cfg = load_config(args.config)
    budget = float(args.budget_seconds if args.budget_seconds is not None else cfg.get("budget_seconds", 600))
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("budget_seconds must be finite and positive.")
    run_dir = make_output_dir(cfg) / (datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    run_dir.mkdir(exist_ok=False)
    evidence = {"schema_version": 1, "status": "running", "run_dir": str(run_dir),
                "created_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
                "budget_seconds": budget, "command": sys.argv, "warnings": [], "experiments": [],
                "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                             "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}}
    atomic_json(run_dir / "evidence.json", evidence)
    try:
        exit_code = run_evaluation(cfg, args, evidence, run_dir, started, budget)
    except KeyboardInterrupt:
        evidence.update(status="interrupted", error="Interrupted; already completed experiments and OOF files were preserved.")
        exit_code = 130
    except Exception as exc:
        evidence.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        exit_code = 2
    finally:
        evidence["elapsed_seconds"] = time.perf_counter() - started
        evidence["budget_exceeded"] = evidence["elapsed_seconds"] > budget
        atomic_json(run_dir / "evidence.json", jsonable(evidence))
    print(json.dumps({"status": evidence["status"], "run_dir": str(run_dir),
                      "evidence": str(run_dir / "evidence.json"),
                      "selected_model": evidence.get("selection", {}).get("model"),
                      "error": evidence.get("error")}, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
