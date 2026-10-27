#!/usr/bin/env python3
"""Leaderboard mode: budgeted, leakage-safe search for the highest validated score.

For tasks ranked by a hidden test score. Compared with evaluate_models.py it adds
mixed-type preprocessing, boosted trees and other families, bounded random search,
repeated cross-validation (to reduce selection noise on small data), a
cross-validated greedy blend, and seed-averaged final refits. Every learned step
still lives inside each training fold, and selection never looks at the public
leaderboard.

Outputs are compatible with validate_submission.py (test_predictions.csv +
evidence.json) and with plot_results.py / build_reports.py.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import platform
import sys
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder
from sklearn.svm import SVC, SVR
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
from evaluate_models import (CLASS_METRICS, LOWER_IS_BETTER, PROBABILITY_METRICS,  # noqa: E402
                             REG_METRICS, atomic_csv, check_regression_target, classification_setup, jsonable,
                             make_splits, score, target_transform, wrap_target)
from exam_io import atomic_json, load_config, load_data, make_output_dir, file_metadata  # noqa: E402

LABELS = {"logreg_l1": "L1 logistic", "logreg_l2": "L2 logistic", "logreg_enet": "Elastic-net logistic",
          "lda": "Shrinkage LDA", "svm_rbf": "RBF SVM", "random_forest": "Random forest",
          "extra_trees": "Extra trees", "lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost",
          "ridge": "Ridge", "lasso": "Lasso", "enet": "Elastic net", "svr_rbf": "RBF SVR", "blend": "CV-selected blend"}
CLASS_FAMILIES = ["logreg_l1", "logreg_l2", "logreg_enet", "lda", "svm_rbf", "random_forest", "extra_trees",
                  "lightgbm", "xgboost", "catboost"]
REG_FAMILIES = ["ridge", "lasso", "enet", "svr_rbf", "random_forest", "extra_trees", "lightgbm", "xgboost", "catboost"]
LINEAR = {"logreg_l1", "logreg_l2", "logreg_enet", "lda", "svm_rbf", "ridge", "lasso", "enet", "svr_rbf"}
STOCHASTIC = {"random_forest", "extra_trees", "lightgbm", "xgboost", "catboost"}


def log_uniform(rng, low, high):
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def default_families(task, n, p):
    """Route by data shape; small n favours strong regularisation over trees."""
    if task == "classification":
        if n < 2000:
            return ["logreg_l1", "logreg_l2", "logreg_enet", "lda", "svm_rbf", "extra_trees", "lightgbm"]
        return ["logreg_l2", "lightgbm", "catboost", "xgboost", "extra_trees"]
    if n < 2000:
        return ["ridge", "lasso", "enet", "svr_rbf", "extra_trees", "lightgbm"]
    return ["ridge", "lightgbm", "catboost", "xgboost", "extra_trees"]


def sample_params(family, rng, first, n_classes):
    """First trial is a sensible default; later trials are random within bounded ranges."""
    if family in ("logreg_l1", "logreg_l2", "logreg_enet"):
        c = {"logreg_l1": 0.1, "logreg_l2": 0.05, "logreg_enet": 0.1}[family] if first else (
            log_uniform(rng, 0.01, 2.0) if family != "logreg_l2" else log_uniform(rng, 1e-3, 1.0))
        ratio = {"logreg_l1": 1.0, "logreg_l2": 0.0}.get(family, 0.5 if first else float(rng.uniform(0.1, 0.9)))
        return {"C": c, "l1_ratio": ratio, "class_weight": None if first or rng.random() < 0.5 else "balanced"}
    if family == "lda":
        return {"shrinkage": "auto" if first else float(rng.uniform(0.05, 0.95))}
    if family in ("svm_rbf", "svr_rbf"):
        return {"C": 1.0 if first else log_uniform(rng, 0.05, 20), "gamma": "scale" if first else log_uniform(rng, 1e-4, 0.1)}
    if family in ("random_forest", "extra_trees"):
        return {"n_estimators": 500, "max_features": "sqrt" if first else str(rng.choice(["sqrt", "log2", "0.3", "0.6"])),
                "min_samples_leaf": 1 if first else int(rng.integers(1, 10))}
    if family == "lightgbm":
        # DART (dropout on trees) is offered in the random phase: slower, often less overfit on wide data.
        return {"boosting_type": "gbdt" if first or rng.random() < 0.7 else "dart",
                "n_estimators": 400 if first else int(rng.integers(150, 900)),
                "learning_rate": 0.05 if first else log_uniform(rng, 0.01, 0.15),
                "num_leaves": 15 if first else int(rng.integers(4, 64)),
                "min_child_samples": 20 if first else int(rng.integers(5, 60)),
                "subsample": 0.8 if first else float(rng.uniform(0.5, 1.0)), "subsample_freq": 1,
                "colsample_bytree": 0.8 if first else float(rng.uniform(0.3, 1.0)),
                "reg_lambda": 1.0 if first else log_uniform(rng, 1e-3, 30)}
    if family == "xgboost":
        return {"n_estimators": 400 if first else int(rng.integers(150, 900)),
                "learning_rate": 0.05 if first else log_uniform(rng, 0.01, 0.15),
                "max_depth": 4 if first else int(rng.integers(2, 8)),
                "subsample": 0.8 if first else float(rng.uniform(0.5, 1.0)),
                "colsample_bytree": 0.8 if first else float(rng.uniform(0.3, 1.0)),
                "min_child_weight": 1.0 if first else log_uniform(rng, 0.5, 20),
                "reg_lambda": 1.0 if first else log_uniform(rng, 1e-3, 30)}
    if family == "catboost":
        return {"iterations": 500 if first else int(rng.integers(200, 1000)),
                "learning_rate": 0.05 if first else log_uniform(rng, 0.01, 0.15),
                "depth": 6 if first else int(rng.integers(3, 9)),
                "l2_leaf_reg": 3.0 if first else log_uniform(rng, 1, 30)}
    if family in ("ridge", "lasso", "enet"):
        alpha = {"ridge": 10.0, "lasso": 0.05, "enet": 0.05}[family] if first else (
            log_uniform(rng, 0.01, 300) if family == "ridge" else log_uniform(rng, 1e-4, 1.0))
        out = {"alpha": alpha}
        if family == "enet":
            out["l1_ratio"] = 0.5 if first else float(rng.uniform(0.1, 0.9))
        return out
    raise ValueError(f"Unknown family {family!r}")


HIGH_CARDINALITY = 30


def preprocessor(family, num_cols, cat_cols, high_card, task, scaled):
    """Fold-local preprocessing. `scaled` means numeric columns were already standardised
    transductively (train+test, no labels), so the linear branch skips its own scaler."""
    if family == "catboost":
        return None
    low_card = [c for c in cat_cols if c not in high_card]
    if family in LINEAR:
        steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("constant", VarianceThreshold())]
        num = Pipeline(steps if scaled else steps + [("scale", StandardScaler())])
        low = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5,
                                                 sparse_output=False))])
        # High-cardinality categories: sklearn TargetEncoder cross-fits internally, so no label leakage in-fold.
        high = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("target", TargetEncoder(target_type="continuous" if task == "regression" else "auto")),
                         ("scale", StandardScaler())])
    else:
        num = SimpleImputer(strategy="median", keep_empty_features=True)
        low = high = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                               ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
    parts = [("num", num, num_cols)] + ([("cat", low, low_card)] if low_card else []) + \
            ([("high", high, high_card)] if high_card else [])
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


def estimator(family, params, task, seed, threads, n_classes, cat_idx):
    p = dict(params)
    if family in ("logreg_l1", "logreg_l2", "logreg_enet"):
        solver = "liblinear" if (p["l1_ratio"] == 1.0 and n_classes == 2) else ("lbfgs" if p["l1_ratio"] == 0 else "saga")
        options = {"C": p["C"], "l1_ratio": p["l1_ratio"], "solver": solver, "class_weight": p["class_weight"],
                   "max_iter": 5000, "random_state": seed}
        # Older sklearn selects the regularizer with penalty; newer versions use
        # l1_ratio and deprecate penalty. Preserve the requested model in both APIs.
        if LogisticRegression().get_params().get("penalty") != "deprecated":
            options["penalty"] = "l1" if p["l1_ratio"] == 1 else "l2" if p["l1_ratio"] == 0 else "elasticnet"
        return LogisticRegression(**options)
    if family == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=p["shrinkage"])
    if family == "svm_rbf":
        return SVC(C=p["C"], gamma=p["gamma"], probability=True, random_state=seed)
    if family == "svr_rbf":
        return SVR(C=p["C"], gamma=p["gamma"])
    if family in ("random_forest", "extra_trees"):
        mf = p["max_features"]
        mf = float(mf) if mf not in ("sqrt", "log2") else mf
        cls = {("random_forest", "classification"): RandomForestClassifier, ("random_forest", "regression"): RandomForestRegressor,
               ("extra_trees", "classification"): ExtraTreesClassifier, ("extra_trees", "regression"): ExtraTreesRegressor}[(family, task)]
        return cls(n_estimators=p["n_estimators"], max_features=mf, min_samples_leaf=p["min_samples_leaf"],
                   n_jobs=threads, random_state=seed)
    if family == "lightgbm":
        import lightgbm as lgb
        cls = lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor
        return cls(**p, n_jobs=threads, random_state=seed, verbose=-1)
    if family == "xgboost":
        import xgboost as xgb
        cls = xgb.XGBClassifier if task == "classification" else xgb.XGBRegressor
        return cls(**p, n_jobs=threads, random_state=seed, tree_method="hist", verbosity=0)
    if family == "catboost":
        import catboost as cb
        cls = cb.CatBoostClassifier if task == "classification" else cb.CatBoostRegressor
        return cls(**p, random_seed=seed, thread_count=threads, verbose=0, cat_features=cat_idx or None,
                   allow_writing_files=False)
    if family == "ridge":
        return Ridge(alpha=p["alpha"])
    if family == "lasso":
        return Lasso(alpha=p["alpha"], max_iter=20000)
    if family == "enet":
        return ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"], max_iter=20000)
    raise ValueError(family)


def build(family, params, task, seed, threads, n_classes, num_cols, cat_cols, high_card=(), scaled=False, transform=None):
    cat_idx = [len(num_cols) + i for i in range(len(cat_cols))]
    model = estimator(family, params, task, seed, threads, n_classes, cat_idx)
    pre = preprocessor(family, num_cols, cat_cols, list(high_card), task, scaled)
    model = model if pre is None else Pipeline([("pre", pre), ("model", model)])
    return wrap_target(model, transform) if task == "regression" else model


class BudgetAbort(Exception):
    """A trial would run into the blend/refit reserve; its partial folds are discarded."""


def trial_size(family, params):
    """Relative cost of a configuration, so a 900-tree trial is not budgeted like a 150-tree one."""
    size = float(params.get("n_estimators") or params.get("iterations") or 1)
    return size * (3.0 if params.get("boosting_type") == "dart" else 1.0)


def adversarial_validation(X, X_test, num_cols, cat_cols, seed, threads):
    """Train-vs-test classifier: AUC near 0.5 means no detectable shift (labels are never used)."""
    try:
        import lightgbm as lgb
    except (ImportError, OSError) as exc:
        return {"status": "skipped", "reason": f"Optional LightGBM unavailable: {type(exc).__name__}: {exc}"}
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    both = pd.concat([X, X_test], ignore_index=True)
    is_test = np.r_[np.zeros(len(X)), np.ones(len(X_test))].astype(int)
    if len(both) > 60000:
        keep = np.random.default_rng(seed).choice(len(both), 60000, replace=False)
        both, is_test = both.iloc[keep].reset_index(drop=True), is_test[keep]
    enc = both.copy()
    for c in cat_cols:
        enc[c] = enc[c].astype("category").cat.codes
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, n_jobs=threads, random_state=seed, verbose=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob = cross_val_predict(model, enc, is_test, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                 method="predict_proba")[:, 1]
        model.fit(enc, is_test)
    auc = float(roc_auc_score(is_test, prob))
    top = pd.Series(model.booster_.feature_importance("gain"), index=enc.columns).sort_values(ascending=False).head(10)
    level = "none" if auc < 0.6 else "moderate" if auc < 0.75 else "strong"
    return {"status": "complete", "auc": auc, "shift": level, "top_shift_features": {str(k): float(v) for k, v in top.items()},
            "train_rows_most_test_like": [int(i) for i in np.argsort(-prob[:len(X)])[:20]] if len(prob) > len(X) else [],
            "advice": ("No action." if level == "none" else
                       "Inspect the top features for IDs, time stamps or leakage; consider validating on the most test-like "
                       "training rows. Drop a feature only with a concrete reason.")}


def prepare(frame, num_cols, cat_cols):
    """Column order and dtypes shared by every family (CatBoost needs string categories)."""
    out = frame[num_cols + cat_cols].copy()
    for c in cat_cols:
        out[c] = out[c].astype("string").fillna("__missing__").astype(str)
    return out


def predict_scores(model, X, task, n_classes, metric=None):
    """Probabilities for classification (n x K), predictions for regression (n,)."""
    if task == "classification" and n_classes == 2 and metric == "roc_auc" and hasattr(model, "decision_function") \
            and isinstance(getattr(model, "named_steps", {}).get("model", model), SVC):
        # SVC's internal Platt scaling can invert the ranking on small data; AUC needs only the decision order.
        d = np.asarray(model.decision_function(X), dtype=float)
        pos = 1 / (1 + np.exp(-d))
        return np.column_stack([1 - pos, pos])
    if task == "classification":
        local = np.asarray(model.predict_proba(X), dtype=float)
        classes = np.asarray(getattr(model, "classes_", np.arange(n_classes)), dtype=int)
        out = np.zeros((len(X), n_classes))
        out[:, classes] = local
        return out
    return np.asarray(model.predict(X), dtype=float)


def metric_from_scores(metric, y, s, n_classes, positive):
    if metric in REG_METRICS:
        return score(metric, y, s, None, n_classes, positive)
    labels = s.argmax(axis=1)
    return score(metric, y, labels, s, n_classes, positive)


def to_rank(s):
    """Rank-normalise positive-class scores to [0, 1] (AUC depends only on order)."""
    r = rankdata(s) - 1
    return r / max(len(s) - 1, 1)


def combine(members, metric, n_classes):
    """Equal-weight average of member scores (members may repeat = weights).
    Binary AUC rank-averages the positive column (index 1); everything else averages probabilities."""
    if metric == "roc_auc":
        pos = np.mean([to_rank(m[:, 1]) for m in members], axis=0)
        return np.column_stack([1 - pos, pos])
    return np.mean(members, axis=0)


def greedy_blend(pool, y, metric, n_classes, positive, rows, max_size=12):
    """Caruana forward selection with replacement on the given rows; returns member names."""
    sign = -1 if metric in LOWER_IS_BETTER else 1
    chosen, best = [], -np.inf
    for _ in range(max_size):
        trial = None
        for name, s in pool.items():
            candidate = combine([pool[c][rows] for c in chosen] + [s[rows]], metric, n_classes)
            value = sign * metric_from_scores(metric, y[rows], candidate, n_classes, positive)
            if trial is None or value > trial[1] + 1e-12:
                trial = (name, value)
        if trial[1] <= best + 1e-12:
            break
        chosen.append(trial[0])
        best = trial[1]
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--budget-seconds", type=float, help="Wall-time for search + blend + refit (default: config budget_seconds).")
    parser.add_argument("--families", help=f"Comma-separated subset of {', '.join(CLASS_FAMILIES)} (classification) "
                        f"or {', '.join(REG_FAMILIES)} (regression). Default routes by data shape.")
    parser.add_argument("--repeats", type=int, default=None, help="CV repeats (default 5 if n<1000, 3 if n<10000, else 1).")
    parser.add_argument("--max-trials", type=int, default=12, help="Upper bound on configurations per family.")
    parser.add_argument("--blend-pool", type=int, default=15, help="Best configurations offered to the blend.")
    parser.add_argument("--seeds", type=int, default=5, help="Seeds averaged in the final refit of stochastic models.")
    parser.add_argument("--transductive", action="store_true",
                        help="Standardise numeric features on train+test (no labels). Only if the rules allow using test "
                             "features; mainly useful when n is small.")
    parser.add_argument("--no-adversarial", action="store_true", help="Skip the train-vs-test shift check.")
    args = parser.parse_args()
    started = time.perf_counter()
    cfg = load_config(args.config)
    budget = float(args.budget_seconds or cfg.get("budget_seconds", 900))
    run_dir = make_output_dir(cfg) / (datetime.now(timezone.utc).strftime("lb_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    run_dir.mkdir(exist_ok=False)
    evidence = {"schema_version": 1, "mode": "leaderboard", "status": "running", "run_dir": str(run_dir),
                "created_utc": datetime.now(timezone.utc).isoformat(), "config": cfg, "budget_seconds": budget,
                "command": sys.argv, "warnings": [], "experiments": [],
                "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                             "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}}
    try:
        code = run(cfg, args, evidence, run_dir, started, budget)
    except KeyboardInterrupt:
        evidence.update(status="interrupted")
        code = 130
    except Exception as exc:
        evidence.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        code = 2
    evidence["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(run_dir / "evidence.json", jsonable(evidence))
    print(json.dumps({"status": evidence["status"], "run_dir": str(run_dir), "selected": evidence.get("selection", {}),
                      "error": evidence.get("error")}, ensure_ascii=False, default=str), flush=True)
    return code


def run(cfg, args, evidence, run_dir, started, budget) -> int:
    task, metric = cfg["task"], cfg.get("metric")
    permitted = CLASS_METRICS if task == "classification" else REG_METRICS
    if metric not in permitted:
        raise ValueError(f"Set the official metric explicitly; {task} supports {sorted(permitted)}.")
    seed, threads = int(cfg.get("seed", 42)), int(cfg.get("threads", 2))
    evidence["data_sources"] = {role: (file_metadata(cfg[k]) if cfg.get(k) else None)
                                for role, k in (("train", "train_path"), ("labels", "labels_path"), ("test", "test_path"))}
    bundle = load_data(cfg, include_test=True)
    kind = (cfg.get("prediction") or {}).get("kind")
    if bundle.X_test is not None and kind not in ("label", "probability", "regression"):
        raise ValueError("Set prediction.kind from the submission contract before a leaderboard run.")
    if bundle.X_test is not None and metric in PROBABILITY_METRICS and kind != "probability":
        raise ValueError(f"{metric} is scored on probabilities/scores; set prediction.kind=probability.")
    num_cols = [c for c in bundle.X.columns if pd.api.types.is_numeric_dtype(bundle.X[c]) or pd.api.types.is_bool_dtype(bundle.X[c])]
    cat_cols = [c for c in bundle.X.columns if c not in num_cols]
    X = prepare(bundle.X, num_cols, cat_cols)
    X_test = prepare(bundle.X_test, num_cols, cat_cols) if bundle.X_test is not None else None
    high_card = [c for c in cat_cols if X[c].nunique() > HIGH_CARDINALITY]
    if X_test is not None and not args.no_adversarial:
        check = adversarial_validation(X, X_test, num_cols, cat_cols, seed, threads)
        evidence["adversarial_validation"] = check
        if check["status"] == "complete":
            print(json.dumps({"adversarial_auc": round(check["auc"], 4), "shift": check["shift"]}), flush=True)
        else:
            evidence["warnings"].append(check["reason"])
            print(json.dumps({"adversarial_validation": check}), flush=True)
    scaled = False
    if args.transductive and X_test is not None and num_cols:
        both = pd.concat([X[num_cols], X_test[num_cols]])
        mu, sd = both.mean(), both.std(ddof=0).replace(0, 1)
        X[num_cols] = (X[num_cols] - mu) / sd
        X_test[num_cols] = (X_test[num_cols] - mu) / sd
        scaled = True
        evidence["transductive_scaling"] = "numeric features standardised on train+test features (labels unused)"
    classes, positive, encoder = [], None, None
    transform = target_transform(cfg)
    if task == "classification":
        encoder = LabelEncoder().fit(bundle.y)
        classes = encoder.classes_.tolist()
        y = encoder.transform(bundle.y)
        positive = classification_setup(metric, classes, cfg)
        if metric == "roc_auc" and positive != 1:
            raise ValueError("positive_class must sort as the second class for this runner's AUC convention.")
    else:
        y = np.asarray(bundle.y, dtype=float)
        check_regression_target(y, metric, transform)
    n_classes = max(len(classes), 1)
    n, p = X.shape
    repeats = args.repeats or (5 if n < 1000 else 3 if n < 10000 else 1)
    method = cfg["split"]["method"]
    k = int(cfg["split"].get("n_splits", 5))
    if method in ("stratified", "kfold"):
        splitter = (RepeatedStratifiedKFold if method == "stratified" else RepeatedKFold)(
            n_splits=k, n_repeats=repeats, random_state=seed)
        folds = list(splitter.split(X, y))
    else:
        folds, _ = make_splits(cfg, bundle, y)  # group/time: one repeat, same safeguards as evaluate_models
        repeats, k = 1, len(folds)
    atomic_json(run_dir / "fold_indices.json", {"schema_version": 1, "n_rows": n, "row_ids": bundle.ids.tolist(),
                                                "split_method": method, "repeats": repeats, "n_splits": k, "seed": seed,
                                                "folds": [{"repeat": i // k, "fold": i % k, "train_indices": tr.tolist(),
                                                           "validation_indices": va.tolist()} for i, (tr, va) in enumerate(folds)]})
    families = [f.strip() for f in args.families.split(",")] if args.families else default_families(task, n, p)
    allowed = CLASS_FAMILIES if task == "classification" else REG_FAMILIES
    if set(families) - set(allowed):
        raise ValueError(f"Unknown families {sorted(set(families) - set(allowed))}; choose from {allowed}.")
    if metric in PROBABILITY_METRICS:
        families = [f for f in families if f != "svr_rbf"]
    requested_families = list(families)
    unavailable = {}
    for family in families:
        if family not in {"lightgbm", "xgboost", "catboost"}:
            continue
        try:
            module = importlib.import_module(family)
            evidence["versions"][family] = module.__version__
        except (ImportError, OSError) as exc:
            unavailable[family] = f"{type(exc).__name__}: {exc}"
            print(json.dumps({"family": family, "status": "skipped_dependency", "reason": unavailable[family]}), flush=True)
    families = [family for family in families if family not in unavailable]
    if unavailable:
        evidence["warnings"].append("Optional model families skipped because their dependencies are unavailable: "
                                    + ", ".join(unavailable))
    sign = -1 if metric in LOWER_IS_BETTER else 1
    evidence.update({"task": task, "metric": metric, "metric_direction": "minimize" if sign < 0 else "maximize",
                     "evaluation_kind": "selection_cv_repeated", "class_order": classes, "n_rows": n, "n_features": p,
                     "n_numeric": len(num_cols), "n_categorical": len(cat_cols), "categorical_columns": cat_cols,
                     "n_test_rows": len(X_test) if X_test is not None else 0, "repeats": repeats, "n_splits": k,
                     "families": families, "families_requested": requested_families, "families_unavailable": unavailable,
                     "feature_columns": bundle.feature_columns, "data_diagnostics": bundle.diagnostics,
                     "metric_details": {"positive_class": classes[positive] if positive is not None else None,
                                        "positive_probability_column": f"p_{positive}" if positive is not None else None,
                                        "label_rule": "arg-max of class probabilities" if task == "classification" else None,
                                        "target_transform": transform},
                     "split_file": "fold_indices.json",
                     "cv_interpretation": ("Repeated selection CV over many configurations: the best score is optimistic "
                                           "(winner's curse). The blend is chosen by nested cross-validation over folds.")})
    reserve = max(5.0, budget * 0.25)  # blend + final refit
    rng = np.random.default_rng(seed)
    trials, pool = [], {}

    def remaining():
        return budget - (time.perf_counter() - started)

    def cv_trial(family, params, tid, make=None):
        """Repeated CV; returns per-fold scores, repeat-averaged OOF scores and seconds.
        Aborts (BudgetAbort) as soon as the projected finish would eat into the reserve."""
        oof_sum = np.zeros((n, n_classes)) if task == "classification" else np.zeros(n)
        oof_cnt = np.zeros(n)
        scores_, t0 = [], time.perf_counter()
        for i, (tr, va) in enumerate(folds):
            if remaining() - reserve <= 0:
                raise BudgetAbort(f"reserve reached before fold {i + 1}/{len(folds)}")
            model = make(i) if make else build(family, params, task, seed + i, threads, n_classes, num_cols, cat_cols,
                                               high_card, scaled, transform)
            with warnings.catch_warnings(), threadpool_limits(limits=threads):
                warnings.simplefilter("ignore")
                model.fit(X.iloc[tr], y[tr])
                s = predict_scores(model, X.iloc[va], task, n_classes, metric)
            scores_.append(metric_from_scores(metric, y[va], s, n_classes, positive))
            oof_sum[va] += s
            oof_cnt[va] += 1
            per_fold = (time.perf_counter() - t0) / (i + 1)
            left = len(folds) - i - 1
            if left and per_fold * left > remaining() - reserve:
                raise BudgetAbort(f"projected {per_fold * len(folds):.1f}s for {len(folds)} folds; "
                                  f"{remaining() - reserve:.1f}s left before the reserve after fold {i + 1}")
        oof = oof_sum / (oof_cnt[:, None] if task == "classification" else oof_cnt)
        per_repeat = [float(np.mean(scores_[r * k:(r + 1) * k])) for r in range(repeats)]
        return scores_, per_repeat, oof, time.perf_counter() - t0

    # Majority/prior (classification) or mean (regression) baseline on the same folds; never blended or selected.
    def make_dummy(_):
        return DummyClassifier(strategy="prior") if task == "classification" else DummyRegressor()
    baseline = None
    try:
        b_scores, _, b_oof, b_secs = cv_trial("dummy", {}, "dummy", make=make_dummy)
        baseline = {"scores": b_scores, "oof": b_oof, "seconds": b_secs}
    except (BudgetAbort, ValueError) as exc:
        evidence["warnings"].append(f"Baseline skipped: {exc}")

    # Round-robin: defaults for every family first, then random configurations, until the search budget ends.
    # est holds seconds per unit of trial_size, so larger random configurations are budgeted proportionally.
    est = {}
    for round_ in range(args.max_trials):
        progressed = False
        for family in families:
            params = sample_params(family, rng, round_ == 0, n_classes)
            expected = est[family] * trial_size(family, params) if family in est else 1.0
            if remaining() - reserve < expected * 1.2:
                continue
            tid = f"{family}#{round_}"
            try:
                fold_scores, per_repeat, oof, secs = cv_trial(family, params, tid)
            except BudgetAbort as exc:
                trials.append({"id": tid, "family": family, "params": params, "status": "aborted_budget",
                               "reason": str(exc)})
                print(json.dumps({"trial": tid, "status": "aborted_budget", "remaining": round(remaining(), 1)}), flush=True)
                continue
            except Exception as exc:
                trials.append({"id": tid, "family": family, "params": params, "status": "failed",
                               "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
                est[family] = math.inf
                continue
            est[family] = secs / trial_size(family, params)
            progressed = True
            mean = float(np.mean(fold_scores))
            se = float(np.std(per_repeat, ddof=1) / math.sqrt(repeats)) if repeats > 1 else float(
                np.std(fold_scores) / math.sqrt(len(fold_scores)))
            trials.append({"id": tid, "family": family, "params": params, "status": "complete", "mean": mean,
                           "repeat_se": se, "fold_scores": fold_scores, "repeat_means": per_repeat, "seconds": secs})
            pool[tid] = oof
            print(json.dumps({"trial": tid, "mean": round(mean, 5), "se": round(se, 5), "seconds": round(secs, 1),
                              "remaining": round(remaining(), 1)}), flush=True)
        if not progressed:
            break
    done = [t for t in trials if t["status"] == "complete"]
    if not done:
        evidence["status"] = "no_complete_trial"
        evidence["trials"] = trials
        return 3
    done.sort(key=lambda t: sign * t["mean"], reverse=True)
    pd.DataFrame([{"id": t["id"], "family": t["family"], "mean": t["mean"], "repeat_se": t["repeat_se"],
                   "seconds": t["seconds"], "params": json.dumps(jsonable(t["params"]))} for t in done]
                 ).to_csv(run_dir / "trials.csv", index=False)
    best = done[0]

    # Blend chosen by nested CV over the first repeat's folds: select members on k-1 folds, score on the held fold.
    candidates = {t["id"]: pool[t["id"]] for t in done[:args.blend_pool]}
    first = folds[:k]
    nested_blend, nested_single = [], []
    for tr, va in first:
        members = greedy_blend(candidates, y, metric, n_classes, positive, tr)
        nested_blend.append(metric_from_scores(metric, y[va], combine([candidates[m][va] for m in members], metric, n_classes),
                                               n_classes, positive))
        # Fair comparison: also choose the single best config on the k-1 folds only.
        single = max(candidates, key=lambda c: sign * metric_from_scores(metric, y[tr], candidates[c][tr], n_classes, positive))
        nested_single.append(metric_from_scores(metric, y[va], candidates[single][va], n_classes, positive))
    members = greedy_blend(candidates, y, metric, n_classes, positive, np.arange(n))
    blend_oof = combine([candidates[m] for m in members], metric, n_classes)
    use_blend = sign * np.mean(nested_blend) > sign * np.mean(nested_single) and len(set(members)) > 1
    weights = {m: members.count(m) / len(members) for m in dict.fromkeys(members)}
    evidence["blend"] = {"members": weights, "nested_cv_blend": float(np.mean(nested_blend)),
                         "nested_cv_single": float(np.mean(nested_single)), "selected": bool(use_blend),
                         "rule": "Blend replaces the best single configuration only if its nested-CV score is better."}

    # Experiments in evaluate_models format (best configuration per family, plus the blend) for plots and reports.
    def write_oof(name, s):
        frame = pd.DataFrame({"row_id": bundle.ids, "fold": np.zeros(n, dtype=int), "y_true": bundle.y})
        for i, (_, va) in enumerate(first):
            frame.loc[va, "fold"] = i
        if task == "classification":
            frame["prediction"] = encoder.inverse_transform(s.argmax(axis=1))
            for j in range(n_classes):
                frame[f"p_{j}"] = s[:, j]
        else:
            frame["prediction"] = s
        atomic_csv(run_dir / f"oof_{name}.csv", frame)
        return f"oof_{name}.csv"

    if baseline is not None:
        evidence["experiments"].append({
            "name": "dummy", "status": "complete", "parameters": {"strategy": "prior" if task == "classification" else "mean"},
            "folds": [{"fold": i, "score": s} for i, s in enumerate(baseline["scores"])],
            "fold_score_mean": float(np.mean(baseline["scores"])), "fold_score_sd": float(np.std(baseline["scores"])),
            "pooled_oof_score": metric_from_scores(metric, y, baseline["oof"], n_classes, positive),
            "elapsed_seconds": baseline["seconds"], "oof_file": write_oof("dummy", baseline["oof"]),
            "note": "Baseline on the same folds; never selected or blended."})
    seen = set()
    for t in done:
        if t["family"] in seen:
            continue
        seen.add(t["family"])
        evidence["experiments"].append({
            "name": t["family"], "label": LABELS[t["family"]], "status": "complete", "trial_id": t["id"],
            "parameters": t["params"], "folds": [{"fold": i, "score": s} for i, s in enumerate(t["fold_scores"])],
            "fold_score_mean": t["mean"], "fold_score_sd": float(np.std(t["fold_scores"])), "repeat_se": t["repeat_se"],
            "pooled_oof_score": metric_from_scores(metric, y, pool[t["id"]], n_classes, positive),
            "elapsed_seconds": t["seconds"], "oof_file": write_oof(t["family"], pool[t["id"]])})
    # The blend's OOF reuses members chosen on all rows (optimistic), so it is listed as a candidate only when the
    # nested-CV rule selected it; otherwise figures and reports would rank it above the selected model.
    if use_blend:
        blend_folds = [metric_from_scores(metric, y[va], blend_oof[va], n_classes, positive) for _, va in folds]
        evidence["experiments"].append({
            "name": "blend", "label": LABELS["blend"], "status": "complete", "parameters": weights,
            "folds": [{"fold": i, "score": s} for i, s in enumerate(blend_folds)],
            "fold_score_mean": float(np.mean(blend_folds)), "fold_score_sd": float(np.std(blend_folds)),
            "pooled_oof_score": metric_from_scores(metric, y, blend_oof, n_classes, positive),
            "elapsed_seconds": 0.0, "oof_file": write_oof("blend", blend_oof),
            "note": "Blend OOF reuses member OOF chosen on all rows, so its score is optimistic; see blend.nested_cv_*."})
    final_members = weights if use_blend else {best["id"]: 1.0}
    evidence["selection"] = {"model": "blend" if use_blend else best["family"], "trial": best["id"] if not use_blend else None,
                             "members": final_members, "score": float(np.mean(nested_blend if use_blend else [best["mean"]])),
                             "metric": metric, "evaluation_kind": "nested_cv_blend" if use_blend else "selection_cv_repeated",
                             "best_single": {"id": best["id"], "mean": best["mean"], "repeat_se": best["repeat_se"]}}
    evidence["trials"] = [{k_: v for k_, v in t.items() if k_ != "fold_scores"} for t in trials]
    atomic_json(run_dir / "evidence.json", jsonable(evidence))

    # Final refit on all training rows; stochastic families averaged over seeds.
    if X_test is None:
        evidence["status"] = "complete_no_test"
        return 0
    by_id = {t["id"]: t for t in done}
    outputs = {}
    refit_started = time.perf_counter()
    # Budget the refit: one full-data fit costs about one CV fit scaled by n / n_train.
    fit_cost = {tid: by_id[tid]["seconds"] / len(folds) * (n / np.mean([len(tr) for tr, _ in folds])) for tid in by_id}

    def refit_cost(members_, seeds_):
        return sum(fit_cost[m] * (seeds_ if by_id[m]["family"] in STOCHASTIC else 1) for m in members_)
    n_seeds, adjustments = max(1, args.seeds), []
    while n_seeds > 1 and refit_cost(final_members, n_seeds) > remaining():
        n_seeds -= 1
    if n_seeds < args.seeds:
        adjustments.append(f"seeds reduced from {args.seeds} to {n_seeds} to fit the remaining budget")
    if len(final_members) > 1 and refit_cost(final_members, n_seeds) > remaining():
        adjustments.append("blend refit would exceed the budget; fell back to the best single configuration")
        final_members = {best["id"]: 1.0}
        evidence["experiments"] = [e for e in evidence["experiments"] if e["name"] != "blend"]
        evidence["selection"].update(model=best["family"], trial=best["id"], members=final_members,
                                     score=best["mean"], evaluation_kind="selection_cv_repeated",
                                     budget_fallback_from_blend=True)
    evidence["final_refit_plan"] = {"seeds": n_seeds, "estimated_seconds": refit_cost(final_members, n_seeds),
                                    "remaining_seconds": remaining(), "adjustments": adjustments}
    for tid in final_members:
        t = by_id[tid]
        seeds = range(n_seeds if t["family"] in STOCHASTIC else 1)
        preds = []
        for s_ in seeds:
            model = build(t["family"], t["params"], task, seed + 1000 + s_, threads, n_classes, num_cols, cat_cols, high_card,
                          scaled, transform)
            with warnings.catch_warnings(), threadpool_limits(limits=threads):
                warnings.simplefilter("ignore")
                model.fit(X, y)
                preds.append(predict_scores(model, X_test, task, n_classes, metric))
        outputs[tid] = np.mean(preds, axis=0)
    members_list = [m for m, w in final_members.items() for _ in range(max(1, round(w * 100)))]
    final = combine([outputs[m] for m in members_list], metric, n_classes) if len(final_members) > 1 else outputs[next(iter(final_members))]
    out = pd.DataFrame({"row_id": bundle.test_ids})
    if task == "classification":
        if kind == "probability":
            for j in range(n_classes):
                out[f"p_{j}"] = final[:, j]
            evidence["prediction_kind"] = "probabilities" if metric != "roc_auc" or len(final_members) == 1 else "rank_scores_in_[0,1]"
        else:
            out["prediction"] = encoder.inverse_transform(final.argmax(axis=1))
            evidence["prediction_kind"] = "labels"
    else:
        out["prediction"] = final
        evidence["prediction_kind"] = "continuous"
    atomic_csv(run_dir / "test_predictions.csv", out)
    evidence.update(test_predictions_file="test_predictions.csv", test_predictions_rows=len(out),
                    final_refit={"status": "complete", "seconds": time.perf_counter() - refit_started,
                                 "seeds_for_stochastic": n_seeds, "members": final_members, "adjustments": adjustments},
                    status="complete")
    evidence["budget_exceeded"] = (time.perf_counter() - started) > budget
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
