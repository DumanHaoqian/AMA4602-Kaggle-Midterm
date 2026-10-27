#!/usr/bin/env python3
"""Publication-style figures from existing exam runs; never refits or alters results.

Reads an evaluate_models.py run directory and/or an analyze_structure.py output
directory and writes vector PDF + 300 dpi PNG figures, English captions and a
provenance manifest into a new directory. Every number in a title or caption is
read from those outputs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from exam_io import file_metadata
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.transforms import blended_transform_factory
from scipy.optimize import linear_sum_assignment

import viz_style as vs
from viz_style import plt

FIGURES = ("models", "confusion", "holdout", "importance", "gap", "structure", "variance", "robustness",
           "associations", "distributions", "heatmap")
METRIC_NAMES = {"accuracy": "Accuracy", "macro_f1": "Macro-F1", "weighted_f1": "Weighted F1", "f1": "F1",
                "balanced_accuracy": "Balanced accuracy", "mcc": "MCC", "qwk": "Quadratic weighted kappa",
                "log_loss": "Log loss", "roc_auc": "ROC AUC", "roc_auc_ovr": "Macro one-vs-rest ROC AUC",
                "roc_auc_ovr_weighted": "Weighted one-vs-rest ROC AUC", "roc_auc_ovo": "Macro one-vs-one ROC AUC",
                "rmse": "RMSE", "mae": "MAE", "r2": "R²", "rmsle": "RMSLE", "mape": "MAPE"}
# Upper-bounded by 1 (MCC and kappa can be negative, but never exceed 1).
BOUNDED_METRICS = {"accuracy", "macro_f1", "weighted_f1", "f1", "balanced_accuracy", "mcc", "qwk",
                   "roc_auc", "roc_auc_ovr", "roc_auc_ovr_weighted", "roc_auc_ovo"}


def cv_design(ev: dict) -> str:
    """'5-fold stratified', or '5-fold stratified × 5 repeats' for leaderboard runs."""
    split = ev["config"]["split"]
    k = ev.get("n_splits") or split.get("n_splits")
    repeats = int(ev.get("repeats") or 1)
    base = f"{k}-fold {split['method']}"
    return base if repeats == 1 else f"{base} × {repeats} repeats"


def oof_note(ev: dict) -> str:
    """How each pooled out-of-fold prediction was formed."""
    repeats = int(ev.get("repeats") or 1)
    if repeats == 1:
        return "Each sample is predicted once, by the model whose training folds excluded it"
    return (f"Each sample's out-of-fold output is the average over {repeats} repeats, each from a model whose "
            f"training folds excluded it")


def model_label(name: str, parameters: dict, task: str, folds: list | None = None) -> str:
    # A log1p target transform wraps the pipeline, prefixing its parameters with "regressor__".
    parameters = {key.removeprefix("regressor__"): v for key, v in parameters.items()}
    k = parameters.get("select__max_features")
    fitted = [f.get("fitted_preprocessing", {}).get("pca_components") for f in folds or []]
    fitted = sorted({c for c in fitted if isinstance(c, int)})
    n = (fitted[0] if len(fitted) == 1 else f"{fitted[0]}–{fitted[-1]}") if fitted else (
        parameters.get("pca__n_components") or parameters.get("pca__max_components"))
    pca = f"PCA-{n}" if isinstance(n, (int, str)) else "PCA"
    top = f"ANOVA top-{k}" if isinstance(k, int) else "ANOVA top-k"
    return {"dummy": "Majority-class baseline" if task == "classification" else "Mean baseline",
            "select_logistic": f"{top} + logistic", "linear_svm": f"{top} + linear SVM",
            "pca_logistic": f"{pca} + logistic", "ridge": "Ridge", "pca_ridge": f"{pca} + ridge",
            "random_forest": "Random forest", "select_rbf_svm": f"{top} + RBF SVM", "select_mlp": f"{top} + MLP",
            "rbf_svr": "RBF SVR", "mlp": "MLP"}.get(name, name)


def join_names(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def class_styles(labels: list[str], counts: dict | None = None) -> dict:
    """Fixed colour/marker per class in sorted label order; >6 classes fold the tail."""
    labels = [str(v) for v in labels]
    if len(labels) <= len(vs.CATEGORICAL):
        return {lab: {**vs.category_style(i), "legend": lab} for i, lab in enumerate(labels)}
    ranked = sorted(labels, key=lambda lab: -(counts or {}).get(lab, 0))
    keep = set(ranked[:5])
    styles, slot = {}, 0
    for lab in labels:
        if lab in keep:
            styles[lab] = {**vs.category_style(slot), "legend": lab}
            slot += 1
        else:
            styles[lab] = {"color": vs.NEUTRAL_LIGHT, "marker": ".", "legend": f"Other ({len(labels) - 5} classes)"}
    return styles


def legend_below(fig, styles: dict, ncol: int | None = None) -> None:
    seen, handles = set(), []
    for style in styles.values():
        if style["legend"] in seen:
            continue
        seen.add(style["legend"])
        handles.append(plt.Line2D([], [], linestyle="none", marker=style["marker"], markersize=6,
                                  markerfacecolor=style["color"], markeredgecolor=vs.SURFACE,
                                  markeredgewidth=0.4, label=style["legend"]))
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol or len(handles),
               columnspacing=1.2, handletextpad=0.2)


# ---------------------------------------------------------------- run figures

def load_run(run_dir: Path) -> dict:
    evidence = json.loads((run_dir / "evidence.json").read_text(encoding="utf-8"))
    rows = []
    for exp in evidence.get("experiments", []):
        scores = [f["score"] for f in exp.get("folds", []) if f.get("score") is not None]
        if exp.get("status") != "complete" or not scores:
            continue
        rows.append({"name": exp["name"], "label": exp.get("label") or model_label(exp["name"], exp.get("parameters", {}),
                                                                               evidence["task"], exp.get("folds")),
                     "scores": scores, "mean": float(np.mean(scores)), "oof_file": exp.get("oof_file")})
    return {"evidence": evidence, "models": rows}


def fig_models(run: dict, out: Path, titles: bool) -> dict | None:
    ev, rows = run["evidence"], run["models"]
    metric, task = ev["metric"], ev["task"]
    lower = ev.get("metric_direction") == "minimize"
    baseline = next((r for r in rows if r["name"] == "dummy"), None)
    cands = [r for r in rows if r["name"] != "dummy"]
    if not cands:
        return None
    cands.sort(key=lambda r: r["mean"], reverse=not lower)
    best = cands[0]["mean"]
    tied = [r["label"] for r in cands if math.isclose(r["mean"], best, rel_tol=0, abs_tol=1e-12)]
    selected = ev.get("selection", {}).get("model")
    metric_name = METRIC_NAMES.get(metric, metric)
    design = cv_design(ev)

    fig, ax = plt.subplots(figsize=(vs.FULL_WIDTH * 0.82, 0.5 * len(cands) + 0.9))
    vs.soften(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    rng = np.random.default_rng(0)
    for i, r in enumerate(cands):
        y = len(cands) - 1 - i
        emph = r["name"] == selected
        color = vs.ACCENT if emph else vs.INK_2
        jitter = rng.uniform(-0.09, 0.09, len(r["scores"]))
        ax.plot([min(r["scores"]), max(r["scores"])], [y, y], color=vs.ACCENT if emph else vs.NEUTRAL,
                linewidth=1.0, solid_capstyle="round", zorder=1)
        ax.scatter(r["scores"], y + jitter, s=11, facecolor=vs.SURFACE, edgecolor=vs.ACCENT if emph else vs.NEUTRAL,
                   linewidth=0.8, zorder=2)
        ax.scatter([r["mean"]], [y], s=30, color=color, edgecolor=vs.SURFACE, linewidth=0.6, zorder=3)
    ax.set_yticks(range(len(cands)))
    ax.set_yticklabels([r["label"] for r in reversed(cands)])
    for tick, r in zip(ax.get_yticklabels(), reversed(cands)):
        tick.set_color(vs.ACCENT if r["name"] == selected else vs.INK_2)
        tick.set_fontsize(vs.FS["label"])
    values = [s for r in cands for s in r["scores"]]
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * 0.12, 0.002 if metric in BOUNDED_METRICS else abs(hi) * 0.02 + 1e-9)
    left, right = lo - pad, hi + pad
    ax.set_xlim(left, right)
    ax.set_ylim(-0.6, len(cands) - 0.4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    span = max(right - left, 1e-12)
    ax.xaxis.set_major_formatter(FormatStrFormatter(f"%.{min(max(2, 1 - math.floor(math.log10(span))), 4)}f"))
    limit = 1.0 if metric in BOUNDED_METRICS else math.inf
    ax.set_xticks([t for t in ax.get_xticks() if left <= t <= min(right, limit) + 1e-12])
    ax.spines["bottom"].set_bounds(left, min(right, limit))
    values_x = blended_transform_factory(ax.transAxes, ax.transData)
    for i, r in enumerate(cands):
        y = len(cands) - 1 - i
        ax.text(1.04, y, f"{r['mean']:.3f}", transform=values_x, ha="left", va="center", fontsize=vs.FS["body"],
                color=vs.ACCENT if r["name"] == selected else vs.INK_2,
                fontweight="medium" if r["name"] == selected else "normal")
    ax.text(1.04, len(cands) - 0.45, "Mean", transform=values_x, ha="left", va="bottom", fontsize=vs.FS["note"],
            color=vs.INK_3)
    ax.set_xlabel(f"{metric_name} per validation fold" + (" (lower is better)" if lower else ""))
    note = f"majority-class baseline = {baseline['mean']:.3f}" if baseline and task == "classification" else (
        f"mean baseline = {baseline['mean']:.3f}" if baseline else "no baseline")
    in_text = metric_name.lower() if metric in ("accuracy", "balanced_accuracy", "log_loss") else metric_name
    title = (f"{join_names(tied)} {'attain' if len(tied) > 1 else 'attains'} the best "
             f"cross-validated {in_text} ({best:.3f})")
    subtitle = (f"{design} selection CV, n = {ev['n_rows']:,} · open dots: folds, "
                f"filled: mean · {note}")
    if titles:
        vs.header(fig, title, subtitle)
    caption = (f"Cross-validated {in_text} of each candidate pipeline on identical {design} "
               f"splits (n = {ev['n_rows']:,}). Open circles are individual folds, filled circles the mean; the line spans "
               f"the fold range. The highlighted pipeline ({next((r['label'] for r in cands if r['name'] == selected), selected)}) was "
               f"selected by mean fold score with the declared tie-break. These are selection-CV estimates used to choose "
               f"the model, not an independent test score; fold spread is not a confidence interval. {note.capitalize()}.")
    return {"fig": fig, "stem": "fig_model_comparison", "title": title, "caption": caption}


def fig_confusion(run: dict, run_dir: Path, out: Path, titles: bool) -> dict | None:
    ev = run["evidence"]
    selected = ev.get("selection", {}).get("model")
    row = next((r for r in run["models"] if r["name"] == selected), None)
    if row is None or not row["oof_file"]:
        return None
    oof = pd.read_csv(run_dir / row["oof_file"], dtype={"y_true": str, "prediction": str} if ev["task"] == "classification" else None)
    oof = oof[oof["fold"] >= 0]
    if ev["task"] != "classification":
        return fig_regression(ev, row, oof, titles)
    classes = [str(c) for c in ev["class_order"]]
    fig, n, errors, worst, off = draw_confusion(oof, classes)
    label = row["label"]
    if errors == 0:
        title = f"All {n:,} out-of-fold predictions are correct"
    else:
        title = (f"{errors} of {n:,} out-of-fold predictions are wrong; most often "
                 f"{classes[worst[0]]} predicted as {classes[worst[1]]} ({off[worst]})")
    subtitle = f"{label} · {cv_design(ev)} out-of-fold · numbers: counts, shade: row share"
    if titles:
        vs.header(fig, title, subtitle)
    caption = (f"Out-of-fold confusion matrix of the selected pipeline ({label}). {oof_note(ev)}; "
               f"counts are pooled over folds (n = {n:,}, {errors} errors). "
               f"Shading is the row-normalised share and the right column gives per-class recall. This is a selection-CV "
               f"diagnostic, not an independent test.")
    return {"fig": fig, "stem": "fig_confusion_matrix", "title": title, "caption": caption}


def draw_confusion(frame: pd.DataFrame, classes: list[str]):
    """Counts heatmap (row-share shading, per-class recall column). Returns fig, n, errors, worst cell, off-diagonal."""
    index = {c: i for i, c in enumerate(classes)}
    counts = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(frame["y_true"], frame["prediction"]):
        counts[index[t], index[p]] += 1
    share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    n, errors = int(counts.sum()), int(counts.sum() - np.trace(counts))
    k = len(classes)
    side = min(0.42 * k + 1.0, vs.FULL_WIDTH * 0.62)
    fig, ax = plt.subplots(figsize=(side + 0.9, side))
    image = ax.imshow(share, cmap=vs.SEQUENTIAL, vmin=0, vmax=1)
    ax.set_xticks(range(k), classes, rotation=0 if max(map(len, classes)) <= 5 else 45,
                  ha="center" if max(map(len, classes)) <= 5 else "right")
    ax.set_yticks(range(k), classes)
    ax.tick_params(length=0, labelcolor=vs.INK_2, labelsize=vs.FS["note"])
    ax.set_xticks(np.arange(-0.5, k), minor=True)
    ax.set_yticks(np.arange(-0.5, k), minor=True)
    ax.grid(which="minor", color=vs.SURFACE, linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    font = vs.FS["body"] if k <= 8 else vs.FS["tiny"]
    for i in range(k):
        for j in range(k):
            if counts[i, j]:
                ax.text(j, i, f"{counts[i, j]}", ha="center", va="center", fontsize=font,
                        color="white" if share[i, j] > 0.72 else vs.INK,
                        fontweight="medium" if i == j else "normal")
        recall = counts[i, i] / max(counts[i].sum(), 1)
        ax.text(k - 0.5 + 0.18, i, f"{recall:.1%}", ha="left", va="center", fontsize=vs.FS["note"], color=vs.INK_2,
                transform=ax.transData, clip_on=False)
    ax.text(k - 0.5 + 0.18, -0.75, "Recall", ha="left", va="bottom", fontsize=vs.FS["note"], color=vs.INK_3, clip_on=False)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.16, ticks=[0, 0.5, 1])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=vs.FS["tiny"])
    bar.set_label("Share of true class", fontsize=vs.FS["note"], color=vs.INK_2)
    off = counts.copy()
    np.fill_diagonal(off, 0)
    worst = np.unravel_index(off.argmax(), off.shape)
    return fig, n, errors, worst, off


def fig_regression(ev: dict, row: dict, oof: pd.DataFrame, titles: bool) -> dict:
    y, p = oof["y_true"].astype(float).to_numpy(), oof["prediction"].astype(float).to_numpy()
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    r2 = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    fig, ax = plt.subplots(figsize=(vs.HALF_WIDTH, vs.HALF_WIDTH))
    vs.soften(ax)
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    ax.plot([lo, hi], [lo, hi], color=vs.NEUTRAL, linewidth=0.8, zorder=1)
    ax.scatter(y, p, s=8, color=vs.ACCENT, alpha=0.6, edgecolor=vs.SURFACE, linewidth=0.3, rasterized=True, zorder=2)
    ax.set_aspect("equal")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Out-of-fold prediction")
    label = row["label"]
    title = (f"Out-of-fold predictions track the target (R² = {r2:.3f})" if r2 >= 0.7 else
             f"Out-of-fold predictions explain {max(r2, 0):.0%} of target variance (R² = {r2:.3f})")
    if titles:
        vs.header(fig, title, f"{label} · RMSE = {rmse:.3g} · grey line: perfect agreement")
    caption = (f"Observed target versus pooled out-of-fold prediction of {label} (n = {len(y):,}, {cv_design(ev)}); "
               f"the grey line is y = x. {oof_note(ev)}. Pooled OOF R² = {r2:.3f}, RMSE = {rmse:.3g}; selection-CV "
               f"estimates, not an independent test.")
    return {"fig": fig, "stem": "fig_oof_regression", "title": title, "caption": caption}


def fig_holdout(run: dict, run_dir: Path, titles: bool) -> dict | None:
    """Single evaluation of the selected pipeline on the untouched holdout (evaluate_models.py with holdout)."""
    ev = run["evidence"]
    h = ev.get("holdout") or {}
    if h.get("status") != "complete" or not (run_dir / h["predictions_file"]).exists():
        return None
    row = next((r for r in run["models"] if r["name"] == h["model"]), None)
    label = row["label"] if row else h["model"]
    metric_name = METRIC_NAMES.get(h["metric"], h["metric"])
    setting = (f"fitted on {h['n_development']:,} development rows, scored once on {h['n_holdout']:,} "
               f"{h['method']} holdout rows")
    if ev["task"] == "classification":
        frame = pd.read_csv(run_dir / h["predictions_file"], dtype={"y_true": str, "prediction": str})
        classes = [str(c) for c in ev["class_order"]]
        fig, n, errors, worst, off = draw_confusion(frame, classes)
        title = (f"All {n:,} holdout predictions are correct" if errors == 0 else
                 f"{n - errors:,} of {n:,} holdout predictions are correct; most often {classes[worst[0]]} "
                 f"predicted as {classes[worst[1]]} ({off[worst]})")
    else:
        frame = pd.read_csv(run_dir / h["predictions_file"])
        y, p = frame["y_true"].astype(float).to_numpy(), frame["prediction"].astype(float).to_numpy()
        fig, ax = plt.subplots(figsize=(vs.HALF_WIDTH, vs.HALF_WIDTH))
        vs.soften(ax)
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color=vs.NEUTRAL, linewidth=0.8, zorder=1)
        ax.scatter(y, p, s=10, color=vs.ACCENT, alpha=0.7, edgecolor=vs.SURFACE, linewidth=0.3, zorder=2)
        ax.set_aspect("equal")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Holdout prediction")
        title = f"Holdout {metric_name} of the selected pipeline is {h['score']:.3f}"
    if titles:
        vs.header(fig, title, f"{label} · {metric_name} = {h['score']:.3f} · {setting}")
    caption = (f"Untouched holdout evaluation of the selected pipeline ({label}): {setting}; {metric_name} = "
               f"{h['score']:.3f}. The holdout was split off before any fold was drawn and was not used for fitting, "
               f"preprocessing or model choice, so this is the report's independent estimate; with "
               f"{h['n_holdout']:,} rows it is still a single small-sample estimate.")
    return {"fig": fig, "stem": "fig_holdout", "title": title, "caption": caption}


def fig_importance(run: dict, run_dir: Path, titles: bool, top: int = 20) -> dict | None:
    """Fold-averaged feature weights of the selected pipeline (or the best candidate that has weights)."""
    ev = run["evidence"]
    exps = [e for e in ev["experiments"] if e.get("status") == "complete" and e.get("importance_file")
            and (run_dir / e["importance_file"]).exists()]
    if not exps:
        return None
    selected = ev.get("selection", {}).get("model")
    lower = ev.get("metric_direction") == "minimize"
    exp = next((e for e in exps if e["name"] == selected), None)
    fallback = exp is None
    if fallback:
        exp = sorted(exps, key=lambda e: e["fold_score_mean"], reverse=not lower)[0]
    label = next((r["label"] for r in run["models"] if r["name"] == exp["name"]), exp["name"])
    frame = pd.read_csv(run_dir / exp["importance_file"]).head(top)
    weight_cols = [c for c in frame if c.startswith("weight_")]
    kind = exp["importance_kind"]
    what = {"standardized_coefficient": "fold-averaged coefficient on standardised features",
            "pca_back_projected_coefficient": "fold-averaged coefficient back-projected through PCA",
            "impurity_importance": "fold-averaged impurity importance"}[kind]
    n_folds = len(exp["folds"])
    freq = frame["selection_frequency"].to_numpy()
    # Selector pipelines always show the fold count: 5/5 is itself the stability evidence.
    has_selector = any("selected_features" in f.get("fitted_preprocessing", {}) for f in exp["folds"])
    show_freq = has_selector or not np.allclose(freq, 1.0)
    names = frame["feature"].astype(str).tolist()
    height = 0.17 * len(frame) + 1.0
    if len(weight_cols) > 1:
        classes = [c.removeprefix("weight_") for c in weight_cols]
        values = frame[weight_cols].to_numpy()
        bound = float(np.abs(values).max()) or 1.0
        fig, ax = plt.subplots(figsize=(min(vs.FULL_WIDTH, 0.55 * len(classes) + 2.6), height))
        image = ax.imshow(values, cmap=vs.DIVERGING, vmin=-bound, vmax=bound, aspect="auto")
        ax.set_xticks(range(len(classes)), classes)
        ax.set_yticks(range(len(names)), names)
        ax.tick_params(length=0, labelsize=vs.FS["tiny"], labelcolor=vs.INK_2)
        ax.set_xticks(np.arange(-0.5, len(classes)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(names)), minor=True)
        ax.grid(which="minor", color=vs.SURFACE, linewidth=1.2)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.18 if show_freq else 0.04)
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=0, labelsize=vs.FS["tiny"])
        bar.set_label("Weight (sign = direction for that class)", fontsize=vs.FS["note"], color=vs.INK_2)
        strongest = np.unravel_index(np.abs(values).argmax(), values.shape)
        finding = (f"{names[strongest[0]]} carries the largest weight "
                   f"({'raises' if values[strongest] > 0 else 'lowers'} the {classes[strongest[1]]} score)")
        right = len(classes) - 0.5 + 0.15
    else:
        values = frame[weight_cols[0]].to_numpy()
        fig, ax = plt.subplots(figsize=(vs.FULL_WIDTH * 0.75, height))
        vs.soften(ax, grid_axis="x")
        ax.spines["left"].set_visible(False)
        ypos = np.arange(len(names))[::-1]
        ax.barh(ypos, values, color=[vs.ACCENT if v >= 0 else vs.CATEGORICAL[1] for v in values], height=0.7)
        ax.axvline(0, color=vs.RULE, linewidth=0.8)
        ax.set_yticks(ypos, names)
        ax.tick_params(axis="y", length=0, labelsize=vs.FS["tiny"])
        ax.set_xlabel("Weight" if kind != "impurity_importance" else "Impurity importance")
        ax.set_ylim(-0.6, len(names) - 0.4)
        finding = f"{names[0]} carries the largest weight"
        right = None
    if show_freq:
        transform = ax.transData if right is not None else blended_transform_factory(ax.transAxes, ax.transData)
        x = right if right is not None else 1.02
        rows_y = range(len(names)) if right is not None else np.arange(len(names))[::-1]
        for yv, f in zip(rows_y, freq):
            ax.text(x, yv, f"{int(round(f * n_folds))}/{n_folds}", ha="left", va="center", fontsize=vs.FS["tiny"],
                    color=vs.INK_2, transform=transform, clip_on=False)
        ax.text(x, -0.9 if right is not None else len(names) - 0.2, "Folds", ha="left", va="bottom",
                fontsize=vs.FS["tiny"], color=vs.INK_3, transform=transform, clip_on=False)
    title = finding
    subtitle = f"{label} · top {len(frame)} by |weight| · {what}" + (" · right: folds selecting the feature" if show_freq else "")
    if titles:
        vs.header(fig, title, subtitle)
    caption = (f"Top {len(frame)} features of {label}{' (the selected pipeline has no feature weights; this is the best candidate that has them)' if fallback else ''}, "
               f"ranked by the largest absolute {what} over the {n_folds} fold models. "
               + ("Weights count as 0 in folds whose selector dropped the feature; the right column gives how many "
                  "folds selected it, a stability check. " if show_freq else "")
               + ("Back-projected weights spread over many correlated features and are not per-feature effects. "
                  if kind == "pca_back_projected_coefficient" else "")
               + "Weights are associations under this model; correlated features share weight, and anonymous feature "
                 "names carry no biological identity.")
    return {"fig": fig, "stem": "fig_feature_importance", "title": title, "caption": caption}


def fig_gap(run: dict, titles: bool) -> dict | None:
    """Mean training-fold score versus mean validation-fold score per candidate (overfitting diagnostic)."""
    ev = run["evidence"]
    rows = [(next((r["label"] for r in run["models"] if r["name"] == e["name"]), e["name"]), e["train_score_mean"],
             e["fold_score_mean"]) for e in ev["experiments"]
            if e.get("status") == "complete" and e["name"] != "dummy" and e.get("train_score_mean") is not None]
    if not rows:
        return None
    lower = ev.get("metric_direction") == "minimize"
    metric_name = METRIC_NAMES.get(ev["metric"], ev["metric"])
    rows.sort(key=lambda r: r[2], reverse=not lower)
    gaps = [(r[2] - r[1]) if lower else (r[1] - r[2]) for r in rows]  # positive = train looks better
    fig, ax = plt.subplots(figsize=(vs.FULL_WIDTH * 0.82, 0.42 * len(rows) + 1.0))
    vs.soften(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for i, (label, tr, va) in enumerate(rows):
        yv = len(rows) - 1 - i
        ax.plot([tr, va], [yv, yv], color=vs.NEUTRAL, linewidth=1.0, zorder=1)
        ax.scatter([tr], [yv], s=26, facecolor=vs.SURFACE, edgecolor=vs.INK_2, linewidth=0.9, zorder=2)
        ax.scatter([va], [yv], s=30, color=vs.ACCENT, edgecolor=vs.SURFACE, linewidth=0.6, zorder=3)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    values_x = blended_transform_factory(ax.transAxes, ax.transData)
    for i, g in enumerate(gaps):
        ax.text(1.03, len(rows) - 1 - i, f"{g:+.3f}", transform=values_x, ha="left", va="center",
                fontsize=vs.FS["note"], color=vs.INK_2)
    ax.text(1.03, len(rows) - 0.45, "Gap", transform=values_x, ha="left", va="bottom", fontsize=vs.FS["note"], color=vs.INK_3)
    ax.set_xlabel(f"Mean {metric_name} over folds" + (" (lower is better)" if lower else ""))
    worst = int(np.argmax(gaps))
    title = (f"{rows[worst][0]} shows the largest train–validation gap ({gaps[worst]:+.3f})" if gaps[worst] > 0.01 else
             f"Training and validation {metric_name} agree within {max(abs(g) for g in gaps):.3f} for every pipeline")
    if titles:
        vs.header(fig, title, f"{cv_design(ev)} · open: training rows of each fold, filled: validation rows · "
                              f"gap = train minus validation, sign-adjusted so positive means optimistic training fit")
    caption = (f"Mean {metric_name} of each pipeline on the training rows of each fold (open) and on the held-out "
               f"validation rows (filled), {cv_design(ev)}. A large positive gap indicates that the pipeline fits the "
               f"training rows much better than unseen rows (overfitting); with p ≫ n a perfect training score is "
               f"expected for flexible models and is not itself a problem. The training score is a diagnostic only "
               f"and played no part in model selection.")
    return {"fig": fig, "stem": "fig_train_validation_gap", "title": title, "caption": caption}


# ---------------------------------------------------------- structure figures

def fig_variance(sdir: Path, titles: bool) -> dict | None:
    """Scree (per-component share) and cumulative explained variance of the clustering PCA."""
    path = sdir / "clustering_variance.csv"
    if not path.exists():
        return None
    var = pd.read_csv(path)
    share, cum = var["explained_variance_ratio"].to_numpy(), var["cumulative_explained_variance_ratio"].to_numpy()
    k = len(share)
    fig, ax = plt.subplots(figsize=(vs.FULL_WIDTH * 0.82, 2.5))
    vs.soften(ax, grid_axis="y")
    x = np.arange(1, k + 1)
    ax.bar(x, share, color=vs.NEUTRAL, width=0.75, label="Single component")
    ax.plot(x, cum, color=vs.ACCENT, linewidth=1.4, marker="o", markersize=2.5, label="Cumulative")
    marks = []
    for level in (0.5, 0.8, 0.9):
        hit = np.nonzero(cum >= level)[0]
        if hit.size:
            c = int(hit[0]) + 1
            marks.append(f"{level:.0%} at PC{c}")
            ax.axhline(level, color=vs.GRID, linewidth=0.8, zorder=0)
            ax.annotate(f"{level:.0%}: {c} PCs", (c, cum[c - 1]), xytext=(6, -10), textcoords="offset points",
                        fontsize=vs.FS["tiny"], color=vs.INK_2)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0.3, k + 0.7)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Share of variance")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.08), frameon=False, fontsize=vs.FS["note"])
    title = (f"The first {k} principal components retain {cum[-1]:.0%} of the variance; PC1 alone {share[0]:.0%}")
    if titles:
        vs.header(fig, title, "Standardised, median-imputed, non-constant features · randomized PCA used for clustering")
    caption = (f"Explained variance of the {k} principal components used by the unsupervised analysis: grey bars give "
               f"each component's share, the line the cumulative share ({cum[-1]:.1%} after {k} components"
               + (f"; {', '.join(marks)}" if marks else "; no 50% threshold within the retained components")
               + "). Variance refers to the processed feature space; it is not a share of predictive or biological "
                 "information, and the component cap was fixed in advance rather than chosen from this curve.")
    return {"fig": fig, "stem": "fig_pca_variance", "title": title, "caption": caption}


def fig_cluster_robustness(sdir: Path, titles: bool) -> dict | None:
    """Silhouette over k for k-means and Ward, and the two methods' agreement with each other and the labels."""
    summary_path, sweep_path = sdir / "clustering_summary.json", sdir / "clustering_k_sweep.csv"
    if not summary_path.exists() or not sweep_path.exists():
        return None
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    sweep = pd.read_csv(sweep_path)
    ward = s.get("ward", {})
    k_used = s["n_clusters_requested"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(vs.FULL_WIDTH, 2.4), gridspec_kw={"width_ratios": [1.25, 1]})
    vs.soften(a, grid_axis="y")
    a.plot(sweep["k"], sweep["kmeans_silhouette"], color=vs.ACCENT, marker="o", markersize=3, linewidth=1.3, label="k-means")
    if sweep["ward_silhouette"].notna().any():
        a.plot(sweep["k"], sweep["ward_silhouette"], color=vs.CATEGORICAL[1], marker="s", markersize=3, linewidth=1.3,
               label="Ward")
    a.axvline(k_used, color=vs.RULE, linewidth=0.8, linestyle="--")
    a.annotate(f"k used = {k_used}", (k_used, a.get_ylim()[0]), xytext=(3, 4), textcoords="offset points",
               fontsize=vs.FS["tiny"], color=vs.INK_2)
    a.set_xlabel("Number of clusters k")
    a.set_ylabel("Silhouette")
    a.set_xticks(sweep["k"])
    a.legend(frameon=False, fontsize=vs.FS["note"], loc="lower right")
    vs.panel_label(a, "a")
    vs.soften(b, grid_axis="x")
    metrics = [("ARI vs labels", s.get("adjusted_rand_index"), ward.get("adjusted_rand_index")),
               ("NMI vs labels", s.get("normalized_mutual_information"), ward.get("normalized_mutual_information")),
               ("Silhouette", s.get("silhouette"), ward.get("silhouette"))]
    ypos = np.arange(len(metrics))[::-1]
    for yv, (name, km, wd) in zip(ypos, metrics):
        if km is not None:
            b.barh(yv + 0.18, km, height=0.34, color=vs.ACCENT)
            b.text(km + 0.02, yv + 0.18, f"{km:.2f}", va="center", fontsize=vs.FS["tiny"], color=vs.INK_2)
        if wd is not None:
            b.barh(yv - 0.18, wd, height=0.34, color=vs.CATEGORICAL[1])
            b.text(wd + 0.02, yv - 0.18, f"{wd:.2f}", va="center", fontsize=vs.FS["tiny"], color=vs.INK_2)
    b.set_yticks(ypos, [m[0] for m in metrics])
    b.tick_params(axis="y", length=0)
    b.spines["left"].set_visible(False)
    b.set_xlim(0, 1.12)
    b.set_xlabel(f"Value at k = {k_used}")
    vs.panel_label(b, "b")
    best_km = s.get("k_sweep", {}).get("best_k_kmeans_silhouette")
    agree = ward.get("agreement_with_kmeans_ari")
    title = (f"Silhouette favours k = {best_km}" + (f"; Ward and k-means agree at k = {k_used} (ARI {agree:.2f})"
                                                   if agree is not None else ""))
    if titles:
        vs.header(fig, title, "Same PCA space · (a) silhouette by k, computed without labels · "
                              "(b) external agreement with labels after fitting, k-means (clay) vs Ward (blue)")
    caption = (f"Robustness of the unsupervised structure. (a) Silhouette of k-means and Ward clustering for k = "
               f"{int(sweep['k'].min())}–{int(sweep['k'].max())} in the same PCA space, computed without labels; the dashed line "
               f"marks the k used ({k_used}, {s.get('k_source', '').replace('_', ' ')}). (b) Agreement of each method with "
               f"the labels (ARI, NMI, computed after fitting) and silhouette at that k"
               + (f"; the two methods agree with each other at ARI {agree:.2f}" if agree is not None else "")
               + ". A silhouette maximum is one internal criterion, not proof of the number of groups.")
    return {"fig": fig, "stem": "fig_cluster_robustness", "title": title, "caption": caption}


def match_clusters(table: pd.DataFrame) -> dict:
    """Map each cluster to a class by maximum overlap (display only)."""
    rows, cols = linear_sum_assignment(-table.to_numpy())
    return {table.columns[c]: table.index[r] for r, c in zip(rows, cols)}


def fig_structure(sdir: Path, titles: bool) -> dict | None:
    summary_path, scores_path = sdir / "clustering_summary.json", sdir / "clustering_scores.csv"
    if not summary_path.exists() or not scores_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scores = pd.read_csv(scores_path, dtype={"observed_label": str})
    table = pd.crosstab(scores["observed_label"], scores["cluster"])
    labels = sorted(table.index.tolist())
    table = table.loc[labels]
    styles = class_styles(labels, scores["observed_label"].value_counts().to_dict())
    mapping = match_clusters(table)
    order = sorted(table.columns, key=lambda c: labels.index(mapping[c]) if c in mapping else len(labels) + c)
    ratio = summary["pca_explained_variance_ratio"]

    fig = plt.figure(figsize=(vs.FULL_WIDTH, 2.75))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9])
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    for ax, key, name in ((axes[0], "cluster", "k-means clusters"),
                          (axes[1], "observed_label", "Reference labels")):
        vs.soften(ax)
        groups = order if key == "cluster" else labels
        for g in groups:
            sub = scores[scores[key] == g]
            if key == "cluster":
                style = styles[mapping[g]] if g in mapping else {"color": vs.NEUTRAL, "marker": "x"}
            else:
                style = styles[g]
            ax.scatter(sub["PC1"], sub["PC2"], s=10, marker=style["marker"], color=style["color"], alpha=0.8,
                       edgecolor=vs.SURFACE, linewidth=0.25, rasterized=True)
        vs.panel_title(ax, name)
        ax.set_xlabel(f"PC1 ({ratio[0]:.1%} of variance)")
        ax.set_ylabel(f"PC2 ({ratio[1]:.1%} of variance)")
    axes[1].sharex(axes[0])
    axes[1].sharey(axes[0])
    ax = axes[2]
    shown = table[order]
    share = shown.to_numpy() / shown.to_numpy().sum(axis=1, keepdims=True)
    ax.imshow(share, cmap=vs.SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order)), [f"C{c}" for c in order])
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0, labelcolor=vs.INK_2)
    ax.set_xticks(np.arange(-0.5, len(order)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
    ax.grid(which="minor", color=vs.SURFACE, linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            if shown.iat[i, j]:
                ax.text(j, i, str(shown.iat[i, j]), ha="center", va="center", fontsize=vs.FS["tiny"],
                        color="white" if share[i, j] > 0.72 else vs.INK)
    vs.panel_title(ax, "Label × cluster")
    ax.set_xlabel("Cluster (matched order)")
    for a, letter in zip(axes, "abc"):
        vs.panel_label(a, letter)
    legend_below(fig, styles)
    ari, nmi = summary["adjusted_rand_index"], summary["normalized_mutual_information"]
    degree = ("closely recovers" if ari >= 0.9 else "largely recovers" if ari >= 0.6
              else "partly recovers" if ari >= 0.3 else "weakly reflects")
    title = f"Label-free clustering {degree} the reference classes (ARI = {ari:.2f}, NMI = {nmi:.2f})"
    k_note = "k fixed in advance" if summary.get("k_source", "").startswith("explicit") else "k chosen from data"
    subtitle = (f"Standardised features, then PCA ({summary['pca_components_used']} PCs, "
                f"{summary['pca_cumulative_explained_variance']:.0%} of variance), then k-means (k = "
                f"{summary['n_clusters_observed']}, {k_note}); labels used only after fitting")
    if titles:
        vs.header(fig, title, subtitle)
    caption = (f"Unsupervised structure. (a) k-means clusters fitted on {summary['pca_components_used']} principal "
               f"components of standardised features without labels, shown on PC1–PC2; cluster colours are matched "
               f"to their best-overlapping reference label for display only. (b) Reference labels overlaid on the same "
               f"projection. (c) Label × cluster counts. ARI = {ari:.3f}, NMI = {nmi:.3f}, silhouette = "
               f"{summary['silhouette']:.3f}. The number of clusters was {'fixed in advance' if k_note.startswith('k fixed') else 'chosen from the data'}. A two-PC projection is descriptive and does not "
               f"validate a classifier; cluster indices have no intrinsic class meaning.")
    return {"fig": fig, "stem": "fig_unsupervised_structure", "title": title, "caption": caption}


def load_associations(sdir: Path) -> tuple[pd.DataFrame, dict] | None:
    path, spath = sdir / "associations.csv", sdir / "associations_summary.json"
    if not path.exists() or not spath.exists():
        return None
    return pd.read_csv(path), json.loads(spath.read_text(encoding="utf-8"))


def fig_associations(sdir: Path, titles: bool, top: int = 20) -> dict | None:
    loaded = load_associations(sdir)
    if loaded is None:
        return None
    assoc, summary = loaded
    alpha = summary["alpha"]
    classes = [c["label"] for c in summary["class_order"]]
    tested = assoc[assoc["test_status"] == "finite_test"].sort_values("F", ascending=False).reset_index(drop=True)
    m, sig = summary["tested_family_size"], summary["q_below_alpha"]

    fig = plt.figure(figsize=(vs.FULL_WIDTH, 3.15))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(grid[0, 0])
    vs.soften(ax, grid_axis="y")
    rank = np.arange(1, len(tested) + 1)
    eta = tested["eta_squared"].to_numpy()
    passed = tested["q_bh"].to_numpy() < alpha
    ax.fill_between(rank, 0, eta, where=passed, color=vs.ACCENT, alpha=0.12, linewidth=0, step="mid")
    ax.plot(rank, eta, color=vs.ACCENT, linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlim(1, len(rank))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Feature rank by ANOVA F (log scale)")
    ax.set_ylabel("Effect size η²")
    if passed.any():
        cut = int(np.flatnonzero(passed).max()) + 1
        if cut < 0.9 * len(rank):
            ax.axvline(cut, color=vs.INK_3, linewidth=0.6)
        ax.text(0.04, 0.08, f"Shaded: {sig:,} of {m:,}\nfeatures with BH q < {alpha:g}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=vs.FS["note"], color=vs.INK_2, linespacing=1.3)
    ax.annotate(f"{tested.at[0, 'feature']}  (η² = {eta[0]:.2f})", xy=(rank[0], eta[0]), xytext=(10, 10),
                textcoords="offset points", fontsize=vs.FS["tiny"], color=vs.INK_2, va="bottom",
                arrowprops=dict(arrowstyle="-", color=vs.INK_3, linewidth=0.5, shrinkA=0, shrinkB=2))
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    vs.panel_label(ax, "a")

    ax = fig.add_subplot(grid[0, 1])
    head = tested.head(top)
    means = head[[f"mean_class_{i}" for i in range(len(classes))]].to_numpy(dtype=float)
    centred = means - means.mean(axis=1, keepdims=True)
    spread = means.std(axis=1, keepdims=True)
    z = np.divide(centred, spread, out=np.zeros_like(centred), where=spread > 0)
    lim = max(2.0, float(np.ceil(np.abs(z).max())))
    image = ax.imshow(z, cmap=vs.DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(head)), head["feature"])
    ax.tick_params(length=0, labelsize=vs.FS["tiny"], labelcolor=vs.INK_2)
    ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(classes)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(head)), minor=True)
    ax.grid(which="minor", color=vs.SURFACE, linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    for i, value in enumerate(head["eta_squared"]):
        ax.text(len(classes) - 0.5 + 0.15, i, f"{value:.2f}", va="center", fontsize=vs.FS["tiny"], color=vs.INK_3, clip_on=False)
    ax.text(len(classes) - 0.5 + 0.15, -0.6, "η²", va="bottom", fontsize=vs.FS["tiny"], color=vs.INK_3, clip_on=False)
    bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.12, ticks=[-lim, 0, lim])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=vs.FS["tiny"])
    bar.set_label("Class mean, z-scored across classes", fontsize=vs.FS["note"], color=vs.INK_2)
    vs.panel_label(ax, "b")
    title = f"{sig:,} of {m:,} tested features differ across classes (BH q < {alpha:g})"
    subtitle = ("Exploratory one-way ANOVA on all labelled samples, BH over the full tested family; "
                "an association screen, not predictive validation")
    if titles:
        vs.header(fig, title, subtitle)
    caption = (f"Exploratory class association. (a) Effect size η² of every tested feature ranked by one-way "
               f"ANOVA F; the shaded region marks features with Benjamini–Hochberg q < {alpha:g} over the complete "
               f"family of {m:,} tests ({summary.get('p_numerically_zero_with_finite_F', 0)} p-values underflow to zero). "
               f"(b) Class-mean profiles of the top {len(head)} features, z-scored across the {len(classes)} class means; "
               f"right column gives η². Omnibus ANOVA indicates that at least one class mean differs; it does not "
               f"identify which class, establish biological function, or validate a predictor. Anonymous feature names are "
               f"reported as given.")
    return {"fig": fig, "stem": "fig_class_associations", "title": title, "caption": caption}


def load_raw(config_path: Path | None):
    if config_path is None:
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from exam_io import load_config, load_data
    data = load_data(load_config(str(config_path)), include_test=False)
    return data.X, np.asarray(data.y).astype(str)


def fig_distributions(sdir: Path, raw, titles: bool, top: int = 6) -> dict | None:
    loaded = load_associations(sdir)
    if loaded is None or raw is None:
        return None
    assoc, summary = loaded
    X, y = raw
    tested = assoc[assoc["test_status"] == "finite_test"].sort_values("F", ascending=False).head(top)
    classes = [c["label"] for c in summary["class_order"]]
    styles = class_styles(classes, pd.Series(y).value_counts().to_dict())
    ncol = 2
    nrow = math.ceil(len(tested) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(vs.FULL_WIDTH, 1.75 * nrow + 0.4), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (_, row) in zip(axes.flat, tested.iterrows()):
        vs.soften(ax, grid_axis="y")
        values = X[row["feature"]].to_numpy(dtype=float)
        for i, cls in enumerate(classes):
            v = values[(y == cls) & np.isfinite(values)]
            st = styles[cls]
            ax.scatter(i + rng.uniform(-0.22, 0.22, len(v)), v, s=2.2, color=st["color"], alpha=0.35,
                       linewidth=0, rasterized=True, zorder=1)
            if len(v):
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                ax.plot([i - 0.28, i + 0.28], [med, med], color=vs.INK, linewidth=1.1, zorder=3,
                        solid_capstyle="butt")
                ax.add_patch(plt.Rectangle((i - 0.28, q1), 0.56, q3 - q1, fill=False, edgecolor=vs.INK_2,
                                           linewidth=0.6, zorder=2))
        ax.set_xticks(range(len(classes)), classes, fontsize=vs.FS["tiny"])
        ax.tick_params(axis="x", length=0)
        ax.set_title(f"{row['feature']}", fontsize=vs.FS["body"], fontweight="medium", color=vs.INK, pad=3)
        ax.text(1.0, 1.0, f"η² = {row['eta_squared']:.2f}", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=vs.FS["tiny"], color=vs.INK_3)
    for ax in axes.flat[len(tested):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Value (supplied scale)")
    title = f"Distributions of the {len(tested)} features with the largest class effect"
    subtitle = ("Points: samples (jittered) · box: interquartile range · bar: median · "
                "features ranked by exploratory all-data ANOVA F")
    if titles:
        vs.header(fig, title, subtitle)
    caption = (f"Per-class distributions of the {len(tested)} features with the largest one-way ANOVA F on all labelled "
               f"samples (η² {tested['eta_squared'].min():.2f}–{tested['eta_squared'].max():.2f}). Points are "
               f"individual samples with horizontal jitter only; boxes show the interquartile range and the bar the median; "
               f"missing values are omitted per feature. Values are on the supplied scale (no transformation). Ranking used "
               f"all labelled data and is exploratory.")
    return {"fig": fig, "stem": "fig_top_feature_distributions", "title": title, "caption": caption}


def fig_heatmap(sdir: Path, raw, titles: bool, top: int = 30) -> dict | None:
    loaded = load_associations(sdir)
    if loaded is None or raw is None:
        return None
    assoc, summary = loaded
    X, y = raw
    classes = [c["label"] for c in summary["class_order"]]
    head = assoc[assoc["test_status"] == "finite_test"].sort_values("F", ascending=False).head(top)
    means = head[[f"mean_class_{i}" for i in range(len(classes))]].to_numpy(dtype=float)
    head = head.assign(peak=means.argmax(axis=1)).sort_values(["peak", "F"], ascending=[True, False])
    sample_order = np.concatenate([np.flatnonzero(y == c) for c in classes])
    values = X[head["feature"].tolist()].to_numpy(dtype=float)[sample_order]
    z = (values - np.nanmean(values, axis=0)) / np.where(np.nanstd(values, axis=0) > 0, np.nanstd(values, axis=0), 1)
    z = np.clip(z, -3, 3).T
    styles = class_styles(classes, pd.Series(y).value_counts().to_dict())

    fig = plt.figure(figsize=(vs.FULL_WIDTH, 3.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.045, 1], width_ratios=[1, 0.018], hspace=0.02, wspace=0.02)
    strip = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0], sharex=strip)
    cax = fig.add_subplot(grid[1, 1])
    ordered = y[sample_order]
    start = 0
    for cls in classes:
        count = int((ordered == cls).sum())
        strip.add_patch(plt.Rectangle((start - 0.5, 0), count, 1, color=styles[cls]["color"], linewidth=0))
        strip.text(start + count / 2 - 0.5, 1.25, cls, ha="center", va="bottom", fontsize=vs.FS["note"], color=vs.INK_2)
        if start:
            ax.axvline(start - 0.5, color=vs.SURFACE, linewidth=1.2)
        start += count
    strip.set_ylim(0, 1)
    strip.axis("off")
    image = ax.imshow(z, cmap=vs.DIVERGING, vmin=-3, vmax=3, aspect="auto", interpolation="nearest", rasterized=True)
    ax.set_xlim(-0.5, len(sample_order) - 0.5)
    ax.set_xticks([])
    ax.set_yticks(range(len(head)), head["feature"], fontsize=vs.FS["tiny"] - 1 if len(head) > 32 else vs.FS["tiny"])
    ax.tick_params(axis="y", length=0, labelcolor=vs.INK_3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(f"Samples (n = {len(sample_order):,}), grouped by reference label")
    bar = fig.colorbar(image, cax=cax, ticks=[-3, 0, 3])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=vs.FS["tiny"])
    bar.set_label("z-score (clipped at ±3)", fontsize=vs.FS["note"], color=vs.INK_2)
    title = f"Class-wise patterns of the {len(head)} highest-F features"
    subtitle = ("Samples ordered by reference label (the ordering uses labels) · features z-scored across samples, "
                "grouped by the class with the highest mean")
    if titles:
        vs.header(fig, title, subtitle)
    caption = (f"Heatmap of the {len(head)} features with the largest exploratory ANOVA F. Each feature is z-scored across "
               f"all {len(sample_order):,} samples and clipped at ±3; missing values appear as grey. Samples are grouped by "
               f"their reference label and features by the class with the highest mean, so block structure partly reflects "
               f"this label-based ordering and does not demonstrate unsupervised recovery.")
    return {"fig": fig, "stem": "fig_feature_heatmap", "title": title, "caption": caption}


# ---------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, help="evaluate_models.py run directory (contains evidence.json).")
    parser.add_argument("--structure", type=Path, help="analyze_structure.py output directory.")
    parser.add_argument("--config", type=Path, help="Exam config; needed only for raw-data figures "
                        "(distributions, heatmap). Defaults to the structure run's config_path.")
    parser.add_argument("--output-dir", type=Path, help="New directory for figures (default: figures_<UTC> next to the inputs).")
    parser.add_argument("--figures", default=",".join(FIGURES), help=f"Comma-separated subset of {', '.join(FIGURES)}.")
    parser.add_argument("--no-titles", action="store_true", help="Omit in-figure titles (paper mode: captions carry them).")
    parser.add_argument("--top-features", type=int, default=6, help="Panels in the distributions figure.")
    parser.add_argument("--heatmap-features", type=int, default=30, help="Rows in the feature heatmap.")
    args = parser.parse_args()
    if not args.run and not args.structure:
        parser.error("Give --run and/or --structure.")
    wanted = [f.strip() for f in args.figures.split(",") if f.strip()]
    unknown = sorted(set(wanted) - set(FIGURES))
    if unknown:
        parser.error(f"Unknown figures: {unknown}")

    vs.apply_style()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = (args.run or args.structure).resolve().parent
    out = (args.output_dir or base / f"figures_{stamp}").resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} is not empty; choose a new --output-dir.")
    titles = not args.no_titles

    inputs, made, skipped = {}, [], {}
    run = None
    if args.run:
        run = load_run(args.run)
        inputs["evidence"] = file_metadata(args.run / "evidence.json")
    config_path = args.config
    if args.structure and config_path is None:
        summary = args.structure / "structure_summary.json"
        if summary.exists():
            candidate = json.loads(summary.read_text(encoding="utf-8")).get("config_path")
            config_path = Path(candidate) if candidate and Path(candidate).exists() else None
    raw = None
    if args.structure and {"distributions", "heatmap"} & set(wanted):
        if config_path is None:
            skipped.update({f: "raw data unavailable: pass --config" for f in ("distributions", "heatmap")})
        else:
            raw = load_raw(config_path)
            inputs["config"] = file_metadata(config_path)

    builders = {
        "models": lambda: fig_models(run, out, titles) if run else None,
        "confusion": lambda: fig_confusion(run, args.run, out, titles) if run else None,
        "holdout": lambda: fig_holdout(run, args.run, titles) if run else None,
        "importance": lambda: fig_importance(run, args.run, titles) if run else None,
        "gap": lambda: fig_gap(run, titles) if run else None,
        "variance": lambda: fig_variance(args.structure, titles) if args.structure else None,
        "robustness": lambda: fig_cluster_robustness(args.structure, titles) if args.structure else None,
        "structure": lambda: fig_structure(args.structure, titles) if args.structure else None,
        "associations": lambda: fig_associations(args.structure, titles) if args.structure else None,
        "distributions": lambda: fig_distributions(args.structure, raw, titles, args.top_features) if args.structure else None,
        "heatmap": lambda: fig_heatmap(args.structure, raw, titles, args.heatmap_features) if args.structure else None,
    }
    if args.structure:
        for name in ("associations.csv", "clustering_scores.csv", "clustering_summary.json",
                     "clustering_variance.csv", "clustering_k_sweep.csv"):
            if (args.structure / name).exists():
                inputs[name] = file_metadata(args.structure / name)
    for name in wanted:
        if name in skipped:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = builders[name]()
            if result is None:
                skipped[name] = "required inputs not present"
                continue
            record = vs.save_figure(result["fig"], out, result["stem"], provenance=inputs)
        record.update(figure=name, title=result["title"], caption=result["caption"])
        made.append(record)

    lines = ["# Figure captions", "", "Generated by plot_results.py; numbers are read from the listed run outputs.", ""]
    for i, rec in enumerate(made, 1):
        lines += [f"**Figure {i}. {rec['title']}.** {rec['caption']}", "", f"Files: `{Path(rec['files']['pdf']).name}`, "
                  f"`{Path(rec['files']['png']).name}`", ""]
    (out / "captions.md").write_text("\n".join(lines), encoding="utf-8")
    vs.write_json(out / "figures_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "inputs": inputs, "figures": made,
        "skipped": skipped, "style": {"categorical": vs.CATEGORICAL, "font_stack": vs.FONT_STACK,
                                      "titles": titles}})
    print(json.dumps({"output_dir": str(out), "created": [r["figure"] for r in made], "skipped": skipped}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
