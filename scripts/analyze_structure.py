#!/usr/bin/env python3
"""Bounded, separate association and unsupervised analyses for an exam dataset.

All-data exploratory outputs from this script must never become preprocessing
inputs to a purportedly held-out prediction evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import warnings
import uuid

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from exam_io import atomic_json, load_config, load_data, make_output_dir, file_metadata


class NotApplicable(ValueError):
    """The bounded runner is inappropriate for the supplied task or data."""


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False, float_format="%.17g")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def validate_column_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{name} must be a list of exact column names.")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicate column names.")
    return value


def select_scope(frame: pd.DataFrame, cfg: dict, output: Path) -> tuple[np.ndarray, list[str], dict]:
    """Use an explicit, outcome-independent scope; never inspect labels here."""
    requested = cfg.get("structure_feature_columns")
    declared = list(frame.columns) if requested is None else validate_column_list(
        requested, "structure_feature_columns"
    )
    excluded = validate_column_list(cfg.get("structure_exclude_columns", []), "structure_exclude_columns")
    unknown = (set(declared) | set(excluded)) - set(frame.columns)
    if unknown:
        raise ValueError(f"Structure columns absent from the loaded predictor set: {sorted(unknown)[:10]}")
    chosen = [name for name in declared if name not in excluded]
    if not chosen:
        raise NotApplicable("No features remain in the declared structure-analysis scope.")
    rows = []
    for name in frame.columns:
        if name in excluded:
            reason = "explicit_structure_exclude_columns"
        elif name not in declared:
            reason = "outside_explicit_structure_feature_columns"
        else:
            reason = "included_in_declared_scope"
        rows.append({"feature": name, "dtype": str(frame[name].dtype), "scope_status": reason})
    atomic_csv(output / "feature_scope.csv", pd.DataFrame(rows))
    nonnumeric = [name for name in chosen if not pd.api.types.is_numeric_dtype(frame[name])]
    if nonnumeric:
        raise NotApplicable(
            "This runner supports numeric predictors and a classification target only. "
            f"Non-numeric columns: {nonnumeric[:10]}. Set structure_feature_columns to a "
            "predeclared numeric scope, or use a method appropriate for mixed data."
        )
    values = frame[chosen].to_numpy(dtype=np.float64, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError("Structure inputs contain infinity; resolve invalid measurements explicitly.")
    metadata = {
        "scope_source": "all_loaded_predictors" if requested is None else "explicit_structure_feature_columns",
        "loaded_predictor_count": len(frame.columns),
        "declared_feature_count": len(declared),
        "configured_excluded_columns": excluded,
        "analyzed_feature_count_before_quality_checks": len(chosen),
        "scope_file": str(output / "feature_scope.csv"),
        "outcome_independence_requirement": (
            "The declared feature scope must be chosen before inspecting label associations; "
            "a list selected using these same labels is not a valid full testing family."
        ),
        "clinical_name_alerts": [name for name in chosen if name.casefold() == "edema"],
    }
    if metadata["clinical_name_alerts"]:
        metadata["clinical_scope_note"] = (
            "Edema is retained because it was not explicitly excluded. Treat it as a clinical "
            "variable, not a gene; define a gene-only scope or analyze it separately when required."
        )
    return values, chosen, metadata


def numeric_quality(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.isfinite(values).sum(axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        low, high = np.nanmin(values, axis=0), np.nanmax(values, axis=0)
    all_missing = counts == 0
    constant = (counts > 0) & (low == high)
    return counts, all_missing, constant


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Correct the entire supplied hypothesis family using SciPy's complete BH."""
    if not hasattr(stats, "false_discovery_control"):
        raise RuntimeError("SciPy >= 1.11 is required for false_discovery_control; prepare it before the exam.")
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or not np.isfinite(p_values).all() or ((p_values < 0) | (p_values > 1)).any():
        raise ValueError("BH requires a finite one-dimensional array of p-values in [0, 1].")
    return stats.false_discovery_control(p_values, method="bh") if len(p_values) else p_values.copy()


def associations(values: np.ndarray, names: list[str], y: np.ndarray, cfg: dict,
                 output: Path, groups: np.ndarray | None = None) -> dict:
    started = time.monotonic()
    if groups is not None and pd.Series(groups).duplicated().any():
        raise NotApplicable(
            "Repeated observations share a configured group. Ordinary independent one-way ANOVA "
            "is inappropriate; use a paired, mixed-effects, or group-aware inference procedure."
        )
    codes, classes = pd.factorize(y, sort=True)
    n_classes = len(classes)
    if n_classes < 2:
        raise NotApplicable("Association analysis requires at least two observed classes.")
    minimum = cfg.get("structure_min_group_samples", 2)
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 2:
        raise ValueError("structure_min_group_samples must be an integer >= 2.")
    alpha = float(cfg.get("structure_fdr_alpha", 0.05))
    if not 0 < alpha < 1:
        raise ValueError("structure_fdr_alpha must lie strictly between 0 and 1.")
    counts, all_missing, constant = numeric_quality(values)
    class_counts = np.stack([np.isfinite(values[codes == i]).sum(axis=0) for i in range(n_classes)])
    insufficient = (class_counts < minimum).any(axis=0)
    reason = np.full(len(names), "tested", dtype=object)
    reason[insufficient] = f"fewer_than_{minimum}_observations_in_at_least_one_class"
    reason[constant] = "globally_constant_among_observed_values"
    reason[all_missing] = "all_values_missing"
    eligible = reason == "tested"
    family = pd.DataFrame({"feature": names, "family_status": reason, "n_effective": counts})
    for i in range(n_classes):
        family[f"n_class_{i}"] = class_counts[i]
    atomic_csv(output / "association_family.csv", family)
    indices = np.flatnonzero(eligible)
    if not len(indices):
        raise NotApplicable(
            "No valid ANOVA hypotheses remain. See association_family.csv for explicit exclusion reasons."
        )
    f_values = np.full(len(indices), np.nan)
    raw_p = np.full(len(indices), np.nan)
    warning_messages: list[str] = []
    # Partition complete and incomplete columns: do not force per-feature slow
    # NaN handling on thousands of complete columns, and do not drop whole rows.
    for offset in range(0, len(indices), 1024):
        block_indices = indices[offset:offset + 1024]
        block = values[:, block_indices]
        complete = np.isfinite(block).all(axis=0)
        for use, nan_policy in ((complete, "raise"), (~complete, "omit")):
            positions = np.flatnonzero(use)
            if not len(positions):
                continue
            grouped = [block[codes == i][:, positions] for i in range(n_classes)]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = stats.f_oneway(*grouped, axis=0, nan_policy=nan_policy)
            warning_messages.extend(str(w.message) for w in caught)
            f_values[offset + positions] = np.atleast_1d(result.statistic)
            raw_p[offset + positions] = np.atleast_1d(result.pvalue)
    invalid = (~np.isfinite(raw_p)) | (raw_p < 0) | (raw_p > 1) | np.isnan(f_values) | (f_values < 0)
    # Retain computationally invalid hypotheses in m with conservative p=1;
    # never reduce the family after observing their test statistics.
    p_for_bh = raw_p.copy()
    p_for_bh[invalid] = 1.0
    adjusted = bh_adjust(p_for_bh)
    effective_n = counts[indices]
    df_between = n_classes - 1
    df_within = effective_n - n_classes
    eta = np.full(len(indices), np.nan)
    finite_nonnegative = np.isfinite(f_values) & (f_values >= 0) & ~invalid
    eta[finite_nonnegative & (f_values == 0)] = 0.0
    positive = finite_nonnegative & (f_values > 0)
    eta[positive] = 1.0 / (1.0 + (df_within[positive] / df_between) / f_values[positive])
    infinite_f = np.isposinf(f_values) & ~invalid
    eta[infinite_f] = 1.0
    status = np.full(len(indices), "finite_test", dtype=object)
    status[infinite_f] = "degenerate_zero_within_class_variance"
    status[invalid] = "invalid_test_retained_with_p1_for_BH"
    table = pd.DataFrame({
        "feature": np.asarray(names)[indices],
        "F": f_values,
        "p_value": raw_p,
        "p_for_bh": p_for_bh,
        "q_bh": adjusted,
        "eta_squared": eta,
        "n_effective": effective_n,
        "df_between": df_between,
        "df_within": df_within,
        "test_status": status,
    })
    for i in range(n_classes):
        table[f"n_class_{i}"] = class_counts[i, indices]
        table[f"mean_class_{i}"] = np.nanmean(values[codes == i][:, indices], axis=0)
    table = table.sort_values(["F", "feature"], ascending=[False, True], na_position="last")
    atomic_csv(output / "associations.csv", table)
    result = {
        "status": "completed",
        "kind": "exploratory_all_data_one_way_ANOVA",
        "samples": len(y),
        "class_order": [{"index": i, "label": label, "sample_count": int((codes == i).sum())}
                        for i, label in enumerate(classes.tolist())],
        "input_feature_count": len(names),
        "tested_family_size": len(indices),
        "excluded_count_by_reason": pd.Series(reason[~eligible]).value_counts().to_dict(),
        "minimum_observed_samples_per_class_per_feature": minimum,
        "invalid_tests_retained_with_p1": int(invalid.sum()),
        "fdr_method": "scipy.stats.false_discovery_control(method='bh') on the entire tested family",
        "alpha": alpha,
        "nominal_p_below_alpha": int((p_for_bh < alpha).sum()),
        "q_below_alpha": int((adjusted < alpha).sum()),
        "p_numerically_zero_with_finite_F": int(((raw_p == 0) & np.isfinite(f_values)).sum()),
        "positive_infinite_F_count": int(infinite_f.sum()),
        "missing_observations_handling": "omit each feature's missing observations; no global row deletion or imputation",
        "effect_size": "eta_squared = F*(K-1)/(F*(K-1) + N_feature-K), evaluated stably",
        "data_scale": "the supplied numeric scale; no implicit log transform or standardization",
        "top_by_F": table.head(10).to_dict(orient="records"),
        "warnings": sorted(set(warning_messages)),
        "limitations": [
            "Exploratory associations on all labeled data are not held-out prediction results or confirmed biomarkers.",
            "Ordinary ANOVA assumes independent observations and a suitable within-group error model including equal variances.",
            "An omnibus result means at least one class mean differs; it does not establish which class or contrast differs.",
            "BH control relies on valid null p-values and independence or suitable positive dependence; those assumptions are not established by this run.",
            "Finite-F p=0 is floating-point underflow, not mathematical certainty. Infinite F indicates a degenerate zero-within-class-variance case.",
            "No biological identity is inferred from anonymous feature names; no F statistic or z-score is relabeled as fold change.",
            "Do not reuse this all-data ranking inside an evaluation that claims its validation labels were unseen.",
        ],
        "files": {"results": str(output / "associations.csv"),
                  "family_and_exclusions": str(output / "association_family.csv")},
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(output / "associations_summary.json", result)
    return result


def fit_unsupervised(values: np.ndarray, names: list[str], cfg: dict, n_clusters: int) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    """Fit using X only. Labels are deliberately absent from this interface."""
    _, all_missing, _ = numeric_quality(values)
    keep = ~all_missing
    if not keep.any():
        raise NotApplicable("Clustering has no observed numeric features.")
    imputer = SimpleImputer(strategy="median")
    filled = imputer.fit_transform(values[:, keep])
    variance = VarianceThreshold(threshold=0.0)
    try:
        filtered = variance.fit_transform(filled)
    except ValueError as exc:
        raise NotApplicable("Clustering has no varying features after median imputation.") from exc
    kept_names = np.asarray(names)[keep][variance.get_support()].tolist()
    n_samples, n_features = filtered.shape
    if not 2 <= n_clusters < n_samples:
        raise NotApplicable(f"n_clusters must satisfy 2 <= k < n_samples; received k={n_clusters}, n={n_samples}.")
    seed = int(cfg.get("seed", 42))
    requested = cfg.get("structure_pca_components", 50)
    if not isinstance(requested, int) or isinstance(requested, bool) or requested < 1:
        raise ValueError("structure_pca_components must be a positive integer.")
    n_components = min(50, requested, n_samples - 1, n_features)
    standardized = StandardScaler().fit_transform(filtered)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    scores = pca.fit_transform(standardized)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clusters = model.fit_predict(scores)
    unique_clusters = np.unique(clusters)
    sample_limit = cfg.get("structure_silhouette_max_samples", 2000)
    if not isinstance(sample_limit, int) or isinstance(sample_limit, bool) or sample_limit < 3:
        raise ValueError("structure_silhouette_max_samples must be an integer >= 3.")
    sample_size = min(sample_limit, n_samples)
    silhouette = None
    silhouette_note = None
    if 2 <= len(unique_clusters) < n_samples:
        try:
            silhouette = float(silhouette_score(
                scores, clusters, metric="euclidean",
                sample_size=sample_size if sample_size < n_samples else None, random_state=seed
            ))
        except ValueError as exc:
            silhouette_note = f"Silhouette unavailable for the sampled cluster configuration: {exc}"
    else:
        silhouette_note = "Silhouette requires at least two nonempty clusters and fewer clusters than samples."
    retained = set(kept_names)
    feature_status = {
        name: ("all_values_missing" if all_missing[i] else
               "constant_after_median_imputation" if name not in retained else "used")
        for i, name in enumerate(names)
    }
    summary = {
        "samples": n_samples, "input_feature_count": len(names), "used_feature_count": n_features,
        "n_clusters_requested": n_clusters, "n_clusters_observed": len(unique_clusters),
        "pca_components_requested": requested, "pca_components_used": n_components,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_cumulative_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "silhouette": silhouette, "silhouette_note": silhouette_note,
        "silhouette_sample_count": sample_size, "silhouette_metric": "euclidean",
        "silhouette_space": f"{n_components} unwhitened PCA scores of median-imputed, nonconstant, standardized features",
        "kmeans_inertia": float(model.inertia_),
        "pipeline": "median imputation -> zero-variance filter -> StandardScaler -> randomized PCA -> KMeans",
        "random_seed": seed, "kmeans_n_init": 10,
        "warnings": sorted({str(w.message) for w in caught}),
    }
    return summary, scores, clusters, feature_status


WARD_MAX_SAMPLES = 5000  # the Ward linkage stores an n x n distance matrix


def clustering_robustness(scores: np.ndarray, clusters: np.ndarray, cfg: dict, n_clusters: int) -> tuple[dict, np.ndarray | None, pd.DataFrame]:
    """Label-free robustness checks in the same PCA space: Ward hierarchical clustering at the same k, and a
    silhouette sweep over k for KMeans and Ward. Labels are deliberately absent from this interface."""
    seed = int(cfg.get("seed", 42))
    n = len(scores)
    sample = min(int(cfg.get("structure_silhouette_max_samples", 2000)), n)
    sample_size = sample if sample < n else None

    def silhouette(assignments):
        if not 2 <= len(np.unique(assignments)) < n:
            return None
        return float(silhouette_score(scores, assignments, sample_size=sample_size, random_state=seed))

    k_max = int(cfg.get("structure_k_max", 10))
    ks = list(range(2, max(2, min(k_max, n - 1)) + 1))
    tree = linkage(scores, method="ward") if n <= WARD_MAX_SAMPLES else None
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scores)
        ward_k = fcluster(tree, k, criterion="maxclust") - 1 if tree is not None else None
        rows.append({"k": k, "kmeans_silhouette": silhouette(km.labels_), "kmeans_inertia": float(km.inertia_),
                     "ward_silhouette": silhouette(ward_k) if ward_k is not None else None})
    sweep = pd.DataFrame(rows)
    ward = fcluster(tree, n_clusters, criterion="maxclust") - 1 if tree is not None else None
    best = lambda col: (int(sweep.loc[sweep[col].idxmax(), "k"]) if sweep[col].notna().any() else None)
    summary = {
        "k_sweep": {"k_values": ks, "best_k_kmeans_silhouette": best("kmeans_silhouette"),
                    "best_k_ward_silhouette": best("ward_silhouette") if tree is not None else None,
                    "note": "Silhouette in the same PCA space for each k; computed without labels. A silhouette "
                            "maximum is one internal criterion, not proof of the true number of groups."},
        "ward": ({"status": "completed", "linkage": "ward", "n_clusters": n_clusters, "silhouette": silhouette(ward),
                  "agreement_with_kmeans_ari": float(adjusted_rand_score(clusters, ward))}
                 if ward is not None else
                 {"status": "skipped", "reason": f"n = {n} exceeds {WARD_MAX_SAMPLES}; the Ward distance matrix would be too large."}),
    }
    return summary, ward, sweep


def optional_plot(scores: np.ndarray, clusters: np.ndarray, y: np.ndarray, ratio: list[float],
                  output: Path, enabled: bool) -> dict:
    if not enabled or scores.shape[1] < 2:
        return {"status": "omitted", "reason": "disabled or fewer than two retained components"}
    if not os.environ.get("MPLCONFIGDIR"):
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="ama4602-mpl-")
    try:
        import viz_style
        from viz_style import plt
        viz_style.apply_style()
    except ImportError:
        return {"status": "omitted", "reason": "matplotlib is not installed; no installation attempted"}
    palette = viz_style.CATEGORICAL + [viz_style.NEUTRAL] * 20
    path = output / "clustering_pca.png"
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
    try:
        label_codes, labels = pd.factorize(y, sort=True)
        for ax, codes, title, legend_labels in (
            (axes[0], clusters, "Clusters fitted without labels",
             {i: f"Cluster {i}" for i in np.unique(clusters)}),
            (axes[1], label_codes, "Labels shown after fitting",
             {i: str(label) for i, label in enumerate(labels.tolist())}),
        ):
            levels = np.unique(codes)
            for i, level in enumerate(levels):
                use = codes == level
                ax.scatter(scores[use, 0], scores[use, 1], color=palette[i],
                           marker=viz_style.MARKERS[i % len(viz_style.MARKERS)],
                           label=legend_labels[level], s=8, alpha=0.8, edgecolor="white", linewidth=0.25)
            ax.set_title(title)
            ax.set_xlabel(f"PC1 ({ratio[0]:.1%} variance)")
            ax.set_ylabel(f"PC2 ({ratio[1]:.1%} variance)")
            if len(levels) <= 10:
                ax.legend(loc="best", fontsize=7, frameon=False)
        fig.savefig(path, dpi=140)
    except Exception as exc:
        return {"status": "omitted", "reason": f"Optional plot failed: {type(exc).__name__}: {exc}"}
    finally:
        plt.close(fig)
    return {"status": "created", "file": str(path),
            "note": "A descriptive two-PC projection is not evidence of predictive accuracy.",
            "label_code_mapping": [{"code": i, "label": label} for i, label in enumerate(labels.tolist())]}


def clustering(values: np.ndarray, names: list[str], y: np.ndarray, ids: np.ndarray,
               cfg: dict, output: Path) -> dict:
    started = time.monotonic()
    n_classes = len(pd.unique(y))
    configured_k = cfg.get("n_clusters")
    n_clusters = n_classes if configured_k is None else configured_k
    if not isinstance(n_clusters, int) or isinstance(n_clusters, bool):
        raise ValueError("n_clusters must be an integer.")
    summary, scores, clusters, feature_status = fit_unsupervised(values, names, cfg, n_clusters)
    # No label-dependent value is passed into representation fitting; these
    # external alignment measures are computed only after fitting has finished.
    summary.update({
        "status": "completed", "kind": "descriptive_all_data_unsupervised_structure",
        "k_source": "observed_label_class_count_as_prior" if configured_k is None else "explicit_config_n_clusters",
        "k_note": ("The known number of label classes supplies k; the number of clusters was not discovered."
                   if configured_k is None else "k was fixed by configuration; record whether it used prior domain knowledge."),
        "adjusted_rand_index": float(adjusted_rand_score(y, clusters)),
        "normalized_mutual_information": float(normalized_mutual_info_score(y, clusters)),
        "labels_used_for_feature_selection": False,
        "labels_used_for_representation_or_cluster_fitting": False,
        "limitations": [
            "Labels are used only for post-fit external evaluation and plotting, apart from the disclosed optional known-k prior.",
            "Cluster numbers have no intrinsic class meaning; ARI and NMI are invariant to cluster relabeling.",
            "Silhouette measures separation in the stated PCA space, not classification accuracy.",
            "Explained variance is variance in the processed feature space, not percentage of predictive or biological information.",
            "All-data exploration is not a held-out generalization evaluation; no biological mechanism or clinical usefulness is established.",
        ],
    })
    robustness, ward, sweep = clustering_robustness(scores, clusters, cfg, n_clusters)
    if ward is not None:  # external evaluation of Ward, again only after fitting
        robustness["ward"].update(adjusted_rand_index=float(adjusted_rand_score(y, ward)),
                                  normalized_mutual_information=float(normalized_mutual_info_score(y, ward)))
        atomic_csv(output / "cluster_label_counts_ward.csv",
                   pd.crosstab(pd.Series(y, name="observed_label"), pd.Series(ward, name="ward_cluster")).reset_index())
    atomic_csv(output / "clustering_k_sweep.csv", sweep)
    summary.update(robustness)
    scores_frame = pd.DataFrame(scores, columns=[f"PC{i + 1}" for i in range(scores.shape[1])])
    scores_frame.insert(0, "observed_label", y)
    if ward is not None:
        scores_frame.insert(0, "ward_cluster", ward)
    scores_frame.insert(0, "cluster", clusters)
    scores_frame.insert(0, "sample_id", ids)
    atomic_csv(output / "clustering_scores.csv", scores_frame)
    atomic_csv(output / "clustering_feature_scope.csv", pd.DataFrame({
        "feature": list(feature_status), "status": list(feature_status.values())
    }))
    ratio = np.asarray(summary["pca_explained_variance_ratio"])
    atomic_csv(output / "clustering_variance.csv", pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(len(ratio))],
        "explained_variance_ratio": ratio, "cumulative_explained_variance_ratio": np.cumsum(ratio)
    }))
    contingency = pd.crosstab(pd.Series(y, name="observed_label"), pd.Series(clusters, name="cluster"))
    atomic_csv(output / "cluster_label_counts.csv", contingency.reset_index())
    summary["plot"] = optional_plot(
        scores, clusters, y, summary["pca_explained_variance_ratio"], output, bool(cfg.get("structure_plot", True))
    )
    summary["files"] = {
        "scores_clusters_ids_labels": str(output / "clustering_scores.csv"),
        "feature_scope": str(output / "clustering_feature_scope.csv"),
        "pca_variance": str(output / "clustering_variance.csv"),
        "cluster_label_counts": str(output / "cluster_label_counts.csv"),
        "k_sweep": str(output / "clustering_k_sweep.csv"),
    }
    summary["elapsed_seconds"] = time.monotonic() - started
    atomic_json(output / "clustering_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Exam configuration JSON")
    parser.add_argument("--mode", choices=("associations", "clustering", "both"), default="both")
    parser.add_argument("--permuted-labels-seed", type=int, help="Diagnostic only: permute y; writes to a separate output subdirectory")
    args = parser.parse_args()
    output = None
    run: dict = {"started_utc": datetime.now(timezone.utc).isoformat(), "mode": args.mode}
    try:
        cfg = load_config(args.config)
        suffix = "structure" if args.permuted_labels_seed is None else f"structure_permuted_{args.permuted_labels_seed}"
        suffix += "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
        output = make_output_dir(cfg) / suffix
        output.mkdir(parents=True, exist_ok=False)
        if cfg["task"] != "classification":
            raise NotApplicable("This structure runner supports a categorical classification target, not regression.")
        bundle = load_data(cfg, include_test=False)
        y = np.asarray(bundle.y)
        if len(pd.unique(y)) < 2:
            raise NotApplicable("At least two observed target classes are required.")
        if args.permuted_labels_seed is not None:
            y = np.random.default_rng(args.permuted_labels_seed).permutation(y)
        values, names, scope = select_scope(bundle.X, cfg, output)
        run.update({
            "task": cfg["task"], "target": cfg["target"], "config_path": cfg["_config_path"],
            "seed": cfg.get("seed", 42), "threads": cfg.get("threads", 2),
            "diagnostic_label_permutation_seed": args.permuted_labels_seed,
            "diagnostic_outputs_not_for_exam_submission": args.permuted_labels_seed is not None,
            "source_files": [file_metadata(cfg[key])
                             for key in ("train_path", "labels_path") if cfg.get(key)],
            "loader_diagnostics": bundle.diagnostics, "feature_scope": scope,
            "design_alerts": (["A time column is configured; temporal independence for ordinary ANOVA has not been validated."]
                              if bundle.times is not None else []),
            "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
            "results": {},
        })
        threads = int(cfg.get("threads", 2))
        if threads < 1:
            raise ValueError("threads must be >= 1.")
        with threadpool_limits(limits=threads):
            if args.mode in ("associations", "both"):
                try:
                    run["results"]["associations"] = associations(values, names, y, cfg, output, bundle.groups)
                except NotApplicable as exc:
                    result = {"status": "not_applicable", "reason": str(exc)}
                    atomic_json(output / "associations_summary.json", result)
                    run["results"]["associations"] = result
            if args.mode in ("clustering", "both"):
                try:
                    run["results"]["clustering"] = clustering(values, names, y, bundle.ids, cfg, output)
                except NotApplicable as exc:
                    result = {"status": "not_applicable", "reason": str(exc)}
                    atomic_json(output / "clustering_summary.json", result)
                    run["results"]["clustering"] = result
        completed = sum(r["status"] == "completed" for r in run["results"].values())
        run["status"] = "completed" if completed == len(run["results"]) else "partial" if completed else "not_applicable"
        atomic_json(output / "structure_summary.json", run)
        print(json.dumps({"status": run["status"], "summary": str(output / "structure_summary.json")}))
        return 0 if completed else 2
    except (ValueError, RuntimeError) as exc:
        run.update({"status": "not_applicable" if isinstance(exc, NotApplicable) else "error", "reason": str(exc)})
        if output is not None:
            atomic_json(output / "structure_summary.json", run)
        print(json.dumps(run if output is None else {"status": run["status"], "reason": str(exc),
                                                    "summary": str(output / "structure_summary.json")}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
