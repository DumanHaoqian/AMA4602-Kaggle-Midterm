"""Interactive Plotly versions of the standard figures for the HTML report.

Same palette, type and honesty rules as viz_style/plot_results; hover adds the
exact values, it never replaces the caption or the static PDF figures.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import plot_results as pr
import viz_style as vs

FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"


def _layout(fig: go.Figure, height: int, **kwargs) -> go.Figure:
    fig.update_layout(
        template="simple_white", height=height, margin=dict(l=10, r=20, t=20, b=10),
        font=dict(family=FONT, size=13, color=vs.INK), paper_bgcolor=vs.SURFACE, plot_bgcolor=vs.SURFACE,
        hoverlabel=dict(font=dict(family=FONT, size=12), bgcolor=vs.SURFACE, bordercolor=vs.RULE),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0.5, xanchor="center", font=dict(size=12)),
        **kwargs)
    fig.update_xaxes(linecolor=vs.RULE, tickcolor=vs.RULE, tickfont=dict(color=vs.INK_2), automargin=True,
                     title_font=dict(size=13, color=vs.INK), showgrid=False, zeroline=False)
    fig.update_yaxes(linecolor=vs.RULE, tickcolor=vs.RULE, tickfont=dict(color=vs.INK_2), automargin=True,
                     title_font=dict(size=13, color=vs.INK), showgrid=False, zeroline=False)
    return fig


def models(run: dict) -> go.Figure:
    ev = run["evidence"]
    lower = ev.get("metric_direction") == "minimize"
    cands = sorted([m for m in run["models"] if m["name"] != "dummy"], key=lambda m: m["mean"], reverse=not lower)
    selected = ev["selection"]["model"]
    metric = pr.METRIC_NAMES.get(ev["metric"], ev["metric"])
    fig = go.Figure()
    for m in cands:
        color = vs.ACCENT if m["name"] == selected else vs.INK_2
        fig.add_trace(go.Scatter(
            x=[min(m["scores"]), max(m["scores"])], y=[m["label"]] * 2, mode="lines", showlegend=False,
            line=dict(color=vs.ACCENT if m["name"] == selected else vs.NEUTRAL, width=2), hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=m["scores"], y=[m["label"]] * len(m["scores"]), mode="markers", showlegend=False,
            marker=dict(size=10, color=vs.SURFACE, line=dict(width=1.6, color=color)),
            customdata=np.arange(1, len(m["scores"]) + 1),
            hovertemplate=f"{m['label']}<br>Fold %{{customdata}}: %{{x:.4f}}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[m["mean"]], y=[m["label"]], mode="markers", showlegend=False,
            marker=dict(size=14, color=color, line=dict(width=1, color=vs.SURFACE)),
            hovertemplate=f"{m['label']}<br>Mean {metric}: %{{x:.4f}}<br>SD: {np.std(m['scores']):.4f}<extra></extra>"))
    base = next((m for m in run["models"] if m["name"] == "dummy"), None)
    title = f"{metric} per validation fold" + (" (lower is better)" if lower else "")
    _layout(fig, 120 + 60 * len(cands), xaxis_title=title)
    fig.update_xaxes(showgrid=True, gridcolor=vs.GRID)
    # Plotly lists categorical y values top-down in reverse array order: best pipeline on top.
    fig.update_yaxes(categoryorder="array", categoryarray=[m["label"] for m in cands], autorange="reversed",
                     ticks="", showline=False, tickfont=dict(size=13, color=vs.INK))
    if base:
        fig.add_annotation(xref="paper", yref="paper", x=1, y=1.08, showarrow=False, xanchor="right",
                           text=f"{base['label']} = {base['mean']:.3f}", font=dict(size=12, color=vs.INK_3))
    return fig, None


def confusion(run: dict, run_dir) -> go.Figure | None:
    ev = run["evidence"]
    if ev["task"] != "classification":
        return None
    sel = next(m for m in run["models"] if m["name"] == ev["selection"]["model"])
    oof = pd.read_csv(run_dir / sel["oof_file"], dtype={"y_true": str, "prediction": str})
    oof = oof[oof["fold"] >= 0]
    classes = [str(c) for c in ev["class_order"]]
    counts = pd.crosstab(oof["y_true"], oof["prediction"]).reindex(index=classes, columns=classes, fill_value=0)
    share = counts.to_numpy() / np.maximum(counts.to_numpy().sum(axis=1, keepdims=True), 1)
    text = [[str(v) if v else "" for v in row] for row in counts.to_numpy()]
    fig = go.Figure(go.Heatmap(
        z=share, x=classes, y=classes, zmin=0, zmax=1, xgap=3, ygap=3,
        colorscale=[[0, "#FBF6EF"], [0.2, "#FDE3CB"], [0.4, "#FBC293"], [0.6, "#F39A62"], [0.8, "#E2703A"], [1, "#C4511B"]],
        text=text, texttemplate="%{text}", textfont=dict(size=14), customdata=counts.to_numpy(),
        colorbar=dict(title=dict(text="Share of<br>true class", font=dict(size=12)), thickness=10, outlinewidth=0),
        hovertemplate="True %{y} → predicted %{x}<br>%{customdata} samples (%{z:.1%} of %{y})<extra></extra>"))
    _layout(fig, 430, xaxis_title="Predicted class", yaxis_title="True class")
    fig.update_yaxes(autorange="reversed", showline=False, ticks="")
    fig.update_xaxes(showline=False, ticks="", side="bottom")
    n, errors = int(counts.to_numpy().sum()), int(counts.to_numpy().sum() - np.trace(counts.to_numpy()))
    caption = (f"Out-of-fold confusion matrix of the selected pipeline ({sel['label']}), pooled over the "
               f"{pr.cv_design(ev)} validation folds (n = {n:,}, {errors} errors). {pr.oof_note(ev)}. "
               f"Cells show counts and are shaded by the share of the true class; "
               f"hover for exact shares. This is a selection-CV diagnostic, not an independent test.")
    return fig, caption


def structure(sdir) -> go.Figure | None:
    path = sdir / "clustering_scores.csv"
    if not path.exists():
        return None
    scores = pd.read_csv(path, dtype={"observed_label": str})
    summary = json.loads((sdir / "clustering_summary.json").read_text(encoding="utf-8"))
    ratio = summary["pca_explained_variance_ratio"]
    labels = sorted(scores["observed_label"].unique())
    styles = pr.class_styles(labels, scores["observed_label"].value_counts().to_dict())
    table = pd.crosstab(scores["observed_label"], scores["cluster"]).loc[labels]
    mapping = pr.match_clusters(table)
    symbols = {"o": "circle", "s": "square", "^": "triangle-up", "D": "diamond", "v": "triangle-down",
               "P": "cross", ".": "circle", "x": "x"}
    fig = go.Figure()
    for lab in labels:
        sub = scores[scores["observed_label"] == lab]
        st = styles[lab]
        fig.add_trace(go.Scatter(
            x=sub["PC1"], y=sub["PC2"], mode="markers", name=st["legend"],
            marker=dict(size=7, color=st["color"], symbol=symbols[st["marker"]], line=dict(width=0.5, color=vs.SURFACE)),
            customdata=np.stack([sub["sample_id"], sub["cluster"]], axis=1),
            hovertemplate="%{customdata[0]}<br>Label " + lab + " · cluster C%{customdata[1]}<extra></extra>"))
    n_label = len(fig.data)
    for c in sorted(scores["cluster"].unique()):
        sub = scores[scores["cluster"] == c]
        st = styles[mapping[c]] if c in mapping else {"color": vs.NEUTRAL, "marker": "x"}
        fig.add_trace(go.Scatter(
            x=sub["PC1"], y=sub["PC2"], mode="markers", name=f"C{c}", visible=False,
            marker=dict(size=7, color=st["color"], symbol=symbols[st["marker"]], line=dict(width=0.5, color=vs.SURFACE)),
            customdata=np.stack([sub["sample_id"], sub["observed_label"]], axis=1),
            hovertemplate="%{customdata[0]}<br>Cluster C" + str(c) + " · label %{customdata[1]}<extra></extra>"))
    n_all = len(fig.data)
    vis_labels = [i < n_label for i in range(n_all)]
    vis_clusters = [i >= n_label for i in range(n_all)]
    _layout(fig, 470, xaxis_title=f"PC1 ({ratio[0]:.1%} of variance)", yaxis_title=f"PC2 ({ratio[1]:.1%} of variance)")
    fig.update_layout(updatemenus=[dict(
        type="buttons", direction="right", x=0, y=1.12, xanchor="left", showactive=True,
        bgcolor=vs.SURFACE, bordercolor=vs.RULE, font=dict(size=12),
        buttons=[dict(label="Reference labels", method="update", args=[{"visible": vis_labels}]),
                 dict(label="k-means clusters (no labels)", method="update", args=[{"visible": vis_clusters}])])],
        margin=dict(t=60))
    k_note = "fixed in advance" if summary.get("k_source", "").startswith("explicit") else "chosen from the data"
    caption = (f"Unsupervised structure on the first two of {summary['pca_components_used']} principal components of "
               f"standardised features. Use the buttons to switch between the reference labels and the k-means clusters "
               f"(k = {summary['n_clusters_observed']}, {k_note}), which were fitted without labels; cluster colours are "
               f"matched to their best-overlapping label for display only. ARI = {summary['adjusted_rand_index']:.3f}, "
               f"NMI = {summary['normalized_mutual_information']:.3f}, silhouette = {summary['silhouette']:.3f}. A two-PC "
               f"projection is descriptive and does not validate a classifier.")
    return fig, caption


def associations(sdir) -> go.Figure | None:
    loaded = pr.load_associations(sdir)
    if loaded is None:
        return None
    assoc, summary = loaded
    tested = assoc[assoc["test_status"] == "finite_test"].sort_values("F", ascending=False).reset_index(drop=True)
    rank = np.arange(1, len(tested) + 1)
    q = tested["q_bh"].to_numpy()
    fig = go.Figure(go.Scatter(
        x=rank, y=tested["eta_squared"], mode="lines", line=dict(color=vs.ACCENT, width=2),
        fill="tozeroy", fillcolor="rgba(217,119,87,0.12)", showlegend=False,
        customdata=np.stack([tested["feature"], tested["F"], np.where(q == 0, "< 1e-300", [f"{v:.2e}" for v in q])], axis=1),
        hovertemplate="Rank %{x}: %{customdata[0]}<br>η² = %{y:.3f} · F = %{customdata[1]:,.1f}<br>BH q = %{customdata[2]}<extra></extra>"))
    _layout(fig, 400, xaxis_title="Feature rank by ANOVA F (log scale)", yaxis_title="Effect size η²")
    fig.update_xaxes(type="log")
    fig.update_yaxes(range=[0, 1.02], showgrid=True, gridcolor=vs.GRID)
    fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.06, xanchor="left", showarrow=False, align="left",
                       text=(f"{summary['q_below_alpha']:,} of {summary['tested_family_size']:,} features "
                             f"with BH q < {summary['alpha']:g}"), font=dict(size=12, color=vs.INK_2))
    caption = (f"Effect size η² of all {summary['tested_family_size']:,} tested features, ranked by one-way ANOVA F in "
               f"the exploratory all-data screen; hover a point for the feature, F and Benjamini–Hochberg q-value. "
               f"{summary['q_below_alpha']:,} features have q < {summary['alpha']:g}. The omnibus test indicates that at "
               f"least one class mean differs; it does not identify which class or validate a predictor.")
    return fig, caption


def distributions(sdir, raw, top: int = 12) -> go.Figure | None:
    loaded = pr.load_associations(sdir)
    if loaded is None or raw is None:
        return None
    assoc, summary = loaded
    X, y = raw
    classes = [c["label"] for c in summary["class_order"]]
    styles = pr.class_styles(classes, pd.Series(y).value_counts().to_dict())
    head = assoc[assoc["test_status"] == "finite_test"].sort_values("F", ascending=False).head(top)
    fig = go.Figure()
    per_feature = len(classes)
    for k, (_, row) in enumerate(head.iterrows()):
        values = X[row["feature"]].to_numpy(dtype=float)
        for cls in classes:
            v = values[(y == cls) & np.isfinite(values)]
            fig.add_trace(go.Box(
                y=v, name=cls, visible=(k == 0), boxpoints="all", jitter=0.45, pointpos=0,
                marker=dict(size=4, color=styles[cls]["color"], opacity=0.45),
                line=dict(color=vs.INK_2, width=1.2), fillcolor="rgba(255,255,255,0)", showlegend=False,
                hovertemplate=f"{cls}: %{{y:.3f}}<extra>{row['feature']}</extra>"))
    buttons = []
    for k, (_, row) in enumerate(head.iterrows()):
        vis = [k * per_feature <= i < (k + 1) * per_feature for i in range(len(fig.data))]
        buttons.append(dict(label=f"{row['feature']} (η² = {row['eta_squared']:.2f})", method="update",
                            args=[{"visible": vis}]))
    _layout(fig, 430, yaxis_title="Value (supplied scale)")
    fig.update_yaxes(showgrid=True, gridcolor=vs.GRID)
    fig.update_layout(margin=dict(t=60), updatemenus=[dict(
        buttons=buttons, direction="down", x=0, y=1.14, xanchor="left", showactive=True,
        bgcolor=vs.SURFACE, bordercolor=vs.RULE, font=dict(size=12))])
    caption = (f"Per-class distributions of the {len(head)} features with the largest exploratory ANOVA F; choose a "
               f"feature from the menu. Points are individual samples, boxes the interquartile range with the median. "
               f"Values are on the supplied scale; missing values are omitted per feature. The ranking used all "
               f"labelled data and is exploratory.")
    return fig, caption
