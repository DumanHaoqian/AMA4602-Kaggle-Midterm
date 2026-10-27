"""Report source parsing, run-derived facts, tables and inline format conversion.

A report source is one Markdown file: a front-matter block, then sections. Every
number in the prose should be a {{fact}} placeholder resolved from the run outputs,
so the PDF, notebook and HTML all quote the same evidence. Unknown placeholders,
citations or cross-references stop the build instead of rendering silently.
"""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import plot_results as pr

FIGURE_NAMES = ("models", "confusion", "holdout", "importance", "gap", "structure", "variance", "robustness",
                "associations", "distributions", "heatmap")
TABLE_NAMES = ("data_summary", "cv_results", "metrics", "class_recall", "top_features", "importance")
FIGURE_STEMS = {"models": "fig_model_comparison", "structure": "fig_unsupervised_structure",
                "associations": "fig_class_associations", "distributions": "fig_top_feature_distributions",
                "heatmap": "fig_feature_heatmap"}


class ReportError(ValueError):
    pass


# ------------------------------------------------------------------- parsing

@dataclass
class Block:
    kind: str                 # heading, para, list, math, figure, table, code, refs
    text: str = ""
    level: int = 0
    label: str | None = None
    ordered: bool = False
    items: list = field(default_factory=list)
    options: dict = field(default_factory=dict)


@dataclass
class Source:
    meta: dict
    abstract: str
    blocks: list
    references: dict          # key -> text, in listed order
    path: Path


def parse_source(path: Path) -> Source:
    text = Path(path).read_text(encoding="utf-8")
    meta = {}
    if text.startswith("---"):
        head, _, text = text[3:].partition("\n---")
        for line in head.strip().splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                key, sep, value = line.partition(":")
                if not sep:
                    raise ReportError(f"Front matter line without ':': {line!r}")
                meta[key.strip()] = value.strip()
    for key in ("title", "author"):
        if not meta.get(key):
            raise ReportError(f"Front matter must set {key}.")
    lines = text.strip("\n").splitlines()
    blocks, refs, abstract = [], {}, []
    section = None
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        heading = re.match(r"^(#{1,3})\s+(.+?)\s*(\{#([\w:-]+)\})?\s*$", stripped)
        if heading:
            title = heading.group(2).strip()
            level = len(heading.group(1))
            section = title.lower() if level == 1 else section
            if level == 1 and section in ("abstract", "references"):
                i += 1
                continue
            blocks.append(Block("heading", title, level=level, label=heading.group(4)))
            i += 1
            continue
        if section == "references":
            ref = re.match(r"^\[([\w:.-]+)\]\s+(.+)$", stripped)
            if not ref:
                raise ReportError(f"Reference lines must look like '[key] Full reference.': {stripped!r}")
            refs[ref.group(1)] = ref.group(2)
            i += 1
            continue
        if stripped.startswith(":::"):
            parts = stripped[3:].split()
            if len(parts) < 2 or parts[0] not in ("figure", "table"):
                raise ReportError(f"Directive must be '::: figure NAME' or '::: table NAME': {stripped!r}")
            kind, name = parts[0], parts[1]
            allowed = FIGURE_NAMES if kind == "figure" else TABLE_NAMES
            if name not in allowed:
                raise ReportError(f"Unknown {kind} {name!r}; choose from {', '.join(allowed)}.")
            options = dict(p.split("=", 1) for p in parts[2:] if "=" in p)
            blocks.append(Block(kind, name, options=options))
            i += 1
            continue
        if stripped.startswith("```"):
            body = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body.append(lines[i])
                i += 1
            blocks.append(Block("code", "\n".join(body)))
            i += 1
            continue
        if stripped.startswith("$$"):
            body = [stripped[2:]]
            if not (stripped.endswith("$$") and len(stripped) > 4):
                i += 1
                while i < len(lines) and "$$" not in lines[i]:
                    body.append(lines[i])
                    i += 1
                body.append(lines[i].replace("$$", "") if i < len(lines) else "")
            else:
                body = [stripped.strip("$")]
            blocks.append(Block("math", "\n".join(b for b in body if b.strip())))
            i += 1
            continue
        item = re.match(r"^(-|\*|\d+\.)\s+(.+)$", stripped)
        if item:
            ordered = item.group(1)[0].isdigit()
            items = []
            while i < len(lines):
                m = re.match(r"^(-|\*|\d+\.)\s+(.+)$", lines[i].strip())
                if m:
                    items.append(m.group(2))
                elif lines[i].strip() and items:
                    items[-1] += " " + lines[i].strip()
                else:
                    break
                i += 1
            blocks.append(Block("list", items=items, ordered=ordered))
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,3}\s|:::|```|\$\$|(-|\*|\d+\.)\s)", lines[i].strip()):
            para.append(lines[i].strip())
            i += 1
        if section == "abstract":
            abstract.append(" ".join(para))
        else:
            blocks.append(Block("para", " ".join(para)))
    if not abstract:
        raise ReportError("The source needs a '# Abstract' section.")
    return Source(meta, "\n\n".join(abstract), blocks, refs, Path(path))


# --------------------------------------------------------------------- facts

@dataclass
class Context:
    run_dir: Path | None
    structure_dir: Path | None
    config_path: Path | None
    run: dict | None
    raw: tuple | None
    facts: dict


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def load_context(run: str | Path | None, structure: str | Path | None, config: str | Path | None,
                 need_raw: bool = True) -> Context:
    run_dir = Path(run).resolve() if run else None
    sdir = Path(structure).resolve() if structure else None
    config_path = Path(config).resolve() if config else None
    run_data = pr.load_run(run_dir) if run_dir else None
    raw = pr.load_raw(config_path) if (config_path and need_raw) else None
    facts = {}
    if run_data:
        ev = run_data["evidence"]
        metric = pr.METRIC_NAMES.get(ev["metric"], ev["metric"])
        models = run_data["models"]
        cands = [m for m in models if m["name"] != "dummy"]
        lower = ev.get("metric_direction") == "minimize"
        cands.sort(key=lambda m: m["mean"], reverse=not lower)
        best = cands[0]["mean"]
        tied = [m["label"] for m in cands if abs(m["mean"] - best) < 1e-12]
        sel = next(m for m in models if m["name"] == ev["selection"]["model"])
        base = next((m for m in models if m["name"] == "dummy"), None)
        facts.update({
            "n_samples": f"{ev['n_rows']:,}", "n_features": f"{ev['n_features']:,}",
            "metric": metric, "metric_lower": metric.lower() if ev["metric"] in ("accuracy", "balanced_accuracy", "log_loss") else metric,
            "metric_direction": "lower is better" if lower else "higher is better",
            "n_folds": str(ev.get("n_splits") or ev["config"]["split"].get("n_splits") or len(sel["scores"])),
            "n_repeats": str(int(ev.get("repeats") or 1)), "cv_design": pr.cv_design(ev),
            "n_fold_scores": str(len(sel["scores"])), "split_method": ev["config"]["split"]["method"],
            "seed": str(ev["config"].get("seed")), "n_candidates": str(len(cands)),
            "candidate_list": pr.join_names([m["label"] for m in cands]),
            "best_model": pr.join_names(tied), "best_score": f"{best:.3f}",
            "selected_model": sel["label"], "selected_score": f"{sel['mean']:.3f}",
            "selected_sd": f"{np.std(sel['scores']):.3f}",
            "baseline_score": f"{base['mean']:.3f}" if base else "n/a",
            "python_version": ev["versions"].get("python", ""), "sklearn_version": ev["versions"].get("scikit_learn", ""),
            "excluded_columns": ", ".join(c for c in ev["data_diagnostics"].get("excluded_train_columns", [])
                                          if c != ev["config"].get("id_column")) or "none",
        })
        for m in models:
            facts[f"score_{m['name']}"] = f"{m['mean']:.3f}"
        if ev.get("mode") == "leaderboard":
            trials = ev.get("trials", [])
            facts["n_trials"] = str(sum(t.get("status") == "complete" for t in trials))
            facts["n_trials_aborted"] = str(sum(t.get("status") == "aborted_budget" for t in trials))
            if ev.get("blend"):
                facts.update({"nested_cv_blend": f"{ev['blend']['nested_cv_blend']:.3f}",
                              "nested_cv_single": f"{ev['blend']['nested_cv_single']:.3f}",
                              "blend_decision": "was selected" if ev["blend"]["selected"] else "was not selected"})
            if ev.get("adversarial_validation"):
                facts.update({"adversarial_auc": f"{ev['adversarial_validation']['auc']:.3f}",
                              "adversarial_shift": ev["adversarial_validation"]["shift"]})
        if ev["task"] == "classification":
            classes = [str(c) for c in ev["class_order"]]
            oof = _oof(run_dir, sel)
            correct = int((oof["y_true"] == oof["prediction"]).sum())
            counts = oof["y_true"].value_counts()
            facts.update({"n_classes": str(len(classes)), "class_list": pr.join_names(classes),
                          "class_counts": ", ".join(f"{c} ({counts.get(c, 0):,})" for c in classes),
                          "largest_class": counts.idxmax(), "largest_class_n": f"{counts.max():,}",
                          "smallest_class": counts.idxmin(), "smallest_class_n": f"{counts.min():,}",
                          "oof_total": f"{len(oof):,}", "oof_correct": f"{correct:,}",
                          "oof_errors": f"{len(oof) - correct:,}", "oof_accuracy": f"{correct / len(oof):.3f}"})
        h = ev.get("holdout") or {}
        if h.get("status") == "complete":
            facts.update({"holdout_n": f"{h['n_holdout']:,}", "development_n": f"{h['n_development']:,}",
                          "holdout_fraction": _pct(h["fraction"], 0), "holdout_score": f"{h['score']:.3f}",
                          "holdout_model": next((m["label"] for m in models if m["name"] == h["model"]), h["model"])})
            hp = run_dir / h["predictions_file"]
            if ev["task"] == "classification" and hp.exists():
                hf = pd.read_csv(hp, dtype={"y_true": str, "prediction": str})
                right = int((hf["y_true"] == hf["prediction"]).sum())
                facts.update({"holdout_correct": f"{right:,}", "holdout_errors": f"{len(hf) - right:,}"})
        exps_by_name = {e["name"]: e for e in ev["experiments"]}
        gaps = [(m["label"], exps_by_name[m["name"]]["train_score_mean"] - m["mean"]) for m in cands
                if exps_by_name.get(m["name"], {}).get("train_score_mean") is not None]
        if gaps:
            gaps = [(label, -g if lower else g) for label, g in gaps]  # positive = optimistic training fit
            label, g = max(gaps, key=lambda t: t[1])
            facts.update({"max_gap": f"{g:+.3f}", "max_gap_model": label})
            selected_train = exps_by_name[sel["name"]].get("train_score_mean")
            if selected_train is not None:
                facts["selected_gap"] = f"{(sel['mean'] - selected_train if lower else selected_train - sel['mean']):+.3f}"
        imp = importance_frame(run_dir, run_data)
        if imp is not None:
            frame, exp, _ = imp
            weight_cols = [c for c in frame if c.startswith("weight_")]
            first = frame.iloc[0]
            facts.update({"top_weight_feature": str(first["feature"]),
                          "top_weight_model": next((m["label"] for m in models if m["name"] == exp["name"]), exp["name"])})
            if len(weight_cols) > 1:
                facts["top_weight_class"] = max(weight_cols, key=lambda c: abs(first[c])).removeprefix("weight_")
            sel_freq = frame.head(20)["selection_frequency"]
            facts["top20_all_folds"] = f"{int((sel_freq >= 1.0).sum())} of {len(sel_freq)}"
    if raw is not None:
        X, _ = raw
        facts.update({"n_missing": f"{int(X.isna().to_numpy().sum()):,}",
                      "n_constant": f"{int((X.nunique(dropna=True) <= 1).sum()):,}"})
    if sdir:
        cpath = sdir / "clustering_summary.json"
        if cpath.exists():
            c = json.loads(cpath.read_text(encoding="utf-8"))
            ratio = c["pca_explained_variance_ratio"]
            facts.update({"ari": f"{c['adjusted_rand_index']:.2f}", "nmi": f"{c['normalized_mutual_information']:.2f}",
                          "silhouette": f"{c['silhouette']:.2f}", "pca_components": str(c["pca_components_used"]),
                          "pca_variance": _pct(c["pca_cumulative_explained_variance"]),
                          "pc1_variance": _pct(ratio[0]), "pc2_variance": _pct(ratio[1]),
                          "k_clusters": str(c["n_clusters_observed"])})
            ward, sweep = c.get("ward", {}), c.get("k_sweep", {})
            if ward.get("status") == "completed":
                facts.update({"ward_ari": f"{ward['adjusted_rand_index']:.2f}",
                              "ward_nmi": f"{ward['normalized_mutual_information']:.2f}",
                              "ward_silhouette": f"{ward['silhouette']:.2f}",
                              "kmeans_ward_ari": f"{ward['agreement_with_kmeans_ari']:.2f}"})
            if sweep.get("best_k_kmeans_silhouette") is not None:
                facts.update({"best_k_silhouette": str(sweep["best_k_kmeans_silhouette"]),
                              "k_sweep_range": f"{min(sweep['k_values'])}–{max(sweep['k_values'])}"})
                if sweep.get("best_k_ward_silhouette") is not None:
                    facts["best_k_ward_silhouette"] = str(sweep["best_k_ward_silhouette"])
            var = sdir / "clustering_variance.csv"
            if var.exists():
                cum = pd.read_csv(var)["cumulative_explained_variance_ratio"].to_numpy()
                for level in (50, 80, 90):
                    hit = np.nonzero(cum >= level / 100)[0]
                    if hit.size:
                        facts[f"pcs_for_{level}pct"] = str(int(hit[0]) + 1)
        loaded = pr.load_associations(sdir)
        if loaded:
            assoc, s = loaded
            top = assoc[assoc["test_status"] == "finite_test"].sort_values("F", ascending=False).iloc[0]
            facts.update({"n_tested": f"{s['tested_family_size']:,}", "n_significant": f"{s['q_below_alpha']:,}",
                          "significant_share": _pct(s["q_below_alpha"] / s["tested_family_size"]),
                          "alpha": f"{s['alpha']:g}", "top_feature": str(top["feature"]),
                          "top_eta2": f"{top['eta_squared']:.2f}",
                          "n_p_underflow": f"{s.get('p_numerically_zero_with_finite_F', 0):,}"})
    return Context(run_dir, sdir, config_path, run_data, raw, facts)


def importance_frame(run_dir: Path, run: dict):
    """(importance table, experiment, fallback flag) for the selected pipeline, else the best candidate with weights."""
    ev = run["evidence"]
    exps = [e for e in ev["experiments"] if e.get("status") == "complete" and e.get("importance_file")
            and (run_dir / e["importance_file"]).exists()]
    if not exps:
        return None
    exp = next((e for e in exps if e["name"] == ev["selection"]["model"]), None)
    fallback = exp is None
    if fallback:
        exp = sorted(exps, key=lambda e: e["fold_score_mean"], reverse=ev.get("metric_direction") != "minimize")[0]
    return pd.read_csv(run_dir / exp["importance_file"]), exp, fallback


def _oof(run_dir: Path, model: dict) -> pd.DataFrame:
    oof = pd.read_csv(run_dir / model["oof_file"], dtype={"y_true": str, "prediction": str})
    return oof[oof["fold"] >= 0]


# -------------------------------------------------------------------- tables

def table(ctx: Context, name: str) -> dict:
    """Return {'caption', 'columns', 'rows', 'align', 'bold'} with string cells."""
    f, run = ctx.facts, ctx.run
    if name == "data_summary":
        need(run, "data_summary needs --run")
        rows = [["Labelled samples", f["n_samples"]], ["Predictors analysed", f["n_features"]]]
        if "n_classes" in f:
            rows += [["Classes", f["n_classes"]], ["Class sizes", f["class_counts"]]]
        if "n_missing" in f:
            rows += [["Missing predictor values (cells)", f["n_missing"]], ["Constant predictors", f["n_constant"]]]
        rows += [["Columns excluded from predictors", f["excluded_columns"]]]
        return {"caption": "Data used in the analysis. Counts are computed from the loaded training files after the ID join.",
                "columns": ["Quantity", "Value"], "rows": rows, "align": "lp", "bold": []}
    if name == "cv_results":
        need(run, "cv_results needs --run")
        ev = run["evidence"]
        exps = {e["name"]: e for e in ev["experiments"]}
        lower = ev.get("metric_direction") == "minimize"
        models = sorted(run["models"], key=lambda m: (m["name"] == "dummy", -m["mean"] if not lower else m["mean"]))
        rows, best = [], None
        for m in models:
            e = exps[m["name"]]
            s = np.asarray(m["scores"])
            rows.append([m["label"], f"{s.mean():.3f}", f"{s.std():.3f}", f"{s.min():.3f}", f"{s.max():.3f}",
                         f"{e['pooled_oof_score']:.3f}" if e.get("pooled_oof_score") is not None else "–",
                         f"{e.get('elapsed_seconds', 0):.1f}"])
        values = [float(r[1]) for r, m in zip(rows, models) if m["name"] != "dummy"]
        target = min(values) if lower else max(values)
        bold = [i for i, (r, m) in enumerate(zip(rows, models)) if m["name"] != "dummy" and float(r[1]) == target]
        metric = f["metric"]
        return {"caption": (f"Selection cross-validation results ({f['cv_design']}, identical folds "
                            f"for every pipeline, n = {f['n_samples']}). Mean, SD (population, across folds), minimum and "
                            f"maximum fold {f['metric_lower']}; pooled OOF is the metric on all out-of-fold predictions together. Bold "
                            f"marks the best mean. These scores selected the model and are not an independent test estimate."),
                "columns": ["Pipeline", "Mean", "SD", "Min", "Max", "Pooled OOF", "Time (s)"],
                "rows": rows, "align": "lrrrrrr", "bold": bold}
    if name == "class_recall":
        need(run and run["evidence"]["task"] == "classification", "class_recall needs a classification run")
        ev = run["evidence"]
        sel = next(m for m in run["models"] if m["name"] == ev["selection"]["model"])
        oof = _oof(ctx.run_dir, sel)
        rows = []
        for c in [str(c) for c in ev["class_order"]]:
            sub = oof[oof["y_true"] == c]
            wrong = sub[sub["prediction"] != c]["prediction"].value_counts()
            tp, predicted = int((sub["prediction"] == c).sum()), int((oof["prediction"] == c).sum())
            precision = tp / predicted if predicted else float("nan")
            recall = tp / len(sub) if len(sub) else float("nan")
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            rows.append([c, f"{len(sub):,}", f"{tp:,}", f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}",
                         f"{wrong.index[0]} ({wrong.iloc[0]})" if len(wrong) else "–"])
        return {"caption": (f"Per-class out-of-fold precision, recall and F1 of the selected pipeline ({sel['label']}). "
                            f"Most frequent error gives the predicted class and count for misclassified samples of each "
                            f"true class."),
                "columns": ["Class", "n", "Correct", "Precision", "Recall", "F1", "Most frequent error"],
                "rows": rows, "align": "lrrrrrl", "bold": []}
    if name == "metrics":
        need(run, "metrics needs --run")
        ev = run["evidence"]
        lower = ev.get("metric_direction") == "minimize"
        models = sorted(run["models"], key=lambda m: (m["name"] == "dummy", -m["mean"] if not lower else m["mean"]))
        rows = []
        if ev["task"] == "classification":
            from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
            columns = ["Pipeline", "Accuracy", "Balanced acc.", "Macro-F1", "Weighted F1"]
            for m in models:
                o = _oof(ctx.run_dir, m)
                rows.append([m["label"], f"{accuracy_score(o['y_true'], o['prediction']):.3f}",
                             f"{balanced_accuracy_score(o['y_true'], o['prediction']):.3f}",
                             f"{f1_score(o['y_true'], o['prediction'], average='macro', zero_division=0):.3f}",
                             f"{f1_score(o['y_true'], o['prediction'], average='weighted', zero_division=0):.3f}"])
        else:
            columns = ["Pipeline", "RMSE", "MAE", "R²"]
            for m in models:
                o = pd.read_csv(ctx.run_dir / m["oof_file"])
                o = o[o["fold"] >= 0]
                yv, pv = o["y_true"].astype(float).to_numpy(), o["prediction"].astype(float).to_numpy()
                r2 = 1 - np.sum((yv - pv) ** 2) / np.sum((yv - yv.mean()) ** 2)
                rows.append([m["label"], f"{np.sqrt(np.mean((yv - pv) ** 2)):.3g}", f"{np.mean(np.abs(yv - pv)):.3g}",
                             f"{r2:.3f}"])
        best = {j: (min if columns[j] in ("RMSE", "MAE") else max)(
            float(r[j]) for r, m in zip(rows, models) if m["name"] != "dummy") for j in range(1, len(columns))}
        bold = [i for i, (r, m) in enumerate(zip(rows, models))
                if m["name"] != "dummy" and float(r[1]) == best[1]]
        return {"caption": (f"Additional metrics on the pooled out-of-fold predictions ({f['cv_design']}). Only "
                            f"{f['metric_lower']} was used for selection; the other columns are reported for "
                            f"completeness and were not used to choose the pipeline. Bold marks the best value in the "
                            f"first column."),
                "columns": columns, "rows": rows, "align": "l" + "r" * (len(columns) - 1), "bold": bold}
    if name == "importance":
        need(run, "importance needs --run")
        imp = importance_frame(ctx.run_dir, run)
        need(imp is not None, "importance needs a run whose candidates store feature weights (evaluate_models.py)")
        frame, exp, fallback = imp
        label = next((m["label"] for m in run["models"] if m["name"] == exp["name"]), exp["name"])
        weight_cols = [c for c in frame if c.startswith("weight_")]
        n_folds = len(exp["folds"])
        rows = []
        for rank, (_, r) in enumerate(frame.head(10).iterrows(), 1):
            strongest = max(weight_cols, key=lambda c: abs(r[c]))
            rows.append([str(rank), str(r["feature"]), f"{r[strongest]:+.4f}",
                         strongest.removeprefix("weight_") if len(weight_cols) > 1 else "–",
                         f"{int(round(r['selection_frequency'] * n_folds))}/{n_folds}"])
        return {"caption": (f"Ten features with the largest fold-averaged weight in {label}"
                            f"{' (best candidate with feature weights; the selected pipeline has none)' if fallback else ''}. "
                            f"Weight is the {exp['importance_kind'].replace('_', ' ')} for the class shown; folds gives "
                            f"how many fold models selected the feature. Weights are model associations, shared among "
                            f"correlated features, not causal effects."),
                "columns": ["Rank", "Feature", "Weight", "Class", "Folds"],
                "rows": rows, "align": "rlrlr", "bold": []}
    if name == "top_features":
        need(ctx.structure_dir and pr.load_associations(ctx.structure_dir), "top_features needs --structure with associations")
        assoc, s = pr.load_associations(ctx.structure_dir)
        classes = [c["label"] for c in s["class_order"]]
        head = assoc[assoc["test_status"] == "finite_test"].sort_values("F", ascending=False).head(10)
        rows = []
        for rank, (_, r) in enumerate(head.iterrows(), 1):
            means = [r[f"mean_class_{i}"] for i in range(len(classes))]
            q = r["q_bh"]
            rows.append([str(rank), str(r["feature"]), f"{r['F']:,.0f}", f"{r['eta_squared']:.3f}",
                         "< 1e-300" if q == 0 else f"{q:.1e}", classes[int(np.argmax(means))]])
        return {"caption": (f"Ten features with the largest one-way ANOVA F statistic in the exploratory all-data screen "
                            f"(BH over {s['tested_family_size']:,} tests). q-values that underflow are shown as < 1e-300. "
                            f"Highest-mean class is descriptive; the omnibus test does not identify which class differs."),
                "columns": ["Rank", "Feature", "F", "η²", "q (BH)", "Highest-mean class"],
                "rows": rows, "align": "rlrrrl", "bold": []}
    raise ReportError(f"Unknown table {name!r}")


def table_frame(ctx: Context, name: str) -> pd.DataFrame:
    t = table(ctx, name)
    return pd.DataFrame(t["rows"], columns=t["columns"])


def need(condition, message: str) -> None:
    if not condition:
        raise ReportError(message)


# ------------------------------------------------------------ numbering

@dataclass
class Numbering:
    figures: dict
    tables: dict
    sections: dict


def number(source: Source) -> Numbering:
    figures, tables, sections, counters = {}, {}, {}, [0, 0, 0]
    appendix = False
    for b in source.blocks:
        if b.kind == "figure" and b.text not in figures:
            figures[b.text] = len(figures) + 1
        elif b.kind == "table" and b.text not in tables:
            tables[b.text] = len(tables) + 1
        elif b.kind == "heading":
            if b.level == 1 and b.text.lower() == "appendix":
                appendix, counters = True, [0, 0, 0]
                continue
            if appendix and b.level > 1 and not b.options.get("promoted"):
                # '## X' under '# Appendix' becomes appendix section A, B, ...
                b.level -= 1
                b.options["promoted"] = True
            counters[b.level - 1] += 1
            for j in range(b.level, 3):
                counters[j] = 0
            first = chr(64 + counters[0]) if appendix else str(counters[0])
            b.options["number"] = ".".join([first] + [str(c) for c in counters[1:b.level]])
            if b.label:
                sections[b.label] = b.options["number"]
    return Numbering(figures, tables, sections)


# ------------------------------------------------------- inline conversion

PLACEHOLDER = re.compile(r"\{\{\s*([\w]+)\s*\}\}")
CITE = re.compile(r"\[(@[\w:.-]+(?:\s*;\s*@[\w:.-]+)*)\]")
XREF = re.compile(r"@(fig|tab|sec):([\w-]+)")
MATH = re.compile(r"(\$[^$]+\$)")


def fill(text: str, facts: dict) -> str:
    def sub(m):
        key = m.group(1)
        if key not in facts:
            raise ReportError(f"Unknown fact {{{{{key}}}}}. Available: {', '.join(sorted(facts))}")
        return facts[key]
    return PLACEHOLDER.sub(sub, text)


def cited_keys(text: str) -> list[str]:
    return [k.strip().lstrip("@") for m in CITE.finditer(text) for k in m.group(1).split(";")]


TEX_UNICODE = {"η²": r"$\eta^2$", "η": r"$\eta$", "²": r"\textsuperscript{2}", "×": r"$\times$", "±": r"$\pm$",
               "≤": r"$\leq$", "≥": r"$\geq$", "→": r"$\rightarrow$", "≈": r"$\approx$", "α": r"$\alpha$",
               "μ": r"$\mu$", "·": r"$\cdot$", "−": "--", "–": "--", "—": "---", "…": r"\ldots{}", "’": "'",
               "“": "``", "”": "''"}


def tex_escape(text: str) -> str:
    out = []
    for ch in text:
        out.append({"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "{": r"\{",
                    "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "$": r"\$"}.get(ch, ch))
    text = "".join(out)
    for k in sorted(TEX_UNICODE, key=len, reverse=True):
        text = text.replace(k, TEX_UNICODE[k])
    return re.sub(r'"([^"]+)"', r"``\1''", text)


def _emphasis(text: str, fmt: str) -> str:
    if fmt == "tex":
        text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
        text = re.sub(r"\*(.+?)\*", r"\\emph{\1}", text)
    else:
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    return text


def inline(text: str, fmt: str, facts: dict, numbering: Numbering, refs: dict) -> str:
    """Convert one prose string to 'tex', 'html' or 'md' (notebook)."""
    text = fill(text, facts)
    for key in cited_keys(text):
        if key not in refs:
            raise ReportError(f"Citation [@{key}] has no entry under '# References'.")
    order = list(refs)

    def cite(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        if fmt == "tex":
            return "\\citep{" + ",".join(keys) + "}"
        nums = [str(order.index(k) + 1) for k in keys]
        if fmt == "html":
            return "[" + ", ".join(f'<a href="#ref-{k}">{n}</a>' for k, n in zip(keys, nums)) + "]"
        return "[" + ", ".join(nums) + "]"

    def xref(m):
        kind, name = m.groups()
        table_ = {"fig": numbering.figures, "tab": numbering.tables, "sec": numbering.sections}[kind]
        if name not in table_:
            raise ReportError(f"Cross-reference @{kind}:{name} has no matching {kind} in the source.")
        word = {"fig": "Figure", "tab": "Table", "sec": "Section"}[kind]
        if fmt == "tex":
            return f"{word}~\\ref{{{kind}:{name}}}"
        if fmt == "html":
            return f'<a href="#{kind}-{name}">{word} {table_[name]}</a>'
        return f"{word} {table_[name]}"

    pieces = MATH.split(text)
    out = []
    for piece in pieces:
        if MATH.fullmatch(piece):
            out.append(piece if fmt != "html" else html.escape(piece, quote=False))
            continue
        tokens = re.split(r"(`[^`]+`|\[@[^\]]+\]|@(?:fig|tab|sec):[\w-]+)", piece)
        conv = []
        for tok in tokens:
            if tok.startswith("`") and tok.endswith("`") and len(tok) > 1:
                code = tok[1:-1]
                conv.append(f"\\texttt{{{tex_escape(code)}}}" if fmt == "tex" else
                            f"<code>{html.escape(code)}</code>" if fmt == "html" else tok)
            elif CITE.fullmatch(tok):
                conv.append(CITE.sub(cite, tok))
            elif XREF.fullmatch(tok):
                conv.append(XREF.sub(xref, tok))
            else:
                body = tex_escape(tok) if fmt == "tex" else html.escape(tok, quote=False) if fmt == "html" else tok
                conv.append(_emphasis(body, fmt) if fmt != "md" else body)
        out.append("".join(conv))
    return "".join(out)
