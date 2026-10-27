# Reports: one source, three formats

## Contents

1. What gets built
2. Writing the source file
3. Building and checking
4. Format-specific behaviour
5. Exam-time order of work

## 1. What gets built

`scripts/build_reports.py` turns one Markdown source plus finished runs into:

| Output | Location | Content |
|---|---|---|
| NeurIPS-style paper | `<out-root>/tex/<slug>.tex` and `<slug>.pdf` | Prose, tables, captioned vector figures, references; no code |
| Executed notebook | `<out-root>/jupyter/<slug>.ipynb` | The same prose as Markdown cells, code cells that load the runs and draw every figure and table, executed outputs |
| Interactive web report | `<out-root>/html/<slug>.html` | The same prose, interactive Plotly figures (hover, view switches, feature menu), contents sidebar, single self-contained file |

Nothing is refitted. Every number comes from the run outputs, so the three formats cannot disagree.

The PDF uses the NeurIPS 2026 style shipped with this skill (`assets/neurips_2026/neurips_2026.sty`, copied next to the generated .tex on every build; the official template `neurips_2026.tex` sits beside it for reference) in `preprint` mode, which is non-anonymous and has no line numbers, and empties the first-page notice box, so neither "Submitted to … Do not distribute." nor "Preprint." appears. The user's workspace needs no LaTeX files of its own. Do not edit the style file. Compilation uses tectonic; its package cache is filled on first use (needs network once, done during preparation).

## 2. Writing the source file

Start from a copy of [assets/report_minimal.md](../assets/report_minimal.md) when the report covers prediction only (any evaluate_models.py or leaderboard.py run, classification or regression; uses only facts every run provides), or of [assets/report_example.md](../assets/report_example.md), a complete source written for the PANCAN rehearsal with association and clustering sections. Follow [academic-writing.md](academic-writing.md). Both templates contain REPLACE_ markers outside the front-matter comments; the build stops and lists them until each is replaced.

~~~markdown
---
title: Leakage-Controlled Classification of Five Tumour Types
author: Full Name From Workspace Rules
student_id: StudentIDFromWorkspaceRules
affiliation: AMA4602 Midterm Project
config: ../config.json            # paths are relative to this file
run: ../artifacts/run_...
structure: ../artifacts/structure_...
out_root: .                        # folder that holds tex/, jupyter/, html/
slug: midterm_report
---

# Abstract
One paragraph, numbers as {{facts}}.

# Introduction {#sec:intro}
Text with a citation [@pedregosa2011] and a reference to @fig:models.

## Subsection
- list items work, as do **bold**, *italic*, `code` and $\eta^2$.

$$
\eta^2 = \frac{SS_{\text{between}}}{SS_{\text{total}}}
$$

::: figure models
::: table cv_results

```python
# notebook-only code cell; omitted from the PDF, collapsible in the HTML
```

# Appendix
## Extra figures
::: figure heatmap

# References
[pedregosa2011] F. Pedregosa, et al. Scikit-learn: Machine learning in Python. *JMLR*, 12:2825–2830, 2011.
~~~

Rules the builder enforces (it stops instead of guessing):

- `{{fact}}` must exist. List them with `--list-facts`. They cover data size and classes, missing/constant counts, metric, folds, every candidate's score (`score_<model>`), the selected and tied-best pipelines, baseline, OOF errors, PCA/k-means/ARI/NMI, and the association screen. Validation facts: `n_folds` (k), `n_repeats`, `cv_design` ("5-fold stratified" or "5-fold stratified × 5 repeats"; use it rather than writing "{{n_folds}}-fold" for a leaderboard run) and `n_fold_scores`. Leaderboard runs add `n_trials`, `n_trials_aborted`, `nested_cv_blend`, `nested_cv_single`, `blend_decision`, `adversarial_auc` and `adversarial_shift`.
- `[@key]` must have a `[key]` line under `# References`; uncited references are reported as warnings.
- `@fig:name`, `@tab:name` and `@sec:label` must point to a figure, table or labelled heading in the source.
- Figures: models, confusion, holdout, importance, gap (run); structure, variance, robustness, associations, distributions, heatmap (structure). Optional `width=0.8` sets the PDF width as a fraction of the text width. holdout needs a run with `holdout` configured; importance and gap need an evaluate_models.py run (leaderboard runs store neither).
- Tables: data_summary, cv_results, metrics (accuracy, balanced accuracy, macro/weighted F1 or RMSE/MAE/R² for every candidate, from pooled OOF), class_recall (per-class precision, recall, F1), top_features, importance (top 10 features by fold-averaged weight with fold counts).
- Extra facts when the inputs exist: holdout_n, development_n, holdout_fraction, holdout_score, holdout_model, holdout_correct, holdout_errors; max_gap, max_gap_model, selected_gap; top_weight_feature, top_weight_class, top_weight_model, top20_all_folds; ward_ari, ward_nmi, ward_silhouette, kmeans_ward_ari, best_k_silhouette, best_k_ward_silhouette, k_sweep_range, pcs_for_50pct (and 80/90 when reached).
- `## ` headings directly under `# Appendix` become appendix sections A, B, …

Figure and table captions are generated from the run (with numbers and limitations); the prose interprets them and does not repeat them.

## 3. Building and checking

~~~sh
PY=<verified python interpreter>
SK=<path to this skill>/scripts
$PY -B $SK/build_reports.py --source reports/midterm_report.md --list-facts
$PY -B $SK/build_reports.py --source reports/midterm_report.md                 # all three formats
$PY -B $SK/build_reports.py --source reports/midterm_report.md --overwrite --formats tex
~~~

Front-matter paths resolve against the source file; `--run`, `--structure` and `--config` given on the command line resolve against the current directory. A full build takes about 20 seconds on the PANCAN rehearsal. Existing outputs for the same slug are protected unless `--overwrite` is given. The command prints the status of each output (`compiled`, `executed`) and writes a hidden `.<slug>_build.json` with the facts used.

Check after every build:

1. Render the PDF pages and look at them (PyMuPDF is in the venv): first page has title, author block and abstract with no footer notice or line numbers; figures and tables are legible and referenced; no `??` from broken references.
2. Open the notebook and confirm every code cell has an output and no error.
3. Open the HTML in a browser: figures respond to hover and buttons, the contents sidebar links work, maths renders (KaTeX loads from a CDN; without network the raw `$...$` text remains readable).
4. Reread the abstract and conclusion against the generated numbers.

## 4. Format-specific behaviour

- **PDF:** figures are drawn without in-figure titles (the caption carries title and description) at the NeurIPS text width of 5.5 in, so printed font sizes equal the design sizes. Tables use booktabs with the best result in bold. Code blocks are omitted.
- **Notebook:** a Setup cell loads the runs through report_source.load_context and defines `show()`. A data-overview cell follows when the config is available. Each figure or table directive becomes a code cell that draws it from the run, followed by its caption. The notebook ends with references and the exact command that produced the model comparison. It is executed during the build with the venv kernel.
- **HTML:** models, confusion, structure, associations and distributions are interactive and carry captions written for the interactive view. Other figures are embedded as high-resolution PNGs. Plotly is embedded inline, so chart interactions work offline (file size depends on the installed Plotly version). KaTeX mathematical typesetting and Google Fonts use external CDNs and need network access. Code blocks are collapsible.

## 5. Exam-time order of work

Reports come after a validated submission exists. With little time left, build only the format the brief requires (`--formats tex` or `--formats ipynb`). A sensible sequence is:

1. Runs finished and selected (evaluate_models, analyze_structure if required).
2. `plot_results.py` once to look at the figures and decide which belong in the main text.
3. Copy assets/report_example.md into the workspace, replace the front matter (identity from the workspace rules, current run paths), rewrite the prose for the actual task using facts, and choose figures/tables.
4. Build, inspect the PDF pages, fix, rebuild with `--overwrite`.
5. Only then optional polish (appendix, extra notebook code).
