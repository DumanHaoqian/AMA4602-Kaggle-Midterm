# Publication-quality figures

## Contents

1. When to draw, and the one-command route
2. The design system (colour, type, size, export)
3. Figure catalogue and what each may claim
4. Titles and captions in English
5. Hand-made figures in a notebook
6. Final review checklist

The aim is the visual standard of Anthropic's interpretability articles on transformer-circuits.pub (for example "Scaling Monosemanticity" and "Superposition, Memorization, and Double Descent"): warm ivory panels, warm-grey chrome, near-black ink, Claude clay as the single accent with everything else in grey, an orange "activation" ramp for magnitude, small upper-case panel captions, and English text whose numbers come from the run being described. Taste here means restraint. Data carries the ink; everything else recedes.

## 1. When to draw, and the one-command route

Draw only when the exam asks for figures or a figure clearly earns marks. Do it after a valid submission exists (the 60–85 minute block). A sentence beats a chart for one or two numbers; a small table beats a chart for a 3–5 item ranking.

All standard figures come from finished runs, without refitting anything:

~~~sh
python scripts/plot_results.py --run RUN_DIR --structure STRUCTURE_DIR
python scripts/plot_results.py --run RUN_DIR --figures models,confusion      # prediction only
python scripts/plot_results.py --run RUN_DIR --structure STRUCTURE_DIR --no-titles  # paper mode
~~~

RUN_DIR is an evaluate_models.py run (contains evidence.json); STRUCTURE_DIR is an analyze_structure.py output. The raw-data figures (distributions, heatmap) reload the training data through the structure run's config_path, or an explicit --config. Output goes to a new figures_<UTC> directory: a vector PDF and a 300 dpi PNG per figure, captions.md, and figures_manifest.json with input paths and sizes. Existing files are never overwritten.

Read the result before using it: open each PNG and check for label collisions, clipped text and a title that matches the data. A figure that rendered is not a figure that was reviewed.

## 2. The design system

All values live in scripts/viz_style.py; import it rather than restating them.

### Colour by the job it does

| Job | Encoding | Value |
|---|---|---|
| Surface and chrome | Ivory figure background, warm greys | surface `#FAF9F5`; regime bands `#F0EFEB`, `#E5E5E1`; grid/borders `#E6E4DD`; axes `#B9B7AE` |
| Ink | Near-black and warm greys | `#1F1F1E` primary, `#5E5D59` secondary, `#8C8B86` tertiary |
| Identity (classes) | Categorical, fixed order, never cycled | clay `#D97757`, sky `#4F8FCF`, fig `#A65A8C`, olive `#6B8E3A`, violet `#6C63B8`, ochre `#C08A2A` |
| Emphasis | Claude clay for the one series that matters, everything else grey | accent `#D97757`; neutral `#A8A69D` |
| Magnitude (shares, counts) | Sequential orange "activation" ramp | `ama_seq`: `#FBF6EF` to `#C4511B` |
| Signed deviation around a real centre (z-scores) | Diverging sky ↔ warm grey ↔ clay | `ama_div`: `#2A5E95` … `#F0EFEB` … `#B5502A` |

Surface, ink, clay and the orange ramp are sampled from the two articles' CSS and figure pixels. The six categorical hues are earth tones in the same family, adjusted until they passed a colour-vision validator on the ivory surface: every adjacent pair keeps OKLab ΔE ≥ 10.2 under simulated deutan/protan/tritan vision and ≥ 17.9 for normal vision, and the first three (clay, sky, fig) also pass all-pairs. Clay and olive form a red–green pair under protanopia, which is why olive sits fourth. Scatter plots always add marker shapes (viz_style.MARKERS). Clay and ochre sit just under 3:1 contrast on ivory, so they always appear with a legend or direct label. White text is used on heatmap cells only above a 0.72 share, where the ramp gives at least about 4.5:1.

Rules that follow from this:

- A class keeps the same colour in every figure (slot = position in sorted label order). Filtering never repaints survivors.
- More than six classes: the five largest keep colours and the rest fold into "Other", or use small multiples. Never generate a seventh hue.
- No rainbow/jet, no red–green pairs carrying meaning, no hue at a diverging midpoint, no diverging map without a meaningful centre.
- Text stays in ink colours (`#1F1F1E`, `#5E5D59`, `#8C8B86`), never in a series colour. The one exception is the highlighted model's label and value.

### Type

macOS Helvetica Neue ships the needed weights, so one family covers everything (fallbacks: Helvetica, Arial, DejaVu Sans):

| Element | Face | Size (printed) |
|---|---|---|
| Figure title | Helvetica Neue Bold (700), near-black | 11 pt |
| Subtitle | Helvetica Neue Regular, grey | 8.5 pt |
| Axis titles | Helvetica Neue Medium (500), near-black | 9 pt |
| Panel captions | Upper-case Helvetica Neue Medium, warm grey (viz_style.panel_title), as in transformer-circuits figures | 7.5 pt |
| Ticks, legends, in-cell numbers | Helvetica Neue Regular, dark grey | 7.5–8.5 pt |
| Panel tags (a), (b) | Helvetica Neue Bold | 10 pt |
| Mathematics | STIX (mathtext), matching STIX Two | — |

Sizes are tokens in viz_style.FS; use them instead of literal numbers. NeurIPS body text is 10 pt, so the smallest figure text stays at least 7.5 pt in print. Avoid rotated labels; flip to horizontal bars or abbreviate instead. Use a real minus sign and thin thousands separators (19,565). PDF output embeds fonts as TrueType (Type 42), so text stays selectable and editable.

### Size and chrome

- Design at final size: full width 5.5 in (the NeurIPS text width), half width 2.7 in. Never scale a figure after export, because that changes every font size.
- Remove top/right spines; axes and ticks are thin light-grey rules (`#B9B7AE`, 0.8 pt). Use a hairline grid (`#E6E4DD`) only on the axis where values must be read.
- Separate cells and adjacent fills with surface-coloured (ivory) gaps, not black borders. Small markers get a thin ivory ring so overlapping points stay legible.
- Points/lines may use a non-zero axis; bars and filled areas start at zero.
- Dense scatters and heatmaps are rasterised inside the PDF while text and axes stay vector.
- No 3D, shadows, gradients, pies, dual y-axes or decorative backgrounds.

## 3. Figure catalogue and what each may claim

| Figure (stem) | Shows | May claim | Must not claim |
|---|---|---|---|
| fig_model_comparison | Per-fold and mean selection-CV score per pipeline, selected one highlighted, majority baseline in subtitle | Ranking on these frozen folds | An independent test score; fold SD as a confidence interval |
| fig_confusion_matrix | Pooled OOF counts, row-share shading, per-class recall | Which classes the selected pipeline confuses under CV | Private-leaderboard performance |
| fig_oof_regression | Observed vs OOF prediction with y = x, R², RMSE | Calibration of OOF predictions | Test performance |
| fig_unsupervised_structure | (a) k-means clusters on PC1–PC2, (b) labels overlaid after fitting, (c) label × cluster counts; ARI/NMI | Degree of label-free structure | That a separated 2-PC plot validates a classifier; that cluster k equals class k (colours are matched for display only) |
| fig_class_associations | (a) ranked η² with the BH-significant share, (b) class-mean profiles of top features | How many features differ across classes in an exploratory screen | Which class drives an omnibus test; biological function; a volcano plot (it is not one) |
| fig_top_feature_distributions | Raw per-class values (jittered points, IQR box, median) of the top features | Shape and overlap of class distributions | Predictive validation (ranking used all labelled data) |
| fig_feature_heatmap | z-scored top features, samples grouped by label | Class-specific blocks given label ordering | Unsupervised recovery (the ordering uses labels) |
| fig_holdout | Confusion matrix (or observed vs predicted) of the selected pipeline on the untouched holdout | The single independent estimate in the run | Precision beyond its small n; that other candidates would score the same |
| fig_feature_importance | Top 20 features by fold-averaged weight; classes × features signed heatmap (diverging, centred at 0) or signed bars; right column: folds that selected the feature | Which features the model leans on, and how stable that is across folds | Causal or biological roles; uniqueness (correlated features share weight); gene importance from PC coefficients without back-projection |
| fig_train_validation_gap | Mean training-row vs validation-row score per pipeline | How much each pipeline overfits its training folds | Generalisation by itself (validation is still selection CV) |
| fig_pca_variance | Per-component and cumulative explained variance of the clustering PCA | How much processed variance the retained PCs keep | Share of predictive or biological information |
| fig_cluster_robustness | (a) silhouette by k for k-means and Ward, (b) ARI/NMI/silhouette of both at the used k | Whether the label-free structure depends on the algorithm and on k | The true number of groups (silhouette is one internal criterion) |

Anonymous names such as gene_9175 stay anonymous. PC importance is never renamed gene importance.

## 4. Titles and captions in English

All figure text is English. A title states the finding; the grey subtitle states method and context; the caption states the evidence and its limits.

- Good title: "All 801 out-of-fold predictions are correct". Weak title: "Confusion matrix".
- Build every number in a title or caption from the run outputs, as plot_results.py does. Never type a remembered accuracy.
- Name ties honestly ("A and B attain the best accuracy") and name uncertainty precisely (folds, SD, CI, n).
- Use "selection CV" for scores used to choose a model, "out-of-fold" for pooled OOF predictions, "exploratory" for all-data tests.
- For a paper-style document, run with --no-titles and put "Figure N. Title. Caption." beneath each image, taken from captions.md.

## 5. Hand-made figures in a notebook

For a figure the script does not cover, reuse the style instead of inventing a new one:

~~~python
import sys; sys.path.insert(0, "/absolute/path/to/ama4602-kaggle-exam/scripts")
import viz_style as vs
from viz_style import plt
vs.apply_style()

fig, ax = plt.subplots(figsize=(vs.HALF_WIDTH, 2.4))
vs.soften(ax, grid_axis="y")
for i, (name, values) in enumerate(groups.items()):
    style = vs.category_style(i)              # colour + marker, fixed order
    ax.scatter(x, values, s=8, label=name, edgecolor="white", linewidth=0.3, **style)
ax.set(xlabel="Component 1", ylabel="Component 2")
vs.header(fig, "Finding stated as a sentence", "Method · n = 801 · what marks mean")
vs.save_figure(fig, "figures_extra", "fig_custom")   # PDF + 300 dpi PNG, no overwrite
~~~

With seaborn 0.13, pass `hue=` whenever you pass `palette=` (palette without hue is deprecated), use `errorbar=("ci", 95)` or `"sd"` instead of the removed `ci=`, and set a `seed` for bootstrap intervals. Prefer axes-level functions (boxplot, stripplot, heatmap) inside a styled matplotlib figure over figure-level ones, which create their own layout and fonts. Seaborn line plots bridge missing values; use explicit NaN gaps when missingness matters.

## 6. Final review checklist

- [ ] The figure is required or clearly useful, and a sentence or table would not do the job.
- [ ] Every number in titles/captions came from the same run as the plotted data.
- [ ] Classes keep their colours across figures; marker shapes are used from the fourth class onward.
- [ ] Sequential for magnitude, diverging only around a meaningful centre, no rainbow.
- [ ] All text is English, horizontal where possible, with no clipped or overlapping labels.
- [ ] Axes are labelled with units or scale; bars start at zero; no dual axes.
- [ ] Uncertainty is named (folds, SD, CI) and n is stated.
- [ ] The caption states the limitation (selection CV, exploratory, label-ordered).
- [ ] The PNG was opened and inspected at final size.
