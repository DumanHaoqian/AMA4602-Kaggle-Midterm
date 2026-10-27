# 2025 · PANCAN RNA-Seq analysis

An archived AMA4602 task combining high-dimensional statistical analysis,
unsupervised learning and multiclass prediction. The repository maintainer
identifies these course files as the **2025 assessment**, the year before last
relative to 2026. This year attribution comes from the supplied course context,
not from file modification dates.

The supplied description identifies the measurements as a subset of the
RNA-Seq (HiSeq) PANCAN data, with TCGA as the data source. Samples are patients;
labels identify five tumor types. This copy is provided for historical study
and reproducible practice.

## Included files

| File | Contents | File size |
|---|---|---:|
| `data.csv.gz` | Gzip-compressed, otherwise unmodified feature CSV | 72,884,238 bytes (69.51 MiB) |
| `labels.csv` | Original sample IDs and tumor labels, copied unchanged | 12,713 bytes |

The uncompressed feature file is **206,188,453 bytes** (about 196.64 MiB).
Compression keeps each committed file below GitHub's 100 MiB regular-file limit;
the dataset does not require Git LFS. Reading the entire gzip stream confirmed
that its decompressed byte count matches the source file size. No checksum is
provided or calculated.

This archive contains only the data and this guide. Another student's notebook,
HTML export, report and personal file names are not included.

## Data layout

The actual feature CSV has **801 rows and 20,533 columns**:

| Column or field | Meaning |
|---|---|
| `Unnamed: 0` | Sample identifier; join key, not a predictor |
| `Edema` | Recorded degree of edema; observed codes are `0`, `1` and `2` |
| `gene_0`–`gene_20530` | 20,531 anonymized gene-expression features |

The task description's 20,532 features refer to the expression measurements plus
`Edema`; the CSV also contains the identifier column. The label file has 801
rows, a blank first column header containing sample IDs, and the target column
`Class`. The skill normalizes the blank ID header to `Unnamed: 0` when loading.
IDs are unique, and both files contain the same sample IDs; join explicitly by
ID rather than relying on their current matching row order.

| Class | Samples |
|---|---:|
| `BRCA` | 300 |
| `KIRC` | 146 |
| `LUAD` | 141 |
| `PRAD` | 136 |
| `COAD` | 78 |
| **Total** | **801** |

The bundled rehearsal configuration excludes `Edema` to make the predictor scope
gene-expression-only. That is an explicit analysis choice, not evidence that
the field is invalid or a universal instruction to discard it in other tasks.
Inspect missing values and document their treatment; do not automatically remove
whole patients just because one measurement is missing.

The anonymized `gene_*` column names do not supply a gene-identity mapping.
Statistical association or model importance alone does not establish a pathway,
clinical mechanism or validated biomarker.

## What the task asks students to do

The supplied `Description.pdf` describes the following objectives:

1. **Association analysis:** identify expression features that differ across
   tumor types, using suitable statistical tests, effect summaries and plots.
   The shared skill's rehearsal applies Benjamini–Hochberg false-discovery-rate
   correction across the complete tested feature family; this correction is the
   skill's implementation choice rather than a named requirement in the brief.
2. **Unsupervised clustering:** apply methods such as k-means or hierarchical
   clustering, visualize the structure, and compare clusters with tumor labels.
   Fit the unsupervised representation without using those labels, then use them
   to interpret the results.
3. **Feature selection and dimensionality reduction:** investigate methods such
   as regularization, supervised feature selection and PCA, and explain what is
   retained or discarded.
4. **Predictive modelling:** classify tumor types, compare regularized logistic
   regression with alternatives such as SVM or random forest, and evaluate with
   appropriate splits or cross-validation, accuracy, F1 and confusion matrices.
   Keep learned preprocessing and feature selection inside each training fold.
5. **Interpretation and reproducible reporting:** explain the selected methods,
   parameters, results and limitations; include the full code, running results,
   visualizations, data-loading instructions and random seeds. If a neural
   network is used, state and justify its architecture and training settings.

The repository's rehearsal defaults to classification accuracy for model
selection. The historical brief discusses several evaluation measures rather
than specifying a Kaggle ranking metric. There is no separate hidden-test file
or Kaggle submission template in this dataset archive.

## Prepare a new workspace

Run the following from the **repository root**, with your Python environment
active. It uses only the Python standard library and works without `gunzip`,
including on Windows. `pancan-practice` must not already exist.

```sh
python -c "from pathlib import Path
import gzip
import shutil
source = Path('datasets/2025-pancan')
workspace = Path('pancan-practice')
workspace.mkdir()
(workspace / 'data').mkdir()
with gzip.open(source / 'data.csv.gz', 'rb') as src, (workspace / 'data' / 'data.csv').open('xb') as dst:
    shutil.copyfileobj(src, dst, length=1024 * 1024)
shutil.copyfile(source / 'labels.csv', workspace / 'data' / 'labels.csv')
shutil.copyfile('assets/pancan_config.example.json', workspace / 'config.json')
print('Ready:', workspace / 'config.json')"
```

The result is:

```text
pancan-practice/
├── config.json
└── data/
    ├── data.csv
    └── labels.csv
```

The paths in `config.json` resolve relative to that configuration file, so
`data/data.csv` means `pancan-practice/data/data.csv`. Outputs are written inside
the practice workspace, leaving this archived dataset unchanged.

## Run the shared workflow

After installing the repository's single `requirements.txt`, use the same
Python environment for each command:

```sh
python -B scripts/profile_data.py --config pancan-practice/config.json
python -B scripts/evaluate_models.py --config pancan-practice/config.json --models select_logistic,linear_svm,pca_logistic --budget-seconds 300
python -B scripts/analyze_structure.py --config pancan-practice/config.json --mode both
```

Use the actual completed model-run and structure-run directories printed by
these commands when making figures:

```sh
python -B scripts/plot_results.py --run MODEL_RUN_DIR --structure STRUCTURE_RUN_DIR
```

Copy `assets/report_example.md` into the practice workspace when preparing a
written report. Replace its placeholders, point it to the current run files,
and rewrite the narrative around the results you actually obtained. Selection
CV, exploratory association tests and label-free clustering answer different
questions; describe their scopes separately.

For the configuration and detailed workflow, see
[the PANCAN example config](../../assets/pancan_config.example.json) and
[the PANCAN reference](../../references/pancan-reference.md). Any historical
illustrative score is not a promised result for a new run.
