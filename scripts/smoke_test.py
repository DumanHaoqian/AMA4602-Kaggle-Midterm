#!/usr/bin/env python3
"""Offline release smoke checks on temporary synthetic data; no Kaggle access.

Run with the same interpreter used for the skill. Add --reports to also check
figures, TeX source, interactive HTML and an executed notebook (report extras).
The temporary workspace is removed on completion, including on failure.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


SCRIPTS = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_frame(path: Path, id_column: str) -> pd.DataFrame:
    """Read numeric missing values normally and obtain ID lexemes independently."""
    frame = pd.read_csv(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.reader(handle)
        position = next(rows).index(id_column)
        frame[id_column] = [row[position] for row in rows]
    return frame


def command(script: str, *arguments: object, fails: bool = False) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", str(SCRIPTS / script), *map(str, arguments)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=240 if script == "build_reports.py" else 150, env=env,
    )
    expected = result.returncode != 0 if fails else result.returncode == 0
    if not expected:
        raise AssertionError(
            f"{script} returned {result.returncode}; expected {'failure' if fails else 'success'}\n"
            + (result.stdout + "\n" + result.stderr)[-6500:]
        )
    return result


def source_path(workspace: Path, cfg: dict, key: str) -> Path:
    path = Path(cfg[key]).expanduser()
    return path if path.is_absolute() else workspace / path


def reference_model(task: str, name: str, seed: int) -> object:
    """Independent sklearn baseline for checking recorded held-out predictions."""
    if name == "dummy":
        return DummyClassifier(strategy="prior") if task == "classification" else DummyRegressor(strategy="mean")
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
             ("variance", VarianceThreshold())]
    if task == "classification":
        require(name == "select_logistic", f"Unexpected classification candidate {name}")
        steps += [("select", SelectKBest(f_classif, k="all")), ("scale", StandardScaler()),
                  ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=seed))]
    else:
        require(name == "ridge", f"Unexpected regression candidate {name}")
        steps += [("scale", StandardScaler()), ("model", Ridge(alpha=1.0, solver="lsqr"))]
    return Pipeline(steps)


def equal_predictions(actual: object, expected: object, task: str, context: str) -> None:
    if task == "classification":
        require(np.array_equal(np.asarray(actual, dtype=str), np.asarray(expected, dtype=str)), context)
    else:
        require(np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                            atol=1e-8, rtol=1e-7), context)


def load_raw(workspace: Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    id_column = cfg["id_column"]
    train = read_frame(source_path(workspace, cfg, "train_path"), id_column)
    test = read_frame(source_path(workspace, cfg, "test_path"), id_column)
    require(not cfg.get("labels_path"), "Synthetic smoke data should contain the target in train.csv")
    excluded = {id_column, cfg["target"], *cfg.get("drop_columns", [])}
    features = cfg.get("feature_columns") or [column for column in train if column not in excluded]
    return train[features], test[features], train[cfg["target"]], train[id_column]


def check_numeric_run(workspace: Path, cfg: dict, run: Path, candidate: str) -> None:
    task = cfg["task"]
    evidence = read_json(run / "evidence.json")
    require(evidence["status"] == "complete", "Model run did not complete")
    require(evidence["final_refit"]["status"] == "complete", "Final refit did not complete")
    X, X_test, y, train_ids = load_raw(workspace, cfg)
    folds = read_json(run / "fold_indices.json")
    require(folds["row_ids"] == train_ids.tolist(), "Recorded folds lost training row identities")
    seen = np.zeros(len(X), dtype=int)
    experiments = {experiment["name"]: experiment for experiment in evidence["experiments"]}
    require(experiments[candidate]["status"] == "complete", "Requested non-dummy model was not evaluated")
    for name in ("dummy", candidate):
        oof = read_frame(run / experiments[name]["oof_file"], "row_id")
        require(oof["row_id"].tolist() == train_ids.tolist(), "OOF rows changed order or IDs")
        require(oof["prediction"].notna().all(), "OOF predictions have uncovered rows")
        equal_predictions(oof["y_true"], y, task, "OOF truth does not match training labels by ID")
        for fold in folds["folds"]:
            tr = np.asarray(fold["train_indices"], dtype=int)
            va = np.asarray(fold["validation_indices"], dtype=int)
            require(len(set(tr)) == len(tr) and len(set(va)) == len(va), "Duplicate indices inside a fold")
            require(not set(tr).intersection(va), "Training and validation rows overlap")
            require(set(tr).union(va) == set(range(len(X))), "Fold does not partition the synthetic training set")
            require((oof.iloc[va]["fold"] == fold["fold"]).all(), "OOF fold assignment is wrong")
            # Fit only these training rows; matching held-out predictions exercises
            # the complete preprocessing boundary, not just the saved split file.
            model = reference_model(task, name, int(cfg.get("seed", 42)))
            with threadpool_limits(limits=1):
                model.fit(X.iloc[tr], y.iloc[tr])
                expected = model.predict(X.iloc[va])
            equal_predictions(oof.iloc[va]["prediction"], expected, task,
                              f"{name} OOF differs from an independent fold-only fit")
            if name == candidate:
                seen[va] += 1
    require((seen == 1).all(), "Each row must receive exactly one held-out prediction")
    selected = reference_model(task, evidence["selection"]["model"], int(cfg.get("seed", 42)))
    with threadpool_limits(limits=1):
        selected.fit(X, y)
        expected_test = selected.predict(X_test)
    prediction = read_frame(run / "test_predictions.csv", "row_id")
    test = read_frame(source_path(workspace, cfg, "test_path"), cfg["id_column"])
    require(prediction["row_id"].tolist() == test[cfg["id_column"]].tolist(), "Test prediction IDs do not follow test data")
    equal_predictions(prediction["prediction"], expected_test, task,
                      "Final predictions differ from an independent full-training fit")


def check_export(workspace: Path, cfg: dict, run: Path) -> None:
    template_path = source_path(workspace, cfg, "sample_submission_path")
    id_column = cfg["submission_id_column"]
    test = read_frame(source_path(workspace, cfg, "test_path"), cfg["id_column"])
    template = read_frame(template_path, id_column)
    # Make the identity check decisive even if the demo's initial order changes.
    template = template.set_index(id_column).loc[test[cfg["id_column"]]].iloc[::-1].reset_index()
    template.to_csv(template_path, index=False)
    require(template[id_column].tolist() != test[cfg["id_column"]].tolist(), "Template must have a different row order")
    output = workspace / "final"
    arguments = ("--config", workspace / "config.json", "--predictions", run / "test_predictions.csv",
                 "--evidence", run / "evidence.json", "--output-dir", output)
    command("validate_submission.py", *arguments)
    receipt = read_json(output / "local_receipt.json")
    require(receipt["status"] == "locally_validated" and receipt["uploaded"] is False,
            "Local receipt must not claim upload")
    require(receipt["filename"] == "DEMO0000_Demo_Student.csv", "Demo uses a non-demo identity")
    final = Path(receipt["file_path"])
    frame = read_frame(final, id_column)
    require(frame.columns.tolist() == template.columns.tolist(), "Final schema differs from the sample submission")
    require(frame[id_column].tolist() == template[id_column].tolist(), "Final CSV does not follow template IDs")
    predictions = read_frame(run / "test_predictions.csv", "row_id").set_index("row_id")
    expected = predictions.loc[frame[id_column], "prediction"]
    equal_predictions(frame[cfg["prediction"]["columns"][0]], expected, cfg["task"],
                      "Export attached predictions to the wrong sample IDs")
    if cfg["task"] == "regression":
        require(np.isfinite(frame[cfg["prediction"]["columns"][0]].astype(float)).all(), "Nonfinite exported values")
    else:
        require(set(frame[cfg["prediction"]["columns"][0]]) <= {"C0", "C1", "C2"}, "Unknown exported class")
    before = final.read_bytes()
    receipt_before = (output / "local_receipt.json").read_bytes()
    command("validate_submission.py", *arguments, fails=True)
    require(final.read_bytes() == before, "Repeated export overwrote the final CSV")
    require((output / "local_receipt.json").read_bytes() == receipt_before, "Repeated export overwrote the receipt")
    require(len(list(output.glob("local_receipt*.json"))) == 1, "Failed duplicate export produced a success receipt")


def new_run(workspace: Path, cfg: dict, before: set[Path]) -> Path:
    root = source_path(workspace, cfg, "output_dir")
    created = {path for path in root.iterdir() if (path / "evidence.json").is_file()} - before
    require(len(created) == 1, f"Expected one new model run, got {len(created)}")
    return created.pop()


def numeric(workspace: Path, task: str) -> tuple[dict, Path]:
    command("create_demo.py", "--output-dir", workspace, "--task", task)
    cfg = read_json(workspace / "config.json")
    command("profile_data.py", "--config", workspace / "config.json")
    root = source_path(workspace, cfg, "output_dir")
    before = set(root.iterdir()) if root.exists() else set()
    candidate = "select_logistic" if task == "classification" else "ridge"
    command("evaluate_models.py", "--config", workspace / "config.json", "--models", candidate,
            "--budget-seconds", 60)
    run = new_run(workspace, cfg, before)
    check_numeric_run(workspace, cfg, run, candidate)
    check_export(workspace, cfg, run)
    print(f"PASS {task}: fold-only OOF, final refit, ID-aligned export, overwrite refusal", flush=True)
    return cfg, run


def mixed(workspace: Path) -> None:
    command("create_demo.py", "--output-dir", workspace, "--task", "classification", "--mixed")
    cfg = read_json(workspace / "config.json")
    command("profile_data.py", "--config", workspace / "config.json")
    root = source_path(workspace, cfg, "output_dir")
    before = set(root.iterdir()) if root.exists() else set()
    command("leaderboard.py", "--config", workspace / "config.json", "--families", "logreg_l2",
            "--repeats", 1, "--max-trials", 1, "--seeds", 1, "--no-adversarial", "--budget-seconds", 60)
    run = new_run(workspace, cfg, before)
    evidence = read_json(run / "evidence.json")
    require(evidence["status"] == "complete", "Mixed-table run did not complete")
    require(evidence["n_categorical"] > 0 and evidence["n_numeric"] > 0, "Mixed predictors were not retained")
    train = read_frame(source_path(workspace, cfg, "train_path"), cfg["id_column"])
    require(any(train[column].nunique() > 30 for column in evidence["categorical_columns"]),
            "Synthetic mixed data did not exercise high-cardinality categorical handling")
    require(any(trial["family"] == "logreg_l2" and trial["status"] == "complete"
                for trial in evidence["trials"]), "Mixed-table linear trial did not complete")
    prediction = read_frame(run / "test_predictions.csv", "row_id")
    require(prediction["prediction"].notna().all(), "Mixed-table predictions are missing")
    check_export(workspace, cfg, run)
    print("PASS mixed table: categorical model, ID-aligned export, overwrite refusal", flush=True)


def reports(workspace: Path, run: Path) -> None:
    command("plot_results.py", "--run", run)
    report_dir = workspace / "reports"
    report_dir.mkdir()
    text = (SCRIPTS.parent / "assets" / "report_minimal.md").read_text(encoding="utf-8")
    replacements = {
        "REPLACE_WITH_TITLE": "Synthetic Classification Smoke Report",
        "REPLACE_WITH_FULL_NAME": "Demo Student",
        "REPLACE_WITH_STUDENT_ID": "DEMO0000",
        "REPLACE_WITH_TARGET_DESCRIPTION": "a synthetic three-class target",
        "REPLACE_WITH_TWO_SENTENCES_ON_THE_TASK_AND_WHY_IT_IS_HARD":
            "This report checks the offline synthetic workflow. The data are generated for software validation",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"^run:.*$", f"run: {run}", text, flags=re.MULTILINE)
    require(not re.search(r"REPLACE_[A-Z_]+", "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))),
            "Report template gained an unsupported placeholder")
    source = report_dir / "report.md"
    source.write_text(text, encoding="utf-8")
    command("build_reports.py", "--source", source, "--formats", "tex,html,ipynb", "--no-compile")
    summary = read_json(report_dir / ".report_build.json")
    require(summary["outputs"]["tex"]["status"] == "not_compiled", "TeX smoke unexpectedly compiled a PDF")
    require(summary["outputs"]["ipynb"]["status"] == "executed", "Notebook execution failed")
    notebook = read_json(report_dir / "jupyter" / "report.ipynb")
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    require(code_cells and all(cell.get("execution_count") is not None for cell in code_cells), "Some notebook code was not executed")
    require(not any(output["output_type"] == "error" for cell in code_cells for output in cell.get("outputs", [])),
            "Notebook contains an error output")
    require(any(cell.get("outputs") for cell in code_cells), "Executed notebook has no visible outputs")
    for path in (report_dir / "tex" / "report.tex", report_dir / "html" / "report.html"):
        require(path.is_file() and path.stat().st_size > 1000, f"Missing or empty report: {path.name}")
        rendered = path.read_text(encoding="utf-8")
        require(not re.search(r"REPLACE_[A-Z_]+|\{\{[a-z_]+\}\}", rendered), "Report has unresolved placeholders")
    require("plotly.js" in (report_dir / "html" / "report.html").read_text(encoding="utf-8"), "HTML lacks embedded Plotly")
    print("PASS reports: figures, TeX source, inline HTML, executed notebook (PDF compilation skipped)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", action="store_true", help="Also check figure/report dependencies and notebook execution.")
    args = parser.parse_args()
    try:
        with tempfile.TemporaryDirectory(prefix="ama4602-smoke-") as directory:
            root = Path(directory)
            _, run = numeric(root / "classification", "classification")
            numeric(root / "regression", "regression")
            mixed(root / "mixed")
            if args.reports:
                reports(root / "classification", run)
        print("PASS offline smoke checks; temporary files removed", flush=True)
        return 0
    except (AssertionError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
