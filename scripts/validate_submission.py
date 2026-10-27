#!/usr/bin/env python3
"""Assemble or validate one local exam CSV; never upload or alter predictions."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from exam_io import atomic_json, load_config, load_data


PROBABILITY_SUM_ATOL = 1e-6


def read_csv_exact(path: str | Path) -> pd.DataFrame:
    """Preserve every ID/label lexeme, including leading zeroes and literal NA."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), None)
    if not header:
        raise ValueError(f"Empty CSV: {path.name}.")
    if any(not name for name in header):
        raise ValueError(f"Empty column name in {path.name}; supply an explicit submission schema.")
    if len(set(header)) != len(header):
        raise ValueError(f"Duplicate column names in {path.name}.")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str,
                        keep_default_na=False, na_filter=False)
    if frame.empty:
        raise ValueError(f"No prediction rows in {path.name}.")
    return frame


def label_key(value) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        raise ValueError("Missing or non-finite class label in metadata.")
    return str(value)


def unique_labels(values, context: str) -> list[str]:
    if not isinstance(values, (list, tuple, np.ndarray, pd.Series)):
        raise ValueError(f"{context} must be a list of actual class labels.")
    keys = [label_key(v) for v in values]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError(f"{context} must contain each class exactly once.")
    return keys


def require_ids(values, context: str) -> list[str]:
    keys = [label_key(v) for v in values]
    if any(not key.strip() for key in keys):
        raise ValueError(f"{context} contains an empty ID.")
    if len(keys) != len(set(keys)):
        raise ValueError(f"{context} contains duplicate IDs; no rows will be dropped.")
    return keys


def require_same_ids(actual: list[str], expected: list[str], context: str) -> None:
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    if len(actual) != len(expected) or missing or extra:
        raise ValueError(f"{context}: ID sets differ ({len(missing)} missing, "
                         f"{len(extra)} extra; {len(actual)} rows versus {len(expected)}).")


def official_filename(cfg: dict) -> str:
    student_id, full_name = cfg.get("student_id"), cfg.get("full_name")
    if not isinstance(student_id, str) or not student_id.strip():
        raise ValueError("Set student_id as the exact nonempty official string; numeric IDs may lose zeroes.")
    if not isinstance(full_name, str) or not full_name.strip():
        raise ValueError("Set full_name to the university-record English name; it will never be invented.")
    if student_id != student_id.strip() or re.search(r"\s", student_id):
        raise ValueError("student_id must not contain whitespace; preserve its official spelling and case.")
    for field, value in (("student_id", student_id), ("full_name", full_name)):
        if re.search(r'[<>:"/\\|?*\x00-\x1f]', value):
            raise ValueError(f"{field} contains an unsafe filename/path character.")
        if value.strip() in (".", ".."):
            raise ValueError(f"{field} cannot be a path component.")
    normalized_name = re.sub(r"\s+", "_", full_name.strip())
    filename = f"{student_id}_{normalized_name}.csv"
    if len(filename.encode("utf-8")) > 240:
        raise ValueError("Official submission filename is too long for a portable filename.")
    return filename


def submission_contract(cfg: dict, bundle) -> dict:
    if bundle.X_test is None or bundle.test_ids is None:
        raise ValueError("test_path and test data are required to validate prediction rows and IDs.")
    test_ids = require_ids(bundle.test_ids, "test data")
    pred = cfg.get("prediction")
    if not isinstance(pred, dict):
        raise ValueError("Set prediction.kind and the exact prediction.columns in config.")
    kind = pred.get("kind")
    if kind not in ("label", "probability", "regression"):
        raise ValueError("prediction.kind must be label, probability, or regression.")
    if (cfg["task"] == "regression") != (kind == "regression"):
        raise ValueError("Regression requires kind=regression; classification requires label or probability.")
    id_column = cfg.get("submission_id_column")
    if id_column is not None and (not isinstance(id_column, str) or not id_column):
        raise ValueError("submission_id_column must be an exact nonempty column name or null.")
    declared = pred.get("columns")
    if declared is not None:
        if not isinstance(declared, list) or not declared or any(not isinstance(v, str) or not v for v in declared):
            raise ValueError("prediction.columns must be a nonempty list of exact column names.")
        if len(set(declared)) != len(declared):
            raise ValueError("prediction.columns contains duplicate column names.")
    template_path = cfg.get("sample_submission_path")
    if template_path:
        template = read_csv_exact(template_path)
        if len(template) != len(test_ids):
            raise ValueError("Sample submission row count does not equal the test row count.")
        if id_column is not None and id_column not in template.columns:
            raise ValueError("Configured submission_id_column is absent from the sample submission.")
        prediction_columns = [c for c in template.columns if c != id_column]
        if declared is not None and declared != prediction_columns:
            raise ValueError("prediction.columns/order disagrees with the authoritative sample submission.")
        columns = template.columns.tolist()
        if id_column is not None:
            target_ids = require_ids(template[id_column], "sample submission")
            require_same_ids(target_ids, test_ids, "Sample submission versus test data")
            alignment = "one_to_one_ID_join_in_sample_submission_order"
        else:
            target_ids = test_ids
            alignment = "test_row_order_no_submission_ID_column"
    else:
        if declared is None:
            raise ValueError("Without a sample submission, prediction.columns is required.")
        prediction_columns = declared
        if id_column in prediction_columns:
            raise ValueError("The submission ID column cannot also be a prediction column.")
        columns = ([id_column] if id_column is not None else []) + prediction_columns
        target_ids = test_ids
        alignment = "test_ID_order_no_sample_template" if id_column else "test_row_order_no_submission_ID_column"
    if not prediction_columns:
        raise ValueError("No prediction columns remain in the submission schema.")
    if kind in ("label", "regression") and len(prediction_columns) != 1:
        raise ValueError("This validator supports one label or regression target column.")
    observed = [] if kind == "regression" else unique_labels(pd.unique(bundle.y), "Observed training classes")
    target_classes = None
    if kind == "probability":
        if len(prediction_columns) == 1:
            positive = cfg.get("positive_class")
            if positive is None or len(observed) != 2:
                raise ValueError("Single-column probabilities require binary classes and explicit positive_class.")
            positive = label_key(positive)
            if positive not in observed:
                # Accept 1 versus 1.0 by numeric equality, mapped to the observed spelling; nothing else is guessed.
                try:
                    numeric = [c for c in observed if float(c) == float(positive)]
                except ValueError:
                    numeric = []
                if len(numeric) != 1:
                    raise ValueError("positive_class is not an observed training class.")
                positive = numeric[0]
            target_classes = [positive]
            if pred.get("class_order") is not None:
                declared_classes = unique_labels(pred["class_order"], "prediction.class_order")
                if declared_classes != target_classes:
                    raise ValueError("For a single probability column, class_order must be [positive_class] or null.")
        else:
            if pred.get("class_order") is not None:
                target_classes = unique_labels(pred["class_order"], "prediction.class_order")
            elif set(prediction_columns) == set(observed):
                # Exact class-name matching is semantic mapping, never an index guess.
                target_classes = list(prediction_columns)
            else:
                raise ValueError("Set prediction.class_order: probability column names do not identify their classes.")
            if len(target_classes) != len(prediction_columns) or set(target_classes) != set(observed):
                raise ValueError("Probability output columns/class_order must cover all observed classes exactly once.")
    return {"kind": kind, "columns": columns, "prediction_columns": prediction_columns,
            "id_column": id_column, "test_ids": test_ids, "target_ids": target_ids,
            "id_alignment": alignment, "observed_classes": observed,
            "target_class_order": target_classes, "sample_submission_path": template_path}


def numeric_values(frame: pd.DataFrame, columns: list[str], context: str) -> np.ndarray:
    try:
        values = frame.loc[:, columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{context} contains missing or nonnumeric predictions.") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{context} contains NaN or infinite predictions.")
    return values


def check_probabilities(values: np.ndarray, context: str) -> None:
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError(f"{context}: probabilities must lie in [0, 1]; clipping is not performed.")
    if values.shape[1] > 1 and not np.allclose(values.sum(axis=1), 1.0, atol=PROBABILITY_SUM_ATOL, rtol=0):
        raise ValueError(f"{context}: class probabilities do not sum to 1 within {PROBABILITY_SUM_ATOL}; no renormalization.")


def assemble(predictions_path: Path, evidence_path: Path | None, contract: dict) -> tuple[pd.DataFrame, dict]:
    source = read_csv_exact(predictions_path)
    if "row_id" not in source:
        raise ValueError("Model predictions require an explicit row_id column.")
    source_ids = require_ids(source["row_id"], "model predictions")
    require_same_ids(source_ids, contract["test_ids"], "Model predictions versus test data")
    source = source.set_index("row_id").loc[contract["target_ids"]]
    mapping = {"source_predictions": str(predictions_path),
               "source_rows_reordered": source_ids != contract["target_ids"]}
    result = pd.DataFrame(index=range(len(source)))
    if contract["id_column"] is not None:
        result[contract["id_column"]] = contract["target_ids"]
    if contract["kind"] in ("label", "regression"):
        if source.columns.tolist() != ["prediction"]:
            raise ValueError("Label/regression model CSV must contain only row_id,prediction.")
        result[contract["prediction_columns"][0]] = source["prediction"].to_numpy()
    else:
        if evidence_path is None:
            evidence_path = predictions_path.parent / "evidence.json"
        if not evidence_path.is_file():
            raise ValueError("Probability assembly requires --evidence or adjacent evidence.json with class_order.")
        with evidence_path.open(encoding="utf-8") as handle:
            evidence = json.load(handle)
        if not isinstance(evidence, dict):
            raise ValueError("Evidence must be a JSON object containing source class_order.")
        source_classes = unique_labels(evidence.get("class_order"), "evidence.class_order")
        if set(source_classes) != set(contract["observed_classes"]):
            raise ValueError("Source class_order must match the observed training classes exactly.")
        probability_columns = [f"p_{i}" for i in range(len(source_classes))]
        if set(source.columns) != set(probability_columns):
            raise ValueError("Probability model CSV must contain row_id and exactly p_0 ... p_(n_classes-1).")
        source_values = numeric_values(source, probability_columns, "Source probabilities")
        check_probabilities(source_values, "Source probabilities")
        for output_column, actual_class in zip(contract["prediction_columns"], contract["target_class_order"]):
            result[output_column] = source_values[:, source_classes.index(actual_class)]
        mapping.update({"source_evidence": str(evidence_path.resolve()),
                        "source_class_order": source_classes,
                        "output_class_order": contract["target_class_order"]})
    return result.loc[:, contract["columns"]], mapping


def validate_frame(frame: pd.DataFrame, contract: dict) -> None:
    if frame.columns.tolist() != contract["columns"]:
        raise ValueError("Submission column names/order differ from the required schema.")
    if len(frame) != len(contract["target_ids"]):
        raise ValueError("Submission row count differs from the required test/template count.")
    if contract["id_column"] is not None:
        actual_ids = require_ids(frame[contract["id_column"]], "submission")
        require_same_ids(actual_ids, contract["target_ids"], "Submission IDs")
        if actual_ids != contract["target_ids"]:
            raise ValueError("Submission IDs are not in required template/test order; validate-only mode will not reorder them.")
    if contract["kind"] == "label":
        labels = [label_key(v) for v in frame[contract["prediction_columns"][0]]]
        unexpected = set(labels) - set(contract["observed_classes"])
        if unexpected:
            raise ValueError(f"Submission contains {len(unexpected)} labels outside the observed training label set.")
    else:
        values = numeric_values(frame, contract["prediction_columns"], "Submission")
        if contract["kind"] == "probability":
            check_probabilities(values, "Submission")


def publish_csv(frame: pd.DataFrame, destination: Path, contract: dict) -> None:
    """Validate serialized bytes before atomic publication, without replacing any file."""
    if destination.exists():
        raise ValueError(f"Refusing to overwrite {destination.name}; select a fresh --output-dir.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".validated-submission-", suffix=".csv", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False, lineterminator="\n")
            handle.flush()
            os.fsync(handle.fileno())
        validate_frame(read_csv_exact(temporary), contract)
        if os.name == "nt":
            # Windows rename fails if destination already exists.
            os.rename(temporary, destination)
        else:
            # An exclusive hard-link publication is atomic and cannot overwrite a race winner.
            os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def receipt_destination(directory: Path) -> Path:
    candidate = directory / "local_receipt.json"
    if candidate.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        candidate = directory / f"local_receipt_{stamp}_{uuid.uuid4().hex[:8]}.json"
    return candidate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", help="Model CSV with row_id and prediction or p_0,p_1,...")
    source.add_argument("--submission", help="Validate an existing correctly named CSV without changing it")
    parser.add_argument("--evidence", help="JSON containing source probability class_order")
    parser.add_argument("--output-dir", help="Destination for assembled CSV and/or a new local receipt")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        filename = official_filename(cfg)
        bundle = load_data(cfg, include_test=True)
        contract = submission_contract(cfg, bundle)
        input_path = Path(args.predictions or args.submission).expanduser().resolve()
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_path.parent
        if args.submission:
            if input_path.name != filename:
                raise ValueError(f"Submission filename must be exactly {filename!r}; validate-only mode does not rename it.")
            frame = read_csv_exact(input_path)
            validate_frame(frame, contract)
            final_path = input_path
            mapping = {"validation_mode": "existing_file_not_modified"}
        else:
            evidence_path = Path(args.evidence).expanduser().resolve() if args.evidence else None
            frame, mapping = assemble(input_path, evidence_path, contract)
            validate_frame(frame, contract)
            final_path = output_dir / filename
            publish_csv(frame, final_path, contract)
            mapping["validation_mode"] = "assembled_and_serialization_checked"
        output_dir.mkdir(parents=True, exist_ok=True)
        note = None
        if contract["id_column"] is None:
            note = ("The submission schema contains no ID column. Rows follow test-file order. "
                    "Assembly aligns model row_id to test_ids; an existing ID-free CSV cannot "
                    "independently prove prediction-to-row identity.")
        receipt = {"status": "locally_validated", "uploaded": False,
                   "validated_at": datetime.now(timezone.utc).isoformat(),
                   "competition": cfg.get("competition"), "filename": filename,
                   "file_path": str(final_path), "size_bytes": final_path.stat().st_size,
                   "row_count": len(frame), "task": cfg["task"],
                   "prediction_kind": contract["kind"],
                   "schema": {"columns": contract["columns"],
                              "prediction_columns": contract["prediction_columns"],
                              "submission_id_column": contract["id_column"],
                              "class_order": contract["target_class_order"]},
                   "id_alignment": contract["id_alignment"],
                   "row_identity_note": note,
                   "sample_submission_path": contract["sample_submission_path"],
                   "config_path": cfg["_config_path"], **mapping}
        receipt_path = receipt_destination(output_dir)
        receipt["receipt_path"] = str(receipt_path)
        atomic_json(receipt_path, receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
