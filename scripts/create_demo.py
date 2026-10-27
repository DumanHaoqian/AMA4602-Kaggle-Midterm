#!/usr/bin/env python3
"""Create a small synthetic rehearsal, with fictional identity and no downloads."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import shutil
import sys


def write_csv(path: Path, columns: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are refused.")
    parser.add_argument("--task", choices=("classification", "regression"), default="classification")
    parser.add_argument("--mixed", action="store_true", help="Include low/high-cardinality categorical predictors.")
    args = parser.parse_args()
    output = args.output_dir.expanduser().absolute()
    created = False
    try:
        if output.exists() or output.is_symlink():
            raise ValueError(f"Output already exists: {output}")
        output.mkdir(parents=True, exist_ok=False)
        created = True
        (output / "data").mkdir()
        rng = random.Random(42)
        columns = ["sample_id"] + [f"x{i}" for i in range(6)] + ["constant"]
        if args.mixed:
            columns += ["site", "batch"]
        train, test = [], []
        for i in range(108):
            cls = i % 3
            features = [round(rng.gauss(cls * (j % 2 + 0.5), 1), 5) for j in range(6)]
            target = f"C{cls}" if args.task == "classification" else round(3 * features[0] - 2 * features[1] + rng.gauss(0, 0.2), 5)
            if i % 13 == 0:
                features[2] = ""
            # Leading-zero IDs exercise lexical preservation; the test includes literal NA.
            row = [("NA" if i == 107 else f"{i:05d}")] + features + [1]
            if args.mixed:
                row += [f"site_{i % 4}", f"batch_{i % 37}"]
            if i < 90:
                train.append(row + [target])
            else:
                test.append(row)
        write_csv(output / "data/train.csv", columns + ["target"], train)
        write_csv(output / "data/test.csv", columns, test)
        write_csv(output / "data/sample_submission.csv", ["sample_id", "target"],
                  [[row[0], "C0" if args.task == "classification" else 0] for row in reversed(test)])
        config = {
            "task": args.task, "train_path": "data/train.csv", "test_path": "data/test.csv",
            "labels_path": None, "target": "target", "id_column": "sample_id",
            "metric": "accuracy" if args.task == "classification" else "rmse",
            "split": {"method": "stratified" if args.task == "classification" else "kfold", "n_splits": 3},
            "sample_submission_path": "data/sample_submission.csv", "submission_id_column": "sample_id",
            "prediction": {"kind": "label" if args.task == "classification" else "regression", "columns": ["target"], "class_order": None},
            "student_id": "DEMO0000", "full_name": "Demo Student", "competition": "",
            "deadline": None, "seed": 42, "threads": 1, "budget_seconds": 60,
            "output_dir": "artifacts", "n_clusters": 3,
            "notes": "Synthetic offline rehearsal only. Identity is fictional; never upload these files to a competition.",
        }
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"config": str(output / "config.json"), "task": args.task,
                          "mixed": args.mixed, "train_rows": 90, "test_rows": 18}, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        if created:
            shutil.rmtree(output)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
