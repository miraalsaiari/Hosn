#!/usr/bin/env python3
"""Train and validate the HOSN conflict-only AI from reviewed measurements.

The input must be reviewed_training_dataset.csv from the measured paired
campaign.  Only the controller's eight pre-decision features are used.  The
four replays belonging to one scenario always stay together in validation,
preventing paired-condition leakage between training and testing.

The model is a fixed, interpretable preprocessing-plus-logistic-regression
pipeline.  No hyperparameter search is performed on this small dataset.  The
cross-validation report is a development estimate, not an external field-test
claim.  A final model is accepted only if grouped validation meets declared
minimum accuracy and per-class recall checks.

Commands:
  python3 hosn_train_conflict_ai.py --self-test
  python3 hosn_train_conflict_ai.py --train
  python3 hosn_train_conflict_ai.py --inspect
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any

import hosn_heterogeneous_controller as heterogeneous


REVISION = "conflict-ai-grouped-logistic-v1"
CAMPAIGN_PREFIX = "measured_dataset_campaign_"
TRAINING_FILE = "reviewed_training_dataset.csv"
TRAINING_METADATA_FILE = "reviewed_training_metadata.json"
MODEL_FILE = "hosn_conflict_ai_model.joblib"
REPORT_FILE = "conflict_ai_training_report.json"
PREDICTIONS_FILE = "conflict_ai_cv_predictions.csv"
RANDOM_SEED = 20261003
FOLD_COUNT = 4
MIN_GROUP_BALANCED_ACCURACY = 0.70
MIN_GROUP_CLASS_RECALL = 0.60


def strict_json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_latest_reviewed_campaign(root: Path) -> Path:
    candidates = sorted((root / "results").glob(CAMPAIGN_PREFIX + "*"), reverse=True)
    for folder in candidates:
        if (folder / TRAINING_FILE).is_file() and (folder / TRAINING_METADATA_FILE).is_file():
            return folder
    raise RuntimeError("No reviewed measured training dataset was found")


def load_reviewed_data(folder: Path) -> dict:
    csv_path = folder / TRAINING_FILE
    metadata_path = folder / TRAINING_METADATA_FILE
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("synthetic_data") is not False:
        raise ValueError("Training metadata must explicitly declare synthetic_data=false")
    if metadata.get("source_data_kind") != heterogeneous.REQUIRED_TRAINING_DATA_KIND:
        raise ValueError("Training source does not match the measured-data contract")
    if tuple(metadata.get("feature_order", ())) != heterogeneous.AI_FEATURE_ORDER:
        raise ValueError("Training feature order does not match the controller contract")
    if metadata.get("label_source") != "HUMAN_PAIRED_OUTCOME_REVIEW":
        raise ValueError("Labels must come from explicit paired-outcome human review")
    if metadata.get("training_csv_sha256") != sha256(csv_path):
        raise ValueError("Training CSV hash does not match its review metadata")

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Reviewed training dataset is empty")
    forbidden = {"forced_experimental_action", "post_loss_pct", "outcome"}
    if forbidden.intersection(rows[0]):
        raise ValueError("Post-decision or forced-action leakage column detected")
    required = {
        "group_id", "replay_id", "target_label", "label_source",
        *heterogeneous.AI_FEATURE_ORDER,
    }
    if not required.issubset(rows[0]):
        raise ValueError("Reviewed training dataset is missing required columns")

    features = []
    labels = []
    groups = []
    replay_ids = []
    for row in rows:
        label = row["target_label"]
        if label not in ("STAY", "HANDOVER"):
            raise ValueError("Target labels must be STAY or HANDOVER")
        if row["label_source"] != "HUMAN_PAIRED_OUTCOME_REVIEW":
            raise ValueError("Mixed or unreviewed label source")
        vector = []
        for name in heterogeneous.AI_FEATURE_ORDER:
            value = float(row[name])
            if not math.isfinite(value):
                raise ValueError("Non-finite feature: " + name)
            vector.append(value)
        features.append(vector)
        labels.append(label)
        groups.append(row["group_id"])
        replay_ids.append(row["replay_id"])

    group_labels: dict[str, set[str]] = defaultdict(set)
    group_sizes = Counter(groups)
    for group, label in zip(groups, labels):
        group_labels[group].add(label)
    if any(len(values) != 1 for values in group_labels.values()):
        raise ValueError("A scenario group contains conflicting human labels")
    if any(size != 4 for size in group_sizes.values()):
        raise ValueError("Every paired scenario group must contain four replays")
    if set(labels) != {"STAY", "HANDOVER"}:
        raise ValueError("Both STAY and HANDOVER classes are required")
    if len(group_labels) < FOLD_COUNT * 2:
        raise ValueError("Too few independent scenario groups for grouped validation")
    if metadata.get("rows") != len(rows) or metadata.get("groups") != len(group_labels):
        raise ValueError("Training metadata counts do not match the CSV")
    return {
        "features": features,
        "labels": labels,
        "groups": groups,
        "replay_ids": replay_ids,
        "rows": rows,
        "metadata": metadata,
        "csv_sha256": sha256(csv_path),
    }


def stratified_group_folds(groups: list[str], labels: list[str],
                           fold_count: int = FOLD_COUNT,
                           seed: int = RANDOM_SEED) -> list[tuple[list[int], list[int]]]:
    if len(groups) != len(labels):
        raise ValueError("groups and labels must have equal length")
    group_label_sets: dict[str, set[str]] = defaultdict(set)
    for group, label in zip(groups, labels):
        if label not in ("STAY", "HANDOVER"):
            raise ValueError("Unknown label")
        group_label_sets[group].add(label)
    if any(len(values) != 1 for values in group_label_sets.values()):
        raise ValueError("Each group must have exactly one label")
    class_groups = {label: [] for label in ("STAY", "HANDOVER")}
    for group, values in group_label_sets.items():
        class_groups[next(iter(values))].append(group)
    if any(len(values) < fold_count for values in class_groups.values()):
        raise ValueError("Each class needs at least one group per fold")
    rng = random.Random(seed)
    folds = [set() for _ in range(fold_count)]
    for label in ("STAY", "HANDOVER"):
        values = sorted(class_groups[label])
        rng.shuffle(values)
        for index, group in enumerate(values):
            folds[index % fold_count].add(group)
    output = []
    all_indices = set(range(len(groups)))
    for test_groups in folds:
        test_indices = [index for index, group in enumerate(groups) if group in test_groups]
        train_indices = sorted(all_indices - set(test_indices))
        if set(groups[index] for index in train_indices).intersection(test_groups):
            raise AssertionError("Group leakage detected")
        if set(labels[index] for index in test_indices) != {"STAY", "HANDOVER"}:
            raise AssertionError("Test fold is missing a class")
        output.append((train_indices, test_indices))
    return output


def build_pipeline():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn before training") from exc
    return Pipeline((
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED,
            solver="liblinear",
        )),
    ))


def confusion(actual: list[str], predicted: list[str]) -> dict:
    matrix = {
        "STAY": {"STAY": 0, "HANDOVER": 0},
        "HANDOVER": {"STAY": 0, "HANDOVER": 0},
    }
    for truth, guess in zip(actual, predicted):
        matrix[truth][guess] += 1
    return matrix


def metrics(actual: list[str], predicted: list[str]) -> dict:
    if not actual or len(actual) != len(predicted):
        raise ValueError("Metric inputs must be nonempty and equal length")
    matrix = confusion(actual, predicted)
    recalls = {}
    for label in ("STAY", "HANDOVER"):
        total = sum(matrix[label].values())
        recalls[label] = matrix[label][label] / total if total else 0.0
    return {
        "accuracy": sum(a == p for a, p in zip(actual, predicted)) / len(actual),
        "balanced_accuracy": sum(recalls.values()) / 2,
        "recall": recalls,
        "confusion_matrix_actual_rows": matrix,
        "count": len(actual),
    }


def _rows(values: list[list[float]], indices: list[int]) -> list[list[float]]:
    return [values[index] for index in indices]


def _items(values: list[str], indices: list[int]) -> list[str]:
    return [values[index] for index in indices]


def cross_validate(data: dict) -> tuple[dict, list[dict]]:
    features = data["features"]
    labels = data["labels"]
    groups = data["groups"]
    folds = stratified_group_folds(groups, labels)
    row_predictions = [None] * len(labels)
    fold_reports = []
    for fold_number, (train_indices, test_indices) in enumerate(folds, start=1):
        model = build_pipeline()
        model.fit(_rows(features, train_indices), _items(labels, train_indices))
        predictions = list(model.predict(_rows(features, test_indices)))
        for index, prediction in zip(test_indices, predictions):
            row_predictions[index] = str(prediction)
        fold_reports.append({
            "fold": fold_number,
            "train_groups": len(set(_items(groups, train_indices))),
            "test_groups": len(set(_items(groups, test_indices))),
            "row_metrics": metrics(_items(labels, test_indices), predictions),
        })
    if any(prediction is None for prediction in row_predictions):
        raise AssertionError("Cross-validation did not predict every row")
    row_predictions = [str(value) for value in row_predictions]

    group_votes: dict[str, list[str]] = defaultdict(list)
    group_truth: dict[str, str] = {}
    for group, truth, prediction in zip(groups, labels, row_predictions):
        group_votes[group].append(prediction)
        group_truth[group] = truth
    group_actual = []
    group_predicted = []
    for group in sorted(group_votes):
        votes = Counter(group_votes[group])
        # Four rows can tie 2-2. A tie is counted conservatively as the wrong
        # class for that group's known truth rather than resolved optimistically.
        if votes["STAY"] == votes["HANDOVER"]:
            prediction = "HANDOVER" if group_truth[group] == "STAY" else "STAY"
        else:
            prediction = votes.most_common(1)[0][0]
        group_actual.append(group_truth[group])
        group_predicted.append(prediction)
    summary = {
        "fold_count": len(folds),
        "row_metrics": metrics(labels, row_predictions),
        "group_metrics": metrics(group_actual, group_predicted),
        "folds": fold_reports,
        "group_leakage_check_pass": True,
    }
    prediction_rows = [
        {
            "group_id": group,
            "replay_id": replay,
            "actual_label": truth,
            "predicted_label": prediction,
            "correct": truth == prediction,
        }
        for group, replay, truth, prediction in zip(
            groups, data["replay_ids"], labels, row_predictions
        )
    ]
    return summary, prediction_rows


def acceptance(cv: dict) -> dict:
    group = cv["group_metrics"]
    checks = {
        "group_balanced_accuracy": (
            group["balanced_accuracy"] >= MIN_GROUP_BALANCED_ACCURACY
        ),
        "stay_group_recall": group["recall"]["STAY"] >= MIN_GROUP_CLASS_RECALL,
        "handover_group_recall": (
            group["recall"]["HANDOVER"] >= MIN_GROUP_CLASS_RECALL
        ),
        "group_leakage_absent": cv["group_leakage_check_pass"],
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "minimum_group_balanced_accuracy": MIN_GROUP_BALANCED_ACCURACY,
            "minimum_group_class_recall": MIN_GROUP_CLASS_RECALL,
        },
    }


def train_campaign(folder: Path) -> dict:
    try:
        import joblib
        import sklearn
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn and joblib before training") from exc
    data = load_reviewed_data(folder)
    cv, prediction_rows = cross_validate(data)
    gate = acceptance(cv)

    with (folder / PREDICTIONS_FILE).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(prediction_rows[0]))
        writer.writeheader(); writer.writerows(prediction_rows)

    model_path = folder / MODEL_FILE
    if gate["accepted"]:
        model = build_pipeline()
        model.fit(data["features"], data["labels"])
        # Metadata is enforced by hosn_heterogeneous_controller.decide().
        model.hosn_schema_version = heterogeneous.MODEL_SCHEMA_VERSION
        model.feature_order = heterogeneous.AI_FEATURE_ORDER
        model.training_data_kind = heterogeneous.REQUIRED_TRAINING_DATA_KIND
        model.training_data_sha256 = data["csv_sha256"]
        model.training_group_count = len(set(data["groups"]))
        model.training_row_count = len(data["labels"])
        model.created_utc = datetime.now(timezone.utc).isoformat()
        temporary = model_path.with_name(model_path.name + ".tmp")
        joblib.dump(model, temporary)
        os.replace(temporary, model_path)
    else:
        model_path.unlink(missing_ok=True)

    report = {
        "revision": REVISION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_campaign": folder.name,
        "source_training_csv": TRAINING_FILE,
        "source_training_csv_sha256": data["csv_sha256"],
        "synthetic_data": False,
        "label_source": "HUMAN_PAIRED_OUTCOME_REVIEW",
        "features": list(heterogeneous.AI_FEATURE_ORDER),
        "rows": len(data["labels"]),
        "independent_scenario_groups": len(set(data["groups"])),
        "row_label_distribution": dict(Counter(data["labels"])),
        "group_label_distribution": dict(Counter(
            next(label for group_value, label in zip(data["groups"], data["labels"])
                 if group_value == group)
            for group in sorted(set(data["groups"]))
        )),
        "model": "StandardScaler + class-balanced logistic regression",
        "hyperparameter_search_performed": False,
        "cross_validation": cv,
        "acceptance_gate": gate,
        "model_saved": gate["accepted"],
        "model_file": MODEL_FILE if gate["accepted"] else None,
        "model_sha256": sha256(model_path) if gate["accepted"] else None,
        "sklearn_version": sklearn.__version__,
        "limitations": [
            "Development validation uses 32 emulated scenario groups, not an external field test.",
            "Cellular is an emulated 5G-like IP path, not a complete 3GPP radio/core.",
            "The model is valid only for genuine rule conflicts under this feature contract.",
            "Clear rule cases must bypass the model.",
        ],
        "next_step": (
            "integrate accepted model and run identical three-arm final comparison"
            if gate["accepted"]
            else "collect/review more measured conflict scenarios before integration"
        ),
    }
    strict_json_write(folder / REPORT_FILE, report)
    return report


def print_report(report: dict) -> None:
    group = report["cross_validation"]["group_metrics"]
    row = report["cross_validation"]["row_metrics"]
    print("Measured rows:", report["rows"])
    print("Independent scenario groups:", report["independent_scenario_groups"])
    print("Group-balanced accuracy: {:.2f}%".format(100 * group["balanced_accuracy"]))
    print("Group STAY recall: {:.2f}%".format(100 * group["recall"]["STAY"]))
    print("Group HANDOVER recall: {:.2f}%".format(100 * group["recall"]["HANDOVER"]))
    print("Row-balanced accuracy: {:.2f}%".format(100 * row["balanced_accuracy"]))
    print("Model accepted:", report["acceptance_gate"]["accepted"])
    print("Model saved:", report["model_saved"])


def run_self_tests() -> int:
    import unittest

    def fixture(folder: Path, groups_per_class: int = 8) -> None:
        rows = []
        for label, sign in (("STAY", -1.0), ("HANDOVER", 1.0)):
            for group_index in range(groups_per_class):
                group = "{}_{}".format(label.lower(), group_index)
                for replay_index in range(4):
                    row = {
                        "group_id": group, "replay_id": "r{}".format(replay_index),
                        "direction": "outbound", "speed_class": "fast",
                        "decision_zone": "edge", "qos_pair": "1",
                        "target_label": label,
                        "label_source": "HUMAN_PAIRED_OUTCOME_REVIEW",
                        "reviewer": "test", "reviewed_utc": "2026-10-03T00:00:00Z",
                    }
                    for feature_index, name in enumerate(heterogeneous.AI_FEATURE_ORDER):
                        row[name] = sign * (10.0 + feature_index) + replay_index * 0.01
                    rows.append(row)
        csv_path = folder / TRAINING_FILE
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        strict_json_write(folder / TRAINING_METADATA_FILE, {
            "synthetic_data": False,
            "source_data_kind": heterogeneous.REQUIRED_TRAINING_DATA_KIND,
            "feature_order": list(heterogeneous.AI_FEATURE_ORDER),
            "label_source": "HUMAN_PAIRED_OUTCOME_REVIEW",
            "rows": len(rows), "groups": groups_per_class * 2,
            "training_csv_sha256": sha256(csv_path),
        })

    class Tests(unittest.TestCase):
        def test_group_folds_have_no_leakage_and_both_classes(self):
            groups, labels = [], []
            for label in ("STAY", "HANDOVER"):
                for group_index in range(8):
                    for _ in range(4):
                        groups.append(label + str(group_index)); labels.append(label)
            folds = stratified_group_folds(groups, labels)
            self.assertEqual(len(folds), 4)
            for train, test in folds:
                self.assertFalse(set(_items(groups, train)) & set(_items(groups, test)))
                self.assertEqual(set(_items(labels, test)), {"STAY", "HANDOVER"})

        def test_metrics_are_exact(self):
            result = metrics(
                ["STAY", "STAY", "HANDOVER", "HANDOVER"],
                ["STAY", "HANDOVER", "HANDOVER", "HANDOVER"],
            )
            self.assertEqual(result["accuracy"], 0.75)
            self.assertEqual(result["balanced_accuracy"], 0.75)

        def test_reviewed_loader_and_group_counts(self):
            with tempfile.TemporaryDirectory() as name:
                folder = Path(name); fixture(folder)
                data = load_reviewed_data(folder)
            self.assertEqual(len(data["labels"]), 64)
            self.assertEqual(len(set(data["groups"])), 16)

        def test_hash_mismatch_is_rejected(self):
            with tempfile.TemporaryDirectory() as name:
                folder = Path(name); fixture(folder)
                metadata_path = folder / TRAINING_METADATA_FILE
                metadata = json.loads(metadata_path.read_text())
                metadata["training_csv_sha256"] = "0" * 64
                strict_json_write(metadata_path, metadata)
                with self.assertRaises(ValueError):
                    load_reviewed_data(folder)

        def test_end_to_end_training_fixture(self):
            with tempfile.TemporaryDirectory() as name:
                folder = Path(name); fixture(folder)
                report = train_campaign(folder)
                self.assertTrue(report["acceptance_gate"]["accepted"])
                self.assertTrue((folder / MODEL_FILE).is_file())
                self.assertEqual(
                    report["cross_validation"]["group_metrics"]["balanced_accuracy"], 1.0
                )

        def test_controller_accepts_saved_model_contract(self):
            with tempfile.TemporaryDirectory() as name:
                folder = Path(name); fixture(folder)
                report = train_campaign(folder)
                import joblib
                model = joblib.load(folder / report["model_file"])
                self.assertEqual(model.hosn_schema_version,
                                 heterogeneous.MODEL_SCHEMA_VERSION)
                self.assertEqual(tuple(model.feature_order),
                                 heterogeneous.AI_FEATURE_ORDER)
                self.assertEqual(model.training_data_kind,
                                 heterogeneous.REQUIRED_TRAINING_DATA_KIND)

    result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    )
    if result.wasSuccessful():
        print("PASS: {} grouped conflict-AI software checks.".format(result.testsRun))
        print("Self-tests use software fixtures only; no project model was trained.")
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Group-validated HOSN conflict-only AI trainer"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return run_self_tests()
    root = Path(__file__).resolve().parent
    try:
        folder = find_latest_reviewed_campaign(root)
        if args.inspect:
            report_path = folder / REPORT_FILE
            if not report_path.is_file():
                raise RuntimeError("No training report exists yet; run --train")
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = train_campaign(folder)
        print_report(report)
        print("Report:", folder / REPORT_FILE)
        if args.train and report["model_saved"]:
            print("Accepted model:", folder / MODEL_FILE)
        return 0 if report["acceptance_gate"]["accepted"] else 1
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.exit(2, "Training stopped: " + str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
