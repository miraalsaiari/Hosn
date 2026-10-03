#!/usr/bin/env python3
"""Human review for measured paired HOSN outcomes.

This program never invents or silently accepts labels.  It finds the latest
completed measured campaign, shows the paired STAY/HANDOVER outcome summary,
and requires an explicit reviewer choice for every scenario.  Progress is
saved after each choice and can be resumed by running the same command.

Only when every scenario has been reviewed does it export
reviewed_training_dataset.csv.  The model features come from pre-decision
measurements; forced experimental actions and post-decision outcome metrics are
not included as model inputs.  Scenario IDs remain as group IDs so later model
validation can keep all four paired replays in the same train/test partition.

Commands:
  python3 hosn_review_labels.py --self-test
  python3 hosn_review_labels.py --status
  python3 hosn_review_labels.py --review
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Optional

import hosn_heterogeneous_controller as heterogeneous


REVISION = "human-paired-outcome-review-v1"
CAMPAIGN_PREFIX = "measured_dataset_campaign_"
REVIEWS_FILE = "human_reviews.json"
TRAINING_FILE = "reviewed_training_dataset.csv"
TRAINING_METADATA_FILE = "reviewed_training_metadata.json"
VALID_LABELS = ("STAY", "HANDOVER", "EXCLUDE")

TRAINING_FIELDS = (
    "group_id", "replay_id", "direction", "speed_class",
    "decision_zone", "qos_pair", *heterogeneous.AI_FEATURE_ORDER,
    "target_label", "label_source", "reviewer", "reviewed_utc",
)


def strict_json_write(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def find_latest_campaign(root: Path) -> Path:
    candidates = sorted((root / "results").glob(CAMPAIGN_PREFIX + "*"), reverse=True)
    for folder in candidates:
        report_path = folder / "campaign_report.json"
        manifest_path = folder / "manifest.json"
        if not report_path.is_file() or not manifest_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if report.get("collection_complete") is True and manifest.get("status") in (
            "collection_complete", "collection_complete_with_exclusions",
        ):
            return folder
    raise RuntimeError("No completed measured-data campaign was found")


def load_campaign(folder: Path) -> tuple[list[dict], list[dict], dict]:
    report = json.loads((folder / "campaign_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if report.get("collection_complete") is not True:
        raise ValueError("Campaign is not complete")
    if report.get("synthetic_data") is not False or manifest.get("synthetic_data") is not False:
        raise ValueError("Refusing a campaign that is not explicitly measured/non-synthetic")
    if report.get("training_labels_assigned") != 0:
        raise ValueError("Campaign unexpectedly contains training labels")
    with (folder / "paired_outcome_review.csv").open(encoding="utf-8", newline="") as handle:
        review_candidates = list(csv.DictReader(handle))
    with (folder / "measured_decision_rows.csv").open(encoding="utf-8", newline="") as handle:
        measured_rows = list(csv.DictReader(handle))
    scenario_ids = [row["scenario_id"] for row in review_candidates]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Duplicate scenario IDs in review table")
    if set(row["scenario_id"] for row in measured_rows) != set(scenario_ids):
        raise ValueError("Measured and paired-review scenario sets differ")
    counts = Counter(row["scenario_id"] for row in measured_rows)
    if any(count != 4 for count in counts.values()):
        raise ValueError("Each scenario must contain exactly four ABBA measured rows")
    return review_candidates, measured_rows, {"report": report, "manifest": manifest}


def load_reviews(folder: Path) -> dict:
    path = folder / REVIEWS_FILE
    if not path.is_file():
        return {
            "revision": REVISION,
            "campaign_directory": folder.name,
            "reviews": {},
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("revision") != REVISION or not isinstance(value.get("reviews"), dict):
        raise ValueError("Existing human review file has an incompatible schema")
    return value


def _number(value: str) -> float:
    number = float(value)
    if not number == number or number in (float("inf"), float("-inf")):
        raise ValueError("Non-finite metric")
    return number


def scenario_display(folder: Path, candidate: dict, position: int, total: int) -> str:
    scenario_id = candidate["scenario_id"]
    report_path = folder / scenario_id / "scenario_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stay = report["aggregate_by_action"]["STAY"]
    handover = report["aggregate_by_action"]["HANDOVER"]

    def line(name: str, values: dict) -> str:
        return (
            "{}: loss={:.3f}% delay={:.2f}ms gap={:.2f}ms "
            "frames={:.2f}% goodput={:.3f}Mbps"
        ).format(
            name,
            values["mean_post_loss_pct"],
            values["mean_post_p95_delay_ms"],
            values["mean_post_max_gap_ms"],
            values["mean_post_complete_frames_pct"],
            values["mean_post_goodput_mbps"],
        )
    return "\n".join((
        "",
        "Scenario {}/{}: {}".format(position, total, scenario_id),
        "Context: {} / {} / {} / QoS pair {}".format(
            candidate["direction"], candidate["speed_class"],
            candidate["decision_zone"], candidate["qos_pair"],
        ),
        "Rule snapshots all conflict: {}".format(candidate["all_snapshots_conflict"]),
        line("STAY", stay),
        line("HANDOVER", handover),
        "Automated recommendation (not a label): {}".format(
            candidate["automated_review_recommendation"]
        ),
        "Materially better metrics: {}".format(
            candidate["materially_better_metrics"] or "none"
        ),
    ))


def review_campaign(folder: Path, reviewer: str,
                    input_fn: Callable[[str], str] = input,
                    output_fn: Callable[[str], None] = print) -> dict:
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("Reviewer/team name cannot be blank")
    candidates, measured_rows, campaign = load_campaign(folder)
    state = load_reviews(folder)
    valid_ids = {row["scenario_id"] for row in candidates}
    unknown = set(state["reviews"]) - valid_ids
    if unknown:
        raise ValueError("Review file contains unknown scenario IDs")

    for position, candidate in enumerate(candidates, start=1):
        scenario_id = candidate["scenario_id"]
        if scenario_id in state["reviews"]:
            continue
        output_fn(scenario_display(folder, candidate, position, len(candidates)))
        recommendation = candidate["automated_review_recommendation"]
        while True:
            choice = input_fn(
                "Choose [A]ccept recommendation, [S]TAY, [H]ANDOVER, "
                "[X]exclude, or [Q]save and quit: "
            ).strip().lower()
            if choice == "q":
                strict_json_write(folder / REVIEWS_FILE, state)
                return build_status(folder, candidates, measured_rows, state, campaign)
            if choice == "a" and recommendation in ("STAY", "HANDOVER"):
                label, decision_method = recommendation, "EXPLICIT_ACCEPT_RECOMMENDATION"
                break
            if choice in ("s", "h", "x"):
                label = {"s": "STAY", "h": "HANDOVER", "x": "EXCLUDE"}[choice]
                decision_method = "EXPLICIT_MANUAL_CHOICE"
                break
            output_fn("Invalid choice. Enter A, S, H, X, or Q.")
        state["reviews"][scenario_id] = {
            "human_label": label,
            "reviewer": reviewer,
            "reviewed_utc": datetime.now(timezone.utc).isoformat(),
            "decision_method": decision_method,
            "automated_recommendation_seen": recommendation,
            "materially_better_metrics_seen": candidate["materially_better_metrics"],
        }
        strict_json_write(folder / REVIEWS_FILE, state)
        output_fn("Saved {} = {}".format(scenario_id, label))

    status = build_status(folder, candidates, measured_rows, state, campaign)
    if status["remaining"] == 0:
        export_training_table(folder, measured_rows, state, campaign)
        status = build_status(folder, candidates, measured_rows, state, campaign)
    return status


def build_status(folder: Path, candidates: list[dict], measured_rows: list[dict],
                 state: dict, campaign: dict) -> dict:
    labels = Counter(
        review["human_label"] for review in state["reviews"].values()
    )
    return {
        "campaign": folder.name,
        "total_scenarios": len(candidates),
        "reviewed": len(state["reviews"]),
        "remaining": len(candidates) - len(state["reviews"]),
        "labels": {label: labels.get(label, 0) for label in VALID_LABELS},
        "measured_rows_available": len(measured_rows),
        "training_file_created": (folder / TRAINING_FILE).is_file(),
        "source_collection_complete": campaign["report"]["collection_complete"],
    }


def export_training_table(folder: Path, measured_rows: list[dict],
                          state: dict, campaign: dict) -> dict:
    scenario_ids = set(row["scenario_id"] for row in measured_rows)
    if set(state["reviews"]) != scenario_ids:
        raise ValueError("Every scenario must be reviewed before training export")
    output_rows = []
    for row in measured_rows:
        review = state["reviews"][row["scenario_id"]]
        if review["human_label"] == "EXCLUDE":
            continue
        if row["rule_status"] != "CONFLICT" or row["pilot_valid"] != "True":
            raise ValueError("Approved rows must be mechanically valid CONFLICT measurements")
        output = {
            "group_id": row["scenario_id"],
            "replay_id": row["replay_id"],
            "direction": row["direction"],
            "speed_class": row["speed_class"],
            "decision_zone": row["decision_zone"],
            "qos_pair": row["qos_pair"],
            "target_label": review["human_label"],
            "label_source": "HUMAN_PAIRED_OUTCOME_REVIEW",
            "reviewer": review["reviewer"],
            "reviewed_utc": review["reviewed_utc"],
        }
        for name in heterogeneous.AI_FEATURE_ORDER:
            output[name] = _number(row[name])
        output_rows.append(output)

    with (folder / TRAINING_FILE).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAINING_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(output_rows)
    distribution = Counter(row["target_label"] for row in output_rows)
    metadata = {
        "revision": REVISION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_campaign": folder.name,
        "source_data_kind": campaign["manifest"]["data_kind"],
        "synthetic_data": False,
        "feature_order": list(heterogeneous.AI_FEATURE_ORDER),
        "label_source": "HUMAN_PAIRED_OUTCOME_REVIEW",
        "rows": len(output_rows),
        "groups": len({row["group_id"] for row in output_rows}),
        "excluded_groups": sum(
            review["human_label"] == "EXCLUDE" for review in state["reviews"].values()
        ),
        "label_distribution_rows": dict(distribution),
        "validation_requirement": "group-aware split by group_id; never split paired replays",
        "training_csv_sha256": hashlib.sha256((folder / TRAINING_FILE).read_bytes()).hexdigest(),
    }
    strict_json_write(folder / TRAINING_METADATA_FILE, metadata)
    return metadata


def print_status(status: dict) -> None:
    print("Campaign:", status["campaign"])
    print("Reviewed: {}/{}".format(status["reviewed"], status["total_scenarios"]))
    print("Remaining:", status["remaining"])
    print("Labels: STAY={} HANDOVER={} EXCLUDE={}".format(
        status["labels"]["STAY"], status["labels"]["HANDOVER"],
        status["labels"]["EXCLUDE"],
    ))
    print("Training file created:", status["training_file_created"])


def run_self_tests() -> int:
    import unittest

    def fixture(root: Path) -> Path:
        folder = root / "results" / (CAMPAIGN_PREFIX + "fixture")
        folder.mkdir(parents=True)
        rows = []
        candidates = []
        for number, recommendation in ((1, "STAY"), (2, "HANDOVER")):
            scenario_id = "scenario_{}".format(number)
            candidates.append({
                "scenario_id": scenario_id, "direction": "outbound",
                "speed_class": "fast", "decision_zone": "edge", "qos_pair": str(number),
                "mechanical_checks_pass": "True", "all_snapshots_conflict": "True",
                "automated_review_recommendation": recommendation,
                "materially_better_metrics": "mean_post_loss_pct",
                "review_status": "REVIEW_CANDIDATE", "human_label": "",
                "human_reviewer": "", "human_notes": "",
            })
            aggregate = {}
            for action, loss in (("STAY", 4.0), ("HANDOVER", 1.0)):
                aggregate[action] = {
                    "mean_post_loss_pct": loss, "mean_post_p95_delay_ms": 20.0,
                    "mean_post_max_gap_ms": 30.0,
                    "mean_post_complete_frames_pct": 98.0,
                    "mean_post_goodput_mbps": 0.95,
                }
            scenario_dir = folder / scenario_id
            scenario_dir.mkdir()
            strict_json_write(scenario_dir / "scenario_report.json", {
                "aggregate_by_action": aggregate,
            })
            for replay_id in ("stay_1", "handover_1", "handover_2", "stay_2"):
                row = {
                    "scenario_id": scenario_id, "replay_id": replay_id,
                    "forced_experimental_action": "STAY",
                    "direction": "outbound", "speed_class": "fast",
                    "decision_zone": "edge", "qos_pair": str(number),
                    "rule_decision": "ASK_AI", "rule_status": "CONFLICT",
                    "pilot_valid": "True", "evidence_directory": "evidence",
                }
                row.update({name: str(index + 1.0) for index, name in
                            enumerate(heterogeneous.AI_FEATURE_ORDER)})
                rows.append(row)
        review_fields = tuple(candidates[0])
        measured_fields = tuple(rows[0])
        for filename, fields, values in (
            ("paired_outcome_review.csv", review_fields, candidates),
            ("measured_decision_rows.csv", measured_fields, rows),
        ):
            with (folder / filename).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerows(values)
        strict_json_write(folder / "campaign_report.json", {
            "collection_complete": True, "synthetic_data": False,
            "training_labels_assigned": 0,
        })
        strict_json_write(folder / "manifest.json", {
            "status": "collection_complete", "synthetic_data": False,
            "data_kind": heterogeneous.REQUIRED_TRAINING_DATA_KIND,
        })
        return folder

    class Tests(unittest.TestCase):
        def test_latest_campaign_requires_completion(self):
            with tempfile.TemporaryDirectory() as name:
                folder = fixture(Path(name))
                self.assertEqual(find_latest_campaign(Path(name)), folder)

        def test_explicit_accept_and_manual_exclude(self):
            with tempfile.TemporaryDirectory() as name:
                folder = fixture(Path(name))
                choices = iter(("a", "x"))
                status = review_campaign(
                    folder, "ATP Team", input_fn=lambda _prompt: next(choices),
                    output_fn=lambda _text: None,
                )
                self.assertEqual(status["remaining"], 0)
                self.assertEqual(status["labels"], {"STAY": 1, "HANDOVER": 0, "EXCLUDE": 1})
                self.assertTrue(status["training_file_created"])
                rows = list(csv.DictReader((folder / TRAINING_FILE).open(encoding="utf-8")))
                self.assertEqual(len(rows), 4)
                self.assertTrue(all(row["target_label"] == "STAY" for row in rows))

        def test_quit_saves_and_resume_finishes(self):
            with tempfile.TemporaryDirectory() as name:
                folder = fixture(Path(name))
                first = review_campaign(
                    folder, "Reviewer", input_fn=lambda _prompt: "q",
                    output_fn=lambda _text: None,
                )
                self.assertEqual(first["remaining"], 2)
                choices = iter(("s", "h"))
                second = review_campaign(
                    folder, "Reviewer", input_fn=lambda _prompt: next(choices),
                    output_fn=lambda _text: None,
                )
                self.assertEqual(second["remaining"], 0)
                self.assertEqual(second["labels"]["STAY"], 1)
                self.assertEqual(second["labels"]["HANDOVER"], 1)

        def test_training_schema_excludes_forced_action_and_outcomes(self):
            self.assertNotIn("forced_experimental_action", TRAINING_FIELDS)
            self.assertNotIn("loss_pct", TRAINING_FIELDS)
            self.assertEqual(
                tuple(name for name in TRAINING_FIELDS if name in heterogeneous.AI_FEATURE_ORDER),
                heterogeneous.AI_FEATURE_ORDER,
            )

        def test_invalid_existing_review_schema_rejected(self):
            with tempfile.TemporaryDirectory() as name:
                folder = fixture(Path(name))
                strict_json_write(folder / REVIEWS_FILE, {"revision": "wrong", "reviews": {}})
                with self.assertRaises(ValueError):
                    load_reviews(folder)

    result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    )
    if result.wasSuccessful():
        print("PASS: {} human-review software checks. No campaign changed.".format(
            result.testsRun
        ))
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explicit human review of measured paired HOSN outcomes"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--review", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return run_self_tests()
    root = Path(__file__).resolve().parent
    try:
        folder = find_latest_campaign(root)
        candidates, measured_rows, campaign = load_campaign(folder)
        state = load_reviews(folder)
        if args.status:
            print_status(build_status(folder, candidates, measured_rows, state, campaign))
            return 0
        if not sys.stdin.isatty():
            parser.exit(2, "Run --review in an interactive terminal.\n")
        reviewer = input("Reviewer or team name: ").strip()
        status = review_campaign(folder, reviewer)
        print_status(status)
        if status["remaining"] == 0:
            print("\nHUMAN REVIEW COMPLETE")
            print("Reviewed training data:", folder / TRAINING_FILE)
            print("Next step: group-aware model training and validation.")
        else:
            print("Run the same --review command later to continue.")
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.exit(2, "Review stopped: " + str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
