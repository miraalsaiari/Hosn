#!/usr/bin/env python3
"""Collect conflict-focused measured HOSN candidates from Mininet-WiFi.

This is the first final-data collection stage, not a synthetic generator and
not a trainer.  Every scenario is replayed in ABBA order with forced STAY and
HANDOVER actions so their measured post-decision outcomes can be compared.

The program writes two deliberately separate CSV files:

* measured_decision_rows.csv contains measured controller inputs and the
  forced experimental action.  It has no target/label column.
* paired_outcome_review.csv contains an automated review recommendation plus
  blank human_label and reviewer fields.  A recommendation is not a label.

Only a later, explicit human-review step may create a training table.  Missing,
invalid, mechanically failed, non-conflict, or inconclusive pairs remain
excluded instead of receiving invented labels.

Commands:
  python3 hosn_measured_dataset.py --self-test
  python3 hosn_measured_dataset.py --plan
  sudo python3 hosn_measured_dataset.py --collect
  sudo python3 hosn_measured_dataset.py --resume-latest
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback
from typing import Any, Optional

import hosn_heterogeneous_controller as heterogeneous
import hosn_paired_outcome_pilot as paired
import hosn_wifi_5g_compare as compare
import hosn_wifi_5g_pilot as base


REVISION = "measured-conflict-campaign-v1"
CAMPAIGN_PREFIX = "measured_dataset_campaign_"

DATASET_PROFILES = {
    "wifi_data_excellent": {
        "delay_ms": 4.0, "jitter_ms": 1.0, "loss_pct": 0.0, "rate_mbit": 25.0,
    },
    "wifi_data_good": {
        "delay_ms": 8.0, "jitter_ms": 2.0, "loss_pct": 0.4, "rate_mbit": 18.0,
    },
    "wifi_data_fair": {
        "delay_ms": 18.0, "jitter_ms": 4.0, "loss_pct": 1.2, "rate_mbit": 12.0,
    },
    "wifi_data_poor": {
        "delay_ms": 45.0, "jitter_ms": 9.0, "loss_pct": 3.5, "rate_mbit": 8.0,
    },
    "wifi_data_bad": {
        "delay_ms": 70.0, "jitter_ms": 15.0, "loss_pct": 7.0, "rate_mbit": 5.0,
    },
    "cell_data_excellent": {
        "delay_ms": 10.0, "jitter_ms": 1.5, "loss_pct": 0.0, "rate_mbit": 25.0,
    },
    "cell_data_good": {
        "delay_ms": 18.0, "jitter_ms": 3.0, "loss_pct": 0.5, "rate_mbit": 18.0,
    },
    "cell_data_fair": {
        "delay_ms": 32.0, "jitter_ms": 6.0, "loss_pct": 1.5, "rate_mbit": 12.0,
    },
    "cell_data_poor": {
        "delay_ms": 60.0, "jitter_ms": 12.0, "loss_pct": 4.5, "rate_mbit": 7.0,
    },
    "cell_data_bad": {
        "delay_ms": 85.0, "jitter_ms": 18.0, "loss_pct": 8.0, "rate_mbit": 5.0,
    },
}

# At the Wi-Fi edge, cellular has the normalized signal advantage but QoS is
# deliberately worse.  Near Wi-Fi, cellular lacks the signal advantage but
# QoS is deliberately better.  Both families should reach rule CONFLICT.
EDGE_QOS_PAIRS = (
    ("wifi_data_excellent", "cell_data_fair"),
    ("wifi_data_good", "cell_data_poor"),
    ("wifi_data_fair", "cell_data_bad"),
    ("wifi_data_excellent", "cell_data_bad"),
)
NEAR_QOS_PAIRS = (
    ("wifi_data_poor", "cell_data_good"),
    ("wifi_data_bad", "cell_data_fair"),
    ("wifi_data_fair", "cell_data_excellent"),
    ("wifi_data_bad", "cell_data_excellent"),
)


def _movement(direction: str, speed_class: str, zone: str,
              decision_profile: str) -> list[dict]:
    times = (0.0, 4.0, 8.0) if speed_class == "fast" else (0.0, 6.0, 12.0)
    if zone == "edge" and direction == "outbound":
        points = ((20.0, "wifi_near"), (40.0, "wifi_moving"), (65.0, decision_profile))
    elif zone == "edge" and direction == "inbound":
        points = ((70.0, "wifi_edge"), (67.0, "wifi_edge"), (65.0, decision_profile))
    elif zone == "near" and direction == "outbound":
        points = ((0.0, "wifi_near"), (10.0, "wifi_near"), (20.0, decision_profile))
    elif zone == "near" and direction == "inbound":
        points = ((65.0, "wifi_edge"), (40.0, "wifi_moving"), (20.0, decision_profile))
    else:
        raise ValueError("Unknown movement context")
    return [
        {"at_s": at_s, "x_m": x_m, "wifi_profile": profile}
        for at_s, (x_m, profile) in zip(times, points)
    ]


def build_scenarios() -> tuple[dict, ...]:
    scenarios = []
    index = 0
    for zone, pairs, role in (
        ("edge", EDGE_QOS_PAIRS, "conflict_signal_favors_handover_qos_favors_stay"),
        ("near", NEAR_QOS_PAIRS, "conflict_signal_favors_stay_qos_favors_handover"),
    ):
        for direction in ("outbound", "inbound"):
            for speed_class in ("fast", "slow"):
                for pair_number, (wifi_profile, cell_profile) in enumerate(pairs, start=1):
                    index += 1
                    scenarios.append({
                        "id": "{:02d}_{}_{}_{}_q{}".format(
                            index, zone, direction, speed_class, pair_number
                        ),
                        "direction": direction,
                        "speed_class": speed_class,
                        "decision_zone": zone,
                        "qos_pair": pair_number,
                        "movement": _movement(direction, speed_class, zone, wifi_profile),
                        "decision_stage_index": 2,
                        "cell_profile": cell_profile,
                        "design_role": role,
                        "duration_s": 18.0 if speed_class == "fast" else 22.0,
                        # Same seed within each ABBA pair, different seed across contexts.
                        "seed_offset": index * 100,
                    })
    return tuple(scenarios)


CAMPAIGN_SCENARIOS = build_scenarios()
ALL_PROFILES = {**paired.ALL_ACCESS_PROFILES, **DATASET_PROFILES}
paired.ALL_ACCESS_PROFILES.update(DATASET_PROFILES)

PLAN = {
    "revision": REVISION,
    "purpose": "measured conflict-candidate collection for later human review",
    "data_origin": "Mininet-WiFi plus emulated cellular/5G-like IP path",
    "synthetic_data": False,
    "encoded_video_claim": False,
    "training_performed": False,
    "training_labels_assigned": 0,
    "automatic_recommendations_are_labels": False,
    "scenario_count": len(CAMPAIGN_SCENARIOS),
    "replays_per_scenario": len(paired.RUN_ORDER),
    "planned_replays": len(CAMPAIGN_SCENARIOS) * len(paired.RUN_ORDER),
    "run_order": [{"id": run_id, "forced_action": action}
                  for run_id, action in paired.RUN_ORDER],
    "feature_order": list(heterogeneous.AI_FEATURE_ORDER),
    "scenarios": list(CAMPAIGN_SCENARIOS),
    "eligibility": (
        "mechanical checks pass; all four rule snapshots are CONFLICT; "
        "paired outcome is not INCONCLUSIVE; human review still required"
    ),
}


MEASURED_FIELDS = (
    "scenario_id", "replay_id", "forced_experimental_action",
    "direction", "speed_class", "decision_zone", "qos_pair",
    *heterogeneous.AI_FEATURE_ORDER,
    "rule_decision", "rule_status", "pilot_valid", "evidence_directory",
)
REVIEW_FIELDS = (
    "scenario_id", "direction", "speed_class", "decision_zone", "qos_pair",
    "mechanical_checks_pass", "all_snapshots_conflict",
    "automated_review_recommendation", "materially_better_metrics",
    "review_status", "human_label", "human_reviewer", "human_notes",
)


def _plan_sha256() -> str:
    encoded = json.dumps(PLAN, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_write_json(path: Path, value: Any) -> None:
    compare.write_json(path, value)


def _load_report(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if value.get("scenario_complete") is True else None


def _attempt_directory(scenario_dir: Path) -> Path:
    existing = sorted(scenario_dir.glob("attempt_*"))
    attempt = scenario_dir / "attempt_{:02d}".format(len(existing) + 1)
    attempt.mkdir(parents=True)
    return attempt


def _collect_reports(folder: Path) -> dict[str, dict]:
    reports = {}
    for scenario in CAMPAIGN_SCENARIOS:
        report = _load_report(folder / scenario["id"] / "scenario_report.json")
        if report is not None:
            reports[scenario["id"]] = report
    return reports


def export_tables(folder: Path, reports: dict[str, dict]) -> dict:
    measured_rows = []
    review_rows = []
    for scenario in CAMPAIGN_SCENARIOS:
        report = reports.get(scenario["id"])
        if report is None:
            continue
        for replay_id, result in report["results"].items():
            rule = result["decision_snapshot"]["rule_evaluation"]
            features = rule["features"]
            row = {
                "scenario_id": scenario["id"],
                "replay_id": replay_id,
                "forced_experimental_action": result["action"],
                "direction": scenario["direction"],
                "speed_class": scenario["speed_class"],
                "decision_zone": scenario["decision_zone"],
                "qos_pair": scenario["qos_pair"],
                "rule_decision": rule["decision"],
                "rule_status": rule["status"],
                "pilot_valid": result["pilot_valid"],
                "evidence_directory": result["evidence_directory"],
            }
            row.update({name: features[name] for name in heterogeneous.AI_FEATURE_ORDER})
            measured_rows.append(row)
        recommendation = report["review_recommendation"]
        eligible = (
            report["mechanical_checks_pass"]
            and report["all_snapshots_conflict"]
            and recommendation["recommendation_for_human_review"] != "INCONCLUSIVE"
        )
        review_rows.append({
            "scenario_id": scenario["id"],
            "direction": scenario["direction"],
            "speed_class": scenario["speed_class"],
            "decision_zone": scenario["decision_zone"],
            "qos_pair": scenario["qos_pair"],
            "mechanical_checks_pass": report["mechanical_checks_pass"],
            "all_snapshots_conflict": report["all_snapshots_conflict"],
            "automated_review_recommendation": recommendation[
                "recommendation_for_human_review"
            ],
            "materially_better_metrics": "|".join(
                recommendation["materially_better_metrics"]
            ),
            "review_status": "REVIEW_CANDIDATE" if eligible else "EXCLUDED_OR_INCONCLUSIVE",
            "human_label": "",
            "human_reviewer": "",
            "human_notes": "",
        })

    for filename, fields, rows in (
        ("measured_decision_rows.csv", MEASURED_FIELDS, measured_rows),
        ("paired_outcome_review.csv", REVIEW_FIELDS, review_rows),
    ):
        with (folder / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
    return {
        "measured_replay_rows": len(measured_rows),
        "paired_review_rows": len(review_rows),
        "review_candidates": sum(row["review_status"] == "REVIEW_CANDIDATE"
                                 for row in review_rows),
        "training_labels_assigned": sum(bool(row["human_label"]) for row in review_rows),
    }


def _new_campaign_folder(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = root / "results" / (CAMPAIGN_PREFIX + stamp)
    folder.mkdir(parents=True)
    _strict_write_json(folder / "plan.json", PLAN)
    return folder


def _latest_resumable_folder(root: Path) -> Path:
    folders = sorted((root / "results").glob(CAMPAIGN_PREFIX + "*"), reverse=True)
    for folder in folders:
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") not in ("collection_complete", "collection_complete_with_exclusions"):
            if manifest.get("plan_sha256") != _plan_sha256():
                raise RuntimeError("Latest incomplete campaign uses a different collection plan")
            return folder
    raise RuntimeError("No incomplete measured-data campaign is available to resume")


def run_collection(root: Path, resume: bool) -> int:
    import hosn_switch as helper
    import mn_wifi.net as wifi_module

    folder = _latest_resumable_folder(root) if resume else _new_campaign_folder(root)
    manifest_path = folder / "manifest.json"
    reports = _collect_reports(folder)
    source_names = (
        "hosn_measured_dataset.py", "hosn_paired_outcome_pilot.py",
        "hosn_wifi_5g_compare.py", "hosn_wifi_5g_pilot.py",
        "hosn_heterogeneous_controller.py", "hosn_switch.py",
    )
    manifest = {
        "revision": REVISION,
        "status": "starting",
        "created_utc": folder.name.removeprefix(CAMPAIGN_PREFIX),
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": _plan_sha256(),
        "data_kind": heterogeneous.REQUIRED_TRAINING_DATA_KIND,
        "synthetic_data": False,
        "scenario_count": len(CAMPAIGN_SCENARIOS),
        "planned_replays": len(CAMPAIGN_SCENARIOS) * len(paired.RUN_ORDER),
        "completed_scenarios": len(reports),
        "completed_replays": len(reports) * len(paired.RUN_ORDER),
        "training_labels_assigned": 0,
        "python": platform.python_version(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "mininet_wifi_version": str(getattr(wifi_module, "VERSION", "unknown")),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in source_names
        },
    }
    _strict_write_json(manifest_path, manifest)
    network = None
    old_cwd = Path.cwd()
    try:
        os.chdir(folder)
        network, station, server, wifi_core, cell_gateway, ap = base.create_network()
        run = helper.NodeCommands(folder / "commands.jsonl")
        wifi_if = station.wintfs[0].name
        base.verified_interface(station, cell_gateway, "sta1-5g0")
        base.verified_interface(server, wifi_core, "h1-wifi0")
        base.verified_interface(server, cell_gateway, "h1-5g0")
        base.configure_ip(station, wifi_if, base.WIFI_UE_IP + "/24", run)
        base.configure_ip(station, "sta1-5g0", base.CELL_UE_IP + "/24", run)
        base.configure_ip(server, "h1-wifi0", base.WIFI_SERVER_IP + "/24", run)
        base.configure_ip(server, "h1-5g0", base.CELL_SERVER_IP + "/24", run)
        seed_capability = compare.detect_netem_seed_support(station, "sta1-5g0", run)
        manifest["netem_seed_capability"] = seed_capability

        for scenario_index, scenario in enumerate(CAMPAIGN_SCENARIOS, start=1):
            if scenario["id"] in reports:
                print("Skipping completed scenario", scenario["id"], flush=True)
                continue
            scenario_dir = folder / scenario["id"]
            scenario_dir.mkdir(exist_ok=True)
            attempt_dir = _attempt_directory(scenario_dir)
            print("\nSCENARIO {}/{}: {}".format(
                scenario_index, len(CAMPAIGN_SCENARIOS), scenario["id"]
            ), flush=True)
            results = {}
            for replay_id, action in paired.RUN_ORDER:
                replay_dir = attempt_dir / replay_id
                replay_dir.mkdir()
                print("  {} ({})".format(replay_id, action), flush=True)
                result = paired.run_one(
                    action, scenario, replay_dir, station, server, cell_gateway, ap,
                    seed_capability["supported"], run, helper,
                )
                result["evidence_directory"] = str(
                    Path(scenario["id"]) / attempt_dir.name / replay_id
                )
                results[replay_id] = result
                post = result["post_decision"]
                print("    loss={:.3f}% gap={:.3f}ms frames={:.2f}%".format(
                    post["loss_pct"], post["max_interarrival_gap_ms"],
                    post["frames"]["complete_pct"],
                ), flush=True)

            aggregate = paired.aggregate_by_action(results)
            recommendation = paired.review_recommendation(aggregate)
            statuses = [
                result["decision_snapshot"]["rule_evaluation"]["status"]
                for result in results.values()
            ]
            report = {
                "revision": REVISION,
                "scenario_complete": True,
                "scenario": scenario,
                "attempt_directory": attempt_dir.name,
                "results": results,
                "aggregate_by_action": aggregate,
                "pairing_diagnostics": paired.pairing_diagnostics(results),
                "review_recommendation": recommendation,
                "mechanical_checks_pass": all(item["pilot_valid"] for item in results.values()),
                "observed_rule_statuses": statuses,
                "all_snapshots_conflict": all(status == "CONFLICT" for status in statuses),
                "training_label_assigned": False,
            }
            _strict_write_json(scenario_dir / "scenario_report.json", report)
            reports[scenario["id"]] = report
            export_summary = export_tables(folder, reports)
            manifest.update(
                status="in_progress",
                last_updated_utc=datetime.now(timezone.utc).isoformat(),
                completed_scenarios=len(reports),
                completed_replays=len(reports) * len(paired.RUN_ORDER),
                export_summary=export_summary,
                training_labels_assigned=0,
            )
            _strict_write_json(manifest_path, manifest)
            print("  Review recommendation:",
                  recommendation["recommendation_for_human_review"], flush=True)

        export_summary = export_tables(folder, reports)
        exclusions = [
            scenario_id for scenario_id, report in reports.items()
            if not report["mechanical_checks_pass"]
            or not report["all_snapshots_conflict"]
            or report["review_recommendation"]["recommendation_for_human_review"]
            == "INCONCLUSIVE"
        ]
        final_report = {
            "revision": REVISION,
            "collection_complete": len(reports) == len(CAMPAIGN_SCENARIOS),
            "synthetic_data": False,
            "training_performed": False,
            "training_labels_assigned": 0,
            "completed_scenarios": len(reports),
            "completed_replays": len(reports) * len(paired.RUN_ORDER),
            "export_summary": export_summary,
            "excluded_or_inconclusive_scenarios": exclusions,
            "next_step": "human review; do not train directly from automated recommendations",
        }
        _strict_write_json(folder / "campaign_report.json", final_report)
        status = "collection_complete" if not exclusions else "collection_complete_with_exclusions"
        manifest.update(
            status=status,
            last_updated_utc=datetime.now(timezone.utc).isoformat(),
            completed_scenarios=len(reports),
            completed_replays=len(reports) * len(paired.RUN_ORDER),
            export_summary=export_summary,
            exclusions=exclusions,
            report_file="campaign_report.json",
            training_labels_assigned=0,
        )
        _strict_write_json(manifest_path, manifest)
        print("\nMEASURED COLLECTION COMPLETE")
        print("Completed replays:", final_report["completed_replays"])
        print("Human-review candidates:", export_summary["review_candidates"])
        print("Training labels assigned: 0")
        print("Saved:", folder)
        return 0
    except KeyboardInterrupt:
        manifest.update(
            status="interrupted",
            last_updated_utc=datetime.now(timezone.utc).isoformat(),
            completed_scenarios=len(reports),
            completed_replays=len(reports) * len(paired.RUN_ORDER),
            training_labels_assigned=0,
        )
        export_tables(folder, reports)
        return 130
    except Exception as exc:
        manifest.update(
            status="failed",
            last_updated_utc=datetime.now(timezone.utc).isoformat(),
            error=type(exc).__name__ + ": " + str(exc),
            completed_scenarios=len(reports),
            completed_replays=len(reports) * len(paired.RUN_ORDER),
            training_labels_assigned=0,
        )
        (folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        export_tables(folder, reports)
        print("\nMEASURED COLLECTION STOPPED:", exc)
        print("After fixing the problem, use --resume-latest.")
        return 1
    finally:
        if network is not None:
            network.stop()
        os.chdir(old_cwd)
        _strict_write_json(manifest_path, manifest)
        try:
            helper.restore_result_owner(folder)
        except Exception:
            pass


def run_self_tests() -> int:
    import tempfile
    import unittest

    class Tests(unittest.TestCase):
        def test_campaign_size_and_balance(self):
            self.assertEqual(len(CAMPAIGN_SCENARIOS), 32)
            self.assertEqual(PLAN["planned_replays"], 128)
            counts = {}
            for item in CAMPAIGN_SCENARIOS:
                key = (item["decision_zone"], item["direction"], item["speed_class"])
                counts[key] = counts.get(key, 0) + 1
            self.assertEqual(set(counts.values()), {4})
            self.assertEqual(len(counts), 8)

        def test_all_scenarios_are_conflict_designs(self):
            self.assertTrue(all(item["design_role"].startswith("conflict_")
                                for item in CAMPAIGN_SCENARIOS))

        def test_scenario_ids_and_seeds_unique(self):
            self.assertEqual(len({item["id"] for item in CAMPAIGN_SCENARIOS}), 32)
            self.assertEqual(len({item["seed_offset"] for item in CAMPAIGN_SCENARIOS}), 32)

        def test_profiles_and_timing_valid(self):
            for scenario in CAMPAIGN_SCENARIOS:
                self.assertIn(scenario["cell_profile"], ALL_PROFILES)
                decision = scenario["movement"][scenario["decision_stage_index"]]
                self.assertIn(decision["wifi_profile"], ALL_PROFILES)
                self.assertGreater(scenario["duration_s"], decision["at_s"] + 5.0)

        def test_configured_conflict_families_match_controller_contract(self):
            policy = compare.PLAN["hosn_policy"]
            for zone, pairs, wifi_signal in (
                ("edge", EDGE_QOS_PAIRS, -86.0),
                ("near", NEAR_QOS_PAIRS, -60.0),
            ):
                for wifi_name, cell_name in pairs:
                    wifi_profile = DATASET_PROFILES[wifi_name]
                    cell_profile = DATASET_PROFILES[cell_name]
                    current = heterogeneous.AccessObservation(
                        "wifi", "Wi-Fi RSSI", wifi_signal,
                        policy["wifi_service_floor_dbm"],
                        wifi_profile["delay_ms"] * 2,
                        wifi_profile["loss_pct"], -1.0,
                    )
                    candidate = heterogeneous.AccessObservation(
                        "emulated_5g", "emulated cellular",
                        policy["emulated_cellular_rsrp_dbm"],
                        policy["emulated_cellular_service_floor_rsrp_dbm"],
                        cell_profile["delay_ms"] * 2,
                        cell_profile["loss_pct"], 0.0,
                    )
                    rule = heterogeneous.evaluate_rules(
                        heterogeneous.build_features(current, candidate)
                    )
                    self.assertEqual(rule.status, "CONFLICT", (zone, wifi_name, cell_name))

        def test_no_training_labels_or_synthetic_data(self):
            self.assertFalse(PLAN["synthetic_data"])
            self.assertFalse(PLAN["training_performed"])
            self.assertEqual(PLAN["training_labels_assigned"], 0)
            self.assertNotIn("label", MEASURED_FIELDS)
            self.assertIn("human_label", REVIEW_FIELDS)

        def test_export_keeps_human_labels_blank(self):
            features = {name: float(index) for index, name in
                        enumerate(heterogeneous.AI_FEATURE_ORDER)}
            scenario = CAMPAIGN_SCENARIOS[0]
            result = {
                "action": paired.STAY, "pilot_valid": True,
                "evidence_directory": "evidence",
                "decision_snapshot": {"rule_evaluation": {
                    "decision": "ASK_AI", "status": "CONFLICT", "features": features,
                }},
            }
            report = {
                "results": {run_id: dict(result, action=action)
                            for run_id, action in paired.RUN_ORDER},
                "mechanical_checks_pass": True,
                "all_snapshots_conflict": True,
                "review_recommendation": {
                    "recommendation_for_human_review": paired.HANDOVER,
                    "materially_better_metrics": ["mean_post_loss_pct"],
                },
            }
            with tempfile.TemporaryDirectory() as name:
                summary = export_tables(Path(name), {scenario["id"]: report})
                rows = list(csv.DictReader(
                    (Path(name) / "paired_outcome_review.csv").open(encoding="utf-8")
                ))
            self.assertEqual(summary["training_labels_assigned"], 0)
            self.assertEqual(rows[0]["human_label"], "")
            self.assertEqual(rows[0]["review_status"], "REVIEW_CANDIDATE")

        def test_plan_is_strict_json(self):
            json.dumps(PLAN, allow_nan=False)

    result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    )
    if result.wasSuccessful():
        print("PASS: {} measured-collector software checks. No network run.".format(
            result.testsRun
        ))
        print("Synthetic rows: 0. Training labels: 0.")
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measured paired HOSN conflict collector; no automatic labels"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--collect", action="store_true")
    mode.add_argument("--resume-latest", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return run_self_tests()
    if args.plan:
        print(json.dumps(PLAN, indent=2, allow_nan=False))
        return 0
    if sys.platform != "linux" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        parser.exit(2, "Run collection in Ubuntu with sudo.\n")
    missing = [name for name in ("ip", "iw", "ping", "tc") if not shutil.which(name)]
    if missing:
        parser.exit(2, "Missing required tools: " + ", ".join(missing) + "\n")
    root = Path(__file__).resolve().parent
    needed = (
        "hosn_paired_outcome_pilot.py", "hosn_wifi_5g_compare.py",
        "hosn_wifi_5g_pilot.py", "hosn_heterogeneous_controller.py", "hosn_switch.py",
    )
    if any(not (root / name).is_file() for name in needed):
        parser.exit(2, "Keep this file beside " + ", ".join(needed) + ".\n")
    return run_collection(root, resume=args.resume_latest)


if __name__ == "__main__":
    raise SystemExit(main())
