#!/usr/bin/env python3
"""Final controlled three-arm HOSN comparison on unseen scenario profiles.

Arms:
  RSSI_BASELINE    Wi-Fi RSSI threshold with two confirmations.
  HOSN_RULES_ONLY  normalized signal/QoS rules; conflicts safely STAY/HOLD.
  HOSN_FULL_AI     identical rules, with the reviewed model called only for
                   complete conflicts.

Eight evaluation scenarios use profile values not present in the training
campaign.  Each strategy runs twice per scenario in a counterbalanced order,
for 48 measured replays.  All arms share the same movement, traffic, probes,
profiles, seed request, and make-before-break executor.  The expected action is
retained only for evaluation; it is never supplied to a controller.

Commands:
  python3 hosn_final_comparison.py --self-test
  python3 hosn_final_comparison.py --plan
  sudo python3 hosn_final_comparison.py --run
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
import platform
import shutil
import statistics
import sys
import traceback
from typing import Any

import hosn_heterogeneous_controller as heterogeneous
import hosn_paired_outcome_pilot as paired
import hosn_strategy_router as router
import hosn_wifi_5g_compare as compare
import hosn_wifi_5g_pilot as base


REVISION = "final-three-arm-unseen-evaluation-v1"
STRATEGIES = router.STRATEGIES
REPLAY_ORDER = (
    ("rssi_1", router.RSSI_BASELINE),
    ("rules_1", router.HOSN_RULES_ONLY),
    ("ai_1", router.HOSN_FULL_AI),
    ("ai_2", router.HOSN_FULL_AI),
    ("rules_2", router.HOSN_RULES_ONLY),
    ("rssi_2", router.RSSI_BASELINE),
)

EVALUATION_PROFILES = {
    "wifi_eval_good": {
        "delay_ms": 10.0, "jitter_ms": 2.5, "loss_pct": 0.6, "rate_mbit": 16.0,
    },
    "wifi_eval_poor": {
        "delay_ms": 38.0, "jitter_ms": 7.5, "loss_pct": 3.0, "rate_mbit": 9.0,
    },
    "wifi_eval_bad": {
        "delay_ms": 58.0, "jitter_ms": 11.0, "loss_pct": 5.5, "rate_mbit": 6.5,
    },
    "cell_eval_good": {
        "delay_ms": 16.0, "jitter_ms": 2.5, "loss_pct": 0.3, "rate_mbit": 18.0,
    },
    "cell_eval_fair": {
        "delay_ms": 28.0, "jitter_ms": 5.0, "loss_pct": 1.2, "rate_mbit": 12.0,
    },
    "cell_eval_poor": {
        "delay_ms": 52.0, "jitter_ms": 10.0, "loss_pct": 3.8, "rate_mbit": 7.5,
    },
}
paired.ALL_ACCESS_PROFILES.update(EVALUATION_PROFILES)


def movement(direction: str, speed: str, zone: str,
             decision_profile: str) -> list[dict]:
    times = (0.0, 4.0, 8.0) if speed == "fast" else (0.0, 6.0, 12.0)
    configurations = {
        ("outbound", "edge"): (
            (20.0, "wifi_near"), (40.0, "wifi_moving"), (65.0, decision_profile),
        ),
        ("inbound", "edge"): (
            (70.0, "wifi_edge"), (67.0, "wifi_edge"), (65.0, decision_profile),
        ),
        ("outbound", "near"): (
            (0.0, "wifi_near"), (10.0, "wifi_near"), (20.0, decision_profile),
        ),
        ("inbound", "near"): (
            (65.0, "wifi_edge"), (40.0, "wifi_moving"), (20.0, decision_profile),
        ),
    }
    points = configurations[(direction, zone)]
    return [
        {"at_s": at_s, "x_m": x_m, "wifi_profile": profile}
        for at_s, (x_m, profile) in zip(times, points)
    ]


def scenario(identifier: str, direction: str, speed: str, zone: str,
             wifi_profile: str, cell_profile: str, case_type: str,
             expected_action: str, seed_offset: int) -> dict:
    return {
        "id": identifier,
        "direction": direction,
        "speed_class": speed,
        "decision_zone": zone,
        "case_type": case_type,
        "expected_action_for_evaluation_only": expected_action,
        "movement": movement(direction, speed, zone, wifi_profile),
        "decision_stage_index": 2,
        "cell_profile": cell_profile,
        "design_role": case_type,
        "duration_s": 18.0 if speed == "fast" else 22.0,
        "seed_offset": seed_offset,
    }


EVALUATION_SCENARIOS = (
    scenario("eval01_edge_out_fast_conflict_stay", "outbound", "fast", "edge",
             "wifi_eval_good", "cell_eval_poor", "conflict", paired.STAY, 4100),
    scenario("eval02_edge_in_slow_conflict_stay", "inbound", "slow", "edge",
             "wifi_eval_good", "cell_eval_poor", "conflict", paired.STAY, 4200),
    scenario("eval03_near_out_slow_conflict_handover", "outbound", "slow", "near",
             "wifi_eval_poor", "cell_eval_good", "conflict", paired.HANDOVER, 4300),
    scenario("eval04_near_in_fast_conflict_handover", "inbound", "fast", "near",
             "wifi_eval_poor", "cell_eval_good", "conflict", paired.HANDOVER, 4400),
    scenario("eval05_edge_out_slow_clear_handover", "outbound", "slow", "edge",
             "wifi_eval_bad", "cell_eval_good", "clear", paired.HANDOVER, 4500),
    scenario("eval06_edge_in_fast_clear_handover", "inbound", "fast", "edge",
             "wifi_eval_bad", "cell_eval_good", "clear", paired.HANDOVER, 4600),
    scenario("eval07_near_out_fast_clear_stay", "outbound", "fast", "near",
             "wifi_eval_good", "cell_eval_poor", "clear", paired.STAY, 4700),
    scenario("eval08_near_in_slow_clear_stay", "inbound", "slow", "near",
             "wifi_eval_good", "cell_eval_poor", "clear", paired.STAY, 4800),
)

PLAN = {
    "revision": REVISION,
    "purpose": "final identical-condition comparison of three distinct strategies",
    "strategies": list(STRATEGIES),
    "scenarios": list(EVALUATION_SCENARIOS),
    "replay_order_per_scenario": [
        {"id": replay_id, "strategy": strategy}
        for replay_id, strategy in REPLAY_ORDER
    ],
    "scenario_count": len(EVALUATION_SCENARIOS),
    "replays_per_strategy_per_scenario": 2,
    "planned_replays": len(EVALUATION_SCENARIOS) * len(REPLAY_ORDER),
    "training_rows_created": 0,
    "training_labels_assigned": 0,
    "evaluation_profiles_used_for_training": False,
    "expected_action_is_controller_input": False,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expose_sudo_invoker_user_packages() -> None:
    """Expose packages installed with `pip --user` to the sudo-run harness."""
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return
    sudo_uid = os.environ.get("SUDO_UID")
    if sudo_uid is None:
        return
    try:
        import pwd
        import site
        user_home = Path(pwd.getpwuid(int(sudo_uid)).pw_dir)
        candidate = user_home / ".local" / "lib" / (
            "python{}.{}".format(sys.version_info.major, sys.version_info.minor)
        ) / "site-packages"
    except (KeyError, ValueError):
        return
    if candidate.is_dir():
        site.addsitedir(str(candidate))


def find_accepted_model(root: Path):
    expose_sudo_invoker_user_packages()
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to load the accepted model") from exc
    folders = sorted((root / "results").glob("measured_dataset_campaign_*"), reverse=True)
    for folder in folders:
        report_path = folder / "conflict_ai_training_report.json"
        model_path = folder / "hosn_conflict_ai_model.joblib"
        if not report_path.is_file() or not model_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("acceptance_gate", {}).get("accepted") is not True:
            continue
        if report.get("model_sha256") != sha256(model_path):
            raise ValueError("Accepted model hash does not match its training report")
        model = joblib.load(model_path)
        if getattr(model, "hosn_schema_version", None) != heterogeneous.MODEL_SCHEMA_VERSION:
            raise ValueError("Model schema version mismatch")
        if tuple(getattr(model, "feature_order", ())) != heterogeneous.AI_FEATURE_ORDER:
            raise ValueError("Model feature order mismatch")
        if (getattr(model, "training_data_kind", None)
                != heterogeneous.REQUIRED_TRAINING_DATA_KIND):
            raise ValueError("Model training-data contract mismatch")
        return model, model_path, report
    raise RuntimeError("No accepted conflict-only AI model was found")


def observations(snapshot: dict):
    policy = compare.PLAN["hosn_policy"]
    current = heterogeneous.AccessObservation(
        "wifi", "Wi-Fi RSSI", snapshot["wifi_rssi_model_dbm"],
        policy["wifi_service_floor_dbm"], snapshot["wifi_rtt_ms"],
        snapshot["wifi_loss_pct"], snapshot["wifi_trend_db_per_s"],
    )
    candidate = heterogeneous.AccessObservation(
        "emulated_5g", "emulated cellular/5G-like IP path",
        snapshot["cell_rsrp_configured_dbm"],
        policy["emulated_cellular_service_floor_rsrp_dbm"],
        snapshot["cell_rtt_ms"], snapshot["cell_loss_pct"],
        snapshot["cell_trend_db_per_s"],
    )
    return current, candidate


def decide(strategy: str, snapshot: dict, ai_model) -> dict:
    current, candidate = observations(snapshot)
    if strategy == router.RSSI_BASELINE:
        state = router.BaselineState(
            threshold_dbm=compare.RSSI_BASELINE_THRESHOLD_DBM,
            required_confirmations=compare.RSSI_CONFIRMATIONS,
        )
        routed = None
        for _ in range(compare.RSSI_CONFIRMATIONS):
            routed = router.route_decision(
                strategy, current, candidate, baseline_state=state,
            )
    elif strategy == router.HOSN_RULES_ONLY:
        routed = router.route_decision(strategy, current, candidate)
    else:
        routed = router.route_decision(
            strategy, current, candidate, ai_model=ai_model,
        )
    value = routed.to_dict()
    return {
        "action": value["decision"],
        "strategy": strategy,
        "source": value["source"],
        "status": value["status"],
        "reason": value["reason"],
        "handover_authorized": value["handover_authorized"],
        "ai_called": value["ai_called"],
        "controller_result": value["controller_result"],
    }


def aggregate(results: list[dict]) -> dict:
    output = {}
    for strategy in STRATEGIES:
        items = [item for item in results if item["strategy"] == strategy]
        post = [item["post_decision"] for item in items]
        decisions = [item["controller_decision"] for item in items]
        correct = [
            item["action"] == item["expected_action_for_evaluation_only"]
            for item in items
        ]
        output[strategy] = {
            "replays": len(items),
            "decision_accuracy_pct": 100.0 * sum(correct) / len(correct),
            "handovers": sum(item["action"] == paired.HANDOVER for item in items),
            "stays": sum(item["action"] == paired.STAY for item in items),
            "ai_calls": sum(bool(item["ai_called"]) for item in decisions),
            "mean_post_loss_pct": statistics.fmean(item["loss_pct"] for item in post),
            "mean_post_p95_delay_ms": statistics.fmean(
                item["p95_one_way_process_delay_ms"] for item in post
            ),
            "mean_post_jitter_ms": statistics.fmean(
                item["rfc3550_interarrival_jitter_ms"] for item in post
            ),
            "mean_post_max_gap_ms": statistics.fmean(
                item["max_interarrival_gap_ms"] for item in post
            ),
            "mean_post_goodput_mbps": statistics.fmean(
                item["application_goodput_mbps"] for item in post
            ),
            "mean_post_complete_frames_pct": statistics.fmean(
                item["frames"]["complete_pct"] for item in post
            ),
            "all_mechanical_checks_pass": all(item["pilot_valid"] for item in items),
        }
    return output


def write_csv(folder: Path, results: list[dict]) -> None:
    fields = (
        "scenario_id", "case_type", "direction", "speed_class", "decision_zone",
        "replay_id", "strategy", "action", "expected_action", "decision_correct",
        "decision_source", "decision_status", "ai_called", "post_loss_pct",
        "post_p95_delay_ms", "post_jitter_ms", "post_max_gap_ms",
        "post_complete_frames_pct", "post_goodput_mbps", "pilot_valid",
    )
    rows = []
    for item in results:
        post = item["post_decision"]
        decision = item["controller_decision"]
        rows.append({
            "scenario_id": item["scenario_id"], "case_type": item["case_type"],
            "direction": item["direction"], "speed_class": item["speed_class"],
            "decision_zone": item["decision_zone"], "replay_id": item["replay_id"],
            "strategy": item["strategy"], "action": item["action"],
            "expected_action": item["expected_action_for_evaluation_only"],
            "decision_correct": item["action"] == item["expected_action_for_evaluation_only"],
            "decision_source": decision["source"], "decision_status": decision["status"],
            "ai_called": decision["ai_called"], "post_loss_pct": post["loss_pct"],
            "post_p95_delay_ms": post["p95_one_way_process_delay_ms"],
            "post_jitter_ms": post["rfc3550_interarrival_jitter_ms"],
            "post_max_gap_ms": post["max_interarrival_gap_ms"],
            "post_complete_frames_pct": post["frames"]["complete_pct"],
            "post_goodput_mbps": post["application_goodput_mbps"],
            "pilot_valid": item["pilot_valid"],
        })
    with (folder / "final_comparison_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def run_final(root: Path) -> int:
    import hosn_switch as helper
    import mn_wifi.net as wifi_module

    ai_model, model_path, training_report = find_accepted_model(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = root / "results" / ("final_three_arm_comparison_" + stamp)
    folder.mkdir(parents=True)
    compare.write_json(folder / "plan.json", PLAN)
    source_names = (
        "hosn_final_comparison.py", "hosn_paired_outcome_pilot.py",
        "hosn_strategy_router.py", "hosn_heterogeneous_controller.py",
        "hosn_wifi_5g_compare.py", "hosn_wifi_5g_pilot.py", "hosn_switch.py",
    )
    manifest = {
        "revision": REVISION, "status": "starting", "created_utc": stamp,
        "data_kind": "final_controlled_evaluation_not_training_data",
        "planned_replays": PLAN["planned_replays"],
        "training_rows_created": 0, "training_labels_assigned": 0,
        "model_loaded": True, "model_sha256": sha256(model_path),
        "model_training_report_sha256": sha256(
            model_path.parent / "conflict_ai_training_report.json"
        ),
        "python": platform.python_version(), "kernel": platform.release(),
        "machine": platform.machine(),
        "mininet_wifi_version": str(getattr(wifi_module, "VERSION", "unknown")),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in source_names
        },
    }
    compare.write_json(folder / "manifest.json", manifest)
    network = None
    old_cwd = Path.cwd()
    results = []
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

        for scenario_index, spec in enumerate(EVALUATION_SCENARIOS, start=1):
            scenario_dir = folder / spec["id"]
            scenario_dir.mkdir()
            print("\nSCENARIO {}/{}: {}".format(
                scenario_index, len(EVALUATION_SCENARIOS), spec["id"]
            ), flush=True)
            for replay_id, strategy in REPLAY_ORDER:
                replay_dir = scenario_dir / replay_id
                replay_dir.mkdir()
                print("  {} ({})".format(replay_id, strategy), flush=True)
                callback = lambda snapshot, selected=strategy: decide(
                    selected, snapshot, ai_model
                )
                result = paired.run_one(
                    paired.STAY, spec, replay_dir, station, server,
                    cell_gateway, ap, seed_capability["supported"], run, helper,
                    decision_callback=callback,
                )
                result.update(
                    replay_id=replay_id, strategy=strategy,
                    scenario_id=spec["id"], case_type=spec["case_type"],
                    direction=spec["direction"], speed_class=spec["speed_class"],
                    decision_zone=spec["decision_zone"],
                    expected_action_for_evaluation_only=spec[
                        "expected_action_for_evaluation_only"
                    ],
                )
                results.append(result)
                post = result["post_decision"]
                print("    action={} source={} loss={:.3f}% gap={:.2f}ms frames={:.2f}%".format(
                    result["action"], result["controller_decision"]["source"],
                    post["loss_pct"], post["max_interarrival_gap_ms"],
                    post["frames"]["complete_pct"],
                ), flush=True)
                compare.write_json(replay_dir / "final_summary.json", result)

        aggregate_by_strategy = aggregate(results)
        all_valid = all(item["pilot_valid"] for item in results)
        ai_sources_valid = all(
            item["controller_decision"]["source"] == "AI"
            for item in results
            if item["strategy"] == router.HOSN_FULL_AI
            and item["case_type"] == "conflict"
        )
        ai_clear_bypass_valid = all(
            not item["controller_decision"]["ai_called"]
            for item in results
            if item["strategy"] == router.HOSN_FULL_AI
            and item["case_type"] == "clear"
        )
        report = {
            "revision": REVISION,
            "final_controlled_evaluation": True,
            "training_data_created": False,
            "training_labels_assigned": 0,
            "completed_replays": len(results),
            "mechanical_checks_pass": all_valid,
            "ai_used_only_for_conflicts": ai_sources_valid and ai_clear_bypass_valid,
            "aggregate_by_strategy": aggregate_by_strategy,
            "model_training_summary": {
                "source_campaign": training_report["source_campaign"],
                "independent_groups": training_report["independent_scenario_groups"],
                "group_balanced_accuracy": training_report[
                    "cross_validation"
                ]["group_metrics"]["balanced_accuracy"],
            },
            "honesty_note": (
                "Evaluation profiles differ from training profiles, but all results remain "
                "controlled emulation on one Mininet-WiFi environment, not an external field trial."
            ),
        }
        compare.write_json(folder / "final_comparison_report.json", report)
        write_csv(folder, results)
        valid = (
            all_valid and ai_sources_valid and ai_clear_bypass_valid
            and len(results) == PLAN["planned_replays"]
        )
        manifest.update(
            status="final_comparison_complete" if valid else "final_comparison_checks_failed",
            completed_replays=len(results),
            report_file="final_comparison_report.json",
            mechanical_checks_pass=all_valid,
            ai_routing_checks_pass=ai_sources_valid and ai_clear_bypass_valid,
        )
        compare.write_json(folder / "manifest.json", manifest)
        print("\nFINAL THREE-ARM COMPARISON {}".format(
            "COMPLETE" if valid else "CHECKS FAILED"
        ))
        print("Completed replays:", len(results))
        for strategy in STRATEGIES:
            item = aggregate_by_strategy[strategy]
            print("{}: decision={:.2f}% loss={:.3f}% gap={:.2f}ms frames={:.2f}%".format(
                strategy, item["decision_accuracy_pct"], item["mean_post_loss_pct"],
                item["mean_post_max_gap_ms"], item["mean_post_complete_frames_pct"],
            ))
        print("Saved:", folder)
        return 0 if valid else 1
    except KeyboardInterrupt:
        manifest.update(status="interrupted", completed_replays=len(results))
        return 130
    except Exception as exc:
        manifest.update(
            status="failed", completed_replays=len(results),
            error=type(exc).__name__ + ": " + str(exc),
        )
        (folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("\nFINAL COMPARISON STOPPED:", exc)
        return 1
    finally:
        if network is not None:
            network.stop()
        os.chdir(old_cwd)
        compare.write_json(folder / "manifest.json", manifest)
        try:
            helper.restore_result_owner(folder)
        except Exception:
            pass


class _TestModel:
    hosn_schema_version = heterogeneous.MODEL_SCHEMA_VERSION
    feature_order = heterogeneous.AI_FEATURE_ORDER
    training_data_kind = heterogeneous.REQUIRED_TRAINING_DATA_KIND

    def __init__(self, action: str):
        self.action = action
        self.calls = 0

    def predict(self, rows):
        self.calls += 1
        return [self.action for _ in rows]


def snapshot(zone: str, wifi_rtt: float, cell_rtt: float,
             wifi_loss: float, cell_loss: float) -> dict:
    wifi_signal = -86.0 if zone == "edge" else -60.0
    return {
        "wifi_rssi_model_dbm": wifi_signal,
        "cell_rsrp_configured_dbm": -90.0,
        "wifi_rtt_ms": wifi_rtt, "cell_rtt_ms": cell_rtt,
        "wifi_loss_pct": wifi_loss, "cell_loss_pct": cell_loss,
        "wifi_trend_db_per_s": -1.0, "cell_trend_db_per_s": 0.0,
    }


def run_self_tests() -> int:
    import inspect
    import unittest

    class Tests(unittest.TestCase):
        def test_final_matrix_is_balanced_and_unseen(self):
            self.assertEqual(len(EVALUATION_SCENARIOS), 8)
            self.assertEqual(PLAN["planned_replays"], 48)
            self.assertEqual(Counter(item["case_type"] for item in EVALUATION_SCENARIOS),
                             Counter({"conflict": 4, "clear": 4}))
            self.assertEqual(Counter(item["direction"] for item in EVALUATION_SCENARIOS),
                             Counter({"outbound": 4, "inbound": 4}))
            self.assertEqual(Counter(item["speed_class"] for item in EVALUATION_SCENARIOS),
                             Counter({"fast": 4, "slow": 4}))
            training_names = {
                "wifi_data_excellent", "wifi_data_good", "wifi_data_fair",
                "wifi_data_poor", "wifi_data_bad", "cell_data_excellent",
                "cell_data_good", "cell_data_fair", "cell_data_poor", "cell_data_bad",
            }
            self.assertFalse(training_names.intersection(EVALUATION_PROFILES))

        def test_replay_order_counterbalances_all_strategies(self):
            self.assertEqual(Counter(strategy for _, strategy in REPLAY_ORDER),
                             Counter({strategy: 2 for strategy in STRATEGIES}))

        def test_edge_conflict_routing(self):
            value = snapshot("edge", 20.0, 104.0, 0.6, 3.8)
            self.assertEqual(decide(router.RSSI_BASELINE, value, None)["action"],
                             paired.HANDOVER)
            self.assertEqual(decide(router.HOSN_RULES_ONLY, value, None)["action"],
                             paired.STAY)
            model = _TestModel(paired.STAY)
            result = decide(router.HOSN_FULL_AI, value, model)
            self.assertEqual((result["action"], result["source"]), (paired.STAY, "AI"))
            self.assertEqual(model.calls, 1)

        def test_near_conflict_routing(self):
            value = snapshot("near", 76.0, 32.0, 3.0, 0.3)
            self.assertEqual(decide(router.RSSI_BASELINE, value, None)["action"], paired.STAY)
            self.assertEqual(decide(router.HOSN_RULES_ONLY, value, None)["action"], paired.STAY)
            result = decide(router.HOSN_FULL_AI, value, _TestModel(paired.HANDOVER))
            self.assertEqual((result["action"], result["source"]),
                             (paired.HANDOVER, "AI"))

        def test_clear_case_bypasses_ai(self):
            value = snapshot("edge", 116.0, 32.0, 5.5, 0.3)
            model = _TestModel(paired.STAY)
            result = decide(router.HOSN_FULL_AI, value, model)
            self.assertEqual((result["action"], result["source"]),
                             (paired.HANDOVER, "RULES"))
            self.assertEqual(model.calls, 0)

        def test_paired_engine_accepts_decision_callback(self):
            self.assertIn("decision_callback", inspect.signature(paired.run_one).parameters)

        def test_plan_is_strict_json_and_no_training(self):
            json.dumps(PLAN, allow_nan=False)
            self.assertEqual(PLAN["training_rows_created"], 0)
            self.assertEqual(PLAN["training_labels_assigned"], 0)
            self.assertFalse(PLAN["expected_action_is_controller_input"])

    result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    )
    if result.wasSuccessful():
        print("PASS: {} final-comparison software checks. No network run.".format(
            result.testsRun
        ))
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Final controlled RSSI vs rules-only vs full-AI comparison"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return run_self_tests()
    if args.plan:
        print(json.dumps(PLAN, indent=2, allow_nan=False))
        return 0
    if sys.platform != "linux" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        parser.exit(2, "Run --run in Ubuntu with sudo.\n")
    missing = [name for name in ("ip", "iw", "ping", "tc") if not shutil.which(name)]
    if missing:
        parser.exit(2, "Missing required tools: " + ", ".join(missing) + "\n")
    root = Path(__file__).resolve().parent
    needed = (
        "hosn_paired_outcome_pilot.py", "hosn_strategy_router.py",
        "hosn_heterogeneous_controller.py", "hosn_wifi_5g_compare.py",
        "hosn_wifi_5g_pilot.py", "hosn_switch.py",
    )
    if any(not (root / name).is_file() for name in needed):
        parser.exit(2, "Keep this file beside " + ", ".join(needed) + ".\n")
    return run_final(root)


if __name__ == "__main__":
    raise SystemExit(main())
