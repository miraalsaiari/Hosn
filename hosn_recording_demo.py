#!/usr/bin/env python3
"""Recording-focused ATP demo for the final HOSN Wi-Fi -> emulated-5G system.

This wrapper reuses the project's accepted measured-data AI, rules-first
controller, media-like workload, and make-before-break executor.  It runs only
three presentation scenarios, one at a time, instead of the full 48-replay
research evaluation.
Each recording plans four real monitoring observations across three movement
stages. The existing rules-first controller evaluates every actual check and
calls AI immediately if those rules find a conflict. An authorized handover
executes immediately; monitoring stops because the Wi-Fi link is disconnected.
Recording-only extra probes share links with the video-like UDP workload, so
these run outcomes should not be compared as identical research replays.

Commands (from /home/mira/Hosn):
    python3 hosn_recording_demo.py --plan
    python3 hosn_recording_demo.py --check-model
    sudo python3 hosn_recording_demo.py --scenario 1
    sudo python3 hosn_recording_demo.py --scenario 2
    sudo python3 hosn_recording_demo.py --scenario 3
    sudo python3 hosn_recording_demo.py --scenario all

Scenario 1: clear degradation; rules authorize HANDOVER and AI is bypassed.
Scenario 2: strong Wi-Fi signal but poor Wi-Fi service; AI resolves conflict.
Scenario 3: weak Wi-Fi signal but good Wi-Fi service and poor candidate service;
            AI resolves the opposite conflict.

The script never loads the old synthetic hosn_ai_model.pkl.  It accepts only a
model with the measured-data contract and an accepted training report, using the
same checks as hosn_final_comparison.py.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import hosn_final_comparison as final
import hosn_paired_outcome_pilot as paired
import hosn_strategy_router as router
import hosn_wifi_5g_compare as compare
import hosn_wifi_5g_pilot as base


REVISION = "atp-recording-three-scenario-v5"
COLOR_MODE = "auto"
COLORS = {
    "info": "\033[94m", "success": "\033[92m", "warning": "\033[93m",
    "ai": "\033[95m", "critical": "\033[91m",
}

SCENARIO_SPECS = {
    "1": {
        "source_id": "eval05_edge_out_slow_clear_handover",
        "title": "CLEAR CASE: RULES AUTHORIZE HANDOVER",
        "story": (
            "The user moves away from Wi-Fi. Wi-Fi signal and service become poor, "
            "while the emulated cellular/5G-like path is clearly better."
        ),
        "point": "Designed to show a clear rules case; the observed result comes from the controller.",
        "expected_source": "RULES",
    },
    "2": {
        "source_id": "eval03_near_out_slow_conflict_handover",
        "title": "CONFLICT: STRONG WI-FI, BUT POOR WI-FI SERVICE",
        "story": (
            "Wi-Fi is still physically near/strong, but its delay and loss are poor. "
            "Signal alone favors staying; service quality favors the candidate path."
        ),
        "point": "Designed to test a conflict; AI runs only if the real rules detect one.",
        "expected_source": "AI",
    },
    "3": {
        "source_id": "eval01_edge_out_fast_conflict_stay",
        "title": "CONFLICT: WEAK WI-FI, BUT CANDIDATE SERVICE IS WORSE",
        "story": (
            "Wi-Fi signal is weak, but its measured service remains good. The candidate "
            "path has a better signal margin but poorer delay/loss."
        ),
        "point": "This shows why blindly choosing the stronger signal can be wrong.",
        "expected_source": "AI",
    },
}


def _scenario_by_id(identifier: str) -> dict:
    for item in final.EVALUATION_SCENARIOS:
        if item["id"] == identifier:
            return item
    raise RuntimeError("Required evaluation scenario is missing: " + identifier)


def _recording_scenario(number: str) -> dict:
    """Choose real movement inputs without changing the research scenario matrix."""
    scenario = copy.deepcopy(_scenario_by_id(SCENARIO_SPECS[number]["source_id"]))
    if number == "1":
        # Stage 2 stays nearer the AP to demonstrate an ordinary rules STAY;
        # the original final degradation and handover conditions stay intact.
        first, middle = scenario["movement"][:2]
        middle["x_m"] = (first["x_m"] + middle["x_m"]) / 2
        middle["wifi_profile"] = first["wifi_profile"]
        scenario["recording_intermediate_position_adjusted"] = True
    return scenario


def _line(char: str = "=", width: int = 78) -> str:
    return char * width


def _color(text: str, kind: str = "info") -> str:
    enabled = COLOR_MODE == "always" or (
        COLOR_MODE == "auto" and sys.stdout.isatty() and "NO_COLOR" not in os.environ
    )
    return COLORS.get(kind, "") + text + "\033[0m" if enabled else text


def _section(title: str, kind: str = "info") -> None:
    print("\n" + _color(_line(), kind))
    print(_color(title, kind))
    print(_color(_line(), kind), flush=True)


def _field(label: str, value: Any) -> None:
    print("  {:<25} {}".format(label, value))


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "unavailable"
    if _finite_number(value):
        return ("{:.%df}" % digits).format(float(value)) + suffix
    return str(value)


def _yes_no(value: Any) -> str:
    return "YES" if bool(value) else "NO"


def _path_label(value: Any) -> str:
    return {"wifi": "Wi-Fi AP1", "5g": "Emulated cellular/5G-like IP path"}.get(
        value, str(value) if value is not None else "unavailable"
    )


def _clear_screen() -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="", flush=True)


def _countdown(seconds: int = 3) -> None:
    if seconds <= 0:
        return
    print("\nRecording begins now. Scenario starts in:", flush=True)
    for value in range(seconds, 0, -1):
        print("  {}".format(value), flush=True)
        time.sleep(1)
    print("  START\n", flush=True)


def _print_scenario_intro(number: str, info: dict, model_info: dict) -> None:
    _section("ATP DEMONSTRATION  |  HOSN  |  SCENARIO {}".format(number))
    _field("Scenario goal", info["title"])
    _field("Application", "Measured UDP video-call-like traffic")
    _field("Current path", "Wi-Fi AP1 (emulated station)")
    _field("Candidate path", "Emulated cellular/5G-like IP path")
    _field("Objective", "Maintain video-like traffic quality during movement")
    print("\nSCENARIO SETUP\n  " + info["story"])
    print("\nINITIALIZING HOSN")
    _field("Topology interfaces", _color("VERIFIED", "success"))
    _field("Existing controller", _color("READY", "success"))
    if number != "1":
        _field("Accepted measured-data AI", _color(
            "LOADED" if model_info["accepted"] else "UNAVAILABLE",
            "success" if model_info["accepted"] else "critical",
        ))
    if number == "1":
        print("\nStarting up to four real monitoring checks. The original HOSN")
        print("controller applies its rules after every check.")
    else:
        print("\nStarting up to four real monitoring checks. Rules run after each")
        print("check; a real conflict calls the trained AI immediately.")
    print("An authorized handover executes immediately and ends Wi-Fi monitoring.")
    print("Recording probes share the media links; results describe this run.", flush=True)


def _print_monitoring_check(snapshot: dict, index: int, total: int,
                            final_check: bool, previous: dict | None) -> None:
    _section("CHECK {} / {}  |  LIVE NETWORK MONITORING".format(index, total))
    _field("Elapsed traffic time", _fmt(snapshot.get("monitor_observed_at_s"), 2, " s"))
    _field("Signal sampled at", _fmt(snapshot.get("monitor_signal_observed_at_s"), 2, " s"))
    _field("Emulated station X", _fmt(snapshot.get("emulated_station_x_m"), 1, " m"))
    _field("Selected sender path", _path_label(snapshot.get("selected_access_before_decision")))
    print("\nHOSN ACTION\n  Existing Wi-Fi and candidate path probes completed.")
    _field("Wi-Fi / candidate pings", "{} / {}".format(
        snapshot.get("wifi_probe_count", "unavailable"),
        snapshot.get("cell_probe_count", "unavailable")))
    print("\nCURRENT NETWORK  |  Wi-Fi AP1")
    _field("Modeled RSSI", _fmt(snapshot.get("wifi_rssi_model_dbm"), 2, " dBm"))
    _field("Measured RTT", _fmt(snapshot.get("wifi_rtt_ms"), 2, " ms"))
    _field("Measured packet loss", _fmt(snapshot.get("wifi_loss_pct"), 2, "%"))
    _field("Modeled signal trend", _fmt(snapshot.get("wifi_trend_db_per_s"), 3, " dB/s"))
    print("\nCANDIDATE NETWORK  |  Emulated cellular/5G-like IP path")
    _field("Configured signal", _fmt(snapshot.get("cell_rsrp_configured_dbm"), 2, " dBm"))
    _field("Measured RTT", _fmt(snapshot.get("cell_rtt_ms"), 2, " ms"))
    _field("Measured packet loss", _fmt(snapshot.get("cell_loss_pct"), 2, "%"))
    _field("Configured signal trend", _fmt(snapshot.get("cell_trend_db_per_s"), 3, " dB/s"))
    print("\nWHAT CHANGED")
    if previous is None:
        print("  This is the first measurement; no earlier check exists.")
    else:
        for title, key, suffix, digits in (
            ("Wi-Fi modeled RSSI", "wifi_rssi_model_dbm", " dBm", 2),
            ("Wi-Fi measured RTT", "wifi_rtt_ms", " ms", 2),
            ("Wi-Fi measured loss", "wifi_loss_pct", "%", 2),
            ("Candidate measured RTT", "cell_rtt_ms", " ms", 2),
            ("Candidate measured loss", "cell_loss_pct", "%", 2),
        ):
            before, after = previous.get(key), snapshot.get(key)
            if _finite_number(before) and _finite_number(after):
                _field(title, "{:+.{digits}f}{suffix} ({} -> {})".format(
                    after - before, _fmt(before, digits, suffix),
                    _fmt(after, digits, suffix), digits=digits, suffix=suffix,
                ))
            else:
                _field(title, "unavailable for comparison")
    print("\nRULES EVALUATION\n  HOSN is deciding from this check's real measurements...", flush=True)


def _print_snapshot(snapshot: dict, decision: dict, check: int, total: int) -> None:
    rule = snapshot.get("rule_evaluation", {})
    _section("RULES EVALUATION  |  CHECK {} / {}".format(check, total),
             "warning" if rule.get("status") == "CONFLICT" else "info")
    _field("Rules status", rule.get("status", "UNKNOWN"))
    _field("Rules result", rule.get("decision", "UNKNOWN"))
    _field("Reason", rule.get("reason", "No reason recorded."))
    controller = decision.get("controller_result") or {}
    if decision.get("ai_called"):
        _section("AI ANALYSIS  |  TRAINED MEASURED-DATA MODEL", "ai")
        print("  Rules found a conflict. The controller called the trained model")
        print("  with the ordered feature vector, including service quality.")
        features = controller.get("features") or {}
        order = final.heterogeneous.AI_FEATURE_ORDER
        for name in order:
            _field(name, _fmt(features.get(name), 6))
        if all(_finite_number(features.get(name)) for name in order):
            row = [[float(features[name]) for name in order]]
            _field("Exact model.predict row", repr(row))
        else:
            _field("Model input", "unavailable in controller trace")
        prediction = controller.get("ai_prediction")
        _field("Actual AI prediction", prediction if prediction is not None
               else "unavailable (model did not return a usable prediction)")
    _section("CONTROLLER DECISION  |  CHECK {} / {}".format(check, total))
    _field("Rules result", "{} | {}".format(
        rule.get("decision", "UNKNOWN"), rule.get("status", "UNKNOWN")))
    _field("AI called", _yes_no(decision.get("ai_called")))
    _field("Decision source", decision.get("source", "UNKNOWN"))
    _field("Controller status", decision.get("status", "UNKNOWN"))
    _field("Handover authorized", _yes_no(decision.get("handover_authorized")))
    _field("Final decision", _color(str(decision.get("action", "UNKNOWN")),
                                     "success" if decision.get("action") == paired.HANDOVER
                                     else "warning"))
    _field("Controller reason", decision.get("reason", "No reason recorded."))
    if decision.get("action") == paired.HANDOVER:
        _section("EXECUTING EXISTING MAKE-BEFORE-BREAK HANDOVER", "warning")
        print("  The executor will duplicate traffic, verify candidate packets,")
        print("  select the 5G-like path, and then disconnect Wi-Fi.", flush=True)
    else:
        print("  HOSN chose STAY. Wi-Fi remains selected; video-like traffic continues.")
        print("  Monitoring continues to the next check." if check < total else
              "  All planned checks are complete; no handover was authorized.", flush=True)


def _handover_switch_verified(result: dict) -> bool:
    """Use the executor's recorded packet and Wi-Fi checks, not the intended action."""
    timing = result.get("timing") or {}
    checks = result.get("checks") or {}
    return result.get("action") == paired.HANDOVER and all((
        timing.get("candidate_verified_ns") is not None,
        timing.get("wifi_break_ns") is not None,
        timing.get("wifi_disconnected_verified") is True,
        checks.get("handover_used_both_paths") is True,
        checks.get("candidate_verified_before_break") is True,
        checks.get("wifi_disconnect_verified") is True,
    ))


def _print_result(result: dict, expected_action: str | None = None,
                  expected_source: str | None = None) -> bool:
    post = result["post_decision"]
    decision = result["controller_decision"]
    timing = result.get("timing", {})
    history = result.get("recording_checks", ())
    ai_called_any = any(bool(item.get("controller_decision", {}).get("ai_called"))
                        for item in history)
    switch_verified = _handover_switch_verified(result)
    _section("HOSN DECISIONS ACROSS THE MONITORING CHECKS")
    for item in history:
        check_decision = item.get("controller_decision") or {}
        _field("Check {} / 4".format(item["number"]),
               "{} via {} | AI called {}".format(
                   check_decision.get("action", "UNKNOWN"),
                   check_decision.get("source", "UNKNOWN"),
                   _yes_no(check_decision.get("ai_called"))))
    _field("AI used in this scenario", _yes_no(ai_called_any))
    _field("Last decision source", decision.get("source", "UNKNOWN"))
    _section("POST-HANDOVER VERIFICATION" if result["action"] == paired.HANDOVER
             else "POST-DECISION VERIFICATION",
             "success" if result.get("pilot_valid") and
             (result["action"] != paired.HANDOVER or switch_verified) else "critical")
    _field("Executed action", result["action"])
    _field("Decision source", decision.get("source", "UNKNOWN"))
    if result["action"] == paired.HANDOVER:
        _field("Handover triggered at", "Check {} / 4".format(result["handover_check"])
               if result.get("handover_check") is not None else "UNAVAILABLE")
        _field("Wi-Fi AP1 -> emulated 5G", _color(
            "SWITCH VERIFIED" if switch_verified else "SWITCH NOT VERIFIED",
            "success" if switch_verified else "critical"))
        _field("Selected traffic path", "Emulated cellular/5G-like IP path"
               if switch_verified else "UNCONFIRMED")
        _field("Candidate packets verified", _yes_no(
            timing.get("candidate_verified_ns") is not None))
        _field("Wi-Fi AP1 disconnected", _yes_no(
            timing.get("wifi_disconnected_verified")))
    else:
        _field("Handover", "NO - controller chose STAY")
        _field("Selected traffic path", "Wi-Fi AP1" if
               result.get("checks", {}).get("stay_used_only_wifi") is True
               else "UNCONFIRMED")
    print("\nMEASURED UDP VIDEO-LIKE TRAFFIC AFTER DECISION")
    _field("Packet loss", _fmt(post.get("loss_pct"), 3, "%"))
    _field("P95 process delay", _fmt(post.get("p95_one_way_process_delay_ms"), 2, " ms"))
    _field("Interarrival jitter", _fmt(post.get("rfc3550_interarrival_jitter_ms"), 2, " ms"))
    _field("Maximum packet gap", _fmt(post.get("max_interarrival_gap_ms"), 2, " ms"))
    _field("Complete frames", _fmt(post.get("frames", {}).get("complete_pct"), 2, "%"))
    _field("Application goodput", _fmt(post.get("application_goodput_mbps"), 3, " Mbps"))
    _field("Traffic/mechanical checks", _color(
        "PASSED" if result.get("pilot_valid") else "FAILED",
        "success" if result.get("pilot_valid") else "critical",
    ))
    if expected_action is not None:
        goal_source = ("an AI decision during monitoring" if expected_source == "AI"
                       else "rules only" if expected_source == "RULES"
                       else str(expected_source))
        _field("Recording goal", "{} with {}".format(expected_action, goal_source))
        _field("Final decision", "{} via {}".format(
            result["action"], decision.get("source", "UNKNOWN")))
    action_met = expected_action is None or result["action"] == expected_action
    if expected_source == "AI":
        source_met = any(
            (item.get("controller_decision") or {}).get("source") == "AI" and
            (item.get("controller_decision") or {}).get("action") == expected_action
            for item in history
        )
    elif expected_source == "RULES":
        source_met = decision.get("source") == "RULES" and all(
            (item.get("controller_decision") or {}).get("source") == "RULES"
            for item in history
        )
    else:
        source_met = expected_source is None or decision.get("source") == expected_source
    ai_met = expected_source is None or ai_called_any == (expected_source == "AI")
    goal_met = action_met and source_met and ai_met
    succeeded = bool(result.get("pilot_valid")) and goal_met and (
        result["action"] != paired.HANDOVER or switch_verified
    )
    if result["action"] == paired.HANDOVER and not switch_verified:
        closing = "SCENARIO ENDED: HANDOVER SWITCH NOT VERIFIED"
    elif not result.get("pilot_valid"):
        closing = "SCENARIO ENDED: VERIFICATION FAILED"
    elif not goal_met:
        closing = "SCENARIO ENDED: GOAL NOT OBSERVED"
    else:
        closing = "SCENARIO COMPLETE"
    _section(closing, "success" if succeeded else "critical")
    return succeeded


def _model_summary(report: dict, model_path: Path) -> dict:
    group = report.get("cross_validation", {}).get("group_metrics", {})
    return {
        "accepted": report.get("acceptance_gate", {}).get("accepted") is True,
        "path": str(model_path),
        "training_data_synthetic": report.get("synthetic_data"),
        "label_source": report.get("label_source"),
        "independent_scenario_groups": report.get("independent_scenario_groups"),
        "group_balanced_accuracy_pct": (
            100.0 * group["balanced_accuracy"]
            if _finite_number(group.get("balanced_accuracy")) else None
        ),
        "stay_recall_pct": (
            100.0 * group.get("recall", {}).get("STAY")
            if _finite_number(group.get("recall", {}).get("STAY")) else None
        ),
        "handover_recall_pct": (
            100.0 * group.get("recall", {}).get("HANDOVER")
            if _finite_number(group.get("recall", {}).get("HANDOVER")) else None
        ),
    }


def _load_accepted_model(root: Path):
    try:
        model, model_path, report = final.find_accepted_model(root)
    except Exception as exc:
        raise RuntimeError(
            "No accepted measured-data AI is ready. Do NOT use hosn_ai_model.pkl. "
            "Run: python3 hosn_train_conflict_ai.py --inspect. Original error: "
            + str(exc)
        ) from exc
    summary = _model_summary(report, model_path)
    if not summary["accepted"] or summary["training_data_synthetic"] is not False:
        raise RuntimeError("The available model did not pass the measured-data acceptance contract.")
    return model, model_path, report, summary


def _print_model_check(summary: dict) -> None:
    print(_line())
    print("ACCEPTED MEASURED-DATA AI: READY")
    print(_line())
    print("Model path                 :", summary["path"])
    print("Synthetic training data    :", summary["training_data_synthetic"])
    print("Label source               :", summary["label_source"])
    print("Independent scenario groups:", summary["independent_scenario_groups"])
    print("Group balanced accuracy    :", _fmt(summary["group_balanced_accuracy_pct"], 2, "%"))
    print("STAY group recall          :", _fmt(summary["stay_recall_pct"], 2, "%"))
    print("HANDOVER group recall      :", _fmt(summary["handover_recall_pct"], 2, "%"))
    print(_line())


def _required_files(root: Path) -> tuple[str, ...]:
    return (
        "hosn_final_comparison.py",
        "hosn_paired_outcome_pilot.py",
        "hosn_strategy_router.py",
        "hosn_heterogeneous_controller.py",
        "hosn_wifi_5g_compare.py",
        "hosn_wifi_5g_pilot.py",
        "hosn_switch.py",
    )


def _preflight(root: Path) -> None:
    missing_files = [name for name in _required_files(root) if not (root / name).is_file()]
    if missing_files:
        raise RuntimeError("Missing project file(s): " + ", ".join(missing_files))
    missing_tools = [name for name in ("ip", "iw", "ping", "tc") if not shutil.which(name)]
    if missing_tools:
        raise RuntimeError("Missing Ubuntu command(s): " + ", ".join(missing_tools))
    if sys.platform != "linux" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise RuntimeError("Run a scenario inside Ubuntu with sudo.")


def _setup_network(run_folder: Path):
    import hosn_switch as helper

    network, station, server, wifi_core, cell_gateway, ap = base.create_network()
    run = helper.NodeCommands(run_folder / "commands.jsonl")
    wifi_if = station.wintfs[0].name
    base.verified_interface(station, cell_gateway, "sta1-5g0")
    base.verified_interface(server, wifi_core, "h1-wifi0")
    base.verified_interface(server, cell_gateway, "h1-5g0")
    base.configure_ip(station, wifi_if, base.WIFI_UE_IP + "/24", run)
    base.configure_ip(station, "sta1-5g0", base.CELL_UE_IP + "/24", run)
    base.configure_ip(server, "h1-wifi0", base.WIFI_SERVER_IP + "/24", run)
    base.configure_ip(server, "h1-5g0", base.CELL_SERVER_IP + "/24", run)
    seed_capability = compare.detect_netem_seed_support(station, "sta1-5g0", run)
    return network, station, server, wifi_core, cell_gateway, ap, run, helper, seed_capability


def _run_one_scenario(root: Path, number: str, model, model_info: dict, parent: Path,
                      countdown_seconds: int, hold_seconds: float) -> dict:
    info = SCENARIO_SPECS[number]
    scenario = _recording_scenario(number)
    run_folder = parent / ("scenario_" + number)
    run_folder.mkdir(parents=True, exist_ok=False)
    old_cwd = Path.cwd()
    network = None
    try:
        os.chdir(run_folder)
        (network, station, server, _wifi_core, cell_gateway, ap,
         run, helper, seed_capability) = _setup_network(run_folder)

        _clear_screen()
        _print_scenario_intro(number, info, model_info)
        _countdown(countdown_seconds)

        callback_state: dict[str, Any] = {}
        previous_observation: dict | None = None

        def observation_callback(snapshot: dict, check: int, total: int,
                                 final_check: bool) -> None:
            nonlocal previous_observation
            _print_monitoring_check(snapshot, check, total, final_check,
                                    previous_observation)
            previous_observation = snapshot

        def decision_callback(snapshot: dict) -> dict:
            check = len(callback_state.setdefault("history", [])) + 1
            decision = final.decide(router.HOSN_FULL_AI, snapshot, model)
            callback_state["snapshot"] = snapshot
            callback_state["decision"] = decision
            callback_state["history"].append(decision)
            _print_snapshot(snapshot, decision, check, 4)
            return decision

        result = paired.run_one(
            paired.STAY,
            scenario,
            run_folder,
            station,
            server,
            cell_gateway,
            ap,
            bool(seed_capability["supported"]),
            run,
            helper,
            decision_callback=decision_callback,
            observation_callback=observation_callback,
        )
        if "snapshot" not in callback_state:
            raise RuntimeError("The decision callback was never reached.")
        check_count = len(result.get("recording_checks", ()))
        if not 1 <= check_count <= 4 or check_count != len(callback_state["history"]):
            raise RuntimeError("Every recording check must have one real controller decision.")
        if check_count < 4 and result.get("handover_check") != check_count:
            raise RuntimeError("Wi-Fi monitoring ended without an executed early handover.")
        goal_observed = _print_result(
            result, scenario["expected_action_for_evaluation_only"], info["expected_source"]
        )
        result["recording_demo"] = {
            "revision": REVISION,
            "scenario_number": number,
            "scenario_title": info["title"],
            "model_expected_action_was_not_supplied": True,
            "fixed_three_check_gate_used": False,
            "real_monitoring_checks": len(result["recording_checks"]),
            "additional_probe_at_initial_position": True,
            "recording_intermediate_position_adjusted": scenario.get(
                "recording_intermediate_position_adjusted", False),
            "handover_check": result.get("handover_check"),
            "recording_goal_observed": goal_observed,
            "recording_probes_share_media_links": True,
        }
        compare.write_json(run_folder / "recording_summary.json", result)
        if hold_seconds > 0:
            print("\nResult remains on screen for {:.0f} seconds...".format(hold_seconds), flush=True)
            time.sleep(hold_seconds)
        return result
    finally:
        if network is not None:
            print("\nStopping this scenario's simulated network...", flush=True)
            network.stop()
        os.chdir(old_cwd)
        try:
            if "helper" in locals():
                helper.restore_result_owner(run_folder)
        except Exception:
            pass


def _plan() -> None:
    print(_line())
    print("ATP HOSN THREE-SCENARIO RECORDING PLAN")
    print(_line())
    for number, info in SCENARIO_SPECS.items():
        source = _scenario_by_id(info["source_id"])
        print("\n{}. {}".format(number, info["title"]))
        print("   " + info["story"])
        print("   Rules/AI role: " + info["point"])
        print("   Movement: {} | speed: {} | decision zone: {}".format(
            source["direction"], source["speed_class"], source["decision_zone"]
        ))
    print("\nThe HOSN controller receives no evaluation answer as an input.")
    print("Up to four checks use real probes and an actual HOSN decision at each check.")
    print("AI runs immediately on a real rules conflict; handover executes when authorized.")
    print("An early handover ends Wi-Fi monitoring before Check 4.")
    print("The old synthetic hosn_ai_model.pkl is never loaded.")
    print(_line())


def run_selected(root: Path, selection: str, countdown_seconds: int,
                 hold_seconds: float) -> int:
    _preflight(root)
    model, model_path, report, model_info = _load_accepted_model(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    parent = root / "results" / ("atp_recording_demo_" + stamp)
    parent.mkdir(parents=True, exist_ok=False)
    manifest = {
        "revision": REVISION,
        "created_utc": stamp,
        "status": "starting",
        "selection": selection,
        "fixed_three_check_gate_used": False,
        "model": model_info,
        "python": platform.python_version(),
        "kernel": platform.release(),
        "scenarios": [],
    }
    compare.write_json(parent / "manifest.json", manifest)
    numbers = tuple(SCENARIO_SPECS) if selection == "all" else (selection,)
    try:
        for number in numbers:
            result = _run_one_scenario(
                root, number, model, model_info, parent,
                countdown_seconds, hold_seconds
            )
            manifest["scenarios"].append({
                "number": number,
                "title": SCENARIO_SPECS[number]["title"],
                "action": result["action"],
                "source": result["controller_decision"].get("source"),
                "ai_called": result["controller_decision"].get("ai_called"),
                "ai_called_any": any(bool(item.get("controller_decision", {}).get("ai_called"))
                                     for item in result["recording_checks"]),
                "completed_checks": len(result["recording_checks"]),
                "handover_check": result.get("handover_check"),
                "pilot_valid": result.get("pilot_valid"),
                "handover_switch_verified": _handover_switch_verified(result),
                "stay_wifi_verified": result.get("checks", {}).get("stay_used_only_wifi") is True,
                "recording_goal_observed": result["recording_demo"]["recording_goal_observed"],
            })
        all_goals_observed = all(item["recording_goal_observed"] for item in manifest["scenarios"])
        manifest["status"] = "complete" if all_goals_observed else "complete_with_unmet_goals"
        compare.write_json(parent / "manifest.json", manifest)
        print(_line())
        print("ATP HOSN RECORDING DEMO COMPLETE" if all_goals_observed else
              "ATP HOSN RECORDING FINISHED: SOME GOALS NOT OBSERVED")
        print(_line())
        for item in manifest["scenarios"]:
            _section("SCENARIO {} | FINAL RESULT".format(item["number"]),
                     "success" if item["recording_goal_observed"] else "critical")
            _field("Checks completed", "{} / 4".format(item["completed_checks"]))
            if item["action"] == paired.HANDOVER:
                _field("Handover executed at", "CHECK {}".format(item["handover_check"])
                       if item["handover_check"] is not None else "UNCONFIRMED")
                _field("Wi-Fi AP1 -> emulated 5G", "SWITCH VERIFIED" if
                       item["handover_switch_verified"] else "SWITCH NOT VERIFIED")
                _field("Selected traffic path", "Emulated cellular/5G-like IP path" if
                       item["handover_switch_verified"] else "UNCONFIRMED")
            else:
                _field("Handover", "NO - controller chose STAY")
                _field("Selected traffic path", "Wi-Fi AP1" if
                       item["stay_wifi_verified"] else "UNCONFIRMED")
            _field("Controller decision", item["action"])
            _field("Final decision made by", "HOSN RULES" if item["source"] == "RULES"
                   else "TRAINED HOSN AI" if item["source"] == "AI" else item["source"])
            _field("AI used in this scenario", _yes_no(item["ai_called_any"]))
            _field("Traffic/mechanical checks", "PASSED" if item["pilot_valid"] else "FAILED")
            _field("Scenario goal observed", _yes_no(item["recording_goal_observed"]))
        print("\nSaved evidence:", parent)
        print(_line())
        return 0 if all_goals_observed else 1
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        compare.write_json(parent / "manifest.json", manifest)
        print("\nRecording demo interrupted. Partial evidence was preserved.")
        return 130
    except Exception as exc:
        manifest.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        compare.write_json(parent / "manifest.json", manifest)
        (parent / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("\nDEMO STOPPED:", exc, file=sys.stderr)
        print("Network switch completion was not verified in this run.", file=sys.stderr)
        print("Saved diagnostics:", parent, file=sys.stderr)
        return 1
    finally:
        try:
            import hosn_switch as helper
            helper.restore_result_owner(parent)
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recording-focused three-scenario HOSN ATP demonstration"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="Show the three demo scenarios")
    mode.add_argument("--check-model", action="store_true", help="Verify the accepted measured-data AI")
    mode.add_argument("--scenario", choices=("1", "2", "3", "all"), help="Run one or all scenarios")
    parser.add_argument("--countdown", type=int, default=0, help="Seconds before traffic starts")
    parser.add_argument("--hold-seconds", type=float, default=6.0,
                        help="Keep each result visible before cleanup")
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto",
                        help="ANSI terminal colors")
    args = parser.parse_args()
    global COLOR_MODE
    COLOR_MODE = args.color
    root = Path(__file__).resolve().parent
    if args.plan:
        _plan()
        return 0
    if args.check_model:
        try:
            _model, _path, _report, summary = _load_accepted_model(root)
            _print_model_check(summary)
            return 0
        except Exception as exc:
            parser.exit(2, "Model check stopped: " + str(exc) + "\n")
    if args.countdown < 0:
        parser.error("--countdown cannot be negative")
    if not math.isfinite(args.hold_seconds) or args.hold_seconds < 0:
        parser.error("--hold-seconds must be finite and nonnegative")
    return run_selected(root, args.scenario, args.countdown, args.hold_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
