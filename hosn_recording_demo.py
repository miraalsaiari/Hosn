#!/usr/bin/env python3
"""Recording-focused ATP demo for the final HOSN Wi-Fi -> emulated-5G system.

This wrapper reuses the project's accepted measured-data AI, rules-first
controller, media-like workload, and make-before-break executor.  It runs only
three presentation scenarios, one at a time, instead of the full 48-replay
research evaluation.

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


REVISION = "atp-recording-three-scenario-v1"

SCENARIO_SPECS = {
    "1": {
        "source_id": "eval05_edge_out_slow_clear_handover",
        "title": "CLEAR CASE: RULES AUTHORIZE HANDOVER",
        "story": (
            "The user moves away from Wi-Fi. Wi-Fi signal and service become poor, "
            "while the emulated cellular/5G-like path is clearly better."
        ),
        "point": "This proves clear cases are handled directly by rules; AI is not called.",
    },
    "2": {
        "source_id": "eval03_near_out_slow_conflict_handover",
        "title": "CONFLICT: STRONG WI-FI, BUT POOR WI-FI SERVICE",
        "story": (
            "Wi-Fi is still physically near/strong, but its delay and loss are poor. "
            "Signal alone favors staying; service quality favors the candidate path."
        ),
        "point": "This is an ambiguous case, so the accepted AI must make the final choice.",
    },
    "3": {
        "source_id": "eval01_edge_out_fast_conflict_stay",
        "title": "CONFLICT: WEAK WI-FI, BUT CANDIDATE SERVICE IS WORSE",
        "story": (
            "Wi-Fi signal is weak, but its measured service remains good. The candidate "
            "path has a better signal margin but poorer delay/loss."
        ),
        "point": "This shows why blindly choosing the stronger signal can be wrong.",
    },
}


def _scenario_by_id(identifier: str) -> dict:
    for item in final.EVALUATION_SCENARIOS:
        if item["id"] == identifier:
            return item
    raise RuntimeError("Required evaluation scenario is missing: " + identifier)


def _line(char: str = "=", width: int = 78) -> str:
    return char * width


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


def _print_scenario_intro(number: str, info: dict) -> None:
    print(_line())
    print("ATP HOSN RECORDING DEMO — SCENARIO {}".format(number))
    print(info["title"])
    print(_line())
    print("\nSCENARIO")
    print(info["story"])
    print("\nWHY IT MATTERS")
    print(info["point"])
    print("\nArchitecture: MEASURE -> RULES -> AI only if CONFLICT -> EXECUTE -> VERIFY")
    print("No fixed three-check gate is used in this recording demo.")
    print(_line("-"))


def _print_snapshot(snapshot: dict, decision: dict) -> None:
    rule = snapshot.get("rule_evaluation", {})
    print("\n" + _line("="))
    print("LIVE DECISION SNAPSHOT")
    print(_line("="))
    print(
        "Current Wi-Fi : signal {} | RTT {} | loss {} | trend {}".format(
            _fmt(snapshot.get("wifi_rssi_model_dbm"), 2, " dBm"),
            _fmt(snapshot.get("wifi_rtt_ms"), 2, " ms"),
            _fmt(snapshot.get("wifi_loss_pct"), 2, "%"),
            _fmt(snapshot.get("wifi_trend_db_per_s"), 3, " dB/s"),
        )
    )
    print(
        "Candidate path: signal {} | RTT {} | loss {} | trend {}".format(
            _fmt(snapshot.get("cell_rsrp_configured_dbm"), 2, " dBm"),
            _fmt(snapshot.get("cell_rtt_ms"), 2, " ms"),
            _fmt(snapshot.get("cell_loss_pct"), 2, "%"),
            _fmt(snapshot.get("cell_trend_db_per_s"), 3, " dB/s"),
        )
    )
    print("\nRULE LAYER")
    print("  Result : {} | {}".format(rule.get("decision", "UNKNOWN"), rule.get("status", "UNKNOWN")))
    print("  Reason : {}".format(rule.get("reason", "No reason recorded.")))
    print("\nFINAL CONTROLLER")
    print("  AI called      : {}".format(_yes_no(decision.get("ai_called"))))
    print("  Decision source: {}".format(decision.get("source", "UNKNOWN")))
    print("  FINAL ACTION   : {}".format(decision.get("action", "UNKNOWN")))
    print("  Reason         : {}".format(decision.get("reason", "No reason recorded.")))
    print(_line("="), flush=True)


def _print_result(result: dict) -> None:
    post = result["post_decision"]
    decision = result["controller_decision"]
    timing = result.get("timing", {})
    print("\n" + _line("="))
    print("ACTION EXECUTED AND OUTCOME MEASURED")
    print(_line("="))
    print("Executed action      : {}".format(result["action"]))
    print("Decision source      : {}".format(decision.get("source", "UNKNOWN")))
    print("AI used              : {}".format(_yes_no(decision.get("ai_called"))))
    if result["action"] == paired.HANDOVER:
        print("Candidate verified   : {}".format(
            _yes_no(timing.get("candidate_verified_ns") is not None)
        ))
        print("Wi-Fi break verified : {}".format(
            _yes_no(timing.get("wifi_disconnected_verified"))
        ))
    else:
        print("Connection choice    : stayed on current Wi-Fi path")
    print("\nPOST-DECISION VIDEO-LIKE TRAFFIC")
    print("  Packet loss         : {}".format(_fmt(post.get("loss_pct"), 3, "%")))
    print("  P95 process delay   : {}".format(
        _fmt(post.get("p95_one_way_process_delay_ms"), 2, " ms")
    ))
    print("  Jitter              : {}".format(
        _fmt(post.get("rfc3550_interarrival_jitter_ms"), 2, " ms")
    ))
    print("  Maximum packet gap  : {}".format(
        _fmt(post.get("max_interarrival_gap_ms"), 2, " ms")
    ))
    print("  Complete frames     : {}".format(
        _fmt(post.get("frames", {}).get("complete_pct"), 2, "%")
    ))
    print("  Application goodput : {}".format(
        _fmt(post.get("application_goodput_mbps"), 3, " Mbps")
    ))
    print("\nMechanical checks passed: {}".format(_yes_no(result.get("pilot_valid"))))
    print(_line("="), flush=True)


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


def _run_one_scenario(root: Path, number: str, model, parent: Path,
                      countdown_seconds: int, hold_seconds: float) -> dict:
    info = SCENARIO_SPECS[number]
    scenario = _scenario_by_id(info["source_id"])
    run_folder = parent / ("scenario_" + number)
    run_folder.mkdir(parents=True, exist_ok=False)
    old_cwd = Path.cwd()
    network = None
    try:
        os.chdir(run_folder)
        (network, station, server, _wifi_core, cell_gateway, ap,
         run, helper, seed_capability) = _setup_network(run_folder)

        _clear_screen()
        _print_scenario_intro(number, info)
        _countdown(countdown_seconds)

        callback_state: dict[str, Any] = {}

        def decision_callback(snapshot: dict) -> dict:
            decision = final.decide(router.HOSN_FULL_AI, snapshot, model)
            callback_state["snapshot"] = snapshot
            callback_state["decision"] = decision
            _print_snapshot(snapshot, decision)
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
        )
        if "snapshot" not in callback_state:
            raise RuntimeError("The decision callback was never reached.")
        _print_result(result)
        result["recording_demo"] = {
            "revision": REVISION,
            "scenario_number": number,
            "scenario_title": info["title"],
            "model_expected_action_was_not_supplied": True,
            "fixed_three_check_gate_used": False,
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
    print("\nThe final HOSN controller receives no evaluation answer as an input.")
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
                root, number, model, parent, countdown_seconds, hold_seconds
            )
            manifest["scenarios"].append({
                "number": number,
                "title": SCENARIO_SPECS[number]["title"],
                "action": result["action"],
                "source": result["controller_decision"].get("source"),
                "ai_called": result["controller_decision"].get("ai_called"),
                "pilot_valid": result.get("pilot_valid"),
            })
        manifest["status"] = "complete"
        compare.write_json(parent / "manifest.json", manifest)
        _clear_screen()
        print(_line())
        print("ATP HOSN RECORDING DEMO COMPLETE")
        print(_line())
        for item in manifest["scenarios"]:
            print(
                "Scenario {number}: action={action} | source={source} | AI={ai_called} | checks={pilot_valid}".format(
                    **item
                )
            )
        print("\nSaved evidence:", parent)
        print(_line())
        return 0
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
    parser.add_argument("--countdown", type=int, default=3, help="Seconds before traffic starts")
    parser.add_argument("--hold-seconds", type=float, default=6.0,
                        help="Keep each result visible before cleanup")
    args = parser.parse_args()
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
