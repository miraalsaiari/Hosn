#!/usr/bin/env python3
"""Small measured STAY-versus-HANDOVER outcome pilot for HOSN.

This program is deliberately pre-dataset.  It runs one controlled scenario four
times in an ABBA action order: STAY, HANDOVER, HANDOVER, STAY.  Each replay uses
the same movement, traffic, access profiles, probe procedure, and (where the
host supports it) netem seeds.  It records all observed packet loss, delay,
jitter, gaps, frame completeness, throughput, and goodput.

The program does not train AI and does not emit training rows.  It reports a
review recommendation only when one action tolerance-dominates the other; an
unclear result remains INCONCLUSIVE instead of receiving a fabricated label.

Commands:
  python3 hosn_paired_outcome_pilot.py --self-test
  python3 hosn_paired_outcome_pilot.py --plan
  sudo python3 hosn_paired_outcome_pilot.py --run
  sudo python3 hosn_paired_outcome_pilot.py --matrix-pilot
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Optional

import hosn_wifi_5g_pilot as base
import hosn_wifi_5g_compare as compare
import hosn_heterogeneous_controller as heterogeneous


REVISION = "paired-outcome-pilot-v2-matrix"
STAY = "STAY"
HANDOVER = "HANDOVER"
RUN_ORDER = (
    ("stay_1", STAY),
    ("handover_1", HANDOVER),
    ("handover_2", HANDOVER),
    ("stay_2", STAY),
)
TRAFFIC_DURATION_S = 18.0
DECISION_STAGE_INDEX = 2

EXTRA_ACCESS_PROFILES = {
    # Strong Wi-Fi signal but deliberately congested service: useful for a
    # conflict where signal favors STAY while measured QoS favors HANDOVER.
    "wifi_congested_near": {
        "delay_ms": 55.0, "jitter_ms": 10.0,
        "loss_pct": 5.0, "rate_mbit": 6.0,
    },
    # Weak Wi-Fi signal with good service: useful for the opposite conflict.
    "wifi_edge_good_qos": {
        "delay_ms": 5.0, "jitter_ms": 1.0,
        "loss_pct": 0.0, "rate_mbit": 20.0,
    },
    "emulated_5g_congested": {
        "delay_ms": 60.0, "jitter_ms": 12.0,
        "loss_pct": 4.0, "rate_mbit": 7.0,
    },
}
ALL_ACCESS_PROFILES = {**compare.ACCESS_PROFILES, **EXTRA_ACCESS_PROFILES}

DEFAULT_SCENARIO = {
    "id": "outbound_fast_edge_wifi_cell_healthy",
    "direction": "outbound",
    "speed_class": "fast",
    "movement": compare.MOVEMENT,
    "decision_stage_index": 2,
    "cell_profile": "emulated_5g_healthy",
    "design_role": "clear_handover_control",
    "duration_s": 18.0,
}

MATRIX_SCENARIOS = (
    DEFAULT_SCENARIO,
    {
        "id": "outbound_slow_weak_wifi_good_qos_cell_congested",
        "direction": "outbound",
        "speed_class": "slow",
        "movement": [
            {"at_s": 0.0, "x_m": 20.0, "wifi_profile": "wifi_near"},
            {"at_s": 6.0, "x_m": 40.0, "wifi_profile": "wifi_moving"},
            {"at_s": 12.0, "x_m": 65.0, "wifi_profile": "wifi_edge_good_qos"},
        ],
        "decision_stage_index": 2,
        "cell_profile": "emulated_5g_congested",
        "design_role": "conflict_signal_favors_handover_qos_favors_stay",
        "duration_s": 22.0,
    },
    {
        "id": "inbound_fast_strong_wifi_congested_cell_healthy",
        "direction": "inbound",
        "speed_class": "fast",
        "movement": [
            {"at_s": 0.0, "x_m": 65.0, "wifi_profile": "wifi_edge"},
            {"at_s": 4.0, "x_m": 40.0, "wifi_profile": "wifi_moving"},
            {"at_s": 8.0, "x_m": 20.0, "wifi_profile": "wifi_congested_near"},
        ],
        "decision_stage_index": 2,
        "cell_profile": "emulated_5g_healthy",
        "design_role": "conflict_signal_favors_stay_qos_favors_handover",
        "duration_s": 18.0,
    },
    {
        "id": "inbound_slow_near_wifi_cell_congested",
        "direction": "inbound",
        "speed_class": "slow",
        "movement": [
            {"at_s": 0.0, "x_m": 65.0, "wifi_profile": "wifi_edge"},
            {"at_s": 6.0, "x_m": 40.0, "wifi_profile": "wifi_moving"},
            {"at_s": 12.0, "x_m": 20.0, "wifi_profile": "wifi_near"},
        ],
        "decision_stage_index": 2,
        "cell_profile": "emulated_5g_congested",
        "design_role": "clear_stay_control",
        "duration_s": 22.0,
    },
)

# These tolerances decide only whether a pilot recommendation is sufficiently
# clear for human review.  They are not labels and are not model thresholds.
DOMINANCE_TOLERANCES = {
    "post_loss_pct": 0.10,
    "post_p95_delay_ms": 5.0,
    "post_max_gap_ms": 10.0,
    "post_complete_frames_pct": 1.0,
    "post_goodput_mbps": 0.01,
}

PLAN = {
    "revision": REVISION,
    "purpose": "small controlled paired-outcome validation before dataset design",
    "dataset_collection": False,
    "training_rows_created": 0,
    "training_labels_assigned": 0,
    "actions": [STAY, HANDOVER],
    "run_order": [{"id": run_id, "action": action} for run_id, action in RUN_ORDER],
    "scenario": {
        "application": "measured UDP video-frame-like workload; not an encoded-video MOS/VMAF claim",
        "movement": DEFAULT_SCENARIO["movement"],
        "wifi_profiles": ALL_ACCESS_PROFILES,
        "decision_stage": DEFAULT_SCENARIO["movement"][DEFAULT_SCENARIO["decision_stage_index"]],
        "cellular_access": "emulated cellular/5G-like IP path; not a 3GPP radio/core",
    },
    "pairing_controls": {
        "same_configured_movement": True,
        "same_configured_access_profiles": True,
        "same_media_workload": True,
        "same_probe_procedure": True,
        "same_decision_stage": True,
        "same_make_before_break_executor_for_handover": True,
        "netem_seed": "same per-profile seeds when supported; otherwise unseeded ABBA repeats",
    },
    "reported_metrics": [
        "raw/unique/duplicate/lost packets",
        "post-decision actual loss",
        "one-way process delay p95",
        "RFC3550 interarrival jitter",
        "post-decision maximum interruption gap",
        "network throughput and application goodput",
        "complete/damaged/missing frames",
        "gaps over 150 ms",
    ],
    "review_rule": {
        "method": "tolerance dominance on mean post-decision outcomes",
        "tolerances": DOMINANCE_TOLERANCES,
        "inconclusive_behavior": "retain INCONCLUSIVE; assign no training label",
    },
    "matrix_pilot": {
        "dataset_collection": False,
        "training_rows_created": 0,
        "training_labels_assigned": 0,
        "scenarios": list(MATRIX_SCENARIOS),
        "replays_per_scenario": len(RUN_ORDER),
        "total_replays": len(MATRIX_SCENARIOS) * len(RUN_ORDER),
    },
}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _percentile(values: list[float], q: float) -> Optional[float]:
    return compare.percentile(values, q)


def configure_access_profile(node, interface: str, profile_name: str,
                             seed: int, seed_supported: bool, run) -> dict:
    """Apply one declared matrix profile and retain tc verification evidence."""
    if profile_name not in ALL_ACCESS_PROFILES:
        raise ValueError("Unknown access profile: " + profile_name)
    profile = ALL_ACCESS_PROFILES[profile_name]
    applied_seed = seed if seed_supported else None
    argv = compare.profile_tc_args(interface, profile, applied_seed)
    text, code = run(node, argv)
    if code:
        raise RuntimeError("tc/netem profile failed: " + text.strip())
    shown, code = run(node, ["tc", "qdisc", "show", "dev", interface])
    if code or "netem" not in shown:
        raise RuntimeError("Could not verify netem on " + interface)
    return {
        "name": profile_name,
        "configured_not_measured": profile,
        "requested_seed": seed,
        "applied_seed": applied_seed,
        "randomization": "seeded" if seed_supported else "unseeded_by_platform_limitation",
        "tc_show": shown.strip(),
    }


def evaluate_snapshot_rules(snapshot: dict) -> dict:
    """Evaluate the real controller rules on a measured decision snapshot."""
    policy = compare.PLAN["hosn_policy"]
    current = heterogeneous.AccessObservation(
        access_name="wifi", technology="Wi-Fi RSSI",
        signal_dbm=snapshot["wifi_rssi_model_dbm"],
        service_floor_dbm=policy["wifi_service_floor_dbm"],
        latency_ms=snapshot["wifi_rtt_ms"], loss_pct=snapshot["wifi_loss_pct"],
        trend_db_per_s=snapshot["wifi_trend_db_per_s"],
    )
    candidate = heterogeneous.AccessObservation(
        access_name="emulated_5g",
        technology="emulated cellular/5G-like IP path",
        signal_dbm=snapshot["cell_rsrp_configured_dbm"],
        service_floor_dbm=policy["emulated_cellular_service_floor_rsrp_dbm"],
        latency_ms=snapshot["cell_rtt_ms"], loss_pct=snapshot["cell_loss_pct"],
        trend_db_per_s=snapshot["cell_trend_db_per_s"],
    )
    features = heterogeneous.build_features(current, candidate)
    rule = heterogeneous.evaluate_rules(features)
    return {
        "decision": rule.decision, "status": rule.status,
        "reason": rule.reason, "missing_fields": list(rule.missing_fields),
        "features": features,
    }


def summarize_outcome(sender: dict, receiver: dict, events: list[dict],
                      action: str, start_ns: int, decision_ns: int,
                      timing: dict, duration_s: float = TRAFFIC_DURATION_S) -> dict:
    """Summarize raw observations without hiding duplicates or loss."""
    if action not in (STAY, HANDOVER):
        raise ValueError("Unknown action")
    if sender.get("status") != "complete" or receiver.get("status") != "complete":
        raise ValueError("sender/receiver incomplete")

    ordered = sorted(events, key=lambda event: event["received_monotonic_ns"])
    first_by_sequence: dict[int, dict] = {}
    for event in ordered:
        first_by_sequence.setdefault(event["sequence"], event)
    unique = sorted(first_by_sequence.values(), key=lambda event: event["received_monotonic_ns"])
    attempted = int(sender["scheduled_sequences"])
    decision_sequence = min(
        attempted,
        max(0, int(math.ceil((decision_ns - start_ns) / 1e9 * compare.PACKETS_PER_SECOND))),
    )
    post = [event for event in unique if event["sequence"] >= decision_sequence]
    post_expected = max(0, attempted - decision_sequence)
    post_received = len({event["sequence"] for event in post})
    post_lost = max(0, post_expected - post_received)
    post_delays = [
        (event["received_monotonic_ns"] - event["sent_monotonic_ns"]) / 1e6
        for event in post
    ]
    post_gaps = [
        (right["received_monotonic_ns"] - left["received_monotonic_ns"]) / 1e6
        for left, right in zip(post, post[1:])
    ]
    paths = Counter(event["declared_path"] for event in ordered)

    # Exclude the frame that straddles the decision boundary so neither action
    # is penalized by packets that were intentionally sent before the decision.
    frame_first = math.ceil(decision_sequence / compare.PACKETS_PER_FRAME)
    final_frame = math.ceil(attempted / compare.PACKETS_PER_FRAME)
    frame_parts = {frame: set() for frame in range(frame_first, final_frame)}
    for event in post:
        if event["frame"] in frame_parts:
            frame_parts[event["frame"]].add(event["part"])
    complete_flags = [
        len(frame_parts[frame]) == compare.PACKETS_PER_FRAME
        for frame in range(frame_first, final_frame)
    ]
    damaged = sum(
        0 < len(frame_parts[frame]) < compare.PACKETS_PER_FRAME
        for frame in range(frame_first, final_frame)
    )
    missing = sum(len(frame_parts[frame]) == 0 for frame in range(frame_first, final_frame))
    longest = run = 0
    for complete in complete_flags:
        run = 0 if complete else run + 1
        longest = max(longest, run)

    post_duration = post_expected / compare.PACKETS_PER_SECOND if post_expected else 0.0
    action_checks = {
        "receiver_parse_clean": receiver.get("parse_errors") == 0,
        "sender_completed": sender.get("status") == "complete",
        "receiver_completed": receiver.get("status") == "complete",
        "expected_action": action,
        "stay_used_only_wifi": action != STAY or (paths["wifi"] > 0 and paths["5g"] == 0),
        "handover_used_both_paths": action != HANDOVER or (paths["wifi"] > 0 and paths["5g"] > 0),
        "candidate_verified_before_break": (
            action != HANDOVER
            or timing["candidate_verified_ns"] <= timing["wifi_break_ns"]
        ),
        "minimum_overlap_met": (
            action != HANDOVER
            or (timing["wifi_break_ns"] - timing["duplicate_start_ns"]) / 1e9
            >= compare.OVERLAP_MIN_S
        ),
        "wifi_disconnect_verified": action != HANDOVER or timing["wifi_disconnected_verified"],
    }
    boolean_checks = [value for value in action_checks.values() if isinstance(value, bool)]
    return {
        "action": action,
        "attempted_packets": attempted,
        "raw_datagrams_received": len(ordered),
        "unique_packets_received": len(unique),
        "duplicate_datagrams_received": len(ordered) - len(unique),
        "actual_lost_sequences": max(0, attempted - len(unique)),
        "actual_loss_pct": 100.0 * max(0, attempted - len(unique)) / attempted if attempted else 100.0,
        "raw_received_by_path": dict(paths),
        "decision_sequence": decision_sequence,
        "post_decision": {
            "expected_packets": post_expected,
            "received_packets": post_received,
            "lost_packets": post_lost,
            "loss_pct": 100.0 * post_lost / post_expected if post_expected else 100.0,
            "p95_one_way_process_delay_ms": _percentile(post_delays, 0.95),
            "rfc3550_interarrival_jitter_ms": compare.rfc3550_jitter_ms(post),
            "max_interarrival_gap_ms": max(post_gaps) if post_gaps else None,
            "gaps_over_150ms": sum(gap > 150.0 for gap in post_gaps),
            "application_goodput_mbps": (
                post_received * compare.PAYLOAD_BYTES * 8 / post_duration / 1e6
                if post_duration else 0.0
            ),
            "frames": {
                "expected": len(complete_flags),
                "complete": sum(complete_flags),
                "damaged": damaged,
                "missing": missing,
                "complete_pct": 100.0 * sum(complete_flags) / len(complete_flags)
                if complete_flags else 0.0,
                "longest_incomplete_run_frames": longest,
                "longest_incomplete_run_ms": 1000.0 * longest / compare.FRAME_RATE,
            },
        },
        "network_throughput_mbps": len(ordered) * compare.PAYLOAD_BYTES * 8 / duration_s / 1e6,
        "checks": action_checks,
        "pilot_valid": all(boolean_checks),
    }


def aggregate_by_action(results: dict[str, dict]) -> dict[str, dict]:
    grouped = {STAY: [], HANDOVER: []}
    for result in results.values():
        grouped[result["action"]].append(result)
    output = {}
    for action, items in grouped.items():
        post = [item["post_decision"] for item in items]
        output[action] = {
            "replays": len(items),
            "mean_post_loss_pct": statistics.fmean(item["loss_pct"] for item in post),
            "mean_post_p95_delay_ms": statistics.fmean(item["p95_one_way_process_delay_ms"] for item in post),
            "mean_post_jitter_ms": statistics.fmean(item["rfc3550_interarrival_jitter_ms"] for item in post),
            "mean_post_max_gap_ms": statistics.fmean(item["max_interarrival_gap_ms"] for item in post),
            "mean_post_goodput_mbps": statistics.fmean(item["application_goodput_mbps"] for item in post),
            "mean_post_complete_frames_pct": statistics.fmean(item["frames"]["complete_pct"] for item in post),
            "total_post_gaps_over_150ms": sum(item["gaps_over_150ms"] for item in post),
            "all_mechanical_checks_pass": all(item["pilot_valid"] for item in items),
        }
    return output


def review_recommendation(aggregate: dict[str, dict]) -> dict:
    stay = aggregate[STAY]
    handover = aggregate[HANDOVER]
    specs = (
        ("mean_post_loss_pct", "post_loss_pct", False),
        ("mean_post_p95_delay_ms", "post_p95_delay_ms", False),
        ("mean_post_max_gap_ms", "post_max_gap_ms", False),
        ("mean_post_complete_frames_pct", "post_complete_frames_pct", True),
        ("mean_post_goodput_mbps", "post_goodput_mbps", True),
    )

    def dominates(left: dict, right: dict) -> tuple[bool, list[str]]:
        no_worse = True
        materially_better = []
        for aggregate_key, tolerance_key, higher_is_better in specs:
            tolerance = DOMINANCE_TOLERANCES[tolerance_key]
            left_value, right_value = left[aggregate_key], right[aggregate_key]
            if higher_is_better:
                if left_value < right_value - tolerance:
                    no_worse = False
                if left_value > right_value + tolerance:
                    materially_better.append(aggregate_key)
            else:
                if left_value > right_value + tolerance:
                    no_worse = False
                if left_value < right_value - tolerance:
                    materially_better.append(aggregate_key)
        return no_worse and bool(materially_better), materially_better

    handover_wins, handover_better = dominates(handover, stay)
    stay_wins, stay_better = dominates(stay, handover)
    if handover_wins and not stay_wins:
        recommendation, evidence = HANDOVER, handover_better
    elif stay_wins and not handover_wins:
        recommendation, evidence = STAY, stay_better
    else:
        recommendation, evidence = "INCONCLUSIVE", []
    return {
        "recommendation_for_human_review": recommendation,
        "materially_better_metrics": evidence,
        "training_label_assigned": False,
        "reason": (
            "One action tolerance-dominates the other on mean measured post-decision outcomes."
            if recommendation != "INCONCLUSIVE"
            else "Neither action tolerance-dominates; retain the pair as inconclusive."
        ),
        "tolerances": DOMINANCE_TOLERANCES,
    }


def pairing_diagnostics(results: dict[str, dict]) -> dict:
    """Report how closely independent replay snapshots match before action."""
    fields = (
        "wifi_rssi_model_dbm", "cell_rsrp_configured_dbm",
        "wifi_signal_margin_db", "cell_signal_margin_db",
        "wifi_rtt_ms", "cell_rtt_ms", "wifi_loss_pct", "cell_loss_pct",
        "wifi_trend_db_per_s", "cell_trend_db_per_s",
    )
    ranges = {}
    missing = []
    for field in fields:
        values = [result["decision_snapshot"].get(field) for result in results.values()]
        if any(value is None for value in values):
            missing.append(field)
            ranges[field] = None
        else:
            ranges[field] = max(values) - min(values)
    return {
        "configured_conditions_identical": True,
        "signal_state_exact_match": all(
            ranges[field] == 0.0
            for field in (
                "wifi_rssi_model_dbm", "cell_rsrp_configured_dbm",
                "wifi_signal_margin_db", "cell_signal_margin_db",
            )
        ),
        "observed_feature_ranges_across_replays": ranges,
        "missing_snapshot_fields": missing,
        "random_realization_note": (
            "Probe and media outcomes may differ across independent unseeded replays; "
            "the configured distributions and ABBA ordering are controlled."
        ),
    }


def run_one(action: str, scenario: dict, replay_dir: Path, station, server,
            cell_gateway, ap, seed_supported: bool, run, helper) -> dict:
    wifi_if = station.wintfs[0].name
    movement = scenario["movement"]
    decision_stage_index = int(scenario["decision_stage_index"])
    duration_s = float(scenario["duration_s"])
    seed_offset = int(scenario.get("seed_offset", 0))
    if seed_offset < 0:
        raise ValueError("seed_offset must be nonnegative")
    station.position = [float(movement[0]["x_m"]), 40.0, 0.0]
    helper.connect_verified(station, ap, (ap,), run)
    profile_log = {
        "cell": configure_access_profile(
            station, "sta1-5g0", scenario["cell_profile"], 9001 + seed_offset,
            seed_supported, run
        )
    }
    wifi_profile_names = list(dict.fromkeys(stage["wifi_profile"] for stage in movement))
    seed_map = {
        name: 1001 + seed_offset + index
        for index, name in enumerate(wifi_profile_names)
    }
    movement_log = []
    sender_script = replay_dir / "sender.py"
    receiver_script = replay_dir / "receiver.py"
    sender_script.write_text(compare.SENDER_CODE, encoding="utf-8")
    receiver_script.write_text(compare.RECEIVER_CODE, encoding="utf-8")
    control = replay_dir / "selected_access.txt"
    control.write_text("wifi\n", encoding="utf-8")
    packets = replay_dir / "packet_arrivals.jsonl"
    sender_summary = replay_dir / "sender_summary.json"
    receiver_summary = replay_dir / "receiver_summary.json"
    sender_output = (replay_dir / "sender_output.txt").open("w", encoding="utf-8")
    receiver_output = (replay_dir / "receiver_output.txt").open("w", encoding="utf-8")
    sender = receiver = None
    try:
        start = time.monotonic() + 1.5
        stop = start + duration_s
        start_ns = int(start * 1e9)
        receiver = server.popen(
            [sys.executable, str(receiver_script), str(base.UDP_PORT), str(stop + 1),
             str(packets), str(receiver_summary)],
            stdout=receiver_output, stderr=subprocess.STDOUT,
        )
        sender = station.popen(
            [sys.executable, str(sender_script), base.WIFI_UE_IP, wifi_if,
             base.WIFI_SERVER_IP, base.CELL_UE_IP, "sta1-5g0",
             base.CELL_SERVER_IP, str(base.UDP_PORT), str(start), str(stop),
             str(1 / compare.PACKETS_PER_SECOND), str(compare.PAYLOAD_BYTES),
             str(control), str(sender_summary)],
            stdout=sender_output, stderr=subprocess.STDOUT,
        )

        decision_ns = None
        snapshot = None
        timing = {
            "duplicate_start_ns": None,
            "candidate_verified_ns": None,
            "wifi_break_ns": None,
            "wifi_disconnected_verified": False,
        }
        previous_rssi = previous_elapsed = None
        for stage_index, stage in enumerate(movement):
            base.sleep_until(start + stage["at_s"])
            station.position = [stage["x_m"], 40.0, 0.0]
            profile_log[stage["wifi_profile"]] = configure_access_profile(
                station, wifi_if, stage["wifi_profile"],
                seed_map[stage["wifi_profile"]], seed_supported, run,
            )
            elapsed = time.monotonic() - start
            rssi = base.modeled_wifi_rssi(station, ap)
            trend = None
            if previous_rssi is not None and elapsed > previous_elapsed:
                trend = (rssi - previous_rssi) / (elapsed - previous_elapsed)
            movement_log.append({
                "scheduled_at_s": stage["at_s"], "applied_at_s": elapsed,
                "x_m": stage["x_m"], "wifi_profile": stage["wifi_profile"],
                "wifi_rssi_model_dbm": rssi, "wifi_trend_db_per_s": trend,
            })
            previous_rssi, previous_elapsed = rssi, elapsed
            if stage_index != decision_stage_index:
                continue

            wifi_probe = base.ping_path(
                station, wifi_if, base.WIFI_SERVER_IP, run,
                replay_dir / "decision_wifi_ping.txt", helper,
                count=compare.PROBE_COUNT,
            )
            cell_probe = base.ping_path(
                station, "sta1-5g0", base.CELL_SERVER_IP, run,
                replay_dir / "decision_cell_ping.txt", helper,
                count=compare.PROBE_COUNT,
            )
            decision_ns = time.monotonic_ns()
            cell_rsrp = compare.PLAN["hosn_policy"]["emulated_cellular_rsrp_dbm"]
            snapshot = {
                "scenario_id": scenario["id"],
                "direction": scenario["direction"],
                "speed_class": scenario["speed_class"],
                "design_role": scenario["design_role"],
                "wifi_rssi_model_dbm": rssi,
                "cell_rsrp_configured_dbm": cell_rsrp,
                "wifi_signal_margin_db": rssi - compare.PLAN["hosn_policy"]["wifi_service_floor_dbm"],
                "cell_signal_margin_db": cell_rsrp - compare.PLAN["hosn_policy"]["emulated_cellular_service_floor_rsrp_dbm"],
                "wifi_rtt_ms": wifi_probe["rtt_avg_ms"],
                "cell_rtt_ms": cell_probe["rtt_avg_ms"],
                "wifi_loss_pct": wifi_probe["loss_pct"],
                "cell_loss_pct": cell_probe["loss_pct"],
                "wifi_trend_db_per_s": trend,
                "cell_trend_db_per_s": 0.0,
                "signal_provenance": {
                    "wifi": "Mininet-WiFi propagation model",
                    "cell": "constant emulated cellular profile, not RF measurement",
                },
            }
            snapshot["rule_evaluation"] = evaluate_snapshot_rules(snapshot)
            if action == HANDOVER:
                control.write_text("duplicate\n", encoding="utf-8")
                duplicate_ns = time.monotonic_ns()
                timing["duplicate_start_ns"] = duplicate_ns
                verified = compare.wait_for_cell_packets(
                    packets, duplicate_ns, compare.MIN_VERIFIED_CELL_PACKETS
                )
                if verified < compare.MIN_VERIFIED_CELL_PACKETS:
                    raise RuntimeError("candidate media path was not verified")
                timing["candidate_verified_ns"] = time.monotonic_ns()
                base.sleep_until(duplicate_ns / 1e9 + compare.OVERLAP_MIN_S)
                control.write_text("5g\n", encoding="utf-8")
                compare.disconnect_wifi(station, ap, run, helper)
                timing["wifi_break_ns"] = time.monotonic_ns()
                timing["wifi_disconnected_verified"] = helper.actual_link(station, run)[0] == ""

        if decision_ns is None or snapshot is None:
            raise RuntimeError("decision snapshot was not captured")
        base.sleep_until(stop + 1.2)
        base.process_wait(sender, "sender")
        base.process_wait(receiver, "receiver")
        sender_output.close(); receiver_output.close()
        sender_output = receiver_output = None
        sender_data = json.loads(sender_summary.read_text(encoding="utf-8"))
        receiver_data = json.loads(receiver_summary.read_text(encoding="utf-8"))
        events = compare.read_events(packets)
        summary = summarize_outcome(
            sender_data, receiver_data, events, action, start_ns, decision_ns,
            timing, duration_s,
        )
        summary.update(
            scenario_id=scenario["id"],
            scenario_design_role=scenario["design_role"],
            decision_snapshot=snapshot,
            timing=timing,
            movement=movement_log,
            profiles=profile_log,
            netem_seed_supported=seed_supported,
        )
        compare.write_json(replay_dir / "summary.json", summary)
        return summary
    finally:
        for process in (sender, receiver):
            if process is not None and process.poll() is None:
                process.kill(); process.wait(timeout=3)
        for handle in (sender_output, receiver_output):
            if handle is not None:
                handle.close()


def run_pilot(root: Path) -> int:
    import hosn_switch as helper
    import mn_wifi.net as wifi_module

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = root / "results" / ("paired_outcome_pilot_" + stamp)
    folder.mkdir(parents=True)
    compare.write_json(folder / "plan.json", PLAN)
    source_names = (
        "hosn_paired_outcome_pilot.py", "hosn_wifi_5g_compare.py",
        "hosn_wifi_5g_pilot.py", "hosn_switch.py",
    )
    manifest = {
        "revision": REVISION, "status": "starting", "created_utc": stamp,
        "data_kind": "paired_pilot_not_training_dataset",
        "training_rows_created": 0, "training_labels_assigned": 0,
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
        results = {}
        for run_id, action in RUN_ORDER:
            replay_dir = folder / run_id
            replay_dir.mkdir()
            print("\nRunning {} ({})...".format(run_id, action), flush=True)
            result = run_one(
                action, DEFAULT_SCENARIO, replay_dir, station, server, cell_gateway, ap,
                seed_capability["supported"], run, helper,
            )
            results[run_id] = result
            post = result["post_decision"]
            print(
                "  post-loss={:.3f}% gap={:.3f}ms frames={:.2f}% goodput={:.3f}Mbps".format(
                    post["loss_pct"], post["max_interarrival_gap_ms"],
                    post["frames"]["complete_pct"], post["application_goodput_mbps"],
                ), flush=True,
            )
        aggregate = aggregate_by_action(results)
        recommendation = review_recommendation(aggregate)
        pairing = pairing_diagnostics(results)
        valid = all(result["pilot_valid"] for result in results.values())
        report = {
            "revision": REVISION,
            "paired_pilot_only": True,
            "final_dataset_started": False,
            "training_rows_created": 0,
            "training_labels_assigned": 0,
            "mechanical_checks_pass": valid,
            "netem_seed_capability": seed_capability,
            "run_order": PLAN["run_order"],
            "results": results,
            "aggregate_by_action": aggregate,
            "pairing_diagnostics": pairing,
            "review_recommendation": recommendation,
            "honesty_note": (
                "Configured conditions are repeated identically. Where netem seed is unsupported, "
                "random loss events cannot be identical; ABBA repeats reduce ordering bias but do not remove it."
            ),
        }
        compare.write_json(folder / "paired_report.json", report)
        manifest["status"] = "pilot_complete" if valid else "pilot_checks_failed"
        manifest["report_file"] = "paired_report.json"
        compare.write_json(folder / "manifest.json", manifest)
        print("\nPAIRED OUTCOME PILOT {} — final dataset NOT started.".format(
            "COMPLETE" if valid else "CHECKS FAILED"
        ))
        print("Review recommendation:", recommendation["recommendation_for_human_review"])
        print("Saved:", folder)
        return 0 if valid else 1
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        return 130
    except Exception as exc:
        manifest.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        (folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("\nPAIRED PILOT STOPPED:", exc)
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


def run_matrix_pilot(root: Path) -> int:
    """Run four small paired scenarios; never emit a training dataset."""
    import hosn_switch as helper
    import mn_wifi.net as wifi_module

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = root / "results" / ("paired_matrix_pilot_" + stamp)
    folder.mkdir(parents=True)
    compare.write_json(folder / "plan.json", PLAN)
    source_names = (
        "hosn_paired_outcome_pilot.py", "hosn_wifi_5g_compare.py",
        "hosn_wifi_5g_pilot.py", "hosn_switch.py",
    )
    manifest = {
        "revision": REVISION, "status": "starting", "created_utc": stamp,
        "data_kind": "multi_scenario_paired_pilot_not_training_dataset",
        "scenario_count": len(MATRIX_SCENARIOS),
        "planned_replays": len(MATRIX_SCENARIOS) * len(RUN_ORDER),
        "training_rows_created": 0, "training_labels_assigned": 0,
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

        scenario_reports = {}
        total_completed = 0
        for scenario in MATRIX_SCENARIOS:
            scenario_id = scenario["id"]
            scenario_dir = folder / scenario_id
            scenario_dir.mkdir()
            print("\nSCENARIO:", scenario_id, flush=True)
            results = {}
            for run_id, action in RUN_ORDER:
                replay_dir = scenario_dir / run_id
                replay_dir.mkdir()
                print("  Running {} ({})...".format(run_id, action), flush=True)
                result = run_one(
                    action, scenario, replay_dir, station, server,
                    cell_gateway, ap, seed_capability["supported"], run, helper,
                )
                results[run_id] = result
                total_completed += 1
                post = result["post_decision"]
                print(
                    "    loss={:.3f}% gap={:.3f}ms frames={:.2f}% goodput={:.3f}Mbps".format(
                        post["loss_pct"], post["max_interarrival_gap_ms"],
                        post["frames"]["complete_pct"],
                        post["application_goodput_mbps"],
                    ), flush=True,
                )
            aggregate = aggregate_by_action(results)
            recommendation = review_recommendation(aggregate)
            report = {
                "scenario": scenario,
                "results": results,
                "aggregate_by_action": aggregate,
                "pairing_diagnostics": pairing_diagnostics(results),
                "review_recommendation": recommendation,
                "training_rows_created": 0,
                "training_labels_assigned": 0,
                "mechanical_checks_pass": all(item["pilot_valid"] for item in results.values()),
            }
            statuses = [
                item["decision_snapshot"]["rule_evaluation"]["status"]
                for item in results.values()
            ]
            expected_conflict = scenario["design_role"].startswith("conflict_")
            report["observed_rule_statuses"] = statuses
            report["scenario_design_check_pass"] = (
                all(status == "CONFLICT" for status in statuses)
                if expected_conflict
                else all(status == "CLEAR" for status in statuses)
            )
            scenario_reports[scenario_id] = report
            compare.write_json(scenario_dir / "scenario_report.json", report)
            print("  Review recommendation:", recommendation["recommendation_for_human_review"], flush=True)

        valid = all(
            report["mechanical_checks_pass"] and report["scenario_design_check_pass"]
            for report in scenario_reports.values()
        )
        matrix_report = {
            "revision": REVISION,
            "matrix_pilot_only": True,
            "final_dataset_started": False,
            "training_rows_created": 0,
            "training_labels_assigned": 0,
            "scenario_count": len(scenario_reports),
            "completed_replays": total_completed,
            "mechanical_checks_pass": valid,
            "netem_seed_capability": seed_capability,
            "scenario_reports": scenario_reports,
            "honesty_note": (
                "Recommendations are review candidates, not training labels. "
                "Inconclusive outcomes remain inconclusive. Unseeded platforms "
                "use ABBA repeats but cannot reproduce identical random loss events."
            ),
        }
        compare.write_json(folder / "matrix_report.json", matrix_report)
        manifest["status"] = "matrix_pilot_complete" if valid else "matrix_pilot_checks_failed"
        manifest["completed_replays"] = total_completed
        manifest["report_file"] = "matrix_report.json"
        compare.write_json(folder / "manifest.json", manifest)
        print("\nPAIRED MATRIX PILOT {} — final dataset NOT started.".format(
            "COMPLETE" if valid else "CHECKS FAILED"
        ))
        print("Completed replays:", total_completed)
        print("Saved:", folder)
        return 0 if valid else 1
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        return 130
    except Exception as exc:
        manifest.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        (folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("\nMATRIX PILOT STOPPED:", exc)
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


def run_self_tests() -> int:
    import unittest

    class Tests(unittest.TestCase):
        def test_no_dataset_or_training_labels(self):
            self.assertFalse(PLAN["dataset_collection"])
            self.assertEqual(PLAN["training_rows_created"], 0)
            self.assertEqual(PLAN["training_labels_assigned"], 0)

        def test_abba_action_order(self):
            self.assertEqual([action for _, action in RUN_ORDER], [STAY, HANDOVER, HANDOVER, STAY])

        def test_both_actions_repeated(self):
            self.assertEqual(Counter(action for _, action in RUN_ORDER), Counter({STAY: 2, HANDOVER: 2}))

        def test_decision_stage_is_moving_edge(self):
            self.assertEqual(PLAN["scenario"]["decision_stage"]["wifi_profile"], "wifi_edge")

        def test_matrix_covers_required_contexts(self):
            self.assertEqual(len(MATRIX_SCENARIOS), 4)
            self.assertEqual(
                {(item["direction"], item["speed_class"]) for item in MATRIX_SCENARIOS},
                {("outbound", "fast"), ("outbound", "slow"),
                 ("inbound", "fast"), ("inbound", "slow")},
            )
            self.assertEqual(
                sum(item["design_role"].startswith("conflict_") for item in MATRIX_SCENARIOS), 2
            )
            self.assertEqual(PLAN["matrix_pilot"]["total_replays"], 16)

        def test_matrix_scenarios_are_executable(self):
            ids = [item["id"] for item in MATRIX_SCENARIOS]
            self.assertEqual(len(ids), len(set(ids)))
            for scenario in MATRIX_SCENARIOS:
                self.assertIn(scenario["cell_profile"], ALL_ACCESS_PROFILES)
                self.assertGreaterEqual(scenario["decision_stage_index"], 1)
                self.assertLess(scenario["decision_stage_index"], len(scenario["movement"]))
                self.assertGreater(
                    scenario["duration_s"],
                    scenario["movement"][scenario["decision_stage_index"]]["at_s"] + 5.0,
                )
                for stage in scenario["movement"]:
                    self.assertIn(stage["wifi_profile"], ALL_ACCESS_PROFILES)

        def test_campaign_seed_offsets_must_be_nonnegative(self):
            self.assertTrue(all(item.get("seed_offset", 0) >= 0 for item in MATRIX_SCENARIOS))

        def test_snapshot_rule_evaluation_detects_conflict(self):
            snapshot = {
                "wifi_rssi_model_dbm": -86.0, "cell_rsrp_configured_dbm": -90.0,
                "wifi_rtt_ms": 10.0, "cell_rtt_ms": 120.0,
                "wifi_loss_pct": 0.0, "cell_loss_pct": 4.0,
                "wifi_trend_db_per_s": -3.0, "cell_trend_db_per_s": 0.0,
            }
            result = evaluate_snapshot_rules(snapshot)
            self.assertEqual(result["status"], "CONFLICT")
            self.assertEqual(result["decision"], "ASK_AI")

        def test_dominance_handover(self):
            aggregate = {
                STAY: {"mean_post_loss_pct": 5.0, "mean_post_p95_delay_ms": 80.0,
                       "mean_post_max_gap_ms": 100.0, "mean_post_complete_frames_pct": 85.0,
                       "mean_post_goodput_mbps": 0.80},
                HANDOVER: {"mean_post_loss_pct": 0.2, "mean_post_p95_delay_ms": 40.0,
                           "mean_post_max_gap_ms": 30.0, "mean_post_complete_frames_pct": 99.0,
                           "mean_post_goodput_mbps": 0.95},
            }
            result = review_recommendation(aggregate)
            self.assertEqual(result["recommendation_for_human_review"], HANDOVER)
            self.assertFalse(result["training_label_assigned"])

        def test_tradeoff_remains_inconclusive(self):
            aggregate = {
                STAY: {"mean_post_loss_pct": 1.0, "mean_post_p95_delay_ms": 20.0,
                       "mean_post_max_gap_ms": 20.0, "mean_post_complete_frames_pct": 99.0,
                       "mean_post_goodput_mbps": 0.96},
                HANDOVER: {"mean_post_loss_pct": 0.0, "mean_post_p95_delay_ms": 60.0,
                           "mean_post_max_gap_ms": 80.0, "mean_post_complete_frames_pct": 100.0,
                           "mean_post_goodput_mbps": 0.97},
            }
            result = review_recommendation(aggregate)
            self.assertEqual(result["recommendation_for_human_review"], "INCONCLUSIVE")
            self.assertFalse(result["training_label_assigned"])

        def test_tolerances_are_finite_nonnegative(self):
            self.assertTrue(all(_finite(value) and value >= 0 for value in DOMINANCE_TOLERANCES.values()))

        def test_pairing_diagnostics_report_observed_ranges(self):
            snapshot = {
                "wifi_rssi_model_dbm": -86.0, "cell_rsrp_configured_dbm": -90.0,
                "wifi_signal_margin_db": -6.0, "cell_signal_margin_db": 15.0,
                "wifi_rtt_ms": 70.0, "cell_rtt_ms": 30.0,
                "wifi_loss_pct": 8.0, "cell_loss_pct": 0.0,
                "wifi_trend_db_per_s": -3.0, "cell_trend_db_per_s": 0.0,
            }
            result = pairing_diagnostics({
                "a": {"decision_snapshot": snapshot},
                "b": {"decision_snapshot": dict(snapshot, wifi_rtt_ms=72.0)},
            })
            self.assertTrue(result["signal_state_exact_match"])
            self.assertEqual(result["observed_feature_ranges_across_replays"]["wifi_rtt_ms"], 2.0)

        def test_stay_summary_keeps_real_packet_accounting(self):
            events = []
            for sequence in range(12):
                sent = 1_000_000_000 + sequence * 10_000_000
                events.append({
                    "sequence": sequence, "sent_monotonic_ns": sent,
                    "received_monotonic_ns": sent + 2_000_000,
                    "declared_path": "wifi", "source_ip": base.WIFI_UE_IP,
                    "frame": sequence // 4, "part": sequence % 4,
                    "payload_bytes": compare.PAYLOAD_BYTES,
                })
            summary = summarize_outcome(
                {"status": "complete", "scheduled_sequences": 12},
                {"status": "complete", "parse_errors": 0}, events, STAY,
                1_000_000_000, 1_040_000_000,
                {"duplicate_start_ns": None, "candidate_verified_ns": None,
                 "wifi_break_ns": None, "wifi_disconnected_verified": False},
            )
            self.assertEqual(summary["actual_lost_sequences"], 0)
            self.assertEqual(summary["duplicate_datagrams_received"], 0)
            self.assertEqual(summary["post_decision"]["frames"]["complete_pct"], 100.0)
            self.assertTrue(summary["pilot_valid"])

        def test_handover_summary_counts_duplicates(self):
            events = []
            for sequence in range(12):
                paths = ["wifi", "5g"] if sequence in (4, 5) else (["wifi"] if sequence < 6 else ["5g"])
                for path in paths:
                    sent = 1_000_000_000 + sequence * 10_000_000
                    events.append({
                        "sequence": sequence, "sent_monotonic_ns": sent,
                        "received_monotonic_ns": sent + 2_000_000 + (1 if path == "5g" else 0),
                        "declared_path": path,
                        "source_ip": base.CELL_UE_IP if path == "5g" else base.WIFI_UE_IP,
                        "frame": sequence // 4, "part": sequence % 4,
                        "payload_bytes": compare.PAYLOAD_BYTES,
                    })
            summary = summarize_outcome(
                {"status": "complete", "scheduled_sequences": 12},
                {"status": "complete", "parse_errors": 0}, events, HANDOVER,
                1_000_000_000, 1_040_000_000,
                {"duplicate_start_ns": 1_040_000_000,
                 "candidate_verified_ns": 1_050_000_000,
                 "wifi_break_ns": 2_040_000_000,
                 "wifi_disconnected_verified": True},
            )
            self.assertEqual(summary["duplicate_datagrams_received"], 2)
            self.assertEqual(summary["actual_lost_sequences"], 0)
            self.assertTrue(summary["pilot_valid"])

        def test_strict_json_plan(self):
            json.dumps(PLAN, allow_nan=False)

    result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    )
    if result.wasSuccessful():
        print("PASS: {} paired-pilot software checks. No network run; no dataset created.".format(result.testsRun))
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Measured paired STAY/HANDOVER pilot; never a dataset collector")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--matrix-pilot", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return run_self_tests()
    if args.plan:
        print(json.dumps(PLAN, indent=2, allow_nan=False))
        return 0
    if sys.platform != "linux" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        parser.exit(2, "Run --run or --matrix-pilot in Ubuntu with sudo.\n")
    missing = [name for name in ("ip", "iw", "ping", "tc") if not shutil.which(name)]
    if missing:
        parser.exit(2, "Missing required tools: " + ", ".join(missing) + "\n")
    root = Path(__file__).resolve().parent
    needed = (
        "hosn_wifi_5g_compare.py", "hosn_wifi_5g_pilot.py",
        "hosn_switch.py", "hosn_heterogeneous_controller.py",
    )
    if any(not (root / name).is_file() for name in needed):
        parser.exit(2, "Keep this file beside " + ", ".join(needed) + ".\n")
    return run_matrix_pilot(root) if args.matrix_pilot else run_pilot(root)


if __name__ == "__main__":
    raise SystemExit(main())
