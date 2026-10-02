#!/usr/bin/env python3
"""HOSN mobility-aware measured collector -- no training or invented labels.

Run from the existing Hosn project in the Ubuntu VM:
    python3 hosn_collect_mobility.py --self-test
    python3 hosn_collect_mobility.py --plan
    sudo python3 hosn_collect_mobility.py --pilot
    sudo python3 hosn_collect_mobility.py --collect

Requires the already tested hosn_switch.py and hosn_rules.py beside this file.
No existing project file, model, CSV, or saved result is overwritten.

This is a separate successor to hosn_collect.py.  The verified original file is
not modified.  The paired measurement protocol remains the same, but movement
histories are now role-aware and explicitly cover stationary, slow, and fast
motion in both directions.  The full default plan remains 128 samples.

Protocol:
* Measured Linux ping RTT/loss; model-derived RSSI and model-history trends.
* Path delays/loss are deliberately imposed with netem, NOT inferred from RSSI.
* One radio samples both APs sequentially, held at the decision position.
* Both STAY and HANDOVER are then tried separately from the same current AP.
  A ping train starts BEFORE the action, so its counts include the action period.
* Raw observations/settings/failures are kept. Labels are BLANK for joint review.
  Neither the rules' answer nor AI_UNAVAILABLE/STAY is a training answer.
* Speed and direction are scenario metadata, never additional AI features.
* This is an offline paired experiment, not a continuous real-device policy.
  The two replays have the same settings, NOT identical random packet events.

Primary references consulted 2026-10-02:
https://mininet-wifi.github.io/advanced/
https://wireless.docs.kernel.org/en/latest/en/users/documentation/iw.html
https://man7.org/linux/man-pages/man8/tc-netem.8.html
https://github.com/mininet/mininet/blob/master/mininet/node.py
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

REVISION = "mobility-paired-measurements-v3-role-aware"
# This is the agreed project interface, not a new set of decision rules.
FEATURE_ORDER = (
    "current_rssi", "candidate_rssi", "current_latency", "candidate_latency",
    "current_loss", "candidate_loss", "current_trend", "candidate_trend",
)
UNITS = dict(zip(FEATURE_ORDER, (
    "dBm (propagation model)", "dBm (propagation model)",
    "ms (measured ping RTT)", "ms (measured ping RTT)",
    "percent of ping requests", "percent of ping requests",
    "dB/second (timestamped model history)", "dB/second (timestamped model history)",
)))
# These numbers configure experimental CONDITIONS, never observation columns.
# Tuple order: current delay ms, candidate delay ms, current loss %, candidate loss %.
PROFILES = {
    "current_delay": (20.0, 0.0, 0.0, 0.0),
    "candidate_delay": (0.0, 20.0, 0.0, 0.0),
    "equal_paths": (2.0, 2.0, 0.0, 0.0),
    "current_loss": (2.0, 2.0, 12.0, 0.0),
    "candidate_loss": (2.0, 2.0, 0.0, 12.0),
    "candidate_faster_lossier": (20.0, 2.0, 0.0, 8.0),
    "candidate_slower_reliable": (2.0, 20.0, 8.0, 0.0),
    "similar_noisy_paths": (4.0, 5.0, 5.0, 5.0),
}
# These histories are defined relative to the current/candidate AP roles.
# This prevents "left-to-right" from always meaning the same AP role.  The two
# current-AP roles create both physical directions while keeping the matrix at:
# 8 profiles x 4 histories x 2 current-AP roles x 2 repeats = 128 samples.
#
# During the fixed two-second model history:
# - slow motion covers 4 m (2 m/s), representing deliberate walking;
# - fast motion covers 12 m (6 m/s), representing faster movement;
# - stationary histories have a measured zero trend rather than an invented one.
#
# These values configure the experiment plan.  They are NOT model features.
HISTORIES = {
    "stationary_current_side": {
        "mobility_class": "stationary", "relation": "current_side", "speed_mps": 0.0,
    },
    "stationary_candidate_side": {
        "mobility_class": "stationary", "relation": "candidate_side", "speed_mps": 0.0,
    },
    "slow_toward_candidate": {
        "mobility_class": "slow", "relation": "toward_candidate", "speed_mps": 2.0,
    },
    "fast_toward_candidate": {
        "mobility_class": "fast", "relation": "toward_candidate", "speed_mps": 6.0,
    },
}
READ_ME = """# HOSN measured data: read this before training

## Mobility-aware v3 scope
This batch follows the verified three-user topology work, but it remains a
controlled one-station paired collector so STAY and HANDOVER can be compared
without mixing several users' outcomes.  The four history families are:

* stationary on the current-AP side;
* stationary on the candidate-AP side;
* slow movement toward the candidate (2 m/s);
* fast movement toward the candidate (6 m/s).

Reversing the current AP reverses the physical travel direction, so both
left-to-right and right-to-left cases are present.  Speed, direction, history
name, AP names, profile, netem settings and split identifiers are experiment
metadata only.  They MUST NOT be supplied to the model.  Only the same eight
pre-action features listed below are model inputs.  The default full plan is
still 128 samples; this redesign changes coverage rather than inflating size.

## For the AI teammate
You do not need Mininet-WiFi to read these CSVs or train a model on your own
computer. You do need to understand the units, labels and limitations below.
The final integration/network test happens in the Ubuntu VM.

The system is RULES FIRST. Clear cases use hosn_rules.py. Only a complete
conflicting comparison reaches the supplied predictor in hosn_controller.py.
The old hosn_engine.py and old model are not used by this collector.
Your component must provide:

    model.predict([[current_rssi, candidate_rssi,
                    current_latency, candidate_latency,
                    current_loss, candidate_loss,
                    current_trend, candidate_trend]])

It must return exactly one string: STAY or HANDOVER, e.g. ['HANDOVER'].
Training/retraining belongs to the AI teammate; this script does NOT train.
Explain these concepts again when joining the chat rather than assuming prior
messages were read: current AP = current connection, candidate AP = alternative,
RSSI = signal level, RTT = round-trip delay, loss = unanswered requests,
trend = modeled signal change per second, label = justified training answer.

## Files
* inputs.csv: the eight pre-action inputs plus identifiers/quality metadata.
* outcomes.csv: a separate STAY and HANDOVER replay for every complete sample.
* labels_to_review.csv: decision is deliberately BLANK. Fill only AFTER review.
* raw/sample_<number>/record.json: scenario settings, timestamps and observations.
* raw/sample_<number>/*.txt and *.jsonl: original ping/iw/tc outputs and commands.
* manifest.json: protocol, versions, hashes, counts, grouping and limitations.
* plan.json: conditions and split assignments fixed BEFORE measurements.

## Evidence and labels are not the same thing
Do NOT run the old training script on these unlabelled rows. Do NOT label a
conflict STAY merely because the controller had no AI. Do NOT copy rule outputs
as targets. Do NOT call a same-target-trained model independent ground truth.

Use the pilot to agree on a service objective (acceptable delay/loss, switching
cost, comparison horizon) BEFORE the full batch or inspecting its held-out test
outcomes. Then review repeated paired outcomes under that recorded objective. A universal
'best AP' cannot be inferred when delay and reliability trade off without an
objective. Leave ties, unstable comparisons, incomplete probes and unusable
inputs unlabelled. Record the label reason, reviewer and label policy/version.
An answer derived from this experiment is task-specific supervision, NOT a
proved universal optimum. The ten-second observation horizon is a protocol
choice, not an agreement about the final service objective.

## Measurement meaning
RTT is measured from replies; no replies means missing RTT, never zero.
Loss is (sent-received)/sent * 100, not the configured netem loss percentage.
Probes use a fixed transmitted count (-c) WITHOUT ping's -w deadline. Python
separately bounds process time. The three-request warmup is not the 100-request
feature measurement. Probe *.txt.json files explain requested/sent counts,
actual AP checks and any rejection. A rejected probe is NOT converted to a
completed measurement or a fictional loss rate.
A short 100-packet window has limited precision (one packet is one percentage
point). Keep counts and repetitions; do not report tiny loss differences as
certain. Zero observed loss does not establish a zero true loss rate.

RSSI/trends come from timestamped Mininet-WiFi logDistance model observations,
NOT a physical Wi-Fi scan. Model history changes position without starting
Mininet's automatic association or RF link updates. Geometry is held still
while each AP is probed. With this backend, weak modeled RSSI is NOT shown to
cause the measured packet delay/loss. Path impairments are separately configured
on the switch's egress toward each AP. This is not RF fading/congestion ground
truth, download-throughput measurement, or evidence on real phones.

One radio visits both APs to collect inputs. Those visits are OFFLINE sampling,
not HOSN decisions, and would interrupt a real single-radio user's connection.
Both features refer to a static measurement stage, not simultaneous live scans.
The signal trend is the preceding modeled history (its timestamps/age are saved).

Each action replay starts from the same current AP and same position/conditions.
A new ping train starts BEFORE requesting the action; HANDOVER uses the verified
connection helper. STAY makes no decision-time switch. STAY and HANDOVER are
separate runs, not simultaneous counterfactuals. The kernel packet losses and
scheduler are not made identical. Repetitions and counterbalanced order help,
but do not guarantee causal comparability. The action window includes initial
old-path packets, the switching period, and new-path packets. Its mean RTT only
summarizes replies, so always assess losses too. Verification includes polling
waits; request_to_observation_s is NOT exact radio outage duration.

## Prevent data leakage
Use ONLY the eight named pre-action input columns as model features.
NEVER feed future outcomes, applied netem settings, labels, rule outputs,
scenario IDs, current/candidate AP names, or split identifiers to the model.

All repeats and reversed-current-AP roles from the same profile/history share a
split_group_id and suggested_split, assigned before collecting. Preserve those
groups across train/validation/test. Pilot rows are excluded from final training
and scoring. Re-running the same plan must not place duplicate scenarios in a
different split. Splits are preliminary: evaluation on NEW scenario families and
continuous switching is still needed. Do not tune your objective/model on test
outcomes. Do not compare AI solely on clear cases that never reach it.

## What the batch does NOT prove
A completed collection is NOT a completed trained model, final dataset size
approval, good AI accuracy, or an improvement over rules alone. Review class
coverage, conflict coverage, measurement quality and held-out outcomes first.
Matching Python/scikit-learn dependencies must be agreed before a model is
exported back to Ubuntu. Do not load an untrusted pickle.

## Beginner summary
Your teammate uses Mininet-WiFi to run the experiments. You use the saved inputs
and reviewed answers to train AI. The controller uses it only for unclear cases.
Raw measured data first, justified labels second, train/evaluate third, final
network comparison last. The original generated hosn_data.csv is untouched.
"""


@dataclass(frozen=True)
class Settings:
    feature_count: int = 100
    feature_interval_s: float = 0.05
    outcome_count: int = 100
    outcome_interval_s: float = 0.1
    history_duration_s: float = 2.0
    history_points: int = 5
    action_start_delay_s: float = 0.3

    def __post_init__(self):
        for n in (self.feature_count, self.outcome_count, self.history_points):
            if isinstance(n, bool) or not isinstance(n, int) or n < 2:
                raise ValueError("Counts must be integers >= 2.")
        for v in (self.feature_interval_s, self.outcome_interval_s,
                  self.history_duration_s, self.action_start_delay_s):
            if isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                raise ValueError("Intervals must be finite and positive.")
        if self.feature_interval_s < 0.02 or self.outcome_interval_s < 0.02:
            raise ValueError("Do not use this collector for flood pings.")
        if (self.outcome_count - 1) * self.outcome_interval_s < 5:
            raise ValueError("The outcome window must last at least five seconds.")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_csv(path: Path, fields, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        if new:
            writer.writeheader()
        writer.writerow(row)  # None becomes an empty cell, not an invented zero.
        handle.flush()
        os.fsync(handle.fileno())


def base_groups():
    return [p + "__" + h for p in PROFILES for h in HISTORIES]


def split_map() -> dict:
    # Fixed independently of --seed, repeat count, measurements and labels.
    groups = sorted(base_groups(), key=lambda s: hashlib.sha256(
        ("hosn-split-v1:" + s).encode()).hexdigest())
    n = max(1, len(groups) // 5)
    return {g: ("test" if i < n else "validation" if i < 2*n else "train")
            for i, g in enumerate(groups)}


def history_trajectory(history: str, current_ap: str,
                       duration_s: float = 2.0) -> dict:
    """Resolve a role-relative history into concrete geometry and metadata."""
    if history not in HISTORIES:
        raise ValueError("Unknown mobility history: " + str(history))
    if current_ap not in ("ap1", "ap2"):
        raise ValueError("Current AP must be ap1 or ap2.")
    if (isinstance(duration_s, bool) or not isinstance(duration_s, (int, float))
            or not math.isfinite(duration_s) or duration_s <= 0):
        raise ValueError("History duration must be finite and positive.")
    definition = HISTORIES[history]
    speed = float(definition["speed_mps"])
    relation = definition["relation"]
    if relation == "current_side":
        decision = 35.0 if current_ap == "ap1" else 65.0
        start = decision
    elif relation == "candidate_side":
        decision = 65.0 if current_ap == "ap1" else 35.0
        start = decision
    elif relation == "toward_candidate":
        # For current AP1 the candidate is AP2 (positive x); reversing the
        # current AP reverses the path while keeping the role relationship.
        sign = 1.0 if current_ap == "ap1" else -1.0
        decision = 55.0 if current_ap == "ap1" else 45.0
        start = decision - sign * speed * float(duration_s)
    else:
        raise ValueError("Unsupported mobility relation: " + str(relation))
    delta = decision - start
    measured_speed = abs(delta) / float(duration_s)
    if delta > 0:
        direction = "left_to_right"
    elif delta < 0:
        direction = "right_to_left"
    else:
        direction = "stationary"
    if not math.isclose(measured_speed, speed, rel_tol=0, abs_tol=1e-9):
        raise ValueError("History geometry does not match its declared speed.")
    return {
        "x_start": start,
        "x_decision": decision,
        "history_duration_s": float(duration_s),
        "mobility_class": definition["mobility_class"],
        "mobility_relation": relation,
        "mobility_speed_mps": speed,
        "mobility_direction": direction,
    }


def make_plan(repeats=2, seed=20261002, pilot=False) -> list:
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be positive.")
    splits = split_map()
    cases = []
    for repeat in range(repeats):
        for index, (profile, history, current) in enumerate(
                (p, h, a) for p in PROFILES for h in HISTORIES for a in ("ap1", "ap2")):
            candidate = "ap2" if current == "ap1" else "ap1"
            group = profile + "__" + history
            reverse = (repeat + index) % 2
            trajectory = history_trajectory(history, current)
            cases.append({
                "profile": profile, "history": history, "current_ap": current,
                "candidate_ap": candidate, "repeat": repeat + 1,
                "split_group_id": group, "suggested_split": splits[group],
                "action_order": ["HANDOVER", "STAY"] if reverse else ["STAY", "HANDOVER"],
                "probe_order": [candidate, current] if reverse else [current, candidate],
                **trajectory,
                "condition_settings": {
                    current: {"delay_ms": PROFILES[profile][0], "loss_pct": PROFILES[profile][2]},
                    candidate: {"delay_ms": PROFILES[profile][1], "loss_pct": PROFILES[profile][3]},
                },
            })
    random.Random(seed).shuffle(cases)  # Only experimental order, NOT observations.
    if pilot:
        # Four pilot cases exercise every history family and both directions.
        # They are diagnostics only and never enter final training/scoring.
        wanted = [
            ("candidate_delay", "stationary_current_side", "ap1"),
            ("current_delay", "stationary_candidate_side", "ap2"),
            ("current_delay", "slow_toward_candidate", "ap1"),
            ("candidate_delay", "fast_toward_candidate", "ap2"),
        ]
        cases = [next(c for c in cases
                      if (c["profile"], c["history"], c["current_ap"]) == key
                      and c["repeat"] == 1) for key in wanted]
        for c in cases:
            c["suggested_split"] = "pilot_only"
    for i, c in enumerate(cases, 1):
        c["sample_number"] = i
    return cases


def ping_args(interface: str, count: int, interval: float, target: str) -> tuple:
    """A fixed request count, with a separate Python process watchdog.

    Do NOT combine ping -c with -w. With a deadline, iputils can continue
    transmitting until the requested number of REPLIES arrives. A delayed
    three-request warmup can therefore transmit five requests, and a lossy
    100-request measurement can transmit more than 100. This breaks the fixed
    measurement window. See iputils doc/ping.xml (-c and -w), and
    ping/ping_common.c pinger(): the transmit-count cap applies when !deadline.

    -W is retained to bound waiting when no replies arrive. It is not a
    per-packet RTT cutoff. The outer Python timeout is a safety watchdog;
    exceeding it invalidates a probe, never invents a completed measurement.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Ping count must be a positive integer.")
    if (isinstance(interval, bool) or not isinstance(interval, (int, float))
            or not math.isfinite(interval) or interval <= 0):
        raise ValueError("Ping interval must be finite and positive.")
    process_timeout = float(math.ceil((count - 1) * interval + 8.0))
    return (["ping", "-n", "-D", "-I", interface, "-c", str(count),
             "-i", str(interval), "-W", "1", target], process_timeout)


def model_snapshot(station, aps) -> dict:
    vals = {}
    for ap in aps:
        value = float(station.wintfs[0].get_rssi(ap.wintfs[0], station.get_distance_to(ap)))
        if not math.isfinite(value) or value >= 0:
            raise ValueError("Invalid modeled RSSI for " + ap.name)
        vals[ap.name] = value
    return {"monotonic_s": time.monotonic(), "position": list(station.position),
            "rssi_dbm": vals}


def model_history(station, aps, start: float, end: float, settings: Settings) -> list:
    """Model geometry only. Do not call setPosition/configLinks: they may roam.

    This deliberately does NOT update an RF channel emulator. Actual packets use
    the static hwsim backend and separately controlled wired path impairments.
    """
    samples = []
    began = time.monotonic()
    for i in range(settings.history_points):
        fraction = i / (settings.history_points - 1)
        scheduled = began + settings.history_duration_s * fraction
        time.sleep(max(0.0, scheduled - time.monotonic()))
        x = start + (end - start) * fraction
        station.position = [x, 40.0, 0.0]
        samples.append(model_snapshot(station, aps))
    return samples


def model_trend(samples: list, ap_name: str) -> float:
    if len(samples) < 2:
        raise ValueError("Signal trend needs measured model history, not a guessed zero.")
    elapsed = samples[-1]["monotonic_s"] - samples[0]["monotonic_s"]
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Invalid signal observation interval.")
    return (samples[-1]["rssi_dbm"][ap_name] - samples[0]["rssi_dbm"][ap_name]) / elapsed


def verified_wire(switch, peer):
    pairs = switch.connectionsTo(peer)
    if len(pairs) != 1:
        raise RuntimeError("Expected exactly one experiment cable to " + peer.name)
    local, remote = pairs[0]
    if (getattr(local, "node", None) is not switch
            or getattr(remote, "node", None) is not peer
            or not local.name.startswith(switch.name + "-eth")
            or getattr(local, "link", None) is None
            or local.link is not getattr(remote, "link", None)):
        raise RuntimeError("Refusing to alter an unverified experiment interface.")
    return local


def configure_paths(switch, aps, conditions: dict, run) -> list:
    reports = []
    for ap in aps:
        interface = verified_wire(switch, ap)
        delay, loss = conditions[ap.name]["delay_ms"], conditions[ap.name]["loss_pct"]
        if not (math.isfinite(delay) and 0 <= delay <= 200
                and math.isfinite(loss) and 0 <= loss <= 50):
            raise ValueError("Unsupported experimental impairment settings.")
        text, code = run(switch, ["tc", "qdisc", "replace", "dev", interface.name,
                                 "root", "netem", "delay", str(delay) + "ms",
                                 "loss", "random", str(loss) + "%"])
        if code:
            raise RuntimeError("netem configuration failed: " + text.strip())
        text, code = run(switch, ["tc", "qdisc", "show", "dev", interface.name])
        if code or "netem" not in text:
            raise RuntimeError("Cannot verify netem configuration.")
        reports.append({"ap": ap.name, "interface": interface.name,
                        "configured_delay_ms": delay, "configured_loss_pct": loss,
                        "actual_qdisc_report": text})
    return reports


def probe(station, ap, run, path: Path, count: int, interval: float, helper) -> dict:
    """Measure a fixed-count ping window; retain diagnostics even when rejected.

    Packet loss, including 100% observed loss, is valid measurement output.
    Unexpected AP identity, incomplete sends, or invalid command output are
    measurement-quality failures. They must not be relabelled as packet loss
    or accepted just to make the collection pass.
    """
    expected = str(ap.wintfs[0].mac).lower()
    argv, timeout = ping_args(station.wintfs[0].name, count, interval, helper.SERVER_IP)
    diagnostic_path = path.with_name(path.name + ".json")
    diagnostic = {"revision": REVISION, "status": "started", "raw_file": path.name,
                  "expected_ap": ap.name, "expected_bssid": expected,
                  "requested": count, "interval_s": interval, "argv": argv,
                  "process_timeout_s": timeout, "fixed_transmit_count": True}
    try:
        before, before_text = helper.actual_link(station, run)
        diagnostic.update(bssid_before=before, iw_before=before_text)
        if before != expected:
            raise RuntimeError("{}: wrong AP before probe (expected {}, observed {}).".format(
                ap.name, expected, before or "disconnected"))
        began = time.monotonic()
        diagnostic["start_monotonic_s"] = began
        text, code = run(station, argv, timeout=timeout)
        ended = time.monotonic()
        path.write_text(text, encoding="utf-8")
        diagnostic.update(returncode=code, end_monotonic_s=ended)
        if code not in (0, 1):
            raise RuntimeError("ping command failed with code {}; inspect {}".format(code, path.name))
        parsed = helper.parse_ping(text)
        after, after_text = helper.actual_link(station, run)
        parsed.update(bssid_before=before, bssid_after=after,
                      ap_verified=before == after == expected,
                      requested=count, complete=parsed["sent"] == count,
                      start_monotonic_s=began, end_monotonic_s=ended)
        diagnostic.update(parsed)
        diagnostic["iw_after"] = after_text
        failures = []
        if not parsed["complete"]:
            failures.append("requested {} requests but ping reported {} sent and {} replies".format(
                count, parsed["sent"], parsed["received"]))
        if not parsed["ap_verified"]:
            failures.append("AP changed: expected {}, before {}, after {}".format(
                expected, before or "disconnected", after or "disconnected"))
        if failures:
            raise RuntimeError("Probe rejected for {} ({}): {}. See {}".format(
                ap.name, path.name, "; ".join(failures), diagnostic_path.name))
        diagnostic["status"] = "complete"
        return parsed
    except BaseException as exc:
        diagnostic.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "rejected",
                          error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        write_json(diagnostic_path, diagnostic)


def action_replay(station, aps, switch, setting: dict, action: str,
                  run, folder: Path, options: Settings, helper) -> dict:
    if action not in ("STAY", "HANDOVER"):
        raise ValueError("Only STAY/HANDOVER are experiment actions.")
    by_name = {ap.name: ap for ap in aps}
    current, candidate = by_name[setting["current_ap"]], by_name[setting["candidate_ap"]]
    expected = current if action == "STAY" else candidate
    expected_bssid = str(expected.wintfs[0].mac).lower()
    # Every replay starts on the SAME current AP, never on the sampling endpoint.
    run.log_path = folder / (action.lower() + "_commands.jsonl")
    conditions = configure_paths(switch, aps, setting["condition_settings"], run)
    reset = helper.connect_verified(station, current, aps, run)
    warmup = probe(station, current, run, folder / (action.lower() + "_warmup.txt"),
                   count=3, interval=0.2, helper=helper)
    current_bssid = str(current.wintfs[0].mac).lower()
    if helper.actual_link(station, run)[0] != current_bssid:
        raise RuntimeError("Current AP changed before outcome replay.")
    argv, timeout = ping_args(station.wintfs[0].name, options.outcome_count,
                             options.outcome_interval_s, helper.SERVER_IP)
    report = {"action": action, "conditions": conditions, "reset": reset,
              "warmup": warmup, "expected_final_ap": expected.name,
              "expected_final_bssid": expected_bssid, "argv": argv,
              "window_description": "ping train begins before action; both paths may contribute",
              "start_monotonic_s": time.monotonic(), "start_epoch_s": time.time()}
    process = None
    path = folder / (action.lower() + "_outcome_ping.txt")
    with path.open("w", encoding="utf-8") as raw:
        try:
            process = station.popen(argv, stdout=raw, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, LC_ALL="C", LANG="C"))
            time.sleep(options.action_start_delay_s)
            report["ping_running_at_action"] = process.poll() is None
            report["action_start_monotonic_s"] = time.monotonic()
            report["action_start_epoch_s"] = time.time()
            if not report["ping_running_at_action"]:
                raise RuntimeError("Ping ended before the action; no comparable outcome window.")
            try:
                if action == "HANDOVER":
                    report["connection_action"] = helper.connect_verified(
                        station, candidate, aps, run, expected_current=current_bssid)
                else:
                    # No manual/policy handover for STAY. It is not reset mid-window.
                    if helper.actual_link(station, run)[0] != current_bssid:
                        raise RuntimeError("Unexpected connection change in STAY replay.")
                    report["connection_action"] = {"connection_changed": False}
                report["action_verified_while_ping_running"] = process.poll() is None
            except Exception as exc:
                report["action_error"] = type(exc).__name__ + ": " + str(exc)
                report["action_verified_while_ping_running"] = False
            report["action_end_monotonic_s"] = time.monotonic()
            remaining = max(1.0, timeout - (time.monotonic() - report["start_monotonic_s"]))
            process.wait(timeout=remaining)
            report["returncode"] = process.returncode
        except BaseException:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=3)
            raise
    report["end_monotonic_s"] = time.monotonic()
    text = path.read_text(encoding="utf-8")
    if report["returncode"] not in (0, 1):
        raise RuntimeError("Outcome ping failed; inspect " + path.name)
    report.update(helper.parse_ping(text))
    report["requested"] = options.outcome_count
    final_bssid, final_text = helper.actual_link(station, run)
    report.update(final_bssid=final_bssid, final_iw=final_text,
                  final_ap_verified=final_bssid == expected_bssid,
                  complete=report["sent"] == options.outcome_count)
    report["valid_for_review"] = bool(
        report["complete"] and report["final_ap_verified"]
        and report["ping_running_at_action"]
        and report.get("action_verified_while_ping_running", False)
        and not report.get("action_error"))
    report["request_to_observation_s"] = report.get("connection_action", {}).get(
        "request_to_observation_s")  # For STAY this is absent, not a fictional timing.
    report["label"] = None
    return report


INPUT_FIELDS = ["sample_id", "split_group_id", "suggested_split", "data_kind", "repeat",
                "current_ap", "candidate_ap", "mobility_class", "mobility_relation",
                "mobility_speed_mps_metadata_only", "mobility_direction_metadata_only",
                *FEATURE_ORDER, "features_complete",
                "current_probe_sent", "current_probe_received", "candidate_probe_sent",
                "candidate_probe_received", "feature_sampling_span_s", "history_duration_s",
                "history_age_at_sampling_end_s", "rule_decision_for_audit", "rule_status_for_audit"]
OUTCOME_FIELDS = ["sample_id", "action", "action_order", "requested", "sent", "received",
                  "loss_pct", "rtt_avg_ms", "final_ap_verified", "final_bssid",
                  "request_to_observation_s", "valid_for_review", "action_error"]
LABEL_FIELDS = ["sample_id", "split_group_id", "suggested_split", "data_kind",
                "measurement_review_ready", "decision", "label_status", "label_policy_version",
                "label_reason", "reviewer"]


def inputs_from_record(record: dict) -> dict:
    case, measurements, history = record["setting"], record["measurements"], record["history"]
    inputs = {}
    for prefix in ("current", "candidate"):
        ap_name = case[prefix + "_ap"]
        measured = measurements[ap_name]
        inputs[prefix + "_rssi"] = history[-1]["rssi_dbm"][ap_name]
        inputs[prefix + "_trend"] = model_trend(history, ap_name)
        inputs[prefix + "_latency"] = measured["rtt_avg_ms"]
        inputs[prefix + "_loss"] = measured["loss_pct"]
    return {name: inputs[name] for name in FEATURE_ORDER}


def valid_features(values: dict) -> bool:
    for name in FEATURE_ORDER:
        value = values.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            return False
        if name.endswith("rssi") and value >= 0:
            return False
        if name.endswith("latency") and value < 0:
            return False
        # 100% loss implies no RTT, so an eight-complete-input AI cannot use it.
        if name.endswith("loss") and not 0 <= value < 100:
            return False
    return True


def input_row(record: dict, rule) -> dict:
    c, v, m, h = record["setting"], record["inputs"], record["measurements"], record["history"]
    cur, cand = m[c["current_ap"]], m[c["candidate_ap"]]
    ended = max(cur["end_monotonic_s"], cand["end_monotonic_s"])
    began = min(cur["start_monotonic_s"], cand["start_monotonic_s"])
    return dict(sample_id=record["sample_id"], split_group_id=c["split_group_id"],
                suggested_split=c["suggested_split"], data_kind=record["data_kind"], repeat=c["repeat"],
                current_ap=c["current_ap"], candidate_ap=c["candidate_ap"],
                mobility_class=c["mobility_class"], mobility_relation=c["mobility_relation"],
                mobility_speed_mps_metadata_only=c["mobility_speed_mps"],
                mobility_direction_metadata_only=c["mobility_direction"], **v,
                features_complete=valid_features(v),
                current_probe_sent=cur["sent"], current_probe_received=cur["received"],
                candidate_probe_sent=cand["sent"], candidate_probe_received=cand["received"],
                feature_sampling_span_s=ended-began,
                history_duration_s=h[-1]["monotonic_s"]-h[0]["monotonic_s"],
                history_age_at_sampling_end_s=ended-h[-1]["monotonic_s"],
                rule_decision_for_audit=rule.decision, rule_status_for_audit=rule.status)


def label_row(record: dict) -> dict:
    c = record["setting"]
    ready = valid_features(record["inputs"])
    ready = ready and len(record.get("outcomes", [])) == 2
    ready = ready and {o["action"] for o in record.get("outcomes", [])} == {"STAY", "HANDOVER"}
    ready = ready and all(o["valid_for_review"] for o in record.get("outcomes", []))
    return dict(sample_id=record["sample_id"], split_group_id=c["split_group_id"],
                suggested_split=c["suggested_split"], data_kind=record["data_kind"],
                measurement_review_ready=ready, decision="", label_status="UNREVIEWED",
                label_policy_version="", label_reason="", reviewer="")


def create_network():
    from mininet.link import Link
    from mininet.log import setLogLevel
    from mininet.node import OVSBridge
    from mn_wifi.net import Mininet_wifi
    from mn_wifi.node import OVSBridgeAP
    from hosn_switch import SERVER_IP, SSID
    setLogLevel("info")
    network = Mininet_wifi(controller=None, switch=OVSBridge, accessPoint=OVSBridgeAP,
                           autoAssociation=False, allAutoAssociation=False)
    try:
        sta = network.addStation("sta1", ip="10.0.0.1/24", mac="02:00:00:00:00:01",
                                 position="50,40,0", range=50)
        host = network.addHost("h1", ip=SERVER_IP + "/24", mac="02:00:00:00:00:64")
        switch = network.addSwitch("s1", dpid="0000000000000010")
        ap1 = network.addAccessPoint("ap1", ssid=SSID, mode="g", channel="1",
                                    mac="02:00:00:00:01:01", position="20,50,0", range=50)
        ap2 = network.addAccessPoint("ap2", ssid=SSID, mode="g", channel="6",
                                    mac="02:00:00:00:02:01", position="80,50,0", range=50)
        network.setPropagationModel(model="logDistance", exp=3.5)
        configure = getattr(network, "configureNodes", None)
        (configure or network.configureWifiNodes)()
        for peer in (host, ap1, ap2):
            network.addLink(switch, peer, cls=Link)
        network.build()
        for bridge in (switch, ap1, ap2):
            bridge.start([])
        time.sleep(1.0)
        return network, sta, (ap1, ap2), switch
    except BaseException:
        network.stop()
        raise


def collect(root: Path, options: Settings, plan: list, pilot: bool, seed: int) -> int:
    import hosn_switch as helper
    from hosn_rules import AI_FEATURE_ORDER, DEFAULT_CONFIG, evaluate_rules
    import mn_wifi.net as wifi_module
    if tuple(AI_FEATURE_ORDER) != FEATURE_ORDER:
        raise RuntimeError("The agreed feature order changed. Review before collecting.")
    for name in ("s1", "ap1", "ap2"):
        result = subprocess.run(["ip", "link", "show", "dev", name],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4)
        if result.returncode == 0:
            raise RuntimeError("Another experiment may still be running (" + name + "). Exit it first.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_id = ("mobility_dataset_pilot_" if pilot else "mobility_dataset_") + stamp
    folder = root / "results" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "raw").mkdir()
    (folder / "README_FOR_AI.md").write_text(READ_ME, encoding="utf-8")
    write_json(folder / "plan.json", plan)
    manifest = {
        "revision": REVISION, "run_id": run_id, "created_utc": stamp,
        "status": "incomplete",
        "data_kind": "mobility_pilot_only" if pilot else "mobility_measured_development_batch",
        "planned_samples": len(plan), "completed_samples": 0,
        "measurement_review_ready": 0, "labels_assigned": 0, "model_trained": False,
        "feature_order": list(FEATURE_ORDER), "units": UNITS, "settings": asdict(options),
        "protocol": "role-aware model mobility, offline sequential AP probes, then counterbalanced paired actions",
        "wireless_backend": "default hwsim, no wmediumd; no physical RSSI measurements",
        "position_updates": "model geometry only, NOT automatic association or RF-channel updates",
        "mobility_histories": HISTORIES,
        "mobility_metadata_excluded_from_ai_features": [
            "history", "mobility_class", "mobility_relation",
            "mobility_speed_mps", "mobility_direction", "x_start", "x_decision",
        ],
        "verified_richer_topology_result": (
            "A separate three-user UDP/ping validation passed before this collector redesign; "
            "that run is evidence for topology feasibility, not training data."
        ),
        "qos": "netem one-way switch egress towards each AP; settings are NOT observations",
        "ping_mode": "fixed transmitted count (-c), no ping -w; separate Python watchdog",
        "probe_diagnostics": "raw/*.txt.json records requested/sent counts and actual AP checks",
        "associations": "verified by Linux iw, not associatedTo alone",
        "outcomes": "ten-second nominal ping trains start before the STAY/HANDOVER action",
        "rules": "audit annotation only; not used to select/rewrite actions or training labels",
        "rule_config": asdict(DEFAULT_CONFIG), "order_seed": seed,
        "packet_loss_rng_seed": None,
        "split_policy": "fixed scenario grouping v1 before observations; roles/repeats stay together",
        "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "python_version": platform.python_version(), "kernel": platform.release(),
        "machine": platform.machine(), "mininet_wifi_version": str(getattr(wifi_module, "VERSION", "unknown")),
        "source_sha256": {n: hashlib.sha256((root / n).read_bytes()).hexdigest()
                          for n in ("hosn_collect_mobility.py", "hosn_switch.py", "hosn_rules.py")},
    }
    write_json(folder / "manifest.json", manifest)
    old_cwd, network, result_code = Path.cwd(), None, 1
    try:
        os.chdir(folder)  # Hostapd/Mininet transient files stay inside this new run.
        print("HOSN mobility-aware measured collector: " + REVISION, flush=True)
        print("{} samples. No model is loaded. Training answers stay BLANK.".format(len(plan)), flush=True)
        print("Each sample measures both paths, then separately tests STAY and HANDOVER.", flush=True)
        network, station, aps, switch = create_network()
        by_name = {ap.name: ap for ap in aps}
        durations, routes, invalid_streak = [], Counter(), 0
        for setting in plan:
            number = setting["sample_number"]
            sample_id = run_id + "__" + str(number).zfill(4)
            sample_dir = folder / "raw" / ("sample_" + str(number).zfill(4))
            sample_dir.mkdir()
            record = {"sample_id": sample_id, "setting": setting, "status": "incomplete",
                      "data_kind": manifest["data_kind"], "measurements": {}, "outcomes": []}
            started = time.monotonic()
            print("\n[{}/{}] {} | {} | current {}".format(number, len(plan), setting["profile"],
                  setting["history"], setting["current_ap"]), flush=True)
            try:
                if not math.isclose(setting["history_duration_s"], options.history_duration_s,
                                    rel_tol=0, abs_tol=1e-9):
                    raise RuntimeError("Plan history duration does not match collector settings.")
                run = helper.NodeCommands(sample_dir / "setup_commands.jsonl")
                record["path_configuration"] = configure_paths(switch, aps, setting["condition_settings"], run)
                record["history"] = model_history(station, aps, setting["x_start"],
                                                   setting["x_decision"], options)
                for name in setting["probe_order"]:
                    ap = by_name[name]
                    run.log_path = sample_dir / ("features_" + name + "_commands.jsonl")
                    helper.connect_verified(station, ap, aps, run)
                    probe(station, ap, run, sample_dir / ("features_" + name + "_warmup.txt"),
                          count=3, interval=0.2, helper=helper)
                    observation = probe(station, ap, run, sample_dir / ("features_" + name + ".txt"),
                                        count=options.feature_count,
                                        interval=options.feature_interval_s, helper=helper)
                    record["measurements"][name] = observation
                    print("  Input {}: RTT {} ms | loss {:.1f}% | {}/{} replies".format(
                          name, observation["rtt_avg_ms"], observation["loss_pct"],
                          observation["received"], observation["sent"]), flush=True)
                record["inputs"] = inputs_from_record(record)
                rule = evaluate_rules(**record["inputs"])
                record["rule_annotation_not_label"] = asdict(rule)
                append_csv(folder / "inputs.csv", INPUT_FIELDS, input_row(record, rule))
                routes[rule.status] += 1
                write_json(sample_dir / "record.json", record)
                for action_index, action in enumerate(setting["action_order"], 1):
                    outcome = action_replay(station, aps, switch, setting, action, run,
                                            sample_dir, options, helper)
                    outcome["order"] = action_index
                    record["outcomes"].append(outcome)
                    row = {k: outcome.get(k) for k in OUTCOME_FIELDS
                           if k not in ("sample_id", "action_order")}
                    row.update(sample_id=sample_id, action_order=action_index)
                    append_csv(folder / "outcomes.csv", OUTCOME_FIELDS, row)
                    print("  {} replay: RTT {} ms | loss {:.1f}% | AP verified {}".format(
                          action, outcome["rtt_avg_ms"], outcome["loss_pct"],
                          outcome["final_ap_verified"]), flush=True)
                    write_json(sample_dir / "record.json", record)
                labels = label_row(record)
                append_csv(folder / "labels_to_review.csv", LABEL_FIELDS, labels)
                ready = labels["measurement_review_ready"]
                record["status"] = "complete_for_review" if ready else "needs_measurement_review"
                manifest["completed_samples"] += 1
                manifest["measurement_review_ready"] += int(ready)
                invalid_streak = 0 if ready else invalid_streak + 1
                elapsed = time.monotonic() - started
                durations.append(elapsed)
                mean = sum(durations) / len(durations)
                print("  Saved. Training answer: UNREVIEWED. Estimated batch time left: {:.1f} min".format(
                      (len(plan) - number) * mean / 60), flush=True)
                if invalid_streak >= 3:
                    raise RuntimeError("Three consecutive incomplete/invalid samples; stop and inspect before more collection.")
            except BaseException as exc:
                record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                              error=type(exc).__name__ + ": " + str(exc))
                (sample_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                raise
            finally:
                record["elapsed_s"] = time.monotonic() - started
                write_json(sample_dir / "record.json", record)
                manifest["observed_rule_status_counts_not_labels"] = dict(routes)
                write_json(folder / "manifest.json", manifest)
        manifest["status"] = "measurements_collected_labels_pending"
        result_code = 0 if manifest["measurement_review_ready"] == len(plan) else 1
        print("\nMEASUREMENTS COLLECTED: {} / {} samples.".format(
              manifest["completed_samples"], len(plan)), flush=True)
        print("Ready for measurement/label review: {}. Assigned training answers: 0.".format(
              manifest["measurement_review_ready"]), flush=True)
        if pilot:
            print("PILOT ONLY: inspect results before the full collection; do not train on this pilot.", flush=True)
    except KeyboardInterrupt:
        manifest["status"], result_code = "interrupted_partial_data_preserved", 130
        print("\nInterrupted. Completed measurements are preserved; labels are not invented.")
    except Exception as exc:
        manifest.update(status="failed_partial_data_preserved", error=type(exc).__name__ + ": " + str(exc))
        (folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("\nCOLLECTION STOPPED:", exc)
    finally:
        try:
            if network is not None:
                print("\nStopping this simulated network...")
                network.stop()
        except Exception as exc:
            manifest["cleanup_error"], result_code = repr(exc), 1
        finally:
            os.chdir(old_cwd)
            write_json(folder / "manifest.json", manifest)
            try:
                helper.restore_result_owner(folder)
            except OSError as exc:
                print("Could not restore result ownership:", exc)
            print("Saved data:", folder)
    return result_code


def run_self_tests() -> int:
    """Pure software tests with fixtures, never exported as a measured dataset."""
    import tempfile
    from types import SimpleNamespace
    import unittest
    from unittest.mock import patch

    class Tests(unittest.TestCase):
        def test_full_plan_size(self):
            self.assertEqual(len(make_plan()), 128)

        def test_pilot(self):
            p = make_plan(pilot=True)
            self.assertEqual(len(p), 4)
            self.assertEqual({c["suggested_split"] for c in p}, {"pilot_only"})
            self.assertEqual({c["history"] for c in p}, set(HISTORIES))
            self.assertEqual({c["current_ap"] for c in p}, {"ap1", "ap2"})

        def test_all_profiles_and_roles(self):
            p = make_plan()
            self.assertEqual({c["profile"] for c in p}, set(PROFILES))
            self.assertEqual({c["history"] for c in p}, set(HISTORIES))
            self.assertEqual({c["current_ap"] for c in p}, {"ap1", "ap2"})

        def test_repeat_and_order_balance(self):
            pairs = {}
            for c in make_plan():
                key = (c["profile"], c["history"], c["current_ap"])
                pairs.setdefault(key, []).append(c["action_order"][0])
            self.assertTrue(all(sorted(v) == ["HANDOVER", "STAY"] for v in pairs.values()))

        def test_split_leakage_guard(self):
            groups = {}
            for c in make_plan(repeats=3):
                groups.setdefault(c["split_group_id"], set()).add(c["suggested_split"])
            self.assertTrue(all(len(v) == 1 for v in groups.values()))
            self.assertEqual(set(split_map().values()), {"train", "validation", "test"})

        def test_order_seed_does_not_change_split(self):
            a, b = make_plan(seed=1), make_plan(seed=2)
            self.assertNotEqual(a[0]["profile"], b[0]["profile"])
            self.assertEqual({(c["split_group_id"], c["suggested_split"]) for c in a},
                             {(c["split_group_id"], c["suggested_split"]) for c in b})

        def test_no_scenario_has_labels_or_fake_readings(self):
            for c in make_plan():
                self.assertNotIn("decision", c)
                for feature in FEATURE_ORDER:
                    self.assertNotIn(feature, c)

        def test_all_model_positions_within_both_ranges(self):
            for case in make_plan(repeats=1):
                for x in (case["x_start"], case["x_decision"]):
                    self.assertLess(math.hypot(x - 20, -10), 50)
                    self.assertLess(math.hypot(x - 80, -10), 50)

        def test_role_reversal_gives_both_directions(self):
            slow_ap1 = history_trajectory("slow_toward_candidate", "ap1")
            slow_ap2 = history_trajectory("slow_toward_candidate", "ap2")
            self.assertEqual(slow_ap1["mobility_direction"], "left_to_right")
            self.assertEqual(slow_ap2["mobility_direction"], "right_to_left")
            self.assertAlmostEqual(slow_ap1["mobility_speed_mps"], 2.0)
            self.assertAlmostEqual(slow_ap2["mobility_speed_mps"], 2.0)
            self.assertAlmostEqual(slow_ap1["x_decision"] - slow_ap1["x_start"], 4.0)
            self.assertAlmostEqual(slow_ap2["x_decision"] - slow_ap2["x_start"], -4.0)

        def test_fast_history_is_faster_but_uses_same_duration(self):
            slow = history_trajectory("slow_toward_candidate", "ap1")
            fast = history_trajectory("fast_toward_candidate", "ap1")
            self.assertEqual(slow["history_duration_s"], fast["history_duration_s"])
            self.assertGreater(fast["mobility_speed_mps"], slow["mobility_speed_mps"])
            self.assertAlmostEqual(fast["x_decision"] - fast["x_start"], 12.0)

        def test_stationary_histories_are_role_relative(self):
            for current in ("ap1", "ap2"):
                current_side = history_trajectory("stationary_current_side", current)
                candidate_side = history_trajectory("stationary_candidate_side", current)
                self.assertEqual(current_side["mobility_direction"], "stationary")
                self.assertEqual(candidate_side["mobility_direction"], "stationary")
                self.assertEqual(current_side["x_start"], current_side["x_decision"])
                self.assertEqual(candidate_side["x_start"], candidate_side["x_decision"])
                self.assertNotEqual(current_side["x_decision"], candidate_side["x_decision"])

        def test_mobility_metadata_is_not_in_eight_features(self):
            self.assertEqual(len(FEATURE_ORDER), 8)
            for name in ("mobility_class", "mobility_relation", "mobility_speed_mps",
                         "mobility_direction", "x_start", "x_decision"):
                self.assertNotIn(name, FEATURE_ORDER)

        def test_no_model_import_for_plan(self):
            self.assertFalse(any(n.startswith("sklearn") for n in sys.modules))

        def test_ping_watchdog_grows_without_ping_deadline(self):
            a, t = ping_args("sta1-wlan0", 100, .1, "10.0.0.100")
            self.assertEqual(a[a.index("-c")+1], "100")
            self.assertGreater(t, 100 * .1)
            self.assertNotIn("-w", a)
            self.assertIn("-W", a)
            _, small_timeout = ping_args("sta1-wlan0", 3, .2, "10.0.0.100")
            self.assertGreater(t, small_timeout)

        def test_warmup_is_fixed_count_not_reply_count(self):
            a, t = ping_args("sta1-wlan0", 3, .2, "10.0.0.100")
            self.assertEqual(a[a.index("-c")+1], "3")
            self.assertNotIn("-w", a)
            self.assertEqual(a[-1], "10.0.0.100")
            self.assertGreater(t, 5.0)

        def test_count_and_interval_validation(self):
            for count in (True, 0, -1, 3.5, "3"):
                with self.assertRaises(ValueError):
                    ping_args("sta1-wlan0", count, .2, "10.0.0.100")
            for interval in (True, 0, -1, float("nan"), float("inf"), "0.2"):
                with self.assertRaises(ValueError):
                    ping_args("sta1-wlan0", 3, interval, "10.0.0.100")

        def fixture_probe(self, directory, text, requested=3, after=None,
                          before=None, code=0):
            import hosn_switch as helper
            station = SimpleNamespace(wintfs={0: SimpleNamespace(name="sta1-wlan0")})
            bssid = "02:00:00:00:02:01"
            ap = SimpleNamespace(name="ap2", wintfs={0: SimpleNamespace(mac=bssid)})
            observations = [(bssid if before is None else before, "iw before"),
                            (bssid if after is None else after, "iw after")]
            def run(node, argv, timeout):
                self.assertNotIn("-w", argv)
                self.assertEqual(argv[argv.index("-c") + 1], str(requested))
                return text, code
            with patch.object(helper, "actual_link", side_effect=observations):
                return probe(station, ap, run, Path(directory)/"features_ap2_warmup.txt",
                             requested, .2, helper)

        def test_delayed_three_request_warmup_is_valid(self):
            text = ("3 packets transmitted, 3 received, 0% packet loss\n"
                    "rtt min/avg/max/mdev = 200.0/608.73/1020.305/200.0 ms\n")
            with tempfile.TemporaryDirectory() as d:
                value = self.fixture_probe(d, text)
                meta = json.loads((Path(d)/"features_ap2_warmup.txt.json").read_text())
                self.assertTrue(value["complete"])
                self.assertTrue(value["ap_verified"])
                self.assertEqual(meta["status"], "complete")
                self.assertEqual(meta["rtt_avg_ms"], 608.73)
                self.assertEqual(meta["requested"], 3)
                self.assertNotIn("-w", meta["argv"])

        def test_five_sent_for_three_requested_is_explicit(self):
            # Regression fixture based on the reported failed warm-up summary.
            # It is NOT measured training data and is never exported as such.
            text = ("5 packets transmitted, 5 received, 0% packet loss, time 821ms\n"
                    "rtt min/avg/max/mdev = 199.738/608.730/1020.305/289.773 ms, pipe 5\n")
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaisesRegex(RuntimeError, "requested 3 requests.*5 sent.*5 replies"):
                    self.fixture_probe(d, text)
                meta = json.loads((Path(d)/"features_ap2_warmup.txt.json").read_text())
                self.assertTrue(meta["ap_verified"])
                self.assertFalse(meta["complete"])
                self.assertEqual(meta["loss_pct"], 0.)
                self.assertEqual(meta["status"], "rejected")
                self.assertEqual((Path(d)/"features_ap2_warmup.txt").read_text(), text)

        def test_five_sent_for_hundred_is_not_ninety_five_percent_loss(self):
            text = ("5 packets transmitted, 5 received, 0% packet loss\n"
                    "rtt min/avg/max/mdev = 1.0/2.0/3.0/0.1 ms\n")
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaisesRegex(RuntimeError, "requested 100 requests.*5 sent"):
                    self.fixture_probe(d, text, requested=100)
                meta = json.loads((Path(d)/"features_ap2_warmup.txt.json").read_text())
                self.assertEqual(meta["loss_pct"], 0.)
                self.assertFalse(meta["complete"])

        def test_complete_lossy_probe_remains_valid(self):
            text = ("100 packets transmitted, 85 received, 15% packet loss\n"
                    "rtt min/avg/max/mdev = 1.0/2.0/3.0/0.1 ms\n")
            with tempfile.TemporaryDirectory() as d:
                result = self.fixture_probe(d, text, requested=100)
                self.assertTrue(result["complete"])
                self.assertEqual(result["loss_pct"], 15.)
                self.assertEqual(result["received"], 85)

        def test_complete_total_loss_probe_keeps_unknown_rtt(self):
            text = "100 packets transmitted, 0 received, 100% packet loss\n"
            with tempfile.TemporaryDirectory() as d:
                result = self.fixture_probe(d, text, requested=100, code=1)
                self.assertTrue(result["complete"])
                self.assertEqual(result["loss_pct"], 100.)
                self.assertIsNone(result["rtt_avg_ms"])

        def test_actual_ap_change_has_distinct_error(self):
            text = ("3 packets transmitted, 3 received, 0% packet loss\n"
                    "rtt min/avg/max/mdev = 1.0/2.0/3.0/0.1 ms\n")
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaisesRegex(RuntimeError, "AP changed"):
                    self.fixture_probe(d, text, after="02:00:00:00:01:01")
                meta = json.loads((Path(d)/"features_ap2_warmup.txt.json").read_text())
                self.assertTrue(meta["complete"])
                self.assertFalse(meta["ap_verified"])

        def test_disconnection_is_not_silently_accepted(self):
            text = ("3 packets transmitted, 3 received, 0% packet loss\n"
                    "rtt min/avg/max/mdev = 1.0/2.0/3.0/0.1 ms\n")
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaisesRegex(RuntimeError, "disconnected"):
                    self.fixture_probe(d, text, after="")
                meta = json.loads((Path(d)/"features_ap2_warmup.txt.json").read_text())
                self.assertFalse(meta["ap_verified"])

        def test_wrong_start_ap_saved_without_sending_probe(self):
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaisesRegex(RuntimeError, "wrong AP before probe"):
                    self.fixture_probe(d, "", before="02:00:00:00:01:01")
                self.assertFalse((Path(d)/"features_ap2_warmup.txt").exists())
                meta = json.loads((Path(d)/"features_ap2_warmup.txt.json").read_text())
                self.assertEqual(meta["status"], "rejected")
                self.assertNotIn("sent", meta)

        def test_command_error_saves_raw_and_diagnostics(self):
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaisesRegex(RuntimeError, "code 2"):
                    self.fixture_probe(d, "ping: socket error\n", code=2)
                meta = json.loads((Path(d)/"features_ap2_warmup.txt.json").read_text())
                self.assertEqual(meta["returncode"], 2)
                self.assertEqual(meta["status"], "rejected")
                self.assertNotIn("sent", meta)

        def test_invalid_summary_saves_error_and_original_text(self):
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError):
                    self.fixture_probe(d, "unparseable output\n")
                meta = json.loads((Path(d)/"features_ap2_warmup.txt.json").read_text())
                self.assertEqual(meta["status"], "rejected")
                self.assertIn("ValueError", meta["error"])
                self.assertEqual((Path(d)/"features_ap2_warmup.txt").read_text(),
                                 "unparseable output\n")

        def test_bad_intervals_rejected(self):
            for value in (0, -1, float("nan"), float("inf"), True):
                with self.assertRaises(ValueError):
                    Settings(feature_interval_s=value)

        def test_short_outcome_rejected(self):
            with self.assertRaises(ValueError):
                Settings(outcome_count=5)

        def test_trend_uses_actual_time(self):
            h = [{"monotonic_s": 1, "rssi_dbm": {"ap1": -70}},
                 {"monotonic_s": 5, "rssi_dbm": {"ap1": -66}}]
            self.assertEqual(model_trend(h, "ap1"), 1)
            h[-1]["monotonic_s"] = 1
            with self.assertRaises(ValueError):
                model_trend(h, "ap1")

        def test_missing_history_not_zero_filled(self):
            with self.assertRaises(ValueError):
                model_trend([], "ap1")

        def test_negative_and_zero_trends(self):
            for last, expected in ((-75, -2.5), (-70, 0)):
                h = [{"monotonic_s": 10, "rssi_dbm": {"ap1": -70}},
                     {"monotonic_s": 12, "rssi_dbm": {"ap1": last}}]
                self.assertEqual(model_trend(h, "ap1"), expected)

        def test_link_lookup_not_addlink_return(self):
            s, ap, link = SimpleNamespace(name="s1"), SimpleNamespace(name="ap1"), object()
            a = SimpleNamespace(node=s, name="s1-eth2", link=link)
            b = SimpleNamespace(node=ap, name="ap1-eth1", link=link)
            s.connectionsTo = lambda peer: [(a, b)]
            self.assertIs(verified_wire(s, ap), a)
            b.link = object()
            with self.assertRaises(RuntimeError):
                verified_wire(s, ap)

        def test_ambiguous_or_absent_cable_rejected(self):
            s, ap = SimpleNamespace(name="s1"), SimpleNamespace(name="ap1")
            for pairs in ([], [(1, 2), (3, 4)]):
                s.connectionsTo = lambda peer: pairs
                with self.assertRaises(RuntimeError):
                    verified_wire(s, ap)

        def test_csv_missing_is_blank_not_zero(self):
            with tempfile.TemporaryDirectory() as d:
                p = Path(d)/"data.csv"
                append_csv(p, ["a", "b"], {"a": None, "b": 0})
                rows = list(csv.DictReader(p.open()))
                self.assertEqual(rows[0], {"a": "", "b": "0"})

        def test_json_cannot_export_nan(self):
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError):
                    write_json(Path(d)/"x.json", {"rtt": float("nan")})

        def test_missing_rtt_preserved(self):
            r = example_record()
            r["measurements"]["ap1"]["rtt_avg_ms"] = None
            r["measurements"]["ap1"]["loss_pct"] = 100
            v = inputs_from_record(r)
            self.assertIsNone(v["current_latency"])
            self.assertEqual(v["current_loss"], 100)
            self.assertEqual(tuple(v), FEATURE_ORDER)

        def test_both_actions_valid_but_labels_still_blank(self):
            r = example_record()
            r["inputs"] = inputs_from_record(r)
            row = label_row(r)
            self.assertTrue(row["measurement_review_ready"])
            self.assertEqual(row["decision"], "")
            self.assertEqual(row["label_status"], "UNREVIEWED")

        def test_unusable_outcome_not_ready(self):
            r = example_record()
            r["inputs"] = inputs_from_record(r)
            r["outcomes"][0]["valid_for_review"] = False
            self.assertFalse(label_row(r)["measurement_review_ready"])
            self.assertEqual(label_row(r)["decision"], "")

        def test_input_export_fields(self):
            r = example_record()
            r["inputs"] = inputs_from_record(r)
            row = input_row(r, SimpleNamespace(decision="ASK_AI", status="CONFLICT"))
            self.assertEqual(set(row), set(INPUT_FIELDS))
            self.assertNotIn("decision", row)
            self.assertNotIn("HANDOVER_loss_pct", row)

        def test_invalid_values_cannot_be_review_ready(self):
            r = example_record()
            r["inputs"] = inputs_from_record(r)
            for bad in (None, float("nan"), float("inf"), -1, True):
                r["inputs"]["candidate_latency"] = bad
                self.assertFalse(label_row(r)["measurement_review_ready"])
            r["inputs"]["candidate_latency"] = 1
            r["inputs"]["candidate_loss"] = 100
            self.assertFalse(label_row(r)["measurement_review_ready"])

        def test_two_stay_replays_do_not_make_a_pair(self):
            r = example_record()
            r["inputs"] = inputs_from_record(r)
            r["outcomes"][1]["action"] = "STAY"
            self.assertFalse(label_row(r)["measurement_review_ready"])

        def test_no_wifi_setposition_or_roaming_during_model_history(self):
            station = SimpleNamespace(position=[35, 40, 0])
            station.setPosition = lambda x: self.fail("must not trigger auto association")
            clock = SimpleNamespace(t=0.)
            def sleep(seconds):
                clock.t += seconds
            def snap(sta, aps):
                return {"monotonic_s": clock.t, "position": list(sta.position),
                        "rssi_dbm": {"ap1": -60.0, "ap2": -70.0}}
            with patch.object(time, "monotonic", side_effect=lambda: clock.t), \
                 patch.object(time, "sleep", side_effect=sleep), \
                 patch.dict(globals(), model_snapshot=snap):
                history = model_history(station, (), 62, 65, Settings())
            self.assertEqual(len(history), 5)
            self.assertEqual(station.position, [65, 40, 0])
            self.assertEqual(history[-1]["monotonic_s"], 2.)

        def test_netem_failure_stops_collection(self):
            interface = SimpleNamespace(name="s1-eth2")
            switch, ap = SimpleNamespace(name="s1"), SimpleNamespace(name="ap1")
            with patch.dict(globals(), verified_wire=lambda s, a: interface):
                with self.assertRaises(RuntimeError):
                    configure_paths(switch, [ap], {"ap1": {"delay_ms": 1, "loss_pct": 0}},
                                    lambda *a, **k: ("error", 1))

        def test_netem_conditions_not_measurements(self):
            interface = SimpleNamespace(name="s1-eth2")
            switch, ap = SimpleNamespace(name="s1"), SimpleNamespace(name="ap1")
            calls=[]
            def run(node, argv):
                calls.append(argv)
                return ("qdisc netem delay 20ms loss 8%", 0)
            with patch.dict(globals(), verified_wire=lambda s, a: interface):
                result = configure_paths(switch, [ap], {"ap1": {"delay_ms": 20, "loss_pct": 8}}, run)
            self.assertIn("random", calls[0])
            self.assertEqual(result[0]["configured_loss_pct"], 8)
            self.assertNotIn("current_loss", result[0])

        def test_real_probe_parser_and_ap_verification(self):
            import hosn_switch as helper
            station = SimpleNamespace(wintfs={0: SimpleNamespace(name="sta1-wlan0")})
            ap = SimpleNamespace(name="ap1", wintfs={0: SimpleNamespace(mac="02:00:00:00:01:01")})
            text = ("100 packets transmitted, 90 received, 10% packet loss\n"
                    "rtt min/avg/max/mdev = 1.0/2.5/5.0/1.0 ms\n")
            run = lambda *a, **k: (text, 0)
            with tempfile.TemporaryDirectory() as d, \
                 patch.object(helper, "actual_link", return_value=(ap.wintfs[0].mac, "")):
                result = probe(station, ap, run, Path(d)/"fixture.txt", 100, .05, helper)
            self.assertEqual(result["loss_pct"], 10.)
            self.assertEqual(result["rtt_avg_ms"], 2.5)
            with tempfile.TemporaryDirectory() as d, \
                 patch.object(helper, "actual_link", side_effect=[(ap.wintfs[0].mac, ""), ("wrong", "")]):
                with self.assertRaises(RuntimeError):
                    probe(station, ap, run, Path(d)/"fixture.txt", 100, .05, helper)

        def test_no_reply_probe_rtt_is_missing(self):
            import hosn_switch as helper
            station = SimpleNamespace(wintfs={0: SimpleNamespace(name="sta1-wlan0")})
            ap = SimpleNamespace(name="ap1", wintfs={0: SimpleNamespace(mac="02:00:00:00:01:01")})
            run = lambda *a, **k: ("100 packets transmitted, 0 received, 100% packet loss\n", 1)
            with tempfile.TemporaryDirectory() as d, \
                 patch.object(helper, "actual_link", return_value=(ap.wintfs[0].mac, "")):
                result = probe(station, ap, run, Path(d)/"fixture.txt", 100, .05, helper)
            self.assertEqual(result["loss_pct"], 100.)
            self.assertIsNone(result["rtt_avg_ms"])

        def test_action_replay_orders_switch_after_probe_start(self):
            import hosn_switch as helper
            events=[]
            ap1 = SimpleNamespace(name="ap1", wintfs={0: SimpleNamespace(mac="02:00:00:00:01:01")})
            ap2 = SimpleNamespace(name="ap2", wintfs={0: SimpleNamespace(mac="02:00:00:00:02:01")})
            current=SimpleNamespace(bssid=ap1.wintfs[0].mac)
            class Process:
                returncode=None
                def poll(self):
                    return self.returncode
                def wait(self, timeout):
                    self.returncode=0
                    events.append("ping_finished")
                    return 0
                def kill(self):
                    self.returncode=-9
            def popen(argv, stdout, **kwargs):
                events.append("ping_started")
                stdout.write("100 packets transmitted, 95 received, 5% packet loss\n"
                             "rtt min/avg/max/mdev = 1.0/2.0/3.0/0.1 ms\n")
                stdout.flush()
                return Process()
            station=SimpleNamespace(wintfs={0: SimpleNamespace(name="sta1-wlan0")}, popen=popen)
            def connect(sta, ap, *args, **kwargs):
                events.append("connect_"+ap.name)
                current.bssid=ap.wintfs[0].mac
                return {"request_to_observation_s": .8, "association_verified": True}
            with tempfile.TemporaryDirectory() as d, \
                 patch.object(helper, "connect_verified", side_effect=connect), \
                 patch.object(helper, "actual_link", side_effect=lambda *a: (current.bssid,"")), \
                 patch.object(time, "sleep", return_value=None), \
                 patch.dict(globals(), configure_paths=lambda *a: [], probe=lambda *a, **k: {"received":3}):
                c=next(c for c in make_plan(pilot=True) if c["current_ap"]=="ap1")
                result=action_replay(station, (ap1,ap2), None, c, "HANDOVER",
                                     SimpleNamespace(log_path=None), Path(d), Settings(), helper)
            self.assertEqual(events, ["connect_ap1", "ping_started", "connect_ap2", "ping_finished"])
            self.assertTrue(result["valid_for_review"])
            self.assertEqual(result["loss_pct"], 5.)
            self.assertIsNone(result["label"])

        def test_handoff_warns_about_labels_and_leakage(self):
            for phrase in ("blank", "only the eight", "counterfactuals", "pilot",
                           "do not label", "this script does not train"):
                self.assertIn(phrase, READ_ME.lower())

    def example_record():
        c = make_plan(pilot=True)[0]
        h = [{"monotonic_s": 1, "rssi_dbm": {"ap1": -77, "ap2": -67}},
             {"monotonic_s": 3, "rssi_dbm": {"ap1": -78, "ap2": -65}}]
        m = {a: {"rtt_avg_ms": 12 if a == "ap1" else 2, "loss_pct": 0,
                 "sent": 100, "received": 100, "start_monotonic_s": 4,
                 "end_monotonic_s": 9} for a in ("ap1", "ap2")}
        return {"sample_id": "SOFTWARE_FIXTURE_NOT_DATA", "setting": c,
                "data_kind": "software_test", "history": h, "measurements": m,
                "outcomes": [{"action": "STAY", "valid_for_review": True},
                             {"action": "HANDOVER", "valid_for_review": True}]}

    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if result.wasSuccessful():
        print("PASS: {} collector software tests. No network run; no training data generated.".format(
              result.testsRun))
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HOSN mobility-aware paired measurements; AI handled separately."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true",
                      help="Four mobility diagnostic samples, not training data.")
    mode.add_argument("--collect", action="store_true",
                      help="128-sample measured batch by default; labels left for review.")
    mode.add_argument("--self-test", action="store_true", help="Local software checks only, no sudo needed.")
    mode.add_argument("--plan", action="store_true", help="Show full collection plan without running a network.")
    parser.add_argument("--repeats", type=int, default=2, help="Full batch repeats per scenario (default 2).")
    parser.add_argument("--seed", type=int, default=20261002, help="Order seed only, not a measurement generator.")
    args = parser.parse_args()
    if args.self_test:
        return run_self_tests()
    if not 1 <= args.repeats <= 20:
        parser.error("Use --repeats from 1 through 20.")
    plan = make_plan(args.repeats, args.seed, args.pilot)
    if args.plan:
        print(json.dumps({"samples": len(plan), "group_splits": dict(Counter(split_map().values())),
                          "labels_assigned": 0, "settings": asdict(Settings()),
                          "cases": plan}, indent=2))
        return 0
    if sys.platform != "linux" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        parser.exit(2, "Run this in Ubuntu with sudo. Start with: "
                      "sudo python3 hosn_collect_mobility.py --pilot\n")
    missing = [n for n in ("ip", "iw", "ping", "tc") if not shutil.which(n)]
    if missing:
        parser.exit(2, "Missing system tools: " + ", ".join(missing) + "\n")
    root = Path(__file__).resolve().parent
    for name in ("hosn_switch.py", "hosn_rules.py"):
        if not (root/name).is_file():
            parser.exit(2, "Keep hosn_collect_mobility.py in your Hosn folder beside "
                          + name + "\n")
    try:
        return collect(root, Settings(), plan, args.pilot, args.seed)
    except Exception as exc:
        print("Cannot start collection: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
