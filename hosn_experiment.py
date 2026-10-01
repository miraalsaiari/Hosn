#!/usr/bin/env python3
"""HOSN first measured integration check (not a trained-AI evaluation).

Ubuntu, inside the Hosn project, with other Mininet experiments stopped:
    sudo python3 hosn_experiment.py
    sudo python3 hosn_experiment.py --case stay
    sudo python3 hosn_experiment.py --case conflict

Reuses hosn_controller.py -> hosn_rules.py, and hosn_switch.py. No old model is
loaded, no thresholds are changed, and no old files or results are overwritten.

This is an OFFLINE, STATIC engineering check, not seamless live roaming:
- One radio briefly joins AP1 and AP2 to measure each path, then returns to AP1.
  Those sampling switches are NOT controller decisions or counted as handovers.
- Only AFTER sampling/restoring does the controller choose an action. Its answer
  is not forced to match the selected case. A mismatch is reported as a failure.
- RSSI is computed by Mininet-WiFi's log-distance model, not a physical scan.
  The default wireless backend is used, without wmediumd. Do NOT infer that
  these RSSI numbers caused the measured delay or loss.
- Linux netem adds a documented 20-ms delay on ONE simulated switch egress
  towards the chosen AP. RTT/loss are then measured with ping, never copied
  from that configured delay. This represents an AP-path impairment, not a
  measured RF effect or demonstrated congestion mechanism.
- Both APs remain in the configured range. Trend comes from two timestamped
  model observations at the same fixed position (normally zero).
- Sequential probes assume this controlled test is stationary. They are NOT a
  deployable way to know an unconnected AP's performance without switching.
- These few cases test wiring/decision execution, NOT AI accuracy or superiority.
  comparison.csv contains controller outputs, not ground-truth training labels.
- Twenty pings per AP are only a short check; they do not establish low loss
  rates accurately. Five post-action pings do not measure switching interruption.

Cases (settings, not fabricated observations):
  clear:    AP2 nearer; AP1-path +20 ms; expect RULES/HANDOVER if probes support it.
  stay:     AP1 nearer; AP2-path +20 ms; expect RULES/STAY if probes support it.
  conflict: AP2 nearer; AP2-path +20 ms; expect HOLD/AI_UNAVAILABLE, since the
            replacement AI is not yet supplied. HOLD is not a preference for AP1.

Primary references checked 2026-10-01:
https://mininet-wifi.github.io/advanced/  (disable automatic association)
https://mininet-wifi.github.io/commands/ (association commands)
https://wireless.docs.kernel.org/en/latest/en/users/documentation/iw.html
https://man7.org/linux/man-pages/man8/tc-netem.8.html
https://github.com/intrig-unicamp/mininet-wifi/blob/master/mn_wifi/net.py
https://github.com/mininet/mininet/blob/master/mininet/node.py
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

from hosn_controller import decide
from hosn_rules import AI_FEATURE_ORDER, DEFAULT_CONFIG
from hosn_switch import (
    NodeCommands, SERVER_IP, SSID, actual_link, connect_verified,
    ping_on_ap, restore_result_owner, wait_for_link,
)

SCRIPT_REVISION = "wired-link-lookup-v2"
AP_PROBES = 20
FINAL_PROBES = 5
MAX_OBSERVATION_AGE_S = 25.0  # Guard for this static diagnostic, not a live policy.
CASES = {
    "clear": {"position": "65,40,0", "delayed_ap": "ap1",
              "expected": ("HANDOVER", "RULES", "CLEAR")},
    "stay": {"position": "35,40,0", "delayed_ap": "ap2",
             "expected": ("STAY", "RULES", "CLEAR")},
    "conflict": {"position": "65,40,0", "delayed_ap": "ap2",
                 "expected": ("STAY", "HOLD", "AI_UNAVAILABLE")},
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def signal_snapshot(station, aps):
    """Two AP model values for the same timestamp/position, not Linux RSSI."""
    values = {}
    for ap in aps:
        distance = station.get_distance_to(ap)
        value = float(station.wintfs[0].get_rssi(ap.wintfs[0], distance))
        if not math.isfinite(value):
            raise ValueError("Invalid modeled RSSI for " + ap.name)
        values[ap.name] = value
    return {"monotonic_s": time.monotonic(), "rssi_model_dbm": values}


def comparison_inputs(first, last, probes):
    elapsed = last["monotonic_s"] - first["monotonic_s"]
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Signal observations need a positive time interval.")
    values = {}
    for prefix, name in (("current", "ap1"), ("candidate", "ap2")):
        probe = probes[name]
        if not probe["ap_verified_at_both_ends"]:
            raise ValueError("Probe could not be attributed to " + name)
        if probe["sent"] != AP_PROBES:
            raise ValueError("Incomplete probe window for " + name)
        # Preserve None when there were no replies. It is never zero delay.
        values[prefix + "_latency"] = probe["rtt_avg_ms"]
        values[prefix + "_loss"] = probe["loss_pct"]
        values[prefix + "_rssi"] = last["rssi_model_dbm"][name]
        values[prefix + "_trend"] = (
            last["rssi_model_dbm"][name] - first["rssi_model_dbm"][name]
        ) / elapsed
    return values


def execute_decision(station, aps, run, decision):
    """Only this step executes a HOSN choice; sampling is kept separate."""
    ap1, ap2 = aps
    current = str(ap1.wintfs[0].mac).lower()
    if decision.decision not in ("STAY", "HANDOVER"):
        raise ValueError("No executable final decision.")
    if decision.decision == "HANDOVER" and decision.source not in ("RULES", "AI"):
        raise ValueError("A HOLD result must not request a handover.")
    if actual_link(station, run)[0] != current:
        raise RuntimeError("Current AP changed after measurement; do not act on stale data.")
    if decision.decision == "HANDOVER":
        result = connect_verified(station, ap2, aps, run, expected_current=current)
        return ap2, result
    wait_for_link(station, current, run)
    return ap1, {"connection_changed": False, "association_verified": True,
                 "verified_bssid": current, "note": "No decision-time switch requested."}


def add_test_delay(switch, peer, run):
    """Find the real wired interface; do not rely on addLink's return value.

    Mininet-WiFi.addLink() can create a wired link but return None. Using
    that return value as a Link caused the earlier '.intf1' failure.
    connectionsTo() instead looks up the created interfaces, with the local
    switch interface first regardless of the link's creation order.
    """
    connections = switch.connectionsTo(peer)
    if len(connections) != 1:
        raise RuntimeError(
            "Expected one wired connection from {} to {}; found {}. "
            "No delay was applied.".format(switch.name, peer.name, len(connections)))
    interface, remote = connections[0]
    if (getattr(interface, "node", None) is not switch
            or getattr(remote, "node", None) is not peer
            or not str(getattr(interface, "name", "")).startswith(switch.name + "-eth")
            or getattr(interface, "link", None) is None
            or interface.link is not getattr(remote, "link", None)):
        raise RuntimeError("Refusing to change an unverified experiment interface.")
    text, code = run(switch, ["tc", "qdisc", "replace", "dev", interface.name,
                              "root", "netem", "delay", "20ms"])
    if code:
        raise RuntimeError("Could not set the controlled delay: " + text.strip())
    observed, code = run(switch, ["tc", "qdisc", "show", "dev", interface.name])
    if code or "netem" not in observed:
        raise RuntimeError("Could not verify the test-delay configuration.")
    return {"interface": interface.name, "configured_delay_ms": 20,
            "direction": "switch egress towards AP; one direction only",
            "actual_qdisc_report": observed}


def run_experiment(case_name):
    from mininet.link import Link
    from mininet.log import setLogLevel
    from mininet.node import OVSBridge
    from mn_wifi.net import Mininet_wifi
    from mn_wifi.node import OVSBridgeAP
    import mn_wifi.net as wifi_module

    setLogLevel("info")
    for name in ("s1", "ap1", "ap2"):
        check = subprocess.run(["ip", "link", "show", "dev", name],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=4)
        if check.returncode == 0:
            raise RuntimeError("Another experiment may be running. Exit it first: " + name)

    setting = CASES[case_name]
    root = Path(__file__).resolve().parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = root / "results" / ("integration_" + case_name + "_" + stamp)
    folder.mkdir(parents=True, exist_ok=False)
    report = {
        "case": case_name, "status": "incomplete", "created_utc": stamp,
        "script_revision": SCRIPT_REVISION,
        "purpose": "Static engineering integration check, NOT AI evaluation/training",
        "position": setting["position"], "settings": setting,
        "backend": "default wireless backend, no wmediumd",
        "rssi_source": "logDistance propagation model, exponent 3.5",
        "probe_method": "same radio, sequential AP1 then AP2, then restore AP1",
        "automatic_association": False, "ai_model_supplied": False,
        "rule_config": asdict(DEFAULT_CONFIG),
        "ground_truth_training_labels_provided": False,
        "python": platform.python_version(), "kernel": platform.release(),
        "mininet_wifi_version": str(getattr(wifi_module, "VERSION", "unknown")),
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ("hosn_experiment.py", "hosn_switch.py",
                                       "hosn_controller.py", "hosn_rules.py")},
        "sampling_connections": [], "observations": {},
    }
    old_cwd, network, exit_code = Path.cwd(), None, 1
    try:
        os.chdir(folder)
        print("HOSN integration check: " + SCRIPT_REVISION, flush=True)
        network = Mininet_wifi(controller=None, switch=OVSBridge,
                               accessPoint=OVSBridgeAP, autoAssociation=False,
                               allAutoAssociation=False)
        sta = network.addStation("sta1", ip="10.0.0.1/24", mac="02:00:00:00:00:01",
                                 position=setting["position"], range=50)
        host = network.addHost("h1", ip=SERVER_IP + "/24", mac="02:00:00:00:00:64")
        switch = network.addSwitch("s1", dpid="0000000000000010")
        ap1 = network.addAccessPoint("ap1", ssid=SSID, mode="g", channel="1",
                                    mac="02:00:00:00:01:01", position="20,50,0", range=50)
        ap2 = network.addAccessPoint("ap2", ssid=SSID, mode="g", channel="6",
                                    mac="02:00:00:00:02:01", position="80,50,0", range=50)
        aps = (ap1, ap2)
        network.setPropagationModel(model="logDistance", exp=3.5)
        configure = getattr(network, "configureNodes", None)
        (configure or network.configureWifiNodes)()
        # Mininet-WiFi may return None even when it creates the wired link.
        # Find its actual endpoints later through switch.connectionsTo(ap).
        for node in (host, ap1, ap2):
            network.addLink(switch, node, cls=Link)
        network.build()
        for bridge in (switch, ap1, ap2):
            bridge.start([])
        time.sleep(1)

        run = NodeCommands(folder / "setup_commands.jsonl")
        delayed_ap = next(ap for ap in aps if ap.name == setting["delayed_ap"])
        report["delay_setup"] = add_test_delay(switch, delayed_ap, run)
        print("\nHOSN MEASURE -> RULES/CONTROLLER -> VERIFIED ACTION", flush=True)
        print("Controlled case: {}. A 20-ms delay is applied towards {}.".format(
            case_name, setting["delayed_ap"]), flush=True)
        print("RTT/loss below come from ping, not from the configured delay.", flush=True)
        print("Sampling visits both APs first; those visits are NOT HOSN decisions.\n",
              flush=True)
        first = signal_snapshot(sta, aps)
        report["first_signal_observation"] = first
        probes, probe_ends = {}, {}
        for ap in aps:
            run.log_path = folder / ("sampling_" + ap.name + "_commands.jsonl")
            print("Measuring {} (temporary sampling connection)...".format(ap.name), flush=True)
            connection = connect_verified(sta, ap, aps, run)
            report["sampling_connections"].append(connection)
            warmup = ping_on_ap(sta, ap, run, folder / ("warmup_" + ap.name + ".txt"), count=1)
            if not warmup["ap_verified_at_both_ends"]:
                raise RuntimeError("AP changed during the warmup.")
            started = time.monotonic()
            measured = ping_on_ap(sta, ap, run, folder / ("measure_" + ap.name + ".txt"),
                                  count=AP_PROBES)
            ended = time.monotonic()
            measured.update(start_monotonic_s=started, end_monotonic_s=ended)
            probes[ap.name], probe_ends[ap.name] = measured, ended
            report["observations"][ap.name] = {"warmup": warmup, "measurement": measured}
            print("  {}: {} / {} replies; RTT {} ms; loss {:.1f}%".format(
                ap.name, measured["received"], measured["sent"],
                measured["rtt_avg_ms"], measured["loss_pct"]), flush=True)

        run.log_path = folder / "restore_current_ap_commands.jsonl"
        print("\nReturning to AP1 BEFORE asking the controller.", flush=True)
        report["restore_ap1"] = connect_verified(
            sta, ap1, aps, run, expected_current=str(ap2.wintfs[0].mac).lower())
        time.sleep(0.5)
        last = signal_snapshot(sta, aps)
        report["last_signal_observation"] = last
        values = comparison_inputs(first, last, probes)
        report["inputs"] = values
        age = time.monotonic() - min(probe_ends.values())
        report["oldest_probe_age_s_at_decision"] = age
        if age > MAX_OBSERVATION_AGE_S:
            raise RuntimeError("Probes are too old for this integration check; no action taken.")

        # There is deliberately no .pkl load or invented substitute AI here.
        decision = decide(**values, ai_model=None)
        report["controller"] = decision.to_dict()
        print("\nSignal model: AP1 {:.2f} dBm; AP2 {:.2f} dBm".format(
            values["current_rssi"], values["candidate_rssi"]), flush=True)
        print("CONTROLLER: {} | {} | {}".format(
            decision.decision, decision.source, decision.status), flush=True)
        print(decision.reason, flush=True)
        print("AI called: {}".format(decision.ai_called), flush=True)
        write_json(folder / "decision.json", report["controller"])
        with (folder / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = list(AI_FEATURE_ORDER) + ["controller_action", "source", "status",
                                              "record_purpose"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(dict(values, controller_action=decision.decision,
                                 source=decision.source, status=decision.status,
                                 record_purpose="engineering_check_not_training_label"))

        run.log_path = folder / "decision_action_commands.jsonl"
        selected, action = execute_decision(sta, aps, run, decision)
        report["decision_time_action"] = action
        report["final_ap"] = selected.name
        run.log_path = folder / "post_action_commands.jsonl"
        final_ping = ping_on_ap(sta, selected, run, folder / "post_action_ping.txt",
                                count=FINAL_PROBES)
        report["post_action_ping"] = final_ping
        print("Actual AP after decision: {}. Replies: {} / {}.".format(
            selected.name, final_ping["received"], final_ping["sent"]), flush=True)
        expected = (decision.decision, decision.source, decision.status) == setting["expected"]
        report["expected_case_route_observed"] = expected
        passed = (expected and not decision.ai_called
                  and final_ping["ap_verified_at_both_ends"]
                  and final_ping["all_requested_replies"])
        report["status"] = "passed" if passed else "not_passed"
        exit_code = 0 if passed else 1
        print("\n{}: measured inputs, controller route, actual AP and post-action pings checked."
              .format("PASS" if passed else "NOT PASSED"), flush=True)
        if not expected:
            print("The measured inputs did NOT produce the intended case route; inspect logs.")
        print("This is a controlled integration check, not AI performance or seamless roaming.")
    except KeyboardInterrupt:
        report.update(status="interrupted", error="Interrupted by user")
        exit_code = 130
        print("\nInterrupted; preserving partial results.")
    except Exception as exc:
        report.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        (folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("\nTEST NOT PASSED:", exc)
    finally:
        try:
            if network is not None:
                print("\nStopping this simulated network...")
                network.stop()
        except Exception as exc:
            report.update(status="cleanup_failed", cleanup_error=repr(exc))
            exit_code = 1
            print("Cleanup failed; share the output before starting another experiment.")
        finally:
            os.chdir(old_cwd)
            write_json(folder / "summary.json", report)
            try:
                restore_result_owner(folder)
            except OSError as exc:
                print("Could not restore result-file ownership:", exc)
            print("Saved results:", folder)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description="Measured HOSN integration check; no trained AI.")
    parser.add_argument("--case", choices=tuple(CASES), default="clear")
    args = parser.parse_args()
    if sys.platform != "linux" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        parser.exit(2, "Run inside Ubuntu: sudo python3 hosn_experiment.py\n")
    missing = [name for name in ("ip", "iw", "ping", "tc") if not shutil.which(name)]
    if missing:
        parser.exit(2, "Missing command(s): " + ", ".join(missing) + "\n")
    try:
        return run_experiment(args.case)
    except Exception as exc:
        print("Could not start experiment:", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
