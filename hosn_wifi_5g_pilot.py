#!/usr/bin/env python3
"""HOSN Wi-Fi -> emulated-5G video-call path validation.

Run beside hosn_switch.py in the Ubuntu Hosn repository:

    python3 hosn_wifi_5g_pilot.py --self-test
    python3 hosn_wifi_5g_pilot.py --plan
    sudo python3 hosn_wifi_5g_pilot.py --run

This is a SHORT VALIDATION, not a dataset collector and not an AI test.  One
station sends one continuous, sequence-numbered UDP media-like stream.  The
stream starts on a verified Wi-Fi association and then changes to a separate
IP path attached to an emulated cellular gateway.  The sequence counter and
receiver remain continuous across the path change.

The second path models mobile/5G access using a dedicated interface, gateway,
subnet, bandwidth, delay, and loss profile.  It does NOT implement or claim a
3GPP 5G radio, gNB, RAN, core, SIM, or physical RF measurements.  The honest
project description is "Wi-Fi to emulated-5G access handover".  Software
simulation/emulation is explicitly allowed by the ATP/EDGE challenge.

The script never loads/trains a model, assigns labels, appends to a CSV, or
changes an existing HOSN source/result.  It writes a new validation directory
under results/ and preserves raw packet arrivals and command evidence.
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
import tempfile
import time
import traceback
from typing import Any, Optional


os.environ.setdefault("MPLBACKEND", "Agg")

REVISION = "wifi-emulated-5g-video-pilot-v1"
SSID = "hosn-wifi"
WIFI_UE_IP = "10.10.0.2"
WIFI_SERVER_IP = "10.10.0.100"
CELL_UE_IP = "10.20.0.2"
CELL_SERVER_IP = "10.20.0.100"
UDP_PORT = 5500
TRAFFIC_DURATION_S = 16.0
PACKETS_PER_SECOND = 100.0
PAYLOAD_BYTES = 1200

# Conditions are explicit emulation settings, not measured observations.
ACCESS_PROFILES = {
    "wifi_near": {"delay_ms": 2.0, "loss_pct": 0.0, "rate_mbit": 40.0},
    "wifi_moving": {"delay_ms": 12.0, "loss_pct": 1.0, "rate_mbit": 15.0},
    "wifi_edge": {"delay_ms": 35.0, "loss_pct": 8.0, "rate_mbit": 3.0},
    "emulated_5g": {"delay_ms": 15.0, "loss_pct": 0.5, "rate_mbit": 25.0},
}

PLAN = {
    "revision": REVISION,
    "purpose": "validate one continuous media-like stream across two access technologies",
    "application": "paced UDP video-call-like traffic (not a video codec or MOS test)",
    "initial_access": "Wi-Fi",
    "target_access": "emulated mobile/5G IP access",
    "real_3gpp_stack": False,
    "training_data": False,
    "labels_assigned": 0,
    "duration_s": TRAFFIC_DURATION_S,
    "packets_per_second": PACKETS_PER_SECOND,
    "payload_bytes": PAYLOAD_BYTES,
    "movement": [
        {"at_s": 0.0, "x_m": 20.0, "wifi_profile": "wifi_near"},
        {"at_s": 4.0, "x_m": 40.0, "wifi_profile": "wifi_moving"},
        {"at_s": 7.0, "x_m": 65.0, "wifi_profile": "wifi_edge"},
    ],
    "path_switch_at_s": 9.0,
    "switch_trigger": "scheduled validation action, not a HOSN controller decision",
    "success_requires": [
        "verified Wi-Fi association",
        "distinct Wi-Fi and emulated-5G interfaces/subnets",
        "packets received from both access paths",
        "continuous sequence numbering",
        "verified first emulated-5G packet after the controller request",
        "at least 95 percent of sent packets received",
        "handover packet gap no greater than 500 ms",
    ],
}


SENDER_CODE = r"""
import json, os, socket, sys, time, traceback
wifi_src, wifi_if, wifi_dst, cell_src, cell_if, cell_dst, port, start_at, stop_at, interval, payload_size, control, summary = sys.argv[1:]
port = int(port); start_at = float(start_at); stop_at = float(stop_at)
interval = float(interval); payload_size = int(payload_size)
result = {"status": "incomplete", "sent": 0, "send_errors": 0, "sent_by_path": {"wifi": 0, "5g": 0}}
sockets = {}
try:
    for path, source, interface in (("wifi", wifi_src, wifi_if), ("5g", cell_src, cell_if)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, (interface + "\0").encode("ascii"))
        sock.bind((source, 0))
        sockets[path] = sock
    destinations = {"wifi": (wifi_dst, port), "5g": (cell_dst, port)}
    time.sleep(max(0.0, start_at - time.monotonic()))
    sequence = 0
    while True:
        scheduled = start_at + sequence * interval
        if scheduled >= stop_at:
            break
        time.sleep(max(0.0, scheduled - time.monotonic()))
        try:
            with open(control, encoding="utf-8") as handle:
                path = handle.read().strip()
        except OSError:
            path = "invalid"
        if path not in sockets:
            result["send_errors"] += 1
            sequence += 1
            continue
        sent_ns = time.monotonic_ns()
        header = (str(sequence) + "|" + str(sent_ns) + "|" + path + "|").encode("ascii")
        payload = (header + b"v" * max(0, payload_size - len(header)))[:payload_size]
        try:
            sockets[path].sendto(payload, destinations[path])
            result["sent"] += 1
            result["sent_by_path"][path] += 1
        except OSError:
            result["send_errors"] += 1
        sequence += 1
    result["scheduled_sequences"] = sequence
    result["status"] = "complete"
except BaseException as exc:
    result["status"] = "failed"
    result["error"] = type(exc).__name__ + ": " + str(exc)
    result["traceback"] = traceback.format_exc()
finally:
    for sock in sockets.values():
        sock.close()
    with open(summary, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
"""


RECEIVER_CODE = r"""
import json, socket, sys, time, traceback
port, stop_at, packets_path, summary_path = sys.argv[1:]
port = int(port); stop_at = float(stop_at)
result = {"status": "incomplete", "received": 0, "parse_errors": 0}
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
sock.settimeout(0.2)
try:
    sock.bind(("0.0.0.0", port))
    with open(packets_path, "w", encoding="utf-8") as output:
        while time.monotonic() < stop_at:
            try:
                payload, source = sock.recvfrom(65535)
            except socket.timeout:
                continue
            received_ns = time.monotonic_ns()
            try:
                sequence_raw, sent_raw, path_raw, _ = payload.split(b"|", 3)
                event = {
                    "sequence": int(sequence_raw),
                    "sent_monotonic_ns": int(sent_raw),
                    "received_monotonic_ns": received_ns,
                    "declared_path": path_raw.decode("ascii"),
                    "source_ip": source[0],
                    "source_port": source[1],
                    "payload_bytes": len(payload),
                }
                output.write(json.dumps(event, allow_nan=False) + "\n")
                output.flush()
                result["received"] += 1
            except (ValueError, UnicodeError):
                result["parse_errors"] += 1
    result["status"] = "complete"
except BaseException as exc:
    result["status"] = "failed"
    result["error"] = type(exc).__name__ + ": " + str(exc)
    result["traceback"] = traceback.format_exc()
finally:
    sock.close()
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
"""


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_events(path: Path, tolerate_partial_final_line: bool = False) -> list[dict]:
    if not path.exists():
        return []
    events = []
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()
        for index, line in enumerate(lines):
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    # The live verifier can observe the receiver's final line
                    # between write() and flush(). Completed-file reads remain
                    # strict and reject malformed evidence.
                    if tolerate_partial_final_line and index == len(lines) - 1:
                        continue
                    raise
    return events


def profile_tc_args(interface: str, profile: dict) -> list[str]:
    for key in ("delay_ms", "loss_pct", "rate_mbit"):
        value = profile.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError("Invalid access-profile value: " + key)
    if profile["rate_mbit"] <= 0 or profile["loss_pct"] > 100:
        raise ValueError("Invalid access-profile range.")
    return [
        "tc", "qdisc", "replace", "dev", interface, "root", "netem",
        "delay", "{}ms".format(profile["delay_ms"]),
        "loss", "{}%".format(profile["loss_pct"]),
        "rate", "{}mbit".format(profile["rate_mbit"]),
    ]


def configure_profile(node, interface: str, profile_name: str, run) -> dict:
    if profile_name not in ACCESS_PROFILES:
        raise ValueError("Unknown access profile: " + profile_name)
    argv = profile_tc_args(interface, ACCESS_PROFILES[profile_name])
    text, code = run(node, argv)
    if code:
        raise RuntimeError("Could not configure {}: {}".format(interface, text.strip()))
    shown, code = run(node, ["tc", "qdisc", "show", "dev", interface])
    if code or "netem" not in shown:
        raise RuntimeError("The access profile was not verified on " + interface)
    return {
        "profile": profile_name,
        "interface": interface,
        "condition_settings_not_measurements": ACCESS_PROFILES[profile_name],
        "tc_show": shown.strip(),
    }


def ping_path(node, interface: str, destination: str, run, raw_path: Path,
              helper, count: int = 10) -> dict:
    text, code = run(
        node,
        ["ping", "-n", "-I", interface, "-c", str(count), "-i", "0.1", "-W", "1", destination],
        timeout=count * 1.1 + 5,
    )
    raw_path.write_text(text, encoding="utf-8")
    if code not in (0, 1):
        raise RuntimeError("Path probe failed; inspect " + raw_path.name)
    result = helper.parse_ping(text)
    result.update(interface=interface, destination=destination, requested=count,
                  complete=result["sent"] == count)
    if not result["complete"]:
        raise RuntimeError("Path probe did not transmit its fixed request count.")
    return result


def modeled_wifi_rssi(station, access_point) -> float:
    value = float(station.wintfs[0].get_rssi(
        access_point.wintfs[0], station.get_distance_to(access_point)
    ))
    if not math.isfinite(value) or value >= 0:
        raise RuntimeError("Invalid modeled Wi-Fi RSSI.")
    return round(value, 3)


def percent(value: int, total: int) -> float:
    return 0.0 if total == 0 else 100.0 * value / total


def summarize(sender: dict, receiver: dict, events: list[dict],
              switch_request_ns: int) -> dict:
    if sender.get("status") != "complete" or receiver.get("status") != "complete":
        raise ValueError("Sender and receiver must complete before summarizing.")
    ordered = sorted(events, key=lambda item: item["received_monotonic_ns"])
    unique_by_sequence = {}
    for event in ordered:
        unique_by_sequence.setdefault(event["sequence"], event)
    unique = sorted(unique_by_sequence.values(), key=lambda item: item["received_monotonic_ns"])
    attempted = int(sender.get("scheduled_sequences", sender["sent"]))
    transmitted = int(sender["sent"])
    received = len(unique)
    received_by_path = Counter(event["declared_path"] for event in unique)
    sources_by_path = {
        path: sorted({event["source_ip"] for event in unique if event["declared_path"] == path})
        for path in ("wifi", "5g")
    }
    gaps_ms = [
        (right["received_monotonic_ns"] - left["received_monotonic_ns"]) / 1e6
        for left, right in zip(unique, unique[1:])
    ]
    first_5g = next((event for event in unique
                     if event["declared_path"] == "5g"
                     and event["received_monotonic_ns"] >= switch_request_ns), None)
    last_wifi = next((event for event in reversed(unique)
                      if event["declared_path"] == "wifi"
                      and (first_5g is None
                           or event["received_monotonic_ns"] < first_5g["received_monotonic_ns"])), None)
    transition_ms: Optional[float] = None
    handover_gap_ms: Optional[float] = None
    if first_5g is not None:
        transition_ms = (first_5g["received_monotonic_ns"] - switch_request_ns) / 1e6
    if first_5g is not None and last_wifi is not None:
        handover_gap_ms = (
            first_5g["received_monotonic_ns"] - last_wifi["received_monotonic_ns"]
        ) / 1e6
    delay_ms = [
        (event["received_monotonic_ns"] - event["sent_monotonic_ns"]) / 1e6
        for event in unique
    ]
    loss_pct = 100.0 * max(0, attempted - received) / attempted if attempted else 100.0
    wifi_sequences = [event["sequence"] for event in unique if event["declared_path"] == "wifi"]
    cell_sequences = [event["sequence"] for event in unique if event["declared_path"] == "5g"]
    checks = {
        "sender_sent_both_paths": all(sender["sent_by_path"].get(path, 0) > 0 for path in ("wifi", "5g")),
        "receiver_saw_both_paths": all(received_by_path[path] > 0 for path in ("wifi", "5g")),
        "wifi_source_verified": sources_by_path["wifi"] == [WIFI_UE_IP],
        "cellular_source_verified": sources_by_path["5g"] == [CELL_UE_IP],
        "first_5g_after_request_verified": first_5g is not None,
        "sequence_did_not_reset_at_switch": bool(
            wifi_sequences and cell_sequences and min(cell_sequences) > max(wifi_sequences)
        ),
        "receive_ratio_at_least_95pct": received >= math.ceil(attempted * 0.95),
        "handover_gap_at_most_500ms": handover_gap_ms is not None and handover_gap_ms <= 500.0,
        "no_parse_errors": receiver.get("parse_errors") == 0,
        "no_sender_errors": sender.get("send_errors") == 0,
    }
    return {
        "attempted_packets": attempted,
        "successfully_transmitted": transmitted,
        "sent": attempted,
        "unique_received": received,
        "duplicate_datagrams": len(events) - received,
        "loss_pct": loss_pct,
        "sent_by_path": sender["sent_by_path"],
        "received_by_path": dict(received_by_path),
        "sources_by_path": sources_by_path,
        "max_interarrival_gap_ms": max(gaps_ms) if gaps_ms else None,
        "mean_one_way_process_delay_ms": statistics.fmean(delay_ms) if delay_ms else None,
        "one_way_delay_jitter_stddev_ms": statistics.pstdev(delay_ms) if delay_ms else None,
        "application_goodput_mbps": (
            sum(int(event.get("payload_bytes", PAYLOAD_BYTES)) for event in unique) * 8
            / TRAFFIC_DURATION_S / 1_000_000
        ),
        "path_switch_request_to_first_5g_packet_ms": transition_ms,
        "handover_packet_gap_ms": handover_gap_ms,
        "checks": checks,
        "validation_pass": all(checks.values()),
    }


def verified_interface(node, peer, expected_name: str):
    pairs = node.connectionsTo(peer)
    if len(pairs) != 1:
        raise RuntimeError("Expected one link between {} and {}.".format(node.name, peer.name))
    local, remote = pairs[0]
    if (local.node is not node or remote.node is not peer
            or local.link is None or local.link is not remote.link
            or local.name != expected_name):
        raise RuntimeError("Could not verify experiment interface " + expected_name)
    return local


def configure_ip(node, interface: str, address: str, run) -> None:
    for argv in (
        ["ip", "addr", "flush", "dev", interface],
        ["ip", "addr", "add", address, "dev", interface],
        ["ip", "link", "set", "dev", interface, "up"],
    ):
        text, code = run(node, argv)
        if code:
            raise RuntimeError("IP configuration failed on {}: {}".format(interface, text.strip()))


def route_evidence(node, source: str, destination: str, expected_interface: str, run) -> dict:
    text, code = run(node, ["ip", "route", "get", destination, "from", source])
    if code or (" dev " + expected_interface + " ") not in (" " + text.replace("\n", " ") + " "):
        raise RuntimeError("Route to {} did not use {}: {}".format(
            destination, expected_interface, text.strip()
        ))
    return {"source": source, "destination": destination,
            "expected_interface": expected_interface, "ip_route_get": text.strip()}


def process_wait(process, name: str, timeout: float = 10.0) -> None:
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)
        raise RuntimeError(name + " did not finish before its watchdog.")
    if code:
        raise RuntimeError("{} exited with code {}.".format(name, code))


def sleep_until(moment: float) -> None:
    time.sleep(max(0.0, moment - time.monotonic()))


def create_network():
    from mininet.link import Link
    from mininet.log import setLogLevel
    from mininet.node import OVSBridge
    from mn_wifi.net import Mininet_wifi
    from mn_wifi.node import OVSBridgeAP

    setLogLevel("info")
    net = Mininet_wifi(
        controller=None,
        switch=OVSBridge,
        accessPoint=OVSBridgeAP,
        autoAssociation=False,
        allAutoAssociation=False,
    )
    try:
        sta = net.addStation(
            "sta1", ip=WIFI_UE_IP + "/24", mac="02:00:00:00:00:01",
            position="20,40,0", range=60,
        )
        server = net.addHost("h1", ip=None, mac="02:00:00:00:00:64")
        wifi_core = net.addSwitch("s1", dpid="0000000000000010")
        cell_gateway = net.addSwitch("g5", dpid="0000000000000050")
        ap = net.addAccessPoint(
            "ap1", ssid=SSID, mode="g", channel="1",
            mac="02:00:00:00:01:01", position="20,50,0", range=60,
        )
        net.setPropagationModel(model="logDistance", exp=3.5)
        configure = getattr(net, "configureNodes", None)
        (configure or net.configureWifiNodes)()

        net.addLink(ap, wifi_core, cls=Link, intfName1="ap1-backhaul", intfName2="s1-ap1")
        net.addLink(server, wifi_core, cls=Link, intfName1="h1-wifi0", intfName2="s1-server")
        net.addLink(sta, cell_gateway, cls=Link, intfName1="sta1-5g0", intfName2="g5-ue0")
        net.addLink(server, cell_gateway, cls=Link, intfName1="h1-5g0", intfName2="g5-server")
        net.build()
        for bridge in (wifi_core, cell_gateway, ap):
            bridge.start([])
        time.sleep(1.0)
        return net, sta, server, wifi_core, cell_gateway, ap
    except BaseException:
        net.stop()
        raise


def run_validation(root: Path) -> int:
    import hosn_switch as helper
    import mn_wifi.net as wifi_module

    for name in ("s1", "g5", "ap1"):
        existing = subprocess.run(
            ["ip", "link", "show", "dev", name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4,
        )
        if existing.returncode == 0:
            raise RuntimeError("Another experiment may still be running ({}).".format(name))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = root / "results" / ("wifi_5g_pilot_" + stamp)
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "plan.json", PLAN)
    manifest = {
        "revision": REVISION,
        "status": "starting",
        "created_utc": stamp,
        "data_kind": "validation_only_not_training_data",
        "application": "paced UDP media-like stream; not encoded video or MOS",
        "access_technologies": ["Wi-Fi", "emulated mobile/5G IP access"],
        "real_3gpp_stack": False,
        "not_present": ["gNB", "5G NR radio", "5G core", "SIM", "physical RF"],
        "cellular_model": ACCESS_PROFILES["emulated_5g"],
        "labels_assigned": 0,
        "model_loaded": False,
        "python_version": platform.python_version(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "mininet_wifi_version": str(getattr(wifi_module, "VERSION", "unknown")),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("hosn_wifi_5g_pilot.py", "hosn_switch.py")
        },
    }
    write_json(folder / "manifest.json", manifest)

    network = None
    old_cwd = Path.cwd()
    result_code = 1
    sender_process = receiver_process = None
    sender_output = receiver_output = None
    try:
        os.chdir(folder)
        print("HOSN Wi-Fi -> emulated-5G video-call validation", flush=True)
        print("VALIDATION ONLY: no AI, labels, or dataset rows.", flush=True)
        print("The 5G path is an access-network model, not a real 3GPP stack.", flush=True)
        network, station, server, wifi_core, cell_gateway, ap = create_network()
        run = helper.NodeCommands(folder / "commands.jsonl")

        wifi_if = station.wintfs[0].name
        verified_interface(station, cell_gateway, "sta1-5g0")
        verified_interface(server, wifi_core, "h1-wifi0")
        verified_interface(server, cell_gateway, "h1-5g0")

        configure_ip(station, wifi_if, WIFI_UE_IP + "/24", run)
        configure_ip(station, "sta1-5g0", CELL_UE_IP + "/24", run)
        configure_ip(server, "h1-wifi0", WIFI_SERVER_IP + "/24", run)
        configure_ip(server, "h1-5g0", CELL_SERVER_IP + "/24", run)

        association = helper.connect_verified(station, ap, (ap,), run)
        routes = {
            "wifi": route_evidence(station, WIFI_UE_IP, WIFI_SERVER_IP, wifi_if, run),
            "emulated_5g": route_evidence(station, CELL_UE_IP, CELL_SERVER_IP, "sta1-5g0", run),
        }
        profiles = {
            "wifi_initial": configure_profile(station, wifi_if, "wifi_near", run),
            "emulated_5g": configure_profile(station, "sta1-5g0", "emulated_5g", run),
        }
        initial_probes = {
            "wifi": ping_path(station, wifi_if, WIFI_SERVER_IP, run,
                              folder / "wifi_initial_ping.txt", helper),
            "emulated_5g": ping_path(station, "sta1-5g0", CELL_SERVER_IP, run,
                                     folder / "cellular_initial_ping.txt", helper),
        }
        if initial_probes["wifi"]["received"] == 0 or initial_probes["emulated_5g"]["received"] == 0:
            raise RuntimeError("Both access paths must work before media starts.")

        sender_script = folder / "udp_path_sender.py"
        receiver_script = folder / "udp_dual_receiver.py"
        sender_script.write_text(SENDER_CODE, encoding="utf-8")
        receiver_script.write_text(RECEIVER_CODE, encoding="utf-8")
        control_path = folder / "selected_access.txt"
        control_path.write_text("wifi\n", encoding="utf-8")
        packets_path = folder / "packet_arrivals.jsonl"
        sender_summary_path = folder / "sender_summary.json"
        receiver_summary_path = folder / "receiver_summary.json"
        sender_output = (folder / "sender_output.txt").open("w", encoding="utf-8")
        receiver_output = (folder / "receiver_output.txt").open("w", encoding="utf-8")

        start_at = time.monotonic() + 1.5
        stop_at = start_at + TRAFFIC_DURATION_S
        receiver_process = server.popen(
            [sys.executable, str(receiver_script), str(UDP_PORT), str(stop_at + 1.0),
             str(packets_path), str(receiver_summary_path)],
            stdout=receiver_output, stderr=subprocess.STDOUT,
        )
        sender_process = station.popen(
            [sys.executable, str(sender_script),
             WIFI_UE_IP, wifi_if, WIFI_SERVER_IP,
             CELL_UE_IP, "sta1-5g0", CELL_SERVER_IP,
             str(UDP_PORT), str(start_at), str(stop_at), str(1.0 / PACKETS_PER_SECOND),
             str(PAYLOAD_BYTES), str(control_path), str(sender_summary_path)],
            stdout=sender_output, stderr=subprocess.STDOUT,
        )

        movement = []
        for entry in PLAN["movement"]:
            sleep_until(start_at + entry["at_s"])
            station.position = [entry["x_m"], 40.0, 0.0]
            profiles[entry["wifi_profile"]] = configure_profile(
                station, wifi_if, entry["wifi_profile"], run
            )
            event = {
                "elapsed_s": entry["at_s"],
                "x_m": entry["x_m"],
                "wifi_profile": entry["wifi_profile"],
                "wifi_rssi_model_dbm": modeled_wifi_rssi(station, ap),
                "actual_wifi_bssid": helper.actual_link(station, run)[0],
            }
            movement.append(event)
            print("  t={:.1f}s x={:.1f}m Wi-Fi RSSI(model)={} dBm profile={}".format(
                entry["at_s"], entry["x_m"], event["wifi_rssi_model_dbm"],
                entry["wifi_profile"]
            ), flush=True)

        sleep_until(start_at + PLAN["path_switch_at_s"])
        switch_request_ns = time.monotonic_ns()
        control_path.write_text("5g\n", encoding="utf-8")
        print("  Scheduled path-switch request: Wi-Fi -> emulated 5G", flush=True)

        deadline = time.monotonic() + 3.0
        transition_event = None
        while time.monotonic() < deadline:
            for event in read_events(packets_path, tolerate_partial_final_line=True):
                if (event["declared_path"] == "5g"
                        and event["source_ip"] == CELL_UE_IP
                        and event["received_monotonic_ns"] >= switch_request_ns):
                    transition_event = event
                    break
            if transition_event is not None:
                break
            time.sleep(0.05)
        if transition_event is None:
            raise RuntimeError("No packet was verified on the emulated-5G path after the request.")
        print("  First emulated-5G packet verified at server", flush=True)

        sleep_until(stop_at + 1.2)
        process_wait(sender_process, "media sender")
        process_wait(receiver_process, "media receiver")
        sender_output.close(); sender_output = None
        receiver_output.close(); receiver_output = None

        sender_summary = read_json(sender_summary_path)
        receiver_summary = read_json(receiver_summary_path)
        events = read_events(packets_path)
        summary = summarize(sender_summary, receiver_summary, events, switch_request_ns)
        summary.update(
            revision=REVISION,
            association=association,
            routes=routes,
            profiles=profiles,
            initial_probes=initial_probes,
            movement=movement,
            path_switch={
                "from": "Wi-Fi",
                "to": "emulated mobile/5G IP access",
                "controller_request_monotonic_ns": switch_request_ns,
                "trigger": "scheduled validation action, not a HOSN decision",
                "real_3gpp_handover": False,
            },
        )
        write_json(folder / "summary.json", summary)
        manifest["status"] = "validation_pass" if summary["validation_pass"] else "validation_failed"
        manifest["summary"] = {
            key: summary[key] for key in (
                "sent", "unique_received", "loss_pct", "handover_packet_gap_ms",
                "path_switch_request_to_first_5g_packet_ms", "validation_pass",
            )
        }
        write_json(folder / "manifest.json", manifest)

        print("\nWI-FI -> EMULATED-5G VIDEO PATH VALIDATION: {}".format(
            "PASS" if summary["validation_pass"] else "FAIL"
        ), flush=True)
        print("  UDP: {}/{} unique packets, {:.3f}% loss".format(
            summary["unique_received"], summary["sent"], summary["loss_pct"]
        ), flush=True)
        print("  Received via Wi-Fi: {} | via emulated 5G: {}".format(
            summary["received_by_path"].get("wifi", 0),
            summary["received_by_path"].get("5g", 0),
        ), flush=True)
        print("  Path-request to first 5G packet: {:.3f} ms".format(
            summary["path_switch_request_to_first_5g_packet_ms"]
        ), flush=True)
        print("  Packet gap across path switch: {:.3f} ms".format(
            summary["handover_packet_gap_ms"]
        ), flush=True)
        print("  Claim: emulated 5G access path; NOT a real 3GPP 5G stack.", flush=True)
        result_code = 0 if summary["validation_pass"] else 1
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        print("\nInterrupted. Validation evidence is preserved; no dataset was created.")
        result_code = 130
    except Exception as exc:
        manifest.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        (folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("\nVALIDATION STOPPED:", exc)
        result_code = 1
    finally:
        for process in (sender_process, receiver_process):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=3)
        for handle in (sender_output, receiver_output):
            if handle is not None:
                handle.close()
        try:
            if network is not None:
                print("\nStopping this simulated network...")
                network.stop()
        except Exception as exc:
            manifest["cleanup_error"] = repr(exc)
            result_code = 1
        finally:
            os.chdir(old_cwd)
            write_json(folder / "manifest.json", manifest)
            try:
                helper.restore_result_owner(folder)
            except OSError as exc:
                print("Could not restore result ownership:", exc)
            print("Saved validation files:", folder)
    return result_code


def run_self_tests() -> int:
    import unittest

    class Tests(unittest.TestCase):
        def test_plan_is_cross_technology_and_not_training(self):
            self.assertEqual(PLAN["initial_access"], "Wi-Fi")
            self.assertIn("5G", PLAN["target_access"])
            self.assertFalse(PLAN["real_3gpp_stack"])
            self.assertFalse(PLAN["training_data"])
            self.assertEqual(PLAN["labels_assigned"], 0)

        def test_profiles_valid(self):
            for profile in ACCESS_PROFILES.values():
                args = profile_tc_args("test0", profile)
                self.assertEqual(args[:6], ["tc", "qdisc", "replace", "dev", "test0", "root"])
                self.assertIn("netem", args)
            for bad in (-1, float("nan"), float("inf"), True):
                profile = dict(ACCESS_PROFILES["wifi_near"], delay_ms=bad)
                with self.assertRaises(ValueError):
                    profile_tc_args("test0", profile)

        def test_media_rate_is_video_call_like_but_not_labeled_video(self):
            bits_per_second = PACKETS_PER_SECOND * PAYLOAD_BYTES * 8
            self.assertGreaterEqual(bits_per_second, 500_000)
            self.assertLessEqual(bits_per_second, 5_000_000)
            self.assertIn("not a video codec", PLAN["application"])

        def test_summary_pass_fixture(self):
            switch_ns = 1_000_000_000
            sender = {"status": "complete", "sent": 100,
                      "sent_by_path": {"wifi": 50, "5g": 50},
                      "scheduled_sequences": 100, "send_errors": 0}
            receiver = {"status": "complete", "received": 99, "parse_errors": 0}
            events = []
            for sequence in range(99):
                path = "wifi" if sequence < 50 else "5g"
                received_ns = 600_000_000 + sequence * 10_000_000
                events.append({
                    "sequence": sequence,
                    "sent_monotonic_ns": received_ns - 2_000_000,
                    "received_monotonic_ns": received_ns,
                    "declared_path": path,
                    "source_ip": WIFI_UE_IP if path == "wifi" else CELL_UE_IP,
                })
            result = summarize(sender, receiver, events, switch_ns)
            self.assertTrue(result["validation_pass"])
            self.assertAlmostEqual(result["handover_packet_gap_ms"], 10.0)

        def test_wrong_cellular_source_fails(self):
            switch_ns = 1_000_000_000
            sender = {"status": "complete", "sent": 2,
                      "sent_by_path": {"wifi": 1, "5g": 1},
                      "scheduled_sequences": 2, "send_errors": 0}
            receiver = {"status": "complete", "received": 2, "parse_errors": 0}
            events = [
                {"sequence": 0, "sent_monotonic_ns": 900_000_000,
                 "received_monotonic_ns": 990_000_000, "declared_path": "wifi",
                 "source_ip": WIFI_UE_IP},
                {"sequence": 1, "sent_monotonic_ns": 1_000_000_000,
                 "received_monotonic_ns": 1_100_000_000, "declared_path": "5g",
                 "source_ip": WIFI_UE_IP},
            ]
            result = summarize(sender, receiver, events, switch_ns)
            self.assertFalse(result["checks"]["cellular_source_verified"])
            self.assertFalse(result["validation_pass"])

        def test_event_reader(self):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "events.jsonl"
                path.write_text('{"sequence": 1}\n{"sequence": 2}\n', encoding="utf-8")
                self.assertEqual([item["sequence"] for item in read_events(path)], [1, 2])

        def test_strict_json_rejects_nan(self):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    write_json(Path(directory) / "bad.json", {"x": float("nan")})

    result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    )
    if result.wasSuccessful():
        print("PASS: {} software checks. No network run; no data generated.".format(
            result.testsRun
        ))
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Wi-Fi to emulated-5G media path; not a dataset collector."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true", help="Software tests only; no sudo.")
    mode.add_argument("--plan", action="store_true", help="Print the validation plan; no network.")
    mode.add_argument("--run", action="store_true", help="Run the short Mininet-WiFi validation.")
    args = parser.parse_args()
    if args.self_test:
        return run_self_tests()
    if args.plan:
        print(json.dumps(PLAN, indent=2, allow_nan=False))
        return 0
    if sys.platform != "linux" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        parser.exit(2, "Run the validation in Ubuntu with sudo.\n")
    missing = [name for name in ("ip", "iw", "ping", "tc") if not shutil.which(name)]
    if missing:
        parser.exit(2, "Missing required tools: {}\n".format(", ".join(missing)))
    root = Path(__file__).resolve().parent
    if not (root / "hosn_switch.py").is_file():
        parser.exit(2, "Keep this file beside hosn_switch.py in the Hosn repository.\n")
    return run_validation(root)


if __name__ == "__main__":
    sys.exit(main())
