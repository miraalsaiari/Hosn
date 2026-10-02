#!/usr/bin/env python3
"""HOSN richer-topology validation: three users, mobility, and live traffic.

Run beside hosn_switch.py in the Ubuntu Hosn repository:

    sudo python3 hosn_rich_topology.py

Safe local checks that do not start Mininet-WiFi:

    python3 hosn_rich_topology.py --self-test
    python3 hosn_rich_topology.py --plan

Purpose
-------
This is a SMALL VALIDATION experiment to run before redesigning or collecting
the full measured dataset.  It creates two Wi-Fi APs, a wired service host,
two mobile stations moving in opposite directions at different speeds, and a
third stationary station.  All three receive continuous constant-rate UDP
traffic while sta1 and sta2 perform scheduled, explicitly verified handovers.

This file deliberately does NOT:

* load or train an AI model;
* create training labels or append to hosn_data.csv;
* call the HOSN controller or claim that scheduled switches are decisions;
* claim to run a real video codec or measure human video QoE;
* claim LTE/5G or cross-technology handover;
* alter any existing HOSN source file or earlier result directory.

The traffic is only "video-call-like" in the limited sense that it is a
continuous, paced UDP media stream.  Linux ping runs concurrently as a simple
RTT/loss probe.  RSSI is calculated by Mininet-WiFi's propagation model, not
measured from physical radios.  Movement updates model geometry; it is not a
physical RF experiment.  The switch action is accepted only after Linux `iw`
reports the requested BSSID, using hosn_switch.connect_verified().

Results are written to a new results/rich_topology_<UTC timestamp>/ directory.
Raw packet arrivals, ping output, mobility history, handover events, manifests,
and summaries are retained so delay spikes and losses are not hidden.
"""

from __future__ import annotations

import argparse
import csv
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
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# No desktop display is required.  Set this before importing Mininet-WiFi.
os.environ.setdefault("MPLBACKEND", "Agg")

SERVER_IP = "10.0.0.100"
SSID = "hosn-wifi"
AP_RANGE_M = 60.0
PROPAGATION_EXPONENT = 3.5

# A short validation, not a dataset run.
TRAFFIC_DURATION_S = 32.0
UDP_PACKETS_PER_SECOND = 25.0
UDP_PAYLOAD_BYTES = 1000
PING_INTERVAL_S = 0.2
MOBILITY_SAMPLE_INTERVAL_S = 0.5

AP_SPECS = {
    "ap1": {"position": (20.0, 50.0, 0.0), "channel": "1",
            "mac": "02:00:00:00:01:01"},
    "ap2": {"position": (80.0, 50.0, 0.0), "channel": "6",
            "mac": "02:00:00:00:02:01"},
}

STATION_SPECS = {
    # Eighty metres in 30 seconds: about 2.67 m/s, left to right.
    "sta1": {
        "ip": "10.0.0.1", "mac": "02:00:00:00:00:01",
        "start": (10.0, 40.0, 0.0), "end": (90.0, 40.0, 0.0),
        "move_start_s": 0.5, "move_duration_s": 30.0,
        "initial_ap": "ap1", "target_ap": "ap2", "handover_s": 15.5,
        "udp_port": 5201,
    },
    # Eighty metres in 15 seconds: about 5.33 m/s, right to left.
    "sta2": {
        "ip": "10.0.0.2", "mac": "02:00:00:00:00:02",
        "start": (90.0, 60.0, 0.0), "end": (10.0, 60.0, 0.0),
        "move_start_s": 2.0, "move_duration_s": 15.0,
        "initial_ap": "ap2", "target_ap": "ap1", "handover_s": 9.5,
        "udp_port": 5202,
    },
    "sta3": {
        "ip": "10.0.0.3", "mac": "02:00:00:00:00:03",
        "start": (25.0, 35.0, 0.0), "end": (25.0, 35.0, 0.0),
        "move_start_s": 0.0, "move_duration_s": 0.0,
        "initial_ap": "ap1", "target_ap": None, "handover_s": None,
        "udp_port": 5203,
    },
}

MOBILITY_FIELDS = (
    "elapsed_s", "station", "x_m", "y_m", "z_m",
    "speed_mps", "direction", "ap1_rssi_model_dbm",
    "ap2_rssi_model_dbm", "last_verified_ap",
)


UDP_SENDER_CODE = r"""
import json, socket, sys, time, traceback
destination, port, start_at, stop_at, interval, payload_size, summary_path = sys.argv[1:]
port = int(port); start_at = float(start_at); stop_at = float(stop_at)
interval = float(interval); payload_size = int(payload_size)
summary = {"status": "incomplete", "sent": 0, "send_errors": 0}
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    time.sleep(max(0.0, start_at - time.monotonic()))
    sequence = 0
    scheduled = start_at
    while scheduled < stop_at:
        time.sleep(max(0.0, scheduled - time.monotonic()))
        sent_ns = time.monotonic_ns()
        header = (str(sequence) + "|" + str(sent_ns) + "|").encode("ascii")
        payload = (header + (b"x" * max(0, payload_size - len(header))))[:payload_size]
        try:
            sock.sendto(payload, (destination, port))
            summary["sent"] += 1
        except OSError:
            summary["send_errors"] += 1
        sequence += 1
        scheduled = start_at + sequence * interval
    summary["status"] = "complete"
except BaseException as exc:
    summary["status"] = "failed"
    summary["error"] = type(exc).__name__ + ": " + str(exc)
    summary["traceback"] = traceback.format_exc()
finally:
    summary["finished_monotonic_ns"] = time.monotonic_ns()
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    sock.close()
"""


UDP_RECEIVER_CODE = r"""
import json, socket, sys, time, traceback
bind_ip, port, stop_at, packet_path, summary_path = sys.argv[1:]
port = int(port); stop_at = float(stop_at)
summary = {"status": "incomplete", "datagrams_received": 0, "parse_errors": 0}
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
sock.settimeout(0.2)
try:
    sock.bind((bind_ip, port))
    with open(packet_path, "w", encoding="utf-8") as packets:
        while time.monotonic() < stop_at:
            try:
                payload, source = sock.recvfrom(65535)
            except socket.timeout:
                continue
            received_ns = time.monotonic_ns()
            try:
                sequence_raw, sent_raw, _ = payload.split(b"|", 2)
                event = {
                    "sequence": int(sequence_raw),
                    "sent_monotonic_ns": int(sent_raw),
                    "received_monotonic_ns": received_ns,
                    "source_ip": source[0],
                    "source_port": source[1],
                    "payload_bytes": len(payload),
                }
                packets.write(json.dumps(event, allow_nan=False) + "\n")
                packets.flush()
                summary["datagrams_received"] += 1
            except (ValueError, TypeError):
                summary["parse_errors"] += 1
    summary["status"] = "complete"
except BaseException as exc:
    summary["status"] = "failed"
    summary["error"] = type(exc).__name__ + ": " + str(exc)
    summary["traceback"] = traceback.format_exc()
finally:
    summary["finished_monotonic_ns"] = time.monotonic_ns()
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    sock.close()
"""


def write_json(path: Path, value: Any) -> None:
    """Write strict JSON; NaN/Infinity must never enter experiment records."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def interpolate_position(spec: dict, elapsed_s: float) -> Tuple[float, float, float]:
    """Return the scheduled position at elapsed_s, clamped to the path."""
    start = tuple(float(value) for value in spec["start"])
    end = tuple(float(value) for value in spec["end"])
    duration = float(spec["move_duration_s"])
    if duration <= 0:
        return start
    fraction = (float(elapsed_s) - float(spec["move_start_s"])) / duration
    fraction = min(1.0, max(0.0, fraction))
    return tuple(a + (b - a) * fraction for a, b in zip(start, end))


def path_speed(spec: dict) -> float:
    duration = float(spec["move_duration_s"])
    if duration <= 0:
        return 0.0
    distance = math.dist(tuple(spec["start"]), tuple(spec["end"]))
    return distance / duration


def path_direction(spec: dict) -> str:
    dx = float(spec["end"][0]) - float(spec["start"][0])
    dy = float(spec["end"][1]) - float(spec["start"][1])
    if dx == 0 and dy == 0:
        return "stationary"
    if abs(dx) >= abs(dy):
        return "left_to_right" if dx > 0 else "right_to_left"
    return "bottom_to_top" if dy > 0 else "top_to_bottom"


def modeled_rssi(station, access_point) -> float:
    distance = station.get_distance_to(access_point)
    value = float(station.wintfs[0].get_rssi(access_point.wintfs[0], distance))
    if not math.isfinite(value) or value >= 0:
        raise RuntimeError("Invalid modeled RSSI for " + access_point.name)
    return round(value, 3)


def percentile(values: Sequence[float], percent: float) -> Optional[float]:
    if not values:
        return None
    if not 0 <= percent <= 100:
        raise ValueError("Percentile must be from 0 through 100.")
    ordered = sorted(float(value) for value in values)
    rank = max(0, math.ceil((percent / 100.0) * len(ordered)) - 1)
    return ordered[rank]


def read_packet_events(path: Path) -> List[dict]:
    events = []
    if not path.exists():
        return events
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            for field in ("sequence", "sent_monotonic_ns", "received_monotonic_ns"):
                if not isinstance(value.get(field), int):
                    raise ValueError("Invalid {} on packet-log line {}".format(field, line_number))
            events.append(value)
    return events


def packet_gap_across_event(events: Sequence[dict], event_ns: Optional[int]) -> Optional[float]:
    if event_ns is None:
        return None
    arrivals = sorted({int(event["received_monotonic_ns"]) for event in events})
    before = [value for value in arrivals if value < event_ns]
    after = [value for value in arrivals if value >= event_ns]
    if not before or not after:
        return None
    return (after[0] - before[-1]) / 1_000_000.0


def summarize_udp(events: Sequence[dict], sender_summary: dict,
                  handover_started_ns: Optional[int]) -> dict:
    """Summarize raw arrivals without deleting spikes or duplicate packets."""
    sent = int(sender_summary.get("sent", 0))
    by_sequence: Dict[int, dict] = {}
    duplicates = 0
    for event in sorted(events, key=lambda item: item["received_monotonic_ns"]):
        sequence = int(event["sequence"])
        if sequence in by_sequence:
            duplicates += 1
            continue
        by_sequence[sequence] = event
    unique = list(by_sequence.values())
    delays_ms = [
        (event["received_monotonic_ns"] - event["sent_monotonic_ns"]) / 1_000_000.0
        for event in unique
    ]
    if any(value < 0 or not math.isfinite(value) for value in delays_ms):
        raise ValueError("One-way delay is invalid; the VM monotonic clock assumption failed.")
    ordered_by_arrival = sorted(unique, key=lambda item: item["received_monotonic_ns"])
    gaps_ms = [
        (b["received_monotonic_ns"] - a["received_monotonic_ns"]) / 1_000_000.0
        for a, b in zip(ordered_by_arrival, ordered_by_arrival[1:])
    ]
    ordered_by_sequence = sorted(unique, key=lambda item: item["sequence"])
    jitter = 0.0
    previous_transit = None
    for event in ordered_by_sequence:
        transit = (event["received_monotonic_ns"] - event["sent_monotonic_ns"]) / 1_000_000.0
        if previous_transit is not None:
            jitter += (abs(transit - previous_transit) - jitter) / 16.0
        previous_transit = transit
    received = len(unique)
    lost = max(0, sent - received)
    before_count = sum(
        event["received_monotonic_ns"] < handover_started_ns
        for event in unique
    ) if handover_started_ns is not None else None
    after_count = sum(
        event["received_monotonic_ns"] >= handover_started_ns
        for event in unique
    ) if handover_started_ns is not None else None
    handover_gap = packet_gap_across_event(unique, handover_started_ns)
    return {
        "sender_status": sender_summary.get("status"),
        "packets_sent": sent,
        "packets_received_unique": received,
        "duplicate_packets": duplicates,
        "packets_lost": lost,
        "loss_pct": round(100.0 * lost / sent, 3) if sent else None,
        "delay_avg_ms": round(statistics.fmean(delays_ms), 6) if delays_ms else None,
        "delay_p95_ms": round(percentile(delays_ms, 95), 6) if delays_ms else None,
        "delay_max_ms": round(max(delays_ms), 6) if delays_ms else None,
        "rfc3550_style_jitter_ms": round(jitter, 6) if delays_ms else None,
        "max_interarrival_gap_ms": round(max(gaps_ms), 6) if gaps_ms else None,
        "handover_packet_gap_ms": round(handover_gap, 6) if handover_gap is not None else None,
        "packets_received_before_handover": before_count,
        "packets_received_after_handover": after_count,
    }


def station_plan() -> dict:
    result = {}
    for name, spec in STATION_SPECS.items():
        result[name] = {
            "ip": spec["ip"],
            "start_m": list(spec["start"]),
            "end_m": list(spec["end"]),
            "speed_mps": round(path_speed(spec), 6),
            "direction": path_direction(spec),
            "initial_ap": spec["initial_ap"],
            "target_ap": spec["target_ap"],
            "scheduled_handover_s": spec["handover_s"],
            "udp_port": spec["udp_port"],
        }
    return result


def create_network():
    """Create only the topology; no traffic or decisions are started here."""
    from mininet.link import Link
    from mininet.log import setLogLevel
    from mininet.node import OVSBridge
    from mn_wifi.net import Mininet_wifi
    from mn_wifi.node import OVSBridgeAP

    setLogLevel("info")
    network = Mininet_wifi(
        controller=None,
        switch=OVSBridge,
        accessPoint=OVSBridgeAP,
        autoAssociation=False,
        allAutoAssociation=False,
    )
    try:
        stations = {}
        for name, spec in STATION_SPECS.items():
            stations[name] = network.addStation(
                name,
                ip=spec["ip"] + "/24",
                mac=spec["mac"],
                position=",".join(str(value) for value in spec["start"]),
                range=AP_RANGE_M,
            )
        server = network.addHost(
            "h1", ip=SERVER_IP + "/24", mac="02:00:00:00:00:64"
        )
        switch = network.addSwitch("s1", dpid="0000000000000010")
        aps = {}
        for name, spec in AP_SPECS.items():
            aps[name] = network.addAccessPoint(
                name,
                ssid=SSID,
                mode="g",
                channel=spec["channel"],
                mac=spec["mac"],
                position=",".join(str(value) for value in spec["position"]),
                range=AP_RANGE_M,
            )
        network.setPropagationModel(model="logDistance", exp=PROPAGATION_EXPONENT)
        configure = getattr(network, "configureNodes", None)
        (configure or network.configureWifiNodes)()
        for peer in (server, aps["ap1"], aps["ap2"]):
            network.addLink(switch, peer, cls=Link)
        network.build()
        for bridge in (switch, aps["ap1"], aps["ap2"]):
            bridge.start([])
        time.sleep(1.0)
        return network, stations, aps, server, switch
    except BaseException:
        network.stop()
        raise


def start_long_process(node, argv: Sequence[str]):
    return node.popen(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=dict(os.environ, LC_ALL="C", LANG="C"),
    )


def finish_process(process, name: str, raw_path: Path, timeout: float) -> None:
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate()
        raw_path.write_text(output, encoding="utf-8")
        raise RuntimeError(name + " exceeded its watchdog; raw output was saved.")
    raw_path.write_text(output, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError("{} exited with code {}; inspect {}".format(
            name, process.returncode, raw_path.name
        ))


def terminate_processes(processes: Iterable[subprocess.Popen]) -> None:
    for process in processes:
        try:
            if process.poll() is None:
                process.terminate()
        except Exception:
            pass
    deadline = time.monotonic() + 2.0
    for process in processes:
        try:
            if process.poll() is None:
                process.wait(timeout=max(0.05, deadline - time.monotonic()))
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def run_validation(keep_cli: bool = False) -> int:
    import hosn_switch as helper
    import mn_wifi.net as wifi_module

    for interface in ("s1", "ap1", "ap2"):
        check = subprocess.run(
            ["ip", "link", "show", "dev", interface],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=4,
        )
        if check.returncode == 0:
            raise RuntimeError(
                "An interface named {} already exists. Exit the previous Mininet "
                "experiment before starting this one.".format(interface)
            )

    project_dir = Path(__file__).resolve().parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = project_dir / "results" / ("rich_topology_" + stamp)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "experiment": "three-user Wi-Fi mobility and continuous-traffic validation",
        "status": "incomplete",
        "created_utc": stamp,
        "training_data_created": False,
        "labels_created": False,
        "ai_model_loaded": False,
        "controller_called": False,
        "handover_trigger": "fixed validation schedule, not a HOSN decision",
        "traffic": {
            "description": "constant-rate UDP media-like downlink plus concurrent ping",
            "duration_s": TRAFFIC_DURATION_S,
            "udp_packets_per_second_per_station": UDP_PACKETS_PER_SECOND,
            "udp_payload_bytes": UDP_PAYLOAD_BYTES,
            "ping_interval_s": PING_INTERVAL_S,
        },
        "topology": {
            "server": "h1 wired to s1",
            "access_points": AP_SPECS,
            "stations": station_plan(),
        },
        "wireless_backend": "Mininet-WiFi default hwsim/traffic-control backend; no wmediumd",
        "propagation_model": "logDistance",
        "propagation_exponent": PROPAGATION_EXPONENT,
        "association_control": "manual requests verified with Linux iw",
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python_version": platform.python_version(),
        "mininet_wifi_version": str(getattr(wifi_module, "VERSION", "unknown")),
        "source_sha256": {
            "hosn_rich_topology.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "hosn_switch.py": hashlib.sha256((project_dir / "hosn_switch.py").read_bytes()).hexdigest(),
        },
        "measurement_limits": [
            "This is a topology/traffic validation, not an AI or policy evaluation.",
            "Scheduled handovers are not training labels and are not controller decisions.",
            "UDP media-like traffic is not a real video codec or subjective video QoE.",
            "RSSI is calculated by the Mininet-WiFi propagation model, not a physical radio.",
            "Movement updates model geometry while actual packets use the emulated Wi-Fi path.",
            "One-way UDP delay uses the shared monotonic clock inside one Ubuntu VM.",
            "A packet gap is not claimed to be exact radio outage duration.",
            "No LTE, 5G, satellite, or cross-technology handover is implemented here.",
        ],
    }
    write_json(manifest_path, manifest)

    old_cwd = Path.cwd()
    network = None
    processes: List[subprocess.Popen] = []
    result_code = 1
    mobility_rows: List[dict] = []
    handover_events: Dict[str, dict] = {}
    ping_processes = {}
    sender_processes = {}
    receiver_processes = {}
    try:
        os.chdir(run_dir)
        network, stations, aps, server, _ = create_network()
        ap_sequence = (aps["ap1"], aps["ap2"])
        command_log = helper.NodeCommands(run_dir / "commands.jsonl")

        print("\nConnecting each station to its initial AP and verifying with iw...")
        for name, station in stations.items():
            target = aps[STATION_SPECS[name]["initial_ap"]]
            connection = helper.connect_verified(station, target, ap_sequence, command_log)
            preflight = helper.ping_on_ap(
                station, target, command_log,
                run_dir / ("preflight_" + name + ".txt"), count=3,
            )
            if preflight["received"] == 0 or not preflight["ap_verified_at_both_ends"]:
                raise RuntimeError(name + " failed initial AP/connectivity verification.")
            handover_events[name] = {
                "initial_connection": connection,
                "scheduled_handover_s": STATION_SPECS[name]["handover_s"],
                "handover": None,
            }

        # Receivers start before traffic.  Their stop time includes a grace
        # period so the final in-flight datagrams can be recorded.
        traffic_epoch = time.monotonic() + 1.0
        traffic_stop = traffic_epoch + TRAFFIC_DURATION_S
        receiver_stop = traffic_stop + 1.0
        for name, station in stations.items():
            spec = STATION_SPECS[name]
            packet_path = run_dir / ("udp_" + name + "_packets.jsonl")
            receiver_summary = run_dir / ("udp_" + name + "_receiver.json")
            process = start_long_process(station, [
                "python3", "-u", "-c", UDP_RECEIVER_CODE,
                spec["ip"], str(spec["udp_port"]), str(receiver_stop),
                str(packet_path), str(receiver_summary),
            ])
            receiver_processes[name] = process
            processes.append(process)

        time.sleep(0.4)
        ping_count = int(math.ceil(TRAFFIC_DURATION_S / PING_INTERVAL_S))
        for name, station in stations.items():
            process = start_long_process(station, [
                "ping", "-n", "-D", "-I", station.wintfs[0].name,
                "-c", str(ping_count), "-i", str(PING_INTERVAL_S),
                "-W", "1", SERVER_IP,
            ])
            ping_processes[name] = process
            processes.append(process)

        for name, spec in STATION_SPECS.items():
            sender_summary = run_dir / ("udp_" + name + "_sender.json")
            process = start_long_process(server, [
                "python3", "-u", "-c", UDP_SENDER_CODE,
                spec["ip"], str(spec["udp_port"]), str(traffic_epoch),
                str(traffic_stop), str(1.0 / UDP_PACKETS_PER_SECOND),
                str(UDP_PAYLOAD_BYTES), str(sender_summary),
            ])
            sender_processes[name] = process
            processes.append(process)

        print("\nLive traffic started. sta2 moves faster right-to-left; sta1 moves left-to-right.")
        print("Switches are scheduled validation actions, not HOSN decisions.")
        started = traffic_epoch
        next_sample = 0.0
        switched = set()
        last_verified = {
            name: STATION_SPECS[name]["initial_ap"] for name in stations
        }
        mobility_path = run_dir / "mobility.csv"
        with mobility_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MOBILITY_FIELDS)
            writer.writeheader()
            handle.flush()
            while True:
                now = time.monotonic()
                elapsed = now - started
                if elapsed >= TRAFFIC_DURATION_S:
                    break

                for name, station in stations.items():
                    position = interpolate_position(STATION_SPECS[name], elapsed)
                    # Direct model-geometry update avoids starting a second,
                    # automatic association mechanism that could fight HOSN.
                    station.position = list(position)

                # Execute any due handover in chronological order.  Traffic
                # and ping child processes continue while verification blocks.
                due = sorted(
                    name for name, spec in STATION_SPECS.items()
                    if spec["handover_s"] is not None
                    and elapsed >= spec["handover_s"] and name not in switched
                )
                for name in due:
                    station = stations[name]
                    spec = STATION_SPECS[name]
                    expected = str(aps[spec["initial_ap"]].wintfs[0].mac).lower()
                    event = {
                        "trigger": "fixed_validation_schedule",
                        "scheduled_elapsed_s": spec["handover_s"],
                        "request_started_monotonic_ns": time.monotonic_ns(),
                        "request_started_elapsed_s": time.monotonic() - started,
                        "from_ap": spec["initial_ap"],
                        "to_ap": spec["target_ap"],
                    }
                    print("  {}: verified switch {} -> {}".format(
                        name, spec["initial_ap"], spec["target_ap"]
                    ))
                    try:
                        result = helper.connect_verified(
                            station, aps[spec["target_ap"]], ap_sequence,
                            command_log, expected_current=expected,
                        )
                        event["result"] = result
                        event["status"] = "verified"
                        last_verified[name] = spec["target_ap"]
                    except BaseException as exc:
                        event["status"] = "failed"
                        event["error"] = type(exc).__name__ + ": " + str(exc)
                        raise
                    finally:
                        event["verification_finished_monotonic_ns"] = time.monotonic_ns()
                        event["verification_finished_elapsed_s"] = time.monotonic() - started
                        handover_events[name]["handover"] = event
                        write_json(run_dir / "handover_events.json", handover_events)
                    switched.add(name)

                now_elapsed = time.monotonic() - started
                if now_elapsed >= next_sample:
                    for name, station in stations.items():
                        x, y, z = (float(value) for value in station.position)
                        row = {
                            "elapsed_s": round(now_elapsed, 6),
                            "station": name,
                            "x_m": round(x, 6), "y_m": round(y, 6), "z_m": round(z, 6),
                            "speed_mps": round(path_speed(STATION_SPECS[name]), 6),
                            "direction": path_direction(STATION_SPECS[name]),
                            "ap1_rssi_model_dbm": modeled_rssi(station, aps["ap1"]),
                            "ap2_rssi_model_dbm": modeled_rssi(station, aps["ap2"]),
                            "last_verified_ap": last_verified[name],
                        }
                        mobility_rows.append(row)
                        writer.writerow(row)
                    handle.flush()
                    next_sample += MOBILITY_SAMPLE_INTERVAL_S
                time.sleep(0.05)

        # Complete and validate every child process.  Raw stdout is preserved.
        for name, process in sender_processes.items():
            finish_process(process, "UDP sender " + name,
                           run_dir / ("udp_" + name + "_sender_stdout.txt"), 6.0)
        for name, process in receiver_processes.items():
            finish_process(process, "UDP receiver " + name,
                           run_dir / ("udp_" + name + "_receiver_stdout.txt"), 6.0)
        for name, process in ping_processes.items():
            finish_process(process, "ping " + name,
                           run_dir / ("ping_" + name + ".txt"), 8.0)

        final_links = {}
        traffic_summary = {}
        for name, station in stations.items():
            observed_bssid, raw_link = helper.actual_link(station, command_log)
            (run_dir / ("final_link_" + name + ".txt")).write_text(raw_link, encoding="utf-8")
            expected_ap = STATION_SPECS[name]["target_ap"] or STATION_SPECS[name]["initial_ap"]
            expected_bssid = str(aps[expected_ap].wintfs[0].mac).lower()
            final_links[name] = {
                "expected_ap": expected_ap,
                "expected_bssid": expected_bssid,
                "observed_bssid": observed_bssid,
                "verified": observed_bssid == expected_bssid,
            }

            sender_summary = json.loads(
                (run_dir / ("udp_" + name + "_sender.json")).read_text(encoding="utf-8")
            )
            receiver_summary = json.loads(
                (run_dir / ("udp_" + name + "_receiver.json")).read_text(encoding="utf-8")
            )
            events = read_packet_events(run_dir / ("udp_" + name + "_packets.jsonl"))
            event = handover_events[name].get("handover")
            event_ns = event.get("request_started_monotonic_ns") if event else None
            udp = summarize_udp(events, sender_summary, event_ns)
            udp["receiver_status"] = receiver_summary.get("status")

            ping_raw = (run_dir / ("ping_" + name + ".txt")).read_text(encoding="utf-8")
            ping = helper.parse_ping(ping_raw)
            ping["requested"] = ping_count
            ping["fixed_request_count_complete"] = ping["sent"] == ping_count
            traffic_summary[name] = {"udp": udp, "ping": ping}

        pass_checks = {
            "all_initial_connections_verified": all(
                value["initial_connection"]["association_verified"]
                for value in handover_events.values()
            ),
            "both_mobile_handovers_verified": all(
                handover_events[name]["handover"]
                and handover_events[name]["handover"]["status"] == "verified"
                for name in ("sta1", "sta2")
            ),
            "all_final_bssids_verified": all(value["verified"] for value in final_links.values()),
            "all_udp_streams_received_packets": all(
                value["udp"]["packets_received_unique"] > 0
                for value in traffic_summary.values()
            ),
            "all_udp_processes_completed": all(
                value["udp"]["sender_status"] == "complete"
                and value["udp"]["receiver_status"] == "complete"
                for value in traffic_summary.values()
            ),
            "mobile_udp_continues_across_each_handover": all(
                traffic_summary[name]["udp"]["packets_received_before_handover"] > 0
                and traffic_summary[name]["udp"]["packets_received_after_handover"] > 0
                for name in ("sta1", "sta2")
            ),
            "all_ping_probes_received_replies": all(
                value["ping"]["received"] > 0 for value in traffic_summary.values()
            ),
            "all_ping_request_counts_complete": all(
                value["ping"]["fixed_request_count_complete"]
                for value in traffic_summary.values()
            ),
            "stationary_user_remained_on_ap1": final_links["sta3"]["verified"],
        }
        summary = {
            "functional_validation_passed": all(pass_checks.values()),
            "checks": pass_checks,
            "final_links": final_links,
            "handover_events": handover_events,
            "traffic": traffic_summary,
            "interpretation": (
                "Passing proves that this emulated topology carried traffic before and after "
                "verified scheduled Wi-Fi handovers. It does not prove seamless human video "
                "QoE, an optimal policy, AI accuracy, or LTE/5G handover."
            ),
        }
        write_json(run_dir / "traffic_summary.json", traffic_summary)
        write_json(run_dir / "summary.json", summary)
        manifest["status"] = "passed" if summary["functional_validation_passed"] else "check_failed"
        result_code = 0 if summary["functional_validation_passed"] else 1

        print("\nRICH TOPOLOGY VALIDATION:", "PASS" if result_code == 0 else "NOT PASSED")
        for name in ("sta1", "sta2", "sta3"):
            udp = traffic_summary[name]["udp"]
            ping = traffic_summary[name]["ping"]
            gap = udp["handover_packet_gap_ms"]
            print(
                "  {}: UDP {}/{} received, {:.3f}% loss, max gap {} ms; "
                "ping {}/{} replies".format(
                    name, udp["packets_received_unique"], udp["packets_sent"],
                    udp["loss_pct"] if udp["loss_pct"] is not None else float("nan"),
                    "N/A" if gap is None else round(gap, 3),
                    ping["received"], ping["sent"],
                )
            )
        print("Scheduled switches were validation actions, not HOSN decisions or labels.")

        if keep_cli:
            from mn_wifi.cli import CLI
            print("\nInteractive mode. CLI actions are not added to validation summaries.")
            CLI(network)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted_partial_results_preserved"
        result_code = 130
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("\nInterrupted. Completed raw observations remain in the result folder.")
    except BaseException as exc:
        manifest["status"] = "failed_partial_results_preserved"
        manifest["error"] = type(exc).__name__ + ": " + str(exc)
        result_code = 1
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("\nVALIDATION STOPPED: {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
    finally:
        terminate_processes(processes)
        try:
            if network is not None:
                print("\nStopping the simulated network...")
                network.stop()
        except Exception as exc:
            manifest["cleanup_error"] = type(exc).__name__ + ": " + str(exc)
            result_code = 1
            print("Cleanup warning. Before retrying, run: sudo mn -c", file=sys.stderr)
        finally:
            os.chdir(old_cwd)
            write_json(manifest_path, manifest)
            try:
                helper.restore_result_owner(run_dir)
            except OSError as exc:
                print("Could not restore result ownership:", exc, file=sys.stderr)
            print("Saved validation files:", run_dir)
    return result_code


def run_self_tests() -> int:
    import tempfile
    import unittest

    class Tests(unittest.TestCase):
        def test_speeds_and_directions_differ(self):
            self.assertGreater(path_speed(STATION_SPECS["sta2"]),
                               path_speed(STATION_SPECS["sta1"]))
            self.assertEqual(path_direction(STATION_SPECS["sta1"]), "left_to_right")
            self.assertEqual(path_direction(STATION_SPECS["sta2"]), "right_to_left")
            self.assertEqual(path_direction(STATION_SPECS["sta3"]), "stationary")

        def test_interpolation_clamps_path(self):
            spec = STATION_SPECS["sta1"]
            self.assertEqual(interpolate_position(spec, -10), spec["start"])
            self.assertEqual(interpolate_position(spec, 100), spec["end"])
            halfway = interpolate_position(spec, spec["move_start_s"] + spec["move_duration_s"] / 2)
            self.assertEqual(halfway, (50.0, 40.0, 0.0))

        def test_handover_occurs_near_midpoint_and_within_ranges(self):
            for name in ("sta1", "sta2"):
                spec = STATION_SPECS[name]
                position = interpolate_position(spec, spec["handover_s"])
                self.assertAlmostEqual(position[0], 50.0)
                for ap in AP_SPECS.values():
                    self.assertLess(math.dist(position, ap["position"]), AP_RANGE_M)

        def test_udp_summary_preserves_spike_and_loss(self):
            events = [
                {"sequence": 0, "sent_monotonic_ns": 0, "received_monotonic_ns": 10_000_000},
                {"sequence": 1, "sent_monotonic_ns": 40_000_000,
                 "received_monotonic_ns": 60_000_000},
                {"sequence": 3, "sent_monotonic_ns": 120_000_000,
                 "received_monotonic_ns": 1_120_000_000},
            ]
            value = summarize_udp(events, {"status": "complete", "sent": 4}, 70_000_000)
            self.assertEqual(value["packets_lost"], 1)
            self.assertEqual(value["loss_pct"], 25.0)
            self.assertEqual(value["delay_max_ms"], 1000.0)
            self.assertEqual(value["handover_packet_gap_ms"], 1060.0)
            self.assertEqual(value["packets_received_before_handover"], 2)
            self.assertEqual(value["packets_received_after_handover"], 1)

        def test_duplicates_are_reported_not_counted_as_delivery(self):
            events = [
                {"sequence": 0, "sent_monotonic_ns": 1, "received_monotonic_ns": 2},
                {"sequence": 0, "sent_monotonic_ns": 1, "received_monotonic_ns": 3},
            ]
            value = summarize_udp(events, {"status": "complete", "sent": 2}, None)
            self.assertEqual(value["duplicate_packets"], 1)
            self.assertEqual(value["packets_received_unique"], 1)
            self.assertEqual(value["packets_lost"], 1)

        def test_missing_packet_log_is_empty(self):
            with tempfile.TemporaryDirectory() as directory:
                self.assertEqual(read_packet_events(Path(directory) / "missing.jsonl"), [])

        def test_json_rejects_nan(self):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    write_json(Path(directory) / "bad.json", {"value": float("nan")})

        def test_feature_contract_is_not_changed_here(self):
            import ast
            text = Path(__file__).read_text(encoding="utf-8")
            tree = ast.parse(text)
            imported_modules = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_modules.add(node.module)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotEqual(node.func.attr, "predict")
            self.assertNotIn("hosn_controller", imported_modules)
            self.assertNotIn("hosn_rules", imported_modules)

    result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    )
    if result.wasSuccessful():
        print(
            "PASS: {} rich-topology software tests. No network was started and no "
            "training data was generated.".format(result.testsRun)
        )
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-test", action="store_true",
                      help="Run local software checks; no sudo or Mininet required.")
    mode.add_argument("--plan", action="store_true",
                      help="Print the topology/mobility plan without starting Mininet.")
    parser.add_argument("--cli", action="store_true",
                        help="Open Mininet CLI after measurements, before cleanup.")
    args = parser.parse_args()
    if args.self_test:
        return run_self_tests()
    if args.plan:
        print(json.dumps({
            "purpose": "pre-dataset rich-topology validation",
            "traffic_duration_s": TRAFFIC_DURATION_S,
            "access_points": AP_SPECS,
            "stations": station_plan(),
            "training_data_created": False,
            "controller_called": False,
        }, indent=2, allow_nan=False))
        return 0
    if sys.platform != "linux" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        parser.exit(2, "Run inside the Ubuntu VM: sudo python3 hosn_rich_topology.py\n")
    missing = [name for name in ("ip", "iw", "ping", "python3") if not shutil.which(name)]
    if missing:
        parser.exit(2, "Missing required system tools: " + ", ".join(missing) + "\n")
    helper_path = Path(__file__).resolve().parent / "hosn_switch.py"
    if not helper_path.is_file():
        parser.exit(2, "Keep hosn_rich_topology.py beside hosn_switch.py in /home/mira/Hosn\n")
    return run_validation(keep_cli=args.cli)


if __name__ == "__main__":
    sys.exit(main())
