#!/usr/bin/env python3
"""HOSN stage 1: a headless, measured two-AP handover baseline.

Run inside the Ubuntu VM: sudo python3 handover_simulation.py
Optional interactive prompt after the experiment: add --cli

This is NOT the HOSN AI policy and NOT a replacement training dataset.
- Association policy: Mininet-WiFi's built-in strongest-signal-first (ssf).
- RSSI: calculated by Mininet-WiFi's log-distance propagation model.
- RTT/loss: measured with Linux ping through the station's active connection.
- Movement: discrete positions, with a pause to measure at each position.
- No candidate-path RTT/loss, invented labels, throughput, or exact handover
  interruption time are produced. In particular, missing RTT is not zero.
- Uses the default traffic-control wireless backend, not wmediumd.
- Results are saved in a NEW results/handover_<UTC timestamp>/ directory.

APs and the wired switch use standalone bridging; no OpenFlow controller
or desktop/plot window is required. Existing HOSN code/data are not modified.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

# Must be set before importing Mininet-WiFi, which can import matplotlib.
os.environ["MPLBACKEND"] = "Agg"

POSITIONS = tuple(range(10, 91, 5))
Y_POSITION = 40.0
PAUSE_SECONDS = 1.0
PING_COUNT = 5
SERVER_IP = "10.0.0.100"
FIELDS = (
    "sample", "x_m", "y_m", "ping_start_s", "ping_end_s",
    "ap_before_ping", "ap_after_ping", "bssid_before", "bssid_after",
    "ap1_rssi_model_dbm", "ap2_rssi_model_dbm",
    "ping_transmitted", "ping_received", "ping_loss_pct", "rtt_avg_ms",
)


def parse_ping(text):
    """Parse actual Linux ping output; reject missing/invalid summaries."""
    counts = re.search(
        r"(\d+) packets transmitted,\s*(\d+) (?:packets )?received", text
    )
    if not counts:
        raise ValueError("Ping did not produce a packet-count summary.")
    sent, received = map(int, counts.groups())
    if sent <= 0 or not 0 <= received <= sent:
        raise ValueError("Ping reported invalid packet counts.")
    rtt = re.search(
        r"(?:rtt|round-trip)[^=\n]*=\s*([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)",
        text,
    )
    avg = float(rtt.group(2)) if rtt else None
    if received and (avg is None or not math.isfinite(avg) or avg < 0):
        raise ValueError("Replies were reported but the RTT summary is invalid.")
    if not received:
        avg = None  # A lost probe has unknown RTT, NOT a zero-ms RTT.
    return {
        "ping_transmitted": sent,
        "ping_received": received,
        "ping_loss_pct": round(100.0 * (sent - received) / sent, 3),
        "rtt_avg_ms": avg,
    }


def parse_link(text):
    """Return the BSSID verified by iw, or '' for an explicit disconnection."""
    connected = re.search(r"Connected to\s+([0-9a-f:]{17})", text, re.I)
    if connected:
        return connected.group(1).lower()
    if "not connected" in text.lower():
        return ""
    raise ValueError("Unrecognized 'iw link' output; cannot verify association.")


def run_node(node, command, timeout=8):
    """Run inside a Mininet node, without a shell, with a bounded wait."""
    env = dict(os.environ, LC_ALL="C", LANG="C")
    process = node.popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, env=env,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except BaseException:
        # Also clean up this child when the user presses Ctrl+C.
        process.kill()
        process.communicate()
        raise
    return output, process.returncode


def read_association(station, bssid_names):
    interface = station.wintfs[0].name
    text, code = run_node(station, ["iw", "dev", interface, "link"])
    if code != 0:
        raise RuntimeError("iw failed: " + text.strip())
    bssid = parse_link(text)
    name = bssid_names.get(bssid, "UNKNOWN_AP") if bssid else "NOT_CONNECTED"
    return name, bssid, text


def probe(station, raw_path):
    """Measure the path currently used by sta1, not an unassociated AP."""
    text, code = run_node(
        station,
        ["ping", "-n", "-c", str(PING_COUNT), "-i", "0.2",
         "-W", "1", "-w", "4", SERVER_IP],
    )
    raw_path.write_text(text, encoding="utf-8")
    if code not in (0, 1):
        raise RuntimeError("ping failed; inspect " + str(raw_path))
    return parse_ping(text)


def modeled_rssi(station, access_point):
    """Model-derived dBm; this is not a physical RF measurement."""
    distance = station.get_distance_to(access_point)
    value = float(station.wintfs[0].get_rssi(access_point.wintfs[0], distance))
    if not math.isfinite(value):
        raise RuntimeError("Propagation model returned a non-finite RSSI.")
    return round(value, 3)


def summarize(rows):
    """Judge only what was observed; never report an assumed handover."""
    transitions = []
    previous = None
    replies = {"ap1": 0, "ap2": 0}
    for row in rows:
        for key in ("ap_before_ping", "ap_after_ping"):
            ap = row[key]
            if ap in replies:
                if previous is not None and ap != previous:
                    transitions.append([previous, ap])
                previous = ap
        before, after = row["ap_before_ping"], row["ap_after_ping"]
        if before == after and before in replies:
            replies[before] += row["ping_received"]
    sent = sum(row["ping_transmitted"] for row in rows)
    received = sum(row["ping_received"] for row in rows)
    passed = bool(
        rows and ["ap1", "ap2"] in transitions
        and replies["ap1"] > 0 and replies["ap2"] > 0
        and rows[-1]["ap_before_ping"] == "ap2"
        and rows[-1]["ap_after_ping"] == "ap2"
        and rows[-1]["ping_received"] > 0
    )
    return {
        "functional_test_passed": passed,
        "samples": len(rows),
        "observed_ap_transitions": transitions,
        "replies_in_stable_ap_windows": replies,
        "ping_transmitted": sent,
        "ping_received": received,
        "ping_loss_pct": round(100.0 * (sent - received) / sent, 3) if sent else None,
    }


def restore_owner(results_root, run_dir):
    """Keep generated files editable by the user who invoked sudo."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid is None or gid is None:
        return
    try:
        for path in [results_root, run_dir, *run_dir.rglob("*")]:
            os.chown(path, int(uid), int(gid), follow_symlinks=False)
    except OSError as exc:
        print("WARNING: could not restore ownership of every result:", exc)


def experiment(keep_cli=False):
    # Import only when actually running, so helpers can be tested without the VM.
    from mininet.link import Link
    from mininet.log import setLogLevel
    from mininet.node import OVSBridge
    from mn_wifi.cli import CLI
    import mn_wifi.net as wifi_module
    from mn_wifi.net import Mininet_wifi
    from mn_wifi.node import OVSBridgeAP

    setLogLevel("info")
    project_dir = Path(__file__).resolve().parent
    results_root = project_dir / "results"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = results_root / ("handover_" + stamp)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "stage": "functional SSF baseline, not HOSN AI evaluation",
        "status": "incomplete",
        "created_utc": stamp,
        "mininet_wifi_version": str(getattr(wifi_module, "VERSION", "unknown")),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "association_policy": "Mininet-WiFi built-in ssf",
        "forwarding": "standalone OVS bridges, no controller",
        "wireless_backend": "default traffic-control backend; no wmediumd",
        "propagation_model": "logDistance",
        "propagation_exponent": 3.5,
        "movement": "stepped; pause and probe at each position; not constant speed",
        "x_positions_m": list(POSITIONS),
        "station_y_m": Y_POSITION,
        "pause_per_position_s": PAUSE_SECONDS,
        "ping_count_requested_per_window": PING_COUNT,
        "ping_interval_s": 0.2,
        "ping_target": SERVER_IP,
        "measurement_limits": [
            "RSSI values are propagation-model outputs, not physical readings.",
            "Both AP RSSIs are calculated even out of range; they are not scan results.",
            "Ping RTT/loss are observed inside this emulated network.",
            "Probes happen after movement steps, not continuously during movement.",
            "APs are sampled before/after probes; brief changes may be missed.",
            "No candidate-path RTT/loss or exact handover interruption is measured.",
            "Five probes per window are a functional check, not a robust loss estimate.",
            "This file is not labeled training data and does not overwrite hosn_data.csv.",
        ],
    }
    manifest_path = run_dir / "run_info.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    net = None
    rows = []
    previous_dir = Path.cwd()
    try:
        # Confine Mininet's temporary configuration files to this run directory.
        os.chdir(run_dir)
        net = Mininet_wifi(
            controller=None, switch=OVSBridge, accessPoint=OVSBridgeAP,
            ac_method="ssf",
        )
        station = net.addStation(
            "sta1", ip="10.0.0.1/24", mac="02:00:00:00:00:01",
            position="10,40,0", range=50,
        )
        server = net.addHost(
            "h1", ip=SERVER_IP + "/24", mac="02:00:00:00:00:64"
        )
        switch = net.addSwitch("s1", dpid="0000000000000010")
        ap1 = net.addAccessPoint(
            "ap1", ssid="hosn-wifi", mode="g", channel="1",
            mac="02:00:00:00:01:01", position="20,50,0", range=50,
        )
        ap2 = net.addAccessPoint(
            "ap2", ssid="hosn-wifi", mode="g", channel="6",
            mac="02:00:00:00:02:01", position="80,50,0", range=50,
        )
        net.setPropagationModel(model="logDistance", exp=3.5)
        configure = getattr(net, "configureNodes", None)
        if configure is None:
            configure = net.configureWifiNodes  # Older Mininet-WiFi releases.
        configure()
        for endpoint in (server, ap1, ap2):
            net.addLink(switch, endpoint, cls=Link)
        net.build()
        switch.start([])
        ap1.start([])
        ap2.start([])

        bssid_names = {str(ap.wintfs[0].mac).lower(): ap.name for ap in (ap1, ap2)}
        manifest["access_points"] = {
            ap.name: {
                "position_m": list(ap.position),
                "bssid": str(ap.wintfs[0].mac),
                "channel": str(ap.wintfs[0].channel),
                "effective_range_m": float(ap.wintfs[0].range),
                "txpower_dbm": float(ap.wintfs[0].txpower),
            } for ap in (ap1, ap2)
        }
        print("\nWaiting for sta1 to associate with ap1...")
        deadline = time.monotonic() + 15
        while True:
            ap, _, raw = read_association(station, bssid_names)
            (run_dir / "initial_link.txt").write_text(raw, encoding="utf-8")
            if ap == "ap1":
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("sta1 did not associate with ap1; see initial_link.txt")
            time.sleep(0.5)
        warmup = probe(station, run_dir / "warmup_ping.txt")
        if warmup["ping_received"] == 0:
            raise RuntimeError("Associated to ap1, but h1 is unreachable; see warmup_ping.txt")

        print("\nMoving in steps from AP1 toward AP2; each row uses 5 pings.")
        print("   x   AP before -> after        RSSI1 / RSSI2       RTT    loss")
        started = time.monotonic()
        with (run_dir / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            handle.flush()
            for sample, x in enumerate(POSITIONS, start=1):
                station.setPosition("{},40,0".format(x))
                time.sleep(PAUSE_SECONDS)
                before, bssid_before, link_before = read_association(station, bssid_names)
                rssi1, rssi2 = modeled_rssi(station, ap1), modeled_rssi(station, ap2)
                start_s = time.monotonic() - started
                measured = probe(station, run_dir / ("ping_{:03d}.txt".format(sample)))
                end_s = time.monotonic() - started
                after, bssid_after, link_after = read_association(station, bssid_names)
                (run_dir / ("link_{:03d}.txt".format(sample))).write_text(
                    "BEFORE PING\n" + link_before + "\nAFTER PING\n" + link_after,
                    encoding="utf-8",
                )
                row = {
                    "sample": sample, "x_m": x, "y_m": Y_POSITION,
                    "ping_start_s": round(start_s, 6), "ping_end_s": round(end_s, 6),
                    "ap_before_ping": before, "ap_after_ping": after,
                    "bssid_before": bssid_before, "bssid_after": bssid_after,
                    "ap1_rssi_model_dbm": rssi1, "ap2_rssi_model_dbm": rssi2,
                    **measured,
                }
                rows.append(row)
                writer.writerow(row)  # None becomes an empty CSV cell, not zero.
                handle.flush()
                rtt = "N/A" if measured["rtt_avg_ms"] is None else "{:.2f}ms".format(measured["rtt_avg_ms"])
                print("{:4}  {:>4} -> {:<13} {:7.2f} / {:7.2f}  {:>8}  {:5.1f}%".format(
                    x, before, after, rssi1, rssi2, rtt, measured["ping_loss_pct"]
                ))

        summary = summarize(rows)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        manifest["status"] = "passed" if summary["functional_test_passed"] else "check_failed"
        print("\nFUNCTIONAL TEST:", "PASS" if summary["functional_test_passed"] else "NOT PASSED")
        print("Observed AP transitions:", summary["observed_ap_transitions"])
        print("Replies received:", summary["ping_received"], "/", summary["ping_transmitted"])
        print("This is a baseline check, not proof of AI performance or seamless handover.")
        if keep_cli:
            print("\nInteractive mode. Type exit to stop. CLI actions are not added to the CSV.")
            CLI(net)
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        manifest["error"] = repr(exc)
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        try:
            if net is not None:
                print("\nStopping the simulated network...")
                net.stop()
        except Exception as exc:
            manifest["cleanup_error"] = repr(exc)
            print("WARNING: cleanup failed:", exc)
            print("Before retrying, run: sudo mn -c")
        finally:
            os.chdir(previous_dir)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            restore_owner(results_root, run_dir)
            print("\nSaved run files:", run_dir)
    return 0 if manifest["status"] == "passed" and "cleanup_error" not in manifest else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", action="store_true", help="Open Mininet CLI after the test")
    args = parser.parse_args()
    if sys.platform != "linux" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("Run inside the Ubuntu VM: sudo python3 handover_simulation.py", file=sys.stderr)
        return 1
    try:
        return experiment(keep_cli=args.cli)
    except KeyboardInterrupt:
        print("\nStopped by you. Completed rows remain in the run folder.")
        return 130
    except Exception as exc:
        print("\nERROR: {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        print("Share this error before changing packages or deleting files.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
