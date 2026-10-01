#!/usr/bin/env python3
"""HOSN switching action: request a connection and verify it with Linux iw.

Run in the existing Ubuntu VM, with no other Mininet experiment running:
    sudo python3 hosn_switch.py

This creates a SHORT, STATIC AP1 -> AP2 action test. It does NOT select the
best AP, call the rules/AI, replay movement, train anything, or replace data.
Both APs are within the station's configured range. Automatic association is
disabled so that a second mechanism cannot compete with our requests.

The reusable connect_verified() function implements the sequence the user
successfully tested manually: disconnect, connect to the requested BSSID,
poll 'iw link', and ONLY THEN synchronize Mininet's bookkeeping.
Never use associatedTo alone as proof of a completed switch.

This helper is for the project's OPEN, emulated Wi-Fi network. It is not a
real-phone controller or a WPA/WPA2 roaming implementation. This action test
uses the default wireless backend, not wmediumd, and does not measure physical
RF, candidate-path quality, movement effects, or exact handover interruption.
It does not load the old .pkl model. No model or dataset is modified.

Later integration: the controller selects an AP; connect_verified() performs
that request. The measurement/controller loop must still enforce fresh data,
cooldown, and single ownership of association. On a failed request this helper
reports failure rather than pretending success or silently making a new choice.

Results: NEW results/switch_check_<UTC timestamp>/, with raw command logs,
ping reports, and summary.json. No old experiment results are overwritten.

Primary API references, checked 2026-10-01:
https://wireless.docs.kernel.org/en/latest/en/users/documentation/iw.html
https://github.com/intrig-unicamp/mininet-wifi/blob/master/mn_wifi/net.py
https://github.com/intrig-unicamp/mininet-wifi/blob/master/mn_wifi/link.py
"""

import argparse
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from typing import Optional

# Set before Mininet-WiFi imports: no desktop display is required.
os.environ.setdefault("MPLBACKEND", "Agg")

SSID = "hosn-wifi"
SERVER_IP = "10.0.0.100"
PING_COUNT = 5
MAC_PATTERN = r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}"


def parse_bssid(text: str) -> str:
    """Actual BSSID, or '' for explicit 'Not connected'; never guess on errors."""
    match = re.search(r"^\s*Connected to\s+(" + MAC_PATTERN + r")(?:\s|$)",
                      text, flags=re.I | re.M)
    if match:
        return match.group(1).lower()
    if re.search(r"^\s*Not connected\.?\s*$", text, flags=re.I | re.M):
        return ""
    raise ValueError("Cannot interpret the iw link output; see commands.jsonl.")


def parse_ping(text: str) -> dict:
    counts = re.search(r"(\d+) packets transmitted,\s*(\d+) (?:packets )?received", text)
    if not counts:
        raise ValueError("No packet-count summary in ping output.")
    sent, received = map(int, counts.groups())
    if sent < 1 or not 0 <= received <= sent:
        raise ValueError("Invalid ping packet counts.")
    match = re.search(r"(?:rtt|round-trip)[^=\n]*=\s*[\d.]+/([\d.]+)/", text)
    average = float(match.group(1)) if match and received else None
    if received and (average is None or not math.isfinite(average) or average < 0):
        raise ValueError("Replies reported without a valid round-trip summary.")
    return {"sent": sent, "received": received,
            "loss_pct": 100.0 * (sent - received) / sent,
            "rtt_avg_ms": average}  # No replies means unknown RTT, NOT zero RTT.


class NodeCommands:
    """Bounded, argument-list commands inside a Mininet node, with raw logs."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path

    def __call__(self, node, args, timeout=8.0):
        event = {"utc": datetime.now(timezone.utc).isoformat(),
                 "node": node.name, "argv": list(args)}
        process = None
        started = time.monotonic()
        try:
            process = node.popen(
                list(args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=dict(os.environ, LC_ALL="C", LANG="C"),
            )
            output, _ = process.communicate(timeout=timeout)
            event.update(returncode=process.returncode, output=output)
            return output, process.returncode
        except BaseException as exc:
            event["error"] = type(exc).__name__ + ": " + str(exc)
            if process is not None:
                if process.poll() is None:
                    process.kill()
                output, _ = process.communicate()
                event.update(returncode=process.returncode, output=output)
            raise
        finally:
            event["elapsed_s"] = time.monotonic() - started
            if self.log_path is not None:
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, allow_nan=False) + "\n")


def actual_link(station, run):
    interface = station.wintfs[0].name
    text, code = run(station, ["iw", "dev", interface, "link"])
    if code:
        raise RuntimeError("iw link failed: " + text.strip())
    return parse_bssid(text), text


def wait_for_link(station, expected, run, timeout=12.0):
    """Require two consecutive observations; timer is NOT outage duration."""
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be positive and finite.")
    deadline = time.monotonic() + timeout
    consecutive = 0
    while True:
        observed, _ = actual_link(station, run)
        consecutive = consecutive + 1 if observed == expected else 0
        if consecutive >= 2:
            return observed
        if time.monotonic() >= deadline:
            raise TimeoutError("Expected {} but last observed {}. See commands.jsonl."
                               .format(expected or "disconnected", observed or "disconnected"))
        time.sleep(0.25)


def sync_observed_record(station, access_points, observed):
    """Synchronize ONLY from an actual iw observation, never a requested AP.

    This updates bookkeeping, not the kernel connection or propagation model.
    Association-changing threads must be disabled by the calling experiment.
    """
    interface = station.wintfs[0]
    target = next((ap.wintfs[0] for ap in access_points
                   if str(ap.wintfs[0].mac).lower() == observed), None)
    for ap in access_points:
        members = ap.wintfs[0].associatedStations
        while interface in members:
            members.remove(interface)
    interface.associatedTo = target
    if target is not None:
        target.associatedStations.append(interface)
        for attr in ("freq", "channel", "mode", "ssid"):
            setattr(interface, attr, getattr(target, attr))
    # Unknown/absent AP -> unknown record, not a fictional successful association.


def connect_verified(station, target_ap, access_points, run=None,
                     timeout=12.0, expected_current=None) -> dict:
    """Execute one open-network association request; raise on failure.

    expected_current, when supplied, guards against a stale decision. Supply
    the actual BSSID from the measurement on which the decision was based.
    A successful return means association verified, NOT end-to-end connectivity.
    The caller should run a connectivity check and handle failures explicitly.
    """
    run = run if run is not None else NodeCommands()
    aps = tuple(access_points)
    if target_ap not in aps:
        raise ValueError("The requested AP is not part of this experiment.")
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be positive and finite.")
    ap_intf = target_ap.wintfs[0]
    bssid = str(ap_intf.mac).lower()
    if not re.fullmatch(MAC_PATTERN, bssid):
        raise ValueError("Invalid target AP address.")
    if getattr(ap_intf, "encrypt", None):
        raise ValueError("This helper supports the OPEN test network only.")
    ssid = str(ap_intf.ssid)
    if not ssid or "\x00" in ssid or len(ssid.encode("utf-8")) > 32:
        raise ValueError("Invalid SSID.")
    if hasattr(station, "position") and hasattr(target_ap, "position"):
        if station.get_distance_to(target_ap) > float(ap_intf.range):
            raise ValueError("Target AP is outside its configured simulated range.")

    before, _ = actual_link(station, run)
    if expected_current is not None and before != expected_current.lower():
        raise RuntimeError("Connection changed since the decision; collect fresh data.")
    known = {str(ap.wintfs[0].mac).lower() for ap in aps}
    if before and before not in known:
        raise RuntimeError("Connected to an unknown AP; no disconnect was requested.")
    sync_observed_record(station, aps, before)
    started = time.monotonic()
    try:
        if before != bssid:
            interface = station.wintfs[0].name
            if before:
                text, code = run(station, ["iw", "dev", interface, "disconnect"])
                # A concurrent disconnect may return an error; only actual state
                # can establish that it is safe to proceed.
                if code and actual_link(station, run)[0]:
                    raise RuntimeError("Disconnect failed: " + text.strip())
                wait_for_link(station, "", run, timeout=min(timeout, 4.0))
                sync_observed_record(station, aps, "")
            text, code = run(station, ["iw", "dev", interface, "connect", ssid, bssid])
            if code:
                raise RuntimeError("Connect request failed: " + text.strip())
        # Even an 'already on target' result requires actual verification.
        verified = wait_for_link(station, bssid, run, timeout=timeout)
        sync_observed_record(station, aps, verified)
        return {"requested_ap": target_ap.name, "before_bssid": before,
                "verified_bssid": verified, "association_verified": True,
                "connection_changed": before != verified,
                "request_to_observation_s": time.monotonic() - started}
    except BaseException:
        try:
            observed, _ = actual_link(station, run)
            sync_observed_record(station, aps, observed)
        except Exception:
            sync_observed_record(station, aps, "")
        raise


def ping_on_ap(station, ap, run, raw_path: Path, count=PING_COUNT) -> dict:
    """Bind probes to the Wi-Fi interface; check AP at both window endpoints."""
    expected = str(ap.wintfs[0].mac).lower()
    before, _ = actual_link(station, run)
    if before != expected:
        raise RuntimeError("Not on the intended AP before ping; no result attributed to it.")
    text, code = run(station, ["ping", "-n", "-I", station.wintfs[0].name,
                              "-c", str(count), "-i", "0.2", "-W", "1",
                              "-w", "6", SERVER_IP], timeout=9.0)
    raw_path.write_text(text, encoding="utf-8")
    if code not in (0, 1):
        raise RuntimeError("ping command failed; see " + raw_path.name)
    result = parse_ping(text)
    after, _ = actual_link(station, run)
    result.update(ap=ap.name, bssid_before=before, bssid_after=after,
                  ap_verified_at_both_ends=before == after == expected,
                  all_requested_replies=(result["sent"] == count == result["received"]))
    return result


def restore_result_owner(run_dir: Path):
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid is not None and gid is not None:
        for path in (run_dir.parent, run_dir, *run_dir.rglob("*")):
            if not path.is_symlink():
                os.chown(path, int(uid), int(gid), follow_symlinks=False)


def run_experiment() -> int:
    # Lazy imports: opening this file does not create any network.
    from mininet.link import Link
    from mininet.log import setLogLevel
    from mininet.node import OVSBridge
    import mn_wifi.net as wifi_module
    from mn_wifi.net import Mininet_wifi
    from mn_wifi.node import OVSBridgeAP

    setLogLevel("info")
    for name in ("s1", "ap1", "ap2"):
        check = subprocess.run(["ip", "link", "show", "dev", name],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=4)
        if check.returncode == 0:
            raise RuntimeError("An interface named {} already exists. Exit the previous "
                               "Mininet experiment before starting this one.".format(name))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = Path(__file__).resolve().parent / "results" / ("switch_check_" + stamp)
    folder.mkdir(parents=True, exist_ok=False)
    report = {"stage": "static verified switching ACTION test, NOT decision evaluation",
              "status": "incomplete", "created_utc": stamp,
              "kernel": platform.release(), "python": platform.python_version(),
              "mininet_wifi_version": str(getattr(wifi_module, "VERSION", "unknown")),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "automatic_association": False,
              "ai_used": False, "rules_used": False, "movement_used": False,
              "backend": "default wireless backend, not wmediumd",
              "station_position_m": [50, 40, 0],
              "ap_positions_m": {"ap1": [20, 50, 0], "ap2": [80, 50, 0]},
              "configured_ap_range_m": 50,
              "limits": ["Fixed AP1 then AP2: not proof either AP is a better choice.",
                         "Pings run after each connection, not continuously during switching.",
                         "Request-to-observation time is NOT exact handover interruption.",
                         "Five pings are a functional check, not a robust loss estimate.",
                         "This is not a training dataset or an AI performance result."],
              "steps": []}
    previous_dir = Path.cwd()
    network = None
    result_code = 1
    try:
        os.chdir(folder)
        network = Mininet_wifi(controller=None, switch=OVSBridge,
                               accessPoint=OVSBridgeAP, autoAssociation=False,
                               allAutoAssociation=False)
        station = network.addStation("sta1", ip="10.0.0.1/24",
                                     mac="02:00:00:00:00:01",
                                     position="50,40,0", range=50)
        server = network.addHost("h1", ip=SERVER_IP + "/24",
                                 mac="02:00:00:00:00:64")
        switch = network.addSwitch("s1", dpid="0000000000000010")
        ap1 = network.addAccessPoint("ap1", ssid=SSID, mode="g", channel="1",
                                    mac="02:00:00:00:01:01", position="20,50,0", range=50)
        ap2 = network.addAccessPoint("ap2", ssid=SSID, mode="g", channel="6",
                                    mac="02:00:00:00:02:01", position="80,50,0", range=50)
        network.setPropagationModel(model="logDistance", exp=3.5)
        configure = getattr(network, "configureNodes", None)
        if configure is None:
            configure = network.configureWifiNodes
        configure()
        for endpoint in (server, ap1, ap2):
            network.addLink(switch, endpoint, cls=Link)
        network.build()
        for bridge in (switch, ap1, ap2):
            bridge.start([])
        time.sleep(1.0)
        run = NodeCommands(folder / "commands.jsonl")
        # Each connection is verified against the actual AP interface address.
        report["ap_bssids"] = {ap.name: str(ap.wintfs[0].mac).lower() for ap in (ap1, ap2)}
        print("\nAUTOMATED SWITCH ACTION TEST - no rules or AI are deciding.\n", flush=True)
        for ap in (ap1, ap2):
            print("Requesting connection to {}...".format(ap.name), flush=True)
            step = {"ap": ap.name}
            report["steps"].append(step)
            expected = str(ap1.wintfs[0].mac).lower() if ap is ap2 else None
            step["connection"] = connect_verified(station, ap, (ap1, ap2), run,
                                                   expected_current=expected)
            print("Verified by Linux: {} ({})".format(
                ap.name, step["connection"]["verified_bssid"]), flush=True)
            if ap is ap1:
                # Separate, preserved address-learning probe; not mixed into
                # the five reported test requests or hidden from the run log.
                step["warmup"] = ping_on_ap(station, ap, run,
                                           folder / "warmup_ap1.txt", count=1)
                if not step["warmup"]["ap_verified_at_both_ends"]:
                    raise RuntimeError("AP changed during the warmup probe.")
            measured = ping_on_ap(station, ap, run, folder / ("ping_" + ap.name + ".txt"))
            step["ping"] = measured
            print("{}: {} / {} replies; packet loss {:.1f}%".format(
                ap.name, measured["received"], measured["sent"], measured["loss_pct"]), flush=True)
            if not (measured["ap_verified_at_both_ends"] and measured["all_requested_replies"]):
                raise RuntimeError("{} was requested, but its five-ping check did not fully pass. "
                                   "Inspect the saved results; this is not an AI failure.".format(ap.name))
        report["status"] = "passed"
        result_code = 0
        print("\nPASS: program connected to AP1, switched to AP2, and received all test replies.")
        print("This tests the switching action, NOT HOSN decisions or seamless handover.")
    except KeyboardInterrupt:
        report.update(status="interrupted", error="Stopped by user")
        print("\nInterrupted. Keeping partial logs.")
        result_code = 130
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
            report["cleanup_error"] = repr(exc)
            result_code = 1
            print("Cleanup reported an error; share the output before running another test.")
        finally:
            os.chdir(previous_dir)
            (folder / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)
                                                 + "\n", encoding="utf-8")
            try:
                restore_result_owner(folder)
            except OSError as exc:
                print("Could not restore result-file ownership:", exc)
            print("Saved results:", folder)
    return result_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    if sys.platform != "linux" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("Run inside Ubuntu: sudo python3 hosn_switch.py", file=sys.stderr)
        return 1
    for program in ("iw", "ip", "ping", "ovs-vsctl"):
        if shutil.which(program) is None:
            print("Required program missing:", program, file=sys.stderr)
            return 1
    # Check the fixed target before it is ever passed to a command.
    ipaddress.IPv4Address(SERVER_IP)
    try:
        return run_experiment()
    except Exception as exc:
        print("ERROR: {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        print("Share this message before changing packages or removing anything.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
