#!/usr/bin/env python3
"""Controlled RSSI-baseline vs HOSN Wi-Fi -> emulated-5G pilot.

Commands:
  python3 hosn_wifi_5g_compare.py --self-test
  python3 hosn_wifi_5g_compare.py --plan
  sudo python3 hosn_wifi_5g_compare.py --run

This is validation only. It creates no training rows and no labels. Both
strategies replay the same configured movement, access profiles, probe load,
traffic, netem seeds, and make-before-break executor. Only the decision policy
differs. HOSN decides from live observations; it is never switched on a timer.

The second access is an IP-path emulator shaped to an explicitly documented
healthy conversational-video service profile. It is not a 3GPP radio/core.
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

import hosn_wifi_5g_pilot as base


REVISION = "wifi-5g-mbb-controlled-comparison-v1"
TRAFFIC_DURATION_S = 16.0
PACKETS_PER_SECOND = 100.0
PAYLOAD_BYTES = 1200
FRAME_RATE = 25
PACKETS_PER_FRAME = 4
OVERLAP_MIN_S = 1.0
MIN_VERIFIED_CELL_PACKETS = 20
PROBE_COUNT = 10
CONTROLLER_PERIOD_S = 0.5
RSSI_BASELINE_THRESHOLD_DBM = -75.0
RSSI_CONFIRMATIONS = 2
SIGNAL_MARGIN_ADVANTAGE_DB = 8.0

# These are test inputs, never reported as measurements. The 5G-like profile
# is selected to sit inside 3GPP 5QI 2's 150 ms PDB and 10^-3 PER targets.
# Netem delay is one-way egress delay; measured end-to-end results are separate.
ACCESS_PROFILES = {
    "wifi_near": {"delay_ms": 2.0, "jitter_ms": 0.5, "loss_pct": 0.0, "rate_mbit": 40.0},
    "wifi_moving": {"delay_ms": 12.0, "jitter_ms": 3.0, "loss_pct": 1.0, "rate_mbit": 15.0},
    "wifi_edge": {"delay_ms": 35.0, "jitter_ms": 8.0, "loss_pct": 8.0, "rate_mbit": 3.0},
    "emulated_5g_healthy": {"delay_ms": 15.0, "jitter_ms": 3.0, "loss_pct": 0.1, "rate_mbit": 25.0},
}

MOVEMENT = [
    {"at_s": 0.0, "x_m": 20.0, "wifi_profile": "wifi_near"},
    {"at_s": 4.0, "x_m": 40.0, "wifi_profile": "wifi_moving"},
    {"at_s": 8.0, "x_m": 65.0, "wifi_profile": "wifi_edge"},
]

PLAN = {
    "revision": REVISION,
    "purpose": "small controlled pre-dataset comparison",
    "dataset_collection": False,
    "synthetic_result_rows": False,
    "strategies": ["conventional_rssi", "hosn_rules_first"],
    "run_order": ["conventional_rssi", "hosn_rules_first"],
    "controlled_conditions": {
        "same_movement": MOVEMENT,
        "same_access_profiles": ACCESS_PROFILES,
        "same_netem_seed_per_profile": True,
        "same_probe_procedure_until_each_policy_decides": True,
        "same_media_workload": True,
        "same_make_before_break_executor": True,
        "independent_replays": True,
    },
    "media_workload": {
        "description": "measured UDP frame/packet workload; not encoded video and not MOS/VMAF",
        "duration_s": TRAFFIC_DURATION_S,
        "packets_per_second": PACKETS_PER_SECOND,
        "payload_bytes": PAYLOAD_BYTES,
        "frames_per_second": FRAME_RATE,
        "packets_per_frame": PACKETS_PER_FRAME,
    },
    "baseline_policy": {
        "input": "modeled Wi-Fi RSSI only",
        "threshold_dbm": RSSI_BASELINE_THRESHOLD_DBM,
        "consecutive_samples": RSSI_CONFIRMATIONS,
    },
    "hosn_policy": {
        "inputs": ["technology-normalized signal margin", "measured RTT", "measured loss"],
        "rules_first": True,
        "ai_only_on_conflict": True,
        "unreviewed_ai_behavior": "HOLD; continue measuring",
        "scheduled_handover": False,
        "wifi_service_floor_dbm": -80.0,
        "emulated_cellular_service_floor_rsrp_dbm": -105.0,
        "emulated_cellular_rsrp_dbm": -90.0,
        "signal_margin_advantage_db": SIGNAL_MARGIN_ADVANTAGE_DB,
    },
    "make_before_break": {
        "candidate_preconfigured_and_probed": True,
        "duplicate_media_on_both_paths": True,
        "minimum_overlap_s": OVERLAP_MIN_S,
        "minimum_verified_candidate_packets": MIN_VERIFIED_CELL_PACKETS,
        "wifi_disconnect_after_candidate_verification": True,
        "receiver_deduplicates_for_goodput_but_raw_duplicates_are_reported": True,
    },
    "emulated_cellular_model": {
        "profile": ACCESS_PROFILES["emulated_5g_healthy"],
        "justification": "healthy test profile chosen within 3GPP 5QI 2 conversational-video delay/error targets",
        "standard_reference": "3GPP TS 23.501 table 5.7.4-1: 5QI 2, PDB 150 ms, PER 1e-3",
        "not_present": ["5G NR", "gNB", "5G core", "SIM", "physical RF"],
    },
    "reported_metrics": [
        "attempted/sent/raw/unique/duplicate packets", "actual loss",
        "one-way process delay", "RFC3550 interarrival jitter",
        "handover-window and overall interruption", "throughput", "goodput",
        "complete/damaged/missing frames", "longest incomplete-frame run",
        "gaps over 150 ms",
    ],
}


SENDER_CODE = r'''
import json, socket, sys, time, traceback
wifi_src, wifi_if, wifi_dst, cell_src, cell_if, cell_dst, port, start_at, stop_at, interval, payload_size, control, summary = sys.argv[1:]
port=int(port); start_at=float(start_at); stop_at=float(stop_at); interval=float(interval); payload_size=int(payload_size)
result={"status":"incomplete","scheduled_sequences":0,"successful_datagrams":0,"send_errors":0,"datagrams_by_path":{"wifi":0,"5g":0},"errors_by_path":{"wifi":0,"5g":0}}
socks={}
try:
  for path,src,intf in (("wifi",wifi_src,wifi_if),("5g",cell_src,cell_if)):
    s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET,socket.SO_BINDTODEVICE,(intf+"\0").encode("ascii")); s.bind((src,0)); socks[path]=s
  dst={"wifi":(wifi_dst,port),"5g":(cell_dst,port)}
  time.sleep(max(0,start_at-time.monotonic())); seq=0
  while start_at+seq*interval < stop_at:
    time.sleep(max(0,start_at+seq*interval-time.monotonic()))
    try:
      with open(control,encoding="utf-8") as h: mode=h.read().strip()
    except OSError: mode="invalid"
    paths={"wifi":["wifi"],"5g":["5g"],"duplicate":["wifi","5g"]}.get(mode,[])
    sent_ns=time.monotonic_ns(); frame=seq//4; part=seq%4
    for path in paths:
      header=(f"{seq}|{sent_ns}|{path}|{frame}|{part}|").encode("ascii")
      payload=(header+b"v"*max(0,payload_size-len(header)))[:payload_size]
      try:
        socks[path].sendto(payload,dst[path]); result["successful_datagrams"]+=1; result["datagrams_by_path"][path]+=1
      except OSError:
        result["send_errors"]+=1; result["errors_by_path"][path]+=1
    if not paths: result["send_errors"]+=1
    seq+=1
  result["scheduled_sequences"]=seq; result["status"]="complete"
except BaseException as exc:
  result.update(status="failed",error=type(exc).__name__+": "+str(exc),traceback=traceback.format_exc())
finally:
  for s in socks.values(): s.close()
  with open(summary,"w",encoding="utf-8") as h: json.dump(result,h,indent=2,allow_nan=False); h.write("\n")
'''

RECEIVER_CODE = r'''
import json, socket, sys, time, traceback
port,stop_at,packets,summary=sys.argv[1:]; port=int(port); stop_at=float(stop_at)
result={"status":"incomplete","raw_datagrams":0,"parse_errors":0}
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,8*1024*1024); s.settimeout(.2)
try:
  s.bind(("0.0.0.0",port))
  with open(packets,"w",encoding="utf-8") as out:
    while time.monotonic()<stop_at:
      try: payload,source=s.recvfrom(65535)
      except socket.timeout: continue
      received=time.monotonic_ns()
      try:
        seq,sent,path,frame,part,_=payload.split(b"|",5)
        event={"sequence":int(seq),"sent_monotonic_ns":int(sent),"received_monotonic_ns":received,"declared_path":path.decode("ascii"),"frame":int(frame),"part":int(part),"source_ip":source[0],"payload_bytes":len(payload)}
        out.write(json.dumps(event,allow_nan=False)+"\n"); out.flush(); result["raw_datagrams"]+=1
      except (ValueError,UnicodeError): result["parse_errors"]+=1
  result["status"]="complete"
except BaseException as exc:
  result.update(status="failed",error=type(exc).__name__+": "+str(exc),traceback=traceback.format_exc())
finally:
  s.close()
  with open(summary,"w",encoding="utf-8") as h: json.dump(result,h,indent=2,allow_nan=False); h.write("\n")
'''


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def profile_tc_args(interface: str, profile: dict, seed: int) -> list[str]:
    for key in ("delay_ms", "jitter_ms", "loss_pct", "rate_mbit"):
        value = profile.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid profile value: " + key)
    if profile["rate_mbit"] <= 0 or profile["loss_pct"] > 100 or not isinstance(seed, int) or seed <= 0:
        raise ValueError("Invalid profile range or seed")
    return ["tc","qdisc","replace","dev",interface,"root","netem",
            "delay",f'{profile["delay_ms"]}ms',f'{profile["jitter_ms"]}ms',"distribution","normal",
            "loss","random",f'{profile["loss_pct"]}%',"rate",f'{profile["rate_mbit"]}mbit',"seed",str(seed)]


def configure_profile(node, interface: str, name: str, seed: int, run) -> dict:
    argv = profile_tc_args(interface, ACCESS_PROFILES[name], seed)
    text, code = run(node, argv)
    if code:
        raise RuntimeError("tc/netem profile failed (iproute2 must support 'seed'): " + text.strip())
    shown, code = run(node, ["tc", "qdisc", "show", "dev", interface])
    if code or "netem" not in shown:
        raise RuntimeError("Could not verify netem on " + interface)
    return {"name": name, "configured_not_measured": ACCESS_PROFILES[name], "seed": seed, "tc_show": shown.strip()}


def heterogeneous_hosn_decision(wifi_rssi: float, wifi_probe: dict, cell_probe: dict) -> dict:
    wifi_margin = wifi_rssi - PLAN["hosn_policy"]["wifi_service_floor_dbm"]
    cell_rsrp = PLAN["hosn_policy"]["emulated_cellular_rsrp_dbm"]
    cell_margin = cell_rsrp - PLAN["hosn_policy"]["emulated_cellular_service_floor_rsrp_dbm"]
    evidence = {"wifi_rssi_model_dbm": wifi_rssi, "cell_rsrp_configured_dbm": cell_rsrp,
                "wifi_link_margin_db": wifi_margin, "cell_link_margin_db": cell_margin,
                "wifi_probe": wifi_probe, "cell_probe": cell_probe}
    if cell_probe["received"] == 0:
        return {"action":"STAY","source":"SAFETY","reason":"candidate_unreachable","evidence":evidence}
    wl, cl = wifi_probe["loss_pct"], cell_probe["loss_pct"]
    wr, cr = wifi_probe["rtt_avg_ms"], cell_probe["rtt_avg_ms"]
    candidate_signal_better = cell_margin >= wifi_margin + SIGNAL_MARGIN_ADVANTAGE_DB
    candidate_qos_no_worse = (wr is None or cr <= wr) and cl <= wl
    if candidate_signal_better and candidate_qos_no_worse:
        return {"action":"HANDOVER","source":"RULES","reason":"candidate_margin_and_qos_clear","evidence":evidence}
    current_has_no_signal_disadvantage = wifi_margin + SIGNAL_MARGIN_ADVANTAGE_DB > cell_margin
    current_qos_no_worse = (cr is None or (wr is not None and wr <= cr)) and wl <= cl
    if current_has_no_signal_disadvantage and current_qos_no_worse:
        return {"action":"STAY","source":"RULES","reason":"current_access_clear","evidence":evidence}
    return {"action":"STAY","source":"AI_UNAVAILABLE_HOLD","reason":"rules_conflict_no_reviewed_ai","evidence":evidence}


def baseline_decision(rssi: float, consecutive: int) -> tuple[dict, int]:
    consecutive = consecutive + 1 if rssi <= RSSI_BASELINE_THRESHOLD_DBM else 0
    action = "HANDOVER" if consecutive >= RSSI_CONFIRMATIONS else "STAY"
    return ({"action":action,"source":"RSSI_BASELINE","reason":"threshold_confirmed" if action == "HANDOVER" else "threshold_not_confirmed",
             "evidence":{"wifi_rssi_model_dbm":rssi,"threshold_dbm":RSSI_BASELINE_THRESHOLD_DBM,"consecutive":consecutive}}, consecutive)


def rfc3550_jitter_ms(events: list[dict]) -> Optional[float]:
    if len(events) < 2:
        return None
    ordered = sorted(events, key=lambda e: e["received_monotonic_ns"])
    jitter = 0.0
    previous = (ordered[0]["received_monotonic_ns"] - ordered[0]["sent_monotonic_ns"]) / 1e6
    for event in ordered[1:]:
        transit = (event["received_monotonic_ns"] - event["sent_monotonic_ns"]) / 1e6
        jitter += (abs(transit - previous) - jitter) / 16.0
        previous = transit
    return jitter


def summarize(sender: dict, receiver: dict, events: list[dict], timing: dict) -> dict:
    if sender.get("status") != "complete" or receiver.get("status") != "complete":
        raise ValueError("sender/receiver incomplete")
    ordered = sorted(events, key=lambda e: e["received_monotonic_ns"])
    unique_by_seq = {}
    for event in ordered:
        unique_by_seq.setdefault(event["sequence"], event)
    unique = sorted(unique_by_seq.values(), key=lambda e: e["received_monotonic_ns"])
    attempted = int(sender["scheduled_sequences"]); received = len(unique)
    delays = [(e["received_monotonic_ns"] - e["sent_monotonic_ns"]) / 1e6 for e in unique]
    gaps = [(b["received_monotonic_ns"] - a["received_monotonic_ns"]) / 1e6 for a,b in zip(unique,unique[1:])]
    decision_ns, break_ns = timing["decision_ns"], timing["wifi_break_ns"]
    window = [e for e in unique if decision_ns - 500_000_000 <= e["received_monotonic_ns"] <= break_ns + 1_000_000_000]
    window_gaps = [(b["received_monotonic_ns"] - a["received_monotonic_ns"]) / 1e6 for a,b in zip(window,window[1:])]
    first_cell = next((e for e in ordered if e["declared_path"] == "5g" and e["received_monotonic_ns"] >= decision_ns), None)
    paths = Counter(e["declared_path"] for e in ordered)
    sources = {p: sorted({e["source_ip"] for e in ordered if e["declared_path"] == p}) for p in ("wifi","5g")}
    total_frames = math.ceil(attempted / PACKETS_PER_FRAME)
    received_parts = {frame:set() for frame in range(total_frames)}
    for e in unique:
        if 0 <= e["frame"] < total_frames: received_parts[e["frame"]].add(e["part"])
    complete = [len(received_parts[f]) == PACKETS_PER_FRAME for f in range(total_frames)]
    damaged = sum(0 < len(received_parts[f]) < PACKETS_PER_FRAME for f in range(total_frames))
    missing = sum(len(received_parts[f]) == 0 for f in range(total_frames))
    longest = run = 0
    for okay in complete:
        run = 0 if okay else run + 1; longest = max(longest, run)
    duration = TRAFFIC_DURATION_S
    loss_packets = max(0, attempted - received)
    checks = {"both_paths_received":paths["wifi"]>0 and paths["5g"]>0,
              "sources_verified":sources["wifi"]==[base.WIFI_UE_IP] and sources["5g"]==[base.CELL_UE_IP],
              "candidate_verified_before_break":timing["candidate_verified_ns"] <= break_ns,
              "minimum_overlap_met":(break_ns-timing["duplicate_start_ns"])/1e9 >= OVERLAP_MIN_S,
              "wifi_actually_disconnected":timing["wifi_disconnected_verified"],
              "no_parse_errors":receiver["parse_errors"]==0,
              "handover_was_policy_decision":timing["decision_source"] in ("RSSI_BASELINE","RULES")}
    return {"attempted_packets":attempted,"successful_datagrams_sent":sender["successful_datagrams"],
            "raw_datagrams_received":len(ordered),"unique_packets_received":received,
            "duplicate_datagrams_received":len(ordered)-received,"actual_lost_sequences":loss_packets,
            "actual_loss_pct":100*loss_packets/attempted if attempted else 100.0,
            "raw_received_by_path":dict(paths),"sources_by_path":sources,
            "one_way_process_delay_ms":{"mean":statistics.fmean(delays) if delays else None,"p95":percentile(delays,.95),"max":max(delays) if delays else None},
            "rfc3550_interarrival_jitter_ms":rfc3550_jitter_ms(unique),
            "overall_max_interarrival_gap_ms":max(gaps) if gaps else None,
            "handover_window_max_interarrival_gap_ms":max(window_gaps) if window_gaps else None,
            "gaps_over_150ms":sum(g>150 for g in gaps),
            "decision_to_first_cell_packet_ms":((first_cell["received_monotonic_ns"]-decision_ns)/1e6 if first_cell else None),
            "mbb_overlap_ms":(break_ns-timing["duplicate_start_ns"])/1e6,
            "network_throughput_mbps":len(ordered)*PAYLOAD_BYTES*8/duration/1e6,
            "application_goodput_mbps":received*PAYLOAD_BYTES*8/duration/1e6,
            "frames":{"expected":total_frames,"complete":sum(complete),"damaged":damaged,"missing":missing,
                      "complete_pct":100*sum(complete)/total_frames if total_frames else 0,
                      "longest_incomplete_run_frames":longest,"longest_incomplete_run_ms":1000*longest/FRAME_RATE},
            "qoe_interpretation":"packet/frame QoE indicators only; no encoded-video MOS or VMAF claim",
            "qoe_proxy_5qi2_pass":bool(delays and percentile(delays,.95) <= 150 and loss_packets/attempted <= .001 and not any(g>150 for g in gaps)),
            "checks":checks,"pilot_valid":all(checks.values())}


def read_events(path: Path, partial: bool=False) -> list[dict]:
    if not path.exists(): return []
    events=[]; lines=path.read_text(encoding="utf-8").splitlines()
    for i,line in enumerate(lines):
        try: events.append(json.loads(line))
        except json.JSONDecodeError:
            if partial and i == len(lines)-1: continue
            raise
    return events


def wait_for_cell_packets(path: Path, after_ns: int, minimum: int, timeout: float=4.0) -> int:
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        count=sum(e["declared_path"]=="5g" and e["received_monotonic_ns"]>=after_ns for e in read_events(path,True))
        if count>=minimum: return count
        time.sleep(.05)
    return 0


def disconnect_wifi(station, ap, run, helper) -> None:
    intf=station.wintfs[0].name
    text,code=run(station,["iw","dev",intf,"disconnect"])
    if code and helper.actual_link(station,run)[0]: raise RuntimeError("Wi-Fi disconnect failed: "+text.strip())
    helper.wait_for_link(station,"",run,timeout=4.0); helper.sync_observed_record(station,(ap,),"")


def run_replay(strategy: str, replay_dir: Path, station, server, wifi_core, cell_gateway, ap, run, helper) -> dict:
    wifi_if=station.wintfs[0].name
    station.position=[20.0,40.0,0.0]
    helper.connect_verified(station,ap,(ap,),run)
    profile_log={"cell":configure_profile(station,"sta1-5g0","emulated_5g_healthy",9001,run)}
    decision_log=[]; movement_log=[]; handover_stage_index=None
    seed_map={"wifi_near":1001,"wifi_moving":1002,"wifi_edge":1003}
    sender_script=replay_dir/"sender.py"; receiver_script=replay_dir/"receiver.py"
    sender_script.write_text(SENDER_CODE,encoding="utf-8"); receiver_script.write_text(RECEIVER_CODE,encoding="utf-8")
    control=replay_dir/"selected_access.txt"; control.write_text("wifi\n",encoding="utf-8")
    packets=replay_dir/"packet_arrivals.jsonl"; send_summary=replay_dir/"sender_summary.json"; recv_summary=replay_dir/"receiver_summary.json"
    send_out=(replay_dir/"sender_output.txt").open("w",encoding="utf-8"); recv_out=(replay_dir/"receiver_output.txt").open("w",encoding="utf-8")
    sender=receiver=None
    try:
        start=time.monotonic()+1.5; stop=start+TRAFFIC_DURATION_S
        receiver=server.popen([sys.executable,str(receiver_script),str(base.UDP_PORT),str(stop+1),str(packets),str(recv_summary)],stdout=recv_out,stderr=subprocess.STDOUT)
        sender=station.popen([sys.executable,str(sender_script),base.WIFI_UE_IP,wifi_if,base.WIFI_SERVER_IP,base.CELL_UE_IP,"sta1-5g0",base.CELL_SERVER_IP,str(base.UDP_PORT),str(start),str(stop),str(1/PACKETS_PER_SECOND),str(PAYLOAD_BYTES),str(control),str(send_summary)],stdout=send_out,stderr=subprocess.STDOUT)
        handover=None; confirmations=0
        for stage_index,stage in enumerate(MOVEMENT):
            base.sleep_until(start+stage["at_s"])
            station.position=[stage["x_m"],40.0,0.0]
            profile_log[stage["wifi_profile"]]=configure_profile(station,wifi_if,stage["wifi_profile"],seed_map[stage["wifi_profile"]],run)
            movement_log.append({"scheduled_at_s":stage["at_s"],"applied_at_s":time.monotonic()-start,
                                 "x_m":stage["x_m"],"wifi_profile":stage["wifi_profile"],
                                 "wifi_rssi_model_dbm":base.modeled_wifi_rssi(station,ap)})
            # Identical probes in both strategies; baseline records but ignores QoS.
            wifi_probe=base.ping_path(station,wifi_if,base.WIFI_SERVER_IP,run,replay_dir/(stage["wifi_profile"]+"_wifi_ping.txt"),helper,count=PROBE_COUNT)
            cell_probe=base.ping_path(station,"sta1-5g0",base.CELL_SERVER_IP,run,replay_dir/(stage["wifi_profile"]+"_cell_ping.txt"),helper,count=PROBE_COUNT)
            for _ in range(RSSI_CONFIRMATIONS):
                rssi=base.modeled_wifi_rssi(station,ap)
                if strategy=="conventional_rssi": decision,confirmations=baseline_decision(rssi,confirmations)
                else: decision=heterogeneous_hosn_decision(rssi,wifi_probe,cell_probe)
                decision.update(elapsed_s=time.monotonic()-start,stage=stage["wifi_profile"]); decision_log.append(decision)
                if decision["action"]=="HANDOVER": handover=decision; handover_stage_index=stage_index; break
                time.sleep(CONTROLLER_PERIOD_S)
            if handover: break
        if not handover: raise RuntimeError(strategy+" never decided to hand over")
        decision_ns=time.monotonic_ns(); control.write_text("duplicate\n",encoding="utf-8"); duplicate_ns=time.monotonic_ns()
        verified=wait_for_cell_packets(packets,duplicate_ns,MIN_VERIFIED_CELL_PACKETS)
        if verified<MIN_VERIFIED_CELL_PACKETS: raise RuntimeError("candidate media path was not verified during overlap")
        candidate_ns=time.monotonic_ns()
        base.sleep_until(duplicate_ns/1e9+OVERLAP_MIN_S)
        control.write_text("5g\n",encoding="utf-8"); disconnect_wifi(station,ap,run,helper); break_ns=time.monotonic_ns()
        # Movement remains identical after an early policy decision. The Wi-Fi
        # profile is still replayed even though application traffic now uses 5G.
        for stage in MOVEMENT[(handover_stage_index or 0)+1:]:
            base.sleep_until(start+stage["at_s"])
            station.position=[stage["x_m"],40.0,0.0]
            profile_log[stage["wifi_profile"]]=configure_profile(station,wifi_if,stage["wifi_profile"],seed_map[stage["wifi_profile"]],run)
            movement_log.append({"scheduled_at_s":stage["at_s"],"applied_at_s":time.monotonic()-start,
                                 "x_m":stage["x_m"],"wifi_profile":stage["wifi_profile"],
                                 "wifi_rssi_model_dbm":base.modeled_wifi_rssi(station,ap),
                                 "after_handover":True})
        base.sleep_until(stop+1.2); base.process_wait(sender,"sender"); base.process_wait(receiver,"receiver")
        send_out.close(); recv_out.close(); send_out=recv_out=None
        timing={"decision_ns":decision_ns,"duplicate_start_ns":duplicate_ns,"candidate_verified_ns":candidate_ns,"wifi_break_ns":break_ns,
                "wifi_disconnected_verified":helper.actual_link(station,run)[0]=="","decision_source":handover["source"]}
        summary=summarize(json.loads(send_summary.read_text()),json.loads(recv_summary.read_text()),read_events(packets),timing)
        summary.update(strategy=strategy,decision=handover,decision_log=decision_log,timing=timing,profiles=profile_log,movement=movement_log)
        write_json(replay_dir/"summary.json",summary); return summary
    finally:
        for proc in (sender,receiver):
            if proc is not None and proc.poll() is None: proc.kill(); proc.wait(timeout=3)
        for handle in (send_out,recv_out):
            if handle is not None: handle.close()


def run_comparison(root: Path) -> int:
    import hosn_switch as helper
    import mn_wifi.net as wifi_module
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder=root/"results"/("wifi_5g_mbb_comparison_"+stamp); folder.mkdir(parents=True)
    write_json(folder/"plan.json",PLAN)
    manifest={"revision":REVISION,"status":"starting","created_utc":stamp,"data_kind":"controlled_pilot_not_training_data","labels_assigned":0,"model_loaded":False,
              "real_3gpp_stack":False,"python":platform.python_version(),"kernel":platform.release(),"machine":platform.machine(),
              "mininet_wifi_version":str(getattr(wifi_module,"VERSION","unknown")),
              "source_sha256":{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ("hosn_wifi_5g_compare.py","hosn_wifi_5g_pilot.py","hosn_switch.py")}}
    write_json(folder/"manifest.json",manifest)
    network=None; old=Path.cwd()
    try:
        os.chdir(folder); network,station,server,wifi_core,cell_gateway,ap=base.create_network(); run=helper.NodeCommands(folder/"commands.jsonl")
        wifi_if=station.wintfs[0].name
        base.verified_interface(station,cell_gateway,"sta1-5g0"); base.verified_interface(server,wifi_core,"h1-wifi0"); base.verified_interface(server,cell_gateway,"h1-5g0")
        base.configure_ip(station,wifi_if,base.WIFI_UE_IP+"/24",run); base.configure_ip(station,"sta1-5g0",base.CELL_UE_IP+"/24",run)
        base.configure_ip(server,"h1-wifi0",base.WIFI_SERVER_IP+"/24",run); base.configure_ip(server,"h1-5g0",base.CELL_SERVER_IP+"/24",run)
        results={}
        for strategy in PLAN["run_order"]:
            replay=folder/strategy; replay.mkdir(); print("\nRunning",strategy,"replay...",flush=True)
            results[strategy]=run_replay(strategy,replay,station,server,wifi_core,cell_gateway,ap,run,helper)
            s=results[strategy]; print("  loss={:.3f}% gap={:.3f}ms goodput={:.3f}Mbps decision={}".format(s["actual_loss_pct"],s["handover_window_max_interarrival_gap_ms"],s["application_goodput_mbps"],s["decision"]["source"]),flush=True)
        all_valid=all(item["pilot_valid"] for item in results.values())
        comparison={"revision":REVISION,"controlled_pilot_only":True,"final_dataset_started":False,
                    "mechanical_checks_pass":all_valid,"results":results,
                    "honesty_note":"Independent real packet replays under identical configured conditions; results may differ because packet scheduling is stochastic despite fixed netem seeds.",
                    "winner_not_forced":True}
        write_json(folder/"comparison.json",comparison); manifest["status"]="pilot_complete" if all_valid else "pilot_checks_failed"; manifest["comparison_file"]="comparison.json"; write_json(folder/"manifest.json",manifest)
        print("\nCONTROLLED PILOT {} — final dataset NOT started.".format("COMPLETE" if all_valid else "CHECKS FAILED"))
        print("Saved:",folder); return 0 if all_valid else 1
    except KeyboardInterrupt:
        manifest["status"]="interrupted"; return 130
    except Exception as exc:
        manifest.update(status="failed",error=type(exc).__name__+": "+str(exc)); (folder/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        print("\nPILOT STOPPED:",exc); return 1
    finally:
        if network is not None: network.stop()
        os.chdir(old); write_json(folder/"manifest.json",manifest)
        try: helper.restore_result_owner(folder)
        except Exception: pass


def run_self_tests() -> int:
    import unittest
    class Tests(unittest.TestCase):
        def test_no_dataset(self): self.assertFalse(PLAN["dataset_collection"]); self.assertFalse(PLAN["synthetic_result_rows"])
        def test_cell_profile_is_explicit(self):
            p=ACCESS_PROFILES["emulated_5g_healthy"]; self.assertEqual(set(p),{"delay_ms","jitter_ms","loss_pct","rate_mbit"}); self.assertEqual(p["loss_pct"],.1)
        def test_seeded_netem(self):
            args=profile_tc_args("x0",ACCESS_PROFILES["wifi_edge"],123); self.assertIn("seed",args); self.assertIn("normal",args)
        def test_baseline_confirmation(self):
            d,n=baseline_decision(-76,0); self.assertEqual(d["action"],"STAY"); d,n=baseline_decision(-76,n); self.assertEqual(d["action"],"HANDOVER")
        def test_hosn_clear_and_conflict(self):
            wp={"sent":20,"received":16,"loss_pct":20.0,"rtt_avg_ms":80.0}; cp={"sent":20,"received":20,"loss_pct":0.0,"rtt_avg_ms":30.0}
            self.assertEqual(heterogeneous_hosn_decision(-86,wp,cp)["source"],"RULES")
            wp.update(received=20,loss_pct=0.0,rtt_avg_ms=20.0); self.assertEqual(heterogeneous_hosn_decision(-75,wp,cp)["source"],"AI_UNAVAILABLE_HOLD")
        def test_summary_counts_duplicates_and_loss(self):
            sender={"status":"complete","scheduled_sequences":8,"successful_datagrams":10}; receiver={"status":"complete","parse_errors":0}
            ev=[]
            for seq in range(7):
                for path in (["wifi","5g"] if seq in (3,4) else ["wifi" if seq<4 else "5g"]):
                    sent=1_000_000_000+seq*10_000_000; ev.append({"sequence":seq,"sent_monotonic_ns":sent,"received_monotonic_ns":sent+2_000_000+(1 if path=="5g" else 0),"declared_path":path,"source_ip":base.WIFI_UE_IP if path=="wifi" else base.CELL_UE_IP,"payload_bytes":1200,"frame":seq//4,"part":seq%4})
            t={"decision_ns":1_020_000_000,"duplicate_start_ns":1_020_000_000,"candidate_verified_ns":1_030_000_000,"wifi_break_ns":2_020_000_000,"wifi_disconnected_verified":True,"decision_source":"RULES"}
            s=summarize(sender,receiver,ev,t); self.assertEqual(s["duplicate_datagrams_received"],2); self.assertEqual(s["actual_lost_sequences"],1); self.assertAlmostEqual(s["actual_loss_pct"],12.5)
        def test_strict_json(self):
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError): write_json(Path(d)/"x",{"bad":float("nan")})
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if result.wasSuccessful(): print(f"PASS: {result.testsRun} software checks. No network run; no dataset created.")
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser=argparse.ArgumentParser(description="Controlled MBB baseline-vs-HOSN pilot; never a dataset collector")
    mode=parser.add_mutually_exclusive_group(required=True); mode.add_argument("--self-test",action="store_true"); mode.add_argument("--plan",action="store_true"); mode.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if args.self_test: return run_self_tests()
    if args.plan: print(json.dumps(PLAN,indent=2,allow_nan=False)); return 0
    if sys.platform!="linux" or not hasattr(os,"geteuid") or os.geteuid()!=0: parser.exit(2,"Run --run in Ubuntu with sudo.\n")
    missing=[x for x in ("ip","iw","ping","tc") if not shutil.which(x)]
    if missing: parser.exit(2,"Missing required tools: "+", ".join(missing)+"\n")
    root=Path(__file__).resolve().parent
    needed=("hosn_switch.py","hosn_wifi_5g_pilot.py")
    if any(not (root/x).is_file() for x in needed): parser.exit(2,"Keep this file beside "+" and ".join(needed)+".\n")
    return run_comparison(root)


if __name__=="__main__": raise SystemExit(main())
