#!/usr/bin/env python3
"""Frozen-model HOSN stress validation for common network problems.

This is an evaluation harness, not a trainer.  It keeps the accepted HOSN
model frozen, creates no training rows or labels, and never supplies the
predeclared expected action to a controller.

The full plan contains 10 common-problem families, five unseen configured
variants per family, five independent netem seed requests, and the three
separate strategies (RSSI baseline, rules-only HOSN, and full HOSN).  That is
50 scenarios and 750 measured replays.  Thirty scenarios are deliberate rule
conflicts, producing 150 full-HOSN replays in which the AI is expected to be
called.  AI-only conflict accuracy is reported separately from full-pipeline
accuracy so clear rule decisions cannot inflate the AI result.

Commands:
  python3 hosn_stress_validation.py --self-test
  python3 hosn_stress_validation.py --plan
  sudo python3 hosn_stress_validation.py --smoke-run
  sudo python3 hosn_stress_validation.py --run
  sudo python3 hosn_stress_validation.py --resume-latest
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import sys
import traceback
from typing import Any, Iterable, Optional

import hosn_final_comparison as final
import hosn_measured_dataset as measured
import hosn_paired_outcome_pilot as paired
import hosn_strategy_router as router
import hosn_wifi_5g_compare as compare
import hosn_wifi_5g_pilot as base


REVISION = "common-problem-frozen-model-stress-v1"
GENERATOR_SEED = 20261004
SEEDS_PER_SCENARIO = 5
STRATEGIES = router.STRATEGIES
FULL_PREFIX = "stress_validation_"
SMOKE_PREFIX = "stress_smoke_"

# Ranges are configured netem conditions, not measured results.  Each family
# has five deterministic random variants so the exact training/evaluation
# profile values are not reused.  "Crowded" and "interference" are emulated by
# delay, jitter, loss, and rate shaping; no physical interferer or population
# of real client devices is claimed.
FAMILY_SPECS = (
    {
        "id": "crowded_wifi",
        "problem": "Crowded Wi-Fi / many active users (emulated load)",
        "zone": "near", "case_type": "conflict", "expected": paired.HANDOVER,
        "basis": "Wi-Fi signal remains strong but congestion makes the candidate path better.",
        "wifi": ((55, 85), (8, 18), (3, 8), (3, 7)),
        "cell": ((12, 25), (1, 5), (0.1, 1.0), (18, 30)),
    },
    {
        "id": "wifi_interference",
        "problem": "Wi-Fi interference and fading",
        "zone": "near", "case_type": "conflict", "expected": paired.HANDOVER,
        "basis": "Strong Wi-Fi RSSI hides substantial loss and jitter.",
        "wifi": ((25, 50), (12, 25), (9, 20), (4, 10)),
        "cell": ((15, 30), (2, 6), (0.1, 1.5), (15, 28)),
    },
    {
        "id": "wifi_latency_spike",
        "problem": "Wi-Fi latency and jitter spike",
        "zone": "near", "case_type": "conflict", "expected": paired.HANDOVER,
        "basis": "Strong Wi-Fi RSSI hides a delay spike that harms conversational traffic.",
        "wifi": ((80, 135), (18, 35), (0.5, 3.0), (5, 12)),
        "cell": ((18, 38), (2, 7), (0.1, 1.5), (14, 25)),
    },
    {
        "id": "backup_congestion",
        "problem": "Congested candidate/backup network",
        "zone": "edge", "case_type": "conflict", "expected": paired.STAY,
        "basis": "Wi-Fi RSSI is weak, but Wi-Fi QoS is better than the congested candidate.",
        "wifi": ((5, 16), (1, 4), (0.0, 1.0), (18, 30)),
        "cell": ((75, 125), (14, 28), (4, 10), (3, 8)),
    },
    {
        "id": "backup_interference",
        "problem": "Candidate/backup packet loss and interference",
        "zone": "edge", "case_type": "conflict", "expected": paired.STAY,
        "basis": "The candidate has more signal margin but substantially worse packet loss.",
        "wifi": ((8, 22), (1, 5), (0.0, 1.5), (15, 28)),
        "cell": ((28, 58), (8, 18), (10, 22), (4, 10)),
    },
    {
        "id": "backup_latency_spike",
        "problem": "Candidate/backup latency and jitter spike",
        "zone": "edge", "case_type": "conflict", "expected": paired.STAY,
        "basis": "The candidate has more signal margin but delay is much worse.",
        "wifi": ((7, 20), (1, 5), (0.0, 1.5), (16, 28)),
        "cell": ((85, 145), (20, 38), (1, 4), (5, 12)),
    },
    {
        "id": "weak_wifi_coverage",
        "problem": "Weak Wi-Fi coverage at the cell edge",
        "zone": "edge", "case_type": "clear", "expected": paired.HANDOVER,
        "basis": "The candidate has more normalized signal margin and better QoS.",
        "wifi": ((50, 85), (10, 20), (5, 12), (3, 8)),
        "cell": ((12, 28), (2, 6), (0.0, 1.0), (18, 30)),
    },
    {
        "id": "severe_wifi_drop",
        "problem": "Severe Wi-Fi quality drop / near-outage",
        "zone": "edge", "case_type": "clear", "expected": paired.HANDOVER,
        "basis": "Wi-Fi remains measurable but suffers severe loss and delay.",
        "wifi": ((90, 150), (20, 40), (20, 40), (1.5, 4)),
        "cell": ((15, 35), (2, 7), (0.1, 1.5), (15, 25)),
    },
    {
        "id": "poor_backup_healthy_wifi",
        "problem": "Healthy Wi-Fi with a poor candidate network",
        "zone": "near", "case_type": "clear", "expected": paired.STAY,
        "basis": "Wi-Fi has stronger normalized margin and better QoS.",
        "wifi": ((4, 15), (0.5, 3), (0.0, 0.8), (22, 40)),
        "cell": ((60, 110), (12, 25), (6, 15), (3, 9)),
    },
    {
        "id": "both_paths_busy",
        "problem": "Both paths busy, with Wi-Fi less degraded",
        "zone": "near", "case_type": "clear", "expected": paired.STAY,
        "basis": "Both paths are impaired, but switching would make QoS worse.",
        "wifi": ((35, 60), (7, 14), (2, 5), (7, 14)),
        "cell": ((70, 115), (15, 30), (8, 18), (3, 8)),
    },
)

# The fifth context pauses at the decision location.  This adds a zero-trend
# case alongside inbound/outbound and fast/slow mobility without inventing a
# second decision or claiming ping-pong testing.
VARIANT_CONTEXTS = (
    ("outbound", "fast", -3.0, False),
    ("outbound", "slow", 0.0, False),
    ("inbound", "fast", 3.0, False),
    ("inbound", "slow", -2.0, False),
    ("outbound", "paused", 2.0, True),
)

SEED_ORDERS = (
    (router.RSSI_BASELINE, router.HOSN_RULES_ONLY, router.HOSN_FULL_AI),
    (router.HOSN_RULES_ONLY, router.HOSN_FULL_AI, router.RSSI_BASELINE),
    (router.HOSN_FULL_AI, router.RSSI_BASELINE, router.HOSN_RULES_ONLY),
    (router.RSSI_BASELINE, router.HOSN_FULL_AI, router.HOSN_RULES_ONLY),
    (router.HOSN_FULL_AI, router.HOSN_RULES_ONLY, router.RSSI_BASELINE),
)


def _draw_profile(rng: random.Random, ranges: tuple[tuple[float, float], ...]) -> dict:
    names = ("delay_ms", "jitter_ms", "loss_pct", "rate_mbit")
    return {
        name: round(rng.uniform(low, high), 3)
        for name, (low, high) in zip(names, ranges)
    }


def build_stress_matrix() -> tuple[tuple[dict, ...], dict[str, dict]]:
    rng = random.Random(GENERATOR_SEED)
    scenarios: list[dict] = []
    profiles: dict[str, dict] = {}
    scenario_index = 0
    for family in FAMILY_SPECS:
        for variant, (direction, speed, x_adjust, paused) in enumerate(
            VARIANT_CONTEXTS, start=1
        ):
            scenario_index += 1
            wifi_name = "stress_wifi_{}_{:02d}".format(family["id"], variant)
            cell_name = "stress_cell_{}_{:02d}".format(family["id"], variant)
            profiles[wifi_name] = _draw_profile(rng, family["wifi"])
            profiles[cell_name] = _draw_profile(rng, family["cell"])
            item = final.scenario(
                "stress_{:02d}_{}_v{}".format(
                    scenario_index, family["id"], variant
                ),
                direction, speed, family["zone"], wifi_name, cell_name,
                family["case_type"], family["expected"],
                10000 + scenario_index * 100,
            )
            # Small position changes prevent five copies of an identical
            # signal state while keeping the declared near/edge context.
            item["movement"][-1]["x_m"] += x_adjust
            if paused:
                item["movement"][-2]["x_m"] = item["movement"][-1]["x_m"]
            item.update(
                problem_family=family["id"],
                common_problem=family["problem"],
                expected_action_basis=family["basis"],
                variant_number=variant,
                configured_wifi_profile=profiles[wifi_name],
                configured_cell_profile=profiles[cell_name],
            )
            scenarios.append(item)
    return tuple(scenarios), profiles


STRESS_SCENARIOS, STRESS_PROFILES = build_stress_matrix()
paired.ALL_ACCESS_PROFILES.update(STRESS_PROFILES)


def execution_specs(scenarios: Iterable[dict] = STRESS_SCENARIOS,
                    seeds: int = SEEDS_PER_SCENARIO) -> list[dict]:
    output = []
    for scenario in scenarios:
        for seed_number in range(1, seeds + 1):
            order = SEED_ORDERS[seed_number - 1]
            for order_position, strategy in enumerate(order, start=1):
                configured = dict(scenario)
                configured["seed_offset"] = (
                    int(scenario["seed_offset"]) + seed_number * 10
                )
                output.append({
                    "scenario": configured,
                    "scenario_id": scenario["id"],
                    "seed_number": seed_number,
                    "strategy": strategy,
                    "order_position": order_position,
                    "replay_id": "seed{:02d}_{}".format(
                        seed_number, strategy.lower()
                    ),
                })
    return output


def _scenario_public(item: dict) -> dict:
    return {
        key: value for key, value in item.items()
        if key != "expected_action_for_evaluation_only"
    } | {
        "expected_action_for_evaluation_only": item[
            "expected_action_for_evaluation_only"
        ]
    }


PLAN = {
    "revision": REVISION,
    "purpose": "frozen-model stress validation under common network problems",
    "data_origin": "Mininet-WiFi plus emulated cellular/5G-like IP path",
    "physical_interference_or_real_multiuser_load": False,
    "synthetic_training_data": False,
    "training_performed": False,
    "training_rows_created": 0,
    "training_labels_assigned": 0,
    "model_frozen": True,
    "expected_action_is_controller_input": False,
    "common_problem_families": [
        {"id": item["id"], "problem": item["problem"]}
        for item in FAMILY_SPECS
    ],
    "scenario_count": len(STRESS_SCENARIOS),
    "variants_per_problem": len(VARIANT_CONTEXTS),
    "seeds_per_scenario": SEEDS_PER_SCENARIO,
    "strategies": list(STRATEGIES),
    "planned_replays": len(STRESS_SCENARIOS) * SEEDS_PER_SCENARIO * len(STRATEGIES),
    "planned_full_hosn_conflict_replays": (
        sum(item["case_type"] == "conflict" for item in STRESS_SCENARIOS)
        * SEEDS_PER_SCENARIO
    ),
    "scenarios": [_scenario_public(item) for item in STRESS_SCENARIOS],
    "honesty_note": (
        "Configured netem profiles emulate common symptoms such as congestion, "
        "interference, and multiuser load. They are not physical interferers, "
        "real user populations, a 3GPP network, or a field trial."
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan_sha256(plan: dict) -> str:
    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _finite_values(values: Iterable[Any]) -> list[float]:
    output = []
    for value in values:
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value))):
            output.append(float(value))
    return output


def _mean(values: Iterable[Any]) -> Optional[float]:
    valid = _finite_values(values)
    return statistics.fmean(valid) if valid else None


def _stdev(values: Iterable[Any]) -> Optional[float]:
    valid = _finite_values(values)
    return statistics.stdev(valid) if len(valid) > 1 else None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict:
    if total <= 0 or successes < 0 or successes > total:
        return {"low_pct": None, "high_pct": None}
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    ) / denominator
    return {
        "low_pct": 100 * max(0.0, center - half),
        "high_pct": 100 * min(1.0, center + half),
    }


def _decision_metrics(items: list[dict]) -> dict:
    correct = sum(
        item["action"] == item["expected_action_for_evaluation_only"]
        for item in items
    )
    total = len(items)
    return {
        "correct": correct,
        "total": total,
        "accuracy_pct": 100 * correct / total if total else None,
        "wilson_95_pct": wilson_interval(correct, total),
    }


def aggregate(results: list[dict]) -> dict:
    by_strategy = {}
    for strategy in STRATEGIES:
        items = [item for item in results if item["strategy"] == strategy]
        post = [item["post_decision"] for item in items]
        decisions = [item["controller_decision"] for item in items]
        ai_items = [
            item for item in items
            if bool(item["controller_decision"].get("ai_called"))
        ]
        bypass_items = [
            item for item in items
            if not bool(item["controller_decision"].get("ai_called"))
        ]
        by_strategy[strategy] = {
            "replays": len(items),
            "full_pipeline_decisions": _decision_metrics(items),
            "ai_only_conflict_decisions": _decision_metrics(ai_items),
            "non_ai_decisions": _decision_metrics(bypass_items),
            "ai_calls": len(ai_items),
            "decision_sources": dict(Counter(
                value.get("source", "UNKNOWN") for value in decisions
            )),
            "mechanical_checks_pass": sum(bool(item["pilot_valid"]) for item in items),
            "mean_post_loss_pct": _mean(item["loss_pct"] for item in post),
            "stdev_post_loss_pct": _stdev(item["loss_pct"] for item in post),
            "mean_post_p95_delay_ms": _mean(
                item["p95_one_way_process_delay_ms"] for item in post
            ),
            "mean_post_jitter_ms": _mean(
                item["rfc3550_interarrival_jitter_ms"] for item in post
            ),
            "mean_post_max_gap_ms": _mean(
                item["max_interarrival_gap_ms"] for item in post
            ),
            "mean_post_complete_frames_pct": _mean(
                item["frames"]["complete_pct"] for item in post
            ),
            "mean_post_goodput_mbps": _mean(
                item["application_goodput_mbps"] for item in post
            ),
        }

    by_problem = {}
    for family in FAMILY_SPECS:
        family_items = [
            item for item in results
            if item["problem_family"] == family["id"]
        ]
        by_problem[family["id"]] = {
            "problem": family["problem"],
            "expected_action": family["expected"],
            "case_type": family["case_type"],
            "by_strategy": {
                strategy: _decision_metrics([
                    item for item in family_items
                    if item["strategy"] == strategy
                ])
                for strategy in STRATEGIES
            },
        }
    return {"by_strategy": by_strategy, "by_problem": by_problem}


CSV_FIELDS = (
    "scenario_id", "problem_family", "common_problem", "case_type",
    "variant_number", "direction", "speed_class", "decision_zone",
    "seed_number", "order_position", "strategy", "action", "expected_action",
    "decision_correct", "decision_source", "decision_status", "ai_called",
    "post_loss_pct", "post_p95_delay_ms", "post_jitter_ms", "post_max_gap_ms",
    "post_complete_frames_pct", "post_goodput_mbps", "pilot_valid",
    "evidence_directory",
)


def write_csv(folder: Path, results: list[dict]) -> None:
    with (folder / "stress_validation_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in results:
            decision = item["controller_decision"]
            post = item["post_decision"]
            writer.writerow({
                "scenario_id": item["scenario_id"],
                "problem_family": item["problem_family"],
                "common_problem": item["common_problem"],
                "case_type": item["case_type"],
                "variant_number": item["variant_number"],
                "direction": item["direction"],
                "speed_class": item["speed_class"],
                "decision_zone": item["decision_zone"],
                "seed_number": item["seed_number"],
                "order_position": item["order_position"],
                "strategy": item["strategy"],
                "action": item["action"],
                "expected_action": item["expected_action_for_evaluation_only"],
                "decision_correct": (
                    item["action"] == item["expected_action_for_evaluation_only"]
                ),
                "decision_source": decision.get("source"),
                "decision_status": decision.get("status"),
                "ai_called": decision.get("ai_called"),
                "post_loss_pct": post.get("loss_pct"),
                "post_p95_delay_ms": post.get("p95_one_way_process_delay_ms"),
                "post_jitter_ms": post.get("rfc3550_interarrival_jitter_ms"),
                "post_max_gap_ms": post.get("max_interarrival_gap_ms"),
                "post_complete_frames_pct": post["frames"].get("complete_pct"),
                "post_goodput_mbps": post.get("application_goodput_mbps"),
                "pilot_valid": item.get("pilot_valid"),
                "evidence_directory": item.get("evidence_directory"),
            })


def _result_path(folder: Path, spec: dict) -> Path:
    return (
        folder / spec["scenario_id"]
        / "seed_{:02d}".format(spec["seed_number"])
        / spec["strategy"].lower()
        / "stress_summary.json"
    )


def _load_completed(folder: Path, specs: list[dict]) -> dict[tuple[str, int, str], dict]:
    completed = {}
    for spec in specs:
        path = _result_path(folder, spec)
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        key = (spec["scenario_id"], spec["seed_number"], spec["strategy"])
        completed[key] = value
    return completed


def _next_attempt(strategy_dir: Path) -> Path:
    attempts = sorted(strategy_dir.glob("attempt_*"))
    path = strategy_dir / "attempt_{:02d}".format(len(attempts) + 1)
    path.mkdir(parents=True)
    return path


def _latest_resumable(root: Path) -> Path:
    folders = sorted((root / "results").glob(FULL_PREFIX + "*"), reverse=True)
    for folder in folders:
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") in {"starting", "interrupted", "failed"}:
            return folder
    raise RuntimeError("No incomplete stress validation was found to resume")


def _new_folder(root: Path, prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = root / "results" / (prefix + stamp)
    folder.mkdir(parents=True)
    return folder


def run_validation(root: Path, *, resume: bool = False,
                   smoke: bool = False) -> int:
    import hosn_switch as helper
    import mn_wifi.net as wifi_module

    if resume and smoke:
        raise ValueError("A smoke run cannot resume a full run")
    selected_scenarios = (
        (STRESS_SCENARIOS[0], STRESS_SCENARIOS[15])
        if smoke else STRESS_SCENARIOS
    )
    selected_seeds = 1 if smoke else SEEDS_PER_SCENARIO
    specs = execution_specs(selected_scenarios, selected_seeds)
    selected_plan = dict(PLAN)
    selected_plan.update(
        smoke_run=smoke,
        scenario_count=len(selected_scenarios),
        seeds_per_scenario=selected_seeds,
        planned_replays=len(specs),
        scenarios=[_scenario_public(item) for item in selected_scenarios],
    )

    model, model_path, training_report = final.find_accepted_model(root)
    model_hash = _sha256(model_path)
    folder = _latest_resumable(root) if resume else _new_folder(
        root, SMOKE_PREFIX if smoke else FULL_PREFIX
    )
    plan_hash = _plan_sha256(selected_plan)
    manifest_path = folder / "manifest.json"
    if resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("plan_sha256") != plan_hash:
            raise RuntimeError("Refusing resume: stress plan changed")
        if manifest.get("frozen_model_sha256") != model_hash:
            raise RuntimeError("Refusing resume: accepted model changed")
        manifest["status"] = "starting"
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
    else:
        compare.write_json(folder / "plan.json", selected_plan)
        source_names = (
            "hosn_stress_validation.py", "hosn_final_comparison.py",
            "hosn_paired_outcome_pilot.py", "hosn_strategy_router.py",
            "hosn_heterogeneous_controller.py", "hosn_wifi_5g_compare.py",
            "hosn_wifi_5g_pilot.py", "hosn_switch.py",
        )
        manifest = {
            "revision": REVISION,
            "status": "starting",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "smoke_run": smoke,
            "plan_sha256": plan_hash,
            "planned_replays": len(specs),
            "completed_replays": 0,
            "model_frozen": True,
            "frozen_model_sha256": model_hash,
            "training_performed": False,
            "training_rows_created": 0,
            "training_labels_assigned": 0,
            "python": platform.python_version(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "mininet_wifi_version": str(getattr(wifi_module, "VERSION", "unknown")),
            "source_sha256": {
                name: _sha256(root / name) for name in source_names
            },
            "resume_count": 0,
        }
    compare.write_json(manifest_path, manifest)

    completed = _load_completed(folder, specs)
    results = list(completed.values())
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
        seed_capability = compare.detect_netem_seed_support(
            station, "sta1-5g0", run
        )
        manifest["netem_seed_capability"] = seed_capability
        compare.write_json(manifest_path, manifest)

        for number, spec in enumerate(specs, start=1):
            key = (spec["scenario_id"], spec["seed_number"], spec["strategy"])
            if key in completed:
                continue
            scenario = spec["scenario"]
            result_path = _result_path(folder, spec)
            strategy_dir = result_path.parent
            strategy_dir.mkdir(parents=True, exist_ok=True)
            attempt = _next_attempt(strategy_dir)
            print(
                "\nREPLAY {}/{}: {} seed={} {}".format(
                    number, len(specs), spec["scenario_id"],
                    spec["seed_number"], spec["strategy"]
                ),
                flush=True,
            )
            callback = lambda snapshot, selected=spec["strategy"]: final.decide(
                selected, snapshot, model
            )
            result = paired.run_one(
                paired.STAY, scenario, attempt, station, server,
                cell_gateway, ap, seed_capability["supported"], run, helper,
                decision_callback=callback,
            )
            result.update(
                replay_id=spec["replay_id"],
                strategy=spec["strategy"],
                scenario_id=spec["scenario_id"],
                seed_number=spec["seed_number"],
                order_position=spec["order_position"],
                problem_family=scenario["problem_family"],
                common_problem=scenario["common_problem"],
                case_type=scenario["case_type"],
                variant_number=scenario["variant_number"],
                direction=scenario["direction"],
                speed_class=scenario["speed_class"],
                decision_zone=scenario["decision_zone"],
                expected_action_for_evaluation_only=scenario[
                    "expected_action_for_evaluation_only"
                ],
                expected_action_basis=scenario["expected_action_basis"],
                evidence_directory=str(attempt.relative_to(folder)),
            )
            compare.write_json(result_path, result)
            completed[key] = result
            results.append(result)
            manifest["completed_replays"] = len(results)
            compare.write_json(manifest_path, manifest)
            post = result["post_decision"]
            print(
                "  action={} source={} ai={} loss={:.3f}% frames={:.2f}%".format(
                    result["action"], result["controller_decision"].get("source"),
                    result["controller_decision"].get("ai_called"),
                    post["loss_pct"], post["frames"]["complete_pct"],
                ),
                flush=True,
            )

        report_aggregate = aggregate(results)
        report = {
            "revision": REVISION,
            "stress_validation_complete": len(results) == len(specs),
            "smoke_run": smoke,
            "completed_replays": len(results),
            "planned_replays": len(specs),
            "model_frozen": True,
            "frozen_model_sha256": model_hash,
            "training_performed": False,
            "training_rows_created": 0,
            "training_labels_assigned": 0,
            "aggregate": report_aggregate,
            "model_training_summary": {
                "source_campaign": training_report.get("source_campaign"),
                "independent_groups": training_report.get(
                    "independent_scenario_groups"
                ),
            },
            "claim_guidance": (
                "Report full-HOSN pipeline accuracy and AI-only conflict accuracy "
                "as separate results. Never describe full-pipeline accuracy as "
                "AI model accuracy."
            ),
            "honesty_note": selected_plan["honesty_note"],
        }
        compare.write_json(folder / "stress_validation_report.json", report)
        write_csv(folder, results)
        complete = len(results) == len(specs)
        manifest.update(
            status="stress_validation_complete" if complete else "stress_validation_incomplete",
            completed_replays=len(results),
            report_file="stress_validation_report.json",
            csv_file="stress_validation_rows.csv",
        )
        compare.write_json(manifest_path, manifest)

        full = report_aggregate["by_strategy"][router.HOSN_FULL_AI]
        ai_only = full["ai_only_conflict_decisions"]
        pipeline = full["full_pipeline_decisions"]
        print("\nHOSN STRESS VALIDATION {}".format(
            "COMPLETE" if complete else "INCOMPLETE"
        ))
        print("Completed replays:", len(results))
        print(
            "Full-HOSN pipeline: {}/{} correct ({:.2f}%)".format(
                pipeline["correct"], pipeline["total"], pipeline["accuracy_pct"]
            )
        )
        print(
            "AI-only conflicts: {}/{} correct ({:.2f}%), Wilson 95% {:.2f}-{:.2f}%".format(
                ai_only["correct"], ai_only["total"], ai_only["accuracy_pct"],
                ai_only["wilson_95_pct"]["low_pct"],
                ai_only["wilson_95_pct"]["high_pct"],
            )
        )
        print("Saved:", folder)
        return 0 if complete else 1
    except KeyboardInterrupt:
        manifest.update(status="interrupted", completed_replays=len(results))
        return 130
    except Exception as exc:
        manifest.update(
            status="failed", completed_replays=len(results),
            error=type(exc).__name__ + ": " + str(exc),
        )
        (folder / "error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        print("\nSTRESS VALIDATION STOPPED:", exc)
        return 1
    finally:
        if network is not None:
            network.stop()
        os.chdir(old_cwd)
        compare.write_json(manifest_path, manifest)
        try:
            helper.restore_result_owner(folder)
        except Exception:
            pass


def run_self_tests() -> int:
    import unittest

    class Tests(unittest.TestCase):
        def test_matrix_size_and_balance(self):
            self.assertEqual(len(FAMILY_SPECS), 10)
            self.assertEqual(len(STRESS_SCENARIOS), 50)
            self.assertEqual(PLAN["planned_replays"], 750)
            self.assertEqual(
                Counter(item["expected_action_for_evaluation_only"]
                        for item in STRESS_SCENARIOS),
                Counter({paired.STAY: 25, paired.HANDOVER: 25}),
            )
            self.assertEqual(
                Counter(item["case_type"] for item in STRESS_SCENARIOS),
                Counter({"conflict": 30, "clear": 20}),
            )
            self.assertEqual(PLAN["planned_full_hosn_conflict_replays"], 150)

        def test_all_profiles_are_unique_and_not_exact_training_profiles(self):
            values = [tuple(profile[name] for name in (
                "delay_ms", "jitter_ms", "loss_pct", "rate_mbit"
            )) for profile in STRESS_PROFILES.values()]
            known = {
                tuple(profile[name] for name in (
                    "delay_ms", "jitter_ms", "loss_pct", "rate_mbit"
                ))
                for profile in (
                    list(measured.DATASET_PROFILES.values())
                    + list(final.EVALUATION_PROFILES.values())
                )
            }
            self.assertEqual(len(values), len(set(values)))
            self.assertFalse(set(values).intersection(known))

        def test_configured_rule_case_matches_declaration(self):
            for item in STRESS_SCENARIOS:
                wifi = item["configured_wifi_profile"]
                cell = item["configured_cell_profile"]
                snap = final.snapshot(
                    item["decision_zone"], wifi["delay_ms"], cell["delay_ms"],
                    wifi["loss_pct"], cell["loss_pct"],
                )
                rule = paired.evaluate_snapshot_rules(snap)
                expected_status = "CONFLICT" if item["case_type"] == "conflict" else "CLEAR"
                self.assertEqual(rule["status"], expected_status, item["id"])
                if item["case_type"] == "clear":
                    self.assertEqual(
                        rule["decision"],
                        item["expected_action_for_evaluation_only"],
                        item["id"],
                    )

        def test_every_seed_runs_each_strategy_once(self):
            specs = execution_specs()
            self.assertEqual(len(specs), 750)
            groups: dict[tuple[str, int], list[str]] = {}
            for spec in specs:
                groups.setdefault(
                    (spec["scenario_id"], spec["seed_number"]), []
                ).append(spec["strategy"])
            self.assertEqual(len(groups), 250)
            self.assertTrue(all(Counter(items) == Counter(STRATEGIES)
                                for items in groups.values()))

        def test_plan_is_evaluation_only_and_strict_json(self):
            json.dumps(PLAN, allow_nan=False)
            self.assertTrue(PLAN["model_frozen"])
            self.assertFalse(PLAN["training_performed"])
            self.assertEqual(PLAN["training_rows_created"], 0)
            self.assertEqual(PLAN["training_labels_assigned"], 0)
            self.assertFalse(PLAN["expected_action_is_controller_input"])

        def test_accuracy_interval_is_not_a_bare_hundred_percent(self):
            interval = wilson_interval(150, 150)
            self.assertLess(interval["low_pct"], 100.0)
            self.assertEqual(interval["high_pct"], 100.0)
            self.assertGreater(interval["low_pct"], 97.0)

        def test_smoke_plan_is_six_replays(self):
            specs = execution_specs(
                (STRESS_SCENARIOS[0], STRESS_SCENARIOS[15]), 1
            )
            self.assertEqual(len(specs), 6)
            self.assertEqual(
                {item["expected_action_for_evaluation_only"]
                 for item in (STRESS_SCENARIOS[0], STRESS_SCENARIOS[15])},
                {paired.STAY, paired.HANDOVER},
            )

    result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    )
    if result.wasSuccessful():
        print(
            "PASS: {} stress-validation checks. No network run, training, "
            "or dataset creation.".format(result.testsRun)
        )
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Frozen-model common-problem HOSN stress validation"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--smoke-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--resume-latest", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return run_self_tests()
    if args.plan:
        print(json.dumps(PLAN, indent=2, allow_nan=False))
        return 0
    if sys.platform != "linux" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        parser.exit(2, "Run network modes in Ubuntu with sudo.\n")
    missing = [name for name in ("ip", "iw", "ping", "tc") if not shutil.which(name)]
    if missing:
        parser.exit(2, "Missing required tools: " + ", ".join(missing) + "\n")
    root = Path(__file__).resolve().parent
    needed = (
        "hosn_final_comparison.py", "hosn_measured_dataset.py",
        "hosn_paired_outcome_pilot.py", "hosn_strategy_router.py",
        "hosn_heterogeneous_controller.py", "hosn_wifi_5g_compare.py",
        "hosn_wifi_5g_pilot.py", "hosn_switch.py",
    )
    missing_files = [name for name in needed if not (root / name).is_file()]
    if missing_files:
        parser.exit(2, "Keep this file beside: " + ", ".join(missing_files) + "\n")
    return run_validation(
        root,
        resume=args.resume_latest,
        smoke=args.smoke_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
