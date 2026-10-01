#!/usr/bin/env python3
"""HOSN: rules first, AI only for a genuinely unclear comparison.

This NEW module does not import or change hosn_engine.py. The future controller
must use THIS module for its rules, not run two competing rule engines.

    result = evaluate_rules(**measurements)
    result.decision: STAY, HANDOVER, or ASK_AI
    result.status: CLEAR, CONFLICT, NEEDS_DATA, or INVALID_DATA

ASK_AI is a request for another decision, NOT permission to switch. STAY with
NEEDS_DATA/INVALID_DATA is a temporary no-action fallback, not evidence that the
current AP is better. A controller should obtain/correct the measurements.

Units: RSSI in dBm; latency in milliseconds (ping RTT for this experiment);
loss in percent, 0..100; trends in dB/second, using the same observation window.
Missing observations must be None, NEVER copied from the other AP or set to 0.
A zero trend is valid only when measured history actually shows no change.

Caller responsibilities: collect fresh, comparable measurements through BOTH
AP paths, check candidate reachability, and track timestamps. The original
handover_simulation.py measures ping only through the connected AP: its CSV
alone CANNOT supply this module's full current/candidate comparison.

This is an editable prototype policy, not a vendor standard or validated optimum.
The 8-dB starting margin is retained from the existing project; delay/loss
comparison tolerances default to zero (equal or better is acceptable). Optional
positive tolerances must be justified from repeated measurements, not selected
on the final evaluation set. Trends are extra AI context, not mandatory
'current weakening AND candidate improving' conditions for clear decisions.

No model is loaded, no network commands are run, and no files are written.
A later controller must add temporal confirmation/cooldown, an approved AI for
CONFLICT cases, and an executor that requests AND verifies an AP switch. Missing
data must not be filled with invented numbers to activate that AI. This module
is not a reconnection routine for a disconnected station.

Run: python3 hosn_rules.py --self-test
Self-tests use explicitly invented inputs for software checks, NOT training
measurements or evidence that HOSN improves a real or simulated network.
"""

import argparse
import math
from dataclasses import dataclass
from numbers import Real
from typing import Optional, Tuple


# Keep these names/order consistent with the friend's training script.
AI_FEATURE_ORDER = (
    "current_rssi", "candidate_rssi",
    "current_latency", "candidate_latency",
    "current_loss", "candidate_loss",
    "current_trend", "candidate_trend",
)


@dataclass(frozen=True)
class RuleConfig:
    """Starting settings, not measured data or universal Wi-Fi thresholds."""

    rssi_margin_db: float = 8.0
    latency_tolerance_ms: float = 0.0
    loss_tolerance_pct: float = 0.0  # Percentage POINTS, not relative percent.

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(name + " must be a finite number.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative.")
        if self.rssi_margin_db == 0:
            raise ValueError("rssi_margin_db must be positive.")
        if self.loss_tolerance_pct > 100:
            raise ValueError("loss_tolerance_pct must not exceed 100.")


@dataclass(frozen=True)
class RuleResult:
    decision: str
    status: str
    reason: str
    missing_fields: Tuple[str, ...] = ()

    @property
    def needs_ai(self) -> bool:
        return self.decision == "ASK_AI" and self.status == "CONFLICT"


DEFAULT_CONFIG = RuleConfig()


def evaluate_rules(
    current_rssi: Optional[float],
    candidate_rssi: Optional[float],
    current_latency: Optional[float],
    candidate_latency: Optional[float],
    current_loss: Optional[float],
    candidate_loss: Optional[float],
    current_trend: Optional[float] = None,
    candidate_trend: Optional[float] = None,
    *,
    config: RuleConfig = DEFAULT_CONFIG,
) -> RuleResult:
    """Return one rule result without calling AI or controlling the network.

    A significant signal advantage plus no worse latency/loss -> HANDOVER.
    No significant signal advantage and no latency/loss advantage -> STAY.
    Other trade-offs -> ASK_AI, provided all eight AI inputs are available.

    Equality is allowed: 0% loss versus 0% loss does NOT veto a clear handover.
    A slightly better signal within the RSSI margin is treated as inconclusive,
    not enough by itself to justify a switch. A QoS advantage in that situation
    goes to AI; without one, stay. A significantly weaker candidate with better
    delay or loss is also a trade-off, not an automatic handover.
    """
    if not isinstance(config, RuleConfig):
        raise TypeError("config must be a RuleConfig.")

    values = dict(zip(AI_FEATURE_ORDER, (
        current_rssi, candidate_rssi, current_latency, candidate_latency,
        current_loss, candidate_loss, current_trend, candidate_trend,
    )))
    for name, value in values.items():
        if value is None:
            continue
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value)):
            return RuleResult("STAY", "INVALID_DATA", name + " is not a finite number.")
        if name.endswith("rssi") and value >= 0:
            return RuleResult("STAY", "INVALID_DATA", name + " must be a negative dBm reading.")
        if name.endswith("latency") and value < 0:
            return RuleResult("STAY", "INVALID_DATA", name + " cannot be negative.")
        if name.endswith("loss") and not 0 <= value <= 100:
            return RuleResult("STAY", "INVALID_DATA", name + " must be a percentage from 0 to 100.")

    missing = tuple(name for name in AI_FEATURE_ORDER[:6] if values[name] is None)
    if missing:
        return RuleResult(
            "STAY", "NEEDS_DATA",
            "Wait for AP-specific measurements; do not guess: " + ", ".join(missing),
            missing,
        )

    # With ping RTT, 100% loss has no measured RTT. Catch accidental zero-fill.
    if current_loss == 100 or candidate_loss == 100:
        return RuleResult(
            "STAY", "INVALID_DATA",
            "100% ping loss cannot also have a measured RTT. Use None for that RTT; "
            "the controller must handle reachability/recovery separately.",
        )

    gap = candidate_rssi - current_rssi
    delay_better = candidate_latency < current_latency - config.latency_tolerance_ms
    delay_worse = candidate_latency > current_latency + config.latency_tolerance_ms
    loss_better = candidate_loss < current_loss - config.loss_tolerance_pct
    loss_worse = candidate_loss > current_loss + config.loss_tolerance_pct

    if gap >= config.rssi_margin_db and not (delay_worse or loss_worse):
        return RuleResult(
            "HANDOVER", "CLEAR",
            "Candidate signal is better by {:.2f} dB; delay and loss are no worse "
            "within the configured tolerances. Equal loss is allowed.".format(gap),
        )

    if gap < config.rssi_margin_db and not (delay_better or loss_better):
        return RuleResult(
            "STAY", "CLEAR",
            "Candidate has neither the required signal advantage nor better delay/loss.",
        )

    # Remaining cases have a performance trade-off or an insufficient signal
    # margin with a delay/loss benefit. These are the ONLY route to the AI.
    if gap >= config.rssi_margin_db:
        conflict = "Candidate signal is significantly stronger, but delay or loss is worse."
    else:
        conflict = "Candidate improves delay or loss, but lacks the required signal advantage."

    missing_trends = tuple(name for name in AI_FEATURE_ORDER[6:] if values[name] is None)
    if missing_trends:
        return RuleResult(
            "STAY", "NEEDS_DATA",
            conflict + " Collect the missing signal history before asking the AI: "
            + ", ".join(missing_trends),
            missing_trends,
        )
    return RuleResult("ASK_AI", "CONFLICT", conflict)


def run_self_tests() -> int:
    """Deterministic invented fixtures: no data files, model, or VM required."""
    base = dict(
        current_rssi=-75.0, candidate_rssi=-60.0,
        current_latency=20.0, candidate_latency=20.0,
        current_loss=0.0, candidate_loss=0.0,
        current_trend=-1.0, candidate_trend=1.0,
    )
    checks = 0

    def check(name, changes, expected, config=DEFAULT_CONFIG):
        nonlocal checks
        observed = evaluate_rules(**dict(base, **changes), config=config)
        actual = (observed.decision, observed.status)
        if actual != expected:
            raise AssertionError("{}: expected {}, got {} ({})".format(
                name, expected, actual, observed.reason))
        if observed.needs_ai != (observed.decision == "ASK_AI"):
            raise AssertionError(name + ": incorrect AI routing flag")
        if not observed.reason:
            raise AssertionError(name + ": missing explanation")
        checks += 1

    clear_switch = ("HANDOVER", "CLEAR")
    clear_stay = ("STAY", "CLEAR")
    ask = ("ASK_AI", "CONFLICT")
    wait = ("STAY", "NEEDS_DATA")
    invalid = ("STAY", "INVALID_DATA")

    cases = [
        ("Equal zero loss and equal latency permit a switch", {}, clear_switch),
        ("All three primary metrics favor candidate", {
            "candidate_latency": 10, "current_loss": 5, "candidate_loss": 1}, clear_switch),
        ("Equal nonzero loss does not block switch", {
            "current_loss": 2, "candidate_loss": 2}, clear_switch),
        ("Exact 8-dB boundary", {"candidate_rssi": -67}, clear_switch),
        ("Just below signal margin", {"candidate_rssi": -67.01}, clear_stay),
        ("Equal primary metrics", {"candidate_rssi": -75}, clear_stay),
        ("Candidate worse in every metric", {
            "candidate_rssi": -90, "candidate_latency": 40, "candidate_loss": 5}, clear_stay),
        ("Stronger signal, worse delay", {"candidate_latency": 50}, ask),
        ("Stronger signal, worse loss", {"candidate_loss": 5}, ask),
        ("Stronger signal, faster but more loss", {
            "candidate_latency": 10, "candidate_loss": 5}, ask),
        ("Stronger signal, less loss but slower", {
            "current_loss": 5, "candidate_loss": 1, "candidate_latency": 50}, ask),
        ("Weaker signal but faster", {"candidate_rssi": -90, "candidate_latency": 5}, ask),
        ("Weaker signal but less loss", {
            "candidate_rssi": -90, "current_loss": 5}, ask),
        ("Small signal advantage and faster", {
            "candidate_rssi": -72, "candidate_latency": 10}, ask),
        ("Equal signal with QoS conflict", {
            "candidate_rssi": -75, "candidate_latency": 10, "candidate_loss": 5}, ask),
        ("Small signal gain with no QoS gain", {
            "candidate_rssi": -72, "candidate_latency": 25}, clear_stay),
        ("Clear case does not require directional trends", {
            "current_trend": 2, "candidate_trend": -2}, clear_switch),
        ("Clear case works without history", {
            "current_trend": None, "candidate_trend": None}, clear_switch),
        ("Conflict needs current trend", {
            "candidate_latency": 50, "current_trend": None}, wait),
        ("Conflict needs candidate trend", {
            "candidate_loss": 5, "candidate_trend": None}, wait),
        ("Clear stay needs no trends", {
            "candidate_rssi": -90, "current_trend": None, "candidate_trend": None}, clear_stay),
        ("Measured zero trends are allowed for AI", {
            "candidate_latency": 50, "current_trend": 0, "candidate_trend": 0}, ask),
        ("Zero latency is distinct from absent latency", {
            "current_latency": 0, "candidate_latency": 0}, clear_switch),
        ("Current no-reply window has missing RTT", {
            "current_loss": 100, "current_latency": None}, wait),
        ("Candidate no-reply window has missing RTT", {
            "candidate_loss": 100, "candidate_latency": None}, wait),
        ("Reject fabricated current RTT after 100% loss", {"current_loss": 100}, invalid),
        ("Reject fabricated candidate RTT after 100% loss", {"candidate_loss": 100}, invalid),
    ]
    for case in cases:
        check(*case)
    for name in AI_FEATURE_ORDER[:6]:
        check("Missing " + name, {name: None}, wait)
    for name in AI_FEATURE_ORDER:
        for value in (float("nan"), float("inf"), True, "bad"):
            check("Invalid " + name, {name: value}, invalid)
    for name, value in (
        ("current_rssi", 0), ("candidate_rssi", 0), ("candidate_rssi", 4),
        ("current_latency", -1), ("candidate_latency", -1),
        ("current_loss", -0.1), ("candidate_loss", 100.1),
    ):
        check("Out-of-domain " + name, {name: value}, invalid)

    relaxed = RuleConfig(latency_tolerance_ms=2.0, loss_tolerance_pct=0.5)
    check("Tolerance boundary", {"candidate_latency": 22, "candidate_loss": 0.5}, clear_switch, relaxed)
    check("Outside delay tolerance", {"candidate_latency": 22.01}, ask, relaxed)
    check("Outside loss tolerance", {"candidate_loss": 0.51}, ask, relaxed)
    check("Configurable signal margin", {"candidate_rssi": -72}, clear_switch, RuleConfig(rssi_margin_db=3))
    for settings in (
        {"rssi_margin_db": 0}, {"rssi_margin_db": -1}, {"rssi_margin_db": True},
        {"rssi_margin_db": float("nan")}, {"latency_tolerance_ms": -1},
        {"loss_tolerance_pct": -1}, {"loss_tolerance_pct": 101},
    ):
        try:
            RuleConfig(**settings)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("Invalid configuration was accepted: " + repr(settings))

    print("PASS: {} rule/input checks.".format(checks))
    print("These were invented SOFTWARE TEST inputs, not Mininet training data.")
    print("No AI was called, no AP was switched, and no files were changed.")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="HOSN rules-only module; no network side effects.")
    parser.add_argument("--self-test", action="store_true", help="Run deterministic software checks.")
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
    else:
        print("Rules file ready. To check it: python3 hosn_rules.py --self-test")
        print("This file alone does not run the AI or change a Wi-Fi connection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
