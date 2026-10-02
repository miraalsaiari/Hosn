#!/usr/bin/env python3
"""Rules-first HOSN controller contract for Wi-Fi/cellular comparisons.

This module implements the DECISION layer only. It does not run Mininet,
change an interface, load a model, train a model, or create a dataset.

Three experiment systems remain distinct:

1. RSSI_BASELINE: implemented by the comparison harness, not this module.
2. HOSN_RULES_ONLY: call decide(..., ai_model=None). Clear rules execute;
   a genuine conflict returns a safe HOLD with status AI_UNAVAILABLE.
3. HOSN_FULL_AI: call decide(..., ai_model=reviewed_model). Clear rules still
   bypass AI; only a complete CONFLICT is sent to model.predict exactly once.

Wi-Fi RSSI and cellular RSRP must not be compared as if their raw dBm values
were interchangeable. Each observation is normalized to a link margin:

    signal_margin_db = measured_signal_dbm - service_floor_dbm

The service floor is an experiment configuration, not a measurement. For the
current pilot the planned starting values are -80 dBm for Wi-Fi RSSI and
-105 dBm for emulated cellular RSRP. They must remain recorded with every run.

The future model contract uses eight ordered, measured/derived features:

    current_signal_margin_db, candidate_signal_margin_db,
    current_latency_ms, candidate_latency_ms,
    current_loss_pct, candidate_loss_pct,
    current_trend_db_per_s, candidate_trend_db_per_s

Only STAY or HANDOVER is accepted from a model. Missing data, an absent model,
model exceptions, and malformed predictions produce HOLD/STAY and never grant
permission to switch. A separate make-before-break executor may act only on a
final HANDOVER decision and must verify the target path before breaking Wi-Fi.

Run without sudo:

    python3 hosn_heterogeneous_controller.py --self-test

The tests use invented software fixtures and test-only predictors. They are
not training examples, experimental evidence, or synthetic dataset rows.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple


AI_FEATURE_ORDER = (
    "current_signal_margin_db",
    "candidate_signal_margin_db",
    "current_latency_ms",
    "candidate_latency_ms",
    "current_loss_pct",
    "candidate_loss_pct",
    "current_trend_db_per_s",
    "candidate_trend_db_per_s",
)

MODEL_SCHEMA_VERSION = "hosn-heterogeneous-v1"
REQUIRED_TRAINING_DATA_KIND = "measured_mininet_wifi_emulated_5g"


class Predictor(Protocol):
    """Interface supplied by a separately trained and reviewed model."""

    hosn_schema_version: str
    feature_order: Sequence[str]
    training_data_kind: str

    def predict(self, rows: Sequence[Sequence[float]]) -> Any:
        ...


@dataclass(frozen=True)
class AccessObservation:
    """One access-path observation and its documented normalization floor."""

    access_name: str
    technology: str
    signal_dbm: Optional[float]
    service_floor_dbm: float
    latency_ms: Optional[float]
    loss_pct: Optional[float]
    trend_db_per_s: Optional[float] = None

    @property
    def signal_margin_db(self) -> Optional[float]:
        if self.signal_dbm is None:
            return None
        return float(self.signal_dbm) - float(self.service_floor_dbm)


@dataclass(frozen=True)
class RuleConfig:
    """Prototype policy settings; not universal network thresholds."""

    signal_margin_advantage_db: float = 8.0
    latency_tolerance_ms: float = 0.0
    loss_tolerance_pct: float = 0.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(name + " must be a finite number.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative.")
        if self.signal_margin_advantage_db == 0:
            raise ValueError("signal_margin_advantage_db must be positive.")
        if self.loss_tolerance_pct > 100:
            raise ValueError("loss_tolerance_pct must not exceed 100.")


@dataclass(frozen=True)
class RuleResult:
    decision: str       # STAY, HANDOVER, or ASK_AI.
    status: str         # CLEAR, CONFLICT, NEEDS_DATA, or INVALID_DATA.
    reason: str
    missing_fields: Tuple[str, ...] = ()

    @property
    def needs_ai(self) -> bool:
        return self.decision == "ASK_AI" and self.status == "CONFLICT"


@dataclass(frozen=True)
class ControllerDecision:
    """Auditable final decision; ASK_AI is never exposed as an action."""

    decision: str       # STAY or HANDOVER.
    source: str         # RULES, AI, or HOLD.
    status: str
    reason: str
    rule_decision: str
    rule_status: str
    ai_called: bool
    ai_prediction: Optional[str]
    features: Dict[str, Optional[float]]
    current_access: str
    candidate_access: str

    @property
    def on_hold(self) -> bool:
        return self.source == "HOLD"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = RuleConfig()


def _finite_number(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, Real)
            and math.isfinite(value))


def build_features(current: AccessObservation,
                   candidate: AccessObservation) -> Dict[str, Optional[float]]:
    """Build the exact feature mapping used by both rules and future AI."""
    for role, observation in (("current", current), ("candidate", candidate)):
        if not isinstance(observation, AccessObservation):
            raise TypeError(role + " must be an AccessObservation.")
        if not observation.access_name or not observation.technology:
            raise ValueError(role + " access_name and technology are required.")
        if not _finite_number(observation.service_floor_dbm):
            raise ValueError(role + " service floor must be finite.")
        if observation.service_floor_dbm >= 0:
            raise ValueError(role + " service floor must be a negative dBm value.")

    return {
        "current_signal_margin_db": current.signal_margin_db,
        "candidate_signal_margin_db": candidate.signal_margin_db,
        "current_latency_ms": current.latency_ms,
        "candidate_latency_ms": candidate.latency_ms,
        "current_loss_pct": current.loss_pct,
        "candidate_loss_pct": candidate.loss_pct,
        "current_trend_db_per_s": current.trend_db_per_s,
        "candidate_trend_db_per_s": candidate.trend_db_per_s,
    }


def evaluate_rules(features: Dict[str, Optional[float]],
                   config: RuleConfig = DEFAULT_CONFIG) -> RuleResult:
    """Evaluate normalized cross-technology features without calling AI."""
    if not isinstance(config, RuleConfig):
        raise TypeError("config must be a RuleConfig.")
    if set(features) != set(AI_FEATURE_ORDER):
        raise ValueError("features must contain exactly AI_FEATURE_ORDER.")

    for name in AI_FEATURE_ORDER:
        value = features[name]
        if value is None:
            continue
        if not _finite_number(value):
            return RuleResult("STAY", "INVALID_DATA",
                              name + " is not a finite number.")
        if name.endswith("latency_ms") and value < 0:
            return RuleResult("STAY", "INVALID_DATA",
                              name + " cannot be negative.")
        if name.endswith("loss_pct") and not 0 <= value <= 100:
            return RuleResult("STAY", "INVALID_DATA",
                              name + " must be between 0 and 100.")

    required = AI_FEATURE_ORDER[:6]
    missing = tuple(name for name in required if features[name] is None)
    if missing:
        return RuleResult(
            "STAY", "NEEDS_DATA",
            "Wait for comparable current/candidate measurements: "
            + ", ".join(missing), missing,
        )

    current_margin = float(features["current_signal_margin_db"])
    candidate_margin = float(features["candidate_signal_margin_db"])
    current_latency = float(features["current_latency_ms"])
    candidate_latency = float(features["candidate_latency_ms"])
    current_loss = float(features["current_loss_pct"])
    candidate_loss = float(features["candidate_loss_pct"])

    if current_loss == 100 or candidate_loss == 100:
        return RuleResult(
            "STAY", "INVALID_DATA",
            "A path with 100% probe loss cannot also have a measured RTT; "
            "handle reachability/recovery separately.",
        )

    gap = candidate_margin - current_margin
    delay_better = candidate_latency < current_latency - config.latency_tolerance_ms
    delay_worse = candidate_latency > current_latency + config.latency_tolerance_ms
    loss_better = candidate_loss < current_loss - config.loss_tolerance_pct
    loss_worse = candidate_loss > current_loss + config.loss_tolerance_pct

    if gap >= config.signal_margin_advantage_db and not (delay_worse or loss_worse):
        return RuleResult(
            "HANDOVER", "CLEAR",
            "Candidate normalized margin is better by {:.2f} dB and its "
            "measured delay/loss are no worse.".format(gap),
        )

    if gap < config.signal_margin_advantage_db and not (delay_better or loss_better):
        return RuleResult(
            "STAY", "CLEAR",
            "Candidate lacks the required normalized-margin advantage and "
            "does not improve measured delay or loss.",
        )

    conflict = (
        "Candidate has a sufficient normalized-margin advantage but worse "
        "delay/loss."
        if gap >= config.signal_margin_advantage_db
        else "Candidate improves delay/loss without the required normalized-"
             "margin advantage."
    )
    missing_trends = tuple(
        name for name in AI_FEATURE_ORDER[6:] if features[name] is None
    )
    if missing_trends:
        return RuleResult(
            "STAY", "NEEDS_DATA",
            conflict + " Collect both signal trends before asking AI: "
            + ", ".join(missing_trends), missing_trends,
        )
    return RuleResult("ASK_AI", "CONFLICT", conflict)


def decide(current: AccessObservation,
           candidate: AccessObservation,
           *,
           ai_model: Optional[Predictor] = None,
           config: RuleConfig = DEFAULT_CONFIG) -> ControllerDecision:
    """Return one decision without loading a model or changing the network."""
    features = build_features(current, candidate)
    rule = evaluate_rules(features, config=config)

    def result(action: str, source: str, status: str, reason: str,
               called: bool = False,
               prediction: Optional[str] = None) -> ControllerDecision:
        return ControllerDecision(
            decision=action,
            source=source,
            status=status,
            reason=reason,
            rule_decision=rule.decision,
            rule_status=rule.status,
            ai_called=called,
            ai_prediction=prediction,
            features=dict(features),
            current_access=current.access_name,
            candidate_access=candidate.access_name,
        )

    if rule.status == "CLEAR" and rule.decision in ("STAY", "HANDOVER"):
        return result(rule.decision, "RULES", "CLEAR", rule.reason)

    if not rule.needs_ai:
        return result("STAY", "HOLD", rule.status, rule.reason)

    if ai_model is None:
        return result(
            "STAY", "HOLD", "AI_UNAVAILABLE",
            rule.reason + " No reviewed AI model is connected; hold the "
            "current path and continue measuring.",
        )

    try:
        schema_ok = getattr(ai_model, "hosn_schema_version", None) == MODEL_SCHEMA_VERSION
        order_ok = tuple(getattr(ai_model, "feature_order", ())) == AI_FEATURE_ORDER
        data_ok = (getattr(ai_model, "training_data_kind", None)
                   == REQUIRED_TRAINING_DATA_KIND)
    except Exception:
        schema_ok = order_ok = data_ok = False
    if not (schema_ok and order_ok and data_ok):
        return result(
            "STAY", "HOLD", "AI_SCHEMA_MISMATCH",
            "Model metadata does not match the heterogeneous normalized-feature "
            "contract or measured-experiment training requirement.",
        )

    try:
        predict = getattr(ai_model, "predict", None)
        if not callable(predict):
            return result(
                "STAY", "HOLD", "AI_INVALID_MODEL",
                "The supplied model has no callable predict method.",
            )
    except Exception as exc:
        return result(
            "STAY", "HOLD", "AI_INVALID_MODEL",
            "Could not access model.predict: " + type(exc).__name__,
        )

    row = [[float(features[name]) for name in AI_FEATURE_ORDER]]
    try:
        predictions = predict(row)
    except Exception as exc:
        return result(
            "STAY", "HOLD", "AI_ERROR",
            "AI prediction failed (" + type(exc).__name__ + "); do not switch.",
            called=True,
        )

    try:
        if isinstance(predictions, (str, bytes)) or len(predictions) != 1:
            raise ValueError("Expected exactly one prediction.")
        label = predictions[0]
        if not isinstance(label, str) or label not in ("STAY", "HANDOVER"):
            raise ValueError("Expected STAY or HANDOVER.")
    except Exception:
        return result(
            "STAY", "HOLD", "AI_INVALID_OUTPUT",
            "AI must return exactly one STAY or HANDOVER label; do not switch.",
            called=True,
        )

    label = str(label)
    return result(
        label, "AI", "AI_DECIDED",
        rule.reason + " AI selected " + label + ".",
        called=True, prediction=label,
    )


# Test-only objects below are never saved as training data or loaded in a run.
_DEFAULT_REPLY = object()


class _TestOnlyPredictor:
    def __init__(self, reply=_DEFAULT_REPLY, error: Optional[Exception] = None):
        self.hosn_schema_version = MODEL_SCHEMA_VERSION
        self.feature_order = AI_FEATURE_ORDER
        self.training_data_kind = REQUIRED_TRAINING_DATA_KIND
        self.reply = ["HANDOVER"] if reply is _DEFAULT_REPLY else reply
        self.error = error
        self.calls = []

    def predict(self, rows):
        self.calls.append(rows)
        if self.error is not None:
            raise self.error
        return self.reply


def _observations():
    current = AccessObservation(
        access_name="wifi", technology="wifi_rssi",
        signal_dbm=-75.0, service_floor_dbm=-80.0,
        latency_ms=20.0, loss_pct=0.0, trend_db_per_s=-1.0,
    )
    candidate = AccessObservation(
        access_name="emulated_5g", technology="cellular_rsrp",
        signal_dbm=-90.0, service_floor_dbm=-105.0,
        latency_ms=20.0, loss_pct=0.0, trend_db_per_s=0.0,
    )
    return current, candidate


def run_self_tests() -> int:
    import unittest

    class Tests(unittest.TestCase):
        def test_normalizes_each_technology_against_its_floor(self):
            current, candidate = _observations()
            features = build_features(current, candidate)
            self.assertEqual(features["current_signal_margin_db"], 5.0)
            self.assertEqual(features["candidate_signal_margin_db"], 15.0)

        def test_clear_handover_bypasses_ai(self):
            current, candidate = _observations()
            trap = _TestOnlyPredictor(error=RuntimeError("must not be called"))
            result = decide(current, candidate, ai_model=trap)
            self.assertEqual((result.decision, result.source),
                             ("HANDOVER", "RULES"))
            self.assertEqual(trap.calls, [])

        def test_clear_stay_bypasses_ai(self):
            current, candidate = _observations()
            candidate = AccessObservation(
                **dict(asdict(candidate), signal_dbm=-102.0,
                       latency_ms=30.0, loss_pct=1.0)
            )
            trap = _TestOnlyPredictor(error=RuntimeError("must not be called"))
            result = decide(current, candidate, ai_model=trap)
            self.assertEqual((result.decision, result.source), ("STAY", "RULES"))
            self.assertEqual(trap.calls, [])

        def test_conflict_without_model_holds(self):
            current, candidate = _observations()
            candidate = AccessObservation(
                **dict(asdict(candidate), latency_ms=40.0)
            )
            result = decide(current, candidate)
            self.assertEqual((result.decision, result.source, result.status),
                             ("STAY", "HOLD", "AI_UNAVAILABLE"))
            self.assertEqual(result.rule_decision, "ASK_AI")
            self.assertFalse(result.ai_called)

        def test_conflict_calls_ai_once_in_exact_order(self):
            current, candidate = _observations()
            candidate = AccessObservation(
                **dict(asdict(candidate), latency_ms=40.0)
            )
            model = _TestOnlyPredictor(["HANDOVER"])
            result = decide(current, candidate, ai_model=model)
            self.assertEqual((result.decision, result.source, result.status),
                             ("HANDOVER", "AI", "AI_DECIDED"))
            expected = [[float(result.features[name]) for name in AI_FEATURE_ORDER]]
            self.assertEqual(model.calls, [expected])
            self.assertTrue(result.ai_called)

        def test_ai_may_choose_stay(self):
            current, candidate = _observations()
            candidate = AccessObservation(
                **dict(asdict(candidate), latency_ms=40.0)
            )
            result = decide(current, candidate,
                            ai_model=_TestOnlyPredictor(["STAY"]))
            self.assertEqual((result.decision, result.source), ("STAY", "AI"))

        def test_missing_trend_does_not_call_ai(self):
            current, candidate = _observations()
            candidate = AccessObservation(
                **dict(asdict(candidate), latency_ms=40.0,
                       trend_db_per_s=None)
            )
            trap = _TestOnlyPredictor(error=RuntimeError("must not be called"))
            result = decide(current, candidate, ai_model=trap)
            self.assertEqual((result.source, result.status),
                             ("HOLD", "NEEDS_DATA"))
            self.assertEqual(trap.calls, [])

        def test_invalid_ai_outputs_hold(self):
            current, candidate = _observations()
            candidate = AccessObservation(
                **dict(asdict(candidate), latency_ms=40.0)
            )
            bad_outputs = ([], ["STAY", "HANDOVER"], ["ASK_AI"],
                           [1], "HANDOVER", None)
            for output in bad_outputs:
                with self.subTest(output=output):
                    result = decide(current, candidate,
                                    ai_model=_TestOnlyPredictor(output))
                    self.assertEqual((result.decision, result.source,
                                      result.status),
                                     ("STAY", "HOLD", "AI_INVALID_OUTPUT"))

        def test_ai_exception_holds(self):
            current, candidate = _observations()
            candidate = AccessObservation(
                **dict(asdict(candidate), latency_ms=40.0)
            )
            result = decide(
                current, candidate,
                ai_model=_TestOnlyPredictor(error=RuntimeError("test")),
            )
            self.assertEqual((result.decision, result.source, result.status),
                             ("STAY", "HOLD", "AI_ERROR"))
            self.assertTrue(result.ai_called)

        def test_old_or_wrong_schema_model_is_rejected(self):
            current, candidate = _observations()
            candidate = AccessObservation(
                **dict(asdict(candidate), latency_ms=40.0)
            )
            model = _TestOnlyPredictor(["HANDOVER"])
            model.feature_order = ("current_rssi", "candidate_rssi")
            result = decide(current, candidate, ai_model=model)
            self.assertEqual((result.decision, result.source, result.status),
                             ("STAY", "HOLD", "AI_SCHEMA_MISMATCH"))
            self.assertEqual(model.calls, [])

        def test_missing_required_measurement_holds(self):
            current, candidate = _observations()
            candidate = AccessObservation(
                **dict(asdict(candidate), latency_ms=None)
            )
            result = decide(current, candidate,
                            ai_model=_TestOnlyPredictor(["HANDOVER"]))
            self.assertEqual((result.decision, result.source, result.status),
                             ("STAY", "HOLD", "NEEDS_DATA"))
            self.assertFalse(result.ai_called)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if result.wasSuccessful():
        print("PASS: {} controller-contract checks.".format(result.testsRun))
        print("No Mininet run, model training, dataset, or network change occurred.")
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Software-only HOSN heterogeneous AI-routing contract."
    )
    parser.add_argument("--self-test", action="store_true", required=True)
    args = parser.parse_args()
    if args.self_test:
        return run_self_tests()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
